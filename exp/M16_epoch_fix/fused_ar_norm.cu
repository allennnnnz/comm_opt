// M15 (bf16 support; otherwise identical to M14) prototype: one kernel = start barrier -> pull reduce-scatter by token rows -> residual add + RMSNorm on the
// rows this GPU owns -> push all-gather of the normalized rows with LL128 lines (8-byte epoch flag per 128-byte line),
// no mid barrier. TP=8, fp16 or bf16 (one kernel instance per type). Built as a PyTorch extension (torch.utils.cpp_extension.load).
//
// Buffers (all but ctr/residual/gamma/out are PyTorch symmetric memory, addressed through per-rank pointer lists):
//   x[k]      rank k's input, T x H fp16 (peers pull from it)
//   land[k]   rank k's landing area: 2 (epoch parity) x 8 sources x rows_per x gpr groups x 64 u64
//   flags[k]  rank k's start-barrier flags: MAXB x 8 u32 (vLLM-style, one slot per block index and peer)
//   ctr       local per-block epoch counters (u64), so the kernel can be captured in a CUDA graph
//   residual  local T x H fp16; only this rank's rows are read and written (residual kept sharded, as LMDeploy)
//   out       local T x H fp16, normalized output for all rows
// Why no mid barrier is needed: a rank finishes only after receiving all 7 peers' all-gather data, and every peer
// sends it only after it has pulled from every rank's input; so no one is still reading x when the next call's
// producer overwrites it, and landing halves alternate by epoch parity (epoch e+2 is written only after the receiver
// finished e, since every rank receives from every rank each epoch).
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <vector>

namespace {

constexpr int NG = 8, THREADS = 512, MAXU = 4, MAXB = 128;  // MAXU: 16-byte units per thread, H <= 16384

// the only type-dependent code: 16-bit pairs <-> fp32
template <typename T>
struct Pair;
template <>
struct Pair<half> {
  using t = __half2;
  static __device__ __forceinline__ float2 f2(t v) { return __half22float2(v); }
  static __device__ __forceinline__ t pk(float a, float b) { return __floats2half2_rn(a, b); }
};
template <>
struct Pair<__nv_bfloat16> {
  using t = __nv_bfloat162;
  static __device__ __forceinline__ float2 f2(t v) { return __bfloat1622float2(v); }
  static __device__ __forceinline__ t pk(float a, float b) { return __floats2bfloat162_rn(a, b); }
};

struct Args {  // element pointers are 16-bit; the kernel moves them as raw 16-byte units
  const uint16_t* x[NG];
  unsigned long long* land[NG];
  uint32_t* flags[NG];
  unsigned long long* ctr;
  uint16_t* residual;
  const uint16_t* gamma;
  uint16_t* out;
  int rank, T, H, rows_per, gpr;
  float eps;
};

__device__ __forceinline__ uint32_t ldv32(const uint32_t* p) { return *(const volatile uint32_t*)p; }
__device__ __forceinline__ void stv32(uint32_t* p, uint32_t v) { *(volatile uint32_t*)p = v; }
__device__ __forceinline__ void st_v2(unsigned long long* p, unsigned long long a, unsigned long long b) {
  asm volatile("st.volatile.global.v2.u64 [%0], {%1,%2};" ::"l"(p), "l"(a), "l"(b) : "memory");
}
__device__ __forceinline__ void ld_v2(const unsigned long long* p, unsigned long long& a, unsigned long long& b) {
  asm volatile("ld.volatile.global.v2.u64 {%0,%1}, [%2];" : "=l"(a), "=l"(b) : "l"(p) : "memory");
}
template <typename T>
__device__ __forceinline__ void add8(float* f, const uint4& v) {
  const typename Pair<T>::t* h = reinterpret_cast<const typename Pair<T>::t*>(&v);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 t = Pair<T>::f2(h[i]);
    f[2 * i] += t.x;
    f[2 * i + 1] += t.y;
  }
}
template <typename T>
__device__ __forceinline__ uint4 pack8(const float* f) {
  uint4 v;
  typename Pair<T>::t* h = reinterpret_cast<typename Pair<T>::t*>(&v);
#pragma unroll
  for (int i = 0; i < 4; ++i) h[i] = Pair<T>::pk(f[2 * i], f[2 * i + 1]);
  return v;
}
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

template <typename T>
__global__ void __launch_bounds__(THREADS, 1) fused_ar_norm_kernel(Args a) {
  const unsigned long long epoch = a.ctr[blockIdx.x] + 1;
  if (threadIdx.x < NG) {  // start barrier: peers' inputs are ready before anyone pulls
    stv32(&a.flags[threadIdx.x][blockIdx.x * NG + a.rank], (uint32_t)epoch);
    while (ldv32(&a.flags[a.rank][blockIdx.x * NG + threadIdx.x]) != (uint32_t)epoch) {
    }
  }
  __syncthreads();
  const int r = a.rank, H = a.H, U = H / 8, PR = H / 4;  // U: 16-byte units per row; PR: payload u64 per row
  const size_t slot = (size_t)a.rows_per * a.gpr * 64;  // u64 per source in a landing half
  const size_t off = (epoch & 1) * (size_t)NG * slot;
  const int lane = threadIdx.x & 31, part = lane & 7, sub = lane >> 3, warp = threadIdx.x >> 5;
  const int first = r * a.rows_per;
  const int my_rows = max(0, min(a.T - first, a.rows_per));
  __shared__ float red[THREADS / 32];
  __shared__ float s_rstd;

  // phase 1: reduce-scatter (pull) + residual + RMSNorm for this rank's rows, then push them
  for (int j = blockIdx.x; j < my_rows; j += gridDim.x) {
    const int row = first + j;
    float acc[MAXU][8];
    float ss = 0.f;
#pragma unroll
    for (int q = 0; q < MAXU; ++q) {
      const int u = threadIdx.x + q * THREADS;
      if (u < U) {
        uint4 v[NG];
#pragma unroll
        for (int k = 0; k < NG; ++k) v[k] = reinterpret_cast<const uint4*>(a.x[k] + (size_t)row * H)[u];
        float f[8] = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
        for (int k = 0; k < NG; ++k) add8<T>(f, v[k]);
        uint4* rp = reinterpret_cast<uint4*>(a.residual + (size_t)row * H) + u;
        add8<T>(f, *rp);
        const uint4 rv = pack8<T>(f);  // residual is stored in fp16; the norm uses the stored values (as vLLM)
        *rp = rv;
        float z[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        add8<T>(z, rv);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          acc[q][i] = z[i];
          ss += z[i] * z[i];
        }
      }
    }
    ss = warp_sum(ss);
    if (lane == 0) red[warp] = ss;
    __syncthreads();
    if (warp == 0) {
      float t = lane < THREADS / 32 ? red[lane] : 0.f;
      t = warp_sum(t);
      if (lane == 0) s_rstd = rsqrtf(t / H + a.eps);
    }
    __syncthreads();
    const float rstd = s_rstd;
#pragma unroll
    for (int q = 0; q < MAXU; ++q) {
      const int u = threadIdx.x + q * THREADS;
      if (u < U) {
        float g[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        add8<T>(g, reinterpret_cast<const uint4*>(a.gamma)[u]);
        float o[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) o[i] = acc[q][i] * rstd * g[i];
        reinterpret_cast<uint4*>(a.out + (size_t)row * H)[u] = pack8<T>(o);
      }
    }
    __syncthreads();  // the whole row is in the local output before it is pushed
    const unsigned long long* src = reinterpret_cast<const unsigned long long*>(a.out + (size_t)row * H);
    for (int g = warp; g < a.gpr; g += THREADS / 32) {
      const int i0 = (g * 4 + sub) * 15 + part * 2;
      const unsigned long long va = i0 < PR ? src[i0] : 0ull;
      const unsigned long long vb = part == 7 ? epoch : (i0 + 1 < PR ? src[i0 + 1] : 0ull);
#pragma unroll
      for (int i = 1; i < NG; ++i) {
        const int k = (r + i) % NG;
        st_v2(a.land[k] + off + (size_t)r * slot + ((size_t)j * a.gpr + g) * 64 + 2 * lane, va, vb);
      }
    }
  }

  // phase 2: all-gather receive (poll the epoch flags in this rank's landing area)
  const int gw = blockIdx.x * (THREADS / 32) + warp, nw = gridDim.x * (THREADS / 32);
  const int per_src = a.rows_per * a.gpr;
  for (int idx = gw; idx < (NG - 1) * per_src; idx += nw) {
    const int s = (r + 1 + idx / per_src) % NG, rem = idx % per_src, jl = rem / a.gpr, g = rem % a.gpr;
    const int row = s * a.rows_per + jl;
    if (row >= a.T) continue;  // warp-uniform
    const unsigned long long* gp = a.land[r] + off + (size_t)s * slot + (size_t)rem * 64;
    unsigned long long va, vb;
    do {
      ld_v2(gp + 2 * lane, va, vb);
    } while (!__all_sync(0xffffffffu, part != 7 || vb == epoch));
    unsigned long long* dst = reinterpret_cast<unsigned long long*>(a.out + (size_t)row * H);
    const int i0 = (g * 4 + sub) * 15 + part * 2;
    if (i0 < PR) dst[i0] = va;
    if (part != 7 && i0 + 1 < PR) dst[i0 + 1] = vb;
  }
  __syncthreads();
  if (threadIdx.x == 0) a.ctr[blockIdx.x] = epoch;
}

}  // namespace

void fused_ar_norm(const std::vector<int64_t>& x_ptrs, const std::vector<int64_t>& land_ptrs,
                   const std::vector<int64_t>& flag_ptrs, torch::Tensor ctr, torch::Tensor residual,
                   torch::Tensor gamma, torch::Tensor out, int64_t rank, int64_t T, int64_t H, double eps,
                   int64_t ctas, int64_t rows_per, int64_t gpr) {
  TORCH_CHECK(x_ptrs.size() == NG && land_ptrs.size() == NG && flag_ptrs.size() == NG, "needs 8 ranks");
  TORCH_CHECK(H % 8 == 0 && H / 8 <= MAXU * THREADS, "unsupported hidden size");
  TORCH_CHECK(ctas >= 1 && ctas <= MAXB, "ctas out of range");
  const auto dt = residual.scalar_type();
  TORCH_CHECK((dt == torch::kHalf || dt == torch::kBFloat16) && out.scalar_type() == dt && gamma.scalar_type() == dt,
              "fp16 or bf16, same type for residual, gamma, out");
  Args a{};
  for (int k = 0; k < NG; ++k) {
    a.x[k] = reinterpret_cast<const uint16_t*>(x_ptrs[k]);
    a.land[k] = reinterpret_cast<unsigned long long*>(land_ptrs[k]);
    a.flags[k] = reinterpret_cast<uint32_t*>(flag_ptrs[k]);
  }
  a.ctr = reinterpret_cast<unsigned long long*>(ctr.data_ptr<int64_t>());
  a.residual = reinterpret_cast<uint16_t*>(residual.data_ptr());
  a.gamma = reinterpret_cast<const uint16_t*>(gamma.data_ptr());
  a.out = reinterpret_cast<uint16_t*>(out.data_ptr());
  a.rank = (int)rank;
  a.T = (int)T;
  a.H = (int)H;
  a.rows_per = (int)rows_per;
  a.gpr = (int)gpr;
  a.eps = (float)eps;
  auto stream = at::cuda::getCurrentCUDAStream();
  if (dt == torch::kHalf)
    fused_ar_norm_kernel<half><<<(int)ctas, THREADS, 0, stream>>>(a);
  else
    fused_ar_norm_kernel<__nv_bfloat16><<<(int)ctas, THREADS, 0, stream>>>(a);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("fused_ar_norm", &fused_ar_norm); }
