// M12: push (LL128-style, 16-byte stores, flag in every 128-byte line) vs pull (vLLM-style 16-byte loads) on 8x A100.
//
// Message S (fp16) per all-reduce; every GPU owns S/8 (its "slice"). Variants:
//   0 ag_pull  all-gather: each thread loads 16 B from the 7 peers' slices (data placed before the kernel), as vLLM
//              stage 2
//   1 ag_push  all-gather: each GPU writes its slice to the 7 peers in LL128 lines (lanes 0..6 of an 8-lane group carry
//              16 B of data, lane 7 carries 8 B of data + an 8-byte epoch flag); receivers poll their own memory and copy
//              the payload to the output once every flag in the warp's 4 lines matches
//   2 rs_pull  reduce-scatter: each thread loads 16 B from the 8 inputs at its own slice, sums in fp32 in order 0..7
//   3 rs_push  reduce-scatter: each GPU writes slice k of its input to owner k's slot[rank] in LL128 lines; the owner
//              polls its 8 slots and sums in fp32 in slot order 0..7 (bitwise identical to rs_pull)
// Pull variants have their data in place before launch: their time is pure data movement, no waiting.
//
// Per iteration: buffers cleared, all GPUs launched, a start barrier aligns the blocks (not timed), each block times
// (clock64) its whole share; reported time = max over blocks and GPUs (cycles -> ns by per-GPU calibration).
//
// Build: nvcc -O3 -arch=sm_80 -std=c++20 -o ll128_probe ll128_probe.cu
// Usage: ./ll128_probe [seed]   -> CSV on stdout
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#define CK(x)                                                                    \
  do {                                                                           \
    cudaError_t err_ = (x);                                                      \
    if (err_ != cudaSuccess) {                                                   \
      fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(err_)); \
      exit(1);                                                                   \
    }                                                                            \
  } while (0)

constexpr int NG = 8, THREADS = 512, MAXB = 128, WARM = 5, ITERS = 50;
constexpr long SPIN_LIMIT = 1l << 22;
const std::vector<int> SIZES = {80 << 10, 160 << 10, 320 << 10, 640 << 10, 1280 << 10};
const std::vector<int> CTAS = {16, 32, 64, 108};
const char* VNAME[] = {"ag_pull", "ag_push", "rs_pull", "rs_push"};

struct Params {
  int rank, variant;
  int P;       // payload u64 per slice (S/8 bytes / 8)
  int groups;  // 4-line groups per slice (LL128 lines = 4 * groups)
  uint64_t epoch;
  const uint64_t* in;       // RS: local full input (8 * P u64, padded)
  const uint64_t* slice[NG];  // AG: every GPU's own slice data (P u64, padded)
  const uint64_t* inputs[NG];  // RS pull: every GPU's full input
  uint64_t* buf[NG];        // push landing area on every GPU: NG * groups * 64 u64 (one 4-line group = 512 B)
  uint64_t* out;            // local output: AG 8 * P u64; RS P u64
  uint32_t* bar[NG];
  long long* times;
  unsigned long long* errs;
};

__device__ __forceinline__ uint32_t ldv32(const uint32_t* p) { return *(const volatile uint32_t*)p; }
__device__ __forceinline__ void stv32(uint32_t* p, uint32_t v) { *(volatile uint32_t*)p = v; }
__device__ __forceinline__ void st_v2(uint64_t* p, uint64_t a, uint64_t b) {
  asm volatile("st.volatile.global.v2.u64 [%0], {%1,%2};" ::"l"(p), "l"(a), "l"(b) : "memory");
}
__device__ __forceinline__ void ld_v2(const uint64_t* p, uint64_t& a, uint64_t& b) {
  asm volatile("ld.volatile.global.v2.u64 {%0,%1}, [%2];" : "=l"(a), "=l"(b) : "l"(p) : "memory");
}

// payload u64 index of (line, part, word) within a slice: 15 payload u64 per line
__device__ __forceinline__ int pidx(int line, int part, int w) { return line * 15 + part * 2 + w; }

__device__ __forceinline__ void add_h4(float* acc, uint64_t v) {
  __half2 h[2];
  memcpy(h, &v, 8);
  float2 a = __half22float2(h[0]), b = __half22float2(h[1]);
  acc[0] += a.x;
  acc[1] += a.y;
  acc[2] += b.x;
  acc[3] += b.y;
}
__device__ __forceinline__ uint64_t pack_h4(const float* acc) {
  __half2 h[2] = {__floats2half2_rn(acc[0], acc[1]), __floats2half2_rn(acc[2], acc[3])};
  uint64_t v;
  memcpy(&v, h, 8);
  return v;
}

// wait until all 4 lines of an LL128 group carry the epoch; returns this lane's 16 bytes
__device__ __forceinline__ bool poll_group(const uint64_t* g, int lane, uint64_t epoch, uint64_t& a, uint64_t& b,
                                           unsigned long long* errs) {
  const int part = lane & 7;
  for (long spin = 0;; ++spin) {
    ld_v2(g + 2 * lane, a, b);
    if (__all_sync(0xffffffffu, part != 7 || b == epoch)) return true;
    if (spin > SPIN_LIMIT) {
      if (lane == 0) atomicAdd(errs, 1ull);
      return false;
    }
  }
}

__global__ void __launch_bounds__(THREADS, 1) probe(Params p) {
  if (threadIdx.x < NG) {  // start barrier per block index (not timed)
    stv32(&p.bar[threadIdx.x][blockIdx.x * NG + p.rank], (uint32_t)p.epoch);
    while (ldv32(&p.bar[p.rank][blockIdx.x * NG + threadIdx.x]) != (uint32_t)p.epoch) {
    }
  }
  __syncthreads();
  const long long t0 = clock64();
  const int tid = blockIdx.x * blockDim.x + threadIdx.x, stride = gridDim.x * blockDim.x;
  const int lane = threadIdx.x & 31, part = lane & 7, sub = lane >> 3;
  const int gw = tid >> 5, nw = stride >> 5;
  const int r = p.rank, P = p.P;

  if (p.variant == 0) {  // ag_pull: 16-byte units, 7 peers in flight per thread
    const int U = P / 2;
    for (int i = tid; i < U; i += stride) {
      uint4 v[NG];
#pragma unroll
      for (int k = 0; k < NG; ++k) v[k] = ((const uint4*)p.slice[k])[i];
#pragma unroll
      for (int k = 0; k < NG; ++k) ((uint4*)(p.out + (size_t)k * P))[i] = v[k];
    }
  } else if (p.variant == 2) {  // rs_pull
    const int U = P / 2;
    for (int i = tid; i < U; i += stride) {
      uint4 v[NG];
#pragma unroll
      for (int k = 0; k < NG; ++k) v[k] = ((const uint4*)(p.inputs[k] + (size_t)r * P))[i];
      float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
      for (int k = 0; k < NG; ++k) {
        add_h4(acc, ((uint64_t*)&v[k])[0]);
        add_h4(acc + 4, ((uint64_t*)&v[k])[1]);
      }
      ((uint64_t*)p.out)[2 * i] = pack_h4(acc);
      ((uint64_t*)p.out)[2 * i + 1] = pack_h4(acc + 4);
    }
  } else if (p.variant == 1) {  // ag_push
    // send: one LL128 group (4 lines) per warp step; own slice also copied to the local output
    for (int g = gw; g < p.groups; g += nw) {
      const int line = g * 4 + sub;
      const uint64_t a = p.slice[r][pidx(line, part, 0)];
      const uint64_t b = part == 7 ? p.epoch : p.slice[r][pidx(line, part, 1)];
#pragma unroll
      for (int i = 1; i < NG; ++i) {
        const int k = (r + i) % NG;
        st_v2(p.buf[k] + ((size_t)r * p.groups + g) * 64 + 2 * lane, a, b);
      }
      if (pidx(line, part, 0) < P) p.out[(size_t)r * P + pidx(line, part, 0)] = a;
      if (part != 7 && pidx(line, part, 1) < P) p.out[(size_t)r * P + pidx(line, part, 1)] = b;
    }
    // receive: 7 sources x groups
    for (int idx = gw; idx < (NG - 1) * p.groups; idx += nw) {
      const int s = (r + 1 + idx / p.groups) % NG, g = idx % p.groups, line = g * 4 + sub;
      uint64_t a, b;
      poll_group(p.buf[r] + ((size_t)s * p.groups + g) * 64, lane, p.epoch, a, b, p.errs);
      if (pidx(line, part, 0) < P) p.out[(size_t)s * P + pidx(line, part, 0)] = a;
      if (part != 7 && pidx(line, part, 1) < P) p.out[(size_t)s * P + pidx(line, part, 1)] = b;
    }
  } else {  // rs_push
    for (int idx = gw; idx < NG * p.groups; idx += nw) {
      const int k = (r + 1 + idx / p.groups) % NG, g = idx % p.groups, line = g * 4 + sub;
      const uint64_t* src = p.in + (size_t)k * P;
      const uint64_t a = src[pidx(line, part, 0)];
      const uint64_t b = part == 7 ? p.epoch : src[pidx(line, part, 1)];
      st_v2(p.buf[k] + ((size_t)r * p.groups + g) * 64 + 2 * lane, a, b);
    }
    for (int g = gw; g < p.groups; g += nw) {
      const int line = g * 4 + sub;
      float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
      for (int s = 0; s < NG; ++s) {
        uint64_t a, b;
        poll_group(p.buf[r] + ((size_t)s * p.groups + g) * 64, lane, p.epoch, a, b, p.errs);
        add_h4(acc, a);
        if (part != 7) add_h4(acc + 4, b);
      }
      if (pidx(line, part, 0) < P) p.out[pidx(line, part, 0)] = pack_h4(acc);
      if (part != 7 && pidx(line, part, 1) < P) p.out[pidx(line, part, 1)] = pack_h4(acc + 4);
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) p.times[blockIdx.x] = clock64() - t0;
}

__global__ void calib(unsigned long long ns, long long* out) {
  unsigned long long g0, g1;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g0));
  long long c0 = clock64();
  do { asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g1)); } while (g1 - g0 < ns);
  out[0] = clock64() - c0;
  out[1] = (long long)(g1 - g0);
}

static double calibrate(int g) {
  long long* d;
  long long h[2];
  CK(cudaSetDevice(g));
  CK(cudaMalloc(&d, 16));
  calib<<<1, 1>>>(20000000ull, d);
  CK(cudaMemcpy(h, d, 16, cudaMemcpyDeviceToHost));
  CK(cudaFree(d));
  return (double)h[0] / h[1];
}

struct Gpu {
  cudaStream_t s;
  uint64_t *in, *slice, *buf, *out;
  uint32_t* bar;
  long long* times;
  unsigned long long* errs;
  double cyc_per_ns;
};

int main(int argc, char** argv) {
  const int seed = argc > 1 ? atoi(argv[1]) : 0;
  setvbuf(stdout, nullptr, _IONBF, 0);
  const int maxS = *std::max_element(SIZES.begin(), SIZES.end());
  const int maxP = maxS / 64, maxGroups = (maxP + 59) / 60 + 1;
  const size_t padP = (size_t)maxGroups * 60 + 64;  // payload u64 readable past P (zero padding)
  std::vector<Gpu> G(NG);
  for (int a = 0; a < NG; ++a) {
    CK(cudaSetDevice(a));
    for (int b = 0; b < NG; ++b)
      if (a != b) CK(cudaDeviceEnablePeerAccess(b, 0));
  }
  // host data: every GPU's full input (8 slices) and its own AG slice, as fp16 bit patterns
  std::vector<std::vector<uint64_t>> hin(NG, std::vector<uint64_t>(NG * padP, 0)), hslice(NG, std::vector<uint64_t>(padP, 0));
  for (int g = 0; g < NG; ++g) {
    std::mt19937 rng(1000 * seed + g);
    std::normal_distribution<float> nd(0.f, 1.f);
    auto rnd64 = [&] {
      __half h[4];
      for (auto& x : h) x = __float2half(nd(rng));
      uint64_t v;
      memcpy(&v, h, 8);
      return v;
    };
    for (int k = 0; k < NG; ++k)
      for (int i = 0; i < maxP; ++i) hin[g][(size_t)k * padP + i] = rnd64();
    for (int i = 0; i < maxP; ++i) hslice[g][i] = rnd64();
    CK(cudaSetDevice(g));
    CK(cudaStreamCreateWithFlags(&G[g].s, cudaStreamNonBlocking));
    CK(cudaMalloc(&G[g].in, NG * padP * 8));
    CK(cudaMalloc(&G[g].slice, padP * 8));
    CK(cudaMalloc(&G[g].buf, (size_t)NG * maxGroups * 64 * 8));
    CK(cudaMalloc(&G[g].out, NG * padP * 8));
    CK(cudaMalloc(&G[g].bar, MAXB * NG * 4));
    CK(cudaMemset(G[g].bar, 0, MAXB * NG * 4));
    CK(cudaMalloc(&G[g].times, MAXB * 8));
    CK(cudaMalloc(&G[g].errs, 8));
    CK(cudaMemset(G[g].errs, 0, 8));
    G[g].cyc_per_ns = calibrate(g);
    printf("calib,%d,%.5f\n", g, G[g].cyc_per_ns);
  }
  uint64_t epoch = 0;
  struct Cfg { int v, S, ctas; };
  std::vector<Cfg> cfgs;
  for (int v = 0; v < 4; ++v)
    for (int S : SIZES)
      for (int c : CTAS) cfgs.push_back({v, S, c});
  std::shuffle(cfgs.begin(), cfgs.end(), std::mt19937(seed));

  for (const Cfg& cf : cfgs) {
    const int P = cf.S / 64, groups = (P + 59) / 60;
    // stage this size's data: RS inputs use stride P between slices (contiguous message), AG slices as is
    for (int g = 0; g < NG; ++g) {
      std::vector<uint64_t> msg(NG * padP, 0);
      for (int k = 0; k < NG; ++k)
        for (int i = 0; i < P; ++i) msg[(size_t)k * P + i] = hin[g][(size_t)k * padP + i];
      CK(cudaSetDevice(g));
      CK(cudaMemcpy(G[g].in, msg.data(), NG * padP * 8, cudaMemcpyHostToDevice));
      CK(cudaMemcpy(G[g].slice, hslice[g].data(), padP * 8, cudaMemcpyHostToDevice));
    }
    // expected outputs
    std::vector<std::vector<uint64_t>> expect(NG);
    for (int g = 0; g < NG; ++g) {
      if (cf.v <= 1) {
        expect[g].resize((size_t)NG * P);
        for (int k = 0; k < NG; ++k)
          for (int i = 0; i < P; ++i) expect[g][(size_t)k * P + i] = hslice[k][i];
      } else {
        expect[g].resize(P);
        for (int i = 0; i < P; ++i) {
          float acc[4] = {0, 0, 0, 0};
          for (int s = 0; s < NG; ++s) {
            __half h[4];
            memcpy(h, &hin[s][(size_t)g * padP + i], 8);
            for (int j = 0; j < 4; ++j) acc[j] += __half2float(h[j]);
          }
          __half o[4];
          for (int j = 0; j < 4; ++j) o[j] = __float2half_rn(acc[j]);
          memcpy(&expect[g][i], o, 8);
        }
      }
    }
    unsigned long long violations = 0, timeouts = 0;
    std::vector<uint64_t> out;
    std::vector<long long> times(cf.ctas);
    for (int it = 0; it < WARM + ITERS; ++it) {
      ++epoch;
      for (int g = 0; g < NG; ++g) {
        CK(cudaSetDevice(g));
        CK(cudaMemsetAsync(G[g].buf, 0, (size_t)NG * groups * 64 * 8, G[g].s));
        CK(cudaMemsetAsync(G[g].out, 0, NG * padP * 8, G[g].s));
      }
      for (int g = 0; g < NG; ++g) {
        CK(cudaSetDevice(g));
        CK(cudaStreamSynchronize(G[g].s));
      }
      for (int g = 0; g < NG; ++g) {
        Params p{};
        p.rank = g;
        p.variant = cf.v;
        p.P = P;
        p.groups = groups;
        p.epoch = epoch;
        p.in = G[g].in;
        for (int k = 0; k < NG; ++k) {
          p.slice[k] = G[k].slice;
          p.inputs[k] = G[k].in;
          p.buf[k] = G[k].buf;
          p.bar[k] = G[k].bar;
        }
        p.out = G[g].out;
        p.times = G[g].times;
        p.errs = G[g].errs;
        CK(cudaSetDevice(g));
        probe<<<cf.ctas, THREADS, 0, G[g].s>>>(p);
        CK(cudaGetLastError());
      }
      for (int g = 0; g < NG; ++g) {
        CK(cudaSetDevice(g));
        CK(cudaStreamSynchronize(G[g].s));
      }
      if (it < WARM) continue;
      double us_max = 0;
      for (int g = 0; g < NG; ++g) {
        CK(cudaSetDevice(g));
        CK(cudaMemcpy(times.data(), G[g].times, cf.ctas * 8, cudaMemcpyDeviceToHost));
        us_max = std::max(us_max, *std::max_element(times.begin(), times.end()) / G[g].cyc_per_ns / 1000.0);
        out.resize(expect[g].size());
        CK(cudaMemcpy(out.data(), G[g].out, out.size() * 8, cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < out.size(); ++i) violations += out[i] != expect[g][i];
      }
      printf("it,%s,%d,%d,%d,%.3f\n", VNAME[cf.v], cf.S, cf.ctas, it - WARM, us_max);
    }
    for (int g = 0; g < NG; ++g) {
      unsigned long long nerr = 0;
      CK(cudaSetDevice(g));
      CK(cudaMemcpy(&nerr, G[g].errs, 8, cudaMemcpyDeviceToHost));
      timeouts += nerr;
      CK(cudaMemset(G[g].errs, 0, 8));
    }
    printf("sum,%s,%d,%d,%llu,%llu,%d,%d\n", VNAME[cf.v], cf.S, cf.ctas, violations, timeouts, P, groups);
  }
  for (int g = 0; g < NG; ++g) printf("calib_end,%d,%.5f\n", g, calibrate(g));
  return 0;
}
