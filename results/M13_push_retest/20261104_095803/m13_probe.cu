// M13: why is LL128 push slow for reduce-scatter, and does push still keep up with pull for large all-gathers when
// run back to back (steady state) instead of one shot with cleared buffers?
//
// Extends M12's ll128_probe.cu. Variants (0..3 as M12):
//   0 ag_pull      all-gather by 16-byte loads from the 8 slices, every GPU in order 0..7
//   1 ag_push      all-gather by LL128 lines (16-byte stores, 8-byte epoch flag per 128-byte line) to the 7 peers
//   2 rs_pull      reduce-scatter by 16-byte loads from the 8 inputs, fp32 sum in order 0..7
//   3 rs_push      reduce-scatter by LL128 lines into the owner's 8 slots; the owner waits slot by slot
//   4 ag_pull_rot  as 0, but GPU r reads in order r, r+1, ... (vLLM's stage-2 order)
//   5 rs_push_par  as 3, but the owner loads all 8 slots at once and retries until every flag matches
// Payload words are XORed with an epoch-dependent mask on push paths, so a line read with a fresh flag but stale data
// decodes to garbage and fails the bitwise check (stale data alone would otherwise look correct in steady mode).
//
// Modes:
//   single  every iteration: clear buffers, launch on all GPUs, start barrier (not timed), clock64 per block
//   steady  per GPU, K launches captured in a CUDA graph and replayed R times; no clearing, no start barrier; the epoch
//           comes from a per-block counter in device memory (as vLLM's per-block flags); push landing areas alternate
//           by epoch parity (every GPU receives from every GPU each epoch, so epoch e+2 is written only after the
//           receiver has finished epoch e); reports event time per op and the median in-kernel time (slowest block)
//
// Build: nvcc -O3 -arch=sm_80 -std=c++20 -o m13_probe m13_probe.cu
// Usage: ./m13_probe [seed]   -> CSV on stdout
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#define CK(x)                                                                      \
  do {                                                                             \
    cudaError_t err_ = (x);                                                        \
    if (err_ != cudaSuccess) {                                                     \
      fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(err_)); \
      exit(1);                                                                     \
    }                                                                              \
  } while (0)

constexpr int NG = 8, THREADS = 512, MAXB = 128, WARM = 5, ITERS = 50, K = 20, R = 20, CAP = 1024;
constexpr long SPIN_LIMIT = 1l << 22;
const std::vector<int> SIZES = {80 << 10, 160 << 10, 320 << 10, 640 << 10, 1280 << 10};
const std::vector<int> CTAS = {16, 32, 64, 108};
const char* VNAME[] = {"ag_pull", "ag_push", "rs_pull", "rs_push", "ag_pull_rot", "rs_push_par"};

struct Params {
  int rank, variant, steady;
  int P, groups;
  uint64_t epoch;                 // single mode only
  unsigned long long* ctr;        // steady mode: per-block epoch counters (local)
  const uint64_t* in;
  const uint64_t* slice[NG];
  const uint64_t* inputs[NG];
  uint64_t* buf[NG];              // landing areas: 2 halves x NG sources x groups x 64 u64
  uint64_t* out;
  uint32_t* bar[NG];
  long long* times;               // [CAP][MAXB] cycles
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
__device__ __forceinline__ int pidx(int line, int part, int w) { return line * 15 + part * 2 + w; }
__host__ __device__ __forceinline__ uint64_t emask(uint64_t e) { return e * 0x9E3779B97F4A7C15ull; }

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

__device__ __forceinline__ void poll_group(const uint64_t* g, int lane, uint64_t epoch, uint64_t& a, uint64_t& b,
                                           unsigned long long* errs) {
  const int part = lane & 7;
  for (long spin = 0;; ++spin) {
    ld_v2(g + 2 * lane, a, b);
    if (__all_sync(0xffffffffu, part != 7 || b == epoch)) return;
    if (spin > SPIN_LIMIT) {
      if (lane == 0) atomicAdd(errs, 1ull);
      return;
    }
  }
}

template <int V>
__global__ void __launch_bounds__(THREADS, 1) probe(Params p) {  // one instantiation per variant
  const uint64_t epoch = p.steady ? p.ctr[blockIdx.x] + 1 : p.epoch;
  if (!p.steady && threadIdx.x < NG) {  // start barrier (single mode only, not timed)
    stv32(&p.bar[threadIdx.x][blockIdx.x * NG + p.rank], (uint32_t)epoch);
    while (ldv32(&p.bar[p.rank][blockIdx.x * NG + threadIdx.x]) != (uint32_t)epoch) {
    }
  }
  __syncthreads();
  const long long t0 = clock64();
  const int tid = blockIdx.x * blockDim.x + threadIdx.x, stride = gridDim.x * blockDim.x;
  const int lane = threadIdx.x & 31, part = lane & 7, sub = lane >> 3;
  const int gw = tid >> 5, nw = stride >> 5;
  const int r = p.rank, P = p.P;
  constexpr int v = V;
  const uint64_t m = emask(epoch);
  const size_t off = (epoch & 1) * ((size_t)NG * p.groups * 64);  // landing half for this epoch

  if constexpr (v == 0 || v == 4) {  // ag_pull / ag_pull_rot
    const int U = P / 2;
    for (int i = tid; i < U; i += stride) {
      uint4 x[NG];
#pragma unroll
      for (int j = 0; j < NG; ++j) {
        const int k = v == 4 ? (r + j) % NG : j;
        x[j] = ((const uint4*)p.slice[k])[i];
      }
#pragma unroll
      for (int j = 0; j < NG; ++j) {
        const int k = v == 4 ? (r + j) % NG : j;
        ((uint4*)(p.out + (size_t)k * P))[i] = x[j];
      }
    }
  } else if constexpr (v == 2) {  // rs_pull
    const int U = P / 2;
    for (int i = tid; i < U; i += stride) {
      uint4 x[NG];
#pragma unroll
      for (int k = 0; k < NG; ++k) x[k] = ((const uint4*)(p.inputs[k] + (size_t)r * P))[i];
      float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
      for (int k = 0; k < NG; ++k) {
        add_h4(acc, ((uint64_t*)&x[k])[0]);
        add_h4(acc + 4, ((uint64_t*)&x[k])[1]);
      }
      ((uint64_t*)p.out)[2 * i] = pack_h4(acc);
      ((uint64_t*)p.out)[2 * i + 1] = pack_h4(acc + 4);
    }
  } else if constexpr (v == 1) {  // ag_push
    for (int g = gw; g < p.groups; g += nw) {
      const int line = g * 4 + sub;
      const uint64_t a0 = p.slice[r][pidx(line, part, 0)], b0 = p.slice[r][pidx(line, part, 1)];
      const uint64_t a = a0 ^ m, b = part == 7 ? epoch : b0 ^ m;
#pragma unroll
      for (int i = 1; i < NG; ++i) {
        const int k = (r + i) % NG;
        st_v2(p.buf[k] + off + ((size_t)r * p.groups + g) * 64 + 2 * lane, a, b);
      }
      if (pidx(line, part, 0) < P) p.out[(size_t)r * P + pidx(line, part, 0)] = a0;
      if (part != 7 && pidx(line, part, 1) < P) p.out[(size_t)r * P + pidx(line, part, 1)] = b0;
    }
    for (int idx = gw; idx < (NG - 1) * p.groups; idx += nw) {
      const int s = (r + 1 + idx / p.groups) % NG, g = idx % p.groups, line = g * 4 + sub;
      uint64_t a, b;
      poll_group(p.buf[r] + off + ((size_t)s * p.groups + g) * 64, lane, epoch, a, b, p.errs);
      if (pidx(line, part, 0) < P) p.out[(size_t)s * P + pidx(line, part, 0)] = a ^ m;
      if (part != 7 && pidx(line, part, 1) < P) p.out[(size_t)s * P + pidx(line, part, 1)] = b ^ m;
    }
  } else {  // rs_push (3) / rs_push_par (5)
    for (int idx = gw; idx < NG * p.groups; idx += nw) {
      const int k = (r + 1 + idx / p.groups) % NG, g = idx % p.groups, line = g * 4 + sub;
      const uint64_t* src = p.in + (size_t)k * P;
      const uint64_t a = src[pidx(line, part, 0)] ^ m, b = part == 7 ? epoch : src[pidx(line, part, 1)] ^ m;
      st_v2(p.buf[k] + off + ((size_t)r * p.groups + g) * 64 + 2 * lane, a, b);
    }
    for (int g = gw; g < p.groups; g += nw) {
      const int line = g * 4 + sub;
      float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
      uint64_t a[NG], b[NG];
      const uint64_t* mine = p.buf[r] + off + (size_t)g * 64 + 2 * lane;
      if constexpr (v == 3) {
#pragma unroll
        for (int s = 0; s < NG; ++s) poll_group(mine - 2 * lane + (size_t)s * p.groups * 64, lane, epoch, a[s], b[s], p.errs);
      } else {
        for (long spin = 0;; ++spin) {
          bool ok = true;
#pragma unroll
          for (int s = 0; s < NG; ++s) {
            ld_v2(mine + (size_t)s * p.groups * 64, a[s], b[s]);
            ok &= part != 7 || b[s] == epoch;
          }
          if (__all_sync(0xffffffffu, ok)) break;
          if (spin > SPIN_LIMIT) {
            if (lane == 0) atomicAdd(p.errs, 1ull);
            break;
          }
        }
      }
#pragma unroll
      for (int s = 0; s < NG; ++s) {
        add_h4(acc, a[s] ^ m);
        if (part != 7) add_h4(acc + 4, b[s] ^ m);
      }
      if (pidx(line, part, 0) < P) p.out[pidx(line, part, 0)] = pack_h4(acc);
      if (part != 7 && pidx(line, part, 1) < P) p.out[pidx(line, part, 1)] = pack_h4(acc + 4);
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    p.times[(epoch % CAP) * MAXB + blockIdx.x] = clock64() - t0;
    if (p.steady) p.ctr[blockIdx.x] = epoch;
  }
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

static void launch(int v, int ctas, cudaStream_t s, const Params& p) {
  switch (v) {
    case 0: probe<0><<<ctas, THREADS, 0, s>>>(p); break;
    case 1: probe<1><<<ctas, THREADS, 0, s>>>(p); break;
    case 2: probe<2><<<ctas, THREADS, 0, s>>>(p); break;
    case 3: probe<3><<<ctas, THREADS, 0, s>>>(p); break;
    case 4: probe<4><<<ctas, THREADS, 0, s>>>(p); break;
    default: probe<5><<<ctas, THREADS, 0, s>>>(p); break;
  }
}

struct Gpu {
  cudaStream_t s;
  uint64_t *in, *slice, *buf, *out;
  uint32_t* bar;
  long long* times;
  unsigned long long *errs, *ctr;
  double cyc_per_ns;
};

int main(int argc, char** argv) {
  const int seed = argc > 1 ? atoi(argv[1]) : 0;
  setvbuf(stdout, nullptr, _IONBF, 0);
  const int maxS = *std::max_element(SIZES.begin(), SIZES.end());
  const int maxP = maxS / 64, maxGroups = (maxP + 59) / 60 + 1;
  const size_t padP = (size_t)maxGroups * 60 + 64;
  std::vector<Gpu> G(NG);
  for (int a = 0; a < NG; ++a) {
    CK(cudaSetDevice(a));
    for (int b = 0; b < NG; ++b)
      if (a != b) CK(cudaDeviceEnablePeerAccess(b, 0));
  }
  std::vector<std::vector<uint64_t>> hin(NG, std::vector<uint64_t>(NG * padP, 0)),
      hslice(NG, std::vector<uint64_t>(padP, 0));
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
    CK(cudaMalloc(&G[g].buf, 2ull * NG * maxGroups * 64 * 8));
    CK(cudaMalloc(&G[g].out, NG * padP * 8));
    CK(cudaMalloc(&G[g].bar, MAXB * NG * 4));
    CK(cudaMemset(G[g].bar, 0, MAXB * NG * 4));
    CK(cudaMalloc(&G[g].times, (size_t)CAP * MAXB * 8));
    CK(cudaMalloc(&G[g].errs, 8));
    CK(cudaMemset(G[g].errs, 0, 8));
    CK(cudaMalloc(&G[g].ctr, MAXB * 8));
    G[g].cyc_per_ns = calibrate(g);
    printf("calib,%d,%.5f\n", g, G[g].cyc_per_ns);
  }
  uint64_t epoch = 0;
  uint32_t bar_epoch = 0;  // single-mode barrier flags keep increasing across configs
  struct Cfg { int mode, v, S, ctas; };
  std::vector<Cfg> cfgs;
  for (int mode = 0; mode < 2; ++mode)
    for (int v = 0; v < 6; ++v)
      for (int S : SIZES)
        for (int c : CTAS) cfgs.push_back({mode, v, S, c});
  std::shuffle(cfgs.begin(), cfgs.end(), std::mt19937(seed));
  const char* MODE[] = {"single", "steady"};

  for (const Cfg& cf : cfgs) {
    const int P = cf.S / 64, groups = (P + 59) / 60;
    for (int g = 0; g < NG; ++g) {
      std::vector<uint64_t> msg(NG * padP, 0);
      for (int k = 0; k < NG; ++k)
        for (int i = 0; i < P; ++i) msg[(size_t)k * P + i] = hin[g][(size_t)k * padP + i];
      CK(cudaSetDevice(g));
      CK(cudaMemcpy(G[g].in, msg.data(), NG * padP * 8, cudaMemcpyHostToDevice));
      CK(cudaMemcpy(G[g].slice, hslice[g].data(), padP * 8, cudaMemcpyHostToDevice));
      CK(cudaMemset(G[g].buf, 0, 2ull * NG * groups * 64 * 8));
      CK(cudaMemset(G[g].ctr, 0, MAXB * 8));
    }
    const bool is_ag = cf.v == 0 || cf.v == 1 || cf.v == 4;
    std::vector<std::vector<uint64_t>> expect(NG);
    for (int g = 0; g < NG; ++g) {
      if (is_ag) {
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
    auto params = [&](int g) {
      Params p{};
      p.rank = g;
      p.variant = cf.v;
      p.steady = cf.mode;
      p.P = P;
      p.groups = groups;
      p.ctr = G[g].ctr;
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
      return p;
    };
    auto sync_all = [&] {
      for (int g = 0; g < NG; ++g) {
        CK(cudaSetDevice(g));
        CK(cudaStreamSynchronize(G[g].s));
      }
    };
    unsigned long long violations = 0, timeouts = 0;
    std::vector<uint64_t> out;
    auto check = [&] {
      for (int g = 0; g < NG; ++g) {
        CK(cudaSetDevice(g));
        out.resize(expect[g].size());
        CK(cudaMemcpy(out.data(), G[g].out, out.size() * 8, cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < out.size(); ++i) violations += out[i] != expect[g][i];
      }
    };
    std::vector<long long> tbuf((size_t)CAP * MAXB);
    if (cf.mode == 0) {
      for (int it = 0; it < WARM + ITERS; ++it) {
        ++epoch;
        ++bar_epoch;
        for (int g = 0; g < NG; ++g) {
          CK(cudaSetDevice(g));
          CK(cudaMemsetAsync(G[g].buf, 0, 2ull * NG * groups * 64 * 8, G[g].s));
          CK(cudaMemsetAsync(G[g].out, 0, NG * padP * 8, G[g].s));
        }
        sync_all();
        for (int g = 0; g < NG; ++g) {
          Params p = params(g);
          p.epoch = bar_epoch;  // single mode: buffers are cleared, so the barrier counter doubles as the epoch
          CK(cudaSetDevice(g));
          launch(cf.v, cf.ctas, G[g].s, p);
          CK(cudaGetLastError());
        }
        sync_all();
        if (it < WARM) continue;
        double us_max = 0;
        for (int g = 0; g < NG; ++g) {
          CK(cudaSetDevice(g));
          CK(cudaMemcpy(tbuf.data(), G[g].times, (size_t)CAP * MAXB * 8, cudaMemcpyDeviceToHost));
          const long long* row = &tbuf[(bar_epoch % CAP) * MAXB];
          us_max = std::max(us_max, *std::max_element(row, row + cf.ctas) / G[g].cyc_per_ns / 1000.0);
        }
        check();
        printf("it,%s,%s,%d,%d,%d,%.3f\n", MODE[cf.mode], VNAME[cf.v], cf.S, cf.ctas, it - WARM, us_max);
      }
    } else {
      std::vector<cudaGraphExec_t> exec(NG);
      for (int g = 0; g < NG; ++g) {
        Params p = params(g);
        CK(cudaSetDevice(g));
        cudaGraph_t graph;
        CK(cudaStreamBeginCapture(G[g].s, cudaStreamCaptureModeRelaxed));
        for (int k = 0; k < K; ++k) launch(cf.v, cf.ctas, G[g].s, p);
        CK(cudaStreamEndCapture(G[g].s, &graph));
        CK(cudaGraphInstantiate(&exec[g], graph, 0));
        CK(cudaGraphDestroy(graph));
      }
      auto replay = [&](int n) {
        for (int i = 0; i < n; ++i)
          for (int g = 0; g < NG; ++g) {
            CK(cudaSetDevice(g));
            CK(cudaGraphLaunch(exec[g], G[g].s));
          }
      };
      replay(2);
      sync_all();
      const int e0 = 2 * K;  // epochs 1..2K were warm-up
      std::vector<cudaEvent_t> a(NG), b(NG);
      for (int g = 0; g < NG; ++g) {
        CK(cudaSetDevice(g));
        CK(cudaEventCreate(&a[g]));
        CK(cudaEventCreate(&b[g]));
        CK(cudaEventRecord(a[g], G[g].s));
      }
      replay(R);
      for (int g = 0; g < NG; ++g) {
        CK(cudaSetDevice(g));
        CK(cudaEventRecord(b[g], G[g].s));
      }
      sync_all();
      for (int g = 0; g < NG; ++g) {
        float ms;
        CK(cudaEventElapsedTime(&ms, a[g], b[g]));
        CK(cudaSetDevice(g));
        CK(cudaMemcpy(tbuf.data(), G[g].times, (size_t)CAP * MAXB * 8, cudaMemcpyDeviceToHost));
        std::vector<double> kern;
        for (int e = e0 + 1; e <= e0 + R * K; ++e) {
          const long long* row = &tbuf[(e % CAP) * MAXB];
          kern.push_back(*std::max_element(row, row + cf.ctas) / G[g].cyc_per_ns / 1000.0);
        }
        std::sort(kern.begin(), kern.end());
        printf("st,%s,%d,%d,%d,%.3f,%.3f,%.3f\n", VNAME[cf.v], cf.S, cf.ctas, g, ms * 1000.0 / (R * K),
               kern[kern.size() / 2], kern[kern.size() * 9 / 10]);
        CK(cudaEventDestroy(a[g]));
        CK(cudaEventDestroy(b[g]));
      }
      check();  // last epoch's output
      for (int g = 0; g < NG; ++g) CK(cudaGraphExecDestroy(exec[g]));
    }
    for (int g = 0; g < NG; ++g) {
      unsigned long long nerr = 0;
      CK(cudaSetDevice(g));
      CK(cudaMemcpy(&nerr, G[g].errs, 8, cudaMemcpyDeviceToHost));
      timeouts += nerr;
      CK(cudaMemset(G[g].errs, 0, 8));
    }
    printf("sum,%s,%s,%d,%d,%llu,%llu\n", MODE[cf.mode], VNAME[cf.v], cf.S, cf.ctas, violations, timeouts);
  }
  for (int g = 0; g < NG; ++g) printf("calib_end,%d,%.5f\n", g, calibrate(g));
  return 0;
}
