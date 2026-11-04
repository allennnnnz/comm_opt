// M11 part A: 8-to-8 reduce-scatter on 8x A100 by cross-GPU atomic adds (with a counter in every 128-byte line),
// versus a store-based backup ("each GPU writes its own slot, the owner sums").
//
// Every GPU's fp16 input (n elements) is cut into 128-byte lines; line block k is owned by GPU k. Variants:
//   0 atom_f16   one warp-wide red.global.add.noftz.f16x2: lane 0 adds (1,1) to the counter word, lanes 1..31 data
//   1 atom_bf16  atomicAdd(__nv_bfloat162*) (input converted from fp16); sm_80 has no bf16x2 atomic, so CUDA emulates it
//                with a per-lane atomicCAS loop (cuda_bf16.hpp) -- kept as is: that is what A100 offers
//   2 atom_f32   same with float (31 floats per line; twice the bytes on the wire)
//   3 store_slot one warp-wide 4-byte store per lane into slot[rank] of the owner: lane 0 = epoch, lanes 1..31 = data;
//                the owner waits for all 8 slots' epoch and sums them in fp32 in slot order 0..7 (deterministic)
// The counter rides in the same warp instruction as the data, so "counter == 8 but data not complete" would mean the
// line is not applied atomically (M8 tested this for uint32, 7 writers, 1 reader).
//
// Per iteration: buffers cleared, all GPUs launched, a start barrier aligns the blocks (not timed), then each block
// times (clock64) sending its share of all 8 owners' lines plus waiting for and snapshotting its share of its own
// lines. Reported time = max over blocks and GPUs, cycles converted with a per-GPU calibration (clock64 vs globaltimer).
//
// Build: nvcc -O3 -arch=sm_80 -std=c++20 -o rs_probe rs_probe.cu
// Usage: ./rs_probe [seed]                 -> CSV on stdout
//        ./rs_probe metrics <ms> [seed]    -> atom_f16 and store_slot at 64 blocks, each size looped for <ms> in NVTX
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#define CK(x)                                                                              \
  do {                                                                                     \
    cudaError_t e = (x);                                                                   \
    if (e != cudaSuccess) {                                                                \
      fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e));           \
      exit(1);                                                                             \
    }                                                                                      \
  } while (0)

constexpr int NG = 8, THREADS = 512, MAXB = 128, WARM = 5, ITERS = 50;
constexpr long SPIN_LIMIT = 1l << 22;
const std::vector<int> SIZES = {80 << 10, 160 << 10, 320 << 10, 640 << 10, 1280 << 10};
const std::vector<int> CTAS = {16, 32, 64, 108};
const char* VNAME[] = {"atom_f16", "atom_bf16", "atom_f32", "store_slot"};

struct Params {
  int rank, variant, lper;
  uint32_t epoch;
  const __half* in;     // local input, n_pad elements
  uint32_t* acc[NG];    // owner accumulation lines (atomic variants), lper * 32 words each
  uint32_t* slot[NG];   // owner slots (store_slot), NG * lper * 32 words each
  uint32_t* bar[NG];    // start-barrier flags, MAXB * NG words each
  uint32_t* snap;       // local: line contents seen when complete, lper * 32 words
  long long* times;     // local: cycles per block
  unsigned long long* errs;  // local: timeouts
};

__device__ __forceinline__ uint32_t ldv(const uint32_t* p) { return *(const volatile uint32_t*)p; }
__device__ __forceinline__ void stv(uint32_t* p, uint32_t v) { *(volatile uint32_t*)p = v; }

__host__ __device__ inline uint32_t bits_h2(__half2 h) { uint32_t u; memcpy(&u, &h, 4); return u; }
__host__ __device__ inline uint32_t bits_b2(__nv_bfloat162 h) { uint32_t u; memcpy(&u, &h, 4); return u; }
__host__ __device__ inline uint32_t bits_f(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }

__global__ void __launch_bounds__(THREADS, 1) rs_kernel(Params p) {
  if (threadIdx.x < NG) {  // start barrier per block index, as vLLM's barrier_at_start
    stv(&p.bar[threadIdx.x][blockIdx.x * NG + p.rank], p.epoch);
    while (ldv(&p.bar[p.rank][blockIdx.x * NG + threadIdx.x]) != p.epoch) {
    }
  }
  __syncthreads();
  const long long t0 = clock64();
  const int lane = threadIdx.x & 31, wpb = THREADS / 32;
  const int gw = blockIdx.x * wpb + (threadIdx.x >> 5), nw = gridDim.x * wpb;
  const __half2* in2 = (const __half2*)p.in;
  // phase 1: every GPU sends its 8 line blocks to their owners (start with the next GPU to spread the load)
  for (int idx = gw; idx < NG * p.lper; idx += nw) {
    const int k = (p.rank + 1 + idx / p.lper) % NG, j = idx % p.lper;
    const size_t gl = (size_t)k * p.lper + j;  // global line index
    if (p.variant == 2) {
      const float v = lane ? __half2float(p.in[gl * 31 + lane - 1]) : 1.f;
      atomicAdd((float*)(p.acc[k] + (size_t)j * 32 + lane), v);
      continue;
    }
    const __half2 x = lane ? in2[gl * 31 + lane - 1] : __halves2half2(__float2half(1.f), __float2half(1.f));
    if (p.variant == 0) {  // fire-and-forget red (atomicAdd(__half2*) compiles to ATOM, which waits for a reply)
      asm volatile("red.global.add.noftz.f16x2 [%0], %1;" ::"l"(p.acc[k] + (size_t)j * 32 + lane), "r"(bits_h2(x))
                   : "memory");
    } else if (p.variant == 1) {
      atomicAdd((__nv_bfloat162*)(p.acc[k] + (size_t)j * 32 + lane), __float22bfloat162_rn(__half22float2(x)));
    } else {
      stv(p.slot[k] + ((size_t)p.rank * p.lper + j) * 32 + lane, lane ? bits_h2(x) : p.epoch);
    }
  }
  // phase 2: wait for the lines this GPU owns, snapshot them
  const uint32_t target = p.variant == 0   ? bits_h2(__halves2half2(__float2half(8.f), __float2half(8.f)))
                          : p.variant == 1 ? bits_b2(__floats2bfloat162_rn(8.f, 8.f))
                                           : bits_f(8.f);
  for (int j = gw; j < p.lper; j += nw) {
    if (p.variant < 3) {
      const uint32_t* line = p.acc[p.rank] + (size_t)j * 32;
      uint32_t v;
      for (long spin = 0;; ++spin) {
        v = ldv(line + lane);  // one warp-wide load of the whole line
        if (__shfl_sync(0xffffffffu, v, 0) == target) break;
        if (spin > SPIN_LIMIT) {
          if (lane == 0) atomicAdd(p.errs, 1ull);
          break;
        }
      }
      p.snap[(size_t)j * 32 + lane] = v;
    } else {
      float2 acc = make_float2(0.f, 0.f);
      for (int s = 0; s < NG; ++s) {
        const uint32_t* line = p.slot[p.rank] + ((size_t)s * p.lper + j) * 32;
        uint32_t v;
        for (long spin = 0;; ++spin) {
          v = ldv(line + lane);
          if (__shfl_sync(0xffffffffu, v, 0) == p.epoch) break;
          if (spin > SPIN_LIMIT) {
            if (lane == 0) atomicAdd(p.errs, 1ull);
            break;
          }
        }
        __half2 h;
        memcpy(&h, &v, 4);
        const float2 f = __half22float2(h);
        acc.x += f.x;
        acc.y += f.y;
      }
      p.snap[(size_t)j * 32 + lane] = lane ? bits_h2(__float22half2_rn(acc)) : p.epoch;
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

struct Gpu {
  cudaStream_t s;
  __half* in;
  uint32_t *acc, *slot, *bar, *snap;
  long long* times;
  unsigned long long* errs;
  double cyc_per_ns;
};

static float host_value(int variant, uint32_t w, int half_idx) {  // decode one element from a 4-byte data word
  if (variant == 0) { __half2 h; memcpy(&h, &w, 4); return __half2float(half_idx ? h.y : h.x); }
  if (variant == 1) { __nv_bfloat162 b; memcpy(&b, &w, 4); return __bfloat162float(half_idx ? b.y : b.x); }
  if (variant == 2) { float f; memcpy(&f, &w, 4); return f; }
  __half2 h; memcpy(&h, &w, 4); return __half2float(half_idx ? h.y : h.x);
}

int main(int argc, char** argv) {
  const bool metrics = argc > 1 && std::string(argv[1]) == "metrics";
  const int metrics_ms = metrics && argc > 2 ? atoi(argv[2]) : 0;
  const int seed = argc > (metrics ? 3 : 1) ? atoi(argv[metrics ? 3 : 1]) : 0;
  setvbuf(stdout, nullptr, _IONBF, 0);
  const int max_n = *std::max_element(SIZES.begin(), SIZES.end()) / 2;
  const int max_lper = (max_n / 31 + NG) / NG + 1;           // generous (31 elements per line for f32)
  const size_t n_pad_max = (size_t)NG * max_lper * 62;       // input elements read at most
  std::vector<Gpu> G(NG);
  for (int a = 0; a < NG; ++a) {
    CK(cudaSetDevice(a));
    for (int b = 0; b < NG; ++b)
      if (a != b) CK(cudaDeviceEnablePeerAccess(b, 0));
  }
  std::vector<std::vector<__half>> hin(NG, std::vector<__half>(n_pad_max));
  for (int g = 0; g < NG; ++g) {
    std::mt19937 rng(1000 * seed + g);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto& x : hin[g]) x = __float2half(nd(rng));
    CK(cudaSetDevice(g));
    CK(cudaStreamCreateWithFlags(&G[g].s, cudaStreamNonBlocking));
    CK(cudaMalloc(&G[g].in, n_pad_max * 2));
    CK(cudaMemcpy(G[g].in, hin[g].data(), n_pad_max * 2, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&G[g].acc, (size_t)max_lper * 128));
    CK(cudaMalloc(&G[g].slot, (size_t)NG * max_lper * 128));
    CK(cudaMalloc(&G[g].bar, MAXB * NG * 4));
    CK(cudaMemset(G[g].bar, 0, MAXB * NG * 4));
    CK(cudaMalloc(&G[g].snap, (size_t)max_lper * 128));
    CK(cudaMalloc(&G[g].times, MAXB * 8));
    CK(cudaMalloc(&G[g].errs, 8));
    CK(cudaMemset(G[g].errs, 0, 8));
    long long* d;
    long long h[2];
    CK(cudaMalloc(&d, 16));
    calib<<<1, 1>>>(20000000ull, d);
    CK(cudaMemcpy(h, d, 16, cudaMemcpyDeviceToHost));
    CK(cudaFree(d));
    G[g].cyc_per_ns = (double)h[0] / h[1];
    if (!metrics) printf("calib,%d,%.5f\n", g, G[g].cyc_per_ns);
  }
  uint32_t epoch = 0;

  struct Cfg { int v, bytes, ctas; };
  std::vector<Cfg> cfgs;
  if (metrics) {
    for (int v : {0, 3})
      for (int b : SIZES) cfgs.push_back({v, b, 64});
  } else {
    for (int v = 0; v < 4; ++v)
      for (int b : SIZES)
        for (int c : CTAS) cfgs.push_back({v, b, c});
    std::shuffle(cfgs.begin(), cfgs.end(), std::mt19937(seed));
  }

  for (const Cfg& cf : cfgs) {
    const int n = cf.bytes / 2, E = cf.v == 2 ? 31 : 62;
    const int lper = ((n + E - 1) / E + NG - 1) / NG;
    const size_t lines = (size_t)NG * lper;
    // exact reference over all elements touched (lines * E), and the store_slot replica (fp32, slot order 0..7)
    std::vector<double> exact(lines * E);
    std::vector<__half> replica(lines * E);
    double max_ref = 0;
    for (size_t i = 0; i < lines * E; ++i) {
      double s = 0;
      float f = 0.f;
      for (int r = 0; r < NG; ++r) {
        s += (double)__half2float(hin[r][i]);
        f += __half2float(hin[r][i]);
      }
      exact[i] = s;
      replica[i] = __float2half_rn(f);
      max_ref = std::max(max_ref, std::fabs(s));
    }
    auto run_once = [&]() {
      ++epoch;
      for (int g = 0; g < NG; ++g) {
        CK(cudaSetDevice(g));
        CK(cudaMemsetAsync(G[g].acc, 0, (size_t)lper * 128, G[g].s));
        if (cf.v == 3) CK(cudaMemsetAsync(G[g].slot, 0, (size_t)NG * lper * 128, G[g].s));
      }
      for (int g = 0; g < NG; ++g) {
        CK(cudaSetDevice(g));
        CK(cudaStreamSynchronize(G[g].s));
      }
      for (int g = 0; g < NG; ++g) {
        Params p{};
        p.rank = g;
        p.variant = cf.v;
        p.lper = lper;
        p.epoch = epoch;
        p.in = G[g].in;
        for (int k = 0; k < NG; ++k) {
          p.acc[k] = G[k].acc;
          p.slot[k] = G[k].slot;
          p.bar[k] = G[k].bar;
        }
        p.snap = G[g].snap;
        p.times = G[g].times;
        p.errs = G[g].errs;
        CK(cudaSetDevice(g));
        rs_kernel<<<cf.ctas, THREADS, 0, G[g].s>>>(p);
        CK(cudaGetLastError());
      }
      for (int g = 0; g < NG; ++g) {
        CK(cudaSetDevice(g));
        CK(cudaStreamSynchronize(G[g].s));
      }
    };
    if (metrics) {
      char name[64];
      snprintf(name, sizeof name, "rs_%s_%dK", VNAME[cf.v], cf.bytes >> 10);
      nvtxRangePushA(name);
      auto t0 = std::chrono::steady_clock::now();
      while (std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() < metrics_ms)
        run_once();
      nvtxRangePop();
      continue;
    }
    unsigned long long violations = 0, diff_words = 0, data_words = 0, timeouts = 0;
    double max_err = 0;
    std::vector<std::vector<uint32_t>> first(NG);
    std::vector<uint32_t> snap((size_t)lper * 32), fin((size_t)lper * 32);
    std::vector<long long> times(cf.ctas);
    for (int it = 0; it < WARM + ITERS; ++it) {
      run_once();
      if (it < WARM) continue;
      double us_max = 0;
      for (int g = 0; g < NG; ++g) {
        CK(cudaSetDevice(g));
        CK(cudaMemcpy(times.data(), G[g].times, cf.ctas * 8, cudaMemcpyDeviceToHost));
        const double us = *std::max_element(times.begin(), times.end()) / G[g].cyc_per_ns / 1000.0;
        us_max = std::max(us_max, us);
        CK(cudaMemcpy(snap.data(), G[g].snap, snap.size() * 4, cudaMemcpyDeviceToHost));
        if (cf.v < 3) CK(cudaMemcpy(fin.data(), G[g].acc, fin.size() * 4, cudaMemcpyDeviceToHost));
        for (int j = 0; j < lper; ++j) {
          const size_t gl = (size_t)g * lper + j;
          bool bad = false;
          for (int w = 1; w < 32; ++w) {
            const uint32_t sw = snap[(size_t)j * 32 + w];
            if (cf.v < 3) {
              bad |= sw != fin[(size_t)j * 32 + w];
            } else {
              const __half2 r2 = __halves2half2(replica[gl * E + 2 * (w - 1)], replica[gl * E + 2 * (w - 1) + 1]);
              bad |= sw != bits_h2(r2);
            }
            const int per_word = cf.v == 2 ? 1 : 2;
            for (int h = 0; h < per_word; ++h) {
              const size_t e = gl * E + (size_t)(w - 1) * per_word + h;
              max_err = std::max(max_err, std::fabs((double)host_value(cf.v, sw, h) - exact[e]));
            }
            // determinism: compare with the first timed iteration (f32 compared after rounding to fp16)
            const uint32_t cmp = cf.v == 2 ? (uint32_t)__half_as_ushort(__float2half_rn(host_value(2, sw, 0))) : sw;
            if (it == WARM) first[g].push_back(cmp);
            else diff_words += cmp != first[g][(size_t)j * 31 + w - 1];
            ++data_words;
          }
          violations += bad;
        }
      }
      printf("it,%s,%d,%d,%d,%.3f\n", VNAME[cf.v], cf.bytes, cf.ctas, it - WARM, us_max);
    }
    for (int g = 0; g < NG; ++g) {
      unsigned long long nerr = 0;
      CK(cudaSetDevice(g));
      CK(cudaMemcpy(&nerr, G[g].errs, 8, cudaMemcpyDeviceToHost));
      timeouts += nerr;
      CK(cudaMemset(G[g].errs, 0, 8));
    }
    const unsigned long long compared = data_words - (unsigned long long)NG * lper * 31;  // excludes first iteration
    printf("sum,%s,%d,%d,%llu,%llu,%.4e,%.6f,%d\n", VNAME[cf.v], cf.bytes, cf.ctas, violations, timeouts,
           max_err / max_ref, compared ? (double)diff_words / compared : 0.0, lper);
  }
  if (!metrics)
    for (int g = 0; g < NG; ++g) {
      long long* d;
      long long h[2];
      CK(cudaSetDevice(g));
      CK(cudaMalloc(&d, 16));
      calib<<<1, 1>>>(20000000ull, d);
      CK(cudaMemcpy(h, d, 16, cudaMemcpyDeviceToHost));
      CK(cudaFree(d));
      printf("calib_end,%d,%.5f\n", g, (double)h[0] / h[1]);
    }
  return 0;
}
