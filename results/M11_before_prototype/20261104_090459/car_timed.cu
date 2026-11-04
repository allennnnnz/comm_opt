// M11 part B/C: where does vLLM's custom all-reduce spend its time on 8x A100?
//
// The two kernels below restate vLLM v0.33.0's cross_device_reduce_1stage / 2stage (vllm_src/custom_all_reduce.cuh)
// line for line; the only additions are clock64 stamps at phase boundaries, guarded by NO_STAMPS. Stamping needs the
// whole block past a point, so the stamped build adds a __syncthreads() before a stamp where vLLM has none (after
// stage loops and after the final barrier of 1-stage). The plain build (-DNO_STAMPS) is the unmodified kernel; M11
// compares the two builds' per-op times to bound the stamping overhead.
//
// One process, 8 GPUs, peer access (vLLM uses one process per GPU + CUDA IPC; same NVLink path). Each GPU captures
// K back-to-back all-reduces into a CUDA graph; R replays are timed with events (as M6).
//
// Build:  nvcc -O3 -arch=sm_80 -std=c++20 -o car_timed car_timed.cu            (stamped)
//         nvcc -O3 -arch=sm_80 -std=c++20 -DNO_STAMPS -o car_plain car_timed.cu (plain)
// Usage:  ./car_timed [seed]                -> CSV on stdout
//         ./car_plain metrics <ms> [seed]   -> each config replayed for <ms> inside an NVTX range (for nsys GPU metrics)
#include "vllm_src/custom_collective_common.cuh"

#include <nvtx3/nvToolsExt.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

using namespace vllm;

constexpr int NG = 8, NSTAMP = 5, K = 20, R = 20, MAX_INST = 1024, THREADS = 512;
const std::vector<int> SIZES = {80 << 10, 160 << 10, 320 << 10, 640 << 10, 1280 << 10};

#define CK(x) CUDACHECK(x)

#ifdef NO_STAMPS
#define STAMP(i)
#define STAMP_SYNC()
#define STAMP_SAVE()
#else
#define STAMP(i) \
  if (threadIdx.x == 0) st[i] = clock64();
#define STAMP_SYNC() __syncthreads();
#define STAMP_SAVE()                                                                    \
  if (threadIdx.x == 0) {                                                               \
    long long* o = stamps + ((size_t)(inst % MAX_INST) * kMaxBlocks + blockIdx.x) * NSTAMP; \
    for (int i = 0; i < NSTAMP; ++i) o[i] = st[i];                                      \
  }
#endif

template <typename T, int ngpus>
__global__ void __launch_bounds__(512, 1)
    car_1stage(RankData* _dp, RankSignals sg, Signal* self_sg, T* __restrict__ result, int rank, int size,
               long long* stamps) {
#ifndef NO_STAMPS
  long long st[NSTAMP];
  const uint32_t inst = self_sg->_flag[blockIdx.x] / 2;  // each call passes two barriers
#endif
  STAMP(0)
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  auto dp = *_dp;
  barrier_at_start<ngpus>(sg, self_sg, rank);
  STAMP(1)
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size; idx += gridDim.x * blockDim.x) {
    ((P*)result)[idx] = packed_reduce<P, ngpus, A>((const P**)&dp.ptrs[0], idx);
  }
  STAMP_SYNC()
  STAMP(2)
  barrier_at_end<ngpus, true>(sg, self_sg, rank);
  STAMP_SYNC()
  STAMP(3)
  STAMP(4)
  STAMP_SAVE()
}

template <typename P>
DINLINE P* tmp_buf(Signal* sg) {
  return (P*)(((Signal*)sg) + 1);
}

template <typename T, int ngpus>
__global__ void __launch_bounds__(512, 1)
    car_2stage(RankData* _dp, RankSignals sg, Signal* self_sg, T* __restrict__ result, int rank, int size,
               long long* stamps) {
#ifndef NO_STAMPS
  long long st[NSTAMP];
  const uint32_t inst = self_sg->_flag[blockIdx.x] / 2;
#endif
  STAMP(0)
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  int part = size / ngpus;
  int start = rank * part;
  int end = rank == ngpus - 1 ? size : start + part;
  int largest_part = part + size % ngpus;
  const P* ptrs[ngpus];
  P* tmps[ngpus];
#pragma unroll
  for (int i = 0; i < ngpus; i++) {
    int target = (rank + i) % ngpus;
    ptrs[i] = (const P*)_dp->ptrs[target];
    tmps[i] = tmp_buf<P>(sg.signals[target]);
  }
  auto tmp_out = tmps[0];
  barrier_at_start<ngpus>(sg, self_sg, rank);
  STAMP(1)
  for (int idx = start + tid; idx < end; idx += stride) {
    tmp_out[idx - start] = packed_reduce<P, ngpus, A>(ptrs, idx);
  }
  STAMP_SYNC()
  STAMP(2)
  barrier_at_end<ngpus>(sg, self_sg, rank);
  STAMP(3)
  for (int idx = tid; idx < largest_part; idx += stride) {
#pragma unroll
    for (int i = 0; i < ngpus; i++) {
      int gather_from_rank = ((rank + i) % ngpus);
      if (gather_from_rank == ngpus - 1 || idx < part) {
        int dst_idx = gather_from_rank * part + idx;
        ((P*)result)[dst_idx] = tmps[i][idx];
      }
    }
  }
  STAMP_SYNC()
  STAMP(4)
  STAMP_SAVE()
}

__global__ void calib(unsigned long long ns, long long* out) {  // cycles per ns of this SM over ~ns nanoseconds
  unsigned long long g0, g1;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g0));
  long long c0 = clock64();
  do { asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g1)); } while (g1 - g0 < ns);
  long long c1 = clock64();
  out[0] = c1 - c0;
  out[1] = (long long)(g1 - g0);
}

struct Gpu {
  cudaStream_t s;
  half *in, *out;
  Signal* sig;
  RankData* rd;
  long long* stamps;
  double cyc_per_ns;
};

static double calibrate(int g) {
  CK(cudaSetDevice(g));
  long long* d;
  long long h[2];
  CK(cudaMalloc(&d, 16));
  calib<<<1, 1>>>(20000000ull, d);
  CK(cudaMemcpy(h, d, 16, cudaMemcpyDeviceToHost));
  CK(cudaFree(d));
  return (double)h[0] / (double)h[1];
}

int main(int argc, char** argv) {
  const bool metrics = argc > 1 && std::string(argv[1]) == "metrics";
  const int metrics_ms = metrics && argc > 2 ? atoi(argv[2]) : 0;
  const int seed = argc > (metrics ? 3 : 1) ? atoi(argv[metrics ? 3 : 1]) : 0;
  setvbuf(stdout, nullptr, _IONBF, 0);
  const int max_bytes = *std::max_element(SIZES.begin(), SIZES.end());
  const int max_elems = max_bytes / 2;
  std::vector<Gpu> G(NG);
  for (int a = 0; a < NG; ++a) {
    CK(cudaSetDevice(a));
    for (int b = 0; b < NG; ++b)
      if (a != b) CK(cudaDeviceEnablePeerAccess(b, 0));
  }
  std::vector<std::vector<half>> host_in(NG, std::vector<half>(max_elems));
  for (int g = 0; g < NG; ++g) {
    std::mt19937 rng(1000 * seed + g);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto& x : host_in[g]) x = __float2half(nd(rng));
    CK(cudaSetDevice(g));
    CK(cudaStreamCreateWithFlags(&G[g].s, cudaStreamNonBlocking));
    CK(cudaMalloc(&G[g].in, max_bytes));
    CK(cudaMalloc(&G[g].out, max_bytes));
    CK(cudaMalloc(&G[g].sig, sizeof(Signal) + max_bytes));
    CK(cudaMemset(G[g].sig, 0, sizeof(Signal) + max_bytes));
    CK(cudaMalloc(&G[g].rd, sizeof(RankData)));
    CK(cudaMalloc(&G[g].stamps, sizeof(long long) * MAX_INST * kMaxBlocks * NSTAMP));
    CK(cudaMemcpy(G[g].in, host_in[g].data(), max_bytes, cudaMemcpyHostToDevice));
  }
  RankData rd;
  RankSignals sg;
  for (int g = 0; g < NG; ++g) {
    rd.ptrs[g] = G[g].in;
    sg.signals[g] = G[g].sig;
  }
  for (int g = 0; g < NG; ++g) {
    CK(cudaSetDevice(g));
    CK(cudaMemcpy(G[g].rd, &rd, sizeof(rd), cudaMemcpyHostToDevice));
    G[g].cyc_per_ns = calibrate(g);
  }
  if (!metrics) {
#ifdef NO_STAMPS
    printf("build,plain\n");
#else
    printf("build,stamped\n");
#endif
    for (int g = 0; g < NG; ++g) printf("calib,%d,%.5f\n", g, G[g].cyc_per_ns);
  }

  std::vector<std::pair<int, int>> configs;  // (algo 1 or 2, bytes)
  for (int algo : {1, 2})
    for (int b : SIZES) configs.push_back({algo, b});
  std::shuffle(configs.begin(), configs.end(), std::mt19937(seed));

  for (auto [algo, bytes] : configs) {
    const int size = bytes / 16;  // packs of 8 halves, as vLLM: size /= d
    const int blocks = std::min(defaultBlockLimit, (size + THREADS - 1) / THREADS);
    auto launch = [&](int g) {
      if (algo == 1)
        car_1stage<half, NG><<<blocks, THREADS, 0, G[g].s>>>(G[g].rd, sg, G[g].sig, G[g].out, g, size, G[g].stamps);
      else
        car_2stage<half, NG><<<blocks, THREADS, 0, G[g].s>>>(G[g].rd, sg, G[g].sig, G[g].out, g, size, G[g].stamps);
    };
    // correctness: one eager call, compare with a host fp32 sum
    for (int g = 0; g < NG; ++g) {
      CK(cudaSetDevice(g));
      launch(g);
      CK(cudaGetLastError());
    }
    double max_err = 0, max_ref = 0;
    std::vector<half> out(bytes / 2);
    for (int g = 0; g < NG; ++g) {
      CK(cudaSetDevice(g));
      CK(cudaStreamSynchronize(G[g].s));
      CK(cudaMemcpy(out.data(), G[g].out, bytes, cudaMemcpyDeviceToHost));
      for (int i = 0; i < bytes / 2; ++i) {
        float ref = 0;
        for (int r = 0; r < NG; ++r) ref += __half2float(host_in[r][i]);
        max_err = std::max(max_err, (double)fabsf(__half2float(out[i]) - ref));
        max_ref = std::max(max_ref, (double)fabsf(ref));
      }
    }
    // graphs: K back-to-back calls per GPU
    std::vector<cudaGraphExec_t> exec(NG);
    for (int g = 0; g < NG; ++g) {
      CK(cudaSetDevice(g));
      cudaGraph_t graph;
      CK(cudaStreamBeginCapture(G[g].s, cudaStreamCaptureModeRelaxed));
      for (int k = 0; k < K; ++k) launch(g);
      CK(cudaStreamEndCapture(G[g].s, &graph));
      CK(cudaGraphInstantiate(&exec[g], graph, 0));
      CK(cudaGraphDestroy(graph));
    }
    auto replay_all = [&](int n) {
      for (int r = 0; r < n; ++r)
        for (int g = 0; g < NG; ++g) {
          CK(cudaSetDevice(g));
          CK(cudaGraphLaunch(exec[g], G[g].s));
        }
    };
    auto sync_all = [&] {
      for (int g = 0; g < NG; ++g) {
        CK(cudaSetDevice(g));
        CK(cudaStreamSynchronize(G[g].s));
      }
    };
    replay_all(3);
    sync_all();
    char name[64];
    snprintf(name, sizeof name, "car_%dstage_%dK", algo, bytes >> 10);
    if (metrics) {
      nvtxRangePushA(name);
      auto t0 = std::chrono::steady_clock::now();
      while (std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() < metrics_ms) {
        replay_all(10);
        sync_all();
      }
      nvtxRangePop();
    } else {
      std::vector<uint32_t> flag0(NG * kMaxBlocks);
      for (int g = 0; g < NG; ++g) {
        CK(cudaSetDevice(g));
        CK(cudaMemcpy(&flag0[g * kMaxBlocks], G[g].sig->_flag, sizeof(uint32_t) * kMaxBlocks,
                      cudaMemcpyDeviceToHost));
      }
      std::vector<cudaEvent_t> e0(NG), e1(NG);
      for (int g = 0; g < NG; ++g) {
        CK(cudaSetDevice(g));
        CK(cudaEventCreate(&e0[g]));
        CK(cudaEventCreate(&e1[g]));
        CK(cudaEventRecord(e0[g], G[g].s));
      }
      replay_all(R);
      for (int g = 0; g < NG; ++g) {
        CK(cudaSetDevice(g));
        CK(cudaEventRecord(e1[g], G[g].s));
      }
      sync_all();
      for (int g = 0; g < NG; ++g) {
        float ms;
        CK(cudaEventElapsedTime(&ms, e0[g], e1[g]));
        printf("time,%d,%d,%d,%d,%.4f,%.3e\n", algo, bytes, blocks, g, ms * 1000.0 / (R * K), max_err / max_ref);
      }
#ifndef NO_STAMPS
      // per GPU and call: mean over blocks of each phase, and max over blocks of the total (cycles -> ns)
      std::vector<long long> h((size_t)MAX_INST * kMaxBlocks * NSTAMP);
      for (int g = 0; g < NG; ++g) {
        CK(cudaSetDevice(g));
        CK(cudaMemcpy(h.data(), G[g].stamps, h.size() * sizeof(long long), cudaMemcpyDeviceToHost));
        for (int c = 0; c < R * K; ++c) {
          // mean over blocks of each phase, and the phases of the slowest block (the critical path)
          double ph[4] = {0, 0, 0, 0}, crit[4] = {0, 0, 0, 0}, tot_max = -1;
          for (int b = 0; b < blocks; ++b) {
            const uint32_t inst = flag0[g * kMaxBlocks + b] / 2 + c;
            const long long* s = &h[((size_t)(inst % MAX_INST) * kMaxBlocks + b) * NSTAMP];
            for (int p = 0; p < 4; ++p) ph[p] += (s[p + 1] - s[p]) / G[g].cyc_per_ns / blocks;
            const double tot = (s[4] - s[0]) / G[g].cyc_per_ns;
            if (tot > tot_max) {
              tot_max = tot;
              for (int p = 0; p < 4; ++p) crit[p] = (s[p + 1] - s[p]) / G[g].cyc_per_ns;
            }
          }
          printf("phase,%d,%d,%d,%d,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f\n", algo, bytes, g, c, ph[0], ph[1],
                 ph[2], ph[3], crit[0], crit[1], crit[2], crit[3], tot_max);
        }
      }
#endif
    }
    for (int g = 0; g < NG; ++g) CK(cudaGraphExecDestroy(exec[g]));
  }
  if (!metrics)
    for (int g = 0; g < NG; ++g) printf("calib_end,%d,%.5f\n", g, calibrate(g));
  return 0;
}
