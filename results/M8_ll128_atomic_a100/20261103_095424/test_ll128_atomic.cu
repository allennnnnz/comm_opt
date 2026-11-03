// M8: does A100 NVLink apply a warp's atomic adds to one 128-byte line atomically?
//
// Premise of the two-shot LL128 atomic all-reduce (Shen et al., arXiv 2607.16100, Sec IV-4): when a reader
// sees a line's flag (element 0) equal to the rank count N, every other element of that line has also been
// summed. Test: GPUs 1..7 each run warp-wide atomicAdd over NVLink on every line in GPU0's memory (lane 0
// adds 1 to the flag, lanes 1..31 add rank+1). An observer kernel on GPU0 reads each line with one warp-wide
// load and counts lines where flag == 7 but some element != 35 (= 2+...+8). GPU0 does not write: a writer
// kernel on GPU0 did not run concurrently with the spinning observer in a first attempt, so every write here
// arrives over NVLink, which is the case the premise is about. One process, peer access enabled.
//
// Build: nvcc -O3 -arch=sm_80 -o test_ll128_atomic test_ll128_atomic.cu
// Usage: ./test_ll128_atomic [log2_lines=20] [runs=3] [observe=1]
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

constexpr int NGPU = 8, WORDS = 32;  // 32 x 4 B = one 128-byte line
constexpr unsigned EXPECT_FLAG = NGPU - 1, EXPECT_DATA = NGPU * (NGPU + 1) / 2 - 1;  // writers are ranks 1..7

__global__ void writer(unsigned* lines, size_t nlines, unsigned rank) {
  const unsigned lane = threadIdx.x % 32;
  const size_t warp = (blockIdx.x * (size_t)blockDim.x + threadIdx.x) / 32;
  const size_t nwarps = (gridDim.x * (size_t)blockDim.x) / 32;
  for (size_t l = warp; l < nlines; l += nwarps)  // all 32 lanes hit the same 128-byte line together
    atomicAdd(&lines[l * WORDS + lane], lane == 0 ? 1u : rank + 1);
}

__global__ void observer(const volatile unsigned* lines, size_t nlines, unsigned long long* violations,
                         unsigned long long* timeouts, unsigned long long* raced) {
  const unsigned lane = threadIdx.x % 32;
  const size_t warp = (blockIdx.x * (size_t)blockDim.x + threadIdx.x) / 32;
  const size_t nwarps = (gridDim.x * (size_t)blockDim.x) / 32;
  for (size_t l = warp; l < nlines; l += nwarps) {
    for (long spin = 0;; ++spin) {
      unsigned v = lines[l * WORDS + lane];             // one warp-wide load of the whole line
      unsigned flag = __shfl_sync(0xffffffffu, v, 0);
      if (flag == EXPECT_FLAG) {
        bool bad = (lane != 0) && (v != EXPECT_DATA);
        if (__any_sync(0xffffffffu, bad) && lane == 0) atomicAdd(violations, 1ull);
        if (spin > 0 && lane == 0) atomicAdd(raced, 1ull);  // saw the line in progress before it completed
        break;
      }
      if (spin > (1l << 20)) { if (lane == 0) atomicAdd(timeouts, 1ull); break; }
    }
  }
}

int main(int argc, char** argv) {
  const int log2_lines = argc > 1 ? atoi(argv[1]) : 20;
  const int runs = argc > 2 ? atoi(argv[2]) : 3;
  const bool observe = argc > 3 ? atoi(argv[3]) != 0 : true;
  setvbuf(stdout, nullptr, _IONBF, 0);
  const size_t nlines = 1ull << log2_lines, bytes = nlines * WORDS * sizeof(unsigned);
  int n = 0;
  CK(cudaGetDeviceCount(&n));
  if (n < NGPU) { fprintf(stderr, "need %d GPUs, found %d\n", NGPU, n); return 1; }
  for (int d = 1; d < NGPU; ++d) {
    int ok = 0;
    CK(cudaDeviceCanAccessPeer(&ok, d, 0));
    if (!ok) { fprintf(stderr, "GPU%d cannot access GPU0\n", d); return 1; }
    CK(cudaSetDevice(d));
    CK(cudaDeviceEnablePeerAccess(0, 0));
  }
  CK(cudaSetDevice(0));
  unsigned* lines;
  unsigned long long *viol, *tmo, *rac;
  CK(cudaMalloc(&lines, bytes));
  CK(cudaMalloc(&viol, 8));
  CK(cudaMalloc(&tmo, 8));
  CK(cudaMalloc(&rac, 8));
  std::vector<cudaStream_t> st(NGPU);
  std::vector<cudaEvent_t> t0(NGPU), t1(NGPU);
  cudaStream_t obs;
  CK(cudaStreamCreateWithFlags(&obs, cudaStreamNonBlocking));
  for (int d = 0; d < NGPU; ++d) {
    CK(cudaSetDevice(d));
    CK(cudaStreamCreateWithFlags(&st[d], cudaStreamNonBlocking));
    CK(cudaEventCreate(&t0[d]));
    CK(cudaEventCreate(&t1[d]));
  }
  const int threads = 256, blocks = 216;  // 2 x 108 SMs
  printf("lines=%zu (%.0f MiB on GPU0), runs=%d\n", nlines, bytes / 1048576.0, runs);
  for (int r = 0; r < runs; ++r) {
    CK(cudaSetDevice(0));
    CK(cudaMemset(lines, 0, bytes));
    CK(cudaMemset(viol, 0, 8));
    CK(cudaMemset(tmo, 0, 8));
    CK(cudaMemset(rac, 0, 8));
    CK(cudaDeviceSynchronize());
    // observer first (half the SMs on GPU0), then writers on all GPUs; GPU0's writer shares its SMs
    if (observe) observer<<<108, threads, 0, obs>>>(lines, nlines, viol, tmo, rac);
    CK(cudaGetLastError());
    for (int d = 1; d < NGPU; ++d) {
      CK(cudaSetDevice(d));
      CK(cudaEventRecord(t0[d], st[d]));
      writer<<<blocks, threads, 0, st[d]>>>(lines, nlines, d);
      CK(cudaGetLastError());
      CK(cudaEventRecord(t1[d], st[d]));
    }
    double max_ms = 0;
    for (int d = 1; d < NGPU; ++d) {
      CK(cudaSetDevice(d));
      CK(cudaStreamSynchronize(st[d]));
      float ms;
      CK(cudaEventElapsedTime(&ms, t0[d], t1[d]));
      if (ms > max_ms) max_ms = ms;
    }
    CK(cudaSetDevice(0));
    CK(cudaStreamSynchronize(obs));
    unsigned long long hv = 0, ht = 0, hr = 0;
    CK(cudaMemcpy(&hv, viol, 8, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(&ht, tmo, 8, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(&hr, rac, 8, cudaMemcpyDeviceToHost));
    // final state check: every line should end at flag 8 and data 36
    std::vector<unsigned> h(nlines * WORDS);
    CK(cudaMemcpy(h.data(), lines, bytes, cudaMemcpyDeviceToHost));
    size_t final_bad = 0;
    for (size_t l = 0; l < nlines; ++l)
      for (int w = 0; w < WORDS; ++w)
        if (h[l * WORDS + w] != (w == 0 ? EXPECT_FLAG : EXPECT_DATA)) {
          if (++final_bad <= 3) printf("  bad line %zu: flag=%u w1=%u w31=%u\n", l, h[l * WORDS], h[l * WORDS + 1], h[l * WORDS + 31]);
          break;
        }
    // per writer GPU: nlines x 128 B of atomic adds over NVLink; aggregate into GPU0 = 7x that
    double per_gpu_GBs = bytes / (max_ms * 1e-3) / 1e9;
    printf("run %d: violations=%llu raced_lines=%llu timeouts=%llu final_bad_lines=%zu writer_max_ms=%.2f "
           "atomic_GBps_per_gpu=%.1f aggregate_into_gpu0_GBps=%.1f\n",
           r, hv, hr, ht, final_bad, max_ms, per_gpu_GBs, per_gpu_GBs * (NGPU - 1));
  }
  return 0;
}
