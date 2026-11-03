// M8 check 1: is a 16-byte vector atomic add (as used for cache-line atomics) available on sm_80?
// Build: nvcc -arch=sm_80 -c vector_red_sm80.cu   (expected to fail: vector red needs sm_90)
__global__ void k(float* p, float a, float b, float c, float d) {
  asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(p), "f"(a), "f"(b), "f"(c), "f"(d) : "memory");
}
