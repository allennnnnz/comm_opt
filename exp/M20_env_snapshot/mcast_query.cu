// M20: print CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED for every visible GPU (driver API).
// Build: nvcc -arch=sm_90 mcast_query.cu -o mcast_query -lcuda
#include <cuda.h>
#include <stdio.h>
int main() {
  CUresult r = cuInit(0);
  if (r != CUDA_SUCCESS) { const char *s; cuGetErrorString(r, &s); printf("cuInit failed: %s\n", s); return 1; }
  int n = 0; cuDeviceGetCount(&n);
  int drv = 0; cuDriverGetVersion(&drv);
  printf("driver_cuda_version=%d device_count=%d\n", drv, n);
  for (int i = 0; i < n; i++) {
    CUdevice d; cuDeviceGet(&d, i);
    char name[128]; cuDeviceGetName(name, sizeof(name), d);
    int mc = -1, fh = -1;
    CUresult r1 = cuDeviceGetAttribute(&mc, CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, d);
    CUresult r2 = cuDeviceGetAttribute(&fh, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, d);
    printf("gpu=%d name=\"%s\" MULTICAST_SUPPORTED=%d (rc=%d) HANDLE_TYPE_FABRIC_SUPPORTED=%d (rc=%d)\n", i, name, mc, (int)r1, fh, (int)r2);
  }
  return 0;
}
