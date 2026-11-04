Files in this folder are copied unmodified from vLLM v0.33.0 (Apache License 2.0):
https://github.com/vllm-project/vllm/blob/v0.33.0/csrc/custom_collective_common.cuh
https://github.com/vllm-project/vllm/blob/v0.33.0/csrc/custom_all_reduce.cuh
Fetched 2026-11-04. M11's car_timed.cu includes custom_collective_common.cuh and re-states the two kernels from
custom_all_reduce.cuh with clock64 stamps added at phase boundaries (no other change).
