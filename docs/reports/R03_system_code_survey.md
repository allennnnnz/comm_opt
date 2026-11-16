# R03 主流系統通訊實作調查（S03）

- **執行**：Sonnet 5.5，2026-11-16　**規格**：`docs/specs/S03_system_code_survey.md`
- **路徑慣例**：下文所有 `檔案:行號` 皆相對於 `/work/allen0117/comm_opt_env/src/`（原規格是 `third_party/src/`，因 `/home` 滿額，依協調者更改指示改放 `/work`；見第 5 節「偏離」）。行號以版本表的 tag 為準。
- **性質**：只讀原始碼，未編譯、未執行、未用 GPU／Slurm。只寫事實，不評價效能。
- **證據標記**：`[src]` = 原始碼（已讀）；`[doc-web]` = 官方文件網頁（經 WebFetch 摘要，非逐字原文，引用時請再核對）。

---

## 4. 版本表（先放最前面，方便對照）

| 系統 | tag | commit | clone 日期 | 備註 |
|---|---|---|---|---|
| vLLM | v0.34.0 | 62d4bc3c7e33b1b6c906e233e6af2d3506c482ca | 2026-11-16 | `vllm/` |
| SGLang | v0.5.23 | 297e54fd105e974cdf401e5b4d20253a22c16f90 | 2026-11-16 | `sglang/`（程式在 `python/sglang/srt/`） |
| TensorRT-LLM | v1.2.1 | 376f7e1bd8ed543f75014309e3fd4b237e9b0e73 | 2026-11-16 | `trtllm/`；最新非 rc 穩定 tag（另有 v1.4.0rc*，未採用） |
| FlashInfer | v0.7.4 | 77039051b0be921788dc2784ec41423b558878ec | 2026-11-16 | `flashinfer/` |
| Megatron-Core | core_v0.20.0 | 685fb8bd970185e9ac3283fe32a5d6667bf4b268 | 2026-11-16 | `megatron/`（Megatron-LM repo） |
| DeepEP | v1.2.1 | 9af0e0d0e74f3577af1979c9b9e1ac2cad0104ee | 2026-11-16 | `deepep/`；該 commit 日期 2025-09-15，repo 只有這一個 release tag |
| PyTorch | v2.15.0 | 31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457 | 2026-11-16 | `pytorch/` |
| NCCL | v2.32.6-1 | ad2d192c1f279aefcaf340f67e07880b65a1116b | 2026-11-16 | `nccl/`；`nccl/makefiles/version.mk:9-12` = 2.32.6 |

全部 `--depth 1` 於 release tag，工作樹乾淨（`git status` 0 項）。

---

## 白話結論

1. 八張 H100 單機上，vLLM、SGLang、TensorRT-LLM 三個系統「小訊息」都不直接走 NCCL，而是各自寫的專用核心（自訂的 all-reduce），訊息大到某個門檻後才交給 NCCL 或 PyTorch 的對稱記憶體路徑。門檻大小各系統各自寫死在程式裡，且彼此不同。
2. 利用 NVSwitch 內建運算功能（NVLS／multimem）的路徑確實存在：vLLM 與 SGLang 用 PyTorch 對稱記憶體做 multimem all-reduce；FlashInfer 的 MNNVL 路徑要求 multicast（H100 SXM 可）；NCCL 本身預設也開 NVLS。但在 vLLM／SGLang 裡這些通常排在自訂核心之後，或需手動開啟。
3. 專家平行（MoE）的 all-to-all：vLLM 預設其實不是 all-to-all，而是「全收集＋分散歸約」；SGLang 預設不啟用任何 all-to-all 後端；Megatron 預設 token dispatcher 也是 allgather。真正的 all-to-all 專用實作（DeepEP 等）都要手動開。
4. DeepEP 在單機內只用 NVLink、不用 NVSHMEM；低延遲核心一律建立 NVSHMEM 並走 RDMA（可選擇讓機內流量走 NVLink）。
5. 各系統的程式註解裡有不少「TODO」與已知限制，集中在第 3 節；NCCL 的版本說明顯示 2.27 之後對稱記憶體、PAT、NVLS 變動頻繁，用哪個版本會影響結果，實驗時必須記錄。

---

## 1. 總表

註：「8×H100」=單機 8 卡 NVSwitch、TP/EP=8、cc 9.0。「預設實作」是依原始碼呼叫順序與預設值整理的**候選鏈**，實際走哪一條取決於執行期條件（是否安裝 flashinfer、multicast 是否可用、dtype、訊息大小）。

### 1.1 all-reduce

| 系統 | 8×H100 單機預設實作 | 切換條件 | NVLS／multimem | 融合／重疊 | 跨機 | 證據 |
|---|---|---|---|---|---|---|
| vLLM | 呼叫鏈依序：(1) NCCL symm mem（預設關）(2) FlashInfer AR (3) custom AR (4) torch symm_mem (5) pynccl/NCCL | (2) 需 2D contiguous、fp16/bf16/fp32、`nbytes ≤ max_workspace_size`（cc90、ws8 預設 0.5 MB）；(3) `world_size==2 or fully_connected`、`inp_size < max_size`、16 B 對齊；custom AR 內部 `<256KB` 用 1stage 否則 2stage；(4) 僅 bf16、≤64 MiB（cc90、ws8）；超出則 NCCL | (4) 用 `multimem_all_reduce_`（cc9.0 ws∈{4,6,8}）；(2) 的 mnnvl backend 需 multicast pointer | `fuse_allreduce_rms`（-O2 預設）在 TP>1、Hopper/Blackwell、有 flashinfer 時啟用：all-reduce+RMSNorm 融合，cc90 ws8 上限 0.5 MB | custom AR 僅限同節點（`mnnvl_only=not same_node`）；跨機走 NCCL（FlashInfer 跨機需 Blackwell MNNVL） | `vllm/vllm/distributed/device_communicators/cuda_communicator.py:389-464`；`…/all_reduce_utils.py:31-37,58-64`；`…/symm_mem.py:25-30,112-139`；`…/flashinfer_all_reduce.py:143-194,442-470,493-515`；`vllm/csrc/custom_all_reduce.cuh:300-307`；`vllm/vllm/config/vllm.py:208-231,330`；`vllm/vllm/compilation/passes/fusion/allreduce_rms_fusion.py:104-109` |
| SGLang | 呼叫順序：(1) CustomAllReduceV2（CUDA 預設）(2) PCIe-IPC (3) quick-AR（ROCm）(4) mscclpp（預設關）(5) torch symm_mem（預設關）(6) pynccl/NCCL | v2 以 sm90 查表：ws8 時 eager 模式 ≤128 KB 用 1shot（push/pull）、≤128 MB 用 2shot pull，multicast 範圍 0–128 MB；CUDA graph 模式 ≤128 KB 1shot、≤32 MB 2shot pull，multicast 範圍 512 KB–128 MB；v2 另受 `SGLANG_CUSTOM_ALL_REDUCE_V2_MAX_SIZE_KB`（預設 16 MB）限制；超出回 NCCL | v2 在 multicast 可用時有 `2shot_pull` multicast 變體；torch symm_mem 為 `--enable-torch-symm-mem`（預設 False），cc9 ws∈{4,6,8} 用 multimem，≤128 MB（ws8） | FlashInfer AR+RMSNorm 融合：特定 model arch、sm90/sm100、TP>1、單機（sm90）、`moe_a2a_backend=="none"` 時自動啟用 backend `auto`（sm90 單機 → trtllm） | v2 跨機僅在「單一 NVLink clique」下允許，其餘回 NCCL | `sglang/python/sglang/srt/distributed/parallel_state.py:949-993`；`…/device_communicators/custom_all_reduce.py:373-395`；`…/configs/custom_all_reduce_v2.py:264-310,316-330`；`…/custom_all_reduce_v2.py:74,388-398,582-597`；`…/torch_symm_mem.py:49-54,~140`；`sglang/python/sglang/srt/environ.py:1532,1535`；`…/arg_groups/fields/exec_.py:715-740,770-790`；`…/arg_groups/overrides.py:980-1001` |
| TensorRT-LLM | `AllReduceStrategy.AUTO`（預設）：先試 MNNVL（僅多機 aarch64）→ `tunable_allreduce`/AUTO 查表：custom one-shot / two-shot kernel 或 NCCL | AUTO 在 C++ 以查表選策略，key 為 (SM→90 或 100, TP, fusion op, hidden size, num_tokens)；SM90 TP=8 `NONE` 表：小 token 數多為 ONESHOT(4)，中段 TWOSHOT(5)，大 token 數 NCCL(0)；訊息 >64 MiB（工作區）或無 P2P／NVLink 則 NCCL | 預設不使用；`SYMM_MEM`（PyTorch multimem）、`NCCL_SYMMETRIC`、`UB` 僅在使用者明確指定策略時啟用；AUTO 表內不會選 SYMM_MEM | 融合：RESIDUAL_RMS_NORM、+FP8/NVFP4 量化、MoE finalize 等（`AllReduceFusionOp`）；MIN_LATENCY 下 `seq_len ≤ 128` 用 oneshot | MNNVL 策略只在 multi-node、aarch64、MNNVL 記憶體支援時啟用，否則 NCCL | `trtllm/tensorrt_llm/functional.py:3874-3897`；`trtllm/tensorrt_llm/_torch/distributed/ops.py:638-750,752-840`；`trtllm/cpp/tensorrt_llm/thop/allreduceOp.cpp:1199-1262,686-700`；`trtllm/cpp/tensorrt_llm/common/customAllReduceUtils.h:37-47,89-123,169-176`；`…/kernels/communicationKernels/allReduceFusionKernels.h:55` |
| FlashInfer | 為函式庫，無「預設系統路徑」。`create_allreduce_fusion_workspace(backend="auto")` 在 trtllm 與 mnnvl 兩者皆可用時選 mnnvl | trtllm backend 要求 `world_size ≤ 16`；mnnvl 要求 multicast（`is_multicast_supported`，註解寫 SM90+ NVLink）；mnnvl-cutedsl 需 Blackwell 且不會被 auto 選 | mnnvl 用 NVLink multicast（工作區建立時 multicast pointer 為空則報錯） | all-reduce+residual+RMSNorm（+量化）融合 API `allreduce_fusion` | trtllm backend 僅單機；mnnvl 跨機（Blackwell MNNVL） | `flashinfer/flashinfer/comm/allreduce.py:285-297,307-340,361-381,599`；`…/comm/trtllm_mnnvl_ar.py:197-204`；`…/comm/mnnvl.py:554-557`；`…/comm/mnnvl_cutedsl_ar.py:173` |
| Megatron-Core | TP all-reduce 直接呼叫 `torch.distributed.all_reduce`（NCCL）；無自訂 all-reduce 核心（未找到，見第 5 節） | 無切換 | 否（訓練路徑）；推論路徑另有 NVLS（見 1.4） | `tp_comm_overlap`（預設 False）需 `sequence_parallel`，把 GEMM 與 AG/RS 重疊；DDP 的 `overlap_grad_reduce`、`overlap_param_gather` 預設 False | NCCL | `megatron/megatron/core/tensor_parallel/mappings.py:39`；`megatron/megatron/core/model_parallel_config.py:300-304`；`megatron/megatron/training/arguments.py:1350-1351`；`megatron/megatron/core/distributed/distributed_data_parallel_config.py:22,25` |
| PyTorch | `torch.distributed.all_reduce`＝ProcessGroupNCCL。symmetric memory 提供另外的運算子，非預設 | `torch.ops.symm_mem.*`：需自行以 `empty` 配置並 `rendezvous` | `multimem_all_reduce_`（僅 sum、需 multicast）、`multimem_one_shot_all_reduce[_out]`；另有 `one_shot_all_reduce`、`two_shot_all_reduce_` | — | symmetric memory 後端預設 `CUDA`（機內）；`NVSHMEM`、`NCCL` 後端需設 `TORCH_SYMMMEM` | `pytorch/torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.cpp:503-523`；`…/CUDASymmetricMemoryOps.cu:164-194`；`…/CUDASymmetricMemoryUtils.cpp:140-157` |
| NCCL | NCCL 內部選 Ring／Tree／NVLS／NVLSTree 等 | `NCCL_ALGO` 可限制；NVLS 預設 `NCCL_NVLS_ENABLE=2`（有支援就用，資源配置失敗則報錯） | NVLS（NVLink SHARP，需 Hopper+、NVSwitch gen3）預設開啟 | 對稱記憶體 kernel（需註冊 window） | 同左；跨機另有 NVLSTree、CollNet（`NCCL_COLLNET_ENABLE` 預設 0） | `nccl/docs/userguide/source/env.rst:865-880,1506-1540,1965-1975`；`nccl/src/transport/nvls.cc:240-267`；`nccl/src/init.cc:2048-2049` |
| DeepEP | 不提供 all-reduce | — | — | — | — | `deepep/README.md:3-7` |

### 1.2 all-gather ／ reduce-scatter

| 系統 | 8×H100 單機預設實作 | 切換條件 | NVLS／multimem | 融合／重疊 | 跨機 | 證據 |
|---|---|---|---|---|---|---|
| vLLM | all-gather：pynccl（`pynccl_comm.all_gather`）；reduce-scatter：pynccl；custom AR 通訊器另有 `custom_all_gather`／`custom_reduce_scatter`（IPC，限同節點，跨機僅 MNNVL 版） | `dim==0 and should_nccl_symm_mem_ag_rs()` 時走 NCCL 對稱記憶體（`VLLM_USE_NCCL_SYMM_MEM` 預設 False） | 對稱記憶體路徑註解稱「NVLS symmetric memory」；custom AG 在 `mnnvl_multicast_ptr` 存在時用 Lamport multicast kernel | 序列平行的 gather-before-GEMM 用 dim=0 | NCCL | `vllm/vllm/distributed/device_communicators/cuda_communicator.py:469-545,552-587`；`…/pynccl_allocator.py:49-51`；`…/all_reduce_utils.py:159-168`；`…/custom_all_reduce.py:594-610`；`vllm/vllm/envs.py:316` |
| SGLang | `all_gather_into_tensor`／`reduce_scatter_along_dim` 經 pynccl；使用 `use_symmetric_memory` context（`--enable-symm-mem`，預設 False） | `--enable-nccl-nvls`（預設 False，「prefill heavy requests」）、`--enable-mscclpp`（預設 False，AllReduce 與 AllGather） | 另有 `triton_symm_mem_ag.py`（檔案存在，未逐行讀） | 見註 | NCCL | `sglang/python/sglang/srt/distributed/parallel_state.py:1079-1100,1363`；`…/arg_groups/fields/exec_.py:715-736`；`…/device_communicators/triton_symm_mem_ag.py` |
| TensorRT-LLM | MoE 的 attention-DP 路徑：`AllGatherReduceScatter`（MoE TP≠1 時強制；auto-selection 的最後備援） | 見 1.4 | — | — | — | `trtllm/tensorrt_llm/_torch/modules/fused_moe/communication/communication_factory.py:98-104,189-190` |
| FlashInfer | `mnnvl_cutedsl_ag.py`（Blackwell CuTe DSL）；`all_gather_matmul/`（目錄） | 未讀細節 | — | all-gather＋matmul 融合目錄存在 | — | `flashinfer/flashinfer/comm/`（`ls`） |
| Megatron-Core | `torch.distributed.all_gather_into_tensor`／`reduce_scatter_tensor`（TP/SP、DP 分散式優化器） | — | 推論 MoE：`inference_moe_token_dispatcher_type` 預設 `'nvls'`，用 multimem AllGather-V／ReduceScatter-V | `tp_comm_overlap`、`overlap_grad_reduce`、`overlap_param_gather` | NCCL | `megatron/megatron/core/tensor_parallel/mappings.py:16-17`；`megatron/megatron/core/transformer/transformer_config.py:1436-1441`；`megatron/megatron/core/transformer/moe/token_dispatcher_inference.py:12-13,32-33` |
| PyTorch | NCCL；symmetric memory 另有 `multimem_all_gather_out`、`reduce_scatter_out`、`fused_all_gather_matmul`、`fused_matmul_reduce_scatter` | 同上 | `multimem_all_gather_out` 要求 output 有 multicast | async TP：`torch/_inductor/config.py` 的 `_micro_pipeline_tp`（預設 False）；`fused_*` 運算子 | — | `…/SymmetricMemory.cpp:503-523,526-530`；`…/CUDASymmetricMemoryOps.cu:392-416`；`pytorch/torch/distributed/_symmetric_memory/__init__.py:533,548`；`pytorch/torch/_inductor/config.py:1350`；`pytorch/torch/_inductor/fx_passes/post_grad.py:340` |
| NCCL | Ring／NVLS／PAT（PAT 僅每節點 1 GPU，2.23 引入；2.31 起預設關閉，需 `NCCL_ALGO=PAT`） | 見第 2.8 節 | NVLS；對稱記憶體 AllGather kernel | — | PAT、NVLS+IB SHARP（2.27） | `nccl/src/config/algorithm_registry.cc:53`；`[doc-web]` NCCL release notes 2.23.4／2.27.3／2.31.2 |
| DeepEP | 無 | — | — | — | — | — |

### 1.3 all-to-all（含 MoE dispatch/combine）

| 系統 | 8×H100 單機預設實作 | 切換條件 | NVLS／multimem | 融合／重疊 | 跨機 | 證據 |
|---|---|---|---|---|---|---|
| vLLM | EP 預設 `all2all_backend="allgather_reducescatter"`（dispatch 用 `all_gatherv`，combine 用 `reduce_scatterv`）；非 NCCL all-to-all | `--all2all-backend` 可選：`deepep_high_throughput`、`deepep_low_latency`、`deepep_v2`、`flashinfer_nvlink_one_sided`、`flashinfer_nvlink_two_sided`、`nixl_ep`、`mori*`、`moonep` 等 | 否（該層面） | DeepEP LL 有 `VLLM_DEEPEP_*` 設定；`deepep_v2` 有 `VLLM_DEEPEP_V2_PREFER_OVERLAP`（預設 0） | DeepEP HT internode 用 RDMA（`num_rdma_bytes`、`num_qps_per_rank`）；`VLLM_DEEPEP_HIGH_THROUGHPUT_FORCE_INTRA_NODE` 預設 0 | `vllm/vllm/config/parallel.py:229-250`；`vllm/vllm/distributed/device_communicators/cuda_communicator.py:213-260`；`…/all2all.py:57-90,118-150,257-290,316-330`；`vllm/vllm/envs.py:303-310` |
| SGLang | `--moe-a2a-backend` 預設 `"none"`（不啟用 EP all-to-all 後端）；另有 `torch.distributed` all_to_all_single 包裝 | 選項：deepep、mooncake、nixl、mori、flashinfer、megamoe、deepep_v2、pplx 等 | 否 | `megamoe` 要求 SM90（block-FP8 experts）且無 two-batch overlap | deepep_v2 `direct`（單機 NVLink）／`hybrid`（多機） | `sglang/python/sglang/srt/arg_groups/fields/exec_.py:836-880`；`…/arg_groups/mega_moe_hook.py:34-44`；`…/distributed/parallel_state.py:1211` |
| TensorRT-LLM | MoE 通訊自動選擇（僅 attention-DP 且 MoE TP=1 時才有通訊策略；否則 None／AllGather+ReduceScatter）：NVLinkOneSided → NVLinkTwoSided → DeepEP → DeepEPLowLatency → AllGatherReduceScatter | 前兩者平台檢查為 `MnnvlMemory.supports_mnnvl()`＝本卡所有 NVLink link 皆 active（原始碼註解 TODO：此判斷不等價於 MNNVL 支援）；DeepEP 需 `TRTLLM_CAN_USE_DEEP_EP=1` 且 bf16；可用 `TRTLLM_FORCE_COMM_METHOD` 強制 | NVLink one/two-sided 使用 MNNVL 記憶體（非 multimem all-reduce） | `payload_in_workspace`、`alltoall_result_do_sum` | DeepEP 跨機 2/4/8/16 節點、每節點 8 rank | `trtllm/tensorrt_llm/_torch/modules/fused_moe/communication/communication_factory.py:60-70,93-125,153-190`；`…/nvlink_one_sided.py:243-247`；`…/deep_ep.py:79-100`；`trtllm/tensorrt_llm/_mnnvl_utils.py:326-356` |
| FlashInfer | `moe_ep/backends/split/comm/` 含 `nvlink_one_sided`、`nvlink_two_sided`、`nccl_ep`、`nixl_ep`、`cake`；one-sided 的 CFT 支援探測要求 compute capability ≥ 10.0（H100=9.0 不通過）；`dcp_alltoall`（decode CP）、`ulysses`（含 PCIe 與 a2a 路徑） | cc<10 回報不支援 | — | — | — | `flashinfer/flashinfer/moe_ep/backends/split/comm/nvlink_one_sided/communication.py:108-116`；`flashinfer/flashinfer/comm/dcp_alltoall.py`；`flashinfer/flashinfer/comm/ulysses.py` |
| Megatron-Core | `moe_token_dispatcher_type` 預設 `'allgather'`；選 `'alltoall'` 時用 `torch.distributed.all_to_all_single`；選 `'flex'` 時預設 backend `deepep`（可選 deepepv2、hybridep、ncclep） | `moe_enable_deepep` 已棄用；`overlap_moe_expert_parallel_comm` 需 alltoall/flex dispatcher、bf16/fp16、關閉 shared-expert overlap | 否 | `overlap_moe_expert_parallel_comm`、`delay_wgrad_compute`（預設關）；`moe_shared_expert_overlap`（預設 False） | DeepEP（NVLink+RDMA）；註解：僅 hybridep 支援 MNNVL | `megatron/megatron/core/transformer/transformer_config.py:794,998-1008,3613-3652,3684-3691`；`megatron/megatron/core/tensor_parallel/mappings.py:454-464,564-568`；`megatron/megatron/core/transformer/moe/fused_a2a.py:52-84` |
| PyTorch | NCCL `all_to_all_single`；symmetric memory 另有 `all_to_all_vdev`、`all_to_all_vdev_2d`、`nvshmem_all_to_all`、`nccl_all_to_all_nd` | 後端見上 | — | — | NVSHMEM／NCCL 後端 | `pytorch/torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.cpp:570-578` |
| NCCL | `ncclAlltoAll` 在 2.28 起有 CE（Copy Engine）集體版減少 SM 使用（`[doc-web]`）；LL 對稱 kernel `alltoall_LL.cuh` | `NCCL_CTA_POLICY` | — | — | 2.29：all2all/send/recv 遵守 `NCCL_NETDEVS_POLICY` | `nccl/src/device/symmetric/alltoall_LL.cuh`；`nccl/docs/userguide/source/env.rst:885-895`；`[doc-web]` 2.28.3、2.29.2 |
| DeepEP | normal kernel（intranode：NVLink；internode：NVLink+RDMA 轉送）；low-latency kernel（純 RDMA，NVSHMEM/IBGDA；`allow_nvlink_for_low_latency_mode=True` 預設，同機流量可走 NVLink） | 見第 2.6 節 | `NVSHMEM_DISABLE_NVLS=1`（註解 "Disable NVLink SHArP"） | LL 有 `return_recv_hook` hook 式重疊，不占 SM | 需 RDMA（IB） | `deepep/README.md:7,45-52`；`deepep/deep_ep/buffer.py:30,34-53,94-112` |

### 1.4 send／recv

| 系統 | 8×H100 單機預設實作 | 證據 |
|---|---|---|
| vLLM | `pynccl_comm.send/recv`（可用時），否則 `torch.distributed.send/recv`；預設目標為 ring 下一個 rank（PP 使用） | `vllm/vllm/distributed/device_communicators/cuda_communicator.py:822-850` |
| SGLang | 同樣經 `pynccl_comm.send/recv` | `sglang/python/sglang/srt/distributed/parallel_state.py:2062-2085` |
| TensorRT-LLM | 未專門調查（未找到 TP/PP 以外的 send/recv 抽象；搜尋詞：`def send`、`isend`） | — |
| FlashInfer | 無獨立 send/recv（未找到；搜尋詞：`send`、`recv` 於 `flashinfer/comm/`） | — |
| Megatron-Core | PP 使用 `torch.distributed` P2P（未逐行調查，見第 5 節） | — |
| PyTorch | NCCL；symmetric memory 有 `nccl_put/get`、`nvshmem_put/get`、`*_signal` | `pytorch/torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.cpp:552-566` |
| NCCL | `ncclSend/ncclRecv`；2.29 起遵守 `NCCL_NETDEVS_POLICY`；2.30 起 one-sided RMA（`ncclPutSignal`／`ncclWaitSignal`）可支援 CUDA graph | `[doc-web]` 2.29.2、2.30.3 |

---

## 2. 各系統細節

### 2.1 vLLM v0.34.0

**通訊實作**：`CudaCommunicator` 持有 `pynccl_comm`、`ca_comm`（custom AR）、`symm_mem_comm`、`fi_ar_comm`（FlashInfer）、`fi_pcie_ipc_ar_comm`、`qr_comm`（ROCm）、`aiter_ar_comm`（ROCm）。all-reduce 呼叫鏈見 `cuda_communicator.py:389-464`，摘錄：

```
cuda_communicator.py:399-404   if (self.pynccl_comm is not None and not use_fi_ar
                                   and should_nccl_symm_mem_allreduce(self.pynccl_comm.world_size, input_)):
                                   out = torch.ops.vllm.all_reduce_symmetric_with_copy(input_)
cuda_communicator.py:419-423   if use_fi_ar: out = fi_ar_comm.all_reduce(input_)
cuda_communicator.py:434-442   ca_comm ... should_custom_ar(input_) ... custom_all_reduce
cuda_communicator.py:454-458   symm_mem_comm ... should_use_symm_mem(input_) ... symm_mem_comm.all_reduce
cuda_communicator.py:459-463   pynccl_comm.all_reduce(input_)   （最後；pynccl 不可用則 torch.distributed）
```

**切換條件與數值**
- custom AR：`custom_all_reduce.py:500-515`：`world_size > 8` 不用；dtype 限 fp32/fp16/bf16；`inp_size % 16 != 0` 不用；`if self.world_size == 2 or self.fully_connected: return self.batch_invariant or inp_size < self.max_size`。註解：`# for 4 or more non NVLink-capable GPUs, custom allreduce provides little performance improvement over NCCL.`
- `max_size`：預設 `8192 * 1024`（`custom_all_reduce.py:138`）；若 torch symm_mem 通訊器啟用，取 `min(CUSTOM_ALL_REDUCE_MAX_SIZES[cc][ws], max_size)`（`custom_all_reduce.py:237-253`）。cc9.0：ws2=64 MiB、ws4=32 MiB、ws6=512 KB、ws8=256 KB（`all_reduce_utils.py:31-37`）。
- custom AR 核心 1stage/2stage：`custom_all_reduce.cuh:300-307`：`if (world_size_ == 2) 1stage; else if ((world_size_ <= 4 && bytes < 512*1024) || (world_size_ <= 8 && bytes < 256*1024)) 1stage; else 2stage`；環境變數強制 `1stage|oneshot|2stage|twoshot`（`custom_all_reduce.cuh:274-286`）。
- torch symm_mem：`symm_mem.py:25-30` `_WORLD_SIZES_MULTIMEM = {"9.0": [4, 6, 8], "10.0": [6, 8], ...}`；`symm_mem.py:112-122` 只收 bf16、`inp_size % 4 == 0`、`inp_size <= max_size`；上限 cc9.0 ws8 = 64 MiB（`all_reduce_utils.py:58-64`）。multimem 用 `torch.ops.symm_mem.multimem_all_reduce_`，否則 `two_shot_all_reduce_`（`symm_mem.py:~140-150`）。`handle.multicast_ptr == 0` 時通訊器停用（`symm_mem.py:96-100`）。
- NCCL symm mem 分派：`all_reduce_utils.py:111-156`：`min_world_size=4`；ws8 在 (16 KiB, 128 KiB) 範圍內「不」用 NCCL symm mem（留給 custom AR）。註解：`# Always use symm mem for world_size > 8`。啟用旗標 `VLLM_USE_NCCL_SYMM_MEM` 預設 False（`envs.py:316`；`pynccl_allocator.py:49-51`）。
- FlashInfer AR：`VLLM_ALLREDUCE_USE_FLASHINFER` 預設 True（`envs.py:290`），`should_use_fi_ar` 條件見 `flashinfer_all_reduce.py:493-515`；backend `auto`（單機）→ `mnnvl`，原始碼註解：`# MNNVL decodes faster than TRTLLM on single-node NVSwitch systems (about 3-5% throughput at TP4/TP8 on H200 and B200). Fall back to TRTLLM where the node lacks NVSwitch multicast...`（`flashinfer_all_reduce.py:174-180`）。standalone 上限：`FI_MNNVL_ALLREDUCE_MAX_SIZE_MB` 無 cc90 項目，故用 `default_fi_allreduce_fusion_max_size_mb`（cc90：ws2=64、ws4=2、ws8=0.5 MB；`allreduce_rms_fusion.py:104-109`；`flashinfer_all_reduce.py:440-470`）。
- 融合：`fuse_allreduce_rms` 於 -O2（預設，`config/vllm.py:465`）以 `enable_allreduce_rms_fusion` 決定：`tensor_parallel_size > 1 and is_cuda and has_flashinfer and (cc family 100 or cc 90)`（`config/vllm.py:208-231`）。
- 環境變數／旗標：`VLLM_ALLREDUCE_USE_SYMM_MEM`（預設 1）（`envs.py:289`）、`VLLM_ALLREDUCE_USE_FLASHINFER`、`VLLM_FLASHINFER_ALLREDUCE_BACKEND`（auto|trtllm|mnnvl，`envs.py:233`）、`VLLM_USE_NCCL_SYMM_MEM`、`VLLM_NCCL_SO_PATH`（`envs.py:21`）、`--disable-custom-all-reduce`（`config/parallel.py:259-260`）、`--all2all-backend`（`config/parallel.py:229`）、`VLLM_DEEPEP_BUFFER_SIZE_MB`（預設 1024）等（`envs.py:303-310`）。custom AR 僅用於 TP 群組（`cuda_communicator.py:57-63`）。
- EP all-to-all：見 1.3。`AgRsAll2AllManager.dispatch` 對各 DP rank 做 `all_gatherv`（`all2all.py:118-150`）。
- 文件：本次未逐頁讀取 vLLM 官方文件網站（未找到對應調查），設定資訊以原始碼為準。

### 2.2 SGLang v0.5.23

- **all-reduce 呼叫流程**（`parallel_state.py:673-770`）：compile 外：先算 `should_use_pymscclpp_allreduce` 與 `should_use_custom_allreduce`；若 pynccl 在且 `is_symmetric_memory_enabled()` 且兩者皆不用 → pynccl 對稱記憶體 all-reduce；再試 `_can_use_flashinfer_allreduce`；再 `_resolve_outplace_all_reduce_method`（`parallel_state.py:949-993`），順序 `ca` → `pcie_ipc`（註解：「After ``ca``: the PCIe-IPC kernels are for hosts where no fabric-specific backend applies」）→ `qr` → `pymscclpp` → `torch_symm_mem` → `pynccl`（piecewise cuda graph）→ None（in-place：pynccl 或 `torch.distributed.all_reduce`）。
- **CustomAllReduceV2**（CUDA 預設）：`dispatch_custom_allreduce` 註解 `On CUDA, the JIT-compiled v2 implementation is used by default. Set SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2=0 to fall back to the legacy CustomAllreduce.`（`custom_all_reduce.py:382-383`）；環境變數預設 True（`environ.py:1532`）。支援的 world size 由 `get_supported_world_sizes` 決定，sm90 為 2–8（`configs/custom_all_reduce_v2.py:264-310`）；其餘架構用預設 2–16（`configs/custom_all_reduce_v2.py:~326-336`）。
  - 演算法：`1shot_push / 2shot_push / 1shot_pull / 2shot_pull（可 multicast）`，選擇函式 `_pick_config`（`custom_all_reduce_v2.py:385-400`），超過所有門檻 → 回傳 None（NCCL）。
  - sm90（註解 `Tuned on H200`）ws8：graph=(128 KB, 128 KB, 32 MB, mc=Range(512 KB, 128 MB))；eager=(128 KB, 128 KB, 128 MB, mc=Range(0, 128 MB))（`configs/custom_all_reduce_v2.py:307-310`）；`num_pull_blocks=64`、`num_mc_blocks=128 // world_size`（ws≥4；`:268-272`）。
  - 最大大小：`SGLANG_CUSTOM_ALL_REDUCE_V2_MAX_SIZE_KB = EnvInt(16 * 1024)`（`environ.py:1535`）；強制門檻 `SGLANG_FORCE_CUSTOM_ALL_REDUCE_V2_{PULL,PUSH}_SIZE_KB`（`custom_all_reduce_v2.py:77-78`）。
  - 單機判斷：`full_nvlink` 為真才可用；跨機僅限單一 NVLink clique 且 VMM allocator（`custom_all_reduce_v2.py:582-597`）。
  - 舊版 `CustomAllreduce.should_custom_ar`：`world_size == 2 or full_nvlink` 且 `inp_size <= max_size`（`custom_all_reduce.py:260-283`）。
- **torch_symm_mem**：`--enable-torch-symm-mem`（預設 False），說明文字 `Only supports CUDA device SM90 and above. SM90 supports world size 4, 6, 8.`（`exec_.py:736-739`）；`_WORLD_SIZES_MULTIMEM = {9: [4, 6, 8], 10: [4, 6, 8]}`（`torch_symm_mem.py:49-54`）；上限 cc9：ws8=128 MB（`all_reduce_utils.py:3-16`）；`multicast_ptr == 0` 時停用。註解說明該檔改寫自 vLLM（`torch_symm_mem.py:3`）。
- **FlashInfer AR 融合**：`--flashinfer-allreduce-fusion-backend`（auto|trtllm|mnnvl|cutedsl）說明：`'auto': choose mnnvl on Blackwell ... and trtllm on SM90 single-node systems.`（`exec_.py:770-789`）。自動啟用條件：支援的 model arch、sm90 或 sm100、`tp_size > 1`、非 attention-DP、單機（或 sm100）、`moe_a2a_backend == "none"`（`overrides.py:980-1001`）；`--enforce-disable-flashinfer-allreduce-fusion` 可強制關閉（`overrides.py:1004-1014`；`exec_.py:766-768`）。
- **其他旗標**：`--disable-custom-all-reduce`、`--enable-mscclpp`（預設 False，`exec_.py:732-735`）、`--enable-symm-mem`（NCCL 對稱記憶體，預設 False，`exec_.py:718-724`）、`--enable-nccl-nvls`（預設 False，`exec_.py:715-717`）、`--moe-a2a-backend`（預設 none，`exec_.py:836-865`）。
- **限制／TODO（原文）**：見第 3 節。

### 2.3 TensorRT-LLM v1.2.1

- **策略列舉**：`AllReduceStrategy`：NCCL=0、MIN_LATENCY=1、UB=2、AUTO=3、ONESHOT=4、TWOSHOT=5、LOWPRECISION=6、MNNVL=7、NCCL_SYMMETRIC=8、SYMM_MEM=9（`functional.py:3874-3884`）；預設 `allreduce_strategy = AUTO`（`_torch/model_config.py:110`）。
- **文件字串**（`ops.py:660-661`）：`AUTO: AUTO chooses the best available strategy. Will try MNNVL, then choose between NCCL and MIN_LATENCY based on a heuristic policy.`
- **`AllReduce.forward` 順序**（`ops.py:752-840`）：(1) 若 `symm_mem_allreduce` 存在（僅 strategy==SYMM_MEM 時建立，`ops.py:696-720`）且 fusion_op==NONE → symm mem；(2) 若 `mnnvl_allreduce` → MNNVL；(3) 否則 `tunable_allreduce`（AUTO 且未設 `TLLM_DISABLE_ALLREDUCE_AUTOTUNE=1` 且非 `_disable_mpi`）或 `all_reduce_op`。
- **MNNVL 條件**（`ops.py:540-550`）：`is_testing or (dtype 支援 and not mapping.has_cp() and mapping.is_multi_node() and MnnvlMemory.supports_mnnvl() and is_on_aarch64)`。故 x86 H100 單機不啟用 MNNVL 策略。
- **C++ 選擇**（`allreduceOp.cpp:1199-1262`）：UB／NCCL／NCCL_SYMMETRIC 由使用者保證；`ifFallbackToNCCL`：`message_size_bytes > max_workspace_size || !mIsP2PSupported || !mIsNVLINKSupported` → NCCL（註解：`TODO: Use NCCL_SYMMETRIC once the memory allocation issue is resolved.`）；否則 AUTO 時 `selectStrategyLookUpTable`。工作區上限 64 MiB，環境變數 `TRTLLM_ALLREDUCE_FUSION_WORKSPACE_SIZE` 可改（`customAllReduceUtils.h:37-47`）。
- **查表**（`customAllReduceUtils.h:89-123`）：SM<100 一律映射到 90；超出表格範圍 → NCCL。SM90 TP=8 `Fusion=NONE` 表（`:169-176`）：每列為 hidden size（128…8192），欄為 token 數（1…16384），值 4=ONESHOT、5=TWOSHOT、0=NCCL；例如 hidden=4096（第 6 列）：`{4,4,4,4,4,4,4,0,5,5,0,0,0,0,0}`＝token 1–64 ONESHOT、128 NCCL、256–512 TWOSHOT、≥1024 NCCL。
- **MIN_LATENCY** 內部 oneshot 判斷：`seq_len <= kOneShotMaxToken(=128) || hidden_size < tp_size`（`allreduceOp.cpp:691-697`；`allReduceFusionKernels.h:55`）。
- **環境變數**：`TLLM_DISABLE_ALLREDUCE_AUTOTUNE`（`ops.py:~815`）、`TRTLLM_ALLREDUCE_FUSION_WORKSPACE_SIZE`、`TLLM_NCCL_MIN_REGISTRATION`（`allreduceOp.cpp:505`）、`TRTLLM_FORCE_COMM_METHOD`、`TRTLLM_CAN_USE_DEEP_EP`、`TLLM_TEST_MNNVL`。
- **融合**：`AllReduceFusionOp` 列舉 9 種（`functional.py:3887-3896`）；`MoEAllReduce`（MoE reduction + add + AR + RMSNorm，`ops.py:~845-870`）。
- **MoE 通訊**：見 1.3。MoE 在 `(not enable_attention_dp) or dp_size == 1` 時回傳 None，無通訊策略（`communication_factory.py:96-98`）。
- 文件：未讀 TensorRT-LLM 官方文件網站（未找到對應調查，僅原始碼）。

### 2.4 FlashInfer v0.7.4

- `flashinfer/comm/` 含：`allreduce.py`（統一融合 API）、`trtllm_ar.py`、`trtllm_mnnvl_ar.py`、`mnnvl.py`、`mnnvl_cutedsl_ar.py`／`_ag.py`、`vllm_ar.py`、`torch_symmetric_memory.py`、`nvshmem*.py`、`pcie_ipc_*`、`cake_*`（SM100/103）、`dcp_alltoall.py`、`ulysses*.py`、`all_gather_matmul/`、`fp8_rowwise_gemm_allreduce/` 等。
- 統一 API：`create_allreduce_fusion_workspace(backend: Literal["trtllm","mnnvl","mnnvl-cutedsl","auto"]="auto", ...)`（`allreduce.py:361-381`）；auto 說明：`"auto"` uses a topology-based heuristic to pick between `"trtllm"` and `"mnnvl"`; it never selects `"mnnvl-cutedsl"`。
- 啟發式（`allreduce.py:307-340`）：兩者皆可用時 `if "mnnvl" in suitable_backends: return ["mnnvl"]`；註解：`MNNVL is either on par (smaller problem sizes) or significantly faster than TRTLLM ... for single-node scenarios. TRTLLM still has the larger specialized-fusion surface, such as MoE patterns and packed group FP8 quantization.`
- 硬條件：trtllm 後端 `world_size <= 16`（`allreduce.py:285-297`）；mnnvl 需 `is_multicast_supported`（`allreduce.py:599`；`mnnvl.py:554-557`，docstring `SM90+ NVLink`）；`trtllm_mnnvl_ar.py:197-204` 在 multicast pointer 為空時報錯，訊息寫 `This requires NVLink multicast (SM90+ with NVLink, e.g. H100 SXM)`。
- 架構限制：`mnnvl_cutedsl_ar.py:173` `MNNVL CuTe DSL kernels require a Blackwell GPU`；`cake_*` 需 SM100/SM103（`trtllm_ar.py:938`）；`nvlink_one_sided` MoE 通訊 cc ≥ 10.0（`communication.py:113-115`）。
- vLLM 與 SGLang 均透過此模組使用 AR 融合（見 2.1、2.2）。
- 文件：未讀 FlashInfer 官方文件網站。

### 2.5 Megatron-Core core_v0.20.0

- **TP/SP**：`tensor_parallel/mappings.py:39` 使用 `torch.distributed.all_reduce`；`mappings.py:16-17` 使用 `all_gather_into_tensor`、`reduce_scatter_tensor`。
- **TP 通訊重疊**：`tp_comm_overlap: bool = False`（`model_parallel_config.py:300-304`），參數驗證要求 `sequence_parallel == True`（`training/arguments.py:1350-1351`）；細項旗標 `tp_comm_bulk_wgrad` 等（`model_parallel_config.py:306-352`）；原始碼中重疊實作委託 TransformerEngine（未逐行驗證，見第 5 節）。
- **DDP 重疊**：`overlap_grad_reduce=False`、`overlap_param_gather=False`（`distributed_data_parallel_config.py:22,25`）。
- **CP**：`cp_comm_type` 可為 `"p2p"`、`"all_gather"`、`"a2a"`、`"a2a+p2p"`（`transformer_config.py:1156-1160`）。
- **EP／MoE dispatcher**：`moe_token_dispatcher_type: Literal['allgather','alltoall','flex'] = "allgather"`（`transformer_config.py:998-1000`）；`moe_flex_dispatcher_backend: Literal['deepep','deepepv2','hybridep','ncclep'] = "deepep"`，註解 `[Experimental] ... Currently only "hybridep" backend supports the MNNVL case.`（`transformer_config.py:1005-1008`）；`moe_enable_deepep` 已棄用（`transformer_config.py:2167-2174`）。`alltoall` 用 `torch.distributed.all_to_all_single`（`mappings.py:454,464`）；`flex`/DeepEP 的 buffer 由 `fused_a2a.py:52-84` 建立（`Buffer(group, num_nvl_bytes, num_rdma_bytes)`）。
- **EP 重疊**：`overlap_moe_expert_parallel_comm` 檢查：需 EP、`alltoall`/`flex`、bf16/fp16、關閉完整重算、關閉 `moe_shared_expert_overlap` 等（`transformer_config.py:3613-3657`）；`delay_wgrad_compute` 需先開 `overlap_moe_expert_parallel_comm`（`transformer_config.py:3684-3691`）。
- **推論**：`inference_moe_token_dispatcher_type: Literal['nccl','nvls'] = 'nvls'`，`'nvls'`＝`Variable-count AllGather-V/ReduceScatter-V via NVLS multimem kernels`（`transformer_config.py:1436-1441`；`token_dispatcher_inference.py:12-13`）。對稱記憶體相關檔案：`inference/communication/torch_symm_triton/`、`inference/symmetric_memory.py`、`nccl_allocator.py`。
- 文件：未讀 Megatron 官方文件。

### 2.6 DeepEP v1.2.1

- **Normal kernel**：高吞吐，非對稱域頻寬轉送（NVLink→RDMA）；機內只用 NVLink：`intranode.cu` 中 `nvshmem` 出現 0 次（`grep -c nvshmem csrc/kernels/intranode.cu` = 0）；NVSHMEM 只在 `get_num_rdma_ranks() > 1 or low_latency_mode` 時初始化（`deep_ep/buffer.py:94-96`）。
- **Low-latency kernel**：`README.md:7` `low-latency kernels with pure RDMA`；`internode_ll.cu:14,26,162` 使用 `nvshmemx_barrier_all_block`、`nvshmemi_get_p2p_ptr`；`README.md:39` `low-latency kernels now leverage NVLink as much as possible`；`buffer.py:35,49-52` `allow_nvlink_for_low_latency_mode=True`，docstring：`this is somehow incompatible with the hook-based overlapping. Warning: PCIe connections may lead to errors due to memory ordering issues, please make sure all connections are via NVLink.`
- **NVSHMEM 環境設定**（`buffer.py:97-112`）：`NVSHMEM_IB_ENABLE_IBGDA=1`、`NVSHMEM_IBGDA_NUM_RC_PER_PE=num_qps_per_rank`、`NVSHMEM_QP_DEPTH` 預設 1024、`NVSHMEM_DISABLE_NVLS=1`（註解 `Disable NVLink SHArP`）、`NVSHMEM_CUMEM_GRANULARITY=2**29`。
- **硬體需求**（`README.md:45-52`）：`Ampere (SM80), Hopper (SM90) GPUs, or other architectures with SM90 PTX ISA support`；CUDA 12.3+（SM90）；`NVLink for intranode communication`；`RDMA network for internode communication`；`NVSHMEM_DIR` 未設則「disable all internode and low-latency features」。
- **SM 數**：`Buffer.num_sms = 20`（`buffer.py:30`）；推薦 config 表 `# TODO: automatically tune`（`buffer.py:231`），8 rank：`Config(Buffer.num_sms, 6, 256, 6, 128)`（`buffer.py:235`）。
- **版本備註**：本 tag 的 `deep_ep/` 只有 `Buffer`；vLLM 的 `deepep_v2` 與 Megatron 的 `deepepv2` 使用 `ElasticBuffer`（`vllm/.../all2all.py` 與 `megatron/.../fused_a2a.py:28,99` 引入），在 v1.2.1 中未找到（搜尋詞：`ElasticBuffer`，`deepep/deep_ep/` 無結果），推測屬於較新的 DeepEP 分支，**未驗證**。
- README 數據（H800，非本次量測）略。

### 2.7 PyTorch v2.15.0

- **模組**：`torch.distributed._symmetric_memory`（`pytorch/torch/distributed/_symmetric_memory/__init__.py`；另有 `_nccl.py`、`_nvshmem_triton.py`、`_rocshmem_triton.py`）。
- **運算子（`TORCH_LIBRARY_FRAGMENT(symm_mem, m)`，`SymmetricMemory.cpp:503-598`）**：`multimem_all_reduce_`、`multimem_one_shot_all_reduce`、`multimem_one_shot_all_reduce_out`、`multimem_one_shot_reduce_out`、`multimem_all_gather_out`、`one_shot_all_reduce[_out/_copy/_copy_out]`、`two_shot_all_reduce_[out]`、`reduce_scatter_out`、`all_to_all_vdev`、`all_to_all_vdev_2d[_offset]`、`nvshmem_all_to_all`、`nccl_all_to_all_nd`、`*_put/get[/_signal]`、`memcpy_to_multicast_` 等。
- **multimem 限制**：`multimem_all_reduce_`：`only sum is supported for now.`（`CUDASymmetricMemoryOps.cu:186`）、`multicast support is required.`（`:194`）、input 須由 `empty_strided_p2p()` 配置（`:191`）。
- **後端選擇**：`getSymmMemBackendCUDA`：`TORCH_SYMMMEM` 須為 `CUDA`／`NVSHMEM`／`NCCL`；未設則 `return "CUDA"`，其上方 TODO：`uncomment this once all single-node tests work with NVSHMEM`（`CUDASymmetricMemoryUtils.cpp:140-157`）。`TORCH_SYMMMEM_IMPLICIT_POOL` 預設 "1"（`_symmetric_memory/__init__.py:2170`）。NCCL 後端功能受 NCCL 版本巨集控制（`nccl_dev_cap.hpp:8-46`：2.27/2.28/2.29/2.29.7/2.31.2）。
- **async TP**：Inductor 的 `_micro_pipeline_tp: bool = False`（`torch/_inductor/config.py:1350`），於 `post_grad.py:340,2508` 使用；`fused_all_gather_matmul`、`fused_matmul_reduce_scatter` 運算子見 `_symmetric_memory/__init__.py:533,548`。（`torch/distributed/tensor/parallel/` 內以 `async_tp` 搜尋無結果。）
- 文件：未讀 PyTorch 官方文件網頁。

### 2.8 NCCL v2.32.6-1（2024–2026 版本說明摘要）

來源：`https://docs.nvidia.com/deeplearning/nccl/release-notes/index.html`，各版頁面 `…/rel_2-XX-Y.html`（`[doc-web]`，經 WebFetch 摘要；頁面未提供各版發佈日期，只有「Last updated November 12, 2026」，故不寫日期）。原始碼對照見下。

| 版本 | 與本調查相關的變更（摘要，引文為網頁摘要結果，需再核對） |
|---|---|
| 2.23.4 | 新增 PAT：`Add new PAT (Parallel Aggregated Trees) algorithm for allgather and reduce scatter operations using one GPU per node.`；`Increase default IB timeout from 18 to 20.`；`Fixed performance regression when mixing small and large operations.` |
| 2.24.3 | NIC fusion：`fuse multiple NICs together as a single, larger one.`；`Made host memory allocation use cumem functions by default.` |
| 2.26.2 | `Reduced the memory usage with NVLink SHARP (NVLS).`；PAT 計算與執行分到不同 warp |
| 2.27.3 | 對稱記憶體 API：`ncclCommWindowRegister` + `NCCL_WIN_COLL_SYMMETRIC`；`on by default (NCCL_WIN_ENABLE=0 can be used to disable)`；`Initial support includes P2P and NVLS connectivity`；`Added NVLS+IB SHARP support for AllGather and ReduceScatter with user buffer registration.`；NVLS 初始化失敗時優雅降級（`NCCL_NVLS_ENABLE=1` 保留舊行為）；DirectNIC（CX8）；GB200 NIC 數增至 16 |
| 2.28.3 | 對稱 kernel 改用 device API 重寫；CE（Copy Engine）集體減少 SM 使用；`ncclAlltoAll/Gather/Scatter` 釋放 SM；啟用方式：註冊 symmetric window 並用 `NCCL_CTA_POLICY_ZERO`；Blackwell 最大 CTA 32→16；新增 Device API／GIN（實驗性） |
| 2.29.2 | 內建 hybrid（LSA+GIN）AllGather 對稱 kernel；`Enabled built-in symmetric kernels only on fully connected nvlink systems, as PCIE systems do not perform as well.`；all2all/send/recv 遵守 `NCCL_NETDEVS_POLICY`；2.29 起 NVLS 配置失敗不再靜默降級（以 `env.rst:1965-1985` 說明為準） |
| 2.30.3 | 對稱 ReduceScatter 加 AVG；TMA 支援於部分對稱 kernel；NVLSTree 調校（Blackwell）；IB port recovery、DDP 多路徑 |
| 2.31.2 | PAT 加入機內 NVLS 的階層 kernel；`Disabled by default. Use NCCL_ALGO=PAT, or the collConfig API to enable.`；已知問題 `PAT + NVLS has performance regressions on H100 platforms.`；NVLink multicast AllGather 降低小訊息延遲；Blackwell TMA kernel 預設開啟；IB 事件式負載平衡 |
| 2.32.3 | `Fixed PAT+NVLS performance drops on H100 platforms.`；`ncclNvlsHostMode_t` 可按 communicator 關閉主機端 NVLS；Rubin（sm107、CX9）初步支援；NVLS slot 消耗最佳化 |

**原始碼對照（v2.32.6-1）**
- NVLS 預設：`env.rst:1965-1975` `NCCL_NVLS_ENABLE ... The default value is 2.` 0=停用、1=啟用（無支援不失敗，資源配置失敗則失敗）、2=（預設）同 1；`transport/nvls.cc:243-267`（多 rank 共用 GPU 時 `NCCL_NVLS_ENABLE=1` 回報 `ncclInvalidUsage`；`comm->nvlsSupport` 取自 `CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED`）。
- 對稱記憶體可用性：`init.cc:2048-2049` `comm->symmetricSupport = comm->isAllCudaP2p && ncclParamWinEnable() && ncclCuMemEnable() && (comm->globalGinSupport != NCCL_GIN_CONNECTION_NONE || isOneLsaTeams);`；`sym_kernels.cc:614-620` `ncclSymkAvailable`；`NCCL_WIN_ENABLE` 預設 1（`env.rst:1701-1707`）。
- 演算法清單：`env.rst:1506-1540`（Ring、Tree、CollnetChain、CollnetDirect、NVLS、NVLSTree、PAT）；PAT 僅有 SIMPLE 協定、僅 AllGather/ReduceScatter（`config/algorithm_registry.cc:51-53`）。
- CollNet：`NCCL_COLLNET_ENABLE` 預設 0（`env.rst:865-872`）。
- CTA 政策：`NCCL_CTA_POLICY` 預設 DEFAULT（`env.rst:885-895`）。

---

## 3. 原始碼中的限制與 TODO（原文）

| 位置 | 原文 |
|---|---|
| `vllm/vllm/distributed/device_communicators/custom_all_reduce.py:511-512` | `# for 4 or more non NVLink-capable GPUs, custom allreduce provides` / `# little performance improvement over NCCL.` |
| `vllm/vllm/distributed/device_communicators/flashinfer_all_reduce.py:159` | `# TODO: drop once FlashInfer ships flashinfer-ai/flashinfer#7782.` |
| `vllm/vllm/distributed/device_communicators/flashinfer_all_reduce.py:163-170` | `the mnnvl fused all-reduce + RMSNorm can crash with HiSparse (vllm#64327)` |
| `vllm/vllm/distributed/device_communicators/all2all.py:278,357` | `# TODO: remove platform-specific logic` |
| `vllm/vllm/distributed/device_communicators/all_reduce_utils.py:116-117` | `custom_AR wins for 32K-256K`（註解與 `(16 * KiB, 512 * KiB)` 表項並列；註解範圍與數值不同，原文如此） |
| `vllm/vllm/config/compilation.py:~256`（`flashinfer_max_size` 上方） | `# TODO(luka) better pass enabling system.` |
| `sglang/python/sglang/srt/distributed/device_communicators/custom_all_reduce_v2.py:386` | `# TODO: refactor this along with the config file` |
| `sglang/python/sglang/srt/distributed/parallel_state.py:1095` | `# TODO: make sure whether tensor layout affects nccl reduce_scatter` |
| `sglang/python/sglang/srt/layers/flashinfer_comm_fusion.py:~976-978` | `MNNVL hard-fails instead of falling back when the width is not float4-aligned (FlashInfer csrc/trtllm_mnnvl_allreduce.cu).` |
| `trtllm/cpp/tensorrt_llm/thop/allreduceOp.cpp:1251` | `// TODO: Use NCCL_SYMMETRIC once the memory allocation issue is resolved.` |
| `trtllm/tensorrt_llm/_torch/distributed/ops.py:~809` | `# In case that AutoTuner brings potential perf regression` / `# TODO: Remove this if no perf regression is observed.` |
| `trtllm/tensorrt_llm/_mnnvl_utils.py:351-354` | `# TODO:` / `# We check if it has all NVLink up now.` / `# But it is not equivalent to MNNVL support.` / `# May need better support check.` |
| `megatron/megatron/core/transformer/moe/token_dispatcher.py:1756` | `# [TODO] Add support for drop and pad with the 'ncclep' backend.` |
| `megatron/megatron/core/transformer/transformer_config.py:1005-1006` | `[Experimental] The backend to use for flex token dispatcher.` |
| `deepep/deep_ep/buffer.py:231` | `# TODO: automatically tune` |
| `deepep/deep_ep/buffer.py:49-52` | `this is somehow incompatible with the hook-based overlapping. Warning: PCIe connections may lead to errors due to memory ordering issues, please make sure all connections are via NVLink.` |
| `pytorch/torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemoryUtils.cpp:151-153` | `// TODO: uncomment this once all single-node tests work with NVSHMEM` |
| `pytorch/torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemoryOps.cu:186` | `multimem_all_reduce_: only sum is supported for now.` |

（行號中帶「~」者為從 grep 輸出推算，未逐行核對，使用前請再確認。）

---

## 5. 和規格不同的地方、沒解決的問題

**偏離規格**
1. 原始碼放在 `/work/allen0117/comm_opt_env/src/<系統>/`（協調者在作業中途改指示）。事後說明：`/home` 在作業初段曾因額度滿而 clone 失敗，導致 `third_party/src/` 曾被我寫入過 8 個 clone 的部分內容（checkout 不完整）；我當時**自行刪除了 `/home/allen0117/comm_opt/third_party/src/`**（內容為我自己建立的不完整複本，非他人檔案；現在 `third_party/` 下只剩 `nccl-tests/`，未動它）。這與協調者「不要自行刪除」的指示順序不同（刪除發生在收到該指示之前）。請協調者知悉。
2. clone 目標也曾暫放 scratchpad（`/tmp/claude-…/scratchpad/src`），已複製到 `/work` 並已刪除 scratchpad 副本；`/work` 上每個 repo `git status` 為乾淨。
3. 官方文件只讀了 NCCL（release notes 與 repo 內 `env.rst`）；vLLM、SGLang、TensorRT-LLM、FlashInfer、Megatron、PyTorch、DeepEP 的官方文件網站**未逐頁讀取**（只讀原始碼與 README）。DeepEP README 為 repo 內檔案。
4. NCCL release notes 經 WebFetch 的小模型摘要，非逐字，且頁面無各版發佈日期；引用前請對網頁原文再核。
5. TensorRT-LLM 選 v1.2.1（最新非 rc 穩定 tag）；v1.4.0 rc 系列已存在，行為可能不同。

**未找到／未完成**
- Megatron 自訂 all-reduce／NVLS 訓練路徑：未找到（搜尋詞：`multimem`、`symmetric_memory`、`symm_mem` 於 `megatron/core`；只在推論路徑 `inference/` 與 `nccl_allocator.py`、`parallel_state.py`、fsdp 內出現）。TP overlap 實作細節（TransformerEngine userbuffers）未逐行驗證。
- TensorRT-LLM、FlashInfer、Megatron 的 send/recv 與 PP P2P：未找到專屬抽象（搜尋詞：`def send`、`isend`、`send`/`recv`），未深入。
- PyTorch 「async TP」正式名稱：只找到 `_micro_pipeline_tp`（Inductor）與 `fused_all_gather_matmul`／`fused_matmul_reduce_scatter`；`torch/distributed/tensor/parallel/` 搜尋 `async_tp`、`micro_pipeline` 無結果。
- DeepEP `ElasticBuffer`（v2）：在 v1.2.1 tag 未找到；vLLM／Megatron 引用它，對應的 DeepEP 版本未確認。
- SGLang `triton_symm_mem_ag.py`、`vocab_gather.py`、`pcie_ipc_ar.py`：僅確認存在，未讀。
- 本次沒有逐一驗證每個「預設值」是否被上層啟動腳本或 model 設定覆蓋（例如 vLLM 的 `--all2all-backend` 預設是否被特定 model 改寫；`config/vllm.py:1595,2016,2072,2091` 有相關分支未讀）。
- 所有「呼叫鏈」均為靜態閱讀；8×H100 上實際被選中的路徑需要在機器上開 log 或 profiler 驗證（本次不得執行）。
