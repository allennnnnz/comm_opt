# R08_B：程式碼與我們環境的可行性（S08 分工 B）

- 撰寫：Sonnet 5.5，2026-11-25。只寫事實，不做選題判斷。不編譯、不安裝、不送 Slurm。
- 路徑簡寫：`V/` = `/work/allen0117/comm_opt_env/venvs/vllm/lib/python3.12/site-packages/vllm/`（版本檔 `V/_version.py` 寫 `0.32.0`）；`SGL/` = `/work/allen0117/comm_opt_env/src/sglang/python/sglang/srt/`（tag v0.5.23）；`TRT/` = `/work/allen0117/comm_opt_env/src/trtllm/tensorrt_llm/`（tag v1.2.1）；`OV/` = `/work/allen0117/comm_opt_env/src/overlap/`。
- 環境前提（來自 S08 與 M25 卡）：nvcc 最高 12.6、驅動 550.127.08（CUDA 12.4）、FlashInfer JIT 需 nvcc ≥ 12.8 所以 M25 關閉 FlashInfer（`exp/M25_topic_survey/README.md:87`）。venv 內 torch 2.13.0+cu129、nvidia-nccl 2.29.7、nvshmem 3.4.5、flashinfer-python 0.7.0.post1、triton 3.7.1、nvidia-cutlass-dsl 4.8.0。

## 白話結論

1. **vLLM 0.32.0 沒有「序列平行（SP）」和「非同步 TP（async TP）」這兩個編譯 pass**：`V/compilation/passes/fusion/` 只有 RMSNorm／量化／rope／allreduce+RMSNorm 等融合；對 `sequence_parallelism|async_tp|enable_sp|fuse_gemm_comms|fused_all_gather_matmul|gemm_reduce_scatter` 的全 venv grep 沒有命中（dense TP 路徑上）。連 `src/vllm`（v0.34.0）也沒有。也就是說 vLLM 目前對 dense 模型的 TP all-reduce 與 GEMM 沒有任何重疊功能。
2. **唯一的 TP 通訊融合是 AllReduce＋RMSNorm（`fuse_allreduce_rms`），而且綁在 FlashInfer**；我們環境因 FlashInfer JIT 要 nvcc ≥ 12.8 而不能用（有依據，見 1.2）。
3. **DBO（dual batch overlap）只在 MoE 專家平行的 all-to-all 路徑有切換點**，要求 DeepEP／nixl_ep 後端；dense 模型的 TP all-reduce 路徑上沒有任何 DBO 切換點，所以 DBO 不會讓 TP all-reduce 與計算重疊。
4. 論文程式碼：只有 TokenWeave（vLLM 分支，torch 2.6／CUDA 12.4／Python 3.12）和 ThunderKittens／ParallelKittens（需 CUDA 12.8+）明確針對 H100 做過 TP 推論或 GEMM+AR 融合；FlashOverlap 只支援 sm_80/86/89；NanoFlow 要 Gurobi 授權、docker、root 式安裝；Syncopate 要自訂 Triton-distributed 分支；ISO 找不到程式碼。多數需要改版本或整個換環境（見 3）。

---

## 1. vLLM 0.32.0 的重疊／融合功能（以 venv 內原始碼為準）

### 1.1 功能總表

| # | 功能 | 怎麼開 | 預設 | 預填／解碼 | 依賴 | 依據 `檔案:行號` |
|---|---|---|---|---|---|---|
| F1 | AllReduce＋RMSNorm（＋量化）融合（FlashInfer allreduce fusion） | `--compilation-config` 的 `pass_config.fuse_allreduce_rms`（`PassConfig` 欄位，註解寫 "Enable flashinfer allreduce fusion"） | 優化等級 O0／O1 為 False；**O2／O3 為 `enable_allreduce_rms_fusion(cfg)`**：TP>1、CUDA、`has_flashinfer()`、capability 100 家族或 90 | 兩者皆可：只在 token 數對應的通訊量 ≤ 門檻時啟用；SM90 TP8 預設上限表為 0.5 MB（`allreduce_rms_fusion.py` 的 `FI_ALLREDUCE_FUSION_MAX_SIZE_MB[90][8]`），`PassConfig` 註解寫 8 卡為 1 MB（兩處不一致，以程式表為準）。可用 `fi_allreduce_fusion_max_size_mb` 覆寫。換算：hidden=8192、bf16 時 0.5 MB ≈ 32 token，所以大型預填步（8k–17k token）超過門檻、不會融合（推算，非實測） | torch.compile（VLLM_COMPILE 模式）、FlashInfer（`flashinfer.comm.allreduce_fusion`）、`has_flashinfer()` | `V/config/compilation.py:136`（欄位）、`:~160-180`（門檻註解）；`V/config/vllm.py:171-195`（`enable_allreduce_rms_fusion`）、`:252,273,294,315`（各 O 等級）；`V/config/vllm.py:2563-2584`（compile range 依門檻切分）；`V/compilation/passes/pass_manager.py:167`；`V/compilation/passes/fusion/allreduce_rms_fusion.py:102-157`（門檻表）、`:87-97`（flashinfer_comm 匯入） |
| F2 | 自製 all-reduce（custom all-reduce）／symmetric memory／NCCL symmetric／FlashInfer all-reduce（單純 all-reduce，不融合計算） | `--disable-custom-all-reduce`（關閉自製）；`VLLM_ALLREDUCE_USE_SYMM_MEM`（預設 1）；`VLLM_ALLREDUCE_USE_FLASHINFER`（預設 1）；`VLLM_USE_NCCL_SYMM_MEM`（預設 0）；`VLLM_DISABLE_PYNCCL`（預設 False） | 見左 | 兩者皆可；與計算依序執行（同 stream），**不是重疊功能**（是 TP all-reduce 本身的實作選擇） | 選擇順序：NCCL symm（需開）→ quick AR → FlashInfer PCIe IPC → FlashInfer AR → ca_comm（自製）→ symm_mem → pynccl | `V/envs.py:277-279,302`；`V/distributed/device_communicators/cuda_communicator.py:362-445`（選擇順序）；`symm_mem.py:41-133`（`SymmMemCommunicator`，SM90 world size 4/6/8 用 multimem，上限 64 MiB：`all_reduce_utils.py:58-64`） |
| F3 | Dual batch overlap（DBO，兩個 microbatch 交錯計算與通訊） | `--enable-dbo`（或 `--ubatch-size N`）；`--dbo-decode-token-threshold`（預設 32）、`--dbo-prefill-token-threshold`（預設 512，註解 "TODO(lucas): tune"）；`VLLM_DBO_COMM_SMS`（預設 20，CUDA） | 關（`enable_dbo=False`、`ubatch_size=0`） | 兩者皆有門檻（解碼批次 token ≥32、含預填 ≥512 才切） | **要求 `--all2all-backend` ∈ {deepep_low_latency, deepep_high_throughput, nixl_ep}**（否則 assert）；V2 model runner 需明確 `VLLM_USE_V2_MODEL_RUNNER`；V2 路徑不支援 LoRA／spec decode／PP／CP／多模態／hybrid／PIECEWISE cudagraph。切換點 `dbo_yield*`／`dbo_switch*` 只出現在 `fused_moe/modular_kernel.py` 與 `prepare_finalize/deepep_ht.py／deepep_ll.py／nixl_ep.py`；在 `model_executor/layers`、`models` 的 dense 路徑（`linear.py`、`qwen2.py`、`llama.py`）grep 無命中 | `V/config/parallel.py:257-270,641-646`；`V/engine/arg_utils.py:561-565,1288-1314`；`V/config/vllm.py:2104-2116`（backend assert）、`:3085-3123`（`_get_dbo_unsupported_features`，`:3095` V2 環境變數檢查）；`V/envs.py:297,2078-2087`；`V/v1/worker/ubatch_utils.py:87,140-148`；`V/v1/worker/ubatching.py:168-184` |
| F4 | MoE 共享專家用輔助 stream 與路由專家重疊 | `VLLM_DISABLE_SHARED_EXPERTS_STREAM`（預設 0＝啟用）；`VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD`（預設 256） | 開 | 兩者；只對有共享專家的 MoE | `aux_stream()` | `V/envs.py:306,308,2114-2128`；`V/utils/torch_utils.py:864-875`。Qwen3-30B-A3B 是否有共享專家：**未核對** |
| F5 | MoE 的 TP 序列平行（`use_sequence_parallel_moe`；DP-attention＋EP 時按 token 切塊） | 由 `all2all_backend` 與 DP／EP 設定決定，非獨立旗標 | 視設定 | 兩者；屬 MoE 專家平行，非 TP dense | 見 `all2all_backend` 清單 | `V/config/parallel.py:219-233,764-773`；`V/model_executor/models/utils.py:1080-1117`（`sequence_parallel_chunk`） |
| F6 | GEMM＋ReduceScatter／AllReduce 融合 kernel（CuTe DSL，symmetric memory） | 模型專用（Kimi-K3、DeepSeek-V4.1 等）；`maybe_init_gemm_rs_ar` 檢查條件 | 模型相關 | 檔頭註明 SM100 BF16／MXFP8 | **SM100 專用**：`current_platform.is_device_capability_family(100)` 不成立即停用；ubatching 開啟也停用；需 nvidia-cutlass-dsl | `V/model_executor/kernels/linear/cute_dsl/gemm_rs_ar.py:1-10,1106-1125`；`V/models/kimi_k3/nvidia/model.py:173-189` |
| F7 | 其他融合（非 TP 通訊）：RMSNorm＋quant、SiluMul＋quant、attention＋quant、QK-norm＋RoPE、RoPE＋KV cache | `pass_config.fuse_norm_quant／fuse_act_quant／fuse_attn_quant／enable_qk_norm_rope_fusion／fuse_rope_kvcache` 等 | 依 O 等級 | 兩者 | — | `V/config/compilation.py:128-146`；`V/config/vllm.py:244-322` |

### 1.2 依我們環境判斷

| 功能 | 判斷 | 依據 |
|---|---|---|
| F1 `fuse_allreduce_rms` | **不能開（有證據）** | (a) `enable_allreduce_rms_fusion` 要 `has_flashinfer()`（`V/config/vllm.py:187-195`）；`has_flashinfer()` 在沒有 flashinfer-cubin、且找不到 nvcc 或 ninja 時回 False（`V/utils/flashinfer.py:88-106`）。venv 內只有 `flashinfer_python-0.7.0.post1`，site-packages 內無 flashinfer-cubin／jit-cache 套件（`ls site-packages` 篩選結果）；登入節點 PATH 無 nvcc。(b) allreduce 融合的 `trtllm` 後端要現場編譯：`flashinfer/comm/trtllm_ar.py:120` 呼叫 `gen_trtllm_comm_module().build_and_load()`（`flashinfer/jit/comm.py:190-201`），編譯旗標含 `--compress-mode=size`（`flashinfer/jit/core.py:693`），該旗標需 nvcc ≥ 12.8（M25 卡的說法，`exp/M25_topic_survey/README.md:87`；本報告未獨立核對 nvcc 版本表）。(c) 驗證工作 497896 因此失敗（同卡）。 |
| F2 各 all-reduce 實作 | **能用**（M25 已用） | M25 用 `VLLM_ALLREDUCE_USE_FLASHINFER=0`，依序走自製 CUSTOM、SYMM_MEM、PYNCCL（同卡）。`SymmMemCommunicator` 需 multicast（`symm_mem.py` 的 `handle.multicast_ptr == 0` 即停用）；M25 已量到 NVLS 可用。`VLLM_USE_NCCL_SYMM_MEM=1` 未在 M25 試過：**不確定** |
| F3 DBO | **dense TP 不適用；MoE 不確定** | dense：DBO 切換點只在 MoE 路徑（見上），且要求 DeepEP／nixl_ep all2all 後端。MoE：DeepEP 需 `src/deepep`（v1.2.1）編譯與 NVSHMEM；是否能在 nvcc 12.6／驅動 550 編譯：**未核對**。DBO 另需 `VLLM_USE_V2_MODEL_RUNNER` 明確設定（`V/config/vllm.py:3095`） |
| F4 | **能開**（預設就開；只與有共享專家的 MoE 相關） | 純 Python／torch stream，無 nvcc 依賴 |
| F6 | **不能**（Hopper） | SM100 專用，`gemm_rs_ar.py:1124-1125` |
| SP／async TP（pass） | **0.32.0 不存在**，無從開啟 | 見白話結論 1 的 grep；`V/compilation/passes/pass_manager.py:140-216` 列出的 pass 沒有 SP／async TP。PyTorch 側有零件：`torch/distributed/_symmetric_memory/__init__.py:853,1206`（`_fused_all_gather_matmul`、`_fused_matmul_reduce_scatter`）與 `torch/_inductor/config.py:1168`（`_micro_pipeline_tp: bool = False`、`torch/_inductor/fx_passes/micro_pipeline_tp.py`），但 vLLM 原始碼 grep `async_tp|AsyncTP|fused_matmul_reduce_scatter|all_gather_matmul|micro_pipeline` 無命中（唯一例外：flashinfer 自帶 `all_gather_matmul`，見 1.3） |

### 1.3 補充：venv 內 FlashInfer 自帶的 TP 相關 kernel（vLLM 未使用）

- `flashinfer/comm/all_gather_matmul/`：`all_gather_matmul`（push-wait）：SM ≥ 100 走 cuTile，**SM < 100 走 Triton 實作**（`all_gather_matmul.py` 的 `backend="auto"` 路由；`all_gather_matmul_triton.py`）。`vllm` 內無引用（`grep all_gather_matmul` 於 `V/` 無命中）。Triton 路徑不需 nvcc（Triton 自帶 ptxas，未實測）；是否能在 sm_90 正確運行：**未驗證**。
- `flashinfer/comm/mnnvl*`、`trtllm_ar.py` 等 all-reduce 後端的 JIT 同 F1 的限制。

---

## 2. SGLang 與 TensorRT-LLM 的同類功能（只列清單）

### SGLang（v0.5.23）

| 功能 | 旗標／欄位 | 依據 | 備註（事實） |
|---|---|---|---|
| Two-batch overlap（TBO） | `enable_two_batch_overlap`、`tbo_token_distribution_threshold`（預設 0.48） | `SGL/arg_groups/fields/exec_.py:1047-1058` | 只有 DeepseekV2／Qwen3Moe／MiMoV2／DeepseekV4 的 decoder layer 有策略，其餘 `raise NotImplementedError`（`SGL/batch_overlap/operations_strategy.py:34-78`）；依 a2a 後端（DeepEP 等）分派（`two_batch_overlap.py:19-27`）。dense 模型無 |
| Single-batch overlap（SBO） | `enable_single_batch_overlap` | `exec_.py:1051`；`SGL/batch_overlap/single_batch_overlap.py`；與 `deepep_v2` 互斥（`SGL/arg_groups/moe_hook.py:420-425`） | MoE 為主 |
| FlashInfer allreduce fusion | `flashinfer_allreduce_fusion_backend`（auto／trtllm／mnnvl／cutedsl；舊旗標 `--enable-flashinfer-allreduce-fusion` 已棄用）；`enforce_disable_flashinfer_allreduce_fusion` | `exec_.py:765-790`；`SGL/arg_groups/serving_hook.py:285-302`；`SGL/layers/flashinfer_comm_fusion.py` | 說明寫 "Requires SM90 or SM10X"，"Fuses allreduce with Residual + RMSNorm for supported MoE models" |
| 自製 all-reduce／symm mem／NVLS／MSCCL++ | `disable_custom_all_reduce`、`enable_symm_mem`（NCCL symmetric）、`enable_torch_symm_mem`（SM90 world size 4/6/8）、`enable_nccl_nvls`（"for prefill heavy requests"）、`enable_mscclpp` | `exec_.py:715-738` | `enable_nccl_nvls` 說明直接寫給預填 |
| LayerNorm 序列平行（SP，Megatron 式） | 模組 `layernorm_sp`；`initialize_layernorm_sp` | `SGL/layers/layernorm_sp.py:13`；`SGL/runtime_context.py:615,701`；用於 `SGL/models/muse_spark.py:1161-1328` | 目前只見 muse_spark 模型使用 |
| SP 集體 GEMM（all_gather_matmul／matmul_reduce_scatter） | `sglang/kernels/ops/communication/sp_collective_gemm.py`；工作區 `SGL/layers/communication/sp_collective_gemm_workspace.py` | 檔頭 "GEMM collectives for Blackwell prefill sequence parallelism"；`sp_collective_gemm.py:99` 檢查 `get_device_capability(...)[0] != 10` | **Blackwell 專用** |
| 其他 | `enable_attn_tp_input_scattered`（TP-only 時分散 attention 輸入）、`enable_dp_attention`、`enable_dp_lm_head`、`muse_spark_gemm_ag.py`（解碼 GEMM＋all-gather，multicast） | `SGL/arg_groups/fields/parallel.py:194-227`；`SGL/layers/communication/muse_spark_gemm_ag.py:1-10`；`k3_sp_collective.py:1-6`（GB200/GB300 調校） | |

### TensorRT-LLM（v1.2.1）

| 功能 | 旗標／欄位 | 依據 | 備註 |
|---|---|---|---|
| AllReduce 策略 | `allreduce_strategy`：AUTO／NCCL／UB／MINLATENCY／ONESHOT／TWOSHOT／LOWPRECISION／MNNVL／NCCL_SYMMETRIC（預設 AUTO，status=beta） | `TRT/llmapi/llm_args.py:2905-2910`；`TRT/functional.py:3874-3884`（enum 另有 SYMM_MEM=9） | 為 all-reduce 實作選擇 |
| AllReduce＋殘差＋RMSNorm（＋FP8／NVFP4 量化、MoE finalize）融合 | `AllReduceFusionOp` | `TRT/functional.py:3887-3896`；`TRT/_torch/compilation/patterns/ar_residual_norm.py`（torch.compile pass）；`TRT/_torch/compilation/backend.py:68-84` | 多卡時註解寫 torch compile 不能與 lamport fusion kernel 共用而設 `DISABLE_LAMPORT_REDUCE_NORM_FUSION=1`（`backend.py:73`） |
| Userbuffers（UB） | `torch_compile_config.enable_userbuffers`（預設 True，需 torch compile 開啟） | `TRT/llmapi/llm_args.py:2744-2748`；`backend.py:77`（`ub_supported()`）；`ar_residual_norm.py:447-545` | |
| GEMM＋AllReduce 融合（NVFP4） | 環境變數 `TRTLLM_GEMM_ALLREDUCE_FUSION_ENABLED=1`，且 `get_sm_version() >= 100`、NVLS 支援、NVFP4、row-parallel | `TRT/_torch/modules/linear.py:2160-2178` | **Blackwell 專用** |
| 多 stream（torch.compile） | `torch_compile_config.max_num_streams`（預設 1） | `llm_args.py:2751-2760` | |
| Overlap scheduler | `disable_overlap_scheduler`（預設 False） | `llm_args.py:2792` | 名稱為排程器重疊（CPU 排程與 GPU 執行）；是否涉及 TP 通訊重疊：**未核對** |
| Attention DP | `enable_attention_dp` | `llm_args.py:1965` | |

---

## 3. 論文程式碼可行性（git clone --depth 1 於 `OV/<名稱>/`；只讀未編譯）

注意：淺層 clone 只能知道最後一個 commit 日期，無法算 commit 頻率；「是否仍在維護」只以最後 commit 日期描述。多數 repo 的 submodule（cutlass、nccl、flashinfer…）在 `--depth 1` 下**未展開**（目錄空），其版本以 `git ls-tree` 的 commit 號記錄。

### 3.1 總表

| 論文／程式碼 | 路徑（`OV/`） | 最後 commit | 需求（原文／檔案） | 對 vLLM 整合 | 預編譯 wheel | 對我們環境判斷 |
|---|---|---|---|---|---|---|
| FLUX（bytedance/flux） | `flux` | 2025-08-28（`19831ca`） | `README.md:46-77`：Hopper 用 `./build.sh --arch 90 --nvshmem`；範例環境 CUDA 12.4、torch 2.6.0、Python 3.11；NVSHMEM "tested with 3.2.5/3.3.9"（`README.md:93`）；submodule cutlass `df8a550d`、nccl `8c6c5951`（`git ls-tree`）；`CMakeLists.txt:41-48`：CUDA ≥ 12.4 預設 arch 80;89;90 | **無**：repo 內 grep "vllm"（不分大小寫）無命中；測試腳本在 `test/python/{ag_gemm,gemm_rs,...}` | PyPI `byte-flux`：torch 2.4／2.5／2.6、Python 3.10／3.11、CUDA 12.4（`README.md:76-82`） | **需要改版本**：我們 torch 2.13、Python 3.12 與 wheel 不符；從原始碼編譯需 nvcc ≥ 12.4（12.6 滿足文件下限）＋ torch 2.13 ABI 相容性未驗證。只有算子層 benchmark，沒有端到端推論整合。需另建 venv |
| TokenWeave（microsoft/tokenweave） | `tokenweave` | 2026-01-29（`f483d1c`） | `README.md:31-33`：編譯 CUDA 12.4、執行 Python 3.12／PyTorch 2.6.0／Ubuntu 22.04、8×H100 NVLink；`requirements/cuda.txt` 釘 `torch==2.6.0`；README 稱源自 vLLM 分支 commit 87aaade（`README.md:135`）；multimem kernel 以 PyTorch CUDASymmetricMemoryOps 為底（`csrc/tokenweave_multimem_utils.cuh:13`），`csrc/tokenweave_fused_kernels.cu:18` 要求 `__CUDA_ARCH__ >= 900` 且 `CUDART_VERSION >= 12010`；`CMakeLists.txt:34` 含 9.0；`README.md:49` 編譯約 18 分鐘 | **本身就是 vLLM 的完整 fork**（≈v0.8.5 時代，Docker 建議 `vllm/vllm-openai:v0.8.5`）；整合 Llama-3.3-70B、Qwen2.5-72B、Mixtral-8x22B（`README.md:18`） | 未見 | **看起來可行但需獨立環境**：nvcc 12.6 ≥ 12.4、sm_90 正好是目標；需 torch 2.6.0 的獨立 venv、重編整個 vLLM fork（`README.md:49`）。cuMulticast 需驅動支援（`vmm` 類 API）：驅動 550（12.4）夠用性**未驗證**，另 M20 曾在 NCCL 建 512 MB multicast 群組時遇 `Cuda failure 999`（`docs/status.md` 2026-11-17 12:30 段），NVLS 在 vLLM 內的 symm_mem 路徑已由 M25 確認可用 |
| NanoFlow（efeslab/Nanoflow） | `nanoflow` | 2025-09-17（`f179a90`） | `README.md:83-92`：Docker `nvcr.io/nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04`、`apt install` pybind11／liburing／openmpi；需 **Gurobi 授權**（`README.md:97-114`）；`setup.sh`：torch 2.7.1+cu128、`dpkg -i` 安裝 Nsight、安裝 cmake 到 `/usr/local`、從原始碼編 FlashInfer 子模組；submodule：cutlass、flashinfer、mscclpp、aiter（`.gitmodules`）；官方效能數據在 A100（`README.md:50`）；`entry/test_multi_gpu_tp*.sh` 以 `TORCH_CUDA_ARCH_LIST="9.0"` 執行（H100 路徑存在） | 自成一套 C++ 後端，非 vLLM 整合 | 未見 | **需要改版本且有阻礙**：要 Docker／root 式安裝（dpkg、/usr/local）、Gurobi 授權、CUDA 12.8 與 FlashInfer 自編（會碰我們 nvcc 12.6 的限制）。不可行或需大改 |
| Syncopate（tie-pilot-qxw/syncopate） | `syncopate` | 2026-07-19（`7892aaf`） | `README.md`：AE commit `ce2b13e4`；`Dockerfile`：基底 `nvcr.io/nvidia/pytorch:25.06-py3`、nvshmem-cu12 3.3.9、cuda.core 0.2.0、**自訂 Triton-distributed 分支 `merge-tlx`（tie-pilot-qxw/Triton-distributed）並重編 Triton**；實驗在 H100／H800 多卡（≥4 卡）；編譯器前端與 CPU 測試只需 Python≥3.10＋PyTorch＋pytest（45 個測試） | 無（kernel 層測試：ag_gemm、gemm_rs、gemm_ar、all2all 等；`tests/test_gemm_ar_collective_nvsharp.py` 等） | 無 | **CPU 部分可行；GPU 部分不可行（需自編特製 Triton）**：Dockerfile 需 root／容器 |
| ParallelKittens／ThunderKittens（HazyResearch） | `thunderkittens` | 2026-09-12（`67845f5`） | `README.md:161`：**CUDA 12.8+**、C++20、PyTorch 2.8+（`:249`）；`kernels/parallel/README.md`：`export ARCH=SM90`，**所有 kernel 假設 8 GPU**（NUM_DEVICES 為編譯期常數），`make run` 用 8 卡 torchrun；Hopper 有 `ag_gemm_h100.cu`、`gemm_rs_h100.cu`、`gemm_ar_h100.cu`、`gemm_ar_h100_lcsc.cu`、`all_reduce`、`reduce_scatter`、`ulysses_attn` 等；`include/types/system/vmm.cuh` 使用 `cuMulticastCreate` | 無 | 無 | **需要改版本**：README 要求 CUDA 12.8+，我們最高 12.6（`include/kittens.cuh` 僅對 ≥13 有條件分支，12.6 能否編譯**未驗證**）。算子層 benchmark，無端到端推論整合 |
| Triton-distributed／TileLink（ByteDance-Seed） | `triton-distributed` | 2026-11-23（`5faca68`） | `docs/build.md:3-6`：Python ≥3.11、CUDA ≥12.4、Torch ≥2.8，建議 NGC 容器；`:44` 起：`nvidia-nvshmem-cu12==3.3.9`、`nvshmem4py-cu12==0.1.2`、`cuda-python==12.4`、`cuda.core==0.2.0`、`Cython==0.29.24`；README：預設分支追蹤 Triton 3.7、以 out-of-tree plugin 形式建置；submodule：triton、cutlass（`.gitmodules`）；`python/triton_dist/kernels/nvidia/` 有 `allgather_gemm.py`、`gemm_reduce_scatter.py`、`gemm_allreduce.py`、`allreduce.py`、`flash_decode.py` 等；README 稱支援 SM90、SM100 | repo 內 grep "vllm" 無命中；有自帶的 dense 端到端 demo（`docs/getting-started/e2e/e2e_dense.md`，README:54 列為 2025-07-24 項目） | README 的 wheel 為 `triton_dist-3.4.0-cp312`（舊的 Triton 3.4 線；預設分支已是 3.7） | **需要改版本／部分不確定**：Python 3.12、torch 2.13（≥2.8 滿足）、venv 內已有 triton 3.7.1（與預設分支追蹤的 3.7 同線）；但需 nvshmem 3.3.9／nvshmem4py（我們 3.4.5）與重新建置 plugin（"Dependencies with other versions may also work well, but this is not guaranteed", `build.md:13`）。ParallelKittens 論文稱其「originally tuned for H800」（R02_T2 #10 引文）。是否需 root：文件示範用 `--privileged` 容器，**未核對**非容器環境 |
| FlashOverlap（infinigence） | `flashoverlap` | 2026-01-21（`89100a6`） | `README.md`「Dependency」：NCCL v2.18.3／v2.19.3 測試；CUTLASS ≥3.6.0；**「The code only supports sm_80, sm_86, sm_89 now」**，CUDA 12.1／12.2，torch 2.5.1；Roadmap 未勾選項：「more platforms (e.g., hopper GPU)」「end2end example」；邊界未處理（`M, N % 128 == 0`） | 無（無端到端範例） | 無 | **不可行（Hopper 未支援）**：官方明寫只支援 sm_80/86/89 |
| Ladder-residual（open-lm-engine/lm-engine） | `lm-engine` | 2026-09-25（`a9c6e2b`） | `pyproject.toml`：`requires-python >=3.12`、`torch==2.10.0`；Ladder 實作在 `lm_engine/training/models/ladder_residual/{base,layer,config,main}.py`（共 137 行）：只有殘差路徑重排（block 回傳 `current_attention_out、current_mlp_out、residual`，延後加入殘差），**這些檔內沒有 async_op／stream／all-reduce 程式碼**；全 repo `async_op=True` 只出現在 `training/dtensors.py:37,64`；版權行 "Copyright (c) 2026, Mayank Mishra"（R02_T2 #21 已記「我未確認是官方釋出」，本次仍無法確認） | 無 | 無 | **模型架構層可讀；重疊本身要自己做**：torch 2.10 釘版與我們 2.13 不同；通訊重疊靠後端 TP 實作的非同步 all-reduce，此 repo 未見。論文提到需約 3B token 重新適配（R02_T2 #21） |
| ISO（arXiv 2409.11155，Xiao & Su） | 無 | — | arXiv 摘要頁（`mcp__arxiv__get_abstract` 與 WebFetch）無程式碼連結；摘要稱 30b／70b 模型，預填階段 4090 約減 35%、A800 約減 15% | — | — | **找不到程式碼**（僅查 arXiv 頁面；未在 GitHub 另做搜尋） |
| MPK（mirage-project/mirage，`mpk` 分支） | `mirage` | 2026-11-22（`ed771bd`，HEAD=origin/mpk） | `INSTALL.md`：CMake ≥3.24、Cython、CUDA ≥11.0、cuDNN ≥8；`python/mirage/kernel.py:156-170`：目標 90 時 JIT 用 `nvcc -arch=sm_90a`（執行期編譯，`:476` 找 nvcc）；submodule：cutlass、z3、json；多 GPU 路徑用 `mpi4py`（`demo/qwen3/demo.py:162-164`）與 NVSHMEM（`include/mirage/persistent_kernel/persistent_kernel.cuh`；`attach_nvshmem_tensor`）；Hopper 有 `tasks/hopper/allreduce.cuh`、`linear_hopper.cuh`、`demo/qwen3/demo_hopper.py`、`demo_30B_A3B_hopper.py` | 無 vLLM 整合（自成一套，以 HF 模型載入編譯） | README 稱預編譯 wheel 在準備中；`INSTALL.md` 有舊版 wheel（cu122，cp310，v0.2.2） | **不確定／可能可行**：Hopper 任務齊全、nvcc 12.6 支援 sm_90a、venv 有 nvshmem 3.4.5 標頭（`site-packages/nvidia/nvshmem/include`）；但 mpi4py／MPI 環境（M25 有 mpirun 經驗）、megakernel 對 TP=8 的實際支援度、與 nvshmem 3.4.5 的相容性**皆未驗證** |

### 3.2 對 vLLM 整合的歸納（事實）

- 有 vLLM 整合：僅 **TokenWeave**（以 vLLM v0.8.5 時代 fork 形式）。
- 論文程式碼中無與 vLLM 0.32.0 直接整合者；FLUX、Triton-distributed、ThunderKittens、FlashOverlap、Syncopate 皆為算子層測試。
- vLLM 0.32.0 本身沒有 SP／async-TP pass（1.2），所以要在 vLLM 內做 TP 重疊，上述論文程式碼都得移植，或使用 TokenWeave 的舊分支。

---

## 4. 未能核實或僅部分核實的項目

1. `--compress-mode=size` 需 nvcc ≥ 12.8：直接引用 M25 卡的說法，未用 CUDA 文件另核。計算節點上 nvcc 12.6 實際行為未在本次檢查（登入節點無 nvcc）。
2. vLLM 0.32.0 與 0.34.0 在 `fuse_allreduce_rms`、DBO 等欄位是否有差異：僅讀 0.32.0（venv）；SP／async-TP 的 grep 在兩版 `src/vllm`（0.34.0）也皆無命中（只做 grep，未逐檔通讀）。
3. DeepEP（v1.2.1）能否在 nvcc 12.6／驅動 550 編譯並與 DBO 搭配：未核對。
4. `VLLM_USE_NCCL_SYMM_MEM=1` 在本環境的行為：未試。
5. FlashInfer 的 Triton 版 `all_gather_matmul` 在 sm_90 的正確性與需求：僅讀程式碼，未驗證。
6. ThunderKittens 在 nvcc 12.6 能否編譯、TokenWeave／ThunderKittens 所需 cuMulticast 在驅動 550 的行為：未驗證。
7. Triton-distributed 在無容器、無 root 下的建置可行性、與 nvshmem 3.4.5 的相容性：未驗證。
8. Qwen3-30B-A3B 是否有共享專家（影響 F4 是否相關）：未核對。
9. 各 repo 的「維護狀態」僅憑最後 commit 日期（淺層 clone 無歷史）；未查 issue／release。
10. ISO 與 Ladder-residual 的官方程式碼：僅查 arXiv 頁面與 PDF 連結（R02_T2 #21），未做 GitHub 全面搜尋。
11. SGLang、TRT-LLM 清單只列 CLI／config 與主要實作檔，未逐項核對各旗標的預設啟用條件與模型覆蓋範圍（例如 SGLang TBO 以外的 dense 模型路徑、TRT-LLM overlap scheduler 的內涵）。

## 5. 本次執行的操作（可重現）

- 唯讀 `grep`／`sed`／`ls` 於 `V/`、`SGL/`、`TRT/`、`/work/allen0117/comm_opt_env/src/vllm`（v0.34.0）、`site-packages/{flashinfer,torch}`。
- `git clone --depth 1` 九個 repo 至 `/work/allen0117/comm_opt_env/src/overlap/{flux,tokenweave,nanoflow,syncopate,thunderkittens,triton-distributed,flashoverlap,lm-engine,mirage}`（來源網址：`github.com/bytedance/flux`、`microsoft/tokenweave`、`efeslab/Nanoflow`、`tie-pilot-qxw/syncopate`、`HazyResearch/ThunderKittens`、`ByteDance-Seed/Triton-distributed`、`infinigence/FlashOverlap`、`open-lm-engine/lm-engine`、`mirage-project/mirage`）。未編譯、未安裝、未送 Slurm、未 commit、未讀禁讀資料夾。
