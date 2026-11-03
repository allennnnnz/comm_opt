# 「加總同步＋正規化合併」在 A100 上有沒有人做過（2026-11-03）

- **觸發**：使用者：「查 TensorRT-LLM 和其他框架：確認『加總＋正規化合併』在 A100 上有沒有人做過。不需要 GPU」
- **方法**：只讀原始碼與文件，沒有跑 GPU。重點看三件事：(1) 有沒有把 all-reduce 和 residual add + RMSNorm 合成一個 kernel；(2) A100（sm80）能不能編譯、執行期會不會啟用；(3) 用幾次全域同步（barrier）、正規化做在哪一步。

## 白話結論

1. **「加總＋正規化合併」在 A100 上已經有人做過，而且不只一家。** 輝達自己的推論框架（TensorRT-LLM）和上海人工智慧實驗室的推論框架（LMDeploy）都有，程式都寫了 A100 能走的路。
2. **我們原本想的「每張卡負責一部分的列 → 自己做正規化 → 再互相交換」，LMDeploy 已經完整做出來了**，連「殘差只保留自己那一份」的技巧都有。所以這個組合本身不能當成我們的新貢獻。
3. **但有一個明顯的空缺：小訊息。** LMDeploy 在 A100 上，訊息 512 KB 以下會主動關掉這個合併、改回分開做（程式裡寫死）；它的做法每次要全體等齊兩次。TensorRT-LLM 在小訊息用的是「一次全送」的免等待做法，資料量是分兩次傳的 4 倍；它分兩次傳的版本也要全體等齊兩次。**沒有人做出「分兩次傳、而且兩次都不用全體等齊」的合併版本**——這正好是 M8 證明 A100 硬體做得到的那一招。
4. **vLLM、SGLang 和它們底層的 FlashInfer，在 A100 上都不開這個合併**（只為新一代 GPU 編譯），所以如果我們在 A100 上用 vLLM 做端到端，比較對象要另外找 TensorRT-LLM 和 LMDeploy 的合併版本。
5. **下一步建議**：先在 A100 上實際量 TensorRT-LLM 和 LMDeploy 的合併版本（這是新實驗，要你核准），確認它們在 64 KB–1 MB 到底多快、LMDeploy 為什麼要在小訊息關掉，再決定要不要做「兩次都免等齊的合併版本」。

## 逐一結果

| 框架（版本） | 有合併 kernel？ | A100 上可用？ | 做法 | 全域同步次數 | 正規化在哪做 | 出處 |
|---|---|---|---|---|---|---|
| **TensorRT-LLM** 新版 `ar_fusion`（PyTorch 後端用的） | 有：AR + residual + RMSNorm（＋FP8/NVFP4 量化） | **程式上可以**：A100 走非 cluster、非 PDL 的路（`#else` 分支、`supports_cluster = SM>=90 && SM<120`）；官方 support matrix 列 Ampere | one-shot Lamport：每卡把整份推給 7 張卡，接收端以 −0.0 當「還沒到」的標記等資料 | **0**（無 barrier，標記嵌在資料） | 每張卡對**全部**列做 | `allReduceFusionKernels.cu` L448–525 |
| 同上 | 有 | 程式上可以 | two-shot sync：先複製到本卡 buffer → barrier → 每卡負責一段列、讀 8 份相加、把**總和**寫回 8 張卡 → barrier → 每卡對**全部**列做 residual + norm | **2** | 全部列（重複做 8 次） | 同檔 L528–590 |
| TensorRT-LLM 舊版 `customAllReduceKernels`（TRT engine plugin） | 有：one-shot AR+norm 一個 kernel；two-shot 是 AR(+residual) kernel 再接一個 norm kernel | one-shot 版可以；Lamport 版限 sm90–sm119（`is_lamport_supported`） | 見左 | one-shot 有 barrier | 全部列 | `customAllReduceKernels.cu` L937、L1172、L1722–1767 |
| **LMDeploy TurboMind**（`cuda_ipc`） | 有：AR + residual + bias + RMSNorm | **可以**（TurboMind 支援 sm70 以上，這段沒有架構限制）；有 multicast 時另走 NVLS 版 | **token 列 reduce-scatter → 自己那段做 residual + norm → all-gather 正規化後的結果**（Push：先把資料推到對方 scratch；Pull：從對方讀） | **2**（Push：推完一次、收完一次） | **只做自己那 1/N 的列** | `fused_allreduce.cu` L26–165、L277–404 |
| 同上，分派規則 | — | — | **訊息 ≤ 512 KB 直接 `return false`**，改回「一般 AR + 另一個 norm kernel」（`#if 1 else if (bytesize <= 1 << 19) return false;`）；512 KB–6 MB 用 Push；更大用 Pull | — | — | `fused_allreduce.cu` L455–460 |
| **vLLM**（0.33.0，本機安裝） | 有（AllReduceFusionPass，呼叫 FlashInfer 的 TRT-LLM 移植版） | **不開**：`enable_allreduce_rms_fusion` 只在 `is_device_capability(90)` 或 `family(100)` 為真；大小表只有 90/100/103/107 | — | — | — | `vllm/config/vllm.py` L179–203；`vllm/config/compilation.py` L182–209；vLLM 文件 fusion 表 SM80 欄空白 |
| **FlashInfer**（0.7.2.post1，本機安裝） | 有（`trtllm_allreduce_fusion`，與 TRT-LLM 同源） | **不編譯**：`gen_trtllm_comm_module` 的 `supported_major_versions=[9, 10, 12]` | — | — | — | `flashinfer/jit/comm.py` L195–205 |
| **SGLang**（main 40a64e1） | 有（呼叫 FlashInfer） | **不開**：`_resolve_backend` 遇到非 SM90/SM10X 直接報錯 | — | — | — | `python/sglang/srt/layers/flashinfer_comm_fusion.py` L46–50 |

### TensorRT-LLM 在 A100 上自動會挑哪個

`selectStrategyLookUpTable`（`customAllReduceUtils.h` L89–123）把 **sm < 100 一律當成 sm90**，所以 A100 用的是**在 H100 上調出來的表**。TP=8、RESIDUAL_RMS_NORM：

| hidden | 1–64 tokens | ≥128 tokens |
|---|---|---|
| 4096 | ONESHOT（合併、免 barrier） | NCCL（不合併，AR 後另跑 norm kernel） |
| 8192 | 1–32 tokens ONESHOT | ≥64 tokens NCCL |

換成 bytes（fp16）：大約 **≤ 512 KB 用合併的 one-shot，以上改用 NCCL＋另跑 norm**；TP=8、hidden ≥ 4096 時表裡從不選合併的 two-shot。這張表沒在 A100 上調過——和 M7 的「NCCL 在 A100 上選錯」是同一類問題。

## 論文端

- arXiv 搜尋（`all-reduce` × `RMSNorm/normalization` × `fused/kernel`，2023 起）只找到 **TokenWeave**（2505.11329，`gond2025tokenweave`）：RS → residual + RMSNorm → AG 的合併 kernel，用 multimem（Hopper 以上），論文寫明「targets H100s and later」
- **《Every µs Matters》**（2607.16100）全文搜尋 “norm” 0 筆：只做 all-reduce，沒有合併正規化
- **SiFAR**（2607.08973）：只做 all-reduce；拿 TRT-LLM oneshot 和 Lamport oneshot 當比較對象
- 沒有找到在 A100 上量測 TRT-LLM 或 LMDeploy 合併 kernel 的公開數據（這次只查到這個程度，不代表一定沒有）

## 對選題的影響

| 想法 | 狀態 |
|---|---|
| A100 上做 AR + residual + RMSNorm 合併 | **已有**（TRT-LLM、LMDeploy） |
| token 列 RS → 只對自己那段做 norm → AG，殘差只保留自己那段 | **已有**（LMDeploy；TokenWeave 在 Hopper） |
| 免 barrier 的合併 **one-shot** | **已有**（TRT-LLM Lamport oneshot；A100 程式上可跑） |
| 免 barrier 的合併 **two-shot**（兩輪都把旗標嵌在資料，A100 用 32-bit 原子加法） | **沒找到**。LMDeploy 的 two-shot 合併要 2 次 barrier，且在 ≤ 512 KB 被關掉；TRT-LLM 的 two-shot 合併要 2 次 barrier、norm 重複做 8 次；《Every µs Matters》的免 barrier two-shot 沒有合併 norm、只在 GB200 |
| 依機器實測挑做法（取代在別的 GPU 上調的表） | TRT-LLM、vLLM、NCCL 都是寫死或用別的 GPU 的表；**沒看到在 A100 上自動校正的** |

## 限制

- 只讀程式碼，沒有在 A100 上執行 TRT-LLM 或 LMDeploy 的合併 kernel；「A100 上可用」是指程式路徑存在且沒被架構檢查擋住，實際能不能跑、跑多快未驗證
- LMDeploy 「殘差只保留自己那段」是從 kernel 只寫自己那段 `res` 推斷的，沒有追完整個模型流程
- 沒查 LightLLM、DeepSpeed-MII、MLC-LLM、AMD aiter（後者不適用 NVIDIA）；Megatron sequence parallel 是訓練用、RS／norm／AG 分成三個運算，不是合併 kernel

## 更正先前的說法

- D012 寫「TensorRT-LLM／FlashInfer 在 SM80 不啟用」：**FlashInfer 正確，TensorRT-LLM 不精確**——TRT-LLM 自己的合併 kernel 有 A100 的程式路徑，只有 Lamport 舊版和 cluster／PDL 這些加速限新一代 GPU
- `survey_20261103.md` 寫「非 Multimem 的 AR+RMSNorm 融合在 A100 上沒測過」：程式已經存在（TRT-LLM、LMDeploy）；「沒有公開量測數據」這部分在這次搜尋範圍內仍成立

## 來源（皆於 2026-11-03 取得）

- TensorRT-LLM main `508f8dddefa817cff2279e3f1632698c9dd93b47`：`cpp/tensorrt_llm/kernels/communicationKernels/allReduceFusionKernels.{cu,h}`、`cpp/tensorrt_llm/kernels/customAllReduceKernels.{cu,h}`、`cpp/tensorrt_llm/thop/allreduceOp.cpp`、`cpp/tensorrt_llm/common/customAllReduceUtils.h`、`cpp/cmake/modules/cuda_configuration.cmake`（預設架構含 80、86）；support matrix：https://nvidia.github.io/TensorRT-LLM/reference/support-matrix.html
- LMDeploy main `3228395546dfb2f6427faf39ec0c0c285f4c378a`：`src/turbomind/comm/cuda_ipc/fused_allreduce.cu`、`fused_allreduce_ex.cu`、`src/turbomind/models/llama/unified_decoder.cc` L137–170
- vLLM 0.33.0、FlashInfer 0.7.2.post1（`.venv-vllm` 安裝版）；vLLM fusion 文件：https://docs.vllm.ai/en/latest/design/fusions/
- SGLang main `40a64e1b5ba00c73b7f8a123a903a2123b7d467c`
- 論文：TokenWeave（`gond2025tokenweave`）、Every µs Matters（`shen2026every`）、SiFAR（`taneja2026sifar`），皆已在 `lit/refs.bib`
