# S03 主流系統的通訊實作調查（L2）：在 H100 上實際走哪些路

- **寫給**：Sonnet 5.5 執行者　**撰寫**：Opus 5.5，2026-11-16（D051）
- **類型**：原始碼調查，不需 GPU
- **報告**：`docs/reports/R03_system_code_survey.md`

## 目標

找出現在主流的訓練與推論系統，在 **8× H100 SXM（NVSwitch）單機** 與 **多台 InfiniBand** 上，實際用哪些通訊實作、預設是什麼、在什麼條件下切換。這些就是之後任何題目的「現成最強對照組」，也能看出哪些地方大家還在用簡單的做法。**只記錄事實，不做評價。**

## 範圍（每個系統用截至 2026-11-16 的最新穩定版 release tag；記錄 tag 與 commit）

| 系統 | 重點 |
|---|---|
| vLLM | custom all-reduce（one-shot／two-shot 切換門檻）；symmetric memory 或 NVLS 路徑；FlashInfer 或其他 all-reduce＋norm 融合；pynccl；EP 的 all-to-all 後端；各自的啟用條件、環境變數、預設值 |
| SGLang | 同上，加上它自己的 custom all-reduce 版本 |
| TensorRT-LLM | all-reduce 策略（例如 one-shot、two-shot、NCCL、Lamport 等名稱以原始碼為準）、融合選項、MoE 通訊 |
| FlashInfer | 通訊相關模組（all-reduce 融合、MNNVL 等，以原始碼為準）與它支援的 GPU 架構 |
| Megatron-Core | TP／SP／CP／EP 通訊與 overlap 選項（例如 TP comm overlap、EP dispatcher 種類）、預設值 |
| DeepEP | normal 與 low-latency 兩種 kernel 的通訊方式：NVLink、RDMA、NVSHMEM；硬體需求 |
| PyTorch | `torch.distributed._symmetric_memory`（或正式名稱）的 collective 與 multimem 支援、async TP |
| NCCL | 2024–2026 release notes 中與 NVLS、symmetric memory、PAT、小訊息延遲、跨機器（IB、SHARP、multi-rail）相關的新功能與預設值改變 |

## 步驟

1. 在登入節點 `git clone --depth 1 --branch <最新 release tag>` 到 `third_party/src/<系統>/`（`third_party/` 不進 git）。只做讀取，不編譯、不執行。若 clone 太大，可改用 GitHub 網頁或 raw 檔。
2. 用 `grep` 等工具找出通訊相關程式碼。每個論點都要附 **檔案路徑:行號**（以你 clone 的 tag 為準）與那幾行的原文摘錄（≤ 5 行）。
3. 讀官方文件（docs 網站、README、release notes）補充設定方式，附網址。
4. 每個系統整理以下內容：
   - 有哪些通訊實作
   - 預設用哪個，以及切換條件：訊息大小門檻、GPU 架構判斷（例如 `sm_90`、`is_hopper`）、GPU 數、是否跨機
   - 能用環境變數或設定改什麼
   - 有沒有和計算融合或重疊
   - 原始碼註解或文件中提到的限制或 TODO（原文）

## 產出：`docs/reports/R03_system_code_survey.md`

1. **總表**：每列一個「系統 × 集體通訊（all-reduce、all-gather、reduce-scatter、all-to-all、send/recv）」組合，欄位：
   - 在 8× H100 單機上的預設實作
   - 切換條件
   - 是否用 NVLS／multimem
   - 是否融合或重疊
   - 跨機時的實作
   - 證據（檔案:行號）
2. **各系統細節**：照步驟 4。
3. **原始碼中的限制與 TODO**：原文引用。
4. **版本表**：系統、tag、commit、clone 日期。
5. **和規格不同的地方、沒解決的問題**。

## 規則

- 只寫原始碼與文件中有的事實，不推測效能，不評價好壞。看不懂或找不到就寫「未找到」，並列出搜過的關鍵字。
- 不 commit、不 push；不執行任何 GPU 或 Slurm 工作；不讀 `~/TP_opt`、`~/tp_opt`、`~/flux`；不改 `exp/`、`results/`。另一位執行者正在使用 `third_party/nccl-tests/`，不要動它。
- 用繁體中文撰寫，技術名詞保留英文。
