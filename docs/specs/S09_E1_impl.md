# S09 E1 實作：腳本、2 卡冒煙驗證、主工作腳本（不送主工作）

- **寫給**：Sonnet 5.5 執行者　**撰寫**：Opus 5.5，2026-11-25（D060）
- **類型**：寫程式＋送 1–2 個 `dev` 冒煙工作
- **報告**：`docs/reports/R09_E1_impl.md`

## 背景

實驗卡 `exp/E1_prefill_overlap/README.md` 已由使用者核准（D060）。**先完整讀它**：A、B、C 三段的設定，以及預期與判讀規則。

你的工作是把它變成可以跑的腳本，在 `dev` 用 2 卡驗證流程，並寫好兩台主工作的 sbatch，**交回 Opus**。主工作由 Opus 檢查驗證結果後送出，你不送。

**實驗卡的設定不可改。** 遇到做不到的地方，停下來在報告中寫明原因和替代方案，交給 Opus 決定。

## 沿用的東西（先讀）

- `exp/M25_topic_survey/m25_lib.sh`：
  - 環境、GPU 監看、同節點工作紀錄（`exp/lib/cotenants.sh`，D054）
  - `b0`：暖機、8 卡 GEMM 燒機、`thermal_check.py`，不合格在 full 模式中止
  - `gpu_clean`
  - 區塊計時（`blk_begin`／`blk_end`／`cap`／`can_run`／`skip_rec`）
  - 實作時的教訓都寫在註解，例如：
    - `wait` 只等自己的 PID（B0 燒機時，裸 `wait` 會連 GPU 監看程序一起等，曾讓 498237 卡住）
    - mpirun 要 `< /dev/null`
    - NVLS 下控制用的歸約要走 gloo
- `exp/M25_topic_survey/README.md` 的「Opus 補充」各節：497945、498225、498237 遇到的問題與修法。
- `exp/M25_topic_survey/vllm_bench.py`（A 段的基礎）、`overlap_bench.py`（B 段的基礎）、`b2_kernels.py` 與 `b2_prefill.py`（kernel 分類與切步）。
- `.claude/skills/slurm-submit/`（`SKILL.md`、`template.sbatch`、`submit.sh`）。
  - `wallet` 壞掉時用 `--skip-wallet`（D057）。
  - `submit.sh` 會拒絕含 `<...>` 樣式的行，stdin 重導向改用 `exec 0</dev/null`。
- 模型 config：`/work/allen0117/comm_opt_env/models/{Qwen2.5-72B-Instruct,Qwen2.5-7B-Instruct,Qwen3-30B-A3B}`（dummy 權重）。

## 要寫的檔案（都在 `exp/E1_prefill_overlap/`）

### 1. `e1_lib.sh`（工作執行時 source）

- 輸出到 `results/E1_prefill_overlap/<YYYYmmdd_HHMMSS>_job<id>_<tag>/`，包含：
  - `job_env.txt`、`cotenants_*`、`gpu_monitor.csv`（結束時 gzip）
  - `summary.txt`、`blocks.tsv`
- 區塊與時間上限（主工作 3 小時）：
  - B0：8 分
  - A：60 分
  - B：50 分
  - C：30 分
  - 超時的項目記 `SKIPPED(time)`，不中斷其他區塊。
- 模式：
  - `E1_MODE=full`：8 卡、B0 溫度不合格就中止。
  - `E1_MODE=smoke`：2 卡、縮小規模、不中止。
- 可以 source `m25_lib.sh` 借用函式。如果它的全域變數（`OUT`、`M25_*`）會互相干擾，就把需要的函式複製過來，並在註解標出處。

### 2. A 段：`e1_vllm.py`（可從 `vllm_bench.py` 擴充或複製）

- **設定**：Qwen2.5-72B、TP=8、graph 模式、FlashInfer 關閉，設定同 M25 B2 的 `q72_tp8`。`max_num_batched_tokens=17408`，`max_model_len` 足夠放 16384＋1。
- **每次引擎啟動**：
  - 依隨機順序（記錄種子）跑 L ∈ {2048, 4096, 8192, 16384}。
  - 每個 L 送 10 次「單一請求、輸出 1 token」，前 2 次丟棄。
  - 記錄每次的 TTFT（wall），以及 vLLM 能提供的排程時間。
  - 每個 L 另外用 torch profiler 抓 3 次請求（rank 0）。
  - 由 trace 算：該步 GPU kernel 時間總和、all-reduce kernel 時間總和與比例、all-reduce 用的 kernel 名稱（分類沿用 `vllm_bench.classify`）、該步 GPU 時間跨度。
- **3 次獨立引擎啟動**。
- 每個 L 保留一個 trace（gzip，沿用 `--keep-trace-mb` 的作法）。
- **輸出**：`a/<launch>/result.json`、`a/<launch>/per_request.tsv`。

### 3. B 段：`e1_pairs.py`（torchrun 8 卡，可從 `overlap_bench.py` 擴充）

- **格子**：卡上 B 段寫的 4 種 T × 5 種計算，形狀照卡上。
  - 注意力：因果 SDPA，每卡 8 個 Q 頭、1 個 KV 頭，用 GQA 或展開 KV 皆可，寫明用哪個。
  - 計算用隨機 bf16 張量（記錄種子）。
- **每格量三種**，每種暖機 10 次、計時 50 次：
  - 通訊單獨
  - 計算單獨
  - 兩者同時啟動：兩條 stream，各自以 CUDA event 計時，並記完成時間。
  - 每次迭代跨 8 卡取最大（R7）。計時彙總用 gloo 群組（NVLS 下 NCCL 沒有 float64 歸約）。
- **通訊方式**：照卡上 7 種，每種用獨立的 torchrun 行程，以環境變數設定。
  - `NCCL_ALGO`、`NCCL_PROTO`、`NCCL_MAX_CTAS` 的實際生效要用 `NCCL_DEBUG=INFO` 的 INIT／TUNING 或 ENV 訊息確認，存 log。
  - symm_mem multimem 照 vLLM 0.32.0 的作法（venv 內 `vllm/distributed/device_communicators/symm_mem.py`），只跑 T ≤ 4096。
- **3 次獨立啟動**：7 種 × 3 次 = 21 個行程，依隨機交錯順序執行（記錄順序與種子，同 M25 B1）。
- **輸出**：`b/<mech>_l<launch>.jsonl`，每格一行，含所有原始迭代時間。
- **B 結束後跑 `pick_c_mechs.py`**，從 B 的結果選出 C 要用的「完成時間最好的兩種」非預設方式。
  - 規則：每種方式對「NCCL 預設」的完成時間比值，取跨格幾何平均，最小的兩種。
  - symm_mem 只有一半格子，只在它有的格子比，並在選擇紀錄中寫明。
  - 結果寫到 `b/c_mechs.txt`。
  - 若 B 失敗，就用 `NCCL_MAX_CTAS=8` 與 Ring＋Simple，並記錄改用的原因。

### 4. C 段：`e1_layer.py`（torchrun 8 卡，新寫）

- 照卡上的 8 層堆疊，版本 C0、C1、C2（k=2、4）。L 照卡上 4 種。
- **C2 的寫法**：
  - token 切 k 塊。第 i 塊的 o_proj 輸出與 down 輸出的 all-reduce 在通訊 stream 上，同時計算第 i+1 塊。
  - 跨層也可以重疊：下一層第 i 塊只依賴本層第 i 塊加總完的結果。
  - 用 CUDA event 管依賴。
  - 注意力的第 j 塊要看到 1…j 塊的 K／V（右下對齊的因果遮罩，例如 `torch.nn.attention.bias.causal_lower_right`，或自行建 mask）。
- **正確性**：每個 C2 設定與 C0 用相同權重與輸入，比較最後輸出，bf16 容差 1e-2（相對或絕對都寫明）。不通過就把該格標 `INVALID`。
- **通訊方式**：NCCL 預設＋`b/c_mechs.txt` 的兩種。每種方式一個 torchrun 行程，3 次獨立啟動，交錯順序。
- 每個設定暖機 5 次、計時 20 次，跨卡取最大。
- **輸出**：`c/<mech>_l<launch>.jsonl`。

### 5. `analyze.py`

- 對照卡上每一條預期（A1、A2、B1–B4、C1–C3）輸出判定所需的數字：
  - 中位數、95% bootstrap 信賴區間
  - 3 次啟動各自的中位數
- 列出判讀規則各項成立與否。
- 不寫判定文字，判定由 Opus 寫。

### 6. sbatch 檔

**冒煙驗證**：`validate.sbatch`

- `dev`、2 卡、30 分、`E1_MODE=smoke`、`--mem=200G`、`--cpus-per-task=24`。
- A 段改用 Qwen2.5-7B TP=2，L 只跑 {2048, 4096}，1 次啟動。
- B、C 段用 world size 2，形狀不變，但每種方式只跑 1 次啟動，計時次數可減少。
- B 只跑 3 種方式：預設、`NCCL_MAX_CTAS=8`、symm_mem。

**主工作**：寫好、不送

- `job_h100.sbatch`：`normal`、`--exclude=hgpn01`、8 卡、`--exclusive`、`--mem=400G`、`--cpus-per-task=96`、`--time=03:00:00`、`E1_MODE=full`、tag `h100`。
- `job_h200.sbatch`：`normal2`，其餘相同，tag `h200`。
- 輸出到 `results/E1_prefill_overlap/slurm/%x-%j.{out,err}`。

## 驗證流程

1. 送 `validate.sbatch`：`submit.sh --not-measurement`；`wallet` 壞掉時加 `--skip-wallet`。
2. 等結果用「背景延遲後查一次」（skill 步驟 4），**不輪詢**。
3. 失敗就修，再送，最多 4 次。每次失敗的原因與修法寫進報告。
4. **通過條件**（全部要成立）：
   - 各區塊 rc=0，`blocks.tsv` 全部 DONE／OK。
   - A 有 `per_request.tsv` 與 trace 分析結果，all-reduce kernel 名稱有記錄。
   - B 的 jsonl 每種方式的格子數正確；`NCCL_DEBUG` log 確認各環境變數有生效；`c_mechs.txt` 產生。
   - C 的正確性檢查全部通過（沒有 `INVALID`）。
   - `analyze.py` 在驗證輸出上跑得完。
   - 依驗證工作的實際耗時，推估 8 卡主工作各區塊時間，都在上限內。推估方法寫進報告。

## 規則

- 實驗卡的設定與預期不可改；做不到就停下回報。
- 只送 `dev` 冒煙工作，**不送主工作**；不輪詢 `squeue`。
- 登入節點不跑任何 GPU 或運算。
- 不改 `results/` 既有數據；不改 `docs/decisions.md`、`docs/status.md`、`docs/project.md`。
- 可以 commit 並 push 你寫的腳本與驗證結果（在 `nano5` 分支，訊息寫清楚；結尾加 `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`）。不要 commit `bot.log`。
- 不讀 `~/TP_opt`、`~/tp_opt`、`~/flux`。
- 用繁體中文寫報告，技術名詞保留英文。
- **最後回報**（給 Opus）：
  - 報告路徑
  - 驗證工作編號與結果
  - 每個通過條件逐條是否成立（附證據路徑）
  - 主工作推估時間
  - 任何與實驗卡不同的地方
