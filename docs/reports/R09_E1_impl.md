# R09 E1 實作報告：腳本、2 卡冒煙驗證、主工作 sbatch

- **撰寫**：Sonnet 5.5，2026-11-25　**規格書**：`docs/specs/S09_E1_impl.md`　**實驗卡**：`exp/E1_prefill_overlap/README.md`
- 未送主工作。依 Opus 補充（使用者要求不因時間不夠跳過項目），主工作 `--time=05:00:00`，full 模式區塊上限 B0 10／A 90／B 90／C 60 分，整體 `GLOBAL_MIN=290`。

## 白話結論

1. 三段實驗（預填基準、成對運算子、八層原型）的腳本都寫好，在 2 卡 `dev` 上完整跑通：各區塊都正常結束、分析程式跑得完、通訊方式的環境變數確實生效。
2. **有一項通過條件不成立，需要 Opus 決定**：C 段「重疊版與依序版數值要一致（相對誤差 1e-2）」。所有 C2 設定（k=2、4）都超過 1e-2（k=2 約 1.07–1.10%，k=4 約 1.31–1.36%），照卡的規則全部會被標作廢。
   證據顯示這是切塊造成的四捨五入差異，不是重疊程式的錯：重疊版與「同樣切塊但依序執行」的版本逐位元相同（誤差 0.0）；只做一層時誤差是 0.3–0.4%，約等於 bf16 的精度，8 層累積到約 1%；且與通訊方式無關（三種方式的數字完全一樣）。
3. 我沒有改容差。為了不讓主工作白跑，輸出同時保留兩種判定（卡上的 1e-2、以及「同樣切塊依序版」的 1e-4），分析程式兩份都會出（`analyze_stdout.txt`／`analyze_stdout_cvalid_seq.txt`），請 Opus 決定採哪個。
4. 主工作時間推估約 70–80 分鐘，遠低於各區塊上限。
5. 驗證用 2 卡 7B 模型，所以 A1、A2、B、C 的數值本身沒有意義，不可用來判讀預期；它只證明流程。

## 驗證工作

| 工作 | 結果 | 說明 |
|---|---|---|
| 500158（第 1 次） | 全部 rc=0，3m17s | 落在 hgpn01，GPU1 過熱降頻（375 MHz，B0 溫度檢查 THERMAL_FAIL，smoke 不中止），B/C 計時被拖慢不可用；C2 全部 INVALID（見上）|
| 500217（第 2 次） | 全部 rc=0，2m45s | 加入第一層誤差診斷；hgpn01 溫度檢查 THERMAL_OK |
| 500244（第 3 次，最終） | 全部 rc=0，約 2m40s | 再加入「同樣切塊依序版」診斷與雙判定分析；證據都用這次 |

最終驗證輸出：`results/E1_prefill_overlap/validation/20261125_211720_job500244_val/`（以下簡稱 V）。三次驗證都不是修錯誤：第 1 次無程式錯誤，後兩次是為了診斷 C 段數值差異而加診斷欄位。另在 hgpn01 發現本次 smoke 顯示的過熱問題只出現在 500158（當時 B0 `thermal.txt`：gpu1 clk_median=375、temp_max=89）。

## 通過條件逐條

| 條件 | 是否成立 | 證據 |
|---|---|---|
| 各區塊 rc=0，`blocks.tsv` 全部 DONE／OK | 成立 | V`/blocks.tsv`（B0 OK、A、B、C 皆 DONE）；V`/summary.txt` 每個 launch rc=0 |
| A 有 `per_request.tsv`、trace 分析、all-reduce kernel 名稱 | 成立 | V`/a/1/per_request.tsv`、`result.json`（profile 含 n_allreduce_calls、allreduce_share、gpu_span）、每 L 一份 trace gz；kernel 名稱：`two_shot_all_reduce_kernel_inplace<bf16,16,2>`（7B TP2 的 vLLM custom 路徑）；backend 清單 `a/1/allreduce_backend.txt` |
| B 的 jsonl 格子數正確 | 成立 | V`/b/default_l1.jsonl` 20、`cta8_l1.jsonl` 20、`symm_l1.jsonl` 15（T≤4096）|
| `NCCL_DEBUG` log 確認環境變數生效 | 成立 | V`/b/env_check.tsv`：`NCCL_MAX_CTAS=8` set_by_environment 2 行（2 rank）；channels 0..7 對預設 0..23（B/C 的 log 在 `b/nccl_logs/*.tar.gz`）。ALGO／PROTO 的檢查程式同一段碼，2 卡 smoke 沒有跑這兩種，主工作會產生 |
| `c_mechs.txt` 產生 | 成立 | V`/b/c_mechs.txt`（mech1=symm、mech2=cta8；smoke 只有 3 種可選，數值無意義）|
| C 正確性全部通過（沒有 INVALID）| **不成立（照卡的 1e-2）** | V`/analyze_stdout.txt` 的「C2 correctness」：k=2 誤差 0.01066–0.01102、k=4 0.01314–0.01358，全部 INVALID。診斷：第 1 層誤差 0.28–0.39%；overlap 對「同切塊依序」誤差 = 0.0（V`/c/*.jsonl` 欄位 `rel_err_vs_seq_same_k`）；三種通訊方式數字完全相同 |
| `analyze.py` 在驗證輸出上跑得完 | 成立 | V`/analyze_stdout.txt`、`analyze_stdout_cvalid_seq.txt`，`analyze rc=0`，無 ERROR |
| 主工作各區塊推估時間在上限內 | 成立 | 見下 |

## 主工作時間推估（8 卡，單台）

方法：用驗證實際耗時（V`/blocks.tsv`、`b/launches.tsv`、`c/launches.tsv`、`a/launches.tsv`）拆成「固定開銷（引擎啟動、NCCL 初始化、配置）」與「計時部分」，計時部分依實際迭代次數放大；A 的 72B 啟動時間用 M25 的實測（`q72_tp8` 100–230 秒，`results/M25_topic_survey/20261125_093304_job498238_all/b2/engines.tsv`）。

| 區塊 | 驗證耗時 | 推估（主工作） | 上限 |
|---|---|---|---|
| B0 | 44 s（燒機縮為 20 s）| 約 5 分（M25 實測 262 s）| 10 分 |
| A | 42 s（7B、1 次、2 個 L）| 每次啟動 約 6–8 分（啟動 2–4 分 + 40 次請求 < 1 分 + 12 次 profile 各約 10 s 含匯出）× 3 ≈ 20–25 分 | 90 分 |
| B | 每個行程 9–10 s（13 次迭代）| 每個行程 約 40–90 s（60 次迭代，8 卡初始化較久）× 21 ≈ 15–30 分 | 90 分 |
| C | 每個行程 13–14 s（7 次迭代）| 每個行程 約 60–120 s（25 次迭代 × 4 個 L × 4 版本 + 正確性檢查）× 9 ≈ 10–18 分 | 60 分 |
| 合計 | 157 s | 約 55–80 分 | 300 分（工作上限）|

全部低於上限，沒有項目會因時間跳過。不確定處：8 卡下 all-reduce 較慢、vLLM 72B 第一次 torch.compile 若沒有快取會更久（M25 當時已有快取；`/work/allen0117/comm_opt_env/cache` 仍在）。

## 與實驗卡／規格書不同或需要 Opus 注意的地方

1. **C2 正確性（上述）**：容差 1e-2 對「切塊 vs 不切塊」過緊。我沒有改，兩種判定並存（`--c-valid card|seq`；預設卡的規則）。若主工作照卡規則，C2 會全部作廢，分析程式的 C2／C3 表會是空的；另一份 `analyze_stdout_cvalid_seq.txt` 才有 C2 數字。
2. **B 段「一次迭代 = R 對」**：卡上寫「一個 chunk 的 all-reduce 和下一個 chunk 的計算同時跑」。單一 16 MB all-reduce 或小 GEMM 只有幾十微秒，事件計時和啟動時間差會主導。我讓每次計時迭代重複 R = 8192/T 對（R×T = 8192 token；T=1024 時 R=8），通訊與計算各發 R 次同時進行。jsonl 的時間是整個迭代（R 對）的時間，欄位 `R` 有記錄；拖慢與重疊效率是比值，不受影響，但「完成時間」是 R 對的總和。若 Opus 要嚴格的單對，改 `e1_pairs.py` 的 `R = HID // T` 即可。
3. **B 的時間彙整**：每格 50 次迭代的 5 個數字，每次迭代跨 8 卡取最大（gloo），與規格相同。
4. **B0 上限**：`m25_lib.sh` 的 `b0` 寫死 8 分；為了 10 分，`e1_lib.sh` 包了一層 `blk_begin`（full 模式 B0=10）。
5. **C 段注意力**：每個版本（含 C0、C1）都用 `SDPA + causal_lower_right(Lc, e)`，K/V 由 1 頭展開成 8 頭（`.contiguous()`）；沒有 RoPE、沒有 KV cache；RMSNorm 用 `F.rms_norm`，SiLU×mul 為兩個 kernel（vLLM 是融合 kernel）。C0 因此不等於 vLLM 的 kernel 組合，只是依序執行的對照。
6. **symm_mem**：2 卡時 vLLM 的規則不用 multimem（只列 4、6、8 卡），smoke 實際走 `two_shot`（jsonl 欄位 `symm_algo`）；8 卡主工作會走 `multimem`，若 multicast 不可用會降成 two_shot 並在欄位記錄。C 段 symm 對 > 64 MB 的 all-reduce 退回 NCCL（同 vLLM）。
7. **A 的設定**：`disable_log_stats=False`（為了取得 `RequestOutput.metrics` 的排程與首 token 時間戳，已確認在 vLLM 0.32.0 有值）；與 M25 B2 其餘設定相同。profile 用 `start_profile(delay_iterations=0)`＋`stop_profile`，每次剛好含一個預填步；每個 L 每次啟動 3 個 trace，第 1 個 gz 存入結果資料夾，其餘留在 `/work/allen0117/comm_opt_env/traces/`。
8. **A2 的取值**：smoke（7B TP2）預填時間比 kernel 總和大 13–32%，因為 L 小、固定開銷占比高；72B 在 L 大時應較小，但這次無法判斷，需主工作看。表中同時列「vLLM 的 first_token − scheduled」與 wall time 兩種差。
9. **驗證 sbatch**：未加 `--exclude=hgpn01`（排隊太久）。hgpn01 的 GPU1 在 500158 過熱，與 M25 497945 的 GPU7 是同一台機器的問題；主工作 H100 已 `--exclude=hgpn01`。
10. 驗證工作的輸出放在 `results/E1_prefill_overlap/validation/`（含 slurm 日誌），主工作輸出在 `results/E1_prefill_overlap/<時間>_job<id>_<h100|h200>/`、日誌在 `results/E1_prefill_overlap/slurm/`。

## 檔案

`exp/E1_prefill_overlap/`：`e1_lib.sh`、`e1_vllm.py`、`e1_pairs.py`、`e1_layer.py`、`pick_c_mechs.py`、`analyze.py`、`validate.sbatch`、`job_h100.sbatch`（normal、exclude hgpn01）、`job_h200.sbatch`（normal2）。
送主工作：`.claude/skills/slurm-submit/submit.sh --skip-wallet exp/E1_prefill_overlap/job_h100.sbatch`（同理 h200；量測要 `--exclusive`，檔內已有）。
