# R05 M20 環境快照（執行者報告，階段性：工作已送出、尚未執行）

撰寫：Sonnet 5.5，2026-11-17。依據：S05、`exp/M20_env_snapshot/README.md`。

## 狀態

腳本已寫好、三個工作已送出並在排隊；**沒有任何量測數據**，P1–P10 尚無觀察。

## 已完成

- `exp/M20_env_snapshot/`：`node.sbatch`（快照 → SHARP 單台預檢 → mcast 屬性 → NVLS (a)(b)(c) → 時脈 5 輪）、`sharp.sbatch`（2 台，設定 (a)–(f) + all_gather／reduce_scatter）、`common.sh`（啟動方式探測 pmix_v4 → pmi2 → mpirun）、`mcast_query.cu`、`analyze.py`（節點對照表、時脈表、`figures/M20_clock_*.png`；尚未用真實數據測試）。語法皆通過 `bash -n`／AST 檢查；`mcast_query.cu` 尚未編譯（登入節點無 nvcc，會在工作內編譯，失敗原文會存進 `nvls/mcast_compile.txt`）。
- 實驗卡「指令」段已補上；「預期」段未動。
- 設計備註：時脈負載的 `-n` 由工作內的校正跑（`-n 200 -w 20`）量出的單次迭代時間算出（n = 60 s /（out-of-place + in-place 時間）），估算過程存於 `clock/calibration_result.txt`；負載與校正用 `-c 0`（不做資料檢查）。負載視窗從 launch 開始，包含 NCCL 初始化的數秒。SHARP 預檢的環境變數只在子命令內設定，不影響後面 NVLS 測試的 NCCL 版本。

## 工作清單（皆以 `--dependency=afterok:475444:475459` 送出）

| 工作 | 編號 | 排隊區 | 狀態（送出時） | 節點 | 結果路徑 |
|---|---|---|---|---|---|
| H100 第 1 台（node.sbatch） | 475733 | dev | PENDING (Dependency) | 未定 | 尚無 |
| 2 台 SHARP（sharp.sbatch） | 475734 | normal -N 2 | PENDING (Dependency) | 未定 | 尚無 |
| H200 第 1 台（node.sbatch） | 475735 | normal2 | PENDING (Dependency) | 未定 | 尚無 |

`squeue --start` 顯示三個工作的預估開始時間為 N/A（因依賴未滿足）。被依賴的 S01 工作預估開始時間已**延後**：475459 → 2026-11-18 11:18、475444 → 2026-11-18 12:29（先前估 11-17 04:00–06:00）。所以 H100 第 1 台最早也要 11-18 中午之後才可能開始，我無法在這個對話內等到。

## 異常（照實記錄）

- 三次送出都出現叢集警告：`sbatch: WARNING: Exclusive mode is not allowed. --exclusive is removed.` 工作仍被接受。意思是叢集會移除 `--exclusive`；我們仍要求 8 GPU 與 8×12=96 CPU（每台 112），但**不保證獨佔整台**（記憶體預設整台）。腳本在開跑前有檢查 GPU 上沒有其他行程（R13），但別人的 CPU／網路工作無法排除。（S01 的工作也同樣用 `--exclusive`，可能同樣被移除，請 Opus 確認。）此事需 Opus／使用者決定是否可接受（D048「量測一律獨佔整台」）。
- 2 台 SHARP 的啟動方式、sharp_hello、NVLS 等皆尚未執行，無原文可引。

## 尚未完成

- H100 第 2、3 台、H200 第 2 台尚未送（依規格書，須第 1 台確認正常後才送）。
- P1–P10 觀察、節點對照表、時脈表、SHARP 預檢原文：全待工作執行後；執行完用 `python3 exp/M20_env_snapshot/analyze.py` 產生表格。
- 未 commit、未 push。
