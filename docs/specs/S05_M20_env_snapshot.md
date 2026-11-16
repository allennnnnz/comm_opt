# S05 執行 M20 環境快照

- **寫給**：Sonnet 5.5 執行者　**撰寫**：Opus 5.5，2026-11-16
- **類型**：寫腳本、送已核准的 Slurm 工作、整理表格（M20 已核准：D051；P10 SHARP：D052）
- **實驗卡（唯一依據）**：`exp/M20_env_snapshot/README.md`——預期 P1–P10、設定、統計方式都在卡上。**不要改卡上的「預期」段落**；只在「指令」段補上實際指令。
- **報告**：`docs/reports/R05_M20_env_snapshot.md`

## 前提

- S01 的 8 卡驗證已通過（Opus 確認後才會交給你）。多行程啟動方式沿用 S01 冒煙測試成功的那一種：看 `results/S01_env_setup/*_job475459/attempts.txt`（依序試過 `srun --mpi=pmix_v4`、`pmi2`、`mpirun`）。
- 環境一律 `source /home/allen0117/comm_opt/exp/lib/nano5_env.sh`；nccl-tests 在 `$NCCL_TESTS`。

## 要做的事

1. **讀**：實驗卡、`.claude/skills/slurm-submit/SKILL.md`（硬規則全部適用）、`template.sbatch`、`exp/S01_env_setup/d4_allreduce_smoke.sbatch`（可參考其寫法）。
2. **寫腳本**（放 `exp/M20_env_snapshot/`）：
   - `node.sbatch`：單台工作，依卡上「每個工作做三件事」執行：快照 → NVLS 檢查（a/b/c 三種設定）→ 時脈行為（5 輪）。從 `template.sbatch` 複製；`--exclusive`；輸出到 `results/M20_env_snapshot/<時間>_job<id>/`，每個指令的原始輸出各存一個檔，檔名清楚。
   - `mcast_query.cu`：在工作內用 `nvcc -arch=sm_90` 編譯，對 8 張卡各印出 `CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED`（用 driver API `cuDeviceGetAttribute`）；編譯失敗要記錄錯誤原文，不要改用別的方法猜。
   - 時脈取樣：`nvidia-smi --query-gpu=... -lms 100` 在背景執行，負載結束後確實結束取樣行程；每輪記錄負載開始與結束的時間戳（同一時鐘）。60 秒負載的 `-n` 先用短的試跑估每次迭代時間再決定，估算過程寫進報告。
   - `sharp.sbatch`（2 台 H100）：**先只寫好、不要送**。SHARP 的環境變數與外掛載入方式要等 Opus 依 `docs/reports/R04_A_two_level_switch_reduction.md` 補充後才送。
   - `analyze.py`：讀 `results/M20_env_snapshot/` 產生卡上「統計方式」要的表（節點對照表、時脈表）與 `figures/M20_clock_*.png`（時脈對時間，每輪一張或疊圖皆可）。所有數字由腳本算，不手抄。
3. **送工作**（用 `.claude/skills/slurm-submit/submit.sh`；**一次只送一個**，前一個結束且確認輸出正常後才送下一個；等待一律「背景延遲後查一次」，不輪詢）：
   1. H100 第 1 台：`dev`。結束後先自己檢查輸出完整（每個檔案非空、NVLS 三種設定都有結果、時脈 5 輪都有），有問題就停下回報，不要自動重送。
   2. H100 第 2、3 台：`dev`，`--exclude=` 已量過的節點。注意 H100 分兩組：hgpn01–06 與 hgpn17–21；三台盡量兩組都有（用 `--exclude` 引導，不用 `--nodelist` 指定，以免排很久；排不到就照實記錄）。
   3. H200 第 1、2 台：`normal2`，第 2 台排除第 1 台。
   4. `sharp.sbatch`：等 Opus 補充後才送。
4. **分析與報告**：跑 `analyze.py`，寫報告。

## 報告 `docs/reports/R05_M20_env_snapshot.md`

1. 每個工作：工作編號、排隊區、節點、開始／結束時間、狀態、結果路徑。
2. 節點對照表（卡上 P8 的各項），標出差異。
3. 對照 P1–P9 逐條列「觀察到什麼」，附 `results/` 路徑與原文（例如 `nvidia-smi topo -m` 的那幾行、NCCL log 中 NVLS 相關行）。**只寫觀察，不寫「預期成立／推翻」**——判定由 Opus 做。
4. 時脈表（卡上指定的指標），附產生它的指令與 commit。
5. 任何異常、失敗、和卡上設定不同的地方（照實寫，附錯誤原文）。

## 規則

- 只送上述工作；任何卡上沒有的量測都不要做。速度數字不分析（那是 M21 的範圍），但原始輸出要保留。
- `results/` 的原始檔不改、不刪。
- 不嘗試修改時脈、驅動設定或任何需要 root 的事；不用 `sudo`。
- 不 commit、不 push（Opus 會做）；不讀 `~/TP_opt`、`~/tp_opt`、`~/flux`。
- 用繁體中文撰寫報告。
