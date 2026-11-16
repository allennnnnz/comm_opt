# S06 準備 M21（單機基準）與 M23（跨機器基準）

- **寫給**：Sonnet 5.5 執行者　**撰寫**：Opus 5.5，2026-11-17
- **類型**：寫腳本、乾跑檢查、送已核准的 Slurm 工作（M21、M23 皆在 D051 範圍內核准）
- **實驗卡（唯一依據）**：`exp/M21_intranode_baseline/README.md`、`exp/M23_internode_baseline/README.md`——預期、設定、統計方式、**「Opus 補充」段落**都在卡上。不改卡上的「預期」段落；只在「指令」段補上實際指令
- **報告**：`docs/reports/R06_M21_M23_prep.md`

## 背景

- 叢集很滿（2026-11-17 01:20 試算）：`dev` 單台數小時；`normal` 約 11-19；`4nodes`（只接受 4 台）約 11-19；`normal2` 約 11-26。
- S01 驗證工作 475444、475459 仍在 `dev` 排隊。M20 的工作由另一位小助手處理（S05），**不要動 M20 的檔案或工作**。
- 環境：`source /home/allen0117/comm_opt/exp/lib/nano5_env.sh`；nccl-tests 在 `$NCCL_TESTS`。多行程啟動方式依序試 `srun --mpi=pmix_v4`、`pmi2`、`mpirun`（照 `exp/S01_env_setup/d4_allreduce_smoke.sbatch`），用第一個成功的並記錄。
- SHARP 外掛的環境設定照 `exp/M20_env_snapshot/README.md` 的「Opus 補充」：**不要** `module load nvhpc-hpcx-cuda12/24.7`；手動設 `LD_LIBRARY_PATH`（venv NCCL 在最前）。

## 要做的事

1. **讀**：兩張實驗卡（含「Opus 補充」）、`.claude/skills/slurm-submit/SKILL.md`、`template.sbatch`、`submit.sh`、`exp/S01_env_setup/d4_allreduce_smoke.sbatch`、`exp/M20_env_snapshot/README.md` 的「Opus 補充」段落。
2. **確認 nccl-tests 參數**：在 `/home/allen0117/comm_opt/third_party/nccl-tests/src/` 原始碼中確認「跨 rank 取最大值」的參數與數值對應（卡上要求跨 rank 取 max）、`-n`／`-w` 的意義、`alltoall_perf` 與 `sendrecv_perf` 的位元組定義。在報告附 `檔案:行號`。
3. **M21 腳本**（放 `exp/M21_intranode_baseline/`）：
   - `node.sbatch`：單台、`dev`、`--exclusive`、`--time` ≤ 1:40。一個工作內做完卡上全部設定（6 種集體通訊 × 卡數 2／4／8 × NVLS 開／關；8 卡 all_reduce 的演算法與 protocol 變體），每個設定 5 次獨立啟動、**順序打亂**（用固定的隨機種子，記錄實際順序）。開頭先跑卡上「Opus 補充」的暖機。
   - 2、4 卡用 `CUDA_VISIBLE_DEVICES` 選前 2／4 張卡，記錄實際卡號。
   - **先估時間**：每個設定的單次執行時間用規格推算不出來時，在報告寫出你的估算方法；若估計總時間 > 1.5 小時，拆成兩個腳本（例如 `node_a.sbatch`、`node_b.sbatch`）並說明怎麼拆。
   - `analyze.py`：卡上要求的中位數與 95% bootstrap 信賴區間（跨 5 次啟動）、圖表。
   - **M21 只做乾跑（`submit.sh --dry-run`），不要送出**——Opus 看過 M20 第 1 台的結果後才送。
4. **M23 腳本**（放 `exp/M23_internode_baseline/`）：
   - `h100_4node.sbatch`：`-p 4nodes -N 4`、`--exclusive`、`--time` 依估算（≤ 12:00）。先跑 4 台的全部設定，再依卡上「Opus 補充」分成兩台一組跑 2 台設定（同組／跨組的判斷用 `SLURM_JOB_NODELIST` 的節點編號：hgpn01–06 一組、hgpn17–21 一組）。
   - `h200_2node.sbatch`：`-p normal2 -N 2`。
   - 每個設定 5 次獨立啟動、順序打亂；SHARP／NVLSTree 等演算法照卡上一律都跑，失敗或退回照實記錄（每次啟動都保留 `NCCL_DEBUG=INFO` 的初始化與演算法選擇日誌）。
   - 單一設定失敗不能讓整個工作中止：每次啟動各自記錄 exit code 後繼續。
   - `analyze.py`：同上；演算法以日誌中實際使用的為準。
   - **送出**：兩個工作都用 `submit.sh <腳本> --dependency=afterok:475444:475459`。送之前先 `--dry-run`。送出後記錄工作編號與 `squeue --start` 的預估時間。**如果送出被權限系統或叢集拒絕**：停下、在報告寫明原文，不要換方法繞過。
5. **寫報告**後交回（不用等 M23 開始跑）。

## 報告 `docs/reports/R06_M21_M23_prep.md`

1. 建立的檔案清單與每個檔案做什麼。
2. nccl-tests 參數確認（`檔案:行號`）。
3. M21：設定總數、估計時間與估算方法、是否拆成多個工作、乾跑輸出。
4. M23：兩個工作的編號、排隊區、申請資源、預估開始時間、乾跑輸出。
5. 和實驗卡不同、或卡上沒寫清楚而由你決定的地方（逐條列出，讓 Opus 審）。

## 規則

- 只送 M23 的兩個工作；M21 不送。任何卡上沒有的量測都不要做。
- 不改實驗卡的「預期」段落；不動 M20 的檔案與工作；不取消任何工作。
- 不 commit、不 push（Opus 會做）；不用 `sudo`；不讀 `~/TP_opt`、`~/tp_opt`、`~/flux`。
- 用繁體中文撰寫報告。
