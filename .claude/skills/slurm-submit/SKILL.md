---
name: slurm-submit
description: Submit any job on the Nano5 (NCHC) cluster through Slurm, the project's standard way. Use for EVERY task that needs a compute node — measurements, GPU tests, builds that need a GPU, smoke tests. Covers the job-script template, the submit helper (checks + mail), waiting without polling, and where outputs go. Never run compute on the login node.
---

# slurm-submit：在 Nano5 上發任務的標準流程

這台是國網中心晶創主機（Nano5）。登入節點（`cbi-lgn01`）只能看資料、改檔案、編譯；**所有運算都經 Slurm 送到計算節點**。
使用說明：https://man.twcc.ai/@AI-Pilot/manual（排隊區列表：`/aPiCU8VXS7SZgFJBOoSqBQ`，注意事項：`/CxQR2qnESZ-n_7vnCklX1g`）。

## 硬規則（違反可能被砍工作或鎖帳號）

1. **不在登入節點跑運算**（登入節點看得到 H100 PCIe，也不能用）。
2. **不輪詢**：不用 `watch`、迴圈、Monitor until-loop 查 `squeue`。等工作只用「背景延遲後查一次」（見步驟 4）或等結束通知信。
3. **送出前要有授權**：只送 `docs/status.md` 已核准的實驗，或使用者在這次對話明確說可以的任務。新增或改變實驗先問（CLAUDE.md 第 6 條）。
4. **量測一律獨佔整台**（`--exclusive`，使用者 2026-11-16 決定）。非量測（編譯、流程測試）可不獨佔，但要在送出時明說。
   - **2026-11-17 發現**：叢集會拿掉 `--exclusive`（`sbatch: WARNING: Exclusive mode is not allowed. --exclusive is removed.`），`dev`／`normal`／`normal2` 都是。實際效果：申請 8 張 GPU 時，叢集上限是每 GPU 12 顆 CPU（共 96／104）、預設記憶體 1600G／約 1.9 TB，別人的 GPU 工作排不進來，但小型純 CPU 工作理論上可能。對策（D054）：量測工作一律 `source exp/lib/cotenants.sh`（模板已含），開始與結束各記錄一次同節點上的所有工作；分析時有其他工作的數據要標出來。
5. **帳號**一律 `MST114546`；**通知信**寄到 `~/.config/comm_opt/slurm.env` 裡的 `SLURM_MAIL_USER`（信箱不寫進 repo，因為 repo 是公開的）。
6. **輸出放 `/home` 底下**（計算節點看不到登入節點的 `/tmp`）。量測數據照專案慣例寫到 `results/<ID>_<名稱>/<時間>/`。
7. 不用 `sudo`／`apt`；Python 套件用自己的 Miniconda 或專案 `.venv`。

## 排隊區（2026-07-01 版說明書）

| 排隊區 | 每計畫 GPU 上限 | 單一工作最長 | 每人同時執行 | GPU |
|---|---|---|---|---|
| `dev` | 8 | 2 h | 2 | H100 80GB HBM3（SXM） |
| `normal` | 16 | 48 h | 2 | H100 |
| `normal2` | 16 | 48 h | 2 | H200 |
| `4nodes` | 32 | 12 h | 2 | H100 |

選法：試跑、5 分鐘內的小工作 → `dev`；正式量測 → `normal`（H100）或 `normal2`（H200）；跨節點 → `4nodes`。
每台：8 GPU、112 CPU、約 1.9 TB 記憶體、8×400 Gbps InfiniBand。不寫 `--mem` 時預設拿整台記憶體。

## 步驟

### 1. 寫工作腳本

複製 `template.sbatch`（同資料夾）到實驗資料夾，例如 `exp/<ID>_<名稱>/job.sbatch`，只改 `<...>` 佔位符：
- `--job-name=<ID>`，`--partition`，`--time`（估計時間 ×1.5，不要灌水，會影響排隊）
- `--output`／`--error` 指到 `results/<ID>_<名稱>/slurm/%x-%j.out|err`
- 主體指令放在 `### BODY` 區塊

模板已內建：`set -euo pipefail`、在結果資料夾寫 `job_env.txt`（工作編號、節點、時間、git commit、模組、`nvidia-smi`、`nvidia-smi topo -m`）、檢查卡上沒有別的行程。

### 2. 用輔助腳本送出（不要直接 `sbatch`）

```bash
.claude/skills/slurm-submit/submit.sh exp/<ID>_<名稱>/job.sbatch            # 量測（必須有 --exclusive）
.claude/skills/slurm-submit/submit.sh --not-measurement path/to/job.sbatch    # 編譯、流程測試
```

`submit.sh` 會：檢查腳本有 `--account=MST114546`、`--partition`、`--time`、`--output`；量測要有 `--exclusive`；查 `wallet` 餘額 > 0；查自己同時執行的工作 < 2；建立輸出資料夾；加上通知信參數後 `sbatch --parsable`；把工作編號、時間、腳本、git commit 記到 `~/.config/comm_opt/jobs.tsv`。任何一項不過就停下，不送。

### 3. 回報給使用者

送出後立刻回報：工作編號、排隊區、申請的資源、預估時間、輸出位置。

### 4. 等結果（不輪詢）

用一個**背景** Bash 指令，延遲約為預估執行時間後查**一次**：

```bash
sleep <秒數>; sacct -j <JOBID> --format=JobID,JobName,Partition,State,Elapsed,NodeList,ExitCode
```

收到完成通知後再讀輸出。還沒結束就再排一次延遲查詢（間隔不少於幾分鐘），不要連續查。其他查法（各查一次即可）：`squeue -u $USER`、`scontrol show job <JOBID>`、`sacct -j <JOBID>`。取消：`scancel <JOBID>`。

### 5. 收尾

- 讀 `.out`／`.err`，確認 `State=COMPLETED`、`ExitCode 0:0`。
- 失敗時先看 `.err` 和 `sacct`，**不要自動重送**；把原因告訴使用者。
- 量測結果照專案流程分析、更新實驗卡、commit、push。

## 多行程／多卡

- 單一行程管 8 卡（如 nccl-tests `-g 8`）：`srun -n 1 --gpus-per-task=8 <bin>`
- 一卡一行程：`#SBATCH --ntasks-per-node=8`、`--gpus-per-node=8`，執行 `srun --mpi=pmix_v4 --gpus-per-task=1 <bin>`（此叢集支援 `pmix_v4`、`pmi2`；**尚未在這裡實測過**，第一次用要先小規模試）。
- `torchrun`：`srun -n 1 torchrun --nproc_per_node=8 ...`
- 互動式（少用，會一直計費直到 `exit`）：`salloc -A MST114546 -p dev --gpus-per-node=8 --exclusive -t 0:30:00`，用完立刻 `exit`。

## 已驗證紀錄

- 2026-11-16 流程測試：工作 475347（`dev`、1 GPU）→ 節點 `hgpn03` → COMPLETED，驅動 550.127.08。
