# S01 環境建置（Nano5）

- **寫給**：Sonnet 5.5 執行者　**撰寫**：Opus 5.5，2026-11-16（D050）
- **類型**：環境建置，**不是量測**。產生的任何速度數字都不分析、不當結論。
- **報告**：`docs/reports/R01_env_setup.md`（格式見文末）

## 目標

讓這台機器上有一套可重現、版本固定的工具，供之後的通訊量測（M20–M24）與推論實驗使用：PyTorch＋vLLM、nccl-tests（與 PyTorch 用同一版 NCCL）、論文搜尋工具。交出「哪些版本能用、裝在哪、怎麼驗證過」的清單，以及一個啟用環境的腳本。

## 背景（已知與已排除）

- 叢集：國網中心 Nano5。登入節點 `cbi-lgn01`；計算節點每台 8× H100 80GB HBM3（`dev` 2 h、`normal` 48 h、`4nodes` 12 h）或 8× H200（`normal2` 48 h）。每台 112 CPU、約 1.9 TB 記憶體、8× InfiniBand NDR 400 Gb/s。說明書：https://man.twcc.ai/@AI-Pilot/manual
- 計算節點驅動 550.127.08（工作 475347 量到）。原生支援到 CUDA 12.4；CUDA 12.x 的小版本相容性**可能**讓 cu126／cu128／cu129 的套件也能跑，CUDA 13 的不行——**這是推測，要實測**。
- 模組：`cuda/12.4`（預設）、`cuda/12.6`、`cuda/12.2`、`openmpi/4.1.6`、`openmpi/5.0.5`、`nvhpc/24.7`、`nvhpc-hpcx-cuda12/24.7`、`miniconda3/24.11.1`、`gcc/11.5.0`。Slurm 支援 `--mpi=pmix_v4`、`pmi2`。
- 登入節點能連 pypi.org 與 huggingface.co；計算節點能不能上網**未知**（步驟 A 查）。
- 檔案系統：`/home`（2.9 PB 共享）、`/work`（3.6 PB 共享，已有 `/work/allen0117/`）。配額與清除政策**未知**。
- repo 內沒有任何 venv，`third_party/` 不存在。A100 時期的環境（`.venv-vllm` 等）不在這台。
- `uv`／`uvx` 沒裝；`.mcp.json` 指向別的使用者路徑 `/home/rogerlee/...`，所以論文搜尋工具連不上。
- 不要用 A100 時期的原型程式（`exp/M1x_*`），那是之後的實驗。

## 必守規則

1. CLAUDE.md 的必守規則全部適用，尤其：**運算一律經 Slurm**（第 7 條）、不碰 `results/` 既有檔案（第 4 條）、從零開始（第 1 條：不讀 `~/TP_opt`、`~/tp_opt`、`~/flux`）。
2. **登入節點只做**：讀檔、寫檔、`git clone`、`pip`／`uv` 下載與安裝（純下載解壓）、Hugging Face 下載、查 `sinfo`／`sacct`／`wallet`。**編譯（`make`、`nvcc`）與任何用到 GPU 的事**都要送 Slurm 工作。
3. 送工作照 `.claude/skills/slurm-submit/SKILL.md`：
   - 用模板改寫工作腳本。
   - 用 `submit.sh` 送出：編譯與單卡測試加 `--not-measurement`；8 卡或多台的測試也加 `--not-measurement`，但腳本裡要寫 `#SBATCH --exclusive`。
   - **不輪詢 `squeue`**，等待時只用背景的「`sleep N; sacct -j <id>`」查一次。
   - 同時執行的工作不超過 2 個。
   - 工作失敗**不自動重送**同一份腳本；要先看 `.err`，修正原因後才可以再送。同一個問題修兩次還不行，就停下回報。
4. 工作腳本放 `exp/S01_env_setup/`；工作輸出（`.out`／`.err`、測試紀錄）放 `results/S01_env_setup/<YYYYmmdd_HHMMSS>_job<id>/`。
5. 不用 `sudo`，不改系統檔，不動別人的檔案。
6. 不 commit、不 push（Opus 驗收後處理）。

## 步驟

### A. 空間與網路

1. 查 `/home`、`/work` 的檔案系統類型（`df -T`、`mount | grep -E "home|work"`）與個人配額（依類型試 `quota -s`、`lfs quota -hu $USER <path>`、`mmlsquota` 等）。查說明書有沒有清除政策：硬體規格頁 `https://man.twcc.ai/dUGG_heYSv2CSLig9yFpVQ`、注意事項頁 `https://man.twcc.ai/CxQR2qnESZ-n_7vnCklX1g`，以及目錄頁的其他相關頁面。
2. **安裝根目錄**（下文稱 `$ENVROOT`）：
   - `/work` 沒有自動清除 → 用 `/work/allen0117/comm_opt_env`
   - 否則 → 用 `/home/allen0117/comm_opt_env`

   在報告寫明理由。
3. 送一個 `dev`、1 GPU、2 分鐘的工作，在計算節點上跑 `curl -sI https://pypi.org/simple/`、`curl -sI https://huggingface.co`、`curl -sI https://github.com`，並記錄 `nvidia-smi` 的 CUDA Version 欄位。

### B. Python 工具

1. 在登入節點執行 `python3 -m pip install --user uv`（裝到 `~/.local/bin`）。記錄版本。

### C. PyTorch＋vLLM

1. 查 PyPI 與 vLLM 官方安裝文件，找出截至今天**最新的穩定版 vLLM**、它要求的 torch 版本，以及有哪些 CUDA 版本的變體（預設 wheel 用哪個 CUDA；有沒有 cu126／cu128 等變體的安裝方式）。把查到的內容附網址寫進報告。
2. 建立 `$ENVROOT/venvs/vllm`（用 `uv venv`，Python 版本照 vLLM 要求，建議 3.12）並安裝。如果預設 wheel 需要 CUDA 13，改裝 CUDA 12.x 的變體。
3. 送 `dev`、8 GPU、`--exclusive` 的測試工作，依序確認：
   1. `torch.cuda.is_available()`、`torch.cuda.device_count()==8`、裝置名稱、`torch.version.cuda`、`torch.cuda.nccl.version()`。
   2. `torchrun --nproc-per-node 8` 做一次 `all_reduce`（每個 rank 放自己的編號，檢查總和為 28）。
   3. vLLM 離線推論，`tensor_parallel_size=8`：
      - **模型**：選一個不需要申請權限的小模型，`config.json` 的 `num_attention_heads` 與 `num_key_value_heads` 都要能被 8 整除。先試 `Qwen/Qwen3-0.6B`，下載前先讀它的 `config.json` 確認；不符合就另找一個並寫明。
      - **下載**：權重在登入節點下載到 `$ENVROOT/hf_cache`（設 `HF_HOME`）。
      - **執行**：生成 16 個 token，記錄 `vllm.__version__` 和輸出文字。
      - **紀錄**：把 vLLM 啟動 log 中跟 all-reduce、通訊後端有關的行**原文**存檔（例如 custom all-reduce、symmetric memory、pynccl、NCCL 版本、FlashInfer 相關字樣）。不解讀，只摘錄。
4. 如果最新版跑不起來：記錄錯誤原文，改試次新的穩定版，最多再試 2 個版本。全部失敗 → 停下回報。

### D. nccl-tests

1. 在登入節點 `git clone https://github.com/NVIDIA/nccl-tests third_party/nccl-tests`，切到最新的 release tag，記錄 tag 與 commit。
2. NCCL 來源：用 C 步驟 venv 裡 PyTorch 帶的 NCCL（`nvidia-nccl-cu12` 之類的套件；找出 `libnccl.so.2` 和 `nccl.h` 的路徑）。
   - 套件裡沒有標頭檔 → 從 NVIDIA 的 NCCL GitHub 取**同版本**的標頭，並記錄做法。
   - **不要**混用其他版本的 NCCL。
3. 送一個編譯工作（`dev`、1 GPU、`--not-measurement`）：
   - 載入 `cuda/12.4`（或 C 步驟確認能用的版本）和 `openmpi/4.1.6`
   - 執行 `make MPI=1 MPI_HOME=<openmpi 路徑> CUDA_HOME=<cuda 路徑> NCCL_HOME=<venv 的 nccl 路徑> NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90"`
4. 單台冒煙測試（`dev`、8 GPU、`--exclusive`）：
   - `NCCL_DEBUG=INFO`、`LD_LIBRARY_PATH` 指向 venv 的 NCCL 目錄
   - 先試 `srun --mpi=pmix_v4 --ntasks-per-node=8 --gpus-per-node=8 third_party/nccl-tests/build/all_reduce_perf -b 8 -e 128M -f 4 -g 1 -n 5 -w 2`
   - 不行再試 `pmi2`，或在工作內用 `mpirun`
   - 記錄：哪種啟動方式可行、執行時印出的 NCCL 版本、`#wrong` 欄位全為 0
5. 跨機器冒煙測試：
   - H100：`normal`、2 台各 8 GPU、`--exclusive`、`--ntasks-per-node=8`，同樣的 all_reduce。從 `NCCL_DEBUG=INFO` 摘錄傳輸方式的行原文（`NET/IB`、`GDRDMA`、使用了哪些 `mlx5_*`）。
   - H200：在 `normal2` 做「單台 8 GPU」與「2 台」各一次。
   - 速度欄位**不解讀**，只確認能跑、`#wrong` 為 0。
6. 單一工作排隊超過 30 分鐘（用一次延遲查詢確認）就先記錄「仍在排隊」，繼續做其他步驟，最後在報告列出。

### E. 啟用環境的腳本

寫 `exp/lib/nano5_env.sh`（之後所有工作腳本 `source` 它），內容：
- `module load` 確定可用的 cuda 與 openmpi 版本
- 啟用 `$ENVROOT/venvs/vllm`
- 設 `HF_HOME`、`LD_LIBRARY_PATH`（venv 的 NCCL 放最前面）、`NCCL_TESTS=<repo>/third_party/nccl-tests/build`

腳本不得含個人信箱或密碼。C、D 的測試工作最後改用這個腳本再跑一次單台 8 GPU 的 all_reduce 與 torch all_reduce，確認腳本本身正確。

### F. 論文搜尋工具

1. 修改 repo 根目錄的 `.mcp.json`：
   - `command` 兩處改為 `/home/allen0117/.local/bin/uvx`
   - arxiv 的 `--storage-path` 改為 `/home/allen0117/comm_opt/papers/arxiv`
   - 版本號（`arxiv-mcp-server@0.8.2`、`paper-search-mcp@0.1.4`、`--with "mcp<2"`）**不變**
2. 在登入節點預先下載並確認能啟動：用 `timeout 20` 執行兩個 server 的啟動指令。這兩個是 stdio server，會等待輸入，被 timeout 結束是正常的，只要沒有匯入錯誤或缺套件錯誤就算通過。記錄輸出。
3. 報告註明：Claude Code 要重新啟動，新的 MCP 設定才會生效。

## 產出

1. `docs/reports/R01_env_setup.md`
2. `exp/S01_env_setup/`：所有工作腳本
3. `exp/lib/nano5_env.sh`
4. `.mcp.json`（改路徑）
5. `results/S01_env_setup/...`：工作輸出與 log

## 驗收標準（Opus 會逐項核對）

- [ ] `$ENVROOT` 的位置與理由；配額與清除政策（查不到要寫「查不到」）
- [ ] 計算節點能否上網
- [ ] vLLM 版本、torch 版本、torch 的 CUDA 版本、NCCL 版本；8 卡 torch all_reduce 正確；vLLM TP=8 生成成功
- [ ] nccl-tests 的 tag／commit、編譯參數；執行時印出的 NCCL 版本**等於**PyTorch 的 NCCL 版本
- [ ] 單台 H100、單台 H200、2 台 H100、2 台 H200 的 all_reduce 冒煙測試：成功或失敗原因；可行的啟動方式
- [ ] `exp/lib/nano5_env.sh` 能用（有工作編號為證）
- [ ] 兩個論文搜尋工具能啟動
- [ ] 每個「成功」都有工作編號與 `results/` 路徑

## 停下回報的情況

- 需要 `sudo` 或管理員權限
- 配額不夠放 venv＋模型（約 30 GB）
- 最新與次新的兩個 vLLM 版本都無法在驅動 550 上執行
- NCCL 版本無法和 PyTorch 一致
- 同一個錯誤修兩次仍失敗
- 任何規格沒寫到、需要自行判斷「要不要做」的事

## 報告格式（`docs/reports/R01_env_setup.md`）

1. **摘要**：每個驗收項目一行，標成功、失敗或未完成
2. **版本與路徑表**：元件、版本、安裝位置、安裝方式、驗證工作編號、`results/` 路徑
3. **各步驟紀錄**：做了什麼、指令、結果；錯誤附原文
4. **證據清單**：每個論點的出處（網址、檔案路徑、工作編號、log 檔與行）
5. **和規格不同的地方**及原因
6. **沒解決的問題**

只寫事實與觀察，不下研究結論，不解讀速度數字。
