# S07 準備並驗證 M25（選題用的單節點量測）

- **寫給**：Sonnet 5.5 執行者　**撰寫**：Opus 5.5，2026-11-17 13:40
- **類型**：修環境、寫腳本、送一個 2 卡的驗證工作（非量測）、整理驗證結果
- **實驗卡（唯一依據）**：`exp/M25_topic_survey/README.md`。區塊、設定、時間預算、預期都在卡上。**不改卡上的「預期」與「判讀規則」**；只補「指令」段
- **報告**：`docs/reports/R07_M25_prep.md`
- **最終目標**：Opus 在你交回後，用 `submit.sh` 把主工作送到 `dev`（2 小時、8 卡）。主工作只有一次機會（整台節點排隊很久），所以**每條程式路徑都要先在驗證工作裡實際跑過一次**

## 背景事實（Opus 已確認）

- `dev` 單一工作上限 2 小時、一次最多 1 台；`normal` 試算要到 11-18 20:05 才排得到，所以主工作用 `dev`
- M20（`results/M20_env_snapshot/20261117_095743_job475780/`）：
  - `srun --mpi=pmix_v4` 和 `pmi2` 都失敗，**只有 `mpirun` 能用**
  - 單機 NVLS 小訊息正常，但建立 512 MB multicast 群組後出現 `transport/nvls.cc:257 NCCL WARN Cuda failure 999 'unknown error'`（`nvls/nvls_a_default.failed_mpirun.log:678`）
  - 計算節點沒有 C++ 編譯器（`cc1plus`）；要編譯請 `module load gcc/12.5.0`
  - 驅動 550.127.08 對應 CUDA 12.4
- **venv 的 Python 壞了**：`/work/allen0117/comm_opt_env/venvs/vllm/bin/python -> /usr/bin/python3.12`，計算節點沒有這個檔案（S01 的 475444 出現 `bad interpreter`）。`pyvenv.cfg`：`home = /usr/bin`、`version_info = 3.12.13`、`uv = 0.13.8`。分析用 venv（`venvs/analysis`）也是 `home = /usr/bin`
- `uv` 在 `~/.local/bin/uv`。`/home` 空間很緊：所有快取、下載都放 `/work/allen0117/comm_opt_env/` 底下
- venv 內：torch 2.13.0+cu129、vLLM 0.32.0+cu129、flashinfer 0.7.0.post1、nvidia-nccl-cu12 2.29.7
- NCCL 2.29.7 有這些環境變數（`strings` 已確認）：`NCCL_CTA_POLICY`、`NCCL_MIN_CTAS`、`NCCL_MAX_CTAS`、`NCCL_NVLS_NCHANNELS`、`NCCL_NVLS_CHUNKSIZE`、`NCCL_LOCAL_REGISTER`、`NCCL_P2P_USE_CUDA_MEMCPY`
- NCCL 原始碼在 `/work/allen0117/comm_opt_env/src/nccl`（目前是 `v2.32.6-1`）。要看 2.29.7 的程式碼，用 `git -C ... show v2.29.7-1:<路徑>`（唯讀，不要切換工作目錄的分支）
- HuggingFace 從登入節點連得到

## 要做的事

### 1. 修 Python（登入節點，不需要 GPU）

1. `UV_PYTHON_INSTALL_DIR=/work/allen0117/comm_opt_env/python ~/.local/bin/uv python install 3.12.13`（版本要和 `version_info` 完全一樣）
2. 把兩個 venv 的 `bin/python` 改指到新的直譯器，並把 `pyvenv.cfg` 的 `home` 改成新直譯器所在的資料夾。**不要重裝套件**。改之前先備份 `pyvenv.cfg`（例如 `pyvenv.cfg.bak_20261117`）
3. 在登入節點只做最輕的檢查：`venv/bin/python -c "import sys; print(sys.executable, sys.version)"`。`import torch`、`import vllm` 留到驗證工作裡在計算節點做

### 2. 下載模型設定檔（登入節點，只下載小檔）

- 放在 `/work/allen0117/comm_opt_env/models/<名稱>/`；`HF_HOME` 用 `/work/allen0117/comm_opt_env/hf_cache`
- 模型：`Qwen/Qwen2.5-72B-Instruct`、`Qwen/Qwen3-30B-A3B`，以及驗證用的 `Qwen/Qwen2.5-7B-Instruct`
- 只下載 `*.json`、`*.txt`、`tokenizer*`、`merges.txt`、`vocab.json`；**不下載權重**（`*.safetensors`、`*.bin`、`*.pt`）。主工作用 vLLM 的 `load_format="dummy"`，隨機權重的效能和真權重相同
- 在報告附上每個資料夾的檔案清單與總大小

### 3. 寫主工作（放 `exp/M25_topic_survey/`）

- `job.sbatch`：照 `.claude/skills/slurm-submit/template.sbatch`。`-p dev`、`--nodes=1`、`--gpus-per-node=8`、`--cpus-per-task=96`、`--exclusive`、`--time=02:00:00`、輸出到 `results/M25_topic_survey/slurm/`。主體只做一件事：`source exp/M25_topic_survey/m25_lib.sh` 後呼叫 `m25_main`。**`m25_lib.sh` 在執行時才讀入**，排隊期間還能修正
- `m25_lib.sh`：
  - 讀 `exp/lib/nano5_env.sh`；所有快取設到 `/work/allen0117/comm_opt_env/cache/`（`VLLM_CACHE_ROOT`、`TORCHINDUCTOR_CACHE_DIR`、`TRITON_CACHE_DIR`、`XDG_CACHE_HOME`、`FLASHINFER_WORKSPACE_BASE` 若有）
  - **一個全域截止時間**：工作開始後 112 分鐘。每個區塊有自己的上限（卡上：B4 8 分、B1 30 分、B2 40 分、B3 25 分）。每次啟動前檢查：剩餘時間不夠就跳過並記錄
  - 每次外部啟動都包 `timeout`；失敗只記 exit code、繼續下一個
  - `$OUT/blocks.tsv`：每個區塊的開始、結束、狀態、跳過了幾個設定
  - 全程背景記錄 GPU 狀態：`nvidia-smi --query-gpu=timestamp,index,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,clocks_throttle_reasons.active --format=csv -lms 500 > $OUT/gpu_monitor.csv`；用 trap 在結束時停掉
  - 區塊順序：B0 → B4 → B1 → B2 → B3（卡上的順序）
- **B0**：模板的環境紀錄、GPU 行程檢查（有其他行程就中止）、`cotenants.sh`；60 秒暖機（`mpirun` 跑 nccl-tests 8 卡 all_reduce 64 MB，用 `-n` 讓總時間約 60 秒）
- **B4（NVLS 當機）**：
  - 從 `exp/M20_env_snapshot/node.sbatch` 第 79–90 行附近找出 `nvls_a_default` 的 `$BIN $ARGS` 與環境變數；用 `mpirun` 完全照樣重跑 2 次
  - 同樣指令但最大訊息改為 `-e 256M`，跑 1 次
  - 用 `git show v2.29.7-1:src/transport/nvls.cc` 找出 2.29.7 第 257 行附近呼叫的是哪個 CUDA 函式。依此最多再加 2 個診斷用的變體（例如改 `NCCL_NVLS_CHUNKSIZE`、`NCCL_LOCAL_REGISTER=0`），在報告寫明理由
  - 每次 `timeout 150`；`NCCL_DEBUG=INFO`
- **B1（通訊與矩陣乘法同時執行）**：寫 `overlap_bench.py`，用 `torchrun --nproc-per-node 8` 啟動（單機不需要 PMI；若 torchrun 有問題，在報告說明替代方式）
  - 卡上的 3 種集體通訊 × 3 種大小 × 2 種矩陣乘法 = 18 格。大小的定義：all_reduce 的輸入、reduce_scatter 的輸入、all_gather 的輸出都等於該大小（bf16）
  - **校準啟動**（不算量測、只跑一次、用預設 NCCL 設定）：決定每種矩陣乘法大小的次數 `n_g`（單獨約 100 ms），以及每格的通訊次數 `n_c`（單獨時間約為矩陣乘法的一半），寫成 `calib.json`。之後所有量測啟動都讀同一個檔，確保工作量一樣
  - **量測啟動**：卡上 10 種 NCCL 設定 × 2 次，用固定種子 20261117 打亂順序，記錄實際順序。每次啟動用環境變數設定 NCCL，跑完全部 18 格
  - 每格三種情況各 3 次：矩陣乘法單獨、通訊單獨、兩者同時。每次開始前所有 rank 先 barrier 並同步。「同時」的做法：在預設 stream 上先放 `torch.cuda._sleep(...)`（約 1 ms，讓 CPU 有時間把工作排進去）再 record 起點事件；矩陣乘法 stream 和通訊 stream 都 `wait_event` 這個起點，然後各自排入工作；用 CUDA event 量各自結束的時間
  - 每個 rank 的時間 gather 到 rank 0，每個指標取跨 rank 的最大值。一行寫一筆 JSONL：設定、啟動編號、格、第幾次、`t_gemm_alone`、`t_comm_alone`、`t_gemm_conc`、`t_comm_conc`、`makespan`
  - NCCL 日誌：`NCCL_DEBUG=INFO`、`NCCL_DEBUG_SUBSYS=INIT,ENV`、`NCCL_DEBUG_FILE` 每行程一檔（從中取得實際 channel／CTA 數）。**不要開 TUNING**（每次呼叫都寫日誌會拖慢 CPU 排工作的速度）
  - 估計每次啟動約 45–60 秒；在報告寫出驗證工作量到的實際時間，以及推算出的 B1 總時間
- **B2（vLLM）**：寫 `vllm_bench.py`，一個引擎一個 Python 行程（引擎之間 GPU 記憶體要完全釋放）
  - 先讀**已安裝的** vLLM 0.32.0 原始碼（`venvs/vllm/lib/python3.12/site-packages/vllm/`），確認：`LLM` 的 `load_format="dummy"`、`enable_expert_parallel`、profiler 的開法（`start_profile`／`stop_profile` 與 profiler 設定在這一版的名稱，可能是環境變數也可能是參數），以及要怎麼知道它選了哪一種加總實作（自製 all-reduce、NCCL、symmetric memory、FlashInfer 等）。在報告附 `檔案:行號`
  - 輸入用隨機 token id（`TokensPrompt`），`SamplingParams(max_tokens=..., ignore_eos=True, temperature=0)`
  - 每字時間：同一批次分別生成 1 個與 128 個 token，各 3 次（之前先暖機 1 次）；每字時間 =（t128 − t1）／127。預填：批次 1、輸入 L、生成 1 個 token，3 次
  - profiler：只錄少數解碼步（例如生成 8 個 token），讓 trace 小一點。**在工作內把 trace 解析成 kernel 時間摘要 CSV**：依 kernel 名稱分成加總類通訊、其他集體通訊（all-gather、reduce-scatter、all-to-all、MoE dispatch／combine）、矩陣乘法、注意力、其他；分類用的名稱規則寫在程式裡並在報告列出。原始 trace 壓縮保存；任何檔案 > 50 MB 就不放進 `results/`，改放 `/work/allen0117/comm_opt_env/traces/` 並在 `results/MANIFEST_large_files.txt` 記 sha256
  - CUDA graph 模式下若 trace 裡沒有個別 kernel（只有 graph launch），主工作改用 eager 模式另開引擎抓比例。**由驗證工作決定是哪一種**，在報告寫明
  - 引擎清單與順序照卡上 B2；每個引擎 `timeout` 12 分鐘。`max_model_len`：需要 16k 預填的引擎用 17408，其他用 2048。`gpu_memory_utilization=0.85`
  - 引擎日誌全部保存；記錄每個引擎的啟動時間（從開始到第一次能生成）
- **B3（nccl-tests）**：能用就沿用 `exp/lib/nccl_bench.sh` 的 `add_cfg`／`shuffle_plan`／`run_plan`（M21／M23 共用的工具）；不適合就照它的格式另寫，在報告說明
  - 設定照卡上 B3；2 次、打亂順序；`-a 3`；8 B–1 GB 分兩段：小段 `-b 8 -e 64M -f 2 -n 200 -w 5`，大段 `-b 128M -e 1G -f 2 -n 20 -w 5`
  - 每個設定另跑一次 `-n 1 -w 0 -c 0` 並開 `NCCL_DEBUG_SUBSYS=TUNING`，留下演算法選擇的日誌
  - 2 卡、4 卡用 `CUDA_VISIBLE_DEVICES`
- **收尾**：`summary.txt`（每個區塊做了幾個、失敗幾個、跳過幾個）、壓縮大日誌
- `analyze.py`：讀 B1 JSONL、B2 摘要 CSV 與每字時間、B3 的 nccl-tests 輸出；照卡上每條預期算出要判定的數字，輸出表格（判定由 Opus 做）

### 4. 驗證工作（2 卡，非量測）

- `validate.sbatch`：`-p dev`、`--gpus-per-node=2`、`--cpus-per-task=24`、`--time=00:40:00`，**不加 `--exclusive`**，用 `submit.sh --not-measurement` 送出。輸出到 `results/M25_topic_survey/validation/`
- 內容：**和主工作同一份 `m25_lib.sh`**，用 `M25_MODE=smoke`、`M25_NGPU=2` 縮小規模，但每條程式路徑都要走過：
  - 計算節點上：`python -c` 印 `sys.executable`、`torch.__version__`、`torch.cuda.is_available()`、`torch.cuda.nccl.version()`、`vllm.__version__`
  - B0（暖機 5 秒）、B4（2 卡、小訊息、各 1 次）、B1（校準＋2 種設定 × 1 次，只跑 2 格）
  - B2：`Qwen2.5-7B-Instruct` TP=2（每字時間、預填 1k、profiler，並判斷 CUDA graph 下抓不抓得到個別 kernel）；`Qwen3-30B-A3B` TP=2 與 TP=2＋專家平行（各生成幾個 token 就好）
  - B3（2 卡 all_reduce 1 個設定、sendrecv 1 個設定）、`analyze.py` 跑過一次
- 等結果：照 `.claude/skills/slurm-submit/SKILL.md` 第 4 步，**背景延遲後查一次 `sacct`，不輪詢**
- 失敗就修、再送一次驗證（最多 3 次）。第 3 次還不行就停下來，在報告寫明原因

### 4b. 2026-11-24 補充（Opus；使用者：維持 8 卡，「務必收集好所有做題目決策需要的資料」）

- **不能因為時間不夠而丟掉任何區塊**。依驗證工作量到的實際耗時推算主工作；若估計總時間 > 105 分鐘，就把主工作拆成兩個 2 小時的 `dev` 工作（例如 `job_a.sbatch`：B0＋B4＋B1＋B3；`job_b.sbatch`：B0＋B2），兩者共用同一份 `m25_lib.sh`，用環境變數選區塊，兩個都要在驗證工作中走過。在報告寫明拆法與每個工作的估計時間
- 每個工作仍保留 B0（環境紀錄、GPU 行程檢查、暖機、時脈紀錄）
- 上次中斷前已完成：venv Python 改指 `/work/allen0117/comm_opt_env/python/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12`、三個模型設定檔已下載、`overlap_bench.py` 與 `vllm_bench.py` 已寫。先檢查這些是否完整正確，再繼續

### 5. 交回

- **不要送主工作**（Opus 審過後送）。先用 `submit.sh --dry-run exp/M25_topic_survey/job.sbatch` 確認能通過檢查
- 寫報告後交回

## 報告 `docs/reports/R07_M25_prep.md`

1. Python 修正：改了什麼、備份在哪、計算節點上的檢查輸出
2. 模型設定檔清單
3. 建立的檔案與用途
4. 驗證工作：編號、每個區塊的結果、實際耗時（尤其 B1 每次啟動與 B2 每個引擎的啟動時間），以及據此推算的主工作各區塊時間；是否塞得進 112 分鐘
5. B4 讀到的 NCCL 2.29.7 原始碼（`檔案:行號`）與加入的診斷變體
6. vLLM 0.32.0 原始碼確認（`檔案:行號`）；profiler 在 CUDA graph 下是否抓得到 kernel、主工作用哪種模式
7. 和實驗卡不同、或卡上沒寫清楚而由你決定的地方（逐條列，讓 Opus 審）

## 規則

- 只送驗證工作（`--not-measurement`）；不送主工作；不動已排隊的工作（475781、475783、475784）
- 不改實驗卡的「預期」「判讀規則」；不 commit、不 push（Opus 會做）；不用 `sudo`；不讀 `~/TP_opt`、`~/tp_opt`、`~/flux`
- 登入節點只做編輯、下載小檔、`git show`、輕量檢查；任何用到 GPU 或大量 CPU 的事都經 Slurm
- 用繁體中文寫報告
