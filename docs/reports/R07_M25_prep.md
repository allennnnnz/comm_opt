# R07 M25 準備報告（Sonnet 執行者）

**狀態：準備完成、驗證工作尚未送出。** 阻塞：`submit.sh` 內的 `wallet` 指令在登入節點壞了（見第 4 節）。我沒有繞過 `submit.sh` 直接 `sbatch`。未 commit、未 push、未送任何工作。

## 1. Python 修正
- `uv python install 3.12.13`（`UV_PYTHON_INSTALL_DIR=/work/allen0117/comm_opt_env/python`，`UV_CACHE_DIR` 在 `/work`）。直譯器：`/work/allen0117/comm_opt_env/python/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12`。
- `venvs/vllm` 與 `venvs/analysis` 的 `bin/python` 改指該直譯器；兩者 `pyvenv.cfg` 的 `home` 改為其所在資料夾；備份 `pyvenv.cfg.bak_20261117`。未重裝套件。
- 登入節點檢查（2026-11-24 重驗）：兩個 venv 的 `python -c "import sys..."` 都印出 3.12.13；`from vllm.inputs import TokensPrompt` 在登入節點可匯入。計算節點上的 torch／vllm／nccl 檢查：寫在 B0 的 `job_env.txt`，**待驗證工作跑完才有輸出**。

## 2. 模型設定檔（`/work/allen0117/comm_opt_env/models/`，各只有 7 個小檔）
- `Qwen2.5-72B-Instruct`（12 MB）、`Qwen2.5-7B-Instruct`（12 MB）、`Qwen3-30B-A3B`（17 MB）：`config.json`、`generation_config.json`、`merges.txt`、`model.safetensors.index.json`、`tokenizer_config.json`、`tokenizer.json`、`vocab.json`。無權重。
- 72B：hidden 8192、80 層、64 頭、KV 頭 8；30B-A3B：hidden 2048、48 層、128 專家、每 token 8 個。

## 3. 建立的檔案（`exp/M25_topic_survey/`）
- `m25_lib.sh`：主程式庫（執行時才讀入）。B0／B4／B1／B2／B3、全域截止（預設 112 分）與各區塊上限、`blocks.tsv`、GPU 監測（`clocks_event_reasons.active`，與 M20 相同欄名；`clocks_throttle_reasons` 是舊別名）、`gpu_clean`。環境變數選區塊：`M25_BLOCKS`、`M25_MODE=smoke|full`、`M25_NGPU`、`M25_B2_EAGER=1`、`M25_LIM_B*`、`M25_GLOBAL_MIN`。
- `job.sbatch`（全部區塊）、`job_a.sbatch`（B4＋B1＋B3）、`job_b.sbatch`（B2）：`dev`、8 卡、96 CPU、2 小時。`validate.sbatch`：2 卡、24 CPU、40 分、不加 `--exclusive`，之後執行 `analyze.py`。
- `overlap_bench.py`（B1）、`vllm_bench.py`（B2）、`analyze.py`。
- `sbatch --test-only` 對四個腳本都通過語法與資源檢查（未實際送出；`dev` 預估：主工作約 11-25 01:19 可開始、驗證工作約 11-24 23:25）。

## 4. 驗證工作
- **未執行**。`submit.sh --dry-run` 在 `wallet MST114546` 失敗：`/opt/walletcmd/bin/walletcmd.sh.x: ... location has changed!`（系統指令自我檢查失敗，重試仍同；與我的檔案無關）。`submit.sh` 取不到餘額就拒送。需要使用者／Opus 決定：等叢集修好 `wallet`、或明確授權以 `sbatch` 直接送（我不自行繞過）。
- 因此尚無各區塊實測時間。**預先估計（未驗證）**：B0 約 4 分；B4 約 2–4 分；B1 校準＋20 次啟動各 45–60 秒約 17–22 分；B3 約 80 次啟動各 10–15 秒約 15–20 分；B2 四個引擎各啟動 4–7 分加量測約 25–40 分，若加三個 eager 引擎再 +10–15 分。總計約 65–95 分，貼近 105 分界線；建議按規格 4b 預備拆成 `job_a`（B0＋B4＋B1＋B3，約 40–50 分）與 `job_b`（B0＋B2，約 40–55 分）。最終拆不拆以驗證實測決定。

## 5. B4：NCCL 2.29.7 原始碼
- 本地 clone 只有 v2.32.6-1；已 `git fetch --depth 1 origin tag v2.29.7-1`（只加 tag，沒動工作目錄）。
- `src/transport/nvls.cc:257` 為 `CUCHECKGOTO(cuMulticastAddDevice(*mcHandle, comm->cudaDev), ...)`（在 `nvlsAllocateMem` 內；rank 0 先 `ncclNvlsGroupCreate`，其他 rank `ncclNvlsGroupConnect` 匯入 handle 後呼叫它）。M20 的 999 出在 rank 6（匯入端）。
- 多播群組大小：`nvls.cc:333-335`：`buffSize = chunkSize × NCCL_STEPS`、`nvlsPerRankSize = nChannels × 2 × buffSize`、`nvlsTotalSize = perRank × nHeads`（M20 日誌 256 MB，群組建立 512 MB）；`NCCL_NVLS_CHUNKSIZE` 預設 128 KB（`nvls.cc:154`）。
- 工作內容：`b4_a1`、`b4_a2` 完全照 M20（`-b 8 -e 64M -f 4 -g 1`、`mpirun -np 8`、`NCCL_DEBUG=INFO`、`SUBSYS=INIT,ENV,NVLS`）；`b4_e256`（`-e 256M`）；診斷變體：`b4_v1_nvlsch4`（`NCCL_NVLS_NCHANNELS=4`，群組約縮為 1/4）、`b4_v2_chunk16k`（`NCCL_NVLS_CHUNKSIZE=16384`，約縮為 1/8）。理由：檢驗失敗是否與群組大小有關（卡上推測 512 MB）。沒用 `NCCL_LOCAL_REGISTER=0`，因為失敗點在 buffer 配置而非註冊。

## 6. vLLM 0.32.0 原始碼（`site-packages/vllm/`）
- `load_format="dummy"`：`config/load.py:30`（`DummyModelLoader`，`model_executor/model_loader/__init__.py:13,53`）。
- `enable_expert_parallel`：`config/parallel.py:189`。
- Profiler：`LLM(profiler_config={...})`（`entrypoints/llm.py:214`）；欄位 `profiler="torch"`、`torch_profiler_dir`、`torch_profiler_with_stack` 等見 `config/profiler.py:70-100`；`LLM.start_profile(prefix, delay_iterations=, max_iterations=)`／`stop_profile`：`entrypoints/llm.py:801,826`；迭代計數：`profiler/wrapper.py:103-190`；每個 worker 一個 trace，檔名含 `rank0`…。CPU 標註 `execute_context_<n>(tok)_generation_<n>(tok)`：`v1/worker/gpu_worker.py:1279` 附近。
- 加總實作：`distributed/device_communicators/cuda_communicator.py:354` 的 `logger.info_once("Using [...] all-reduce backends (in dispatch order) ...")`，可能值 NCCL_SYMM_MEM／QUICK_REDUCE／CUSTOM／SYMM_MEM／PYNCCL；分派邏輯 `:365-430`。`vllm_bench` 把此行存成每引擎的 `allreduce_backend.txt`。
- 前綴快取預設開（`config/cache.py:141`），所以 `vllm_bench` 設 `enable_prefix_caching=False`。
- **CUDA graph 下 profiler 是否抓得到 kernel：未驗證**（需驗證工作）。`M25_B2_EAGER=1` 會在 B2 追加三個 eager 引擎（72B TP8、30B TP8、30B TP8+EP，只做 profiler）。
- Kernel 分類規則（小寫名稱，先符合先贏）：加總＝`cross_device_reduce, allreduce, all_reduce, oneshot, twoshot, one_shot, two_shot`；其他集體＝`nccl, allgather, all_gather, reducescatter, reduce_scatter, alltoall, all_to_all, dispatch, combine, sendrecv, broadcast`；注意力＝`fmha, flash, attention, attn, paged, mla, fa3, flashinfer, prepare_inputs, softmax`；矩陣乘法＝`gemm, cutlass, cublas, nvjet, xmma, gemv, splitk, fused_moe, grouped, matmul, ampere_, hopper_, sm90_, sm80_`；其餘＝其他。每個 profile 另存前 40 名 kernel 名稱供複核。

## 7. 我自行決定、卡上沒寫清楚的地方（請審）
1. B1「同時」：GEMM 與通訊各一條 stream，都等同一個起點事件（預設 stream 先 `_sleep` 約 1 ms）；時間為起點到各 stream 結束事件；每格開始前丟棄一次同時執行的暖機；全域暖機一次。通訊次數 `n_c` 依 `n_g` 不同而不同（每格 op|size|gemm 各一個），因為矩陣乘法大小不同。小訊息（4 MB）的通訊可能受 CPU 排工作速度限制（單獨與同時都一樣，已在 `calib.json` 的次數中反映）。
2. B1 的 `NVLS` 設定用 `NCCL_ALGO=NVLS`（對三種集體都強制）；若 NVLS 在 2.29.7 初始化即當機（M20），這些啟動會記為失敗，不重試。`NCCL_DEBUG_SUBSYS=INIT,ENV`、`NCCL_DEBUG_FILE` 每行程一檔，每次啟動打包成 `.tar.gz`。
3. B1 煙霧測試用 3 格（all_reduce 4MB×8192、reduce_scatter 32MB×4096、all_gather 256MB×8192）而非 2 格，才能走過三種集體。
4. B2：`enable_prefix_caching=False`；`max_num_seqs=128`；72B 引擎 `max_num_batched_tokens=17408`（16k 預填單步完成）；每字時間 3 次、暖機 1 次（warm-up 與 t128 同批次）；profiler 在解碼批次之後以 `delay_iterations = 預填步數+2`、`max_iterations=6` 只錄解碼步，並以 trace 內 `execute_context_*` 標註的分布驗證（寫在 result.json）。profile 的 trace 一律寫到 `/work/.../traces/<job>/`，≤50 MB 才複製進 `results/`，超過則記 sha256（寫入 result.json，需由整理者追加到 `results/MANIFEST_large_files.txt`；腳本沒有直接改該檔）。
5. B2 預填不另錄 profiler（卡上通訊比例只針對解碼）。72B TP=4 只做解碼 1／32 與預填，沒有 profile 之外的項目被省略。
6. B3：不沿用 `exp/lib/nccl_bench.sh` 的 `run_one`／`add_cfg`（它們固定 5 次、用 `launch.sh`，只轉送固定的環境變數），只沿用 `shuffle_plan`／`wrongcols`；另寫 `b3()`，格式相近。每個設定 2 次獨立啟動（`small`＋`big` 兩段）＋1 次選擇日誌（`INIT,TUNING`；`-b 8 -e 1G -f 2 -n 1 -w 0 -c 0`）。`alltoall` 小段從 512 B 起。
7. 啟動器：所有 nccl-tests 用 `mpirun`（M20：只有它能用），在 `-x` 轉送要設定的環境變數；2／4 卡用 `CUDA_VISIBLE_DEVICES`。
8. 暖機：先校準 `-n`（64 MB all_reduce，8 卡）使 60 秒（煙霧 5 秒）；`mpirun` 直接跑，不經 `rankwrap`。
9. GPU 行程檢查：full 模式發現他人行程就中止；smoke 模式只記錄（2 卡工作可能與他人同節點）。`gpu_clean` 在每次啟動後等 GPU 釋放，必要時 `pkill -u $USER -f 'VLLM::'`（只殺本人帳號下的 vLLM 殘留）。
10. 為了不重複的風險：未改動 `job.sbatch` 以外的已排隊工作（475781、475783、475784）。
