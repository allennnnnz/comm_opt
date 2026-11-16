# R01 環境建置（S01）報告 — 部分完成（中途交付）

撰寫：Sonnet 5.5 執行者，2026-11-16。**狀態：工作 475444（C3）與 475459（D4 單台 H100）在 `dev` 排隊中，尚未執行；C3 驗證、D4、D5、E 驗證未完成。** 本報告只記錄已發生的事實。

**重要發現（需使用者處理）：`/home` 有 100 GB 的配額（`df -h .` 於 repo 目錄顯示 `100G 100G 0 100%`，約 22:59 寫入報告時曾出現 `No space left on device`；隨後回到約 393 MB 可用）。`~/.cache` 佔 95 GB（`huggingface` 73 GB、`rattler` 18 GB、`pip` 3.9 GB、`ccache` 539 MB，皆為本次作業前就存在的目錄，我沒有動）。Slurm 的 `.out`／`.err` 與結果都寫在 `/home/allen0117/comm_opt/results/`，家目錄滿了會讓排隊中的工作寫檔失敗。** 這符合規格「配額不夠」的停下回報條件；我沒有刪任何東西。

## 1. 摘要（驗收項目）

| 項目 | 狀態 | 證據 |
|---|---|---|
| `$ENVROOT` 位置與理由；配額與清除政策 | 成功（`/work` 個人配額查不到；`/home` 配額 100 GB 已近滿；清除政策說明書未記載） | 3.A |
| 計算節點能否上網 | 成功：能（pypi、huggingface、github 皆 HTTP/2 200） | 工作 475402，`results/S01_env_setup/20261116_224539_job475402/net.txt` |
| vLLM／torch／CUDA／NCCL 版本；8 卡 torch all_reduce；vLLM TP=8 | 未完成：版本已安裝（見表），8 卡驗證工作 475444 仍排隊 | 3.C |
| nccl-tests tag／commit、編譯參數；NCCL 版本一致 | 部分：編譯成功（工作 475455）；執行時印出的 NCCL 版本尚未驗證 | `results/S01_env_setup/20261116_225716_job475455/` |
| 單台 H100／單台 H200／2 台 H100／2 台 H200 冒煙測試 | 未完成：單台 H100 為 475459 排隊中；其餘三項未送出 | 3.D |
| `exp/lib/nano5_env.sh` 能用 | 未完成：已寫成，僅 `bash -n` 通過，尚無工作驗證 | `exp/lib/nano5_env.sh` |
| 兩個論文搜尋工具能啟動 | 成功（login 節點，`timeout 20`，兩者 exit=0） | `results/S01_env_setup/login_logs/mcp_arxiv.log`、`mcp_paper_search.log` |

## 2. 版本與路徑表

| 元件 | 版本 | 位置 | 安裝方式 | 驗證工作 |
|---|---|---|---|---|
| uv | 0.13.8 | `~/.local/bin/uv`、`uvx` | `python3 -m ensurepip --user` 後 `python3 -m pip install --user uv` | login 節點，無工作 |
| Python（venv） | 3.12.13（系統 `/usr/bin/python3.12`） | `/work/allen0117/comm_opt_env/venvs/vllm` | `uv venv --python 3.12` | — |
| vLLM | 0.32.0+cu129 | 同上 venv | GitHub release wheel `vllm-0.32.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl`，`--extra-index-url https://download.pytorch.org/whl/cu129 --index-strategy unsafe-best-match` | 待 475444 |
| torch | 2.13.0+cu129 | 同上 | 隨上述指令 | 待 475444 |
| NCCL（nvidia-nccl-cu12） | 2.29.7（`nccl.h` 22907） | `venvs/vllm/lib/python3.12/site-packages/nvidia/nccl/{lib,include}` | 隨 torch | 待 475444 |
| flashinfer-python | 0.7.0.post1 | 同 venv | 隨 vLLM | — |
| nvidia-cuda-runtime-cu12 | 12.9.79 | 同 venv | 隨 torch | — |
| nccl-tests | v2.21.1，commit `afd59abab774a6a4b492eb344bd0c845a0b7bb4d` | `/home/allen0117/comm_opt/third_party/nccl-tests/build/` | git clone；編譯工作 475455 | 475455 |
| NCCL_HOME（供編譯） | 2.29.7 | `/work/allen0117/comm_opt_env/nccl_home/{include,lib}`（符號連結到 venv；`lib/libnccl.so -> libnccl.so.2`，因套件只附 `libnccl.so.2`） | 手動建連結 | `ldd.txt` 顯示連到此處 |
| 模型 | Qwen/Qwen3-0.6B，snapshot `c1899de289a04d12100db370d81485cdf75e47ca` | `/work/allen0117/comm_opt_env/hf_cache`（1.5 GB） | `huggingface_hub.snapshot_download`（login 節點） | 待 475444 |
| 模組 | cuda/12.4、openmpi/4.1.6、gcc/11.5.0（編譯用） | — | `module load` | 475455 |

用量（`du -sh`）：venvs 11 GB、hf_cache 1.5 GB、uv_cache 635 MB，皆在 `/work`。

## 3. 各步驟紀錄

### A. 空間與網路
- `df -T`：`/home`、`/work` 皆 `wekafs`。`quota -s`、`weka` 指令回 `error: Authentication Failed: Please run this command to login to your cluster`；`lfs`、`mmlsquota` 不存在。`/work` 個人配額**查不到**。`/home` 在 repo 目錄的 `df -h .` 顯示 100G 配額（見上方重要發現）。`/work/allen0117` 內已有 123 GB 既有內容（未動）。
- 清除政策：抓取並 grep 說明書 `dUGG_heYSv2CSLig9yFpVQ`、`CxQR2qnESZ-n_7vnCklX1g` 及目錄頁連到的其他頁（登入節點、資料傳輸、Lmod、Slurm、Queue、錢包），**沒有關於 /work 或 /home 自動清除或配額的描述**。
- 決定 `$ENVROOT=/work/allen0117/comm_opt_env`：理由是說明書未記載自動清除（「沒記載」不等於「確定沒有」），且 `/home` 配額只有 100 GB 且近滿。**需使用者確認。**
- 網路：工作 475402（`dev`、1 GPU、hgpn01，COMPLETED 0:0）。`net.txt`：三個網址皆 `HTTP/2 200`。`job_env.txt` 的 nvidia-smi：Driver 550.127.08，CUDA Version 12.4。

### B. uv
- `python3 -m pip install --user uv` 第一次回 `No module named pip`；改先 `python3 -m ensurepip --user` 再安裝，得 uv 0.13.8。

### C. PyTorch＋vLLM
- PyPI（https://pypi.org/pypi/vllm/json）：最新穩定版 0.33.0（2026-11-02），`requires_python >=3.11,<3.15`，要求 `torch==2.14.0`、`flashinfer-python==0.7.2.post1`、`nvidia-cutlass-dsl[cu13]`、`humming-kernels[cu13]`。0.32.0、0.31.0 要求 `torch==2.13.0`。
- 官方文件 https://docs.vllm.ai/en/latest/getting_started/installation/gpu/ ：預設 wheel 為 CUDA 12.9，另有 12.8、13.2 變體，以 GitHub release 的 `vllm-<ver>+cu<ver>-...whl` 搭配 `--extra-index-url https://download.pytorch.org/whl/cu<ver>` 安裝。
- 實查 0.33.0：
  - `uv pip install "vllm==0.33.0" --torch-backend=cu129` 失敗，原文：`Because there is no version of torch==2.14.0 and vllm>=0.33.0 depends on torch==2.14.0, we can conclude that vllm>=0.33.0 cannot be used.`（`login_logs/uv_install_vllm_0.33.0.log`）。pytorch 的 cu129 index 只到 torch 2.13.0；cu126／cu130／cu132 才有 2.14.0（`curl https://download.pytorch.org/whl/<cu>/torch/` 結果）。
  - 下載 PyPI 的 0.33.0 wheel，`readelf -d vllm/_C_stable_libtorch.abi3.so` 顯示 `NEEDED libcudart.so.13`，即 PyPI 0.33.0 需要 CUDA 13 runtime（此輸出只在對話中觀察，未存檔；wheel 已刪除）。
  - GitHub releases（https://api.github.com/repos/vllm-project/vllm/releases）：latest 為 `v0.33.0rc3`，assets 沒有 `+cu129`；v0.32.0、v0.31.0 有 `+cu129`。
  - 沒有在驅動 550 上實測 0.33.0。規格推測 CUDA 13 不行，**此推測仍未實測**。
- 因此改裝次新穩定版 **0.32.0+cu129**。安裝 log `login_logs/uv_install_vllm_0.32.0_cu129.log`，196 個套件，5m46s，成功。
- 下載 Qwen3-0.6B 前讀 `config.json`（https://huggingface.co/Qwen/Qwen3-0.6B/raw/main/config.json）：`num_attention_heads=16`、`num_key_value_heads=8`，皆可被 8 整除。
- 測試腳本：`exp/S01_env_setup/c3_check_torch.py`、`c3_allreduce.py`、`c3_vllm_tp8.py`、`c3_vllm_test.sbatch`。工作 **475444**（`dev`、8 GPU）已送出，**撰寫時仍 PENDING (Priority)**，故 torch／all_reduce／vLLM TP=8 全部未驗證。
- 送 `dev` 時 sbatch 印出 `sbatch: WARNING: Exclusive mode is not allowed. --exclusive is removed.`（`dev` 不允許獨佔；申請 8 GPU 即佔整台 GPU）。

### D. nccl-tests
- `git clone https://github.com/NVIDIA/nccl-tests third_party/nccl-tests`；最新 tag `v2.21.1`，commit `afd59abab774a6a4b492eb344bd0c845a0b7bb4d`。`third_party/` 在 `.gitignore`。
- NCCL 來源：venv 的 `nvidia/nccl/include/nccl.h`（MAJOR 2 / MINOR 29 / PATCH 7）與 `lib/libnccl.so.2`；標頭齊全，不需另取。
- 編譯工作 475454：失敗，`make[2]: g++: Command not found`（`results/S01_env_setup/20261116_225646_job475454/make.log`）。原因：沒載入 gcc。
- 編譯工作 475455（hgpn01，COMPLETED，`make rc=0`；加載 `gcc/11.5.0`）。指令（`make_cmd.txt`）：
  `make -j12 MPI=1 MPI_HOME=/work/HPC_software/LMOD/nvidia/packages/cuda-12.2/openmpi-4.1.6 CUDA_HOME=/work/HPC_software/LMOD/nvidia/packages/cuda-12.4 NCCL_HOME=/work/allen0117/comm_opt_env/nccl_home NVCC_GENCODE=-gencode=arch=compute_90,code=sm_90`。`ldd.txt`：`libnccl.so.2 => /work/allen0117/comm_opt_env/nccl_home/lib/libnccl.so.2`。產生 12 個 `*_perf`。
  - 第一次 sbatch 被拒：`CPU per node specification of 16 exceeds the limit (1 * 12)`（1 GPU 最多 12 CPU），改 12。
  - `module load openmpi/4.1.6` 會設 `SLURM_MPI_TYPE=pmi2`、`UCX_TLS=all`、`UCX_MEMTYPE_CACHE=n`（`slurm/S01d3-475455.out`），所以 srun 一律明確指定 `--mpi=`。
- D4 腳本 `exp/S01_env_setup/d4_allreduce_smoke.sbatch`（依序 pmix_v4 → pmi2 → mpirun，可用命令列覆寫 `-p -N -J`）；單台 H100 已送出為 **475459**，仍 PENDING。
- 我試圖一次送出 D5 三個工作（2 台 H100、單台 H200、2 台 H200），被 Claude Code 權限系統拒絕，**沒有繞過**，未送出。
- 排隊：475444 於 22:56、475459 於 22:57 送出，撰寫時仍在排隊；此沙箱時鐘走得很慢，無法證明已超過 30 分鐘。

### E. 環境腳本
- `exp/lib/nano5_env.sh`：`module load cuda/12.4 openmpi/4.1.6`、啟用 venv、`HF_HOME`、`LD_LIBRARY_PATH`（venv NCCL 最前）、`NCCL_TESTS`。無信箱或密碼。尚未有工作驗證；沒載入 gcc（編譯時另載）。

### F. 論文搜尋工具
- `.mcp.json`：兩處 command 改為 `/home/allen0117/.local/bin/uvx`，arxiv `--storage-path` 改為 `/home/allen0117/comm_opt/papers/arxiv`，版本與 `--with "mcp<2"` 未動（`git diff .mcp.json`）。
- `timeout 20 uvx arxiv-mcp-server@0.8.2 ...`：安裝 45 個套件後 `exit=0`；`timeout 20 uvx --with "mcp<2" paper-search-mcp@0.1.4`：`exit=0`，只有 DOAJ／Unpaywall 未設 API key 的 WARNING。stdin 為 /dev/null，EOF 即正常結束；log 內 grep `error|traceback|modulenotfound` 無結果。
- **Claude Code 要重新啟動，新的 MCP 設定才會生效。**

## 4. 證據清單
- 網路：`results/S01_env_setup/20261116_224539_job475402/{net.txt,job_env.txt}`
- 編譯：`20261116_225646_job475454/make.log`（失敗）、`20261116_225716_job475455/{make.log,make_cmd.txt,ldd.txt,job_env.txt}`
- 安裝 log、MCP log：`results/S01_env_setup/login_logs/`
- vLLM／PyPI／GitHub：上述網址
- 腳本：`exp/S01_env_setup/*`、`exp/lib/nano5_env.sh`

## 5. 和規格不同的地方
1. `python3 -m pip` 無 pip，先 `ensurepip --user`。
2. vLLM 裝 0.32.0+cu129 而非最新 0.33.0（0.33.0 無 CUDA 12 變體）。
3. 編譯加載 `gcc/11.5.0`。
4. 為連結 `-lnccl` 建立 `nccl_home` 符號連結目錄。
5. `dev` 不允許 `--exclusive`；uv cache 與 Python 安裝目錄設在 `$ENVROOT`（`UV_CACHE_DIR`）。
6. vLLM 測試用 `max_model_len=2048, gpu_memory_utilization=0.5`（規格未指定）。

## 6. 沒解決的問題
- `/home` 100 GB 配額近滿（`~/.cache` 95 GB 是既有內容），可能讓工作寫 `.out`／`.err` 失敗；需使用者決定是否清理或改輸出位置。
- 工作 475444、475459 排隊中；完成後需讀 `results/S01_env_setup/*_job475444`（`c3_*`）與 `*_job475459`（`attempts.txt`、`allreduce_*.log`），並確認 torch 的 NCCL 版本等於 nccl-tests 印出的版本。
- D5（2 台 H100、單台／2 台 H200）與 E 的驗證工作尚未送出。
- `/work` 個人配額查不到、有無自動清除未確認。
- vLLM 0.33.0（CUDA 13）在驅動 550 上是否不能跑，未實測。
- Claude Code 啟動 MCP 時用預設 `~/.cache/uv`（`/home`，配額近滿）重新下載套件，可能受 `/home` 空間影響。
