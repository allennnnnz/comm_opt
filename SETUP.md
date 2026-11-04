# 交接與環境建置

給接手這個專案的人：照這份文件，可以在另一台 8×A100 機器上重建環境、重跑任何一個實驗，並找到每個數字的來源。

## 1. 這個 repo 是什麼

一篇 GPU 間通訊優化論文的完整研究過程：量測 → 找現象 → 選題 → 原型 → 寫作。

- **目前進度**：[docs/status.md](docs/status.md)
- **各階段結論**：[docs/progress_log.md](docs/progress_log.md)，每個階段在 git 有一個 tag（`p01`–`p18`）
- **研究脈絡總整理**：[docs/findings_overview.md](docs/findings_overview.md)
- **每個決策的理由**：[docs/decisions.md](docs/decisions.md)

主線是 **8×A100 上，生成階段 tensor parallel 的 all-reduce＋residual＋RMSNorm**。原型 kernel 在 [exp/M17_epoch_ab/fused_ar_norm_nf.cu](exp/M17_epoch_ab/fused_ar_norm_nf.cu)。

## 2. 硬體與系統需求

產生 `results/` 的機器（完整快照：[env/system_snapshot.txt](env/system_snapshot.txt)）：

| 項目 | 這台機器 | 需求 |
|---|---|---|
| GPU | 8× A100-SXM4-80GB，NVSwitch（每兩張之間 NV12） | **必要**：8 張 A100 經 NVLink 全互連。M8、M11–M18 的程式寫死 8 張 GPU、`sm_80` |
| Driver | 615.71.10 | 支援 CUDA 13 即可 |
| CUDA toolkit | 13.4，裝在 `/usr/local/cuda`（含 `nvcc`、`nsys`） | **必要**：13.x。原型、探針、FlashInfer JIT 都要用 `nvcc` 現場編譯 |
| NCCL（系統） | 2.32.3（`libnccl2`、`libnccl-dev` 2.32.3-1+cuda13.4） | M0–M7 的 nccl-tests 用它；PyTorch 自帶的是 2.30.7 |
| MPI | Open MPI 4.1.9a1（`/usr/mpi/gcc/openmpi-4.1.9a1`） | nccl-tests 用 MPI 版 |
| 編譯工具 | gcc 13.3、cmake 3.28、git、curl | cmake ≥ 3.25（LMDeploy） |
| Python | 3.12，以 [uv](https://docs.astral.sh/uv/) 管理 | **必要** |
| OS | Ubuntu 24.04.5 | 其他 Linux 應可 |

**時脈注意**：這台機器的 GPU 工作時固定在 1155 MHz（最高 1410），閒置時降到 210 MHz。換機器後，絕對數字會不同；比較一律在同一次 run 內交錯量測（見第 6 節）。

## 3. 建置步驟

```bash
git clone https://github.com/allennnnnz/comm_opt.git && cd comm_opt

# (1) 系統套件（需要 root；版本依第 2 節）
#     CUDA 13.x toolkit → /usr/local/cuda
#     sudo apt install libnccl2=2.32.3-1+cuda13.4 libnccl-dev=2.32.3-1+cuda13.4
#     Open MPI、cmake、gcc、git、curl
#     uv: curl -LsSf https://astral.sh/uv/install.sh | sh

# (2) 三個 Python 環境（版本與原機完全相同）
bash env/setup_python.sh                 # 約 10–20 分鐘；只想確認版本能否安裝：CHECK_ONLY=1 bash env/setup_python.sh

# (3) 外部工具（放在 third_party/，不進 git）
bash env/build_third_party.sh            # 全部；或指定：nccl-tests mscclpp low-latency-nccl nccl-tests-llnccl lmdeploy

# (4) 模型與訓練資料（只有 M5 需要；Qwen2.5-32B 約 65 GB）
bash env/get_models_assets.sh            # 不要 32B 模型：--skip-qwen

# (5) 檢查
python3 env/check_env.py                 # 列出每一項 OK／MISSING 和補救方法
python3 env/check_env.py --gpu           # 另外在 8 張 GPU 上跑原型的正確性測試（約 1 分鐘）
```

| 環境 | 內容 | 用在 |
|---|---|---|
| `.venv` | pandas、matplotlib 等（`requirements-analysis.txt`） | 所有 `analyze.py` |
| `.venv-vllm` | vLLM 0.33.0、torch 2.14.0（CUDA 13.0）、FlashInfer 0.7.2.post1（`requirements-vllm.txt`） | M5 推論、M6、M7、M9–M18 |
| `.venv-titan` | torchtitan 0.4.0、torch 2.15.0（`requirements-titan.txt`） | M5 訓練 |

| 外部工具 | 版本 | 用在 |
|---|---|---|
| nccl-tests | `afd59ab`，MPI 版，`sm_80`，連系統 NCCL | M0–M7 |
| MSCCL++ | `d5ce6c72`，NCCL 相容介面 `libmscclpp_nccl.so`（以 `LD_PRELOAD` 使用） | M6、M7 |
| low-latency-nccl | `ss16118/low-latency-nccl` 分支 `siyshen/low-latency` @ `5357eff` | M7 |
| nccl-tests-llnccl | nccl-tests `afd59ab`，改連 low-latency-nccl 的 headers | M7 |
| LMDeploy | `3228395` ＋ [exp/M9_fused_baselines/lmdeploy_m9.patch](exp/M9_fused_baselines/lmdeploy_m9.patch)，`test_comm` 編兩份（`build/`、`build_force/`） | M9、M10 |

腳本驗證情況（2026-11-04）：三個 Python 環境的全部版本可以解析安裝；nccl-tests 在乾淨目錄從零下載編譯成功；其他外部工具在乾淨目錄下載、切換版本、套用 patch 成功（完整編譯用的是與原機相同的指令，但沒有在乾淨目錄重編一次）。

## 4. 每個實驗需要什麼

| 實驗 | 需要 |
|---|---|
| M0、M2、M4 | nccl-tests、系統 NCCL、MPI、`.venv` |
| M5 | `.venv-vllm`、`.venv-titan`、模型與 assets |
| M6 | nccl-tests、MSCCL++、`.venv-vllm` |
| M7 | nccl-tests、MSCCL++、low-latency-nccl、nccl-tests-llnccl、`.venv-vllm` |
| M8、M11、M12、M13 | 只要 `nvcc`（M11 另需 `nsys`） |
| M9、M10 | `.venv-vllm`（FlashInfer 的 TRT-LLM 合併程式會現場另編 `sm_80`）、LMDeploy（M10 另需 `nsys`） |
| M14–M18 | `.venv-vllm`（原型以 PyTorch extension 現場編譯） |

## 5. 怎麼跑一個實驗

每個實驗一個資料夾 `exp/<ID>_<名稱>/`：`README.md`（實驗卡：問題、預期、判準、結果、結論）、`run.py` 或 `run.sh`（量測）、`analyze.py`（分析）。

```bash
python3 exp/M17_epoch_ab/run.py                 # 量測 → results/M17_epoch_ab/<時間>/
.venv/bin/python exp/M17_epoch_ab/analyze.py    # 分析 → figures/M17_epoch_ab/<時間>/
```

- `run.py` 開始前會檢查機器上有沒有其他 GPU 工作，有就拒絕執行（守則 R13：不打擾別人的工作）
- 每次量測都會在結果資料夾寫 `env.txt`（機器快照）與 `schedule.tsv`（執行順序、亂數種子、每次的時脈與溫度）
- `results/` 是原始數據，不可修改或刪除；大型 profiler 檔（`.nsys-rep`）只放硬碟不進 git，sha256 記在 [results/MANIFEST_large_files.txt](results/MANIFEST_large_files.txt)

## 6. 研究守則重點

完整版與出處：[docs/research_rules.md](docs/research_rules.md)。最常用的幾條：

- **先寫預期與判準再量**（實驗卡），量完不改預期；結果和預期矛盾要照實記錄
- **重複量測**：結論至少 3 次獨立 launch（M18 起 5 次），報中位數與 [最小, 最大]
- **比較放在同一次 run 內交錯量測**：同一份程式在不同 run 之間可以差 10% 以上（M15 vs M16）
- **挑參數要校準與驗證分開**：例如最佳 block 數先在一次 run 挑、再用另一次 run 量效果
- 每個數字都要能追到 `results/` 路徑、腳本與 commit；2026-11-04 改寫過 git 歷史，舊 commit 編號對照見 [docs/git_history_rewrite_20261104.md](docs/git_history_rewrite_20261104.md)

## 7. 已知的坑

| 現象 | 說明 |
|---|---|
| FlashInfer 在 A100 上不提供 TRT-LLM 合併 kernel | M9 起的測試程式用同一批原始碼另編 `sm_80`，第一次約 4 分鐘，之後用快取 |
| TRT-LLM 合併在 hidden ≥ 6400 啟動失敗 | A100 沒有 thread block cluster，一列一個 block 超過暫存器上限（M9、M10） |
| MSCCL++ 的 NCCL 介面讓 nccl-tests 停用 bf16 | 它回報 NCCL 版本 0.9.0；M6 起以 fp16 量測 |
| low-latency-nccl 的新做法在 A100 上算錯 | 新 kernel 只為 `sm_90` 以上編譯，A100 上是空殼（M7、M8） |
| A100 沒有 bf16 的 atomicAdd 指令 | CUDA 以 CAS 迴圈模擬，跨 GPU 會出錯（M11） |
| 原型的 epoch 必須全域一致 | 每個 block 各自計數時，交錯使用不同 block 數會卡住（M16 修正、M17 確認代價） |
| 量測時脈 | GPU 閒置會降頻（210 MHz）。M14–M17 中排在每次 run 最前面量的設定偶爾慢 2–5 µs；M18 步驟 a 正在驗證「先暖機 2 秒＋每個設定 50 次重播」能否消除（結果見 `exp/M18_incremental/README.md`） |
| GitHub 連線 | 這台機器的 port 22 不通，`origin` 走 `ssh://git@ssh.github.com:443/allennnnnz/comm_opt.git` |

## 8. 用 Claude Code 接手（選用）

- [CLAUDE.md](CLAUDE.md)：專案規則、溝通慣例（回覆一律繁體中文、先給白話結論）、git 慣例（commit 後 push、每個研究階段加 tag）
- `.claude/skills/`：研究寫作與繪圖等 skills（來源見 [.claude/SKILL_SOURCES.md](.claude/SKILL_SOURCES.md)）
- `.mcp.json`：論文搜尋 MCP（arXiv、paper-search）；引用只用 arXiv MCP 或 DOI 匯出的 BibTeX
