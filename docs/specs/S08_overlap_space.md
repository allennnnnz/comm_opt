# S08 計算與通訊重疊的研究空間：論文細節與我們環境的可行性

- **寫給**：Sonnet 5.5 執行者，兩位並行（分工 A、B）　**撰寫**：Opus 5.5，2026-11-25（D059）
- **類型**：文獻＋原始碼＋文件調查，不需 GPU、不送 Slurm 工作
- **報告**：`docs/reports/R08_A_papers.md`、`docs/reports/R08_B_code_env.md`

## 背景（為什麼做）

M25 在 8×H200（hgpn46）與 8×H100（hgpn18）上量到（`exp/M25_topic_survey/README.md`）：

- vLLM 0.32.0 推論 72B（Qwen2.5-72B 結構、dummy 權重）TP=8：
  - **預填步**（每步 8k–17k token）all-reduce 佔 GPU kernel 時間 30–35%，與計算依序執行、沒有重疊。
  - **解碼步**佔 10–17%。
- 30B MoE（Qwen3-30B-A3B）預填 25–35%。
- 讓 GEMM 與 NCCL collective 同時跑時：通訊變慢 15–280%、GEMM 變慢 0–38%，程度隨訊息大小與 GEMM 大小改變。
- 換 NCCL 設定（ring／NVLS／CTA 數）對總完成時間大多 <5%。

Opus 據此提出研究方向：**8 卡單機（H100／H200，NVSwitch＋NVLS）上，TP 推論（以預填／大批次為主）的計算與通訊重疊，以及重疊時兩者互相拖慢的問題**。使用者要知道這個領域還剩多少研究空間、我們可以做哪些區塊。**選題判斷由 Opus 做，你只提供有出處的事實。**

**先讀（不要重做）**：
- `docs/reports/R02_T2_overlap_fusion.md`：25 篇重疊／融合論文，已核實過基本資料。
- `docs/reports/R04_CD_smtax_model.md`、`docs/reports/R04_CD_sm_tax_and_selection.md`：SM 佔用與干擾相關論文、NCCL／vLLM 旋鈕。

已核實的論文直接引用其編號（例如「R02_T2 #23」）；你要補的是**它們沒寫到的細節**，以及**沒收的論文**。

## 我們的環境

- 每台 8× H100 80GB SXM 或 8× H200，NVLink4＋NVSwitch，NVLS 可用（M25 實測）。驅動 550.127.08（CUDA 12.4 driver API）。沒有 root。
- 計算節點最高 nvcc 12.6。已知問題：FlashInfer JIT 需要 nvcc ≥ 12.8，所以 M25 把 vLLM 的 FlashInfer 全部關掉，也關了 `fuse_allreduce_rms`（見 `exp/M25_topic_survey/README.md`「設定」與 status 的 2026-11-25 00:30 段）。
- vLLM venv：`/work/allen0117/comm_opt_env/venvs/vllm/`，內含：
  - vllm 0.32.0+cu129
  - torch 2.13.0+cu129
  - nvidia-nccl 2.29.7
  - nvidia-nvshmem-cu12 3.4.5
  - flashinfer-python 0.7.0.post1
- 原始碼（release tag，`--depth 1`）：`/work/allen0117/comm_opt_env/src/{nccl,vllm,sglang,trtllm,flashinfer,megatron,deepep,pytorch}`。
  - 注意：`src/vllm` 是 v0.34.0，我們裝的是 0.32.0。0.32.0 的 Python 原始碼就在 venv 的 `site-packages/vllm/` 內，**以 venv 內的為準**。

## 分工

### A：論文細節（`R08_A_papers.md`）

**論文清單**：
- R02_T2 已收：#23 FLUX、#21 Ladder-residual、#20 Comet、#15 TokenWeave、#7 Syncopate、#10 ParallelKittens、#19 TileLink／#17 Triton-distributed、#18 FlashOverlap、#22 Domino、#8 MPK、#13 Lee et al.
- R04_CD 已收：Resource-aware Overlap 2606.09200、Weave 2609.21483、Lagom（R02_T2 #6）
- **R02 沒收、要新增核實**：
  - NanoFlow（arXiv 2408.12757）
  - ISO（arXiv 2409.11155）
  - PyTorch Async-TP（找官方論文或文件）
  - 用 `citation_graph` 從 FLUX、NanoFlow、TokenWeave、Lee et al. 往後找到的，2024 年至今、**推論 TP 重疊或重疊干擾**相關的論文（最多再加 10 篇）
  - 2026-11-16 之後的新論文

**每篇都要回答四件事，數字一律附原文位置（節、表、圖號）**：
1. **在 H100／H200（或 H800）上的實際數字**：分預填與解碼；記下 token 數或批次、TP 度、模型、加速倍數。沒有 Hopper 數據就寫「無」，並記下它用什麼硬體。
2. **比較對象的細節**：
   - 對 vLLM 比的，記下 vLLM 版本，以及是否開 custom all-reduce、CUDA graph、NVLS。
   - 對 NCCL 比的，記下版本，以及是否依序執行。
   - 有沒有報告「沒有通訊的理想上限」，以及基準中通訊佔比是多少。
3. **怎麼處理重疊時的互相拖慢**：
   - 有沒有量通訊被拖慢多少、GEMM 被拖慢多少。
   - 用什麼方法緩解：SM 切分、stream 優先權、tile 大小、wave 對齊的切分、copy engine。
   - 是固定設定、離線調參，還是執行時自適應。
   - 是否考慮拖慢程度隨訊息大小、GEMM 大小、批次而變。
4. **作者自述的限制與未來工作**（引文 ≤2 句＋位置），特別是和「預填」「H100」「干擾」「自動調整」有關的。

**最後一節「空白格」**：
- 做一張表：列 = 情境（預填 TP／解碼 TP／混合批次／MoE），欄 = 「Hopper 8 卡上有實測」「基準含 vLLM custom all-reduce＋CUDA graph」「量了雙向拖慢」「拖慢隨大小自適應」。每格填論文編號，沒有就寫「未找到」。
- 找不到不代表不存在，要附第幾條查詢。

### B：程式碼與我們環境的可行性（`R08_B_code_env.md`）

1. **vLLM 0.32.0 已內建的重疊／融合功能**（以 venv 內原始碼為準，附 `檔案:行號`）：
   - 涵蓋：sequence parallelism、async TP（collective fusion pass）、AllReduce＋RMSNorm 融合、dual batch overlap／microbatching、其他任何讓 TP 通訊和計算重疊的功能。
   - 每個功能寫：怎麼開（參數名稱）；預設開不開；支援預填還是解碼；依賴什麼（FlashInfer？symmetric memory？torch.compile？nvcc 版本？）。
   - 依我們環境的限制（nvcc 12.6、FlashInfer JIT 不能用、驅動 550），判斷**哪些看起來能開、哪些不能、哪些不確定**，並寫出判斷依據。
2. **SGLang、TensorRT-LLM 的同類功能**：只要清單和 `檔案:行號`（用 `src/` 內的原始碼）。
3. **論文程式碼可行性**：
   - 對象：FLUX、TokenWeave、NanoFlow、Syncopate、ParallelKittens／ThunderKittens、Triton-distributed／TileLink、FlashOverlap、Ladder-residual、ISO（有的話）、MPK。
   - 方法：`git clone --depth 1` 到 `/work/allen0117/comm_opt_env/src/overlap/<名稱>/`，只讀不編譯。
   - 每個要記：
     - 最後 commit 日期，是否仍在維護
     - 需要的 CUDA／nvcc、torch、NVSHMEM、CUTLASS 版本、GPU 架構（sm_90）
     - 是否需要 root 或特定驅動功能
     - 有沒有對 vLLM 的整合，以及綁哪一版
     - 有沒有預編譯 wheel，支援哪些 CUDA
   - 對照我們的環境，判斷「看起來可行／需要改版本／不可行」，並寫出依據。
   - **不要編譯、不要 pip install 到現有 venv**。

## 步驟與工具

1. 讀「先讀」的檔案。
2. 文獻：
   - 工具：arXiv MCP（`mcp__arxiv__search_papers`、`get_abstract`、`citation_graph`、`download_paper`、`read_paper`）、paper-search MCP、WebSearch、WebFetch。工具是延遲載入的，先用 ToolSearch 載入。
   - `search_papers` 要用欄位前綴與詞組（例如 `abs:"tensor parallel" AND abs:overlap`），排序用 relevance。
   - PDF 存 `/home/allen0117/comm_opt/papers/<arXiv編號>.pdf`，下載後確認大小不是 0。
3. 核實：
   - 每篇新論文要打開 arXiv abs 或會議頁，確認標題、作者、日期。
   - 數字要從全文來，不能只看摘要；只看過摘要的，標「僅摘要」。
4. 登入節點只跑幾秒內結束的唯讀指令（`git clone --depth 1`、`grep`、`sed -n`、`ls`）。**不要**跑量測、GPU 程式、編譯或安裝。

## 規則

- 只寫事實與原文；**不寫**「這是好題目」「建議做」之類的判斷。
- 每個論點都要有網址、原文位置或 `檔案:行號`。不可憑記憶寫論文、數字、版本號或參數名稱。
- 不對共用檔案系統做大範圍搜尋（不用 `find /`、`find /work`、`locate`）。
- 不改 `lit/refs.bib`；不 commit、不 push；不送任何 Slurm 工作；不讀 `~/TP_opt`、`~/tp_opt`、`~/flux`（`~/flux` 是本專案以前的研究資料夾，和 FLUX 論文無關；FLUX 一律從 arXiv 和 GitHub 讀）。
- 用繁體中文撰寫，技術名詞、論文標題、參數名稱保留英文。
- 最後回報（給 Opus）：
  - 報告路徑
  - 最重要的 5 個事實（附出處）
  - 「空白格」表（A）或可行性總表（B）
  - 未能核實的項目
