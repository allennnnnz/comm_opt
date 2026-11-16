# S04 候選方向深入調查：既有工作、現成軟體、在 Nano5 的前置條件

- **寫給**：Sonnet 5.5 執行者，三位並行（分工見下表）　**撰寫**：Opus 5.5，2026-11-16（D052）
- **類型**：文獻＋原始碼＋文件調查，不需 GPU、不送 Slurm 工作
- **報告**：`docs/reports/R04_<分工代號>_<slug>.md`

## 背景

S02（文獻地圖）與 S03（主流系統原始碼）已完成，Opus 依此提出五個候選方向（`docs/notes_synthesis_nano5.md` 的「初步候選方向」表），使用者決定全部深入（D052）。**這一輪的目的是回答三個問題，選題判斷仍由 Opus 做，你只提供有出處的事實：**

1. **最接近的既有工作是什麼**：誰已經做了這個方向的哪一部分、沒做哪一部分（以原文為證）。
2. **現成軟體已經做到哪**：NCCL、vLLM、SGLang、TensorRT-LLM、Megatron 是否已經內建、預設是否開啟（附原始碼 `檔案:行號` 或官方文件網址）。
3. **在 Nano5 上做這個方向需要什麼前置條件**：需不需要管理員權限、特定服務、驅動參數、版本；每項怎麼檢查。

**先讀**（避免重複）：`docs/notes_synthesis_nano5.md`；和你方向相關的 `docs/reports/R02_T*.md` 段落與 `docs/reports/R03_system_code_survey.md`。已在 R02 核實過的論文可以直接引用其編號，不必重新核實；你要找的是 **R02 沒涵蓋的** 與 **更細的事實**。

## 我們的環境

- 每台 8× H100 80GB SXM（或 8× H200），NVLink4＋NVSwitch，每台 8× InfiniBand NDR 400 Gb/s；最多 H100 4 台（32 GPU）或 H200 2 台
- 驅動 550（CUDA 12.x）；沒有 root；Slurm
- 已裝：NCCL 2.29.7（venv 內）、vLLM 0.32.0+cu129、torch 2.13.0+cu129、nccl-tests v2.21.1（`/home/allen0117/comm_opt/third_party/nccl-tests`）
- 原始碼（2026-11-16 的 release tag，`--depth 1`）：`/work/allen0117/comm_opt_env/src/{nccl,vllm,sglang,trtllm,flashinfer,megatron,deepep,pytorch}`。注意 NCCL 原始碼是 v2.32.6，我們實際跑的是 2.29.7；引用時標明版本，若可能看 git tag 間差異就寫出來，查不到就寫「未確認 2.29.7 是否相同」
- HPC-X（含 SHARP）：`/work/HPC_software/LMOD/nvidia/packages/hpc_sdk-24.7/Linux_x86_64/24.7/comm_libs/12.5/hpcx/latest/`，其中有 `sharp/`、`nccl_rdma_sharp_plugin/`（Opus 已確認存在）

## 分工

| 代號 | slug | 方向（Opus 的定義） | 一定要回答的事實問題 |
|---|---|---|---|
| A | two_level_switch_reduction | **兩層交換器加總**：機器內用 NVSwitch 的 NVLS（multimem），機器之間用 InfiniBand 交換器的 SHARP，讓小型 H100 叢集（≤4 台）的 LLM 集體通訊更快、佔用更少 GPU 計算資源；同時看頻寬導向（訓練）與延遲導向（推論）| (1) NCCL 有哪些演算法同時用到 NVLS 與網路 SHARP（例如 NVLSTree、CollNet＋NVLS 的組合）？啟用條件、限制、預設開關各是什麼（原始碼 `檔案:行號`）？(2) 2023 年後有沒有論文或廠商資料實測「NVLS＋IB SHARP」或「IB SHARP」在 H100／NDR 上的 LLM 集體通訊？規模、訊息大小、有沒有小訊息延遲、有沒有量 GPU 計算資源（SM）用量？(3) SHARP（Quantum-2 世代）有哪些資源限制：同時能有幾個加總樹、幾個 communicator、streaming aggregation 與 low-latency 模式的差異、訊息大小限制？(4) 使用者端需要什麼才能用：外掛、`sharp_am` 等服務是否由管理員端提供、環境變數；有哪些不需 root 的檢查方法？(5) 主流推論／訓練框架有沒有文件或程式碼提到 SHARP／CollNet？(6) 交換器內加總的學術研究（可程式化交換器、Rina 這類）2023 年後有哪些，和 NVIDIA SHARP 的關係 |
| BE | crossnode_inference | **B：跨機器 TP decode 的小訊息 all-reduce**（模型一台放不下、TP 跨兩台以上時）；**E：prefill／decode 分離時的 KV cache 傳輸**在 H100＋IB 上是不是瓶頸 | B：(1) 實務上有沒有人跨機器做 TP 推論？找框架官方文件或部落格的具體設定（例如某模型用 TP=16 跨 2 台），以及「跨機器建議改用 PP／EP」的官方說法；(2) vLLM／SGLang／TRT-LLM 在 TP 跨機器時，all-reduce 實際走哪條路徑（自訂 all-reduce 是否只限單機、跨機時退回 NCCL）——附原始碼 `檔案:行號`；(3) decode 時 all-reduce 的訊息大小範圍（隱藏維度×batch×位元組），有沒有論文報告實測分佈；(4) 低延遲跨機 all-reduce 的既有工作（NVRAR、Every-µs 的多節點部分、PCCL、其他 R02 沒有的）各自硬體；(5) GPU 直接發起 RDMA（IBGDA）與 GDRCopy 在**沒有 root** 的情況下能不能用：NVSHMEM 官方文件對驅動參數（例如 PeerMappingOverride）與核心模組的要求。E：(6) vLLM 0.32／SGLang 支援哪些 KV 傳輸連接器（NIXL、Mooncake、LMCache 等），需要什麼前置條件（UCX、GPUDirect RDMA、nvidia_peermem 或 dmabuf）；(7) 在 H100＋IB 上實測 KV 傳輸佔比的論文或廠商數據，條件為何；R02_T5 §3-F 的矛盾有沒有新資料 |
| CD | sm_tax_and_selection | **C：通訊的「SM 稅」**——各種集體通訊實作（ring、NVLS／symmetric kernel、copy engine、SHARP）佔用多少 GPU 計算單元、讓同時執行的 GEMM 慢多少，並據此分配；**D：用效能模型挑選集體通訊做法**（演算法、協定、通道數） | C：(1) NCCL 控制佔用計算單元數的參數（例如 `NCCL_MAX_CTAS`、`NCCL_MIN_CTAS`、`NCCL_NCHANNELS*`、CTA policy、symmetric kernel 的 CTA 數）與 copy engine 相關功能，在原始碼中的位置、預設值、從哪一版開始；(2) 2023 年後在 H100 上量「集體通訊佔用計算單元 vs 頻寬 vs GEMM 變慢」的論文（Lee et al.、Zero-SM NCCL、ConCCL、ParallelKittens、其他 R02 沒有的），各自量了什麼、沒量什麼；(3) 「用 DMA／copy engine 做集體通訊」的既有實作與限制（是否支援 reduction、需要什麼權限）。D：(4) NCCL tuner plugin 介面（原始碼位置、能控制什麼、哪一版開始）與既有 tuner 實作（官方範例、雲端廠商、AutoCCL、Lagom 等）；(5) NCCL 內部的效能模型在哪（`檔案:行號`），有沒有論文評估它的準確度；(6) 用模型或學習方法選集體通訊參數的論文（2023 年後），硬體與效益 |

## 步驟

1. 讀上面「先讀」的檔案。
2. **原始碼與文件**：在指定的原始碼目錄內用 `grep -rn`／`sed -n` 讀；只在指定目錄內搜尋。NVIDIA 官方文件（NCCL、NVSHMEM、HPC-X／SHARP 文件、NCCL release notes）用 WebFetch 讀，附網址。
3. **文獻**：
   - 工具：arXiv MCP（`mcp__arxiv__*`）、paper-search MCP（`mcp__paper-search__*`，例如 `search_semantic`、`search_dblp`、`search_google_scholar`，適合找會議論文）、WebSearch、WebFetch。工具是延遲載入的，先用 ToolSearch 載入（例如 `select:mcp__arxiv__search_papers,mcp__arxiv__citation_graph,mcp__arxiv__get_abstract`）。
   - **arXiv MCP 的用法提醒（Opus 實測）**：`search_papers` 要用欄位前綴與詞組，例如 `abs:"in-network aggregation"`、`ti:"SHARP"`，搭配 `sort_by: "relevance"`；不加前綴又按日期排序時，關鍵字幾乎被忽略，只會回傳最新的無關論文。`citation_graph` 可以找「引用某篇關鍵論文的後續論文」，用它從 R02 的關鍵論文往後找。
   - 範圍 2023–2026，重點 2024-01 至今（2026-11-16）。每個方向 **10–25 篇**，寧缺勿濫。
4. **核實**：每篇都要實際打開 arXiv abs 頁、DOI 頁或會議頁確認標題、作者、年份、出處；打不開的不列。廠商部落格或文件標明「非同儕審查」。
5. **前置條件檢查**：只寫「怎麼檢查」與「文件怎麼說」。可以在登入節點跑**幾秒內結束的唯讀指令**（例如 `ls`、`cat` 指定檔案、`module show`、讀 HPC-X 內的文件）；**不要**在登入節點跑任何量測、GPU 程式或長時間指令。需要在計算節點確認的項目，寫成「待 M20／M23 在計算節點確認」並給出指令。

## 產出：`docs/reports/R04_<代號>_<slug>.md`（每個方向一節；BE、CD 各有兩節）

1. **方向定義**：照抄上表 Opus 的定義。
2. **最接近的既有工作**（表格）：編號、標題／作者／年／出處、做了什麼、和這個方向重疊的部分、**原文明確沒做或自述限制的部分**（引文 ≤2 句＋位置）、硬體與規模、程式碼網址、出處網址。R02 已有的論文標「見 R02_T<k> P<n>」並只補新事實。
3. **現成軟體的支援程度**：每個系統一段，附 `檔案:行號`（標版本）或文件網址；寫清楚「有沒有」「預設開不開」「啟用條件」。
4. **在 Nano5 的前置條件**（表格）：條件、文件出處、檢查方法（指令＋在哪裡跑）、目前狀態（已知／未知）、是否需要管理員。
5. **「找不到有人量過」的清單**：只列事實，格式為「在〈硬體／規模〉上量〈什麼〉的論文或資料：未找到（查詢見第 8 節第 n 條）」。找到了就列出來，不要寫成未找到。
6. **互相矛盾或不確定之處**：列出，不裁決。
7. **arXiv ID／DOI 清單**：給 Opus 匯出 BibTeX 用。
8. **搜尋紀錄**：查詢字串（含用的是哪個工具）、大約看了幾筆、排除了哪些與原因。

## 規則

- 只寫事實與原文；**不寫**「這是好題目」「建議做」之類的判斷。
- 每個論點都要有網址、原文位置或 `檔案:行號`。不可憑記憶寫論文、版本號、環境變數名稱或預設值——一律在原始碼或文件中找到才寫。
- **不要對大型共用檔案系統做大範圍搜尋**（例如 `find /`、`find /work`、`find /home`、`locate`）；只在上面列出的目錄內搜尋。
- 下載 PDF 存到 `/home/allen0117/comm_opt/papers/<arXiv編號>.pdf`（不進 git）；下載後確認檔案大小不是 0（之前家目錄滿時留下過 0 位元組的檔案，遇到就重新下載覆蓋）。
- 不改 `lit/refs.bib`；不 commit、不 push；不送任何 GPU 或 Slurm 工作；不讀 `~/TP_opt`、`~/tp_opt`、`~/flux`。
- 用繁體中文撰寫，技術名詞、論文標題、環境變數保留英文。
- 最後回報（給 Opus）：報告路徑、每節的重點各 2–3 句、核實篇數、未能核實的項目。
