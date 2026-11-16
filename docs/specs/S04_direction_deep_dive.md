# S04 候選方向深入調查：既有工作、現成軟體、可行性前提

- **寫給**：Sonnet 5.5 執行者（三位並行，各負責一組方向）　**撰寫**：Opus 5.5，2026-11-16（D052）
- **類型**：文獻＋原始碼＋登入節點唯讀檢查；**不需 GPU、不送 Slurm 工作**
- **報告**：`docs/reports/R04_<組>_<slug>.md`（見下表）

## 背景

S02（文獻地圖，`docs/reports/R02_T1`–`T5`）與 S03（主流系統原始碼，`docs/reports/R03_system_code_survey.md`）已完成。Opus 整理出候選方向，使用者決定全部繼續深入（D052）。整理筆記：`docs/notes_synthesis_nano5.md`（**先讀**，尤其「初步候選方向」表）。

你的任務：針對分到的方向，回答三件事——**(1) 最接近的既有工作做到哪裡、沒做什麼；(2) 現成軟體（NCCL、vLLM、SGLang、TensorRT-LLM、Megatron）已經支援到哪裡；(3) 在我們環境要做這個方向，有哪些前提條件、各怎麼檢查。** 判斷「值不值得做」由 Opus 負責，你只提供有出處的事實。

## 我們的環境

- 國網中心 Nano5：每台 8× H100 80GB SXM（或 8× H200），NVLink4＋NVSwitch，每台 8× InfiniBand NDR 400 Gb/s；最多 H100 4 台（32 GPU）、H200 2 台
- 驅動 550（CUDA 12.x）；**沒有 root**；Slurm
- 已安裝：NCCL 2.29.7（在 vLLM venv 內）、vLLM 0.32.0、torch 2.13.0（見 `docs/reports/R01_env_setup.md`）
- 原始碼（S03 下載，release tag，唯讀）：`/work/allen0117/comm_opt_env/src/{nccl,vllm,sglang,trtllm,flashinfer,megatron,deepep,pytorch}`；NCCL 是 v2.32.6-1（**比我們裝的 2.29.7 新**，引用功能時要確認 2.29 是否已有：查 release notes 或 git tag 日期）
- HPC-X（系統模組 `nvhpc-hpcx-cuda12/24.7`）：`/work/HPC_software/LMOD/nvidia/packages/hpc_sdk-24.7/Linux_x86_64/24.7/comm_libs/12.5/hpcx/latest/`，內含 `sharp/`、`nccl_rdma_sharp_plugin/`、`ucc/`、`ucx/`、`README.txt`

## 分組

| 組 | 報告檔名 | 方向 |
|---|---|---|
| A | `R04_A_switch_reduction.md` | **A 兩層交換器加總**：機內 NVLS（NVLink SHARP，multimem）＋機間 InfiniBand SHARP，用於小型 H100 叢集的 LLM 集體通訊（含延遲導向的小訊息） |
| BE | `R04_BE_crossnode_inference.md` | **B 跨機 TP decode 的小訊息 all-reduce**（模型一台放不下時）；**E** prefill／decode 分離時 KV cache 在 H100＋IB 上的傳輸（E 比重較輕，約全篇四分之一） |
| CD | `R04_CD_smtax_model.md` | **C 通訊的「SM 稅」**：各種 collective 實作佔多少 SM、對同時執行的 GEMM 拖慢多少、如何減少；**D** 用效能模型選擇 collective 演算法與參數 |

## 各組必答問題

### A 組
1. **NCCL 怎麼組合 NVLS 與 IB SHARP**：在 NCCL 原始碼找出相關演算法（例如 NVLS、NVLS_TREE、COLLNET_DIRECT、COLLNET_CHAIN）各自在機內、機間用什麼路徑；哪些 collective 支援（all_reduce、all_gather、reduce_scatter）；啟用條件與環境變數；附 `檔案:行號`。也查 release notes：各功能在哪一版加入，2.29.7 是否已有。
2. **外掛相容性（登入節點唯讀檢查）**：HPC-X 內 `nccl_rdma_sharp_plugin/lib/libnccl-net.so` 匯出的外掛介面版本（`nm -D --defined-only <檔案> | grep -i -E "net|collnet"`），與 NCCL 2.29 能載入的版本（NCCL 原始碼 `src/include/plugin/` 等處）是否相容；`sharp/` 的版本（`README.txt` 或檔名）。
3. **SHARP 的硬體與服務前提**：Quantum-2（NDR）SHARP 需要哪些服務（例如 Aggregation Manager、subnet manager 設定）、使用者端環境變數、資源上限（可同時用的 aggregation tree 數、每個 job 限制、streaming aggregation 與低延遲模式的訊息大小範圍）。來源：NVIDIA 官方文件（標版本）、HPC-X 本地文件。哪些需要管理員開、使用者端怎麼確認有沒有開——寫成「給 M20 計算節點的檢查項目」。
4. **既有工作**：
   - 網路內加總（in-network aggregation）用於分散式訓練或推論：可程式交換器路線（SwitchML、ATP、NetReduce、Rina 等）與 SHARP 路線（含 Mellanox／NVIDIA 的 SHARP 論文、SHARP 用於 MPI／深度學習的量測）
   - NVLS／multimem 的量測或使用（R02 已列的 MSCCL++、TokenWeave、Every-µs、SiFAR、ForestColl 等，**只補充與 A 有關的部分，不重抄**）
   - **兩層都用交換器加總**（NVLink SHARP＋IB SHARP，或 NVL72 等級的多層）的任何量測、分析、廠商資料
   - SHARP 用於推論或小訊息延遲的任何資料
5. 每篇既有工作要寫：它量了什麼、在什麼硬體（是否 H100、是否 IB NDR、是否真的開了 SHARP）、與 A 重疊的部分、它沒涵蓋的條件（原文自述優先）。

### BE 組
1. **實務上有沒有人跨機做 TP 推論**（關鍵，影響 B 的動機）：vLLM、SGLang、TensorRT-LLM 的官方文件或部署指南，對「模型一台 8× H100 放不下」（例如 Llama-3.1-405B BF16、DeepSeek-V3／R1）建議用什麼平行方式（跨機 TP、TP＋PP、EP、DP-attention 等）；附網址與原文。也找業界部署報告。
2. **TP 跨機時三個系統實際用什麼 all-reduce**：在原始碼找出 TP 跨節點時的路徑（自製 all-reduce 是否只在單機啟用、跨機是否回到 NCCL、有沒有階層式），附 `檔案:行號`（用 S03 已下載的 release tag；注意我們裝的是 vLLM 0.32.0，若關鍵行為在 0.32.0 與 0.34.0 之間有變，用 `git log`／release notes 標出）。
3. **跨機小訊息 all-reduce 的既有工作**：NVRAR、PCCL（Big Send-off）、Every-µs（GB200 多機 NVLink）、NCCL 的 tree／LL／LL128、MSCCL++ 多機、NVSHMEM 或 GIN 實作、SHARP 低延遲模式。每篇寫硬體、訊息大小、延遲數字（附條件）、與 H100＋IB NDR 的差異。
4. **GPU 直接發 RDMA（IBGDA／GIN GDAKI）在沒有 root 時的可行性**：官方文件列出的前提（驅動參數、kernel module 如 `nvidia_peermem`、`gdrdrv`、dmabuf）；登入節點唯讀檢查（例如 `lsmod`、`cat /proc/driver/nvidia/params`，**標明這是登入節點，計算節點要在 M20 確認**）。
5. **E（較輕）**：vLLM 0.32、SGLang 的 PD 分離連接器（NIXL、Mooncake、LMCache 等）在 H100＋IB 上的支援與前提；有沒有論文在 H100＋IB NDR 實測 KV 傳輸佔端到端的比例；R02_T5 §3-F 的矛盾有沒有後續論文處理。

### CD 組
1. **C：干擾與 SM 用量的既有量測**：在 H100（或 Hopper）上量化「collective 同時執行時 GEMM 變慢多少」「各 collective 實作用幾個 SM／CTA」的論文（R02 已列的 Lee et al.、Comet、ParallelKittens、Zero-SM NCCL，加上 AMD 的 DMA 引擎路線如 ConCCL、FiCCO 等）；每篇寫量測條件與數字。
2. **C：NCCL 減少 SM 用量的機制**：例如 `NCCL_MAX_CTAS`／`NCCL_MIN_CTAS`、communicator 設定中的 CTA 政策、symmetric kernel、NVLS、copy engine 類 collective；在 NCCL v2.32.6 原始碼附 `檔案:行號`，並標出 2.29.7 是否已有。
3. **C：主流系統分給通訊多少 SM**：TransformerEngine／Megatron 的 userbuffers、DeepEP 的 SM 數、vLLM／SGLang 自製 all-reduce 的 block 數，附 `檔案:行號`。
4. **D：效能模型與選擇**：NCCL 內建的演算法／協定選擇（`src/graph/tuning.cc` 等）怎麼算、tuner plugin 介面（`nccl_tuner.h`）能改什麼；既有的選擇或調參研究（AutoCCL、Lagom、PICO、MPI 集體通訊選擇模型、機器學習選擇器等）；作者自述限制。特別找：有沒有考慮 NVLS、SHARP、或「同時有計算」情況的模型。

## 方法

- **先讀** `docs/notes_synthesis_nano5.md` 與你方向相關的 R02、R03 段落；R02 已列的論文用 R02 編號引用，只補新資訊。
- **搜尋工具**（先用 ToolSearch 載入）：
  - arXiv MCP（`mcp__arxiv__search_papers`）：**一定要用欄位前綴**（`abs:"..."`、`ti:"..."`）並用 `sort_by: relevance`；沒有前綴又按日期排序時，關鍵字幾乎被忽略、只回傳最新論文。`mcp__arxiv__citation_graph` 可查某篇論文被誰引用，用來找關鍵論文的後續工作。
  - paper-search MCP（`mcp__paper-search__search_semantic`、`search_dblp`、`search_google_scholar`）：找會議論文與非 arXiv 論文。
  - WebSearch（先 standard）、WebFetch：官方文件、release notes、部落格。
- **核實**：每篇都要實際打開 abs 頁、DOI 頁或會議頁；打不開就不列。需要全文時下載 PDF 到 `papers/`（不進 git，檔名用 arXiv 編號），用 `pdftotext` 讀；**下載後確認檔案大小不是 0**。
- **數量**：每組 10–25 篇關鍵既有工作，重深度不重數量。
- **登入節點**：只做輕量、唯讀、指定路徑的檢查（`ls`、`cat`、`nm -D`、`lsmod`、`grep` 特定檔案）；**不要用 `find /` 或掃描整個共用檔案系統**；不跑任何運算。

## 產出：`docs/reports/R04_<組>_<slug>.md`

1. **重點事實**（8–12 條，每條附出處）
2. **最接近的既有工作表**：工作／出處（venue＋年份＋網址）／做了什麼／硬體與規模／與本方向重疊的部分／沒涵蓋的條件（原文自述附引文與位置；或明顯的條件差異，例如「只在 GB200」「沒開 SHARP」）
3. **現成軟體的支援程度**：每個系統一段，附 `檔案:行號` 與版本；標出我們安裝的版本（NCCL 2.29.7、vLLM 0.32.0）是否已有
4. **在 Nano5 的前提條件**：表格——前提／為什麼需要／怎麼檢查／登入節點檢查結果（附指令與輸出原文）或「需在計算節點檢查（給 M20）」
5. **尚未有人量過或作者自述未解的問題**：每項附出處；只列事實，不評價
6. **矛盾或不確定之處**：列出，不裁決
7. **搜尋紀錄**：查詢字串、工具、約看幾筆、排除的項目與原因
8. **引用清單**：arXiv 編號或 DOI（Opus 之後用 arXiv MCP 或 DOI 匯出 BibTeX）

## 規則

- 只寫事實與原文；不寫「這是好題目」「建議做」之類的判斷。
- 每個論點要有網址、原文位置或 `檔案:行號`。
- 不改 `lit/refs.bib`、`exp/`、`results/`；不 commit、不 push；不送 Slurm 工作、不在登入節點跑運算。
- 不讀 `~/TP_opt`、`~/tp_opt`、`~/flux`。
- 用繁體中文撰寫，技術名詞與論文標題保留英文。
- 交回時用 3–6 點摘要：最重要的發現、無法核實的部分、偏離規格之處。
