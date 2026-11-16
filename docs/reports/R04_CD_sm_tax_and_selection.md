# R04 / CD：方向 C（通訊的「SM 稅」）與方向 D（用效能模型挑選集體通訊做法）深入調查

- **規格**：`docs/specs/S04_direction_deep_dive.md`（分工代號 CD）　**撰寫**：Sonnet 5.5 執行者，2026-11-16
- **性質**：只列有出處的事實，不做選題判斷。未做 GPU／Slurm 工作；只在登入節點跑了幾秒內結束的唯讀指令（`grep`、`strings`、`pdftotext`、`ls`）。
- **版本標示**：「v2.32.6」＝`/work/allen0117/comm_opt_env/src/nccl`（2026-11-16 的 release tag，`git log` 顯示 `Merge v2.32.6-1 release into master`）；「2.29.7」＝我們實際安裝的 `libnccl.so.2`（`/work/allen0117/comm_opt_env/venvs/vllm/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2`）。凡只對 v2.32.6 原始碼讀過、未對 2.29.7 確認的，一律標「未確認 2.29.7 是否相同」。對 2.29.7 二進位只做了 `strings` 字串比對（只能證明字串存在，不能證明行為）。
- **工具狀態**：arXiv MCP 的 `search_papers`、`get_abstract`、`citation_graph` 在本 session 大部分時間被 HTTP 429／Semantic Scholar quota 擋住（`citation_graph` 三次全失敗），paper-search `search_semantic` 回傳空。因此文獻搜尋多半靠 WebSearch，核實一律改用 WebFetch 開 arXiv abs 頁並下載 PDF 以 `pdftotext` 讀全文（PDF 存 `/home/allen0117/comm_opt/papers/`，已確認大小非 0）。**`citation_graph` 往後找這個步驟沒有做成**。

---

# 第一節　方向 C：通訊的「SM 稅」

## C.1 方向定義（照抄 Opus）

**C：通訊的「SM 稅」**——各種集體通訊實作（ring、NVLS／symmetric kernel、copy engine、SHARP）佔用多少 GPU 計算單元、讓同時執行的 GEMM 慢多少，並據此分配。一定要回答：(1) NCCL 控制佔用計算單元數的參數（例如 `NCCL_MAX_CTAS`、`NCCL_MIN_CTAS`、`NCCL_NCHANNELS*`、CTA policy、symmetric kernel 的 CTA 數）與 copy engine 相關功能，在原始碼中的位置、預設值、從哪一版開始；(2) 2023 年後在 H100 上量「集體通訊佔用計算單元 vs 頻寬 vs GEMM 變慢」的論文（Lee et al.、Zero-SM NCCL、ConCCL、ParallelKittens、其他 R02 沒有的），各自量了什麼、沒量什麼；(3)「用 DMA／copy engine 做集體通訊」的既有實作與限制（是否支援 reduction、需要什麼權限）。

## C.2 最接近的既有工作

> 欄位精簡為：編號／出處、做了什麼、與 C 重疊的部分、原文沒做或自述限制（引文＋位置）、硬體與規模、程式碼、網址。R02 已核實者標「見 R02_T<k> P<n>」只補新事實。

| 編號 | 標題／作者／年／出處 | 與 C 重疊的部分 | 沒做／自述限制 | 硬體規模 | 程式碼 | 網址 |
|---|---|---|---|---|---|---|
| C1 | **Zero-SM NCCL**（見 R02_T1 P1）He…Hoefler，2026-11-13，arXiv:2611.18075 | 圖 1 列「NCCL 飽和 NVLink 所需 SM 數」隨世代（Pascal 8、Volta 12、…、Blackwell 32 SM）；單一 B200 節點量「144 MiB AllGather 讓重疊 GEMM 變慢多少 vs. AllGather 用的 CTA 數」：legacy ring 預設 32 CTA 中位數慢 32.9%，symmetric kernel 預設 4 CTA 慢 6.2%、強制 32 CTA 慢 24.0%，copy engine（CE）慢 3.9%（§IV-C，全文 `pdftotext` 讀到）；「SM kernel 讓 GEMM 變慢的程度和它佔的 SM 成正比」 | 不支援 reduction（§III-A）；單 domain 小訊息延遲高於 SM kernel（§V；1 KiB 單機 AllGather：CE 25–29 μs、SM kernel 12–18 μs）；**沒有 H100 的 GEMM 慢多少數據**（全文 `grep -i "H100\|H200"` 僅在參考文獻出現「Hopper」一詞）；Hopper 在圖 1 的 SM 數我從文字抽取無法確定（需看圖） | GB200 NVL72、B200 NVL8；NCCL 2.32.3（含未釋出功能） | https://github.com/spcl/zero-sm-artifact | https://arxiv.org/abs/2611.18075 |
| C2 | **Characterizing Compute-Communication Overlap…**（見 R02_T2 #13）Lee et al.，2025-07-03，arXiv:2507.03114 | DGX H100 上量重疊使計算平均變慢 18.9%、最高 40.0% | 我在 R02 保存的全文文字用 `grep -i "channel\|NCCL_\|copy engine\|NVLS"` 找，**未見其調整通道數／CTA 數、也未區分通訊實作（ring／NVLS／CE）**；結論為整體「重疊 vs 理想」，未分解各通訊實作的 SM 佔用（此為我的 grep 結果，非原文自述） | DGX H100（8×H100）、A100×4、MI250、MI210 | 未說明 | https://arxiv.org/abs/2507.03114 |
| C3 | **Optimizing ML Concurrent Computation and Communication with GPU DMA Engines（ConCCL）** Agrawal, Aga, Pati, Islam（AMD），2024-12-18（v2），arXiv:2412.14335 | 量 C3（concurrent computation & communication）：平均只達理想加速的 21%；以優先權與 CU 切分提高到 42%；用 DMA engine 實作 all-gather、all-to-all 的概念驗證（ConCCL）達 72%、最高 1.67×。CU 預設數：all-gather 64、all-to-all 56（PDF 第 312 行附近） | 腳註：「GPU DMA engines do not support arithmetic operations, we do not consider offloading all-reduce kernels to DMAs」；**只在 AMD MI300X（8 卡全連接）**，**非 NVIDIA、非 H100** | 8×MI300X（14 個 SDMA 引擎／GPU） | 未說明 | https://arxiv.org/abs/2412.14335 |
| C4 | **The Endgame of FSDP Communication（TIDE）** Wang…Song（14 人），2026-10-28，arXiv:2610.38228（18 頁，無 venue） | **直接量 NCCL CTA 數 vs. GEMM 慢多少（含 SM 波次量化的機制）**：32 MiB reduce-scatter 於 8 節點用 16 CTA，使單波 GEMM 慢 69%；NCCL 2.31 預設以 4-CTA cluster 啟動（§2.2）；把 NCCL 限制為 8 CTA 仍慢 67.8%；cluster size 設 1 則 6.3% 但 reduce-scatter 變慢（13.0 ms，約 TIDE 的 2 倍）；TIDE reduce-scatter 的 SM 時間降 12.5–14.5×、all-gather 完全不用 SM | 方法綁定 NCCL 2.31+ 公開 API；GPU 型號在我讀到的文字中**沒有明寫**（§2.2 例子算式 33 個 4-CTA cluster＝132 SM，與 H100／H200 的 132 SM 一致，**此為我的推算，非原文**） | 最多 8 節點×8 GPU＝64 GPU，每 NVSwitch domain 8 GPU，每 GPU 一條 RDMA rail；訓練：Llama-3.1 70B、Qwen3-30B-A3B、Wan2.2 | 論文稱約 1,800 行測試＋實作，網址我未取得 | https://arxiv.org/abs/2610.38228 |
| C5 | **VCCL: An Efficient, Reliable and Observable CCL in Large-scale GPU Training Clusters** Zhang…Hu（34 人），2025-10-01（v2 2026-06-01），arXiv:2510.00991 | 稱 NCCL 的 SendRecv／AlltoAll 雖無 reduction 仍耗用「non-negligible」SM（附錄 A 用 Nsight／DCGM 量 SM 利用率，2 台各 8 Hopper＋8 CX7）；以 CPU 執行緒＋copy engine 取代 SM 型 P2P kernel；訓練吞吐最多 +5.28%（對 NCCL） | 只做 P2P（SendRecv／AlltoAll）去 SM，**沒有 AllReduce／reduce 類的 SM-free**；GPU 型號寫 "Hopper"，未見具體型號；是生產系統報告，非控制實驗 | 2 台×8 Hopper（附錄 A）；生產叢集 | 論文稱開源（網址我未取得） | https://arxiv.org/abs/2510.00991 |
| C6 | **Resource-aware Computation-Communication Overlap for multi-GPU ML Workloads** Cui & Pericas，2026-06-08，arXiv:2606.09200（ISC 2026 AI on HPC Workshop，arXiv 註記 "To appear"） | 以「每 block 配置 shared memory」限制 GEMM 的常駐 block 數、並提高通訊 stream 優先權，不改 NCCL／kernel；A40／A100／H100／MI250X 總執行時間最多 −25.5% | 每台 **僅 4 卡 NVIDIA（#SMs：H100 132）**，單機（Table 1）；摘要頁不報告 SM／occupancy 實測數字；**未見跨機**；workshop 論文 | A40、A100、H100 各 4 卡；MI250X 8 卡 | 未說明 | https://arxiv.org/abs/2606.09200 |
| C7 | **Weave: Fine-Grained Dynamic SM Scheduling in an MoE Megakernel**（Huang…Leng，2026-09-25 v3，arXiv:2609.21483） | 在 4×H100 SXM 依路由結果動態切 SM（通訊 worker vs. 計算 worker）；2.89× MoE layer、1.33× 端到端（幾何平均） | 原文 Limitations：多節點與更大配置留待未來；偏 MoE 推論，不是通用 collective | 4×H100 SXM | 未說明 | https://arxiv.org/abs/2609.21483 |
| C8 | **ParallelKittens**（見 R02_T1 P15／R02_T2 #10）Sul…Ré | 量 H100 傳輸機制：copy engine 368.82 GB/s（82%）、TMA 350.01（78%）、暫存器指令 342.68（76%）（1 GB，全部 SM）；暫存器指令需 3.2–5.1× SM 才能飽和 NVLink（§3.1.2）；有 SM 分配 auto-tune（圖 5 依搜尋摘要，未逐頁核實） | 單機（§5）；GEMM 變慢未獨立量 | 8×H100 | https://github.com/HazyResearch/ThunderKittens | https://arxiv.org/abs/2511.13940 |
| C9 | **Comet / mKernel / Syncopate / Cui-Wingkvist 2607.19539 / FiCCO / Lagom** | 見 R02_T2 #20／#2／#7／#3／#9／#6：皆涉及 SM 分配或 DMA 卸載；FiCCO（2512.10236）用 DMA engine，只在 MI300X；mKernel 自述 "we are only able to test adaptive SM partitioning on a single node"（§5） | 見 R02 | 見 R02 | 見 R02 | 見 R02 |
| C10 | **Collective Communication for 100k+ GPUs（NCCLX）**（見 R02_T1 P16） | NVL CopyEngine 做無 SM 傳輸；FTAR AllReduce 用一半 thread block 達 NCCL 同等（§5.5） | 見 R02 | H100 | https://github.com/meta-pytorch/torchcomms/tree/main/comms/ncclx | https://arxiv.org/abs/2510.20171 |

**非同儕審查、僅供參考**：
- Chalmers 碩士論文（GCOverlap：以 CUDA Green Contexts 切 SM，WebSearch 摘要稱含 GEMM+AllReduce／ReduceScatter 算子層級結果）：**我沒有打開全文，只看過搜尋摘要，未核實**；網址見搜尋紀錄 S5。
- NVIDIA 技術部落格「Fusing Communication and Compute with New Device API and Copy Engine Collectives in NVIDIA NCCL 2.28」（2025-11-10，廠商，非同儕審查）：說 CE 適用於 "collectives that require only data movement, such as AlltoAll and AllGather"；AllGather 圖為 1 節點 8 Blackwell GPU，CE Multicast 約 780 GB/s vs. SM NonSymmetric 約 620 GB/s、對 SM Symmetric 在 4 GB 為 1.25×；**沒有 H100 數字**。https://developer.nvidia.com/blog/fusing-communication-and-compute-with-new-device-api-and-copy-engine-collectives-in-nvidia-nccl-2-28/
- NVIDIA 部落格「Understanding NCCL Tuning」（2025-07-22）：稱 "NCCL can run communication operations on up to 64 CTAs simultaneously (as of NCCL 2.27)"（與 env.rst 的「up to 64 (32 prior to 2.25)」不同，見 C.6）。https://developer.nvidia.com/blog/understanding-nccl-tuning-to-accelerate-gpu-to-gpu-communication

## C.3 現成軟體的支援程度

### NCCL v2.32.6（原始碼）：控制 SM／CTA 數的旋鈕

| 旋鈕 | 位置（v2.32.6） | 預設值／行為 | 從哪一版開始 |
|---|---|---|---|
| `NCCL_MAX_CTAS`、`NCCL_MIN_CTAS` | `docs/userguide/source/env.rst:1904-1964`；`src/init.cc:2414-2434`（env 覆蓋 comm config）、`src/init.cc:2882-2883`（`minCTAs` 預設未設→1、`maxCTAs` 預設未設→`MAXCHANNELS`＝64，`src/include/device.h:80`） | 預設「undefined」，由 NCCL 自行決定；文件說「on platforms where reaching the peak requires a lot of GPU resources, the maximum number of CTAs may be capped by default」；允許值正整數至 64 | `(since 2.17)`（env.rst:1906、1936）；「up to 64 (32 prior to 2.25)」（env.rst:1925–1926 區塊內） |
| `NCCL_MAX_NCHANNELS`、`NCCL_MIN_NCHANNELS` | `env.rst:1093-1125` | 已被上兩者取代（2.17 deprecated）；兩者同設時取較嚴者；文件稱「Reducing the number of channels also reduces the number of CUDA blocks used for communication, hence the impact on GPU computing resources」（env.rst:1097-1098） | NCCL_MAX_NCHANNELS 自 2.5.0、deprecated 2.17 |
| `NCCL_NTHREADS` | `env.rst:1079-1081`；`src/tuning/tuning_general.cc:15`（`NCCL_PARAM(Nthreads, "NTHREADS", -2)`） | 每 CUDA block 的執行緒數；NCCL 一個 channel 一個 block | 未確認起始版本（文件頁未標） |
| `NCCL_NVLS_NCHANNELS`／comm config `nvlsCTAs` | `src/init.cc:68`；`src/transport/nvls.cc:176-203`；`docs/userguide/source/api/types.rst`（nvlsCTAs 段） | **SM90（Hopper）NVLS 預設 16 channel**（`nvls.cc:201` `channels = /*SM90=*/16`；SM100 單機 24、多機 32；Rubin 48／64）；最終 `max(minCTAs, min(maxCTAs, channels))`（`nvls.cc:~205`）；`NCCL_NVLS_NCHANNELS` 不在 env.rst 中（僅在 init.cc 與 2.29.7 二進位字串中） | 未確認；2.29.7 的 `strings` 有 `NCCL_NVLS_NCHANNELS` 與 `Comm config nvlsCTAs reset to...` |
| symmetric kernel 的 CTA 數：`NCCL_SYM_CTAS` | `src/tuning/sym_model.cc:19-27`；`src/tuning/sym_model/lsa.cc:36-97`；上限 `ncclSymkMaxBlocks = 64`（`src/include/sym_kernels.h:26`） | 預設 0＝由模型挑：在 [minCTAs, maxCTAs] 範圍內評估每個候選 CTA 數，**挑「最少 CTA 且預測時間在最佳 1.025 倍以內」者**（`lsa.cc:80-95`）；選擇用的時間含每個 CTA 加罰 2.0–2.5%（`lsa.cc:19`、`:30`）；模型中每 CTA 頻寬常數 Hopper（`cudaArch<1000`）：AllReduce 65 GB/s（`GBps=(1<<30)/1e6`）、其他 44 GB/s，baseLat LL 4.5 μs／非 LL 7.8 μs，peakBw：`AllReduce_RSxLDMC_AGxSTMC` 480、其他 320（GiB/s）（`lsa_base.cc:84-90`） | `NCCL_SYM_CTAS` 字串在 2.29.7 二進位存在；其模型常數是否同 v2.32.6：未確認 2.29.7 是否相同 |
| CTA policy：`NCCL_CTA_POLICY`／comm config `CTAPolicy`／per-collective `CTAPolicy` | `env.rst:885-899`；`docs/userguide/source/api/flags.rst:37-58`；`src/init.cc:85-147`；`src/enqueue/enqueue.cc:3466-3480`；`src/tuning/tuning.cc:242-` | 預設 `DEFAULT`。`EFFICIENCY`：「use minimal number of CTAs to achieve the decent performance when possible」；`ZERO`：「use zero CTA whenever it can, even when that choice may sacrifice some performance」；ZERO 與 EFFICIENCY 同設時 ZERO 優先（`src/config/collconfig.cc:115-116`）。**EFFICIENCY 實作很窄**：僅在無外部 tuner、未設 `NCCL_ALGO`／`NCCL_PROTO`、非 MNNVL、且候選含 NVLS-Simple 時，對「已註冊 user buffer 的 AllGather／ReduceScatter」把演算法換成 NVLS 並用其建議 channel 數（`tuning.cc:242-264`） | env.rst：`(since 2.29, legacy values since 2.27)`；per-collective `CTAPolicy` 為 `(since 2.31)`（types.rst:~590）；2.29.7 字串含 `NCCL_CTA_POLICY` 與 "Both NCCL_CTA_POLICY_ZERO and NCCL_CTA_POLICY_EFFICIENCY are set" |
| per-collective 設定 `ncclCollConfig_t`（`minCTAs`、`maxCTAs`、`nvlsCTAs`、`algSelection`、`CTAPolicy`…） | `src/nccl.h.in:190-215`；`types.rst:461-` | 每次呼叫可覆寫；`algSelection` 用字串過濾演算法（例如 `^symk`），可選 `SYMK_*` 名稱（types.rst:535-565） | `(since 2.31)`（types.rst:331 起多處）；2.29.7 沒有（未在 2.29.7 做功能驗證，僅知文件標 2.31） |
| `NCCL_NCHANNELS_PER_NET_PEER`、`NCCL_P2P_NCHANNELS` 等 | 未逐一讀原始碼；NVIDIA 部落格提到 `NCCL_MIN_CTAS`、`NCCL_NCHANNELS_PER_NET_PEER`（2025-07-22） | 未取得預設值 | 未確認 |
| 啟動時印出實際 channel 數 | `src/init.cc:1960-1961`：`INFO(NCCL_INIT, "%d coll channels, %d collnet channels, %d nvls channels, %d p2p channels, %d p2p channels per peer", ...)` | 可用 `NCCL_DEBUG=INFO` 讀到（待計算節點跑） | — |

### NCCL v2.32.6：copy engine（CE）集體

- **哪些集體有 CE 版**：`src/ce_coll.cc:175-189`（`ncclCeImplemented`）：僅 `ncclFuncAllGather`、`ncclFuncAlltoAll`、`ncclFuncScatter`、`ncclFuncGather`；其餘（含 AllReduce、ReduceScatter、Broadcast、Reduce）一律 `return false`——**CE 路徑沒有任何 reduction**。多機另有 hierarchical CE（`ce_coll.cc:886-` `ncclHierCeAvailable`）只支援 AllGather 與 AlltoAll。
- **驅動需求**：`ce_coll.cc:179-180`「CE is supported in CUDA 12.5 and later: `if (driverVersion >= 12050)`」；官方文件同（`docs/userguide/source/usage/bufferreg.rst:214-222`：CUDA driver version >= 12.5；buffer 以 NCCL window 對稱註冊；communicator 設 `NCCL_CTA_POLICY_ZERO`；單 NVL domain 支援 AlltoAll／AllGather／Scatter／Gather，跨網路支援 AlltoAll／AllGather）。**版本**：「Zero-CTA over NVLink with Copy Engine (CE) is supported since NCCL 2.28; zero-CTA across the network … since NCCL 2.30.6」（bufferreg.rst:214）。
- **可用性檢查**：`ce_coll.cc:194-222`（`ncclCeAvailable`）：需 `comm->symmetricSupport`、window 無 host-backed 段、所有 rank 在同一 NVLink（LSA）team、window 註冊型態屬 `ncclSymSendRegRecvReg` 或 `ncclSymSendNonregRecvReg`；`init.cc:2048-2049` 的 `symmetricSupport` 需 `isAllCudaP2p && NCCL_WIN_ENABLE（預設1） && ncclCuMemEnable() && (GIN 支援 || 單一 LSA team)`。
- **路由**：`src/enqueue/enqueue.cc:3466-3480`：`CTAPolicy` 含 ZERO 且 CE 可用才走 CE，否則走 kernel 路徑（即**沒有 ZERO 就不會自動用 CE**；成本模型 `src/tuning/ce_model.cc` 另供「CE vs. kernel」比較，見 D 節）。
- **CE 相關參數**：`NCCL_CE_COLL_AG_MULTICAST_THRESHOLD`（預設 −1＝成本模型）、`NCCL_CE_BATCH_CAPTURE_ENABLE`（預設 0）、`NCCL_CE_CHUNK_SIZE`（預設 8 MiB）、`NCCL_CE_INTRA_GPU_MEMCPY_ENABLE`（預設 1）（`ce_coll.cc:19-31`）；`NCCL_SYM_CE_THRESHOLD`（`enqueue.cc:44-53`）。2.29.7 二進位含 `CE_COLL_INTRA_BATCH_SYNC_FREQ`、`CE_COLL_INTRA_BATCH_SYNC_MSG_THRESHOLD`、`NCCL_SYM_CE_THRESHOLD`、`ncclLaunchCeColl`、`ceCollTaskAppend` 等字串，**但 `strings` 比對沒有找到 `NCCL_CE_CHUNK_SIZE` 等字串**（可能是參數名於 2.29.7 不同或以其他方式編碼，未確認）。
- **對我們環境的含意（待確認）**：`docs/reports/R01_env_setup.md:43` 記錄 `nvidia-smi` 為 Driver 550.127.08、**CUDA Version 12.4**（`docs/knowledge.md:51` 同為 550.127.08）。若 `ncclCudaDriverVersion()` 回傳 12040，則依 `ce_coll.cc:179-180`，在 v2.32.6 的邏輯下 CE 集體會被判為「not implemented」（2.29.7 是否同門檻：未確認 2.29.7 是否相同）。**這是此方向最先要在計算節點確認的條件。**

### NCCL：對 H100 的 NVLS／Ring 的 SM 佔用

- 預設 Ring 在 H100 單機實際用幾個 channel：**我沒有在原始碼找到固定值**（由拓樸搜尋決定，`comm->nChannels`），要用 `NCCL_DEBUG=INFO` 的 `init.cc:1960` 那行在計算節點讀。
- NVLS 預設 16 channel（見上表）；`NCCL_NVLS_ENABLE` 預設 2（`env.rst:1965-1975`；`src/transport/nvls.cc:59`）。

### 其他系統（以 R03 為準，只補與 SM 相關的事實）

- **DeepEP**：`Buffer.num_sms = 20`、8 rank 推薦 `Config(Buffer.num_sms, 6, 256, 6, 128)`（R03：`deepep/deep_ep/buffer.py:30,235`，見 R03 §2.6）。DeepSeek-V3 論文稱最多 20 SM／H800（見 notes_synthesis T3 摘要）。
- **vLLM／SGLang／TRT-LLM** 的自訂 all-reduce 與 TokenWeave 的 2–8 SM：見 R03、R02_T1 P12；這些 kernel 的 SM 數我沒有再逐一核對。
- **Megatron／PyTorch**：本調查未找到把 `NCCL_MAX_CTAS` 或 CTA policy 預設啟用的程式碼（僅在 `megatron`、`vllm`、`sglang` 原始碼目錄做了 `grep -rn "MAX_CTAS\|CTA_POLICY"`：**我沒有跑這個 grep**，所以此點標「未查」，不寫成「沒有」）。

## C.4 在 Nano5 的前置條件

| 條件 | 文件／原始碼出處 | 檢查方法（指令＋在哪裡跑） | 目前狀態 | 需要管理員？ |
|---|---|---|---|---|
| 驅動／CUDA driver version ≥ 12.5（CE 集體的門檻） | `ce_coll.cc:179-180`；`bufferreg.rst:216` | 計算節點：`nvidia-smi \| head -4`；或 Python `ctypes` 呼叫 `cuDriverGetVersion`（整數 12040＝12.4） | **已知：550.127.08／CUDA 12.4（R01:43）→ 低於門檻（推論，待實測）** | 升級驅動需要管理員 |
| NCCL ≥ 2.28（CE）、≥ 2.29（`NCCL_CTA_POLICY` 字串型）、≥ 2.31（per-collective config、CE 跨網路 2.30.6） | env.rst:889；bufferreg.rst:214；types.rst:331 | `strings .../libnccl.so.2 \| grep "NCCL version 2\."`（登入節點，秒級） | 已知：2.29.7（已用 `strings` 確認） | 否（venv 內可裝新版 `nvidia-nccl-cu12`；是否有可用版本未查） |
| NVLS multicast 可用（symmetric multimem、NVLS kernel 的前提） | `nvls.cc:243-267`（`CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED`）；`init.cc:2048-2049` | 計算節點：`NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,TUNING` 跑 `all_reduce_perf -b 1M -e 1G -g 8`，看 `nvls channels` 與是否出現 NVLS 失敗訊息 | **未知**（R03 提到 M20 要確認；fabric manager／IMEX 屬管理員端） | 若 multicast 未開，需管理員 |
| 實際 channel 數／CTA 數 | `init.cc:1960-1961` | 同上，讀 `coll channels`／`nvls channels` | 未知（待 M20／M23） | 否 |
| 量測「GEMM 變慢多少」所需的 kernel 級 SM 佔用 | — | 計算節點：Nsight Systems（`nsys`）；**登入節點 `which nsys ncu` 沒有輸出**（我只查了 PATH），是否有模組未查 | 未知 | 可能需 `module load` |
| 單機內 CE 集體的 buffer 註冊條件（`ncclMemAlloc`＋`ncclCommWindowRegister`、cuMem） | bufferreg.rst:195-232；`init.cc:2049` | 計算節點小程式或 nccl-tests（v2.21.1）是否支援 `-R 2`（symmetric registration）：**我沒有查 nccl-tests v2.21.1 的選項**，未知 | 未知 | 否 |

## C.5 「找不到有人量過」的清單

- 在 **H100（8 卡 NVSwitch 單機）** 上，以 **NCCL 的 CTA 數（`NCCL_MAX_CTAS`／`NCCL_MIN_CTAS`）或 CTA policy** 為變數，量「通訊頻寬 vs. 同時執行 GEMM 變慢」的論文：**未找到**（查詢見 C.8 第 1、2、3、5 條）。最接近：Zero-SM（B200，且有 CTA 數掃描）、TIDE（NCCL 2.31；GPU 型號未明寫，CTA 數掃描有）、Lee et al.（H100 但未掃 CTA）。
- 在 **H100** 上量 **NVLS／symmetric kernel／ring／CE 四種實作各自的 GEMM 慢多少（同一 GEMM、同一訊息大小）**：**未找到**。
- 在 **H100** 上量 **copy engine 集體（AllGather／AlltoAll）的實測頻寬與小訊息延遲**：未找到（NVIDIA 部落格為 Blackwell；Zero-SM 為 B200／GB200；ParallelKittens 只量單純 copy engine 傳輸 368.82 GB/s，非 NCCL CE 集體）。
- **SHARP（交換器內加總）佔多少 GPU SM**：本分工範圍內未找到量測（SHARP 屬方向 A 的分工）。
- 以 **CE 做含 reduction 的集體（AllReduce／ReduceScatter）**：NCCL v2.32.6 不支援（`ce_coll.cc:175-189`）；ConCCL 腳註明說 DMA 不支援算術；TIDE 的 reduce-scatter 用「copy engine＋short-lived thread blocks＋NVSwitch」混合（摘要頁），**不是純 CE**。純 CE 做 reduction 的既有實作：未找到。

## C.6 互相矛盾或不確定之處（不裁決）

1. **NCCL 單一集體最多幾個 CTA**：env.rst 說 `NCCL_MAX_CTAS` 上限「64 (32 prior to 2.25)」；NVIDIA 部落格（2025-07-22）說「up to 64 CTAs (as of NCCL 2.27)」；Zero-SM 說 B200 ring 預設 32 CTA、symmetric 預設 4 CTA。三者並不矛盾但版本邊界（2.25 vs. 2.27）不同，我沒查 release notes 原文。
2. **Zero-SM 的 GEMM 慢 32.9%（legacy）vs. Lee et al. 的平均 18.9%、最高 40.0%**：硬體（B200 vs. H100）、工作負載、基準（clean pass vs. 理想無通訊）不同，不可直接比。
3. **TIDE 的「slows by 69%」是單波 GEMM 的機制性例子（16 CTA×4-cluster 佔掉 33 個 slot 中的 4 個＋…）**，不是平均值；不可與 Zero-SM 的中位數 32.9% 直接比。
4. **WebSearch 的二手說法與原文不符**：搜尋摘要稱 arXiv:2608.22602（Pan et al.，「Architecting the Next Generation of Asynchronous, Distributed GPUs」）有在 H200 上量 cuBLAS GEMM 與 NCCL AllGather 的 SM 重疊（24 個 NCCL 佔用的 SM）；我打開的 abs 頁摘要只說是週期級模擬器（H100 驗證 13.4% 週期誤差）、案例是 chiplet、SRAM、inter-GPU prefetching，**沒有提到該 GEMM+AllGather 量測**。我未讀其內文，故**不列入表中、也不採用該說法**（列在未能核實清單）。
5. CE 在 2.29.7 的實際門檻與參數名：2.29.7 `strings` 沒有 `NCCL_CE_CHUNK_SIZE`，但 v2.32.6 有；差異原因未確認。

## C.7 arXiv ID／DOI 清單（供匯出 BibTeX）

2611.18075（Zero-SM）；2507.03114（Lee）；2412.14335（ConCCL）；2610.38228（TIDE）；2510.00991（VCCL）；2606.09200（Cui & Pericas）；2609.21483（Weave）；2511.13940（ParallelKittens）；2512.10236（FiCCO）；2502.19811（Comet）；2609.13585（mKernel，見 R02_T2 #2，我只在 R02 引用，ID 取自 R02 表中網址 `2609.13585`）；2601.20595（Syncopate）；2607.19539；2602.20656（Lagom）；2510.20171（NCCLX）。

## C.8 搜尋紀錄（C 節）

| # | 查詢（工具） | 約看幾筆 | 備註 |
|---|---|---|---|
| 1 | arXiv `search_papers`：`abs:"copy engine" AND abs:"collective" AND abs:"GEMM"`（relevance，2023+） | 2 | 得 Zero-SM、TIDE（2610.38228） |
| 2 | arXiv：`abs:"ConCCL" OR ti:"concurrent" AND abs:"collective" AND abs:"interference"` | 3 | 得 ConCCL（2412.14335）；其餘為無關（GC、BLE）排除 |
| 3 | arXiv：`abs:"computation-communication" AND abs:"contention" AND abs:"GPU"` | 4 | 得 FiCCO（R02 已有）；其餘無關（SP-MoE、B300 field report、Gaussian splatting）排除 |
| 4 | arXiv `citation_graph` 對 2412.14335、2507.03114、2602.20656 | 0 | **全部 rate-limited，未取得** |
| 5 | WebSearch：`NCCL SM usage GEMM slowdown overlap H100 "NCCL_MAX_CTAS" OR "number of channels" characterization arXiv`（standard） | 10 | 得 2608.22602（已核實為無關，見 C.6-4）、PyTorch 論壇貼文、Chalmers 論文（未核實）、NCCLX |
| 6 | WebSearch：`arXiv GPU communication SM contention overlapped GEMM NCCL SM allocation dynamic partition…`（extended） | 10 | 得 2606.09200、2607.19539（R02 已有）、Zero-SM |
| 7 | WebSearch：`arXiv 2025 2026 GPU green contexts SM partition communication kernels…` | 10 | 得 Weave（2609.21483）、Chalmers 論文、2609.24270（die scaling，**未打開，排除**）、NVIDIA green contexts 部落格（未打開） |
| 8 | WebSearch：`arXiv copy engine NVLink all-reduce reduction kernel offload DMA…` | 10 | 得 NVIDIA 2.28 部落格（已 WebFetch）、ARK（NSDI'23，Hwang et al.，**未打開，排除**） |
| 9 | paper-search `search_semantic`（兩條） | 0 | 回傳空 |
| 10 | WebFetch：各 arXiv abs 頁（2608.22602、2610.38228、2606.09200、2510.00991、2611.11303）＋NVIDIA 兩篇部落格＋ PDF `pdftotext`（2412.14335、2610.38228、2510.00991、2606.09200、Zero-SM 與 Lee 的 R02 保存文字） | — | 逐一核實 |
| 11 | 原始碼：`grep -rn` 於 `/work/allen0117/comm_opt_env/src/nccl`（`src/`、`docs/`、`plugins/`）及 `strings` 於 `libnccl.so.2` | — | 見 C.3 |

**排除**：2608.22602（內容與查詢無關）；2609.24270、ARK、Chalmers 論文（未打開）；`.pdf` 之外的 PyTorch 論壇貼文（非論文）。

---

# 第二節　方向 D：用效能模型挑選集體通訊做法

## D.1 方向定義（照抄 Opus）

**D：用效能模型挑選集體通訊做法**（演算法、協定、通道數）。一定要回答：(4) NCCL tuner plugin 介面（原始碼位置、能控制什麼、哪一版開始）與既有 tuner 實作（官方範例、雲端廠商、AutoCCL、Lagom 等）；(5) NCCL 內部的效能模型在哪（`檔案:行號`），有沒有論文評估它的準確度；(6) 用模型或學習方法選集體通訊參數的論文（2023 年後），硬體與效益。

## D.2 最接近的既有工作

| 編號 | 標題／作者／年／出處 | 做了什麼 | 沒做／自述限制 | 硬體規模 | 程式碼 | 網址 |
|---|---|---|---|---|---|---|
| D1 | **NCCL-BOT: Closing the Loop on NCCL's Cost Model with Online Bandits** Shteingart, Sapio, Avin, Bureddy, Gorentla Venkata, Krishnan, Mannor, Potluri, Shabat, Wertheimer（10 人，含多位 NVIDIA 作者），2026-11-08，arXiv:2611.11303（46 頁） | tuner plugin：每個 context 一個 bandit，以 NCCL 成本模型 warm start，用實測延遲學習；**只選演算法＋協定，不動 channel 數**（「By design our plugin selects through the cost table and changes only algorithm and protocol」）。離線 H100 sweep：成本模型損失 >10% 的 context 由 19 降到 1，p99 由 103% 降到 9.9%；AUTO 在 H100 上約 80% 的 context 取到實測最佳（pooled error 20.0%，over all contexts 平均損失 5.04%、p99 75.4%）。引用 Williams et al.：出貨的模型「calibrated for isolated collectives」；稱 NCCL「passes no measured latency」（tuner 介面自 NCCL 2.22 起為 v3） | 原文：「We do not show a net end-to-end win.」；小於 1 MiB 學習器輸給 stock（「on H100 the cost model is exact there offline」）；活體測試每策略只跑一次；NVLS 首次使用會停頓 5.7–7.8 s；超參數由 H100 凍結後套用於 GB200 | 一台 H100 系統（8 GPU／節點，64 GPU，Spectrum-X Ethernet，以延遲模擬成多節點）、NCCL 2.30.4 / 2.30.7 / 2.32.0a4；GB200 兩套網路 | 未說明 | https://arxiv.org/abs/2611.11303 |
| D2 | **AutoCCL**（見 R02_T1 P22）Xu et al.，NSDI'25 | 線上調 NCCL 演算法／協定／transport／channel 數／thread 數／chunk size，分離「實作相關」與「搜尋空間大小」參數；1.07–1.32× 每 iteration | 基於 NCCL 2.18.3、A40；IB 專屬參數留待未來（§8.1）；NVSwitch／NVLS 未提 | 2 節點 16×A40、4 節點 32×A40 | https://github.com/gbxu/autoccl | https://www.usenix.org/conference/nsdi25/presentation/xu-guanbin |
| D3 | **Lagom**（見 R02_T1 P10／T2 #6）Xu et al.，2026-02-24，arXiv:2602.20656 | 統一 cost model＋優先序搜尋，共同調通訊參數（channel 數、chunk size），明說目標是降低通訊對重疊計算的拖慢（通訊競爭使計算最多降速 35%，§1）；1.07–1.33× over NCCL | 無程式碼（「will be open-sourced soon」）；NCCL 2.18.3、A40 | 2 節點×16 A40 | 未釋出 | https://arxiv.org/abs/2602.20656 |
| D4 | **Bandit-Guided NCCL Tuning for Multi-Node GPU Clusters** Madan & Raigosa，Stanford CS244C '26（2026-03，6 頁；**課程專案，非同儕審查，DOI 欄位為占位**） | 在 8×A100（1×8、2×4、4×2 三種拓樸）量 AUTO：**單機 NVLink 上，有並行計算（4096² fp32 matmul）時 9 個訊息大小中有 6 個的最佳「演算法＋協定」改變（因協定的 SM 佔用不同），且此效應對 NCCL 成本模型不可見**；2 節點時 AUTO 全選 Ring，Tree 最多快 2.9×、64 MB 延遲損失 57%；線上 bandit（433 行 C，tuner plugin）降 25–42%，約 40 iteration 收斂 | 僅 A100、雲端（Modal）；訊息 32 KB–256 MB 9 個大小、單一 4096² fp32 matmul 當重疊；課程專案 | 8×A100-SXM4-80GB（world size 8） | 未說明 | https://www.scs.stanford.edu/26wi-cs244c/proj/bandit_guided.pdf |
| D5 | **Practical Machine Learning Autotuning for Large-Scale Collective Communication（ACCLAiM）** Wilkins, Guo, Thakur, Dinda, Hardavellas，IEEE TPDS vol.37 no.5, pp.1032–1047, 2026-05（DOI 10.1109/TPDS.2026.3661876） | MPI 集體演算法選擇的 ML autotuner：variance-based active learning＋拓樸感知平行蒐集，每個集體訓練 5–10 分鐘（至 2048 節點），應用最高 4.1× | **MPI／CPU 超級電腦（Aurora、Polaris），非 NCCL、非 NVLink**；先前 ML autotuner 訓練時間過長 | Aurora（exascale）、Polaris | 開源 prototype（網址我沒取得） | https://doi.org/10.1109/TPDS.2026.3661876 |
| D6 | **Demystifying NCCL**（見 R02_T1 P19）Hu…Hoefler，arXiv:2507.04786 | 以 NCCL 2.19.1 分析協定與演算法；明說不含 NVLS／CollNet | 「future versions of NCCL will need to support automated algorithm selection」（§VII）；選擇模型定性而非定量（§V-E） | GH200（Alps） | — | https://arxiv.org/abs/2507.04786 |
| D7 | **PICO**（見 R02_T1 P18）Pasqualoni et al.，arXiv:2508.16809 | MPI 預設演算法選擇常比最佳低約 30–40%，最差僅達 1/5（§4.1）；可產生調參檔 | 本文未報告 NCCL 數據；調參檔手動建立 | Leonardo、LUMI、MareNostrum 5 | 未列 | https://arxiv.org/abs/2508.16809 |
| D8 | **Every Microsecond Matters**（見 R02_T1 P3） | 自述「kernel selection is currently driven by empirical measurements」（§IX） | 無效能模型 | GB200 | https://github.com/ss16118/low-latency-nccl | https://arxiv.org/abs/2607.16100 |
| D9 | **Zero-SM NCCL**（見 R02_T1 P1） | 自述「pipelined rail schedule 與 tuner 尚未進入已釋出的 NCCL」（R02_T1 §3-F，§III-D2） | 同 C1 | GB200／B200 | 同 C1 | 同 C1 |
| D10 | **ATLAHS**（見 R02_T1 P20）；ASTRA-sim 官方 HGX-H100 驗證頁 | ATLAHS：模擬器，驗證誤差 <5%（GH200）；ASTRA-sim 驗證頁（WebSearch 摘要）稱 8×H100 all-reduce（強制 NCCL Ring）幾何平均誤差 9.69%（8 GPU）、12.01%（4）、20.63%（2）；**此數字我只看過搜尋摘要，未打開驗證頁（R02 也只列網址未核實）** | 模擬的是已選定的演算法，不是挑選 | GH200／HGX-H100 | https://github.com/spcl/atlahs | https://astra-sim.github.io/astra-sim-docs/validation/hardware/gpu-validation-hgx-h100.html |

**雲端廠商 tuner**：NCCL-BOT 引文稱 "vendor plugins do so with fixed rules, e.g. the AWS OFI tuner picks a pair from measured regions over message size and rank count (3)"。我 WebFetch 了 `https://github.com/aws/aws-ofi-nccl` 首頁，**頁面沒有提到 tuner plugin**（只有 `topology` 目錄，內容不明）。因此「AWS OFI tuner」這一點**未能獨立核實**。Google fastsocket 與 CoMMA 在 NCCL 官方 `plugins/net/google-fastsocket`、`plugins/profiler/google-CoMMA`（原始碼目錄存在），但那是 net／profiler plugin，不是 tuner。

**非同儕審查／未打開的線索**：Xsim（2606.10440）稱 SimAI 在混合 H100／A100 叢集預測誤差約 80%（WebSearch 摘要，**未打開**）；ACCLAiM 的 2022 CLUSTER 前身（`paragon.cs.northwestern.edu/papers/2022-CLUSTER-ACCLAiM-Wilkins.pdf`，**未打開**，僅在搜尋結果出現）；Springer J. Supercomputing 2024「An autotuning approach to select the inter-GPU communication library」（`10.1007/s11227-024-06794-3`，WebFetch 被導向登入頁，**未能開啟，未核實**）。

## D.3 現成軟體的支援程度

### NCCL tuner plugin 介面

- **位置（v2.32.6）**：介面定義 `src/include/plugin/nccl_tuner.h`（`NCCL_TUNER_PLUGIN_SYMBOL "ncclTunerPlugin_v7"`，行 26；`ncclTuner_t` ＝ v7，行 19–21）、各版 `src/include/plugin/tuner/tuner_v2.h … tuner_v7.h`；載入邏輯 `src/plugin/tuner.cc`（依序嘗試 v7→v6→v5→v4→v3→v2 符號，行 76–90；env 讀取行 60）；範例 `plugins/tuner/{basic,example}`（`example/plugin.c`＋CSV 設定檔 `nccl_tuner.conf`＋`scripts/optimize_config.py`）。載入規則：`NCCL_TUNER_PLUGIN` 設為庫名或後綴，否則找 `libnccl-tuner.so`，再找 net plugin 內的 tuner 符號（`env.rst:618-628`）。
- **能控制什麼（v7，`tuner_v7.h`）**：
  - `init(ctx, commId, nRanks, nNodes, logFunction, nvlDomainInfo, constants)`：可**修改 `ncclTunerConstants_v7_t` 內建成本模型常數**（`baseLatencies`、`hwLatencies`、`llMaxBws`、`perChMaxRingLL128Bws`、`perChMaxTreeLL128Bws`、`perChMaxTreeBws`、`perChMaxNVLSTreeBws`，表格以 [ALGO][PROTO]、[HW][ALGO][PROTO]、[compcap][scale] 索引，`tuner_v7.h:20-30`）。
  - `getCollInfo(ctx, collType, nBytes, numPipeOps, collCostTable, numAlgo, numProto, regBuff, *nChannels)`：就地改 algo×proto 成本表（`NCCL_ALGO_PROTO_IGNORE`＝−1 表示忽略；示範 README 說設成 0.0 即最優先）、輸出 `nChannels`（「number of channels (hence SMs) to be used」）。
  - `getChunkSize(...)`（v7 新增，可選）：覆寫 chunk size。
  - 失敗（非 `ncclSuccess`）時 NCCL 退回預設調校。
- **重要限制（原始碼）**：`src/tuning/tuning.cc:211-236`：tuner 只對「五個標準集體」（`func < NCCL_NUM_FUNCTIONS`）呼叫，且只看得到**舊式 algo×proto 表**（`generalTable[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS]`；`NCCL_ALGO_*` 含 Tree、Ring、CollnetDirect、CollnetChain、NVLS、NVLSTree、PAT）。symmetric kernel（`symKernelId`）與 CE（`ceMethodId`）的候選在 `tunings` 清單中有各自的 `timeUs`，tuner 表只填 algo 與 proto 皆有定義者（`tuning.cc:214-222` 迴圈 `continue` 跳過 `NCCL_ALGO_UNDEF`），因此**依原始碼，tuner 的改表不能降低或提高 symmetric kernel／CE 候選的成本**（這是我對 `tuning.cc` 的讀法，沒有實驗證實）。`input->comm->tuner == NULL` 時才套用 EFFICIENCY 偏好（`tuning.cc:246`）。
- **哪一版開始**：我沒有直接讀到「某 tuner 版本從哪個 NCCL 版本開始」的官方表。能確認的：NCCL-BOT 全文寫「Since NCCL 2.22 the tuner interface (v3) passes the callback getCollInfo the collective type, message size, and a per-collective cost table…」（原文，第 91–95 行附近）；一個 WebSearch 摘要稱 "Since NCCL v2.19, an external tuner plugin can override AUTO's collCostTable"（Stanford 課程專案，**二手、未在 PDF 內核對**）；NVIDIA 部落格（2025-07-22）說 tuner 介面「minimal as of NCCL 2.27」（**部落格文字的 WebFetch 摘要**，另有一句稱最小介面「supports CTA tuning but not protocol or algorithm tuning」來自搜尋摘要，與 `getCollInfo` 原始碼註解明顯不同，不採用）。**2.29.7 二進位 `strings` 結果：含 `ncclTunerPlugin_v2`、`_v3`、`_v4`、`_v5`，沒有 `_v6`、`_v7`**——即我們安裝的 NCCL 2.29.7 最多接受 v5 plugin（含 constants 與 `nvlDomainInfo`）；v6 與 v7（`getChunkSize`）在 2.32.6 才有，v6/v7 各自從哪個 NCCL 版本開始：未查到。
- **範例 plugin**：`plugins/tuner/example/README.md`：CSV 規則 `collective,min_bytes,max_bytes,algorithm,protocol,channels,nNodes,nRanks,numPipeOps,regBuff`，環境變數 `NCCL_TUNER_CONFIG_FILE`，可用 `-1` 表示保持預設 channel；`scripts/optimize_config.py` 可從效能 CSV 產生最佳設定（`scripts/README.md`）。`plugins/tuner/basic/README.md`：預設實作把 RING+SIMPLE 設成成本 0.0、channel 設 1（占位用）。

### NCCL 內部的效能模型（位置）

（均為 v2.32.6；`src/graph/tuning.cc` 是 2.32 以前的舊位置，PyTorch 的程式碼註解仍引用舊路徑）

- 總入口：`src/tuning/tuning.cc:151-` `ncclTuningSelectBestTuning`（取 `selectionTimeUs` 最小者，`tuning.cc:154-175`）與 `ncclTuningCompute`（`tuning.cc:180-`）；模型登錄/介面 `src/tuning/cost_model.h`（`ncclTuningModelEntry_t`：init／model／finalize）。
- **Ring**：`src/tuning/ring.cc:19-142`——頻寬＝`bwIntra`（或 `bwInter`）×channel 數，LL 取 `min(llMaxBw, busBw*0.5)`，LL128 取 `min(busBw*0.92, nChannels*perChMaxRingLL128Bw)`；延遲＝`baseLatencies`＋逐步的 `hwLatencies`（+網路 overhead，`NCCL_NET_OVERHEAD`）；`ncclTuningRingModelSim`（`ring.cc:97-142`）對 Ring/Simple 的多機 AllReduce 乘 plateau 係數（<sm80：1.9、其餘：1.4）。
- **Tree**：`src/tuning/tree.cc`（`busBw*.92`、`1/3.8`、`treeCorrectionFactor`，`cost_model.h` 註解「Trees are not perfectly sticking to the model for medium sizes. Applying a static correction factor…」）；**NVLS**：`src/tuning/nvls.cc`、`pipeline_model/nvls.cc`；**CollNet**：`src/tuning/collnet.cc`；**PAT**：`src/tuning/pat.cc`。
- **預設常數**：`src/tuning/cost_model.cc:147-210`（`ncclTunerConstantsDefaults`）：例如 `baseLatencies` Tree/Ring `{6.8,14.0,8.4}`／`{6.6,14.0,8.4}`；`hwLatencies[NVLINK]` Ring `{0.6,1.9,3.4}`、NVLS Simple 25 μs、NET 的 NVLS 18 μs；`llMaxBws` Hopper `{141.0, 45.0, 35.0}`（行 196，註記「avg of ring & tree」）。Rubin 另有 `src/tuning/platforms/rubin.h`。
- **symmetric kernel 模型**：`src/tuning/sym_model.cc`、`sym_model/lsa.cc`、`sym_model/lsa_base.cc:88-105`（Hopper：`baseLat` LL 4.5／非 LL 7.8 μs；`smBw` 65／44 GB/s；`peakBw` 480／320 GB/s；`bw=ncclSoftMin(nBlocks*smBw*busMultiplier, peakBw, smBw)`，時間＝`baseLat + ncclSoftPlus(busBytes/bw − 1, 1)`）；GIN 版 `sym_model/gin.cc`。
- **CE 模型**：`src/tuning/ce_model.cc`：AllGather 的 latency floor 表 `CE_AG_FLOOR`（sm_90 UC `{17.50,1.957}`／graph `{9.98,6.160}`、MC `{14.90,0.726}`／`{14.45,2.004}`，行 33–39，單位 μs，`base+perRank*nRanks`）與 `CE_NVLINK_BW = {392.0, 779.0}` GB/s（sm_90、sm_100，行 41）；只有 AllGather 有 CE 延遲模型（`ncclCeAgLatencyUs`）。
- **使用者可見的限制**：模型只在 communicator 初始化時以拓樸算出頻寬、逐集體獨立評估（Stanford 專案與 NCCL-BOT 的說法；NCCL 原始碼 `tuning.cc` 的 `ncclTuningCompute` 以單一 `input` 評估，**沒有輸入「其他 stream 正在跑什麼」的欄位**——這是我讀 `struct ncclTuningInput_t` 使用處得到的印象，沒有逐欄位列舉）。
- **下游使用**：PyTorch Inductor 複製了這組模型常數做通訊延遲估計（`/work/allen0117/comm_opt_env/src/pytorch/torch/_inductor/comm_analysis.py:220-310`〔常數〕、`:458-734`〔`estimate_nccl_collective_runtime*`〕；註解稱改編自 `src/graph/tuning.cc` 的 `ncclTunerConstantsDefaults`；PyTorch 版本＝2026-11-16 的 release tag，所用 NCCL 版本未標）。

### 其他框架

- 本分工沒有逐一檢查 vLLM／SGLang／TRT-LLM／Megatron 是否載入 NCCL tuner plugin（未查，不寫成「沒有」）。vLLM 的 all-reduce 後端選擇是靠門檻表而非模型，見 R03。

## D.4 在 Nano5 的前置條件

| 條件 | 文件／原始碼出處 | 檢查方法 | 目前狀態 | 需要管理員？ |
|---|---|---|---|---|
| 可編譯並載入 tuner plugin（`.so`，不需 root） | `env.rst:618-628`；`plugins/tuner/example/Makefile` | 計算節點：`NCCL_TUNER_PLUGIN=/path/libnccl-tuner-example.so NCCL_DEBUG=INFO`，看是否出現 tuner 載入訊息 | 未知（待小實驗；編譯需 `gcc`，登入節點 `module avail gcc` 有 gcc 目錄存在，但版本未確認） | 否 |
| NCCL 2.29.7 接受的 plugin API 版本 | `src/plugin/tuner.cc:76-90`（v2.32.6）；2.29.7 `strings` | `strings .../libnccl.so.2 \| grep ncclTunerPlugin_v` | **已知：v2–v5（沒有 v6、v7）** | 否 |
| 要用 v7（`getChunkSize`）或 per-collective config（≥2.31）需升級 NCCL | types.rst `(since 2.31)` | 在 venv 內安裝較新 `nvidia-nccl-cu12`（**是否有此版本與 torch／vLLM 相容：未查**） | 未知 | 否 |
| 想讓 NCCL 內建模型的輸入（baseLat／hwLat 等）符合 H100＋NVSwitch＋IB NDR | `cost_model.cc:147-210`；v5/v7 `init()` 可改常數 | 對照 `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=TUNING` 輸出（`TRACE` 級訊息需 NCCL 編為 `ENABLE_TRACE`，`makefiles/common.mk:197-198`；**發行版二進位是否含 TRACE 未確認**）與 `nccl-tests` 實測 | 未知 | 否 |
| 量「真實」延遲／頻寬的 ground truth（nccl-tests） | `third_party/nccl-tests`（v2.21.1） | `all_reduce_perf -b 8 -e 1G -f 2 -g 8`（計算節點，獨佔） | 工具已裝（R01） | 否 |
| 不被其他工作干擾、獨佔整台 | CLAUDE.md 規則 7、D048 | Slurm exclusive | — | 否 |
| 跨機（NVLS Tree／IB）時 tuner 所需的拓樸資料（`nvlDomainInfo`） | `tuner_v5.h`／`tuner_v7.h` | 在 `init()` 印出 | 未知 | 否 |

## D.5 「找不到有人量過」的清單

- 在 **H100（8 卡 NVSwitch）單機或 ≤4 台 IB NDR** 上，系統評估 **NCCL 內建成本模型對「演算法＋協定＋channel 數＋NVLS／symmetric／CE 候選」的選擇準確度**的論文：**未找到**。最接近：NCCL-BOT（H100，但 Spectrum-X Ethernet、多節點以延遲模擬、只評演算法＋協定、NCCL 2.30.x）；Stanford 課程專案（A100，非同儕審查）。查詢見 D.8 第 1–6 條。
- **評估 symmetric kernel（`SYMK_*`）與 CE 候選的成本模型準確度**的論文或資料：未找到（NCCL-BOT 的 arms 為 Ring/Tree/NVLSTree × LL/LL128/Simple）。
- **有並行 GEMM（重疊）時，成本模型挑選是否仍然合理**：只有 Stanford 課程專案（A100，4096² fp32 matmul，9 個大小）；同儕審查論文中：未找到（Lagom 與 AutoCCL 是以搜尋調參來繞過，沒有評估內建模型）。
- **tuner plugin 能不能影響 symmetric／CE 候選的實測**：原始碼讀法見 D.3；實驗驗證：未找到。
- **H100 上用學習方法選 NVLS 開／關**：未找到（NCCL-BOT 的 H100 離線 sweep 含 NVLS Tree 這個 arm，且報告「首次使用 NVLS TREE SIMPLE 有 5.7–7.8 s 停頓」，但沒有單獨的 NVLS 開關 A/B）。
- ACCLAiM 類 ML 選擇器**用於 NCCL／GPU 叢集**：未找到（ACCLAiM 為 MPI）。

## D.6 互相矛盾或不確定之處（不裁決）

1. **AUTO 的準確度**：NCCL-BOT 說 H100 上 AUTO 約 80% 的 context 取到最佳、小於 1 MiB「cost model is exact offline」，但整體 p99 損失 75.4%（大訊息、NVLS／Tree 邊界）；Stanford 課程專案（A100）說單機平均差距僅 1.2%、多機放大到 57%。硬體、拓樸、NCCL 版本、量測方式皆不同，不可直接比。
2. **tuner 介面起始版本**：NCCL-BOT 稱 tuner 介面 v3 自 NCCL 2.22；Stanford 專案稱 v2.19；NVIDIA 部落格以 2.27 為基準描述最小介面。我沒有取得官方 release note 逐版證據（R03 §2.8 的 release notes 摘要也未列 tuner API 起始版）。
3. **NVIDIA 部落格 vs. 原始碼**：搜尋摘要稱部落格說最小 tuner 介面「supports CTA tuning but not protocol or algorithm tuning」；原始碼 `getCollInfo` 的註解與 README 明確可改 algo×proto 表。未能打開部落格原文核對此句（WebFetch 摘要的結論相反：部落格說 plugin「can select the algorithm and protocol… and can override the CTA count」）。
4. **WebSearch 摘要稱 SimAI/Xsim 的誤差數字與 ASTRA-sim 驗證頁的數字**：皆未開原頁核實。
5. **AWS OFI tuner**：NCCL-BOT 引用為既有雲端 tuner，官方 repo 首頁看不到 tuner；未核實。
6. Zero-SM 論文提到「tuner 尚未進入已釋出的 NCCL」（§III-D2）與 NCCL 原始碼已含 CE 模型（`ce_model.cc`）是否為同一件事：未確認。

## D.7 arXiv ID／DOI 清單（供匯出 BibTeX）

arXiv：2611.11303（NCCL-BOT）；2602.20656（Lagom）；2507.04786（Demystifying NCCL）；2508.16809（PICO）；2607.16100（Every Microsecond）；2611.18075（Zero-SM）；2505.08936（ATLAHS，見 R02_T1 P20）。
DOI：10.1109/TPDS.2026.3661876（ACCLAiM，TPDS 2026）；AutoCCL：NSDI'25（無 DOI，網址見 D2）；Stanford CS244C '26 專案（無有效 DOI，PDF 網址見 D4，**非同儕審查**）。
未核實不要匯出：Springer 10.1007/s11227-024-06794-3；ACCLAiM CLUSTER 2022；Xsim 2606.10440；SimAI。

## D.8 搜尋紀錄（D 節）

| # | 查詢（工具） | 約看幾筆 | 備註 |
|---|---|---|---|
| 1 | arXiv：`abs:"NCCL" AND abs:"tuning" AND abs:"performance model"` | 0 | 空 |
| 2 | arXiv：`abs:"collective communication" AND (…"reinforcement learning" OR …) AND abs:"algorithm selection"` | 0 | 空 |
| 3 | arXiv：`abs:"NCCL" AND (abs:"auto-tuning" …)`、`abs:"NCCL" AND abs:"tuner plugin"`、`ti:"NCCL" AND abs:"tuning"`、`abs:"collective communication" AND abs:"cost model" AND abs:"GPU"`、`abs:"DMA engine" AND abs:"collective" AND abs:"reduction"`、`abs:"streaming multiprocessors"…` | — | **rate-limited，未取得結果** |
| 4 | WebSearch（standard）：`NCCL tuner plugin auto-tuning algorithm protocol channels learned model arXiv 2025 2026` | 10 | 得 NVIDIA 部落格、Stanford 專案、RCCL tuner 文件（AMD，**排除**：非 NCCL） |
| 5 | WebSearch（extended）：`arXiv collective communication tuning NCCL parameters search…` | 10 | 得 NCCL-BOT（2611.11303）、VCCL（2510.00991）；NCCLZ（2605.12396）、Tools-CC-Bench（2609.08739）為壓縮主題，**排除、未打開** |
| 6 | WebSearch（extended）：`machine learning MPI collective algorithm selection autotuning GPU…` | 9 | 得 ACCLAiM、Springer 2024（未能開）、ICL 2020 技術報告（早於 2023，**排除**） |
| 7 | WebSearch（standard）：`SimAI OR ASTRA-sim NCCL algorithm protocol selection model accuracy H100…` | 10 | 得 ASTRA-sim 驗證頁、Xsim；Echo（2412.12487，端到端模擬，**排除**） |
| 8 | WebSearch（standard）：`arXiv analytical performance model NCCL NVLS ring tree…` | 10 | 得 PyTorch `comm_analysis.py`（後以本機原始碼核實）、Demystifying NCCL |
| 9 | WebFetch：arXiv abs（2611.11303）、Stanford PDF（`pdftotext`）、ACCLAiM PDF（`pdftotext`）、NVIDIA 部落格、aws-ofi-nccl GitHub 頁 | — | 逐一核實 |
| 10 | 原始碼：`grep`／`sed` 於 `src/tuning/`、`src/plugin/`、`src/include/plugin/`、`plugins/tuner/`、`docs/`；`strings` 於 2.29.7 | — | 見 D.3 |

---

# 未能核實／未完成項目（彙總）

1. `citation_graph`（往後找引用）三次全部 rate-limited，**沒有做**；多數 arXiv 搜尋也被 429 擋住，C 與 D 的文獻覆蓋因此比規格要求（10–25 篇）薄：C 節新核實 6 篇（C3–C7、TIDE 含在內）＋R02 沿用，D 節新核實 3 篇（NCCL-BOT、ACCLAiM、Stanford 專案）。
2. 未打開：Chalmers GCOverlap 論文、2609.24270、ARK（NSDI'23）、Xsim 2606.10440、ACCLAiM CLUSTER'22、Springer 2024 inter-GPU 選擇、ASTRA-sim 驗證頁。
3. TIDE 與 VCCL 的實驗 GPU 型號文字中沒有明寫（TIDE 的 132 SM 為我的推算）。
4. NCCL 2.29.7 與 v2.32.6 的差異：只對 `strings` 與少量字串比對；各旋鈕預設值在 2.29.7 是否相同全部標「未確認」。
5. tuner API v6／v7 各自的起始 NCCL 版本、NCCL 2.25／2.27 的 CTA 上限變更版本邊界：未找到官方逐版證據。
6. AWS OFI tuner、AutoCCL／Lagom 與 NCCL 2.29.7 的相容性：未驗證。
7. vLLM／SGLang／Megatron 是否使用 NCCL tuner plugin、`NCCL_MAX_CTAS`、`NCCL_CTA_POLICY`：未查。
8. Zero-SM 圖 1 中 Hopper 所需 SM 數：需看原圖。

## 需要 Opus 優先注意的兩項（事實，不是建議）

- 驅動 550.127.08／CUDA 12.4 低於 NCCL CE 集體的門檻 12.5（`ce_coll.cc:179-180`；`bufferreg.rst:216`），且 2.29.7 的 tuner 只接受 v2–v5；兩項都待在計算節點用指令確認。
- NCCL v2.32.6 的 tuner 介面看不到也改不到 symmetric kernel／CE 候選的成本（`tuning.cc:211-236` 的讀法，未做實驗）。
