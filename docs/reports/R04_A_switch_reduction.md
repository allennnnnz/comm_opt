# R04-A　兩層交換器加總（機內 NVLS＋機間 InfiniBand SHARP）深入調查

- **對應規格書**：`docs/specs/S04_direction_deep_dive.md`（A 組）　**執行者**：Sonnet 5.5　**日期**：2026-11-16
- **範圍**：只提供有出處的事實；不評斷方向好壞。NCCL 原始碼為 v2.32.6-1（`/work/allen0117/comm_opt_env/src/nccl`，以下簡稱「源碼」），我們安裝的是 NCCL 2.29.7。
- **重要限制**：(1) 登入節點與計算節點硬體不同（見 §4 備註），登入節點結果不能代表計算節點；(2) arXiv MCP 後半段被 429 限流，部分論文改用 WebFetch 開 arXiv abs 頁核實；(3) NVIDIA 官方 SHARP 文件的 LLT（低延遲樹）訊息大小上限**沒有找到**。

---

## 1. 重點事實（附出處）

1. **NCCL 與 SHARP 的關係有兩條路，且分屬不同環境變數。** 機內 NVLS（NVLink SHARP）預設開（`NCCL_NVLS_ENABLE` 預設 2，`docs/userguide/source/env.rst:1965-1975`；`src/transport/nvls.cc:59`）；機間 IB SHARP 走 CollNet 外掛，`NCCL_COLLNET_ENABLE` **預設 0**（`env.rst:865-872`）。env.rst 原文：「NVLS and NVLSTree enable NVLink SHARP offload」（`env.rst:1506-1540` 的演算法表附近）。
2. **演算法與路徑**（源碼 `src/include/device.h:655` 註解 `nAlgos = 6; // TREE RING COLLNET_DIRECT COLLNET_CHAIN NVLS NVLS_TREE`）：
   - `NVLS`：機內用 NVLS（multicast／reduce 由 NVSwitch 做）；**機間只有在 CollNet（IB SHARP）可用時才有效**。`src/tuning/nvls.cc:149-153`：`if (tuning->algo == NCCL_ALGO_NVLS && !inputs->collNetSupport && inputs->comm->nNodes > 1) { tuning->valid = 0; ...}`；`src/tuning/pipeline_model/nvls.cc:93` 同。設置端：`src/graph/connect.cc:282` `if (comm->config.collnetEnable && channel->nvls.headRank != -1) channel->nvls.out = comm->nRanks;`（head rank 把機內歸約結果經 `nvls.out` 送給 CollNet）。裝置端：`src/device/all_reduce.h:390-523`（`work->oneNode` 為真走純機內；否則 scatter → NVLS reduce 後「send to network」→ 「Recv from network, broadcast」，`all_reduce.h:491-521`）。
   - `NVLS_TREE`：機內 NVLS＋機間用 tree（走一般網路 p2p，不需要 CollNet／SHARP）。`Demystifying NCCL`（arXiv 2507.04786）對 2.19 的描述：「NVLS continues the reduction via CollNet and SHARP-enabled switches, while NVLS Tree uses a tree-based fan-out」（`papers/2507.04786.pdf` pdftotext 約第 838–846 行）。
   - `COLLNET_DIRECT`／`COLLNET_CHAIN`：機內 GPU 之間用 NVLink／p2p 先匯到各 head GPU（Direct 為「all-to-all within the node」，Chain 為線性鏈），機間用 SHARP（同上文獻第 833–836 行）。上限：CollNetDirect 最多 `NCCL_MAX_DIRECT_ARITY + 1`＝8 個本地 GPU（`src/include/device.h:178`；`src/tuning/collnet.cc:~128-134` 註解 "CollNetDirect is only supported for up to 8 local GPUs"）。
3. **支援的 collective**（裝置端有實作的組合，grep `RunWorkColl<...>` 結果）：
   - AllReduce：`COLLNET_DIRECT`（`device/all_reduce.h:248`）、`NVLS`（390）、`NVLS_TREE`（527）、`COLLNET_CHAIN`（638）。
   - AllGather：`NVLS`（`device/all_gather.h:274`）、`COLLNET_DIRECT`（502）；ReduceScatter：`NVLS`（`device/reduce_scatter.h:233`）、`COLLNET_DIRECT`（444）。NVLS 的 AG／RS 多機分支使用 `nvls->out`（`all_gather.h:447-484`、`reduce_scatter.h:390-426`）。`NVLS_TREE` 只有 AllReduce 實作。
   - 非 AllReduce 時 NVLS 另有限制：`graphs[NVLS].nChannels > NCCL_MAX_NVLS_ARITY(32)` 或 `localRanks > 32` 則無效（`tuning/nvls.cc:140-148`）。
4. **啟用門檻（源碼）**：CollNet 需 `nNodes >= NCCL_COLLNET_NODE_THRESHOLD`（預設 2；`init.cc:1069`、`init.cc:1781-1785`）；任何 rank 的 (op,datatype) 組合只要有一個 head 不支援就整體不支援（`transport/coll_net.cc:1816-1843` `collNetSupportMatrix`）；外掛載入失敗時 `comm->config.collnetEnable = 0`（`coll_net.cc` fail 分支）。NVLS 需 `CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED`（`transport/nvls.cc:263-264`），且同一個 NVML 裝置不能有多個 rank（`nvls.cc:243-252`）。
5. **版本時間線**（來源：env.rst 的 "(since …)"；NCCL Release Notes PDF 2.19.3 版 `https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2193/pdf/NCCL-Release-Notes.pdf`，我用 pdftotext 讀）：
   - `NCCL_COLLNET_ENABLE` since 2.6；CollnetChain／CollnetDirect since 2.14；NVLS since 2.17；NVLSTree since 2.18（env.rst 演算法表）。
   - 2.17.1：「Add support for NVLink SHARP Reduction / Broadcast to accelerate intra-node…」；2.18.1：「Add inter-node algorithms for NVLink SHARP: NVLink SHARP + IB SHARP (NVLS), NVLink SHARP + Tree (NVLSTREE)」；2.19.3：「Added local user buffer registration for NVLink SHARP」。
   - 2.27.3：「Added NVLS+IB SHARP support for AllGather and ReduceScatter with user buffer registration」（R03 §2.8 表第 199 行引用 NCCL release notes；Opus 抽查建議：回 `docs.nvidia.com` 對 2.27.3 頁面再對一次——我用 WebFetch 打開 2.27.7 頁僅找到 symmetric multicast 一行，未取得 2.27.3 頁全文）。NVIDIA 部落格（2.27，2025-07-14）：「This new version brings SHARP support to AllGather (AG) and ReduceScatter (RS) collectives」、ring 實作「can consume 16 or more SMs」，NVLink＋IB SHARP 降到「6 or fewer」（宣稱，無原始數據）。
   - 2.29 起 NVLS 資源配置失敗不再靜默降級（env.rst:1980 附近；R03 第 201 行）。
   - **2.29.7 是否有上述功能**：我對安裝的 `libnccl.so.2`（`/work/allen0117/comm_opt_env/venvs/vllm/lib/python3.12/site-packages/nvidia/nccl/lib/`）做 `strings`，確認含 `NCCL_COLLNET_ENABLE`、`NCCL_COLLNET_NODE_THRESHOLD`、`NCCL_NVLS_ENABLE`、`NCCL_NVLS_NCHANNELS`、`NCCL_NVLS_CHUNKSIZE`、`NCCL_NVLSTREE_MAX_CHUNKSIZE`、`CollNetDirect` 等字串，且版本字串 `NCCL version 2.29.7+cuda12.9`。所以這些功能（均 ≤2.27）在 2.29.7 都存在。
   - **注意**：源碼 v2.32.6 的調校程式碼已重構成 `src/tuning/` 目錄（`tuning.cc`、`nvls.cc`、`collnet.cc`、`pipeline_model/`）；上面的 `檔案:行號` 屬 2.32.6，2.29.7 的行號與檔案配置**未驗證**（我沒有 2.29.7 原始碼）。
6. **HPC-X SHARP 外掛的介面版本與 NCCL 2.29 載入能力相容（版本層面）。** `nm -D --defined-only .../nccl_rdma_sharp_plugin/lib/libnccl-net.so` 輸出：`ncclNetPlugin_v5/v6/v7/v8` 與 `ncclCollNetPlugin_v5/v6/v7/v8`。安裝的 libnccl.so.2 含 `ncclNetPlugin_v6…v11`、`ncclCollNetPlugin_v6…v11`（`strings`）；源碼 `src/plugin/net.cc:42-46` 由新到舊嘗試 v12→v6（2.32.6）。因此 NCCL 會退到 v8（net 與 collnet 皆）載入。外掛版本檔：HPC-X v2.19（`hpcx/latest/VERSION`），`nccl_rdma_sharp_plugin 2.7 (4045cc8)`，`sharp 3.7.0 (8e5eca8)`，建置日期 2024-04-26（`sharp/share/doc/sharp/SHARP_VERSION`）。**版本相容不等於功能相容**：v8 外掛不支援 NCCL 較新的功能（例如 v9 以後的新增介面），實際是否能與 2.29.7 正常建立 CollNet 未實測。
7. **外掛本身的執行期需求**：`ldd`（登入節點）顯示外掛 NEEDED `libsharp_coll.so.5、libsharp.so.6、libalog.so.1、libsmx-3.7.so、libsharprdmacm-3.7.so、libucp/libuct/libucs、libmlx5、librdmacm、libibumad、libibverbs、libcudart.so.12`；其中 `libsmx-3.7.so`、`libsharprdmacm-3.7.so` 在未設 `LD_LIBRARY_PATH` 時「not found」（它們在 `hpcx/latest/sharp/lib/`）。外掛字串含可調環境變數：`NCCL_SHARP_DISABLE、NCCL_SHARP_MAX_COMMS、NCCL_SHARP_GROUP_SIZE_THRESH、NCCL_SHARP_V3_DATATYPES、NCCL_SHARP_DISABLE_ALLGATHER、NCCL_SHARP_DISABLE_REDUCE_SCATTER、SHARP_COLL_ENABLE_SAT、SHARP_COLL_LOCK_ON_COMM_INIT、SHARP_COLL_LOG_LEVEL、SHARP_COLL_NUM_COLL_GROUP_RESOURCE_ALLOC_THRESHOLD`（`strings libnccl-net.so.0.0.0`）；匯出的 `sharp_coll_do_allreduce_nb / allgather_nb / reduce_scatter_nb` 符號表示外掛介面層包含這三種 collective。
8. **SHARP 的服務前提（官方文件）**：
   - Aggregation Manager（AM）要跑在**專用伺服器、與 master Subnet Manager 同處、以 root 執行**：「Run Aggregation Manager using a "root user" as trusted entities.」「Make sure onboard Subnet Manager is disabled in the managed switches.」「Aggregation Manager is a central entity running on a dedicated server with a master Subnet Manager.」；若用 UFM：「When using Aggregation Manager from UFM, NVIDIA SHARP support has to be enabled in UFM.」（`https://networking-docs.nvidia.com/sharpum/390/setting-up-nvidia-sharp-environment`，Rev 3.9.0）。
   - 計算節點：「Enabled IPoIB interface in compute servers in order to enable using UD multicast for result distribution in SHARP.」（同頁）。
   - 外掛端前提（Rev 3.15 NCCL 頁，`https://networking-docs.nvidia.com/sharpum/3150/using-nvidia-sharp-with-nvidia-nccl`）：ConnectX-6 HDR 以上、Quantum HDR 交換器以上、GPUDirect RDMA 核心模組已載入、NCCL ≥ 2.7.3；設 `NCCL_COLLNET_ENABLE=1`；SHARP 端 `SHARP_COLL_LOCK_ON_COMM_INIT=1`、`SHARP_COLL_NUM_COLL_GROUP_RESOURCE_ALLOC_THRESHOLD=0`（「enabled by default with NCCL SHARP Plugin version >= 2.1.x」）；拓樸要求：「NCCL creates an aggregation streaming flow (NCCL Ring/Channel) per HCA rail」、各伺服器的同一個 HCA rail 要接到同一組 leaf 交換器。
9. **SHARP 資源上限（官方文件，版本不同數字不同，標版）**：Rev 3.13.0 `https://networking-docs.nvidia.com/sharpum/3130/general-information`：Quantum-2「Supports up to 1023 aggregation trees in the subnet (511 low latency trees, and 511 streaming aggregation trees)」，且「The limit is one active tree per port」（該句描述 streaming aggregation）。另一版本文件摘要為 126 棵（63 LLT＋63 SAT）（WebSearch 摘要，未開原頁，**僅列為線索**）。SAT 最大訊息 1073741792 bytes（1 GB 少 32 B）、`SHARP_COLL_ENABLE_SAT` 預設 0、`SHARP_COLL_SAT_THRESHOLD`（generic allreduce 轉用 SAT 的訊息大小）預設 16384（Rev 3.13 `.../3130/nvidia-sharp-collective-library`）。單一 job 配額旗標：`SHARP_COLL_JOB_QUOTA_OSTS`、`..._MAX_GROUPS`、`..._MAX_QPS_PER_PORT`（預設 0＝配置預設值）。完整清單用 `sharp_coll_dump_config`（本機 `hpcx/.../sharp/bin/` 內有）。**LLT 的訊息大小上限：沒有在官方頁面找到。**
10. **LLT 與 SAT 的定位（NVIDIA 簡報）**：Bureddy, MUG'20「SHARP: In-Network Scalable Streaming Hierarchical Aggregation and Reduction Protocol」（`papers/SHARP_MUG20_bureddy.pdf`，簡報頁尾印有 "NVIDIA CONFIDENTIAL. DO NOT DISTRIBUTE."，但為公開 MUG 網站檔案）：「EDR InfiniBand Switch-IB-2 switch introduced support for short message reductions. Referred to as Low Latency Transmission (LLT) SHARP」「HDR InfiniBand Quantum switch added support for long vector reduction. Referred to as Streaming Aggregation (SAT) SHARP」「Support for two simultaneous streaming operations per switch (limited resource)」（HDR，簡報 15 頁）。另有「Communications Latencies of 30-40us（onload）vs 3-4us（in-network computing）」（簡報 9 頁，廠商宣稱，無條件）。
11. **驗證 SHARP 是否生效的官方做法**：`ibdiagnet --sharp`（看 "IB Aggregation Nodes"、樹數 `grep -c TreeID /var/tmp/ibdiagnet2/ibdiagnet2.sharp`）、`sharp_hello -d mlx5_0:1 -v 3`（建立單一 SHARP job 並送 barrier，輸出「Test Passed.」；加 `SHARP_COLL_ENABLE_SAT=1` 可看到 `tree_type:SAT`）（`https://networking-docs.nvidia.com/sharpum/390/testing-nvidia-sharp-setup`）。其中 `ibdiagnet` 通常需要 root／管理端，使用者端可行的只有 `sharp_hello` 與 NCCL 日誌。
12. **登入節點觀察**（輕量唯讀，詳見 §4）：`/opt/mellanox/sharp/`（SHARP 3.5.1.MLNX20240219）存在且 `conf/` 為空；未找到 `/etc/sharp*`。這表示系統映像裝了 SHARP 使用者端套件，**不代表** AM 有跑或交換器有開。

---

## 2. 最接近的既有工作表

> 欄位：工作／出處／做了什麼／硬體與規模／與 A 的重疊／沒涵蓋的條件。R02 已列者用 R02 編號，只補與 A 有關的資訊。

### 2-1 SHARP 路線（IB 交換器內加總）

| 工作 | 出處 | 做了什麼 | 硬體與規模 | 與 A 重疊 | 沒涵蓋的條件 |
|---|---|---|---|---|---|
| **SHArP: A Hardware Architecture for Efficient Data Reduction**（Graham 等） | COM-HPC'16；DOI 10.1109/COMHPC.2016.006；摘要見 https://superfri.org/index.php/superfri/article/view/140 與 NVIDIA 預印本 https://network.nvidia.com/pdf/solutions/hpc/paperieee_copyright.pdf（WebSearch 結果，**我未開啟全文**） | SHARP 原始設計：交換器 ASIC 內聚合單元，大 radix 歸約樹，多個同時進行的 job | Switch-IB 2（EDR）；MPI | SHARP 協定基礎 | 早於 GPU／NCCL／LLM；未開全文，細節未核 |
| **SHARP Streaming-Aggregation Hardware Design and Evaluation**（Graham 等） | 2020，DOI 10.1007/978-3-030-50743-5_3（Springer，HPC 研討會論文集；**venue 名稱未確認**）；PMC 頁 https://pmc.ncbi.nlm.nih.gov/articles/PMC7295336/（開啟時為 reCAPTCHA 驗證頁，**我未讀到內文**） | HDR Quantum 上的 streaming aggregation（大向量），與 LLT 比較資料量交界（來自 WebSearch 摘要，未核原文） | Mellanox HDR（原文細節未核） | SAT 原理、LLT／SAT 分界 | 無 NDR／無 LLM／無 NCCL 內數據（未核） |
| **MUG'20 簡報（Bureddy）** | https://mug.mvapich.cse.ohio-state.edu/static/media/mug/presentations/20/bureddy-mug-20.pdf（廠商簡報，非同儕審查） | LLT／SAT 定義、NCCL＋SHARP 使用方式、「2.5X Higher Performance」圖（條件在圖內，我未取得） | HPC-X v2.6／2.7＋NCCL v2.6／2.7（簡報文字） | 只有 NCCL 機間 SHARP | 無 NVLS（2020 年尚無）、無 H100 |
| **HPCwire 2019**「Accelerating HPC and AI Workflows with Mellanox's SHARP」 | https://www.hpcwire.com/2019/05/06/accelerating-hpc-and-ai-workflows-with-mellanoxs-scalable-hierarchical-aggregation-and-reduction-protocol-sharp/ （WebSearch 摘要：宣稱 NCCL 效能「factor of two」；**廠商文章，未開全文**） | 廠商說法 | — | 廠商 | 條件不明 |
| **R2CC**（R02_T4 P8，arXiv 2512.25059） | NSDI'27 camera-ready | 容錯集體通訊 | 4×8 H100、IB 400G；**未啟用 SHARP** | 同規模 H100 IB 叢集 | 原文：「InfiniBand SHARP … R2CC does not currently handle it. Supporting SHARP … is left to future work.」（Appendix I，引自 R02_T4） |
| **Nezha**（R02_T4 P28，arXiv 2405.17870） | 無正式出處 | 多 rail 協調 SHARP／GLEX／TCP | V100、本地 8／128 節點；非 H100／NDR | TCP-SHARP 異質配置 | 基準為 Gloo；非 NCCL |
| **NCCL 2.27 部落格**（R02_T4 P15） | https://developer.nvidia.com/blog/enabling-fast-inference-and-resilient-training-with-nccl-2-27/ | NVLink SHARP＋IB SHARP 支援 AG／RS；SM 用量 ≥16 → ≤6 | 平台與 GPU 數未說明（我重讀確認：「It does not list specific GPU or switch models」） | **唯一明確談「兩層」的官方資料** | 廠商文章；圖表條件未載明；無延遲數字 |

### 2-2 可程式交換器／FPGA 路線（SwitchML 系列）

| 工作 | 出處 | 做了什麼 | 硬體與規模 | 與 A 重疊 | 沒涵蓋的條件 |
|---|---|---|---|---|---|
| **SwitchML**（Sapio 等，arXiv 1903.06701） | NSDI'21（https://www.usenix.org/conference/nsdi21/presentation/sapio ；venue 由 WebSearch 結果取得） | 以 P4 可程式交換器聚合梯度 | 可程式交換器（機型 abs 未載）；abs：訓練最多加速 5.5× | 同為「交換器內加總」 | 梯度同步（大訊息）、非 NCCL、非 GPU 機內 NVSwitch |
| **ATP**（Lao 等） | NSDI'21，https://www.usenix.org/conference/nsdi21/presentation/lao（我只讀 WebSearch 摘要，無 arXiv 版） | 多租戶、多機架的交換器內聚合；盡力而為地在最早的可程式交換器聚合 | P4 交換器；多 job 時吞吐 +38%～66% | 同上 | 梯度同步、PS 為終端 |
| **NetReduce**（arXiv 2009.09736） | 2020，無正式 venue（abs 頁） | RDMA 相容 FPGA 內歸約，不修改 NIC／交換器；訓練最多 1.7×（CV）／1.5×（NLP） | FPGA 原型；機架內，延伸到 spine-leaf（模擬） | 同上 | 梯度同步、RoCE |
| **Rina**（arXiv 2407.19721） | 2024，無正式 venue | Ring-AllReduce＋INA；機架內 worker 抽象為一個；比 ATP 吞吐 >50% | testbed＋模擬 | 同上 | 梯度同步、可程式交換器 |
| **Efficient Data-Plane Memory Scheduling for INA**（arXiv 2201.06398） | 2022 | INA 的資料平面記憶體排程 | abs 未載 | 同上 | 未深入 |
| **EPIC**（arXiv 2605.18683） | 2026，v2 2026-07 | Ethernet 上的 INC 協定規格＋參考系統；Tofino testbed、FPGA/RTL 驗證 | Tofino、FPGA | 交換器內集體通訊（Ethernet 路線） | 非 IB、非 NVLink |

### 2-3 NVLS／NVSwitch 內運算（只補與 A 有關者；R02_T1 已列的不重抄）

| 工作 | 出處 | 與 A 相關的資訊 | 沒涵蓋的條件 |
|---|---|---|---|
| **Demystifying NCCL**（R02_T1 P?，arXiv 2507.04786） | 2025 | 描述 2.19 的 CollNet／NVLS 機制，但**明確排除**：「this paper will not include analyses of NVLS and CollNet, as their implementations rely heavily on specific hardware (NVSwitch and SHARP-enabled networks), making them less representative」（`papers/2507.04786.pdf`；R02_T1 亦引用） | 無 NVLS／CollNet 數據 |
| **Every Microsecond Matters**（R02_T1 P3，arXiv 2607.16100） | 2026 | GB200 NVL72／B200；用 NVLS multicast，vLLM 實驗含「No MC」關閉設定；本文 §IV–V 提到 NVLink SHARP（pdftotext 第 284、301、363、394 行有 "SHARP"、"NVLS Multicast"）；全文中**未出現 "IB SHARP"／"CollNet"**（我 grep 無結果） | GB200；沒有 IB SHARP；多機 vLLM 8 GPU＝2 節點×4 GPU |
| **GPU-Initiated Networking for NCCL**（R02_T1 P13，arXiv 2511.15076） | 2025 | NCCL 2.28 Device API 的 Multimem（NVLink SHARP）模式；EOS DGX H100 2／4／8 節點；**實驗未使用 NVLS**（R02_T1） | 無 NVLS 與 IB SHARP 結合的量測 |
| **TokenWeave**（P12，arXiv 2505.11329）、**MSCCL++**（P11，arXiv 2504.09014）、**SCIN**（P8，arXiv 2603.28239） | 見 R02_T1 | 均為單機 NVLS（SCIN 為交換器內硬體提案，以模擬的 NVLS 當 baseline，誤差 5.5%） | 單機；無 IB SHARP |
| **CAIS: Towards Compute-Aware In-Switch Computing for LLMs Tensor-Parallelism on Multi-GPU Systems**（Zhang 等，arXiv 2605.05628；R02 **未列**） | 2026-05-07；標題頁註記「HPCA 2026 Submission #847」（非正式出版） | 指出 NVLS 的通訊模式（AllGather 用 `multimem.st` push、ReduceScatter 用 `multimem.ld_reduce` pull）與 GEMM 記憶體語意不匹配，造成全域 barrier；提議 ISA／交換器微架構延伸。評估為**週期級模擬**（Accel-Sim＋BookSim2，模擬 8-GPU、4 交換器、DGX-H100；NVLS 模擬以 NCCL 實測驗證 1–16 GB AllReduce）；相對 NVLS 方案平均 1.38× 端到端訓練加速（abstract） | 單機模擬；非實機；無 IB |
| **DySHARP: Accelerating MoE with Dynamic In-Switch Computing on Multi-GPUs**（Zhang 等，arXiv 2605.05607；R02 **未列**） | 2026-05-07 | 稱 NVLS「supports only static collectives with regular patterns」，提議動態 multimem addressing；模擬 GH200 NVL32（內文第 45、122 行）；相對 SOTA 最高 1.79×（abs 摘要） | 模擬；MoE；單一 NVLink domain |
| **NVIDIA 部落格 MultiShot**（"3x Faster AllReduce with NVSwitch and TensorRT-LLM MultiShot"） | https://developer.nvidia.com/blog/3x-faster-allreduce-with-nvswitch-and-tensorrt-llm-multishot/ | NVSwitch multicast 把 ring 的 2N−2 步降到 2 步，「reduces the O(N) latency of Ring AllReduce by up to 3x」；文中**無絕對延遲數字、未提 NVLS／multimem 字樣**（WebFetch 確認） | 無 IB；廠商文章 |

### 2-4 「兩層都用交換器加總」的資料搜尋結果

- 直接量測（NVLS＋IB SHARP 同時啟用）的論文或技術報告：**本次搜尋未找到。** 找到的只有：NCCL 2.18.1 release note（2.18 新增演算法 NVLS＝NVLink SHARP＋IB SHARP）、2.27.3 對 AG／RS 的擴充、NVIDIA 2.27 部落格（無數據）、Demystifying NCCL 的機制描述（無數據）。
- NCCL 2.18.1 release notes 的 Known Issues（pdftotext 約第 336–343 行）：「Running allreduce on H100 platforms with IB SHARP with multiple GPUs per process can result in data corruption if all GPUs on the node are not part of the same process」；2.18.3 修正「data corruption with IB SHARP on H100 platforms when combining multiple…」（原句在 PDF 換行處中斷）與「hang with IB SHARP and bfloat16 on systems with less than one NIC per GPU」。
- SHARP 用於推論或小訊息延遲：除 MUG'20 簡報的「LLT：Latency optimized」定位與「3–4 µs vs 30–40 µs」宣稱（無條件）外，**沒有找到**以 LLM 推論為情境的 SHARP 量測。

---

## 3. 現成軟體的支援程度

**NCCL（安裝 2.29.7；源碼 2.32.6）**
- 機內 NVLS：2.17 起；預設開；`NCCL_NVLS_ENABLE=0` 可關（`env.rst:1965-1985`）；NVLS 通道數可用 `NCCL_NVLS_NCHANNELS`（`init.cc:68`、`init.cc:2596`）、chunk 用 `NCCL_NVLS_CHUNKSIZE`（預設 128 KiB，`transport/nvls.cc:60`）。安裝的 2.29.7 已含這些參數（strings）。
- 機間 IB SHARP：走 CollNet 外掛；預設關；需外掛。NCCL 內建沒有 SHARP 實作，`ncclNetPluginLoad` 以 `NCCL_NET_PLUGIN`（或預設 `libnccl-net.so`，`src/plugin/net.cc:293-297`）找外掛，再逐版嘗試 `getNcclCollNet[i]`（`net.cc:~120-130`）。
- NVLS×CollNet 組合的條件已在 §1 第 2 點列出。源碼 `graph/connect.cc:279` 有一行註解「`NVLS+SHARP not yet implemented.`」緊接著下方有「若 collnetEnable 就設 `nvls.out`」的程式——**註解與程式碼看似矛盾**（見 §6）。
- 對稱記憶體 symmetric kernels 與 `NCCL_ALGO` 篩選：`NCCL_ALGO` 格式見 `env.rst:1506-1600`；`NCCL_ALGO=allreduce:nvls` 之類寫法依 2.24 起語法，**在 2.29.7 是否完全同 2.32.6 的解析，未驗證**。

**HPC-X（`nvhpc-hpcx-cuda12/24.7`，HPC-X v2.19）**
- SHARP 3.7.0（含 `sharp_am`、`sharp_hello`、`sharp_cmd`、`sharp_coll_dump_config`、`libsharp_coll.so.5.0.1`、`libsharp.so.6.1.0`）；`nccl_rdma_sharp_plugin` 2.7（匯出 v5–v8）；UCC 1.4.0、UCX 1.17.0、HCOLL 4.8.3227。
- HPC-X 的 `VERSION` 內記 `OFED: mlnx_ofed`、`CUDA: V12.4.99`、`Linux: redhat7`（編譯環境），與 Nano5 的 RHEL8／驅動 550 不同——外掛與本機 libibverbs／libucx 的相容性未驗證（外掛 `ldd` 顯示連到系統 `/lib64/libucp.so.0`、`/lib64/libmlx5.so.1` 與 `/opt/mellanox/sharp/lib/*`）。
- 本機 `sharp/share/doc/sharp/README` 與 `CHANGELOG.md` 版號標 2.1.0／2.5.0，與套件 3.7.0 不一致（文件未更新）；CHANGELOG 的 "NDR support" 小節（commit 列「Support MTU 4K」「Aggregation trees prioritization according to aggregation rate」「Build max trees supported by network devices」）位於標 v2.5.0 區段內，所以**該 CHANGELOG 不可作為 3.7.0 對 NDR 的支援依據**。

**vLLM 0.32.0／SGLang／TensorRT-LLM／Megatron**（R03）：都沒有特別為 IB SHARP 設定。
- vLLM 與 SGLang 內的 multimem／NVLS 路徑僅限機內（torch `symm_mem`，`multimem_all_reduce_` 限 sum、需 multicast，`pytorch/.../CUDASymmetricMemoryOps.cu:186,194`；R03）。
- SGLang 有 `--enable-nccl-nvls`（預設 False，`exec_.py:715-717`，R03）。
- Megatron 推論路徑 `inference_moe_token_dispatcher_type` 預設 `'nvls'`（`transformer_config.py:1436-1441`，R03）；訓練路徑走 `torch.distributed`（NCCL）。
- 這些系統跨機時都回到 NCCL（R03），因此是否使用 NVLS＋IB SHARP 取決於 NCCL 的選擇與 `NCCL_COLLNET_ENABLE` 是否為 1。DeepEP 在設定中設 `NVSHMEM_DISABLE_NVLS=1`（註解 "Disable NVLink SHArP"，`deepep/.../buffer.py:97-112`，R03）。

---

## 4. 在 Nano5 的前提條件

| 前提 | 為什麼需要 | 怎麼檢查 | 登入節點檢查結果／需在計算節點檢查 |
|---|---|---|---|
| 驅動／CUDA 支援 multicast（NVLS） | `nvlsSupport` 取自 `CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED`（`transport/nvls.cc:263-264`） | 計算節點上跑 `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT` 的小 all_reduce，找 `NVLS multicast support is available on dev …` | 登入節點 `nvidia-smi`：driver `550.127.08`（`/proc/driver/nvidia/version`：「NVRM version: NVIDIA UNIX x86_64 Kernel Module 550.127.08 Fri Oct 25 22:14:46 UTC 2024」）。**需在計算節點確認**（M20）。 |
| NVSwitch 與 Fabric Manager 正常（SXM 機型） | NVLS 需要 NVSwitch 的 multicast | 計算節點 `nvidia-smi topo -m`、`nvidia-smi -q -i 0` 的 Fabric 區塊、NCCL INIT 日誌 | **登入節點不具代表性**：`nvidia-smi -L` 顯示 `2× NVIDIA H100 PCIe`，`topo -m` 只有 GPU0/GPU1 與 NIC0–NIC3、無 NVLink；`Fabric State: N/A`；存在 `/dev/nvidia-nvswitchctl`（意義未驗證）。需在計算節點確認。 |
| 每顆 GPU 單獨一個 rank（不可多 rank 共用 GPU） | `NCCL_NVLS_ENABLE=1` 時多 rank 同 NVML 裝置回 `ncclInvalidUsage`（`nvls.cc:243-252`） | 啟動方式檢查（8 GPU／8 rank） | 不需檢查登入節點。 |
| 外掛介面版本相容 | NCCL 2.29.7 載入 net／collnet v6–v11；HPC-X 外掛匯出 v5–v8 | `nm -D --defined-only .../libnccl-net.so \| grep -i -E "net\|collnet"`；`strings libnccl.so.2 \| grep -E "ncclNetPlugin_v\|ncclCollNetPlugin_v"` | 已做（見 §1 第 6 點）。輸出原文：`D ncclCollNetPlugin_v5/v6/v7/v8`、`D ncclNetPlugin_v5/v6/v7/v8`；libnccl 含 v6–v11。**實際載入結果需計算節點日誌**（`NCCL_DEBUG=INFO` 應出現 `Successfully loaded external network plugin` 或 `External network plugin … is unsupported`，字串見 `src/plugin/net.cc`）。 |
| 外掛相依函式庫可解析 | `libsmx-3.7.so`、`libsharprdmacm-3.7.so` 在未設 `LD_LIBRARY_PATH` 時 not found | 計算節點：`module load nvhpc-hpcx-cuda12/24.7` 後 `ldd .../libnccl-net.so \| grep "not found"`（設 `LD_LIBRARY_PATH` 含 `hpcx/latest/sharp/lib`、`ucx/lib`） | 登入節點 `ldd` 輸出含：`libsmx-3.7.so => not found`、`libsharprdmacm-3.7.so => not found`；其餘 `libsharp_coll.so.5 => /opt/mellanox/sharp/lib/libsharp_coll.so.5` 指向系統版（3.5.1）。**兩版本混用風險**：需確保 LD_LIBRARY_PATH 優先找 HPC-X 3.7.0（未驗證）。 |
| 交換器端有 SHARP Aggregation Manager 在跑，且已建立聚合樹（含 Quantum-2 的 LLT 樹） | 沒有 AM 則 `sharp_coll_init` 失敗，CollNet 會被關掉（`coll_net.cc` fail 分支 `collnetEnable = 0`） | 計算節點：(a) `SHARP_COLL_LOG_LEVEL=3 sharp_hello -d mlx5_0:1 -v 3`（期待「Test Passed.」）；(b) `NCCL_COLLNET_ENABLE=1 NCCL_DEBUG=INFO` 後找 `CollNet` 相關訊息（`Rank %d: %d CollNet devices` 於源碼 `init.cc:~1557`，2.29.7 字串未驗證）；(c) 管理端 `ibdiagnet --sharp`（通常需 root，**向 NCHC 詢問**） | 登入節點：`/opt/mellanox/sharp/{bin,conf,include,lib,sbin,share,systemd}` 存在；`ls /opt/mellanox/sharp/conf` 為空；`cat /opt/mellanox/sharp/share/doc/sharp/SHARP_VERSION`：`PACKAGE VERSION: 3.5.1.MLNX20240219.7fcef5af`；`ls /etc/sharp*` 與 `/var/log/sharp*`：不存在（No such file）。`ps aux \| grep -i sharp` 無輸出（登入節點本來就不會跑 AM，無法推論）。**AM 是否運作：未知，需在計算節點執行 sharp_hello 才能判斷（我未在登入節點執行，因為它會向 AM 建立 job）。** |
| 計算節點 IB 為 NDR／ConnectX-7、韌體與 Quantum-2 交換器 | SAT／LLT 在 Quantum-2 的樹數、兩層拓樸的 rail 對應 | 計算節點 `ibstat`、`ibv_devinfo`、`ibdiagnet`（管理端） | 登入節點（**非計算節點**）`ibstat`：`CA 'mlx5_0'…'mlx5_3'`（4 張）、`CA type: MT4129`（ConnectX-7）、`Firmware version: 28.40.1000`、`Rate: 200／100`（混合）、`link_layer: InfiniBand`。計算節點（每台 8 張 NDR 400）需在 M20 確認。 |
| 拓樸：每台同一 rail 的 HCA 接到同一組 leaf 交換器（rail-optimized） | SHARP 為每個 HCA rail 建一條 aggregation streaming flow；跨 rail 接線會使聚合樹不通 | 管理端／NCHC 提供拓樸；或計算節點上 `ibnetdiscover`（可能需權限）、NCCL `NCCL_DEBUG_SUBSYS=GRAPH` 輸出 | 需詢問 NCHC，**未知**。 |
| GPUDirect RDMA（`nvidia_peermem`／dmabuf） | SHARP 外掛說明頁要求「verify that the GPUDirect RDMA kernel module is properly loaded」；SHARP 與 GPU buffer 搭配 | 計算節點 `lsmod \| grep -E "nvidia_peermem\|nv_peer_mem"`、`NCCL_DEBUG=INFO` 看 `GDR` | 登入節點 lsmod 未做（此項屬 BE 組）；**需在計算節點檢查（M20）**。 |
| 計算節點的 IPoIB（SHARP 以 UD multicast 分送結果） | 官方 Rev 3.9.0 設定頁要求 | 計算節點 `ip addr show \| grep ib` | **需在計算節點檢查（M20）。** |
| 使用者在無 root 下可設定的 SHARP／NCCL 環境變數 | 全部是進程環境變數，不需 root | 在 sbatch 腳本內 export：`NCCL_COLLNET_ENABLE=1`、`NCCL_NET_PLUGIN`（或把外掛目錄加入 `LD_LIBRARY_PATH`）、`SHARP_COLL_ENABLE_SAT`（SAT 預設 0）、`SHARP_COLL_LOG_LEVEL`、`NCCL_ALGO` 等 | 不需檢查。AM 的配額（`SHARP_COLL_JOB_QUOTA_*`）是否被 NCHC 設限：未知。 |
| 官方文件的最低需求：ConnectX-6 HDR／Quantum HDR 以上、NCCL ≥ 2.7.3 | 官方 Rev 3.15 NCCL 頁 | — | 已滿足（版本層面）。 |

---

## 5. 尚未有人量過或作者自述未解的問題

- **兩層同時啟用（NVLS＋IB SHARP）的實機量測：本次搜尋未找到任何論文**（R02_T4 §1 亦同：「本次搜尋沒有找到 2024 年以後以 SHARP 在 LLM 訓練上做獨立量測的論文」）。
- R2CC：「Supporting SHARP … is left to future work」（Appendix I，引自 R02_T4 P8）。
- Demystifying NCCL：明言排除分析 NVLS 與 CollNet——「this paper will not include analyses of NVLS and CollNet, as their implementations rely heavily on specific hardware」（arXiv 2507.04786，pdftotext 第 846 行附近）。
- Nezha（arXiv 2405.17870）結論：「In the future, we plan to evaluate Nezha's performance on more advanced large-scale clusters and assess its capabilities for other collective communication operations.」（引自 R02_T4 P28）。
- NVLS 的 AG／RS 與 GEMM 語意不匹配（CAIS，arXiv 2605.05628 §I）；NVLS 只支援靜態規則集體（DySHARP，arXiv 2605.05607 §I）——兩篇皆為模擬、單一 NVLink domain。
- 本次未找到：SHARP 低延遲樹（LLT）在 NDR 上的**官方訊息大小上限**、NVLS＋IB SHARP 小訊息延遲的任何公開數字、SHARP 用於 LLM 推論的任何量測。
- Every Microsecond Matters（arXiv 2607.16100）自述：「kernel selection is currently driven by empirical measurements」（R02_T1 P3 引 §IX）。

---

## 6. 矛盾或不確定之處

1. **源碼註解與程式碼矛盾**：`src/graph/connect.cc:279` 註解 `// NVLS+SHARP not yet implemented.`，下一段 `connect.cc:282` 卻在 `collnetEnable` 時設 `nvls.out`；device 端 `all_reduce.h:491-521` 也有完整 network send／recv 分支；NCCL 2.18.1 release note 說 NVLS＝NVLink SHARP＋IB SHARP。**註解可能是過時殘留**，未裁決。
2. **SHARP 樹數上限不一致**：Rev 3.13.0 說 Quantum-2 最多 1023 棵（511 LLT＋511 SAT）；WebSearch 摘要另有版本寫 126 棵（63＋63）。是否為不同版本／不同世代，未核。「one active tree per port」那句的適用範圍（僅 SAT 或全部）以官方頁字面為「Multiple SAT operations can run in parallel on a Quantum-2 switch, with the limit being one active tree per port.」，我只在 WebFetch 摘要中看到。
3. **MUG'20 簡報稱 HDR Quantum「Support for two simultaneous streaming operations per switch (limited resource)」**，與 Rev 3.13 的「one active tree per port」（Quantum-2）及 511 棵 SAT 樹之間的關係不清（世代與計數方式可能不同）。
4. **NCCL 外掛載入結果未實測**：介面 v8 在 NCCL 2.29.7 應被接受，但 HPC-X 外掛 2.7 是 2024-04 建置、NCCL 2.29.7 是 2025 後版本；外掛未提供 v9 以後的新增介面（如 NCCL 較新的 net 功能），實際行為未知。
5. **登入節點有 GPU（2× H100 PCIe）、NIC 速率 200／100，與計算節點（8× H100／H200 SXM、8×NDR）不同**；本報告所有登入節點結果僅供參考。
6. **ibstat 的 `Rate: 200／100`** 是否表示登入節點接的是 HDR／EDR 或降速，未驗證。
7. **R02_T1 對 NVLS 開關 A/B 缺口的描述**（「沒有任何一篇報告」）本次搜尋未推翻，但 TokenWeave 與 Every-µs 含 "vLLM-Multimem／No MC" 類比較（R02_T1 已列），細節在 R02。
8. **SHARP 論文全文未核**：SHArP（COM-HPC'16）與 Streaming-Aggregation（2020）僅有 WebSearch 摘要；PMC 頁面受 reCAPTCHA 擋住，內文沒讀到。其硬體與數據欄位標為「未核」。
9. **NVIDIA 官方 NCCL release notes 2.27.3 頁**我沒有打開成功（WebFetch 2.27.7 頁只找到 symmetric multicast 一行；index 頁無內容）；2.27.3 的「NVLS+IB SHARP AG／RS」一行引自 R03 §2.8 表，未獨立核實。

---

## 7. 搜尋紀錄

arXiv MCP（均加欄位前綴、`sort_by: relevance`；約 12:00 起被 429 限流，之後多改用 WebFetch 開 arxiv abs 頁）：
1. `abs:"in-network aggregation" AND abs:"SHARP"`：1 筆，無關（排除）。
2. `abs:"in-network aggregation" AND (abs:"distributed training" OR abs:"all-reduce")`：3 筆（HomoComp 2402.07529、ESA 2201.06398、THC 2302.08545）；採用 2201.06398，其餘為壓縮類（排除）。
3. `abs:"NVLink SHARP" OR abs:"NVLS" OR abs:"multimem"`：12 筆（看到 CAIS 2605.05628、DySHARP 2605.05607、SCIN 2603.28239、GIN 2511.15076、MSCCL++、TokenWeave 等；其餘如 MultiMem 2606.22220、perovskite 2404.14231 為雜訊，排除）。
4. `ti:SwitchML OR ti:"Scaling Distributed Machine Learning…" OR ti:"NetReduce" OR ti:"ATP…" OR ti:"Rina"`：4 筆；採 SwitchML、NetReduce、Rina；ATP 無 arXiv 版；RINAS 2312.02368 無關（排除）。
5. 之後 4 次 arXiv 搜尋（`abs:"SHARP" AND abs:"InfiniBand"…`、`ti:"Demystifying NCCL"…`、`abs:"in-network" AND abs:"InfiniBand" AND abs:"allreduce"…`）全被 429，未得結果。
- `mcp__arxiv__citation_graph`：**未使用**（rate limit 與時間；SCIN／CAIS 的後續引用未查）。
paper-search MCP：`search_semantic`（SHARP 兩次）、`search_dblp`（兩次）、`search_google_scholar`（一次）皆回傳空陣列。
WebSearch（standard 為主）：
- `NVIDIA SHARP Quantum-2 NDR aggregation manager requirements NCCL SHARP plugin …`：取得官方頁 URL 清單（SHARP Rev 2.6.1–3.15）。
- `NVIDIA SHARP user manual "low latency" "streaming aggregation" message size limit …`：取得 SAT 上限與 `SHARP_COLL_SAT_THRESHOLD`。
- `Graham "Scalable Hierarchical Aggregation Protocol (SHARP)" COM-HPC 2016 …`、`…Streaming-Aggregation Hardware Design`：取得 DOI 與摘要。
- `NCCL SHARP CollNet NVLS multi-node allreduce performance H100 InfiniBand NDR NVLSTree vs CollNet measurement`：結果多為部落格（spheron、gpusmith 等，**未列為出處**）；取得 2.18.1 release note 線索。
- `SHARP in-network reduction deep learning training evaluation …`（extended）：HPCwire 2019、MUG'20、NetReduce、EPIC。
- `NVLink SHARP NVLS allreduce latency small message H100 measurement paper …`：取得 NVIDIA MultiShot 部落格、MSCCL++ 文件；DLRM／NVSHMEM 的 2512.05831 與 A 無關（排除）。
- `ATP: In-network Aggregation for Multi-tenant Learning …`：USENIX 頁。
WebFetch：NCCL 官方 release notes index 與 2.27.7 頁（無 SHARP 內容）；SHARP 官方文件 Rev 3.9.0（setup、testing）、3.13.0（general-information、collective library）、3.15（NCCL 頁）；NVIDIA 2.27 部落格；MultiShot 部落格；arXiv abs 頁（2605.05607、2604.00317、2605.18683、1903.06701、2009.09736、2407.19721）。排除：2604.00317（NIMBLE，已在 R02_T4，與 A 關聯低）。
被擋或失敗：PMC7295336（reCAPTCHA）、2.27 部落格第一次逾時（第二次成功）、`docs.nvidia.com/networking/display/sharpv370/...` 重新導向後的 `networking-docs.nvidia.com/sharpum/370/...` 回 404。
下載到 `papers/`（已確認大小非 0）：`SHARP_MUG20_bureddy.pdf`（1,671,353 B）、`2605.05628.pdf`（3,356,423 B）、`2603.28239.pdf`（2,784,819 B）、`2605.05607.pdf`（4,480,238 B）；既有 `2507.04786.pdf`、`2607.16100.pdf`、`2511.15076.pdf` 用於 grep。
登入節點檢查（皆唯讀、指定路徑）：`ls`／`nm -D`／`readelf -d`／`ldd`（外掛）／`strings`（外掛與 libnccl.so.2）／`cat` SHARP_VERSION／`ibstat`／`ibv_devinfo`／`nvidia-smi -L／-q／topo -m`／`ps aux | grep sharp`；未執行 `sharp_hello`、未跑任何 NCCL 或 CUDA 程式。

---

## 8. 引用清單（Opus 之後用 arXiv MCP 或 DOI 匯出 BibTeX）

arXiv：
- 2507.04786（Demystifying NCCL）
- 2607.16100（Every Microsecond Matters）
- 2511.15076（GPU-Initiated Networking for NCCL）
- 2505.11329（TokenWeave）
- 2504.09014（MSCCL++）
- 2603.28239（SCIN）
- 2605.05628（CAIS）
- 2605.05607（DySHARP）
- 1903.06701（SwitchML）
- 2009.09736（NetReduce）
- 2407.19721（Rina）
- 2201.06398（Efficient Data-Plane Memory Scheduling for INA）
- 2605.18683（EPIC）
- 2512.25059（R2CC，R02_T4）
- 2405.17870（Nezha，R02_T4）

DOI／正式出處：
- SHArP：10.1109/COMHPC.2016.006
- SHARP Streaming-Aggregation：10.1007/978-3-030-50743-5_3
- ATP：NSDI'21，https://www.usenix.org/conference/nsdi21/presentation/lao
- SwitchML：NSDI'21，https://www.usenix.org/conference/nsdi21/presentation/sapio

非同儕審查／文件（不進 BibTeX，只作事實出處）：
- NVIDIA NCCL 2.27 部落格 https://developer.nvidia.com/blog/enabling-fast-inference-and-resilient-training-with-nccl-2-27/
- NVIDIA MultiShot 部落格 https://developer.nvidia.com/blog/3x-faster-allreduce-with-nvswitch-and-tensorrt-llm-multishot/
- NCCL 2.19.3 Release Notes PDF https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2193/pdf/NCCL-Release-Notes.pdf
- SHARP 官方文件 Rev 3.9.0／3.13.0／3.15（`networking-docs.nvidia.com/sharpum/390|3130|3150/…`）
- MUG'20 簡報 https://mug.mvapich.cse.ohio-state.edu/static/media/mug/presentations/20/bureddy-mug-20.pdf
- HPCwire 2019 https://www.hpcwire.com/2019/05/06/accelerating-hpc-and-ai-workflows-with-mellanoxs-scalable-hierarchical-aggregation-and-reduction-protocol-sharp/

本地檔案出處：
- NCCL 源碼：`/work/allen0117/comm_opt_env/src/nccl`（v2.32.6-1）
- HPC-X：`/work/HPC_software/LMOD/nvidia/packages/hpc_sdk-24.7/Linux_x86_64/24.7/comm_libs/12.5/hpcx/latest/`
- 安裝的 NCCL：`/work/allen0117/comm_opt_env/venvs/vllm/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2`
