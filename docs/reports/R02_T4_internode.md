# R02 / T4：跨機器通訊（internode）文獻地圖

- **執行者**：Sonnet 5.5　**日期**：2026-11-16　**規格**：`docs/specs/S02_lit_map.md`（D051）
- **範圍**：IB／RoCE 集體通訊、階層式演算法、網路內運算（SHARP）、拓樸感知合成、multi-rail、GPUDirect RDMA、DP／FSDP／ZeRO 通訊、梯度壓縮、壅塞與落後節點、容錯。
- **核實方式**：每篇都開過 arXiv abs 頁（WebFetch）或 USENIX／PPoPP／ACM 會議頁；「作者自述限制」與「硬體」欄另以 `curl` 下載 PDF 後用 `pdftotext` 讀全文（暫存於 scratchpad，未放入 `papers/`），引文為 PDF 文字原句（雙欄排版，章節號以 PDF 可見標題為準，看不到章節號者寫「位置」）。
- **標記**：「未說明」＝原文沒有；「摘要」＝只讀了摘要頁、沒讀全文，該欄資訊僅限摘要；不下任何「該不該做」的判斷。
- **共 29 項**：28 篇論文／預印本（含 1 篇 survey）＋ 1 篇廠商技術部落格（非同儕審查）。

---

## 1. 主題概況

近年跨機器通訊的工作大致分成幾條線：(a) **排程／演算法合成**，以 SMT／MILP／生成樹為每個拓樸產生集體通訊排程（SyCCL、HeteCCL、ForestColl），多與 NCCL、TACCL、TE-CCL 比較；(b) **函式庫與參數調校**，如 AutoCCL 自動調 NCCL 參數、PCCL（Big Send-off）在 Frontier／Perlmutter 上以階層式與演算法自動選擇取代 RCCL／NCCL、NCCLX（Meta）在 10 萬 GPU 級叢集的 CTran 傳輸層；(c) **multi-rail、壅塞與容錯**，如 NIMBLE（多路徑平衡）、R2CC／OptCC（NIC 失效後的 AllReduce）、C4／CCL-D／MegaScale（落後節點與 hang 診斷）；(d) **訓練端通訊量削減**，包含 ZeRO／FSDP 變體（AsyncHZP、FCDP、Frontier 的 ZeRO++ 研究）、梯度壓縮（TAGC、EDGC、PacTrain、LoCo）與低通訊訓練（DiLoCo、DeToNATION）；(e) **量測與剖析**（Demystifying NCCL、NCCL 剖析海報）。
網路內運算（SHARP）方面，本次搜尋**沒有找到** 2024 年以後以 SHARP 在 LLM 訓練上做獨立量測的論文；僅有 NVIDIA 技術部落格（NCCL 2.27）與 survey 的敘述，以及 R2CC 明講其測試機未啟用 SHARP。

---

## 2. 論文表（依年份新到舊）

編號用 P1…P29，後文引用。「重現」判斷依據：我們最多 4 台 H100（32 GPU，每台 8×NDR400 IB）／2 台 H200，驅動 550，無 root。

| # | 論文／出處 | 問題 | 方法要點 | 硬體與規模 | Baseline | 報告的效益（附條件） | 作者自述限制／未來工作（引文與位置） | 程式碼 | 在我們環境能否重現 | 出處網址 |
|---|---|---|---|---|---|---|---|---|---|---|
| P1 | **Collective Communication for Distributed LLM Systems: Planning, Runtime Adaptation, and Computation Coordination**（Song 等；arXiv 2026-08-15；IEEE Network 風格 survey，無正式出處欄） | 整理 LLM 系統中 AR／RS／AG／A2A 的規劃、執行、計算協同 | tutorial 式 taxonomy，三層：排程規劃、執行與適應、計算協同 | 不適用（survey） | 不適用 | 不適用 | 摘要稱「discuss open challenges and future directions」；未逐條整理（只讀摘要＋grep） | 未說明 | 不適用 | https://arxiv.org/abs/2608.15118 |
| P2 | **Don't Let a Few Network Failures Slow the Entire AllReduce**（Chen 等；arXiv 2026-06-01；無正式出處） | NIC 失效後 NCCL 改道，退化的伺服器仍在 ring 關鍵路徑上拖慢整個 AllReduce | 推導非對稱頻寬下 AllReduce 完成時間下界；OptCC：四階段 pipelined AllReduce | **SimAI 模擬**，A100 伺服器每台 8 GPU＋8×ConnectX-6 200G HDR IB；未做 H100 實機 | NCCL 無故障 ring、ICCL（NCCL ring＋降頻寬拓樸）、R2CCL | 頻寬損失最多 50% 時，OptCC 在 NCCL 無故障 ring 的 2–6% 內；現有容錯方案最多 57% 額外開銷（模擬） | 「Extending the bounds to latency-sensitive settings and dynamic bandwidth fluctuations is left to future work.」（結論段，PDF 約 line 636–638） | 未說明 | 部分：演算法可在 2–4 台機器實作，但原文為模擬、需自行實作並製造 NIC 降速 | https://arxiv.org/abs/2606.01680 |
| P3 | **HeteCCL: Synthesizing Near-Optimal Collective Communication Schedules for Heterogeneous GPU Clusters**（Hei 等；NSDI '26，pp. 2533–2551，2026-05） | 異質 GPU 叢集上排程效率低、合成慢 | 以 SMT 編碼頻寬／路由限制，CEGIS 剪枝；把問題寫成加權有向圖上的 maximum parallel transfer | 異質測試床，各 32 GPU，混用 H20 與 V100；H20 伺服器 4×400G NIC | NCCL、TACCL、TE-CCL | 頻寬最高 2.8× NCCL、4.4× TACCL、2.6× TE-CCL（對應順序依摘要列序推得）；合成快達兩個數量級；端到端訓練效率 +23%–37%（摘要未給 workload 細節） | 「applying SMT solving directly to datacenter-scale clusters (thousands of GPUs) is computationally intractable」，以階層組合處理；直接合成評估至 64 GPU（§7 Discussions and Limitations） | 未說明 | 部分：同為 NVSwitch＋IB 機器，但需異質 GPU 才是主場景；同質 H100 退化為一般合成 | https://www.usenix.org/conference/nsdi26/presentation/hei |
| P4 | **ForestColl: Throughput-Optimal Collective Communications on Heterogeneous Network Fabrics**（Zhao 等；NSDI '26，pp. 2067–2093，2026-05） | 異質網路下集體通訊排程難以達到吞吐量最優 | 建 spanning tree 做 broadcast／aggregation，多項式時間產生吞吐量最優排程；以 MSCCL++ 自製 CUDA kernel 執行 | AMD MI250；2×DGX A100；**16×DGX H100（128 H100，NVSwitch＋8×IB NIC，每 NIC 50GB/s）** | NCCL、TACCL、RCCL、Blink+Switch | 16×8 H100：allgather +32%、reduce-scatter +14%（1GB）；allreduce 在小訊息 NCCL tree 較佳；2×DGX A100 FSDP 訓練迭代時間 −14%（Gemma-2-27b）、−20%（Llama-2-70b／Llama-3-119b） | 「ForestColl prioritizes throughput over latency」；「designed as an offline schedule generator」；「leveraging topology symmetry is left for future work」（§3 Limitations，PDF line 243–260） | 未說明（正文未見網址） | 部分：H100 部分需 16 台；2–4 台可做縮小版；MSCCL++ 需自行建置 | https://www.usenix.org/conference/nsdi26/presentation/zhao-liangyu |
| P5 | **From Skew to Symmetry: Node-Interconnect Multi-Path Balancing with Execution-time Planning for Modern GPU Clusters (NIMBLE)**（Yao 等；arXiv 2026-03-31；無正式出處） | 流量偏斜時少數連結過載、多 rail 閒置；NCCL／MPI 靜態路由 | 容量正規化最小壅塞最佳化（multiplicative-weights）；GPU kernel 驅動 RDMA pipelining，經中介 GPU 與 rail-matched NIC 繞行 | H100-SXM（每節點 4 GPU，NVLink4）＋4×NDR400 IB（GPUDirect）；偏斜 All-to-Allv 為 2 節點 8 GPU；總節點數未說明 | 單路徑、NCCL、MPI | 節點內頻寬最高 2.3×；節點間吞吐最高 3.8×（對單路徑）；偏斜 All-to-Allv 最高 5.2×（對 NCCL／MPI）；MoE 端到端最高 1.35×；平衡流量與基準相當 | 「Future directions include refining the orchestration engine and investigating an InfiniBand GPUDirect Async (IBGDA) based implementation」（結論段） | 未說明 | 可（部分）：需 ≥2 節點 IB；其節點為 4 GPU／4 rail，我們為 8 GPU／8 rail，需調整 | https://arxiv.org/abs/2604.00317 |
| P6 | **FCDP: Fully Cached Data Parallel for Communication-Avoiding Large-Scale Training**（Park 等；arXiv 2026-02-06；無正式出處） | 商用叢集上 ZeRO-3 的跨機 all-gather 瓶頸 | 前向參數快取在 host memory，反向以節點內 all-gather 重用，跨機 all-gather −50%；PEFT 只傳可訓練參數 | 4 節點×8×A40（48GB），節點內 NVLink bridge 成對＋PCIe4，節點間 IB 100Gbps | ZeRO-3、ZeRO++（討論 MiCS、ZeRO-Offload、ZeRO-Infinity） | 最高 100× ZeRO-3、51× ZeRO++（商用叢集；10Gbps／IPoIB 等低頻寬條件，細節見論文）；PEFT 跨機流量 −99% | 全文 grep 未找到 limitation／future work 段落：未說明 | 未說明 | 可：4 節點 32 GPU 規模相符；其主場景為低頻寬，NDR400 下效益預期不同（僅陳述條件差異，不下判斷） | https://arxiv.org/abs/2602.06499 |
| P7 | **CCL-D: A High-Precision Diagnostic System for Slow and Hang Anomalies in Large-Scale Model Training**（Gu 等；PPoPP 2026 主會，2026-02-03，DOI 10.1145/3774934.3786429） | 集體通訊 slow／hang 異常診斷慢且不準 | rank-level 即時 probe＋決策分析器，定位故障 GPU rank | 部署於 4,000 GPU 叢集一年；GPU 型號、網路未說明 | 「existing solutions」未具名 | 涵蓋近乎全部已知 slow／hang 異常；6 分鐘內定位受影響 rank（部署條件；對照數字未說明） | 未說明（僅讀會議頁摘要） | 未說明 | 不可：依賴 4,000 GPU 生產部署資料；小規模只能做縮小版 | https://ppopp26.sigplan.org/details/PPoPP-2026-papers/19/CCL-D-A-High-Precision-Diagnostic-System-for-Slow-and-Hang-Anomalies-in-Large-Scale- |
| P8 | **R2CC: Reliable and Resilient Collective Communication for Distributed Learning**（Wang 等；arXiv 2025-12-31，v2 2026-10-27；abs 頁註明 NSDI '27 camera-ready） | 網路錯誤使 NCCL 中止整個 job | 利用 multi-NIC：連線遷移、頻寬感知負載重分配、容錯集體演算法 | **4 台 8×H100（SXM5）IB 伺服器（每台 8×ConnectX，400G）**，另以 SimAI 模擬至 1,024 GPU；**未啟用 SHARP** | vanilla NCCL、SHIFT、ICCL、AdapCC、DéjàVu | 活躍故障下訓練額外開銷 <1.1%、推論 <3%；較現有容錯方案故障開銷最多 −92%（訓練）／−98%（推論） | 「InfiniBand SHARP … R2CC does not currently handle it. Supporting SHARP … is left to future work.」（Appendix I）；NVLink／網路分割／GPU 故障列為 out-of-scope | https://github.com/r2cc-project/R2CC | 可：四台 H100 IB 與我們規模相符 | https://arxiv.org/abs/2512.25059 |
| P9 | **Understanding and Improving Communication Performance in Multi-node LLM Inference**（Singhania 等；arXiv 2025-11-12，v4 2026-05-20；ACM Conference on AI and Agentic Systems 2026） | 多機 TP 推論中小訊息 NCCL all-reduce 慢 | NVRAR：NVSHMEM 實作的階層式 recursive-doubling all-reduce | Perlmutter（A100，Slingshot-11，每節點 4 GPU）；Vista（GH200，每節點 1 GPU，InfiniBand）；最多約 16 GPU 規模 | NCCL（all-reduce） | 256KB–2MB 訊息：Slingshot 1.06×–1.92×、InfiniBand 1.14×–3.57× 低延遲（v4 結論）；YALIS 整合後端到端批次延遲最多縮減 1.72×（Llama 3.1 405B，多機 TP，decode-heavy） | 「We leave heuristic-based hyperparameter tuning to future work.」（附錄）；「we leave detailed investigation to future work」（§5，MoE／相關討論旁） | https://github.com/hpcgroup/nvrar ；https://github.com/axonn-ai/yalis | 部分：NVSHMEM 可在 H100＋IB 重建，但原文無 H100 數據；與 T1／T5 重疊 | https://arxiv.org/abs/2511.09557 |
| P10 | **GPU-Initiated Networking for NCCL**（Hamidouche 等，NVIDIA；arXiv 2025-11-19；無正式出處） | host-initiated 通訊對 MoE 等細粒度低延遲負載有額外開銷 | NCCL 2.28 Device API 三模式：LSA、Multimem（NVLink SHARP）、GIN（網路 RDMA）；GIN 兩後端（GDAKI／Proxy）；整合 DeepEP | NVIDIA EOS 叢集 DGX H100（H100 80GB HBM3），2／4／8 節點（16–64 GPU），NVSHMEM 3.4.5、NCCL 2.28 | NVSHMEM IBGDA、NCCL GIN Proxy | 摘要僅稱 GIN 提供 device-initiated 通訊；數據見論文 §V 圖表（2／4／8 節點 DeepEP dispatch／combine 頻寬與延遲，我未逐條擷取，不引數字） | 「Future work will focus on the broader adoption of GIN in production applications such as PyTorch distributed training, TensorRT-LLM…」（結論）；另計畫擴充 one-sided primitives | 未說明（原文未給網址；NCCL 2.28 開源） | 部分：需 NCCL ≥ 2.28 與 ConnectX-6 Dx 以上；Proxy 後端可在標準 RDMA 上跑；與 T3 重疊 | https://arxiv.org/abs/2511.15076 |
| P11 | **EDGC: Entropy-driven Dynamic Gradient Compression for Efficient LLM Training**（Yi 等；arXiv 2025-11-13，v2 2026-09-30；無正式出處） | 靜態梯度壓縮無法隨梯度演化調整 | 以梯度熵動態調整壓縮 rank，跨 pipeline stage 視窗調整 | 叢集 1：8 節點×4 V100（NVLink 300Gbps，乙太網 32Gbps）；叢集 2：16 節點×4 **H100**（NVLink 900Gbps，**IB NDR 400Gbps**），共 64 H100；GPT-2 2.5B／12.1B；Llama-34B 32 GPU | 摘要未說明 | 通訊延遲最多 −46.45%、端到端 −16.13%（兩叢集、GPT-2 2.5B／12.1B；32Gbps 與 400Gbps 皆有評估） | 「The benefit may be smaller in compute-bound or extremely high-bandwidth environments.」（Limitations，System dependency）；另「Broader long-run evaluation on newer architectures … remains future work.」 | 未說明 | 部分：H100＋IB NDR 為同類硬體，規模 32 GPU 可及；壓縮 kernel 需自行實作 | https://arxiv.org/abs/2511.10333 |
| P12 | **Collective Communication for 100k+ GPUs (NCCLX)**（Si 等，Meta；arXiv 2025-10-23，v4 2026-01-09；無正式出處） | 訓練擴至數十萬 GPU，傳統通訊在吞吐與延遲受限 | NCCLX＋CTran 傳輸層（零拷貝、無 SM）；容錯 AllReduce 等 | 以 Meta 超大規模 H100 叢集（RoCE）；CTran AllToAll 實驗 32 節點×8 H100；Llama4 規模訓練 | NCCL | 摘要僅稱 Llama4 通訊效率大幅改善；數字見論文表（未逐條擷取） | 「we mainly focus on the underlying host-driven common communication stack and leave the rest of device-initiated model support as future work.」（§ 設計章節，PDF line 336） | https://github.com/meta-pytorch/torchcomms/tree/main/comms/ncclx | 部分：CTran 開源但為 Meta 內部網路設計，非 IB；規模不可 | https://arxiv.org/abs/2510.20171 |
| P13 | **AsyncHZP: Hierarchical ZeRO Parallelism with Asynchronous Scheduling for Scalable LLM Training**（Bai 等，ByteDance Seed；arXiv 2025-10-23；tech report） | ZeRO 受通訊開銷限制、ND parallelism 複雜 | 對參數／梯度／優化器狀態自適應分層重分片；多 stream 非同步排程重疊 all-gather／reduce-scatter | 256–1,024「cards」（Seed-OSS 256、MoE 384），GPU 型號與網路**未說明**；基於 Megatron-LM | 經典 ND parallelism（TP／PP／EP 最佳配置） | Seed-OSS-9B MFU 40.41% vs 33.83%（相對 119.5%）；36B 113.2%；MoE-100B 110.2%（Table 2） | 全文未見獨立 limitation 段：未說明 | 未說明 | 不可：規模 ≥256 卡 | https://arxiv.org/abs/2510.20111 |
| P14 | **SyCCL: Exploiting Symmetry for Efficient Collective Communication Scheduling**（Cao 等，Alibaba；ACM SIGCOMM '25，2025-09-08~11，DOI 10.1145/3718958.3750499） | MILP 合成器（TECCL、TACCL）搜尋空間爆炸 | 利用集體與拓樸對稱性分解子需求，搜尋 sketch 並合併子排程 | 32-A100 實機（4 台）；H800 rail 拓樸僅以參數**模擬** | NCCL、TECCL、TACCL | 排程效能最高 +127%（摘要）；合成時間 −2 至 4 個數量級；32-A100 實機較 NCCL 最多 +6.3%（§1） | 「SyCCL does not ensure that the synthesized schedules are optimal.」；規模增大使建模準確度下降（§8 Discussions and limitations） | https://github.com/aliyun/syccl（網址取自搜尋結果；PDF 正文印為 aliyun/symccl，Apache 2.0 為搜尋結果所述，未在 repo 頁面核對） | 部分：NCCL 對照可做；輸出需 MSCCL runtime | https://ennanzhai.github.io/pub/sigcomm25-syccl.pdf ；https://github.com/aliyun/syccl |
| P15 | **Enabling Fast Inference and Resilient Training with NCCL 2.27**（Bachan 等，NVIDIA 技術部落格，2025-07-14，**非同儕審查**） | NCCL 2.27 新功能 | 對稱記憶體（單一 NVLink domain 內）；NVLink SHARP＋IB SHARP 支援 AllGather／ReduceScatter | 單 NVL8（DGX／HGX）或 NVL72；圖表平台與 GPU 數未說明 | NCCL 2.26 | 小訊息 AllReduce 延遲最多 9×（Figure 1，條件未載明）；NVL8 小到中訊息最多 2.5×；SHARP 使 AG／RS 所需 SM 從 ≥16 降至 ≤6（宣稱，無數據） | 未說明（廠商文章） | NCCL 開源（部落格未列網址） | 部分：8×H100 單機可驗證對稱記憶體；IB SHARP 需交換器支援，Nano5 是否有待確認 | https://developer.nvidia.com/blog/enabling-fast-inference-and-resilient-training-with-nccl-2-27/ |
| P16 | **Demystifying NCCL: An In-depth Analysis of GPU Communication Protocols and Algorithms**（Hu 等；arXiv 2025-07-07，v3 2026-03-02；無正式出處） | NCCL 內部設計不透明 | 分析 Simple／LL／LL128 協定、intra／inter-node 資料移動、ring／tree；作為 ATLAHS 模擬器基礎 | 論文基準測試在 Alps（CSCS）16 節點 GH200，Slingshot（25GB/s 每向） | 未說明 | ATLAHS 模擬誤差 <5%（摘要所述）；基準測試為協定／演算法隨訊息大小的相對表現 | 全文 grep 無明確 limitation 段：未說明 | 未說明（NCCL 為開源） | 可（量測方法）：協定行為可在 IB 重量測 | https://arxiv.org/abs/2507.04786 |
| P17 | **Collective Communication Profiling of Modern-day Machine Learning Workloads**（Gupta 等；arXiv 2025-07-03；NSDI 2025 poster） | 集體通訊的高頻寬、突發流量造成壅塞與丟包 | 擴充 NCCL logging，分析多種模型的集體種類／大小分佈 | **4 台各 8×H100，節點內 NVLink，rail-optimized 拓樸**，16 GPU（DeepSeek V3、GPT-2、Llama） | 未說明 | 定性：只有少數 GPU 彼此通訊，進入網路的流量很少，spine 層流量可忽略；無數字比較 | 提出「whether we need a rail optimized topology」的疑問（海報末段）；未列為限制 | 未說明 | 可：4×8 H100 規模相符，NCCL logging 可重做 | https://arxiv.org/abs/2507.07117 |
| P18 | **PacTrain: Pruning and Adaptive Sparse Gradient Compression for Efficient Collective Communication in Distributed Deep Learning**（Wang 等；arXiv 2025-05-24；無正式出處） | 梯度聚合開銷；多數壓縮不相容 all-reduce | 主動剪枝使梯度稀疏，所有 worker 共享稀疏結構，保持 all-reduce 相容 | 單一 8×A40 伺服器＋虛擬交換器／VM 模擬 WAN（100Mbps／500Mbps／1Gbps）；非真實跨機 IB | 「state-of-the-art compression-enabled systems」（摘要未具名） | 吞吐 1.25×–8.72×（頻寬受限、視覺與語言任務） | 未說明 | 未說明 | 部分：可在 IB 上實作，但原文是低頻寬模擬場景 | https://arxiv.org/abs/2505.18563 |
| P19 | **Prime Collective Communications Library — Technical Report**（Keiblinger 等；arXiv 2025-05-20；無正式出處） | 公網上同儕節點故障／加入時的集體通訊 | 容錯 CCL，C99 API＋Python（相容 PyTorch／FSDP），多 TCP 連線 all-reduce，支援 DiLoCo | 跨洲 WAN TCP（歐洲／北美），節點數與硬體未說明；非 IB | Gloo | 歐洲內 up to 45 Gbit/s、北美–歐洲 25 Gbit/s 頻寬利用率 | 「PCCL is more sensitive to OS scheduler-induced latencies … Future versions of PCCL may address these issues.」（§6.6 附近，PDF line 874–879） | MIT 授權（abs 頁所述）；repo 網址未在頁面給出 | 不可：目標為 WAN／公網，與 IB 叢集情境不同 | https://arxiv.org/abs/2505.14065 |
| P20 | **The Big Send-off: Scalable and Performant Collectives for Deep Learning (PCCL)**（Singh 等；arXiv 2025-04-25，v2 2026-03-15；無正式出處） | NCCL／RCCL／Cray-MPICH 在超算上擴展性不佳 | 兩層階層式 all-gather／reduce-scatter／all-reduce、跨機 recursive doubling／halving、SVM 自適應選演算法 | Frontier（MI250X，Slingshot，最多 2048 GCD）與 Perlmutter（A100，Slingshot）；**非 InfiniBand** | RCCL（Frontier）、NCCL（Perlmutter）、Cray-MPICH | Frontier 2048 GCD vs RCCL：reduce-scatter 168×、all-gather 33×、all-reduce 10×（up to）；Perlmutter vs NCCL 5.7×；ZeRO-3 訓練最多 4.9×、DDP 2.4× | 「In future work, we plan to benchmark PCCL on clusters with InfiniBand interconnects to evaluate its performance beyond Slingshot-based systems.」（結論） | 未說明（頁面無 repo 網址） | 部分：演算法可重做；原文無 IB／H100 數據 | https://arxiv.org/abs/2504.18658 |
| P21 | **TAGC: Optimizing Gradient Communication in Distributed Transformer Training**（Polyakov 等；arXiv 2025-04-08；EuroMLSys '25） | 零冗餘平行下梯度同步瓶頸 | 支援分片模型的無損同態壓縮＋逐層選擇壓縮、動態稀疏化；整合 PyTorch FSDP | 2 台各 1×GeForce RTX 3080，10 Gbit/s 網路；GPT-2 約 162M 參數 | FSDP | 訓練最多 +15%，品質影響極小 | 「intra-node communication … TAGC will only slow down the training」；未來方向：「adaption of a hybrid strategy where compression applies only for inter-node communication」（§5 Discussion and Future Work） | https://github.com/ipolyakov/TAGC | 可：實作簡單；但原文僅在 2 GPU＋10GbE 驗證 | https://arxiv.org/abs/2504.05638 |
| P22 | **Enhancing Large-Scale AI Training Efficiency: The C4 Solution for Real-Time Anomaly Detection and Communication Optimization**（Dong 等，Alibaba；HPCA 2025；arXiv 2406.04594 v1 2024-06-07、v2 2025-05-23） | 硬體錯誤與流量碰撞造成訓練效率低 | C4D：擴充 ACCL 即時偵測異常、隔離節點重啟；C4P：利用集體流量可預測性做路徑規劃 | 生產叢集（含 4,096 GPU 工作的一個月資料）；測試規模 16–512 GPU；GPU 型號未於讀取處說明 | 未說明 | 效率 30%→45%；錯誤開銷 −30%、通訊成本 −15%（摘要） | 全文未見獨立 limitation 段：未說明 | 未說明（ACCL 為 Alibaba 內部） | 不可：依賴 ACCL 與生產叢集 | https://arxiv.org/abs/2406.04594 |
| P23 | **AutoCCL: Automated Collective Communication Tuning for Accelerating Distributed and Parallel DNN Training**（Xu 等；NSDI '25，pp. 667–683，2025-04） | 多數研究假設 CCL 已調好，忽略底層參數 | 分治＋子空間座標下降搜尋 NCCL 參數；線上調校並考量通訊與計算干擾；實作於 NCCL | 叢集 A：2 節點 16×A40，節點內 NVLink，節點間 2×400G IB；叢集 B：4 節點 32×A40，PCIe，100G IB | NCCL、AFNFA（學術調校器） | 微基準 1.24–1.29× vs NCCL；含計算干擾最高 1.80×；端到端每迭代 1.07–1.32× | 「further exploration of their optimization is left for future work」（transport 專屬參數如 NCCL_NET_GDR_LEVEL，§討論末） | https://github.com/gbxu/autoccl | 可：基於 NCCL，NCCL 參數搜尋可在 H100＋IB 重做 | https://www.usenix.org/conference/nsdi25/presentation/xu-guanbin |
| P24 | **DeToNATION（含 FlexDeMo）: Decoupled Torch Network-Aware Training on Interlinked Online Nodes**（From 等；arXiv 2502.06728，v4 2025-11-17；AAAI 2026 Main Track） | DeMo 假設模型能放進單一加速器 | FlexDeMo：節點內完整分片，節點間只同步快速變動的梯度成分；框架統一 DeMo／FlexDeMo／DiLoCo | Leonardo BOOSTER（每節點 A100 64GB，2×雙埠 HDR 400Gbps 聚合，Dragonfly+）；LUMI（每節點 4×MI250x） | AdamW＋完整同步的 hybrid-sharded DP | 驗證損失與基準相近，「substantially faster」（摘要，無數字）；replicator 方案訓練時間約 2.6× 差異（正文） | 「different replicators are optimal for different tasks. Future work may study whether the ideal type of replicator depends on the task or architecture」（結論）；FSDP 初始 reduce-scatter 的問題列為 Future Work | https://github.com/schneiderkamplab/DeToNATION | 可：PyTorch 層級，可在 IB 叢集跑 | https://arxiv.org/abs/2502.06728 |
| P25 | **Scaling Large Language Model Training on Frontier with Low-Bandwidth Partitioning**（Xu 等，OSU；arXiv 2025-01-08，v2 2025-02-04；無正式出處） | ZeRO-3 通訊量大；Frontier 頻寬分三層 | 以 ZeRO++ 為基礎的三層階層分割（GCD／GPU／節點） | Frontier（MI250X，RCCL），最多 384 GCD；亦分析 DGX A100（8×A100，8×200Gbps IB 埠）拓樸 | ZeRO++ | 20B GPT，384 GCD：每 GPU TFLOPS 1.71× ZeRO++；scaling efficiency 0.94 | 「Our current implementation works optimally with models whose parameters fit onto two GCDs … up to approximately 36 billion parameters.」（§ Limitations B「Model Size Constraints」） | 未說明 | 部分：概念可移植，但原文為 AMD Frontier | https://arxiv.org/abs/2501.04266 |
| P26 | **Streaming DiLoCo with overlapping communication: Towards a Distributed Free Lunch**（Douillard 等，Google DeepMind；arXiv 2025-01-30；無正式出處） | DiLoCo 仍需相同峰值頻寬 | 分批同步參數子集、同步與計算重疊、外層梯度量化至 4 bit | 十億級參數模型；加速器型號與 worker 數未說明；§3.1 用**自建 simulator** 估計 compute utilization | 未說明（對照 Data-Parallel、DiLoCo） | 與 Data-Parallel 相近品質下頻寬降兩個數量級（正文結論稱 400× 少頻寬） | 結論段「Conclusion and Future Work」（§5）；未見逐條限制 | 未說明 | 部分：演算法層級，真實效益需訓練實驗，與 IB 效能關聯弱 | https://arxiv.org/abs/2501.18512 |
| P27 | **LoCo: Low-Bit Communication Adaptor for Large-scale Model Training**（Xie 等；arXiv 2024-07-05，v2 2024-11-29；無正式出處） | 低位元梯度通訊損失品質 | 壓縮前以誤差移動平均補償；相容 Adam、FSDP、Megatron-LM | 32／64／128 GPU，A100（RoCE v2）與 A800（InfiniBand）叢集 | 全精度 Adam、ZeRO++ | Adam 訓練速度 +14%–40% 無效能損失（LLaMA、MoE）；正文稱 GPU 愈多加速愈大（128 GPU 下 LLaMA2 之數字見論文 §5 圖表，未逐條擷取） | 「removing the bounded gradient assumption for the convergence analysis of Adam-family methods remains an open and challenging problem. We acknowledge this limitation and leave it for future work.」（§4.1 末） | 未說明 | 可（部分）：需自行實作低位元通訊 | https://arxiv.org/abs/2407.04480 |
| P28 | **Nezha: Breaking Multi-Rail Network Barriers for Distributed DNN Training**（Yu 等；arXiv 2024-05-28，v2 2026-02-09；無正式出處） | 靜態單 rail 綁定浪費 multi-rail 頻寬；TCP／RDMA 並存延遲 | 協議無關的 multi-rail 系統，協調 SHARP、GLEX、TCP；冷熱啟動狀態機；單 rail 失效 200ms 內改道 | 本地節點（2×V100、多張乙太網／IB／TH NIC）、雲端節點（1×V100）、超算節點；8 節點與 128 節點；**非 H100／NDR** | Gloo、MPTCP | 延遲 1.7–4.3× 低於 Gloo；8 節點 TCP-TCP 吞吐 +74%（vs MPTCP）；TCP-SHARP +80%；128 節點訓練效率 2.36×（vs Gloo） | 「In the future, we plan to evaluate Nezha's performance on more advanced large-scale clusters and assess its capabilities for other collective communication operations.」（結論） | https://github.com/nezha-Anonymous/nezha（PDF 第 1 頁註腳） | 不可（原文基準）：以 Gloo 為基準、V100／GLEX；其 SHARP 部分需交換器支援 | https://arxiv.org/abs/2405.17870 |
| P29 | **MegaScale: Scaling Large Language Model Training to More Than 10,000 GPUs**（Jiang 等，ByteDance；NSDI '24，pp. 745–760，2024-04；arXiv 2402.15627） | 超大規模 LLM 訓練的效率與穩定性（故障與落後節點） | 全堆疊協同設計：重疊、運算元最佳化、網路調校（ECMP 衝突、壅塞控制、重傳逾時）、診斷與容錯 | 12,288 GPU（NVIDIA Ampere），Tomahawk4 交換器、每台 8×200G NIC | Megatron-LM | 175B 模型 55.2% MFU，1.34× Megatron-LM | 未說明（未逐段讀取；摘要稱分享營運經驗） | 未說明；footnote 稱正開源部分元件：https://github.com/volcengine/veScale | 不可：規模 12,288 GPU | https://www.usenix.org/conference/nsdi24/presentation/jiang-ziheng |

---

## 3. 作者自述的限制與未來工作彙整

依出現次數歸類（引文已在表中，此處只列編號與主旨）：

1. **評估平台限制／尚未在 InfiniBand、更大叢集或更新硬體上驗證**（P20、P28、P11、P25）：P20「plan to benchmark PCCL on clusters with InfiniBand interconnects」；P28「evaluate Nezha's performance on more advanced large-scale clusters」；P11「Broader long-run evaluation on newer architectures … remains future work」；P25 僅在 Frontier、模型限約 36B。
2. **延遲敏感情境、小訊息與動態情況未涵蓋**（P2、P4、P9、P5）：P2「Extending the bounds to latency-sensitive settings and dynamic bandwidth fluctuations is left to future work」；P4「prioritizes throughput over latency」；P9「We leave heuristic-based hyperparameter tuning to future work」；P5 未來要做 IBGDA 版本以降低延遲。
3. **合成／最佳化的規模與最優性限制**（P3、P14、P4）：P3 SMT 直接用於數千 GPU「computationally intractable」；P14「does not ensure that the synthesized schedules are optimal」；P4 為離線一次性排程產生器。
4. **效益依賴硬體條件（頻寬、拓樸、是否已重疊）**（P11、P21）：P11「benefit may be smaller in compute-bound or extremely high-bandwidth environments」；P21 節點內壓縮只會拖慢，建議僅用於跨機。
5. **不支援網路內運算（SHARP）或相關功能待補**（P8；另 P10、P12 的裝置端 API）：P8「R2CC does not currently handle it [SHARP] … left to future work」；P10、P12 皆把 device-initiated／更廣泛應用留待未來。
6. **理論假設未解**（P27）：Adam 家族收斂分析仍需有界梯度假設。

（P7、P13、P16、P22、P29、P6 全文／摘要中未找到明確限制段，已在表中標「未說明」。）

---

## 4. 常見比較對象

- **NCCL**（預設設定）：P2、P3、P4、P5、P8、P9、P12、P14、P15、P20、P23 皆以 NCCL 為對照；是出現最多的基準，也未見有人宣稱已超越「調過參數的 NCCL」後再做公認對照——僅 P23（AutoCCL）把 NCCL 調校器（AFNFA）當對照。
- **排程合成器**：TACCL、TE-CCL（P3、P4、P14）；HeteCCL（P3）在 §7 引用 SyCCL 並採同類階層組合（Similar to SyCCL [7] and Canvas [14]），但其實驗基準列為 NCCL、TACCL、TE-CCL，未列 SyCCL。
- **其他函式庫**：RCCL（P20）、Gloo／MPTCP（P28）、Cray-MPICH（P20）。
- **容錯**：ICCL、SHIFT、AdapCC、R2CCL／R2CC（P2、P8）。
- **訓練端**：ZeRO-3、ZeRO++、PyTorch FSDP、Megatron-LM（P6、P11、P13、P21、P25、P29）。
- 沒有單一「公認最強對照」；集體通訊以 NCCL 為共同底線，ZeRO 類以 ZeRO++ 為較新對照。

---

## 5. 硬體覆蓋

- **8× H100 NVSwitch 單機或 ≤4 台 IB 叢集上有實驗**：
  - P8（R2CC：4×8 H100，IB 400G，另 SimAI）
  - P5（NIMBLE：H100-SXM，4 rail NDR400；偏斜 All-to-Allv 為 2 節點 8 GPU）
  - P11（EDGC：16 節點×4 H100，IB NDR；32–64 H100）
  - P17（4×8 H100 剖析海報）
  - P10（GIN：EOS DGX H100，2／4／8 節點；8 節點超過 4 台上限）
  - P4（ForestColl：16×DGX H100＝128 GPU，另有較小規模 1–16 節點掃描；超過 4 台）
  - P12（NCCLX：32 節點×8 H100 的 CTran 實驗；超過 4 台）
- **只在 A100、AMD、GH200、V100／A40、模擬器、WAN 上做過**：
  - A100／A800：P14（32 A100，H800 僅模擬）、P27、P24、P9（Perlmutter）
  - AMD：P20、P25
  - GH200：P16、P9（Vista）
  - A40／V100／RTX3080：P23、P6、P28、P21、P18
  - 模擬器：P2（SimAI）、P26（自建 simulator）
  - 公網 WAN：P19
  - H20＋V100 異質：P3
- **「NVLS 開與關之間比較」**：本次讀到的論文中，只有 **P4（ForestColl）** 的 Figure 12 圖例同時出現「ForestColl w/ NVLS」「ForestColl w/o NVLS」「NCCL NVLS」「NCCL NVLSTree」（H100 單機 1×8 到 16×8 的 allgather 掃描，PDF 圖例文字；文中指出 1×8 時 NVLS 版在小訊息較佳，因為 zero-copy 實作延遲低）。P10 介紹了 Multimem（NVLink SHARP）模式，P15 描述 NVLS＋IB SHARP 的 AG／RS，但未見 NVLS 開關的對照數據。P23 的搜尋空間參數列出 NVLS、COLL_SHARP 為可選傳輸，但我沒有核實其是否有開關的實驗結果。
- **SHARP／IB 網路內運算實測**：P8 明言測試機未啟用 SHARP；P28 用過 SHARP 於 TCP-SHARP 異質配置；其餘僅廠商說法（P15）。

---

## 6. 開源工具與程式碼清單

| 名稱 | 網址 | 授權 | 備註 |
|---|---|---|---|
| R2CC | https://github.com/r2cc-project/R2CC | 未說明 | 容錯 NCCL 衍生（fork NCCL 2.23.4，見 PDF） |
| NVRAR | https://github.com/hpcgroup/nvrar | 未說明 | NVSHMEM 階層 all-reduce |
| YALIS | https://github.com/axonn-ai/yalis | 未說明 | 論文用的研究推論引擎 |
| NCCLX／CTran | https://github.com/meta-pytorch/torchcomms/tree/main/comms/ncclx | 未說明 | Meta |
| SyCCL | https://github.com/aliyun/syccl | Apache 2.0（搜尋結果所述，未核對 repo） | 以 SCIP 求解、MSCCL runtime |
| AutoCCL | https://github.com/gbxu/autoccl | 未說明 | 論文 §5 稱 open-sourced |
| TAGC | https://github.com/ipolyakov/TAGC | 未說明 | PyTorch FSDP 整合 |
| DeToNATION／FlexDeMo | https://github.com/schneiderkamplab/DeToNATION | 未說明 | |
| Nezha | https://github.com/nezha-Anonymous/nezha | 未說明 | 匿名 repo 名稱，PDF 註腳 |
| PCCL（Prime） | 未說明（repo 網址未取得） | MIT（abs 頁所述） | |
| veScale（MegaScale 相關） | https://github.com/volcengine/veScale | 未說明 | MegaScale footnote 稱開源部分元件 |
| nccl-tests | https://github.com/NVIDIA/nccl-tests | 未說明 | 多篇引用的基準測試工具（P2 參考文獻 [15]） |
| MSCCL／MSCCL++ | https://github.com/Azure/msccl | 未說明 | 被 P4、P14 作為 runtime（MSCCL++ 網址未核對） |

---

## 7. 搜尋紀錄

WebSearch 共 14 次（皆 standard，除標明者）：

1. `arXiv 2025 collective communication synthesis topology-aware multi-node GPU InfiniBand all-reduce`：10 筆；採用 SyCCL、NCCLX；TACCL／TACOS（2021／2023）不列（早於範圍）；PCCL（2606.07019）搜尋摘要提到但未開啟核實，**未列入**。
2. `arXiv gradient compression large-scale training inter-node bandwidth 2024 2025 communication-efficient data parallel`：10 筆；採用 TAGC、EDGC、PacTrain、LoCo；2312.02493 為綜述性質、未列入。
3. `arXiv NCCL InfiniBand SHARP in-network reduction GPU training evaluation`：10 筆；採用 GIN、Nezha；Lambda 與 NVIDIA SHARP 部落格、spheron 等為廠商／部落格，未列（只列 NCCL 2.27 官方部落格）。
4. `arXiv FSDP ZeRO inter-node communication bottleneck hierarchical sharding 2024 2025`：10 筆；採用 Frontier ZeRO++、AsyncHZP、FCDP、DeToNATION；AMSP（2311.00257）屬 2023 提出，未列；PaRO（2023）未列；2407.01614 未開啟核實，未列。
5. `arXiv stragglers congestion collective communication LLM training NCCL fail-slow detection`：10 筆；採用 OptCC、CCL-D、MegaScale、C4；Mycroft、GUARD（MLSys 2026 Industry）只出現在搜尋摘要，**未開啟原文核實，未列**；CoreWeave 部落格未列。
6. `arXiv 2024 2025 hierarchical all-reduce multi-NIC GPUDirect RDMA H100 InfiniBand cluster collective communication library`：10 筆；採用 Big Send-off、NVRAR、NIMBLE；OmNICCL（NAIC'24）、OSU IPDPS'25 poster 未核實，未列。
7. `DiLoCo streaming low-communication distributed training across datacenters arXiv`：10 筆；採用 Streaming DiLoCo（原版 DiLoCo 2311.08105 為 2023-11，早於 2024 重點範圍，未列）。
8. `arXiv NCCL profiling tracing network performance LLM training InfiniBand characterization H100 cluster arXiv 2025`：10 筆；採用 2507.07117、Demystifying NCCL；CCLInsight（ICSE 2026）、TU Wien／Heidelberg 論文皆未開啟核實，未列。
9. `OSDI NSDI SOSP 2024 2025 collective communication GPU cluster network scheduling congestion DNN training`：10 筆；採用 HeteCCL、ForestColl、AutoCCL；CASSINI（NSDI'24）、Crux（SIGCOMM'24）、FORESIGHT（IFIP'25）、Ethereal、FAST（NSDI'26，屬 all-to-all，歸 T3）未核實或不屬本主題，未列。**搜尋結果未見 OSDI／SOSP 2024–2025 相關論文**。
10. `arXiv ZeRO FSDP RDMA inter-node all-gather reduce-scatter optimized implementation PCCL fault-tolerant collective library`：10 筆；採用 Prime PCCL。
11. `SyCCL SIGCOMM 2025 exploiting symmetry …`：10 筆，用於補 SyCCL repo 與 DOI（ACM 頁面本身 404／未開啟，DOI 以 PDF 頁尾文字為準）。
12. `C4 HPCA 2025 communication-driven fault tolerance …`：10 筆；確認 C4 與 HPCA 2025 對應。
13. `arXiv 2025 TE-CCL OR "NCCL" tuning "H100" multi-node InfiniBand all-reduce algorithm selection SHARP CollNet evaluation`：10 筆；僅有部落格與一篇 AsHES@IPDPS'25 workshop（未核實），未列。**沒有找到 CollNet／SHARP 在 H100 叢集上的同儕審查量測**。
14. `in-network aggregation SHARP GPU allreduce InfiniBand switch arXiv 2024 2025 …`（extended）：10 筆；Rina（2407.19721）、NetReduce（2020）、Flare（2021）等未開啟或早於範圍，未列；2407.20018 survey 為更廣的訓練基礎設施調查，未列。
15. `arXiv 2025 multi-rail NIC GPU cluster NCCL PXN rail-optimized …`：10 筆；採用 R2CC（2512.25059）；Rail-only（2307.12169，2023）、HOTI'24 版、KIPS 論文未核實，未列。

WebFetch：arXiv abs 頁 21 次、USENIX／PPoPP 會議頁 5 次；`curl`＋`pdftotext` 下載 24 份 PDF 全文做限制與硬體擷取。無法核實／失敗項目：SyCCL 的 ACM 落地頁（自行猜測的 DOI 回 404，未採用；以作者個人站 PDF 與搜尋結果為準）；WebFetch 直接讀 PDF 會得到壓縮串流，改用 `pdftotext`。
