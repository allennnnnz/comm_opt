# R02 / T5 inference_serving：推論服務中的通訊（文獻地圖）

- 執行者：Sonnet 5.5，2026-11-16；依據 `docs/specs/S02_lit_map.md`。純事實整理，**不含選題判斷**。
- 核實方式：每篇都用 WebFetch 打開 arXiv abs 頁（部分另開會議頁），再把 PDF 下載後用 pdftotext 讀全文擷取硬體、數字與限制原文。PDF 暫存於 scratchpad（`papers/` 目錄對本次 session 不可寫入，未存入 `papers/`）。
- 共列 **24 篇已核實論文**（編號 1–24）＋ 2 項非同儕審查的廠商／專案資料（編號 V1、V2）。
- 「原文引用」皆為英文原句，位置以「節號」或「PDF 抽出文字的段落」標示；沒有明確 limitations 節的寫「未說明」。
- 「可重現」判斷基準：Nano5 最多 H100 4 台（32 GPU）或 H200 2 台（16 GPU），NDR IB，無 root，NVLS 待驗證。

---

## 1. 主題概況

1. **Prefill／decode 分離（PD disaggregation）與 KV cache 傳輸**：DistServe (2024)、Splitwise (2023)、Mooncake (2024) 奠定架構；之後的工作分成三支——(a) 傳輸機制本身（KVDirect 的 pull 模式 RDMA、fabric-lib 的 one-sided WRITE、TENT 的多軌切片噴灑、LMCache 的 KV 搬運層）、(b) 降低傳輸量或隱藏傳輸（Lynx 逐位元漸進傳輸、HMA-Serve 逐層管線、PDD 的 relay decode、PrfaaS 的跨資料中心選擇性 prefill 外包）、(c) 量化「頻寬是否夠用」（Beyond the Buzz 的解析模型 vs. PDD／Lynx 認為傳輸是主要瓶頸）。
2. **TP 生成階段（decode）的小訊息延遲**：NVRAR（跨節點 NVSHMEM all-reduce）、SiFAR（8×H200 單機、multimem in-switch reduction＋去同步）、SCIN（提出替換 NVLS 的 switch-centric 架構，FPGA＋模擬）、TokenWeave（融合 AllReduce–RMSNorm，用 NVLS/multimem 只佔 2–8 SM 做 overlap）。
3. **長上下文／context parallel**：Meta 的 CP（pass-KV／pass-Q ring attention，最多 128 H100）、Medha（sequence pipeline＋KV-cache parallelism＋可搶佔排程）、Helix（KV parallelism 後在 FFN 重用同批 GPU，GB200 模擬器）。
4. **MoE／EP 的生成階段通訊**：MegaScale-Infer（attention 與 expert 分離，M2N 通訊庫）、DeepSeek-V3 硬體反思（EP all-to-all 頻寬解析，IB 與 NVL72 對比）、SGLang 96×H100 部落格（PD 分離＋大規模 EP）；fabric-lib 亦含 MoE dispatch/combine。
5. **RL 訓練的權重同步（trainer → rollout）**：Laminar（relay worker 非同步拉取）、fabric-lib（P2P one-sided WRITE，宣稱 1T 模型 1.3 s）、TENT（替換 checkpoint-engine 的 P2P 後端）、SparseRL-Sync（稀疏 delta 無損傳輸）、checkpoint-engine（專案，非論文）。

---

## 2. 論文表（依年份新到舊）

欄位：硬體與規模含 GPU 型號／單機互連／是否用 NVLS／跨機網路／GPU 數。「作者自述限制」為原文引用（≤2 句）。

| # | 論文（年／出處） | 解決的問題 | 方法要點 | 硬體與規模 | 比較對象 | 報告的效益（含條件） | 作者自述的限制或未來工作 | 程式碼 | 在我們環境可重現？ | 出處網址 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **PDD: Unleashing Economical and Flexible Heterogeneous LLM Inference via Cross-Datacenter Prefill-Decode Disaggregation** (arXiv 2026；arXiv 頁面 v1 日期 2026-07-18；Wang, Li, Ma 等，Infinigence-AI／清華) | 跨資料中心 PD 分離時，KV cache 經廣域乙太網傳輸造成 TTFT 過高（agent 類短輸出請求傳輸佔 e2e 43–56%，§1 Fig.1） | 三層：Prefill、Relay Decode（RLD，同機房 RDMA 收 KV 立刻開始 decode）、Main Decode（遠端經 TCP 收 KV）；decode 端 RadixCache、Extend-Decode Handoff、多階段管線 | H100 43 節點（344 GPU，放 P 與 RLD）＋ H200 32 節點（256 GPU，放 MD）；兩機房 2×10 Gbps 廣域乙太網；機房內 400 Gbps RDMA（§5 實驗設定）；未說明 NVLS | 同機房同質 PD 部署（IDC-PD）；摘要稱 BCR 最高 +37.5% SLA 達標 goodput | 摘要：相對 intra-DC homogeneous PD baseline，BCR（性價比指標）最高提升 37.5%（摘要） | 「First, due to hardware availability, our RLD deployment was constrained to a 3-node H100 setup.」「Second, system-level optimizations such as MTP were disabled in our current tests due to DRC compatibility constraints.」（§5 Limitations and Future Improvements） | 未說明 | 不可：需 600 GPU 與跨資料中心鏈路 | https://arxiv.org/abs/2609.13161 |
| 2 | **SiFAR: Synchronization-Free All-Reduce for Low-Latency LLM Inference** (arXiv 2026-07-09；Taneja, Saxena, Revinipati 等，Georgia Tech／NVIDIA) | 低批量 decode 的 TP all-reduce：oneshot／twoshot 的前後 barrier 佔小訊息延遲 32–62%（摘要） | 去除 oneshot 的 bottom barrier（雙緩衝）、「redundant pull」用 multimem.ld_reduce（in-switch reduction）每個 GPU 拉整份 payload、speculative reduction 減少 top barrier；整合進 Megakernels | 單機 8× H200（141 GB HBM3e），NVSwitch，**使用 multimem in-switch reduction（NVLS 同類機制）**；CUDA 12.9；TP=2/4/8；Llama-3.1-8B、Qwen3.5-397B-17B（FP8）（§7 Evaluation Hardware） | 最佳 oneshot／twoshot、TRT-LLM oneshot、Lamport oneshot、NCCL 2.30（auto，含 LL 低延遲 kernel） | All-Reduce 延遲最多 -52%；TP=8 端到端吞吐 +18.6%（Llama-3.1-8B）、+13.1%（Qwen3.5-397B-17B）；比 NCCL 2.30 低 46–62%；移除 All-Reduce 可使 Llama-3.1-8B 於 8×H200 吞吐 +43%（摘要、§7） | 「correctness assumes that if GPU1 is ahead of GPU0 at time T1, it remains ahead at time T2 > T1, i.e., relative GPU progress is monotonic.」（§8.1 Discussion）；僅評估單機 8 GPU | 未說明 | **可**（單機 8×H200 可用）；但需 Megakernels 整合，程式未公開，故實作需自行重做 → 部分 | https://arxiv.org/abs/2607.08973 |
| 3 | **Lynx: Progressive Speculative Quantization for accelerating KV Transfer in Long-Context Inference** (arXiv 2026-07-02；Han, Yeung, Barletta 等，UCL／Huawei；arXiv 註記原投 SIGCOMM '26) | PD 分離時 decode 必須等 KV 全到齊；量化降傳輸量但仍需整份到達 | 將 KV 拆成高位元 Anchor 串流與低位元 Residual 串流；Anchor 到就開始 speculative decode，Residual 並行傳輸後驗證等價於高精度 | Ascend NPU（vLLM-Ascend＋LMCache-Ascend）；以 rate-limiter 把節點間頻寬限在 10–50 Gbps 模擬長上下文高傳輸/計算比（§6） | 未壓縮 BF16、INT8、INT4、CacheGen（best-effort 移植）（§6 Baselines） | 相對 INT8 的 TTFT 最多 -30.5%（摘要/§1：改善 1.43×）；準確度比 SOTA 高最多 5.1% | 「Lynx can support alternative bit-width configurations for the anchor and residual streams ... We leave a systematic exploration of these configurations to future work.」「while Lynx is implemented on the Ascend NPU platform, its design is hardware-agnostic ... We leave such extensions to future work.」（§8 討論） | 未說明 | 不可：Ascend NPU | https://arxiv.org/abs/2607.01831 |
| 4 | **HBM Is Not All You Need: Efficient Disaggregated LLM Serving across Memory-heterogeneous Accelerators**（HMA-Serve）(arXiv 2026-06-29；Wei, Wang, Yen 等，上海交大) | PD 分離時 prefill 不需昂貴 HBM；跨廠牌（GDDR 加速器＋HBM GPU）缺共同 KV 格式與軟體堆疊 | phase-wise 量化（prefill 原生低精度、decode 保持 BF16）、逐層 KV 傳輸與後續層 prefill 重疊、延遲反量化（傳原始量化位元組） | 4 張 Tenstorrent Blackhole p150（prefill）＋ A100-80GB（decode），100 Gb RoCE，vLLM 0.19.1（§評估 Testbed） | DistServe-Homo（2×A100 同質 1P1D，GPUDirect-RDMA KV）等 | 4 個 Qwen3（4B–32B）、3 條 production trace：goodput 最高 3.2×、goodput/美元 4.8× | 未說明（無獨立 limitations 節；摘要提「cross-vendor」假設） | 未說明 | 不可：Tenstorrent 硬體 | https://arxiv.org/abs/2606.29986 |
| 5 | **SparseRL-Sync: Lossless Weight Synchronization with ~100× Less Communication** (arXiv 2026-05-08；Hu, Zhao, Zhu 等，Scitix) | RL 解耦式 trainer→rollout 權重同步在頻寬受限（跨機房、TCP）下成為瓶頸 | 觀察到 RL 每步權重變動在元素層級 99%+ 稀疏；傳 (indices, values) 無損稀疏 payload，推論端精確重建，配 bucketing | 8×H100 SXM5 節點、每節點 4 張 RDMA NIC；NVLink 約 319 GB/s/GPU；IB on 每 GPU 約 35 GB/s、IB off（TCP）約 2.84 GB/s；106B 模型實測於 128 H100（64 Trainer＋64 Rollout），671B 為**投影**（§4） | 全量更新 baseline（NCCL 廣播） | 106B：IB off 全量廣播 45.6 s → SparseRL-Sync 2.26 s；稀疏約 99%+（§4、摘要）；500 步 bit-exact 驗證 | 「the additional lossless index/value compression of Section 3.3 would shrink the SparseRL-Sync payloads further, which we leave to future measurements.」（§4 Communication Savings Across Model Scales） | 摘要稱 will be released: https://github.com/scitix/helix（repo 存在，Apache-2.0；是否已含本文程式未核實） | 部分：實測用 128 GPU，我們最多 32；稀疏性觀察與小規模廣播可做 | https://arxiv.org/abs/2605.07330 |
| 6 | **Prefill-as-a-Service: KVCache of Next-Generation Models Could Go Cross-Datacenter**（PrfaaS）(arXiv 2026-04-16；Qin, He, Wang 等，Moonshot AI／清華) | 混合注意力模型 KV 變小，使跨資料中心 PD 成為可能，但仍有突發流量、前綴快取不均、頻寬波動 | 選擇性將長 context prefill 外包給獨立計算密集叢集，KV 走一般乙太網傳回本地 PD 叢集；頻寬感知排程、快取感知請求配置 | 案例研究：PrfaaS 叢集 32× H200＋本地 PD 叢集 64× H20（800 Gbps RDMA/節點），跨叢集 VPC 約 100 Gbps；內部 1T 參數混合模型；**使用 profiling 式吞吐模型求解（§3.4、§4）** | 96×H20 同質 PD；無排程的樸素異質部署 | 相對同質 PD：吞吐 +54%、P90 TTFT -64%；同成本約 +15% 吞吐；跨資料中心平均出口僅約 13 Gbps（§4） | 未說明（無獨立 limitations 節）；結論註明「H200 and H20 serve here as a representative hardware pair」 | 未說明 | 不可：內部模型＋96 GPU 異質叢集 | https://arxiv.org/abs/2604.15039 |
| 7 | **TENT: A Declarative Slice Spraying Engine for Performant and Resilient Data Movement in Disaggregated LLM Serving** (arXiv 2026-04-01；Ren, Qin, Ma 等，清華／Moonshot AI／阿里雲／螞蟻) | Mooncake Transfer Engine 的「命令式、早綁定」路徑選擇造成頻寬擱淺、遇灰色故障需人工介入 | 宣告式傳輸意圖，slice 時刻晚綁定，依即時遙測與預測成本在多軌（multi-rail）噴灑細粒度 slice，亞 50 ms 繞過故障 | H800：2 節點各 8×H800＋8×200 Gbps RoCE；H20：4 節點各 8×H20＋4×400 Gbps RoCE；另有 GB200 NVL72 與 Ascend（§5 Testbed） | Mooncake TE、NIXL、UCCL-P2P | SGLang HiCache：吞吐最高 1.36×、P90 TTFT -26%（vs Mooncake TE）；RL：Moonshot Checkpoint Engine v0.2.0 參數更新時間 -19.7%（Qwen3-235B）／-26.1%（GLM-4.5-Air）（8×H800 TP8，Table 3） | 未說明（結論未列限制） | https://github.com/kvcache-ai/Mooncake（論文 §1 聲明已開源；Apache-2.0） | 部分：開源、2 節點規模可做；測試用 RoCE 非 IB，且需 H800/H20 以外的 H100 重測 | https://arxiv.org/abs/2604.00368 |
| 8 | **A Switch-Centric In-Network Architecture for Accelerating LLM Inference in Shared-Memory Network**（SCIN）(arXiv 2026-03-30；Jiang, Zhu, Zhang 等，南京大學／中國移動研究院) | TP all-reduce 在 NVLS 下仍佔 H200 低並行度設定的 prefill 時間最多 49%、decode 19%（§1 自家 profiling）；NVLS 為 GPU 驅動，有多餘 switch–GPU–switch 資料搬移且無法做帶 scale 的量化 all-reduce | switch 端編排的 in-network 集體通訊，pipelined 反量化–歸約–再量化資料路徑，近乎減半通訊量 | 多 FPGA 原型＋以 BookSim2 做的 cycle-level 模擬（拓樸同 DGX H200：8 加速器、4 switch）；profiling 在 8× H200 的 TensorRT-LLM、FP8、TP=8、**NVLS 開啟**（§1） | NVLS all-reduce | 模擬 8-GPU：小/大訊息 all-reduce 最高 1.8×／2.6×（vs NVLS）；TPOT 最高 1.12×、TTFT 最高 1.42×（摘要） | 「Although our FPGA prototype demonstrates the feasibility and efficiency of SCIN, its scale is limited.」（§5 評估方法，說明改用模擬）；其餘未說明 | 未說明 | 不可（硬體架構，需 FPGA／模擬）；僅「NVLS 下 all-reduce 佔比」之 profiling 可在 8×H200 重現 | https://arxiv.org/abs/2603.28239 |
| 9 | **Not All Prefills Are Equal: PPD Disaggregation for Multi-turn LLM Serving** (arXiv 2026-03-09；Li, Liu, Xu 等；**ICML 2026** 接受，依 arXiv 頁 Comments) | 多輪對話中 PD 的 KV 傳輸只能 P→D，第 2 輪起必須在 P 重算整段歷史 | Prefill-capable Decode（PPD）：動態決定第 2 輪以後請求是否在 decode 節點本地 append-prefill；以 TTFT–TPOT 權重控制 | 單節點 4× H100 80GB HBM3、NVLink；以 TetriInfer 方法注入延遲模擬較慢的跨機網路（§6.4）；Llama-3.1-8B、Qwen2.5-14B、Qwen3-30B-A3B；基於 vLLM 的 PD 基礎設施 | 傳統 PD 及 17 種配置（含 R／P／D 混合） | 第 2 輪以後 TTFT 降 ~68%，TPOT 持平（摘要） | 「The principal limitation of PPD is that its lookup table is built offline and degrades gracefully but suboptimally when hardware or workload distributions drift far from the calibration set」（§7 Discussion） | 未說明 | **可**：4×H100 NVLink 單節點即可 | https://arxiv.org/abs/2603.13358 |
| 10 | **Understanding and Improving Communication Performance in Multi-node LLM Inference**（原名 *LLM Inference Beyond a Single Node: From Bottlenecks to Mitigations with Fast All-Reduce Communication*；arXiv v1 2025-11-12，v4 2026-05-20；Singhania, Singh, Hough 等，UMD／LLNL；**CAIS '26**，DOI 10.1145/3786335.3813165） | 多節點 TP 推論的 all-reduce 在小訊息時瓶頸（NCCL 次優） | NVRAR：以 NVSHMEM 實作 hierarchical recursive-doubling all-reduce；另做 vLLM／SGLang／自家 YALIS 引擎的強規模分析 | Perlmutter：A100（40/80GB）每節點 4 張，3rd gen NVLink，Slingshot-11；Vista：GH200 每節點 1 張，InfiniBand（Table 1）；未使用 NVLS | NCCL all-reduce（Tree/Ring）；vLLM V1 0.11.0、SGLang 0.5.1、YALIS | 128 KB–2 MB 訊息比 NCCL 低 1.9×–3.6×（摘要）；整合後 Llama 3.1 405B 多節點 decode-heavy 端到端批次延遲最多 -1.72×（摘要）；**結論段寫 1.06×–1.92×（Slingshot）、1.14×–3.57×（InfiniBand），256 KB–2 MB**（見 §5 矛盾） | 「We plan to investigate and reduce idle time in future versions of NVRAR.」（§5.2.2）；「We leave heuristic-based hyperparameter tuning to future work.」（§5 消融） | https://github.com/axonn-ai/yalis（Apache-2.0）、https://github.com/hpcgroup/nvrar（MIT） | 部分：開源；我們 H100＋NDR IB 2–4 節點可測，但原作在 A100／GH200 | https://arxiv.org/abs/2511.09557 |
| 11 | **fabric-lib: RDMA Point-to-Point Communication for LLM Systems** (arXiv 2025-10-31，v2 2026-04-13；Licker, Hu, Zaytsev, Chen 等，Perplexity；**MLSys 9th**，依 PDF 頁首) | 現有 P2P 通訊綁定特定 NIC，不利於 PD 分離、MoE、非同步 RL | 統一 one-sided WRITEIMM＋ImmCounter 完成通知，不假設傳輸排序，透明管理每 GPU 多 NIC；三個應用：KV 傳輸、RL 權重更新、MoE dispatch/combine | 8×H200 節點（每 GPU 2×200 Gbps AWS EFA）與 8×H100 節點（每 GPU 400 Gbps ConnectX-7）；節點數未明列；NVLink 用於節點內；自家 PyTorch 引擎（§7） | DeepEP、pplx-kernels；既有 RL 框架的 Rank0 廣播 | 峰值 400 Gbps（ConnectX-7、EFA）；RL 權重更新 256 訓練 GPU（bf16）→128 推論 GPU（fp8）1T 模型 1.3 s；MoE decode 延遲於 ConnectX-7 超過 DeepEP（摘要、§5、§6） | 「GPU-initiated RDMA (GDA) ... is not yet available on most cloud instances ... and remains preliminary on p5en; our host-proxy design serves this current-generation hardware」「Porting to additional NICs requires per-hardware tuning, not a redesign of the library」（§8 Discussion）；§6 另記 prefill 因缺少 chunking 而記憶體負擔限制可部署模型 | https://github.com/perplexityai/pplx-garden（MIT） | 部分：ConnectX-7 400G H100 路徑可測；256+128 GPU 的 RL 規模不可 | https://arxiv.org/abs/2510.27656 |
| 12 | **Laminar: A Scalable Asynchronous RL Post-Training Framework** (arXiv 2025-10-14；Sheng, Tong, Wan 等，HKU／ByteDance Seed；依搜尋結果與 HKU 檔名為 EuroSys '26，ACM 頁面 403 未能直接核實，DOI 見下) | 非同步 RL 的全域權重同步造成僵硬更新排程，長尾軌跡造成 GPU 閒置 | 以 relay worker 層取代全域同步（rollout 隨時從 relay 拉權重，鏈式管線 RDMA 廣播）、動態 repack 長尾軌跡、解耦故障 | 128 機 ×8 H800-80GB（共 1024 GPU），400 GB/s NVLink，機間 8×400 Gbps；CUDA 12.6、NCCL 2.26.2、vLLM 0.9.0；Qwen2.5 7B/32B/72B（§8 Testbed） | verl v0.5.0（同步）、one-step staleness、stream generation、partial rollout 等四類 | 1024 GPU 訓練吞吐最高 5.48×；72B 模型自 master 廣播到 127 個 relay 小於 1.6 s（§4、附錄 D） | 「developing effective sampling strategies for utilizing generated experiences represents a promising area for future work in large-scale RL systems.」（Discussion） | 未說明 | 不可（1024 GPU）；relay 鏈式廣播機制可在 16–32 GPU 小規模驗證 → 部分 | https://arxiv.org/abs/2510.12633 （DOI：https://doi.org/10.1145/3767295.3803580，ACM 頁面 403 未驗證） |
| 13 | **LMCache: An Efficient KV Cache Layer for Enterprise-Scale LLM Inference** (arXiv 2025-10-08；Liu, Cheng, Yao 等，Tensormesh／UChicago) | KV cache 需搬離 GPU 以跨查詢與引擎重用，並支援 PD 分離的跨 GPU 傳輸 | 批次化資料搬移、計算與 I/O 管線、模組化 connector、pin/lookup/move/compress 控制 API；PD 傳輸以 NIXL 支援 | 8×H100 伺服器（GMI Cloud）；PD 分離 prefiller／decoder 實驗；文中引述「兩台 AMD 節點、8×Broadcom Thor-2 400G，傳輸量需 ≥16 MB 才飽和」為他人量測（§3.1.1） | vLLM（無 LMCache）、商業端點等（§評估 Baselines） | 與 vLLM 結合，多輪 QA／文件分析吞吐最高 15×（摘要） | 「Though not supported, the same dynamic offloading strategies can also be extended to other storage tiers beyong CPU and GPU.」（§3，double-buffer 說明處） | https://github.com/LMCache/LMCache（Apache-2.0） | **可**（8×H100 單機；PD 傳輸需 2 節點） | https://arxiv.org/abs/2510.09665 |
| 14 | **Helix Parallelism: Rethinking Sharding Strategies for Interactive Multi-Million-Token LLM Decoding** (arXiv 2025-07-07；Bhatia, More, Borkar 等，NVIDIA) | TP 寬度超過 KV head 數後 KV 重複；長 KV 的 DRAM 讀取隨批量線性增長 | KV parallelism（沿序列維切 KV）做 attention，之後同批 GPU 重組為 TP（稠密）或 TP×EP（MoE）做 FFN；HOP-B 以批次級 overlap 隱藏 KVP all-to-all | **GB200 NVL72、FP4，自家高保真模擬器**（非實機）（§5、§1 Fig.） | TP、PP、EP、vanilla KVP（Medha 式）的完整配置掃描 | DeepSeek-R1：固定批量 TTL 最多 -1.5×；同延遲預算下批量最多 32×（摘要） | 「One natural extension is to support sparse attention mechanisms such as Natively Sparse Attention (NSA) [20], which reduce KV read bandwidth but not overall memory capacity requirements.」（§6 Future work） | 論文未提；TensorRT-LLM 文件有 Helix 功能頁 https://nvidia.github.io/TensorRT-LLM/latest/features/helix.html（取自搜尋結果，未逐字核實內容） | 不可：GB200 NVL72 模擬 | https://arxiv.org/abs/2507.07120 |
| 15 | **Beyond the Buzz: A Pragmatic Take on Inference Disaggregation** (arXiv 2025-06-05；Mitra, Borkar, Bhatia 等，NVIDIA) | 何時、如何做 PD 分離缺乏系統性指引 | 以專有高保真 GPU 模擬器掃描數十萬設計點；並解析推導 KV 傳輸所需 egress/ingress 頻寬 | Blackwell 系統、FP4，**模擬**（§2）；DeepSeek-R1 等 | 非分離（piggybacked／同位）配置 | PD 分離在 prefill-heavy 流量與較大模型最有效；「Our analysis indicates that existing provisioned datacenter bandwidth is sufficient to support KV cache transfer without becoming a bottleneck.」（§5.1） | 「the impacts of KV cache reuse, speculation, inference-time compute techniques, and model architecture evolution appear to be particularly promising directions to pursue.」（§7 Future work）；模擬採固定 ISL/OSL 近似（以 P50 的 2 的冪），見附錄 C | 專有模擬器，未公開 | 不可：專有 Blackwell 模擬 | https://arxiv.org/abs/2506.05508 |
| 16 | **TokenWeave: Efficient Compute-Communication Overlap for Distributed LLM Inference** (arXiv 2025-05-16；Gond, Kwatra, Ramjee，Microsoft Research India；**MLSys 2026** oral) | TP 推論即使在 NVLink 上通訊仍占 ~20%；小 token 數下切分重疊反而變慢 | 融合 AllReduce–RMSNorm kernel（用 NVSHARP/multimem，僅 2–8 SM）、wave-aware token 分割，一份切片的計算與另一份的融合通訊重疊 | 8×H100 DGX（NVLink/NVSwitch）；vLLM 0.8.5；附錄含 8×B200 DGX（vLLM 0.14.1）；**使用 NVLS／multimem** | vLLM-Default、**vLLM-Multimem（NVLS 版 AllReduce）**、vLLM-nocomm（無通訊參考）、TileLink | 延遲最多 1.28×、吞吐最多 1.19×；多個設定優於「完全移除通訊」的等效模型（摘要） | 「Under small decode batches, the overhead introduced by splitting tokens into two parts offsets the performance gains obtained from compute-communication overlap.」「torch.compile does not support symmetric memory ... [full solution] does not support CUDA Graphs.」（附錄 C） | https://github.com/microsoft/tokenweave（GitHub API 授權：NOASSERTION） | **可**：8×H100 單機；開源 | https://arxiv.org/abs/2505.11329 |
| 17 | **Insights into DeepSeek-V3: Scaling Challenges and Reflections on Hardware for AI Architectures** (arXiv 2025-05-14；Zhao, Deng, Ruan 等，DeepSeek-AI；**ISCA '25 Industry Track**，DOI 10.1145/3695053.3731412) | 硬體對模型／系統共同設計的瓶頸（記憶體、算力、互連） | 描述 MLA、MoE、FP8、multi-plane 網路、node-limited routing、IBGDA；§2.3.2 以算術模型推 EP decode 上限 | 2,048× H800 訓練；§2.3.2 假設 CX7 400 Gbps IB 與 GB200 NVL72 做對比（解析，非實測） | 解析對比：400 Gbps IB vs NVL72（900 GB/s） | EP 兩次 all-to-all 每步 120.96 µs（IB）→ TPOT 上限約 14.76 ms（67 tok/s）；NVL72：6.72 µs → 0.82 ms（約 1200 tok/s）（§2.3.2；作者註明為理論上限） | 「this figure is purely theoretical and does not account for the substantial drop in GPU efficiency at small batch sizes; in real deployments, actual throughput will be significantly lower.」（§2.3.2） | 無（論文）；引用 open-source profiling data | 不可（實驗）；算術分析可複算 | https://arxiv.org/abs/2505.09343 |
| 18 | **MegaScale-Infer: Serving Mixture-of-Experts at Scale with Disaggregated Expert Parallelism** (arXiv 2025-04-03；Zhu, Jiang, Jin 等，ByteDance Seed／北大) | MoE decode 的 FFN 因稀疏啟動而計算利用率低 | attention 與 expert（FFN）分離到不同 GPU（attention 資料平行、FFN 專家平行）、ping-pong pipeline、自製 M2N 通訊庫取代 NCCL all-to-all | 叢集 1：8 節點 ×8 Ampere 80GB，機內 400 GB/s NVLink，8×200 Gbps IB；叢集 2：H20（900 GB/s NVLink、4×400G NIC）與 L40S（PCIe、2×400G NIC）（§評估 Testbed）；未用 NVLS | vLLM、TensorRT-LLM；NCCL（M2N 對比） | 每 GPU 吞吐最高 1.90×（vs SOTA）；M2N 比 NCCL 吞吐 4.2×、延遲 -68.2%；異質叢集每美元吞吐 1.7×（摘要） | 未說明（無獨立限制節）；§評估提到 baseline 受「certain implementation limitations in a multi-node environment」影響 | 未說明 | 不可：Ampere＋H20/L40S 叢集；程式未公開 | https://arxiv.org/abs/2504.02263 |
| 19 | **KVDirect: Distributed Disaggregated LLM Inference** (arXiv 頁面 v1 日期 2024-12-13，ID 為 2501.14743；Chen, Jiang, Yu 等，Rutgers／ByteDance) | 跨節點 PD 的 KV 傳輸以 message-passing 方式往返同步，頻寬利用低（4KB 區塊僅 1.8%，§3） | tensor-centric 通訊：decode 端以 RDMA「pull」模式讀取 prefill 端 KV；自製通訊庫支援動態增減 GPU | 傳輸微基準：2 張 H100 於兩節點、400 Gbps NIC（§5.3）；端到端實驗硬體敘述不完整（Mistral-Large-Instruct-2407 123B，隨機權重；§5.1） | vLLM（同硬體總 QPS 折半）；UCX message-based 傳輸；作者稱無公開可比的分散式 PD 系統 | 每請求延遲 -55%（arXiv）／-24%（ShareGPT）（§5.2.1）；頻寬 22.23 GB/s vs UCX（Fig.15） | 「We observe the underutilization of GPU memory on the prefill worker, especially with very long prompts ... we plan to use the idling memory as a prefix cache.」「we propose to occasionally switch the role of prefill and decode workers.」（§7 Discussion and Future Work） | 「is open-sourced and will be available upon publication」（§1）；網址未提供 | 部分：2 節點 H100＋IB 可測；程式未公開 | https://arxiv.org/abs/2501.14743 |
| 20 | **Context Parallelism for Scalable Million-Token Inference** (arXiv 2024-11-04；Yang, Yang, Ibrahim 等，Meta；**MLSys 2025**) | 長上下文 prefill 延遲 | 將 ring attention 改為推論用的 pass-KV／pass-Q 兩種精確變體，涵蓋 full prefill、persistent-KV prefill、decode；以啟發式選擇 | Meta Grand Teton：H100 8 卡 NVLink 每機（96GB HBM2e）；GTT 後端 RDMA 400 Gb/s/GPU；GTI 前端 TCP/IP 100 Gb/s/GPU；最多 16 節點／128 GPU；Llama3 405B（§4）；未用 NVLS | 多節點 TP（TP16/TP32）；CP 與 TP 比較 | 1M context prefill 77 s（93% 平行效率、63% FLOPS 利用率，16 節點）；128K prefill 3.8 s；CP vs 多節點 TP：2 節點差 ~15%，8 節點差達 100%（§4.2.2） | 無獨立 limitations；§4.2.2 說「For future GB200 with NVLink connecting multiple hosts, tensor parallelism can still benefits with reasonable scalability.」 | https://github.com/pytorch/FBGEMM/tree/main/fbgemm_gpu/experimental/gen_ai（論文腳註；授權 NOASSERTION） | 部分：CP2–CP4 與 TP 對比可在 4×H100 節點小規模複現；全規模不可 | https://arxiv.org/abs/2411.01783 |
| 21 | **No Request Left Behind: Tackling Heterogeneity in Long-Context LLM Inference with Medha**（v1–v4 原名 *Medha: Efficient LLM Inference on Multi-Million Context Lengths Without Approximation* / *...Efficiently Serving Multi-Million Context Length LLM Inference Requests Without Approximations*）(arXiv v1 2024-09-25，v5 2025-11-26；Agrawal, Qiu, Chen 等，GaTech／Microsoft) | 異質長度請求造成 convoy effect；長 prefill 阻塞短請求；長上下文 decode 慢 | 細粒度可搶佔排程＋Sequence Pipeline Parallelism（降 TTFT）＋KV-Cache Parallelism（降 TPOT）；chunk 可小到 40 token 仍近最佳 | 16×A100（2 台 DGX-A100）與 128 GPU（16 台 DGX-H100）；機內 NVLink、跨機 InfiniBand（§6） | LoongServe、vLLM 等 | 異質負載：吞吐 5.7×、中位延遲 -30×、P99 -174×（vs LoongServe，摘要）；128 H100 上 SPP 相對 CP 1M token prefill 延遲低 1.64×（§6） | 未說明（結論無限制或未來工作） | arXiv 頁面未列程式碼 | 不可：128 H100／16 A100 | https://arxiv.org/abs/2409.17264 |
| 22 | **Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving**（arXiv 2024-06-24；v5 2026-10-08；Qin, Li, He 等，Moonshot AI／清華；**會議版為 FAST '25 Best Paper，標題為「Mooncake: Trading More Storage for Less Computation — A KVCache-centric Architecture for Serving LLM Chatbot」**，作者增加 Cui、Ren） | Kimi 服務的 PD 分離與 KV cache 重用，含過載情境 | Conductor 全域排程（KV 快取感知）；利用 GPU 叢集閒置的 CPU／DRAM／SSD 做分散式 KV 快取；預測式提前拒絕；chunked pipeline parallelism 處理長上下文 | 每節點 8× A800-80GB SXM4（NVLink），RDMA NIC 合計最高 800 Gbps；使用 dummy 模型（LLaMA2-70B 架構）以保護專有資訊（§評估 Testbed） | vLLM | 模擬情境吞吐最高 +525%（SLO 內）；真實負載讓 Kimi 多處理 75% 請求（摘要） | 「The exploration of request-level prediction is left for future work.」（§過載排程）；Future Work：「we plan to explore ... the potential use of heterogeneous accelerators.」 | https://github.com/kvcache-ai/Mooncake（Apache-2.0；程式碼連結出自論文 §1） | 部分：Transfer Engine／Store 開源，2–4 節點 H100＋IB 可測；整體系統與 Kimi 負載不可 | https://arxiv.org/abs/2407.00079 （會議頁：https://www.usenix.org/conference/fast25/presentation/qin） |
| 23 | **DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving** (arXiv 2024-01-18；Zhong, Liu, Chen 等，北大／UCSD；**OSDI '24**) | prefill–decode 干擾及耦合的資源／並行配置 | 兩階段放在不同 GPU，依叢集頻寬放置；搜尋各階段最佳並行策略；低節點親和放置（KV 傳輸走節點內 NVLink） | 4 節點 32 GPU，每節點 8× A100-80GB SXM（NVLink），跨節點僅 25 Gbps（§6.1 Testbed）；未用 NVLS | vLLM、DeepSpeed-MII | 在 SLO 達標下可多服務 7.4× 請求或 12.6× 更嚴 SLO（摘要）；OPT-175B 上 KV 傳輸占總延遲 <0.1%，>95% 請求傳輸延遲 <30 ms（§6.3） | 「DistServe does not implement advanced runtime policies like preemption [26] and fault tolerance [58] ... We leave both as future work.」（§4.4 線上排程） | https://github.com/LLMServe/DistServe（論文 §1 腳註；Apache-2.0）；OSDI 頁面未列 repo，但標示 artifacts available/functional/reproduced 徽章 | **可**：NVLink 單機即可；開源 | https://arxiv.org/abs/2401.09670 （會議頁：https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin） |
| 24 | **Splitwise: Efficient Generative LLM Inference Using Phase Splitting** (arXiv 2023-11-30；Patel, Choukse, Zhang 等，UW／Microsoft；**ISCA 2024**，依 Microsoft Research 頁) | prompt 階段（算力密集）與 token 生成階段（記憶體密集）共置浪費 | 分兩池機器；KV cache 經 InfiniBand 傳輸，大 prompt 用逐層（per-layer）傳輸與計算重疊，小 prompt 用序列化傳輸 | DGX-A100 與 DGX-H100，InfiniBand（Table I：A100 200 Gbps／H100 400 Gbps 級）；真實 2 機驗證＋叢集模擬（Splitwise-AA/HH/HA/HHcap） | Baseline-A100、Baseline-H100（同質 DGX 叢集） | 同成本 1.4× 吞吐且成本低 20%；或同功耗 2.35× 吞吐（摘要）；KV 傳輸：序列化最多占 E2E 3%，Splitwise 僅 0.8%；第二個 token 延遲 +16.5% vs 序列化 +64%（Fig.15） | 「Since we do not have access to such hardware and/or optimized LLM implementations, we leave this to future work.」（Related Work §VIII 前的討論段，指 AMD MI-250／CPU 作為 token 機） | vLLM PR https://github.com/vllm-project/vllm/pull/2809；模擬器開源（論文 §VII 稱）；軌跡 https://github.com/Azure/AzurePublicDataset | 部分：2 節點 H100＋IB 可測；叢集模擬不需 GPU | https://arxiv.org/abs/2311.18677 |

### 非同儕審查的廠商／專案資料

| # | 資料 | 內容（事實） | 限制或 TODO（原文） | 程式碼 | 可重現？ | 網址 |
|---|---|---|---|---|---|---|
| V1 | **Deploying DeepSeek with PD Disaggregation and Large-Scale Expert Parallelism on 96 H100 GPUs**（SGLang Team，LMSYS 部落格，2025-05-05；**非同儕審查**） | 12 節點 ×8 H100（96 GPU，IB，NIC 型號未說明）；PD 分離＋DeepEP＋EPLB；KV 以 RDMA（支援 Mooncake、NIXL）；每節點 52.3k 輸入／22.3k 輸出 tokens/s（2000-token 輸入）；decode 9 節點 EP72 達 TP16 基準約 5.2× | 「Currently, our implementation supports only the NVIDIA Hopper architecture.」；序列長度受 96 GPU 限制；TTFT 2–5 s；MTP 尚未整合 DP attention | https://github.com/sgl-project/sglang/issues/6017（重現說明）、DeepEP（MIT） | 部分：96 GPU 遠超 32；縮小 EP 規模可測 | https://lmsys.org/blog/2025-05-05-large-scale-ep/ |
| V2 | **checkpoint-engine**（MoonshotAI，GitHub 專案，MIT；**非論文**） | RL 權重就地更新中介層，Broadcast 與 P2P（P2P 用 mooncake-transfer-engine）兩模式；Kimi-K2 1T 約 20 s（數千 GPU，README 自述）；Kimi-K2-Instruct FP8 16×H20 TP16 廣播 14.36 s／P2P 17.49 s；256×H20 廣播 16.04 s／P2P 16.75 s；Qwen3-235B（BF16）8×H800 廣播 6.22 s／P2P 7.10 s（vLLM v0.10.2rc1；P2P 僅涵蓋最多兩節點） | 「Tested only with vLLM and SGLang」；README 稱 paper 中的三階段管線尚未實作；FP8 需額外 vLLM patch | https://github.com/MoonshotAI/checkpoint-engine | 可：開源、2 節點即可 | https://github.com/MoonshotAI/checkpoint-engine |

---

## 3. 作者自述的限制與未來工作彙整（依出現次數）

（出現次數為本表 24 篇＋V1/V2 中，作者明文或明確以「leave／plan／future」表達者；一篇可屬多類。）

**A. 硬體或規模受限、其他硬體／加速器尚未涵蓋（約 9 篇）**
- #24 Splitwise：「Since we do not have access to such hardware and/or optimized LLM implementations, we leave this to future work.」
- #22 Mooncake：「we plan to explore ... the potential use of heterogeneous accelerators.」
- #1 PDD：「due to hardware availability, our RLD deployment was constrained to a 3-node H100 setup.」「our evaluation paired H100 with H200 GPUs.」
- #3 Lynx：「We leave such extensions to future work.」（延伸到 NVIDIA GPU）
- #8 SCIN：「its scale is limited.」（FPGA 原型，故改用模擬）
- #14 Helix：未來擴展至 sparse attention（NSA）；實驗僅為 GB200 NVL72 模擬
- #11 fabric-lib：GDA（GPU-initiated RDMA）尚未普及於雲端執行個體；新增 NIC 需逐硬體調校
- V1 SGLang：「Currently, our implementation supports only the NVIDIA Hopper architecture.」及序列長度受 96 GPU 限
- #20 CP：GB200 多機 NVLink 的 TP 行為列為未來硬體情境

**B. 缺乏搶佔／容錯／動態排程／線上適應（約 5 篇）**
- #23 DistServe：「does not implement advanced runtime policies like preemption and fault tolerance ... We leave both as future work.」
- #22 Mooncake：「The exploration of request-level prediction is left for future work.」
- #19 KVDirect：prefill/decode 角色互換、閒置記憶體當 prefix cache 為計畫中
- #9 PPD：「its lookup table is built offline and degrades gracefully but suboptimally when hardware or workload distributions drift」
- #10 NVRAR：「We plan to investigate and reduce idle time in future versions of NVRAR.」「We leave heuristic-based hyperparameter tuning to future work.」

**C. 小批量／小訊息／低並行度下方法效益消失，或與常用編譯／圖執行相容性不足（約 4 篇）**
- #16 TokenWeave：「Under small decode batches, the overhead introduced by splitting tokens into two parts offsets the performance gains」；torch.compile 不支援 symmetric memory、完整方案不支援 CUDA Graphs
- #2 SiFAR：正確性依賴「relative GPU progress is monotonic」假設，依賴 CUDA Graphs／Megakernels 消除 host 端變異
- #15 Beyond the Buzz：以固定 ISL/OSL（P50 近似）模擬；已指出小模型或 generation-heavy 流量下 PD 分離助益有限（§8 Conclusions）
- V1 SGLang：TTFT 2–5 s、MTP 未整合 DP attention

**D. 綁定特定框架或版本（約 4 篇）**
- V2 checkpoint-engine：僅測 vLLM／SGLang；FP8 需 patch
- #1 PDD：MTP 因 DRC 相容性被關閉
- #3 Lynx：僅 vLLM-Ascend
- #11 fabric-lib：僅支援 ConnectX 與 EFA

**E. 結果來自模擬／解析／投影而非實機（約 6 篇，屬「方法限制」而非作者自述）**
- #14 Helix、#15 Beyond the Buzz（專有模擬器）、#8 SCIN（cycle-level 模擬）、#6 PrfaaS（profiling 式吞吐模型）、#5 SparseRL-Sync（671B 為投影）、#17 DeepSeek-V3（理論上限，「purely theoretical」）

**F. 兩篇以上數字或結論互相不一致（列出，不裁決）**
- **KV 傳輸是否為瓶頸**：#15 Beyond the Buzz 結論「existing provisioned datacenter bandwidth is sufficient」（§5.1，Blackwell 解析模型）；#23 DistServe 實測 OPT-175B KV 傳輸 <0.1% 總延遲（A100，NVLink 節點內傳輸）；#24 Splitwise 實測序列化傳輸最多 3% E2E（IB）；但 #1 PDD 的 agent 負載傳輸占 e2e 43–56%（跨資料中心 Ethernet）、#3 Lynx 稱「KV transfer latency, not compute, becomes the dominant limiter」（§1，長上下文）。條件（機房內 vs 跨機房、模型大小、context 長度）不同。
- **#10 NVRAR 自身**：摘要「1.9×–3.6×，128 KB–2 MB，Slingshot 與 InfiniBand」，結論段「1.06×–1.92×（Slingshot）、1.14×–3.57×（InfiniBand），256 KB–2 MB」。
- **#12 Laminar vs #11 fabric-lib 的權重同步量級**：Laminar 稱 72B、128 relay 廣播 <1.6 s；fabric-lib 稱 1T 模型 256→128 GPU 1.3 s；V2 checkpoint-engine 稱 1T 約 20 s。硬體、模型大小、定義（廣播 vs P2P）皆不同。
- **#22 Mooncake**：arXiv v5 標題與 FAST '25 會議版標題、作者名單不同（arXiv 頁面未列 FAST 或最佳論文）。

---

## 4. 常見比較對象

- **vLLM**：幾乎所有系統層論文的 baseline（#23 DistServe、#22 Mooncake、#18 MegaScale-Infer、#19 KVDirect、#16 TokenWeave 以 vLLM 0.8.5 為底、#13 LMCache、#9 PPD、#10 NVRAR）。
- **TensorRT-LLM**：#18 MegaScale-Infer、#8 SCIN 的 profiling、#2 SiFAR 的 oneshot 對照（TRT-LLM oneshot）。
- **SGLang**：#10 NVRAR 比較引擎；V1 實作基礎；#7 TENT 用 SGLang HiCache 評估。
- **DeepSpeed-MII**：#23 DistServe。
- **DistServe／Splitwise／Mooncake**：作為 PD 分離「概念上的前驅」被後續工作（#1、#4、#6、#9、#19）當對照或同質 PD baseline（#4 稱 DistServe-Homo 為「strongest baseline」）。
- **NCCL（含 NCCL 2.30 auto、Tree/Ring）**：通訊層論文（#10 NVRAR、#2 SiFAR、#18 M2N 對 NCCL）。
- **NVLS／multimem all-reduce**：#16 TokenWeave 的 vLLM-Multimem、#8 SCIN 直接對 NVLS。
- **Mooncake Transfer Engine、NIXL、UCCL-P2P、UCX**：傳輸層論文（#7 TENT 三者；#19 KVDirect 對 UCX；#13 LMCache 使用 NIXL）。
- **DeepEP**：#11 fabric-lib 的 MoE dispatch/combine 對照；V1 使用。
- **verl、AReaL**：RL 框架對照（#12 Laminar）；V2 checkpoint-engine 為 #7 TENT 的測試載體。
- **公認最強對照**：文獻中**未見單一公認最強**；各子題各自為政（PD 分離：同質 DistServe 式 PD；TP all-reduce：NCCL 2.30／NVLS；RL 權重同步：checkpoint-engine／verl Rank0 廣播）。#16 TokenWeave 比較了「完全移除通訊」的理想上限（vLLM-nocomm），#2 SiFAR 亦以「移除 All-Reduce」作為上限。

---

## 5. 硬體覆蓋

**在 8× H100／H200（NVSwitch）單機，或 ≤ 4 台 IB／RoCE 叢集上有實測**
- #16 TokenWeave：8×H100 DGX（＋8×B200）
- #2 SiFAR：單機 8×H200
- #9 PPD：4×H100 NVLink 單節點
- #13 LMCache：8×H100 單機（PD 傳輸另有 prefiller/decoder 實驗）
- #8 SCIN：僅 profiling 在 8×H200（NVLS 開）；效益為模擬
- #24 Splitwise：DGX-H100／DGX-A100，2 機真實驗證（其餘模擬）
- #19 KVDirect：2 張 H100 跨 2 節點，400 Gbps NIC（微基準）
- #11 fabric-lib：8×H100（ConnectX-7 400G）、8×H200（EFA）；節點數未明列
- #7 TENT：2 節點 8×H800（200G RoCE）、4 節點 8×H20（400G RoCE）
- #5 SparseRL-Sync：8×H100 節點，實測 128 GPU（超過 4 台）
- #23 DistServe：4 節點 32 GPU，但為 A100、25 Gbps

**僅在 A100／H800／H20／AMD／GB200／模擬器／非 NVIDIA 加速器上做過**
- A100／A800：#23 DistServe、#22 Mooncake（A800）、#10 NVRAR（A100＋GH200）、#21 Medha（A100 與 128 H100）、#18 MegaScale-Infer（Ampere＋H20/L40S）
- H800／H20：#12 Laminar（1024 H800）、#17 DeepSeek-V3（H800）、#7 TENT（H800/H20，另含 GB200 NVL72、Ascend）、V2（H20/H800）
- GB200／Blackwell 模擬：#14 Helix、#15 Beyond the Buzz
- 其他加速器：#3 Lynx（Ascend NPU）、#4 HMA-Serve（Tenstorrent＋A100）、#8 SCIN（FPGA＋BookSim2 模擬）
- 規模大於 Nano5 可用上限：#1 PDD（600 GPU）、#6 PrfaaS（96 GPU 異質）、#20 CP（128 H100）、#21 Medha（128 H100）、V1（96 H100）

**在 NVLS 開／關之間比較**
- 在 24 篇＋V1/V2 全文以關鍵字（NVLS、multimem、NVLink SHARP）檢索：只有 #16 TokenWeave（vLLM-Default vs vLLM-Multimem，後者用 NVLS/multimem）、#8 SCIN（以 NVLS 為對照並於 H200 NVLS-on 下 profiling）、#2 SiFAR（使用 multimem.ld_reduce in-switch reduction，對 NCCL 2.30 auto 比較）明確涉及。**未見任何一篇明確報告「NCCL NVLS 設為開／關」的 A/B 數字**（SiFAR 的對照為各 all-reduce 演算法，不是 NVLS 開關）。#10、#20、#23 等未使用或未提 NVLS。

---

## 6. 開源工具與程式碼清單

授權欄來自 GitHub REST API（https://api.github.com/repos/<repo> 的 `license.spdx_id`，2026-11-16 查詢）；NOASSERTION 表示 API 無法判定，非「無授權」。

| 名稱 | 網址 | 授權 | 與論文 |
|---|---|---|---|
| DistServe | https://github.com/LLMServe/DistServe | Apache-2.0 | #23 |
| Mooncake（含 Transfer Engine、TENT、Store、EP） | https://github.com/kvcache-ai/Mooncake | Apache-2.0 | #22、#7 |
| TokenWeave | https://github.com/microsoft/tokenweave | NOASSERTION | #16 |
| LMCache | https://github.com/LMCache/LMCache | Apache-2.0 | #13 |
| pplx-garden（fabric-lib／TransferEngine） | https://github.com/perplexityai/pplx-garden | MIT | #11 |
| YALIS | https://github.com/axonn-ai/yalis | Apache-2.0 | #10 |
| NVRAR | https://github.com/hpcgroup/nvrar | MIT | #10 |
| FBGEMM gen_ai（CP 實作所在） | https://github.com/pytorch/FBGEMM | NOASSERTION | #20 |
| checkpoint-engine | https://github.com/MoonshotAI/checkpoint-engine | MIT | V2 |
| helix（SparseRL-Sync 聲明釋出處） | https://github.com/scitix/helix | Apache-2.0 | #5（是否已含本文程式未核實） |
| SGLang | https://github.com/sgl-project/sglang | Apache-2.0 | V1；#7 |
| vLLM | https://github.com/vllm-project/vllm | Apache-2.0 | 多篇 baseline／基底；Splitwise PR #2809 |
| DeepEP | https://github.com/deepseek-ai/DeepEP | MIT | V1、#11 |
| NIXL | https://github.com/ai-dynamo/nixl | NOASSERTION | #7、#13 |
| TensorRT-LLM | https://github.com/NVIDIA/TensorRT-LLM | NOASSERTION | 多篇 baseline；Helix 功能頁 |
| Azure Public Dataset（Splitwise 軌跡） | https://github.com/Azure/AzurePublicDataset | 未查 | #24 |

論文自述程式碼未公開或未說明者：#1、#2、#3、#4、#6、#8、#9、#12、#14、#15、#17、#18、#19（自稱發表後釋出）、#21（arXiv 頁面無連結）。

---

## 7. 搜尋紀錄

WebSearch（standard，除註明 extended）；每個查詢大致看前 10 筆結果標題／摘要，再以 WebFetch／curl 取原文核實。

| 查詢字串 | 約看筆數 | 結果／排除 |
|---|---|---|
| DistServe disaggregating prefill and decoding goodput-optimized LLM serving arXiv | 10 | 採用 #23 |
| Mooncake KVCache-centric disaggregated architecture LLM serving arXiv | 10 | 採用 #22 |
| Splitwise phase splitting LLM inference arXiv KV cache transfer | 10 | 採用 #24；Medium 二手說明不採用 |
| context parallelism long-context LLM inference prefill arXiv 2024 2025 ring attention H100 | 10 | 採用 #20 |
| RL post-training weight synchronization trainer to inference engine fast broadcast RDMA arXiv | 0（429 too_many_requests） | 改以後續查詢補 |
| low-latency all-reduce tensor parallel LLM inference decode small message NVLS arXiv | 10 | 採用 #2、#8、#10 |
| expert parallelism MoE decode disaggregated serving large-scale expert parallel inference arXiv 2025 MegaScale-Infer | 10 | 採用 #18；其他 MoE 主題屬 T3 |
| DeepSeek-V3 inference prefill-decode disaggregation large-scale expert parallelism SGLang 96 H100 GPUs | 10 | 採用 V1；Atlas Cloud 等第三方摘要不採用 |
| KV cache transfer layer-wise RDMA prefill decode disaggregation NVLink bandwidth bottleneck arXiv 2025 | 10 | 採用 #19、#11、#3、#15（另 #13）；LMCache 部落格（非 arXiv）不採用 |
| weight update RL rollout inference engines seconds trillion-parameter checkpoint-engine P2P arXiv | 10 | 採用 #7、V2；HackerNoon、Medium 等二手來源，及 NeMo-DCR（未找到可核實的 arXiv 頁面）未列入；Awex（僅部落格）未列 |
| Helix Parallelism KV parallelism TP decode long context arXiv NVIDIA | 10 | 採用 #14 |
| Medha million-token context serving KV cache parallelism arXiv | 10 | 採用 #21 |
| TokenWeave OR "NanoFlow" OR "Domino" tensor parallel inference communication overlap LLM serving arXiv | 10 | 採用 #16；NanoFlow（OSDI '25）為裝置內 overlap、非通訊重點，未列；Domino 無結果 |
| Insights into DeepSeek-V3: Scaling Challenges and Reflections on Hardware for AI Architectures arXiv | 10 | 採用 #17 |
| prefill-decode disaggregation heterogeneous KV cache transfer cross-cluster arXiv 2025（extended） | 10 | 採用 #1、#6、#4、#9；PrefillShare（2602.12029）、Prefill/Decode-Aware Evaluation on emerging accelerators（2606.17104）、HySparse2（2609.26368）、LMCache tech report PDF 僅見摘要，未逐篇核實，未列 |
| asynchronous RL post-training LLM weight synchronization communication NCCL broadcast bottleneck system arXiv 2025 2026（extended） | 10 | 採用 #12、#5；#Role-Based Fault Tolerance（2512.22492）已下載閱讀，其重點為容錯而非通訊優化，排除；DeepSeek-V4.1-Flash（2609.19969）與主題無關，排除；Hugging Face「Keep the Tokens Flowing」與 Medium 為部落格／二手，未列 |

額外直接核實（非搜尋）：各論文 arXiv abs 頁（WebFetch）、USENIX OSDI '24／FAST '25 頁、MLSys 2025／2026 頁、Microsoft Research Splitwise 頁、LMSYS 部落格、checkpoint-engine GitHub README；GitHub REST API 查授權。

**無法核實或未完成的項目**
- #12 Laminar 的 EuroSys '26 與 DOI 10.1145/3767295.3803580：ACM 頁面回傳 HTTP 403，僅有搜尋結果與 HKU 檔名佐證。
- #10 NVRAR 的 CAIS '26 DOI 頁面：dl.acm.org 403；venue 依 arXiv 頁 Comments 與 PDF 頁首。
- #1 PDD：arXiv ID 為 2609.13161，但 abs 頁顯示 v1 提交日期 2026-07-18（ID 前綴與日期不一致，已原樣記錄）。
- #12 Laminar、#18 MegaScale-Infer、#2 SiFAR、#8 SCIN 等未說明程式碼。
- #19 KVDirect 端到端實驗的 GPU 型號與節點數論文未明列，僅微基準提及 2 張 H100／400 Gbps NIC。
- #5 SparseRL-Sync 之 scitix/helix repo 是否已含論文程式，未開啟 repo 內容核實。
- #14 Helix 的 TensorRT-LLM 實作資訊僅來自搜尋結果摘要，未逐字核實。
- 本地 scratchpad 的 `pdf/` 目錄內另有他人下載的 PDF（非本任務），未讀取。
