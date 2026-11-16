# R02 / T1：單機集體通訊與通訊函式庫（intranode_collectives）

- **規格**：`docs/specs/S02_lit_map.md`　**撰寫**：Sonnet 5.5 執行者，2026-11-16
- **性質**：只列有出處的事實，不做選題判斷。每篇論文都已用 WebFetch 打開 arXiv／會議頁（或下載 PDF 以 pdftotext 讀）核實標題、作者、出處。
- **取得內文的方式**：多數用 arXiv HTML 全文；AutoCCL、HeteCCL、TE-CCL、SyCCL、TACCL 用 PDF 轉文字。ACM DL 頁面回 403，TE-CCL 的 SIGCOMM'24 出處改由 Microsoft Research 出版頁核實（見 P24）。
- **未核實的部分一律標「未說明」或「未核實」**。部分論文的 HTML 因長度被截斷，報告中「未說明」表示「我讀到的範圍內沒有」，不代表原文沒有。

---

## 1. 主題概況

近年 T1 範圍內的文獻可分為幾個分支（只描述，不評價）：

1. **函式庫內部與新介面**：NCCL 2.28 起提供 device-side API（LSA、Multimem、GIN 三種模式）與 symmetric memory（P13、P3）；NVSHMEM 與 MSCCL++ 提供 GPU-initiated、one-sided 的 primitive（P4、P11）；Meta 的 NCCLX 在 NCCL 之下加了自己的 transport 層 CTran（P16）。另有 SM-free（copy engine）通訊的 Zero-SM NCCL（P1）。
2. **單機（NVLink domain）延遲／頻寬優化的集體通訊 kernel**：one-shot／two-shot all-reduce、NVLS（multimem）、barrier-free 同步（P3、P11、P12、P15）；多節點小訊息 all-reduce 的 NVRAR（P14）；用 PCIe／RDMA 補 NVLink 頻寬的 FlexLink（P17）。
3. **演算法／排程合成**：TACCL、TE-CCL、SyCCL、ForestColl、HeteCCL 以 MILP／SMT／生成樹合成集體通訊排程（P5、P6、P23、P24、P27）。
4. **NCCL 調參與效能分析／建模／模擬**：AutoCCL、Lagom 調 NCCL 參數（P22、P10）；Demystifying NCCL 以文字分析 NCCL 協定與演算法（P19）；PICO、ATLAHS 做 benchmark／模擬（P18、P20）；Exploring GPU-to-GPU 做超級電腦互連量測（P25）。
5. **量測 GPU-initiated 通訊與 LLM 自動產生通訊 kernel**：Dissecting（P2）量 GPU→NIC 路徑；CUCo（P9）用 agent 產生通訊 kernel；SCIN（P8）提出交換器內運算架構並以 NVLS 為對照。另有壓縮（UCCL-Zip，P7）、PCIe A100 的 HFReduce（P26）、Triton-distributed 與 ParallelKittens 的 overlap 編程（P21、P15）。

本主題下的論文多數把 NCCL 當作 baseline；在 H100／H200 單機上同時比較 NVLS 開關的文獻很少（見第 5 節）。

---

## 2. 論文表（依年份新到舊）

> 欄位：硬體與規模、baseline、效益都只寫原文數字與條件。「可重現」依 S02 的資源（8×H100/H200 NVSwitch 單機、≤4 台 IB NDR、無 root）判斷。論文編號 P1–P27 供後續章節引用。
> MSCCL++（P11）與 TokenWeave（P12）的 arXiv 首版在 2025，但正式出處年份為 2026，故列在 2026。

### 2026

**P1　Zero-SM NCCL: A Systematic Guide to Designing and Deploying SM-Free GPU Communication**
- 作者／年／出處：Zhenhao He, Siyuan Shen, Ke Wen, …, Jeff Hammond, Torsten Hoefler；2026-11-13；arXiv:2611.18075 [cs.DC]（預印本，未註明會議）
- 問題：集體通訊 kernel 佔用 SM，與同時執行的 GEMM 互相拖慢。
- 方法：不啟動任何 SM kernel；用 CUDA stream memory operations 做同步、copy engine 做域內傳輸、stream-ordered CPU proxy 做 RDMA；提供 AllGather、AlltoAll 與 one-sided put／wait-signal（§III）。
- 硬體與規模：GB200 NVL72（18 節點×4 GPU，72 GPU 同一 NVLink domain）與 B200 NVL8（每節點 8 GPU，8×400 Gb/s NIC）；用到 NVLS multicast（§III-D1）；最多 72 GPU（微基準），端到端 32 或 64 GPU（§IV-A、Table I）。NCCL 2.32.3（含未釋出功能）。
- baseline：NCCL 舊式 ring kernel、NCCL symmetric kernel（ST 單播、STMC 多播）、GIN、VCCL、NCCLX AllGatherP、MSCCL++（§IV-B～D）。
- 效益：AllGather 頻寬最高 1.36×（abstract）；單 NVLink domain 4 GiB：copy engine 714–746 GB/s，對 symmetric 537–678、舊式 613–698 GB/s（§IV-B2）；GEMM 同時執行的中位數減速由 32.9%（舊式）降到 3.9%（§IV-C）；Megatron FSDP 最多快 3.5%、pipeline parallel 最多快 14.9%，但 B200 上 Qwen3-32B PP 慢 3.9%（§IV-D）。
- 作者自述限制：「Within a scale-up domain, Zero-SM NCCL has higher latency than SM kernels for small messages」（§V）；「Reduction collectives, such as AllReduce and ReduceScatter … are not yet supported」（§III-A）。1 KiB 單機 AllGather：copy engine 25–29 μs，SM kernel 12–18 μs（§IV-B2）。
- 程式碼：https://github.com/spcl/zero-sm-artifact（patch、benchmark、recipe；部分功能尚未進入已釋出的 NCCL，§III-D2）
- 可重現：**不可**（核心實驗需 GB200 NVL72／B200；H100 上的 copy-engine 部分未見報告）
- 出處網址：https://arxiv.org/abs/2611.18075

**P2　GPU-Initiated Communication: Dissecting Down to the Bone**
- 作者／年／出處：Javid Baydamirli, Ismayil Ismayilov, Kaan Oktay, Didem Unat；2026-10-01；arXiv:2610.01380（cs.DC/cs.NI/cs.PF；15 頁；未註明會議）
- 問題：GPU 直接對 NIC 發 RDMA（NVSHMEM、NCCL GIN、DeepEP）的提交路徑各段成本不明。
- 方法：實作兩個最小傳輸層 mini-gda（GPU 路徑）與 mini-proxy（CPU proxy 路徑），逐段量測並與函式庫比較（abstract）。
- 硬體與規模（Table 2）：P-IB 4 節點×4 H200、200 Gb/s IB；P-RoCE 2 節點×8 B200、400 Gb/s RoCEv2；P-H100 100 節點×4 H100、200 Gb/s IB；P-GB200 2 節點×1 GB200。NVLS：未說明（聚焦 GPU→NIC，非 NVLink）。
- baseline：NVSHMEM IBGDA／IBRC（3.4.5、3.7.2）、NCCL GIN GDAKI／Proxy（2.30.7、2.31.2）、DeepEP V1、UCCL-EP、MSCCL++ PortChannel、fabric-lib（§4）。
- 效益：單次 8 B 操作，mini-gda 發出 0.70 μs、完成 4.03 μs；NVSHMEM 公開 API 發出 5.31 μs（§4.1、Table 3）；all-to-all 流量在約 3,000 條連線時損失 59% 訊息率（abstract、§4.3.3）。
- 作者自述限制：「These results require a dedicated CPU worker pinned to a core on the NIC's NUMA node」（§4.1）；跨平台比較「although CPU, link, firmware, and RDMA provider all differ between the pairs」（§4.2）。無獨立限制章節。
- 程式碼：https://github.com/ParCoreLab/Dissecting-GPU-Communication-Experiments（授權：未說明）
- 可重現：**部分**（H100 節點×IB NDR 可重做 GPU→NIC 量測；題目屬跨機路徑，非單機 NVLink）
- 出處網址：https://arxiv.org/abs/2610.01380

**P3　Every Microsecond Matters: Achieving Near Speed-of-Light Latency in GPU Collectives**
- 作者／年／出處：Siyuan Shen, Anton Korzh, John Bachan, …, Jeff R. Hammond, Torsten Hoefler；2026-07-17；arXiv:2607.16100 [cs.DC]（未註明會議）
- 問題：集體通訊多為頻寬導向，延遲導向工作（長上下文 decode）依賴大量小訊息集體通訊。
- 方法：以 NCCL device-side API 建低延遲介面；原則為 barrier-free 同步與有效使用 symmetric memory／multicast；新增 NCCL symmetric 集體通訊（one-shot、two-shot、LL128 atomic）（§III–V）。
- 硬體與規模：GB200 NVL72（2～64 GPU，4 GPU/節點）；vLLM 用 4 GPU（TP=4）與 8 GPU（2 節點）；cuSOLVERMp 用 Alps（4×GH200/節點、單節點）（§VII-A、VIII）。NVLS：有；vLLM 實驗含「No MC」（關閉 NVLS multicast）設定（§VIII-A）。
- baseline：NCCL 舊式 ring／tree、NCCL symmetric kernel（v2.29.1）、NVSHMEM 3.5.21、MSCCL++ 0.8.0、vLLM custom AllReduce 0.15.1、NCCLX CTran（§VII-C）。
- 效益：4×GB200 小訊息 AllReduce 平均延遲 11.0 μs（NCCL ring）→ 2.37 μs（§I）；Llama-3.1-70B 4×GB200 ITL 降 8.7%；vLLM ITL 降 7–13%（4 GPU）、9–11%（8 GPU）（§VIII-A）；speed-of-light 下界 1.404 μs，約 7% 額外成本（2 GPU，§VI、VII-C）。
- 作者自述限制：「kernel selection is currently driven by empirical measurements」（§IX）；「the algorithm is non-deterministic because floating-point atomic ordering is not guaranteed」（§IV-5）。論文也說 multicast 在小規模可能有些許額外成本（Obs. 3，§VII-C；原文為摘要，未逐字引用）。
- 程式碼：https://github.com/ss16118/low-latency-nccl（§V 註腳 1）
- 可重現：**部分**（H100 NVSwitch 單機可跑 NCCL device API 與 NVLS；原實驗為 GB200，且需 NCCL ≥2.28）
- 出處網址：https://arxiv.org/abs/2607.16100

**P4　Demystifying NVSHMEM: A System-Level Analysis on Symmetric Memory and Device-Initiated Operations in GPU Communication**
- 作者／年／出處：Yijun Ma, Siyuan Shen, Tiancheng Chen, Akhil Langer, Jiri Kraus, Benjamin Glick, Craig Belusar, Jeff Hammond, Torsten Hoefler；2026-06-04；arXiv:2606.05951 [cs.DC]（未註明會議）
- 問題：NVSHMEM 的系統層級行為散在文件、原始碼與經驗中。
- 方法：分析 symmetric memory（建於 CUDA VMM API）、one-sided 操作與 device-side 集體通訊；以 DeepEP 為案例（abstract、§V–VIII）。
- 硬體與規模（§VII-A）：H200 SXM5 144GB、NVLink-4、支援 NVLS；每節點 8 GPU、8×ConnectX-7 IB；節點數未說明；NVSHMEM 3.3.9、CUDA 13.0.88；NCCL 版本未說明（用 nccl-tests）。
- baseline：NCCL Ring（強制）、NCCL NVLS（單機 `RSxLDMC_AGxSTMC` symmetric kernel；跨機 NVLS Tree）、NVSHMEM host-side `_on_stream` 與 device-side 單 CTA 路徑（§VII-C）。DeepEP 不重跑，引用 GIN 論文（P9）。
- 效益：單機 AllReduce：NVSHMEM on-stream 264 GB/s、NCCL NVLS 276 GB/s；小訊息（≤64 KiB）device 路徑約 3.8–7.1 μs、NCCL Ring 約 4.7–8.9 μs、NCCL NVLS 約 5.6–5.9 μs；跨機：NVSHMEM 兩種變體 <0.20 GB/s，NCCL 180 GB/s（Ring）與 252 GB/s（NVLS Tree）（§VII-C）。單機 bulk put 313 GB/s（NVLink 參考 450 GB/s，§VII-B）。
- 作者自述限制：內建集體通訊「limited multi-CTA support」被稱為弱點（§IX）；結論稱主要限制「come from its execution model and collective implementations」（§X）；NVSHMEM 集體通訊集中在 NVLS 型演算法，部分跨機量測逾時（§VII-C）。
- 程式碼：論文未給 NVSHMEM 倉庫網址（指向官方文件 https://docs.nvidia.com/nvshmem/api/api.html）；DeepEP：https://github.com/deepseek-ai/DeepEP；nccl-tests：https://github.com/NVIDIA/nccl-tests
- 可重現：**可**（H100／H200 單機 NVSwitch 可跑 NVSHMEM 與 nccl-tests；跨機部分需 IB 與 NVSHMEM 設定）
- 出處網址：https://arxiv.org/abs/2606.05951

**P5　HeteCCL: Synthesizing Near-Optimal Collective Communication Schedules for Heterogeneous GPU Clusters**
- 作者／年／出處：Chenyang Hei, Fuliang Li, …, Ennan Zhai（Northeastern Univ.、Alibaba Cloud 等）；NSDI '26，2026-05，pp. 2533–2551
- 問題：混合 GPU 叢集的集體通訊排程效率低、合成慢。
- 方法：把排程視為加權有向圖上的最大平行傳輸問題；以 SMT 公式編碼頻寬與路由限制，用 counterexample-guided inductive synthesis 剪枝（abstract）。
- 硬體與規模：32-GPU 測試床（H20 與 V100 混合）；另有同質 A100 與 A100 異質叢集（§6.3、附錄 G）；NVLS：未說明。
- baseline：NCCL、TACCL、TE-CCL。
- 效益：頻寬最高 2.8×（vs NCCL）、4.4×（vs TACCL）、2.6×（vs TE-CCL）；合成速度最高快兩個數量級；端到端訓練效率 +23%–37%（abstract）。
- 作者自述限制：直接用 SMT 解數千 GPU 的資料中心規模叢集「computationally intractable」（§7，原文為摘要）；以階層式組合處理規模。
- 程式碼：頁面未列
- 可重現：**不可**（需異質 H20／V100 叢集，且無公開程式碼）
- 出處網址：https://www.usenix.org/conference/nsdi26/presentation/hei ；PDF https://www.usenix.org/system/files/nsdi26-hei.pdf

**P6　ForestColl: Throughput-Optimal Collective Communications on Heterogeneous Network Fabrics**
- 作者／年／出處：Liangyu Zhao, Saeed Maleki, Yuanhong Wang, Zezhou Wang, Ziyue Yang, Hossein Pourreza, Arvind Krishnamurthy；arXiv:2402.06787（v1 2024-02，v4 2025-09-19）；NSDI '26，pp. 2067–2093
- 問題：任意拓樸（含交換器、直連）下產生吞吐最佳的集體通訊排程。
- 方法：建立廣播與聚合 spanning tree forest，多項式時間；把不支援 in-network 多播／聚合的交換器轉成等價的無交換器邏輯拓樸（abstract）。
- 硬體與規模：16 台 DGX H100（128 GPU，NVSwitch 450 GB/s/GPU、8 NIC×50 GB/s）；2 台 DGX A100；MI250（16/32 GPU）；排程生成最多 1024 GPU（僅生成）。NVLS：只在單機（1×8）H100 比較中使用，「ForestColl NVLS」在較小訊息較佳（v4 實驗章節）。
- baseline：NCCL、RCCL、TACCL、Blink+Switch、MultiTree；TE-CCL／SyCCL 僅作生成速度的理論比較；NCCL ring（MSCCL 實作）為對照。
- 效益：H100 128 GPU、1GB：allgather +32%、reduce-scatter +14%、allreduce +25%（但 NCCL tree 在小訊息、延遲敏感時較好）；A100 16 GPU、1GB：allgather +32%、reduce-scatter +30%、allreduce +26%；FSDP 訓練 iteration 時間降 14%–20%（A100，大模型）。
- 作者自述限制（§3）：「ForestColl prioritizes throughput over latency.」「ForestColl does not exploit network symmetry to speedup scheduling.」
- 程式碼：頁面與論文內容均未見 ForestColl 程式碼連結
- 可重現：**部分**（排程可在 H100 單機 NVSwitch 上以 MSCCL++ 執行；但無公開生成器，128 GPU 實驗不可）
- 出處網址：https://arxiv.org/abs/2402.06787 ；https://www.usenix.org/conference/nsdi26/presentation/zhao-liangyu

**P7　UCCL-Zip: Lossless Compression Supercharged GPU Communication**
- 作者／年／出處：Shuang Ma, Chon Lam Lao, Zhiying Xu, …, Ion Stoica, Yida Wang, Yang Zhou；2026-04-19（v2 04-21）；arXiv:2604.17172 [cs.DC/cs.AI]
- 問題：降低通訊量的方法多為有損壓縮，會影響收斂與精度。
- 方法：Uzip-P2P（split-send pipeline 重疊壓縮與傳輸）與 Uzip-NCCL（把壓縮融入 NCCL persistent kernel）（abstract）。
- 硬體與規模（§5.1、Table 2）：AWS p5en 2 節點×8 H200、16×EFA 200 Gbps；AMD 2 節點×8 MI355X；AWS g6e 單節點 8×L40S。NVLink 世代與 NVLS：未說明。NCCL 2.23.4。
- baseline：NCCL 2.23.4（限 4 SM）、UCCL-P2P、naive encode-send 等。
- 效益：RL 權重同步最高 +47.5%（GLM4-9B，214 MB）；vLLM 端到端最多降 10%；two-shot all_reduce 13.3%（32 MB）至 35.7%（1 GB）；ring all_reduce 反而比 NCCL 差（§5.2.2）。
- 作者自述限制（§6）：「We acknowledge that Uzip-NCCL over NVLink gives negative gains」；壓縮在 ≤1 MB 訊息關閉（§5.1）。
- 程式碼：https://github.com/uccl-project/uccl/tree/main/p2p
- 可重現：**部分**（H100 可跑壓縮 kernel；但原文指出 NVLink 上為負收益，且主要實驗在 EFA）
- 出處網址：https://arxiv.org/abs/2604.17172
- 備註：WebSearch 摘要曾把此編號誤標為「CCCL」，實際標題為 UCCL-Zip，以 arXiv 頁面為準。

**P8　A Switch-Centric In-Network Architecture for Accelerating LLM Inference in Shared-Memory Network（SCIN）**
- 作者／年／出處：Aojie Jiang, Kang Zhu, Zhiheng Zhang, Zhengxu Su, Juntao Liu, Yuan Du, Li Du；2026-03-30（v4 2026-08-04）；arXiv:2603.28239 [cs.AR]（未註明會議）
- 問題：NVLS 由 GPU 端驅動 in-network 運算，造成多餘資料搬移，且無法做 scale-aware 量化 All-Reduce。
- 方法：交換器內加速器主導 All-Reduce，並加入 dequantization–reduction–requantization 管線（abstract）。
- 硬體與規模（§3.6、§4.1）：4 顆 FPGA 模擬端點＋1 顆 FPGA 當交換器的原型；BookSim2 週期級模擬器，模擬 8 加速器 4 交換器（對應 DGX H200，900 GB/s）；模擬的 NVLS 與 NCCL 實測平均誤差 5.5%；LLaMA-2 與 Mixtral-8×7B 在 H200 上以 TensorRT-LLM v1.3.0 取得計算延遲。NVLS：模擬並作為主要 baseline。
- baseline：NVLS（8 GPU 與 4 GPU 設定）、ring 量化 All-Reduce（精度對照）。
- 效益（v4）：小訊息 All-Reduce 最高 1.8×、大訊息 1.34×（未量化）至 2.58×（INT8）over NVLS；FP8 TP=8 端到端最高 TTFT 1.42×、TPOT 1.12×（§4.3、4.5）；FPGA 原型 4 KB All-Reduce 延遲 3.34 μs（§3.6）。
- 作者自述限制：未設獨立限制章節；「SCIN could potentially be extended to multi-layer networks by adopting an approach similar to a reduction tree」（§3.5）。
- 程式碼：未說明
- 可重現：**不可**（需交換器內硬體或其 FPGA／模擬器；但其中的 NVLS 模擬模型可對照 H100 實測）
- 出處網址：https://arxiv.org/abs/2603.28239
- 數字版本差異：WebSearch 摘要曾引述同篇「最高 8.7×（小訊息）與 3.8×（大訊息）」；我直接讀取的 v4 摘要為「最高 1.8× 與 2.6×」，推測為不同版本，未能直接核對舊版。

**P9　CUCo: An Agentic Framework for Compute and Communication Co-design**
- 作者／年／出處：Yoga Sri Varshan Varadharajan, Bodun Hu, Saurabh Agarwal, Aditya Akella；2026-03-02（v2 2026-06-03）；arXiv:2603.02376 [cs.DC]
- 問題：計算與通訊的 co-design kernel 需要深度系統專業與硬體專屬調校。
- 方法：agent 框架，把 host 驅動通訊改寫成 device-initiated CUDA kernel；correctness-first agent 先產生基準，evolution-driven agent 再搜尋高效策略（abstract）。
- 硬體與規模（§4）：2 台伺服器、各 4×A100 80GB（NVLink 機內、RoCE 跨機），CUDA 13.1、NCCL 2.28.9，工作負載用 2 或 4 GPU；NVLS：未提及。
- baseline：host 驅動 NCCL、DeepEP（DeepSeek-V3 MoE layer）、消融版本。
- 效益：最高 1.57× over host-driven baseline（abstract）；對 host NCCL 延遲降 5.3%–36.2%（§4.1）；DeepSeek-V3 MoE layer（2 GPU）8.42 ms（DeepEP）→ 7.38 ms（§4.3、Table 5）。每個 workload 的 LLM API 成本 $4.98–$10.96（§4）。
- 作者自述限制：「co-design is less likely to help when compute dominates wall time」（§1）。無獨立限制章節。
- 程式碼：https://anonymous.4open.science/r/CUCo-06EC/（匿名連結，§1；是否為最終釋出：未核實）
- 可重現：**部分**（需 LLM API 費用；GPU 需求低，A100→H100 需重做搜尋）
- 出處網址：https://arxiv.org/abs/2603.02376
- 內部矛盾（原文）：摘要稱 1.57×，§4.2 稱 11.3% 為「largest speedup」，論文未說明差異。

**P10　Lagom: Unleashing the Power of Communication and Computation Overlapping for Distributed LLM Training**
- 作者／年／出處：Guanbin Xu, ZhenGuo Xu, Yuzhe Li, Youhui Bai, Ping Gong, Chaoyi Ruan, Cheng Li；2026-02-24；arXiv:2602.20656 [cs.DC]（6 頁，未註明會議）
- 問題：計算為瓶頸時，通訊參數（channel 數、chunk size）會拖慢重疊的計算，既有調參未考慮。
- 方法：統一 cost model 加優先權搜尋，共同調校通訊參數（abstract）。
- 硬體與規模（§4.1）：兩個 2 節點叢集、各 16×A40：A 為 NVLink 機內＋2×400 Gbps IB，B 為 PCIe 4.0＋100 Gbps IB；NCCL v2.18.3-1；NVSwitch／NVLS：未提及。
- baseline：NCCL v2.18.3-1、AutoCCL。
- 效益：1.07–1.33× over NCCL、1.03–1.27× over AutoCCL（abstract）；通訊競爭使計算最多降速 35%（§1）。
- 作者自述限制：無獨立限制章節；「The code for Lagom will be open-sourced soon.」（§1）。
- 程式碼：未釋出（論文僅稱將開源）
- 可重現：**部分**（方法可在 H100 重做，但無程式碼；結果與 A40 叢集相關）
- 出處網址：https://arxiv.org/abs/2602.20656

**P11　MSCCL++: Rethinking GPU Communication Abstractions for AI Inference**
- 作者／年／出處：Changho Hwang, Peng Cheng, Roshan Dathathri, …, Jithin Jose（Microsoft）；ASPLOS '26（2026-03-22～26，Pittsburgh）；arXiv:2504.09014 v4（v1 2025-04-11；標題由 "…for Cutting-edge AI Applications" 改為 "…for AI Inference"）
- 問題：既有集體通訊函式庫抽象無法滿足 AI 應用對細粒度、可融合通訊的需求。
- 方法：三層介面——Primitive API（put、signal、wait、flush；zero-copy、one-sided、asynchronous）、DSL、NCCL 相容的 Collective API（§3–4）。
- 硬體與規模（§5.3、Table 2）：A100-40G/80G（NVLink 3.0、HDR IB）、H100（8 GPU/節點、NVLink 4.0、單一 NVSwitch、CX7 400 Gb/s）、MI300x；單機與 2、4 節點；§7.3 用 2 台 H100（16 GPU）。NVLS：有；H100 上 2PA AllReduce 用 SwitchChannel（multimem），§7.2 報告比等效 MemoryChannel 版本高最多 56% 頻寬。
- baseline：NCCL 2.26.2、RCCL 2.20.5、MSCCL 2.23；端到端：vLLM（NCCL）、SGLang、IBGDA 版 DeepEP（§7）。
- 效益：集體通訊 geomean 1.7×（最高 5.4×）、AI 推論 1.2×（最高 1.38×）（abstract）；H100 單機 AllReduce 小訊息最高 2.8× over NCCL、大訊息 2.4×（§7.2）；SwitchChannel AllReduce 在 1 KB–1 GB 平均 >2.2× over NCCL／MSCCL（§7.4）；DeepSeek-V3 decode（SGLang、2 台 H100）1.31×（§7.3）。
- 作者自述限制：「MSCCL++ could be used for accelerating AI training workloads too, but we leave that for future work.」（§9）；DSL 版本平均比 Primitive 版差 3%，最差 18%（§7.1）。
- 程式碼：https://github.com/microsoft/mscclpp（授權：未說明）
- 可重現：**可**（H100 8 GPU NVSwitch 單機可直接安裝；2 節點實驗在資源範圍內）
- 出處網址：https://arxiv.org/abs/2504.09014

**P12　TokenWeave: Efficient Compute-Communication Overlap for Distributed LLM Inference**
- 作者／年／出處：Raja Gond, Nipun Kwatra, Ramachandran Ramjee（Microsoft Research India）；MLSys 2026；arXiv:2505.11329
- 問題：TP 推論的 AllReduce 與計算重疊在小 batch 有開銷。
- 方法：融合 AllReduce–RMSNorm kernel，利用 Multimem 與 NVSHARP（2–8 個 SM）；切分 token 做重疊（§4）。
- 硬體與規模（§5.1、附錄 C）：8×H100 DGX（NVSHARP）；TP-4 用其中 4 張；附錄 C 有 8×B200。NVLS：有；baseline 之一 vLLM-Multimem 也使用。
- baseline：vLLM-Default、vLLM-Multimem、vLLM-nocomm（理論上限）、TileLink、NanoFlow。
- 效益：延遲最高 1.28× over vLLM-Multimem（Llama-3.3-70B、8×H100）；吞吐最高 1.19×；融合 kernel 單獨相對序列 AllReduce+RMSNorm 1.34–1.39×（64–32K tokens，§4.2、Table 1）。
- 作者自述限制：「a net overhead for the TokenWeave full implementation in Mixtral at smaller sequence lengths」（§5.2.2）；「at present, torch.compile does not support symmetric memory allocation」（附錄 C.2）；完整方案不支援 CUDA Graph（C.3）。
- 程式碼：https://github.com/microsoft/tokenweave ；封存 https://doi.org/10.5281/zenodo.18844243
- 可重現：**可**（8×H100 NVSwitch 單機，與我們環境相同）
- 出處網址：https://arxiv.org/abs/2505.11329（註：此篇主題同時涉及 T2）

### 2025

**P13　GPU-Initiated Networking for NCCL**
- 作者／年／出處：Khaled Hamidouche, John Bachan, Pak Markthub, …, Manjunath Gorentla Venkata（均為 NVIDIA）；2025-11-19（v2 11-24）；arXiv:2511.15076 [cs.DC]（13 頁）
- 問題：NCCL 缺少 GPU 端發起、低延遲細粒度通訊（如 MoE）。
- 方法：NCCL 2.28 的 GIN，三層架構（host API、device API、網路 plugin），兩種後端 GDAKI（DOCA GPUNetIO）與 lock-free GPU→CPU proxy。NCCL 2.28 Device API 另含 LSA（NVLink/PCIe）與 Multimem（NVLink SHARP）模式（§III）。
- 硬體與規模（§V、Table III）：NVIDIA EOS，DGX H100（8×H100、NVLink 4、8×400 Gb/s IB），最多 8 節點（64 GPU）；NVLS：未使用於實驗。
- baseline：NVSHMEM 3.4.5（IBGDA、IBRC）、DeepEP 1.2.1、GIN Proxy。
- 效益：2 GPU 4–128 B 往返延遲 GDAKI 16.7 μs、IBRC 16.0 μs、IBGDA 24.3 μs、Proxy 18.0 μs（§V-A）；DeepEP LL dispatch 單機 NVLink 啟用：185.28 GB/s／40.62 μs（GIN）vs 182.15 GB/s／41.43 μs（NVSHMEM）（§V-C）；多數指標在 NVSHMEM 的 1–2% 內。
- 作者自述限制：「the current implementation in NCCL 2.28 enforces symmetric sizes」（§III-A）；GIN 實作仍在開發，計畫批次 WQE 等（§V-D）；Proxy 後端以可攜性換峰值性能（§III-C）。
- 程式碼：論文未給專屬倉庫；為 NCCL 2.28 的一部分（NCCL 產品頁 https://developer.nvidia.com/nccl）
- 可重現：**可**（EOS 為 DGX H100，與我們同型號；跨機部分在 ≤4 台範圍內；需 NCCL ≥2.28.7 與支援 GDAKI 的 NIC）
- 出處網址：https://arxiv.org/abs/2511.15076

**P14　LLM Inference Beyond a Single Node: From Bottlenecks to Mitigations with Fast All-Reduce Communication（NVRAR）**
- 作者／年／出處：Prajwal Singhania, Siddharth Singh, …, Abhinav Bhatele；arXiv:2511.09557 v3（2025-12-15；標註「Preliminary work. Under review」）；亦有 CAIS '26 版本（檔名 2026-05-yalis-nvrar-cais.pdf，未直接打開核實）
- 問題：多節點 TP decode 的小訊息 all-reduce 延遲高。
- 方法：以 NVSHMEM 實作階層式 recursive all-reduce（NVRAR），整合進 YALIS 推論框架（§4–5）。
- 硬體與規模（§3.2、Table 1）：Perlmutter（A100、4 GPU/節點、NVLink 3、Slingshot-11）、Vista（GH200、1 GPU/節點、IB）；70B 用 4–32 GPU、405B 用 16–128 GPU。NVLS：未提及。
- baseline：NCCL all-reduce、vLLM V0／V1、SGLang、Cray-MPICH。
- 效益：microbenchmark 相對 NCCL：Perlmutter 64–128 KB 為 0.7–0.8×、>256 KB 為 1.13–1.92×；Vista 最高 3.5×（256 KB–1 MB）；YALIS 405B 端到端 1.17–1.72×（§5）。
- 作者自述限制：「We plan to investigate and reduce idle time in future versions of NVRAR.」（§5.2）；「In the future, we will explore the use of NVRAR in other latency-bound applications.」（§7）。
- 內部不一致（原文）：加速倍率在摘要（1.9–3.6×）、緒論（1.9×／2.8×）、結論（1.06–1.92×／1.14–3.57×）互不相同；端到端最高值為 1.72×（摘要）、1.9×（緒論）、1.92×（結論）。
- 程式碼：YALIS https://github.com/axonn-ai/yalis ；NVRAR https://github.com/hpcgroup/nvrar
- 可重現：**部分**（NVSHMEM 在 H100 可用；原實驗為 A100／GH200 與 Slingshot）
- 出處網址：https://arxiv.org/abs/2511.09557

**P15　ParallelKittens: Systematic and Practical Simplification of Multi-GPU AI Kernels**
- 作者／年／出處：Stuart H. Sul, Simran Arora, Benjamin F. Spector, Christopher Ré；2025-11-17；arXiv:2511.13940 [cs.DC/cs.LG]
- 問題：多 GPU 計算＋通訊重疊 kernel 難寫。
- 方法：基於 ThunderKittens 的 8 個核心 primitive 與統一 template；分析 copy engine／TMA／register 指令三種 NVLink 傳輸機制（§3.1）。
- 硬體與規模（§4、附錄 A）：8×H100 80GB SXM（NVLink 4/NVSwitch、CUDA 12.6、PyTorch 2.8.0）；8×B200（附錄 A）。NVLS：使用 `multimem.ld_reduce`／`multimem.red`（§3.1.2；附錄 F）。
- baseline：cuBLAS＋NCCL、Triton Distributed、Flux、CUTLASS、Comet、xDiT、YunChang（§4）。
- 效益：data／tensor parallel 1.06–1.68× over 非重疊 baseline；Ring Attention 1.07–4.08× over xDiT；EP 0.92–1.22× of Comet（§4）。傳輸機制（Table 1，1 GB、全部 SM）：H100 copy engine 368.82 GB/s（82%）、TMA 350.01（78%）、register 342.68（76%）；register 指令需 3.2–5.1× 於 TMA 的 SM 數才能飽和 NVLink（§3.1.2）；intra-SM 同步約 64 ns、inter-SM 經 HBM 約 832 ns（§3.1.3）。
- 作者自述限制（§5）：「As this work focuses on intra-node execution」；延伸到跨機屬未來工作（轉述）。
- 程式碼：https://github.com/HazyResearch/ThunderKittens（授權：未說明）
- 可重現：**可**（8×H100 NVSwitch，與主要實驗相同）
- 出處網址：https://arxiv.org/abs/2511.13940（註：同時涉及 T2）

**P16　Collective Communication for 100k+ GPUs（NCCLX）**
- 作者／年／出處：Min Si, Pavan Balaji, James Hongyi Zeng 等約 40 人（Meta）；arXiv:2510.20171（v1 2025-10-23，v4 2026-01-09）；未註明會議；各版本作者清單不同
- 問題：支援 Llama 4 訓練／推論的 10 萬 GPU 等級集體通訊。
- 方法：以 NCCL 為基礎的 NCCLX；transport 層 CTran（NVLink、IB/RoCE、socket 後端）；NVL CopyEngine 做無 SM 傳輸（§4.2、§5.2）。
- 硬體與規模：H100、8 GPU/節點；推論實驗最多 32 節點（256 GPU）；RoCE/IB 三層 Clos；NVLS：未提及。
- baseline：NCCL（copy-based 與 zero-copy send/recv、AllReduce）；NVSHMEM、DeepEP 僅在文中討論未量測。
- 效益：Llama4 訓練穩態 step 延遲最多降 12%（§1）；96K 規模初始化最多快 11×；FTAR AllReduce 用一半 thread block 達 NCCL 同等，同 thread block 數時延遲低 9–18%（§5.5）；Llama4 Maverick decode AllToAll 端到端改善 15%–80%（§1、Table 3）。
- 作者自述限制：「leave the rest of device-initiated model support as future work」（§3）；CUDA Graph 的 tensor registration pooling 尚未在預訓練啟用（§4.3.1）。
- 程式碼：https://github.com/meta-pytorch/torchcomms/tree/main/comms/ncclx
- 可重現：**部分**（元件可在 H100 試用；規模與 Meta 叢集不可複製）
- 出處網址：https://arxiv.org/abs/2510.20171

**P17　FlexLink: Boosting your NVLink Bandwidth by 27% without accuracy concern**
- 作者／年／出處：Ao Shen, Rui Zhang, Junping Zhao（Ant Group）；arXiv:2510.15882 [cs.AR]（頁面標記日期 2025-08-30）；未註明會議
- 問題：NVLink 頻寬不足（H800 為 400 GB/s）。
- 方法：NCCL API 相容，把部分流量分到 PCIe 與 RDMA NIC（§1、§4）。
- 硬體與規模（§5.1）：8×H800；測 2、4、8 GPU；PCIe 5.0 x16；ConnectX-6 NIC；NCCL 2.27.3、nccl-tests。NVLS：未提及。
- baseline：NCCL 2.27.3。
- 效益：8 GPU H800 上 AllReduce 最高 +26%、AllGather +27%（abstract）；2 GPU 256 MB AllReduce 175 vs 139 GB/s；8 GPU AllReduce 256 MB 僅 +2%，作者歸因 ring 的 2(N−1) 步延遲（§5.3、Table 2）。
- 作者自述限制（§6）：「the current RDMA implementation, which relies on the NVSHMEM CPU API, is suboptimal」；未來計畫「extend FlexLink to support a broader range of communication primitives, such as AllToAll」。
- 程式碼：未說明
- 可重現：**部分**（H100 NVLink 頻寬較高，PCIe 補償比例不同；無程式碼）
- 出處網址：https://arxiv.org/abs/2510.15882

**P18　PICO: Performance Insights for Collective Operations**
- 作者／年／出處：Saverio Pasqualoni, Lorenzo Piarulli, Daniele De Sensi（Sapienza）；2025-08-22；arXiv:2508.16809 [cs.DC]
- 問題：集體通訊 benchmark 工具缺乏跨函式庫、步驟級洞察。
- 方法：輕量、可擴充的 benchmark 框架，含 tracing，可產生函式庫調參檔（§3）。
- 硬體與規模（§4.1）：Leonardo、LUMI、MareNostrum 5（MPI 為主）；GPU 型號、NVLink、GPU 數：未說明；最多 2048 節點 Allreduce。NVLS：未提及。
- baseline：MPI 函式庫預設演算法選擇 vs ring、recursive doubling、Rabenseifner 等；OSU、IMB、NCCL Tests、CommBench 僅定性討論（§5）。
- 效益：預設演算法選擇常比最佳低約 30–40%，最差僅達 1/5（§4.1）；UCX_MAX_RNDV_RAILS 2→4 在 2048 節點 Rabenseifner 有 >2× 差異（§4.3）。
- 作者自述限制：「Today, PICO enables users to create these tuning artifacts manually.」（§4.1，後接計畫「integrate an automated workflow」）。
- 程式碼：頁面未列網址
- 可重現：**部分**（概念可用；原實驗為 CPU／MPI 超級電腦，本文未報告 NCCL 數據）
- 出處網址：https://arxiv.org/abs/2508.16809

**P19　Demystifying NCCL: An In-depth Analysis of GPU Communication Protocols and Algorithms**
- 作者／年／出處：Zhiyi Hu, Siyuan Shen, Tommaso Bonato, Sylvain Jeaugey, Cedell Alexander, Eric Spada, James Dinan, Jeff Hammond, Torsten Hoefler；arXiv:2507.04786 v3（v3 2026-03-02）；HOTI '25（2025-08，依搜尋摘要中的 HOTI'25 頁面；未直接打開該頁核實）
- 問題：NCCL 內部協定與演算法缺乏系統文件。
- 方法：以 NCCL 2.19.1 分析 API／channel、Simple／LL／LL128 協定、資料傳輸模型、ring 與 tree 演算法（§II–V）。
- 硬體與規模（§V-F）：Alps 上 16 節點 GH200，機內 150 GB/s、Slingshot 25 GB/s/方向。NVLS：不在範圍內——「this paper will not include analyses of NVLS and CollNet」（§V-A）。
- baseline：協定與演算法互比，無與其他函式庫的數值比較。
- 效益（定性結論）：跨機 LL／LL128 在 <64 KiB 最佳；機內 LL128 在大訊息約比 Simple 慢 5%；ring 大訊息佳、tree 小訊息佳（§V-F）。ATLAHS 模擬誤差 <5%（§VI）。
- 作者自述限制：分析基於 NCCL 2.19.1，後續版本細節可能改變（§I）；選擇定性而非定量複雜度模型（§V-E）；「we believe that future versions of NCCL will need to support automated algorithm selection, robust failure handling」（§VII）。
- 程式碼：NCCL https://github.com/NVIDIA/nccl（引用 [1]）；ATLAHS 倉庫見 P20
- 可重現：**可**（概念與分析；實驗為 GH200，單機 H100 可自行驗證）
- 出處網址：https://arxiv.org/abs/2507.04786

**P20　ATLAHS: An Application-centric Network Simulator Toolchain for AI, HPC, and Distributed Storage**
- 作者／年／出處：Siyuan Shen, Tommaso Bonato, Zhiyi Hu, Pasquale Jordan, Tiancheng Chen, Torsten Hoefler；2025-05-13；arXiv:2505.08936 [cs.DC]
- 問題：以應用 trace（含 NCCL 集體通訊）驅動網路模擬。
- 方法：追蹤應用，轉成 GOAL 排程，在多種網路模擬器上重播（abstract）。
- 硬體與規模（§5.1、Table 1）：AI trace 取自 CSCS Alps（GH200，機內 150 GB/s、Slingshot）；驗證 4–256 GPU。NVLS：未提及。
- baseline：AstraSim 2.0（SimAI 因原始碼未完整公開而未比較）。
- 效益：AI 與 HPC 驗證誤差 <5%；4 節點案例 5.50 s vs AstraSim 76.63 s（13.9× 快）（§5.2）。
- 作者自述限制（§7）：「Due to the nature of the GOAL format, ATLAHS currently does not support dynamically scheduled communication operations.」；不捕捉跨 stream 的 CUDA kernel 相依性，可能影響重疊精度（§7）。
- 程式碼：https://github.com/spcl/atlahs.git（§3.1.2 註腳 1）
- 可重現：**部分**（模擬器可在登入節點之外運作；驗證資料取自 GH200 叢集，無 H100 校準）
- 出處網址：https://arxiv.org/abs/2505.08936

**P21　Triton-distributed: Programming Overlapping Kernels on Distributed AI Systems with the Triton Compiler**
- 作者／年／出處：Size Zheng, Wenlei Bao, Qi Hou, …, Xin Liu；2025-04-28（v3 2025-06-05）；arXiv:2504.19442 [cs.DC]
- 問題：分散式 AI 重疊 kernel 難以編寫。
- 方法：在 Triton 編譯器加入 OpenSHMEM 相容通訊 primitive，編譯器共同優化計算、記憶體存取與通訊（abstract）。
- 硬體與規模（§3.5、§4）：H800（8 GPU/節點、NVSwitch；NVLink 約 170–200 GB/s）、L20（PCIe）、AMD MI308X；8–64 GPU（§4.2 提到 128 GPU 比較 DeepEP）。NVLS：論文未稱 NVLS，但低延遲 AllGather 用 PTX `multimem` 做 NVLink 廣播（§3.4）。
- baseline：PyTorch+NCCL／RCCL、FLUX、cuBLAS／CUTLASS、DeepEP、NVSHMEM fcollect。
- 效益：相對 NCCL／RCCL 1.09×–44.97×（§1）；機內 AG+GEMM 1.42× vs PyTorch+NCCL、1.09× vs FLUX（8×H800，§4.1）；8×L20 低延遲 AllGather 3.11× vs NCCL（§4.2）。
- 作者自述限制：「This indicates that a dedicated ReduceScatter kernel is required for MoE+RS to achieve better performance, which is left for future.」（§4.2）；「To further improve our performance, we need to use IBGDA to re-implement our kernel.」（§4.2）。
- 程式碼：https://github.com/ByteDance-Seed/Triton-distributed
- 可重現：**部分**（H100 與 H800 同架構、NVLink 頻寬更高，可重做；NVSHMEM bitcode 無 IBGDA 支援）
- 出處網址：https://arxiv.org/abs/2504.19442（註：同時涉及 T2）

**P22　AutoCCL: Automated Collective Communication Tuning for Accelerating Distributed and Parallel DNN Training**
- 作者／年／出處：Guanbin Xu, Zhihao Le, Yinhe Chen, Zhiqi Lin, Zewen Jin, Youshan Miao, Cheng Li；NSDI '25（2025-04，Philadelphia），pp. 667–683
- 問題：NCCL 參數（演算法、協定、transport、channel 數、thread 數、chunk size）預設值非最佳，且與計算互相干擾。
- 方法：將實作相關參數與搜尋空間大小相關參數分離，divide-and-conquer 加子空間 coordinate descent；線上調校並把成本藏在訓練前幾個 iteration（abstract、§5）。
- 硬體與規模（§4）：2 節點 16×A40（機內 NVLink、2×400 Gbps IB）與 4 節點 32×A40（機內 PCIe、100 Gbps）；基於 NCCL 2.18.3；NVSwitch／NVLS：未提及。
- baseline：NCCL v2.18.3-1 與另一個 NCCL tuner。
- 效益：microbenchmark 1.24–1.29× over NCCL、1.15–1.22× over 另一 tuner；有並行計算時最高 1.80×、1.49×；端到端每 iteration 1.07–1.32×（abstract）。
- 作者自述限制（§8.1）：「further exploration of their optimization is left for future work」（IB 等 transport 專屬參數，認為預設值已近最佳）；調參可能導致失敗（§8.2，原文為摘要，未逐字引用）。
- 程式碼：https://github.com/gbxu/autoccl（PDF 內文所列，§4／參考文獻）
- 可重現：**部分**（程式碼基於 NCCL 2.18.3，H100 與新版 NCCL 需移植）
- 出處網址：https://www.usenix.org/conference/nsdi25/presentation/xu-guanbin ；PDF https://www.usenix.org/system/files/nsdi25-xu-guanbin.pdf

**P23　SyCCL: Exploiting Symmetry for Efficient Collective Communication Scheduling**
- 作者／年／出處：Jiamin Cao, Shangfeng Shi, …, Ennan Zhai（Alibaba Cloud、Tsinghua）；ACM SIGCOMM '25（2025-09-08～11，Coimbra）；DOI 10.1145/3718958.3750499
- 問題：MILP 型合成器（TE-CCL、TACCL）在生產規模搜尋空間爆炸。
- 方法：利用集體通訊與拓樸對稱性，把需求拆成較小子需求，分別合成子排程再整合（abstract）。
- 硬體與規模（§7、Figure 13）：32-A100 測試床（4 伺服器）；H800 叢集（64 伺服器、每台 8×H800、8×400 Gbps NIC，部分實驗縮為 4 GPU×6 台）；另有生產規模模擬。NVLS：未提及。
- baseline：NCCL、TECCL（TE-CCL）、TACCL。
- 效益：集體通訊性能最高 +127%、合成時間縮短 2–4 個數量級（abstract）；32-A100 上最高 +91% vs TECCL、最高 +6.3% vs NCCL（§1）。
- 作者自述限制（§8）：「SyCCL does not ensure that the synthesized schedules are optimal.」；多租戶、拓樸頻繁變動或鏈路降級的叢集「SyCCL may be less effective」。
- 程式碼：論文內寫 https://github.com/aliyun/symccl ；WebSearch 結果顯示倉庫為 https://github.com/aliyun/syccl（兩者拼法不同，我未開啟倉庫確認）；授權 Apache-2.0 與使用 SCIP 取代內部 solver 的說明來自搜尋摘要，未核實。
- 可重現：**部分**（排程合成為純 CPU；H100 單機 NVSwitch 的實測驗證需自行接 MSCCL++／NCCL XML）
- 出處網址：https://ennanzhai.github.io/pub/sigcomm25-syccl.pdf

### 2024

**P24　Rethinking Machine Learning Collective Communication as a Multi-Commodity Flow Problem（TE-CCL）**
- 作者／年／出處：Xuting Liu, Behnaz Arzani, Siva Kesava Reddy Kakarla, Liangyu Zhao, Vincent Liu, Miguel Castro, Srikanth Kandula, Luke Marshall；ACM SIGCOMM '24（2024-08）；DOI 10.1145/3651890.3672249（出處由 Microsoft Research 出版頁核實；ACM DL 頁回 403）；arXiv:2305.13479 v1（2023-05-22，內文以 PDF 讀取）
- 問題：既有集體通訊排程器（SCCL、TACCL）無法擴展或解品質不佳。
- 方法：以 traffic engineering 的 multi-commodity flow 建模，加入 in-network copy 與時間擴展網路，MILP 與 LP／A* 變體（§3–5）。
- 硬體與規模：以 DGX1／DGX2／NDv2 拓樸與兩個內部拓樸做模擬；實體原型僅單一 8-GPU DGX1 驗證延遲估計；NVLS：未提及。
- baseline：TACCL、SCCL、RCCL。
- 效益：作者 GPU 測試床上 algorithm bandwidth 為 TACCL 的 2.14×、RCCL 的 3.18×（MSR 出版頁摘要）。
- 作者自述限制：「We do not have access and the budget to run hardware experiments at scale on different kinds of GPUs. Thus, the effect of factors such as congestion, message batch sizes and other GPU implementation artefacts on the collective latency remains an unknown.」（arXiv v1「Unexplored avenues」段，節號未標）
- 程式碼：PDF 內未見網址
- 可重現：**部分**（MILP 排程可在 CPU 求解；GPU 實測部分需自行實作）
- 出處網址：https://www.microsoft.com/en-us/research/publication/rethinking-machine-learning-collective-communication-as-a-multi-commodity-flow-problem/

**P25　Exploring GPU-to-GPU Communication: Insights into Supercomputer Interconnects**
- 作者／年／出處：Daniele De Sensi, Lorenzo Pichetti, …, Torsten Hoefler；arXiv:2408.14090 v2（2024-11-15）；SC '24
- 問題：量測超級電腦的機內與跨機 GPU 互連性能。
- 方法：在 Alps、Leonardo、LUMI 比較 host-staged copy、device copy、NCCL/RCCL、GPU-aware MPI 的 p2p 與集體通訊（alltoall、allreduce）（§III–VI）。
- 硬體與規模（§II）：Alps（4×GH200/節點、NVLink 4.0、Slingshot-11，用 512 節點）；Leonardo（4×A100、NVLink 3.0、IB HDR）；LUMI（4×MI250X、Slingshot-11）；最多 4,096 GPU。不含 NVSwitch 機型（H100 SXM 8 卡）。
- baseline：各通訊機制互比；NCCL/RCCL 與 GPU-aware MPI。
- 效益：調參最多約 10× 改善；LUMI 設 NCCL_NCHANNELS_PER_PEER=32 使機內 p2p 提升 3.5×；跨機小傳輸 MPI 比 *CCL 最高快一個數量級；網路雜訊使 Leonardo 在 1,024 GPU 的 allreduce goodput 最多降 50%。
- 作者自述限制：「It is worth remarking, however, that Alps is currently under deployment and still affected by some instability.」（§V-C）；「Addressing this problem would thus require improvements in the adaptive routing algorithm.」（§VI-B）。
- 程式碼：https://zenodo.org/doi/10.5281/zenodo.13312325（benchmark 程式、腳本與資料）
- 可重現：**部分**（benchmark 可在 H100 節點重跑；原系統為 GH200／A100／MI250X）
- 出處網址：https://arxiv.org/abs/2408.14090

**P26　Fire-Flyer AI-HPC: A Cost-Effective Software-Hardware Co-Design for Deep Learning**
- 作者／年／出處：Wei An, Xiao Bi, Guanting Chen 等 52 人；2024-08-26（v2 08-31）；arXiv:2408.14158；SC '24 接受（頁面註記）
- 問題：以較低成本建造 AI-HPC 叢集。
- 方法：HFReduce——在 CPU 先做機內 reduction，再以 double binary tree 做跨機 allreduce（§IV）；HaiScale、3FS、HAI-Platform。
- 硬體與規模（§III-A、V-B1）：每節點 8×PCIe A100、1×200 Gbps ConnectX-6；後期加 NVLink Bridge（兩兩 600 GB/s）；10,000 GPU 叢集；allreduce 測試 16–1,440 GPU。NVLS：不適用。
- baseline：NCCL。
- 效益：186 MiB allreduce 的跨機頻寬 HFReduce 6.3–8.1 GB/s、NCCL 1.6–4.8 GB/s；加 NVLink 後 >10 GB/s（§IV-B2、IV-C）。
- 作者自述限制：HFReduce 理論上限約 12 GB/s，實測「slightly over 8GB/s」，歸因於 EPYC 的 PCIe root complex 約 37.5 GB/s 上限（§IV-D3）。
- 程式碼：僅 HAI Platform：https://github.com/HFAiLab/hai-platform；HFReduce 程式碼：未說明
- 可重現：**不可**（硬體為 PCIe A100 無 NVSwitch；無 HFReduce 程式碼）
- 出處網址：https://arxiv.org/abs/2408.14158

### 2023

**P27　TACCL: Guiding Collective Algorithm Synthesis using Communication Sketches**
- 作者／年／出處：Aashaka Shah, Vijay Chidambaram, Meghan Cowan, Saeed Maleki, Madan Musuvathi, Todd Mytkowicz, Jacob Nelson, Olli Saarikivi, Rachee Singh；NSDI '23（2023-04，Boston），pp. 593–612
- 問題：為給定硬體合成集體通訊演算法，搜尋空間過大。
- 方法：以 communication sketch 縮小搜尋空間；三步驟：relaxed routing、heuristic ordering、contiguity 與 exact scheduling（§9）。
- 硬體與規模：DGX-2（含 NVSwitch）與 Azure NDv2（V100）；最多 2 節點 32 GPU 的評估；NVLS：不適用（V100 世代）。
- baseline：NCCL、SCCL。
- 效益：演算法最高比 NCCL 快 6.7×；Transformer-XL 與 BERT 訓練加速 11% 至 2.3×（USENIX 頁面摘要）；8 台 DGX-2（128 GPU）合成約 11 小時（§9）。
- 作者自述限制（§9）：合成技術「still based on solving an NP-hard problem」；「As a future work, we would like to scale TACCL further by hierarchically composing synthesized algorithms.」
- 程式碼：https://github.com/microsoft/taccl（PDF 內註腳 1）
- 可重現：**部分**（可在 CPU 合成；V100 拓樸需改寫為 H100 NVSwitch 並以 MSCCL 執行）
- 出處網址：https://www.usenix.org/conference/nsdi23/presentation/shah ；arXiv:2111.04867

---

## 3. 作者自述的限制與未來工作彙整（依出現次數歸類）

**A. 小訊息延遲／延遲與吞吐的取捨（8 篇）**
- P1：「Within a scale-up domain, Zero-SM NCCL has higher latency than SM kernels for small messages」（§V）
- P6：「ForestColl prioritizes throughput over latency.」（§3）
- P14：microbenchmark 64–128 KB 為 0.7–0.8×（慢於 NCCL）（§5.1）；idle time 略高於 NCCL，「We plan to investigate and reduce idle time」（§5.2）
- P17：8 GPU AllReduce 僅 +2%，歸因 ring 延遲（§5.3）
- P7：≤1 MB 訊息關閉壓縮（§5.1）
- P21：LL protocol 使訊息加倍，不適合大訊息（§3.4，轉述）
- P3：barrier 每次 >1 μs，兩次 barrier 約佔 ~5 μs 小 AllReduce 的 40%（§III-B；為問題陳述，非限制）
- P19：LL／LL128 跨機大訊息性能劇降（§V-F）

**B. 範圍限單機／NVLink domain，或跨機路徑弱（6 篇）**
- P15：「As this work focuses on intra-node execution」（§5）
- P4：NVSHMEM 跨機 AllReduce <0.20 GB/s，NCCL 180–252 GB/s（§VII-C）；集體通訊集中於 NVLS 型演算法（§VII-C）
- P7：「We acknowledge that Uzip-NCCL over NVLink gives negative gains」（§6）
- P11：訓練工作負載留待未來（§9）
- P16：device-initiated 模型支援留待未來（§3）
- P2：GPU→NIC 路徑與共享佇列延遲增加一至三個數量級（abstract；為量測發現）

**C. 調參／kernel 選擇靠經驗或手動，缺少效能模型（5 篇）**
- P3：「kernel selection is currently driven by empirical measurements」（§IX）
- P18：「Today, PICO enables users to create these tuning artifacts manually.」（§4.1）
- P22：transport 專屬參數的優化「left for future work」（§8.1）
- P19：「future versions of NCCL will need to support automated algorithm selection」（§VII）
- P14：計畫以啟發式調 block／chunk size（附錄 B，轉述）

**D. 合成式演算法的規模與最佳性（5 篇）**
- P27：NP-hard，「scale TACCL further by hierarchically composing」（§9）
- P23：「does not ensure that the synthesized schedules are optimal」（§8）
- P5：SMT 在數千 GPU 規模不可行，改用階層式組合（§7）
- P6：不利用網路對稱性加速排程（§3）
- P24：缺乏大規模硬體實驗，擁塞與 GPU 實作因素「remains an unknown」（arXiv v1）

**E. 功能缺口與工具鏈相容性（5 篇）**
- P1：不支援 reduction collectives（§III-A）；CUDA graph capture 成本更高（§V）；scale-out proxy 佔一條 CPU thread（§V）
- P12：不支援 CUDA Graph（C.3）；torch.compile 不支援 symmetric memory 配置（C.2）
- P3：LL128 atomic 非決定性（§IV-5）
- P13：NCCL 2.28 強制 symmetric window 大小（§III-A）；GIN 實作仍在開發（§V-D）
- P8：多層網路擴充僅屬構想（§3.5）

**F. 程式碼或功能尚未公開（4 篇）**
- P1：pipelined rail schedule 與 tuner 尚未進入已釋出的 NCCL（§III-D2）
- P10：「The code for Lagom will be open-sourced soon.」（§1）
- P9：僅有匿名程式碼連結（§1）
- P5、P6：頁面均未列程式碼

**G. 測試環境或樣本限制（4 篇）**
- P19：僅 NCCL 2.19.1，不含 NVLS／CollNet／PAT（§I、§V-A）
- P25：Alps 仍在部署、有不穩定（§V-C）
- P2：跨平台比較時 CPU、link、韌體與 RDMA provider 皆不同（§4.2）
- P4：微基準為挑選而非全面；僅針對 NVSHMEM 3.3.9（§X 附近，轉述）

---

## 4. 常見比較對象

- **NCCL 是絕大多數論文的 baseline**：P1、P3、P4、P5、P6、P7、P8、P9、P10、P11、P12、P13、P14、P16、P17、P21、P22、P24（RCCL）、P26、P27 均與 NCCL（或 RCCL）比較。版本橫跨 2.18.3（P10、P22）到 2.32.3（P1），**各論文所用的 NCCL 版本不同**，且 NVLS／symmetric kernel 是否啟用多半未明說。
- **NCCL 較新的 baseline 型態**：NCCL symmetric kernel（ST／STMC，P1、P3）、NCCL NVLS 與 NVLS Tree（P4）、vLLM custom AllReduce（P3）、vLLM-Multimem（P12）。
- **其他通訊函式庫**：MSCCL／MSCCL++（P1、P3、P11）、NVSHMEM（P1、P2、P3、P13、P21）、NCCLX／CTran（P1、P3）、DeepEP（P2、P9、P11、P13、P21）、MPI（P14、P18）。
- **合成式演算法類互比**：TACCL（P5、P6、P23、P24、P27）、TE-CCL（P5、P6、P23）、AutoCCL（P10 的 baseline）。
- **「公認最強對照」**：論文自身的說法不一致。P3 的 baseline 組合（NCCL symmetric、NVSHMEM、MSCCL++、vLLM custom AllReduce、NCCLX）是我讀到範圍內最廣的集合；P11 以 NCCL 2.26.2 與 MSCCL 為對照。是否有公認最強對照：**未說明**，不做裁決。

---

## 5. 硬體覆蓋

**在 8×H100／H200 NVSwitch 單機，或 ≤4 台 IB 叢集上做過實驗：**
- P11 MSCCL++：H100 8 GPU 單機與 2、4 節點，用 NVLS（SwitchChannel）
- P12 TokenWeave：8×H100 DGX（NVSHARP）；附錄 8×B200
- P15 ParallelKittens：8×H100 NVSwitch；附錄 8×B200
- P4 Demystifying NVSHMEM：8×H200 NVSwitch 節點（節點數未說明），含 NVLS
- P6 ForestColl：單機 1×8 H100（含 NVLS）；16 節點 H100 屬 128 GPU
- P13 GIN：DGX H100，1～8 節點（含 1 節點、2 節點的比較）
- P2 Dissecting：P-IB 4 節點×4 H200（每節點僅 4 GPU，非 8 卡 NVSwitch）
- P21 Triton-distributed：H800（8 GPU NVSwitch；H800 的 NVLink 頻寬低於 H100）
- P17 FlexLink：8×H800（同上）
- P16 NCCLX：H100 8 GPU/節點，單機與 32 節點

**只在 A100、AMD、GB200、GH200 或模擬器上做過：**
- A100／A40：P9（A100 2 節點×4）、P10、P22（A40）、P23（A100 與 H800 叢集）、P26（PCIe A100）、P14（A100 與 GH200）
- GH200：P19、P20、P25（Alps）
- GB200／B200：P1、P3
- 模擬器：P8（BookSim2 + FPGA 原型）、P20（ATLAHS）、P24（以拓樸模擬為主）
- AMD：P25（MI250X）、P21（MI308X）、P11（MI300x）、P7（MI355X）
- V100／H20：P27、P5

**有在「NVLS 開與關之間比較」的（讀到範圍內）：**
- P11 MSCCL++：H100 上 SwitchChannel（multimem）對 MemoryChannel，最高 +56% 頻寬（§7.2）
- P3：vLLM 設定含「No MC」（關閉 NVLS multicast）；§VII-C Observation 3 稱 multicast 小規模可能有些許額外成本，「並未報告獨立的 on/off 延遲表」（GB200）
- P6 ForestColl：單機 H100 比較「ForestColl NVLS」與非 NVLS 版本
- P4：NCCL NVLS（symmetric kernel）對強制 NCCL Ring（H200 單機 AllReduce）
- P12：vLLM-Multimem 對 vLLM-Default（H100）
- P1：NCCL symmetric 的 ST（單播）與 STMC（多播）（GB200）
- P8：模擬的 NVLS 與其提議架構比較（非實機開關）
- 在 H100 單機把 NVLS 開關作為唯一變因並系統量測的論文：**我讀到的範圍內沒有找到**

---

## 6. 開源工具與程式碼清單

| 名稱 | 網址 | 授權 | 來源論文 |
|---|---|---|---|
| MSCCL++ | https://github.com/microsoft/mscclpp | 未說明 | P11 |
| low-latency-nccl（Every Microsecond） | https://github.com/ss16118/low-latency-nccl | 未說明 | P3 |
| Zero-SM NCCL artifact | https://github.com/spcl/zero-sm-artifact | 未說明 | P1 |
| Dissecting-GPU-Communication-Experiments（mini-gda、mini-proxy） | https://github.com/ParCoreLab/Dissecting-GPU-Communication-Experiments | 未說明 | P2 |
| NCCLX／torchcomms | https://github.com/meta-pytorch/torchcomms/tree/main/comms/ncclx | 未說明 | P16 |
| TokenWeave | https://github.com/microsoft/tokenweave | 未說明 | P12 |
| ThunderKittens／ParallelKittens | https://github.com/HazyResearch/ThunderKittens | 未說明 | P15 |
| Triton-distributed | https://github.com/ByteDance-Seed/Triton-distributed | 未說明 | P21 |
| NVRAR | https://github.com/hpcgroup/nvrar | 未說明 | P14 |
| YALIS | https://github.com/axonn-ai/yalis | 未說明 | P14 |
| AutoCCL | https://github.com/gbxu/autoccl | 未說明 | P22 |
| ATLAHS | https://github.com/spcl/atlahs.git | 未說明 | P20 |
| UCCL（含 Uzip-P2P） | https://github.com/uccl-project/uccl/tree/main/p2p | 未說明 | P7 |
| TACCL | https://github.com/microsoft/taccl | 未說明 | P27 |
| SyCCL | https://github.com/aliyun/symccl（論文內拼法）／ https://github.com/aliyun/syccl（搜尋結果拼法） | Apache-2.0（僅來自搜尋摘要，未核實） | P23 |
| Exploring GPU-to-GPU artifact | https://zenodo.org/doi/10.5281/zenodo.13312325 | 未說明 | P25 |
| HAI Platform（Fire-Flyer） | https://github.com/HFAiLab/hai-platform | 未說明 | P26 |
| NCCL（含 GIN、Device API） | https://github.com/NVIDIA/nccl ；產品頁 https://developer.nvidia.com/nccl | 未說明 | P13、P19 |
| nccl-tests | https://github.com/NVIDIA/nccl-tests | 未說明 | P4、P17 |
| DeepEP | https://github.com/deepseek-ai/DeepEP | 未說明 | P2、P4、P13 |
| NVSHMEM 文件 | https://docs.nvidia.com/nvshmem/api/api.html | 未說明 | P4 |
| CUCo（匿名） | https://anonymous.4open.science/r/CUCo-06EC/ | 未說明 | P9 |
| SimAI（Alibaba）、ASTRA-sim | 搜尋結果提及，**未核實**；ASTRA-sim 有 HGX-H100 NCCL 驗證頁 https://astra-sim.github.io/astra-sim-docs/validation/hardware/gpu-validation-hgx-h100.html（搜尋摘要稱 all-reduce 8×H100 NVSwitch 幾何平均誤差 12.01%，未打開核實） | 未說明 | — |

**無程式碼連結**：P5 HeteCCL、P6 ForestColl、P8 SCIN、P10 Lagom（承諾開源）、P17 FlexLink、P18 PICO（頁面未列）、P24 TE-CCL（PDF 內未見）。

---

## 7. 搜尋紀錄

**工具狀態**：arXiv MCP 與 paper-search MCP 在本 session 連線失敗（uvx 路徑不存在），改用 WebSearch／WebFetch。WebFetch 對 arXiv PDF 常回壓縮二進位，改抓 `arxiv.org/html/<id>`；對 USENIX／ACM 的 PDF，下載後以 pdftotext 轉文字（暫存於 session scratchpad，不進 repo）。

**WebSearch 查詢（約 30 次，每次回傳約 10 筆結果）：**

| 查詢（縮寫） | 模式 | 結果用途 |
|---|---|---|
| NVLS NVLink SHARP multimem all-reduce H100 arXiv | standard | 取得 TokenWeave、MSCCL++、SCIN、Fire-Flyer |
| MSCCL++ GPU communication abstractions arXiv | standard | 確認 MSCCL++ 多版本與 ASPLOS'26 |
| NCCL performance model H100 NVSwitch … 2025 | standard | PICO、ASTRA-sim 驗證頁；未找到 H100 專屬 NCCL 效能模型論文 |
| symmetric memory NVSHMEM PyTorch all-reduce … | standard | 僅得 PyTorch 文件頁，未列入（非論文） |
| collective … synthesis TACCL TE-CCL | standard | TACCL、HeteCCL |
| Demystifying NCCL … | standard | P19 |
| NCCL … H100 H200 GB200 characterization … | extended | Zero-SM NCCL、Dissecting、GIN |
| small message all-reduce latency LLM inference … | standard | NVRAR、Every Microsecond、NCCLX |
| NCCL simulator … ATLAHS SimAI | standard | ATLAHS；SimAI 為二手資訊，未核實故未列 |
| NVSHMEM GPU-initiated … evaluation | standard | Demystifying NVSHMEM、GROMACS NVSHMEM halo exchange（SC Workshops '25，arXiv:2509.21527；應用導向，**排除**） |
| NCCL device API LSA symmetric memory 2.28 | standard | GIN 論文與 NCCL 文件 |
| NCCL auto-tuning … | standard | AutoCCL、Lagom、TU Wien 論文（thesis／workshop，**排除**） |
| NVLink SHARP in-network reduction … 2026 | standard | SCIN；NVIDIA MultiShot 部落格（廠商，未列表） |
| one-shot two-shot all-reduce … vLLM … | standard | Every Microsecond；QuickReduce（AMD 部落格，非論文，**排除**） |
| collective library … NCCL alternative … | extended | FlexLink、UCCL-Zip（搜尋摘要誤標為 CCCL） |
| Meta NCCLX … Llama4 | standard | P16 |
| LLM-generated collective kernels … | standard | CUCo；ParallelKernelBench（2026-06，arXiv ID 未能核實，**排除**）；CommBench（部落格，**排除**）；Ditron（arXiv:2605.02953，主題偏 T2，**排除**，未打開核實） |
| ForestColl … | standard | P6 |
| TE-CCL … SIGCOMM 2024 | standard | P24 |
| Hopper H100 NVLink characterization … | standard | Exploring GPU-to-GPU（P25） |
| Demystifying NVSHMEM arXiv 2606.05951 | standard | 確認 |
| Triton-distributed / TileLink / Iris / Mercury … | standard | Triton-distributed；Iris（arXiv:2511.12500，AMD 為主，**排除**，未打開核實） |
| NCCL PAT / NVLS / symmetric kernels … 2026 | extended | NCCL EP（arXiv:2603.13606，主題為 T3，**排除**）、NCCL 發行說明（文件，不列表；摘要稱 2.31.2 的 PAT+NVLS 在 H100 有性能退化，**未打開核實**） |
| GPU collective kernel NVLink … copy engine TMA … | standard | Syncopate（arXiv:2601.20595，T2，**排除**）、ParallelKittens |
| SyCCL … SIGCOMM 2025 | standard | P23 |
| GB200 NVL72 … characterization … | standard | 無新論文；NVIDIA 多節點 NVLink 調參指南為文件 |
| NCCL profiling tracing … | standard | PICO；ncclsee、NCCL Inspector、碩士論文（非同儕審查或廠商，**排除**） |

**主要排除原因**：非論文（廠商部落格、文件、個人部落格）、學位論文或 workshop 摘要、偏他主題（T2、T3、T4、T5）、無法取得內文核實（Iris、ParallelKernelBench、Ditron 等僅看過搜尋摘要）。

**WebSearch 摘要與原文不符的紀錄**（顯示核實的必要性）：
1. arXiv:2604.17172 的搜尋摘要稱為「CCCL」，實為 UCCL-Zip。
2. SCIN 的搜尋摘要數字（8.7×／3.8×）與 v4 摘要（1.8×／2.6×）不同。
3. Demystifying NCCL 作者清單：搜尋摘要未含 James Dinan，HTML v3 含。

**核實層級**（供 Opus 引用 BibTeX 前參考）：
- 已打開 arXiv abs 頁：P1、P2、P3、P4、P7、P8、P9、P10、P13、P15、P21、P25、P26。
- 已打開 arXiv HTML／PDF 全文（title／authors 由全文確認，abs 頁未另開）：P5（USENIX 頁）、P6、P11、P12、P14、P16、P17、P18、P19、P20。
- USENIX 頁面與 PDF：P5、P6、P22、P27。
- SIGCOMM 論文以 PDF 全文（P23）與 Microsoft Research 出版頁（P24）核實；ACM DL 頁面 403，**DOI 頁未直接核實**。
- HOTI'25 的出處（P19）僅來自搜尋摘要，**未打開 HOTI 頁**。
- NVRAR 的 CAIS '26 版本（P14）僅見搜尋結果檔名，**未打開**。
