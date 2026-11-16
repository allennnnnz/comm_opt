# R02 / T2 overlap_fusion：計算與通訊重疊／融合 文獻地圖

- **撰寫**：Sonnet 5.5 執行者，2026-11-16（依 `docs/specs/S02_lit_map.md`）
- **範圍**：2024-01 至 2026-11-16（另含 1 篇 2024-04 的 Centauri 與 1 篇 2024-01 的 T3 作為起點）。只寫事實與原文，不做選題判斷。
- **核實方式**：每篇都開過 arXiv abs 頁（取標題、作者、日期、comments 欄）並下載 PDF 以 `pdftotext` 讀全文（硬體、baseline、限制皆取自全文；PDF 暫存在 scratchpad，未進 `papers/`，因 `papers/` 目錄寫入異常）。Centauri 以 Crossref DOI API 加作者網站 PDF 核實。MPK 的 OSDI 出處見表中註記。引用格式（BibTeX）留給 Opus 用 arXiv MCP／DOI 匯出。
- **注意**：本次 session 的 `arxiv`、`paper-search` MCP 連線失敗（uvx 找不到），因此全程改用 WebSearch／WebFetch／curl 直接讀 arXiv 與 Crossref。
- 表中「引文」為原文（英文）逐字摘錄；「未說明」＝我在全文中找不到。

## 1. 主題概況

近年「計算與通訊重疊／融合」主要分成幾條做法分支：（a）**單一 kernel 內 tile 粒度融合**（Flux、Comet、ParallelKittens、FlashMoE、mKernel、Syncopate），把集體通訊寫進 GEMM 的 epilogue／prologue 或用專用 SM 角色，而不是在 stream 層級拆 kernel；（b）**以 compiler／DSL 產生分散式 kernel**（Triton-distributed、TileLink、Syncopate、Mercury 於 SOSP'25 但未找到 arXiv 版，見 §7）；（c）**不改 GEMM 的訊號式重疊**（FlashOverlap：由 GEMM 發訊號，再呼叫 NCCL）；（d）**算子融合與 NVLS／multimem 利用**（TokenWeave 的 AllReduce+RMSNorm、ParallelKittens 的 in-network reduction、Runway 的 FP8 GEMM+AllReduce、CAIS 的 compute-aware in-switch）；（e）**改模型架構換取可重疊性**（Ladder-Residual、FarSkip-Collective）與**batch 切片／排程層重疊**（Domino、Centauri、Lagom），以及（f）**硬體／模擬器路線**（T3、CAIS、FiCCO）與**特性量測**（Lee et al. 2025 量到重疊造成的計算變慢）。論文實驗硬體集中在 8×H800／H100 單機 NVLink；跨機實驗多為 2 台 16 GPU；A100、AMD、模擬器各佔一部分（見 §5）。

## 2. 論文表（依年月新到舊）

欄位：# ｜ 標題／作者／年份／出處 ｜ 解決的問題 ｜ 方法要點 ｜ 硬體與規模 ｜ 比較對象 ｜ 報告的效益 ｜ 作者自述限制（引文＋位置） ｜ 程式碼 ｜ 本環境可重現？ ｜ 出處網址

| # | 標題／作者／年份／出處 | 問題 | 方法要點 | 硬體與規模 | baseline | 效益（附條件） | 作者自述限制（引文＋位置） | 程式碼 | 可重現？ | 網址 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **Runway Characters: Serving a 57B Video Diffusion Transformer for Real-Time Conversation**；Agarwal et al.；2026-11-03；arXiv preprint（industry 論文） | 即時影片擴散模型的 TP-all-reduce 佔步驟時間 | 自寫 FP8 GEMM+all-reduce fused kernel，用 NVLS `multimem.ld_reduce`／`multimem.st`，支援奇數 TP | 4×H100（TP-2~4 在 4 卡機，TP-5~8 在 8 卡單機）；也提 Blackwell；NVSwitch；NVLS 有用 | NCCL all-reduce；NVLS multimem all-reduce（單獨替換 NCCL） | 整體 diffusion step 159.3→147.4 ms（約 1.63× 對原始 240.9 ms，不含 fused 為 1.51×）；單獨把 NCCL 換成 NVLS all-reduce 每步只省 0.4 ms | "workers ran out of memory binding NVLS multicast buffers; reverting to NCCL was a configuration change (§8)"；另 "multimem.ld_reduce has no FP8 form on Hopper" | 未說明（本文未見網址） | 部分：H100 單機可；其 kernel 未公開 | https://arxiv.org/abs/2611.05642 |
| 2 | **mKernel: Fast Multi-GPU, Multi-Node Fused Kernels**；Mao, Zhang, Chew, Ma, Raiciu, Zhou, Shenker, Stoica；2026-09-11；arXiv preprint | 既有 fused kernel 侷限單一 NVLink domain | 持久 kernel 內 SM 分成計算／通訊角色，GPU 上控制器動態調 SM 分配；階層式資料搬移；以 RDMA verbs＋host proxy 驅動網路（IBGDA 幾乎無額外好處） | 2 台×8 H200（16 GPU）×2 個叢集：ConnectX-7 InfiniBand 與 AWS EFA | cuBLAS／FlashAttention 未融合、Triton-distributed、Flux、Mercury、DeepEP+DeepGEMM 等共 7 個 | 最高 1.72× GEMM+AllReduce、1.88× Ring Attention（2×8 H200）；MoE dispatch+GEMM 在 EFA 上 3.1–4.5× | "Due to testbed limitations, we are only able to test adaptive SM partitioning on a single node."（§5 Adaptive SM partitioning） | https://github.com/uccl-project/mKernel | 可：同為 2 台×8 H200＋CX7；EFA 部分不適用 | https://arxiv.org/abs/2609.13585 |
| 3 | **Fine-grained Computation-Communication Overlap via Tile-level Signaling and Scheduling for MoE**；Cui, Wingkvist, Ericsson；2026-07-21；ICPP '26（arXiv comments） | MoE 第二次 all-to-all 暴露在關鍵路徑 | 持久計算 kernel（producer）＋專用少量 SM 的持久通訊 kernel（consumer），tile 就緒即發 segment | 4×A100 單機 NVLink | 4 個 SOTA MoE 系統（未逐一列） | 最高 2.64× 端到端、2.74× MoE layer（4×A100） | "Production systems could benefit from an runtime-adaptive SM partition selector, which we leave to future work."；"Future work includes extension to the backward pass for training…"（評估／結論段） | 未說明 | 部分：方法可移植，需 Hopper 重調 | https://arxiv.org/abs/2607.19539 |
| 4 | **Towards Compute-Aware In-Switch Computing for LLMs Tensor-Parallelism on Multi-GPU Systems (CAIS)**；Zhang C. et al.；2026-05-07；HPCA 2026 | NVLS 以通訊為中心，與計算 kernel 的記憶體語意不合，重疊受限 | 擴充 ISA／微架構、merging-aware TB 協調、graph-level dataflow 最佳化 | **模擬器**：Accel-Sim（Hopper 特性）＋BookSim2，模擬 H100 SuperPOD、900 GB/s NVLink/NVSwitch；LLM 縮小為 50% 尺寸 | 9 個 baseline，含 TP-NVLS、SP-NVLS 等 3 個 NVLS 版本及 T3 | 對 SOTA NVLS 方案 1.38× 端到端訓練、對 T3 1.61×（模擬） | 受限於模擬器："Due to simulator memory constraints and the long simulation time, simulating full-scale state-of-the-art models is infeasible."（§Benchmark） | 未說明 | 不可：需硬體／ISA 改動，僅模擬 | https://arxiv.org/abs/2605.05628 |
| 5 | **CommFuse: Hiding Tail Latency via Communication Decomposition and Fusion for Distributed LLM Training**；Karim et al.（Huawei Toronto Ascend）；2026-04-27；arXiv preprint | 資料切片式重疊有 tail latency | 以分解的 P2P 取代 reduce-scatter／all-gather，並排程分塊計算 | 未說明（全文未見實驗硬體；作者群屬 Ascend 團隊） | 未說明 | 摘要只稱 "consistently achieves lower latency, superior MFU"，無數字 | 首頁註記 "This manuscript is an early preprint and has not been rigorously revised for typos or minor errors." | 未說明 | 未說明（硬體未說明，無法判定） | https://arxiv.org/abs/2604.24013 |
| 6 | **Lagom: Unleashing the Power of Communication and Computation Overlapping for Distributed LLM Training**；Xu G. et al.；2026-02-24；arXiv preprint（6 頁） | 重疊時通訊與計算爭搶 GPU 資源；調 NCCL 參數成本高 | 統一成本模型＋優先序搜尋，協同調通訊參數（含 chunk size、channel 數） | 叢集 A：2 台×8 A40（NVLink，節點間 2×400 Gbps IB）；叢集 B：2 台×8 A40（PCIe 4.0，100 Gbps IB）；CUDA 12.8 | NCCL v2.18.3、AutoCCL | 1.07–1.33×（對 NCCL）、1.03–1.27×（對 AutoCCL） | 未說明 | 未說明 | 部分：調參方法可在 H100 上重做，成本模型要重新擬合 | https://arxiv.org/abs/2602.20656 |
| 7 | **Syncopate: Efficient Multi-GPU AI Kernels via Automatic Chunk-Centric Compute-Communication Overlap**（v1 標題為 *AutoOverlap*）；Qiang, Guan, Hu, Zhou, Ding, Aziz；2026-01-28（v4：2026-07-01）；OSDI 2026（arXiv comments） | stream 層級重疊的 kernel 啟動／同步開銷與 tail | 以 Triton source-to-source compiler，引入 chunk 抽象，圍繞單一融合計算 kernel 產生細粒度重疊 | 單機 4／8×H100，NVLink 900 GB/s；CUDA 12.9、NVSHMEM 3.3.9、PyTorch 2.7 | AsyncTP、Flux、Triton-distributed、ThunderKittens、Triton+NCCL、RingAttention 等 | 平均端到端 1.3×、最高 4.7×（8×H100 單機） | "Supporting this setting [multi-node] requires additional backend and runtime support for inter-node communication, which we leave to future work."；TMA "does not currently support inter-node communication or in-network (switch-based) collective reduction"（§8 Discussion／§2） | https://github.com/tie-pilot-qxw/syncopate | 可：單機 8×H100 NVSwitch 與其實驗相同 | https://arxiv.org/abs/2601.20595 |
| 8 | **MPK: A Compiler and Runtime for Mega-Kernelizing Tensor Programs**；Cheng X. et al.；2025-12-22（v2 2026-06-10）；OSDI（論文文末致謝 "anonymous OSDI reviewers"；PDF 檔名 osdi26-cheng.pdf，我未能從頁面直接讀出年份） | kernel-per-operator 造成 kernel 邊界、無法細粒度重疊計算與通訊；PDL 需大幅改寫 kernel | 把整個推論編成單一 mega-kernel：SM 級任務圖（tGraph）、事件驅動非同步 runtime，通訊用 NVSHMEM | A100、H100、B200 三代；單／多 GPU（GPU 數與節點數我未確認） | vLLM、SGLang | 對現有系統 1.0–1.7×（單與多 GPU 推論） | 未說明 | https://github.com/mirage-project/mirage | 部分：H100 可；多 GPU 規模待確認 | https://arxiv.org/abs/2512.22219 |
| 9 | **Design Space Exploration of DMA based Finer-Grain Compute Communication Overlap (FiCCO)**；Pal, Aga, Pati, Islam, John（AMD／UT Austin）；2025-12-11（v3 2026-06-03）；arXiv preprint | shard 粒度重疊受限於網路拓樸與資料流 | 比 shard 更細一層的重疊；通訊改走 GPU DMA engine 以降低干擾；提供選排程的 heuristic | 8×AMD MI300X（全連接）；NVIDIA 未測 | shard 粒度重疊、序列執行 | 摘要稱 "potential 1.7× speedup"；我未逐頁核實 heuristic 的具體加速數字 | 未說明 | 未說明 | 不可：AMD 硬體 | https://arxiv.org/abs/2512.10236 |
| 10 | **ParallelKittens: Systematic and Practical Simplification of Multi-GPU AI Kernels**；Sul, Arora, Spector, Ré；2025-11-17；MLSys 2026 | 多 GPU 重疊 kernel 難寫且難達峰值 | 8 個核心原語＋統一 template；整理出傳輸機制（copy engine／TMA／暫存器級含 in-network reduction）、排程、設計開銷三原則 | 8×H100 SXM（NVLink4／NVSwitch）、B200；皆單機；NVSwitch in-network reduction 有用 | Flux、Comet、CUTLASS、Triton-distributed、xDiT、YunChang、NCCL 非重疊 | 對最強 baseline 最高 2.33×（DP/TP）、4.08×（SP）、1.22×（EP）；勝 Triton-distributed 1.07–5.63× | "As this work focuses on intra-node execution, extending these abstractions to inter-node communication remains an important direction for future work."（§5）；另稱 Triton-distributed "originally tuned for H800 GPUs, fails to adapt efficiently to other architectures such as H100s"（§1） | https://github.com/HazyResearch/ThunderKittens | 可：8×H100 NVSwitch 單機 | https://arxiv.org/abs/2511.13940 |
| 11 | **FarSkip-Collective: Unhobbling Blocking Communication in Mixture of Experts Models**；Dukler et al.（AMD）；2025-11-14（v3 2026-05-27）；MLSys 2026 | MoE 阻塞式通訊 | 修改模型 skip connection 架構以允許重疊，並用自蒸餾轉換 16B–109B 模型 | AMD MI325X（4 節點×8，節點間 400 Gbps）、MI300X；NVIDIA 為次要（HIP/CUDA graphs 兩者皆提） | 原始架構 | 推論 DeepSeek-V3 TTFT 加速 32.6%、prefill 重疊 97.3%；訓練 all-to-all 重疊 88.9% | "…closing any remaining performance gaps in the model conversion which we leave for future work."（§轉換結果段）；亦自述效能分析有限制（附錄 "We would like to note the limitations of the performance analysis we provide."） | https://github.com/AMD-AGI/FarSkip-Collective | 部分：重疊實作在 PyTorch 層，但需模型轉換訓練、主要在 AMD 驗證 | https://arxiv.org/abs/2511.11505 |
| 12 | **Eliminating Multi-GPU Performance Taxes: A Systems Approach to Efficient Distributed LLMs**；Trifan et al.（AMD 等）；2025-11-04；arXiv preprint | BSP 模型的 "Three Taxes"（同步、inter-kernel 資料局部性、kernel 啟動） | 以 Iris（Triton，ROCm）in-kernel 通訊做 tile 級 producer-consumer 管線（All-Gather+GEMM、Flash Decode） | AMD MI300X／MI325X，單機 8 GPU | BSP 式（RCCL＋kernel） | 端到端延遲 10–20%（對 BSP 方法） | "we leave further optimization to the GEMM kernel to future work."；另評 Triton Distributed 有 "significant code complexity burden"（§相關工作） | https://github.com/ROCm/iris | 不可：AMD 專用 | https://arxiv.org/abs/2511.02168 |
| 13 | **Characterizing Compute-Communication Overlap in GPU-Accelerated Distributed Deep Learning: Performance and Power Implications**；Lee, Oh, Kim, Go, Park, Mahajan；2025-07-03；arXiv preprint（量測論文） | 重疊的實際代價（資源爭用、功耗）缺乏量化 | 在多種 GPU 上量測 LLaMA 類模型的重疊 | DGX H100（8×H100，FSDP 等）、A100×4、MI250、MI210 | 無重疊序列執行、"no communication" 理想值 | 重疊使計算平均變慢 18.9%、最高 40.0%（相對無通訊理想）；序列執行比重疊平均慢 10.2%、最高 26.6% | 未說明 | 未說明 | 可：H100 單機量測 | https://arxiv.org/abs/2507.03114 |
| 14 | **FlashMoE: Fast Distributed MoE in a Single Kernel**；Aimuyo, Oh, Singh；2025-06-05（v3 2025-11-08）；NeurIPS 2025 | MoE 的 CPU 排程、host 啟動通訊、頻繁 kernel 啟動 | 將專家計算與 inter-GPU 通訊融合進單一持久 GPU kernel，用 NVSHMEM device-initiated 傳輸 | 8×H100 單機（另 2×A100 對照）；未做多節點 | 未列於我摘錄處（SOTA MoE baselines，baseline 用 FP16、本文用 FP32） | 最高 9× GPU 使用率、6× 延遲、5.7× 吞吐、4× 重疊效率（8×H100，≤128 experts、16K tokens） | "Fully fused, persistent kernels demand deep GPU + distributed-systems expertise…"；"Training support. This work targets inference"（Limitations and Future Work） | https://github.com/osayamenja/FlashMoE | 可：8×H100 單機 | https://arxiv.org/abs/2506.04667 |
| 15 | **TokenWeave: Efficient Compute-Communication Overlap for Distributed LLM Inference**；Gond, Kwatra, Ramjee（Microsoft Research India）；2025-05-16（v5）；MLSys 2026 | TP 推論小 batch 下切分重疊反而變慢，且 RMSNorm 被忽略 | fused AllReduce–RMSNorm kernel（multimem／NVLS）只用 2–8 SM；wave-aware token 切分 | 8×H100 DGX（NVSwitch，NVLS 有用）；附錄含 8×B200；vLLM 0.8.5／0.14.1 | vLLM（含 vLLM-Multimem、vLLM-nocomm 反事實基準）、TileLink、NanoFlow | 最高 1.28× 延遲、1.19× 吞吐；fused kernel 對分開執行約 1.24–1.39× | "Under small decode batches, the overhead introduced by splitting tokens into two parts offsets the performance gains obtained from compute-communication overlap."（附錄 B200 段）；"torch.compile does not support symmetric memory allocation"（同段） | https://github.com/microsoft/tokenweave | 可：8×H100 NVSwitch | https://arxiv.org/abs/2505.11329 |
| 16 | **MegaScale-MoE: Large-Scale Communication-Efficient Training of MoE Models in Production**；Jin et al.（ByteDance）；2025-05-16；arXiv preprint | MoE 大規模訓練通訊占 43.6% | 通訊效率並行策略、tile 級融合 scatter／GEMM／all-to-all 等、通訊壓縮（低精度） | 1,440 張 NVIDIA Hopper（H800）；也有 A100、H20 對照 | Megatron-LM（commit f1f03922） | 352B MoE、1,440 GPU：1.41M tokens/s，比 Megatron-LM 1.88× | "We leave automatic optimization for future work."（Discussion "Holistic vs. automatic"） | 未說明（不開源） | 不可：需 1,440 GPU | https://arxiv.org/abs/2505.11432 |
| 17 | **Triton-distributed: Programming Overlapping Kernels on Distributed AI Systems with the Triton Compiler**；Zheng et al.（ByteDance 等）；2025-04-28；arXiv preprint | 重疊 kernel 難寫 | 在 Triton 加入 OpenSHMEM 風格原語＋signal，tile swizzle；覆蓋 AG+GEMM、GEMM+RS、AllToAll、MoE、Flash Decode | 8×H800（NVLink，CX7 400 Gb/s）、AMD MI308X；至多 64 GPU；也提 multimem 用於 low-latency AllGather | PyTorch+NCCL、Flux、DeepEP | AG+GEMM 平均 1.42×（對 PyTorch+NCCL）、1.09×（對 Flux）；GEMM+RS 跨機約 Flux 的 96.36% | "…a dedicated ReduceScatter kernel is required for MoE+RS to achieve better performance, which is left for future."；"current NVSHMEM bitcode library doesn't support IBGDA. So we leave this for future work."（§4 評估） | https://github.com/ByteDance-Seed/Triton-distributed | 部分：H800→H100 需重調（見 #10 引文） | https://arxiv.org/abs/2504.19442 |
| 18 | **Efficient and Adaptable Overlapping for Computation and Communication via Signaling and Reordering (FlashOverlap)**；Hong et al.；2025-04-28；EuroSys '26（DOI 10.1145/3767295.3769370，見 PDF） | 融合式重疊需手寫 kernel；分解式傷效率 | GEMM 以訊號通知各 wave group 的 tile 完成，再呼叫標準 NCCL；通訊前重排資料為連續位址 | A800、RTX 4090（4×4090 PCIe）；A800 TP 至 8；未測 Hopper | non-overlap、VanillaDecomposition、cuBLASMp、Async-TP、Flux（4090 無 P2P 故無 Flux） | 最高 1.65×（A800／4090） | "in our implementation, we use multi-processing for communication, which is limited to intra-node parallelism."（附錄 A.6）；"FLUX requires peer-to-peer access, which is not supported on the tested RTX 4090 server."（§評估） | https://github.com/infinigence/FlashOverlap | 部分：需 Hopper 版 CUTLASS 設定（A.6.1） | https://arxiv.org/abs/2504.19519 |
| 19 | **TileLink: Generating Efficient Compute-Communication Overlapping Kernels using Tile-Centric Primitives**；Zheng et al.（ByteDance）；2025-03-26；MLSys 2025 | 重疊 kernel 程式設計複雜 | 前端以 tile-centric 原語解耦通訊／計算設計，後端編譯為低階通訊指令 | 8×H800；2 台×8 H800（16 GPU） | 非重疊 baseline、Flux、RingAttention、PyTorch | 1.17×–20.76×（對非重疊）；8 個 LLM 端到端平均 1.32×（8 GPU）、1.29×（16 GPU） | "Currently, TileLink targets only NVIDIA GPUs."；"We leave this [model-level/pipeline parallelism] for future work."（§7.4 Discussion） | https://github.com/ByteDance-Seed/Triton-distributed（TileLink 頁面指向此庫） | 部分：H800→H100 | https://arxiv.org/abs/2503.20313 |
| 20 | **Comet: Fine-grained Computation-communication Overlapping for Mixture-of-Experts**；Zhang S. et al.（ByteDance Seed／SJTU）；2025-02-27；MLSys 2025 | MoE 層通訊占 47% | 依資料相依分析重排任務；thread-block 專門化 kernel；自適應工作量分配 | 8×H800 NVLink 單機（另 L20 叢集）；CUDA 12.3、NVSHMEM 2.11 | Megatron-TE、Megatron-Cutlass、FasterMoE、Tutel | 單層 1.96×、端到端平均 1.71× | "the warps for communication also interfere with the computation warps within the same thread block."（§3.2.1 Hardware resource restriction）；FasterMoE 不支援 TP | https://github.com/bytedance/flux（Comet 頁面所列專案頁） | 部分：H800→H100 | https://arxiv.org/abs/2502.19811 |
| 21 | **Ladder-Residual: Parallelism-Aware Architecture for Accelerating Large Model Inference with Communication Overlapping**；Zhang M. et al.（含 Tri Dao）；2025-01-11；ICML 2025 | TP 推論通訊占比高 | 改殘差路徑讓模組讀「稍舊」輸入，使 all-reduce 可與計算並行 | 8×H100（TP 至 8）；訓練 HSDP 於 8×H100 節點內；另有無 P2P 的情形 | 標準 Transformer、Parallel-Attn、"Communication-Free Upper Bound" | 70B、TP=8 約 29–30% 推論加速；無 P2P 最高 60% | "There is a chance that with longer adaptation… we can obtain a Ladder-Llama with more layers adapted. We leave the further exploration to future work."（§4.2） | https://github.com/open-lm-engine/lm-engine（PDF 連結，我未確認是官方釋出） | 部分：推論加速可驗證，但需約 3B tokens 重新適配模型 | https://arxiv.org/abs/2501.06589 |
| 22 | **Domino: Eliminating Communication in LLM Training via Generic Tensor Slicing and Overlapping**；Wang G., Zhang, Shen, Li, Ruwase（Microsoft DeepSpeed）；2024-09-23；arXiv preprint | TP 訓練通訊占比高（跨機尤甚） | 把單一 batch 切成獨立片段，管線化並重疊通訊與計算 | DGX-H100 1／2／4 台（8–32 H100，NVLink/NVSwitch，8×CX7 IB，共 400 GB/s／節點）；模型 2.7B–30B | Megatron-LM | 最高 1.3× 訓練加速；通訊隱藏約 50–70%（column-wise split） | "we leave this rotary embedding issue as a future optimization direction"（§5，Llama-2 增益較低）；"we defer its optimization to future work"（額外 MemCpy，§3） | https://github.com/microsoft/DeepSpeed（文中稱將釋出於此） | 可：H100 至 4 台（32 GPU）與可用資源相符 | https://arxiv.org/abs/2409.15241 |
| 23 | **FLUX: Fast Software-based Communication Overlap On GPUs Through Kernel Fusion**；Chang et al.（ByteDance／PKU）；2024-06-11（v5 2024-10-23）；arXiv preprint | TP 通訊重疊在 GPU 上效率低 | 把通訊與計算過度分解成 tile 並融合進單一 kernel（CUTLASS 3.4.1＋NVSHMEM 2.10.1） | A100 PCIe、A100 NVLink、H800 NVLink（8 GPU／台）；訓練至 128 GPU；跨機 16 GPU；H800 8×400 Gbps IB | PyTorch、TransformerEngine 1.4.0、vLLM、Megatron-LM | 最高重疊 96%；訓練 1.24×（128 GPU，對 Megatron-LM）；推論 1.66×／1.30×（prefill／decode，8 GPU，對 vLLM）；H800 NVLink 對 TE 1.10–1.51× | "Flux could perform worse than the non-overlapping baseline in a few extremely small m cases"（§6 Discussion）；小 m（decode）H800 上 0.95× 至 1.03× | https://github.com/bytedance/flux（來源：Comet 論文頁面；Flux 本文未見） | 部分：H800→H100；NVSHMEM 版本需對齊 | https://arxiv.org/abs/2406.06858 |
| 24 | **T3: Transparent Tracking & Triggering for Fine-grained Overlap of Compute & Collectives**；Pati, Aga, Islam, Jayasena, Sinclair；2024-01-30；ASPLOS 2024 | TP 通訊與計算序列化 | 以 producer 輸出位址空間設定融合 GEMM 與 reduce-scatter；硬體 tracker 觸發 DMA；near-memory compute | **模擬器**：Accel-Sim 擴充為多 GPU，ring 互連 150 GB/s；端到端用 MI210 單卡資料＋分析模型推估 | Sequential（GEMM 再 all-reduce）等 | 通訊重的子層幾何平均 30%（最高 47%）；資料搬移減 22%（最高 36%）；T-NLG／~500B 模型 geomean 29% | 未說明（無獨立限制段；Discussion §7 為延伸性討論） | 未說明 | 不可：需硬體改動，僅模擬 | https://arxiv.org/abs/2401.16677 |
| 25 | **Centauri: Enabling Efficient Scheduling for Communication-Computation Overlap in Large Model Training via Communication Partitioning**；Chen C., Li, Zhu, Duan, Sun, Zhang, Yang；2024-04-27；ASPLOS '24（DOI 10.1145/3620666.3651379，pp. 178–191） | 混合並行（DP／TP／PP／SP）的通訊重疊排程 | 通訊分割空間（三維抽象）＋operation／layer／model 階層式排程，實作於 Megatron-LM | Cluster A：8×A100-80G NVSwitch 600 GB/s，8 卡共用一個 200 Gb/s HDR IB；Cluster B：每台 4 個 IB HDR；CUDA 11.7、NCCL 2.14 | Megatron-LM 及其他現有方法 | 最高 1.49× | 未說明 | 未說明 | 部分：A100 時代設定，程式碼未說明 | https://dl.acm.org/doi/10.1145/3620666.3651379 |

（#9 FiCCO 與 #5 CommFuse 的細節核實度較低，已於表中標註。）

## 3. 作者自述的限制與未來工作彙整（依出現次數）

**A. 僅限單機（NVLink 域內），跨機尚未支援（6 篇：#7、#10、#18、#19、#14、#2）**
- #7 Syncopate：「Supporting this setting requires additional backend and runtime support for inter-node communication, which we leave to future work.」（§8）
- #10 ParallelKittens：「As this work focuses on intra-node execution, extending these abstractions to inter-node communication remains an important direction for future work.」（§5）
- #18 FlashOverlap：「…we use multi-processing for communication, which is limited to intra-node parallelism.」（附錄 A.6）
- #19 TileLink 與 #14 FlashMoE 在 8 GPU 單機以外的實驗很少；#2 mKernel 自述「only able to test adaptive SM partitioning on a single node」（§5）。

**B. 小 batch／小矩陣（decode）重疊反而變慢或增益消失（3 篇：#15、#23、#21 附帶）**
- #15 TokenWeave：「Under small decode batches, the overhead introduced by splitting tokens into two parts offsets the performance gains…」（附錄）
- #23 Flux：「Flux could perform worse than the non-overlapping baseline in a few extremely small m cases」（§6）
- 補充 #22 Domino 也要求 micro-batch ≥ 4（§5.1.1）。

**C. 通訊與計算爭奪 SM／記憶體頻寬（5 篇：#20、#13、#6、#10、#9）**
- #20 Comet：「the warps for communication also interfere with the computation warps within the same thread block.」（§3.2.1）
- #13 Lee et al.：重疊使計算平均變慢 18.9%、最高 40.0%（摘要／§1）。
- #6 Lagom 的動機是「communication operations contend with computation kernels for limited GPU resources」（§1）。
- #10 ParallelKittens 的分析指出暫存器級傳輸需要比 TMA 多 3.2–5.1× 的 SM 才能飽和 NVLink（由搜尋摘要得出，我未在全文中逐字核實該倍數，因此此條不作引文）。

**D. 需要專家級手寫、通用性／可移植性不足，或需要重調（4 篇：#14、#17、#12、#10）**
- #14 FlashMoE：「Fully fused, persistent kernels demand deep GPU + distributed-systems expertise; future work may investigate compiler/DSL abstractions to lower this barrier.」
- #10 ParallelKittens：Triton-distributed「originally tuned for H800 GPUs, fails to adapt efficiently to other architectures such as H100s」（§1）。
- #12：對 Triton Distributed 的批評為「significant code complexity burden」。

**E. 動態形狀、自動排程、訓練反向傳播等尚未涵蓋（4 篇：#7、#3、#16、#14）**
- #7：「A full dynamic-shape runtime … is left to future work.」
- #3：「Future work includes extension to the backward pass for training, adaptive SM partition selection…」
- #16：「We leave automatic optimization for future work.」
- #14：「Training support. This work targets inference」。

**F. 各論文互相不一致或不可比之處（只列出，不裁決）**
- #15 TokenWeave 稱 "none of the open-source serving systems like vLLM, SGLang … and TensorRT-LLM enable compute-communication overlap by default for tensor-parallel inference"（§1），而 #23 Flux 報告了對 vLLM 的重疊加速——兩者比較設定不同（TokenWeave 針對 token 數小至 1024 的推論重疊）。
- #1 Runway 稱 NVLS multimem all-reduce 每步只省 0.4 ms，並因 FP8 GEMM 無法寫入 symmetric buffer 而需拷貝；#15 TokenWeave 稱 multimem 融合 kernel 比分開執行快 1.24–1.39×。兩者模型、精度、層級不同。

## 4. 常見比較對象

- **PyTorch＋NCCL（含 cuBLAS）非重疊序列執行**：幾乎所有論文的基準（#17、#19、#10、#7、#23、#2…）。
- **Flux**：作為「手寫融合 kernel」最強對照被 #17（1.09×）、#19、#10、#7、#2 比較。
- **TransformerEngine（UserBuffer）／cuBLASMp／Async-TP（PyTorch）**：「中粒度」分解式重疊的對照（#23、#18、#7）。
- **Triton-distributed／TileLink**：compiler 路線對照（#10、#7、#2、#15）。
- **Megatron-LM／vLLM／SGLang**：端到端系統基準（#22、#23、#16、#20、#8、#15）。
- **DeepEP／DeepGEMM**：MoE 通訊對照（#17、#2）。
- 公認的最強對照：**我沒有找到論文明確宣稱「公認最強」**。#10 自述其「matches the performance of the strongest hand-optimized kernels (Flux, Comet, CUTLASS)」（§1）。

## 5. 硬體覆蓋

**在 8×H100／H200 NVSwitch 單機，或 ≤ 4 台 IB 叢集上做過實驗**
- 8×H100 單機：#10（H100＋B200）、#7（4／8×H100）、#14（8×H100）、#15（8×H100，附錄 B200）、#1（4／8×H100）、#13（DGX H100）、#21（H100，TP=8）。
- 多台 H100：#22 Domino（1／2／4 台 DGX-H100，8–32 GPU）。
- H200：#2 mKernel（2 台×8，共 16 GPU，兩個叢集）。
- H800（Hopper，NVLink 頻寬低於 H100）：#23 Flux、#19 TileLink（含 2 台 16 GPU）、#17 Triton-distributed（含 2 台、至 64 GPU）、#20 Comet；#16 為 1,440 GPU。
- #8 MPK：A100／H100／B200，GPU 與節點數我未確認。

**只在 A100／其他 NVIDIA、AMD、GB200 或模擬器上做過**
- A100／A800／4090／A40：#25 Centauri（A100，IB）、#18 FlashOverlap（A800、RTX 4090）、#3（4×A100）、#6 Lagom（A40）。
- AMD：#9 FiCCO（MI300X）、#12（MI300X／MI325X）、#11 FarSkip（MI325X／MI300X）、#13（含 MI250／MI210）。
- 模擬器：#24 T3（Accel-Sim）、#4 CAIS（Accel-Sim＋BookSim2，模擬 H100 SuperPOD）。
- GB200／NVL72：我檢索到的 25 篇中沒有以 GB200 NVL72 做實驗的；B200 出現在 #10、#15、#8、#1。
- #5 CommFuse：硬體未說明。

**有沒有「NVLS 開與關之間比較」**
- #1 Runway：以 NVLS multimem all-reduce 取代 NCCL all-reduce，每步省 0.4 ms（§3）。
- #4 CAIS：有 NVLS 增強版 baseline（TP-NVLS、SP-NVLS）與無 NVLS 的 T3 對照，但為模擬。
- #15 TokenWeave：含 "vLLM-Multimem" 與預設基準（Fig. 2），我未逐項核實是否為 NCCL-NVLS 開／關對比。
- #10 ParallelKittens：對 in-network reduction 與否的傳輸機制做分析（Table 1 為 P2P Reduction／in-network 能力矩陣）。
- 沒有找到明確以 `NCCL_NVLS_ENABLE=0/1` 做開關比較的論文（以全文關鍵字 NVLS／multimem 檢索）。

## 6. 開源工具與程式碼清單

| 名稱 | 網址 | 授權 |
|---|---|---|
| Flux（及 Comet 同庫） | https://github.com/bytedance/flux | 未說明 |
| Triton-distributed（含 TileLink 頁面指向） | https://github.com/ByteDance-Seed/Triton-distributed | 未說明 |
| FlashOverlap | https://github.com/infinigence/FlashOverlap | 未說明 |
| TokenWeave | https://github.com/microsoft/tokenweave | 未說明 |
| ParallelKittens／ThunderKittens | https://github.com/HazyResearch/ThunderKittens | 未說明 |
| Syncopate（原 AutoOverlap） | https://github.com/tie-pilot-qxw/syncopate | 未說明 |
| mKernel | https://github.com/uccl-project/mKernel | 未說明 |
| FlashMoE | https://github.com/osayamenja/FlashMoE | 未說明 |
| MPK／Mirage | https://github.com/mirage-project/mirage | 未說明 |
| Iris（AMD，Triton） | https://github.com/ROCm/iris | 未說明 |
| FarSkip-Collective | https://github.com/AMD-AGI/FarSkip-Collective | 未說明 |
| Domino（DeepSpeed） | https://github.com/microsoft/DeepSpeed | 未說明 |
| Ladder-Residual（PDF 內連結） | https://github.com/open-lm-engine/lm-engine | 未說明（我未確認是官方釋出） |

授權欄位我沒有逐一開 repo 確認，故皆為「未說明」。

## 7. 搜尋紀錄

使用 WebSearch（standard 為主，2 次 extended），每次約回傳 10 筆。查詢字串：
1. `arXiv fine-grained computation communication overlap GEMM kernel fusion tensor parallelism 2025`（10 筆；採 Flux、MegaScale-MoE）
2. `arXiv Triton-distributed compiler overlapping computation communication kernels`（10；採 #17、AutoOverlap／Syncopate）
3. `arXiv FLUX fast software-based communication overlap on GPUs through kernel fusion`（10；#23）
4. `arXiv TileLink OR "ThunderKittens" OR "Iris" OR "CUTLASS" distributed GEMM all-gather fused kernel H100 overlap`（10；#10、#15 線索）
5. `arXiv fused all-reduce RMSNorm kernel LLM inference tensor parallel overlap NVSwitch`（10；#15）
6. `arXiv Domino OR "Ladder-residual" OR "Flash Communication" tensor parallel communication hiding LLM`（10；#22、#21；**Flash Communication 無結果，未列入**）
7. `arXiv TileLink: generating efficient compute-communication overlapping kernels using tile-centric primitives`（10；#19）
8. `arXiv FlashOverlap lightweight design for efficiently overlapping communication and computation`（10；#18；注意：另有一篇同名 "FlashOverlap: Minimizing Tail Latency in Communication Overlap for Distributed LLM Training" 的轉載頁面，**未核實，未列入**）
9. `arXiv ParallelKittens multi-GPU kernels simplified inter-GPU overlap`（10；#10）
10. `arXiv 2602.11808 compute communication overlap`、`arXiv 2511.02168 overlap communication`（各 10；用來查 ID；2602.11808 為 "Deep Kernel Fusion for Transformers"，經開全文確認是單卡 SwiGLU MLP 算子融合、通訊沿用 SGLang 原有 all-reduce，**排除：不屬通訊重疊**；2511.02168 為 #12）
11. `ASPLOS 2024 T3 transparent tracking triggering fine-grained overlap compute collectives`（10；#24）
12. `arXiv distributed GEMM communication overlap Hopper TMA multimem NVLink in-network reduction fused kernel 2025 2026`（extended；#2；另得 FP8 GEMM+AR 即 #1 Runway）
13. `arXiv Mercury loop-level IR fine-grained remote memory access overlapping GPU kernels distributed`（10；Mercury 為 SOSP '25，DOI 10.1145/3731569.3764798，只找到 SOSP artifact 副本，**未找到 arXiv／ACM 頁面可核實，未列入表，僅被 #2 引用為 baseline**）
14. `arXiv Centauri OR "Comet" fine-grained computation-communication overlapping MoE training MLSys 2025`（10；#20）
15. `arXiv Design Space Exploration of DMA based Finer-Grain Compute Communication Overlap FiCCO`（10；#9）
16. `arXiv concurrent computation communication slowdown contention GPU overlap characterization LLM training Lee 2025`（10；#13；Chopper 2512.08242、Characterizing efficiency 2509.10371 為功耗／特性量測，**未列入（屬 T1 量測性質，非重疊／融合方法）**）
17. `Programmatic Dependent Launch PDL overlap kernel launch communication arXiv LLM inference multi-GPU 2025`（10；#8 MPK；其他為部落格：Hazy "No Bubbles"（https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles）與 LMSYS SGLang low-latency mode 部落格，**非同儕審查，未逐篇核實，未列入表**；**未找到專門討論 PDL 與通訊重疊的論文**）
18. `arXiv NanoFlow OR "TileScale" OR "Pallas" OR "NVSHMEM" fused GEMM all-reduce kernel Hopper multimem 2026`（extended；無新增）
19. `Mercury: Unlocking Multi-GPU Operator Optimization for LLMs via Remote Memory Scheduling SOSP 2025 arXiv`（10；同 13）
20. `Centauri: Enabling Efficient Scheduling for Communication-Computation Overlap … ASPLOS 2024`（10；#25，另得 FlashMoE #14、2607.19539 #3）
21. `arXiv 2026 tensor parallel inference communication overlap fused kernel NVLink SHARP decode latency H100 GEMM all-reduce`（extended；#4 CAIS、#5 CommFuse；2511.09557 為跨機推論通訊，**屬 T5，未列入**）

另做的核實動作：對 25 篇逐一以 curl 取 arXiv abs 頁（標題、作者、日期、comments）與 PDF 全文；Centauri 取 Crossref API（https://api.crossref.org/works/10.1145/3620666.3651379）。

**未核實／未列入的項目**
- NCCL GIN（arXiv 2511.15076）、Demystifying NVSHMEM（2606.05951）：已確認存在且為 device-initiated 通訊，與 T1 重疊，未納入本表。
- Mercury（SOSP '25）：見上，無 arXiv／官方頁面可驗證。
- MegaScale-MoE 的正式發表場合：arXiv comments 欄空白，未說明。
- MPK 的 OSDI 年份：以 PDF 檔名 osdi26 及致謝推斷，未能直接讀到頁首。
- PK 的「SM 需 3.2–5.1× 才能飽和 NVLink」數字僅出自搜尋摘要，未在全文逐字核實，故未作為引文。
