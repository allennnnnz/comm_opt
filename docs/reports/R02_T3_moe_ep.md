# R02 – T3 moe_ep：MoE 專家平行通訊文獻地圖

- **撰寫**：Sonnet 5.5 執行者，2026-11-16（依 `docs/specs/S02_lit_map.md`）
- **範圍**：MoE expert parallelism 的 all-to-all dispatch／combine、DeepEP 類函式庫、單機 NVLink＋跨機 IB 階層式、負載不均、與計算重疊
- **核實方式**：每篇都用 WebFetch 實際打開 arXiv abs／html、會議論文集頁或廠商頁；表中「未說明」代表我打開的頁面沒有該資訊。
- **重要限制（誠實聲明）**：WebFetch 回傳的是「小模型對頁面的摘要」，且多篇全文頁被截斷（約只讀前 100,000 字元）。所以「原文引用」是摘要工具轉述的原文片段，節號為其所標；**引用進論文前請 Opus 對原文再核一次**。「未說明」也可能只是落在被截斷的後段。
- 本報告只列事實，不判斷題目好壞。

---

## 1. 主題概況

2024–2026 年 MoE EP 通訊的研究大致分成幾條支線：(a) **通訊函式庫／kernel**：DeepEP（NVLink＋RDMA，分 high-throughput 與 low-latency 兩組 kernel）為事實上的參考實作，其後有 NVIDIA 的 Hybrid-EP、NCCL EP（建立在 NCCL GIN 上）、與針對可攜性的 UCCL-EP、Perplexity fabric-lib；(b) **計算與通訊重疊／融合**：從 Lancet（MLSys 2024）、Comet、MegaScale-MoE 到 FlashMoE 的單一 persistent kernel、Triton-distributed 的 compiler 生成 kernel，以及 DisagMoE 的 AF-pipe；(c) **減少流量**：階層式去重（HierMoE）、樣本／專家放置（NetMoE、Occult、GRACE-MoE、Speculative MoE、2604.23150）、node-limited routing（DeepSeek-V3）；(d) **負載不均**：專家 re-layout／replication／token 重新導向（LAER-MoE、FineMoE、UltraEP、TEMPO、LLEP、EasyBalance）；(e) **整合與實務**：Megatron Core MoE 報告、SGLang／vLLM 的 wide-EP 部署。實驗硬體大宗是 H800／H100／A100／A800 的 8 GPU 節點加 IB，GB200 NVL72 主要出現在 NVIDIA 自己的報告。

---

## 2. 論文表（依年份新到舊）

編號：P＝論文／預印本，V＝廠商技術文章或專案頁（非同儕審查）。
「可重現」判斷依據：我們環境為 8×H100／H200 NVSwitch 節點、最多 4 台（32 GPU）、每台 8×NDR 400G IB、驅動 550（CUDA 12.x）、無 root。

| # | 標題 / 作者 / 年 / 出處 | 解決的問題 | 方法要點 | 硬體與規模（NVLS?） | 比較對象 | 報告的效益（附條件） | 作者自述限制（原文，位置） | 程式碼 | 我們環境可重現？ | 出處網址 |
|---|---|---|---|---|---|---|---|---|---|---|
| P1 | **NCCL EP: Towards a Unified Expert Parallel Communication API for NCCL**；Goldman, Boker, … Bloch（NVIDIA）；2026-03（v1 03-13，v3 04-02）；arXiv preprint，頁面未列 venue | 提供 NCCL 內建的 EP dispatch/combine API | 建在 NCCL Device API；LL 模式（1–128 tokens，直接 NVLink＋RDMA mesh、double buffer）與 HT 模式（NVLink 先聚合再 RDMA）；LL kernel 衍生自 DeepEP | NVIDIA EOS：8×H100 80GB/節點、NVLink4、8×400G IB；LL 測 1–8 節點（最多 64 GPU）、vLLM 測 1/2/4 節點；NVLS：未提及（multimem 僅在 §II-C 作為 Device API 模組提到，未說 NCCL EP 使用） | DeepEP（kernel 測試時跑在 NCCL 2.29 上；vLLM 測試用 v1.2.1 NVSHMEM/IBGDA） | LL dispatch 在 2–8 節點持平或優於 DeepEP（256 experts、hidden 7168、128 tokens、top-k 8、BF16），單節點落後約 4%；LL combine 1–4 節點落後 5–12%；vLLM Qwen3-30B-A3B 吞吐 755.1 vs 811.2 tok/s（1 節點）、563.3 vs 617.0（4 節點）（NCCL EP vs DeepEP，§VII-B Table VII） | §IX "NCCL EP is currently an experimental API."；§IX "training performance remains limited by ongoing high-throughput kernel optimization for multi-node setups."；§VII-A "NCCL EP currently only supports the hybrid NVLink-RDMA path." | 論文連結 https://github.com/NVIDIA/nccl；授權論文未說明 | **可**：H100＋IB、≤4 節點即其測試規模（需 NCCL 版本含 GIN；本機 NCCL 版本未驗證） | https://arxiv.org/abs/2603.13606 ；https://arxiv.org/html/2603.13606v3 |
| P2 | **Scalable Training of Mixture-of-Experts Models with Megatron Core**；Yan, Bai, Yao, … June Yang（NVIDIA）；2026-03-08（v2 03-10）；arXiv 2603.07685 | 整理 MoE 訓練在記憶體／通訊／計算三面的系統優化 | 通訊面：AllGather／all-to-all／Flex（DeepEP、HybridEP）三種 dispatcher＋EP all-to-all overlap；另有 Parallel Folding、FP8/NVFP4 | GB200 NVL72、GB300、H100；H100 為 1,024 GPU（NVL8，跨節點 EP64，Table 17）；跨機為 IB（NDR 50 GB/s 作為例）；NIC／節點數頁面未說明；NVLS：未提及；§4.1.7 提到 SHARP IB | 通訊 benchmark：HybridEP vs all-to-all（Table 7）；端到端：H100 與 GB200 組態互比 | DeepSeek-V3-685B：368 TFLOPS/GPU（1,024 H100）、1,048 TFLOPS/GPU（256 GB200，Table 17）；Table 7（hidden 7168、seq 4096、256 experts）H100 dispatch EP64：HybridEP 4626 µs vs all-to-all 9164 µs；DeepEP 下 EP 通訊佔迭代時間 30–40%，overlap 後 <5%（§4.2.3） | §4.2.3 "The first FWD and last BWD remain on the end-to-end critical path, so their all-to-all cannot be hidden."；§4.2.3 保留 SM 給 all-to-all 會降 GEMM 效率（轉述）；App. B.1 設定 "may not be globally optimal" | 以 Megatron-LM／Megatron-Bridge／Megatron-MoE-ModelZoo 公開（App. B.3）；授權頁面未說明 | **部分**：dispatcher 比較可在 H100 8–32 GPU 做；論文規模 1,024 GPU 與 GB200 不可 | https://arxiv.org/abs/2603.07685 ；https://arxiv.org/html/2603.07685v2 |
| P3 | **UltraEP: Unleash MoE Training and Inference on Rack-Scale Nodes with Near-Optimal Load Balancing**；Wei, Jin, Dai, … Luo；2026（頁面顯示 v3 為 2026-06-18，原始提交日未顯示）；arXiv 2606.04101 | 專家負載不均（預測式放置仍有殘餘不均） | quota 驅動的 planner：專家 replication＋token reroute，求精確平衡 | 每 rack 64 GPU／16 台；serving 1 rack、訓練 2–4 rack（128–256 GPU）；GPU 型號、NIC 未說明；scale-up 頻寬為 scale-out RDMA 的 8–10 倍；NVLS：未提及 | Megatron-LM（commit e93814b）、SGLang（v0.5.9）、EPLB、LPLB、EPLB++、Ideal（強制平衡）；通訊：torch.distributed batch send/recv、DeepEP | 平均達 force-balanced 理想吞吐的 94.3%；訓練比 Megatron-LM 高 42%（GLM4.5-106B／Qwen3-235B／DeepSeek-V3）；serving prefill 比 SGLang 高 1.56×；expert replication 比 torch.distributed／DeepEP 快 3.1–5.5×（Qwen3-235B、EP64，§8.5） | §3 範圍僅訓練與 serving prefill，不含 decode；§8.2 剩餘差距主要歸因於 "uneven routing in realistic MoE training"；§10 "the same abstraction can naturally extend to reinforcement learning (RL) pipelines" | 未提供 | **不可**：需 64–256 GPU rack-scale | https://arxiv.org/abs/2606.04101 ；https://arxiv.org/html/2606.04101 |
| P4 | **TEMPO: Makespan-Aware Expert-Parallel Load Balancing Across Memory- and Compute-Bound Regimes**；Li, Jia, Shen, … Song（KlingAI）；2026-08-14（v2）；arXiv 2608.13057 | token 數量作為負載代理不準，dispatch 應最小化 makespan | 非同步 CPU 求解器（~2 ms／次）做 makespan-aware 的 replica 分配；證明 NP-hard | Testbed A：8 GPU 單節點（型號匿名）；Testbed B：8 GPU 節點，單機 EP8、雙機 EP16，RoCE 8×400G，CUDA 13.2；NVSwitch／NVLS 未提 | 靜態放置、EPLB-even、token-LP（LPLB/UltraEP 目標）、METRO、LLEP-R、SGLang LPLB、MoonEP、10 秒 MILP | 以校準模擬器得相位圖：最多勝 15.5%；Qwen3-235B-FP8 在 Testbed B +4–6% 吞吐、p99 ITL −15.6%；DeepSeek-V3 在 Testbed B 屬於「無優勢區」，約 −2% 到 −3% | §7 L1 "Phase/scale figures come from the calibrated simulator"；§7 L2 完整 DeepSeek-V3 規模（EP32+）為外推 | 頁面可見範圍未提供 | **部分**：EP8／EP16（1–2 節點）可做；EP32+ 為外推需 32 GPU（我們最多 32） | https://arxiv.org/abs/2608.13057 ；https://arxiv.org/html/2608.13057v2 |
| P5 | **EasyBalance: Cross-Layer Load Balancing in Distributed MoE Inference**；Wu, Gao, Li, Wu（ISCAS）；2026-08-08；arXiv 2608.07964（頁面未列 venue） | 推論時各層負載不均造成 GPU 閒置 | 跨層組合 micro-batch 排程（MaxUtil 等啟發式）以降低 under-utilization | 單節點 8×A800-SXM4 80GB、NVLink；NIC 未說明；NVLS 未提 | w/o EB、EPLB（正交疊加）、CumUtil／DiffPeak | EP=8 下 GPU under-utilization 多數降 40% 以上（≥0.35→≈0.2，§5.1）；Qwen3-30B 2wikimqa 0.37→0.21（Table 4） | App. D "EasyBalance currently requires micro-batching"；App. D "The experiments are conducted only on A800-SXM4 80GB GPUs" | https://github.com/yize-wu/EasyInfra ；授權未說明 | **可**：單節點 8 GPU | https://arxiv.org/abs/2608.07964 ；https://arxiv.org/html/2608.07964 |
| P6 | **Least-Loaded Expert Parallelism: Load Balancing An Imbalanced Mixture-of-Experts**；Nguyen, Pandit, Xu, Xiong, Joty；ICML 2026（PMLR 306, pp. 92659–92673） | post-training／推論時路由不均使少數裝置過載 | LLEP：動態把多餘 token 與對應專家參數搬到低負載裝置，輸出與標準 MoE 精確一致 | PMLR 頁與 GitHub README 皆未說明 GPU 型號；範例 8 GPU | 標準 EP | 摘要：最多 5× 加速、4× 降低 peak memory，gpt-oss-120b 快 1.9×；（README 寫「最多 6×」，與摘要的 5× 不同，見 §8 矛盾紀錄） | PMLR 頁無限制章節；README：僅供研究用途、"shines" 於不可預測的不均 | https://github.com/SalesforceAIResearch/LeastLoadedEP ；Apache-2.0（README 側欄） | **部分**：硬體需求未說明；範例 8 GPU 應可跑，未驗證 | https://proceedings.mlr.press/v306/nguyen26m.html |
| P7 | **Scaling Multi-Node Mixture-of-Experts Inference Using Expert Activation Patterns**；Bambhaniya, Jeong, Park, …, Krishna（Georgia Tech、Meta）；2026-04-25；arXiv 2604.23150 | 多節點 EP decode 時節點間 all-to-all 成為主要成本 | 依 expert activation pattern 聚類請求並放置專家 | 資料蒐集 8×H100；評估組態 DP8TP8→EP64（64 GPU）；NVLink／NIC／節點數未說明；NVLS 未提 | vLLM linear placement、EPLB | all-to-all 資料量最多降 20%；Llama-Maverick decode all-to-all 時間 −9%、層 runtime −5.5%；Qwen3-235B 相對 EPLB all-to-all −12%、層 −6%（§5.3） | §5.3 "all-to-all kernel pads data to the largest EP rank's size"（限制增益）；§7 Future Works "Further optimization of all-to-all communication kernel" | 未連結 | **部分**：方法為放置策略，可在 ≤32 GPU 驗證小規模；論文為 64 GPU | https://arxiv.org/abs/2604.23150 ；https://arxiv.org/html/2604.23150v1 |
| P8 | **DisagMoE: Computation-Communication overlapped MoE Training via Disaggregated AF-Pipe Parallelism**；Zeng, Chang, …, Jiang（ByteDance Seed、UW、Cornell）；2026-05-10；arXiv 2605.11005 | operator 層級切塊的重疊仍有暴露通訊（視窗受 FFN 計算限制） | 把 attention 與 FFN 分到不同 GPU 群（disaggregated），以 many-to-many token 交換作為 pipeline 階段 | 16 節點×8 H800 80GB；NVLink（§6.1 寫 400 GB/s，§3.1 寫 "near 900GB/s"）；ConnectX-7 8×400G；NVSwitch／NVLS 未提 | Megatron-LM 1F1B interleaved、Tutel、Comet、DualPipe | 最多 1.81× vs Megatron-1F1B（4K–32K 序列、16 節點）；最多 1.34× vs 最強 overlap 系統；vs DualPipe 平均 1.05–1.13×（§6.2） | §7 "DisagMoE assumes pretraining-style stable shapes such as fixed sequence length and micro-batch size"；§7 不對稱 pipeline 深度為 "a promising extension" | 未提供 | **不可**：128 GPU 規模；縮小規模比較有限 | https://arxiv.org/abs/2605.11005 ；https://arxiv.org/html/2605.11005 |
| P9 | **LAER-MoE: Load-Adaptive Expert Re-layout for Efficient Mixture-of-Experts Training**；Liu, Wang, Fu, Xiao, Li, Li, Cui；2026-02-12；ASPLOS '26，DOI 10.1145/3779212.3790180 | 負載不均使 all-to-all 時間占比由 <10% 升至 >40%（搜尋結果摘要，原文章節未直接核對） | Fully Sharded Expert Parallel：依負載重新排布專家 | 4 節點×8 A100 80GB（32 GPU）；NVLink 300 GB/s 單向；IB 800 Gbps；NVSwitch、NIC 型號未提；NVLS 未提 | Megatron、FSDP+EP（自建）、FlexMoE（自行重現） | 最多 1.69× vs Megatron、1.50× vs FSDP+EP、1.39× vs FlexMoE；Mixtral-8x7B 個案 all-to-all 最多快 2.68×（8K context、50 iter 平均） | §7 "While empirical experiments on large-scale clusters are limited by resource constraints"；§5.2 "We also aim to develop more efficient and effective planners as part of our future work." | https://github.com/PKU-DAIR/Hetu-Galvatron/tree/laer-moe ；授權未說明 | **部分**：32 GPU 規模相同，但原測 A100；可在 H100 重做 | https://arxiv.org/abs/2602.11686 ；https://arxiv.org/html/2602.11686 |
| P10 | **UCCL-EP: Portable Expert-Parallel Communication**；Mao, Zhang, …, Shenker, Raiciu, Zhou, Stoica；2025-12-22（v2 2026-01-22）；arXiv 2512.19849 | DeepEP 的 GPU 直接發 RDMA 綁死特定 GPU–NIC | GPU 把壓縮的路由指令送到 CPU proxy 執行緒，由 CPU 發 GPUDirect RDMA；用 RDMA immediate data 模擬排序語意 | 6 組 testbed：H200×8＋EFAv3 200G×16；B200×8＋EFAv4；**H100×8＋ConnectX-7 400G×8 IB（4 台，Nebius）**；GH200；MI300X＋CX7／Broadcom；EP32 用 4 個 CPU proxy thread／GPU；NVLS 未提 | NCCL／RCCL、DeepEP、Perplexity Kernels（PPLX）、CPU-assisted IBGDA、理論最佳 | CX7＋IB、EP32、HT：dispatch 與 DeepEP 差 <5%；vs PPLX dispatch 2.1×、combine 1.6×；EFA 最多 2.1×；SGLang Qwen3-235B EP32 62K vs 44K tok/s（約 40%，NV_EFA3） | §5.4 "current EFA firmware is not able to process small tokens quickly enough"；§6 "UCCL-EP's low-latency kernel could be further optimized by packing tokens in a best-effort manner" | https://github.com/uccl-project/uccl/tree/main/ep ；授權未說明 | **可（NV_IB 組態）**：4 台 H100＋CX7 IB 與我們最大規模相同 | https://arxiv.org/abs/2512.19849 ；https://arxiv.org/html/2512.19849v2 |
| P11 | **GPU-Initiated Networking for NCCL**；Hamidouche, Bachan, Markthub, …, Venkata（NVIDIA）；2025-11-19（v2 11-24）；arXiv 2511.15076 | 讓 CUDA kernel 在 RDMA 網路上直接發遠端記憶體操作（MoE 低延遲需求） | NCCL 2.28 Device API 的 GIN；兩個 backend：GDAKI（DOCA GPUNetIO）與 Proxy（GPU→CPU lock-free queue）；整合進 DeepEP | EOS：H100 80GB×8、NVLink4、8×400G IB；用 1–8 節點（8–64 GPU）；微基準 2 GPU；NVLS 未提 | NVSHMEM 3.4.5（IBGDA、IBRC）、DeepEP v1.2.1、NCCL GIN 兩 backend | 2 GPU ping-pong（4–128 B）：GDAKI 16.7 µs、IBRC 16.0 µs、IBGDA 24.3 µs、Proxy 18.0 µs；HT dispatch（BF16、2 節點）84.36 GB/s（GIN）vs 84.97 GB/s（NVSHMEM）；LL dispatch（2 節點，含 NVLink）142.51 µs vs 157.00 µs | §V-D 實作 "remains under active development"；§III-A NCCL 2.28 要求各 rank symmetric window 大小一致（未來放寬）；§VII 未來推廣到 PyTorch／TensorRT-LLM／vLLM／SGLang／JAX/Triton | 論文未給 GIN 程式碼網址；連到 DeepEP 與 NCCL 專案頁 | **可**：1–8 節點 H100＋IB；需 NCCL≥2.28 | https://arxiv.org/abs/2511.15076 ；https://arxiv.org/html/2511.15076 |
| P12 | **fabric-lib: RDMA Point-to-Point Communication for LLM Systems**（Perplexity；Licker, Hu, Zaytsev, Chen）；2025-10-31（v2 2026-04-13）；arXiv 2510.27656 | 新型 LLM 系統（MoE 路由、分離式推論、RL 權重更新）需要彈性 P2P，現有實作綁特定 NIC | 統一介面；one-sided WriteImm＋ImmCounter 通知；多 NIC 管理；MoE kernel 用 host proxy 而非 GPU 直發 RDMA | Cluster A：8×H200、2×200G EFA／GPU；Cluster B：8×H100、400G CX7／GPU；MoE 測 8–64 GPU；NVLink 用於節點內；NVLS 未提 | DeepEP、pplx-kernels（NVSHMEM 3.4.5）、NIXL 0.6.1、ib_write_bw、fi_rma_bw | MoE decode（DeepSeek-V3＋MTP、EP=DP=64）：CX7 78.42 vs DeepEP 73.76 tok/s（batch=2）；EFA 66.75 vs pplx 20.97（Table 6）；節點內約比 DeepEP 慢 2 µs、跨節點 16／32 rank 較快（§7.4.3）；RL 權重更新 1.2 s（§7.3）／1.3 s（摘要） | §8 "GDA is not yet available on most cloud instances (AWS `p5`/`p5e`, eRDMA) and remains preliminary on `p5en`"；§8 "Porting to additional NICs requires per-hardware tuning, not a redesign of the library" | https://github.com/perplexityai/pplx-garden/ ；授權未說明 | **部分**：CX7＋H100 可；EP=64 超出 32 GPU，較小 EP 可做 | https://arxiv.org/abs/2510.27656 ；https://arxiv.org/html/2510.27656v2 |
| P13 | **FineMoE: Fine-grained Load Balancing for Mixture-of-Experts with Token Scheduling**；Zhao, Wu, Song, Xu, Yuan；2025-11（v2 2026-01-15；原始日未顯示）；arXiv 2511.16947（頁面未列 venue） | 專家負載不均（micro-batch 層級） | token 在 GPU 間排程，以線性規劃做細粒度平衡；asymmetric 專家放置 | 4 節點×8 H100 80GB SXM；NVLink 900 GB/s（NVSwitch 未提）；每節點僅 2×400G IB；NVLS 未提 | Megatron-LM、SmartMoE、FlexMoE、DeepSpeed ZeRO-1 | 最多 +47.6%（平均 36.9%）吞吐 vs Megatron-LM；vs FlexMoE 平均 +13.9%（GPT 32×1.3B、Mixtral；DP8、EP4；§7.2） | §9 "We plan to integrate FineMoE with FSDP in future work."；§7.5 "Our testbed consists of 8 GPUs but only 2 NICs per node, resulting in limited inter-node network bandwidth." | 頁面可見範圍未提供 | **可**：4×8 H100，與我們最大規模幾乎相同（NIC 數不同） | https://arxiv.org/abs/2511.16947 ；https://arxiv.org/html/2511.16947v2 |
| P14 | **GRACE-MoE: Grouping and Replication with Locality-Aware Routing for Efficient Distributed MoE Inference**；Han, Pan, Peng, Tao, Zhang, Zhang（USTC）；2025-09（v3 2026-01-24）；arXiv 2509.25041 | 跨裝置 all-to-all 與專家負載（推論） | 依親和度分組專家、複製熱門專家、locality-aware routing | 每台 8×RTX 3090（PCIe）或 8×RTX 4090；25 Gb Ethernet；1–2 節點；無 NVLink | DeepSpeed、Tutel、Megablocks、vLLM、C2R | 摘要 "up to 3.79×"；§5.2 總加速 0.95–3.79×、端到端延遲最多 −73.6%（WikiText-2、batch 128、prefill 64、decode 16） | §6 "its reliance on expert replication increases memory demands, which may limit deployment on memory-constrained GPUs."；§6 "Extending GRACE-MoE to larger models and industrial-scale clusters will further validate its efficiency and scalability." | "will be released upon acceptance" | **部分**：方法可移植，但原測為消費級 PCIe GPU，結論不能直接套用 NVSwitch | https://arxiv.org/abs/2509.25041 ；https://arxiv.org/html/2509.25041v3 |
| P15 | **HierMoE: Accelerating MoE Training with Hierarchical Token Deduplication and Expert Swap**；Lin, Pan, Zhang, Shi, Wang, Chu；2025-08-13；arXiv 2508.09591（頁面未列 venue） | 節點間 IB 遠慢於 NVLink，all-to-all 有重複 token | 階層式 AlltoAll 在各網路層先去重 token，另有 expert swap | 4 節點×8 A6000 48GB（32 GPU）；NVLink 112.5 GB/s（4x）；Mellanox 200 Gb/s IB；NVLS 未提 | Megatron-LM、SmartMoE、Tutel-2DH | AlltoAll 快 1.55–3.32×（vs HD2-MoE-Smart／Megatron-LM／Tutel-2DH，§V-D）；端到端 1.18–1.27× vs Megatron-LM（§V-C；EP=32、micro-batch 1、seq 1024、模型縮小） | 無專屬限制章節；§V-A "Due to the GPU memory constraints of our testbed" 模型被縮小 | 未提供 | **部分**：32 GPU 規模相同；A6000 與 H100 頻寬比不同，結論需重測 | https://arxiv.org/abs/2508.09591 ；https://arxiv.org/html/2508.09591v1 |
| P16 | **FlashMoE: Fast Distributed MoE in a Single Kernel**（先前標題 FlashDMoE）；Aimuyo, Oh, Singh（Cornell）；arXiv 2506.04667（v3 2025-11-08）；頁面含 NeurIPS checklist，未明示 venue | PyTorch 呼叫式通訊造成啟動開銷與 SM 閒置 | 單一 persistent kernel 融合 dispatch、expert 計算、combine；NVSHMEM device-initiated | **單機 8×H100 80GB NVLink**；NVSwitch、NIC 未提；多節點為推估；NVLS 未提 | Comet（僅 ≤4 GPU）、FasterMoE、Megatron-CUTLASS、Megatron-TE、DeepEP＋Megatron（利用率比較） | 8×H100、≤128 experts、16K tokens，FlashMoE FP32 對 FP16 baselines：最多 9× GPU 利用率、6× 延遲降低、5.7× 吞吐（摘要）；§4.2 SM 利用率 93.17% vs Megatron-TE 59.11%；§4.3 17.7 MTokens/s（8 GPU） | §5 "This work targets inference"（訓練需融合反向與梯度通訊）；§5 "Fully fused, persistent kernels demand deep GPU + distributed-systems expertise." | https://github.com/osayamenja/FlashMoE ；授權未說明 | **可**：單機 8×H100 | https://arxiv.org/abs/2506.04667 ；https://arxiv.org/html/2506.04667v3 |
| P17 | **MegaScale-MoE: Large-Scale Communication-Efficient Training of Mixture-of-Experts Models in Production**；Jin, Jiang, Bai, …, Liu（北大、ByteDance Seed）；2025（v3 2025-10-17）；**EuroSys '26**，DOI 10.1145/3767295.3769325 | 大規模 MoE 訓練的通訊效率 | attention 與 FFN 用不同平行策略以降通訊量；operator 間與 operator 內（tile 級融合）重疊；低精度通訊；selective activation rematerialization | H800（另 A100、H20）；節點內 NVLink 域 8（H800 400 GB/s）；NIC 未指定；240–1,440 GPU；NVLS 未提 | Megatron-LM（commit f1f03922） | 352B MoE、1,440 GPU：1.41M tokens/s、1.88× vs Megatron-LM（Table 3）；強縮放 1.65–1.88×；operator 內重疊使通訊＋計算時間降 1.2–4.7×、迭代時間降 7.1–12.9%（§6.2）；註：引言稱 1.88× 為 MFU，摘要與 Table 3 為吞吐 | §7 "We leave automatic optimization for future work."；§7 MoE 計算 operator 是 straggler 主要來源（未解決） | 頁面未給 | **不可**（全系統）；**部分**：tile 級融合概念可在單機驗證 | https://arxiv.org/abs/2505.11432 ；https://arxiv.org/html/2505.11432v3 |
| P18 | **Occult: Optimizing Collaborative Communication across Experts for Accelerated Parallel MoE Training and Inference**；Luo, Li, Peng, …, Chen；2025-05-19；**ICML 2025** | all-to-all 佔大規模執行時間比例高（搜尋結果稱 >40%，原文未直接核對） | 修剪專家協作＋依統計重排專家位置＋稀疏矩陣 all-to-all kernel | 單節點 PCIe 連接 A6000 48GB（EP 多為 4 GPU；訓練圖含 8／16 向）；無 NVLink／NIC 資訊 | vLLM、HuggingFace（推論）；FSDP、DeepSpeed、Tutel、MegaBlocks（訓練） | 摘要 ">1.5×"；prefill（2^14 tokens）OLMoE vs Tutel 8.66×、vs MegaBlocks 1.70×；Qwen-MoE／DeepSeek-MoE vs MegaBlocks 0.71×／0.66×（較慢）；訓練 1.54× vs MegaBlocks | §6 "some problems remain unsolved, such as workload balancing." | https://github.com/UNITES-Lab/Occult ；授權未說明 | **可**（軟體）：但原測為 PCIe，NVSwitch 上增益可能不同 | https://arxiv.org/abs/2505.13345 ；https://arxiv.org/html/2505.13345 |
| P19 | **Insights into DeepSeek-V3: Scaling Challenges and Reflections on Hardware for AI Architectures**；Zhao, Deng, …, Liang, …, Wei（DeepSeek-AI）；2025（v2 2025-12-23）；**ISCA '25**，DOI 10.1145/3695053.3731412 | 記錄 EP 對硬體互連的需求與 node-limited routing | §4.3 node-limited routing：256 專家分 8 組、每 token 最多送 4 節點；dispatch FP8／combine BF16；建議 scale-up／scale-out 統一與通訊 co-processor | 2,048 H800；每節點 8 GPU＋8 IB NIC；NVLink 800→400 GB/s（§4.1）；§4.3 有效約 160 GB/s NVLink、約 40 GB/s IB（4:1）；NVLS 未提 | 無對照實驗（設計與觀點文） | 通訊最多佔用 20 SM／H800（§4.4.1）；IBGDA 使 GPU 直接填 work request（§5.2.3） | §4.4.1 通訊佔用 SM "complicates communication pipeline kernel implementations"；§4.5.1 "current hardware lacks the flexibility to dynamically allocate bandwidth" | 頁面未列連結（提及 DeepEP、DeepGEMM 開源） | 不適用（觀點文）；node-limited routing 概念可在 4 節點 H100 驗證 | https://arxiv.org/abs/2505.09343 ；https://arxiv.org/html/2505.09343 |
| P20 | **Triton-distributed: Programming Overlapping Kernels on Distributed AI Systems with the Triton Compiler**；Zheng, Bao, Hou, …, Liu（ByteDance Seed 等）；2025（v3 2025-06-05；原始日未顯示）；arXiv 2504.19442 | 手寫融合通訊 kernel 難度高 | 在 Triton 加入通訊原語（NVSHMEM bitcode），自動產生 overlap kernel，含 MoE（AG+MoE、MoE+RS）與跨機 all-to-all | 8×H800 NVSwitch（NVLink 約 170 GB/s／GPU）、CX7 400G×1／GPU；另 AMD MI308X、L20；8–64 GPU，128 GPU 對 DeepEP；NVLS 未提 | PyTorch+NCCL／RCCL、FLUX、DeepEP、NVSHMEM fcollect | AllToAll（推論 vs DeepEP，8–64 GPU）：dispatch 平均 1.18×、combine 1.44×；64 GPU dispatch 與 128 GPU 仍是 DeepEP 較好；AG+MoE 44.97×（對弱 baseline，作者自述） | §4.2 "a dedicated ReduceScatter kernel is required for MoE+RS to achieve better performance, which is left for future."；§4.2 "current NVSHMEM bitcode library doesn't support IBGDA"，all-to-all 因而用 IBRC | https://github.com/ByteDance-Seed/Triton-distributed ；授權未說明 | **可**：8–16 H100（NVSHMEM、IB）；64 GPU 以上不可 | https://arxiv.org/abs/2504.19442 ；https://arxiv.org/html/2504.19442 |
| P21 | **Speculative MoE: Communication Efficient Parallel MoE Inference with Speculative Token and Expert Pre-scheduling**；Li, Zheng, Chen, Xu, Lai, Du, Wang（Huawei）；2025-03（頁面 v3 2025-03-19；後續至 v5 2026-02，來自搜尋結果，未核對）；arXiv 2503.04398 | 推論時 EP all-to-all 為瓶頸 | s-TS：在 gating 前預測路由並預先搬 token；s-EG：離線把共同啟用的專家分到同裝置／伺服器 | GPU-Server-A：8 GPU 96GB、900 GB/s 互連；GPU-Server-B：8 GPU 48GB GDDR6、PCIe；皆單機（GPU 品牌頁面未明示）；NVLS 未提 | DeepSpeed-MoE（含 TED、Tutel）、SGLang | s-MoE vs DeepSpeed-MoE：TTFT 1.69–2.37×、TPOT 1.14–2.93×、p90-TBT 1.48–6.54×；路由預測平均準確率 89% | §5.5 "a 100% LAR may not be achieved in theory" 因 token 偏好衝突與 GPU 記憶體；§4 未來支援 vLLM／TensorRT-LLM | 頁面未給 | **部分**：放置／預測演算法可在單機驗證 | https://arxiv.org/abs/2503.04398 ；https://arxiv.org/html/2503.04398v3 |
| P22 | **Comet: Fine-grained Computation-communication Overlapping for Mixture-of-Experts**；Zhang, Zheng, Lin, Jiang, Bao, …, Liu（ByteDance Seed、上海交大）；2025（arXiv 2502.19811；頁面未顯示提交日）；MLSys 2025（僅由搜尋結果得知：proceedings.mlsys.org，**未直接打開該頁**） | 粗粒度 pipeline 重疊會損害計算效率 | 以 shared tensor 資料相依分析與任務重排，把通訊與計算融合進同一 GPU kernel，調整各自分配的 thread block 數 | 主測 8×H100 80GB NVLink（平均 377 GB/s）單機；另 8×L20 PCIe（約 25 GB/s）；NIC、節點數未說明；NVLS 未提 | Megatron-Cutlass、Megatron-TE、FasterMoE、Tutel | 單層 1.96×、端到端平均 1.71×（Mixtral 8x7B、Qwen2-MoE、Phi3.5-MoE）；平均隱藏 86.5% 通訊延遲 vs FasterMoE 29.2%、Tutel 68.6%；L20：1.19–1.46× | §3.2.1 "the thread number restriction of warps constrict the communication operator to fully utilize the communication bandwidth."；§1 提出可用 Triton／TVM 等 compiler 實作 | https://github.com/bytedance/flux（論文所列；授權未說明） | **可**：單機 8×H100 | https://arxiv.org/abs/2502.19811 ；https://arxiv.org/html/2502.19811v2 |
| P23 | **NetMoE: Accelerating MoE Training through Dynamic Sample Placement**；Liu, Wang, Fu, Miao, Zhu, Nie, Cui；**ICLR 2025** | 同一訓練樣本的 token 傾向路由到相似專家，可用來減少 all-to-all | 動態重排樣本到 GPU，整數規劃（多項式時間）求最佳放置；expert residual inlining | 摘要：32 GPU；搜尋結果稱 NVLink 400 GB/s、IB 100 GB/s（**僅來自搜尋摘要，未在原文核對**）；NVLS 未提 | FastMoE、FasterMoE、SmartMoE（來自搜尋結果） | 32 GPU 上最多 1.67× 效率提升（ICLR 頁摘要） | 未取得全文，限制章節未核對 | 未核對 | **部分**：規模 32 GPU 相同；細節待讀全文 | https://proceedings.iclr.cc/paper_files/paper/2025/hash/e0c256700465c158de71081b4cf5e8c3-Abstract-Conference.html |
| P24 | **MoNTA: Accelerating Mixture-of-Experts Training with Network-Traffic-Aware Parallel Optimization**；Guo, Liu, Meng, Tao, Liu, Chen, Li（Enflame）；2024-11-01；arXiv 2411.00662 | TP 下 all-to-all 有資料冗餘，且未考慮網路拓樸 | 節點間與節點內 all-to-all 流水線化，配合 D2D copy（O1/O2/O3 變體） | 2 節點×8 A800 SXM4 80GB（16 GPU）；NVLink（§設定寫 400 GB/s，Table 5 寫 200 GB/s 單向，作者未調和）；IB 200 Gb/s；NVLS 未提 | DeepSpeed 0.14.2（all-to-all）、自家未優化 baseline | 摘要：8-way TP 下 all-to-all 最多約 8×（內文最佳為 baseline 的 0.253 延遲，256K 序列）；2x70B MoE、16 A800、8K 序列、TP=8／EP=2：端到端延遲 −13% | §7 "The next step will be analyzing the impact of kernel scheduling on MoE parallel optimization performance"；§4.4 O2/O3 在 8K 較差，歸因 AllGather 與 D2D copy kernel 記憶體衝突 | https://github.com/EnflameTechnology/DeepSpeed ；授權未說明 | **部分**：2 節點 16 GPU 可做，但針對 TP+EP 組合 | https://arxiv.org/abs/2411.00662 ；https://arxiv.org/html/2411.00662 |
| P25 | **Lancet: Accelerating Mixture-of-Experts Training via Whole Graph Computation-Communication Overlapping**；Jiang, Tian, Jia, Zheng, Wu, Wang；2024-04-30；**MLSys 2024** | all-to-all 與 weight-gradient 計算、非 MoE 前向計算的重疊 | 整圖層級：拆 backward 的 weight gradient 計算與 all-to-all 重疊；對 forward 做 partition 並用 cost model 規劃 | 8 節點×8 A100 80GB、4×100G NIC（64 GPU）；8 節點×8 V100、1×100G NIC（64 GPU）；NVLink／NVSwitch 未提；NVLS 未提 | DeepSpeed 0.5.8、Tutel 0.3、RAF | 非重疊通訊時間最多降 77%（摘要）；§7.1：A100 對 RAF −69%、對 Tutel −66%；Switch gate 最多 1.21×（A100）／1.3×（V100） | §8 未來工作 "Investigating the efficient orchestration and overlapping of communication arising from various sharding techniques"；§8 不規則 all-to-all 可再改善、padding 有額外成本 | 頁面未給 | **部分**：概念可重做，但原測 A100／V100 與 64 GPU | https://arxiv.org/abs/2404.19429 ；https://arxiv.org/html/2404.19429v1 |
| V1 | **Optimizing Communication for Mixture-of-Experts Training with Hybrid Expert Parallel**（NVIDIA 技術部落格，**非同儕審查**）；Yu, Liu, Sun；2026-02-02 | Hybrid-EP：MoE EP 通訊且最少用 SM | NVLink 用 TMA、RDMA 用 IBGDA；combine 先節點內部分累加再節點間聚合 | 單機 8×H100；4 台 DGX Hopper（32 GPU，CX7 400G/rail）；GB200 NVL36；NVLS 未提 | DeepEP、all-to-all（Megatron Core） | 8 SM 可填滿 H100 NVLink、約 4 SM 接近 NIC 上限（4 節點）；GB200 上 DeepSeek-V3 MXFP8：829→943 TFLOPS/GPU（1.14×）；Qwen3-235B MXFP8 728→800（1.10×） | "Some additional integration work is required to use Hybrid-EP in the PyTorch-based Megatron Core framework."；"This sync can be avoided if the host predefines a sufficiently large buffer size." | https://github.com/deepseek-ai/DeepEP/tree/hybrid-ep ；授權頁面未說明（DeepEP 主 repo 為 MIT） | **可**：單機與 4 台 H100 為其 benchmark 規模之一（Megatron 結果為 GB200，不可） | https://developer.nvidia.com/blog/optimizing-communication-for-mixture-of-experts-training-with-hybrid-expert-parallel |
| V2 | **DeepEP**（deepseek-ai/DeepEP，GitHub 專案頁，非論文）；日期頁面未顯示 | EP dispatch/combine GPU kernel 函式庫 | HT（NVLink＋RDMA）與 LL（純 RDMA）兩組 kernel；README 稱 V2.5 移除 NVSHMEM 依賴 | 需 Hopper 以上、NVLink 節點內、RDMA 節點間（IB 完整測試、RoCE 理論相容）；CUDA 13.1+、PyTorch 2.10+、NCCL 2.32.3+（**注意：與我們 CUDA 12.x 驅動 550 的相容性是潛在問題，未驗證**） | — | 頁面無 benchmark 數字 | "EP dispatch and combine require GPU SMs; zero-SM RDMA EP is not supported"；Bucket／Engram／PP 為實驗性 | MIT；https://github.com/deepseek-ai/DeepEP | **待驗證**：README 要求 CUDA 13.1+，我們驅動 550（CUDA 12.x）可能不支援；P1、P11 所用 v1.2.1 需 NVSHMEM | https://github.com/deepseek-ai/DeepEP |
| V3 | **Deploying DeepSeek with PD Disaggregation and Large-Scale Expert Parallelism on 96 H100 GPUs**（SGLang Team，LMSYS 部落格，**非同儕審查**）；2025-05-05 | 用開源元件重現 DeepSeek 大規模 EP serving | PD 分離＋DeepEP（prefill 用 normal、decode 用 low-latency）＋EPLB＋two-batch overlap（TBO） | 96×H100（12 節點×8）、節點間 IB（NIC 型號未給）；decode EP72（9 節點）、prefill EP32（4 節點）；NVLS 未提 | SGLang TP16×6、DeepSeek 官方數字與 profile | 每節點 52.3k input／22.3k output tokens/s（2,000-token 輸入）；decode 比 TP16 快 5.2×；EPLB prefill 1.49×、decode 2.54×；TBO prefill +27–35%；decode 每裝置 32 tokens 時 TBO 反而 −27% | 作者稱 TTFT 2–5 s、ITL 約 100 ms 需改善；"Limited to shorter sequences due to the use of 96 GPUs."；EPLB 只測 in-distribution；僅支援 Hopper | https://github.com/sgl-project/sglang/issues/6017 ；授權未說明 | **部分**：96 GPU 超出；4 節點 prefill EP32 可做 | https://lmsys.org/blog/2025-05-05-large-scale-ep/ |
| V4 | **vLLM Large Scale Serving: DeepSeek @ 2.2k tok/s/H200 with Wide-EP**（vLLM Team 部落格，**非同儕審查**）；2025-12-17 | vLLM wide-EP 部署 | EP＋DP；整合 DeepEP HT／LL；Dual Batch Overlap | CoreWeave H200、ConnectX-7 IB、「multi-node」，節點數未說明；NVLS 未提 | 先前 vLLM 約 1.5k tok/s/GPU | 持續 2.2k tokens/s／H200（llm-d 社群 benchmark，具體 checkpoint 頁面未明示） | Roadmap：elastic EP、disaggregated 獨立 TP size、GB200 優化（無獨立限制章節） | https://github.com/vllm-project/vllm ；授權未說明 | **部分**：H200 2 台（16 GPU）為小規模版本 | https://vllm.ai/blog/large-scale-serving |

---

## 3. 作者自述的限制與未來工作彙整（依出現次數）

（次數為本報告 29 項中「作者在原文自述」的篇數；括號內為編號。）

**A. 實驗規模或硬體受限、大規模結果為模擬或外推（約 9 項）**
- P9 LAER-MoE §7：「While empirical experiments on large-scale clusters are limited by resource constraints」
- P4 TEMPO §7 L1／L2：「Phase/scale figures come from the calibrated simulator」；完整 DeepSeek-V3 規模（EP32+）為外推
- P5 EasyBalance App. D：「The experiments are conducted only on A800-SXM4 80GB GPUs」
- P14 GRACE-MoE §6：「Extending GRACE-MoE to larger models and industrial-scale clusters will further validate its efficiency and scalability.」
- P13 FineMoE §7.5：「Our testbed consists of 8 GPUs but only 2 NICs per node, resulting in limited inter-node network bandwidth.」
- P15 HierMoE §V-A：因 GPU 記憶體限制縮小模型
- P16 FlashMoE：多節點結果為推估、僅單機 8×H100（頁面摘要）
- V3 SGLang："Limited to shorter sequences due to the use of 96 GPUs."
- P18 Occult：僅單節點 PCIe A6000（硬體事實，非自述限制）

**B. 實作尚不成熟、或綁定特定傳輸／NIC（約 8 項）**
- P1 NCCL EP §IX：「NCCL EP is currently an experimental API.」；HT 多節點效能待優化
- P11 GIN §V-D：實作「remains under active development」；§III-A 需 symmetric window
- P10 UCCL-EP §5.4：「current EFA firmware is not able to process small tokens quickly enough」
- P12 fabric-lib §8：「GDA is not yet available on most cloud instances …」；「Porting to additional NICs requires per-hardware tuning, not a redesign of the library」
- P20 Triton-distributed §4.2：「current NVSHMEM bitcode library doesn't support IBGDA」
- V1 Hybrid-EP：需額外整合工作、需 host sync 決定 buffer 大小
- V2 DeepEP README：zero-SM RDMA EP 不支援
- P1 §VII-A：「NCCL EP currently only supports the hybrid NVLink-RDMA path.」

**C. 通訊占用 SM／計算資源，或重疊受限於資源劃分（4 項）**
- P19 §4.4.1：通訊最多佔 20 SM／H800，"complicates communication pipeline kernel implementations"
- P22 Comet §3.2.1：「the thread number restriction of warps constrict the communication operator to fully utilize the communication bandwidth.」
- P2 Megatron Core §4.2.3：保留 SM 給 all-to-all 降低 GEMM 效率；「The first FWD and last BWD remain on the end-to-end critical path, so their all-to-all cannot be hidden.」
- V2 DeepEP README："EP dispatch and combine require GPU SMs; zero-SM RDMA EP is not supported"

**D. 負載不均仍未完全解決，或平衡手法需額外記憶體／假設穩定形狀（約 7 項）**
- P18 Occult §6：「some problems remain unsolved, such as workload balancing.」
- P14 GRACE-MoE §6：replication 增加記憶體需求
- P8 DisagMoE §7：「assumes pretraining-style stable shapes such as fixed sequence length and micro-batch size」
- P17 MegaScale-MoE §7：MoE 計算 operator 是 straggler 主因
- P21 Speculative MoE §5.5：「a 100% LAR may not be achieved in theory」
- P3 UltraEP §8.2：剩餘差距主因為 "uneven routing in realistic MoE training"；範圍不含 decode（§3）
- P5 EasyBalance App. D：需要 micro-batching

**E. 自動化不足、需手動調校（4 項）**
- P17 MegaScale-MoE §7：「We leave automatic optimization for future work.」
- P25 Lancet §8：sharding 技術下的通訊 orchestration 待研究
- P24 MoNTA §7：下一步為分析 kernel scheduling 影響
- P20 Triton-distributed §4.2：MoE+RS 需專用 ReduceScatter kernel（未來）

**F. 範圍限制：只做推論／只做訓練／不含 decode（4 項）**
- P16 FlashMoE §5：「This work targets inference」
- P3 UltraEP §3：不含 decode
- P1 NCCL EP §IX：訓練（HT）效能受限
- P11／P12：皆提及後續需推廣到更多框架／NIC

**G. all-to-all kernel 的 padding 開銷（2 項）**
- P7 §5.3：「all-to-all kernel pads data to the largest EP rank's size」
- P25 Lancet §8：不規則 all-to-all 與 padding 的額外成本

---

## 4. 常見比較對象

- **DeepEP**：P1、P3、P10、P11、P12、P20、V1、V3、V4 皆以它為比較對象或整合對象；P1／P10／P12 的 kernel 級比較都以 DeepEP 為最強對照（P20 的跨機 all-to-all 在 64 GPU dispatch 與 128 GPU 仍是 DeepEP 較優，P1 combine 在 1–4 節點也落後 DeepEP 5–12%）。
- **Megatron-LM**（含 Megatron-TE、Megatron-CUTLASS）：P2、P3、P8、P9、P13、P15、P17、P22。
- **Tutel**：P8、P14、P15、P18、P22、P25。
- **FasterMoE**：P16、P22、P23；**SmartMoE／FlexMoE**：P9、P13、P15、P23。
- **EPLB（DeepSeek）／LPLB**：P3、P4、P5、P7。
- **DeepSpeed-MoE／DeepSpeed**：P14、P18、P21、P24、P25。
- **NCCL／RCCL 標準 all-to-all**：P2、P10、P20；**NVSHMEM／pplx-kernels**：P10、P11、P12。
- **公認最強對照**：kernel／函式庫層以 DeepEP（含 v1.2.1 與 hybrid-ep 分支）；負載平衡層以 EPLB 與 LPLB；端到端訓練以 Megatron-LM。（此為本次讀到的使用頻率，不是裁決。）

---

## 5. 硬體覆蓋

**在 8×H100／H200 NVSwitch 單機，或 ≤4 台 IB 叢集上做過實驗：**
- P1 NCCL EP（EOS，1–8 節點 H100；vLLM 1/2/4 節點）
- P10 UCCL-EP（4 台 H100＋CX7 IB；4 台 H200＋EFA；4 台 B200＋EFA）
- P11 GIN（EOS，1–8 節點 H100）
- P12 fabric-lib（H100＋CX7、H200＋EFA，MoE 8–64 GPU）
- P13 FineMoE（4×8 H100 SXM，2×400G IB／節點）
- P16 FlashMoE（單機 8×H100）
- P22 Comet（單機 8×H100）
- P20 Triton-distributed（H800，8–16 GPU 為主）
- V1 Hybrid-EP（單機與 4 台 DGX Hopper）
- V3 SGLang（96 H100，含 4 節點 prefill）
- V4 vLLM（H200 multi-node，節點數未說明）
- P2 Megatron Core（1,024 H100，含小規模未明示）

**只在 A100／A800／A6000／消費級 GPU／AMD／GB200／模擬器做過：**
- A100／A800：P9（32 A100）、P24（16 A800）、P25（A100／V100）、P5（單機 A800）
- A6000／消費級：P15（A6000）、P18（A6000 PCIe）、P14（RTX 3090／4090）
- GB200／GB300：P2 主要結果、V1 的 Megatron 表
- AMD：P10 的 MI300X 部分
- 模擬器：P4 的相位圖為校準模擬器
- GPU 型號未說明：P3、P21（8 GPU 96GB／48GB，品牌未明示）、P6

**H800（非 H100，NVLink 被降為 400 GB/s）：** P8、P17、P19、P20。

**「NVLS 開與關之間比較」：** 在我打開並讀到的範圍內，**沒有任何一篇**做 NVLS 開關比較。P1 僅在 §II-C 提到 multimem 為 Device API 模組、未說 NCCL EP 使用；其他多數頁面未提 NVLS。（注意：多篇全文被截斷，可能漏掉後段；此結論「未發現」不等於「不存在」。）

---

## 6. 開源工具與程式碼清單

| 名稱 | 網址 | 授權 |
|---|---|---|
| DeepEP | https://github.com/deepseek-ai/DeepEP | MIT（README） |
| DeepEP hybrid-ep 分支 | https://github.com/deepseek-ai/DeepEP/tree/hybrid-ep | 分支本身未說明（README 稱為實驗分支） |
| NCCL（含 NCCL EP、GIN 的實作所在） | https://github.com/NVIDIA/nccl | 本次讀到的論文頁未說明 |
| UCCL-EP | https://github.com/uccl-project/uccl/tree/main/ep | 未說明 |
| pplx-garden（Perplexity fabric-lib／MoE kernels） | https://github.com/perplexityai/pplx-garden/ | 未說明 |
| FlashMoE | https://github.com/osayamenja/FlashMoE | 未說明 |
| Comet（論文所列網址） | https://github.com/bytedance/flux | 未說明 |
| Triton-distributed | https://github.com/ByteDance-Seed/Triton-distributed | 未說明 |
| Occult | https://github.com/UNITES-Lab/Occult | 未說明 |
| Least-Loaded EP | https://github.com/SalesforceAIResearch/LeastLoadedEP | Apache-2.0 |
| LAER-MoE（Hetu-Galvatron 分支） | https://github.com/PKU-DAIR/Hetu-Galvatron/tree/laer-moe | 未說明 |
| EasyBalance（EasyInfra） | https://github.com/yize-wu/EasyInfra | 未說明 |
| MoNTA（Enflame DeepSpeed） | https://github.com/EnflameTechnology/DeepSpeed | 未說明 |
| Megatron-LM | https://github.com/NVIDIA/Megatron-LM | 未說明 |
| SGLang（large-scale EP 重現說明） | https://github.com/sgl-project/sglang/issues/6017 | 未說明 |
| vLLM | https://github.com/vllm-project/vllm | 未說明 |
| 未公開／頁面未給網址 | P3 UltraEP、P4 TEMPO、P7、P8 DisagMoE、P13 FineMoE、P14 GRACE-MoE（稱接受後釋出）、P15 HierMoE、P17 MegaScale-MoE、P21、P25 Lancet | — |

---

## 7. 搜尋紀錄

**WebSearch（standard）查詢與使用情形（每次回傳約 10 筆）：**
1. `MoE expert parallelism all-to-all communication optimization arXiv 2025` → 取 Occult、MegaScale-MoE、NetMoE、MoNTA（2411.00662）；DeepSpeed-MoE（2022）排除（年份太舊）
2. `DeepEP efficient expert-parallel communication library NVLink RDMA` → NCCL EP、DeepEP 專案頁；各 fork（coderzc、Wohox、gitee）排除（非原作）
3. `MoE training overlap all-to-all computation fine-grained arXiv` → MegaScale-MoE、FlashMoE、DisagMoE；LoongForge 文件排除（框架文件，非研究）
4. `hierarchical all-to-all MoE inter-node intra-node NVLink InfiniBand` → HierMoE、NCCL EP、Perplexity 部落格；MoESys（2022）排除（年份太舊）；Perplexity EFA 部落格未獨立收錄（內容併入 fabric-lib 論文）
5. `MoE load imbalance expert parallel token dispatch system arXiv 2026` → UltraEP、TEMPO、LLEP、EasyBalance、LAER-MoE、FineMoE、Capacity-Aware Inference（2503.05066；**搜尋有看到但未實際打開核實，故未列**）
6. `Hybrid-EP NVIDIA expert parallel TMA IBGDA Megatron-Core` → NVIDIA 部落格、Megatron Core 論文（2603.07685）
7. `Comet fine-grained computation-communication overlapping Mixture-of-Experts MLSys` → Comet；slideslive 等轉載排除
8. `device-initiated GPU communication MoE dispatch combine NVSHMEM kernel latency decode` → GIN（2511.15076）、fabric-lib（2510.27656）；Nebius 部落格排除（二手）
9. `MoE inference expert parallelism communication Hopper H100 all-to-all low latency arXiv 2025 2026` → 2604.23150；MoE-Inference-Bench（OSTI）排除（非通訊聚焦）
10. `TCCL OR UCCL-EP OR "MoE-Lightning" OR "Speculative MoE" ...` → UCCL-EP；TCCL、MoE-Lightning 無結果
11. `Speculative MoE ... pre-scheduling arXiv` → 2503.04398
12. `Lancet OR ScMoE OR FlowMoE OR Pro-Prophet ...` → Lancet、MoNTA；ScMoE、FlowMoE、Pro-Prophet 未出現，未列
13. `Triton-distributed OR TileLink ...` → Triton-distributed（列入）；TileLink（2503.20313）與 AutoOverlap（2601.20595）**搜尋有看到但未打開，故未列**
14. `wide expert parallelism DeepSeek-V3 inference ... SGLang ...` → DeepSeek-V3 Insights、SGLang、vLLM 部落格；2512.16134 與 Atlas Cloud 文章未打開，未列
15. `MoE all-to-all token deduplication or compression or routing locality ...` → HierMoE、GRACE-MoE、2604.23150；LSH-MoE（2411.08446）未打開，未列
16. `NetMoE: Accelerating MoE Training through Dynamic Sample Placement ICLR 2025` → NetMoE ICLR 頁

**排除的已打開項目：** 2510.20171（Collective Communication for 100k+ GPUs／NCCLX，Meta；非 MoE 專屬，已核實存在，建議歸入 T1／T4）；2504.19925（SYMI，專家狀態解耦，與通訊關聯較弱，已核實存在）。

**WebFetch：** 約 40 次，開了每篇 arXiv abs／html（或 ICLR／PMLR 頁、廠商頁、GitHub README）。

---

## 8. 數據或敘述互相不一致之處（僅列出，不裁決）

1. **DeepEP 是否依賴 NVSHMEM**：P1、P11 稱 DeepEP（v1.2.1）使用 NVSHMEM/IBGDA；DeepEP README（V2.5）稱 "NVSHMEM is no longer a dependency"。可能為版本差異；README 還要求 CUDA 13.1+。
2. **H800 NVLink 頻寬**：P19 §4.1 稱由 900 降至 400 GB/s，§4.3 又用約 200 GB/s（可達約 160 GB/s）；P8 §6.1 寫 400 GB/s、§3.1 寫 "near 900GB/s"；P24 §設定寫 400 GB/s、Table 5 寫 200 GB/s（A800）。
3. **P17 MegaScale-MoE**：引言稱 1.88× 為 MFU，摘要與 Table 3 稱為吞吐。
4. **P6 LLEP**：PMLR 摘要寫最多 5× 加速，README 寫最多 6×。
5. **P24 MoNTA**：摘要稱 all-to-all 約 8×，內文最佳為 baseline 的 0.253 延遲（約 4×）。
6. **P1 NCCL EP 計時方式**：作者自述其為 host-side C API 時間（含啟動開銷），DeepEP 為 GPU kernel 時間（Kineto），因此 LL dispatch／combine 的差距方向受影響（§VII-A）。
7. **P20 Triton-distributed vs DeepEP**：作者稱 dispatch／combine 平均 1.18×／1.44× 快於 DeepEP，但 64 GPU dispatch 與 128 GPU 仍是 DeepEP 較優。
8. **P12 fabric-lib**：RL 權重更新時間摘要寫 1.3 s、§7.3 寫 1.2 s。

---

## 9. 未能核實或僅部分核實的項目

- P22 Comet 的「MLSys 2025」僅見於搜尋結果（proceedings.mlsys.org 連結），**未直接打開該頁**；arXiv html 頁面本身未列 venue 與提交日。
- P23 NetMoE：沒找到 arXiv 版本，只打開了 ICLR 論文集的摘要頁；硬體與限制章節未核對（搜尋摘要提到的 NVLink／IB 頻寬標為未核對）。
- P9 LAER-MoE 的「all-to-all 時間占比 <10% → >40%」數字來自搜尋結果摘要，未在原文章節核對；P18 Occult 的「>40%」同。
- P21 Speculative MoE 的後續版本（至 v5）資訊來自搜尋結果，頁面僅見 v3。
- P2 Megatron Core：全文約 41 萬字元，只讀到部分（1–20 萬、30–40 萬、40 萬後段）；Section 8 的詳細硬體、baseline 與限制章節未取得。
- P6 LLEP：PMLR 頁面無 arXiv 連結、PDF 為壓縮檔無法解析，硬體與限制章節未取得。
- 多篇（P1、P3、P4、P9、P10、P12、P13、P17）全文頁被截斷，後段（可能含附錄與限制）未讀。
- 所有「原文引用」為 WebFetch 的摘要工具轉述，建議引用前再對原文核一次。
