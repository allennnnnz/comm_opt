# R04_CD：C「通訊的 SM 稅」與 D「collective 選擇的效能模型」深入調查

- **對應規格書**：`docs/specs/S04_direction_deep_dive.md`（CD 組）　**執行**：Sonnet 5.5，2026-11-16
- **範圍**：只提供有出處的事實，不判斷題目好壞。論文編號沿用 R02（T1-Pxx、T2-#xx），新增的以 arXiv 編號標示。
- **NCCL 原始碼**：`/work/allen0117/comm_opt_env/src/nccl`（v2.32.6-1，repo 只有這一個 tag，無法用 `git show` 讀 2.29 原始碼）。**我們安裝的 NCCL 2.29.7**（`venvs/vllm/.../nvidia/nccl/lib/libnccl.so.2`，`nccl.h` 22907），因此「2.29.7 是否已有」用三種方式判斷：(a) 安裝版 `nccl.h` 標頭；(b) 對 `libnccl.so.2` 做 `strings`（只查字串存在與否，不執行）；(c) 官方 release notes／env 文件的「since」標註。三種都沒有證據的標「未能確認」。
- **登入節點檢查**：只做了 `nvidia-smi`（唯讀）、`strings`、`grep` 指定路徑，沒有跑運算。

---

## 白話結論（給使用者）

1. **「通訊佔用計算資源」這件事在 H100 上已有人量過，但多數是別種條件**：在 8 張 H100 的機器上量到「同時做通訊，計算平均慢 18.9%、最慢 40%」的是 Lee et al.（R02 已列）。Zero-SM NCCL（2026-11-13）量到「舊式通訊讓同時執行的計算慢約 33%，新式多播核心慢 6%，改用拷貝引擎只慢 3.9%」，但那是 Blackwell 機器、而且還不支援加總類操作。
2. **有一批新論文（2026 年 6–11 月）已經在做「依工作量動態決定通訊用多少 SM」**，其中 Weave（4 張 H100）、Analytical Resource Management（A100）、OverlapAdvisor（A100／H20／H800）都帶有成本模型；它們都針對 MoE，不是針對一般 all-reduce／all-gather 與 GEMM 同時執行。
3. **NCCL 本身已有減少 SM 的開關，但我們的環境可能用不了其中一部分**：限制 SM 數的環境變數與「效率優先」政策在 2.29.7 都有；但「拷貝引擎（零 SM）」版的 collective，原始碼要求驅動程式支援 CUDA 12.5 以上，而這個叢集登入節點顯示驅動 550.127.08、CUDA 12.4。計算節點是否相同要在 M20 確認。
4. **NCCL 內建的「選演算法」模型只考慮單獨一個 collective 在空閒 GPU 上的時間**，不知道旁邊有計算在跑；NVIDIA 官方外掛介面（tuner plugin）也不回報實測時間。2026-11-08 的 NCCL-BOT（NVIDIA，H100）自己寫明「沒有測計算與通訊重疊」。「同時有計算」情況下的演算法選擇模型，這次調查沒找到。
5. **仍有一部分沒能查完**：arXiv 搜尋在後半段被限流，部分搜尋改走 Semantic Scholar／網頁搜尋；見第 7 節。

---

## 1. 重點事實（附出處）

**F1　H100 上重疊使計算變慢的既有量測。** Lee et al. 在 DGX H100（8×H100）等平台量到：重疊使計算平均變慢 18.9%、最高 40.0%（相對無通訊理想值）；序列執行比重疊平均慢 10.2%、最高 26.6%；功率上限下重疊最多讓時間增加 107%（T2-#13；`papers/2507.03114.pdf` 第 1 頁摘要與引言，我用 `pdftotext` 核對 "18.9%"、"40.0%"、"107%" 三個數字都在文中）。

**F2　Zero-SM NCCL 的量測（GB200／B200，非 H100）。** 一台 B200 上，144 MiB AllGather 使同時執行的 Qwen3-30B-A3B GEMM「約慢 33%（NCCL 舊式 ring）、6%（symmetric multicast kernel）、3.9%（copy engine）」；Fig. 1A 說明 NCCL 飽和 NVLink 所需 SM 數「從 Pascal 的 8 增加到 Blackwell 的 32」（`papers/2611.18075.pdf` 第 1 頁圖說與引言；arXiv:2611.18075）。圖 1A 中 Hopper 那一格的 SM 數，我從文字抽取無法確定對應哪個數字（出現「24 SMs (18%)」「24 SMs (22%)」），**未核實**。
- 作者自述：Reduction（AllReduce、ReduceScatter）「not yet supported」（§III-A）；機內小訊息 copy engine 延遲較高（1 KiB AllGather：copy engine 25–29 µs、SM kernel 12–18 µs，§IV-B2；R02_T1-P1）。

**F3　AMD DMA 路線 ConCCL。** C3（concurrent computation and communication）平均只達理想加速的 21%；以 CU 分割＋排程優先權提高到 42%；再把通訊交給 GPU DMA engine（ConCCL）達 72%、最高 1.67×。硬體是 8×AMD MI300X（`papers/2412.14335.pdf`；arXiv:2412.14335 摘要；§II-A 明寫 "we study C3 using AMD Instinct MI300X"）。ConCCL 原型涵蓋 all-gather 與 all-to-all（文中 §II-A 附近稱 "do not consider offloading all-reduce kernels to DMAs" 的是既有工作描述，我未逐字核實歸屬，只作參考）。

**F4　H100 上用「共享記憶體占用率」調控 GEMM 駐留量的做法。** Resource-aware Computation-Communication Overlap（Cui & Pericas，arXiv:2606.09200）：不改 NCCL 與 kernel，用 per-block 共享記憶體配置限制 GEMM 的 resident blocks，並把通訊 stream 設高優先權；平台為 A40／A100／H100／MI250X，每台 4 張（MI250X 8 張，Table 1），集體通訊資料量固定 896 MB；總時間最多減少 25.5%。thread-block 數受限時 TimeRatio 約 0.3，數量增加後趨近 1（§4.2）；「Future work includes exploring adaptive runtime policies that automatically tune occupancy and priority settings」（結論）。

**F5　H100（4 張）上的動態 SM 分配。** Weave（arXiv:2609.21483，4×H100 SXM、NCCL 2.21.5、NVSHMEM 3.6.5）：指出 DeepEP 固定把 132 SM 中的 20 個給通訊；同一個 MoE 層內「per-GPU optimal c* ranges from 12 to 36」，DeepEP 固定 c=20、Comet 每 iteration c=18 都對不上任何一張 GPU 的最佳值（§2.3.1）；megakernel 內跑輕量成本模型決定 SM 分工，MoE-layer 幾何平均加速 2.89×、端到端 1.33×。作者自述：因硬體限制僅單機，多機留待未來（Limitations and future work 段）。

**F6　分析式 SM／CTA 管理模型。** Analytical Resource Management（Liu, Cui, Pericas，arXiv:2609.07536）：「wave-quantized analytical model」在每次 launch 前選通訊 CTA 數，相對量測最佳解的平均 regret 3.22%（15 個真實 p90 工作負載），solver 開銷 0.157 µs；平台 4×A100-SXM4-40GB、NCCL 2.26.2；作者說 C ≤ 16 的合法範圍來自一次性 A100 characterization，「not an A100 hardware limit」。OverlapAdvisor（arXiv:2611.11857）：統一效能模型預測 MoE 重疊策略，MAPE 3.3%–15.9%、launch 時選擇約 6 µs；平台 A100／H20／H800／GB200（Table）；「Future work will extend OverlapAdvisor to multi-node deployments, model SM- and warp-level resource contention among overlapping kernels」（§8）。

**F7　NCCL 提供的減 SM 機制（v2.32.6 原始碼；2.29.7 狀態見第 3 節）。**
- `NCCL_MIN_CTAS`／`NCCL_MAX_CTAS`：`src/init.cc:2405-2434`（env 覆蓋 comm config）；預設 `minCTAs=1`、`maxCTAs=MAXCHANNELS`（`init.cc:2883-2886`）；文件：env.rst 「NCCL_MAX_CTAS (since 2.17) … up to 64 (32 prior to 2.25)」（`docs/userguide/source/env.rst:1906-1960`）。
- `NCCL_CTA_POLICY`（DEFAULT／EFFICIENCY／ZERO）：解析 `init.cc:86-142`；env.rst:885「(since 2.29, legacy values since 2.27)」；旗標說明 `docs/userguide/source/api/flags.rst`（EFFICIENCY：「use minimal number of CTAs to achieve the decent performance … suitable for applications which require better compute and communication overlap」；ZERO：「use zero CTA whenever it can, even when that choice may sacrifice some performance」）。
- EFFICIENCY 的實作：在沒有 tuner plugin、沒設 `NCCL_ALGO`／`NCCL_PROTO` 時，對已註冊 buffer 的 AllGather／ReduceScatter 偏向 NVLS 並取 `recChannels` 較少者（`src/tuning/tuning.cc:243-262`）。
- ZERO（copy engine）：`src/enqueue/enqueue.cc:3478` 條件 `CTAPolicy & ZERO` 且 CE 可用才走 `ceCollTaskAppend`；CE 只支援 AllGather、AlltoAll、Scatter、Gather（`src/ce_coll.cc:175-191`），文件需要「CUDA driver version >= 12.5」「buffer is symmetrically registered」（`docs/userguide/source/usage/bufferreg.rst:209-231`）。
- NVLS 的 CTA 數：Hopper（SM90）預設 16（`src/transport/nvls.cc:201` `channels = /*SM90=*/16`），Blackwell 單機 24、多機 32；可用 `nvlsCTAs` 或 `NCCL_NVLS_NCHANNELS` 覆蓋（`nvls.cc:181-205`；`init.cc:2596`）；release notes 2.28.3：「Decreased max CTA count from 32 to 16 on Blackwell … decreased SM overhead by 50%」（NVIDIA release notes 2.28.3，WebFetch 摘要）。
- Symmetric kernel：`NCCL_SYM_CTAS` 可指定 CTA 數（`src/tuning/sym_model.cc:19-27`），上限 `ncclSymkMaxBlocks = 64`、每 block 最多 512 threads（`src/include/sym_kernels.h:26-27`）。
- ring／tree 的 channel 數：Hopper 多機且 `bwIntra > 45` 且 `nChannels < 16` 時加倍（`src/graph/connect.cc:507-509`）；ring 與 tree 各複製一份使 `nChannels` 乘 2（`connect.cc:479`）。

**F8　NCCL 內建演算法選擇模型的結構（v2.32.6）。** 每個 (algo, proto) 的時間 = `lat × latCount + nBytes / (1000 × bw)`（`src/tuning/tuning_general.cc:58-71`），lat 與 bw 來自 `baseLatencies`、`hwLatencies`、`llMaxBws`、`perChMax*Bws` 等表（`src/include/plugin/tuner/tuner_v7.h:19-29`）：Ring 的 `busBw = bwIntra × nChannels`，LL 取 `min(llMaxBw, busBw×0.5)`，LL128 取 `min(busBw×0.92, nChannels×perChMaxRingLL128Bw)`（`src/tuning/ring.cc:38-43`）；NVLS 的頻寬乘 `nvlsEfficiency[compCap]`（`src/tuning/nvls.cc:63`）；AllReduce 路徑另有 Ring plateau 修正與 NVLSTree 校正表（`ring.cc:101-125`、`nvls.cc:155-165`）。模型輸入沒有「同時有計算」的欄位（`struct ncclTuningInput_t` 使用處 `tuning.cc:180-240` 僅見 comm、func、nBytes、numPipeOps、regBuff 等）。H100（SM90）的 copy-engine AllGather 另有一組寫死的延遲底限與 NVLink 頻寬 392 GB/s（`src/tuning/ce_model.cc:33-41`）。

**F9　Tuner plugin 介面能改什麼。** v7（2.32.6）：`init` 可讀寫 `ncclTunerConstants`（base／hw 延遲、各算力的頻寬上限）；`getCollInfo(collType, nBytes, numPipeOps, collCostTable, …, regBuff, &nChannels)` 可改 cost table 並輸出 `nChannels`（註解「number of channels (hence SMs) to be used」）；v7 新增 `getChunkSize` 可改 chunk size（`tuner_v7.h:56-118`）。**plugin 看不到實測延遲**（NCCL-BOT 原文：「NCCL never returns a chosen pair's measured latency to the plugin」，`papers/2611.11303.pdf` 引言）。實作上 plugin 回傳的 `nChannels` 被存為 `maxChannels`（`src/tuning/tuning.cc:225-237`）。

**F10　既有「調參」研究的條件。** AutoCCL（NSDI'25，T1-P22）與 Lagom（T1-P10／T2-#6）都在 A40 與 NCCL 2.18.3 上做，未提 NVSwitch／NVLS；Lagom 稱通訊競爭使計算最多降速 35%（§1）。NCCL-BOT（arXiv:2611.11303，NVIDIA）在 H100（HGX、Spectrum-X 乙太網路、NCCL 2.30.4）上量到：NCCL AUTO「loses 6.08% of latency on average but 103% at the 99th percentile」（held-out 較大規模切分）；自述 replay 限制：「The sweeps are isolated microbenchmarks: no cross-arm connection reuse, no compute–communication overlap or concurrent communicators」；NVLS 首次使用有 5.7–7.8 s 的設置停頓（H100）。
**F11　NCCL 2.29.7 在 NVLS 與 Ring 之間的選擇有實測矛盾。** NCCLbpf（arXiv:2603.11438，8×B300、NCCL 2.29.7）：「NCCL 2.29.7 defaults to the NVLS algorithm … for all message sizes. A parameter sweep reveals that Ring with 32 channels outperforms NVLS by 5–27% in the 4–128 MiB range, while NVLS is superior at 256 MiB and above」。硬體是 Blackwell，不是 H100。

**F12　主流系統分給通訊的 SM／block 數。**
- vLLM custom all-reduce：`kMaxBlocks = 36`、`defaultBlockLimit = 36`（CUDA），每 block 512 threads（`vllm/csrc/custom_collective_common.cuh:35-39`；`custom_all_reduce.cuh:233-240`；註解稱 36 blocks 是在 A100、A10、A30、T4、V100 上調出，並說明「NCCL kernels also only take a small amount of SMs」）。這是 v0.34.0 的 tag，我們裝 0.32.0，**未核對該行在 0.32.0 是否相同**。
- SGLang CustomAllReduceV2（sm90，標註 "Tuned on H200"）：`num_push_blocks=num_sm`、`num_pull_blocks=64`、`num_mc_blocks = 128 // world_size`（ws≥4）（`sglang/python/sglang/srt/distributed/device_communicators/configs/custom_all_reduce_v2.py:266-273`）。
- TensorRT-LLM 自製 all-reduce：`MAX_ALL_REDUCE_BLOCKS = 24`（`trtllm/cpp/tensorrt_llm/kernels/customAllReduceKernels.h:34`），`cta_num = min(norm_num, MAX_ALL_REDUCE_BLOCKS)`（`customAllReduceKernels.cu:1233`）。
- DeepEP v1.2.1：`Buffer.num_sms = 20`（`deepep/deep_ep/buffer.py:30`，docstring「the SMs used in high-throughput kernels」），必須為偶數（`buffer.py:152`）；Megatron 以 `moe_flex_dispatcher_num_sms`（None 表示各 backend 預設）控制，舊欄位 `moe_deepep_num_sms` 註明「historical default 20」（`megatron/megatron/core/transformer/transformer_config.py:1081-1089`）。
- Megatron／TransformerEngine userbuffers：Megatron 原始碼只暴露 `tp_comm_overlap` 一組布林（`megatron/megatron/core/model_parallel_config.py:300-370`）；SM 數（`num_sm`、`cga_size`、`set_sm_margin`）定義在 Megatron-Bridge 的 overlap 設定 dataclass：Ring-exchange 預設 `num_sm=1, cga_size=1, set_sm_margin=False`，bulk 與 pipeline 預設 None（`https://docs.nvidia.com/nemo/megatron-bridge/0.5.0/apidocs/bridge/bridge.training.comm_overlap.html`，WebFetch 摘要）。TransformerEngine 原始碼本次**未下載**，userbuffers 實際使用 SM 數的邏輯未讀。

**F13　主流系統已經在用 NCCL 的 SM 控制。** vLLM 在 EPLB 搭配 DeepGEMM Mega MoE 時自動設 `NCCL_MAX_CTAS=8`，原因是「cooperative launch blocks until enough SMs are freed, causing a deadlock」（`vllm/vllm/distributed/eplb/eplb_utils.py:100-123`）；SGLang 在 KV-shard 的 gather communicator 設 `NCCL_CTA_POLICY_ZERO`（`sglang/python/sglang/srt/mem_cache/page_interleave_pool.py:220`；`pynccl.py:117-124`）；PyTorch c10d 暴露 `min_ctas`、`max_ctas`、`cta_policy`、`nvls_ctas`（`pytorch/torch/csrc/distributed/c10d/init.cpp:4043-4056`）。

**F14　TMA 路線。** T-CCL（arXiv:2610.07098）：以 TMA 搬資料與做 reduction，用 NCCL 2.28 為 baseline；平台 2×H100 NVL（NV12）與 4×GH200（NV6），限制 CTA 預算為 8（H100）／9（GH200）時最多比 NCCL 快 3.42×；GEMM+collective 重疊 case study 平均加速由 1.12×（NCCL）升到 1.25×（2 GPU）、1.04×→1.14×（4 GPU）；vLLM 整合端到端最高 1.31×（GH200 TP=4）。作者自述：「plan to evaluate T-CCL on larger GPU topologies, such as 8- and 16-GPU systems」，不支援 CUDA Graph（§IV 末段，文字為 "Graph support … future work"）。

**F15　SM 之外的干擾來源（H100）。** Elvinger et al.（arXiv:2501.16909，H100 NVL 132 SMs，ETH）把 GPU 干擾分為 block scheduler、SM 內計算單元、L1/L2、記憶體頻寬等層，並用 CUDA MPS／Green Contexts 控制 SM 分配（§2、§4.3）。VCCL（arXiv:2510.00991，Hopper 叢集）觀察到「P2P primitives (e.g., SendRecv and AlltoAll) that do not involve reductions also consume non-negligible SM resources in NCCL」；其 SM-free P2P 使端到端訓練 TFLOPS 最多 +5.28%；自述機內小訊息延遲較高（「contention for limited copy-engine resources」，§4.1）。Arnold（arXiv:2509.15940）在 9600 GPU 規模觀察到：增加 NCCL channel 使通訊 kernel 變快、計算 kernel 變慢，推測是「less available SMs」（§附錄分析，推測性）。

---

## 2. 最接近的既有工作表

| 工作 | 出處（venue／年／網址） | 做了什麼 | 硬體與規模 | 與 C／D 重疊 | 沒涵蓋的條件（原文自述或明顯差異） |
|---|---|---|---|---|---|
| Lee et al., *Characterizing Compute-Communication Overlap…* (T2-#13) | arXiv 2507.03114，2025-07；https://arxiv.org/abs/2507.03114 | 量重疊對計算的拖慢與功率 | DGX H100（FSDP）、4×A100、MI250／MI210 | C：H100 上量化重疊拖慢（18.9%／40.0%） | 沒有按 collective 實作拆解 SM／CTA 用量；「未說明」限制章節（R02 §3） |
| Zero-SM NCCL (T1-P1) | arXiv 2611.18075，2026-11-13 | SM-free collective（copy engine＋stream memop＋CPU proxy） | GB200 NVL72、B200 NVL8；NCCL 2.32.3 含未釋出功能 | C：直接量 NCCL 各 kernel 對 GEMM 的拖慢與 SM 用量 | 無 reduction；機內小訊息較慢；非 H100（R02） |
| ConCCL (Agrawal et al.) | arXiv 2412.14335，2024-12；https://arxiv.org/abs/2412.14335 | 特徵化 C3，CU 分割＋優先權，DMA engine collective | 8×MI300X | C：干擾量化與減干擾方法（DMA 路線） | AMD；all-gather／all-to-all 為主；非 NVIDIA |
| FiCCO (T2-#9) | arXiv 2512.10236，2025-12 | DMA 為基礎的更細粒度重疊，附排程 heuristic | 8×MI300X | C：DMA、排程選擇 | AMD；NVIDIA 未測（R02） |
| Resource-aware Overlap (Cui & Pericas) | arXiv 2606.09200，2026-06；https://arxiv.org/abs/2606.09200 | 共享記憶體占用率調控 GEMM 駐留＋通訊 stream 高優先權 | A40／A100／H100（各 4 張）、MI250X | C：H100 上無需改 NCCL 的 SM 分配 | 只測 allreduce／alltoall 各一個 896 MB 工作；無端到端；自述未來做 adaptive policy |
| Weave | arXiv 2609.21483，2026-09；https://arxiv.org/abs/2609.21483 | MoE megakernel 內動態 SM 分工＋成本模型 | 4×H100 SXM，六個 MoE 模型 | C＋D：H100 上 SM 劃分最佳點隨層與 GPU 變動 | 僅 MoE；單機；多機留待未來 |
| Analytical Resource Management | arXiv 2609.07536，2026-09 | wave-quantized 分析模型選通訊 CTA 數 | 4×A100，COMET 實作 | C＋D：CTA 選擇的分析模型（regret 3.22%） | A100；範圍為 MoE GEMM2+GatherRS；C ≤ 16 來自一次性校準 |
| OverlapAdvisor | arXiv 2611.11857，2026-11-09 | MoE EP 重疊策略的統一效能模型與 launch 時選擇 | A100／H20／H800／GB200；SGLang、vLLM | D：含重疊的效能模型（MAPE 3.3–15.9%） | 僅 MoE EP；多機與 SM／warp 層級爭用留待未來 |
| T-CCL | arXiv 2610.07098，2026-10 | TMA 搬資料與 reduction 的機內 collective | 2×H100 NVL、4×GH200 | C：機內含 reduction 的低 SM 路線；CTA 預算受限實驗 | 2 與 4 GPU；無 CUDA Graph；8 GPU 未測（自述） |
| VCCL | arXiv 2510.00991，v2 2026-06 | NCCL 之上的 SM-free P2P、可觀測性 | Hopper＋ConnectX-7 叢集（至 24K GPU 部署） | C：SM-free P2P、SM 競爭觀察 | P2P 為主，非 reduction collective；機內小訊息較慢 |
| ParallelKittens (T1-P15) | arXiv 2511.13940 | 量三種 NVLink 傳輸機制與所需 SM 數 | 8×H100、B200 | C：H100 上 copy engine／TMA／register 的 SM 需求 | 單機（§5） |
| Comet (T2-#20)；mKernel (T2-#2) | arXiv 2502.19811；2609.13585 | 通訊 warp 與計算 warp 干擾；自適應 SM 分區 | H800 單機；2×8 H200 | C：自適應 SM 分區 | mKernel 自述自適應分區僅單機測（§5） |
| AutoCCL (T1-P22) | NSDI'25；https://www.usenix.org/conference/nsdi25/presentation/xu-guanbin | 線上調 NCCL 參數（含與計算並行情況） | A40，NCCL 2.18.3 | D：調參；有並行計算時最高 1.80×／1.49× | 未提 NVSwitch／NVLS；IB 專屬參數留待未來（§8.1） |
| Lagom (T1-P10) | arXiv 2602.20656 | 重疊感知的成本模型＋優先序搜尋 | 2 叢集×16 A40，NCCL 2.18.3 | D＋C：把與計算爭用放進模型 | 未提 NVSwitch／NVLS；程式碼未釋出 |
| NCCL-BOT | arXiv 2611.11303，2026-11-08；NVIDIA | 以 tuner plugin＋bandit 從實測延遲修正 AUTO 選擇 | H100 HGX（Spectrum-X、至 32 節點）；GB200 | D：H100 上量 AUTO 誤選；plugin 路線 | 自述 replay 無計算重疊；只選 algo／proto，不改 channel 數；不證明端到端增益 |
| NCCLbpf | arXiv 2603.11438，2026-03 | eBPF 驗證執行的 NCCL plugin；案例：NVLS 與 Ring 依訊息大小切換 | 8×B300，NCCL 2.29.7 | D：同版 NCCL 的 NVLS vs Ring 實測 | Blackwell；單機；AllReduce |
| Demystifying NCCL (T1-P19) | arXiv 2507.04786 | 文字分析 NCCL 協定與演算法 | GH200 | D：建模素材 | 搜尋結果指出其排除 NVLS／CollNet（見第 6 節，我未在全文核對） |
| PICO (T1-P18) | arXiv 2508.16809 | 跨 MPI／NCCL benchmark，預設選擇常比最佳慢 30–40% | Leonardo、LUMI、MareNostrum 5 | D：選擇性能差距 | 調參檔需手動建立（§4.1）；NCCL 數據未報告（R02） |
| Nuriyev & Lastovetsky | arXiv 2004.11062，2020 | 以分析式模型在執行時選 MPI collective 演算法 | CPU 叢集 | D：MPI 路線的模型選擇 | 非 GPU |
| PML-MPI；HAT-MPI | IPDPSW 2024 DOI 10.1109/IPDPSW63119.2024.00140；ACM DOI 10.1145/3806645.3816737（2026） | ML 預測 MPI collective 演算法／UCX 門檻 | CPU／IB 叢集 | D：ML 選擇器（最高 6.3%；準確率 84.9%） | 非 GPU；以 摘要核實，未讀全文 |

---

## 3. 現成軟體的支援程度

### NCCL（原始碼 v2.32.6；我們裝 2.29.7）

| 機制 | v2.32.6 原始碼位置 | 2.29.7 是否已有 | 依據 |
|---|---|---|---|
| `NCCL_MIN_CTAS`／`NCCL_MAX_CTAS`、`minCTAs`／`maxCTAs` | `init.cc:2405-2434,2883-2886`；env.rst:1906 | **有** | env.rst「since 2.17」；`strings libnccl.so.2` 含 `NCCL_MAX_CTAS`、`NCCL_MIN_CTAS`；安裝版 `nccl.h:91-92` |
| `CTAPolicy`（DEFAULT／EFFICIENCY／ZERO）、`NCCL_CTA_POLICY` | `init.cc:86-142,2588-2646`；`flags.rst` | **有** | 安裝版 `nccl.h:70-72,98`；字串含 `NCCL_CTA_POLICY`、"Both NCCL_CTA_POLICY_ZERO and … EFFICIENCY are set"；env.rst:885「(since 2.29, legacy values since 2.27)」 |
| `nvlsCTAs`、`NCCL_NVLS_NCHANNELS` | `nvls.cc:181-205`；`init.cc:68,2596` | **有** | 安裝版 `nccl.h:100`；字串含 `NCCL_NVLS_NCHANNELS`；release notes 2.27.3（WebFetch）提到 `nvlsCTAs` |
| 其他 config：`nChannelsPerNetPeer`、`nvlinkCentricSched`、`graphUsageMode` | `init.cc:2890-2900` | **有**（至 `numRmaCtx` 前） | 安裝版 `nccl.h:101-103,128`；`NCCL_NCHANNELS_PER_NET_PEER` 字串存在 |
| Symmetric kernel、`NCCL_SYM_CTAS`、`NCCL_SYM_KERNEL`、`NCCL_SYM_CE_THRESHOLD`、`NCCL_WIN_ENABLE` | `sym_model.cc:19`；`enqueue.cc:36`；`tuning/cost_model.cc:367` | **有** | 字串皆存在；`ncclCommWindowRegister`、`NCCL_WIN_COLL_SYMMETRIC` 在安裝版標頭（`nccl.h:64,341`）；release notes 2.27.3 |
| Copy-engine collective（`ncclCeAllGather` 等，需 `CTAPolicy=ZERO`） | `ce_coll.cc`；`enqueue.cc:3478` | **符號存在**（`ncclCeAllGather`、`ncclCeAlltoAll`、`ncclCeScatter`、`ncclCeGather`）；release notes 2.28.3 | 但 `ce_coll.cc:175-181` 要求 driver ≥ 12050；本機登入節點驅動 CUDA 12.4 → **可能用不了，需在計算節點確認** |
| `ncclAlltoAll`／`ncclGather`／`ncclScatter` host API | env／API | **有** | 安裝版 `nccl.h:532-564`；release notes 2.28.3 |
| NCCL 2.28 減 Blackwell max CTA 32→16 | release notes 2.28.3 | 有，但與 H100 無關 | WebFetch release notes |
| Tuner plugin 介面 | `src/include/plugin/tuner/tuner_v7.h`（v7，含 `getChunkSize`） | **只到 v5** | 安裝版 libnccl 字串只有 `ncclTunerPlugin_v2` … `_v5`（`strings` 結果，無 v6／v7）；v5 有 `init`（可改 constants）與 `getCollInfo`（輸出 `nChannels`），無 `getChunkSize`（`tuner_v5.h:36-84`）。安裝版 `include/` 沒有 tuner 標頭，要寫 plugin 需用原始碼 repo 內的 `tuner_v5.h` |
| 演算法／模型原始碼組織 | `src/tuning/*`（`ring.cc`、`nvls.cc`、`sym_model.cc`、`ce_model.cc`、`cost_model.cc`） | **不同**：2.32.6 把選擇拆成此資料夾；規格書提的 `src/graph/tuning.cc` 在 2.32.6 不存在（`src/graph/` 僅有 connect/paths/rings/search/topo/trees/xml） | 2.29 的原始碼無法讀（repo 只有一個 tag）；二進位字串顯示 2.29.7 已有 symmetric 與 CE 符號，但模型細節（含 `ce_model.cc` 的 H100 常數）是否存在 **未能確認** |
| `NCCL_ALGO`／`NCCL_PROTO`、`NCCL_THREAD_THRESHOLDS`、`NCCL_NTHREADS` | `tuning_general.cc:14-100`；env.rst | **有** | 字串存在 |
| `NCCL_CE_*` 一類環境變數 | `ce_coll.cc:19-31` | **未在 2.29.7 字串找到**（`NCCL_CE_COLL_AG_MULTICAST_THRESHOLD` 等無）；有 `CE_COLL_INTRA_BATCH_SYNC_FREQ` 字串 | `strings` 篩選結果；表示 2.29.7 的 CE 可調旋鈕較少 |

### vLLM（v0.34.0 原始碼；我們裝 0.32.0）
自製 all-reduce：預設與上限皆 36 blocks、512 threads（`vllm/csrc/custom_collective_common.cuh:35-39`；`custom_all_reduce.cuh:240`）；EPLB＋Mega MoE 時強制 `NCCL_MAX_CTAS=8`（`eplb_utils.py:100-123`）。0.32.0 是否相同：**未核對**。vLLM 沒有找到對 `CTAPolicy` 的使用（grep `CTAPolicy|NCCL_CTA` 在 vllm/vllm 僅命中 `NCCL_MAX_CTAS`）。

### SGLang
CustomAllReduceV2 sm90：push blocks＝SM 數、pull 64、multicast 128/ws（`configs/custom_all_reduce_v2.py:266-273`）；有 `NCCL_CTA_POLICY_ZERO` 常數與 per-communicator `cta_policy`（`pynccl_wrapper.py:91-92`；`pynccl.py:37-48,117-124`），用於 KV-shard gather（`page_interleave_pool.py:220`，`environ.py:618` 註解）。

### TensorRT-LLM
自製 AR `MAX_ALL_REDUCE_BLOCKS=24`（`customAllReduceKernels.h:34`）；R03 已述 AUTO 查表。

### Megatron／TransformerEngine
Megatron 只暴露 `tp_comm_overlap` 布林組合；SM 數由 TE userbuffers 設定（`num_sm`、`cga_size`、`set_sm_margin`，見 F12）；DeepEP 的 SM 數經 `moe_flex_dispatcher_num_sms`。Megatron 內沒有找到 `max_ctas`／`cta_policy` 設定（grep `CTAPolicy|max_ctas|NCCL_MAX_CTAS` 於 `megatron/megatron` 無命中）。

### DeepEP（v1.2.1）
`Buffer.num_sms=20`（`buffer.py:30`）；推薦 config 表標「# TODO: automatically tune」（`buffer.py:231`，R03）。

### PyTorch
c10d `ncclConfig_t` 暴露 `min_ctas`、`max_ctas`、`cta_policy`、`nvls_ctas`（`init.cpp:4043-4056`）；文件示例 `nccl_options.config.max_ctas = 4`（`init.cpp:4140`）。torch 2.13.0（我們裝的）的 NCCL 設定欄位是否與此 tag 一致：**未核對**。

---

## 4. 在 Nano5 的前提條件

| 前提 | 為什麼需要 | 怎麼檢查 | 登入節點檢查結果／狀態 |
|---|---|---|---|
| 驅動支援 CUDA ≥ 12.5 | NCCL 的 copy-engine collective（零 SM）要求（`ce_coll.cc:175-181`；`bufferreg.rst:214`） | `nvidia-smi` 表頭 CUDA Version；`cat /proc/driver/nvidia/version` | 登入節點：`NVIDIA-SMI 550.127.08  Driver Version: 550.127.08  CUDA Version: 12.4`；`NVRM version: NVIDIA UNIX x86_64 Kernel Module  550.127.08  Fri Oct 25 22:14:46 UTC 2024`。**登入節點結果，計算節點需在 M20 確認** |
| `NCCL_CTA_POLICY`／`MAX_CTAS` 在 2.29.7 可用 | 實驗的主要旋鈕 | `strings` 查字串；跑 `NCCL_DEBUG=INFO` 看 "Parsed environment variable NCCL_CTA_POLICY" 訊息 | 字串存在（見第 3 節）；實際行為需計算節點驗證 |
| NCCL 實際使用的 channel／CTA 數與演算法 | 要先知道「基準」各 collective 用幾個 SM | 計算節點 `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,TUNING`；`nsys` 看 kernel grid 大小（`NCCL_DEBUG=INFO` 會印 "NVLS tuning: nChannels …"，`nvls.cc:233`） | 需在計算節點檢查（給 M20） |
| H100 SM 數 | GEMM wave 與 SM 分配計算 | `nvidia-smi -q` 或 `torch.cuda.get_device_properties` | 文獻中 H100 SXM／NVL 皆為 132 SMs（Weave Table；2501.16909 §4.2「H100 NVL (132 SMs…)」）；Nano5 的實際型號與 SM 數需在計算節點確認 |
| Symmetric 注冊與 cuMem | symmetric kernel 與 CE 都需要 `ncclMemAlloc`＋`ncclCommWindowRegister` | 小測試程式；`NCCL_CUMEM_ENABLE` | 需在計算節點檢查（給 M20）；安裝版標頭有相關函式 |
| NVLS／multicast 可用 | NVLS 路徑與 NVLS 的 CTA 數 | `NCCL_DEBUG=INFO` 是否印 "NVLS multicast support is available"；`NCCL_NVLS_ENABLE` | 需在計算節點檢查（給 M20；與 A 組共用） |
| Tuner plugin 可載入 | D 的實作入口 | 用 `tuner_v5.h`（來自 v2.32.6 repo）編譯 `.so`，`NCCL_TUNER_PLUGIN=` 指向它，看 `NCCL_DEBUG=INFO` 載入訊息 | 需在計算節點檢查；安裝版沒有 tuner 標頭（`ls venvs/vllm/.../nvidia/nccl/include` 只有 `nccl.h`、`nccl_device.h`、`nccl_device_wrapper.h`、`nccl_device/`） |
| 能量測 kernel 的 SM 占用 | 量「SM 稅」需看 kernel grid／實際 co-residency | 計算節點是否能用 `nsys`／`ncu`（無 root；`ncu` 需要效能計數器權限，常需 `NVreg_RestrictProfilingToAdminUsers=0`） | 登入節點：`ls /proc/driver/nvidia/params` 本次**未查**；需在計算節點檢查（給 M20） |
| 無其他 GPU 工作同機干擾 | R13、量測乾淨度 | 獨佔整台（D048） | 以 Slurm 獨佔處理，非登入節點檢查項目 |
| 版本一致：vLLM 0.32.0 與原始碼 v0.34.0 | 引用 block 數、路徑 | 在 vLLM venv 內讀 `vllm/distributed/device_communicators/custom_all_reduce.py` 與 `csrc` 對應常數（`.so` 內 `strings`） | 未核對（可在登入節點用 `grep` 讀 venv 內 `.py`，本次沒做） |

---

## 5. 尚未有人量過，或作者自述未解的問題（只列事實）

1. **H100 上「collective 實作 × SM 數 × 對 GEMM 拖慢」的系統性量測**：Zero-SM NCCL 在 B200 上量過（F2），Lee et al. 在 H100 上量的是端到端重疊拖慢、未按 collective 實作拆解（F1）。本次檢索沒有找到在 H100 上同時比較 ring／NVLS／symmetric／copy engine 的論文。（依據：本次搜尋；Zero-SM §IV-C 的 B200 平台；R02 T1 §5。）
2. **NVLS（與 NVLS 的 CTA 數）對同時執行 GEMM 的影響**：R02 T1/T2/T5 皆稱「沒有 NVLS 開／關 A/B」；NCCL-BOT 與 NCCLbpf 都量過 NVLS，但皆為隔離的 collective 微基準（F10、F11）。
3. **含 reduction 的 SM-free 通訊**：Zero-SM NCCL「Reduction collectives … are not yet supported」（§III-A）；T-CCL 以 TMA 做 reduction，但只到 2 與 4 GPU（F14）。
4. **多機的 SM 動態分配**：mKernel「we are only able to test adaptive SM partitioning on a single node」（T2-#2）；Weave 與 OverlapAdvisor 皆把多機列為未來工作（F5、F6）。
5. **自適應策略**：Resource-aware Overlap 結論「Future work includes exploring adaptive runtime policies that automatically tune occupancy and priority settings」（F4）；Cui et al. ICPP'26（T2-#3）「Production systems could benefit from an runtime-adaptive SM partition selector, which we leave to future work」。
6. **模型涵蓋「同時有計算」＋「NVLS／SHARP」的 collective 選擇**：NCCL-BOT 明寫其評估沒有重疊與並行 communicator（F10）；OverlapAdvisor 與 Analytical Resource Management 的模型針對 MoE 重疊 pipeline，不是 NCCL 的 algo/proto 選擇；AutoCCL／Lagom 在 A40、NCCL 2.18.3，未提 NVLS。（依據：各論文如上；本次搜尋未找到同時涵蓋者，**不代表不存在**，因 arXiv 搜尋被限流，見第 7 節。）
7. **NCCL tuner 看不到實測延遲**：NCCL-BOT 引言（F9）。
8. **PAT＋NVLS 在 H100 上的效能退步與修正**：release notes 2.31.2 已知問題「PAT + NVLS has performance regressions on H100 platforms」，2.32.3 修正（R03 §2.8，WebFetch 摘要；我們裝的 2.29.7 沒有 PAT＋NVLS 階層 kernel，該功能自 2.31.2 加入）。
9. **DeepEP 的 SM 數**：`# TODO: automatically tune`（`buffer.py:231`）；Weave 指出固定 20 與每層最佳 12–36 不合（F5）。
10. **PICO**：「Today, PICO enables users to create these tuning artifacts manually」（§4.1，R02_T1-P18）。

---

## 6. 矛盾或不確定之處（不裁決）

1. **「SM-free 一定降干擾」與「SM-free 小訊息較慢」並存**：Zero-SM NCCL §V、VCCL §4.1 都自述機內小訊息延遲較高；同時 T-CCL 稱用 TMA 在受限 CTA 下仍優於 NCCL（F14）。三者平台與 collective 種類不同（B200 AllGather／Hopper P2P／H100 NVL＋GH200 含 reduction）。
2. **NVLS 預設與 Ring 在中等訊息的優劣**：NCCLbpf（B300、NCCL 2.29.7）稱 NVLS 為預設且 Ring 32 channel 在 4–128 MiB 勝 5–27%；NCCL-BOT（H100，不開 NVLS，AUTO 在多數情境「exact」）結論不同層面。兩者平台與規模不同，H100 單機是否重現：**未確認**。
3. **A100 上調出的 36 blocks 與 H100 上的最佳值**：vLLM 註解稱 36 blocks 為 A100／A10／A30／T4／V100 的格網搜尋結果；SGLang 在 sm90 用不同數字（pull 64、mc 128/ws，註解 "Tuned on H200"）；是否各自最佳：**本次無量測**。
4. **Zero-SM NCCL 圖 1A 的 Hopper 數字**：文字抽取只見「24 SMs」兩次（18%、22%）、「8」「12」「32」，無法確定 Hopper 為 24 個還是其他（`papers/2611.18075.pdf`）。
5. **ConCCL 的 DMA 範圍描述**：我在 §II 附近讀到「do not consider offloading all-reduce kernels to DMAs」一句，歸屬（描述既有工作或自身限制）**未逐字核實**，因此不將「ConCCL 不支援 all-reduce」列為事實。
6. **Demystifying NCCL 是否排除 NVLS**：此說只來自網頁搜尋摘要，我未在全文確認，不採為事實。
7. **R03 與本次的 NCCL 版本分歧**：規格書指向 `src/graph/tuning.cc`；v2.32.6 的選擇邏輯實際在 `src/tuning/`；2.29.7 的組織方式未知。
8. **CE 在 H100 上可用性**：release notes 說 CE 自 2.28 起支援（機內 NVLink）；原始碼要求驅動 ≥ 12.5；本機驅動報 12.4。NVIDIA 技術部落格的 CE 數據只在 8×Blackwell。三者是否衝突取決於計算節點驅動與是否有相容性模式：**未確認**。
9. **NCCL 2.29 的 symmetric kernel 預設**：R03 註「2.29.2 起內建 symmetric kernels 僅在 fully connected NVLink 系統啟用」；H100 HGX 符合，但實際會不會被選取依 cost model 與 buffer 是否註冊而定：**未實測**。

---

## 7. 搜尋紀錄

**工具狀態**：arXiv MCP（`search_papers`）在前半段可用，之後多次回「rate_limited（HTTP 429）」；`citation_graph` 完全被 Semantic Scholar 限流，未能取得任何引用圖；`paper-search` 的 `search_semantic` 先回有結果、後來回空（疑似同樣限流）；`search_dblp` 一次回空。因此**未能用 citation_graph 找後續工作**（規格書要求）。後半段改用 WebSearch／WebFetch 補充，其結果屬較低層級證據，已在文中標註。

**arXiv（欄位前綴）**
| 查詢 | 結果 | 取用 |
|---|---|---|
| `abs:"ConCCL"` | 1 筆 | ConCCL（2412.14335） |
| `ti:"concurrent computation and communication" OR ti:"computation-communication overlap" OR ti:"communication interference"`（sort relevance，看 12 筆，另有 11 筆未翻頁） | 12 筆 | 取 2606.09200、2609.07536、2611.11857、2609.21483（新）；2512.10236、2505.11329、2502.19811、2607.19539、2503.20313（R02 已有）；排除 2611.17991（DiLoCo 類，非同題）、2007.00156（2020 模擬平台） |
| `ti:"collective" AND ti:"performance model"` | 1 筆 | 2004.11062（MPI 分析式選擇） |
| 多個含 `abs:"NCCL" AND abs:"tuner"`／`"cost model"`／`"auto-tuning"` 的查詢 | 全數 rate_limited，**無結果** | — |
| 多個含四個以上 AND 的查詢 | 0 筆（查詢過嚴） | — |

**paper-search（Semantic Scholar／dblp）**：`NCCL collective algorithm protocol selection performance model tuning`（約 10 筆）→ 取 NCCL-BOT（2611.11303）、PML-MPI、HAT-MPI；其餘為不相關醫療／感測網論文，排除。`GPU collective communication performance model concurrent GEMM contention prediction`（8 筆）→ 取 T-CCL（2610.07098）；排除 Harmonics（多租戶叢集排程，非本題）。另三個查詢回空。`search_dblp "machine learning MPI collective algorithm selection"` 回空。

**WebSearch／WebFetch（standard）**
- `arXiv 2026 NCCL algorithm protocol selection performance model concurrent computation overlap tuner` → NCCLbpf（2603.11438）；Stanford CS244C 課程專案 "Bandit-Guided NCCL Tuning"（**排除**：課程專案、非同儕審查；其「compute overlap changes optimal pair at 6 of 9 message sizes」為搜尋摘要轉述，**未開啟原文，不採為事實**）。
- `arXiv GPU communication kernels SM usage slows GEMM H100 …` → VCCL（2510.00991）、Arnold（2509.15940）、Elvinger（2501.16909）；Chalmers 碩士論文（Green Contexts，**排除**：學位論文，未開啟）；NVIDIA 論壇與 PyTorch 論壇貼文（**排除**：軼事性使用者回報，不當作量測）。
- `learned collective algorithm selector NCCL …` → 未找到新論文；Spheron 部落格與個人部落格（**排除**）。
- `NCCL NVLS uses SMs CTAs default …` → NVIDIA 2.28 CE 部落格（`https://developer.nvidia.com/blog/fusing-communication-and-compute-with-new-device-api-and-copy-engine-collectives-in-nvidia-nccl-2-28/`，已 WebFetch：CE 數據僅 8×Blackwell AllGather，無 GEMM 重疊數字）。
- WebFetch：release notes 2.28.3、2.27.3；NVIDIA 技術部落格 *Understanding NCCL Tuning*（`https://developer.nvidia.com/blog/understanding-nccl-tuning-to-accelerate-gpu-to-gpu-communication/`：「Tuner plugins are the recommended method to spot-fix tuning」；「NCCL aims to take just enough CTAs to saturate the line rate …」；文章未寫明模型是否假設隔離）；Megatron-Bridge 0.5.0 API 文件（F12）。TransformerEngine 官方 userbuffers 文件 URL 回 404，未取得。

**本地原始碼檢查（唯讀）**：`grep`／`sed` 讀 `/work/allen0117/comm_opt_env/src/{nccl,vllm,sglang,trtllm,megatron,deepep,pytorch}` 指定檔案；`strings` 於 `venvs/vllm/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2`；`ls`／`grep` 於同目錄 `include/nccl.h`；`nvidia-smi`、`cat /proc/driver/nvidia/version`（登入節點）。

**PDF 下載**（`/home/allen0117/comm_opt/papers/`，皆已確認大小非 0）：2412.14335（1,033,202 B）、2606.09200（569,957）、2609.07536（584,540）、2611.11857（1,269,033）、2609.21483（629,434）、2507.03114（9,604,660）、2611.18075（956,158）、2502.19811（1,586,538）、2004.11062（770,650）、2611.11303（2,229,003）、2508.16809（1,095,875）、2507.04786（2,862,712）、2610.07098（951,672）、2603.11438（506,307）、2510.00991（2,288,573）、2501.16909（1,046,571）、2509.15940（1,271,578）。

**未做／未完成**：(1) `citation_graph` 追蹤 Lee et al.、Comet、ParallelKittens、AutoCCL、Lagom 的後續工作（限流）；(2) AutoCCL／Lagom／PICO／PML-MPI／HAT-MPI 的全文閱讀（沿用 R02 或摘要）；(3) TransformerEngine 原始碼與 userbuffers 內部 SM 用量；(4) vLLM 0.32.0 與 0.34.0 的差異；(5) 規格書要求 10–25 篇的「C 干擾與 SM 用量」類，AMD DMA 路線只核實 ConCCL 與 FiCCO 兩篇。

---

## 8. 引用清單

**arXiv**（Opus 之後用 arXiv MCP 匯出 BibTeX）
- 2507.03114 Lee et al.（H100 重疊特徵化）
- 2611.18075 Zero-SM NCCL
- 2412.14335 ConCCL（Optimizing ML Concurrent Computation and Communication with GPU DMA Engines）
- 2512.10236 FiCCO
- 2606.09200 Resource-aware Computation-Communication Overlap for multi-GPU ML Workloads
- 2609.21483 Weave
- 2609.07536 Analytical Resource Management for Fine-grained MoE Computation-Communication Overlap
- 2611.11857 OverlapAdvisor
- 2610.07098 T-CCL
- 2510.00991 VCCL（An Efficient, Reliable and Observable Collective Communication Library…）
- 2511.13940 ParallelKittens
- 2502.19811 Comet；2609.13585 mKernel；2607.19539（Cui et al.，ICPP'26）
- 2602.20656 Lagom；AutoCCL（NSDI'25，無 arXiv 編號，網址見第 2 節）
- 2611.11303 NCCL-BOT
- 2603.11438 NCCLbpf
- 2507.04786 Demystifying NCCL；2508.16809 PICO；2505.08936 ATLAHS
- 2004.11062 Nuriyev & Lastovetsky（MPI collective 選擇）
- 2501.16909 Understanding GPU Resource Interference One Level Deeper
- 2509.15940 Arnold（Topology-Aware Communication Alignment on More Than 9600 GPUs）

**DOI**
- 10.1109/IPDPSW63119.2024.00140（PML-MPI）
- 10.1145/3806645.3816737（HAT-MPI）
- 10.1109/HOTI66940.2025.00024（Demystifying NCCL，Semantic Scholar 列出）
- 10.23919/ISC.2026.11520486（PICO，Semantic Scholar 列出）

**網頁與文件**
- NCCL release notes 2.27.3／2.28.3：https://docs.nvidia.com/deeplearning/nccl/release-notes/index.html
- NVIDIA blog：CE collectives（2.28）、*Understanding NCCL Tuning*（網址見第 7 節）
- Megatron-Bridge 0.5.0 `bridge.training.comm_overlap` API 文件（網址見 F12）
- NCCL v2.32.6-1 repo 內文件：`docs/userguide/source/env.rst`、`api/flags.rst`、`usage/bufferreg.rst`
