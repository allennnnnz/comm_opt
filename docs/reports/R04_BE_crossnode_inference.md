# R04 / BE crossnode_inference：方向 B（跨機 TP decode 小訊息 all-reduce）與方向 E（PD 分離的 KV 傳輸）深入調查

- **執行**：Sonnet 5.5，2026-11-16　**規格**：`docs/specs/S04_direction_deep_dive.md`（分工 BE）
- **性質**：文獻＋原始碼＋官方文件調查；未送 GPU／Slurm 工作；登入節點只跑了幾秒內的唯讀指令（`ls`、`cat /proc/driver/nvidia/*`、`lsmod`）。只寫事實，不含選題判斷。
- **版本標記**：vLLM「安裝版」= 0.32.0+cu129（`/work/allen0117/comm_opt_env/venvs/vllm/lib/python3.12/site-packages/vllm/`，下稱 `[0.32.0]`）；vLLM「原始碼」= v0.34.0（`/work/allen0117/comm_opt_env/src/vllm/`，下稱 `[v0.34.0]`）；SGLang v0.5.23（`src/sglang/`）；TRT-LLM v1.2.1（`src/trtllm/`）。
- **證據標記**：`[src]` 原始碼已讀；`[doc-web]` 官方文件網頁，經 WebFetch 小模型擷取（非逐字原文，引用前請再對網頁核）；`[pdf]` 已下載 PDF 並用 pdftotext 讀全文；`[abs]` 開啟 arXiv abs 頁確認。
- **工具限制紀錄**：arXiv MCP 的 `search_papers` 與 `citation_graph` 在本次執行中被 rate limit（HTTP 429／Semantic Scholar quota 用盡），重試數次仍失敗；因此**沒有用 `citation_graph` 往後追 NVRAR／DistServe／Splitwise**，改用 paper-search 的 `search_openalex` 與 WebSearch 找，再逐篇開 arXiv abs 頁核實。這是本報告文獻覆蓋度的已知缺口（見第 B-5、E-5 節）。

---

# 第一部分：方向 B

## B-1　方向定義（照抄 Opus）

**B：跨機器 TP decode 的小訊息 all-reduce**（模型一台放不下、TP 跨兩台以上時）。一定要回答：(1) 實務上有沒有人跨機器做 TP 推論？框架官方文件或部落格的具體設定，以及「跨機器建議改用 PP／EP」的官方說法；(2) vLLM／SGLang／TRT-LLM 在 TP 跨機器時，all-reduce 實際走哪條路徑（附 `檔案:行號`）；(3) decode 時 all-reduce 的訊息大小範圍，有沒有論文報告實測分佈；(4) 低延遲跨機 all-reduce 的既有工作（NVRAR、Every-µs 的多節點部分、PCCL、其他 R02 沒有的）各自硬體；(5) GPU 直接發起 RDMA（IBGDA）與 GDRCopy 在沒有 root 時能不能用：NVSHMEM 官方文件對驅動參數與核心模組的要求。

## B-2　最接近的既有工作

「R02」欄指 `docs/reports/R02_T*.md` 已有的編號，只補新事實。

| 編號 | 標題／作者／年／出處 | 做了什麼 | 和方向 B 重疊 | 原文明確沒做或自述限制（≤2 句＋位置） | 硬體與規模 | 程式碼 | 出處網址 |
|---|---|---|---|---|---|---|---|
| B1 | **NVRAR**：Understanding and Improving Communication Performance in Multi-node LLM Inference；Singhania, Singh, Hough 等；arXiv 2511.09557 v4（2026-05-20）；CAIS '26，DOI 10.1145/3786335.3813165。見 R02_T5 #10、R02_T4 P9 | 多機 TP 的強規模分析（vLLM、SGLang、自家 YALIS）＋NVSHMEM 階層式 recursive-doubling all-reduce | 與方向 B 幾乎同題：多機 TP decode 小訊息 all-reduce | 「We plan to investigate and reduce idle time in future versions of NVRAR.」（§5.2.2）；「We leave heuristic-based hyperparameter tuning to future work.」（§5 消融）。**新事實 [pdf]**：(a) intra-node reduce-scatter 與 all-gather 用 NVSHMEM host API，原文寫「which internally calls NCCL reduce-scatter」（§4.1）；(b) 全文 grep 不到 IBGDA、GDRCopy、PeerMappingOverride，未說明所用 NVSHMEM transport；(c) TP vs HP（hybrid = TP×PP）：「TP and HP do not scale ideally. HP is advantageous in compute-bound regimes, whereas TP is better for memory-bound and decode-heavy cases.」（Observation 1，§3） | Perlmutter（A100，每節點 4 GPU，Slingshot-11）、Vista（GH200，每節點 1 GPU，IB）；強規模掃 4–128 GPU（§3）；無 H100 | https://github.com/hpcgroup/nvrar（MIT）、https://github.com/axonn-ai/yalis | https://arxiv.org/abs/2511.09557 |
| B2 | **Every Microsecond Matters**（Every-µs）；arXiv 2607.16100；NVIDIA＋ETH。見 R02_T1 P3 | barrier-free、symmetric memory、multicast、LL128 atomic 的單 NVLink 域小訊息 all-reduce | 只有「多節點」的部分與 B 相關 | **新事實 [pdf]**：實驗是「1 node with 4 GPUs (TP=4) and 2 nodes with 8 GPUs (TP=8)」，皆在 GB200 NVL72 內（NVLink 域），不經 IB。對比較對象的限制原文：「The same single-node limitation applies to vLLM's custom AllReduce.」；「since multi-node collectives are not supported at the time of writing」（指 MSCCL++ 非 multicast 版）。 | 4×GB200 NVL72 內最多 64 GPU（Fig.11） | 未取得 | https://arxiv.org/abs/2607.16100 |
| B3 | **PCCL**（The Big Send-off）；Singh 等；arXiv 2504.18658。見 R02_T4 P20 | 兩層階層式 all-reduce 等＋SVM 自動選演算法 | 跨機階層式 recursive doubling | 「In future work, we plan to benchmark PCCL on clusters with InfiniBand interconnects...」（結論，R02 已引）。無新事實 | Frontier（MI250X）、Perlmutter（A100），Slingshot；非 IB | 未說明 | https://arxiv.org/abs/2504.18658 |
| B4 | **mKernel**：Fast Multi-GPU, Multi-Node Fused Kernels；Mao 等；arXiv 2609.13585；見 R02_T2 | 多機融合 kernel（GEMM+AllReduce 等），tile 級 NVLink＋RDMA 重疊 | 多機 TP 的 GEMM+AllReduce；**IBGDA 對比**（B-5 用） | **新事實 [pdf]**：「In the scenarios we tested, GPUDirect Async provides little performance gain over host-assisted GPU-initiated communication.」（§3 附近，Fig.4）；「GPUDirect Async (IBGDA) can also be used in small-message, latency-sensitive workloads (Markthub et al., 2022; Zhao et al., 2025b).」（即未在小訊息場景測 IBGDA 優勢）。 | 兩個 16-GPU H200 叢集（2×8 H200，ConnectX-7 IB 與 AWS EFA）；GEMM+AllReduce 最高 1.72× | 未取得 | https://arxiv.org/abs/2609.13585 |
| B5 | **GPU-Initiated Communication: Dissecting Down to the Bone**；Baydamirli, Ismayilov, Oktay, Unat；arXiv 2610.01380（2026-10-01）[abs]；**R02 沒有** | 量 GPU 直發 RDMA（NVSHMEM IBGDA、NCCL GIN、DeepEP、UCCL-EP、MSCCL++、fabric-lib）與 CPU proxy 的 issue／完成延遲 | 直接回答「IBGDA 對小訊息延遲值不值得」與「需要什麼系統條件」 | **數字 [pdf]**（Table 3，P-IB：4 節點×4 H200，ConnectX-7 200 Gb/s IB，NVSHMEM 3.4.5／NCCL 2.30.7）：單次 8 B 操作 mini-gda issue 0.70 µs／put+完成 4.03 µs；NVSHMEM public API issue 5.31 µs／put+完成 11.10 µs；NVSHMEM IBRC（CPU proxy）issue 0.99 µs／put+完成 6.40 µs；調校過的 CPU proxy 0.13／4.10 µs。**系統條件（原文，§3.1）**：「the NVIDIA kernel module permits it only with the option PeerMappingOverride=1」；「NVSHMEM's CPU-assisted mode keeps GPU-built WQEs and has a host thread forward the doorbell value to the UAR for systems where the mapping is disallowed」。另有 P-H100 平台（100 節點×4 H100，CX-7 200 Gb/s）「without NIC doorbell mapping support」（§4 平台說明） | H100／H200／B200／GB200 小規模（見 Table 2）；未測 collective all-reduce | https://github.com/ParCoreLab/Dissecting-GPU-Communication-Experiments | https://arxiv.org/abs/2610.01380 |
| B6 | **Eliminating Hidden Serialization in Multi-Node Megakernel Communication**（Perseus）；Oh, Singh；arXiv 2605.00686（2026-05-01）[abs]；**R02 沒有** | 找出 proxy 式 RDMA 的 fence 序列化，提出 decoupled signaling＋NIC-side ordering；MoE megakernel | 多機、GPU 直發 vs proxy 的實測比較（H100＋CX-7＋IB，2–4 節點） | 摘要原文：「Perseus on IBRC matches or exceeds IBGDA GPU-direct by up to 1.2×, which shows that serialization, rather than the choice between proxy-based and GPU-direct transport, is what bounds multi-node megakernel performance.」；對象是 MoE，不是 all-reduce | 商用雲 8×H100／節點，ConnectX-7，2–4 節點（16–32 GPU）；另有 Perlmutter 64 A100（§5） | 未取得 | https://arxiv.org/abs/2605.00686 |
| B7 | **GPU-Initiated Networking for NCCL**（GIN）；Hamidouche 等；arXiv 2511.15076。見 R02_T4 P10 | NCCL 2.28 Device API（LSA／Multimem／GIN），GIN 有 GDAKI 與 Proxy 兩後端 | GIN Proxy 後端可在不用 IBGDA 的標準 RDMA 上跑 | 結論（R02 已引）：未來要擴到 PyTorch、TRT-LLM 等。無新事實 | DGX H100，2／4／8 節點，NVSHMEM 3.4.5、NCCL 2.28 | NCCL 開源 | https://arxiv.org/abs/2511.15076 |
| B8 | **NCCL EP**；Goldman 等（NVIDIA）；arXiv 2603.13606（2026-03-13）[abs]；R02 無（R02_T3 可能有，未核對編號） | 以 NCCL Device API 建 MoE dispatch/combine，LL 模式針對 decode（1–128 token） | 提供「NCCL Device API 在 H100 多機＋vLLM 整合」的實例 | 摘要：「competitive LL kernel performance and ... end-to-end results with vLLM integration」；非 all-reduce | H100 多機叢集（節點數見論文，未擷取） | NCCL 內 | https://arxiv.org/abs/2603.13606 |
| B9 | **Llama 3 Herd of Models**；Grattafiori 等；arXiv 2407.21783；R02 無 | Llama 3 405B 的推論並行方式 | 提供「跨機 TP 不是官方做法」的原廠自述 | 「Llama 3 405B does not fit in the GPU memory of a single machine with 8 Nvidia H100 GPUs... we parallelize model inference using BF16 precision across 16 GPUs on two machines. Within each machine, the high NVLink bandwidth enables the use of tensor parallelism. Across nodes, however, connectivity has lower bandwidth and higher latency, so we use pipeline parallelism instead.」（§6.1 Pipeline Parallelism，PDF 抽出文字） | 2 機×8 H100 | — | https://arxiv.org/abs/2407.21783 |
| B10 | **SiFAR**；arXiv 2607.08973；R02_T5 #2 | 去 barrier 的單機 all-reduce | 提供 decode 小批量 payload 大小 | **新事實 [pdf]**：「payloads... 4–32 KB in low-batch inference」；僅單機 8×H200 | 8×H200 單機 | 未公開 | https://arxiv.org/abs/2607.08973 |
| B11 | **flash-comm**；Wang, Li, Zhao；arXiv 2611.06634（2026-11-04）[abs]；R02_T1 可能有（未核對） | 自同步集體通訊（單機）；摘要稱 all-reduce 佔 baseline decode 時間 11.4% | 提供單機 decode 中 all-reduce 佔比 | 摘要原文：「all-reduce accounts for 11.4% of the baseline decoding time, which bounds the latency reduction possible from optimizing this operation alone.」 | 8×B300 單機 | 未取得 | https://arxiv.org/abs/2611.06634 |
| B12 | **Parallelization Strategies for Dense LLM Deployment**；Topcu 等；arXiv 2603.05692（2026-03-05）[abs，由 arXiv MCP get_abstract 取得]；R02 無 | Llama-3.1-70B／405B 的 TP/PP/hybrid 取捨，聚焦 intra-node | 官方之外的同儕文獻：TP 利延遲、PP 利吞吐 | 摘要：「Tensor Parallelism (TP) improves the latency objectives while Pipeline Parallelism (PP) is better-suited for throughput-oriented applications.」；摘要自述焦點為 intra-node | 單機多卡（硬體見論文，未擷取） | 未取得 | https://arxiv.org/abs/2603.05692 |
| B13 | **Characterizing Parallelism Strategies in LLM Inference**；Mirzaei, Mitra；arXiv 2610.05305（2026-10-04）[abs]；R02 無 | TP/PP/hybrid 的解析延遲模型（運算＋通訊＋bubble） | 同上，含 decode 通訊模型 | [pdf]：實驗在 8×A100 SXM4 單機（NVLink 600 GB/s）；§「TP decode」觀察到實際通訊頻寬約理論 NVLink 的 25%、「decoding is fundamentally latency-limited」（約 PDF 第 505–513 行） | 8×A100 單機 | 未取得 | https://arxiv.org/abs/2610.05305 |
| B14 | **Floor-First**（Think Before You Grid-Search）；Yihua Liu；arXiv 2607.05876（2026-07-07）[abs]；R02 無 | 以資源向量估 decode 步驟下限；案例 671B MoE 於 16×H20 | 含「TP16 decode vs EP16+DP-attention」的計算結論 | 摘要：「single-stream latency favors TP by 2.4x」；為解析估算，非實測 | 16×H20（案例） | 專案附計算器（摘要稱） | https://arxiv.org/abs/2607.05876 |
| B15 | **TokenWeave**；arXiv 2505.11329；R02_T1 P12／T5 #16 | 融合 AllReduce-RMSNorm、2–8 SM | 單機；小 decode batch 效益消失 | R02 已引，無新事實 | 8×H100 DGX | https://github.com/microsoft/tokenweave | https://arxiv.org/abs/2505.11329 |
| B16 | **fabric-lib**（pplx-garden）；arXiv 2510.27656；R02_T5 #11 | P2P RDMA 通訊庫（KV 傳輸、權重更新、MoE） | 自述 GPU-initiated RDMA（GDA）尚未普及於雲端 | R02 已引；無新事實 | ConnectX-7 400G H100；EFA H200 | https://github.com/perplexityai/pplx-garden | https://arxiv.org/abs/2510.27656 |

另有兩篇有關但非主要的壓縮式 TP 通訊論文（只改訊息大小，不改路徑）：**Flash Communication**（Li 等，arXiv 2412.04964 [abs]；摘要稱 intra-node 通訊快 >3×、TTFT 降 2×）與 **Communication Compression for TP LLM Inference**（Hansen-Palmus 等，arXiv 2411.09510 [abs]；壓縮 3.5–4.5×、TTFT 最高 2×；abs 頁未給硬體）。以及 **cross-instance MLA attention 的 IBGDA 實測**（Ma 等，arXiv 2606.01502 [abs]；「constants we measure on real IBGDA」於多節點 H100 叢集，非 all-reduce，僅證明 IBGDA 在某多節點 H100 叢集上有人跑過）。

## B-3　現成軟體的支援程度

### B-3.1　官方建議的跨機設定（B(1)）

- **vLLM**（官方文件 [doc-web]，https://docs.vllm.ai/en/latest/serving/parallelism_scaling/；本機同內容於 v0.34.0 `docs/`）：
  - 「if the model is too large for a single node, combine *tensor parallelism* with *pipeline parallelism*.」；「Set `tensor_parallel_size` to the number of GPUs per node and `pipeline_parallel_size` to the number of nodes.」
  - 兩機各 8 GPU 的範例是 `--tensor-parallel-size 8 --pipeline-parallel-size 2 --distributed-executor-backend ray`；**同一頁也給了純 TP 跨機的範例** `vllm serve ... --tensor-parallel-size 16 --distributed-executor-backend ray`（「all 16 GPUs as TP」）。
  - 並寫「Efficient tensor parallelism requires fast internode communication, preferably through high-speed network adapters.」；驗證方式：`NCCL_DEBUG=TRACE vllm serve ...`，看到 `[send] via NET/IB/GDRDMA` 表示走 IB＋GPUDirect RDMA，看到 `NET/Socket` 表示退回 TCP（「inefficient for cross-node TP」）。
  - vLLM EP 文件（https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/ [doc-web]）：多機建議 `--data-parallel-size 16 --data-parallel-size-local 8`（DP＋EP，範例 `--tensor-parallel-size 1`）、decode 用 `deepep_low_latency`；故障排除段提到「init failed for transport: IBGDA」→ 「The InfiniBand GDA kernel modules are missing」，修法是在每個 GPU 節點跑 `tools/ep_kernels/configure_system_drivers.sh` 並重開機（即需要管理員）。
- **SGLang**（v0.5.23 repo 內文件 `docs/docs/references/multi_node_deployment/multi_node.mdx:6-32`，[src]）：「Run 405B (fp16) on Two Nodes」`--tp 16 --dist-init-addr 172.16.4.52:20000 --nnodes 2 --node-rank 0/1`；「Note that LLama 405B (fp8) can also be launched on a single node」（`--tp 8`，同檔 :28-32）；同檔 :38-80 有 SLURM 版 `Llama-405B-Online-Inference-TP16-SGL`（2 節點、每節點 `gres=gpu:8`、`tp_size=16`）。官方也有反向說法：`docs/docs/advanced_features/pipeline_parallelism.mdx:8`（同檔 Why Pipeline Parallelism 段）原文「Although Tensor Parallelism (TP) remains the conventional approach for intra-node scaling, it frequently encounters communication bottlenecks during multi-node deployments. On the other hand, pipeline parallelism only requires cross-node communication at the boundaries of each pipeline stage...」。
- **TensorRT-LLM**（https://nvidia.github.io/TensorRT-LLM/latest/examples/llm_mgmn_trtllm_serve.html [doc-web]）：「IMPORTANT: Total MPI processes (nodes × ntasks-per-node) must equal tp_size.」；「For tensor_parallel_size=16, you may use 2 nodes with 8 gpus for each」；範例 `#SBATCH -N 2 --ntasks-per-node=8`，`srun -l --mpi=pmix ... trtllm-llmapi-launch trtllm-serve $LOCAL_MODEL --tp_size 16`。該頁無 `--pp_size`。（WebSearch 結果另提到第三方鏡像頁說 TP 要同節點，與官方不一致，未採用。）
- **Meta Llama 3**：見 B9（原廠自述 TP 在機內、PP 跨機）。
- **NVIDIA 官方部落格**（TP vs PP，單機 HGX H200）：「Boosting Llama 3.1 405B Throughput by Another 1.5x on NVIDIA H200 Tensor Core GPUs and NVLink Switch」https://developer.nvidia.com/blog/boosting-llama-3-1-405b-throughput-by-another-1-5x-on-nvidia-h200-tensor-core-gpus-and-nvlink-switch ；本次只從 WebSearch 摘要看到「TP 最低延遲 5.6× 優於 PP、PP 最大吞吐 1.5×」，**未開啟原文核對**，標「待核」。非跨機資料。

### B-3.2　TP 跨機時 all-reduce 實際路徑（B(2)）

**vLLM**（TP=16，2 機×8 H100 為例）：
- custom all-reduce：`[v0.34.0] vllm/vllm/distributed/device_communicators/custom_all_reduce.py:202-203`：`same_node = all(in_the_same_node_as(group, source_rank=0)); self.mnnvl_only = not same_node`；`:232-237`：`if not same_node and not _group_can_attempt_mnnvl(group, device): logger.warning("Custom collectives are disabled because this multi-node group does not have MNNVL-capable GPUs on every rank."); return`（MNNVL 判斷函式 `_group_can_attempt_mnnvl` 在 `:65`，註解「MNNVL is available only on Blackwell-class GPUs」）；`:501`：`if self.disabled or self.mnnvl_only or self.world_size > 8: return False`。**世界大小 16 本身也超過 8**（`:501`）。
- `[0.32.0]` 同檔 `:204-205` 有 `mnnvl_only`，`:234` 有多機停用，`:500-501` 為 `if self.disabled or self.world_size > 8: return False`（0.32.0 的 `should_custom_ar` 少了 `mnnvl_only` 條件，其他相同；多機 H100 仍被 `:234` 停用）。
- torch symm_mem：`[v0.34.0] all_reduce_utils.py:58-64`：`SYMM_MEM_ALL_REDUCE_MAX_SIZES["9.0"]` 只有世界大小 2、4、6、8；`symm_mem.py` 初始化時「World size %d not supported, communicator is not available」（`symm_mem.py:75`）；若 multicast ptr 為 0 則「multicast operations are not supported」停用（`:109`）。故 TP=16 的 H100 不走 symm_mem。
- FlashInfer AR：`flashinfer_all_reduce.py:157-190`：`node_count == 1` 才選 `mnnvl`（單機 NVSwitch）；多機且 `has_device_capability(100)`（Blackwell）才選 `mnnvl`；否則 `logger.debug_once("FlashInfer allreduce fusion is disabled: multi-node process groups require an MNNVL-capable Blackwell system.")` 並回傳 None；`trtllm` backend 在 `node_count > 1` 直接報不支援（`:225-229`、`:294-298`）。`[0.32.0]` 同檔 `:157-172` 有同樣邏輯。
- NCCL symmetric memory：`all_reduce_utils.py:111-156`：`min_world_size=4`、`always_use_above_world_size: 8`（註解「Always use symm mem for world_size > 8」），但整體由 `VLLM_USE_NCCL_SYMM_MEM`（預設 False，`envs.py:316`）與 `is_symmetric_memory_enabled()` 控制；預設關閉。
- 結果：H100 TP=16 預設走 `cuda_communicator.py:459-463` 的 `pynccl_comm.all_reduce`（NCCL）。**這是靜態閱讀推論，未在機器上開 log 驗證**（待 M20／M23，指令見 B-4）。

**SGLang**（v0.5.23）：
- `python/sglang/srt/distributed/device_communicators/custom_all_reduce_v2.py:579-597`：`can_use_custom_all_reduce_v2`：世界大小須在 `get_supported_world_sizes()`（sm90 為 2–8，`configs/custom_all_reduce_v2.py:264-310`）；跨機時 `if not all(in_the_same_node_as(group, source_rank=0)): return is_one_nvlink_clique(group, device) and _is_vmm_backed_allocator(device)`。`is_one_nvlink_clique` 在 `custom_all_reduce_utils.py:396`。H100 的 NVSwitch 域限單機，故 16 卡跨機為 False → 回 NCCL。
- `parallel_state.py:717-766`：FlashInfer AR 融合 `_can_use_flashinfer_allreduce` 的自動啟用條件含「單機（sm90）」（R03 `overrides.py:980-1001`）。
- 世界大小 16 時的舊版 `CustomAllreduce.should_custom_ar`：`world_size == 2 or full_nvlink`（`custom_all_reduce.py:260-283`，R03）。

**TensorRT-LLM**（v1.2.1）：
- `cpp/tensorrt_llm/thop/allreduceOp.cpp:1113-1148`：`setGroupTopology()`，`is_inter_node = (mGroup.size() != local_group.size())`；跨機時 `mIsMNNVLSupported = mIsNVLINKSupported && all_ranks_support_mnnvl; mIsP2PSupported = false; // P2P doesn't work across nodes`。
- `allreduceOp.cpp:1248-1257`（`ifFallbackToNCCL`）：`message_size_bytes > max_workspace_size || !mIsP2PSupported || !mIsNVLINKSupported` → `NCCL`。故跨機 TP（P2P=false）一律回 NCCL（AUTO 與 ONESHOT／TWOSHOT 都先走這個檢查，見 `selectImplementation` 在 `:1199-1243`）。
- Python 端 MNNVL 策略：`_torch/distributed/ops.py:535-550`：`is_testing or (dtype 支援 and not mapping.has_cp() and mapping.is_multi_node() and MnnvlMemory.supports_mnnvl() and is_on_aarch64)`；x86 H100 不啟用。

**NCCL 本身**：R03 §1.1 NCCL 列（NVLSTree、CollNet 預設關等）；演算法與協定的選擇屬方向 A、D，本節不重複。

### B-3.3　decode all-reduce 訊息大小（B(3)）

- 公式：每層 2 次 all-reduce（attention 後、MLP 後），訊息 = batch（decode 每請求 1 token）× hidden × bytes。NVRAR 原文：「message size [is] B × H, where B is the batch size and H the hidden dimension. For the 70B model with B=8 and H=8192, this message size is 128 KB. Across our workloads, message sizes range from 128 KB to 1 MB.」（§4 開頭，[pdf]）；另「NVRAR primarily benefits small messages (128 KB–4 MB)」，且 C=256 併發下「pushes the decode message sizes towards the upper bound of this range」（§5.2.3）。
- hidden 大小（Llama 3 Table 3，[pdf] 2407.21783）：8B 4096、70B 8192、405B 16384。bf16（2 B）：每 token 8 KB／16 KB／32 KB；batch 8 時 64 KB／128 KB／256 KB（**此為算術換算，非文獻直接報告**；NVRAR 405B 的「256 KB 與 1024 KB」對應其實驗的 batch，見 §5.2.1）。
- 低批量：SiFAR「4-32 KB payloads in low-batch inference」（[pdf]，單機）。
- **實測分佈（訊息大小隨時間或隨 trace 的直方圖）**：沒有找到（見 B-5）。

## B-4　在 Nano5 的前置條件（B(5)）

登入節點觀察（2026-11-16，主機 `cbi-lgn01`，只讀指令，**計算節點狀態未知**）：驅動 `550.127.08`（`/proc/driver/nvidia/version`）；核心 `4.18.0-553.144.1.el8_10.x86_64`；`/proc/driver/nvidia/params`：`RegistryDwords: ""`、`EnableStreamMemOPs: 0`、`DmaRemapPeerMmio: 1`；`/dev/gdrdrv` 存在（`crw-rw-rw-`，root:root，Nov 12），`lsmod` 有 `gdrdrv`、**沒看到 `nvidia_peermem`**；`ofed_info -s` = `MLNX_OFED_LINUX-23.10-2.1.3.1.201`；`/sys/class/infiniband/` 有 `mlx5_0`–`mlx5_3`。venv 內已裝 `nvidia-nvshmem-cu12 3.4.5`（`.../site-packages/nvidia_nvshmem_cu12-3.4.5.dist-info`，lib 含 `nvshmem_transport_ibgda.so.3`、`ibrc`、`ucx`、`libfabric`、`ibdevx`）；NVHPC 24.7 內另有 NVSHMEM（`comm_libs/nvshmem/lib/libnvshmem_host.so.3.0.6`）；venv **沒有** `nixl`、`mooncake`、`lmcache`、`ucx` 套件。

| 條件 | 文件出處 | 檢查方法（指令＋在哪跑） | 目前狀態 | 需要管理員 |
|---|---|---|---|---|
| IBGDA（預設模式）需 `nvidia.ko` 以 `PeerMappingOverride=1` 載入，且 nvidia.ko ≥ 510.40.3、MLNX_OFED ≥ 5.0 | NVSHMEM Installation Guide「Software Requirements → IBGDA」，https://docs.nvidia.com/nvshmem/release-notes-install-guide/install-guide/abstract.html [doc-web]：「In the default case, `nvidia.ko` must be loaded with `PeerMappingOverride=1`」，設定在 `/etc/modprobe.d/nvidia.conf`：`options nvidia NVreg_RegistryDwords="PeerMappingOverride=1;"`。Dissecting（B5）§3.1 同樣說明（原文見 B-2 B5 列） | 計算節點上：`srun ... cat /proc/driver/nvidia/params \| grep -i RegistryDwords`（待 M20／M23） | 登入節點：`RegistryDwords: ""`（**未設**）；計算節點未知 | 是（改驅動參數需 root 並重載模組） |
| IBGDA「CPU-assisted」模式：「`PeerMappingOverride` is not required」；但需 GDRCopy | 同上：「GDRCopy ... required if CPU-assisted IBGDA is enabled on the system」 | 計算節點：`ls -l /dev/gdrdrv`；`lsmod \| grep gdrdrv` | 登入節點有 `gdrdrv`；計算節點未知 | 否（使用者只需 `/dev/gdrdrv` 可讀寫；載入 `gdrdrv` 模組需 root，已由站方載入於登入節點） |
| GDRCopy 一般需求：核心模組需 root 載入 | GDRCopy README，https://github.com/NVIDIA/gdrcopy [doc-web]：「Root privileges are necessary to load/install the kernel-mode device driver.」；README 擷取未說明使用者空間函式庫是否不需 root（擷取結果寫「does not state」）；另述 DMA-BUF mmap 後端「does not require the GDRCopy kernel module」但需「CUDA driver 13.3 or newer」（我們驅動 550 → CUDA 12.4 世代，不符） | 同上 | 同上 | 載入模組：是；使用已載入的模組：否（`/dev/gdrdrv` 登入節點為 0666） |
| NVSHMEM IBRC／UCX transport 需 GDRCopy 與 nvidia-peermem（atomics 與 GPUDirect RDMA） | 同 install guide：「(Optional) GDRCopy v2.0 or later... required for atomics support on non-NVLink connections」；「Both the IBRC and UCX transports make use of GDRCopy in order to perform atomic operations.」；nvidia-peermem「required when `NVSHMEM_IBRC_SUPPORT=0` and `NVSHMEM_UCX_SUPPORT=0` are not set at compile time」 | 計算節點：`lsmod \| grep nvidia_peermem` | 登入節點**未見** `nvidia_peermem`；計算節點未知 | 是（`modprobe nvidia-peermem` 需 root：CUDA GPUDirect RDMA 文件 https://docs.nvidia.com/cuda/gpudirect-rdma/ [doc-web]「can be loaded ... with root privileges」「there is no service to automatically load nvidia-peermem」） |
| IBGDA 若用 dma-buf 取代 peermem：需 rdma-core ≥ 44.0 與上游核心 ≥ 6.1 | install guide IBGDA 節：「For dma-buf support, it requires `rdma-core` >= 44.0 and an upstream kernel >= 6.1.」；缺 dma-buf 時「requires `nvidia-peermem` >= 510.40.3」 | `uname -r`；`rpm -q rdma-core`（計算節點） | 登入節點核心 4.18 < 6.1（RHEL 8 核心；backport 狀況未知） | 是（核心／套件） |
| NVSHMEM 的環境變數 | NVSHMEM 2.8.0 環境變數文件，https://docs.nvidia.com/nvshmem/archives/nvshmem-280/api/gen/env.html [doc-web]：`NVSHMEM_IB_ENABLE_IBGDA` 預設 false（「Set to enable GPU-initiated communication transport」）；`NVSHMEM_REMOTE_TRANSPORT` 預設 `ibrc`；`NVSHMEM_DISABLE_GDRCOPY` 預設 false。**這是 2.8.0 版文件，3.4.5 的預設值未核對**（最新版網址 `docs/api/gen/env.html` 本次擷取得空內容） | 使用者可自行設，不需 root | 已知（文件） | 否 |
| NVLink SHARP（驅動） | install guide：「Requires nvidia.ko version 530.30.02 or later.」 | `/proc/driver/nvidia/version` | 550.127.08 滿足 | 否 |
| 跨機 NCCL 用 GPUDirect RDMA | vLLM 官方文件（B-3.1）：以 `NCCL_DEBUG=TRACE` 看 `NET/IB/GDRDMA`。NCCL GDRDMA 依賴 peermem 或 dma-buf | 計算節點：`NCCL_DEBUG=INFO` 小工作（M20／M23）看 `via NET/IB/... GDRDMA` | 未知 | 判讀不需；若需 peermem 則需 |
| 登入節點與計算節點相同假設 | — | — | **不能假設相同**；上述登入節點數據只是線索 | — |

備註：mKernel（B4）與 Perseus（B6）都**沒有用 root 做特殊設定的描述**（grep 全文無 root／PeerMappingOverride／modprobe），但兩者的平台是商用雲或 Perlmutter，管理員端設定不明；因此「無 root 能否用 IBGDA」目前只有 NVSHMEM 文件與 Dissecting 的原文可據（見 B-5 與第 6 節）。

## B-5　「找不到有人量過」的清單

- 在 **H100＋InfiniBand（NDR 400G）、2–4 機**上量多機 TP **decode all-reduce（NCCL 對 NVSHMEM／自製）延遲**：NVRAR 只有 A100（Slingshot）與 GH200（IB）；PCCL 只有 Slingshot；Every-µs 在 GB200 NVLink 域內；SiFAR 只單機 H200。**未找到**（查詢見第 8 節 Q1–Q6、Q9）。
- 在 **H100／H200＋IB** 上量 **IBGDA vs IBRC 對「多機 all-reduce」（非 MoE）延遲**：Dissecting（B5）量了單次 put 延遲，mKernel（B4）量了 GEMM+AllReduce 吞吐（IBGDA 與 host proxy 差異「little」，且它自述未量小訊息），Perseus（B6）量 MoE。沒有找到專門量 all-reduce 的（Q7、Q8）。
- **decode all-reduce 訊息大小的實測分佈**（直方圖或按 trace 統計）：僅有 NVRAR 的範圍陳述（128 KB–1 MB）與 SiFAR 的 4–32 KB 陳述，**沒有找到**分佈圖或統計（Q9、Q10）。
- **官方文件明說「跨機 TP 不建議、請改 PP／EP」的 vLLM／TRT-LLM 官方原文**：vLLM 文件只說「combine TP with PP」與「TP=GPUs per node」，並未寫「不建議跨機 TP」，且同頁仍給了 TP=16 的範例；TRT-LLM 官方 Slurm 範例直接給 `--tp_size 16`。SGLang `pipeline_parallelism.mdx` 有「TP frequently encounters communication bottlenecks during multi-node deployments」。「vLLM／TRT-LLM 明文建議避免跨機 TP」**未找到**（Q11–Q13）。
- **在無 root、IBGDA 預設模式**下實際跑成功的第一手報告：**未找到**（Q14）。
- 以 `citation_graph` 追 NVRAR 後續引用：**因 Semantic Scholar rate limit 未能執行**。

## B-6　互相矛盾或不確定之處

1. **vLLM 官方文件自己**：同一頁既說「Set `tensor_parallel_size` to the number of GPUs per node and `pipeline_parallel_size` to the number of nodes」，又給出 `--tensor-parallel-size 16` 跨 2 機的範例。
2. **NVRAR 自身數字**：摘要「1.9×–3.6×、128 KB–2 MB」與結論段「1.06×–1.92×（Slingshot）、1.14×–3.57×（InfiniBand）、256 KB–2 MB」不同（R02 已記）；另 §5.2.1 自述 128 KB 訊息的微基準 NVRAR **慢於** NCCL（歸因於微基準無計算穿插）。
3. **IBGDA 是否值得**：NVSHMEM 文件把 IBGDA 描述為 GPU 直發；Dissecting 在 H200＋CX-7 量到 NVSHMEM public IBGDA put+完成 11.10 µs，高於 NVSHMEM IBRC proxy 的 6.40 µs（單次 8 B，Table 3）；mKernel 與 Perseus 也都報告 IBGDA 不比 proxy 好。條件不同（200 Gb/s、單 QP 單次操作、MoE／融合 kernel），不裁決。
4. **vLLM 0.32.0 與 v0.34.0 的差異**：`should_custom_ar` 在 0.34.0 多了 `mnnvl_only` 條件（行號見 B-3.2）；其他多機判斷相同。我們實際跑 0.32.0。
5. **NVSHMEM 環境變數預設值**：只讀到 2.8.0 文件；我們的 venv 是 3.4.5，NVHPC 內是 3.0.6。IBGDA 預設模式與 CPU-assisted 模式在 3.x 的環境變數名稱，文件擷取結果沒有列出 CPU-assisted 的變數（2.8.0 頁面「does not describe any CPU-assisted IBGDA setting」），Dissecting 腳註 1 提到 `NVSHMEM_IBGDA_SUPPORT_GPUMEM_ONLY=ON`，**用途未核對**。
6. **NVIDIA TP vs PP 部落格數字**只見搜尋摘要，未開原文。

---

# 第二部分：方向 E

## E-1　方向定義（照抄 Opus）

**E：prefill／decode 分離時的 KV cache 傳輸**在 H100＋IB 上是不是瓶頸。一定要回答：(6) vLLM 0.32／SGLang 支援哪些 KV 傳輸連接器（NIXL、Mooncake、LMCache 等），需要什麼前置條件（UCX、GPUDirect RDMA、nvidia_peermem 或 dmabuf）；(7) 在 H100＋IB 上實測 KV 傳輸佔比的論文或廠商數據，條件為何；R02_T5 §3-F 的矛盾有沒有新資料。

## E-2　最接近的既有工作

R02_T5 已列：DistServe（#23）、Splitwise（#24）、Mooncake（#22）、KVDirect（#19）、Beyond the Buzz（#15）、PDD（#1）、Lynx（#3）、TENT（#7）、PrfaaS（#6）、LMCache（#13）、fabric-lib（#11）。這裡只補新事實與 R02 沒有的。

| 編號 | 標題／作者／年／出處 | 做了什麼 | 和方向 E 重疊 | 原文明確沒做或自述限制／重要條件 | 硬體與規模 | 程式碼 | 出處網址 |
|---|---|---|---|---|---|---|---|
| E1 | **Revisiting Disaggregated LLM Serving for Performance and Energy Implications**；Li, Zhu, Lee, Nahrstedt；arXiv 2601.08833（頁面 v1 日期 2025-11-14）[abs]；OpenAlex 顯示 DOI 10.1145/3805621.3807662；R02 無 | 在不同 KV 傳輸路徑（含 NIXL）下比較分離式與並置式的效能與能耗 | 直接比較「KV 傳輸媒介」對分離式效益的影響 | 摘要原文（OpenAlex 全文）：「performance benefits from prefill-decode disaggregation are not guaranteed and depend on the request load and KV transfer mediums.」；**硬體 [pdf]**：單一節點 2×A100 40GB，同一 PCIe Gen3 bridge，GPU P2P 傳 KV；不是 H100、不是 IB | 2×A100（PCIe）單節點 | 未取得 | https://arxiv.org/abs/2601.08833 |
| E2 | **TraCT**：Disaggregated LLM Serving with CXL Shared Memory KV Cache at Rack-Scale；Yoon 等；arXiv 2512.18194（2025-12-20）[abs] | 以 CXL 共享記憶體取代 RDMA 作 KV 傳輸基底；建在 Dynamo | 作者把 RDMA KV 傳輸視為瓶頸，並與 NIXL/UCX 比 | 摘要原文：「KV transfer dominates both time-to-first-token (TTFT) and peak throughput, and remains highly sensitive to network contention」（OpenAlex 摘要；arXiv 摘要同義）；**條件 [pdf]**：兩台伺服器各 1×NVIDIA A6000（48 GB）、**100 Gbps** Mellanox MT2892 NIC，baseline 為 NIXL/UCX；非 H100 | 2 節點×1 A6000，100 Gb/s | 未取得 | https://arxiv.org/abs/2512.18194 |
| E3 | **FlowKV**；Li 等；arXiv 2504.03775（2025-04-03）[abs] | 降低 KV 傳輸呼叫次數（連續記憶體配置），並做負載感知排程 | 量到分離式的 KV 傳輸延遲與其佔總延遲比 | 摘要：「reduces the average transmission latency of KV cache by 96%, from 0.944s to 0.053s, almost eliminating the transfer time relative to the total request latency」；**硬體 [pdf]**：A100-SXM4-80GB×8（NVLink）、L20／H20；非 H100＋IB | 8×A100（NVLink），L20／H20 異質 | 未取得 | https://arxiv.org/abs/2504.03775 |
| E4 | **From Tensor Buffer to Distributed Memory Hierarchy: A Survey of KV Cache Management for LLM Serving**；Li, Wang, Chen；arXiv 2607.02574（2026-06-30）[abs]；R02 無 | 分類 30 多個 KV 管理系統並稽核評估缺口 | 稽核「KV 移動」的量測是否被報告 | 原文（§5.5，[pdf]）：「Systems that move KV rarely report the event distribution underneath their end-to-end numbers: bytes moved per token, remote blocks touched per token, operation sizes, or critical-path stalls (MG1).」；「Transfer or access latency sometimes appears for a single microbenchmark, but not in a form that can explain P99 TTFT or TPOT.」 | 調查，無實驗 | — | https://arxiv.org/abs/2607.02574 |
| E5 | **ARK**：Avoiding Routing Collisions for KV Cache Transfer in Disaggregated LLM Inference；Lin, Hsu, Ho, Saeed；DOI 10.1145/3789240.3828750（OpenAlex，2026-08-11）；**僅 OpenAlex 摘要，未開 DOI 頁核實** | 以來源埠選擇避免 ECMP 碰撞的 KV 傳輸流 | KV 傳輸視為「mega elephant flows」 | OpenAlex 摘要：「Packet-level RDMA simulations」→ 模擬結果，非實機 | 模擬 | 未取得 | https://doi.org/10.1145/3789240.3828750（**未核實**） |
| E6 | **An Optical KV Cache Network for Disaggregated LLM Serving**；Yoon, Yu；arXiv 2610.18444（2026-10-13）[abs] | 光交換器傳 KV，隱藏重組延遲 | 以「H100 實測 prefill 時間」搭配模擬 | 摘要：「In simulations of GPU clusters on real serving traces with prefill times measured on H100 GPUs」→ 傳輸為模擬 | 模擬（prefill 時間來自 H100 實測） | 未取得 | https://arxiv.org/abs/2610.18444 |
| E7 | **KVDirect**（arXiv 2501.14743，R02_T5 #19）、**Beyond the Buzz**（2506.05508，#15）、**DistServe**（2401.09670，#23）、**Splitwise**（2311.18677，#24） | R02 已列 | R02 §3-F 矛盾的原始來源 | R02 已引；本節無新事實 | 見 R02_T5 | 見 R02_T5 | 見 R02_T5 |

**廠商／專案資料（非同儕審查）**：
- **vLLM 官方部落格「Taking vLLM Apart: A Practical Guide to Disaggregated Serving」**（2026-09-29，IBM Research 作者），https://vllm.ai/blog/2026-09-29-disaggregated-serving-guide [doc-web]：測試機為 **2×L40S（PCIe、無 NVLink、無 P2P）**、Qwen2.5-7B、約 8k token prompt；「~470 MB of KV cache (56 KiB per token for Qwen2.5-7B) took about 1.3 s to reach decode」；0.2 req/s 時中位 TTFT「2.2 s against 0.7 s collocated」；文中估算 10k token Llama-3.1-70B（約 3 GB KV）在 400 Gb/s line rate「roughly 65 ms ... before any overhead」（**假設值，非實測**）；文章結論「Every first token pays for the transfer」。需要 vLLM ≥ v0.31.0、`UCX_NET_DEVICES=all`、每個 worker 獨立 `VLLM_NIXL_SIDE_CHANNEL_PORT`；未提 GPUDirect RDMA。
- **LMCache PD 基準部落格**（2025-04-29），https://blog.lmcache.ai/2025-04-29-pdbench/ [doc-web]：8×H100 單機（NVLink）、Llama-3.1-8B；7500 token KV 經 NVLink 傳輸「20ms, if page size is 16 tokens」、「8ms, if page size is 128 tokens」；結論「vLLM v1 (w/ PD) outperforms vLLM (w/o PD) and achieves similar performance as Dynamo across TTFT and ITL」；NIC 型號未指定；非 IB。
- **vLLM 官方文件（v0.34.0 repo `docs/features/disagg_prefill.md`）**：「Disaggregated prefill DOES NOT improve throughput.」（[src]，該檔「Why disaggregated prefilling?」節末尾 note）。
- **vLLM 官方部落格「Bi-Directional KV Transfer」**（2026-10-12）https://vllm.ai/blog/2026-10-12-bidirectional-kvxfer-multiturn-agentic-workload ：只從 WebSearch 摘要得知「AWS EFA、P5en、1P1D、TTFT 最高降 1.6×」，**未開啟原文**，且為 EFA 非 IB。
- **llm-d RDMA 文件** https://llm-d.ai/docs/resources/rdma/rdma-configuration ：WebSearch 摘要稱「KV cache transfer time lands directly on TTFT」，**未開啟原文**。
- **DRANET NIXL KV transfer 範例** https://dranet.sigs.k8s.io/docs/user/nixl-kv-transfer/ ：WebSearch 摘要：2 節點×8 H100 80GB、8×Mellanox ConnectX VF NIC，1 GiB NIXL WRITE，同 NUMA 對齊 39.07 GB/s／27.49 ms，未對齊 27.54 GB/s／38.99 ms（4-GPU 集合）；**未開原文**、非 LLM 端到端佔比。

## E-3　現成軟體的支援程度（E(6)）

**vLLM**
- v0.34.0 `vllm/vllm/distributed/kv_transfer/kv_connector/factory.py:154-276` 註冊的 connector：`ExampleConnector`、`ExampleHiddenStatesConnector`、`LMCacheConnectorV1`、`LMCacheMPConnector`、`NixlConnector`（含 `NixlPullConnector`／`NixlPushConnector`，`:125-130` 依 `transfer_mode`=pull|push 轉換）、`MultiConnector`、`HiSparseConnector`、`MoRIIOConnector`（ROCm）、`OffloadingConnector`、`DecodeBenchConnector`、`MooncakeConnector`、`MooncakeStoreConnector`、`UMBPStoreConnector`、`FlexKVConnectorV1`、`SimpleCPUOffloadConnector`、`HF3FSKVConnector`。
- **安裝的 0.32.0** 同檔（`.../site-packages/vllm/distributed/kv_transfer/kv_connector/factory.py:154-238`）有：Example、ExampleHiddenStates、LMCacheConnectorV1、LMCacheMPConnector、NixlConnector、NixlPullConnector、NixlPushConnector、MultiConnector、HiSparseConnector、MoRIIOConnector、OffloadingConnector、DecodeBenchConnector、MooncakeConnector、MooncakeStoreConnector、FlexKVConnectorV1。**差異（相對 v0.34.0）**：0.32.0 沒有 `UMBPStoreConnector`、`SimpleCPUOffloadConnector`、`HF3FSKVConnector`，且 `NixlConnector` 名稱在 0.32.0 的處理方式少了 `transfer_mode` 轉換。
- 預設：**沒有預設連接器**；需 `--kv-transfer-config`（JSON）指定，例 `{"kv_connector":"NixlConnector","kv_role":"kv_producer"}`（`docs/features/nixl_connector_usage.md:102-122`）；`docs/features/disagg_prefill.md` 稱「experimental」。
- NIXL：`requirements/kv_connectors.txt:5`：`nixl == 1.5.2`；文件（`nixl_connector_usage.md:11`）：`uv pip install nixl`；「UCX is the primary default transport library used by NIXL」，需設定 `UCX_TLS`、`UCX_NET_DEVICES`（`:30-31`）；「Transfers across hosts use RDMA」（`:91`）；GB 系列需 `UCX_CUDA_IPC_ENABLE_MNNVL`（`:428`）；`kv_connector_extra_config.backends` 可選 LIBFABRIC 等（`:41-60`）。
- Mooncake：`docs/features/mooncake_connector_usage.md`：`uv pip install mooncake-transfer-engine-cuda13`（vLLM 預設 CUDA 13；CUDA 12 環境裝 `mooncake-transfer-engine`）；環境變數 `WITH_NVIDIA_PEERMEM`：「Default: 1, which uses `ibv_reg_mr()` and requires the `nvidia-peermem` kernel module to be loaded」「Set to 0 to use the DMA-BUF path, which does not need that module」（`:60-63`）。**我們的 vLLM 0.32.0+cu129 是 CUDA 12.9 版**。
- LMCache：`LMCacheConnectorV1` 經 NIXL 傳 KV（`disagg_prefill.md` 對應條目）；LMCache 套件非 vLLM 內建。

**SGLang v0.5.23**
- `python/sglang/srt/arg_groups/choices.py:122-129`：`DISAGG_TRANSFER_BACKEND_CHOICES = ["mooncake","nixl","ascend","fake","mori","mooncake_tcp"]`；`fields/disagg.py:65-70`：`--disaggregation-transfer-backend` 預設 `mooncake`；`:81-83` `--disaggregation-ib-device`（「Supported by Mooncake and NIXL UCX」）；`disaggregation/utils.py:681-686` 列舉 `TransferBackend`。
- 文件（`docs/docs/advanced_features/pd_disaggregation.mdx`）：「Currently, we support Mooncake and NIXL as the transfer engine.」（:21）；Mooncake 安裝 `uv pip install mooncake-transfer-engine`（:231-235）；NIXL `pip install nixl` 或自建 UCX（:590-606）；`SGLANG_MOONCAKE_CUSTOM_MEM_POOL` 的 `INTRA_NODE_NVLINK` 建議用於「A100, H20, H100, etc.」（:425-431）；`mooncake_tcp` 為 TCP 後備。**該文件擷取內容未見 nvidia_peermem／dmabuf 的明文需求**（`grep -i "gdr\|peermem\|dmabuf" pd_disaggregation.mdx` 無結果）。

**TensorRT-LLM**：本次**未調查**其 KV 傳輸（只讀了 R03 範圍）。R03 沒有 TRT-LLM PD 項目；NVIDIA Dynamo／TRT-LLM `kv-cache-transfer` 文件（https://docs.nvidia.com/dynamo/v1.0.0/additional-resources/tensor-rt-llm-details/kv-cache-transfer.md，僅 WebSearch 結果列出）**未開啟**。

**NIXL 本身**（https://github.com/ai-dynamo/nixl [doc-web]）：「NIXL was tested with UCX version 1.23.x.」；「GDRCopy ... is necessary for maximum performance, but UCX and NIXL will work without it.」；README 擷取**沒有**提 GPUDirect RDMA／nvidia_peermem／dmabuf／root 需求。HPC-X 的 UCX 在本機為 1.16.0（`ucx_info -v`，`/lib64/libucs.so.0` 與 API headers 1.17.0），**低於 NIXL 測試版 1.23.x**（NIXL pip 版是否自帶 UCX 未核對）。

## E-4　在 Nano5 的前置條件

| 條件 | 文件出處 | 檢查方法（指令＋在哪跑） | 目前狀態 | 需要管理員 |
|---|---|---|---|---|
| NIXL 套件（vLLM 0.32.0 用 `nixl == 1.5.2`） | `vllm/requirements/kv_connectors.txt:5` [v0.34.0]；0.32.0 版所需 nixl 版本**未確認** | `uv pip install nixl`（登入節點可裝，不跑運算）；`python -c "import nixl"` | venv 目前無 `nixl`（2026-11-16 `ls site-packages`） | 否 |
| Mooncake 套件 | vLLM／SGLang 文件（見 E-3） | 同上 | venv 目前無 `mooncake` | 否 |
| GPUDirect RDMA 路徑：Mooncake 預設需 `nvidia-peermem`；可設 `WITH_NVIDIA_PEERMEM=0` 走 DMA-BUF | `mooncake_connector_usage.md:60-63` | 計算節點：`lsmod \| grep nvidia_peermem`；跑小 KV 傳輸時看有沒有 `Failed to register memory ... Bad address [14]` | 登入節點未見 `nvidia_peermem`；DMA-BUF 路徑所需核心／驅動版本，Mooncake 文件**未說明**；NVSHMEM IBGDA 的 dma-buf 條件是核心 ≥ 6.1（見 B-4），Mooncake 是否同條件**未核對** | peermem：是；DMA-BUF：視核心 |
| UCX 傳輸可用 RDMA（rc／ud）與 GPU 記憶體 | vLLM NIXL 文件：`UCX_TLS`、`UCX_NET_DEVICES`；驗證用 `UCX_PROTO_INFO=y`（`nixl_connector_usage.md:91` 對 cuda_ipc）；另 vLLM 官方 IB 指引：`NCCL_DEBUG=TRACE` 看 GDRDMA（B-3.1） | 計算節點，兩機各 1 GPU 的小工作：`UCX_PROTO_INFO=y vllm serve ...`／`ucx_info -d`（HPC-X 內有 `ucx_info`） | 未知 | 否（若 peermem 缺，GPUDirect RDMA 不通，UCX 可能退回主機中轉，需實測） |
| GDRCopy（NIXL／UCX 非必須，但「necessary for maximum performance」） | NIXL README [doc-web] | 計算節點：`ls -l /dev/gdrdrv` | 登入節點有（0666） | 否（已存在時） |
| 跨機網路拓樸：需兩台以上同時獨佔（D048） | `CLAUDE.md` 規則 7 | `sbatch -N 2` | 已知可行（M 系列已用） | 否 |
| vLLM 版本差異：0.32.0 的 connector 清單與 v0.34.0 不同 | 見 E-3 | `python -c "import vllm; print(vllm.__version__)"` | 已確認 0.32.0 | 否 |
| NIXL 與 UCX 版本相容 | NIXL README：測試版 UCX 1.23.x | `pip show nixl`、`ucx_info -v` | HPC-X UCX 1.16.0（本機） | 否（若 pip 版自帶 UCX 則無此問題，**未核對**） |

## E-5　「找不到有人量過」的清單

- 在 **H100＋InfiniBand NDR（400 Gb/s）**上量 **PD 分離的 KV 傳輸佔 TTFT 或 e2e 比例**（含 NIXL／Mooncake 實際傳輸）：**未找到**。已找到的條件都不同：KVDirect 為 2 張 H100 的微基準（R02）；Splitwise 為 IB（DGX-H100，R02，傳輸 ≤3% e2e）；TraCT 為 A6000＋100 Gb/s；vLLM 官方部落格為 L40S PCIe；LMCache 為 H100 單機 NVLink；E1 為 2×A100 PCIe；FlowKV 為 A100 NVLink／L20／H20（查詢見第 8 節 Q15–Q20）。
- 在 **H100＋IB** 上量 **layer-wise／pipelined KV 傳輸與 prefill 計算的重疊程度**（隱藏了多少）：**未找到**（Q21）。
- **vLLM 0.32.0** 上 NixlConnector／MooncakeConnector 的跨機 **bandwidth 與 per-transfer 延遲**官方數字：**未找到**；vLLM 提供指標 `vllm:nixl_xfer_time_seconds`、`vllm:nixl_post_time_seconds`（`nixl_connector_usage.md:477-478`）可用來量，但沒有發佈數字（Q22）。
- SGLang **mooncake／nixl** 在 **H100＋IB** 的 KV 傳輸延遲分解：SGLang 部落格 V1（96×H100，R02_T5）不是以 KV 傳輸為焦點；其他**未找到**（Q23）。
- **新資料是否解決 R02_T5 §3-F 的矛盾**：新找到的條件都在非 H100＋IB 的環境（見上），因此**沒有消除矛盾**，只是增加了「PCIe／低速網路下傳輸佔比大」的例子（E2、vLLM 部落格）與「NVLink 下傳輸可忽略」的例子（LMCache、E3）。

## E-6　互相矛盾或不確定之處

1. **傳輸是否為瓶頸**：TraCT 摘要「KV transfer dominates both TTFT and peak throughput」（100 Gb/s、A6000）；vLLM 官方文件「Disaggregated prefill DOES NOT improve throughput」；LMCache 部落格「vLLM v1 (w/ PD) outperforms vLLM (w/o PD)」（8×H100 NVLink 單機）；FlowKV「almost eliminating the transfer time」；Beyond the Buzz「existing provisioned datacenter bandwidth is sufficient」（R02）。硬體、模型、context 長度、傳輸庫版本都不同。
2. **vLLM 部落格自己的數字**：1.3 s 傳 470 MB（約 0.36 GB/s）遠低於任何 PCIe 4.0 頻寬，作者自述「Bandwidth alone doesn't explain 1.3 s」並歸因於「overhead around the copy」；此為單一部落格的測量，條件為 L40S 無 P2P。
3. **KV 傳輸量**：vLLM 部落格 56 KiB／token（Qwen2.5-7B）與 3 GB／10k token（Llama-3.1-70B，估算）屬不同模型；用於 Llama 405B 的數字**未找到**。
4. **vLLM 0.32.0 vs v0.34.0 connector 清單不同**（E-3），任何 0.32.0 以外的連接器行為要先在 0.32.0 驗證。
5. **E1 的 arXiv 日期**：abs 頁寫 2025-11-14 但編號為 2601；OpenAlex 的發表日 2026-04-27；原因未查。
6. **Mooncake 的 DMA-BUF 路徑所需核心／驅動版本**：文件只說「does not need that module」，未列需求；NVSHMEM 的 dma-buf 條件（核心 ≥ 6.1）是否適用 Mooncake 未知。

---

# 共用部分

## 7　arXiv ID／DOI 清單（供匯出 BibTeX）

**已核實（開 abs 頁、arXiv MCP 或 PDF 確認）**
- 2511.09557（NVRAR；DOI 10.1145/3786335.3813165）
- 2607.16100（Every-µs，R02 已核實）
- 2504.18658（PCCL，R02 已核實）
- 2609.13585（mKernel，R02 已核實；本次讀 PDF 全文）
- 2610.01380（Dissecting GPU-Initiated Communication；[abs]）
- 2605.00686（Perseus；[abs]）
- 2511.15076（GIN for NCCL，R02 已核實）
- 2603.13606（NCCL EP；[abs]）
- 2407.21783（Llama 3；[abs]）
- 2607.08973（SiFAR，R02 已核實）
- 2611.06634（flash-comm；[abs]）
- 2603.05692（Parallelization Strategies for Dense LLM Deployment；arXiv MCP get_abstract）
- 2610.05305（Characterizing Parallelism Strategies；[abs]）
- 2607.05876（Floor-First；[abs]）
- 2412.04964（Flash Communication；[abs]）
- 2411.09510（Communication Compression for TP；[abs]）
- 2606.01502（Move the Query, Not the Cache；[abs]）
- 2505.11329（TokenWeave，R02 已核實）
- 2510.27656（fabric-lib，R02 已核實）
- 2601.08833（Revisiting Disaggregated LLM Serving；[abs]）
- 2512.18194（TraCT；[abs]）
- 2504.03775（FlowKV；[abs]）
- 2607.02574（KV cache survey；[abs]）
- 2610.18444（Optical KV Cache Network；[abs]）
- 2506.05508、2401.09670、2311.18677、2501.14743、2407.00079、2604.00368、2609.13161、2607.01831、2510.09665（皆 R02_T5 已核實，本次未重複開頁）

**未核實，勿直接引用**
- DOI 10.1145/3789240.3828750（ARK，只在 OpenAlex 見摘要）
- DualPath，DOI 10.1145/3789240.3829159（只在 OpenAlex 見摘要，本報告未用）
- NVIDIA 技術部落格 Llama 3.1 405B TP vs PP（只見 WebSearch 摘要）
- 2610.14277（Echo，4-bit TP 通訊，A100；只在 OpenAlex 見摘要，本報告未用）

**非同儕審查資料（網址）**
- vLLM 官方文件 https://docs.vllm.ai/en/latest/serving/parallelism_scaling/ 、https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/
- vLLM 部落格 https://vllm.ai/blog/2026-09-29-disaggregated-serving-guide
- LMCache 部落格 https://blog.lmcache.ai/2025-04-29-pdbench/
- SGLang 文件（repo 內）`docs/docs/references/multi_node_deployment/multi_node.mdx`、`docs/docs/advanced_features/pipeline_parallelism.mdx`、`docs/docs/advanced_features/pd_disaggregation.mdx`
- TRT-LLM https://nvidia.github.io/TensorRT-LLM/latest/examples/llm_mgmn_trtllm_serve.html
- NVSHMEM install guide https://docs.nvidia.com/nvshmem/release-notes-install-guide/install-guide/abstract.html ；NVSHMEM 2.8.0 env https://docs.nvidia.com/nvshmem/archives/nvshmem-280/api/gen/env.html
- GDRCopy https://github.com/NVIDIA/gdrcopy ；NIXL https://github.com/ai-dynamo/nixl ；CUDA GPUDirect RDMA https://docs.nvidia.com/cuda/gpudirect-rdma/

## 8　搜尋紀錄

工具縮寫：AX = arXiv MCP `search_papers`；OA = paper-search `search_openalex`；SS = paper-search `search_semantic`；GS = paper-search `search_google_scholar`；WS = WebSearch；WF = WebFetch。

- Q1 OA「multi-node LLM inference all-reduce InfiniBand tensor parallelism bottleneck NCCL」，約 12 筆；採用 NVRAR（已有）、Flash Communication、GIN、mKernel、MSCCL++（未列，單機為主）；排除 Nereus、LLMEasyQuant（無關）。
- Q2 OA「all-reduce latency LLM inference tensor parallelism small messages GPU」，約 10 筆；採用 SCIN（R02 已有）、2610.05305、2411.09510；排除 CPU-induced slowdown、EnergyLens、Echo（A100，壓縮式）。
- Q3 SS「low-latency all-reduce multi-node tensor parallel LLM decode small messages」：0 筆；Q4 GS「multi-node tensor parallel LLM decode all-reduce latency InfiniBand small message NVSHMEM」：0 筆（兩個工具本次回傳空）。
- Q5 AX `abs:"all-reduce" AND abs:"decode" AND abs:"tensor parallel" AND abs:"multi-node"`：回傳 1 筆（NVRAR）。
- Q6–Q8 AX（`abs:"all-reduce" AND abs:"LLM inference" AND abs:"low-latency"`、`abs:"tensor parallelism" AND abs:"across nodes"`、`abs:"KV cache transfer" AND abs:"InfiniBand"`、`abs:"all-reduce" AND abs:"tensor parallelism" AND abs:"inter-node"`）：全部 HTTP 429 rate limited，重試數次均失敗，**無結果**。
- Q9 `mcp__arxiv__citation_graph(2511.09557)`：Semantic Scholar rate limited，**無結果**。
- Q10 OA「decode all-reduce message size distribution tensor parallel LLM serving hidden size batch size profiling」，約 12 筆；採用 flash-comm、Floor-First；排除 PagedAttention、LLM-PQ、ConServe 等（無關）。
- Q11 OA「GPU-initiated RDMA IBGDA NVSHMEM low-latency collective small message H100 InfiniBand」，約 12 筆；採用 Dissecting、Perseus、mKernel、NCCL EP、2606.01502；排除 SigmaTrace（觀測工具）、FlexLink、UCCL-EP、MSCCL++（R02 範圍或單機）。
- Q12 WS「SGLang Llama 3.1 405B two nodes --tp 16 --nnodes 2 ...」；WF docs.sglang.ai（301 轉址）、docs.sglang.io/references/multi_node.html（404）→ 改讀 repo 內文件。
- Q13 WS「TensorRT-LLM multi-node tensor parallelism ... trtllm-llmapi-launch slurm」；WF TRT-LLM Slurm 範例頁。
- Q14 WS「NVIDIA technical blog Llama 3.1 405B H200 NVLink Switch TP vs PP」；只讀搜尋摘要。
- Q15 WS「NVSHMEM IBGDA PeerMappingOverride NVreg_RegistryDwords nvidia.conf modprobe requirement documentation」；WF NVSHMEM install guide、NVSHMEM env、GDRCopy GitHub、CUDA GPUDirect RDMA 文件、NIXL README。
- Q16 OA「prefill-decode disaggregation KV cache transfer overhead measured InfiniBand GPUDirect RDMA NIXL Mooncake evaluation」，3 筆；Q17 OA「layer-wise KV cache transfer overlap prefill disaggregated serving network bandwidth requirement analysis」，約 15 筆；採用 E1–E6（TraCT、FlowKV、Revisiting、ARK、Optical、Survey）；排除 LayerKV、BanaServe、semi-PD、ShadowServe、SwiftCache、KVStream、OrbitFlow、DUET、Janus、HAWK（異質、模擬或單機記憶體管理，與 H100＋IB 傳輸無直接量測）。
- Q18 WS「NIXL KV cache transfer latency measured H100 ConnectX-7 InfiniBand disaggregated serving Dynamo blog GB/s per GPU」；Q19 WS「vLLM blog disaggregated prefill NIXL KV transfer overhead TTFT H100 InfiniBand measurement 2026」；WF vLLM 部落格、LMCache 部落格。
- Q20 對原始碼：`grep` 於 `/work/allen0117/comm_opt_env/src/{vllm,sglang,trtllm}` 的 `device_communicators/*.py`、`kv_transfer/kv_connector/factory.py`、`arg_groups/`、`disaggregation/`、`docs/`；與 venv 內 `venvs/vllm/lib/python3.12/site-packages/vllm/` 的對應檔。
- Q21–Q23：針對「layer-wise 傳輸與 prefill 重疊」「vLLM 0.32.0 NixlConnector 發佈數字」「SGLang mooncake/nixl H100 IB 延遲分解」：在 Q16、Q17、Q18、Q19 結果中未見，**無專門查詢，視為未找到**。
- 登入節點唯讀指令：`ls -la /dev/gdrdrv`、`ls /dev | grep ...`、`cat /proc/driver/nvidia/{version,params}`、`lsmod`、`ofed_info -s`、`ls /sys/class/infiniband`、`uname -r`、`ucx_info -v`、`ls`（venv 與 NVHPC 目錄）。
- PDF 下載（`/home/allen0117/comm_opt/papers/`，皆非 0 位元組）：2511.09557（既有）、2610.01380、2605.00686、2610.05305、2601.08833、2512.18194、2504.03775、2607.08973、2505.11329、2607.02574。
- 核實篇數：本報告新核實 17 篇（[abs] 或 MCP get_abstract：2610.01380、2605.00686、2603.13606、2407.21783、2611.06634、2603.05692、2610.05305、2607.05876、2412.04964、2411.09510、2606.01502、2601.08833、2512.18194、2504.03775、2607.02574、2610.18444；另重讀 PDF 補新事實：2511.09557、2607.16100、2609.13585、2607.08973）；其餘為 R02 已核實的編號。
