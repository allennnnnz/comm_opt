# R04 BE 組：跨機 TP decode 小訊息 all-reduce（B）與 PD 分離的 KV 傳輸（E）

- **執行**：Sonnet 5.5，2026-11-16／17　**規格**：`docs/specs/S04_direction_deep_dive.md`（BE 組）
- **性質**：文獻＋原始碼＋官方文件＋登入節點唯讀檢查。未跑任何運算、未送 Slurm。只寫事實，不評價題目。
- **路徑慣例**：`檔案:行號` 的根目錄為 `/work/allen0117/comm_opt_env/src/`（release tag：vLLM v0.34.0、SGLang v0.5.23、TRT-LLM v1.2.1、NCCL v2.32.6-1）。「0.32.0」指我們安裝的 `/work/allen0117/comm_opt_env/venvs/vllm/lib/python3.12/site-packages/vllm/`。
- **證據標記**：`[src]` 原始碼已讀；`[doc-local]` 隨 release tag 下載的官方文件原檔（逐字）；`[pdf]` 已下載 PDF 並用 `pdftotext` 讀過（PDF 在 `papers/`，檔名為 arXiv 編號）；`[web-sum]` WebFetch／WebSearch 摘要（非逐字，引用前要再核對）；`[login]` 登入節點 `cbi-lgn01` 的唯讀檢查（**計算節點可能不同**）。
- **工具限制（誠實揭露）**：本次 arXiv MCP 的 `search_papers` 與 `citation_graph` 全程回傳 HTTP 429（arXiv／Semantic Scholar 限流），paper-search MCP 的 semantic／arxiv 搜尋回傳空結果。改用 WebSearch（部分限定 `arxiv.org`）找論文，再用 `curl` 抓 arXiv PDF 逐篇打開核實。因此**沒有做 citation_graph 的後續追蹤**（見第 7 節）。

---

## 白話結論（先看這裡）

1. 三個主流系統的官方文件，遇到「一台 8 張 H100 放不下」的模型，**都不建議把 TP 拉到跨機**：vLLM 文件寫「TP 設成每台 GPU 數、PP 設成機器數」，TRT-LLM 文件直接寫「跨機的 TP 會被慢速互連拖住，建議機內 TP＋機間 PP」，Meta 的 Llama 3.1 論文也是兩台 16 張 H100 用 pipeline parallelism。**但** SGLang 官方文件的 405B 兩機範例與 Nemotron 3 Ultra 的 H100 配方就是 `--tp 16`（跨機 TP），vLLM 文件也列出 `--tensor-parallel-size 16` 的寫法。所以「有人跨機 TP」是事實，只是官方預設建議不是它。
2. 有一篇實測（NVRAR，UMD／LLNL）發現：**decode 為主的負載，TP 比「TP＋PP」好**（因為 PP 切 micro-batch 後 decode 的小矩陣乘法沒變快），但跨機 TP 的 all-reduce 通訊時間在 8→16 張卡時增加約 1.6 倍。另一篇（TKNP，64 張 H100、8×400G InfiniBand）量到：跨機 TP64 時單次 all-reduce 比 TP8 貴 6.5 倍。這兩篇是目前找到最直接的「跨機 TP 通訊很貴」的實測。
3. 在我們會用的 H100＋InfiniBand（非 NVL72）環境，vLLM、SGLang、TRT-LLM 的自製 all-reduce（custom AR、FlashInfer AR、torch symm_mem）**跨機一律關閉，退回 NCCL**（附行號）。也就是說，跨機 TP 的小訊息 all-reduce 目前實際走的是 NCCL 的 Tree／Ring＋LL／LL128。NCCL 2.29.7 已有新的對稱記憶體 all-reduce 核心，但它們是「單一 NVLink 領域」用的；GPU 直發 RDMA（GIN）的專用核心在 v2.32.6 只有 reduce-scatter 與 all-gather，**沒有 all-reduce**。
4. 與既有工作的條件差異：NVRAR 只在 A100（Slingshot）與 GH200（每台 1 張 GPU、InfiniBand）測過；PCCL 只在 Frontier／Perlmutter；Every-µs 全在 GB200 NVL72；MSCCL++ 的多機 all-reduce 只在 A100 圖表，H100 只有單機。**沒有找到** H100 八卡機加 InfiniBand NDR、TP 跨機的小訊息 all-reduce 實測。
5. 「GPU 直接發 RDMA」要不要 root：NVSHMEM 官方安裝指南寫預設模式需要把 `nvidia.ko` 以 `PeerMappingOverride=1` 載入（要改 `/etc/modprobe.d`，需管理員），但 3.4.5 起有「CPU-assisted IBGDA」不需要管理員；NCCL GIN 也有不需要 GPU 直發的 Proxy 後端。登入節點檢查：驅動 550.127.08、`nvidia_peermem` 模組檔存在但**沒載入**、`RegistryDwords` 為空、`gdrdrv` 已載入、核心 4.18。計算節點要在 M20 再確認。
6. E 方向：PD 分離的 KV 傳輸在 H100＋IB 上的實測很少。H100＋400 Gbps 的直接資料點有 KVDirect（2 台 H100 間 22.23 GB/s，UCX 基線 4.05 GB/s）與 Splitwise（用 IB 傳輸量測外推到 DGX-H100：序列化傳輸最多佔端到端 3%，逐層傳輸 0.8%，但第二個 token 延遲 +16.5%）。R02_T5 §3-F 的矛盾，後續論文沒有找到正面處理；新找到的都是「條件不同」（模擬、A100／A40 單機）。

---

## 1. 重點事實（附出處）

**B1：實務上怎麼放 8×H100 放不下的模型**

1. **vLLM 官方文件**：`vllm/docs/serving/parallelism_scaling.md:8-9` [doc-local]：「if the model is too large for a single GPU but fits on a single node with multiple GPUs, use *tensor parallelism*」；「**Multi-node multi-GPU using tensor parallel and pipeline parallel inference:** if the model is too large for a single node, combine *tensor parallelism* with *pipeline parallelism*. Set `tensor_parallel_size` to the number of GPUs per node and `pipeline_parallel_size` to the number of nodes.」同檔 `:122`「The common practice is to set the tensor parallel size to the number of GPUs in each node, and the pipeline parallel size to the number of nodes.」；但 `:130-136` 也列：「Alternatively, you can set `tensor_parallel_size` to the total number of GPUs in the cluster」並給 `--tensor-parallel-size 16`。`:163`「Efficient tensor parallelism requires fast internode communication, preferably through high-speed network adapters such as InfiniBand.」`:219`「`[send] via NET/Socket` … *is not* efficient for cross-node tensor parallelism.」
2. **vLLM 對 Llama-3.1-405B**：`vllm/docs/deployment/frameworks/lws.md:43` [doc-local]：`vllm serve meta-llama/Meta-Llama-3.1-405B-Instruct --tensor-parallel-size 8 --pipeline-parallel-size $(LWS_GROUP_SIZE) --nnodes ...`（LeaderWorkerSet 範例；TP=8、PP=機器數）。
3. **vLLM 對 DeepSeek-V3（H100）**：`vllm/docs/serving/expert_parallel_deployment.md:87` [doc-local]：「It will work on a H200 (or H20) node with 8 GPUs. For H100, you can try to serve a smaller model or refer to the multi-node deployment section.」多機範例 `:151` 用 `deepep_low_latency` 模式（`--data-parallel-size 16`＋EP，TP=1；摘要見 [web-sum]，請讀 `:139-200` 確認）。`context_parallel_deployment.md:46` [doc-local]：Kimi-K2 用 `-tp 16` 時 KV cache 重複 16 倍，建議加 `-dcp`，「the communication overhead is smaller since the DCP communication only happens inside one node.」（`-tp 16` 在這裡指兩台 H100 級機器的跨機 TP）。
4. **SGLang 官方文件**：`sglang/docs/docs/references/multi_node_deployment/multi_node.mdx:6-31` [doc-local]：「**Run 405B (fp16) on Two Nodes**」，兩台各執行 `--tp 16 --dist-init-addr ... --nnodes 2 --node-rank <0|1>`；並註「Note that LLama 405B (fp8) can also be launched on a single node」附 `--tp 8`。`sglang/docs/cookbook/autoregressive/NVIDIA/Nemotron3-Ultra.mdx:80,85` [doc-local]：「BF16: `--tp 16` on H100/H200」「The 16×H100 BF16 setup spans two nodes. … keep `--tp 16`.」→ SGLang 文件有官方的**跨機 TP=16 H100 配方**。
5. **SGLang 對 DeepSeek-V3.x**：`cookbook/autoregressive/DeepSeek/DeepSeek-V3.mdx:51,55`（權重類型表）列 FP8「2×8× H100/H800/H20」、BF16「4×8× H100/H800」；單節點 DP-attention 指令 `--tp 8 --attn-dp-size 8`。`DeepSeek-V3_2.mdx:461-468`：兩機 `--tp 16 --attn-dp-size 2 ... EAGLE`（附 benchmark 註解：dp2＋mtp 6842.51 toks/sec）；`:486`：`--tp 8 --pp-size 2 --moe-a2a-backend deepep --ep-size 16`，文字「CP is currently enabled with PP=2 on Hopper platform and we can reduce TP=16 to TP=8 from standalone deployment」（兩者並存）。
6. **TensorRT-LLM 官方文件**：`trtllm/docs/source/legacy/performance/performance-tuning-guide/deciding-model-sharding-strategy.md`（「If your model is sharded across multiple nodes」段）[doc-local]：「Inter-node connections are typically significantly slower than intra-node connections, so if you have tensor parallelism across nodes it will be bottlenecked by the slow interconects. Consequently, a good starting point is having tensor parallelism within the node and pipeline parallelism between nodes. An exception is if you are running on NVL36 or NVL72 Blackwell systems.」範例：405B、`tensor_parallel_size=8, pipeline_parallel_size=2`。`trtllm/examples/models/core/llama/README.md:1437-1454`：405B BF16 用 `tp8-pp2` engine、Slurm 多機。註：此頁位於 `legacy/`。
7. **Meta Llama 3 論文**（arXiv 2407.21783 §6.1）[pdf]：「Llama 3 405B does not fit in the GPU memory of a single machine with 8 Nvidia H100 GPUs. To address this issue, we parallelize model inference using BF16 precision across 16 GPUs on two machines.」「Across nodes, however, connectivity has lower bandwidth and higher latency, so we use pipeline parallelism … instead.」
8. **DeepSeek-V3 技術報告**（arXiv 2412.19437 §3.4）[pdf]：prefill 最小單元 4 節點 32 GPU，「attention … TP4 with SP, combined with DP8 … Its small TP size of 4 limits the overhead of TP communication」，MoE 用 EP32；decode 最小單元 40 節點 320 GPU，「TP4 with SP, combined with DP80 … EP320 … direct point-to-point transfers over IB … leverage the IBGDA」（H800）。
9. **NVRAR 論文對 TP vs TP＋PP 的實測**（arXiv 2511.09557，CAIS '26）[pdf]：「HP outperforms TP for the most compute-bound and prefill-heavy workload (Figure 1, left), but TP starts to outperform HP as workloads become more memory-bound and decode-heavy.」「Observation 1: … HP is advantageous in compute-bound regimes, whereas TP is better for memory-bound and decode-heavy cases.」原因（Table 4 合成 GEMM）：decode 小 M 時把 M 減半（PP micro-batch）幾乎不減時間，把 K 減半（TP）則明顯減少。TP 的通訊代價：「the communication time in YALIS (TP) increases by ∼1.6×, offsetting the gains from reduced computation time when going from eight to 16 GPUs」（70B、Perlmutter A100）。
10. **TKNP 論文（Decode Token Parallelism，arXiv 2611.13293）** [pdf]：「at batch 128, one all-reduce costs 6.5× more at TP64 than at TP8. Despite 8× less local compute per GPU, end-to-end MLP latency rises by 55%.」硬體：「Each node contains 8×80GB H100 GPUs connected via 4th-generation NVLink … nodes interconnected by 8×400 Gb/s InfiniBand」；軟體「NCCL 2.27.5, vLLM 0.19.0」；Llama 3.1 405B（FP8）。→ 這是 H100＋NDR InfiniBand、vLLM 上、跨機 TP 的實測；作者的對策是讓 dense all-reduce 留在 NVLink 內。

**B2：TP 跨節點時的 all-reduce 路徑**

11. **vLLM（v0.34.0 與 0.32.0 行為相同）**：custom AR 在跨機且無 MNNVL 時整個停用，退回 NCCL。`vllm/vllm/distributed/device_communicators/custom_all_reduce.py:202-203`（`same_node = all(in_the_same_node_as(group, source_rank=0))`、`self.mnnvl_only = not same_node`）；`:232-237`：`if not same_node and not _group_can_attempt_mnnvl(group, device): logger.warning("Custom collectives are disabled because this multi-node group does not have MNNVL-capable GPUs on every rank."); return`（`self.disabled` 初值為 True，`:165`）；`:350-356` 同樣訊息（無 MNNVL multicast）。另 `should_custom_ar` 在 world_size > 8 時直接 False（`:501`，0.32.0 為 `:500`，0.34.0 多了 `self.mnnvl_only`）。torch symm_mem：`symm_mem.py:73-79` world size 不在表內就停用，cc9.0 只有 {2,4,6,8}（`all_reduce_utils.py:58-64`），故 TP=16 停用。FlashInfer AR：`flashinfer_all_reduce.py:174-192`：多機且非 Blackwell → `"FlashInfer allreduce fusion is disabled: multi-node process groups require an MNNVL-capable Blackwell system."`；`:225-227`、`:294-297`：trtllm backend 不支援多機。0.32.0 同樣有這些判斷（`flashinfer_all_reduce.py:157-175, 204, 273`）。→ H100＋IB 跨機 TP 的 all-reduce 在 vLLM 走 `pynccl_comm.all_reduce`（`cuda_communicator.py:459-463`，R03）。
12. **vLLM 0.32.0 → 0.34.0 的差異（與此議題相關的）**：用 `diff` 比對 `custom_all_reduce.py`（22 行）、`all_reduce_utils.py`（13 行）、`flashinfer_all_reduce.py`（55 行）、`cuda_communicator.py`（90 行）：差異為（a）0.34.0 的 `should_custom_ar` 多一個 `mnnvl_only` 判斷、docstring 明寫「CUDA IPC collectives require all ranks to be on the same node; cross-node groups only support MNNVL all-gather and reduce-scatter」；（b）FlashInfer AR 多了 HiSparse 例外與 16 B 對齊檢查；（c）cc10.0／10.3 的 MNNVL 上限表；（d）ROCm AITER／MoRI 相關。**H100 跨機 all-reduce 路徑沒有行為差異**。`symm_mem.py` 兩版逐字相同。
13. **SGLang v0.5.23**：`custom_all_reduce.py:384-385`：「Multi-node v2 is admitted only for a single NVLink clique (see can_use_custom_all_reduce_v2); other cross-node groups fall back to NCCL.」`custom_all_reduce_v2.py:582-597`：跨機時 `return is_one_nvlink_clique(group, device) and _is_vmm_backed_allocator(device)`；`custom_all_reduce_utils.py:396-418`：「True iff every rank's GPU is in the same NVLink fabric clique (one NVL72 / MNNVL domain)」。H100＋IB 兩台不在同一 clique → 退回 NCCL（pynccl／`torch.distributed`，R03 `parallel_state.py:673-770`）。FlashInfer AR 融合另限單機（sm90）（R03：`overrides.py:980-1001`）。
14. **TensorRT-LLM v1.2.1**：`trtllm/cpp/tensorrt_llm/thop/allreduceOp.cpp:1013-1017, 1187`：跨節點時 `mIsP2PSupported = false; // P2P doesn't work across nodes`，`mIsMNNVLSupported = mIsNVLINKSupported && all_ranks_support_mnnvl`；`:1199-1262`（R03）：`!mIsP2PSupported` → `ifFallbackToNCCL`。Python 層 `ops.py:540-550`：MNNVL 策略需 `is_multi_node() and MnnvlMemory.supports_mnnvl() and is_on_aarch64`（R03）。→ x86 H100 跨機 TP 走 NCCL。
15. **NCCL（跨機小訊息）**：NVLSTree 只有 SIMPLE 協定：`nccl/src/config/algorithm_registry.cc:48-50`（註解「NVLS_TREE x LL — combo not implemented」「NVLS_TREE x LL128 — combo not implemented」，`NVLSTREE_SIMPLE` 為 `F_AR`）；`tuning/nvls.cc:22`：NVLS／NVLS_TREE 非 SIMPLE 時排除；`:27`：單節點不用 NVLS_TREE。跨機小訊息候選為 Ring／Tree 的 LL／LL128（`tuning/ring.cc`、`tuning/tree.cc`）。
16. **NCCL 對稱記憶體 all-reduce 與 GIN**：v2.32.6 `src/sym_kernels.cc:60-67` AR 核心集合（`AllReduce_AGxLL_R`、`AGxLLMC_R`、`RSxLD_AGxST`、`RSxLDMC_AGxSTMC`、`RSxTmaLD_AGxTmaST` 等）；`:92-94`：「GinOnly」核心只有 `ReduceScatter_RailA2A_LsaLD`、`ReduceScatter_RailA2A_LsaLDMC`、`AllGather_RailRing_LsaSTMC`——**沒有 all-reduce 的 GIN 專用核心**；`src/init.cc:2048-2050`：`symmetricSupport = isAllCudaP2p && WinEnable && cuMemEnable && (globalGinSupport != NONE || isOneLsaTeams)`。我們安裝的 NCCL 2.29.7 的 `libnccl.so.2`（`/work/allen0117/comm_opt_env/venvs/vllm/lib/python3.12/site-packages/nvidia/nccl/lib/`）用 `strings -a | grep -c` 查到 `AllReduce_AGxLL_R`（27）、`AllReduce_RSxLD_AGxST`（36）、`RailA2A_LsaLD`（91）、`AllGather_RailRing`（16）、`ncclGin`（617）、`NCCL_GIN_TYPE_GDAKI`、`NCCL_GIN_GDAKI_QP_DEPTH` 等字串，`NCCL version 2.29.7+cuda12.9`。（字串存在 ≠ 功能可用；`NCCL_GIN_BACKEND` 在 2.29.7 查不到，GIN 論文用此名，2.29.7 的名稱是 `NCCL_GIN_TYPE`。）

**B3：既有工作（細節見第 2 節）、B4：GPU 直發 RDMA、E**

17. **NCCL GIN 實測（arXiv 2511.15076，NVIDIA）** [pdf]：2 張 H100（EOS DGX H100）間 put-with-signal ping-pong：「NCCL GIN GDAKI achieves 16.7 µs round-trip latency, comparable to NVSHMEM IBRC at 16.0 µs, while NVSHMEM IBGDA achieves 24.3 µs … the Proxy backend achieves 18.0 µs」（4–128 B）。
18. **NVSHMEM 分析（arXiv 2606.05951）** [pdf]：CoreWeave **H200**＋ConnectX-7（每節點 8 NIC）、NVSHMEM 3.3.9：「With IBGDA, inter-node put/get take about 9.4–9.5 µs at 256 B」。
19. **MSCCL++（arXiv 2504.09014，ASPLOS '26）Table 1** [pdf]：H100＋Quantum-2 CX7 400 Gb/s：InfiniBand 延遲 MSCCL++ 4.89 µs vs 最佳可達（perftest）3.76 µs（PortChannel，即 CPU proxy 路徑）；NVLink 吞吐／延遲與最佳可達持平。
20. **KVDirect（arXiv 2501.14743）** [pdf]：最多 4 台 H100、每 GPU 一張 400 Gbps RDMA NIC；「We transfer 1024 blocks between 2 H100 GPUs on two nodes via 400 Gbps NIC … KVDirect achieves 22.23 GB/s bandwidth … while UCX only uses 4.05 GB/s when using 4 connections.」摘要：「reduces per-request latency by 55% compared to the baseline」；vLLM v0.5.3、CUDA 12.4。
21. **Splitwise（arXiv 2311.18677 §VI）** [pdf]：以 IB 實作的 KV 傳輸量測建模（DGX-A100 200 Gbps；DGX-H100 400 Gbps），「The latency impact of serially transferring the KV-cache grows up to 3% of the E2E with large prompts. However, Splitwise only incurs 0.8% of E2E.」「Splitwise adds a 16.5% latency to the second token, as compared to the 64% overhead from a serialized transfer.」
22. **DistServe（arXiv 2401.09670 §6.3）** [pdf]：「even for OPT-175B, the KV Cache transmission only accounts for less than 0.1% of the total latency」，但原因是放置演算法讓 prefill／decode 在同一台機器、走 NVLink；其測試機「cross-node bandwidth is 25Gbps」，A100。

---

## 2. 最接近的既有工作表

欄位：工作／出處／做了什麼／硬體與規模／與本方向重疊／沒涵蓋的條件。R02 已列者用 R02 編號並只補新資訊；「新」表示本次新找到。

### 2.1 B：跨機小訊息 all-reduce 與跨機 TP

| 工作 | 出處 | 做了什麼 | 硬體與規模 | 重疊 | 沒涵蓋／條件差異 |
|---|---|---|---|---|---|
| **NVRAR**（R02_T5 #10／R02_T4 P9；本次核實 PDF） | Singhania et al., arXiv 2511.09557 v4；CAIS '26；https://arxiv.org/abs/2511.09557 | NVSHMEM 階層式 all-reduce：機內 reduce-scatter → 機間 recursive doubling（`put_nbi`，8 B 資料＋旗標融合 payload，序號同步）→ 機內 all-gather；整合進 YALIS／vLLM | Perlmutter：A100，每節點 4 GPU，Slingshot-11；Vista：GH200，**每節點 1 GPU**，InfiniBand；NCCL 2.27.3；最多約 32 GPU（Perlmutter） | 直接重疊：多機 TP decode 的小訊息（128 KB–2 MB）all-reduce。原文：摘要「1.9×–3.6×」，內文「Between 256 KB and 1 MB, NVRAR achieves significant speedups over NCCL (1.06×–1.92×)」（Perlmutter）；Vista「up to 3.5× lower latency」；端到端 405B 最多 1.72×（Perlmutter） | 無 H100、無 8 GPU／機的 InfiniBand 資料。原文自述：「We attribute these higher speedups to the architecture of Vista, where each node has one GPU. As a result, NVRAR only executes the inter-node recursive-doubling phase」；Perlmutter 上「NVRAR is slower than NCCL for 64 KB and 128 KB messages, partly due to kernel launch overheads from its three-phase design」；「We plan to investigate and reduce idle time in future versions」「We leave heuristic-based hyperparameter tuning to future work」。機內階段用 NVSHMEM host API reduce-scatter／all-gather。 |
| **TKNP（Decode Token Parallelism）**（新） | arXiv 2611.13293；https://arxiv.org/abs/2611.13293（本次僅下載 PDF，未見 venue） | 讓 dense 層 TP 留在 NVLink 內、attention 跨節點依請求分擔；vLLM 實作 | 最多 64×H100（8×80GB，NVLink4，8×400 Gb/s InfiniBand）；Llama 3.1 405B、Nemotron-4 340B（FP8）；NCCL 2.27.5、vLLM 0.19.0 | 量到跨機 TP 的 all-reduce 成本：「one all-reduce costs 6.5× more at TP64 than at TP8 … MLP latency rises by 55%」；與我們硬體最接近的一篇 | 目標是 long-context／高併發 decode 吞吐，不是改進 all-reduce；原文限制：「At small batches or short contexts, communication and synchronization can outweigh the benefit」；PD 分離設定（2 prefill 節點＋4／8 decode 節點） |
| **Every µs Matters**（R02 T1-P3） | Shen et al.（NVIDIA＋ETH），arXiv 2607.16100；https://arxiv.org/abs/2607.16100；本次讀 PDF | NCCL device API 上的 barrier-free LLBuffer 單發／雙發對稱 all-reduce；NVLS multicast；vLLM 整合 | **GB200 NVL72**（每節點 4 GPU，72 GPU 同一 NVLink 領域）；vLLM 0.15.1、CUDA 13.1；TP=4（1 節點）與 TP=8（2 節點）皆在 NVLink 領域內 | 小訊息 all-reduce 延遲逼近 SoL（單機 2 GPU 約 7% 開銷）；vLLM ITL TP=8 改善 9–11% | 「multi-node」指 NVL72 內跨節點 NVLink，非 InfiniBand；原文對比：vLLM custom AR、NCCLX ctdirect「only single-node」，「MSCCL++ … multi-node collectives are not supported at the time of writing」；kernel 限制「It requires NVLink to guarantee cache-line–level atomic addition」；未來工作「kernel selection … driven by empirical measurements and could be improved with an accurate performance model」 |
| **MSCCL++**（R02 T4／T1 提及；本次讀 PDF） | Hwang et al.，arXiv 2504.09014 v4，ASPLOS '26；https://arxiv.org/abs/2504.09014 | PortChannel（CPU proxy RDMA）／MemoryChannel／SwitchChannel（NVLS）；DSL；多機 all-reduce 用兩階段階層式（2PH，小訊息用 LL） | A100-40G、A100-80G、H100（8 GPU／節點，**Quantum-2 CX7 400 Gb/s，每 GPU 一 NIC**）、MI300x；baseline NCCL 2.26.2 | H100＋NDR 的 IB 延遲 4.89 µs（Table 1）；H100 單機 all-reduce 小訊息最多 2.8×、SwitchChannel 比 MemoryChannel 最多 +56% 頻寬 | 多機 all-reduce 圖表（Fig. 7，`2n16g` 等）為 **A100-40G**；H100 圖（Fig. 9）標題為「single-node (8 GPUs)」；Every-µs 對 MSCCL++ 的評語見上列 |
| **PCCL: The Big Send-off**（R02_T4 P20；本次核實結論句） | Singh et al.，arXiv 2504.18658；https://arxiv.org/abs/2504.18658 | 兩層階層式 all-gather／reduce-scatter／all-reduce，跨機 recursive doubling／halving，自適應選演算法 | Frontier（MI250X，Slingshot，至 2048 GCD）、Perlmutter（A100，Slingshot） | 階層式＋跨機 recursive doubling 的同類設計 | 原文：「In future work, we plan to benchmark PCCL on clusters with InfiniBand interconnects」；重點為大訊息（訓練 all-gather／reduce-scatter），不是 decode 小訊息 |
| **GIN: GPU-Initiated Networking for NCCL**（R02_T4 P10；本次讀 PDF） | Hamidouche et al.（NVIDIA），arXiv 2511.15076；https://arxiv.org/abs/2511.15076 | NCCL 2.28 Device API：LSA／Multimem／GIN（GDAKI＝DOCA GPUNetIO 直發；Proxy＝CPU 代發）；整合 DeepEP | EOS DGX H100，2／4／8 節點；NVSHMEM 3.4.5、NCCL 2.28 | H100＋IB 上 GPU 直發的點對點延遲（見重點事實 17） | 應用為 MoE all-to-all（DeepEP），不是 all-reduce；原文要求：「GPU-initiated networking requires the nv_peer_mem kernel module … and appropriate driver stacks」 |
| **Demystifying NVSHMEM**（新） | Langer et al.（NVIDIA＋ETH），arXiv 2606.05951；https://arxiv.org/abs/2606.05951 | NVSHMEM 的對稱記憶體、單邊操作、device 端 collective 的系統級量測（含 AllReduce） | CoreWeave **H200**×8／節點，ConnectX-7（8 NIC／節點）；NVSHMEM 3.3.9、CUDA 13.0 | IBGDA 跨機小訊息 put/get ≈ 9.4–9.5 µs（256 B）；含 NVSHMEM AllReduce 結果 | H200、CUDA 13；「few inter-node NVSHMEM measurements are missing due to timeouts」；非 TP decode 情境 |
| **GPU-Initiated Communication: Dissecting Down to the Bone**（新） | Baydamirli et al.，arXiv 2610.01380；https://arxiv.org/abs/2610.01380 | 拆解 GPU 直發 RDMA 的每操作成本（GPU 發 doorbell 的 mini-gda 與 host proxy 比較）；跨 H100／H200／B200／GB200 | P-IB：4×4 H200、CX-7 200 Gb/s IB；P-H100：100×4 H100、CX-7 200 Gb/s IB，驅動 580.95／CUDA 13.0；NVSHMEM 3.4.5／3.7.2、NCCL 2.30.7／2.31.2 | 說明 GPU 直發的前提：「the NVIDIA kernel module permits it only with the option PeerMappingOverride=1 [23]」「NVSHMEM's CPU-assisted mode … for systems where the mapping is disallowed」；並指出「P-H100 is a larger cluster without NIC doorbell mapping support」 | 驅動 580+、CUDA 13；非 all-reduce |
| **SiFAR**（R02_T5 #2；本次讀 PDF） | Taneja et al.（GaTech＋NVIDIA），arXiv 2607.08973 | 去同步 all-reduce（雙緩衝、in-switch 「redundant pull」、跨 all-reduce 去 barrier） | 單機 8×H200，TP=2/4/8 | 量到 Llama-3.1-8B TP=8 all-reduce 占 TPOT 30% | 單機；Hopper multimem；無跨機 |
| **TokenWeave**（R02_T5 #16） | arXiv 2505.11329，MLSys '26 | 融合 AllReduce＋RMSNorm、計算通訊重疊 | 8×H100 單機 | 小 decode batch 效益消失（R02 引文） | 單機、不支援 CUDA Graph（R02） |
| **Flash Communication**（新） | Li et al.，arXiv 2412.04964；https://arxiv.org/abs/2412.04964 | TP all-reduce 低位元壓縮（INT8 等） | 機內為主（摘要：「substantially boosts intranode communication speed by more than 3×」） | 減少 TP all-reduce 的資料量（另一條路線） | 機內；準確度取捨 |
| **SCIN**（R02_T5 #8） | arXiv 2603.28239 | 交換器中心的網路內架構（FPGA＋模擬） | FPGA 原型＋cycle-level 模擬（R02） | 網路內加總（A 組範圍） | 規模受限（R02 引文） |
| **Prime PCCL**（R02_T4 P19） | arXiv 2505.14065 | 公網容錯 CCL | WAN TCP | — | 與 IB 無關（R02 已排除） |

### 2.2 E：PD 分離的 KV cache 傳輸

| 工作 | 出處 | 做了什麼 | 硬體與規模 | 與 E 重疊 | 沒涵蓋／條件 |
|---|---|---|---|---|---|
| **KVDirect**（R02_T5 #19；本次讀 PDF） | Chen et al.（Rutgers／ByteDance），arXiv 2501.14743 | pull 式 RDMA 張量通訊庫，比 UCX 訊息式傳輸省 CPU | **最多 4 台 H100（8×H100-80G），每 GPU 400 Gbps RDMA NIC**；vLLM v0.5.3 | 兩台 H100 經 400 Gbps NIC：KVDirect 22.23 GB/s、UCX（4 連線）4.05 GB/s | 原文未提 InfiniBand 或 RoCE 的字樣（grep 未命中）；vLLM 0.5.3 很舊；自述未來：閒置記憶體當 prefix cache、prefill／decode 角色互換 |
| **Splitwise**（R02_T5 #24；本次讀 PDF） | Patel et al.，arXiv 2311.18677 | 階段分離；KV 序列化傳輸 vs 逐層傳輸 | DGX-A100（200 Gbps IB）、DGX-H100（400 Gbps IB）；傳輸開銷由 IB 實作量測建模（§VI-A） | 見重點事實 21：端到端 ≤3%（序列化）／0.8%（逐層），第二 token +16.5%（逐層）vs +64%（序列化） | 傳輸用「建模」而非即時多機服務；原文：「Since we do not have access to such hardware and/or optimized LLM implementations, we leave this to future work」（R02 引文） |
| **DistServe**（R02_T5 #23） | arXiv 2401.09670 | 搭配放置演算法；傳輸占比 | 4 節點 32×A100-80GB，跨機 25 Gbps | <0.1% 總延遲（OPT-175B）；但經 NVLink | 自述「does not implement advanced runtime policies like preemption and fault tolerance」（R02）；非 H100／IB |
| **Mooncake**（R02_T5 #22） | arXiv 2407.00079；FAST '25 | KV 中心架構、Transfer Engine | 每節點 8×A800，RDMA NIC 合計最高 800 Gbps | Transfer Engine 為 vLLM／SGLang 的 KV 連接器後端 | A800；dummy 模型 |
| **Beyond the Buzz**（R02_T5 #15；本次核實引文） | arXiv 2506.05508 | 解析模型：PD 分離的適用條件 | 專有高保真 GPU 模擬器；Blackwell、FP4 | 「Our analysis indicates that existing provisioned datacenter bandwidth is sufficient to support KV cache transfer without becoming a bottleneck.」 | 模擬；Blackwell；R02 已記 |
| **Revisiting Disaggregated LLM Serving for Performance and Energy**（新） | Li et al.（UIUC／IBM），arXiv 2601.08833；https://arxiv.org/abs/2601.08833 | 比較 KV 傳輸路徑（GPU P2P via NIXL cuda_ipc、CPU DRAM 等）的效能與能耗 | **2×A100，同一台、PCIe bridge**；vLLM V1＋LMCacheConnectorV1＋NIXL | 摘要：「performance benefits from prefill–decode disaggregation are not guaranteed and depend on the request load and KV transfer mediums」 | A100、單機、無 IB |
| **ConServe（Observation, Not Prediction）**（新） | Ding et al.，arXiv 2606.01839；https://arxiv.org/abs/2606.01839 | 以「對話」為排程單位；KV 傳輸占 TTFT 的量測 | **4×A40**，Qwen3-0.6B（§實驗） | 「While KV-transfer contributes significantly to TTFT at low input tokens (> 20%), it scales linearly … overshadowed by quadratic prefill latency at long inputs」（Fig. 3） | A40＋小模型；傳輸介質未在我讀到的段落說明 |
| **NetKV**（新） | Ojewale，arXiv 2606.03910；https://arxiv.org/abs/2606.03910 | 以網路成本選 decode 實例 | **64 GPU 四層 fat-tree 模擬器**，Mooncake trace；參數「64-H100 class」（B0=450 GB/s…） | TTFT 最多 −21.2%（模擬） | 全為模擬；「no changes to the transport」 |
| **TENT**（R02_T5 #7）、**fabric-lib**（#11）、**PDD**（#1）、**Lynx**（#3）、**LMCache**（#13，arXiv 2510.09665，已下載 PDF） | 見 R02_T5 | 多軌噴灑／one-sided WRITE／跨資料中心／量化傳輸／KV 搬運層 | R02 已列：H800／H20 RoCE、H200 EFA 等 | — | 無 H100＋IB NDR 資料（R02 §5） |

---

## 3. 現成軟體的支援程度

**NCCL（我們：2.29.7；原始碼：2.32.6-1）**
- 跨機 all-reduce 的小訊息路徑為 Ring／Tree × LL／LL128（見重點事實 15）；NVLSTree 僅 SIMPLE（`algorithm_registry.cc:48-50`），故小訊息不會走 NVLSTree。2.29.7 的 NVLS 預設行為見 R03 §2.8（`NCCL_NVLS_ENABLE` 預設 2）。
- 對稱記憶體 all-reduce 核心（AGxLL 等）與 GIN 相關字串都在 2.29.7 的二進位內（重點事實 16，`strings` 檢查）。2.27.3 起對稱記憶體 API、2.28.3 起 device API／GIN（R03 §2.8）。**但**我們沒有跑，所以「2.29.7 能否在 H100＋IB 兩台上啟用這些核心」是未驗證：`init.cc:2048-2050` 的 `symmetricSupport` 要求 `isAllCudaP2p`（跨機時的值未追到）且「GIN 連線或單一 LSA team」。
- GIN 後端版本相容表：`src/gin/gin_host.cc:23-26`：proxy 後端 `{0, 2.30.3, 2.30.5, 2.32.0}`、gdaki `{0, 2.30.3, 2.30.5}`（陣列索引為後端版本，值為最低相容 NCCL 版本）。此表的語意（後端外掛版本 vs NCCL 版本）未逐行追，**2.29.7 內建 GIN 後端的行為需實測**。
- 前提文件：`nccl/docs/userguide/source/usage/deviceapi.rst:25-55`（見第 4 節）。

**vLLM（我們：0.32.0；原始碼：0.34.0）**：跨機 TP 在 H100＋IB 全走 NCCL（重點事實 11–12）；官方文件建議 TP＝每機 GPU 數、PP＝機器數（重點事實 1），並提供 `--tensor-parallel-size 16`＋Ray 的替代寫法（`parallelism_scaling.md:130-136`）。symm_mem 的 `multimem_all_reduce_` 只在 ws∈{4,6,8}。兩版在此路徑**無行為差異**。

**SGLang（v0.5.23）**：官方兩機 `--tp 16` 範例（重點事實 4）；all-reduce 跨機退回 NCCL（重點事實 13）。另有 `--enable-nccl-nvls`、`--enable-symm-mem`（NCCL 對稱記憶體，預設 False，R03 `exec_.py:715-724`）。

**TensorRT-LLM（v1.2.1）**：官方建議機內 TP＋機間 PP（重點事實 6）；跨機 TP 走 NCCL（重點事實 14）。

**FlashInfer（v0.7.4）**：trtllm backend 僅單機；mnnvl backend 跨機需 Blackwell MNNVL（R03 §1.1）。

**NVSHMEM／NVRAR**：NVRAR 為獨立實作（`https://github.com/hpcgroup/nvrar`，MIT；`https://github.com/axonn-ai/yalis`，Apache-2.0；R02 §6），**未整合進** vLLM／SGLang／TRT-LLM 主線（本次對 vLLM `vllm/`、SGLang `python/`、TRT-LLM `tensorrt_llm/` 與 `cpp/` 做不分大小寫 `grep -rli nvrar`，無命中）。

---

## 4. 在 Nano5 的前提條件

登入節點為 `cbi-lgn01`（`hostname` 輸出）。登入節點有兩張 GPU（`/proc/driver/nvidia/gpus`：`0000:a8:00.0`、`0000:b8:00.0`）與 mlx5 網卡，**與 H100 計算節點的設定不一定相同**，以下僅供參考。

| 前提 | 為什麼需要 | 怎麼檢查 | 登入節點結果（[login]）／狀態 |
|---|---|---|---|
| 驅動 ≥ 510.40.3 | NCCL GIN：「NVIDIA GPU drivers >= 510.40.3」（`deviceapi.rst:33`）；NVSHMEM IBGDA：「Requires nvidia.ko >= 510.40.3」[web-sum] | `cat /proc/driver/nvidia/version` | `NVRM version: NVIDIA UNIX x86_64 Kernel Module  550.127.08  Fri Oct 25 22:14:46 UTC 2024` → 符合；計算節點需於 M20 確認同版 |
| `nvidia.ko` 以 `PeerMappingOverride=1` 載入（NVSHMEM 預設 IBGDA 模式；GIN GDAKI 的 GPU 直發 doorbell 也屬此類機制） | NVSHMEM 安裝指南 [web-sum]：「In the default case, `nvidia.ko` must be loaded with `PeerMappingOverride=1` … `/etc/modprobe.d/nvidia.conf` … options nvidia `NVreg_RegistryDwords="PeerMappingOverride=1;"`」；CPU-assisted 模式「PeerMappingOverride is not required」；Dissecting 論文 [pdf]：「the NVIDIA kernel module permits it only with the option PeerMappingOverride=1」 | `cat /proc/driver/nvidia/params \| grep -E "RegistryDwords\|EnableStreamMemOPs"`；計算節點同 | `RegistryDwords: ""`、`RegistryDwordsPerDevice: ""`、`EnableStreamMemOPs: 0` → **登入節點未設 PeerMappingOverride**；無 root 無法自行更改；計算節點需在 M20 確認 |
| `nvidia_peermem` 已載入（或改用 DMA-BUF） | GIN：「If using nvidia-peermem, Mellanox OFED >= 5.0 or DOCA」（Proxy，`deviceapi.rst:45`）；GDAKI：「If using DMA-BUF, linux kernel >= 6.1. If using nvidia-peermem on baremetal, linux kernel >= 5.12 and either Mellanox OFED or DOCA driver.」（`:50-54`）；Mooncake：`WITH_NVIDIA_PEERMEM` 預設 1 需 `nvidia-peermem`，設 0 走 DMA-BUF（`vllm/docs/features/mooncake_connector_usage.md:62-65`） | `lsmod \| grep -i peermem`；`modinfo nvidia_peermem` | `lsmod` 無 `nvidia_peermem`（未載入）；`modinfo nvidia_peermem` 顯示 `/lib/modules/4.18.0-553.144.1.el8_10.x86_64/extra/nvidia-peermem.ko.xz`、`version: 550.127.08`、`license: Linux-OpenIB` → 模組檔存在、未載入。計算節點需在 M20 確認（可能由 Slurm prolog 載入） |
| `gdrdrv`（GDRCopy） | NVSHMEM：「GDRCopy v2.0 or later … required for atomics support on non-NVLink connections … required if CPU-assisted IBGDA is enabled」[web-sum]；GIN Proxy：完成狀態可放 GPU 可見記憶體（GDRCopy 可用時）[pdf 2511.15076] | `lsmod \| grep gdrdrv`；`ls /dev/gdrdrv` | `gdrdrv 24576 0`（`nvidia` 的使用者之一）、`/dev/gdrdrv` 存在 → 登入節點有 |
| Linux 核心版本 | GIN GDAKI：DMA-BUF 需 kernel ≥ 6.1；nvidia-peermem 裸機需 ≥ 5.12（`deviceapi.rst:50-54`）；NVSHMEM DMA-BUF：「rdma-core >= 44.0 and an upstream kernel >= 6.1」[web-sum] | `uname -r` | `4.18.0-553.144.1.el8_10.x86_64`（RHEL 8 核心，上游版號 4.18；RHEL 有大量 backport，文件的版本門檻是否以上游版號計未知）。**GDAKI 在此環境的可用性無法由文件單獨判定** |
| Mellanox OFED／rdma-core 版本 | GIN：rdma-core ≥ 34.0（Proxy＋DMA-BUF）、≥ 44.0（GDAKI）；OFED ≥ 5.0（Proxy）／≥ 5.8（GDAKI）（`deviceapi.rst:41-55`） | `ofed_info -s`；`rpm -q rdma-core` | `ofed_info -s`：`MLNX_OFED_LINUX-23.10-2.1.3.1.201:`（≥ 5.8 符合）；`ibstat`：`CA 'mlx5_0' CA type: MT4129`（ConnectX-7）、韌體 `28.40.1000`。`rdma-core` 版本未查（`rpm -q` 未執行） |
| NIC 型號 ≥ ConnectX-6 Dx（GDAKI）／CX4（Proxy） | `deviceapi.rst:41,50`；GIN 論文 [pdf]：「ConnectX-6 Dx or newer」 | `ibstat`／`ibv_devinfo` | 登入節點為 MT4129（ConnectX-7）；計算節點每台 8× NDR（plan 文件）需於 M20 確認型號 |
| CUDA 版本 | GIN 編譯：「CUDA 12.2 or later when compiling」（`deviceapi.rst:35`）；NCCL 2.29.7 為 cuda12.9 版；NVSHMEM 3.4.5 說明：CPU-assisted IBGDA 免 GDRCopy、免 x86 regkey [web-sum] | 計算節點 `nvidia-smi`、模組 | 我們的環境為 CUDA 12.x（驅動 550）；較新的論文（2606.05951、2610.01380）用 CUDA 13／驅動 580+，其結果與 12.x 的相容性未知 |
| 同節點 GPU↔NIC 親和（PIX） | GIN 論文 [pdf]：「Optimal performance requires co-location of GPUs and NICs on the same PCIe root complex」 | 計算節點 `nvidia-smi topo -m` | 需在計算節點檢查（給 M20） |
| IB 交換器為 NDR、有無 SHARP | 與 B 關聯：SHARP 屬 A 組 | — | 見 A 組報告；需在計算節點檢查 |

NVSHMEM 官方釋出說明（3.4.5）[web-sum]：「Added support for CPU-Assisted IBGDA without the use of GDRCopy or the x86 regkey setting. It enables the use of IBGDA on a broad range of systems without the need for administrator intervention.」「IBGDA does not work with CX-4 when the link layer is Ethernet (RoCE).」

---

## 5. 尚未有人量過或作者自述未解的問題（只列事實）

1. **H100（每機 8 GPU）＋InfiniBand 的跨機小訊息 all-reduce**：在本次檢視的文獻中未找到。NVRAR（A100／GH200 單 GPU 節點）、PCCL（Slingshot）、Every-µs（GB200 NVL72）、MSCCL++（多機圖表僅 A100）、GIN（H100 但為 MoE all-to-all）、TKNP（H100＋IB 但只量 NCCL all-reduce）各自缺此條件。
2. **NVRAR 作者自述未解**：「We plan to investigate and reduce idle time in future versions of NVRAR」（§5.2.2，Perlmutter 上 NVRAR 空閒時間略高）；「We leave heuristic-based hyperparameter tuning to future work」（附錄）；vLLM（HP）的 GPU 空閒時間異常高：「we leave detailed investigation to future work」。
3. **PCCL**：「we plan to benchmark PCCL on clusters with InfiniBand interconnects」（結論段，arXiv 2504.18658）。
4. **GIN**：結論段稱未來將推廣至 PyTorch 分散式訓練、TensorRT-LLM 等（R02_T4 P10 引文）；本次檢視的版本只提供 MoE（DeepEP）數據。NCCL v2.32.6 的 GIN 專用核心只有 reduce-scatter／all-gather（重點事實 16）。
5. **Every-µs**：kernel 選擇靠經驗，「could be improved with an accurate performance model」；僅限 NVLink 領域（「It requires NVLink to guarantee cache-line–level atomic addition」）。
6. **NIMBLE**（R02_T4 P5）：未來工作含 IBGDA 版本（R02 引文）。
7. **TKNP**：「At small batches or short contexts, communication and synchronization can outweigh the benefit of distributed attention」；「larger NVLink domains shift the balance」；DCP 需要跨機 TP 的問題留待未來（文中列為延伸方向）。
8. **E（PD 分離 KV 傳輸）**：KVDirect 自述的未來：「use the idling memory as a prefix cache」「occasionally switch the role of prefill and decode workers」；Splitwise 在無硬體時「we leave this to future work」（R02 引文）；NetKV 與 ConServe 為模擬／小模型。**沒有找到** 用 NIXL／UCX／Mooncake 在 H100＋IB NDR 上量 KV 傳輸占端到端比例的論文；UCCL 部落格稱「nearly no benchmarks of open-source KV transfer engines」（[web-sum]，https://uccl-project.github.io/posts/kv-transfer-engine ，未打開全文）。

---

## 6. 矛盾或不確定之處（不裁決）

1. **文件間的跨機 TP 建議不一致**：vLLM／TRT-LLM 文件建議機內 TP＋機間 PP（TRT-LLM 明講跨機 TP 會被慢速互連拖住）；SGLang 文件的 405B 與 Nemotron 3 Ultra 配方是 `--tp 16`；SGLang 的 DeepSeek-V3.2 文件同時列 `--tp 16 --attn-dp-size 2` 與 `--tp 8 --pp-size 2`。TRT-LLM 的建議頁位於 `legacy/`，是否仍為現行建議未確認。
2. **NVRAR 的 PP vs TP 結論取決於條件**：HP（TP＋PP）在 prefill 重的負載較好、TP 在 decode 重的負載較好（同一篇）。NVRAR 論文中的 vLLM HP 使用 V0 引擎，因為 V1 的 Ray PP 出現 hang（原文：「we observed persistent hangs with the V1 engine when using Ray-based PP」）。
3. **NVRAR 數字的版本差異**：摘要「1.9×–3.6×」「128 KB–2 MB」（Slingshot／InfiniBand 合併敘述）；內文：Perlmutter「256 KB–1 MB … 1.06×–1.92×」、Vista「up to 3.5×」、同論文另處出現「1.14×–3.57×」（InfiniBand，圖表標示）。R02 已記此矛盾。
4. **GIN 延遲數字的「往返」定義**：GIN 論文的 16.7 µs 為 ping-pong「round-trip」；NVSHMEM 分析論文的 9.4–9.5 µs 為「inter-node put/get」（單向或完成？本次未逐段確認），硬體（H100 vs H200）、NVSHMEM 版本（3.4.5 vs 3.3.9）、CUDA 版本（12.x vs 13.0）皆不同，數字不可直接比較。
5. **NCCL GIN 環境變數名**：GIN 論文寫 `NCCL_GIN_BACKEND`；2.29.7 二進位內查到 `NCCL_GIN_TYPE`（`NCCL_GIN_BACKEND` 字串數 0）。
6. **NCCL 2.29.7 vs 2.32.6**：R03 版本表指出 2.29.2 起內建 hybrid AllGather 對稱核心；`gin_host.cc:23-26` 的後端版本表最低出現 2.30.3。2.29.7 的 GIN 能力僅以字串存在為證，未執行驗證。
7. **IBGDA 的管理員需求**：NVSHMEM 安裝指南（[web-sum]）寫預設模式需 `PeerMappingOverride=1`（改 `/etc/modprobe.d`），3.4.5 說明（[web-sum]）稱 CPU-assisted IBGDA 免管理員；實際在 Nano5 計算節點上哪個模式可用，未知。
8. **WebFetch 摘要與原檔**：vLLM 官方網站 `https://docs.vllm.ai/en/latest/serving/parallelism_scaling/` 的 WebFetch 摘要與 `[doc-local]` 的逐字內容一致（`multi-node`、`InfiniBand`、`NCCL_IB_HCA=mlx5` 等）；SGLang 的 `docs.sglang.ai` 會 301 到 `docs.sglang.io`，其 `multi_node.html` 回 404，故改以 `[doc-local]` 為準。
9. **vLLM 0.32.0 實際跑 TP=16 時的路徑**：由 `diff` 與原始碼判斷與 0.34.0 相同，但**未實際執行**（無 GPU／Slurm）；日誌上應可看到「Custom collectives are disabled because this multi-node group does not have MNNVL-capable GPUs on every rank.」可作為 M20 的檢查項。

---

## 7. 搜尋紀錄

**官方文件／原始碼**
- 本地 `grep`／`sed`：`vllm/docs/serving/{parallelism_scaling,expert_parallel_deployment,context_parallel_deployment}.md`、`vllm/docs/deployment/frameworks/lws.md`、`vllm/docs/features/{disagg_prefill,nixl_connector_usage,mooncake_connector_usage}.md`、`sglang/docs/docs/references/multi_node_deployment/multi_node.mdx`、`sglang/docs/cookbook/autoregressive/{DeepSeek,NVIDIA}/…`、`sglang/docs/docs/advanced_features/pd_disaggregation.mdx`、`trtllm/docs/source/{legacy/performance/performance-tuning-guide/deciding-model-sharding-strategy.md,features/disagg-serving.md,blogs/tech_blog/blog5_Disaggregated_Serving_in_TensorRT-LLM.md,deployment-guide/*}`、`trtllm/examples/models/core/llama/README.md`、`nccl/docs/userguide/source/usage/deviceapi.rst`、`nccl/src/{sym_kernels.cc,init.cc,gin/gin_host.cc,config/algorithm_registry.cc,tuning/*.cc}`。
- `diff` vLLM 0.32.0（venv）vs 0.34.0（src）：`custom_all_reduce.py`、`all_reduce_utils.py`、`cuda_communicator.py`、`flashinfer_all_reduce.py`、`symm_mem.py`。
- `strings -a libnccl.so.2 | grep`（2.29.7）。
- WebFetch：`docs.vllm.ai/en/latest/serving/parallelism_scaling.html`（重導）、`.../parallelism_scaling/`；`docs.sglang.ai/...`（301→404，棄用）；`docs.sglang.io/basic_usage/deepseek_v3.html`；`docs.nvidia.com/nvshmem/release-notes-install-guide/install-guide/abstract.html`；`.../prior-releases/release-3405.html`。皆為 `[web-sum]`。

**文獻（工具與查詢）**
- arXiv MCP `search_papers`（欄位前綴查詢，如 `abs:"all-reduce" AND abs:"multi-node" AND abs:"decode"`、`abs:"KV cache transfer" AND abs:"disaggregated" AND abs:"InfiniBand"`、`ti:"MSCCL++"…`）與 `citation_graph(2511.09557)`：**皆回 429，無結果**；paper-search `search_semantic`（3 個查詢）／`search_arxiv`（1 個）：空結果。
- WebSearch（standard）：「SGLang multi-node tensor parallel Llama 3.1 405B 2 nodes --tp 16」「TensorRT-LLM Llama 3.1 405B multi-node tensor parallelism pipeline parallelism 16 GPUs H100」「vLLM DeepSeek-V3 deployment multi-node 2x8 H100 …」「arXiv MSCCL++ GPU-driven communication … multi-node allreduce small message」「arXiv cross-node tensor parallelism LLM decode all-reduce latency H100 InfiniBand NVSHMEM custom kernel 2026」「arXiv KV cache transfer time fraction of TTFT … H100 InfiniBand NIXL UCX」「multi-node tensor parallelism inference decode all-reduce latency across nodes InfiniBand low-latency collective（限 arxiv.org）」「NVSHMEM GPU-initiated allreduce hierarchical inter-node latency … H100 IBGDA（限 arxiv.org）」「Mooncake Transfer Engine KV cache transfer H100 InfiniBand NDR …」「NVSHMEM IBGDA PeerMappingOverride … requirement」「NVIDIA SHARP low-latency aggregation small message allreduce …」。每查詢看約 10 筆標題＋摘要。
- 逐篇核實（curl 下載 PDF 到 `papers/`，檔案大小皆非 0，`pdftotext` 後 grep 原文）：2511.09557、2407.21783、2412.19437、2511.15076、2607.16100、2504.18658、2607.08973、2603.28239（僅下載）、2505.11329（僅下載）、2501.14743、2506.05508、2311.18677、2401.09670、2510.09665（僅下載）、2407.00079（僅下載）、2510.27656（僅下載）、2604.00368（僅下載）、2505.09343（僅下載）、2606.03910、2606.01839、2601.08833、2504.09014、2606.29986（僅下載）、2611.13293、2606.05951、2610.01380、2607.07862、2412.04964、2606.01502。「僅下載」者未逐段讀，表中引用內容皆來自 R02 或已讀段落。
- **排除／未列入**：2607.07862（CTA-Pipelining，8×H200／B200 單機，與跨機無關）；2606.01502（Move the Query，H100 多機 IBGDA，標題與摘要顯示為跨實例注意力搬運，**非 all-reduce**；其「real multi-node H100 cluster … real IBGDA」可能對 B4 的可行性有參考價值，**本次只讀摘要，未讀全文**）；SHARP 低延遲：屬 A 組，只做了一次 WebSearch（只找到 NVIDIA 官方文件與部落格，無 SHARP 開關對照的論文；`docs.nvidia.com/networking/display/sharpv370/…` 的 OSU allreduce 表為 [web-sum]，未核實硬體世代）；UCCL KV transfer engine 部落格（未開全文）；第三方 skill 頁面（TRT-LLM 405B TP/PP 配置互相矛盾，WebSearch 摘要已註明）。

---

## 8. 引用清單（arXiv 編號或 DOI；BibTeX 由 Opus 之後以 arXiv MCP／DOI 匯出）

- 2511.09557 NVRAR／Understanding and Improving Communication Performance in Multi-node LLM Inference（CAIS '26，DOI 10.1145/3786335.3813165，取自 R02）
- 2611.13293 Decode Token Parallelism (TKNP)
- 2607.16100 Every Microsecond Matters
- 2504.09014 MSCCL++（ASPLOS '26）
- 2504.18658 The Big Send-off (PCCL)
- 2511.15076 GPU-Initiated Networking for NCCL
- 2606.05951 Demystifying NVSHMEM
- 2610.01380 GPU-Initiated Communication: Dissecting Down to the Bone
- 2607.08973 SiFAR
- 2505.11329 TokenWeave
- 2412.04964 Flash Communication
- 2603.28239 SCIN（switch-centric in-network architecture）
- 2407.21783 The Llama 3 Herd of Models
- 2412.19437 DeepSeek-V3 Technical Report
- 2501.14743 KVDirect
- 2311.18677 Splitwise
- 2401.09670 DistServe
- 2407.00079 Mooncake
- 2506.05508 Beyond the Buzz
- 2601.08833 Revisiting Disaggregated LLM Serving for Performance and Energy Implications
- 2606.01839 Observation, Not Prediction（ConServe）
- 2606.03910 NetKV
- 2510.09665 LMCache、2510.27656 fabric-lib、2604.00368 TENT（R02 已列）
- 官方文件（非論文）：vLLM `docs/serving/parallelism_scaling.md`、`expert_parallel_deployment.md`（v0.34.0 tag）；SGLang `docs/references/multi_node_deployment/multi_node.mdx`（v0.5.23 tag）；TRT-LLM `deciding-model-sharding-strategy.md`（v1.2.1 tag）；NCCL `deviceapi.rst`（v2.32.6-1 tag）；NVSHMEM Install Guide 與 3.4.5 Release Notes（`https://docs.nvidia.com/nvshmem/release-notes-install-guide/…`，[web-sum]）
