# R08_A　計算與通訊重疊：論文細節補充（依 S08 分工 A）

- **撰寫**：Sonnet 5.5 執行者，2026-11-25（依 `docs/specs/S08_overlap_space.md` 分工 A）
- **範圍**：補 R02_T2、R04_CD 沒寫到的細節（Hopper 預填／解碼實際數字、比較對象細節、雙向拖慢處理、作者自述限制），並新增 NanoFlow、ISO、PyTorch Async-TP 與少量新論文。只寫事實，不做選題判斷。
- **核實方式**：PDF 存在 `/home/allen0117/comm_opt/papers/<arXiv編號>.pdf`（本次新下載 14 份，皆確認大小非 0），以 `pdftotext` 讀全文後 grep 對照。數字後面括號是原文位置（節、表、圖號；找不到節號的寫段落標題）。「僅摘要」＝只看過摘要。「未說明」＝我在全文中找不到。
- **工具狀況**：arXiv MCP 的 `search_papers`、`get_abstract` 可用；`citation_graph` 全程被 Semantic Scholar 限流（兩次 RATE_LIMITED，另以 curl 直連也回 429），**沒有做到「用 citation_graph 往後找」**，改用 11 條 arXiv 關鍵字查詢（見 §6）。
- 「論文編號」：沿用 R02_T2 的 `#N`；新增論文用 arXiv 編號。

---

## 0. 讀前先看：對 R02_T2 數字的更正與補充（我從全文核實到的差異）

1. **FLUX（#23）推論數字**：R02 寫「推論 1.66×／1.30×（prefill／decode，8 GPU，對 vLLM）」。全文（§5 Evaluation 末段）實際是分三種叢集：A100 PCIe 對 vLLM 1.46× prefill／1.28× decode；A100 NVLink 1.45×／1.30×；**H800 NVLink：prefill 1.66×，decode「no decoding speedups」**。摘要的 1.66×／1.30× 是跨叢集的最大值，不是同一台機器。
2. **Syncopate（#7）「端到端 1.3×」**：全文的 "end-to-end" 是 **operator 層級延遲**（§評估 "consistently reduces end-to-end operator latency"），工作負載是 FFN／attention 單一算子（形狀取自 Llama-3、Qwen），沒有 vLLM 或完整模型服務。限制段落在 §7 "Discussion and Future Work"（我下載的是 arXiv 最新版；R02 標 §8，可能是版本差異，未查）。
3. **ParallelKittens（#10）「3.2–5.1× SM」**：R02 標「未在全文核實」。全文有：「register-level … operations require 3.2–5.1× more SMs than TMA」（§2 傳輸機制段，Table 2／Figure 3 附近）。PK 的實驗也全是 **operator 層級微基準**（矩陣 N 至 32768），沒有 LLM 整體推論。
4. **TokenWeave（#15）CUDA graph／eager 的說法只出現在 B200 附錄**（App. C.2、C.3，vLLM 0.14.1）；H100 主實驗（vLLM 0.8.5）有沒有開 CUDA graph，全文找不到。

---

## 1. 逐篇四問（R02 已收）

### #23 FLUX（arXiv 2406.06858，v5）
1. **Hopper 數字**：硬體是 **H800 SXM5**（8 GPU／台，8×400 Gbps IB），不是 H100（§5 Evaluation 開頭）。模型級推論：GPT-3 175B、Llama-2 70B，TP=8；預填 batch 8、seq 2048；解碼 batch 64 或 512（§5.3 段落「In inference, we use vLLM 0.2.1…」）。H800：預填 **1.66×**（對 vLLM）、1.18×（對 TransformerEngine）；解碼 **對 vLLM 無加速**、對 TE 1.76×。算子層級（m=1024–8192 代表預填，m=64、512 代表解碼）：H800 NVLink 平均 overlap efficiency 72%；m=64 的 ReduceScatter 出現 0.95× 退步（§6 Discussion；Figure 13、14）。
2. **比較對象**：vLLM **0.2.1**（§5）。是否開 custom all-reduce、CUDA graph、NVLS：全文皆未說明。NCCL 版本未說明（GEMM 通訊側以 NVSHMEM 2.10.1；CUTLASS 3.4.1；H800 用 NVCC 12.2，§5）。TE 1.4.0 UserBuffer 取「所有設定中最好」。**無「無通訊理想上限」基準**；用自定義 overlap efficiency（§2.3）。基準通訊佔比：Figure 1（8 GPU 推論、各叢集的 TP 通訊占比）只在圖中，文字沒有數字。
3. **拖慢處理**：未量化「通訊被 GEMM 拖慢／GEMM 被通訊拖慢」。處理方式是把通訊寫進 GEMM 的 prologue／epilogue（kernel 融合），以 tile 座標 swizzle 避免 memory controller 競爭（§3，Figure 7）；通訊 tiling 與 GEMM tiling 分開調（§4 tuning 段）。離線 autotune（template 參數）。有討論小 m 變差，但原因歸於 warp 數少與 TMA store 變小（§6），不是 SM 干擾。
4. **自述限制**：「Flux could perform worse than the non-overlapping baseline in a few extremely small m cases」（§6）；解碼 batch 64 有 5 個案例慢於非重疊 vLLM 基準（§6）。

### #21 Ladder-Residual（2501.06589，v5）
1. **Hopper 數字**：8×H100，TP=8，70B，**batch 1**，prompt 1024 / 生成 512（§3.3.1）。Table 2（P2P=1，即正常 NVLink）：預填延遲改善 Ladder **5.78%**、Parallel-attn 5.42%、**無通訊上限 30.54%**；解碼延遲改善 Ladder **23.71%**、上限 **30.00%**；token/s 改善 Ladder 30.79%、上限 42.90%。另掃 TP∈{1,2,4,8}×batch∈{1,4,16,64}（Figure 2）。
2. **比較對象**：不是 vLLM。基準是自己基於 gpt-fast 的 Llama 實作，用 PyTorch compile（"reduce-overhead" 模式，即 CUDA graph）做預填與解碼（§3.2）。NCCL 版本未說明；all-reduce 用 PyTorch async NCCL（獨立 stream）。有「Communication-Free Upper Bound」（§3.3.1），有通訊佔比間接指標（上限 30%）。NVLS：未提。
3. **拖慢處理**：未量測雙向拖慢；全文沒有 contention／interference 討論。方法是改模型架構（殘差路徑讀舊輸入）以製造可重疊的獨立工作，並用 `NCCL_P2P_DISABLE=1` 模擬慢互連（§3.3.1）。
4. **自述限制**：「There is a chance that with longer adaptation… we can obtain a Ladder-Llama with more layers adapted. We leave the further exploration to future work.」（§4.2）。需約 3B tokens 重新適配。

### #20 Comet（2502.19811）
1. **Hopper 數字**：8×H800 單機（另 L20）；MoE 層，訓練導向（Megatron-LM；input token M 例如 16384）；單層 1.96×、端到端 1.71×。**不是 TP 推論**，且 FasterMoE 不支援 TP（R02）。無預填／解碼區分。
2. **比較對象**：Megatron-Cutlass、Megatron-TE、FasterMoE、Tutel；NVSHMEM 2.11、PyTorch 2.4.0（§評估環境）。非 vLLM。無通訊上限基準；MoE 通訊占比 47%（§1，R02 已有）。
3. **拖慢處理**：thread-block 專門化（計算／通訊各自的 block，同 SM 只放一個 block），明說「the warps for communication also interfere with the computation warps within the same thread block」（§3.2.1），故不採同 block 內分 warp。**自適應**：預先編譯多組 kernel（不同通訊 block 數 nc），**離線 profile 後存成 metadata，執行時依輸入 token 長度 M 選用**（§3.2.2）；明確觀察最佳分割點隨 M 改變。未量化「通訊慢多少、GEMM 慢多少」。
4. **自述限制**：R02 §3 已列；我未找到與預填／H100／干擾直接相關的新句子。

### #15 TokenWeave（2505.11329，v5，MLSys 2026）
1. **Hopper 數字**：8×H100 DGX（NVSwitch，NVLS 可用），Llama-3.3-70B、Qwen2.5-72B、Mixtral-8x22B（Mixtral 用 4×H100 或 8×H100，附錄 B 為 4×H100）。
   - **預填單步延遲**：1K 起 **1.16–1.28×**（對 vLLM-Multimem，Figure 13；§5.2.2 "consistently achieves between 1.16× and 1.28× improvement from 1K onward for dense models on 8×H100"）；Figure 2：序列長度 ≥4K 時超過 vLLM-nocomm。MoE（Mixtral）1K、2K 反而有淨負擔，改只開 fused kernel（§5.2.2、Figure 3）。
   - **混合批次吞吐**：chunked prefill 2K（vLLM 0.8.5 預設）、ShareGPT／arXiv／固定長度 trace，約 **1.19×**（Figure 11、§5.2.1）；chunk 1024–8192 掃描，最高 1.26×（Figure 12）。
   - **解碼**：H100 上「decode-only 小批次只用 fused kernel，不做拆分重疊」（§5.2.1）；B200 附錄 C.2 解碼只測 fused-only，延遲改善 1.01–1.05×。H100 無獨立的解碼端到端表。
   - 通訊佔比：vLLM 基準 AllReduce 9–23%、RMSNorm 4–9%（§1 與 Figure 1，8×H100）。fused kernel 對分開執行 1.24–1.39×（Table 1）。
2. **比較對象**：vLLM **0.8.5**（H100）、**0.14.1**（B200 附錄）。基準三種：vLLM-Default（預設 AllReduce）、vLLM-Multimem（NVLS multimem 最佳化 AllReduce）、**vLLM-nocomm**（拿掉所有通訊的反事實上限）（§5 Baselines）。PyTorch 2.6.0。**是否開 custom all-reduce：全文未說明；H100 主實驗是否開 CUDA graph：未說明**。B200 附錄：解碼實驗關掉 torch.compile、對基準與 TokenWeave 都開 CUDA graph；預填實驗因完整方案不支援 CUDA graph 而用 eager（App. C.2、C.3）。另比 TileLink（單層，Figure 14）、NanoFlow（無法同框架，只比「回收的通訊開銷」，NanoFlow 只支援 A100）。
3. **拖慢處理**：**全文找不到「通訊被拖慢多少／GEMM 被拖慢多少」的量測**（對 "slow" 做關鍵字搜尋無命中）。策略是**把通訊本身變得極省 SM**：fused AllReduce–RMSNorm 用 multimem，只用 **2–8 SM**（Figure 10；§4「only 2–8 in our experiments on 8×H100」）；文中論點為 "memory-bandwidth-bound communication and normalization operations do not interfere with each other"（§1）。SM 數固定（附錄提到小批次可用更少 SM，如 4 個；「some overhead … remains hidden as long as its execution time does not exceed that of the overlapped compute kernel」§4.3）。拆分用 wave-aware smart split，**偏移量用解析或離線 profile 決定**（App. A.2）；是否拆分用離線 profile 得出的門檻（H100 dense 1K，Mixtral 4K；App. §切分門檻段）。即：離線調參＋固定門檻，**非執行時依干擾自適應**。有隨 token 數變化的設計（≤1K 不拆、wave 對齊）。
4. **自述限制**：「Under small decode batches, the overhead introduced by splitting tokens into two parts offsets the performance gains obtained from compute-communication overlap.」（App. C.2）；「torch.compile does not support symmetric memory allocation」與完整方案不支援 CUDA graph（App. C.2、C.3）。

### #7 Syncopate（2601.20595）
1. **Hopper 數字**：4／8×H100 單機（NVLink 900 GB/s）；**operator 層級**：AG-GEMM、GEMM-RS、GEMM-AR、head-parallel／sequence-parallel attention、Ring-Attn，形狀取自 Llama-3／Qwen（§評估 Workloads）。平均 1.3×、最高 4.7×（摘要）。**無預填／解碼區分，無整體模型**。
2. **比較對象**：AsyncTP、Flux、Triton-distributed、ThunderKittens、Triton+NCCL（NCCL **2.26.2**，reference [24]）；另把 Domino、Alpa、Mercury 的排程轉成 chunk 表示再比。是否依序執行：Triton+NCCL 為無重疊序列。無 vLLM、無通訊上限基準。
3. **拖慢處理**：未量測雙向拖慢，全文無 contention／interference 命中。緩解手段：**通訊後端三選一（copy engine／TMA／專用或同位 SM 的 load/store）**，並可調通訊所用 SM 數（§5.3，Figure 7、Figure 11 的 SM 與 chunk size 敏感度）；**離線 auto-tune**（chunk 大小、後端、SM 配置、tile 排程）；Motivation 段指出 TMA 約 16 SM 可飽和 NVLink 而 copy engine 不佔 SM，且 SM 數與傳輸量影響吞吐（Figure 2(c)(d)）。
4. **自述限制**：「A full dynamic-shape runtime that automatically manages shape buckets and metadata updates is left to future work.」；「Supporting this setting [multi-node] requires additional backend and runtime support… which we leave to future work.」（§7）。

### #10 ParallelKittens（2511.13940，MLSys 2026）
1. **Hopper 數字**：8×H100 SXM（NVLink4／NVSwitch；§評估「All experiments were conducted using 8×Nvidia H100 80GB SXM」）。operator 層級：AG+GEMM、GEMM+RS、GEMM+AR（Figures 7–9），PK 對非重疊 cuBLAS+NCCL **1.06–1.68×**，對 Triton Distributed 1.07–5.63×，對 Flux 0.97–2.33×。**無模型級預填／解碼**；僅在 §1 引用「prefill 等階段通訊可佔執行時間 >50%」（引自 [3]，非自己量測）。
2. **比較對象**：cuBLAS+NCCL 非重疊、Triton Distributed、Flux、CUTLASS；NCCL 版本於本文中未說明（reference 為 NCCL 2025 網頁）。Flux／CUTLASS 沒有 GEMM-AR，故省略。無 vLLM、無無通訊上限；§4.2 提「在足夠大的 reduction axis 時，PK 的未重疊通訊時間 <1%」。NVLS／in-network reduction：**有用**，GEMM+AR 以少數專用 SM 做一次 in-network all-reduce（§3.1.3）。
3. **拖慢處理**：未量化「慢多少」。分析指出 register 級傳輸比 TMA 多需 3.2–5.1× SM（§2）、約 76 SM 才能飽和（§1）、TMA 約 16 SM。**SM 分割最佳點隨輸入大小改變**：「larger workloads favor more compute SMs, while smaller ones need proportionally more SMs for communication」（§3.1.3，Figure 5）；**「PK allows users to automatically search for the optimal SM allocation at runtime through a unified program template」**（§3.1.3）。
4. **自述限制**：「As this work focuses on intra-node execution, extending these abstractions to inter-node communication remains an important direction for future work.」（§5）；指出 Triton Distributed「originally developed for H800 GPUs, sometimes performs below the non-overlapped baseline on H100s」（§4.1）。

### #19 TileLink（2503.20313，MLSys 2025）與 #17 Triton-distributed（2504.19442）
1. **Hopper 數字**：皆為 **H800**（TileLink 8×H800 與 2 台 16×H800；Triton-distributed 至 64 GPU）。TileLink 端到端 8 個 LLM：**batch 4、seq 8192**（§7.3），單機平均 1.32×、雙機 1.29×，對 **PyTorch**（非 vLLM）。MLP 層：AG+GEMM 1.27×（達 Flux 的 94.5%）、GEMM+RS 1.25×（§7.2 Single Layer Performance，Figure 8）。MoE 層對 vLLM 的融合算子平均 1.51×／1.14×（§7.2）— 這是 operator 對 operator，不是服務層級。**無預填／解碼分開報告**。
2. **比較對象**：cuBLAS+NCCL 非重疊、CUTLASS+NCCL、Flux、Async-TP、vLLM-Op（僅 MoE 融合算子）。NCCL 版本、custom all-reduce、CUDA graph、NVLS：TileLink 未說明；Triton-distributed 有用 multimem 做 low-latency AllGather（§3.4）。無通訊上限。
3. **拖慢處理**：TileLink：**固定 20 SM 給通訊**（§4 範例「we use 20 SMs for communication」）；同時指出通訊用 driver API 會與 host 互相干擾（§3 附近 "host interference"）。Triton-distributed：資源分割為固定設定（§3.8 與 Figure 9：H800 上 GEMM 用 116 SM、intra-node scatter 用 copy engine、本地 reduction 16 SM），並說「driver APIs seem to interfere … so we fused scatter part with producer kernel」（§3.3）；有分散式 autotuner（tile 與通訊 tile 分開調，§3.8）。未量化雙向拖慢；未依訊息大小自適應 SM。
4. **自述限制**：TileLink「Currently, TileLink targets only NVIDIA GPUs.」「We leave this [model-level/pipeline parallelism] for future work.」（§7.4）；Triton-distributed「current NVSHMEM bitcode library doesn't support IBGDA. So we leave this for future work.」（§4）。

### #18 FlashOverlap（2504.19519，EuroSys '26）
1. **Hopper 數字**：**無**。硬體是 A800（TP 至 8）與 RTX 4090（4 張 PCIe）；**端到端（vLLM／Megatron-LM／xDiT）只在 A800**（§6.1.2 與 Table 4）。§2.3 稱 LLM 服務與 T2V 中 GEMM+AR 占端到端 31.6%–42.2%（§2.3.1，A800 profiling）。
2. **比較對象**：基準＝cuBLAS GEMM 與 NCCL 依序執行；另比 Async-TP、VanillaDecomposition、Flux（4090 無 P2P 故不比 Flux）。NCCL **2.19.3**、CUDA 12.1、PyTorch 2.5.1、CUTLASS 3.6.0（§6.1.2）。vLLM 版本與 custom all-reduce／CUDA graph／NVLS：未說明。
3. **拖慢處理**：**有明確的資源爭用建模**：通訊 kernel 設**較高優先權**，通訊用掉的 SM 數確定後，GEMM 的剩餘 SM 數決定其 wave 數與延遲（§4.1.4）；預測 GEMM 延遲「mainly affected by the computational resource contention」、通訊延遲取決於分段；**離線採樣頻寬曲線＋線上 predictive search 選 wave group**（§4.1.3–4.2）；動態任務（LLM 推論）用「預先搜尋代表性 GEMM 尺寸＋最近鄰匹配」（§4.2.2）。通訊頻寬隨資料量變化被明確建模（Figure 8）。未單獨報告通訊／GEMM 各被拖慢幾 %。
4. **自述限制**：「in our implementation, we use multi-processing for communication, which is limited to intra-node parallelism.」（附錄 A.6）；§4.2.3 Tuning Stability：「if the contention pattern changes too fast or is uncontrollable, the predictive search accuracy will degrade」（§4.2.3）。

### #22 Domino（2409.15241）
1. **Hopper 數字**：DGX-H100 1／2／4 台（8–32 H100）；**只有訓練**（Megatron-LM 對照，模型 2.7B–30B）。§1 提到 vLLM 把 TP 當唯一選項，但**無推論實驗**。無預填／解碼。
2. **比較對象**：Megatron-LM；NCCL **2.18**（§評估）。無通訊上限對照，但報告「約 50–70% 通訊被隱藏」。與 vLLM、custom all-reduce、CUDA graph、NVLS 無關。
3. **拖慢處理**：未量測。用 batch 切片（含 hybrid 的 weight 切片）＋CUDA multi-stream 固定 stream 池以降低切換開銷（§4.3.1 MultiStream）；固定設定。
4. **自述限制**：R02 已列（§5、§3）。

### #8 MPK（2512.22219，OSDI）
1. **Hopper 數字**：A100／H100／B200；單 GPU 解碼服務（最大 batch 1–16 掃描，Figure 9）；**多 GPU TP：Qwen3-1.7B on H100，8 H100 對 vLLM／SGLang 1.1–1.4×**（§6.5，Figure 11）。**重疊消融**：Qwen3-1.7B、**4×H100**，細粒度計算－通訊重疊使每次迭代延遲降低 **1.1×**（§6.6，Figure 13）。皆為**解碼服務**，無預填專門數據。
2. **比較對象**：PyTorch（含 CUDA Graphs＋torch.compile）、vLLM、SGLang（各在獨立 venv，AE 附錄）。**vLLM／SGLang 版本、custom all-reduce、NVLS：主文未說明**；§1 稱 kernel-per-operator 系統「即使用 CUDA Graphs」仍有 kernel 切換開銷。無通訊上限基準。
3. **拖慢處理**：未量測雙向拖慢。以 SM 級任務圖與事件驅動讓通訊任務與計算任務共存（NVSHMEM）；MoE 用「hybrid workload balancer」依執行時 token 分布調整（§6.4 Case Study：MoE 與 Figure 10）；SM 分工靜態＋執行時修正。dense 通訊重疊為固定任務圖。
4. **自述限制**：R02 標「未說明」；全文我只找到「…trade-offs is an interesting direction for future work」（§4 附近，與 page 配置相關），無預填／H100／干擾相關自述。

### #13 Lee et al.（2507.03114）
1. **Hopper 數字**：DGX H100（8×H100）、A100×4、MI250、MI210；**訓練**（FSDP 與 pipeline parallelism，GPT-3 與 LLaMA 1.3B–13B，batch 8–64；§III、§IV）。**不是 TP 推論**。重疊使「計算 kernel」平均變慢 18.9%、最高 40.0%（摘要）；H100 FSDP 小模型最多約 5%，大模型（GPT-3 13B、LLaMA 13B）近 40%（§V.A，Figure 4，Takeaway 2；原文「slowdowns nearing 40%」可能含其他 GPU，我未逐圖核對哪張卡）。有一組 **GEMM 與 1 GB all-reduce 同時執行** 的微基準與功耗（§V.C，Takeaway 6）。
2. **比較對象**：NCCL（Nvidia）／RCCL（AMD）；理想上限＝**無通訊的計算時間**（把重疊時多出的計算時間扣掉求得「E2E ideal」，§IV.D）；NCCL／PyTorch 版本在表格中，我未核對。
3. **拖慢處理**：**量了單向：計算被拖慢**；通訊被拖慢沒有單獨數字（全文以計算 slowdown 與 overlap ratio 為主）。有討論隨模型大小、batch、並行策略、精度（FP16/TF32 可增減爭用）、功耗上限變化（Takeaways 1–6）。**無緩解方法**（純特性量測）；結論建議 "workload-specific optimization strategies"。
4. **自述限制**：我在結論（§VII）找到：「their benefits are constrained by resource contention, causing higher power consumption and performance slowdowns」，無直接的「未來工作」句；未說明預填／解碼。

### R04：Resource-aware Overlap（2606.09200）
1. **Hopper 數字**：**4×H100**（132 SM、L2 50 MB；Table 1）；工作負載是 **cb-ar／mb-ar／cb-a2a／mb-a2a**：GEMM 8192×8192×8192 或 8192×57344×8192，集體通訊資料量固定 896 MB（Table 1）。總時間最多減少 25.5%，**這個數字是 A40 上的 cb-ar**（§4.2 Priority-aware，原文 "reduced by up to 25.5% for cb-ar on A40"）；H100 的個別數字在 Figure 3(c)，文字沒有列。重疊率最高約 90%（Figure 4）。**無模型級預填／解碼**。
2. **比較對象**：基準＝同一份程式不用優先權（共享記憶體占用率調 GEMM 駐留 block）的「baseline overlap」；NCCL 版本與 vLLM：未說明（本文不涉推論引擎）。無通訊上限，但有 TimeRatio（重疊時間／循序時間）。
3. **拖慢處理**：核心就是此題。用每 block 共享記憶體配置限制 GEMM 常駐 block 數，並以 `cudaStreamCreateWithPriority` 讓通訊 stream 高優先權（§3.3）；明說基準「communication progress becomes fragile under resource contention」（§4.2）。**未報告「通訊慢多少、GEMM 慢多少」的分開數字**（以總時間與重疊率表示）。是**固定設定（掃 thread-block 數與 tile 大小），非自適應**。有分析 tile 大小（opt1：TILEk=32；opt2：TILEk=64）的交互作用，MI250X 上優化版有時反而變差（§4.2）。
4. **自述限制**：「Future work includes exploring adaptive runtime policies that automatically tune occupancy and priority settings across diverse workloads, and conducting end-to-end distributed training performance evaluations.」（§5 Conclusion）；tile 設計空間「we leave for future work」（§4.3）。

### R04：Weave（2609.21483）
1. **Hopper 數字**：**4×H100 SXM、EP=4**，六個 MoE 模型，序列 2k／4k／8k（batch 1）**預填**；MoE 層幾何平均 2.89×、端到端 1.33×（Abstract，§5）。**不是 TP，不是 8 卡**；**無解碼**。
2. **比較對象**：DeepEP（固定 20 SM 給通訊）、Comet、ParallelKittens、Triton-distributed 等（§5）；NCCL **2.21.5**、NVSHMEM 3.6.5、driver 575.57.08（§5.1）。無 vLLM、無通訊上限基準（本文未見）。
3. **拖慢處理**：**以 SM 分割作為主要變數並量測其最佳點隨層與 GPU 變動**：同一 MoE 層內「per-GPU optimal c* ranges from 12 to 36」，DeepEP 固定 c=20 在瓶頸 GPU 上慢 1.33×、Comet c=18 慢 1.18×（§2.3.1，Figure 5）。方法：megakernel 內**執行時成本模型**（路由結果出來後才知道通訊與計算量），聯合選通訊 SM 數 c 與 chunk 數 K（§3.3）；成本模型選擇準確度落差 8.2%（§5.5，Figure 12）。**這是目前找到唯一「依負載在線自適應 SM 分工」的 Hopper 結果，但限 MoE EP。** 未單獨報告通訊／GEMM 被拖慢幅度。
4. **自述限制**：「Due to hardware constraints, Weave is currently validated on a single 4×H100 …」，多機與更大配置留待未來（§Limitations and future work）。

### #6 Lagom（2602.20656）
1. **Hopper 數字**：**無**；2 台×8 **A40**，**訓練**（FSDP、TP/EP）。
2. **比較對象**：NCCL v2.18.3、AutoCCL；CUDA 12.8。1.07–1.33×（對 NCCL）。
3. **拖慢處理**：把通訊參數（含 channel 數 NC、chunk size）與計算爭用放進統一成本模型，並指出調整 NC 會「prolong both communication and computation time」（§3 動機圖，2 AllReduce 與 7 MatMul 同時）；**在線 coordinate descent 搜尋**（Lagom 在 Phi-2-2B 的 FSDP 訓練約 33 次迭代收斂，§評估）。
4. **自述限制**：未說明。

---

## 2. 逐篇四問（R02 未收、本次新增）

### NanoFlow（arXiv 2408.12757，v2，Zhu et al.，2024-08-22；已開 arXiv abs 並讀全文）
1. **Hopper 數字**：**無**。所有評估在 **8×A100 80GB SXM**（LLaMA-3-8B 為單張 A100）（§6 評估設定段「All evaluations are done on 8×A100 80GB SXM GPUs…」）。理論最優吞吐成本模型有列 H100／H200 的參數表（Table 1 與 §3），但**沒有實測**。預填／解碼：用 prefill-only（input 512／output 0）與 decode-heavy（512／1024）：只重疊網路密集與計算密集 kernel 得 **1.07×**，再加記憶體密集（decode attention）重疊得 **1.17×**（相對非重疊基準，§6.4，Figure 9）；整體離線吞吐是 vLLM 的 **1.91×（平均，最高 2.66×）**，達理論最優 50%–72%（摘要、§6.2）。
2. **比較對象**：vLLM（commit **38c4b7e**，2024；reference [52]，以 commit 而非版本號給出）、DeepSpeed-FastGen、TensorRT-LLM。custom all-reduce、CUDA graph、NVLS：未說明。**有理論最優吞吐上限**（§3.5，每 GPU 1857 tokens/s/GPU 例）；現有系統只達最優的 22.0%／22.9%／37.8%（§3.5 "Gap to Optimal Throughput"）。基準通訊佔比：以成本模型表示，非單獨百分比。
3. **拖慢處理**：**有，且是核心**。**成對（pairwise）干擾剖析**：以 GEMM 效能當資源代理 R，量測與之同時跑的 GEMV 或網路 kernel 的效能 P（Figure 5、Table 3），「exchange rate」映射；限制 GEMV／網路 kernel 的 thread block 數在 8–128 之間（步長 8）縮小剖析空間；**離線**由 auto-search 先排「無干擾」管線，再用干擾剖析重新規劃（§4.1.1、§4.1.2 第二階段）。假設成對映射在三種 kernel 同時重疊時仍成立（§4.1.1）。實際效果：干擾使計算利用率低於理論（68.5% 平均，§6.6）。是離線調參；形狀（dense batch size 2048 等）預先決定。未報告通訊本身慢幾 %。
4. **自述限制**：我在全文找到「NanoFlow focuses on pairwise interference analysis… assumes that the R to P mapping profiled with pairwise interference holds when overlapping three kernels」（§4.1.1）；實作只支援 NVIDIA A100（TokenWeave §5.2.3 亦指出，見 #15）。未找到明確的「Future work」段。

### ISO（arXiv 2409.11155，v1，Xiao & Su，Baichuan，2024-09-04；已開 abs 並讀全文）
1. **Hopper 數字**：**無**。硬體為 **4090（4／8 卡）與 A800（4／8 卡）**，30B（MHA）與 70B（GQA）密集模型，**僅預填**（Table 1；§4）；int8 權重／KV cache／GEMM，4090 通訊用 int8。預填時間減少：4090 約 35%、A800 約 15%（摘要、§4）。**提到「A 或 H 系列計算占比常 >75%」只是推論，並無 H 系列實測**（§3.2 Computation dominates）。
2. **比較對象**：同一框架下「只差有無 ISO」的基準（§4.1）；未說明 NCCL 版本、vLLM、custom all-reduce、CUDA graph。也試過 Flux 風格的矩陣計算與通訊重疊：A800 上只有 2%–5%，4090 為負（§4 結果段）。無無通訊上限基準；§3.2 報告計算占比 >75%（A、H 類 GPU）與通訊占比約 75%→50%（4090，int8 後）。
3. **拖慢處理**：**有一個量化數字：「NCCL communication tends to consume extra SMs, thereby prolonging the computation time. The impact on A800 ranges from 15% to 20%, whereas it is negligible on the 4090」**（§3.2）。處理：把矩陣計算切成幾段、多次 kernel 啟動，使通訊結束後矩陣計算可用滿 SM；固定設計，非自適應。通訊被拖慢：未量。
4. **自述限制**：「The concurrent execution of computation and communication can hinder gemm operations, thus limiting the potential improvements.」（§4 結果段）；未實作 request-level overlap；結論自述 "preliminary optimization efforts"。

### PyTorch Async-TP（無獨立論文；官方文件與部落格）
- **來源（已用 WebFetch 讀取，內容由 fetch 工具的小模型摘要，非我逐字全文核對，引文為其摘錄，標「二手摘要」）**：
  - 「[Distributed w/ TorchTitan] Introducing Async Tensor Parallelism in PyTorch」，Yifu Wang 等，2024-09-12，https://discuss.pytorch.org/t/distributed-w-torchtitan-introducing-async-tensor-parallelism-in-pytorch/209487
  - 「PyTorch SymmetricMemory: Harnessing NVLink Programmability with Ease」，yifuwang 等，2025-02-19，https://dev-discuss.pytorch.org/t/pytorch-symmetricmemory-harnessing-nvlink-programmability-with-ease/2798
  - TorchTitan 論文 arXiv 2410.06511（已確認存在，v3；**未讀全文，其中 async-TP 內容未核實**）。
1. **Hopper 數字**：**訓練**，64×H100（每台 8 GPU＋NVSwitch，貼文稱為 HBM2e、較低 TDP 的非標準 H100）；Llama3 8B 前向最高約 29%、端到端約 8%；70B 前向最高約 20%、端到端約 8%（皆 torch.compile 開啟）。TP 度、序列長度、batch 未說明。**推論／預填／解碼：沒有數字**，只說「正在探索更細粒度管線來改善推論這類小問題規模」。
2. **比較對象**：torch.compile 的非 async-TP 版本；是否為 NCCL 基準未明說。無 vLLM；無無通訊上限；NVLS：SymmetricMemory 貼文列有 `multimem_all_reduce`、`multimem_all_gather`，稱 NVSwitch V3+ 時 NVLS 集體「likely strictly better」（貼文原文，二手摘要）。
3. **拖慢處理**：貼文**明確指出 SM 爭用**：NCCL send/recv 用 SM 搬資料，讓重疊的 matmul 少了 SM，且「慢多少可能超過通訊 kernel 佔用資源的比例」；對策是改用 **copy engine（CUDA P2P）搬連續資料，不需 SM**，並用兩條對稱 stream 輪流；SymmetricMemory 貼文稱低爭用實作「leave more SM resources for ambient matmul kernels, at the cost of being slightly slower」。**未量化雙向拖慢；固定設定（非自適應）**。stream 優先權：未提。
4. **自述限制**：「Currently, async-TP in PyTorch performs best with large matmul operations」；「relies on NVSwitch for optimal performance」；「currently only works for intra-node setups」（皆 2024-09 貼文，二手摘要）。另「Decomposition-based async-TP has overhead at small problem sizes」（SymmetricMemory 貼文）。

---

## 3. 新增論文（用查詢找到，2024 年至今，與推論 TP 重疊／干擾相關）

`citation_graph` 不可用，以下由 arXiv 關鍵字查詢找到（查詢編號見 §6）。逐篇打開 abs 並（有 PDF 者）讀全文。

| arXiv | 標題／日期 | 與我們問題的關係 | Hopper 8 卡數字 | 重疊干擾 | 備註 |
|---|---|---|---|---|---|
| 2611.12362 | *Release Sooner, Expose Less: Offloading MoE Dispatch in NCCL-EP under Dual-Batch Overlap*（Yin et al.，2026-11-09，v2） | MoE dispatch 在 DBO（兩個 microbatch 交錯）下 send 持有 GPU 的時間仍暴露；比較 GDA-KI 預設與 Free Flow | **無**：GB300 NVL72（aarch64、CUDA 13.0）；vLLM 0.30.1rc0 fork，分離式（2P2D）Qwen3-30B-A3B，TPOT 降 15–25%（對 NCCL-EP 預設）、16–42%（對 DeepEP v1）（摘要；§評估） | 不量 GEMM 對通訊；關鍵發現是「32–96 µs 的獨立計算」可遮蔽延遲進度（摘要，§ 延遲實驗） | 只看過摘要與關鍵字 grep，未逐節讀 |
| 2606.01927 | *Scaling LLM Inference Beyond Amdahl's Limits…*（Albireo；Zhao, He, Xu，2026-06-01） | **重疊對象是排程與 I/O 與計算，不是 TP all-reduce**；談 TP 度的 Amdahl 限制 | 8×H100（NVLink，H100N），vLLM **0.11.2**，Qwen-2.5-32B、t=4、batch 128 的任務時間剖析（§3）；對 vLLM 最高 1.9× 吞吐（摘要） | 未談 GEMM／通訊爭用 | 與 TP 通訊重疊問題只有間接關係（列為「非通訊重疊」） |
| 2610.05305 | *Characterizing Parallelism Strategies in LLM Inference: Fundamental Compute-Communication Trade-offs*（Mirzaei & Mitra，Dell，2026-10-04） | 分析模型：TP／PP 在預填與解碼的差別；Nsight Systems 分解 GEMM／Attention／NCCL／Other | **無**：8×**A100** SXM4，vLLM **v0.15.0**，Qwen2.5-14B／32B；TP8 時 NCCL 約占 TTFT **近 40%**（§3 結果）；kernel 列有 `vllm::cross_device_reduce_1stage`（即 vLLM custom all-reduce 實際被用到） | 非重疊論文，無緩解 | 可作為「A100 上 vLLM 預填通訊占比」旁證 |
| 2609.12923 | *Dissecting GPU Utilization for LLM Inference on Nvidia Hopper*（Siavashi et al.，2026-09-11） | 單卡 H100 NVL、vLLM＋FlashAttention-3＋cuBLASLt 的預填（冷／暖）與解碼的 GEMM 利用率剖析 | 僅單卡，無 TP 通訊（**僅摘要**） | 無 | 與重疊無關；提供「解碼小 batch GMMA 64 列片段填充低」的背景 |
| 2605.05628 | CAIS（R02 #4） | 模擬 | 模擬 H100 SuperPOD | 見 R02 | 已在 R02 |
| 2611.18075 | Zero-SM NCCL（He et al.，2026-11-13） | NCCL 無 SM 的集體（R04 已收） | — | — | 已在 R04 |

**2026-11-16 之後的新論文**：查詢 Q4（`date_from` 2026-11-10，題目／摘要含 overlap／fusion＋communication＋GPU／LLM）共 13 筆，只有 2611.23512 DexAttention（2026-11-17，分散式稀疏 attention，**16 GPU 訓練**，persistent kernel 依賴與重疊；與推論 TP 重疊無直接關係，僅摘要）落在 11-16 之後且與重疊有關；其餘與主題無關。未找到 11-16 後專談推論 TP 重疊或干擾的新論文（**找不到不代表不存在**）。

**未收進的候選**（只看標題或摘要、與主題距離遠）：2608.09160 SwiftQK（TP 下 query-key normalization 的通訊，僅標題）、2601.08800 MixServe（MoE 混合並行融合通訊演算法，僅標題）、2604.19654 FEPLB（copy engine 做 MoE 負載平衡，訓練，僅標題）、2605.05607（MoE in-switch，硬體）、2603.28239（交換器架構，硬體）。

---

## 4. 空白格表

列＝情境；欄＝四個條件。格內填論文編號（#N 為 R02_T2 編號；arXiv 編號為新增），並附條件。查詢編號指 §6 的 Q#。

| 情境 | 「Hopper 8 卡上有實測」 | 「基準含 vLLM custom all-reduce＋CUDA graph」 | 「量了雙向拖慢」 | 「拖慢隨大小自適應」 |
|---|---|---|---|---|
| **預填 TP** | **#15 TokenWeave**（8×H100，Figure 13：1.16–1.28×）；**#23 FLUX**（8×**H800**，對 vLLM 0.2.1：1.66×）；**#21 Ladder**（8×H100，batch 1，預填改善 5.78%，上限 30.54%）；**#19 TileLink**（8×H800，batch 4×seq 8192，對 PyTorch，非 vLLM）；operator 層級：**#10 PK、#7 Syncopate**（8×H100）、#17（H800） | **未找到同時明載兩者的論文**。#15：基準為 vLLM 0.8.5 的 Default／Multimem，custom all-reduce 與 H100 的 CUDA graph 未明載（B200 附錄：解碼用 CUDA graph、預填 eager）。#23：vLLM 0.2.1，皆未說明。Q3、Q5、Q8 | **只量單向、且皆非 Hopper-TP-推論**：#13 Lee（訓練，DGX H100，計算慢 18.9%／最高 40.0%，通訊未單獨量）；2409.11155 ISO（A800 預填，計算慢 15–20%）；2408.12757 NanoFlow（A100，成對 GEMM／網路 kernel 剖析）；2606.09200（4×H100 微基準，只報總時間與重疊率）。**Hopper 8 卡預填 TP 的兩向拖慢：未找到**。Q2、Q3 | **#7 Syncopate**（離線 auto-tune，operator）；**#10 PK**（§3.1.3：SM 分割最佳點隨大小變，執行時搜尋，operator）；**#18 FlashOverlap**（離線＋線上 predictor，A800 非 Hopper）；**#15 TokenWeave**（離線門檻＋wave 對齊 split，非干擾自適應）。**Hopper 預填 TP 的端到端執行時自適應：未找到**。Q1、Q3 |
| **解碼 TP** | **#21 Ladder**（8×H100，batch 1：解碼改善 23.71%，上限 30.00%）；**#8 MPK**（8×H100，Qwen3-1.7B，1.1–1.4× vs vLLM／SGLang；4×H100 重疊消融 1.1×）；**#23 FLUX**（8×H800：對 vLLM 無加速，batch 64 有 5 個案例慢於非重疊）；#15 TokenWeave 僅 B200 附錄的 fused-only（1.01–1.05×） | **#21 Ladder：有 CUDA graph（PyTorch compile）但非 vLLM**；#8 MPK：PyTorch 基準有 CUDA Graphs＋torch.compile，vLLM／SGLang 設定未明載。**無論文同時明載 vLLM＋custom all-reduce＋CUDA graph**。Q5、Q9 | **未找到**（Hopper 解碼場景） | **未找到** |
| **混合批次** | **#15 TokenWeave**（8×H100，chunked prefill 2K，約 1.19× 吞吐，Figure 11；1K 以上混合批次才開拆分） | 同上列（#15 未明載） | #15：**未量**；2408.12757 NanoFlow（A100，混合預填／解碼，pairwise 剖析） | NanoFlow（離線剖析後固定管線，A100）；#15（以 token 數門檻切換是否拆分，離線決定） |
| **MoE** | **#20 Comet**（8×H800，訓練）；**2609.21483 Weave**（4×**H100**，EP=4，預填 2k–8k token）；#14 FlashMoE（R02，8×H100）；#15 TokenWeave 的 Mixtral-8x22B（8×H100 TP 預填，1K、2K 為負） | **未找到**（2611.12362 的 vLLM 為 0.30.1rc0 fork、分離式、GB300，非 Hopper） | **2609.21483 Weave**：量了「通訊 SM 數」對層延遲的影響（最佳 c* 12–36，4×H100），但不是通訊與 GEMM 分開被拖慢幾 %；#20 Comet：只引述 warp 干擾論點 | **Weave（執行時成本模型，4×H100 EP）**、**Comet（離線預編譯，依 M 選用，H800）** |

**查詢編號對照**：見 §6。**限制**：以下情況都可能存在漏網：(1) `citation_graph` 未能執行；(2) 查詢只含 arXiv，未含 MLSys、OSDI、SOSP、EuroSys 等僅出現在會議頁而 arXiv 無的論文（例：R02 已記 Mercury SOSP'25 無 arXiv 版）；(3) 工業部落格與框架文件（vLLM／SGLang／TensorRT-LLM 的 DBO、async-TP、sequence parallel 實作）不在本分工範圍，B 分工處理。

---

## 5. 作者自述限制彙整（只列與「預填」「H100」「干擾」「自動調整」有關，引文皆已核對原文）

- **預填**：#15「Under small decode batches, the overhead… offsets the performance gains」（App. C.2）；#23「Flux could perform worse than the non-overlapping baseline in a few extremely small m cases」（§6）；2409.11155「The concurrent execution of computation and communication can hinder gemm operations」（§4）。
- **H100**：#10「Triton Distributed, originally developed for H800 GPUs, sometimes performs below the non-overlapped baseline on H100s」（§4.1）；2609.21483「Weave is currently validated on a single 4×H100」。
- **干擾**：#20「the warps for communication also interfere with the computation warps within the same thread block」（§3.2.1）；#18「if the contention pattern changes too fast or is uncontrollable, the predictive search accuracy will degrade」（§4.2.3）；Async-TP 貼文（二手摘要）「NCCL send/recv kernels utilize SMs to move data across NVLink」。
- **自動調整**：2606.09200「Future work includes exploring adaptive runtime policies that automatically tune occupancy and priority settings」（§5）；R02 #3「runtime-adaptive SM partition selector… future work」；#16「We leave automatic optimization for future work」；#7「A full dynamic-shape runtime… is left to future work」（§7）；#10「PK allows users to automatically search for the optimal SM allocation at runtime」是已有能力（§3.1.3），非限制。

---

## 6. 搜尋與核實紀錄

**arXiv `search_papers`（均 `abstract_mode: none`）**：
- Q1：`abs:"tensor parallel" AND abs:overlap AND (abs:inference OR abs:serving)`，2024-06 起，依日期：16 筆（採 ISO 2409.11155、Albireo 2606.01927；Kraken 2408.07802、Helix 2507.07120 為架構／切分而非重疊，**未讀**）。
- Q2：`(abs:"all-reduce" OR abs:"communication") AND (abs:interference OR abs:contention OR abs:slowdown) AND abs:GEMM AND abs:overlap`：2 筆（Zero-SM、Argus 2609.12299；Argus 為量測框架，**僅標題，未讀**）。
- Q3：`abs:"nano-batch" OR abs:"micro-batch overlap" OR abs:"dual batch overlap" OR abs:"compute-communication overlap" AND abs:LLM AND abs:prefill`：5 筆（採 2611.12362、NanoFlow、ISO）。
- Q4：2026-11-10 起，`(overlap OR overlapping OR fusion) AND (communication OR all-reduce OR collective) AND (GPU OR LLM)`：13 筆（含 11-16 後新論文 2611.23512；其餘與主題無關）。
- Q5：`abs:"all-reduce" AND (tensor parallelism…) AND (decode OR prefill OR serving) AND (fused OR fusion OR overlap)`，2025-06 起：2 筆（Runway 2611.05642＝R02 #1、MixServe 2601.08800 僅標題）。
- Q6：`abs:H100 AND abs:"communication" AND abs:"GEMM" AND (concurrent OR co-running OR interference OR "SM partition")`，2025 起：1 筆（FEPLB，訓練，僅標題）。
- Q7：`symmetric memory OR async tensor parallel OR two-batch overlap OR multimem OR NVLS`，2025 起：26 筆（總數 26）；與主題相關的皆為已知（TokenWeave、CAIS、Zero-SM、NVSHMEM 分析 2606.05951、Every Microsecond Matters 2607.16100、T-CCL 2610.07098、MSCCL++ 2504.09014 等，後數篇屬集體通訊本身，R04／R02 已涵蓋，**本次未再讀**）；2609.12923 為單卡剖析。
- Q8：`abs:"tensor parallelism" AND (communication-computation …) AND abs:inference`，2025-06 起：4 筆（採 2610.05305）。
- Q9：`all-reduce AND (low-latency OR one-shot OR fused kernel) AND (LLM inference OR serving) AND NVLink`，2025 起：1 筆（TokenWeave）。
- Q10：`(fused OR fusion) AND GEMM AND (reduce-scatter OR all-gather OR all-reduce) AND Hopper`，2025 起：**0 筆**。
- Q11：`ti:torchtitan`：1 筆（2410.06511）。
- `WebSearch`（standard）：`PyTorch async tensor parallel torchtitan "async TP" symmetric memory…`：得兩份 PyTorch 論壇貼文網址（已以 `WebFetch` 讀取）。
- `citation_graph`：對 2505.11329、2406.06858 各一次，皆 `RATE_LIMITED`；改用 curl 對 Semantic Scholar 直連（TokenWeave、NanoFlow）皆 HTTP 429。**因此「從 FLUX、NanoFlow、TokenWeave、Lee et al. 往後找」沒有完成**。

**新下載 PDF（皆確認非 0 位元組）**：2406.06858、2408.12757、2409.11155、2409.15241、2501.06589、2503.20313、2504.19442、2504.19519、2511.13940、2512.22219、2601.20595、2602.20656、2606.01927、2611.12362（位於 `/home/allen0117/comm_opt/papers/`）。已有的 2505.11329、2507.03114、2606.09200、2609.21483、2502.19811、2610.05305 沿用現有檔案。

**未能核實或只部分核實的項目**：
1. `citation_graph` 與 Semantic Scholar 皆限流 → 「citation 往後找」未做。
2. FLUX 的 vLLM 0.2.1 是否含 custom all-reduce、CUDA graph：全文未說明；Figure 1 的各叢集通訊占比只在圖中，未取得數字。
3. TokenWeave H100 主實驗是否開 CUDA graph、vLLM 預設 all-reduce 是否為 custom all-reduce：全文未說明（只有 B200 附錄提到）。TokenWeave 的 GitHub 程式碼未讀（屬 B 分工）。
4. NanoFlow 的 vLLM 只有 commit 38c4b7e，我沒有把 commit 對回版本號。
5. MPK 多 GPU 實驗中 vLLM／SGLang 的版本與設定、MPK 通訊是否使用 NVLS：主文未說明；AE 附錄有腳本路徑，未讀腳本。
6. Lee et al. 的 H100 單獨數字（Figure 4）：圖中，文字只給「GPT-3 XL 在 H100 FSDP 最高約 5%」，大模型「近 40%」未能對應到哪張卡。
7. PyTorch Async-TP 兩份來源為 WebFetch 小模型的摘要，非逐字全文；TorchTitan 論文 2410.06511 內容未讀。
8. 2611.12362、2606.01927、2609.12923、2611.23512、2608.09160、2601.08800、2604.19654 只讀摘要或標題（2611.12362、2606.01927 另做了關鍵字 grep，未逐節讀）。
9. Domino 全文未含推論實驗，此點以關鍵字 grep 判斷（"inference" 只出現在背景敘述），非逐節確認。
10. Mercury（SOSP'25）仍無 arXiv 版可核實（R02 §7 已記）。
11. 未讀 vLLM／SGLang／TensorRT-LLM 官方文件或部落格對 DBO、async-TP 的說明（B 分工）。
