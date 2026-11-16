# Opus 讀報告筆記（Nano5 選方向用，進行中）

> Opus 自己的工作筆記：從 R02_T1–T5、R03 讀出的重點與初步判斷。判斷標「判斷」，事實附報告與論文編號。關鍵論點在用於決策前要回原文抽查（守則第八節 5）。

> **讀法（2026-11-17）**：2026-11-16 深夜兩個 Claude 對話同時執行 S04，各自寫了 R04 的讀後段落，下面有重複。**以最後一節「整合判斷」為準。**

## T1 單機集體通訊（R02_T1，27 篇）

- 小訊息低延遲 collective 是熱點，而且大廠／ETH 在做：
  - **Every Microsecond Matters**（T1-P3，arXiv 2607.16100，NVIDIA＋ETH，GB200）：barrier-free、symmetric memory、multicast、LL128 atomic；4×GB200 小 AllReduce 11.0→2.37 µs；vLLM ITL −7～13%；自述 kernel 選擇靠經驗（無效能模型）
  - **TokenWeave**（T1-P12／T2-#15，MLSys'26，8×H100）：融合 AllReduce＋RMSNorm（multimem），只用 2–8 SM；小 decode batch 的切分開銷抵消重疊；不支援 CUDA Graph
  - MSCCL++（ASPLOS'26）：H100 SwitchChannel（multimem）AllReduce 比 MemoryChannel 最多 +56%
  - 判斷：A100 時期的「融合 all-reduce＋norm、去掉 barrier」在 H100 上已有 TokenWeave、Every-µs、flash-comm 三篇高度重疊 → **不適合再當主題**（與使用者「不被過去綁住」一致）
- **NCCL 2.28+ device API**（LSA／Multimem／GIN）＋ symmetric memory 讓自製 kernel 容易很多（T1-P13）；我們的 NCCL 2.29.7 ≥ 2.28
- **SM-free（copy engine）通訊**：Zero-SM NCCL（T1-P1，2026-11-13，GB200／B200）不支援 reduction；H100 上的 copy engine 表現沒報告；ParallelKittens 量到 H100 copy engine 1 GB 達 368.8 GB/s（82%），且 register 指令需 3.2–5.1× SM 才能飽和 NVLink
- **缺口（事實層面）**：
  - 在 H100 單機把 NVLS 開／關當唯一變因系統量測：沒有找到（T1 §5、T2 §5、T5 同）
  - 調參與 kernel 選擇缺效能模型（T1 §3-C：P3、P18、P19、P22）
  - 合成類（TACCL、TE-CCL、SyCCL、ForestColl、HeteCCL）多以頻寬為主，延遲導向少

## T2 計算通訊重疊與融合（R02_T2，25 篇）

- 主流：單 kernel tile 級融合（Flux、Comet、ParallelKittens、FlashMoE、mKernel、Syncopate）、compiler（Triton-distributed、TileLink、Syncopate、MPK mega-kernel）
- **重疊的代價**：Lee et al.（2507.03114，DGX H100）量到重疊使計算平均變慢 18.9%、最高 40%；Comet：通訊 warp 干擾計算 warp
- **跨機是普遍的未來工作**（Syncopate、ParallelKittens、FlashOverlap、mKernel 的自適應 SM 分配只在單機測）
- **decode 小 batch 重疊無效**（TokenWeave、Flux 小 m 0.95×）
- Runway（2611.05642）：Hopper 上 `multimem.ld_reduce` 沒有 FP8 形式；NVLS multicast buffer 綁定導致 OOM
- 判斷：「SM 分配給通訊 vs 計算」與「跨機的 tile 級重疊」是反覆出現的缺口；mKernel（2×8 H200）已開始做跨機＋動態 SM 分配

## T3 MoE 專家平行（R02_T3，29 項）

- 非常擁擠：DeepEP（事實參考實作）、NVIDIA Hybrid-EP、NCCL EP（GIN）、UCCL-EP、fabric-lib、FlashMoE、Comet、MegaScale-MoE，加上一大批負載平衡論文
- 反覆出現的限制：通訊佔 SM（DeepSeek-V3 最多 20 SM／H800；DeepEP 不支援 zero-SM RDMA EP）；負載不均；規模受限靠模擬
- **環境限制**：DeepEP 最新 README 要求 CUDA 13.1+（R02_T3 §8-1，未驗證）→ 驅動 550 可能跑不了最新版，公平比較困難
- 判斷：除非有很具體的缺口，否則不建議以 MoE EP 為主線

## T4 跨機器（R02_T4，29 項）

- **SHARP（InfiniBand 交換器內加總）在 2024 年後沒有找到獨立的 LLM 量測**（R02_T4 §1）；R2CC（4×8 H100、IB 400G）明說測試機沒開 SHARP、留待未來
- 小訊息、延遲導向的跨機 collective 是共同缺口：NVRAR 無 H100 資料（A100／GH200、Slingshot）；PCCL 想在 IB 上評估；ForestColl「throughput over latency」；OptCC 延遲敏感情境留待未來
- 與我們規模相符的實機研究：R2CC（4×8 H100 IB）、NIMBLE（H100 4 rail NDR）、Gupta 剖析（4×8 H100）
- 判斷：Nano5 的 4 台×8 H100＋8×NDR 正好是「小型 H100 IB 叢集」，若交換器支援 SHARP，**NVLS（機內）＋SHARP（機間）兩層交換器加總**可能是少有人實測的區域 → 要靠 M20／M23 確認 SHARP 是否可用（`nvhpc-hpcx-cuda12/24.7` 模組內可能含 SHARP 函式庫）

## T5 推論服務（R02_T5，24＋2）

- 子題：PD 分離的 KV 傳輸（KV 傳輸是不是瓶頸，文獻結論互相矛盾，條件不同）；TP decode 小訊息（NVRAR、SiFAR 8×H200 multimem、SCIN、TokenWeave）；長上下文 CP；MoE decode；RL 權重同步
- NVLS 開／關的 A/B 數字：沒有任何一篇報告
- 判斷：TP decode all-reduce 在單機已擠（TokenWeave、SiFAR、Every-µs、flash-comm）；**跨機 TP decode（大模型放不下單機時）在 H100＋IB 上的小訊息延遲**較少人做（NVRAR 只有 A100／GH200）

## 初步候選方向（判斷，等 R03、M20 後再定）

| 代號 | 方向 | 支持的缺口 | 風險 |
|---|---|---|---|
| A | 兩層交換器加總：機內 NVLS＋機間 IB SHARP，用於小型 H100 叢集的 LLM collective（含延遲導向） | T4：2024 後無 SHARP 獨立量測；R2CC 沒開 SHARP；T1–T5 都沒有 NVLS 開關 A/B | **取決於 Nano5 交換器是否開 SHARP**（M20／M23）；無 root |
| B | 跨機 TP decode 的小訊息 all-reduce（H100＋IB） | NVRAR 無 H100；Every-µs 是 GB200 全 NVLink；PCCL 未在 IB | GPU 直發 RDMA（IBGDA）可能需要驅動設定（無 root）；與 NVRAR 重疊 |
| C | 通訊對計算的「SM 稅」：H100 上各種 collective 實作（ring／NVLS／symmetric／copy engine）佔多少 SM、拖慢 GEMM 多少，並據此排程 | Lee et al. 18.9%／40%；Comet、DeepEP 的 SM 限制；mKernel 自適應 SM 只在單機 | Zero-SM NCCL（2026-11-13）直接相關（但為 GB200、無 reduction） |
| D | 有效能模型的 collective 選擇（NVSwitch＋NVLS＋IB） | T1 §3-C 多篇自述「靠經驗選 kernel」 | 容易變成小改進；NCCL 已有 tuner plugin |
| E | KV cache 傳輸在 H100＋IB 上是否真的是瓶頸（PD 分離） | T5 §3-F 文獻結論矛盾 | 系統整合重；偏量測 |

- 不建議：單機 all-reduce＋norm 融合（已擠）、MoE EP 主線（擁擠）

## R03 主流系統原始碼（8 個系統，2026-11-16 的 release tag）與 Opus 抽查

- vLLM v0.34.0 的 all-reduce 選擇順序：NCCL symmetric（預設關）→ FlashInfer AR（cc90、8 卡 ≤0.5 MB）→ custom AR（<256 KB one-stage，否則 two-stage）→ torch symm_mem multimem → NCCL（R03）。**Opus 抽查**：`cuda_communicator.py` 389–464 的判斷順序與報告一致；`all_reduce_utils.py` 有依 world size 的門檻表（例如 symm mem 8 卡 256 KB）——細節門檻依架構不同，引用前要再對一次
- SGLang：CustomAllReduceV2（8 卡 sm90：≤128 KB one-shot、≤128 MB two-shot pull，multicast 0–128 MB）；TRT-LLM：AUTO 查表（小 ONESHOT、中 TWOSHOT、大 NCCL）
- EP 預設多半**不是**真正的 all-to-all：vLLM 預設 allgather＋reducescatter；SGLang `--moe-a2a-backend none`；Megatron 預設 allgather
- **版本落差**：我們裝的是 vLLM 0.32.0（cu129），原始碼調查是 v0.34.0；做實驗前要對 0.32.0 的路徑再確認
- **DeepEP**：唯一 tag v1.2.1 要求「CUDA 12.3 and above for SM90」＋NVSHMEM（Opus 抽查 README 第 48–49 行）→ 在驅動 550 下應可建置；T3 報告說的 CUDA 13.1 是主線 README，不適用 tag 版
- **SHARP 軟體存在**：`/work/HPC_software/LMOD/nvidia/packages/hpc_sdk-24.7/Linux_x86_64/24.7/comm_libs/12.5/hpcx/latest/` 下有 `sharp/lib/libsharp_coll*` 與 `nccl_rdma_sharp_plugin/lib/libnccl-net.so`（Opus 讀檔確認）。交換器端（Aggregation Manager）有沒有開，要在計算節點實測 → 加進 M20／M23


## R04_CD（C 通訊的 SM 稅、D 效能模型選 collective）讀後與抽查（2026-11-17）

- **Opus 抽查**：
  - NCCL `src/ce_coll.cc:174-190`（v2.32.6）：copy-engine collective 要 `driverVersion >= 12050`，且只支援 AllGather／AlltoAll／Scatter／Gather（**沒有 reduction**）。登入節點 `nvidia-smi` 驅動 550.127.08（CUDA 12.4）→ 零 SM 的 copy-engine 路線在我們環境大概不能用（計算節點待 M20 確認）
  - NCCL-BOT（arXiv 2611.11303，NVIDIA，2026-11-08，讀 PDF 摘要）：H100 上用 tuner plugin＋bandit 修正 AUTO；AUTO 平均多 6.1%、p99 多 103%；「We do not show a net end-to-end win」；小於 1 MiB 時 cost model 在 H100 上已準
  - Weave（2609.21483，4×H100，MoE megakernel 動態 SM 分工）、OverlapAdvisor（2611.11857，MoE EP 重疊效能模型，MAPE 3.3–15.9%）：摘要核實
- **判斷**：
  - **D 不建議當主線**：NVIDIA 自己剛在 H100 上做（NCCL-BOT），且其結論是「小訊息時內建模型已準、端到端沒有淨收益」→ 空間小
  - **C 單獨做也擁擠**：MoE 的動態 SM 分工（Weave、OverlapAdvisor、Analytical Resource Mgmt、Resource-aware Overlap）2026 下半年密集出現；SM-free 路線（Zero-SM、VCCL、T-CCL）也在做，且 copy engine 在驅動 550 上可能不可用
  - **但 C 可以併入 A**：交換器內加總（NVLS 機內、SHARP 機間）把「加法」從 GPU 運算單元搬到交換器，本來就該減少 SM 用量與對 GEMM 的干擾（NCCL 2.27 部落格宣稱 SHARP 讓 AG／RS 需要的 SM 從 ≥16 降到 ≤6，無數據）。R04_CD §5-1、5-2：H100 上「collective 實作 × SM 數 × GEMM 拖慢」沒有系統性量測，NVLS 對同時執行的 GEMM 的影響也沒人量 → **A 的主要量測軸可以是「延遲」與「SM 稅」兩個**
  - 這也繞開了 copy engine 在驅動 550 不可用的問題：交換器加總不需要 copy engine

## R04_A（兩層交換器加總）讀後與抽查（2026-11-17）

- **Opus 抽查**（NCCL v2.32.6 原始碼、登入節點唯讀）：
  - `src/tuning/nvls.cc:149`：`NCCL_ALGO_NVLS` 在 `nNodes > 1` 且沒有 `collNetSupport` 時無效 → 跨機時「NVLS 演算法」＝機內 NVLS＋機間 IB SHARP；沒有 SHARP 時跨機只能走 NVLS_TREE（機內 NVLS＋機間一般樹）
  - `src/graph/connect.cc:279` 註解「NVLS+SHARP not yet implemented」已過時，`:282` 在 `collnetEnable` 時設 `nvls.out`
  - 外掛相容：HPC-X 外掛 `libnccl-net.so` 匯出 `ncclNetPlugin_v5–v8`、`ncclCollNetPlugin_v5–v8`；安裝的 NCCL 2.29.7 字串含 v6–v11 → v8 重疊（ABI 層面相容，實際載入待計算節點）
- **報告事實（未逐條抽查）**：HPC-X v2.19 內含 SHARP 3.7.0、plugin 2.7；系統另有 `/opt/mellanox/sharp` 3.5.1（版本混用風險）；外掛的 `libsmx-3.7.so` 等沒有 `LD_LIBRARY_PATH` 時找不到；SHARP 需要 root 在 subnet manager 端跑 Aggregation Manager（或 UFM）、計算節點要有 IPoIB；NCCL 2.18 有「H100＋IB SHARP 多 GPU 同進程資料錯誤」的已知問題（2.18.3 修正；我們一 GPU 一進程，不受影響）
- **文獻**：兩層同時啟用的實機量測——沒有找到；SHARP 用於 LLM 推論或小訊息延遲——沒有找到；Demystifying NCCL 明確排除 NVLS 與 CollNet；新增 CAIS（2605.05628）、DySHARP（2605.05607）都是 NVLS 的週期級模擬，指出 NVLS 的 AG／RS 與 GEMM 記憶體語意不匹配、只支援靜態規則集體
- **判斷**：
  - 文獻空白確定比 B／C／D 大；軟體層面看起來可行
  - **單一最大風險＝Nano5 的 IB 交換器有沒有跑 SHARP Aggregation Manager**。使用者端只能用 `sharp_hello`／NCCL 日誌判斷 → M20 P10 一定要做；**同時建議使用者直接問 NCHC**（AM 是否啟用、每個 job 的 SHARP 配額、是否 rail-optimized 拓樸）
  - 若 SHARP 不可用：A 退化成「機內 NVLS × SM 稅 × 跨機 NVLS_TREE」的特性量測，仍有空白（R04_CD §5-1、5-2），但主張變弱

## R04_CD（方向 C、D 深入調查）與 Opus 抽查（2026-11-17）

- **NCCL 內建的 copy engine 集體通訊在 Nano5 用不了**（Opus 讀原始碼確認）：`src/ce_coll.cc:175-189`（v2.32.6）`ncclCeImplemented` 要求 `driverVersion >= 12050`，且只支援 AllGather／AlltoAll／Scatter／Gather，沒有 reduction；我們的驅動 550.127.08 回報 CUDA 12.4（R01:43）。2.29.7 是否同門檻未確認。自己用 `cudaMemcpyPeerAsync` 等寫 copy engine 傳輸不受此門檻限制（判斷，未驗證）
- **方向 C 正在快速變擠**（Opus 以 WebFetch 開 arXiv 頁確認）：
  - TIDE（arXiv 2610.38228，2026-10-28）：FSDP 用 copy engine＋RDMA＋NVSwitch multicast 的零拷貝集體通訊，64 GPU，reduce-scatter 的 SM 時間降 12.5–14.5×，step time 對 FSDP2 最多 −27.1%（GPU 型號摘要未寫）
  - Zero-SM NCCL（2611.18075，GB200／B200）、VCCL（2510.00991）、ConCCL（AMD）
  - 仍未找到：H100 上以 CTA 數或 CTA policy 為變數、四種實作（ring／NVLS／symmetric／CE）對同一個 GEMM 拖慢的系統對照（R04_CD §C-5）
- **方向 D 已有 NVIDIA 本身在做**（Opus 確認）：NCCL-BOT（arXiv 2611.11303，2026-11-08，作者含 NCCL 團隊）：H100 上用 per-context bandit 修正 NCCL 成本模型，損失 >10% 的 context 從 19 降到 1；<1 MiB 輸給 stock；自述「do not show a net end-to-end win」；只選 algo×proto
- 其他事實（R04_CD，未抽查）：2.29.7 的 tuner plugin 只有 v2–v5（`strings` 比對）；tuner 只能改舊式 algo×proto 表，改不了 symmetric／CE 候選（讀碼推論）；NVLS 在 SM90 預設 16 channel（`nvls.cc:201`）

### 判斷更新

- **D 降級**：NVIDIA 已在做模型修正，且 2.29.7 的 tuner 介面受限；除非與 C 結合（「有並行計算時最佳選擇會變」——只有一個課程專案提過，非同儕審查），否則不單獨做
- **C 保留但只當 A／B 的評估維度**：「省多少 GPU 計算資源」是 A（交換器加總理論上不用 SM 做加法）的重要指標，可在 A 的實驗中一起量；單獨做 C 與 TIDE／Zero-SM 重疊高、且 NCCL CE 路徑受驅動限制
- 主推仍為 A，等 R04_A、R04_BE

## R04_BE（方向 B、E 深入調查）與 Opus 抽查（2026-11-17）

- **跨機 TP 在 H100 上一律退回 NCCL**（Opus 讀已安裝的 vLLM 0.32.0 確認）：`site-packages/vllm/distributed/device_communicators/custom_all_reduce.py:204-238`——跨機群組要通過 `_group_can_attempt_mnnvl`，其條件為 `has_device_capability(100)`（Blackwell）且有 multicast；H100（9.0）不符 → 自訂 all-reduce 停用。註：0.32.0 的 `_SUPPORTED_WORLD_SIZES` 含 16（R04_BE 說「16 超過 8」的理由不正確，但結論相同）。TRT-LLM `allreduceOp.cpp:1250-1257` `ifFallbackToNCCL` 在 `!mIsP2PSupported || !mIsNVLINKSupported` 時回 NCCL（Opus 讀碼確認）
- **跨機 TP 的實務**：vLLM、SGLang、TRT-LLM 官方都有 TP=16 跨 2 台的範例；但 vLLM 同頁建議 TP＝每台卡數、PP＝台數，Llama 3 論文 §6.1 跨機用 PP（R04_BE，未抽查）→ 方向 B 的動機要靠「為什麼要跨機 TP」（例如 decode 延遲、PP 的 bubble）撐住，需要更強證據
- **decode all-reduce 訊息大小**：NVRAR 原文 B×H，70B、B=8 時 128 KB，工作負載 128 KB–1 MB；SiFAR 低批量 4–32 KB（R04_BE）
- **IBGDA 需要管理員或 GDRCopy**（Opus 以 WebFetch 讀 NVSHMEM install guide 確認）：預設模式需 `nvidia.ko` 以 `PeerMappingOverride=1` 載入（改 `/etc/modprobe.d/nvidia.conf`）；CPU-assisted 模式不需要，但要 GDRCopy。dma-buf 需 rdma-core ≥44 且核心 ≥6.1；沒有 dma-buf 時需 `nvidia-peermem`
  - 登入節點：核心 4.18（→ dma-buf 路徑不可用）、`/dev/gdrdrv` 存在、未見 `nvidia_peermem`（R04_BE；計算節點未知）
  - **M20 P6（GPUDirect RDMA）是所有跨機方向的關卡**：計算節點若沒有 `nvidia_peermem`，NCCL 跨機只能繞主機記憶體
- 新的低延遲 GPU 發起通訊量測（R04_BE，未抽查）：Dissecting（2610.01380，H200＋CX-7）NVSHMEM IBGDA put＋完成 11.10 µs vs IBRC 6.40 µs（報告者提醒表格可能錯位）；Perseus（2605.00686，H100＋CX-7，2–4 節點）IBRC 持平或勝過 IBGDA；mKernel 自述 IBGDA 增益小
- **方向 E**：vLLM 0.32.0 有 NIXL／Mooncake／LMCache 連接器但需另裝；在 H100＋IB 上量 KV 傳輸佔比的資料仍未找到；矛盾未消除（R04_BE）

### 判斷更新

- **B 保留，但動機需補強**：「H100 跨機 TP 只能用 NCCL」已確認，空間存在；但業界主流跨機用 PP／EP，要先證明跨機 TP decode 有真實需求或優勢。B 與 A 可合併：A 的 SHARP 正好是跨機小訊息加總的硬體路徑
- **E 降級**：需另裝多套軟體、系統整合重，且偏量測；不當主線
- **下一個關卡**：M20 P4（NVLS）、P6（GPUDirect RDMA）、P10（SHARP）

## R04_BE（B 跨機 TP decode、E PD 分離 KV 傳輸）讀後與抽查（2026-11-17）

- **Opus 抽查**：
  - vLLM v0.34.0 `docs/serving/parallelism_scaling.md`：多機時「set `tensor_parallel_size` to the number of GPUs per node and `pipeline_parallel_size` to the number of nodes」，並稱這是「common practice」；`--tensor-parallel-size 16` 只列為「Alternatively」
  - vLLM `custom_all_reduce.py:202-234`：不在同一台（且非 MNNVL）時停用自製 all-reduce → 跨機 TP 回到 NCCL
  - 登入節點：`gdrdrv` 已載入、`nvidia_peermem` 沒載入；kernel 4.18 → 報告說 NCCL GIN GDAKI 需要 kernel ≥ 5.12（peermem）或 ≥ 6.1（DMA-BUF）→ GPU 直接發 RDMA 的路線在這裡大概不可行（計算節點待 M20）
  - TKNP（arXiv 2611.13293，2026-11-09，64 H100，摘要核實）：設計原則是「dense all-reduces remain within NVLink」，刻意不讓 dense all-reduce 跨機
- **報告事實（未逐條抽查）**：SGLang 文件有 405B 兩台 `--tp 16` 範例；TRT-LLM（legacy 文件）說跨機 TP 會被慢的互連卡住；NVRAR 發現 decode-heavy 時 TP 勝過 TP+PP；TKNP 量到 TP64 的單次 all-reduce 是 TP8 的 6.5×；H100 8 卡＋IB NDR 的跨機小訊息 all-reduce 量測沒有找到；E：沒有論文在 H100＋IB NDR 上量 KV 傳輸佔端到端比例
- **判斷**：
  - **B 單獨成題的動機弱**：主流建議與最新研究（TKNP）都在避開跨機 dense all-reduce；GPU 直接發 RDMA 的優化手段在這裡大概不可用（kernel 4.18、無 root）
  - **B 作為 A 的應用很合適**：「為什麼大家避開跨機 TP？因為跨機 all-reduce 慢。若兩層交換器加總（SHARP 低延遲模式）讓它變快，這個取捨會不會改變？」——這是可檢驗的問題，而且有實際後果（405B 在 2 台 H100 上要用 TP16 還是 TP8×PP2）
  - E 有量測空白但工程重、與 A 無關 → 列為備案

## 綜合建議（2026-11-17，給使用者決定）

- **主線 A＋（C 的量測軸）＋（B 的應用場景）**：在 H100 小型叢集上，量化並利用兩層交換器加總（機內 NVLS、機間 IB SHARP）——延遲（小訊息）、頻寬、SM 用量與對同時計算的干擾；應用到 (1) 跨機 TP decode 的平行方式選擇、(2) 訓練的 reduce-scatter／all-gather
- **不做**：D（NCCL-BOT 已覆蓋）、單機 all-reduce＋norm 融合
- **關卡**：SHARP 是否可用（M20 P10＋詢問 NCHC）。不可用時的備案：C 縮小為「單機 NVLS × SM 稅 × 同時 GEMM」＋跨機 NVLS_TREE 的特性量測，或 E

## 整合判斷（Opus，2026-11-17 01:10，合併兩個對話的結論；以本節為準）

- **兩個對話獨立得出相同結論**：主線 A（兩層交換器加總：機內 NVLS＋機間 IB SHARP），C（SM 稅、對同時計算的干擾）作為量測軸，B（跨機 TP decode）作為應用場景；D 不做（NCCL-BOT 已覆蓋）；E 備案
- **再次抽查（本對話，01:05）**：`src/tuning/nvls.cc:149` 跨機 NVLS 需要 collNetSupport；HPC-X 外掛 `nm -D` 匯出 net／collnet v5–v8，2.29.7 `strings` 含 v6–v11（重疊 v6–v8）
- **實驗設計注意**：載入 HPC-X 外掛（`libnccl-net.so`）後，NCCL 的網路傳輸也改由外掛提供 → 「SHARP 開／關」的公平對照是「外掛載入＋`NCCL_COLLNET_ENABLE=1`」對「外掛載入＋`NCCL_COLLNET_ENABLE=0`」，另加「不載外掛（NCCL 內建 IB）」作為參考
- **研究問題的雛形**（待使用者決定是否採用）：在 ≤4 台 H100 的小型叢集上，兩層交換器加總在小訊息延遲、大訊息頻寬、GPU 計算資源用量三方面各帶來多少？能不能讓「跨機 TP」這個業界刻意避開的選項（vLLM 文件、TKNP）在 decode 時變得划算？
- **關卡（依序）**：
  1. M20 P6：計算節點有沒有 GPUDirect RDMA（核心 4.18 不能走 dma-buf，只能靠 `nvidia_peermem`）——所有跨機方向都受影響
  2. M20 P10：SHARP 能不能用（`sharp_hello`＋NCCL 日誌）
  3. 向 NCHC 詢問：Aggregation Manager 是否啟用、每個 job 的 SHARP 配額、是否 rail-optimized 拓樸、計算節點是否載入 `nvidia_peermem`
- **SHARP 不可用時的備案**：單機 NVLS × SM 稅 × 同時 GEMM ＋ 跨機 NVLS_TREE 的特性量測（主張較弱），或 E
