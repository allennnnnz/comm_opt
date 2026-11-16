# Opus 讀報告筆記（Nano5 選方向用，進行中）

> Opus 自己的工作筆記：從 R02_T1–T5、R03 讀出的重點與初步判斷。判斷標「判斷」，事實附報告與論文編號。關鍵論點在用於決策前要回原文抽查（守則第八節 5）。

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

