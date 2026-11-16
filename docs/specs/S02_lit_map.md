# S02 文獻地圖（L1）：Hopper／Blackwell 世代的 GPU 通訊優化研究

- **寫給**：Sonnet 5.5 執行者（每個主題一位，並行）　**撰寫**：Opus 5.5，2026-11-16（D051）
- **類型**：文獻調查，不需 GPU
- **報告**：每個主題一份 `docs/reports/R02_T<k>_<slug>.md`（k 與 slug 見下表）

## 目標

使用者要在新環境（下述）「找到一個優化通訊的破口」，**不受 A100 時期題目的限制**。你的任務是為**一個主題**建立事實地圖：近年做了什麼、在什麼硬體上做、比較對象是誰、作者自己承認的限制、哪些情況還沒被量過。**選題判斷由 Opus 做，你只提供有出處的事實。**

## 我們的環境（用來判斷「這篇在我們這裡能不能重現」）

- 國網中心 Nano5：每台 8× H100 80GB HBM3 SXM（或 8× H200），NVLink4＋NVSwitch（預期支援 NVLS，待驗證），每台 8× InfiniBand NDR 400 Gb/s
- 最多可用：H100 4 台（32 GPU，單一工作 12 h）或 2 台（16 GPU，48 h）；H200 2 台（16 GPU）
- 驅動 550（CUDA 12.x）；沒有 root；Slurm 排程

## 主題分配

| k | slug | 範圍（包含但不限於） |
|---|---|---|
| T1 | intranode_collectives | 單機集體通訊與通訊函式庫：NCCL 內部與調參、NVLS／multimem、symmetric memory、NVSHMEM、MSCCL++ 類函式庫、自製 all-reduce、小訊息延遲、單機演算法合成；**另含量測與建模**：H100／H200／GB200 通訊特性量測論文、NCCL 效能模型、模擬器 |
| T2 | overlap_fusion | 計算與通訊重疊／融合：TP overlap（async TP、kernel 內細粒度重疊）、GEMM＋collective 融合、通訊＋正規化等算子融合、PDL、以 compiler 或 DSL 產生的分散式 kernel |
| T3 | moe_ep | MoE 專家平行通訊：all-to-all dispatch／combine、DeepEP 類函式庫、單機 NVLink＋跨機 IB 的階層式、負載不均、與計算重疊 |
| T4 | internode | 跨機器通訊：IB／RoCE 集體通訊、階層式演算法、SHARP 等網路內運算、拓樸感知演算法合成、multi-rail、GPUDirect RDMA、訓練的 DP／FSDP／ZeRO 通訊、梯度壓縮、壅塞與落後節點 |
| T5 | inference_serving | 推論服務中的通訊：TP／EP 生成階段延遲、prefill／decode 分離的 KV cache 傳輸、context parallel／長上下文、跨機器推論、RL 訓練中的權重同步 |

## 步驟

1. **搜尋**：
   - 用 WebSearch（先 `standard`，結果太少或太舊再用 `extended`），並行送多個查詢。工具若尚未載入，先用 ToolSearch 載入 `WebSearch`、`WebFetch`。
   - 範圍：2023–2026（重點 2024-01 至今天 2026-11-16）。
   - 優先順序：OSDI、SOSP、NSDI、ASPLOS、EuroSys、USENIX ATC、SC、PPoPP、HPDC、ICS、MLSys、ISCA、MICRO、HPCA 的論文，以及 arXiv 預印本；另加主要廠商的技術部落格或白皮書（標明不是同儕審查）。
   - 數量：每個主題 **15–30 篇**，寧缺勿濫。
2. **核實**：每篇都要實際打開 arXiv abs 頁、DOI 頁或會議頁（用 WebFetch），確認標題、作者、年份、出處。**不可憑記憶列論文**；打不開、無法核實的不要列。
3. **擷取**（每篇；只寫原文有的，沒有就寫「未說明」）：
   - 解決的問題（一句）
   - 方法要點（1–2 句）
   - 硬體與規模：GPU 型號、單機互連（NVSwitch／PCIe）、有沒有用 NVLS、跨機網路、GPU 數
   - 比較對象（baseline）
   - 報告的效益：數字附條件
   - **作者自述的限制或未來工作**：原文引用 ≤ 2 句，附位置（節號或頁碼）
   - 程式碼是否公開：網址
   - 在我們環境能否重現：依上面的資源判斷「可／部分／不可」，附一句理由（例如「需要 GB200 NVL72」或「需要 64 GPU」）

## 產出：`docs/reports/R02_T<k>_<slug>.md`

1. **主題概況**：3–6 句，純事實描述這個主題近年的主要做法分支。
2. **論文表**：每篇一列，欄位如步驟 3，加上 `出處網址`。依年份新到舊排列。
3. **作者自述的限制與未來工作彙整**：依出現次數歸類，每類附論文編號與原文引用。
4. **常見比較對象**：大家都跟誰比；有沒有公認的最強對照。
5. **硬體覆蓋**：
   - 哪些論文在 8× H100／H200 NVSwitch 單機，或 ≤ 4 台 IB 叢集上做過實驗
   - 哪些只在 A100、AMD、GB200 或模擬器上做過
   - 有沒有「在 NVLS 開與關之間比較」的
6. **開源工具與程式碼清單**：名稱、網址、授權（若有寫）。
7. **搜尋紀錄**：用過的查詢字串、每個查詢大約看了幾筆結果、排除了哪些及原因。

## 規則

- 只寫事實與原文；**不寫**「這是個好題目」「建議研究」之類的判斷。如果發現兩篇論文的數字互相矛盾，列出來就好，不裁決。
- 每個論點都要有網址或原文位置。
- 不改 `lit/refs.bib`（引用 BibTeX 之後由 Opus 用 arXiv MCP 或 DOI 匯出）。
- 需要下載 PDF 的話，存到 `papers/`（不進 git），檔名用 arXiv 編號。
- 不 commit、不 push；不執行任何 GPU 或 Slurm 工作；不讀 `~/TP_opt`、`~/tp_opt`、`~/flux`。
- 用繁體中文撰寫，技術名詞與論文標題保留英文。
