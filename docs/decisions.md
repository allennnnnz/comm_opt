# 決策紀錄

> append-only：只能在最後新增條目，不改寫舊條目。需要推翻舊決策時，新增一條並註明推翻哪一條。
> 格式：`## D<編號>（日期）標題` → 背景、決定、理由、影響。需要使用者裁決但尚未裁決的條目加 `[需裁決]`。

## D001（2026-11-02）從零開始，不沿用舊研究

- **決定**：本 repo 的題目、量測與寫作全部重新開始；不讀、不沿用 `~/TP_opt`、`~/tp_opt`、`~/flux` 的研究內容。
- **理由**：使用者指示「用 nccl test 重新發展一個新的題目，不要看原本做的研究，整套論文寫作重新開始」。
- **影響**：硬體事實一律重新量測；選題只依本 repo 的量測與文獻。

## D002（2026-11-02）工具：只用不需 API key 的 skills 與 MCP

- **決定**：在 Claude Code 內完成研究；裝 arXiv 與 paper-search 兩個 MCP，加上 23 個 skills（來源見 `.claude/SKILL_SOURCES.md`）。
- **理由**：使用者沒有 LLM API key，gpt-researcher 和 AI-Scientist 這類需要 key 的工具無法使用。

## D003（2026-11-02）研究守則與專案結構

- **決定**：守則見 `docs/research_rules.md`，依據 Hoefler & Belli SC'15、Heiser benchmarking crimes、SIGPLAN checklist、Mytkowicz ASPLOS'09、Sandve 2013、ACM artifact badging、Levin & Redell 1983、Peyton Jones。結構為「每個實驗一個資料夾」。
- **使用者的選擇**：每個實驗一個資料夾；已核准的實驗 Claude 可自己跑，改方向才問；筆記用中文、論文用英文；每完成一個段落就 commit，不 push。

## D004（2026-11-02）M0 核准，但要等其他工作結束

- **決定**：核准 M0_baseline。開跑條件：機器上其他 GPU 工作的 driver 腳本結束，且所有 GPU 連續 3 分鐘沒有程式。
- **理由**：使用者指示「不要中斷已經在執行的其他任務，等他完成再開始測量」。該工作有獨佔檢查，且對延遲敏感。

## D005（2026-11-02）M0 完成；分析環境與事後新增的偵測

- **結果**：M0 18 組全部成功（`results/M0_baseline/20261102_134841/`），觀察到 7 個候選現象 F1–F7（見 `exp/M0_baseline/README.md`）。
- **分析環境**：`.venv`（uv，Python 3.12），版本鎖在 `requirements-analysis.txt`（D3）。
- **事後新增的偵測**：`gpu_count_gap`（同一大小不同 GPU 數的 busbw 落差 > 30%）是看到 sendrecv 平台之後才加進 `analyze.py` 的。因此 F1–F7 屬於事後觀察，不能直接當結論；M4 要在量測前寫好判準再驗證。
- **使用者開始量測的依據**：使用者表示已暫停其他項目；量測前確認 8 張卡記憶體 0 MiB、無任何程式（R13）。

## D006（2026-11-02）核准 M4_repeat 與 M2_p2p_channels；F1 的機制在原始碼中找到

- **決定**：使用者核准（「好 開始做」）M4_repeat（重複驗證 F1–F5）與 M2_p2p_channels（F1 能否靠設定解決）。兩張實驗卡的預期與判準在量測前 commit。
- **新資訊**：讀 NCCL 2.32.3 原始碼（tag `v2.32.3-1`）找到 F1 的機制：每個 P2P 操作的 channel 下限依 communicator 的 rank 數減半（`scheduleP2pTasksToPlan`），且只有每個 channel 負責超過 16 MiB 時才加倍（`addP2pToPlan`）。`exp/M2_p2p_channels/predict_channels.py` 依此重現 M0 的所有 channel 觀測值。
- **影響**：F1 是確定性的設計行為，不是雜訊；它是否值得研究，取決於 M2 的結果（能不能靠設定解決、代價是什麼）。

## D007（2026-11-02）M4 與 M2_p2p_channels 完成；G1 報告送出 `[需裁決]`

- **M4**：F1、F2a、F2b、F4、F5 依量測前判準成立；F3 不成立（只在 2 MiB 明顯）；兩組 GPU 無差異。見 `exp/M4_repeat/README.md`。
- **M2**：原始碼 channel 模型 143/143 點正確；F1「部分是預設值問題」——16 MiB 最佳設定只到 2 卡的 69%，64 MiB 調小 chunk 可達 104%，但所有能改善 8 卡的 chunk 設定都讓 2 卡 send/recv 或 alltoall 變慢。見 `exp/M2_p2p_channels/README.md`。
- **G1**：報告在 `docs/gates/G1_report.md`，建議以 F1 為主線。`[需裁決]`：使用者選擇要追哪個現象。

## D008（2026-11-02）G1 裁決：F1 先記錄，改量端到端的通訊佔比

- **使用者決定**：「這個觀測可以先記錄下來，我想要其他面向的觀測，我想先知道目前通訊佔用端到端多少的時間」。F1 保留為已確認的觀測（`docs/gates/G1_report.md`），暫不選為主題。
- **下一步**：設計 M5（真實訓練／推論工作負載中，通訊占端到端時間的比例），設計與工作負載選擇需使用者核准。

## D009（2026-11-02）M5 範圍：推論 TP=8 與訓練張量平行，兩種模型大小

- **使用者的選擇**：推論「模型切到 8 張卡」（vLLM，TP=8）；訓練「模型切到多張卡」（torchtitan，TP=8、TP=4×FSDP=2、TP=2×FSDP=4）；中型與小型模型都量。
- **量法依據**：Mirzaei & Mitra 2610.05305（延遲拆成計算、通訊、空檔；分讀入與生成階段）、Chopper 2512.08242（多層級 GPU timeline 對齊、記錄時脈）、Lee et al. 2507.03114（重疊造成的計算變慢）。
- **環境**：`.venv-vllm`（vLLM 0.33.0）與 `.venv-titan`（torchtitan 0.4.0），兩者都用 PyTorch 2.14.0。模型權重用 Hugging Face 共用快取裡已下載的 Qwen2.5-32B-Instruct 與 TinyLlama-1.1B；訓練用隨機初始化。

## D010（2026-11-02）M5 完成：通訊占端到端 8–30%

- **結果**：推論（vLLM TP=8）Qwen2.5-32B 生成 8–27%、讀入 24–30%；訓練（torchtitan）Llama-3 8B 11–22%（加上重疊造成的計算變慢為 16–22%）。小模型受 CPU 主導。見 `exp/M5_e2e_comm_share/README.md`。
- **量測方法的修正**（量測前已寫入實驗卡）：端到端時間用不開 profiler 的量測；通訊 kernel 時間拆成真正傳輸與等待；讀入階段改以每步 token 數為變因；訓練環境改用 PyTorch 2.15.0（torchtitan 0.4.0 需要）。
- **新的候選現象**：O1–O6（實驗卡「結論」），連同 F1–F5 交給下一次 G1。

## D011（2026-11-03）依實測數據做文獻對照；建議方向待使用者裁決 `[需裁決]`

- **使用者要求**：根據量到的數據搜尋相關論文與結果，再判斷實作方向；有開源的順便參考做法。
- **做法**：三個助手分頭調查（生成階段 all-reduce、通訊重疊、NCCL 內部與 P2P），彙整在 `lit/notes/survey_20261103.md`；16 篇 BibTeX 以 arXiv MCP 匯出加入 `lit/refs.bib`。
- **關鍵發現**：
  - 生成階段中等大小 all-reduce 在 A100（無交換器內歸約）上是明確的空白；最接近的 SiFAR 只做 H200、≤32 KB；MSCCL++ 在 A100 上用的是舊版 vLLM。
  - 讀入與訓練的通訊重疊在 A100 上已很擁擠（FLUX、NanoFlow、ISO、FlashOverlap、Entwine 等）。
  - 一對一傳輸的通道問題已被 AMD RCCL PR #8349 描述並修正（論文價值低）。
  - **量測方法的缺口**：M0／M4 的 nccl-tests 用預設 `-R 0`（沒登記記憶體），NCCL ≥ 2.27 的 symmetric kernels 不會啟用；也沒用 CUDA graph（`-G`）。「NCCL 小資料慢」在重量前不能當結論。
- **建議**（待裁決）：以「生成階段中等大小 all-reduce」為主線，但先做一次現成實作的對照量測（M6），依量測前寫好的判準決定是否值得實作。

## D012（2026-11-03）核准 M6：現成 all-reduce 做法對照

- **使用者核准**：「好 先去測試這個」（測試生成階段加總同步的現成做法與時間拆解）。
- **納入**：NCCL 2.32.3（一般、CUDA graph、登記記憶體的 symmetric kernels）、MSCCL++（commit d5ce6c72，NCCL 相容介面）、PyTorch 自帶 NCCL 2.30.7、vLLM custom AR、PyTorch SymmetricMemory one-shot／two-shot。
- **未納入**：low-latency-nccl（ss16118，個人帳號專案）的下載與編譯被安全檢查擋下，**需使用者決定是否放行**；SiFAR 無程式碼；TensorRT-LLM／FlashInfer 在 SM80 不啟用。
- **量測前的修正**：資料型別由 bf16 改為 fp16（MSCCL++ 介面下 nccl-tests 不支援 bf16）。
