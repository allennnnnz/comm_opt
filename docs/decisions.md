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

## D013（2026-11-03）M6 完成：判準結果「值得實作」 `[需裁決]`

- **結果**：64 KB–1 MB 最快的現成做法（vLLM、MSCCL++）比可達水準慢 1.53–2.07 倍，5/5 個大小 ≥ 1.5 → 依量測前判準「值得實作」。我的預期是「部分」，結果比預期更有空間。
- **新現象**：P1 搬資料階段只有可達頻寬的 10–50%（不只是同步問題）；P2 沒有單一最佳做法；P3 NCCL symmetric kernels 在 16 KB 以上選擇不佳；P4 NCCL 2.32.3 比 2.30.7 快 20–40%。見 `exp/M6_allreduce_bakeoff/README.md`。
- **偏差**：SymmetricMemory 沒有 fp16 kernel，以 bf16 補跑（相同 bytes）；不影響判準。
- `[需裁決]`：是否以此方向進入選題（G2）；是否放行 low-latency-nccl。

## D014（2026-11-03）核准 M7（找出慢在哪）；放行 low-latency-nccl

- **使用者決定**：「先找慢在哪，我放行可以研究低延遲版本 nccl」。
- **low-latency-nccl**：ss16118/low-latency-nccl 分支 `siyshen/low-latency`（NCCL 2.29.2 為基礎），只在本機編譯與執行。另以它的 headers 重編一份 nccl-tests（`third_party/nccl-tests-llnccl`）；用系統 2.32.3 headers 編的 nccl-tests 搭配它會得到錯誤結果。
- **試跑發現**：它新增的 Lamport／LLBuffer 做法在 A100 上結果錯誤（預設也挑到這類），只有保留的 AGxLL_R、RSxLD_AGxST 正確。
- **M7 設計**：NCCL SYM 的做法 × CTA 數掃描、low-latency-nccl、vLLM 1/2-stage、MSCCL++ symmetric memory，共 22 種 × 3 次。

## D015（2026-11-03）M7 完成：需要新的做法 `[需裁決]`

- **結果**：NCCL SYM 在 A100 上選錯 CTA 數（16K–256K 自選比最佳慢 1.35–2.33 倍，判準成立）；但所有做法的最佳組合在 64K–1M 仍比可達水準慢 1.44–1.91 倍（4/5 ≥ 1.5）→ 依判準「需要新的做法」。
- **機制**：one-shot 一次同步但每卡讀 7×S；two-shot 資料量 1.75×S 但兩輪同步（固定約 13 µs vs 5.4 µs）。沒有現成做法同時做到一次同步與 two-shot 的資料量。
- `[需裁決]`：是否以此進入選題；先確認《Every µs Matters》的 LL128 原子加 two-shot 在 A100 上是否可行（新穎性風險）。

## D016（2026-11-03）核准 M8：LL128 atomic 在 A100 上是否可行

- **使用者決定**：「好 先做第一步」。
- **讀論文與原始碼已確認**：LL128 atomic two-shot 仍是 2 次同步（論文 Table I）；公開分支沒有 LL128 atomic 的實作；分支中所有新增做法都只為 sm_90 以上編譯（A100 上是空殼）。
- **測試**：向量原子加法能否在 sm_80 編譯；8 張 GPU 對同一 128 byte 行做 warp 寬度原子加法時，是否會出現「旗標已滿但資料未加完」；跨 GPU 原子加法頻寬。
- **修正先前說法**：D015 的「一次同步 + two-shot 資料量」不精確——two-shot 本質上需要兩輪；研究空間是讓兩輪都不需要全域 barrier（且在 A100 上可行）。

## D017（2026-11-03）M8 完成：LL128 atomic 的前提在 A100 上很可能成立 `[需裁決]`

- **結果**：sm_80 沒有 16 byte 向量原子加法；但 7 張 GPU 以 warp 寬度 32-bit 原子加法寫同一 128 byte 行，3 × 2^20 行 0 次違反（約 25% 的行在寫入途中被觀察到），匯入總頻寬 235–241 GB/s。見 `exp/M8_ll128_atomic_a100/README.md`。
- **與預期相反**：我預期會出現違反。
- **影響**：《Every µs Matters》的 two-shot LL128 atomic 很可能可以移植到 A100（改用 32-bit 原子加法）；只做移植的論文貢獻偏弱。`[需裁決]`：重新決定題目的定位。

## D018（2026-11-03）文獻與程式碼查核：A100 上的「加總同步＋正規化合併」已有人做 `[需裁決]`

- **使用者要求**：「查 TensorRT-LLM 和其他框架：確認『加總＋正規化合併』在 A100 上有沒有人做過。不需要 GPU」。只讀程式碼與文件，沒有用 GPU。見 `lit/notes/fused_ar_rmsnorm_a100.md`。
- **結果**：TensorRT-LLM（one-shot Lamport 免 barrier；two-shot 2 次 barrier、norm 對全部列重複做）與 LMDeploy TurboMind（token 列 RS → 只對自己那段 residual + norm → AG，2 次 barrier）都有 A100 可走的合併 kernel。vLLM、SGLang、FlashInfer 在 A100 上不開。
- **影響**：「分工加總＋各自正規化＋互相交換」本身不是新貢獻。沒找到的是：**兩輪都免 barrier 的合併 two-shot**（LMDeploy 在 ≤ 512 KB 把合併關掉；TRT-LLM 的 two-shot 合併要 2 次 barrier；《Every µs Matters》不合併 norm、只在 GB200），以及**在 A100 上依實測挑做法**（TRT-LLM 把 sm80 當 sm90 查 H100 調出的表）。
- **更正**：D012 的「TensorRT-LLM 在 SM80 不啟用」不精確——FlashInfer 版不啟用，TRT-LLM 自己的合併 kernel 有 A100 程式路徑（未實測）。
- `[需裁決]`：是否先在 A100 上量 TRT-LLM 與 LMDeploy 的合併 kernel（新實驗），再決定是否做「兩輪免 barrier 的合併 two-shot」。

## D019（2026-11-04）核准 M9：在 A100 上量現有的合併做法；放行編譯 LMDeploy

- **使用者決定**：「好 開始量 並且放行」（看過實驗卡簡介之後）。
- **範圍**：照 `exp/M9_fused_baselines/README.md`：vLLM／NCCL 不合併、TRT-LLM 合併 one-shot／two-shot（FlashInfer 0.7.2.post1 同源原始碼另編 sm_80）、LMDeploy 合併（原分派與強制合併）。
- **LMDeploy**：commit 3228395，本機編譯（會下載 fmt、cutlass、yaml-cpp、xgrammar、Catch2），只用於量測。

## D020（2026-11-04）M9 完成：A100 上現有合併沒有比較快；離可達水準仍有 2 倍以上 `[需裁決]`

- **結果**（hidden 5120，80K–1.25M）：最快的合併做法是可達水準的 2.16–3.05 倍（5/5 ≥ 1.3）→ 依判準「有空間」。但最快的任何做法有 4/5 是不合併的 vLLM（`U_vllm`）；TRT-LLM two-shot 合併慢 1.8–2.0 倍；TRT-LLM 合併在 hidden 8192 無法啟動（A100 沒有 cluster，block 資源超限）。見 `exp/M9_fused_baselines/README.md`。
- **與預期不同**：P1（one-shot 合併比 vLLM 快 1.2 倍）不符，P2 大訊息方向相反。
- **使用者提出的策略**（2026-11-04）：「合併別人的方法，並針對他們各自的缺陷去做優化，達到更好的效能」。M9 量到的缺陷清單正好作為設計依據。
- `[需裁決]`：是否開始原型（E1）的設計與實驗卡。

## D021（2026-11-04）M10：查清楚 A100 少了什麼功能，造成合併做法沒有變快

- **使用者要求**：「調查清楚 a100 是少了什麼功能，造成預期比較快的方法在 a100 沒有比較快」。
- **量測前已確認**：TRT-LLM 合併 kernel 每執行緒 76 個暫存器 → hidden 8192（1024 執行緒）超過 A100 每 SM 65,536 → M9 的啟動失敗；A100 沒有 cluster，一列 token 只能給一個 block。
- **設計**：固定資料量、改變 hidden（5120／2560／1280），看「一列一個 block」的 TRT-LLM 合併是否隨 block 變多而變快；vLLM 與 LMDeploy（block 數不變）當對照。另驗暫存器上限（hidden 6400 vs 6656）與 nsys 記錄實際啟動參數。見 `exp/M10_why_fused_slow/README.md`。

## D022（2026-11-04）M10 完成：「沒有 cluster → block 太少」不是合併變慢的主因 `[需裁決]`

- **結果**：同樣資料量給 4 倍 block，TRT-LLM two-shot 合併快 1.27–1.62 倍（仍比 vLLM 慢），one-shot 合併 1.00–1.20 倍；對照組不變 → 依判準 H1 不成立（P1 1/6）。沒有 cluster 確實造成大 hidden 無法啟動（6400 也失敗，與我預期的 6656 不同）。見 `exp/M10_why_fused_slow/README.md`。
- **與預期相反**：P1、P3 不符。
- **影響**：新做法的省時間空間要靠「拿掉 barrier＋少搬資料」，不能只靠多派 block。`[需裁決]`：先量線路使用率與時間組成。

## D023（2026-11-04）核准 M11：新做法動手前的三項量測

- **使用者決定**：「開始量」（看過 `exp/M11_before_prototype/README.md` 的摘要之後）。
- **範圍**：A. 8 對 8 跨卡原子加法（fp16／bf16／fp32）與備案「各寫各的格子」的正確性、誤差、是否每次相同、速度；B. vLLM 0.33.0 加總 kernel 的時間組成（原樣複製、只加計時）；C. nsys 記錄 NVLink 使用率。

## D024（2026-11-04）M11 完成：原子加法正確但試作版比 vLLM 慢；路沒塞滿 `[需裁決]`

- **結果**：8 對 8 半精度原子加法 0 次違反、誤差 1.5e-3，但 5 個大小都比 vLLM「起點 barrier＋第一階段」慢（0/5）；備案 store_slot 也 0/5 → 依判準「兩者都不行，停下來討論設計」。bf16 在 A100 上以 CAS 模擬，出現 347,886 次違反。vLLM 每次加總：barrier 占 kernel 內 34–42%（2-stage），每次之間另有 3.2–4.4 µs 空檔；NVLink 使用率 10–31%。見 `exp/M11_before_prototype/README.md`。
- **與預期不同**：A1（bf16）、A4、B2 不符。
- **資料管理修正**：`.nsys-rep`、`.sqlite` 不再進 git（`.gitignore`）；raw `.nsys-rep` 留在 `results/`，sha256 記在 `results/MANIFEST_large_files.txt`；M10 已 commit 的 `.nsys-rep`／`.sqlite` 改為不追蹤（檔案仍在硬碟，git 歷史中仍有舊版本）；M11 由 nsys 轉出的 13 GB `.sqlite` 放在 `figures/`。
- `[需裁決]`：新做法的設計要怎麼調整。

## D025（2026-11-04）核准 M12：主動送（LL128 式 16 byte）vs 自己去拿

- **使用者決定**：「先做 a 看看，等等跟我講結果」（選項 A：先改良送資料的方式，再以 vLLM 為底逐步改）。
- **設計**：同一個程式內比較 all-gather 與 reduce-scatter 的「送（LL128 格式＋回合記號）」與「拿（vLLM 式 16 byte 讀取）」；另與 M11 的 vLLM 數字對照。見 `exp/M12_push_vs_pull/README.md`。

## D026（2026-11-04）M12 完成：收齊用「送過去＋到齊記號」夠快；分工加總維持「自己去拿」

- **結果**：`ag_push` 與 `ag_pull` 在 80K–640K 差 ≤ 17%（4/5）；`ag_push` 在 80K–320K 比 M11 vLLM「中間等待＋第二階段」快 5.5–7.6 µs（3/5）；`rs_push` 比 `rs_pull` 慢 1.6–2.6 倍（0/5）；16 byte 版比 4 byte 版快 1.8–2.0 倍；0 次違反 → 依判準「主動送夠快」。見 `exp/M12_push_vs_pull/README.md`。
- **與預期不同**：P3 不符；P4 只有 3/5（預期 5/5），640K 以上本程式連 pull 都比 vLLM 慢一倍，疑為測試情境造成，未驗證。
- **影響**：新做法 = vLLM 式分工加總（拿）→ 負責人做 residual＋RMSNorm → 送過去＋到齊記號。下一步：以 vLLM 為底的原型（需使用者核准實驗卡）。

## D027（2026-11-04）M13：查分工加總主動送為何慢；重測大資料量

- **使用者要求**：「為何分工加總主動送比較慢？大資料量（640 KB 以上）這次測不準，這部份你要再去測一次」。
- **設計**：M12 程式加兩種做法（`rs_push_par` 同時等 8 格、`ag_pull_rot` 錯開讀取順序）與連續模式（CUDA graph、不清除緩衝區、回合記號由 block 計數器遞增、接收區雙份輪流）。見 `exp/M13_push_retest/README.md`。

## D028（2026-11-04）M13 完成：依大小切換；M12 大資料量的問題來自測試方式 `[需裁決]`

- **結果**：`rs_push_par` 比 `rs_push` 快 12–30% 但仍比 `rs_pull` 慢 1.26–1.77 倍（P1 0/5，H1 只部分成立）；錯開讀取順序無效（P2）；連續執行下 `ag_pull` 從 16.85／24.79 降到 5.32／8.68 µs（M12 的問題是單次執行＋清除緩衝區）；連續模式 `ag_push` 80K–320K 4.0–4.5 µs，比 vLLM「中間等待＋第二階段」12.2–12.4 µs 快（P5）；640K、1.25M 為 `ag_pull_rot` 的 1.14、1.22 倍（P4 1/2）→ 依判準「依大小切換」。見 `exp/M13_push_retest/README.md`。
- **與預期不同**：P1、P2、P4 不符。
- **判準未涵蓋的事實**：計入「拿」之前必要的全體等待後，`ag_push` 在 640K、1.25M 仍比 vLLM 快 6.9、3.4 µs。`[需裁決]`：大訊息是否也用主動送。

## D029（2026-11-04）核准新做法原型（M14）；大訊息 all-gather 先全部用 push

- **使用者決定**：「好 開始做新方法原形」，並要求列出 A 之後的改善（列在 `exp/M14_fused_prototype/README.md` 的改善路線）。
- **設計**：單一 kernel：start barrier → pull reduce-scatter（依 token 列）→ residual＋RMSNorm（只做自己的列，residual 分片保存）→ push all-gather（LL128＋flag，不用 mid barrier）。D028 的大訊息問題依我的建議先全部用 push，由本實驗與 vLLM 的實際比較做最後確認。
- **判準**（使用者同意的成功標準）：hidden 5120 的 160K／320K／640K 都比 `U_vllm` 快 ≥ 1.3 倍 → go；五個大小都 < 1.15 倍 → no-go；其他 → 討論。

## D030（2026-11-04）M14 完成：原型 go `[需裁決]`

- **結果**（hidden 5120，同一次 run 內和 vLLM 比）：160K／320K／640K 快 1.30／1.56／1.37 倍（區間不重疊、全部正確）→ 依判準 go；80K 1.09、1.25M 1.22 倍。hidden 8192 也能跑（1.11–1.37 倍），TRT-LLM 在此無法啟動。實測時間和 D029 粗估差 ≤ 7%。見 `exp/M14_fused_prototype/README.md`。
- **注意**：160K 剛好在門檻（1.301）；原型有一次 launch 整體慢 1.5–5.4 µs，原因未查；本次 vLLM 比 M9 快約 10%。
- **同日盲點檢討**（已向使用者說明）：實際批次可能讀入與生成混合、只測 fp16／TP8／token ≥ 8、比較對象不是最強、epoch 設計在不同 block 數交替時會出錯、記號可靠性只有觀察、新穎性需要更清楚的機制說明。
- `[需裁決]`：先處理哪些盲點，再進 G2／G3。

## D031（2026-11-04）核准 M15：原型補 bf16

- **使用者要求**：「補一下 bf16」。
- **設計**：kernel 依資料格式產生 fp16／bf16 兩個版本（只有轉換函式不同），測試程式加 `--dtype`；fp16 同時重跑以確認 M14 可重現。判準同 M14 的 go 標準。見 `exp/M15_bf16/README.md`。

## D032（2026-11-04）M15 完成：原型支援 bf16

- **結果**：bf16 全部正確（最大誤差 4.93e-3，vLLM 6.70e-3）；hidden 5120 的 160K／320K／640K 比 vLLM（bf16）快 1.33／1.60／1.39 倍、區間不重疊 → 依判準「bf16 可用」。bf16 與 fp16 倍數相差 ≤ 0.1 的有 9/10。fp16 重現 M14（1.35／1.60／1.34 倍）。見 `exp/M15_bf16/README.md`。
- **與預期不同**：P2 有 1 格（hidden 8192、256K）相差 0.165。
- 原型後續以 `exp/M15_bf16/fused_ar_norm.cu` 為準（fp16／bf16 兩用）。

## D033（2026-11-04）核准 M16：修原型的 epoch 設計

- **使用者要求**：「修漏洞」。
- **修法**：每張 GPU 一個全域 epoch，最後完成的 block 遞增；以交錯測試（不同 block 數、token 數 1–128、eager 與 CUDA graph）驗證，並用舊版重現漏洞；速度回歸檢查對照 M15。見 `exp/M16_epoch_fix/README.md`。
- **使用者另問「進度是否都進 repo」**：本機全部已 commit；從未 push（守則）；GitHub 單檔上限 100 MB，而 git 歷史含 M10 的 291 MB `.sqlite`，日後若要 push 需先處理。

## D034（2026-11-04）改為 push 到 GitHub；改寫 git 歷史移除 291 MB 衍生檔

- **使用者要求**：「把必要的實驗跟程式碼都 push 上 GitHub，我需要每個版本的改動都有紀錄」。之前的守則是「不 push」，依使用者要求改為「commit 後 push」（CLAUDE.md 已更新）。
- **使用者的三個決定**（AskUserQuestion）：repo 維持公開；從歷史中移除超過 GitHub 上限的衍生檔；以 deploy key（本機 `~/.ssh/id_ed25519.pub`，write access）讓本機 push。
- **做法**：先備份完整歷史（`/home/rogerlee/comm_opt_git_backup_20261104.bundle`），再以 `git filter-branch` 移除 `results/M10_why_fused_slow/20261104_083556/*.sqlite`（291 MB、16 MB，衍生檔，仍在本機）。48 個 commit 全部保留，M10 之前的編號不變，之後的編號改變，對照表見 `docs/git_history_rewrite_20261104.md`。remote 改為 `ssh://git@ssh.github.com:443/allennnnnz/comm_opt.git`（本機連不到 port 22）。

## D035（2026-11-04）M16 完成：漏洞修好但量到變慢 `[需裁決]`

- **結果**：舊版在交錯測試中卡住（逾時）；新版 eager／graph 各 60 次交錯呼叫全部正確（含 token 1、3、13）。速度比 M15 慢 9–25%，160K 對 vLLM 1.27／1.24 倍（< 1.3）→ 依判準「修好但變慢」。但未修改的 vLLM 本次也比 M15 慢 3–13%（機器狀態不同）。見 `exp/M16_epoch_fix/README.md`。
- **與預期不同**：P3 不符。
- `[需裁決]`：建議去掉可能不必要的 `__threadfence()`，在同一次 run 內比較 M15／M16／無 fence 版／vLLM。

## D036（2026-11-04）核准 M17：同一次 run 內比較 m15／m16／去掉 fence 的 m16

- **使用者要求**：「查清楚漏洞是什麼、為何修好後影響效能了、修之前量的數據是怎樣」；GitHub 等使用者回去再處理（期間照常本機 commit）。
- **設計**：三個 kernel 版本與 vLLM 在同一次 launch 內交錯量測；去掉 fence 的版本先過交錯正確性測試。見 `exp/M17_epoch_ab/README.md`。

## D037（2026-11-04）M17 完成：採用去掉 fence 的修正版（m16nf）`[需裁決]`

- **結果**：同一次 run 內，`m16` 比 `m15` 慢 1.8–7.2%（只有 1/5 > 5%，P2 不符 → M16 的大幅變慢主要來自機器狀態）；`m16nf` 與 `m15` 差 0.3–3.5%（5/5）且交錯測試 120/120 正確 → 依判準採用 `m16nf`（`exp/M17_epoch_ab/fused_ar_norm_nf.cu`）。
- **對 vLLM 的倍數（修正版）**：80K–1.25M 為 1.02／1.28／1.55／1.32／1.21；160K 從舊版的 1.30–1.35 降到 1.28，略低於 M14 的 go 門檻 1.3。
- **方法上的修正**：同一份程式跨 run 可差 10% 以上，之後的比較一律在同一次 run 內交錯量測。
- `[需裁決]`：是否接受 160K「約 1.3 倍」，或先嘗試把 160K 拉回來。

## D038（2026-11-04）核准 M18：五項改善逐步加入，找最佳組合

- **使用者要求**：「這些先做（改善 1–5）。我要你每個都一點點加入，確定逐漸優化都確實有效才往下加下一個，找出這幾個優化項目中最好的組合」；並問 GitHub 要做什麼才能 push（已回覆：加 deploy key 並勾 write access）。
- **做法**：見 `exp/M18_incremental/README.md`。起點 v0 = m16nf 固定 64 blocks（之前「最佳 block 數」屬事後挑選）；每步先過交錯正確性測試，再於同一次 run 內 5 次 launch A/B；依預先寫好的保留規則（≥ 3% 且區間不重疊才算有效；變差的大小不用新做法）逐步推進。挑選類參數先校準、再用另一次 run 驗證。
- **授權範圍的理解**：使用者要求逐步自動推進，因此每步依保留規則決定、照實記錄，不逐步停下；正確性失敗或無法解釋的現象才停下來問。
- **順序**：a 查清偶爾變慢（先修量測方法，否則小幅改善無法分辨）→ b block 數 → c one-shot → d start barrier → e 大訊息 → 最佳組合確認。

## D039（2026-11-04）交接用的環境建置文件與腳本

- **使用者要求**：「運作環境也要放進去，我希望交接時別人可以順利跑起來」；並要求依進度分 commit 推上 GitHub（已完成：18 個階段 tag `p01`–`p18` 依序 push，進度表 `docs/progress_log.md`）。
- **做法**：不上傳虛擬環境本身（數 GB、綁定機器），改為可重建：`SETUP.md`（交接說明）、`env/setup_python.sh`（依 `requirements-*.txt` 以 uv 重建三個環境）、`env/build_third_party.sh`（依原始指令與 commit 下載編譯外部工具，含 LMDeploy patch）、`env/get_models_assets.sh`（模型釘死 revision）、`env/check_env.py`（逐項檢查、可選 GPU 正確性測試）、`env/system_snapshot.txt`（系統軟體版本）。
- **驗證**：本機 `check_env.py` 全部 OK；三份 requirements 全部可解析安裝；nccl-tests 在乾淨目錄從零編譯成功；其他外部工具在乾淨目錄下載、切換版本、套用 patch 成功（未在乾淨目錄完整重編）。

## D040（2026-11-04）使用者稽核；M18 步驟 a 第一次執行中止與修正

- **使用者稽核**（commit `993afd5`，使用者以 Copilot 進行）：判定目前結果只足以支持「A100、TP=8、CUDA graph、單獨 fused kernel 可行」，不足以支持端到端加速、跨 TP／工作負載泛化、或「無 fence／LL128 旗標」的硬體保證。G2／G3 前的阻擋項：記憶體排序與跨卡發布的壓力測試、baseline 與 block 選擇的公平性（校準／驗證分離）、可達下限的重新界定、M18 步驟 a 完成、實際 vLLM 整合。並在 `analyze_a.py` 加上資料不完整時不下結論的保護。這些阻擋項納入後續計畫。
- **步驟 a 第一次執行中止**：`new_s1` 在暖機時卡住（各 GPU 依自己的時間決定暖機輪數，輪數不同導致全體 all-reduce 對不上），NCCL watchdog 10 分鐘後中止；之後 run 隨工作階段結束中斷。資料保留於 `results/M18_incremental/a_20261104_152116/`。修正暖機迴圈（每輪全體決定是否繼續）後重跑步驟 a。
- **待確認**：小試跑中暖機後 SM 時脈為 1410 MHz（`SETUP.md` 寫的「工作時固定 1155 MHz」可能有誤，待步驟 a 數據確認後更正）。

## D041（2026-11-04）M18 步驟 a 完成：偶爾變慢來自時脈，採用暖機量測

- **結果**：不暖機時 SM 時脈 1155–1395 MHz 且越早量越慢；先全體暖機 2 秒後全部 1410 MHz。v0 64 blocks 在 2–64 tokens 的 launch 間最大／最小：舊 1.06–1.21、新 1.02–1.05 → 依判準採用新方式。另外每次 launch 的第一個設定一定特別慢（舊約 6 倍、新 22–50 倍），步驟 b 起先量一個不記錄的暖身設定。見 `exp/M18_incremental/README.md`。
- **更正**：`SETUP.md` 原寫「GPU 工作時固定 1155 MHz」有誤，已更正。M14–M17 是在較低時脈下量的；同一次 run 內的比較仍有效。
- **參考（非預先判準）**：1410 MHz 下 v0 64 blocks 對 vLLM：1–4 tokens 0.85–0.97 倍（較慢），16／32／64／128 tokens 1.22／1.71／1.41／1.38 倍。

## D042（2026-11-04）M18 步驟 b 完成：依大小選 block 數無效，不採用；依使用者要求暫停

- **結果**：校準（3 次）得到每個 token 數的最佳 block 數表；驗證（5 次新的 launch）中，表格版對固定 64 blocks 在 8 個大小都沒有「快 ≥ 3% 且區間不重疊」（最好 −3.4%，64 tokens，區間重疊）→ 依保留規則不採用。預期（≥ 2 個有效、16 tokens 對 vLLM ≥ 1.3）不成立：16 tokens 1.23–1.24 倍。見 `exp/M18_incremental/README.md`。
- **使用者指示**：「跑完 b 之後暫停所有事，我們需要重新盤點當前的狀態，好好整理並做出下一步的策略」。M18 步驟 c–e 暫停，改做全面盤點（`docs/strategy_review_20261104.md`）。

## D043（2026-11-04）全面盤點與策略提案 `[需裁決]`

- **使用者要求**：步驟 b 之後暫停，重新盤點並定策略。
- **文件**：`docs/strategy_review_20261104.md`（證據分為扎實／有條件／被推翻；稽核阻擋項狀況；方法教訓；定位風險；四個策略選項）。
- **建議**：先 1–2 天補地基（壓力測試、最強對照組、可達水準重新定義、使用情境調查）並加入小訊息 one-shot，再做 vLLM 端到端，用端到端結果決定系統論文或分析論文，之後過 G2。
- `[需裁決]`：策略、第一段項目、端到端門檻、是否補主線的正式 G1。
