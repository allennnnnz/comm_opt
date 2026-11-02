# 研究守則

本守則整理自下列公認的研究、量測與寫作方法。每條規則後面的標記是出處；標「本專案」的是我們依這台機器與合作方式自訂的。完整出處見文末。

| 標記 | 來源 | 主要用在 |
|---|---|---|
| H | Hoefler & Belli, *Scientific Benchmarking of Parallel Computing Systems*, SC'15（12 條規則） | 量測統計與報告 |
| B | Heiser, *Systems Benchmarking Crimes* | 評估與比較 |
| S | Berger, Blackburn, Hauswirth, Hicks, *SIGPLAN Empirical Evaluation Checklist*, 2018 | claim 與實驗設計 |
| M | Mytkowicz, Diwan, Hauswirth, Sweeney, *Producing Wrong Data Without Doing Anything Obviously Wrong!*, ASPLOS'09 | 量測偏差 |
| T | Sandve, Nekrutenko, Taylor, Hovig, *Ten Simple Rules for Reproducible Computational Research*, PLOS CB 2013 | 可重現性 |
| A | ACM *Artifact Review and Badging* v1.1 | artifact 準備 |
| L | Levin & Redell, *How (and How Not) to Write a Good Systems Paper*, OSR 1983 | 選題與寫作 |
| P | Peyton Jones, *How to Write a Great Research Paper* | 寫作 |
| N | Hu et al., *Demystifying NCCL*, 2025；Pasqualoni et al., *PICO*, 2025 | collective 量測的變因 |

---

## 一、研究流程與關卡

研究分成七個階段，每個階段結束有一個關卡（G）。過關卡需要使用者核准，結果記在 `docs/decisions.md`。

| 階段 | 做什麼 | 過關條件 |
|---|---|---|
| 0 量測 | 用 nccl-tests 等工具量出本機通訊特性（M 系列） | — |
| 1 找現象 | 從量測找出「異常、浪費、與理論上限的差距」 | **G1**：現象清單，每項附證據路徑與重複量測結果；使用者選定要追哪個 |
| 2 選題 | 用 brainstorming skill 發散再收斂；做新穎性搜尋 | **G2**：兩句話 pitch、新穎性搜尋紀錄、可行性評估；核准後寫入 `docs/project.md` |
| 3 設計 | 定 claims、baseline、評估方法 | **G3**：每個 claim 對應一個實驗與成功標準 |
| 4 實作 | 寫系統或方法 | — |
| 5 評估 | 依 G3 的計畫跑 E 系列實驗 | **G4**：逐條判定 claim 是否被支持，不支持就記錄並回到階段 3 |
| 6 寫作 | 從 G2 起就開始寫（見 W4） | — |
| 7 自我審查 | 逐項過第六節的清單；準備 artifact | **G5**：清單全部通過才投稿 |

---

## 二、量測規則

- **R1 先寫預期再量測。** 每個實驗卡在跑之前寫下「預期結果」和「什麼結果會推翻預期」；跑完不得修改預期，只能在「結論」欄對照。（本專案；呼應 S 的「claims 要明確」）
- **R2 完整記錄環境與變因。** 記錄所有變動的因子及其水準、固定的條件、完整軟硬體環境（GPU、拓樸、driver、CUDA、NCCL、nccl-tests commit、時脈、溫度）。（H9、B「缺平台規格」、T3）
- **R3 區分確定與非確定的數據。** 非確定性數據要重複量測並報信賴區間。預設停止條件：重複到中位數的 99% 信賴區間落在 ±5% 以內；做不到就報實際的區間。（H5）
- **R4 不假設常態分布。** 用中位數與無母數信賴區間，或先做常態性檢查再用平均數加減標準差。（H6）
- **R5 比較要有統計依據。** 判定 A 比 B 快，要用不重疊的信賴區間或檢定，不能只比兩個平均值。（H7、B「沒有顯著性」）
- **R6 依問題選統計量。** 延遲類問題要看 p99 或最大值，不只看中位數。（H8）
- **R7 寫清楚平行量測怎麼彙總。** 跨 rank 取 max、平均還是 rank 0，有沒有先同步，都要寫明（nccl-tests 的 `-a`、`-z` 參數）。（H10）
- **R8 標出理論上限。** 圖表要畫出硬體上限（例如 NVLink 頻寬），讓讀者看得出離上限多遠。（H11）
- **R9 正確地彙總。** 成本（時間）用算術平均；速率（頻寬）用調和平均；盡量不要彙總比值，非彙總不可才用幾何平均。（H3、H4、B「用算術平均彙總分數」）
- **R10 一定附絕對數字。** 報加速比或百分比時，同時給基準的絕對值。（H1、B「只報相對數字」）
- **R11 防範量測偏差。** 改變看似無關的設定（GPU 組合、執行順序、程序啟動順序），確認結論不變；正式實驗的執行順序要隨機化。（M）
- **R12 一次只改一個變因**，其他固定並寫明。（本專案）
- **R13 不打擾其他工作。** 量測前確認機器上沒有任何其他 GPU 工作，即使不在同一張卡；有就等它結束，不中斷、不搶用。（本專案）
- **R14 探索性量測與結論分開。** 單次跑的探索性量測只能用來找方向，要寫成結論前必須依 R3 重複。（本專案）

## 三、評估與比較規則

- **E1 不挑對自己有利的情境。** 不能只挑會進步的 benchmark 或輸入範圍；會退化的情況也要報。（B「選擇性 benchmark」、S）
- **E2 只用子集合要給理由。**（H2、B）
- **E3 micro-benchmark 不能代表整體。** nccl-tests 的結論不能直接外推到端到端效能；系統層級的 claim 要用真實 workload 驗證。（B）
- **E4 baseline 要是最新且調過參數的。** 例如最新版 NCCL 加上合理調整的環境變數，不只跟自己的舊版本比。（B、S）
- **E5 對手系統用它的最佳設定，並公開設定。**（B）
- **E6 校準和驗證用不同的資料。** 模型或門檻用來擬合的數據，不能再拿來證明它準。（B）
- **E7 報細項，不只報總分。**（B）
- **E8 寫出限制。**（S）

## 四、可重現性與數據管理

- **D1 每個結果都能追溯產生方式**：腳本、參數、git commit、環境快照。（T1）
- **D2 不手動修改數據。** `results/` 下的原始數據不可手改、不可刪；重跑就新增時間戳目錄。（T2）
- **D3 記錄所有外部程式的確切版本。**（T3）
- **D4 所有腳本都進版本控制。**（T4）
- **D5 保存中間結果，用標準格式（JSON、CSV）。**（T5）
- **D6 有隨機性就記錄 seed。**（T6）
- **D7 每張圖都保存背後的原始數據。** 圖一律由腳本從 `results/` 產生，不手動編輯。（T7）
- **D8 分析輸出分層**：總表 → 細項 → 原始數據，可逐層往下查。（T8）
- **D9 文字裡的每個數字都連到對應結果。** 筆記和論文提到的數字要附 `results/` 路徑。（T9）
- **D10 以 ACM「Artifacts Available」與「Artifacts Evaluated – Functional」為目標**，從第一天就照這個標準整理腳本與文件。（A、T10）

## 五、選題與寫作

- **W1 點子要能用一兩句話講清楚，而且和既有工作有實質差異。** 用 brainstorming skill 的「兩句話測試」檢查。（L）
- **W2 宣稱新穎之前先搜尋。** 至少搜過 arXiv 與 dblp 或 OpenAlex，把關鍵字和結果記在 `lit/`。（本專案）
- **W3 引用要準確。** BibTeX 只用 arXiv MCP 或 DOI 匯出的，不憑記憶寫；讀過的論文在 `lit/notes/` 留筆記。（L、本專案）
- **W4 早點開始寫。** G2 一通過就建 paper 骨架，先寫 contributions 清單；每條 contribution 對應一個證據。（P）
- **W5 讀者優先、說一個故事。** 不照研究過程的時間順序寫。（P）
- **W6 圖只畫足以解讀的資訊。** 只有在表示趨勢且內插有意義時才把點連成線。（H12）

## 六、投稿前自我審查清單（G5）

逐項勾選，每項附證據或理由：

- [ ] Hoefler 12 條（H1–H12）
- [ ] Heiser benchmarking crimes 五大類：選擇性 benchmark、結果處理不當、用錯 benchmark、比較不當、資訊缺漏
- [ ] SIGPLAN checklist：claim 明確、實驗設計合適、指標相關、baseline 合適、寫出限制
- [ ] 量測偏差（M）：換 GPU 組合和執行順序後，結論不變
- [ ] Artifact：照 README 從頭重跑，能重現每張圖

## 七、合作方式（使用者與 Claude）

**Claude 可以自己做：**
- 執行 `docs/status.md` 裡標為「已核准」的實驗，並分析、畫圖、寫實驗卡的結論
- 搜尋文獻、讀論文、寫筆記
- 修 bug、改進腳本（不改變實驗設計）

**必須停下來問使用者：**
- 新增實驗，或改變已核准實驗的設計（量什麼、怎麼量）
- 修改 `docs/project.md` 的研究問題、claims 或 non-goals
- 結果和預期或既有 claim 矛盾時：寫進 `docs/decisions.md` 並標 `[需裁決]`，不自己改 claim
- 任何關卡（G1–G5）
- 任何會影響其他使用者或其他工作的操作

---

## 參考資料

- Hoefler & Belli, SC'15 — [論文 PDF](https://htor.inf.ethz.ch/publications/img/hoefler-scientific-benchmarking.pdf)、DOI [10.1145/2807591.2807644](https://doi.org/10.1145/2807591.2807644)
- Heiser — [Systems Benchmarking Crimes](https://gernot-heiser.org/benchmarking-crimes.html)
- SIGPLAN — [Empirical Evaluation Guidelines](https://www.sigplan.org/Resources/EmpiricalEvaluation)
- Mytkowicz et al., ASPLOS'09 — DOI [10.1145/1508244.1508275](https://doi.org/10.1145/1508244.1508275)
- Sandve et al., PLOS CB 2013 — [PMC3812051](https://pmc.ncbi.nlm.nih.gov/articles/PMC3812051/)
- ACM — [Artifact Review and Badging 說明](https://reviewers.acm.org/training-course/artifact-review-and-badging)、[2020 年術語變更](https://www.acm.org/publications/badging-terms)
- Levin & Redell, OSR 1983 — [重印版 PDF](https://www.cs.wm.edu/~smherwig/readings/papers/83-osr-how_to_write_a_systems_paper.pdf)
- Peyton Jones — [How to write a great research paper](https://www.microsoft.com/en-us/research/?p=186470)
- Hu et al., *Demystifying NCCL* — [arXiv:2507.04786](https://arxiv.org/abs/2507.04786)；Pasqualoni et al., *PICO* — [arXiv:2508.16809](https://arxiv.org/abs/2508.16809)
