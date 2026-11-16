# comm_opt

一篇 GPU 間通訊優化論文的完整研究過程：量測 → 找現象 → 選題 → 實驗 → 寫作。題目尚未決定（見 `docs/project.md`）。

**2026-11-16 起在 `nano5` 分支、國網中心 Nano5（8× H100／H200 每台）重新開始**（計畫：`docs/plan_nano5.md`，D050）。A100 時期（M0–M19）保留在 `main`、標記 `a100-final`。

## 每次 session

- **開始**：讀 `docs/status.md`。需要背景時再讀 `docs/project.md`、`docs/decisions.md`、`docs/research_rules.md`。
- **結束**：更新 `docs/status.md` 的「目前狀態」與「下一步」；有新決策就附加到 `docs/decisions.md`（只能新增，不改舊條目）；commit。

## 必守規則（完整版與出處：`docs/research_rules.md`）

1. **從零開始**：不讀、不沿用 `~/TP_opt`、`~/tp_opt`、`~/flux` 的研究內容。硬體事實一律重新量測。
2. **不打擾其他工作**：量測前確認機器上沒有任何其他 GPU 工作（即使在別張卡）；有就等它結束，不中斷、不搶用（R13）。
3. **先寫預期再量測**，跑完不改預期（R1）。單次探索不能當結論，結論要重複量測並報信賴區間（R3、R14）。
4. **數據不可手改**：`results/` 原始數據不改、不刪；每個數字要能追到 `results/` 路徑、腳本、commit（D1、D2、D9）。
5. **引用只用 arXiv MCP 或 DOI 匯出的 BibTeX**，不憑記憶寫（W3）。
6. **決策權**：已核准的實驗可自己跑和分析；新增或改變實驗、改研究問題或 claims、過關卡（G1–G5）、結果和預期矛盾時，都要停下來問使用者（第七節）。
7. **在 Nano5 叢集上，運算一律經 Slurm 送到計算節點**，照 skill `.claude/skills/slurm-submit/`（模板 `template.sbatch`、送出用 `submit.sh`）；登入節點不跑運算；不輪詢 `squeue`；量測一律獨佔整台（D048）。
8. **分工：Opus 規劃、Sonnet 執行（D049，完整版見 `docs/research_rules.md` 第八節）**：規劃、寫規格書、寫實驗預期、解讀結果、所有研究決策與寫入 `docs/decisions.md`／`status.md`／`project.md`，一律由主對話的 Opus 5.5 做。需要大量篇幅但不需深度推理的執行工作（文獻閱讀與摘要、程式碼掃描、寫或改腳本、安裝編譯、送已核准的工作、整理數據）交給 Sonnet 5.5 小助手（Agent 工具一律指定 `model: "sonnet"`），照 `docs/specs/S<NN>_<名稱>.md` 規格書執行、產出 `docs/reports/R<NN>_<名稱>.md` 報告。影響決策的論點，Opus 必須自己回原始碼、原文或 `results/` 抽查後才能用。第 6 條使用者的決策權不變。

## 慣例

- **語言**：**給使用者的所有回覆一律用繁體中文**（使用者已兩次要求）。守則、實驗卡、筆記、決策用繁體中文，技術名詞保留英文；`paper/` 用英文。
- **結論寫法**：每個實驗卡和每次回報結果，都先給「白話結論」——用中文、高層次、3–5 點，讓使用者不看數字也能快速懂：發現了什麼、代表什麼、下一步要做什麼。數字、表格、細節放在後面。
- **白話結論不用英文術語與代號**（不寫 channel、chunk、SM、F1、G1、commit 這類詞）；非用不可的概念用中文說明，可以用比喻（例如 channel →「傳輸通道，像高速公路的車道」）。回報給使用者的訊息整體也照這個原則。
- **Git**：每完成一個段落（一個實驗、一次分析、一個決策）就在 **`nano5` 分支**上 commit（2026-11-16 起；`main` 停在 `a100-final`，不再改動），訊息寫清楚做了什麼、數據在哪，**然後 push 到 GitHub**（`origin` = `allennnnnz/comm_opt`，公開 repo，經 `ssh.github.com:443`，本機 SSH 別名 `github-comm-opt`；使用者 2026-11-04 要求「每個版本的改動都有紀錄」）。單檔 > 100 MB 不能進 git（`.nsys-rep`／`.sqlite` 已在 `.gitignore`，raw 檔的 sha256 記在 `results/MANIFEST_large_files.txt`）。2026-11-04 曾改寫歷史，舊 commit 編號對照見 `docs/git_history_rewrite_20261104.md`。每完成一個研究階段，加一個 annotated tag（`pNN-<名稱>`）並更新 `docs/progress_log.md`。
- **實驗 ID**：`M*` = 環境特性量測；`E*` = 驗證題目假說的實驗。`S*` = 規格書、`R*` = 對應的報告（同編號，例如 S01 → R01）。

## 資料夾地圖

| 路徑 | 內容 |
|---|---|
| `docs/project.md` | 研究問題、claims、non-goals（G2 後才填） |
| `docs/status.md` | 目前進度、下一步、已核准的實驗 |
| `docs/plan_nano5.md` | Nano5 階段的研究計畫 |
| `docs/decisions.md` | 決策紀錄（append-only） |
| `docs/research_rules.md` | 研究守則與出處 |
| `docs/specs/S<NN>_<名稱>.md` | Opus 寫給小助手的工作規格書 |
| `docs/reports/R<NN>_<名稱>.md` | 小助手照規格書交回的報告 |
| `exp/<ID>_<名稱>/` | 一個實驗一個資料夾：`README.md`（實驗卡，照 `exp/TEMPLATE.md`）、`run.sh`、`analyze.py` |
| `results/<ID>_<名稱>/<時間>/` | 原始數據（不可手改） |
| `figures/` | 由腳本產生的圖 |
| `lit/refs.bib`、`lit/notes/` | 參考文獻與閱讀筆記 |
| `paper/` | 論文（LaTeX，英文） |
| `third_party/` | nccl-tests 等外部工具（不進 git，版本記在實驗卡） |
| `papers/` | 下載的 PDF（不進 git） |
| `.claude/skills/`、`.mcp.json` | 研究用 skills 與論文搜尋 MCP（來源見 `.claude/SKILL_SOURCES.md`） |
