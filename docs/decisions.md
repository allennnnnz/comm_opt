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
