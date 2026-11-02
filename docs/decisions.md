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
