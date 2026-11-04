# 研究進度紀錄（依階段）

每個階段在 git 有一個標籤（tag），指向該階段最後一個 commit；在 GitHub 上點標籤即可看到當時的完整狀態。
commit 編號為 2026-11-04 歷史改寫後的編號（改寫前後對照見 `docs/git_history_rewrite_20261104.md`）。

| 階段標籤 | commit 範圍 | 內容 | 主要結論 | 主要文件 |
|---|---|---|---|---|
| `p01-setup` | `5e904c4`（第 1 個） | 研究守則與專案架構 | 建立研究守則（含出處）、資料夾結構、實驗卡範本 | `docs/research_rules.md` |
| `p02-M0-baseline` | `82bc907`–`033c798`（第 2–3 個） | M0：nccl-tests 基準量測 | 機器健康；找到 7 個候選現象，最強的是 8 卡一對一傳輸只有 2 卡的三分之一 | `exp/M0_baseline/README.md` |
| `p03-M4-M2-G1` | `08a6628`–`23d1386`（第 4–7 個） | M4／M2：重複量測與通道機制；G1 現象清單 | F1 確認（36–41%），通道分配機制從 NCCL 原始碼完全解釋；改設定只能部分修好 | `docs/gates/G1_report.md` |
| `p04-M5-e2e-share` | `f6733b8`–`81b17dd`（第 8–11 個） | M5：端到端通訊佔比 | 通訊佔推論 8–30%、訓練 11–22%；生成階段加總同步佔每步 8–27% | `exp/M5_e2e_comm_share/README.md` |
| `p05-lit-direction` | `719e829`–`dfdd918`（第 12–14 個） | 文獻對照，選定方向 | 量測對照文獻，方向定為「生成階段的加總同步（all-reduce）」 | `lit/notes/survey_20261103.md` |
| `p06-M6-bakeoff` | `e59e9ba`–`a4162f9`（第 15–16 個） | M6：現有 all-reduce 做法比較 | 64 KB–1 MB 最快的現成做法仍比可達水準慢 1.5–2.1 倍 | `exp/M6_allreduce_bakeoff/README.md` |
| `p07-M7-where-slow` | `cf85f43`–`289dacc`（第 17–19 個） | M7：找慢在哪 | NCCL 在 A100 選錯運算單元數；最佳組合仍慢 1.44–1.91 倍 → 需要新做法 | `exp/M7_where_slow/README.md` |
| `p08-M8-ll128-atomic` | `9be9471`–`60aa7ed`（第 20–22 個） | M8：「到齊記號」原子加法在 A100 是否成立 | 成立（0 次違反），但 A100 沒有 16 byte 向量原子加法 | `exp/M8_ll128_atomic_a100/README.md` |
| `p09-lit-fused-check` | `419dd0f`（第 23 個） | 文獻：加總＋正規化合併是否已有人做 | TRT-LLM、LMDeploy 有 A100 可走的合併；vLLM、SGLang 在 A100 不開 | `lit/notes/fused_ar_rmsnorm_a100.md` |
| `p10-M9-fused-baselines` | `1f52d7e`–`1299baa`（第 24–27 個） | M9：量別人的合併做法 | 在 A100 上沒有比 vLLM 分開做快；離可達水準 2.2–3.1 倍 | `exp/M9_fused_baselines/README.md` |
| `p11-M10-why-fused-slow` | `757526c`–`f8176ab`（第 28–29 個） | M10：合併為何沒變快 | 缺 cluster 不是主因，但造成大模型無法啟動；問題在搬資料方式 | `exp/M10_why_fused_slow/README.md` |
| `p12-M11-before-prototype` | `9cdb6ce`–`dc1c762`（第 30–33 個） | M11：動手前檢查＋整體整理 | 路沒塞滿（10–31%）；時間花在等待與空檔；4 byte 原子推送比 vLLM 慢 | `exp/M11_before_prototype/README.md` |
| `p13-M12-push-vs-pull` | `f4a2748`–`930a82b`（第 34–36 個） | M12：主動送 vs 自己去拿 | 收齊適合主動送（不用中間等待）；分工加總維持自己去拿 | `exp/M12_push_vs_pull/README.md` |
| `p14-M13-push-retest` | `02a5e60`–`f9d4ef5`（第 37–39 個） | M13：重測大資料量 | M12 大資料量的問題來自測試方式；連續執行下主動送收齊比 vLLM 快約 8 µs | `exp/M13_push_retest/README.md` |
| `p15-M14-prototype` | `d645bf3`–`68340d7`（第 40–42 個） | M14：新做法原型 | 成功：160／320／640 KB 比 vLLM 快 1.30／1.56／1.37 倍 | `exp/M14_fused_prototype/README.md` |
| `p16-M15-bf16` | `ade7db9`–`e06439f`（第 43–46 個） | M15：原型支援 bf16＋盲點清單 | bf16 快 1.33–1.60 倍；整理 12 項盲點 | `exp/M15_bf16/README.md` |
| `p17-M16-epoch-fix` | `0146bc6`–`d9e289d`（第 47–50 個） | M16：修交錯使用時的漏洞；改為 push 到 GitHub | 漏洞重現並修好；跨次量測看起來變慢（後證實多為機器狀態） | `exp/M16_epoch_fix/README.md` |
| `p18-M17-epoch-ab` | `70bc4ff`–`badc968`（第 51–52 個） | M17：同一次量測比較新舊版 | 修法代價 0.3–3.5%；正式修正版對 vLLM 1.28／1.55／1.32 倍 | `exp/M17_epoch_ab/README.md` |
| （進行中，尚未標籤） | `aec5efd`– | M18：五項改善逐步加入 | 進行中 | `exp/M18_incremental/README.md` |
