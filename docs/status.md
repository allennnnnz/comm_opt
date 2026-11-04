# 目前狀態

> 每次 session 開始先讀本檔，結束時更新「目前狀態」與「下一步」。

## 目前狀態（2026-11-04，暫停中）

- **暫停**：依使用者要求，M18 步驟 b 之後暫停所有實驗，完成全面盤點與策略提案：**`docs/strategy_review_20261104.md`**（等待使用者決定）
- **階段**：1 找現象 → 選題前。原型可行性已通過，尚未過 G2（題目）／G3（claims）
- **研究主線**：生成階段 TP=8 的 all-reduce＋residual＋RMSNorm，8×A100（沒有 NVLS）
- **原型**（`exp/M17_epoch_ab/fused_ar_norm_nf.cu`，fp16／bf16，全域 epoch）：start barrier → pull reduce-scatter → residual＋RMSNorm → LL128 push all-gather。1410 MHz、同一次量測：16／32／64／128 tokens 比 vLLM 快 1.23／1.70／1.39／1.38 倍；1–4 tokens 較慢（0.87–0.98）；固定 64 blocks（依大小選 block 數無效，M18 步驟 b）
- **量測方法**：M18 起先全體暖機 2 秒（時脈 1410 MHz）、丟掉每次第一個設定、比較一律同一次量測內交錯
- **稽核阻擋項**（D040）：記憶體順序壓力測試、最強對照組、可達水準重新定義、vLLM 整合——未做；步驟 a——完成
- **已完成的量測**：M0–M17、M18 步驟 a／b；摘要見 `docs/findings_overview.md`、`docs/progress_log.md`
- **GitHub**：已設定並同步（每次 commit 後 push）

## 已核准的實驗

| ID | 核准日期 | 狀態 |
|---|---|---|
| M0_baseline | 2026-11-02（D004） | 完成 |
| M4_repeat | 2026-11-02（D006） | 完成 |
| M2_p2p_channels | 2026-11-02（D006） | 完成 |
| M5_e2e_comm_share | 2026-11-02（D009） | 完成 |
| M6_allreduce_bakeoff | 2026-11-03（D012） | 完成 |
| M7_where_slow | 2026-11-03（D014） | 完成 |
| M8_ll128_atomic_a100 | 2026-11-03（D016） | 完成 |
| M9_fused_baselines | 2026-11-04（D019） | 完成 |
| M10_why_fused_slow | 2026-11-04（D021） | 完成 |
| M11_before_prototype | 2026-11-04（D023） | 完成 |
| M12_push_vs_pull | 2026-11-04（D025） | 完成 |
| M13_push_retest | 2026-11-04（D027） | 完成 |
| M14_fused_prototype | 2026-11-04（D029） | 完成 |
| M15_bf16 | 2026-11-04（D031） | 完成 |
| M16_epoch_fix | 2026-11-04（D033） | 完成（修好但變慢，D035） |
| M17_epoch_ab | 2026-11-04（D036） | 完成（採用 m16nf，D037） |
| M18_incremental | 2026-11-04（D038） | 暫停（步驟 a、b 完成；D041、D042） |

## 下一步

1. **等待使用者**：決定 `docs/strategy_review_20261104.md` 的策略（建議：B 精簡版＋小訊息 one-shot → vLLM 端到端 → 依結果決定系統論文或分析論文）與第 7 節的四個問題
2. 是否清除 git 裡未被引用的殘留物件（約 2.5 GB，本機）
