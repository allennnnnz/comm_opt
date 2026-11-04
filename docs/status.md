# 目前狀態

> 每次 session 開始先讀本檔，結束時更新「目前狀態」與「下一步」。

## 目前狀態（2026-11-04）

- **階段**：1 找現象 → 選題前。新做法原型已通過可行性（M14 go、M15 bf16 可用），尚未過 G2（題目）／G3（claims）
- **研究主線**：生成階段 TP=8 的 all-reduce＋residual＋RMSNorm（80 KB–1.3 MB），A100（沒有 NVLS）
- **原型**（`exp/M15_bf16/fused_ar_norm.cu`，fp16／bf16）：start barrier → pull reduce-scatter（依 token 列）→ residual＋RMSNorm（residual 分片）→ LL128 push all-gather（不用 mid barrier）。比 vLLM（custom AR＋fused_add_rms_norm）快 1.30–1.60 倍（160–640 KB），8 人與 128 人約 1.1–1.24 倍；hidden 8192 可用（TRT-LLM 不行）
- **原型未完成項目**：epoch 設計（不同 block 數交替會對不上，整合前必修）、token < 8 或非 8 倍數、偶爾整次 launch 慢 1.5–5 µs、只測 TP=8、只在 standalone 測試程式（未進 vLLM）
- **已完成的量測**：M0、M4、M2、M5、文獻（D011、D018）、M6–M15，摘要見 `docs/findings_overview.md`
- **等待使用者**：原型待辦與盲點的處理順序（D030 列表）；是否清除 git 裡未被引用的約 2.5 GB 殘留物件

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

## 下一步

1. 修原型的 epoch 設計（改成所有 block 一致的全域 epoch），之後才能整合進 vLLM
2. 盲點檢討（D030）中影響方向的項目：實際批次組成（讀入與生成混合、TP4 vs TP8，可先不用 GPU 調查）、最強對照組（M7 的最佳現成做法＋norm）、token < 8 的處理、偶爾變慢的原因
3. 以上確認後：G2（題目、兩句話 pitch、新穎性與可行性）→ G3（claims 與成功標準）→ E 系列（vLLM 端到端）
4. 之後的改善路線（`exp/M14_fused_prototype/README.md`）：依大小挑做法、依機器選 block 數、拿掉 start barrier、FLUX 式與前後 GEMM 合併等
