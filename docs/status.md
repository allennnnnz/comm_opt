# 目前狀態

> 每次 session 開始先讀本檔，結束時更新「目前狀態」與「下一步」。

## 目前狀態（2026-11-04）

- **階段**：1 找現象 → 選題前。新做法原型已通過可行性（M14 go、M15 bf16 可用），尚未過 G2（題目）／G3（claims）
- **研究主線**：生成階段 TP=8 的 all-reduce＋residual＋RMSNorm（80 KB–1.3 MB），A100（沒有 NVLS）
- **原型**（正式版 `exp/M17_epoch_ab/fused_ar_norm_nf.cu`，fp16／bf16，全域 epoch、已修交錯漏洞）：start barrier → pull reduce-scatter（依 token 列）→ residual＋RMSNorm（residual 分片）→ LL128 push all-gather（不用 mid barrier）。比 vLLM（custom AR＋fused_add_rms_norm）快 1.30–1.60 倍（160–640 KB），8 人與 128 人約 1.1–1.24 倍；hidden 8192 可用（TRT-LLM 不行）
- **原型未完成項目**：token < 8 或非 8 倍數的速度（功能已確認正確）、160K 倍數 1.28 略低於 1.3、偶爾整次 launch 慢 1.5–5 µs、只測 TP=8、只在 standalone 測試程式（未進 vLLM）
- **本次全面盤點的判定**：目前的性能結果足以支持「A100、TP=8、CUDA graph、isolated fused kernel 的可行性」，尚不足以支持端到端加速、跨 TP／工作負載泛化、或無 fence／LL128 旗標的硬體保證。G2/G3 前必須先處理記憶體排序與正確性壓力測試、baseline／block 選擇的公平性、可達下限的定義，以及實際 vLLM 整合。
- **已完成的量測**：M0、M4、M2、M5、文獻（D011、D018）、M6–M15，摘要見 `docs/findings_overview.md`
- **M18 step a 狀態**：目前只有 `results/M18_incremental/a_20261104_152116/old_s1.json`，old/new 五次交錯量測尚未完成；不能據此判定暖機是否解釋偶發變慢
- **等待使用者**：D037（160K 是否接受約 1.3 倍）；GitHub deploy key 由使用者回去後設定，之後 push（期間只在本機 commit）；是否清除 git 裡未被引用的殘留物件

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
| M18_incremental | 2026-11-04（D038） | 進行中（步驟 a） |

## 下一步

1. ~~修原型的 epoch 設計~~（已完成：M16、M17，採用 m16nf）
2. 先完成盤點列出的阻擋項：m16nf 的 memory-ordering／跨卡發布壓力測試；以校準／驗證分離重做 block 與 baseline 比較；重新界定 attainable bound；M18 step a 完成五次 old/new 結果並分析時脈混淆
3. 盲點檢討（D030）中影響方向的項目：實際批次組成（讀入與生成混合、TP4 vs TP8，可先不用 GPU 調查）、最強對照組（M7 的最佳現成做法＋norm）、token < 8 的處理、偶爾變慢的原因
4. 以上確認後：G2（題目、兩句話 pitch、新穎性與可行性）→ G3（claims 與成功標準）→ E 系列（vLLM 端到端）
4. 之後的改善路線（`exp/M14_fused_prototype/README.md`）：依大小挑做法、依機器選 block 數、拿掉 start barrier、FLUX 式與前後 GEMM 合併等
