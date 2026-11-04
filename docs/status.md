# 目前狀態

> 每次 session 開始先讀本檔，結束時更新「目前狀態」與「下一步」。

## 目前狀態（2026-11-03）

- **階段**：1 找現象 → 選題前
- **已完成**：M0、M4、M2_p2p_channels、M5_e2e_comm_share、文獻對照（D011）、M6_allreduce_bakeoff（D013：值得實作）、M7_where_slow（D015：需要新的做法）、M8_ll128_atomic_a100（D017：LL128 atomic 前提在 A100 上很可能成立）、「加總＋正規化合併」查核（D018：A100 上已有 TensorRT-LLM 與 LMDeploy 的實作；空缺是兩輪免 barrier 的合併 two-shot）、M9_fused_baselines（D020：A100 上現有合併沒有比較快，離可達水準仍有 2 倍以上）、M10_why_fused_slow（D022：block 太少不是主因）、M11_before_prototype（D024：原子加法正確但試作版比 vLLM 慢；路沒塞滿）、M12_push_vs_pull（D026：收齊用主動送夠快，分工加總維持拿）
- **進行中**：M13（分工加總主動送為何慢、大資料量重測）

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
| M13_push_retest | 2026-11-04（D027） | 進行中 |

## 下一步

1. 使用者裁決 D022：量線路使用率（nsys GPU metrics）與時間組成（kernel 內時間戳記）；量跨卡原子加法 8 對 8、半精度的速度與準確度
2. 之後：新做法原型（設計草案已向使用者說明：分工直接加進負責人記憶體＋計數器、負責人只正規化自己那 1/8、帶記號發回、依機器挑做法；第二階段加 FLUX 式矩陣計算內直接傳）
3. 若進入選題：用 brainstorming skill 寫研究問題與 claims（G2）；候選核心＝兩輪免 barrier 的合併 two-shot（A100，32-bit 原子加法）＋依機器實測挑做法
