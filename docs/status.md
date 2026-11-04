# 目前狀態

> 每次 session 開始先讀本檔，結束時更新「目前狀態」與「下一步」。

## 目前狀態（2026-11-03）

- **階段**：1 找現象 → 選題前
- **已完成**：M0、M4、M2_p2p_channels、M5_e2e_comm_share、文獻對照（D011）、M6_allreduce_bakeoff（D013：值得實作）、M7_where_slow（D015：需要新的做法）、M8_ll128_atomic_a100（D017：LL128 atomic 前提在 A100 上很可能成立）、「加總＋正規化合併」查核（D018：A100 上已有 TensorRT-LLM 與 LMDeploy 的實作；空缺是兩輪免 barrier 的合併 two-shot）
- **等待使用者**：看過 `exp/M9_fused_baselines/README.md`（提案）才開始量；是否放行在本機編譯 LMDeploy（被安全檢查擋下）

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
| M9_fused_baselines | 2026-11-04（D019） | 進行中 |

## 下一步

1. M9：使用者看過實驗卡後開始量；FlashInfer 的 TRT-LLM 合併原始碼另編 sm_80 中（試編）；LMDeploy 編譯待放行
2. 待議（使用者先前提出）：M9 GPU 核心與連線使用率量測（ncu／nsys）
3. 若進入選題：用 brainstorming skill 寫研究問題與 claims（G2）；候選核心＝兩輪免 barrier 的合併 two-shot（A100，32-bit 原子加法）＋依機器實測挑做法
