# 目前狀態

> 每次 session 開始先讀本檔，結束時更新「目前狀態」與「下一步」。

## 目前狀態（2026-11-03）

- **階段**：1 找現象 → 準備選題
- **已完成**：M0、M4、M2_p2p_channels、M5_e2e_comm_share；依實測數據的文獻對照（`lit/notes/survey_20261103.md`，D011）
- **等待使用者**：是否以「生成階段中等大小 all-reduce」為主線，並核准先做 M6（現成實作對照量測）

## 已核准的實驗

| ID | 核准日期 | 狀態 |
|---|---|---|
| M0_baseline | 2026-11-02（D004） | 完成 |
| M4_repeat | 2026-11-02（D006） | 完成 |
| M2_p2p_channels | 2026-11-02（D006） | 完成 |
| M5_e2e_comm_share | 2026-11-02（D009） | 完成 |

## 提案中（需核准）

- **M6 現成 all-reduce 實作對照**：8 卡、10 KB–5 MB，比較 NCCL 預設、NCCL 登記記憶體後的低延遲模式（`-R 2`）、CUDA graph 模式（`-G`）、vLLM custom AR、MSCCL++、PyTorch SymmetricMemory、low-latency-nccl；拆出同步、傳輸、發射各花多少

## 下一步

1. 使用者裁決方向與 M6
2. 若核准：先寫 M6 實驗卡（含「值不值得實作」的判準），再建置與量測
