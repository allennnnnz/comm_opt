# G1 報告：已確認的現象（2026-11-02）

## 白話結論

1. **最值得研究的是 F1**：8 張卡一對一傳資料時，中等大小（幾 MB 到幾十 MB）的速度只有 2 張卡的 36–41%。這是 NCCL 的設計造成的，原因已經查到原始碼的哪幾行；改設定只能修好一部分，而且會讓別的情況變慢。
2. **F1 有研究空間**：問題出在 NCCL 依「GPU 總數」而不是「實際在傳的對象數」分配通道；沒有一個固定設定能在所有大小都最好。網路上只有零星的使用者問答，沒找到研究它的論文。
3. **F1 的風險**：真實系統（例如 pipeline 平行）不一定用大的通訊群組做一對一傳輸；而且多用通道會多佔 GPU 計算資源。選 F1 的話，第一步要先確認這兩點。
4. **其他現象比較弱**：broadcast 不穩定（F5）是真的但原因未知、用途較少；2 卡在 2 MiB 附近變慢（F3）範圍很窄；其餘（F2、F4）效果小或可能是硬體本身的特性。

## 現象總表

| 現象 | 狀態 | 效果大小 | 機制 | 研究潛力 | 證據 |
|---|---|---|---|---|---|
| **F1** 8 卡 sendrecv 在中等訊息被限縮 channel | **確認**（M4 兩組 GPU、5 次） | 2 卡的 36–41%（16、64 MiB） | **已知**：channel 下限 = P2P 總數 / communicator rank 數（NCCL `scheduleP2pTasksToPlan`）；模型預測 143/143 點正確 | 高：設定只能部分修好且有副作用（M2）；無論文 | `exp/M4_repeat`、`exp/M2_p2p_channels`、`lit/notes/nccl_p2p_channels.md` |
| F2a alltoall 8 卡 4 GiB 比 1 GiB 慢 | 確認 | −5% | 推測：channel 從 8 增到 32 後效率下降 | 低：效果小，且 MoE 的 alltoall 多在 MB 級 | `exp/M4_repeat` |
| F2b alltoall 2 卡 in-place 比 out-of-place 快 | 確認 | −13% 時間 | 推測：out-of-place 多一次本地複製 | 低：可解釋 | `exp/M4_repeat` |
| F3 2 卡 collective 在 1–2 MiB 比 8 卡低 | **依判準不成立**（2 MiB 成立、4 MiB 變弱） | 2 MiB：2 卡為 8 卡的 0.48–0.61 | 未知 | 中低：範圍窄 | `exp/M4_repeat` |
| F4 單向傳輸比雙向高 | 確認 | +14% | 推測：NVLink 雙向同時傳輸的效率限制 | 低：可能是已知硬體特性 | `exp/M4_repeat` |
| F5 broadcast 8 卡在 1–16 MiB 不穩定 | 確認 | cv 31–52%（all_reduce 1–12%） | 未知 | 中：原因未知，但 broadcast 在訓練／推論的穩態使用較少 | `exp/M4_repeat` |

## 建議

選 **F1** 作為主線。若核准，下一步（需另外核准）：

1. **確認真實系統會不會遇到**：查主流框架（PyTorch、Megatron-LM、vLLM 等）做一對一傳輸時用多大的通訊群組（讀程式碼與文件，不需要 GPU）。
2. **量 SM 代價**：通訊和計算同時跑時，P2P channel 數對兩者各自的影響（新的 M 系列實驗）。

這兩點都通過，才進 G2（用 brainstorming skill 正式選題）。
