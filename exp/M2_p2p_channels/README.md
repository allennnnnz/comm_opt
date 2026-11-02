# M2_p2p_channels

- **狀態**：已核准（D006），尚未執行
- **核准**：2026-11-02（D006）
- **類型**：結論性（每個設定 5 次獨立 launch、隨機順序）

## 白話結論

（實驗完成後填）

## 問題

F1（8 卡 sendrecv 在中等訊息只有 4 個 channel、速度只剩 2 卡的約三分之一）能不能靠調 NCCL 的設定解決？如果能，這只是預設值沒調好；如果不能，才值得當研究題目。

## 背景：NCCL 2.32.3 原始碼怎麼決定 P2P channel 數

讀 `NVIDIA/nccl` tag `v2.32.3-1`：

1. `src/graph/paths.cc` `ncclTopoComputeP2pChannels`：P2P channel 總數 = `pow2Up(max(min(collective channel 數, NCCL_MAX_P2P_NCHANNELS), NCCL_MIN_P2P_NCHANNELS))`。本機 collective channel 是 24，所以預設是 32。
2. `src/enqueue/enqueue.cc` `scheduleP2pTasksToPlan`：每個操作的 channel 下限從每個 peer 的 channel 數（32）開始，**只要「下限 × communicator 的 rank 數」超過總數就減半**。8 卡 → 4、4 卡 → 8、2 卡 → 16。這裡用的是 communicator 的大小，不是這次真正在傳資料的 peer 數。
3. `src/enqueue/enqueue.cc` `addP2pToPlan`：從下限開始，只有當每個 channel 要負責的資料超過 `chunk × 32`（預設 512 KiB × 32 = 16 MiB）時才加倍。

`predict_channels.py` 把這三步移植成 Python，對 M0 的所有 channel 觀測值都吻合。

## 預期（量測前寫，量測後不改）

各設定在 8 卡的預測 channel 數（`python3 exp/M2_p2p_channels/predict_channels.py` 的輸出）：

| sendrecv ×8 | 1M | 2M | 4M | 8M | 16M | 32M | 64M | 128M | 256M | 512M | 1G |
|---|---|---|---|---|---|---|---|---|---|---|---|
| default | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 8 | 16 | 32 | 32 |
| minp2p64（`NCCL_MIN_P2P_NCHANNELS=64`） | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 16 | 32 | 32 |
| chunk128k（`NCCL_P2P_NVL_CHUNKSIZE=131072`） | 4 | 4 | 4 | 4 | 4 | 8 | 16 | 32 | 32 | 32 | 32 |
| chunk64k（`NCCL_P2P_NVL_CHUNKSIZE=65536`） | 4 | 4 | 4 | 4 | 8 | 16 | 32 | 32 | 32 | 32 | 32 |
| minp2p64_chunk128k | 8 | 8 | 8 | 8 | 8 | 8 | 16 | 32 | 32 | 32 | 32 |

| alltoall ×8 | 1M | 2M | 4M | 8M | 16M | 32M | 64M | 128M | 256M | 512M | 1G |
|---|---|---|---|---|---|---|---|---|---|---|---|
| default | 2 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 8 |
| minp2p64 | 2 | 4 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 |
| chunk128k | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 8 | 16 | 32 |
| chunk64k | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 8 | 16 | 32 | 32 |
| minp2p64_chunk128k | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 16 | 32 |

2 卡 sendrecv：default 在 1M–256M 為 16、chunk128k 從 128M 起變 32、chunk64k 從 64M 起變 32。

對效能的預期：
- channel 數變多，busbw 會上升，但不會超過 2 卡 default 的水準
- 64 MiB：chunk64k 預測 32 個 channel，應接近 2 卡水準（≥ 90%）
- 16 MiB：最多只到 8 個 channel，預期只能部分改善（2 卡的 50–90%）
- 整體預期是「部分是預設值問題」：大一點的訊息調 chunk 就能解決，中等訊息靠現有設定解決不了
- 縮小 chunk 可能讓每個 channel 的效率變差；在 2 卡上要看有沒有退步

## 判準（量測前寫，量測後不改）

統計方式同 M4：5 次獨立 launch 的中位數與 [最小, 最大]，「不重疊」= 區間不相交。

| 項目 | 條件 |
|---|---|
| 原始碼模型是否正確 | `-U` 回報的 channel 數與上表完全相同 → 模型確認；任何不符都要記錄 |
| F1 完全是預設值問題 | 某個設定讓 8 卡 sendrecv 在 16 MiB **和** 64 MiB 都 ≥ 2 卡 default 中位數的 90%，且與 8 卡 default 不重疊 |
| 部分是 | 最佳設定在 16 MiB 或 64 MiB 介於 2 卡 default 的 50%–90% |
| 不是 | 最佳設定在 16 MiB 和 64 MiB 都 < 8 卡 default 的 1.5 倍 |
| 副作用 | alltoall 8 卡或 sendrecv 2 卡在任一大小比 default 差 > 5% 且不重疊 → 記錄為取捨 |
| SM 代價 | 每個設定的 channel 數就是佔用的 CTA（SM）數，一併報告 |

## 設定

- **變因**：5 種設定（上表），只改 NCCL 環境變數
- **固定條件**：8 卡用 GPU 組 A（`0–7`），2 卡用 `0,1`；大小 1 MiB–1 GiB（×2）
- **量測**：sendrecv ×8、alltoall ×8 各 5 設定 × 5 次；sendrecv ×2 在 default、chunk128k、chunk64k 各 5 次；共 65 次 launch，seed 20261103 打亂順序
- **iteration**：`-w 10 -n 40 -a 1 -U 1 -I 1`
- **工具與版本**：同 M4；NCCL 原始碼 tag `v2.32.3-1`（commit 12df1a1）

## 指令

```bash
python3 exp/M2_p2p_channels/predict_channels.py   # 預測表 + 對 M0 自我檢查
python3 exp/M2_p2p_channels/run.py
```

## 結果

（尚未執行）

## 結論

（尚未執行）

## 下一步
