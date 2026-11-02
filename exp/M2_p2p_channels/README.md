# M2_p2p_channels

- **狀態**：完成（2026-11-02 14:25–14:36，run `20261102_142517`）
- **核准**：2026-11-02（D006）
- **類型**：結論性（每個設定 5 次獨立 launch、隨機順序）

## 白話結論

1. **我們完全搞懂了 NCCL 怎麼決定 P2P 用幾個通道。** 照原始碼寫的預測模型，143 個量測點的通道數全部預測正確。
2. **改設定只能「部分」修好。** 64 MiB 時把 chunk 調小，8 卡可以完全追上 2 卡的速度；但 16 MiB 時最好的設定也只到 2 卡的 69%——在 8 卡下，NCCL 的上限讓通道數最多只能到 8。
3. **沒有「免費」的設定。** 讓 8 卡變快的設定，會讓 2 卡的 send/recv 慢 10–27%、alltoall 大訊息慢 6–12%。最好的設定隨訊息大小和 GPU 數改變，一個全域環境變數不可能同時兼顧。
4. **這代表 F1 不只是預設值沒調好，而是設計上的限制**：通道下限綁在「GPU 總數」，而不是「這次實際在傳的對象數」。這留下研究空間——依每次傳輸的情況動態決定通道數。但多用通道會多佔 GPU 的計算單元（SM），下一步要先量這個代價。

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

- 原始數據：`results/M2_p2p_channels/20261102_142517/`（65 次 launch）
- 分析：`.venv/bin/python exp/M2_p2p_channels/analyze.py results/M2_p2p_channels/20261102_142517` → `figures/M2_p2p_channels/20261102_142517/`（`channel_model_check.csv`、`sendrecv8_vs_2gpu.csv`、`verdicts.csv`、`side_effects.csv`、`busbw_summary.csv`、`fig_p2p_settings.png`）
- 65 次全部 rc=0，`#wrong` 全為 0

**原始碼模型**：143/143 個（collective, GPU 數, 設定, 大小）點，`-U` 回報的 channel 數與預測完全相同。

**8 卡 sendrecv 相對 2 卡 default（中位數，GB/s）**

| 設定 | 16 MiB | 占 2 卡 | 64 MiB | 占 2 卡 |
|---|---|---|---|---|
| 2 卡 default（目標） | 121.4 | 100% | 163.0 | 100% |
| 8 卡 default | 45.9 | 38% | 58.2 | 36% |
| minp2p64 | 84.3 | 69% | 107.1 | 66% |
| chunk128k | 35.8 | 30% | 139.7 | 86% |
| chunk64k | 52.1 | 43% | 168.8 | 104% |
| minp2p64_chunk128k | 65.7 | 54% | 139.3 | 85% |

（所有設定與 8 卡 default 的 [最小, 最大] 都不重疊。完整表：`sendrecv8_vs_2gpu.csv`）

**副作用**（比 default 差 > 5% 且不重疊，共 14 項，`side_effects.csv`）
- sendrecv 2 卡：chunk64k 在 8–32 MiB 慢 12–27%、512 MiB 以上慢 15–16%；chunk128k 在 16–64 MiB 慢 10–15%、512 MiB 以上慢 5–6%
- alltoall 8 卡：chunk64k 在 64 MiB、512 MiB、1 GiB 慢 6–11%；chunk128k 在 1 GiB 慢 12%
- minp2p64 沒有任何副作用項目

## 結論

| 判準 | 結果 |
|---|---|
| 原始碼模型正確 | **成立**（143/143） |
| F1 完全是預設值問題 | **不成立**：沒有設定在 16 MiB 和 64 MiB 都達到 90% |
| 部分是 | **成立**：16 MiB 最佳 minp2p64 69%；64 MiB 最佳 chunk64k 104% |
| 不是 | 不成立 |

對照預期：
- 「64 MiB chunk64k ≥ 90%」→ 符合（104%）
- 「16 MiB 只能部分改善（50–90%）」→ 符合（最佳 69%）
- 「整體是部分」→ 符合
- 「縮小 chunk 可能讓每個 channel 效率變差」→ 符合，而且比預期嚴重：
  - **意外 1**：chunk128k 在 16 MiB 和 default 一樣用 4 個 channel，卻慢 22%（35.8 vs 45.9）
  - **意外 2**：chunk64k 讓大訊息的上限從約 217 降到約 181 GB/s

機制解讀：
- 在 4–8 個 channel 的區間，頻寬幾乎和 channel 數成正比（每個 channel 約 10–15 GB/s），所以 channel 數就是瓶頸
- 8 卡下，channel 下限 = P2P 總數 / 8，而 P2P 總數最多 64（`MAXCHANNELS`），所以下限最多是 8；要再多只能縮 chunk，但縮 chunk 會降低每個 channel 的效率
- 最佳設定隨訊息大小翻轉：16 MiB 最好的是 minp2p64，64 MiB 最好的是 chunk64k

限制：
- 只量了 nccl-tests 的 sendrecv，所有 rank 同時各自傳給下一個 rank；真實的 pipeline 平行傳輸模式可能不同
- 沒有量通訊和計算同時進行時的影響：多用 channel 會多佔 SM，可能拖慢計算

## 下一步（提案，需核准）

- G1：把 F1 列為主要候選
- 若選 F1：先量「通訊和計算同時跑時，P2P channel 數對兩者的影響」（SM 代價），再決定題目方向
