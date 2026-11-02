# NCCL P2P channel 數：已知資訊與新穎性搜尋（F1 相關）

搜尋日期：2026-11-02。目的：W2——在把 F1 當成研究題目之前，確認有沒有人系統性研究過。

## 搜尋紀錄

| 來源 | 查詢 | 結果 |
|---|---|---|
| arXiv MCP | `abs:NCCL AND (abs:"point-to-point" OR abs:"send/recv" OR abs:"pipeline parallel") AND (abs:channel OR abs:bandwidth OR abs:"SM")`，cs.DC/cs.LG/cs.PF，2022 起 | 0 篇 |
| GitHub issues（NVIDIA/nccl） | `"MIN_P2P_NCHANNELS"` | 0 |
| | `sendrecv bandwidth channels NVLink` | 6 則，相關的是 #1598 |
| | `"P2P_NVL_CHUNKSIZE"` | 0 |
| | `"p2pnChannelsPerPeer"` | 2 則：#1198、#1680 |
| Web | NCCL send/recv bandwidth、p2p channels per peer、pipeline parallel | NCCL 文件（ncclConfig_t） |

## 已知的資訊

- **NCCL 文件（`ncclConfig_t` 的 `nChannelsPerNetPeer`）**：預設值是為 AlltoAll 模式最佳化的，想提高 send/recv 頻寬可以調大；但它會影響所有 send/recv 類操作（all-to-all、scatter、gather）。這個設定是給跨機網路用的，不是單機 NVLink。
  - <https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2303/user-guide/docs/api/types.html>
- **#1198「Why P2P requires more channels?」**：NCCL 維護者 sjeaugey 解釋，P2P 排程需要 channel 數是 2 的冪次，因為它用 bit mirroring 讓每個 peer 的起始 channel 分散，2 或 4 個 channel per peer 時才能平均用到所有 channel；這主要影響小到中等訊息。
  - <https://github.com/NVIDIA/nccl/issues/1198>
- **#1680「How to set griddim for ncclsend/recv?」**：使用者發現 send/recv kernel 的 grid 只有 2，設 `NCCL_MIN/MAX_P2P_NCHANNELS` 和 `minCTAs` 都沒變大；回覆建議用 `NCCL_NCHANNELS_PER_NET_PEER`（未確認在單機是否有效）。
  - <https://github.com/NVIDIA/nccl/issues/1680>
- **#1598**：`NCCL_MAX_NCHANNELS` 管不到 SendRecv kernel 的 grid；P2P 的 channel 數要用 `NCCL_MAX_P2P_NCHANNELS`，`NCCL_MAX_CTAS` 應該能限制所有情況（使用者回報有時沒生效）。
  - <https://github.com/NVIDIA/nccl/issues/1598>

## 和本研究的關係

- 「P2P 用的 channel 很少」這件事，使用者零星注意過，但只停留在 GitHub 問答；沒有找到量化這個行為對效能的影響、或提出修正方法的論文。
- 原始碼（2.32.3）顯示：單機情況下，每個操作的 channel 下限依 **communicator 的 rank 數**決定（`scheduleP2pTasksToPlan`），不看這次實際有幾個 peer 在傳；`config.maxP2pPeers` 只影響多機分支（`ncclTopoComputeP2pChannels`）。這一點在上述 issue 裡都沒有被提到。
- 反方向的考量：channel 數 = 佔用的 SM 數。P2P 用少量 channel 可能是刻意省 SM，讓通訊和計算重疊時少搶計算資源。任何「多給 channel」的方案都要量這個代價。
