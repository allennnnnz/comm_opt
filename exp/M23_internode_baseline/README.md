# M23_internode_baseline

- **狀態**：已核准（D051「通用量測照常進行」），待 M20 完成後執行
- **核准**：2026-11-16，D051
- **類型**：結論性（依 R3 重複量測）
- **依賴**：M20（NVLS、GPUDirect RDMA、SHARP 是否可用，決定要強制哪些演算法）；S01（nccl-tests、NCCL 2.29.7、`exp/lib/nano5_env.sh`）
- **分工**：Opus 寫本卡與預期、做結論；Sonnet 依本卡寫腳本、送工作、整理表格（規格書另開）

## 白話結論

（實驗完成後寫）

## 問題

在 Nano5 上，2 台與 4 台 H100（以及 2 台 H200）之間，各種集體通訊在不同資料大小下的頻寬與延遲是多少？NCCL 預設選了哪種做法？「網卡直接讀寫 GPU 記憶體」（GPUDirect RDMA）與「交換器幫忙加總」（NVLS、SHARP）各帶來多少差別？

## 為什麼要做

- 候選方向 A（兩層交換器加總）與 B（跨機器生成的小訊息延遲）都需要跨機器的基準數字當出發點（D052）。
- 每台 8 張網卡、合計 400 GB/s，和一張 GPU 的 NVLink 頻寬差不多，跨機器很可能是瓶頸 [背景；`docs/knowledge.md`]，但這台還沒有任何實測。
- H100 分兩組（hgpn01–06、hgpn17–21，2026-11-16 `sinfo`），可能接在不同的網路交換器下；要知道跨組會不會比較慢，之後的工作才知道要不要固定節點（R11）。

## 預期（量測前寫，量測後不改）

寫於 2026-11-16。依據：InfiniBand NDR 每埠 50 GB/s、每台 8 埠 [文件]；nccl-tests 的 busbw 定義 [`docs/knowledge.md`]；以下推算為背景推論，**未在本機驗證**。信心標在每條後面。

推算要點：
- all_reduce 用 ring 跨機器時，NCCL 的 8 個傳輸通道各自走不同網卡，所以 busbw 的上限約等於整台的網卡總頻寬 400 GB/s，同時也受每張 GPU 的 NVLink 頻寬限制（M20 P2 預期約 478 GB/s）。
- all_to_all：16 張卡時每張卡有一半資料要出機器、4 台時有四分之三，每張卡出機器的速度上限是它那張網卡的 50 GB/s。所以 algbw 上限 2 台約 100 GB/s、4 台約 66.7 GB/s；busbw = algbw × (n−1)/n，約為 94 與 65 GB/s。

| # | 預期 | 什麼結果會推翻 |
|---|---|---|
| Q1 | 2 台 H100、all_reduce、強制 Ring、≥1 GB：busbw 在 300–400 GB/s（上限 400 的 75–100%）。信心中 | busbw < 300 或 > 400 GB/s |
| Q2 | 若 M20 確認 NVLS 可用：2 台 H100 all_reduce ≥1 GB 時，NCCL 預設選的不是 Ring（例如 NVLSTree），而且 busbw 高於強制 Ring 的 1.1 倍以上。信心低 | 預設選 Ring，或預設 busbw ≤ 強制 Ring 的 1.1 倍 |
| Q3 | 2 台 H100 all_reduce 8 B–8 KB（預設設定）每次時間在 10–40 µs；4 台比 2 台慢，但不超過 2 倍。信心低 | 任一大小落在 10–40 µs 之外，或 4 台超過 2 台的 2 倍 |
| Q4 | all_to_all ≥1 GB：busbw 達到上述上限的 60–100%（2 台 56–94 GB/s；4 台 39–65 GB/s）。信心中 | 落在範圍外 |
| Q5 | 跨機器一對一傳輸（兩台各一張卡，sendrecv）≥1 GB：40–50 GB/s（一張網卡的 80–100%）；8 B 單向延遲 3–10 µs。信心中 | 落在範圍外 |
| Q6 | 關掉 GPUDirect RDMA（資料改繞主機記憶體）後，2 台 all_reduce ≥1 GB 的 busbw 下降至少 20%。前提是 M20 P6 確認 GPUDirect RDMA 可用。信心中 | 下降 < 20% |
| Q7 | 2 台在同一組 vs 跨組（hgpn01–06 對 hgpn17–21），all_reduce ≥1 GB busbw 與 8 B 時間差異都 < 5%。信心低 | 差異 ≥ 5%（看 95% 信賴區間是否分開） |
| Q8 | 2 台 H200 與 2 台 H100 的跨機器數字（Q1、Q4、Q5 的指標）差異 < 5%（網卡相同）。信心中 | 差異 ≥ 5% |
| Q9 | 若 M20 P10 確認 SHARP 可用：2 台 all_reduce 時，強制 CollNet 類演算法在 ≥64 MB 的 busbw 高於強制 Ring；在 ≤8 KB 的時間低於強制 Ring。信心低 | 任一不成立 |

## 設定

- **變因與水準**：
  - 集體通訊：all_reduce、all_gather、reduce_scatter、all_to_all（nccl-tests 的 `alltoall_perf`）、跨機器 sendrecv（兩台各一張卡，以及兩台各 8 張卡成對）
  - 資料大小：8 B–8 GB，每次 ×2（`-b 8 -e 8G -f 2`）；sendrecv 與 all_to_all 依記憶體可承受的上限調整，實際範圍記錄在報告
  - 機器數：H100 2 台、4 台；H200 2 台
  - NCCL 演算法（只對 all_reduce）：預設；強制 `Ring`、`Tree`；M20 確認 NVLS 可用時加 `NVLSTree`；M20 P10 確認 SHARP 可用時加 `CollnetDirect`、`CollnetChain`。強制後 NCCL 實際用的演算法以 log 為準
  - GPUDirect RDMA：開（預設）、關（`NCCL_NET_GDR_LEVEL=LOC`；實作前在 NCCL 文件確認此設定的意義）——只在 2 台 H100、all_reduce 與 sendrecv 做
  - 節點組合（只在 2 台 H100）：同組、跨組（用 `--exclude` 引導，排不到就照實記錄實際節點）
- **固定條件**：NCCL 2.29.7（S01 venv）、nccl-tests v2.21.1、`exp/lib/nano5_env.sh`；一卡一個行程；每台 8 張卡
- **GPU**：每台整台 8 張，`--exclusive`；H100 4 台用 `4nodes`，2 台用 `normal`；H200 用 `normal2`
- **工具與版本**：nccl-tests v2.21.1（commit `afd59ab`）；每個工作的 `job_env.txt` 記錄節點、NCCL 版本、模組
- **統計方式**：
  - 每個設定重複 5 次**獨立啟動**（每次重新建立 NCCL 連線），順序打亂
  - 每次啟動內：nccl-tests `-w 5`（暖機）、`-n 20`；小訊息的 `-n` 可加大，以總時間 ≥ 10 ms 為準，記錄實際值
  - 跨 rank 取最大值（nccl-tests 的平均方式參數，實作前在 v2.21.1 原始碼確認數值對應）
  - 報告：每個大小的中位數與 95% bootstrap 信賴區間（跨 5 次啟動）；指標為 busbw（GB/s）與時間（µs）
  - 每次啟動保留 `NCCL_DEBUG=INFO` 的初始化與演算法選擇紀錄
- **理論上限**（R8）：每台網卡總頻寬 400 GB/s；單一網卡 50 GB/s；all_to_all 上限見預期推算；NVLink 以 M20 實測為準

## 指令

（Sonnet 依本卡實作後補上；輸出到 `results/M23_internode_baseline/<時間>_job<id>/`）

## 結果

- 原始數據：`results/M23_internode_baseline/`
- 分析：`exp/M23_internode_baseline/analyze.py` → 各集體通訊的 busbw／時間對大小圖、演算法對照圖、節點組合對照表

## 結論

（對照 Q1–Q9 逐條判定）

## 下一步

- Q2、Q9 的結果決定方向 A 是否值得做（交換器加總在小型叢集到底帶來多少）
- Q3、Q5 是方向 B（跨機器生成的小訊息延遲）的起點數字
- Q7 決定之後是否固定節點組合
