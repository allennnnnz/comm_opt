# M21_intranode_baseline

- **狀態**：已核准（D051「通用量測照常進行」），待 M20 完成後執行
- **核准**：2026-11-16，D051
- **類型**：結論性（依 R3 重複量測）
- **依賴**：M20（NVLS 是否可用、時脈暖機行為、節點是否一致）；S01（nccl-tests、NCCL 2.29.7、`exp/lib/nano5_env.sh`）
- **分工**：Opus 寫本卡與預期、做結論；Sonnet 依本卡寫腳本、送工作、整理表格（規格書另開）

## 白話結論

（實驗完成後寫）

## 問題

在單台 8× H100（NVSwitch）上，六種集體通訊在 8 B–8 GB、2／4／8 張卡時的頻寬與延遲是多少？「交換器幫忙加總」（NVLS）開與關差多少？NCCL 預設選的做法和強制其他做法比起來如何？

## 為什麼要做

- 所有候選方向都需要單機的對照基準（D051、D052）；文獻五個主題都沒有找到在 H100 單機把 NVLS 開／關當唯一變因的系統量測（`docs/notes_synthesis_nano5.md`）。
- A100 時期的數字不能沿用（CLAUDE.md 第 1 條）。

## 預期（量測前寫，量測後不改）

寫於 2026-11-16。依據：H100 SXM 每張 18 條 NVLink4、經 NVSwitch 全互連（M20 P1、P2 的預期，尚未驗證）；nccl-tests 的 busbw 定義 [`docs/knowledge.md`]。以下為背景推論，**未在本機驗證**。

推算要點：ring 類做法的 busbw 上限約等於每張 GPU 的 NVLink 單向頻寬（M20 P2 預期約 478 GB/s）；NVLS 由交換器做加總，每張卡送出的資料量減少，all_reduce 的 busbw 可以超過 ring 的上限。

| # | 預期 | 什麼結果會推翻 |
|---|---|---|
| R1 | 8 卡 all_reduce、強制 Ring、≥1 GB：busbw 330–420 GB/s（NVLink 上限的 70–88%）。信心中 | 落在範圍外 |
| R2 | 若 M20 確認 NVLS 可用：8 卡 all_reduce ≥256 MB，NVLS 開（預設）的 busbw ≥ NVLS 關（`NCCL_NVLS_ENABLE=0`）的 1.15 倍。信心中 | < 1.15 倍 |
| R3 | 8 卡 all_reduce 8 B–1 KB（預設）每次時間 5–20 µs。信心低 | 任一大小落在範圍外 |
| R4 | 小訊息（8 B–64 KB）時，NVLS 開與關的時間差異 < 20%（NVLS 的好處主要在大訊息）。信心低 | 任一大小差異 ≥ 20% 且 95% 信賴區間分開 |
| R5 | 8 卡 all_gather、reduce_scatter ≥1 GB 預設 busbw 330–450 GB/s；all_to_all ≥1 GB 300–450 GB/s。信心低 | 落在範圍外 |
| R6 | ≥1 GB 時，2、4、8 卡的 all_reduce busbw（強制 Ring）彼此差異 < 10%（經 NVSwitch，每張卡的頻寬不因卡數改變）。信心低 | 任兩者差異 ≥ 10% |
| R7 | 重複量測穩定：≥1 MB 時，5 次獨立啟動的 95% 信賴區間半寬 < 中位數的 2%。信心中 | 任一設定 ≥ 2% |

## 設定

- **變因與水準**：
  - 集體通訊（6 種）：all_reduce、all_gather、reduce_scatter、broadcast、reduce、all_to_all（nccl-tests `alltoall_perf`）
  - 資料大小：8 B–8 GB，每次 ×2（`-b 8 -e 8G -f 2`）；all_to_all 依記憶體上限調整，實際範圍記錄在報告
  - 卡數：2、4、8（2、4 卡用同一台的前幾張卡；記錄實際卡號）
  - NVLS：開（預設）、關（`NCCL_NVLS_ENABLE=0`）——M20 確認 NVLS 不可用時只做預設
  - 演算法（只對 8 卡 all_reduce）：預設；強制 `Ring`、`Tree`、`NVLS`（若可用）；強制後實際使用的演算法以 NCCL log 為準
  - protocol（只對 8 卡 all_reduce、強制 Ring）：`LL`、`LL128`、`Simple`
- **固定條件**：NCCL 2.29.7（S01 venv）、nccl-tests v2.21.1、`exp/lib/nano5_env.sh`；一卡一個行程；暖機長度依 M20 P7 的結果決定
- **GPU**：單台 H100 整台，`--exclusive`；正式量測用 `normal`。H200 不在本卡範圍（若 M20 顯示 H200 拓樸不同再另議）
- **Opus 補充（2026-11-17 01:45，量測前寫；R1–R7 不變，只補設定）**：
  - **排隊區改用 `dev`**：2026-11-17 01:20 試算 `normal` 單台約 11-19 才能開始，`dev` 約數小時；兩者是同一批 H100（`dev`＝hgpn01–06、17–21，`normal` 少 hgpn01）。`dev` 單一工作上限 2 小時、每計畫 8 張卡，所以每台節點一個工作（≤1.5 小時），依序送
  - 每個工作開頭先跑一個短暫的暖機負載，長度依 M20 P7 的結果決定（Opus 審 M20 第 1 台後告知）；未定前用 30 秒 all_reduce 64 MB
  - 送出時機：Opus 看過 M20 H100 第 1 台結果（P4 NVLS、P7 時脈）後才送
- **工具與版本**：nccl-tests v2.21.1（commit `afd59ab`）；`job_env.txt` 記錄節點、NCCL 版本、模組
- **統計方式**：
  - 每個設定重複 5 次**獨立啟動**，順序打亂；至少在 2 台不同節點上各做一部分（節點當成區組，R11），若 M20 P8 顯示節點有差異則每台都做完整一份
  - 每次啟動內：`-w 5`、`-n 20`；小訊息加大 `-n` 使總時間 ≥ 10 ms，記錄實際值
  - 跨 rank 取最大值（nccl-tests 的平均方式參數，實作前在 v2.21.1 原始碼確認數值對應）
  - 報告：每個大小的中位數與 95% bootstrap 信賴區間；指標為 busbw（GB/s）與時間（µs）
- **理論上限**（R8）：每張 GPU 的 NVLink 單向頻寬 = M20 實測 link 數 × 每條速度

## 指令

（Sonnet 依本卡實作後補上；輸出到 `results/M21_intranode_baseline/<時間>_job<id>/`）

## 結果

- 原始數據：`results/M21_intranode_baseline/`
- 分析：`exp/M21_intranode_baseline/analyze.py` → busbw／時間對大小圖（每種集體通訊）、NVLS 開關對照圖、演算法／protocol 對照圖

## 結論

（對照 R1–R7 逐條判定）

## 下一步

- R2、R4 是方向 A 的單機那一層的依據（交換器加總在大小訊息各帶來多少）
- 作為方向 C（通訊佔用 GPU 計算資源）與方向 D（效能模型）的基準資料
