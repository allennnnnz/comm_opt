# M19_small_batch

- **狀態**：進行中
- **核准**：2026-11-04（D044，使用者：「好，先去拆解找出為何我們的方法在某些情況沒有優勢」）
- **類型**：診斷（時間拆解），5 次獨立 launch

## 白話結論

（完成後填）

## 問題

M18 步驟 b（1410 MHz、同一次量測）：原型在 1–16 tokens 幾乎固定 11.5–12.4 µs，像有一個「最低消費」；vLLM（all-reduce＋另跑 RMSNorm）從 9.96 µs（1 token）開始隨大小增加。結果 1–4 tokens 原型較慢（0.87–0.98 倍），8 tokens 1.06 倍、16 tokens 1.23 倍。**原型在小資料時多出來的約 1.5 µs 花在哪一段？** 答案決定要用哪種修法。

## 已知（量測前）

- 原型（`exp/M18_incremental/kernels/v0.cu`）流程：start barrier → 擁有該列的 block 從 8 張 GPU pull、加總、加 residual → block 內求平方和 → 寫出正規化結果 → 以 LL128 push 給 7 個 peer → 所有 block 輪詢接收其他 GPU 送來的列 → 最後一個 block 更新全域 epoch
- 依 token 列分工：token 數 < 8 時，只有前 T 張 GPU 擁有列；其他 GPU 在 phase 1 沒事做，只等別人送來
- vLLM 在 < 256 KB 時用 1-stage：start barrier → 每張 GPU 讀 8 份全部資料相加 → end barrier；之後另一個 kernel 做 `fused_add_rms_norm`
- M11（較低時脈）：vLLM 1-stage 80K 的 kernel 內約 8.5 µs、每次之間空檔約 3.8 µs

## 假說

- **H1 接力段數**：原型的關鍵路徑上有三段跨 GPU 等待（start barrier → pull → push 後輪詢），vLLM 1-stage 的關鍵路徑上 pull 之後只有一次輕量的 end barrier，正規化在本地另一個 kernel
- **H2 閒置**：token 數 < 8 時，沒有列的 GPU 大部分時間在等擁有列的 GPU 做完 pull＋norm＋push
- **H3 固定成本**：kernel 啟動、block 起跑不齊、結尾更新 epoch 等固定成本較大

## 設計

**A. 原型的時間拆解（torch 測試程式，同 M18：暖機 2 秒、丟掉第一個設定、CUDA graph K=20 × R=20）**

- `kernels/v0_stamped.cu`：v0 原封不動，只在每個 block 的階段邊界記 `clock64`：進入、start barrier 後、pull＋residual 後、平方和後、寫出後、push 後（只有擁有列的 block 有這幾段）、接收完成、結束（含 epoch 更新）。每次呼叫、每個 block 一筆
- 換算成 µs 用該設定量測前讀到的 SM 時脈（NVML；M18 起穩定 1410 MHz）
- 每次呼叫取「kernel 內時間最長的 GPU 上最長的 block」為關鍵路徑；每次之間的空檔 = 事件量到的每次時間 − 關鍵路徑
- 另外記錄擁有列的 GPU 與沒有列的 GPU 各自的時間分配
- 計時負擔檢查：同一次量測內比較 stamped 與未 stamped 的 v0，每次時間差 ≤ 10% 才採用拆解

**B. vLLM 的時間拆解**

- 在同一個 torch 測試程式內加量 `A_vllm`（只做 custom all-reduce，不做 norm）：`U_vllm − A_vllm` = norm kernel ＋ 它帶來的空檔
- vLLM all-reduce kernel 內的階段：沿用 M11 的 `car_timed.cu`（vLLM 0.33.0 kernel 原樣＋時間戳記，單一程序），加上與 M18 相同的暖機（全 GPU 忙碌 2 秒）與丟掉第一個設定，在 1410 MHz 下重量 1-stage 與 2-stage

**範圍**：fp16、hidden 5120、TP=8、token 1／2／4／8／16／32／64／128；5 次獨立 launch；每次之前確認沒有其他 GPU 工作

## 量測前的修正（小試跑後、正式數據前）

- 小試跑（不採用）顯示：1、8 tokens 的計時負擔約 1%，32 tokens 約 14%（stamped 版在 push 後多一次 `__syncthreads`）。依 P4 規則，超過 10% 的大小不採用拆解；本實驗重點在 1–8 tokens
- 「最長的 block」在小資料時常是等待接收的 block，看不到擁有列的 block 在做什麼；加記每張 GPU 的 block 0（擁有該 GPU 第一列的 block）的各段時間
- vLLM 的 kernel 內拆解使用 `car_timed_warm.cu`：stamped 與 plain 兩個版本各 5 次，用來檢查計時負擔（同 M11）

## 預期（量測前寫，量測後不改）

- **P1（H1）**：原型在 1–4 tokens 時，關鍵路徑中「start barrier ＋ push 之後等待接收」佔 kernel 內時間 ≥ 50%
- **P2（H2）**：1 token 時，沒有列的 GPU 的 kernel 內時間中，≥ 60% 在等別人送資料
- **P3**：vLLM 在 1–4 tokens 的 all-reduce kernel 內時間比原型 kernel 內時間短 ≥ 2 µs；但 vLLM 的 norm＋空檔（`U_vllm − A_vllm`）≥ 1.5 µs，部分抵消
- **P4（計時負擔）**：stamped 與未 stamped 的原型每次時間差 ≤ 10%

## 判讀規則（量測前寫）

把「原型 − vLLM」在 1–4 tokens 的差距，拆到各段（啟動與空檔、start barrier、pull＋加總、norm、push、等待接收、結尾），找出佔差距 ≥ 50% 的段落：

| 主因 | 對應的修法（之後另開實驗卡驗證） |
|---|---|
| start barrier ＋ 等待接收（接力段數） | 小資料改用 one-shot（只剩一段傳遞、不需 start barrier） |
| 沒有列的 GPU 閒置、擁有列的 GPU 太慢 | 小資料改成依「同一列的不同片段」分工，讓 8 張 GPU 都有事做 |
| 啟動、結尾等固定成本 | 精簡 kernel 開頭與結尾 |
| 以上都不是或無法分辨 | 停下來討論 |

本實驗只診斷，不修改原型。
