# M11_before_prototype

- **狀態**：進行中
- **核准**：2026-11-04（D023，使用者：「開始量」）
- **類型**：結論性（每個設定 3 次獨立執行、隨機順序）

## 白話結論

（完成後填）

## 問題

動手做新做法之前，回答三件事：

- **A. 地基夠不夠穩**：8 張 GPU **同時互相**做跨卡原子加法（新做法的「分工加總」）時，(1) 計數器到齊時資料是否一定加完、(2) 半精度小數相加的誤差與「每次結果是否相同」、(3) 速度能不能比 vLLM 的對應步驟快
- **B. 時間花在哪**：vLLM（A100 上目前最快）一次加總的時間裡，「等待」「搬資料＋加總」「兩次之間的空檔」各占多少——決定「不用等」最多能省多少
- **C. 路有沒有塞滿**：vLLM 連續加總時，8 張卡之間的連線實際用了多少（回答使用者的問題：是沒用滿運算資源，還是通訊瓶頸）

## 已確認的事實（量測前）

- M8：7 張 GPU 對 1 張做 warp 寬度 32-bit 整數原子加法（每 128 byte 一行、lane 0 是計數器），3 × 2^20 行 0 次「計數器到齊但資料未加完」；匯入總頻寬 235–241 GB/s。**沒測過**：8 對 8 同時進行、半精度（`f16x2`／`bf16x2`）與 fp32 原子加法、小資料的完成時間
- vLLM 0.33.0 原始碼（`csrc/custom_all_reduce.cuh`、`custom_collective_common.cuh`，tag v0.33.0）：
  - 1-stage：起點 barrier → 每卡讀 8 份相加 → 終點 barrier
  - 2-stage：起點 barrier → 每卡讀 8 份中自己那 1/8 相加（reduce-scatter）→ 中間 barrier → 從 8 張卡收齊（all-gather）；沒有終點 barrier（靠下一次的起點 barrier）
  - TP=8 時 < 256 KB 用 1-stage；block 數 = min(36, 資料量 / 8 KB)，每 block 512 執行緒
- nsys 2026.3.2 的 `ga100` GPU 指標組可以取樣「NVLink RX/TX User Data [Throughput %]」與「SMs Active」（2026-11-04 在空閒的 GPU 0 上試過，不需要管理員權限）
- M9（hidden 5120）：vLLM 加總＋另跑 norm 為 14.3／18.4／23.4／23.4／27.9 µs（80K／160K／320K／640K／1.25M）；可達水準 6.0–15.3 µs

## 設計

共同設定：8×A100、fp16 資料、大小 = hidden 5120 × token 8／16／32／64／128（80K／160K／320K／640K／1.25M）；量測前確認沒有其他 GPU 工作（R13）。A、B 是**單一程序、8 張 GPU、開 peer access** 的獨立 CUDA 程式（`nvcc -arch=sm_80`），和 M8 相同的做法；vLLM 實際上是每張 GPU 一個程序、用 IPC 共用記憶體，但資料都走同樣的 NVLink，記在限制。

### A. 8 對 8 原子加法（`rs_probe.cu`）

每張 GPU 的輸入切成 8 段，第 k 段送去第 k 張 GPU（包含自己），做「分工加總」：

| 代號 | 做法 | 每行 128 byte 的內容 |
|---|---|---|
| `atom_f16` | warp 寬度原子加法，半精度成對（`atomicAdd(__half2*)`） | lane 0：計數器（+1）；lane 1–31：62 個 fp16 |
| `atom_bf16` | 同上，bf16 成對（`atomicAdd(__nv_bfloat162*)`） | 同上 |
| `atom_f32` | 同上，fp32（累加緩衝區用 fp32，傳輸量約 2 倍） | lane 0：計數器；lane 1–31：31 個 fp32 |
| `store_slot`（備案） | 不用原子加法：每張卡把資料寫進負責人的「第 r 格」，每行帶回合記號；負責人看到 8 格都到齊後自己用 fp32 相加 | lane 0：回合記號；lane 1–31：62 個 fp16 |

- 每次：所有 GPU 先做一次起跑同步（讓 8 張卡對齊，**不計時**），之後各 block 用 `clock64` 計時：送出自己的 8 段 → 等自己負責的那段全部到齊（計數器 = 8 或 8 格記號都對）→ 讀出結果寫到輸出區
- 每個 GPU 的時間 = 各 block 計時的最大值；一次的時間 = 8 張 GPU 的最大值；cycle 換算成 µs 用同一次執行裡的校正 kernel（`clock64` 對 `%globaltimer`，10 ms）
- block 數：16、32、64、108
- 每次 launch：暖機 5 次、計時 50 次；每次之間把緩衝區清零
- **檢查**：
  - 「到齊時資料是否加完」：負責人看到計數器到齊時讀出的值，和全部結束後緩衝區裡的最終值比對，不同即「違反」
  - 誤差：和用 fp64 算的精確和比較，max |誤差| / max |精確值|
  - 每次是否相同：同樣輸入的 50 次結果，與第 1 次逐位元比較，報不同元素的比例

### B. vLLM 一次加總的時間組成（`car_timed.cu`）

- 把 vLLM 0.33.0 的 `cross_device_reduce_1stage`／`2stage` 原樣複製，**只加**每個 block 在階段邊界記 `clock64`（起點 barrier 前後、第一階段後、中間 barrier 後、結束）；barrier 與加總程式不改
- 每張 GPU 把 K=20 次加總錄成一個 CUDA graph 連續執行（和 M6 相同的「連續加總」情境），計時 20 次重播
- 報：起點 barrier 等待、第一階段（讀＋加）、中間 barrier、第二階段（收齊）各自的 µs；**每次之間的空檔** = 事件計時的每次時間 − kernel 內量到的總時間
- block 數用 vLLM 自己的規則；1-stage 和 2-stage 兩種都量（每個大小都量兩種，不只 vLLM 預設會選的那種）
- **自我檢查**：同一個程式另編一份不加計時的版本，兩者每次時間差 ≤ 10% 才採用 B 的拆解；並和 M7 的 vLLM 數字對照量級

### C. 連線使用率（nsys GPU 指標）

- 在 nsys `--gpu-metrics-set=ga100`、8 張 GPU、取樣 100 kHz 下，執行 B 的不計時版本（每個設定連續跑約 0.5 秒，用 NVTX 標記區間）與 A 的 `atom_f16`、`store_slot`
- 報：每個設定區間內 NVLink TX／RX User Data 的平均使用率（%）與 SMs Active（%），取 8 張 GPU 的平均與最大
- 只跑一次（指標取樣，不用來下快慢結論）

## 預期（量測前寫，量測後不改）

- **A1（到齊即加完）**：四種做法、所有大小與 block 數，違反數都 = 0
- **A2（誤差）**：`atom_f32`、`store_slot` 誤差 ≤ 2^-10；`atom_f16` ≤ 2^-8；`atom_bf16` ≤ 2^-5（都在 M6 的「正確」門檻 2^-5 內）
- **A3（每次是否相同）**：`store_slot` 100% 相同；`atom_f16`、`atom_bf16` 有元素不同（> 0%）；`atom_f32` 輸出轉回 fp16 後不同的比例 < 1%
- **A4（速度）**：`atom_f16` 在最佳 block 數下，至少 3/5 個大小比 vLLM 2-stage 的「起點 barrier＋第一階段」快
- **A5**：`atom_f16` 在 1.25M 的速度 ≥ `store_slot` 的 50%（原子加法比一般寫入慢，但不會慢一倍以上）
- **B1（等待的比重）**：vLLM 2-stage 的兩次 barrier 合計，在 80K–320K 占 kernel 時間 ≥ 40%、在 640K–1.25M ≥ 25%；1-stage 的兩次 barrier 在 80K–160K ≥ 30%
- **B2（空檔）**：連續加總時每次之間的空檔 ≤ 2 µs
- **C1（路沒塞滿）**：vLLM 連續加總時，8 張 GPU 的 NVLink TX User Data 平均使用率在所有 5 個大小都 < 50%

## 判準（量測前寫，量測後不改）

| 結果 | 條件 | 代表 |
|---|---|---|
| **原子加法做法可行** | A1 全為 0，且 `atom_f16` 或 `atom_f32` 的誤差 ≤ 2^-5，且該做法 A4 成立（≥ 3/5） | 新做法用原子加法版本 |
| **改用備案** | 原子加法版本不符上一列，但 `store_slot` A1 為 0 且至少 3/5 個大小比 vLLM「起點 barrier＋第一階段」快 | 新做法用「各寫各的格子、負責人自己加」 |
| **兩者都不行** | 其他 | 停下來討論設計 |

- **「不用等」最多能省多少**：取 B 量到的 barrier 時間，作為之後訂成功標準（G3）的依據；B1 不成立（等待比重小）時，新做法的重點改成「少搬資料」，要先和使用者討論
- C 的結果只用來回答「是不是通訊瓶頸」，不影響上面的判定

## 指令（暫定）

```bash
nvcc -O3 -arch=sm_80 -o build/rs_probe exp/M11_before_prototype/rs_probe.cu
nvcc -O3 -arch=sm_80 -o build/car_timed exp/M11_before_prototype/car_timed.cu           # 加計時
nvcc -O3 -arch=sm_80 -DNO_STAMPS -o build/car_plain exp/M11_before_prototype/car_timed.cu
python3 exp/M11_before_prototype/run.py      # R13 檢查、隨機順序、3 次、環境紀錄、nsys 指標
.venv/bin/python exp/M11_before_prototype/analyze.py
```

## 量測前的修正與發現（看到任何正式數據之前）

- **A100 沒有 bf16 成對原子加法指令**：`atomicAdd(__nv_bfloat162*)` 在 sm_80 由 CUDA 標頭（`cuda_bf16.hpp`）以每個 lane 各自的 `atomicCAS` 迴圈模擬（`atom.add.noftz.bf16x2` 要 sm_90）。編譯產物確認：`atom_bf16` 是 `ATOMG.E.CAS` 迴圈。這代表 lane 0 的計數器可能比資料先完成，**A1 對 `atom_bf16` 很可能不成立**；預期不改，照原樣判定
- **`atom_f16` 改用只送不等的 `red`**：`atomicAdd(__half2*)` 編成 `ATOM.E.ADD.F16x2`（要等回覆）；新做法的設計是只送不等，所以測試程式用 inline PTX `red.global.add.noftz.f16x2`（sm_60 起支援）。`atom_f32` 本來就編成 `RED.E.ADD.F32`
- **B 的記錄方式**：試跑（數字不採用）發現 vLLM 2-stage 在小訊息時只有少數 block 真的在做第一階段（每卡只負責 1/8，但 block 數依全部資料量決定），「各 block 平均」會把關鍵的那個 block 平均掉。改成**同時記錄最慢那個 block 的各階段時間**（關鍵路徑），分析以它為主
- **B 的試跑**：加計時版比不加的慢 3–10%（10 個設定中最大 9.6%），接近卡片的 10% 門檻；正式量測照判準檢查

## 已知風險

- 單一程序 ＋ peer access，和 vLLM 的多程序 IPC 不完全相同（同樣走 NVLink）
- `clock64` 只能在同一個 block 內比較；跨 block 的時間用「各 block 最大值」近似，起跑同步本身的誤差（約 1–2 µs）會算進第一個 block 的時間
- A 量的是「分工加總＋知道到齊」，不含正規化與發回；完整新做法的時間要等原型
- 原子加法的「到齊即加完」仍是觀察，不是 NVIDIA 文件保證（同 M8）

## 結果

（量測後填）

## 結論

（量測後填）
