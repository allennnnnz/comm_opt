# M8_ll128_atomic_a100

- **狀態**：已核准（D016），建置中
- **核准**：2026-11-03（D016，使用者：「好 先做第一步」——確認《Every µs Matters》的 two-shot LL128 atomic 在 A100 上是否可行）
- **類型**：可行性測試（3 次獨立執行）

## 白話結論

（實驗完成後填）

## 問題

《Every µs Matters》（Shen et al., arXiv 2607.16100）的 two-shot LL128 atomic all-reduce 能不能在 A100 上正確運作？它的前提是（論文 §IV-4）：「NVLink 會在 cache line（128 bytes）層級原子地套用這些原子加法」與「NVLink 原子地執行 128 byte 的寫入」——所以接收端看到該行的旗標（第一個元素）等於 rank 數 N 時，該行其他元素一定也已加總完成。論文只在 GB200 上評估，沒有提到 A100。

## 已確認的事實（量測前，讀論文與原始碼）

- 論文 Table I：two-shot LL128 atomic 的同步次數是 **2**（不是 1），資料量約 2(N−1)M/N；它去掉的是全域 memory barrier，改用嵌在資料裡的旗標
- 公開的 low-latency-nccl 分支（commit 5357eff）**沒有** LL128 atomic 的實作（搜尋 `LL128_Atomic` 0 筆）；有 LL 與 sentinel（Poison）兩種同步模式的 one-shot／two-shot
- 該分支把所有新增做法（包含不需要 multicast 的 LL／sentinel 版本）列在 `nvls_algos_by_coll`，因此只在 `__CUDA_ARCH__ >= 900` 時編譯 kernel 內容；A100（sm80）上 kernel 是空殼——這解釋了 M7 試跑「1.5 µs、結果全錯」
- NCCL 本身在 A100 + NVLink 上會選用 LL128 協定（M0 tuner 在 2–8 MiB 選了 LL128），代表一般的 128 byte **寫入**在 A100 NVLink 上被 NCCL 視為原子；**原子加法**是否有同樣性質未知

## 測試設計

單一程序、8 張 GPU、開啟 peer access（`test_ll128_atomic.cu`，nvcc `-arch=sm_80`）：

1. **向量原子加法是否存在**：嘗試為 sm_80 編譯 `red.global.add.v4.f32`（16 bytes 一次）——預期 ptxas 拒絕（需要 sm_90）
2. **cache line 層級原子性**：GPU0 上配置 T = 2^20 個 128 byte 的行（每行 32 個 uint32）。8 張 GPU 同時對每一行做 warp 寬度的原子加法：lane 0 對第 0 個元素（旗標）加 1，lane 1–31 對自己的元素加 (rank+1)。GPU0 同時跑觀察者：每個 warp 以一次 warp 寬度的讀取讀整行，旗標 = 8 時檢查其餘 31 個元素是否都 = 36（1+…+8）。違反 = 「旗標已到 8，但資料還沒加完」
3. **跨 GPU 原子加法的速度**：計時 8 張 GPU 的寫入 kernel，換算每張 GPU 的原子加法頻寬

## 預期（量測前寫，量測後不改）

- 1：sm_80 不支援 `red.global.add.v4.f32`（16 byte 向量原子加法是 sm_90 起才有）
- 2：會出現違反（> 0）。A100 上每個 32-bit 原子加法是獨立的 NVLink 交易，沒有理由在 cache line 層級一起生效
- 3：跨 GPU 原子加法頻寬遠低於一般寫入（低於 50 GB/s／GPU）

## 判準（量測前寫，量測後不改）

| 結果 | 條件 | 代表 |
|---|---|---|
| LL128 atomic 前提在 A100 上不成立 | 3 次執行中任一次違反數 > 0 | 這個做法不能直接搬到 A100；A100 需要其他方式做到免 barrier 的 two-shot |
| 前提在 A100 上可能成立 | 3 次執行、每次 2^20 行，違反數都 = 0 | 新穎性風險高；要再確認效能 |

- 違反數 = 0 不能證明「保證成立」（只是沒觀察到）；違反數 > 0 則足以否定

## 指令

```bash
nvcc -O3 -arch=sm_80 -o /tmp/... exp/M8_ll128_atomic_a100/test_ll128_atomic.cu
./test_ll128_atomic   # 單一程序使用 8 張 GPU
```

## 結果

（尚未執行）

## 結論

（尚未執行）
