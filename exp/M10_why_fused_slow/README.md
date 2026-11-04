# M10_why_fused_slow

- **狀態**：進行中
- **核准**：2026-11-04（使用者：「調查清楚 a100 是少了什麼功能，造成預期比較快的方法在 a100 沒有比較快」）
- **類型**：結論性（對照實驗 3 次獨立執行）＋一次追蹤（記錄實際啟動參數，不用來下快慢結論）

## 白話結論

（完成後填）

## 問題

M9 量到：TRT-LLM 與 LMDeploy 的「加總＋正規化合併」在 A100 上沒有比 vLLM 的「分開做」快，TRT-LLM two-shot 合併還慢 2 倍、hidden 8192 無法啟動。**A100 少了哪個新一代 GPU 的功能，造成這個結果？**

## 已確認的事實（量測前，讀原始碼與編譯產物）

這些合併 kernel 用到、而 A100（sm80）沒有的功能，以及在 A100 上的替代路徑：

| 功能（新一代 GPU 才有） | 誰用 | 用途 | A100 上的替代 |
|---|---|---|---|
| **Thread block cluster ＋ 分散式共享記憶體**（sm90+） | TRT-LLM 合併（one-shot、two-shot） | 一列 token 拆給最多 8 個 block 合作，RMSNorm 的平方和用 cluster 內共享記憶體加總 | 一列只能給 **1 個 block**（`IndexHelper` 的 `#else`：token_id = blockIdx.x） |
| **NVSwitch 交換器內加總（multimem／NVLS）**（sm90＋第三代 NVSwitch） | LMDeploy 合併的 NVLS 版、TokenWeave | 交換器替你把 8 份加好，不用 barrier 與搬 8 份 | LMDeploy 改走 Push／Pull 版：**2 次全域 barrier**，一個 block 處理一列，block 數 = ⌈token/8⌉（hidden 5120 時 1–16 個） |
| **Programmatic Dependent Launch**（sm90+） | TRT-LLM、FlashInfer | 下一個 kernel 提早開始，藏住啟動延遲 | 沒有（`cudaGridDependencySynchronize` 被編掉）。**本實驗的微基準沒有前後 kernel，不受影響**，所以不是 M9 結果的原因 |
| 16 byte 向量原子加法 `red.v4`（sm90+） | 《Every µs Matters》 | — | M9 的對象都沒用到，不是原因 |

**編譯產物（cuobjdump，sm_80）**：TRT-LLM 合併 kernel（`kARResidualRMSNorm`、fp16、8 卡）每個執行緒 **76 個暫存器**（one-shot 與 two-shot 相同）。hidden 8192 時一列要 1024 個執行緒 → 76 × 1024 = 77,824 > A100 每個 SM 的 65,536 → 無法啟動（M9 的 `cudaErrorLaunchOutOfResources`）。

**由程式推得的 block 數（hidden 5120，fp16）**：

| token | 大小 | `F_trt_one`（A100） | `F_trt_one`（若有 cluster，H100 路徑） | `F_trt_two`（A100） | LMDeploy 合併 | 
|---|---|---|---|---|---|
| 8 | 80K | 8 | 32 | — | 1 |
| 16 | 160K | 16 | 64 | 2 | 2 |
| 32 | 320K | 32 | 128 | 4 | 4 |
| 64 | 640K | 64 | 128 | 8 | 8 |
| 128 | 1.25M | 108（SM 上限，迴圈） | 128 | 16 | 16 |

（H100 欄是用 TRT-LLM launcher 的規則在 cluster 可用時推算的：cluster 從 8 開始，block 執行緒數不足 128 或總數超過 SM 數就減半，**沒有實測**。）M7 量到 A100 上 NCCL 在 64K–1M 需要 8–32 個 block 才跑得快，且「很少 block」時明顯變慢。

## 假說

- **H1（主要）**：A100 沒有 cluster，所以「一列一個 block」的設計在生成階段的大小只開得出很少的 block，用不到大部分 SM 與 NVLink 頻寬 → 慢。TRT-LLM two-shot 最嚴重（2–16 個 block）
- **H2**：hidden 8192 無法啟動是暫存器上限（已由編譯產物證實，本實驗用 hidden 6400／6656 再直接驗一次）
- **H3**：LMDeploy 合併在 A100 上沒有交換器內加總，只能走 2 次 barrier 的版本，且 block 數同樣是 ⌈token/8⌉

## 對照實驗設計（不改任何人的程式）

**固定資料量、改變 hidden**：大小相同時，hidden 越小、token 越多 → 一列一個 block 的做法 block 數越多（相當於「用很多小 block」模擬 cluster 把一列拆開的效果）；而 vLLM 的 block 數只看資料量，不受影響，當作對照。

- 大小：160K、320K、640K、1.25M（fp16）
- hidden：5120、2560、1280（token 數依大小換算；例如 320K → 32、64、128）
- 做法：`U_vllm`、`F_trt_one`、`F_trt_two`（torch 腳本，同 M9）、`F_lmd_force`（LMDeploy 強制合併，同 M9）
- **LMDeploy 當第二個對照組**：它在 hidden 較小時把多列塞進同一個 block（vdim ≤ 256 → 4 列、≤ 512 → 2 列），所以同樣大小時 block 數**不隨 hidden 改變**（例如 320K：5120／2560／1280 都是 4 個 block）。它不能用來驗 H1，但可以當「block 數不變 → 時間也應該差不多」的對照
- 另加 hidden 6400（800 執行緒）與 6656（832 執行緒）的 `F_trt_one` token=8，驗 H2
- 3 次獨立 launch、隨機順序；計時與正確性檢查同 M9
- **追蹤（一次）**：用 nsys 記錄 hidden 5120 時每個 kernel 實際的 block 數、每個 block 的執行緒數、暫存器數，確認上表

## 預期（量測前寫，量測後不改）

- **P1（H1）**：`F_trt_one`、`F_trt_two` 在 320K、640K、1.25M，hidden 1280 比 hidden 5120 快 ≥ 1.5 倍（不重疊）
- **P2（對照）**：`U_vllm` 與 `F_lmd_force` 在同樣大小，hidden 1280 與 5120 的中位數差 ≤ 15%（M9 的 3 次之間最大到約 15%）
- **P3（H2）**：hidden 6400 的 `F_trt_one` 可以啟動且正確；6656 以 `cudaErrorLaunchOutOfResources` 失敗（暫存器以每個 warp 256 個為單位配置：76 → 80／執行緒，65,536 / (80×32) = 25 個 warp = 800 執行緒）
- **P4**：nsys 記錄的 block 數與上表一致（`F_trt_one` = token 數、上限 108；`F_trt_two` = ⌈token/8⌉；LMDeploy 合併 = ⌈token/8⌉）
- 註：P2 的門檻原本寫 10%，在寫判準時改為 15%（理由見 P2），仍在量測前

## 判準

| 結果 | 條件 | 代表 |
|---|---|---|
| H1 成立 | P1 在 3 個大小 × 2 種做法中 ≥ 4/6 成立，且 P2 的 6 格中 ≥ 5 格成立 | 「缺少 cluster → block 太少」是主因；我們的設計必須讓一列由多個 block 分擔，並在 A100 上用全域記憶體或原子操作完成平方和的跨 block 加總 |
| H1 不成立 | P1 ≤ 2/6 | block 數不是主因，要另找（例如 barrier 成本、NVLink 讀取方式） |
| 中間 | 其他 | 停下來討論 |

## 指令

```bash
python3 exp/M10_why_fused_slow/run.py
.venv/bin/python exp/M10_why_fused_slow/analyze.py
```

## 結果

（量測後填）

## 結論

（量測後填）
