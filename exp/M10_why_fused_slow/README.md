# M10_why_fused_slow

- **狀態**：完成（run `20261104_083556`）
- **核准**：2026-11-04（使用者：「調查清楚 a100 是少了什麼功能，造成預期比較快的方法在 a100 沒有比較快」）
- **類型**：結論性（對照實驗 3 次獨立執行）＋一次追蹤（記錄實際啟動參數，不用來下快慢結論）

## 白話結論

1. **「A100 少了讓多個運算單元合作處理同一列的功能」只解釋了一部分，不是主因。** 我們讓同樣的資料量能派出 4 倍的運算單元：TensorRT-LLM「分兩次傳」快了 1.3–1.6 倍，但仍比 vLLM 慢；「一次全送」幾乎沒變快。依事先定的標準，這個假說**不成立**（6 格只有 1 格達標）。
2. **它確實是大模型跑不起來的原因**：沒有這個功能，一整列只能塞給一個運算單元，資源不夠就無法啟動。實際邊界比我算的還低（資料寬 6400 也失敗）。
3. **真正讓合併沒變快的，是它們搬資料的方式在 A100 上本來就比較差。** 合併最多只能省掉 vLLM 另外做正規化的那一步，大約 5 微秒；但它們的加總部分在 A100 上慢更多：「一次全送」要搬 7 倍的資料，「分兩次傳」要多抄一次、多等兩次、還每張卡把正規化重做 8 遍。
4. **對我們的新做法的意義**：光是「多派運算單元」不夠。省時間要靠「不用等齊」加上「少搬資料」，這兩點是設計的重點。
5. **下一步**：直接量「線路用了多少、等待佔多少時間」，才能確定省時間的空間到底在哪（之前提的第 2、4 項觀察）。


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

- 原始數據：`results/M10_why_fused_slow/20261104_083556/`（26 次 launch 全部 rc=0；`torch_h*`、`lmdforce_h*`、`reglimit_h*`、`nsys_*.nsys-rep`、`schedule.tsv`、`env.txt`）
- 分析：`exp/M10_why_fused_slow/analyze.py` → `figures/M10_why_fused_slow/20261104_083556/`（`summary.csv`、`kernel_launch_configs.csv`、`fig_hidden_sweep.png`）
- 所有計時設定 3/3 正確

**同樣資料量、不同 hidden（中位數 µs，8 個 rank 取最大）：**

| 做法 | 大小 | hidden 5120 | 2560 | 1280 | 5120→1280 |
|---|---|---|---|---|---|
| `F_trt_one` | 160K | 18.9 | 16.5 | 15.9 | 1.19× |
| | 320K | 26.5 | 24.4 | 24.9 | 1.06× |
| | 640K | 42.6 | 42.6 | 42.5 | 1.00× |
| | 1.25M | 93.7 | 76.5 | 78.1 | 1.20× |
| `F_trt_two` | 160K | 67.8 | 45.8 | 34.6 | 1.96× |
| | 320K | 55.2 | 40.6 | 34.1 | **1.62×** |
| | 640K | 51.6 | 41.6 | 36.6 | 1.41× |
| | 1.25M | 55.6 | 47.0 | 43.8 | 1.27× |
| `U_vllm`（對照） | 160K–1.25M | 17.8–27.8 | 18.0–28.5 | 17.6–26.9 | 0.97–1.10 |
| `F_lmd_force`（對照） | 160K–1.25M | 23.4–33.8 | 23.2–35.5 | 23.5–35.0 | 0.95–1.05 |

**暫存器上限**：hidden 6400 與 6656 的 `F_trt_one` 都 3/3 `cudaErrorLaunchOutOfResources`。

**nsys 記錄的啟動參數（GPU 0，hidden 5120，token 16／32／64／128）**：
- `F_trt_one`：16／32／64／108 個 block × 640 執行緒，76 暫存器 —— 與推算一致
- `F_trt_two`：2／4／8／16 個 block × 640 執行緒，76 暫存器 —— 與推算一致
- LMDeploy 合併：2／4／8／16 個 block × 1024 執行緒，56 暫存器 —— 與推算一致
- vLLM 加總：20 或 36 個 block × 512 執行緒；vLLM 另跑的 norm kernel 每次約 4.5–6.1 µs（nsys 記錄的 kernel 時間；這次追蹤不是 CUDA graph，只用來看量級）

## 結論

| 預期 | 結果 | 判定 |
|---|---|---|
| P1：`F_trt_one`、`F_trt_two` 在 320K–1.25M，hidden 1280 比 5120 快 ≥ 1.5 倍 | 1/6（只有 `F_trt_two` 320K 的 1.62 倍）；`F_trt_one` 1.00–1.20 倍；`F_trt_two` 1.27–1.62 倍 | **不符** |
| P2：對照組變化 ≤ 15% | 6/6（0.97–1.10） | 符合 |
| P3：hidden 6400 可啟動、6656 失敗 | 兩者都失敗 | **不符**（邊界在 640 與 800 執行緒之間；我的暫存器配置算法不對，原因未查） |
| P4：block 數與推算一致 | 一致 | 符合 |

- **判準結果：H1 不成立**（P1 1/6 ≤ 2/6）。對照組穩定，代表實驗本身有效
- **解讀**：
  - 「沒有 cluster → 一列一個 block」**有影響但不是主因**：它讓 two-shot 合併在 160K 只有 2 個 block，給更多 block 會快 1.3–2.0 倍，但仍比 vLLM 慢 1.5–1.9 倍；one-shot 合併幾乎不受影響
  - 「沒有 cluster」**確實造成大 hidden 無法啟動**（M9 的 8192、這次的 6400）
  - 剩下的差距來自做法本身在 A100 上的代價（推論，未直接量測）：one-shot 每張卡收 7 倍資料；two-shot 先把輸入抄進自己的緩衝區、2 次 barrier、再把 norm 對全部列重做。合併最多只能省 vLLM 那個約 5 µs 的 norm kernel，抵不過這些代價
  - LMDeploy 合併的 block 數不隨 hidden 改變、時間也不變，與它「一列一個 block、多列塞一個 block」的設計一致；它在 A100 上缺的是交換器內加總（程式確認），本實驗無法驗證這點的影響
- **對設計的意義**：新做法不能只靠「多派 block」；必須同時拿掉 barrier、不多抄資料、只收 1/8。要知道這些各值多少，需要直接量時間組成與線路使用率

### 限制

- 用「改 hidden」模擬「多 block」，同時也改了每個 block 的執行緒數（640→160）與每列的長度；不是完全等同於 cluster
- 沒有 H100 可對照，「新一代 GPU 上這些做法比較快」是 TRT-LLM／vLLM 的設定表與文件的說法，不是我們量的

## 下一步

`[需裁決]` 量線路使用率與一次加總的時間組成（等待／搬資料／啟動），作為新做法設計的依據。
