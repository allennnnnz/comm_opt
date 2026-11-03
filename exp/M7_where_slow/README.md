# M7_where_slow

- **狀態**：完成（run `20261103_091734`）
- **核准**：2026-11-03（D014，使用者：「先找慢在哪，我放行可以研究低延遲版本 nccl」）
- **類型**：結論性（每個設定 3 次獨立 launch、隨機順序）

## 白話結論

1. **NCCL 的低延遲模式在 A100 上挑錯了「用幾個 GPU 核心」。** 在 16 KB–1 MB，它只用 1–8 個核心，但 A100 需要 8–32 個才跑得快；改成最佳設定後快 1.35–2.33 倍。這是「挑選」的問題，不是做法本身的問題。
2. **但就算每個大小都挑最好的設定，還是追不上可達水準。** 把目前所有做法、所有設定裡最快的拿來比，64 KB–1 MB 仍慢 1.4–1.9 倍（5 個大小中有 4 個超過 1.5 倍）→ 依事先定的判準，**需要新的做法**。
3. **原因是兩種現有做法各有一個補不掉的缺點。** 「一次傳完」只需要確認一次（約 5.4 µs），但每張卡要讀其他 7 張卡的全部資料，資料一大就很慢；「分兩次傳」要搬的資料少很多，但要確認兩次，固定花掉約 13 µs。在 64 KB–1 MB，剛好兩邊都吃虧。
4. **我們算的「可達水準」等於「只確認一次、又只搬分兩次傳那麼少的資料」**——目前沒有任何做法同時做到這兩件事。這就是可以研究的空間。
5. 其他：GPU 核心超過 16–32 個就不再變快；vLLM 在 256 KB 切換做法的位置大致正確；MSCCL++ 的「直接用登記記憶體」開關沒有差別；低延遲版 NCCL 新增的做法在 A100 上算錯，只有兩個標準做法可用。

## 問題

M6 顯示 64 KB–1 MB 最快的現成做法離可達水準還差 1.5–2.1 倍。差在哪裡？
1. 現成做法是不是**選錯了內部做法或 GPU 核心數**？如果改用最佳設定就能追上，問題在「挑選」；追不上，才需要新的做法
2. 加總時間隨使用的 GPU 核心數（CTA 數）怎麼變？——判斷搬資料是不是被「用的核心太少」限制
3. NVIDIA/ETH 的 low-latency-nccl 在 A100 上能不能用、有沒有比較快

## 已知線索（讀原始碼，2026-11-03）

- NCCL 2.32.3 的低延遲模式（symmetric kernels）用成本模型挑「內部做法 + CTA 數」（`src/tuning/sym_model/lsa.cc`），可用 `NCCL_SYM_KERNEL`、`NCCL_SYM_CTAS` 強制指定；CTA 上限 64（`ncclSymkMaxBlocks`）。A100 上可用的只有 `AllReduce_AGxLL_R`（先收齊再加，one-shot 類）與 `AllReduce_RSxLD_AGxST`（先分工加再交換，two-shot 類）；其他三種需要新一代 GPU
- vLLM custom AR：每個 block 512 threads，block 數上限固定；TP=8 時 256 KB 以下用 1-stage、以上用 2-stage，可用 `VLLM_CUSTOM_ALLREDUCE_ALGO` 強制
- MSCCL++ 的 NCCL 相容介面有 `MSCCLPP_NCCL_SYMMETRIC_MEMORY` 開關（M6 用預設）

## 已看過的試跑（寫預期之前，全部單次）

- M6 試跑的 NCCL SYM 自選 CTA 數：4 KB 用 4、16 KB 用 1、64 KB 用 2、256 KB 用 6、1 MB 用 8
- low-latency-nccl（NCCL 2.29.2 分支，commit 見設定）：**新增的做法（Lamport 系列、LLBuffer 系列）在 A100 上結果全錯**（`#wrong` 幾乎等於元素數、時間約 1.5 µs），預設挑的 `AllReduce_LLBuffer_R8` 也是；保留的 `AGxLL_R`、`RSxLD_AGxST` 正確
- low-latency-nccl 強制 `RSxLD_AGxST`：16 KB 12.7、64 KB 12.7、256 KB 14.2 µs；強制 `AGxLL_R`：8.3、13.2、33.1 µs

## 預期（量測前寫，量測後不改）

- **NCCL SYM 選擇不佳**：在 16 KB–256 KB，最佳強制設定比 NCCL 自選快 ≥ 1.3 倍（試跑已暗示，所以這條不是盲猜）
- **CTA 數的影響**：小資料（≤ 16 KB）CTA 越多越慢或不變（同步成本隨 CTA 增加）；中等資料（64 KB–1 MB）CTA 增加到 8–16 時明顯變快，之後趨平
- **最佳設定能不能追上可達水準**：預期不能。所有做法的最佳設定合起來（下稱「最佳組合」），在 64 KB–1 MB 仍比可達水準慢 ≥ 1.3 倍
- vLLM 在 256 KB 的 1-stage／2-stage 切換點接近最佳（差距 ≤ 10%）
- MSCCL++ 開 symmetric memory 後中等大小變快（省一次複製），幅度 10–30%

## 判準（量測前寫，量測後不改）

統計：每個設定 3 次獨立 launch 的中位數與 [最小, 最大]；「不重疊」= 區間不相交。

| 項目 | 條件 |
|---|---|
| NCCL SYM 選擇不佳成立 | 16K、32K、64K、128K、256K 中至少 3 個大小：最佳強制設定 ≤ NCCL 自選的 1/1.3，且區間不重疊 |
| 「挑選」就能補上大部分差距 | 最佳組合（M6 ∪ M7 所有正確的設定）在 64K–1M 五個大小都 R_env ≤ 1.2（R_env = 最佳組合 ÷ M6 的可達水準） |
| 需要新的做法 | 最佳組合在 64K–1M 至少 3 個大小 R_env ≥ 1.5 |
| 介於中間 | 其他 → 和使用者討論 |

## 設定

- 8 張 A100、fp16、1 KB–8 MB（×2），CUDA graph（`-G 20`），登記記憶體（`-R 2`），`-a 3`
- **A. NCCL 2.32.3 SYM 掃描**：`NCCL_SYM_KERNEL` ∈ {AGxLL_R, RSxLD_AGxST} × `NCCL_SYM_CTAS` ∈ {1, 2, 4, 8, 16, 32, 64}，加上自選（不設定）→ 15 種
- **B. low-latency-nccl（2.29.2 分支，commit 5357eff）**：自選、強制 AGxLL_R、強制 RSxLD_AGxST → 3 種（用針對它的 headers 重編的 `third_party/nccl-tests-llnccl`）；新增的 Lamport／LLBuffer 做法已確認在 A100 上錯誤，不量時間
- **C. vLLM**：`VLLM_CUSTOM_ALLREDUCE_ALGO` ∈ {1stage, 2stage} → 2 種（`bench_torch.py`）
- **D. MSCCL++**：`MSCCLPP_NCCL_SYMMETRIC_MEMORY` ∈ {0, 1} → 2 種
- 共 22 種 × 3 次 = 66 次 launch，seed 20261106 打亂順序；每個 launch 前確認沒有其他 GPU 工作
- 正確性：nccl-tests `-c 1`、`bench_torch.py` 對 fp32 精確和；**結果錯誤的點不納入任何時間比較**

## 指令

```bash
python3 exp/M7_where_slow/run.py
.venv/bin/python exp/M7_where_slow/analyze.py
```

## 結果

- 原始數據：`results/M7_where_slow/20261103_091734/`（66 次 launch；63 次 rc=0，`B_auto` 的 3 次因結果錯誤被 nccl-tests 中止（rc=7），與試跑一致）
- 分析：`.venv/bin/python exp/M7_where_slow/analyze.py results/M7_where_slow/20261103_091734` → `figures/M7_where_slow/20261103_091734/`（`nccl_sym_regret.csv`、`cta_sweep_table.csv`、`envelope.csv`、`summary.csv`、`fig_cta_sweep.png`）
- 正確性：除 `B_auto` 外所有點都正確

**A. NCCL SYM 自選 vs 最佳強制設定**（µs，3 次中位數）

| 大小 | 自選（CTA 數） | 最佳強制設定 | 時間 | 自選 ÷ 最佳 |
|---|---|---|---|---|
| 8K | 6.53（4） | AGxLL_R × 8 | 6.02 | 1.08 |
| **16K** | 16.67（1） | AGxLL_R × 8 | 7.16 | **2.33** |
| **32K** | 17.66（2） | AGxLL_R × 32 | 9.07 | **1.95** |
| **64K** | 17.92（2） | AGxLL_R × 32 | 12.94 | **1.39** |
| **128K** | 17.19（4） | RSxLD_AGxST × 32 | 12.77 | **1.35** |
| **256K** | 20.62（6） | RSxLD_AGxST × 64 | 14.05 | **1.47** |
| 512K | 31.42（6） | RSxLD_AGxST × 64 | 15.44 | 2.04 |
| 1M | 28.99（8） | RSxLD_AGxST × 32 | 19.86 | 1.46 |

→ 16K–256K 五個大小都 ≥ 1.3 且區間不重疊 → **「NCCL SYM 選擇不佳」成立**

**時間 vs CTA 數**（µs）

| | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| AGxLL_R @64K | 76.5 | 42.8 | 23.5 | 15.6 | 13.9 | 12.9 | 13.2 |
| AGxLL_R @1M | 1099.7 | 547.6 | 277.1 | 148.5 | 108.2 | 98.2 | 93.8 |
| RSxLD_AGxST @64K | 25.4 | 17.7 | 17.3 | 13.2 | 13.0 | 13.0 | 13.3 |
| RSxLD_AGxST @1M | 83.5 | 47.7 | 30.5 | 29.0 | 28.5 | 19.9 | 19.9 |

- AGxLL_R 在 1 KB 最低 5.4 µs，但資料成本是每張卡讀 7 × S；RSxLD_AGxST 最低約 13 µs（兩輪同步），資料量只有 2 × 7/8 × S
- CTA 超過 16–32 不再變快

**最佳組合（M6 ∪ M7 所有正確設定）vs 可達水準**

| 大小 | 最佳組合 | 時間 | 可達水準 | R_env | 比 M6 最佳快 |
|---|---|---|---|---|---|
| 64K | vLLM（M6） | 9.02 | 5.89 | **1.53** | 1.00 |
| 128K | MSCCL++ | 10.73 | 6.38 | **1.68** | 1.04 |
| 256K | low-latency-nccl RSxLD_AGxST | 14.05 | 7.37 | **1.91** | 1.09 |
| 512K | low-latency-nccl RSxLD_AGxST | 15.10 | 9.35 | **1.61** | 1.24 |
| 1M | low-latency-nccl RSxLD_AGxST | 19.24 | 13.32 | 1.44 | 1.14 |

→ R_env ≥ 1.5 的點 4/5 → **「需要新的做法」**

**C. vLLM 1-stage vs 2-stage**（µs）：64K 10.5 vs 19.3；128K 13.7 vs 19.3；256K 18.5 vs 19.9；512K 27.2 vs 21.5；1M 44.0 vs 25.0
**D. MSCCL++ `MSCCLPP_NCCL_SYMMETRIC_MEMORY` 0 vs 1**：所有大小差距 ≤ 10%（例：128K 10.7 vs 11.1、512K 18.1 vs 19.8）

## 結論

### 對照預期

| 預期 | 結果 | 判定 |
|---|---|---|
| NCCL SYM 在 16–256 KB 選擇不佳（≥ 1.3 倍） | 1.35–2.33 倍，5/5 | 符合 |
| 小資料 CTA 越多不變或越慢；中等資料到 8–16 個明顯變快後趨平 | 小資料大致不變；中等資料到 16–32 個才趨平 | 大致符合（需要的 CTA 比預期多） |
| 最佳組合仍比可達水準慢 ≥ 1.3 倍 | 1.44–1.91 倍 | 符合 |
| vLLM 切換點接近最佳（≤ 10%） | 256 KB 時 1-stage 快 7%，512 KB 起 2-stage 較快 | 符合 |
| MSCCL++ 開 symmetric memory 快 10–30% | 沒有差別 | 不符 |

### 機制解讀

- 64 KB–1 MB 的差距主要不是「核心不夠」（超過 16–32 個就飽和），而是演算法結構：one-shot 只同步一次但每卡讀 7 × S；two-shot 每卡只搬 1.75 × S，但多一輪同步，固定成本約 13 µs（one-shot 約 5.4 µs）
- 可達水準 L0 + 1.75·S/B 正好是「一次同步 + two-shot 的資料量」；沒有現成做法同時做到
- NCCL 2.32.3 的成本模型在 A100 上低估了需要的 CTA 數（自選 1–8 個，最佳 8–64 個）；這是可以單獨修正的小問題

### 限制

- 同 M6：連續無計算的加總、fp16、單機
- 「兩輪同步」的成本是從 RSxLD_AGxST 在小資料的最低時間推得，沒有在 kernel 內直接量
- low-latency-nccl 的新做法在 A100 上無法評估（結果錯誤），只比較了它保留的標準做法

## 下一步（提案，需核准）

- 新穎性檢查：NVIDIA/ETH《Every µs Matters》的「LL128 原子加 two-shot」可能就是「一次同步的 two-shot」，但它需要 NVLink 的 cache-line 原子加法，A100 是否支援未確認——要先確認
- 進入選題（G2）：研究問題暫定為「在沒有交換器加速的 GPU 上，讓中等大小的加總同步同時做到一次同步與 two-shot 的資料量」，並附帶修正 NCCL 在 A100 上的 CTA 選擇
