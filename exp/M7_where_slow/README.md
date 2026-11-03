# M7_where_slow

- **狀態**：已核准（D014），建置中
- **核准**：2026-11-03（D014，使用者：「先找慢在哪，我放行可以研究低延遲版本 nccl」）
- **類型**：結論性（每個設定 3 次獨立 launch、隨機順序）

## 白話結論

（實驗完成後填）

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
- **B. low-latency-nccl（2.29.2 分支）**：自選、強制 AGxLL_R、強制 RSxLD_AGxST → 3 種（用針對它的 headers 重編的 `third_party/nccl-tests-llnccl`）；新增的 Lamport／LLBuffer 做法已確認在 A100 上錯誤，不量時間
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

（尚未執行）

## 結論

（尚未執行）

## 下一步
