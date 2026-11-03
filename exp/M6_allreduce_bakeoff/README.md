# M6_allreduce_bakeoff

- **狀態**：完成（run `20261103_083417`）
- **核准**：2026-11-03（D012，使用者：「好 先去測試這個」）
- **類型**：結論性（每個做法 5 次獨立 launch、隨機順序）

## 白話結論

1. **值得做，依事先定的判準成立。** 在生成階段常見的資料大小（64 KB–1 MB），最快的現成做法仍比這台機器做得到的水準慢 1.5–2.1 倍，五個大小全部超過 1.5 倍的門檻。
2. **沒有一套做法在所有大小都最快。** 很小的資料（≤ 8 KB）是 NCCL 的新低延遲模式最快；中等大小是 vLLM 和微軟的 MSCCL++ 輪流領先。NCCL 的低延遲模式一過 8 KB 就突然變慢 2.5 倍，像是選錯了內部做法。
3. **慢的原因不只是「同步確認」，搬資料本身也沒效率。** 最快的兩套在搬資料時只用到這台機器頻寬的一到五成；資料越大，同步占的比例越小（256 KB 以上只占兩到三成）。這和 SiFAR 專攻的「小資料的同步」是不同的問題。
4. **NCCL 預設的做法最慢**：比最快的做法慢 2–4 倍；新版 NCCL（2.32.3）比 PyTorch 自帶的舊版（2.30.7）快約兩成。
5. **對整體的意義**：如果把這段加總同步做到可達水準，生成文字（一次 8–128 句）預估可快約 6–14%。

## 問題

在 8 張 A100（沒有交換器內歸約）上，現成的加總同步做法在生成階段常見的資料大小（64 KB–1 MB，對應一次處理 8–128 句）能做到多快？離「這台機器做得到的水準」還差多少？時間花在同步確認還是搬資料？這決定「生成階段加總同步」值不值得當實作題目（D011）。

## 比較的做法

| 名稱 | 實作 | NCCL 版本 | 計時工具 |
|---|---|---|---|
| `nccl_eager` | NCCL 預設路徑（不登記記憶體） | 系統 2.32.3 | nccl-tests |
| `nccl_graph` | 同上，CUDA graph 模式（`-G 20`） | 2.32.3 | nccl-tests |
| `nccl_symm_graph` | NCCL symmetric kernels：登記記憶體（`-R 2`）+ CUDA graph | 2.32.3 | nccl-tests |
| `mscclpp_graph` | MSCCL++ 的 NCCL 相容介面（`LD_PRELOAD=libmscclpp_nccl.so`）+ CUDA graph | — | nccl-tests |
| `torch_nccl` | `torch.distributed.all_reduce`，CUDA graph | PyTorch 自帶 2.30.7 | `bench_torch.py` |
| `vllm_car` | vLLM 0.33.0 custom all-reduce，CUDA graph | — | `bench_torch.py` |
| `symm_one_shot`、`symm_two_shot` | PyTorch 2.14 SymmetricMemory one-shot／two-shot，CUDA graph | — | `bench_torch.py` |

- **沒有納入**：NVIDIA/ETH 團隊的 low-latency-nccl（ss16118/low-latency-nccl）——下載與編譯被安全檢查擋下，需使用者決定是否放行；SiFAR（沒有公開程式碼）；TensorRT-LLM／FlashInfer（vLLM 在 SM80 上不啟用，且需大型建置）
- **計時方式**：都在 CUDA graph 中連續做多次加總（生成階段每步也是連續做 128 次），取每次平均。nccl-tests 用 `-a 3`（跨 rank 取最大），`bench_torch.py` 也報跨 rank 的最大值
- **和 M5 的差別**：M5 的「傳輸時間」取 8 張卡中最短的那次，排除了同步等待；本實驗量的是包含同步在內的每次成本，所以同一做法會比 M5 的數字大（例：vLLM 4 KB，M5 6 µs，本實驗試跑 10 µs）

## 預期（量測前寫，量測後不改）

**揭露（補記於量測前）**：在寫完預期之後、正式量測之前，又做了 nccl-tests 三種 NCCL 模式與 MSCCL++ 的試跑（4 KB–1 MB，各 1 次），看到 NCCL 登記記憶體後確實啟用 symmetric kernels（`impl=SYM`，4 KB 5.8 µs），MSCCL++ 4 KB 6.2 µs。預期段落沒有因此修改。

**揭露**：寫這段之前，已看過 `bench_torch.py` 的一次試跑（4 KB–1 MB、每點 1 次；`torch_nccl` 31–50 µs、`vllm_car` 10–30 µs、`symm_one_shot` 19–58 µs、`symm_two_shot` 16–28 µs）。所以下列關於這四種做法的預期不是盲寫的；nccl-tests 系列與 MSCCL++ 的預期是盲寫的。

- NCCL 的 CUDA graph 模式（`nccl_graph`）在小資料只比一般模式快一點（試跑中 `torch_nccl` 的 graph 模式仍要 31 µs）
- NCCL symmetric kernels（`nccl_symm_graph`）若在 A100 上真的啟用，小資料會明顯變快，可能到 10–15 µs；也可能根本不啟用（記錄 `-U 1` 回報的實作名稱判斷）
- MSCCL++ 在小資料與 vLLM 相當或更快（論文：在舊版 vLLM 上幾何平均快 1.4 倍）
- 64 KB–1 MB 最快的現成做法約 13–28 µs
- 依下面的判準，預期結果是「部分」：同步與啟動的固定開銷（約 10 µs）占了大部分時間，搬資料的部分相對有效率

## 判準（量測前寫，量測後不改）

- **可達水準** bound(S) = L0 + 2·(7/8)·S / B
  - B = 231.5 GB/s：M0 量到的 8 卡 all_reduce 最高 busbw（`figures/M0_baseline/20261102_134841/peaks.csv`）
  - L0 = 本實驗所有做法在 1 KB 時最快的每次時間（這台機器上「一次完整加總」的實測最低固定開銷）
- **最快現成做法** best(S) = 各做法 5 次 launch 中位數的最小值
- **R(S) = best(S) / bound(S)**，在 S = 64K、128K、256K、512K、1M 五個點計算

| 結果 | 條件 | 代表 |
|---|---|---|
| 值得實作 | R ≥ 1.5 的點 ≥ 3 個 | 搬資料的效率還有明顯空間 |
| 空間已被補上 | 五個點都 R ≤ 1.2 | 現成做法已接近可達水準 |
| 部分 | 其他 | 和使用者討論 |

- **另外報告**（不影響判準）：
  - 每個做法的固定開銷 t(1K) 與「搬資料部分」t(S) − t(1K)
  - 固定開銷占 64 KB–1 MB 總時間的比例 → SiFAR 那類「減少同步」技巧的潛在空間
  - 若改用 M5 推估的 6 µs 當 L0，R 會是多少（對照 D011 的「2.2–2.5 倍」說法）

## 設定

- 8 張 A100、**fp16**、大小 1 KB–8 MB（×2，共 14 點）
- **量測前的修正**：原本設計用 bf16，但 MSCCL++ 的 NCCL 相容介面回報版本 0.9.0，nccl-tests 因此認為不支援 bf16 並改用 fp32（試跑時發現）。為了讓所有做法條件相同，全部改用 fp16（每個數值同樣 2 bytes，資料量相同）
- 每個 launch：nccl-tests 系列每次跑一個做法；`bench_torch.py` 每次跑四個做法（內部順序以 seed 打亂）
- 共 (4 個 nccl-tests 做法 + 1 個 torch 組) × 5 次 = 25 次 launch，seed 20261105 打亂順序（R11）
- 每個 launch 前確認沒有其他 GPU 工作（R13）
- 正確性：`bench_torch.py` 每個大小檢查一次（對 fp32 精確和，最大相對誤差 ≤ 2^-5）；nccl-tests 用 `-c 1` 檢查

## 指令

```bash
python3 exp/M6_allreduce_bakeoff/run.py
.venv/bin/python exp/M6_allreduce_bakeoff/analyze.py
```

## 結果

- 原始數據：`results/M6_allreduce_bakeoff/20261103_083417/`（25 次 launch 全部 rc=0；另有 SymmetricMemory 的 bf16 補跑 5 次 `torch_bf16symm_r*.json`）
- 分析：`.venv/bin/python exp/M6_allreduce_bakeoff/analyze.py results/M6_allreduce_bakeoff/20261103_083417` → `figures/M6_allreduce_bakeoff/20261103_083417/`（`summary.csv`、`ratio.csv`、`fixed_vs_data.csv`、`launches.csv`、`fig_bakeoff.png`）
- 正確性：所有做法、所有大小都通過（nccl-tests `#wrong` = 0；`bench_torch.py` 最大相對誤差 ≤ 1%）
- **量測後的補跑（偏差）**：PyTorch SymmetricMemory one-shot／two-shot 沒有 fp16 kernel（`NotImplementedError ... not implemented for 'Half'`），fp16 的 28 個點全部失敗；以 bf16（相同 bytes）補跑 5 次。這兩種做法在任何大小都不是最快，不影響判準結果

每次 all-reduce 的時間（µs，5 次 launch 中位數，跨 rank 取最大）：

| 大小 | NCCL 一般 | NCCL graph | NCCL 低延遲（SYM） | torch NCCL 2.30.7 | MSCCL++ | vLLM | symm one-shot* | symm two-shot* |
|---|---|---|---|---|---|---|---|---|
| 1K | 29.0 | 23.4 | **5.4** | 30.5 | 5.6 | 5.9 | 14.4 | 14.2 |
| 8K | 29.6 | 23.5 | 6.6 | 31.3 | 7.0 | 6.6 | 15.9 | 14.1 |
| 16K | 30.8 | 24.2 | 16.7 | 32.5 | 7.8 | **6.8** | 15.6 | 14.1 |
| 64K | 35.8 | 26.5 | 17.7 | 39.7 | 10.5 | **9.0** | 16.8 | 15.7 |
| 128K | 36.5 | 24.5 | 17.2 | 39.3 | **11.2** | 11.7 | 19.0 | 20.3 |
| 256K | 35.9 | 24.6 | 20.2 | 39.4 | **15.3** | 17.2 | 25.2 | 21.2 |
| 512K | 35.7 | 24.7 | 31.4 | 40.8 | 19.1 | **18.8** | 33.6 | 21.6 |
| 1M | 46.9 | 33.8 | 29.0 | 52.6 | 28.2 | **22.0** | 51.2 | 24.3 |
| 8M | 142.2 | 137.0 | 108.8 | 145.4 | 93.3 | **79.4** | 320.0 | 96.2 |

\* bf16 補跑。完整 14 個大小見 `summary.csv`；重複之間的差距多在 1% 以內，vLLM 與 symm 系列 11–21%

判準（量測前寫）：L0 = 5.39 µs（NCCL SYM @ 1K），B = 231.5 GB/s

| 大小 | 最快做法 | 時間 | 可達水準 | R | R（L0 = 6 µs） |
|---|---|---|---|---|---|
| 64K | vLLM | 9.02 | 5.88 | **1.53** | 1.39 |
| 128K | MSCCL++ | 11.19 | 6.38 | **1.75** | 1.60 |
| 256K | MSCCL++ | 15.27 | 7.37 | **2.07** | 1.91 |
| 512K | vLLM | 18.78 | 9.35 | **2.01** | 1.88 |
| 1M | vLLM | 21.98 | 13.31 | **1.65** | 1.58 |

→ R ≥ 1.5 的點 5/5 → **值得實作**

固定開銷與搬資料（固定開銷 = 該做法在 1 KB 的時間；搬資料效率 = 1.75·S ÷（總時間 − 固定開銷））：

| 大小 | vLLM 固定占比 | vLLM 搬資料效率 | MSCCL++ 固定占比 | MSCCL++ 搬資料效率 | NCCL SYM 搬資料效率 |
|---|---|---|---|---|---|
| 64K | 66% | 37 GB/s | 54% | 24 GB/s | 9 GB/s |
| 256K | 35% | 41 GB/s | 37% | 47 GB/s | 31 GB/s |
| 1M | 27% | 114 GB/s | 20% | 81 GB/s | 78 GB/s |

（這個拆法是近似：同一做法在不同大小可能換內部演算法，例如 vLLM 在 256 KB 前後從 1-stage 換成 2-stage）

## 結論

### 對照預期

| 預期 | 結果 | 判定 |
|---|---|---|
| NCCL graph 模式只比一般模式快一點 | 快約 20%（23–34 vs 29–47 µs） | 符合 |
| NCCL symmetric kernels 若啟用，小資料 10–15 µs | 啟用（`SYM`），≤ 8 KB 只要 5.4–6.6 µs，是所有做法中最快 | 符合，且比預期好 |
| — | **意外**：NCCL SYM 在 16 KB 突然跳到 16.7 µs，64 KB–1 MB 比 vLLM／MSCCL++ 慢 1.3–2 倍 | 未預期 |
| MSCCL++ 小資料與 vLLM 相當或更快 | ≤ 8 KB 相當；128–256 KB 較快；其他大小 vLLM 較快 | 大致符合 |
| 64 KB–1 MB 最快 13–28 µs | 9–22 µs | 比預期快 |
| 判準結果是「部分」 | **「值得實作」**（5/5） | 不符（預期過於保守） |

### 觀察到的現象

- **P1 搬資料階段的效率低**：64 KB–1 MB 最快的做法，扣掉固定開銷後的搬資料效率只有可達頻寬的 10–50%；固定開銷在 256 KB 以上只占 20–37%。改進空間在搬資料本身，不只在同步
- **P2 沒有單一最佳做法**：最快的做法隨大小換了三次（NCCL SYM → vLLM → MSCCL++ → vLLM）
- **P3 NCCL symmetric kernels 在中等大小的選擇不佳**：8 KB → 16 KB 時間跳 2.5 倍且之後一直落後；同一時期 NVIDIA/ETH 的 low-latency-nccl 分支有「選擇低延遲 symmetric kernel 的新規則」的修改（未驗證，被安全檢查擋下）
- **P4 NCCL 版本差異**：同樣是 CUDA graph 模式，系統 NCCL 2.32.3 比 PyTorch 自帶的 2.30.7 快約 20–40%

### 限制

- 只量了連續、無計算的加總（模擬生成階段每步的 128 次加總），沒有和計算交錯；真實模型中每次加總之間有計算，同步等待的行為可能不同
- fp16（SymmetricMemory 為 bf16），與 vLLM 實際使用的 bf16 位元組數相同
- 「固定開銷 vs 搬資料」是用 1 KB 的時間近似，不是在 kernel 內部直接量測
- low-latency-nccl 沒有納入

## 下一步（提案，需核准）

- 決定是否放行 low-latency-nccl（它可能已改善 P3）
- 拆解 P1：為什麼 64 KB–1 MB 搬資料只有 10–50% 效率（用多少 GPU 核心、每次搬多少、同步方式），作為題目的機制依據
- 進入選題關卡：以「A100 上生成階段中等大小加總同步」為題，用 brainstorming skill 整理研究問題與 claims
