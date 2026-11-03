# M6_allreduce_bakeoff

- **狀態**：已核准（D012），建置中
- **核准**：2026-11-03（D012，使用者：「好 先去測試這個」）
- **類型**：結論性（每個做法 5 次獨立 launch、隨機順序）

## 白話結論

（實驗完成後填）

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

（尚未執行）

## 結論

（尚未執行）

## 下一步
