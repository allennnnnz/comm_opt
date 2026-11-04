# M9_fused_baselines

- **狀態**：進行中
- **核准**：2026-11-04（D019，使用者：「好 開始量 並且放行」）
- **類型**：結論性（每個設定 3 次獨立執行、隨機順序，同 M6／M7）

## 白話結論

（完成後填）

## 問題

在 8×A100（沒有交換器內加總）上，現有的「加總同步＋殘差相加＋RMSNorm 合併」做法，在生成階段的大小（每次 80 KB–1.3 MB）**離可達水準還有多遠**？夠遠，才值得做「兩輪都免 barrier 的合併 two-shot」；已經很近，就放掉這個方向。

順帶回答：(1) 在 A100 上，合併到底比「加總完再另跑 norm」省多少；(2) LMDeploy 為什麼在 ≤ 512 KB 把合併關掉。

## 已確認的事實（量測前，讀原始碼；見 `lit/notes/fused_ar_rmsnorm_a100.md`、D018）

- TensorRT-LLM 的合併 kernel 有兩種：one-shot Lamport（無 barrier，每卡推 7 份整份資料）與 two-shot sync（2 次 barrier，norm 對全部列重複做）。程式有 A100 路徑，但 TRT-LLM 本身沒在本機安裝
- FlashInfer 0.7.2.post1（`.venv-vllm` 已安裝）帶有**同源的** `trtllm_allreduce_fusion` 原始碼，也有 A100 路徑；FlashInfer 只是不為 sm80 編譯（`supported_major_versions=[9, 10, 12]`）。本實驗在測試腳本裡**用同一批原始碼另外編一份 sm_80 版本**，不改 FlashInfer 套件
- LMDeploy TurboMind 的合併 kernel（token 列 RS → 自己那段 norm → AG，2 次 barrier）在 ≤ 512 KB 會改走「一般 all-reduce＋另跑 norm」；它自帶 `src/turbomind/comm/test_comm.cu` 量測程式（hidden 寫死 2048）
- vLLM 0.33.0 在 A100 上不開合併：生成階段實際是「vLLM custom AR → `fused_add_rms_norm`」兩個 kernel
- **sm_80 試編成功**（2026-11-03，只編譯、未執行）：FlashInfer 的 `trtllm_allreduce.cu`、`trtllm_allreduce_fusion.cu`、`trtllm_moe_allreduce_fusion.cu` 以 `-gencode=arch=compute_80,code=sm_80` ＋ FlashInfer 的 `common_nvcc_flags` 編譯，CUDA 13.4，耗時 257 秒；產物只含 sm_80 程式碼，one-shot Lamport／two-shot sync 的 kernel 實例都在裡面（快取於 `~/.cache/flashinfer/0.7.2.post1/80/cached_ops/trtllm_comm_sm80/`）

## 要量的做法

| 代號 | 做法 | 來源與修改 |
|---|---|---|
| `U_vllm` | vLLM custom AR（自動選 1/2-stage）＋ vLLM `fused_add_rms_norm` | vLLM 0.33.0 原樣；**這是 vLLM 在 A100 上的實際行為** |
| `U_nccl` | torch NCCL all-reduce ＋ `fused_add_rms_norm` | PyTorch 自帶 NCCL 2.30.7 |
| `F_trt_one` | TRT-LLM 合併 one-shot Lamport（AR＋residual＋RMSNorm） | FlashInfer 0.7.2.post1 原始碼，另編 sm_80，`use_oneshot=True` |
| `F_trt_two` | TRT-LLM 合併 two-shot sync | 同上，`use_oneshot=False`（需 token 數 ≥ 8） |
| `F_lmd_auto` | LMDeploy `AllreduceResidualBiasRMSnorm`（原本的分派：≤ 512 KB 不合併） | LMDeploy 3228395 的 `test_comm`；只改 `main()` 讓 hidden 與 token 數可由參數指定、輸出每個 rank 的時間 |
| `F_lmd_force` | 同上，但把 ≤ 512 KB 的 `return false` 拿掉，強制合併 | 另一份編譯，只改 `fused_allreduce.cu` 的 `#if 1` → `#if 0` |

**LMDeploy 兩項的前提**：要先在本機編譯 LMDeploy（會再下載 fmt、cutlass、yaml-cpp、xgrammar、Catch2 等外部套件）。這一步被安全檢查擋下，**需要使用者放行**；不放行就只量前四項，並在結論註明。

## 預期（量測前寫，量測後不改）

- **P1（TRT one-shot 合併 vs vLLM 實際行為）**：80–320 KB，`F_trt_one` 比 `U_vllm` 快 ≥ 1.2 倍（省一次 kernel 啟動和一次 barrier）；1.3 MB 時 `F_trt_one` 比 `F_trt_two` 慢（每卡要收 7 份整份資料）
- **P2（TRT two-shot 合併）**：80–160 KB 比 `F_trt_one` 慢（2 次 barrier 的固定成本，M7 量到 two-shot 約 13 µs 起跳）；640 KB–1.3 MB 比 `U_vllm` 快 ≥ 1.1 倍
- **P3（LMDeploy 為什麼關掉）**：80–320 KB，`F_lmd_force` 比 `F_lmd_auto` 慢（2 次 barrier 的成本大於省下的 kernel 啟動）
- **P4（主問題）**：所有合併做法中每個大小最快的（下稱「合併最佳組合」），在 80 KB–1.3 MB 的 5 個大小中，**至少 3 個仍 ≥ 1.3 倍可達水準**——也就是還有空間

## 判準（量測前寫，量測後不改）

焦點大小：hidden 5120（Qwen2.5-32B，M5 用的模型）、token 數 8／16／32／64／128（fp16：80／160／320／640／1280 KB）。可達水準沿用 M6：`L0 + 2(N−1)/N × S / B`，L0 = 5.39 µs、B = 231.5 GB/s、N = 8。norm 本身的計算不另加（合併 kernel 裡 norm 只讀寫本卡記憶體，1.3 MB 約 1–2 µs）；這會讓可達水準偏低一點、比值偏高一點，判讀時註明。

| 結果 | 條件 | 代表 |
|---|---|---|
| **有空間** | 合併最佳組合 ≥ 1.3 倍可達水準的大小 ≥ 3/5，且這些大小的 3 次最小值也 ≥ 1.3 倍 | 「兩輪免 barrier 的合併 two-shot」值得做原型 |
| **沒有空間** | 合併最佳組合 ≤ 1.15 倍可達水準的大小 ≥ 3/5 | 放掉合併方向，回去討論題目定位 |
| **中間** | 其他 | 停下來和使用者討論 |

- P1–P3 各自用「3 次的 [最小, 最大] 不重疊」判定快慢；重疊就判「分不出來」
- 任何做法只要結果不正確（見下），該設定不納入合併最佳組合，並在結論列出

## 設定

- **變因**：做法（上表 6 種）× hidden（主：5120；次：8192）× token 數（1、8、16、32、64、128；8192 另加 256）
- **固定條件**：TP=8（GPU 0–7）、fp16、殘差相加＋RMSNorm（有 weight、eps=1e-6、無 bias）、一個 rank 一張卡
- **計時**：
  - torch 這邊（`U_*`、`F_trt_*`）：沿用 M6 方法——每個設定先暖機，再把 K=20 次「AR(＋norm)」連續錄進一個 CUDA graph，重播 R=20 次，每次操作時間 = 總時間 / (R×K)，每個 rank 各自計時、取 8 個 rank 的最大值。每次操作前不重設輸入（同 M6），但每次的 residual 會被更新，數值範圍用 `fused_add_rms_norm` 的輸出自然維持
  - LMDeploy：用它自帶的 `test_comm` 計時（CUDA event 包住連續 N 次呼叫取平均，每個 rank 一個執行緒），取 8 個 rank 的最大值。**計時方式和 torch 這邊不同**（沒有 CUDA graph、有 CPU 啟動成本），所以另跑一次 `test_comm` 的一般 all-reduce，和 M6/M7 的 vLLM／NCCL 數字對照，確認兩邊的計時差距；結論中 LMDeploy 的數字以它和自己的 `F_lmd_auto` 比較為主
- **正確性**：每個設定量完後做一次檢查——norm 輸出與更新後的 residual 對照 fp32 參考值（由各 rank 的輸入在 fp32 加總後算 residual＋RMSNorm），相對誤差 ≤ 2^-5 才算正確（同 M6）。LMDeploy 用 `test_comm` 自帶的驗證
- **重複**：每個設定 3 次獨立 launch，順序隨機（seed 記錄在 `schedule.tsv`），報中位數與 [最小, 最大]
- **環境紀錄**：`env.txt`（nvidia-smi、driver、CUDA、torch、vLLM、FlashInfer、LMDeploy commit、sm_80 編譯參數）；量測前確認機器上沒有其他 GPU 工作（R13）
- **理論上限**：見判準

## 指令（暫定）

```bash
# torch 這邊（.venv-vllm）
torchrun --nproc-per-node 8 exp/M9_fused_baselines/bench_fused.py --hidden 5120 --tokens 1,8,16,32,64,128 \
  --impls U_vllm,U_nccl,F_trt_one,F_trt_two --seed <s> --out results/M9_fused_baselines/<時間>/<設定>_r<k>.json
# LMDeploy（放行後）
third_party/lmdeploy/build{,_force}/bin/test_comm --hidden 5120 --tokens 1,8,16,32,64,128 cuda-ipc
exp/M9_fused_baselines/run.py      # 排程、隨機順序、R13 檢查、環境紀錄
exp/M9_fused_baselines/analyze.py  # 判準、圖
```

## 量測前的修正（看到任何正式數據之前）

- **試跑**（2026-11-04，hidden 5120、token 1／8／128、K=R=5，只確認能跑，數字不採用）：`F_trt_one`、`F_trt_two` 在 A100 上可執行且結果正確（相對誤差 ≤ 7.2e-4）；FlashInfer 的 cluster 大小問題沒有發生
- `F_trt_two`：FlashInfer 要求 token 數**大於** 8（assertion），所以 token = 8 也跳過，不是原先寫的「≥ 8」
- LMDeploy `test_comm` 的原始計時是「每次呼叫各自用一對 CUDA event、呼叫前先同步」，**只印 rank 0**，不是卡片原先寫的「連續 N 次取平均」。修正：保留原始計時（記為 isolated），另外加一段「連續 K=20 次呼叫包在一對 event 裡」的計時（記為 b2b，和 M6 的 graph 連續計時較接近），兩者都印出每個 rank。分析以 b2b、8 個 rank 取最大值為主
- LMDeploy 測試原本帶 bias、eps=1e-5；改成無 bias、eps=1e-6，與 torch 這邊一致
- 所有對 LMDeploy 的修改存在 `exp/M9_fused_baselines/lmdeploy_m9.patch`（測試程式＋`fused_allreduce.cu` 的 `#if 1` 改為 `#ifndef M9_FORCE_FUSED`；不加該旗標時行為與原版相同）
- **讀程式時的新發現**（量測前）：LMDeploy 一般 all-reduce 在小訊息（打包後 ≤ 1 MB）用的是 `Allreduce_LL16_V2`——分兩次傳、**資料與旗標放在同一個 8 byte 封包裡、不需要 barrier**。也就是說，A100 上「免 barrier 的 two-shot」LMDeploy 已經有了（LL 格式，資料量加倍），只是沒有和 norm 合併

## 已知風險

- `F_trt_*` 是 FlashInfer 的移植版、另編 sm_80，不是 TRT-LLM 官方建置。已比對：one-shot Lamport 與 two-shot sync 的 kernel 主體和 TRT-LLM `508f8dd` 的 `allReduceFusionKernels.cu` 逐段相同（推送／清除／輪詢 −0.0；複製 → barrier → 分段加總寫回 → barrier → 全部列做 norm）。**不同處在啟動參數**：FlashInfer 依暫存器用量調整 block 大小，且用 `cudaOccupancyMaxPotentialClusterSize` 查 cluster 大小（TRT-LLM 在 SM < 90 直接設 1）。若 A100 上查到 > 1，kernel 的索引會錯（A100 路徑假設一個 block 一列），正確性檢查會抓出
- 如果 `F_trt_*` 因上述原因結果錯誤，改法是在我們另編的那份原始碼複本裡把 cluster 大小固定為 1（和 TRT-LLM 官方的 A100 行為一致），套件本身不動；這是修改，會先問使用者並記錄在結論
- TRT one-shot Lamport 在 A100 上的正確性未知（新一代 GPU 才是官方測試平台）；若結果錯誤，照判準排除並記錄
- LMDeploy 的計時方式不同，跨框架的絕對時間比較要保守解讀

## 結果

（量測後填）

## 結論

（量測後填）
