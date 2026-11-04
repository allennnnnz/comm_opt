# M14_fused_prototype

- **狀態**：進行中
- **核准**：2026-11-04（D029，使用者：「好 開始做新方法原形」）
- **類型**：可行性（feasibility）＋結論性比較（3 次獨立 launch、隨機順序）

## 白話結論

（完成後填）

## 問題

把 A（M12、M13）確認過的做法組合成**一個** kernel：

```
start barrier → reduce-scatter（pull，依 token 列分工）→ residual add + RMSNorm（只做自己負責的列）→ all-gather（push＋flag，不用 mid barrier）
```

和 vLLM 在 A100 上的實際做法（custom all-reduce → 另跑 `fused_add_rms_norm`）相比，在生成階段的大小快多少？

## 設計

### 新做法（`fused_ar_norm.cu`，PyTorch 擴充模組）

- TP=8，fp16，T 個 token、hidden H。第 r 張 GPU 負責第 r·⌈T/8⌉ 起的 ⌈T/8⌉ 列
- **reduce-scatter（pull）**：負責人以 16 byte 讀取，同時從 8 張 GPU 讀自己那幾列，fp32 相加（M12／M13 的 `rs_pull` 寫法）
- **residual add＋RMSNorm**：加上本地 residual（fp32），寫回 residual（只寫自己的列），算整列平方和（block 內 reduce），乘 rsqrt 與 gamma，寫到本地輸出
- **all-gather（push＋flag）**：把正規化後的列以 LL128 格式（每 128 byte 帶 8 byte epoch flag，每列補齊到整數個 4 行群組）寫到其他 7 張 GPU；收的一方看到 flag 就把資料搬進輸出（M13 的 `ag_push` 寫法，接收區依 epoch 單雙數輪流兩份）
- **start barrier**：沿用 vLLM 的每 block flag 寫法（pull 前必須確認對方的輸入已就緒）；epoch 由每個 block 的計數器遞增
- 不需要 mid barrier 的安全性：一張 GPU 要收齊 7 個 peer 的 all-gather 資料才算完成，而每個 peer 都是讀完所有人的輸入之後才送 all-gather，所以下一次寫輸入時不會有人還在讀
- 輸入 x、接收區、barrier flag 放在 PyTorch symmetric memory（每張 GPU 都能直接讀寫）
- **residual 只保留自己負責的列**（和 LMDeploy 相同）：同一個模型裡每一層用同樣的分工，下一層也只用到自己那幾列，所以語意正確；vLLM 對照組會更新整個 residual（多做的部分只是本地讀寫）

### 比較（同一個 torch 測試程式，計時方式同 M9）

| 代號 | 說明 |
|---|---|
| `U_vllm` | vLLM custom all-reduce＋`fused_add_rms_norm`（vLLM 0.33.0，A100 上的實際做法） |
| `F_trt_one`、`F_trt_two` | TRT-LLM 合併（FlashInfer 0.7.2.post1 同源程式另編 sm_80，同 M9） |
| `P_ours` | 新做法，block 數 16／32／64／108，取最佳（每個大小的最佳 block 數也記錄） |

- hidden 5120（主要）與 8192（次要），token 8／16／32／64／128（hidden 5120：80K–1.25M）
- 計時：每個設定 K=20 次錄成 CUDA graph、重播 R=20 次，每次時間 = 總時間／400，8 個 rank 取最大（同 M6、M9）；3 次獨立 launch、隨機順序
- 正確性：輸出（所有列、所有 rank）與 residual（自己的列）對照 fp32 參考值，max |誤差| / max |參考| ≤ 2^-5 才算正確（同 M9）
- 量測前確認沒有其他 GPU 工作（R13）

## 預期（量測前寫，量測後不改）

- **P1（正確）**：`P_ours` 在所有設定都正確；hidden 8192 也能啟動（TRT-LLM 在這裡失敗）
- **P2（速度，依 D029 前的粗估）**：hidden 5120 時，`P_ours` 比 `U_vllm` 快約 1.1／1.4／1.7／1.4／1.1 倍（80K／160K／320K／640K／1.25M）；粗估的新做法時間約 12.5／12.9／13.6／17.1／24.9 µs
- **P3**：`P_ours` 在 hidden 5120 的每個大小都比 `F_trt_one`、`F_trt_two` 快

## 判準（量測前寫，量測後不改；成功標準已和使用者討論過）

| 結果 | 條件 | 代表 |
|---|---|---|
| **go** | hidden 5120 的 160K、320K、640K 三個大小，`P_ours` 都比 `U_vllm` 快 ≥ 1.3 倍（中位數），且 [最小, 最大] 不重疊，且全部正確 | 進入 G2／G3（題目與 claims），再做 vLLM 端到端 |
| **no-go** | hidden 5120 的五個大小都 < 1.15 倍 | 回頭重新討論方向 |
| **中間** | 其他 | 停下來討論 |

## 已知簡化與限制

- token 數 < 8 時無法讓 8 張 GPU 都分到列（之後的改善 1：小資料改用 one-shot）
- start barrier 保留（之後的改善 3）
- 只量單一 all-reduce＋norm，不含前後的矩陣計算（之後的改善 4、5、6）

## 之後的改善路線（2026-11-04 和使用者討論）

| 層級 | 改善 | 要省的 |
|---|---|---|
| 同一個 kernel | 1. 依大小挑做法（小資料用 one-shot） | 小資料時 two-shot 不划算 |
| | 2. 依機器自動選 block 數 | M7：NCCL 選錯慢 1.35–2.33 倍 |
| | 3. 拿掉 start barrier | 約 2 µs；需先查清 M13 的 push reduce-scatter 為何慢 |
| 和前後計算合併 | 4. 前一步 GEMM 算完一塊就 push 出去（FLUX 式） | start barrier＋kernel 間空檔＋部分搬資料 |
| | 5. all-gather 收到一塊，下一步 GEMM 就開始算 | all-gather 時間 |
| | 6. 藏住 kernel 之間的空檔 | 每次約 3.5 µs（M11） |
| 系統層 | 7. 一批拆兩半交替（NanoFlow／TokenWeave） | 計算與通訊重疊；M5：計算會慢約兩成 |
| | 8. 壓縮傳輸資料 | 要驗證準確度 |
| | 9. 推廣到 TP4、其他沒有 NVLS 的 GPU | 需要其他硬體 |

## 指令

```bash
python3 exp/M14_fused_prototype/run.py
.venv/bin/python exp/M14_fused_prototype/analyze.py
```

## 結果

（量測後填）

## 結論

（量測後填）
