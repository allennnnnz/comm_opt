# M15_bf16

- **狀態**：進行中
- **核准**：2026-11-04（D031，使用者：「補一下 bf16」）
- **類型**：結論性（3 次獨立 launch、隨機順序）

## 白話結論

（完成後填）

## 問題

M14 的原型只支援 fp16，但主流模型（Llama-3、Qwen2.5 等）預設用 bf16。補上 bf16 後，正確性與速度優勢是否不變？同時重跑 fp16，確認 M14 的結果能重現。

## 改動（相對 M14，量測前）

- `fused_ar_norm.cu`：kernel 改成依資料格式產生兩個版本（fp16、bf16）。只有「轉成 fp32 相加」與「轉回 16 位元」兩個函式不同（bf16 用 `__bfloat1622float2`、`__floats2bfloat162_rn`）；搬資料、barrier、LL128 push 完全相同（每個元素都是 2 byte）。host 端依 residual 的資料格式選版本
- `bench_proto.py`：加 `--dtype`（float16／bfloat16），x、residual、gamma、輸出都用該格式；vLLM custom AR、`fused_add_rms_norm`、FlashInfer TRT-LLM 合併本來就支援 bf16
- 其他設定與 M14 相同：TP=8、hidden 5120／8192、token 8–128、CUDA graph K=20 × R=20、8 個 rank 取最大、正確性對照 fp32 參考值（≤ 2^-5）

## 預期（量測前寫，量測後不改）

- **P1（正確）**：bf16 所有設定都正確；最大誤差比 fp16 大（bf16 只有 8 位元尾數），但 ≤ 2^-5
- **P2（速度不變）**：bf16 的 `P_ours` 對 `U_vllm` 倍數，和同一次 run 的 fp16 倍數相差 ≤ 0.1（每個大小）——資料量相同，只有轉換指令不同
- **P3（重現 M14）**：fp16 在 hidden 5120 的 160K／320K／640K 仍 ≥ 1.3 倍

## 判準（量測前寫，量測後不改）

| 結果 | 條件 | 代表 |
|---|---|---|
| **bf16 可用** | bf16 全部正確，且 hidden 5120 的 160K／320K／640K 都比 `U_vllm`（bf16）快 ≥ 1.3 倍、區間不重疊（同 M14 的 go 標準） | 原型支援 bf16 |
| **bf16 有問題** | 任何設定不正確，或上述三個大小有任何一個 < 1.15 倍 | 停下來查原因 |
| **中間** | 其他 | 停下來討論 |

## 指令

```bash
python3 exp/M15_bf16/run.py
.venv/bin/python exp/M15_bf16/analyze.py
```

## 結果

（量測後填）

## 結論

（量測後填）
