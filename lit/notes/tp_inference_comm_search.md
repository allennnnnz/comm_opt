# 張量平行推論的通訊：新穎性初查（2026-11-03）

目的：W2——依 M5 的結果評估候選方向前，先看各方向的擁擠程度。只讀了摘要，尚未精讀。

## 搜尋紀錄（arXiv MCP，cs.DC/cs.LG/cs.PF/cs.AR，2023-06 起）

| 查詢 | 結果數 | 相關 |
|---|---|---|
| all-reduce × tensor parallel × (decode/decoding/latency) × inference | 8 | SiFAR 2607.08973、Singhania et al. 2511.09557（NVRAR）、TokenWeave 2505.11329 |
| (quantized/compression/low-bit) × (all-reduce/communication) × tensor parallel × inference | 6 | 2411.09510、2411.07942、Echo 2610.14277 |
| overlap × communication × computation × tensor parallel × (inference/serving/prefill) | 13 | ISO 2409.11155、TokenWeave、FLUX 2406.06858、T3 2401.16677、Kraken 2408.07802、Ladder-residual 2501.06589、CommFuse 2604.24013、Helix 2507.07120、Albireo 2606.01927 |

## 各方向的擁擠程度

- **藏住通訊（讓通訊與計算重疊）**：擁擠，至少 9 篇，涵蓋 kernel 融合、序列切分、架構修改。
- **壓縮張量平行的通訊資料**：3 篇，需處理精度損失。
- **降低生成階段 all-reduce 的延遲**：
  - **SiFAR**（Taneja et al., 2026-07）：針對低延遲推論（最小批次），把 oneshot/twoshot 演算法前後的 barrier 開銷去掉——雙緩衝去掉下方 barrier、交換器內歸約（in-switch reduction）讓 oneshot 能擴展、speculative reduction 減少上方 barrier。在 8 張 H200 上 all-reduce 延遲最多降 52%，端到端吞吐提升 13–19%。**與本研究最接近**；其中交換器內歸約需要新一代 NVSwitch，A100 沒有。
  - **NVRAR**（Singhania et al., 2025-11）：多機推論，用 NVSHMEM 做 recursive doubling，128 KB–2 MB 比 NCCL 快 1.9–3.6 倍。針對多機網路，不是單機 NVLink。
  - **Albireo**（Zhao et al., 2026-06）：處理張量平行中不隨 GPU 數縮小的開銷（排程、I/O、取樣），對應 M5 的 O3（小模型受 CPU 主導）。

## 和 M5 結果的關係

- M5：Qwen2.5-32B、TP=8、A100。生成階段批次 8–128 時，每次 all-reduce 為 82 KB–1.3 MB，傳輸時間是保守估計可達水準（實測峰值 231 GB/s、每次固定開銷 6 µs）的 2.2–2.5 倍；這個區間的通訊占每步 17–27%。
- SiFAR 主打最小批次；M5 的空間最大處在批次 8–128（吞吐導向的服務）。兩者的區間不同，但 SiFAR 的雙緩衝與 speculative reduction 與硬體無關，可能也適用於這個區間——**精讀 SiFAR 是選這個方向前的必要步驟**。
- 尚未查：NCCL 2.27 以後的 symmetric kernels、MSCCL++、TensorRT-LLM 與 FlashInfer 的 all-reduce 實作、PyTorch symmetric memory（這些是此方向必須比較的 baseline）。
