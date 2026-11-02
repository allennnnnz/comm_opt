# M0_baseline

- **狀態**：已核准，等待 GPU 空出（D004）
- **核准**：2026-11-02（D004）
- **類型**：探索性（每個點單次 run，用來找值得深入的區間；結論要等 M4 重複量測，R14）

## 問題

在這台 8× A100（NVSwitch）上，六種 collective 在不同訊息大小與 GPU 數下能達到多少頻寬與延遲？NCCL 在每個大小選了什麼演算法與協定？

## 預期（量測前寫，量測後不改）

依據：A100 NVLink3 每張 GPU 單向 12 × 25 = 300 GB/s；Hu et al.（2507.04786）對協定與演算法的分析。

- 大訊息（≥ 256 MiB）的 all_reduce、all_gather、reduce_scatter，busbw 落在 200–250 GB/s（理論上限 300 GB/s 的 65–85%）。
- 因為經 NVSwitch 全互連，2、4、8 卡的大訊息 busbw 差距在 15% 以內。
- alltoall 的 busbw 低於 all_gather。
- 小訊息（≤ 8 KiB）的 8 卡 all_reduce 延遲在 10–30 µs。
- tuner 選擇：小訊息用 LL，中等用 LL128，大訊息用 Simple；單機內以 Ring 為主。
- **推翻預期的情況**：大訊息 busbw < 150 GB/s；或 2、4、8 卡差距 > 30%；或 busbw 隨訊息大小出現非單調的下跌。這些都會列為「現象」進入 G1 候選。

## 設定

- **變因與水準**：collective ∈ {all_reduce, all_gather, reduce_scatter, alltoall, broadcast, sendrecv}；訊息大小 8 B–4 GiB（×2 遞增）；GPU 數 ∈ {2, 4, 8}
- **固定條件**：GPU 2 卡用 0,1、4 卡用 0–3、8 卡用 0–7（擺放位置的影響留給 M3）；資料型別 float；op = sum；in-place 與 out-of-place 都量（nccl-tests 預設）
- **執行方式**：一個 GPU 一個 MPI rank（`mpirun -np N`，`-g 1`）；`--bind-to none`，CPU 親和性由 NCCL 依拓樸決定
- **工具與版本**：nccl-tests v2.21.1 @ afd59ab；NCCL 2.32.3；Open MPI 4.1.9a1；每次 run 存 `env.txt`（GPU 時脈、溫度、拓樸、版本）
- **統計方式**：`-w 5` 暖身、`-n 20` 次；`-a 1`（跨 rank 取平均，與 nccl-tests 預設一致，方便和他人數字比較）；`-I 1` 逐次計時並報 p50/p99；`-U 1` 記錄 tuner 選擇；單次 run
- **理論上限**：NVLink 單向 300 GB/s／GPU

## 指令

```bash
exp/M0_baseline/run.sh            # 有其他 GPU 程式時會拒絕執行
DRY_RUN=1 exp/M0_baseline/run.sh  # 只印出 18 條指令
```

## 結果

（尚未執行）

## 結論

（尚未執行）

## 下一步（提案，需核准）

- **M1**：強制 `NCCL_ALGO` × `NCCL_PROTO`，算 tuner 預設選擇比最佳選擇慢多少
- **M2**：`NCCL_MIN/MAX_NCHANNELS` 與 CTA policy，量頻寬對 SM 數的曲線
- **M3**：2、4 卡的不同 GPU 組合（同 NUMA、跨 NUMA、不同 PCIe switch）
- **M4**：M0 關鍵點重複 5 次，報信賴區間；CUDA Graph 與一般模式比較
