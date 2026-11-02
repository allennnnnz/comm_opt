# M0_baseline

- **狀態**：完成（2026-11-02 13:48–13:51，run `20261102_134841`）
- **核准**：2026-11-02（D004）
- **類型**：探索性（每個點單次 run，用來找值得深入的區間；結論要等 M4 重複量測，R14）

## 白話結論

1. **這台機器的 GPU 互傳基本健康。** 傳大量資料時能用到 NVLink 理論頻寬的七成多，跟一般 A100 機器差不多，沒有哪裡壞掉或特別慢。
2. **真正有趣的在「中等大小」的資料（幾 MB 到幾十 MB），不在大資料。** 這正是實際訓練和推論裡很常見的傳輸大小。
3. **最強的發現：8 張卡一對一傳資料時，NCCL 自己把傳輸通道砍少，速度只剩 2 張卡時的約三分之一。** 每張卡其實只跟一張卡傳（像 pipeline 平行那樣），NCCL 卻像是以為大家在跟所有人講話，保守地分配資源。
4. **反過來也有怪事：只用 2 張卡做集體通訊時，中等大小反而比 8 張卡慢三到五成**，原因還不知道。另外 broadcast 在中等大小時快時慢，很不穩定。
5. **這些都只量了一次，還不能當結論。** 下一步是重複量測確認不是偶然，並測試第 3 點是不是改個設定就能解決（如果是，研究價值就低）。

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

- 原始數據：`results/M0_baseline/20261102_134841/`（18 個 JSON、18 個 log、每個 rank 一份 NCCL debug log、`env.txt`）
- 分析：`.venv/bin/python exp/M0_baseline/analyze.py results/M0_baseline/20261102_134841` → `figures/M0_baseline/20261102_134841/`
  - `summary.csv`（全部點）、`peaks.csv`（每組峰值與最小訊息延遲）、`anomalies.csv`（自動標記的異常）
  - `fig_busbw.png`、`fig_latency.png`、`fig_tuner.png`
- 執行前狀態：8 張卡閒置，SM clock 210 MHz（上限 1410）、溫度 30–32°C（`env.txt`）
- 18 組全部成功，`#wrong` 全為 0

**大訊息峰值 busbw（out-of-place，GB/s，括號內為 NVLink 300 GB/s 的百分比）**

| collective | 2 GPU | 4 GPU | 8 GPU |
|---|---|---|---|
| all_reduce | 212.2 (71%) | 229.0 (76%) | 231.5 (77%) |
| all_gather | 166.5 (56%) | 207.1 (69%) | 220.1 (73%) |
| reduce_scatter | 182.2 (61%) | 220.1 (73%) | 227.2 (76%) |
| alltoall | 192.7 (64%) | 214.8 (72%) | 217.8 (73%)，峰值在 1 GiB |
| broadcast | 263.3 (88%) | 230.3 (77%) | 230.2 (77%) |
| sendrecv | 220.7 (74%) | 220.2 (73%) | 220.6 (74%) |

最小訊息的 p50 延遲為 14–25 µs（8 卡 all_reduce 8 B：24.6 µs）。來源：`peaks.csv`。

## 結論

### 對照預期

| 預期 | 結果 | 判定 |
|---|---|---|
| 大訊息 AR/AG/RS busbw 200–250 GB/s | 4、8 卡符合；2 卡 AG 166、RS 182 低於範圍 | 部分不符 |
| 2/4/8 卡大訊息差距 ≤ 15% | AR 8%、sendrecv <1%；AG 2 vs 8 卡差 24%、RS 20% | 部分不符 |
| alltoall busbw 低於 all_gather | 8 卡兩者相當（218 vs 220）；2 卡 alltoall 反而較高（193 vs 167） | 不符 |
| 8 卡 all_reduce 小訊息延遲 10–30 µs | 24.6 µs | 符合 |
| tuner：小 LL、中 LL128、大 Simple，Ring 為主 | 大致符合；但 LL128 只在 4/8 卡的 2–8 MiB 出現，2 卡完全不用 LL128；從未選 Tree | 大致符合 |
| 推翻條件「2/4/8 卡差距 > 30%」 | 大訊息沒有；**中等訊息有**（見 F1、F3） | 觸發 |
| 推翻條件「busbw 非單調下跌」 | 沒有 >10% 的下跌；sendrecv 8 卡出現長平台，alltoall 8 卡在 1 GiB 之後下降約 6% | 部分觸發 |

### 觀察到的現象（G1 候選；全部是單次 run，需重複量測確認，R14）

- **F1 sendrecv 在大 communicator 的中等訊息被限縮通道。** 8 卡在 4–64 MiB 只有 4 個 channel，busbw 停在 41–58 GB/s；2 卡同大小用 16 個 channel，達 63–162 GB/s，64 MiB 時相差 2.8 倍。4 卡在 128 MiB 也是 8 vs 16 channel（106 vs 168 GB/s）。到 256 MiB 以上 channel 數才升到 16–32，三者會合到約 220 GB/s。每個 rank 只有一個傳送對象，但 channel 數卻隨 communicator 大小下降。（`anomalies.csv` 的 `gpu_count_gap`、`fig_busbw.png` 右下、`fig_tuner.png`）
- **F2 alltoall 的 channel 數也隨 GPU 數下降。** 8 卡在 2 MiB–512 MiB 都只用 4 個 channel（512 MiB：202.4 GB/s）；1 GiB 用 8 個 channel 達峰值 217.8，2 GiB 用 16 個 217.0，4 GiB 用 32 個反而降到 205.8（−6%）。2 卡 in-place 比 out-of-place 快 10–25%（64 MiB–4 GiB）。
- **F3 2 卡 collective 在 1–8 MiB 的 busbw 比 8 卡低 30–54%**，而且 channel 數相同（22–24）。到 4 GiB，2 卡 all_gather 仍只有 166 GB/s（8 卡 220）。2 卡 all_gather 的 in-place 比 out-of-place 快 11–16%，可能是 out-of-place 多了一次本地複製（占輸出的一半）。
- **F4 雙向流量的上限。** 所有雙向的 collective 最高停在 220–231 GB/s（73–77%）；只有單向的 2 卡 broadcast 達 263 GB/s（88%）。
- **F5 broadcast 在 1–32 MiB 極不穩定。** 8 卡 cv 34–69%、p99/p50 達 2.7–4.8（64 MiB 降到 cv 19%）；4 卡 1–32 MiB cv 16–31%。
- **F6 LL128 只在很窄的區間被選用**（4/8 卡、2–8 MiB）。是不是最佳選擇要靠 M1 判斷。
- **F7 小訊息延遲的尾端。** p50 為 14–25 µs，但 p99 偶有 60–500 µs 的尖峰。每點只有 20 次，p99 幾乎等於最大值，不能下結論。

### 限制

- 每個點只跑一次、20 次 iteration；p99 等於最大值附近
- 執行順序固定（2→4→8 卡，collective 依序），沒有隨機化（R11）
- 執行前 GPU 處於 210 MHz 閒置時脈，每個程序只暖身 5 次
- 只用了一組 GPU（0,1 / 0–3 / 0–7）
- 分析腳本在本次 run 之後才寫完（`gpu_count_gap` 偵測是看到 sendrecv 平台後加的），所以這些現象是「事後觀察」，要在重複量測時預先寫好判準再驗證

## 下一步（提案，需核准）

- **M4（建議先做）**：F1–F5 涉及的點各重複 5 次、隨機順序、換一組 GPU，報信賴區間；判準在跑之前寫進 M4 實驗卡。G1 需要這一步
- **M2-P2P**：針對 F1/F2，量 P2P channel 數（`NCCL_MIN_P2P_NCHANNELS`、`NCCL_MAX_P2P_NCHANNELS`、`NCCL_NCHANNELS_PER_PEER`）對 sendrecv、alltoall 的影響，確認 F1 是不是只是預設值的問題

原本的其他提案：

- **M1**：強制 `NCCL_ALGO` × `NCCL_PROTO`，算 tuner 預設選擇比最佳選擇慢多少
- **M2**：`NCCL_MIN/MAX_NCHANNELS` 與 CTA policy，量頻寬對 SM 數的曲線
- **M3**：2、4 卡的不同 GPU 組合（同 NUMA、跨 NUMA、不同 PCIe switch）
- **M4**：M0 關鍵點重複 5 次，報信賴區間；CUDA Graph 與一般模式比較
