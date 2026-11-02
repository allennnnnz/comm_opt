# M4_repeat

- **狀態**：完成（2026-11-02 14:02–14:25，run `20261102_140252`）
- **核准**：2026-11-02（D006）
- **類型**：結論性（每點 5 次獨立 launch、隨機順序、兩組 GPU）

## 白話結論

1. **最強的發現站得住：8 卡一對一傳資料時，中等訊息的速度只有 2 卡的 36–41%。** 兩組 GPU、5 次重複都一樣，重複之間差不到 3%，是 NCCL 固定的行為，不是雜訊。
2. **「2 卡反而比 8 卡慢」只在很窄的範圍成立。** 2 MiB 時很明顯（2 卡只有一半左右），到 4 MiB 就縮小，依事先定的判準不算成立。
3. **其他三個都確認了：** alltoall 在 4 GiB 比 1 GiB 慢 5%；單向傳輸比雙向快 14%；broadcast 在中等大小很不穩（變異 30–50%，同條件的 all_reduce 只有 1–12%）。
4. **換一組 GPU 結果幾乎一樣**（沒有任何點差超過 10%），這台機器的 8 張卡是對稱的，之後不必擔心挑到哪幾張卡。

## 問題

M0 單次量到的現象 F1–F5 是真的，還是偶然？換一組 GPU、打亂順序之後還在不在？

## 預期（量測前寫，量測後不改）

- F1（sendrecv 8 卡中等訊息只有 2 卡的一半以下）：**會成立**。NCCL 原始碼已顯示 8 卡時 channel 下限是 4（`exp/M2_p2p_channels/predict_channels.py` 完全重現 M0），所以這是確定性的行為，不是雜訊。
- F3（2 卡 collective 在 1–8 MiB 明顯低於 8 卡）：**大概會成立**，幅度可能比 M0 小。
- F4（單向 broadcast 比雙向 collective 高 ≥10%）：**會成立**。
- F2a（alltoall 8 卡 4 GiB 比 1 GiB 慢）：不確定，M0 只差 6%。
- F5（broadcast 8 卡在 1–16 MiB 很不穩定）：不確定；M0 每點只有 20 次，可能是雜訊。
- 兩組 GPU 的結果：預期差距在 10% 以內（NVSwitch 全互連）。

## 判準（量測前寫，量測後不改）

統計方式：每個（collective, GPU 數, GPU 組, 大小）有 5 次獨立 launch。每次取 nccl-tests 回報的 out-of-place busbw（40 次 iteration 的平均，`-a 1` 跨 rank 平均）。5 次之間報**中位數**與 **[最小, 最大]**；對中位數而言，5 個樣本的 [最小, 最大] 是約 94% 的無母數信賴區間（R3、R4）。「不重疊」= 兩組的 [最小, 最大] 沒有交集（R5）。每個判準要在 GPU 組 A、B **都**成立才算成立。

| 現象 | 成立條件 |
|---|---|
| F1 | sendrecv 在 16 MiB 與 64 MiB：8 卡中位數 ≤ 0.5 × 2 卡中位數，且不重疊 |
| F2a | alltoall 8 卡：4 GiB 的中位數比 1 GiB 低 ≥ 3%，且不重疊 |
| F2b | alltoall 2 卡 1 GiB：in-place 時間 ≤ 0.95 × out-of-place 時間，且不重疊 |
| F3 | 在 2 MiB 與 4 MiB，三種 collective（all_gather、reduce_scatter、all_reduce）中至少兩種：2 卡中位數 ≤ 0.75 × 8 卡中位數，且不重疊 |
| F4 | 2 卡 broadcast 在 2 GiB 的中位數 ≥ 1.10 × 8 卡 all_gather、reduce_scatter、all_reduce 在 4 GiB 的最高中位數，且不重疊 |
| F5 | broadcast 8 卡在 1、2、4、8、16 MiB 中至少 3 個大小：單次 run 內 cv 的中位數 ≥ 20%，且 ≥ 2 × 同大小 all_reduce 8 卡的 cv |
| GPU 組偏差（R11） | 任一點 A、B 中位數相差 > 10% 且不重疊 → 標記為「GPU 組有影響」，交給 M3 |

## 設定

- **量測點**：sendrecv 2/4/8 卡 4 MiB–256 MiB；alltoall 2/8 卡 64 MiB–4 GiB；all_gather、reduce_scatter、all_reduce 2/8 卡 1 MiB–4 GiB；broadcast 2/8 卡 1 MiB–2 GiB（皆 ×2 遞增）
- **GPU 組**：A = M0 用的組（`0,1`／`0–3`／`0–7`）；B = 另一組實體卡或 rank 順序（`6,7`／`4–7`／`0,2,4,6,1,3,5,7`），定義在 `exp/lib/ncclrun.py`
- **重複**：每個組合 5 次獨立 launch，共 130 次；執行順序用 seed 20261102 打亂（R11、D6），實際順序存在 `schedule.tsv`
- **iteration**：`-w 10 -n 40`；`-I 1` 存每次 iteration 的時間；`-U 1` 記錄 tuner 選擇
- **每個 job 之前**：確認沒有其他 GPU 程式（有就停止，R13），並記錄每張卡的 SM clock 與溫度
- **工具與版本**：nccl-tests v2.21.1 @ afd59ab；NCCL 2.32.3；Open MPI 4.1.9a1
- **理論上限**：NVLink 單向 300 GB/s／GPU

## 指令

```bash
python3 exp/M4_repeat/run.py            # 有其他 GPU 程式時拒絕執行
DRY_RUN=1 python3 exp/M4_repeat/run.py  # 只印出指令
```

## 結果

- 原始數據：`results/M4_repeat/20261102_140252/`（130 次 launch，`schedule.tsv` 記錄實際順序、每個 job 前的 GPU clock 與溫度）
- 分析：`.venv/bin/python exp/M4_repeat/analyze.py results/M4_repeat/20261102_140252` → `figures/M4_repeat/20261102_140252/`（`verdicts.csv`、`busbw_summary.csv`、`runs.csv`、`fig_confirm.png`）
- 130 次全部 rc=0，`#wrong` 全為 0，每個點都有 5 次

| 判準 | 組 A | 組 B | 關鍵數字（組 A；中位數 [最小, 最大]，GB/s） |
|---|---|---|---|
| F1 | 成立 | 成立 | 16 MiB：8 卡 45.6 [44.5, 46.1] vs 2 卡 112.2 [111.9, 120.4]（0.41）；64 MiB：58.1 vs 162.0（0.36） |
| F2a | 成立 | 成立 | alltoall 8 卡 4 GiB 206.1 vs 1 GiB 217.6（0.947） |
| F2b | 成立 | 成立 | alltoall 2 卡 1 GiB in-place 2487 µs vs out-of-place 2865 µs（0.868） |
| F3 | **不成立** | **不成立** | 2 MiB 三種都 ≤ 0.75（0.48、0.50、0.61）；4 MiB 只有 all_gather（0.63），reduce_scatter 0.76、all_reduce 0.85 |
| F4 | 成立 | 成立 | broadcast 2 卡 2 GiB 263.3 vs all_reduce 8 卡 4 GiB 231.5（1.137） |
| F5 | 成立 | 成立 | 5/5 個大小 broadcast 8 卡 cv 31–52%，all_reduce 8 卡 1–12% |
| GPU 組偏差 | 無 | — | 沒有任何點 A、B 差 > 10% 且不重疊 |

## 結論

| 預期 | 結果 |
|---|---|
| F1 會成立 | 符合。而且 [最小, 最大] 區間很窄，和「確定性行為」的預期一致 |
| F3 大概會成立，幅度可能比 M0 小 | 依判準不成立；幅度確實比 M0 小，只在 2 MiB 附近明顯 |
| F4 會成立 | 符合（+13.7%） |
| F2a、F5 不確定 | 兩者都成立。F5 在 40 次 iteration 下仍然存在，不是 M0 樣本太少造成的 |
| 兩組 GPU 差距 ≤ 10% | 符合 |

- F3 的機制仍未知；目前只確定它集中在 1–2 MiB，且三種 collective 都有
- F5 的機制未知；broadcast 在 NCCL 是 ring 鏈狀傳遞，可能和鏈上每一段的 pipeline 有關，但沒有證據
- 限制：只有一台機器；GPU 時脈沒有鎖定（無 sudo），只記錄每個 job 之前的時脈

## 下一步

- G1：整理已確認的現象給使用者選擇（F1 最強，F2、F4、F5 次之）
