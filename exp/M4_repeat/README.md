# M4_repeat

- **狀態**：已核准（D006），尚未執行
- **核准**：2026-11-02（D006）
- **類型**：結論性（每點 5 次獨立 launch、隨機順序、兩組 GPU）

## 白話結論

（實驗完成後填）

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

（尚未執行）

## 結論

（尚未執行）

## 下一步
