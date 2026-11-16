# 研究專案定義

> 本檔定義研究的不變量。研究問題、claims、non-goals 只能在使用者核准後修改（G2、G3），並在 `docs/decisions.md` 記錄理由。

## 研究方向（已定）

GPU 間通訊優化。先用 nccl-tests 量測本機的通訊特性，從中找出值得優化的現象，再決定題目。

## 研究問題

（待 G2 決定）

## Claims

（待 G3 決定。每個 claim 要對應一個實驗與成功標準）

## Non-goals

（待 G2 決定）

## 硬體與軟體（2026-11-02 盤點，正式數值以 M0 的環境快照為準）

| 項目 | 值 |
|---|---|
| GPU | 8× A100-SXM4-80GB，每對之間 NV12（12 條 NVLink3），經 NVSwitch 全互連 |
| CPU / NUMA | 2× Xeon Platinum 8358；GPU0–3 在 NUMA 0，GPU4–7 在 NUMA 1 |
| 網卡 | 8× ConnectX（mlx5_0–7），RoCE，各 100 Gb/s；每兩張 GPU 與兩張網卡在同一個 PCIe switch 下 |
| Driver / CUDA | 615.71.10 / CUDA 13.x |
| NCCL | 2.32.3（系統安裝） |
| nccl-tests | v2.21.1 @ afd59ab（MPI 版，只編 sm_80） |

## 硬體與軟體：Nano5（2026-11-16 起，正式數值以 M20 的環境快照為準）

| 項目 | 值 |
|---|---|
| 平台 | 國網中心 Nano5，Slurm 排程（帳號 MST114546） |
| GPU | 每台 8× H100 80GB HBM3（`dev`／`normal`／`4nodes`）或 8× H200（`normal2`） |
| 節點 | 2× Xeon 8480，約 1.9 TB 記憶體；跨節點 8× InfiniBand NDR 400 Gb/s |
| Driver | 550.127.08（測試工作 475347） |
| 其他 | 待 M20 量測 |

## 目標產出

（待 G2 決定投稿類型：系統會議、HPC 會議、ML systems 或 workshop）
