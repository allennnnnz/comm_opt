# M20_env_snapshot

- **狀態**：已核准，待 S01 完成後執行
- **核准**：2026-11-16（D051：使用者「通用的就去測試」）；P10（SHARP）於 2026-11-16 經使用者同意加入（D052）
- **類型**：環境事實（拓樸、版本、設定：確定性，逐台記錄）＋結論性（時脈行為：依 R3 重複）
- **依賴**：S01 環境建置完成（`exp/lib/nano5_env.sh`、nccl-tests、NCCL 版本）
- **分工**：Opus 寫本卡與預期、做結論；Sonnet 依本卡寫 `run.sh`／`collect.sh`／`analyze.py`、送工作、整理表格（規格書 S 編號另開）

## 白話結論

（實驗完成後寫）

## 問題

Nano5 的 H100 與 H200 節點，在「卡怎麼互連、交換器能不能幫忙加總、卡和網卡怎麼接、時脈會不會變動、每台是不是都一樣」上的實際狀況是什麼？之後所有量測（M21–M24）的設定與解讀都以這裡為準。

## 為什麼要做

- A100 時期的硬體事實不能沿用（CLAUDE.md 第 1 條）；H100 的互連世代不同，最大的變數是 NVLS（NVSwitch 內的加總）有沒有開。
- A100 時期學到：時脈會隨暖機變化，影響量測（M18 步驟 a）。要在這台重新確認。
- 跨機器量測（M23）要知道每張卡最近的網卡是哪張、速度多少、GPUDirect RDMA 能不能用。
- 之後的工作會被排到不同的節點；要確認節點之間沒有差異，否則要固定節點或把節點當成變因（R11）。

## 預期（量測前寫，量測後不改）

寫於 2026-11-16，依據：H100／H200 SXM 的公開規格（背景知識，未在本機驗證）與工作 475347 的單卡拓樸輸出。

| # | 預期 | 什麼結果會推翻 |
|---|---|---|
| P1 | H100 節點：`nvidia-smi topo -m` 中每對 GPU 都是 `NV18`（經 NVSwitch 全互連） | 任何一對不是 NV18 |
| P2 | 每張 H100 的 `nvidia-smi nvlink -s` 有 18 條 link，全部 active，每條 26.562 GB/s（單向合計約 478 GB/s） | link 數 < 18、有 inactive、或速度不同 |
| P3 | H200 節點的 P1、P2 與 H100 相同 | 任何差異 |
| P4 | **NVLS 可用**：fabric manager 狀態為完成且成功（`nvidia-smi -q` 的 Fabric 欄位）；CUDA 屬性 `CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED`=1；NCCL 在 8 卡初始化時回報 NVLS 可用，指定 `NCCL_ALGO=NVLS` 時 all_reduce 能跑且 `#wrong`=0 | 任何一項不成立（信心中等：叢集可能沒開 NVLS） |
| P5 | 每張 GPU 至少有一張 400 Gb/s InfiniBand 網卡，和它在同一個 PCIe switch 下（`PXB` 或 `PIX`）；狀態為 Active 的 NDR（400 Gb/s）埠 ≥ 8；其餘網卡速度較低（儲存或管理用） | 有 GPU 沒有近的 400G 網卡，或 NDR Active 埠 < 8 |
| P6 | GPUDirect RDMA 可用：`nvidia_peermem` 已載入，或驅動支援 dma-buf | 兩者都沒有 |
| P7 | 使用者**不能**鎖定時脈（不嘗試修改，只查權限與設定）；持續通訊負載下，SM 時脈在 2 秒內升到最高（約 1980 MHz）並維持在 ±2% 內 60 秒；沒有功耗或溫度降頻 | 穩定時脈 < 1900 MHz、波動 > 2%、升頻要 > 5 秒，或出現降頻原因 |
| P8 | 抽樣的同類節點（H100 3 台、H200 2 台）完全一致：GPU 型號、VBIOS 系列、驅動、ECC 開、MIG 關、persistence mode 開、拓樸 | 任何一項不同 |
| P9 | GPU 與 400G 網卡都是 PCIe Gen5 x16 | 任何一個不是 |
| P10 | **InfiniBand SHARP 可用**（寫於 2026-11-16，加入時尚未量測；信心低，約五五波）：2 台 H100 上載入 HPC-X 的 SHARP 外掛後，NCCL 初始化回報 CollNet 可用，指定 `NCCL_ALGO=CollnetDirect`（或 `CollnetChain`）時 all_reduce 能跑且 `#wrong`=0 | 外掛載入失敗、NCCL 回報 CollNet 不可用或退回其他演算法、或結果錯誤 |

## 設定

- **節點抽樣**：
  - H100：3 台不同節點（`dev`，第 2、3 次用 `--exclude` 排除已量過的節點）
  - H200：2 台不同節點（`normal2`）
  - 每次都用 `--exclusive`
- **每個工作做三件事**（約 10 分鐘）：
  1. **快照**（確定性，存原始輸出）：
     - GPU 與連線：`nvidia-smi -q`、`nvidia-smi topo -m`、`nvidia-smi nvlink -s`、`nvidia-smi nvlink -c`
     - 網卡：`ibstat`、`ibv_devinfo -v`
     - 系統：`lscpu`、`numactl -H`、`lspci -tv`（不用 root 能讀到的部分）、`lsmod | grep -E "nvidia|peermem|mlx|gdr"`、`uname -a`、`cat /proc/cmdline`
     - 軟體：`module list`、NCCL 版本（S01 的 venv）
  2. **NVLS 檢查**：
     - 編譯一個小 CUDA 程式查 multicast 屬性（`sm_90`，在工作內編譯）
     - nccl-tests `all_reduce_perf -b 8 -e 64M -f 4 -g 1`，8 個行程，在 `NCCL_DEBUG=INFO`、`NCCL_DEBUG_SUBSYS=INIT,ENV,NVLS` 下各跑一次：
       - (a) 預設
       - (b) `NCCL_ALGO=NVLS`
       - (c) `NCCL_NVLS_ENABLE=0`
     - 只記錄是否成功、`#wrong`、log 中 NVLS 相關行。**速度數字不分析**（M21 的範圍）
  3. **時脈行為**（非確定性，R3）：
     - 每輪：先閒置 10 秒，再跑 60 秒持續通訊負載（nccl-tests all_reduce 64 MB、8 卡，用足夠大的 `-n` 撐滿 60 秒）
     - 全程用 `nvidia-smi --query-gpu=timestamp,index,clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_event_reasons.active --format=csv -lms 100` 背景取樣
     - 每台節點跑 5 輪
     - 另外查一次 `nvidia-smi -q -d CLOCK,PERFORMANCE,POWER`（只讀，不嘗試 `-lgc`／`-ac`）
- **固定條件**：S01 定下的 vLLM venv 中的 NCCL、nccl-tests 版本、`exp/lib/nano5_env.sh`
- **GPU**：每次整台 8 張
- **SHARP 檢查**（P10，2 台 H100，`normal`，`--exclusive`，約 10 分鐘，**不看速度**）：
  - 外掛位置（Opus 已讀檔確認存在）：`/work/HPC_software/LMOD/nvidia/packages/hpc_sdk-24.7/Linux_x86_64/24.7/comm_libs/12.5/hpcx/latest/nccl_rdma_sharp_plugin/lib/libnccl-net.so`；SHARP 函式庫在同層 `sharp/lib/`
  - 記錄：`ibstat`、SHARP 相關環境與服務（例如 `sharp_hello` 或 HPC-X 附的檢查工具，若存在）、外掛與 NCCL 版本相容性訊息
  - nccl-tests all_reduce `-b 8 -e 64M -f 4`，16 個行程，在 `NCCL_DEBUG=INFO`、`NCCL_DEBUG_SUBSYS=INIT,NET,COLL` 下：(a) 預設；(b) `NCCL_COLLNET_ENABLE=1`＋外掛；(c) (b) 加 `NCCL_ALGO=CollnetDirect`；(d) (b) 加 `NCCL_ALGO=CollnetChain`
  - 只記錄能否啟用、`#wrong`、log 中 CollNet／SHARP 相關原文
- **工具與版本**：以 S01 報告 R01 為準，並在每個工作的 `job_env.txt` 記錄
- **統計方式**：
  - 快照：逐台列表，標出節點之間的差異
  - 時脈：每輪算三個指標：
    - 升頻時間：從負載開始，到時脈進入並維持在穩定中位數 98% 以上的時間
    - 穩定段（第 5–60 秒）的中位數、p1、p99
    - 有沒有降頻原因
  - 跨輪（5 輪 × 每台節點）報中位數與最小／最大值；跨 GPU 取最慢的那張（max 升頻時間、min 時脈）
- **理論上限**（R8，之後的量測用）：每 GPU NVLink 單向頻寬＝ link 數 × 每條速度（P2 實測值）；InfiniBand 每埠 400 Gb/s＝50 GB/s

## 指令

（Sonnet 依本卡實作後補上；腳本放本資料夾，輸出到 `results/M20_env_snapshot/<時間>_job<id>/`）

## 結果

- 原始數據：`results/M20_env_snapshot/`
- 分析：`exp/M20_env_snapshot/analyze.py` → 節點對照表、時脈表、`figures/M20_*.png`（時脈對時間）

## 結論

（對照 P1–P9 逐條判定）

## 下一步

- 依 P4（NVLS 有沒有開）決定 M21 是否把「NVLS 開／關」列為主要變因
- 依 P7 決定之後量測的暖機長度與丟棄策略
- 依 P8 決定之後是否要固定節點
