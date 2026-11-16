# 共通知識（整個研究都會用到）

> Opus 維護。寫規格書時，引用這裡相關的小節給小助手讀，不要每次重寫背景。
> 本檔**不是** append-only：量測驗證後要更新標記。重大更正同時記到 `docs/decisions.md`。

**可信度標記**（每一條都要有一個）：

| 標記 | 意思 | 能不能當論文或決策依據 |
|---|---|---|
| `[實測:<路徑>]` | 本專案量到，可追到 `results/` 或實驗卡 | 可以（注意適用範圍） |
| `[文件:<出處>]` | 官方文件、說明書或原始碼，附網址或檔案行號 | 可以，但硬體行為仍以實測為準 |
| `[背景]` | Opus 的背景知識，還沒在本機或原文核對 | **不行**；要用之前先驗證，驗證後改標記 |
| `[A100]` | A100 時期量到的，硬體不同，不能直接套用到 Nano5 | 只能當對照或假設來源 |

---

## 一、研究流程速查

- **流程與關卡**（G1–G5）：`docs/research_rules.md` 第一節。
- **量測規則**（R1–R14）：第二節。
- **比較規則**（E1–E8）：第三節。
- **數據管理**（D1–D10）：第四節。
- **分工**：Opus 規劃與決策、Sonnet 照規格書執行（CLAUDE.md 第 8 條、守則第八節）。
- **編號**：
  - `M*` 環境特性量測（Nano5 從 M20 起）；`E*` 驗證假說的實驗
  - `S*`／`R*` 規格書與報告（同號對應）
  - `L*` 文獻調查（見 `docs/plan_nano5.md`）
- **每個實驗**：實驗卡（`exp/<ID>/README.md`，先寫預期、經使用者核准）→ 小助手寫腳本並送出 → 原始數據寫到 `results/<ID>/<時間>_job<id>/` → `analyze.py` 產生表與圖 → Opus 寫結論。

## 二、叢集操作速查（Nano5）

- **完整流程**：skill `.claude/skills/slurm-submit/`（`SKILL.md`、`template.sbatch`、`submit.sh`）。
- **硬規則**：
  - 登入節點不跑運算，編譯也送 Slurm
  - 不輪詢 `squeue`（違反可能被鎖帳號）
  - 量測一律 `--exclusive`
  - 輸出放 `/home`
  - 每人同時執行最多 2 個工作

  出處：[文件: https://man.twcc.ai/@AI-Pilot/manual 注意事項頁]、D048。
- **排隊區**：
  - `dev`：H100，2 h，每計畫最多 8 GPU
  - `normal`：H100，48 h，16 GPU
  - `4nodes`：H100，12 h，32 GPU
  - `normal2`：H200，48 h，16 GPU，所以 H200 最多 2 台

  出處：[文件: 說明書 Queue 列表，2026-07-01 版]。
- **環境啟用**：`source exp/lib/nano5_env.sh`（S01 建立後生效；版本見 `docs/reports/R01_env_setup.md`）。
- **每個工作自動記錄**：`job_env.txt`（工作編號、節點、git commit、`nvidia-smi`、拓樸），由 `template.sbatch` 產生。
- **節點規格**：每台 8 GPU、112 CPU、約 1.9 TB 記憶體 [實測: `sinfo`，2026-11-16]；8× InfiniBand NDR 400 Gb/s 用於計算，另有 2× 200 Gb/s 用於檔案系統 [文件: 說明書硬體規格頁]。
- **驅動**：計算節點 550.127.08 [實測: 工作 475347]。
- **單卡拓樸**：一個節點的 `nvidia-smi topo -m` 列出 12 張 mlx5 網卡 [實測: 工作 475347，`/home/allen0117/slurm_test/flow-test-475347.out`]。每張網卡的用途待 M20 確認。

## 三、硬體概念

### 3.1 卡與卡之間（單機）

| 項目 | A100（舊） | H100／H200（Nano5） |
|---|---|---|
| NVLink 世代與條數 | NVLink3，每對 GPU 為 NV12 [A100: `docs/project.md`] | NVLink4，預期每對 NV18 [背景；待 M20 P1] |
| 每條速度 | — | 預期每條單向 26.562 GB/s，18 條合計約 478 GB/s [背景；待 M20 P2] |
| NVSwitch 交換器內加總（NVLS） | 沒有 [A100: `docs/findings_overview.md` 第三節] | 預期有 [背景；待 M20 P4] |

- **NVLS／multimem** [背景]：
  - 原理：GPU 對一個「多播位址」做讀取時，NVSwitch 把各卡的值加總後回傳（`multimem.ld_reduce`）；寫入時，交換器把資料複製到所有卡（`multimem.st`）。
  - 需要的條件：fabric manager、CUDA multicast 物件（`cuMulticastCreate` 一類的 API）、sm_90 以上。
  - 對 all-reduce 的意義：每張卡要搬的資料量大約減半，而且可以不用「分工加總 → 收齊」兩段式。
- **A100 缺、Hopper 可能有的功能**：A100 時期整理的清單見 `docs/findings_overview.md` 第三節，包括：
  - 交換器內加總
  - thread block cluster（多個運算單元合作處理同一列）
  - PDL（讓下一個 kernel 提前開始）
  - 16 byte 向量原子加法
  - bf16 原子加法

  這些在 H100 上是否可用、效果如何，**全部** [背景]，要逐項驗證才能引用。
- **`nvidia-smi topo -m` 圖例** [實測: 工作 475347 輸出]：
  - `NV#`：經 # 條 NVLink
  - `PIX`：最多經一個 PCIe 橋
  - `PXB`：經多個 PCIe 橋，但不經 host bridge
  - `PHB`：經 PCIe host bridge
  - `NODE`：同一個 NUMA 內，跨 host bridge
  - `SYS`：跨 NUMA（經 UPI）

### 3.2 機器之間

- **InfiniBand NDR**：每埠 400 Gb/s，等於 50 GB/s；每台 8 埠，合計 400 GB/s [文件: 說明書硬體規格頁]。單台內一張 GPU 的 NVLink 頻寬（約 450–480 GB/s 單向 [背景]）大約等於整台機器對外的總頻寬，所以跨機器通常是瓶頸 [背景；待 M23]。
- **GPUDirect RDMA**：網卡直接讀寫 GPU 記憶體，不經 CPU 記憶體。需要 `nvidia_peermem` 或 dma-buf [背景；待 M20 P6]。
- **網卡要選離 GPU 最近的**（同一個 PCIe switch，`PIX`／`PXB`）。NCCL 預設會自動選 [背景]。
- **SHARP**：InfiniBand 交換器內加總，需要交換器與軟體支援 [背景；可用性待 M23]。

## 四、NCCL 與 nccl-tests

> 以下參數名稱與公式要和 S01 裝的版本核對（`third_party/nccl-tests/doc/PERFORMANCE.md`、`all_reduce_perf --help`、NCCL 官方環境變數文件）；核對後改標記。

- **演算法**（`NCCL_ALGO`）：`Ring`、`Tree`、`NVLS`、`NVLSTree`、`CollNetDirect`、`CollNetChain`、`PAT` [背景]。
- **協定**（`NCCL_PROTO`）[背景]：
  - `LL`：每 8 byte 含 4 byte 資料加 4 byte 旗標，延遲最低、頻寬最差
  - `LL128`：每 128 byte 含 120 byte 資料加 8 byte 旗標，依賴「128 byte 寫入一起到達」
  - `Simple`：大資料用，頻寬最好
- **A100 時期的 LL128 觀察**：「沒看到違反」只是觀察，不是硬體保證 [A100: `exp/M8_ll128_atomic_a100/README.md`]。
- **常用環境變數** [背景]：
  - 演算法與協定：`NCCL_ALGO`、`NCCL_PROTO`、`NCCL_NVLS_ENABLE`
  - 通道：`NCCL_MIN_NCHANNELS`／`NCCL_MAX_NCHANNELS`
  - 緩衝與執行緒：`NCCL_BUFFSIZE`、`NCCL_NTHREADS`
  - 網路：`NCCL_P2P_LEVEL`、`NCCL_NET_GDR_LEVEL`、`NCCL_IB_HCA`
  - 記憶體：`NCCL_CUMEM_ENABLE`
  - 除錯：`NCCL_DEBUG=INFO`、`NCCL_DEBUG_SUBSYS`、`NCCL_DEBUG_FILE`
- **頻寬換算**（nccl-tests）[背景；待核對 PERFORMANCE.md]：
  - `algbw` 等於資料大小 S 除以時間 t
  - `busbw` 等於 `algbw` 乘上一個係數，n 是 GPU 數：
    - all_reduce：2(n−1)/n
    - all_gather、reduce_scatter、alltoall：(n−1)/n
    - broadcast、reduce、sendrecv：1
  - `busbw` 可以直接和連線的硬體頻寬比較。
- **nccl-tests 參數**：A100 時期用過 `-b -e -f -g -n -w -a 1 -U 1 -I 1 -J` [A100: `exp/M0_baseline/README.md`、`run.sh`]：
  - `-a` 跨 rank 怎麼彙總，對應 R7
  - `-I 1` 逐次計時
  - `-U 1` 記錄 tuner 的選擇
  - `-J` 輸出 JSON

  新版參數以 `--help` 為準。
- **NCCL 版本**：nccl-tests 和被比較的系統（vLLM、PyTorch）要用**同一版**（S01 規定）。

## 五、推論中的通訊（M24 與選題用）

- **TP（把一層模型切給多張卡）**：每一層有 2 次 all-reduce，分別在 attention 輸出投影之後和 MLP down 投影之後 [背景；A100 時期在 vLLM 0.33.0 + Qwen2.5-32B 觀察到相同結構，見 M5]。
  - 每次的資料量：tokens × hidden × 每個元素的 byte 數。例如 hidden 5120、fp16，每個 token 10 KB。
  - 生成階段（decode）資料小，延遲為主；讀入階段（prefill）資料大，頻寬為主。
- **EP（MoE 專家平行）**：每層兩次 all-to-all（dispatch、combine）[背景]。
- **PP（流水線平行）**：層與層之間 send/recv [背景]。
- **vLLM 的 all-reduce 路徑**：
  - A100 + vLLM 0.33.0：custom all-reduce（one-shot／two-shot），也有 pynccl [A100: M5–M11]。
  - H100 上新版走哪條（custom、symmetric memory／NVLS、FlashInfer 合併）：**待 L2 掃原始碼、S01 摘錄啟動紀錄**。
- **A100 時期的推論數字**（只當對照）：
  - 通訊占每步 8–27% [A100: M5]
  - vLLM 一次 all-reduce 中，全體等待占 34–42%，每次之間空檔 3–4 µs [A100: M11]

## 六、量測方法：A100 時期的教訓（在 Nano5 要重新確認）

| 教訓 | 做法 | 出處 |
|---|---|---|
| 時脈會隨暖機變化：不暖機時跑在 1155–1395 MHz，越早量的越慢 | 開始前所有 GPU 一起忙 2 秒；量測前記錄 SM 時脈（NVML） | [A100: `exp/M18_incremental/README.md` 步驟 a] |
| 每次 launch 量的第一個設定特別慢 | 丟掉第一個設定（或先跑一個暖身設定） | 同上 |
| 不同次 launch 的結果可以差 10% 以上 | 要比較的東西放在**同一次 launch 內隨機交錯**；跨次比較只看趨勢 | [A100: M15 vs M16、M17] |
| 事後從多個選項挑最好的會高估 | 校準和驗證用不同的 launch（E6） | [A100: M18 步驟 b] |
| 測試程式本身會出錯：暖機迴圈各卡輪數不同，全體通訊對不上而卡死；單次執行加清快取，讓大資料看起來變慢 | 每輪由全體共同決定是否繼續（例如 all-reduce 取最小值）；正式量測前先小規模試跑 | [A100: M18 步驟 a 中止紀錄、M12→M13] |
| 計時方式 | CUDA graph，K=20 次 × R=20 輪，8 個 rank 取最大；5 次獨立 launch，每次取中位數後再取中位數，區間＝5 次的 [最小, 最大] | [A100: M18 卡「速度」] |
| 「沒看到錯誤」不等於「保證正確」 | 正確性論證分清楚證明、觀察、假設；需要時做壓力測試 | [A100: M8、`docs/method_design.md`] |

## 七、統計與上限的算法

- **非確定性數據**（R3、R4）：
  - 用中位數加無母數信賴區間，例如以二項分布取順序統計量，或用 bootstrap。
  - 停止條件：中位數的 99% 信賴區間在 ±5% 以內；做不到就報實際區間。
- **彙總**（R9）：
  - 時間用算術平均，速率用調和平均
  - 比值盡量不彙總，非彙總不可才用幾何平均
  - 跨 rank 取最大（最慢的那張卡決定完成時間），並寫明（R7）
- **延遲類問題**看 p99 或最大值（R6）。
- **理論上限**（R8）：
  - 頻寬上限：每 GPU NVLink 單向頻寬（M20 實測 link 數乘上每條速度）；InfiniBand 每埠 50 GB/s。
  - ring 或兩段式 all-reduce 的時間下限約為 2(n−1)/n × S ÷ 每 GPU 單向頻寬 [背景]。
  - NVLS 的下限公式待 M21 依實測與文件定義，不先寫死。
  - A100 時期把「最小延遲＋最大頻寬」拼成「可達水準」被證明有問題（D046），Nano5 上不再這樣拼。

## 八、白話用語對照（回報給使用者時用）

CLAUDE.md 規定白話結論不用英文術語。下表是固定譯法，第一次出現時可加比喻。

| 術語 | 白話 | 比喻（需要時） |
|---|---|---|
| all-reduce | 全體加總（每張卡都拿到總和） | 每個人報自己的數字，最後每個人手上都有總數 |
| reduce-scatter | 分工加總 | 每個人負責加總其中一段 |
| all-gather | 收齊 | 每個人把自己那段發給所有人 |
| all-to-all | 互相交換 | 每個人給每個人各一份不同的資料 |
| send/recv、P2P | 點對點傳送 | — |
| NVLink | 卡與卡之間的專用高速線 | — |
| NVSwitch | 連接所有卡的交換器 | 路口的交流道 |
| NVLS / multimem | 交換器內加總 | 收費站直接幫你把各車道的數字加好 |
| InfiniBand | 機器之間的高速網路 | — |
| GPUDirect RDMA | 網路卡直接讀寫 GPU 記憶體 | 不用先放到倉庫再轉運 |
| channel | 傳輸通道 | 高速公路的車道 |
| SM | 運算單元 | — |
| kernel | GPU 上執行的一段程式 | — |
| barrier | 全體等待（大家都到了才繼續） | — |
| latency / bandwidth | 延遲／頻寬 | 車程時間／路寬 |
| TP | 把一層模型切給多張卡一起算 | — |
| EP / MoE | 把不同「專家」放在不同卡上 | — |
| decode / prefill | 生成階段（一次產一個字）／讀入階段（一次讀整段提示） | — |
| 信賴區間 | 數字可能落在的範圍 | — |
| commit / branch / push | 存一個版本／版本線／上傳 | — |
| Slurm、job | 排隊系統、工作 | — |
| spec / report | 規格書／報告 | — |
