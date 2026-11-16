# R06 M21／M23 準備報告

- 執行者：Sonnet 5.5，2026-11-17；規格書 `docs/specs/S06_M21_M23_baselines.md`
- 未 commit、未 push。M21 未送出；M23 兩個工作已送出（皆在 Dependency 排隊）。

## 1. 建立的檔案

| 檔案 | 用途 |
|---|---|
| `exp/lib/nccl_bench.sh` | 共用：環境（venv NCCL 在最前；SHARP 外掛只在 plugin=1 時加 `LD_LIBRARY_PATH`，不 load hpcx 模組）、啟動方式探測（pmix_v4 → pmi2 → mpirun，第一個 rc=0 且有結果列者）、`add_cfg`（每設定 5 次 run＋1 次 probe）、種子洗牌、`run_plan`（每段各自記 exit code 到 `runs.tsv`，失敗繼續；超過時間預算的啟動跳過並記 `SKIPPED_TIME_BUDGET`） |
| `exp/lib/launch.sh` | 以 `exec` 啟動 srun／mpirun（讓 `timeout` 的訊號直達），含 `--distribution`、`--nodelist` |
| `exp/lib/rankwrap.sh` | 每個 rank 內設定 `CUDA_VISIBLE_DEVICES=$GPU_LIST`（Slurm 會覆蓋外部設定，所以放在 rank 內），並印出實際值 |
| `exp/lib/analyze_common.py` | 解析 nccl-tests 輸出、中位數＋95% bootstrap CI（跨 5 次啟動，10000 次重抽，固定種子）、cotenants 旗標、畫圖（matplotlib 缺就略過圖） |
| `exp/M21_intranode_baseline/{m21_lib.sh,node_a.sbatch,node_b.sbatch,analyze.py}` | M21 |
| `exp/M23_internode_baseline/{m23_lib.sh,h100_4node.sbatch,h200_2node.sbatch,analyze.py}` | M23 |
| 兩張實驗卡的「指令」段 | 只補指令，未動「預期」 |

Opus 追加要求（cotenants）：四個 sbatch 都在 `$OUT` 建立後立刻 `source /home/allen0117/comm_opt/exp/lib/cotenants.sh`，之後沒有再設 EXIT trap；`--exclusive` 保留（叢集實際會移除，送出時有警告）。`analyze_common.py` 讀 `cotenants_summary.txt`，summary CSV 的 `shared_flag` 欄：1＝開始或結束時有其他工作在我們的節點、0＝沒有、`unknown`＝該工作沒有此檔。**M23 兩個工作是在加入此行之後才送的，腳本內已含 cotenants，不需處理。**

## 2. nccl-tests 參數確認（v2.21.1，commit afd59ab，`third_party/nccl-tests/src/`）

- 跨 rank 取最大值：`-a 3`。`common.cu:121` 註解 `(0=RANK0,1=AVG,2=MIN,3=MAX)`；`common.cu:357-359` `average==3 ? MPI_MAX`；選項解析 `common.cu:1333-1334`；說明 `common.cu:1449`。（預設 `average=1`＝平均。此路徑為 MPI_Allreduce，需 MPI 版建置；S01 已用 srun/pmix 跑過。）
- `-w`／`-n`：`common.cu:96-97`（預設 warmup 1、iters 20）、解析 `common.cu:1277-1279`、`1287-1289`；暖機迴圈 `common.cu:728-730`，計時迴圈 `common.cu:757-764`，每次時間＝總時間／(iters×agg_iters) `common.cu:796-800`。
- busbw／位元組定義：all_reduce `all_reduce.cu:59-65`（algbw＝size/time，busbw＝algbw×2(n−1)/n）；alltoall `alltoall.cu:17-23`（paramcount＝(count/nranks)&~(16/eltSize−1)，**`-b` 的 size 是每張卡的總送出量**，每對各 size/n）與 `alltoall.cu:47-53`（algbw＝`count*nranks*typesize/sec`，count 即 paramcount，所以 algbw≈每張卡總送出量/時間；busbw＝algbw×(n−1)/n，與卡的預期推算一致）；sendrecv `sendrecv.cu:10-16`（送出＝接收＝count）與 `sendrecv.cu:38-43`（busbw＝algbw，factor 1）。
- sendrecv 的配對：`sendrecv.cu` RunColl（約 `:95-105`）每個 rank 送給 (rank+1)%n、收自 (rank−1+n)%n，即環狀位移，不是兩兩配對。見第 5 節。
- 另：每個 rank 的 GPU＝localRank（`common.cu:1660`），所以 2／4 卡用 `CUDA_VISIBLE_DEVICES` 取前幾張。

NCCL 文件確認：`NCCL_NET_GDR_LEVEL=LOC` 的定義我在本次未能確認（WebFetch 只拿到前 100000 字元，未涵蓋該節；僅見 `NCCL_MULTI_RANK_GPU_ENABLE` 段提到「lower limits on NCCL_NET_GDR_LEVEL=LOC」）。我的背景知識是 LOC＝永不使用 GPUDirect RDMA，**但這是記憶，未驗證**；M23 的 `*_gdroff` 設定實際效果請從 probe 日誌的 `GDR` 行驗證（日誌會記 `NET/IB : GPU Direct RDMA Disabled/Enabled`）。Opus 若要，可讀文件該節或讓我續讀 offset 100000。

## 3. M21

- 設定數：42 個設定（節點 `node_a` 24 個、`node_b` 18 個）。node_a：all_reduce／all_gather／reduce_scatter × {2,4,8 卡} × NVLS {開,關}＝18，加 8 卡 all_reduce 的 NCCL_ALGO＝Ring／Tree／NVLS（3）與 Ring×PROTO＝LL／LL128／Simple（3）；node_b：broadcast／reduce／alltoall × 同上＝18。每設定 5 次 run＋1 次 probe：node_a 144 次啟動、node_b 108 次。
- 估算方法（無實測，需 Opus 審）：每次啟動＝2 段，每段含行程啟動與 NCCL 初始化（假設單機 8 卡 ≈ 10 s，含 NVLS 可能更久）＋ sweep（大段 8 GB 一次約 0.03–0.1 s×25 次迭代×2 位置，各大小加總約數秒；小段 `-n 2000` 各大小加總約數秒）。取單次 run ≈ 30 s、probe ≈ 15 s。node_a ≈ 120×30＋24×15＋暖機與探測約 2 分 ≈ 66–70 分鐘；node_b ≈ 90×30＋18×15 ≈ 50 分鐘。總計約 2 小時 > 1.5 小時，所以拆成兩個工作（以集體通訊類別拆：a＝all_reduce 系列、b＝其餘三種）。`--time`：a＝1:40:00、b＝1:20:00。腳本內有時間預算保護：a 超過 5700 s、b 超過 4300 s 後的啟動不跑並記錄（因順序已洗牌，缺的會均勻分散）。初始化時間若遠大於 10 s，a 會被預算截斷，請在 M20 第 1 台看過 NCCL 初始化時間後再調整。
- 暖機：開頭 30 s all_reduce 64 MB（`WARMUP_SECS` 可改，M20 P7 結果出來再定），前後各記 `nvidia-smi` 時脈（`warmup_clocks_*.csv`）。暖機以 `timeout` 結束，rc=124 為預期。
- 乾跑輸出（`submit.sh --dry-run`）：兩者皆通過：`wallet=68937.5213 running=0 measurement=1` / `DRY RUN: sbatch --parsable --mail-type=END,FAIL --mail-user=<...> exp/M21_intranode_baseline/node_a.sbatch`（node_b 同）。**未送出。**
- 離線測試：偽造的 run 目錄通過 `analyze_common` 解析與統計；`m21_plan`／`m23_*` 的計畫產生器已離線跑過（設定數與啟動數如上）。腳本本身（launch、srun）**尚未在計算節點上跑過**。

## 4. M23

| 工作 | 編號 | 排隊區 | 資源 | --time | 預估開始 |
|---|---|---|---|---|---|
| H100 | **475773** | `4nodes` | 4 節點、每節點 8 GPU、`--exclusive`（叢集會移除） | 9:00:00 | `squeue --start` 回 N/A（Dependency 未滿足，無法預估）；規格書 01:20 的試算約 11-19 |
| H200 | **475774** | `normal2` | 2 節點、每節點 8 GPU | 3:00:00 | 同上；規格書試算約 11-26 |

兩者皆 `--dependency=afterok:475444:475459`，送出沒有被拒絕（只有 `--exclusive is removed` 的警告）。送出前乾跑通過。

- H100 工作：4 台設定 12 個（all_reduce 9：default／Ring／Tree／NVLSTree／外掛＋COLLNET=0 的 default、Ring／外掛＋COLLNET=1 的 default、CollnetDirect、CollnetChain；加 all_gather、reduce_scatter、alltoall），再加每個節點對 18 個 2 台設定（all_reduce 9＋ag／rs／a2a＋sr1／sr8＋GDR 關 4 個），兩個節點對。總 48 個設定；啟動 288 次（240 run＋48 probe）。估算 run ≈ 90 s、probe ≈ 45 s（4 台初始化假設 ~30–40 s/段）≈ 6.6 小時，`--time` 9:00:00，預算保護 29700 s。
- H200 工作：14 設定（2 台設定不含 GDR 關），84 次啟動，估算約 1.5 小時，`--time` 3:00:00，預算 9600 s。
- 節點配對由 `SLURM_JOB_NODELIST` 的編號決定（hgpn01–06＝A 組、17–21＝B 組），寫到 `pairs.tsv`：優先「同組一對＋不重疊的跨組一對」；若 4 台全在同組，第二對標 `same2_nocross`（Q7 不可判定，分析的 Q7 表只列 same／cross 標籤）。
- 每次啟動保留 `NCCL_DEBUG=INFO`，子系統 `INIT,NET`；probe 啟動另加 `TUNING`（libnccl 字串確認含 `%s: %ld Bytes -> Algo %s proto %s channel{Lo..Hi}=...`，但這行屬哪個子系統我是推測，未實跑驗證）。若 probe 沒印出 Algo 行，分析的 `algo_proto_probe` 欄會空白，需改為 `COLL` 或讀 init 的 tuning 表。

## 5. 與卡不同、或卡沒寫清楚而由我決定的地方（請 Opus 審）

1. **每次啟動拆成兩段 nccl-tests**（8 B–64 MB 用 `-n 2000`；128 MB–8 GB 用 `-n 20`），為了「小訊息總時間 ≥ 10 ms」。代價是每次啟動重建 NCCL 兩次（M23 的「每次重新建連線」仍成立）。大段從 128 MB 起（-n 20，單次約 0.5 ms 以上，約 10 ms 以上）；8 B 在 -n 2000 約 16–20 ms，皆約略滿足，實際以 `runs.tsv` 與輸出時間為準（未實測）。實際 `-n` 記在 `runs.tsv` 的 args 欄。
2. **probe 啟動**：卡沒提到，我為了取得「實際使用的演算法」每設定多加 1 次 `-n 1 -w 0 -c 0` 的啟動（不計入 5 次統計）。
3. **alltoall 的最小大小 512 B**（小段 `-b 512`）：8 B 在 16–32 rank 時每對 0 位元組（`alltoall.cu:18`），無意義。
4. **sendrecv 的 sr8**：sendrecv_perf 是環狀位移而非兩兩配對，所以用 `--distribution=cyclic` 讓 16 個 rank 輪流排在兩台上，rank r 送給 r+1（另一台）、收自 r−1（另一台），16 張卡同時跨機；`sr1` 為兩台各 rank 0（GPU 0）。「兩台各 8 張成對」在不改 nccl-tests 的前提下就是這個意義。sendrecv 只在 2 台設定做。
5. **M23 的 4 台設定只做 default 演算法的 ag／rs／a2a**，SHARP AG／RS 未列（卡沒要求）。外掛的對照：`arP0_default`／`arP0_ring`（載外掛＋COLLNET=0），`arP1_*`（COLLNET=1）；**沒有跑外掛下的 Tree／NVLSTree／NVLS（跨機 NVLS＋SHARP，M20 的 (f)）**，因卡上只列 CollnetDirect／CollnetChain。
6. **GDR 關**：all_reduce 做 default 與 Ring 兩種（Q6 比 default，Q1 用 Ring，兩者都要對照），sendrecv 做 sr1 與 sr8，只在 H100 2 台的兩個節點對上各一份。
7. **兩個節點對各做完整的 2 台設定**（共 2×18 個），而非只做 Q7 需要的子集；占時約 2 小時。若要縮短可改成只做 Q7 子集。
8. **H200 不做 GDR 關**（卡規定 GDR 關只在 2 台 H100）。
9. **M23 沒有暖機**（卡沒要求）；M21 有（卡的 Opus 補充）。
10. **指標取 out-of-place**（nccl-tests 預設呈現的主要欄位）；in-place 也在 CSV。分析比較以 `-a 3` 的最大值時間為準。
11. **M21 把 NVLS 開／關套用到全部 6 種集體通訊**（照卡），其中 broadcast／reduce／alltoall 理論上 NVLS 不影響，這 18 個設定可能只是重複量測雜訊。
12. **所有 NCCL_ALGO 強制皆可能被 NCCL 忽略／退回**；分析以 probe 的 TUNING 日誌為準（需驗證，見上）。
13. 每段 timeout：M21 300 s、M23 600 s（防止 CollNet／外掛卡死拖垮整個工作；逾時記 rc=124 後繼續）。
14. `nano5_env.sh` 載 openmpi 會設 `SLURM_MPI_TYPE=pmi2`、`UCX_TLS=all` 等；我每次顯式 `--mpi=`，但 UCX 變數仍在環境中（NCCL 不使用，僅註記）。
15. 分析環境：venv 只有 numpy，沒有 pandas／matplotlib；`analyze.py` 只依賴 numpy，圖在 matplotlib 缺席時略過（CSV 照寫）。要出圖需另備 matplotlib（我沒有安裝）。
16. `exp/lib/` 內原有 `ncclrun.py`、`nccljson.py`（A100 時期遺留，未被使用，也未修改）。

## 6. 風險與待驗證

- 所有 shell 腳本只通過 `bash -n` 與離線計畫產生，**未在計算節點實跑**；M23 工作一旦開始才會暴露問題（如 pmix 與 `--distribution=cyclic`、`--nodelist` 的交互）。launcher 探測失敗時工作會直接結束並在 `summary.txt` 記錄。
- 洗牌是整份計畫（含 4 台與兩個 2 台節點對）一起洗，所以不同節點對的啟動交錯進行。
