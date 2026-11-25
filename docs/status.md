# 目前狀態

> 每次 session 開始先讀本檔，結束時更新「目前狀態」與「下一步」。

## 2026-11-25 19:55 更新（以本節為準）

- **M25 H100 一輪完成**（498238，hgpn18，09:33–10:36，B3 因時間跳過 7 項含 `ar_default run2`）。與 H200 幾乎一致：T1、T2、T4、T5 不成立（T2／T4 用修正量法）；T3 在 H100 只有 8 MB 過 5%（4 MB 4.9%、16 MB 不重現）→ 邊緣。M20 的 NVLS 當機在另一台 H100 也不重現。詳見實驗卡「H100 一輪」
- **量測後補充（20:30）**：預填步 all-reduce 佔 kernel 時間 72B TP8 30–35%、30B MoE 25–35%（兩台一致；實驗卡末節）。文獻（FLUX 2406.06858、ISO 2409.11155、NanoFlow 2408.12757、Comet 2502.19811、Ladder-residual 2501.06589）的加速主要靠計算與通訊重疊，且多在預填／大批次
- **待使用者**：是否把研究問題轉向「預填／大批次的計算通訊重疊及其互相干擾」；I2 量法；是否回報 hgpn01 第 7 張卡過熱；D058 確認

## 2026-11-25 04:30 更新（部分過時）

- **M25 H200 一輪完成**（498237，hgpn46，5 段全部成功）。逐條判定寫在實驗卡「結論」。五個題目：T1 不成立（只 1 格），T2 取決於通訊比例的量法（修正後 ≤17.5%，<20%），T3 勉強成立（4–16 MB NCCL 選 NVLS，Ring LL128 快 5–6%），T4 取決於量法（修正後 16%），T5 不成立（1.2%）
- **量測後新增分析**：`b2_kernels.py`（只算解碼步、扣掉每步第一次 all-reduce 的等待），需使用者知悉並決定 I2／I3／I4 用哪個量法
- **H100 498238**（排除 hgpn01）排隊，預估 11-26 11:43
- **待使用者**：選題；I2 量法；是否保留 498238；是否回報 hgpn01 第 7 張卡過熱

## 2026-11-25 03:05 更新（部分過時）

- **M25 第一次主量測 497945（hgpn01）8 卡數據不能用**：第 7 張卡在負載下過熱降頻（中位數 345 MHz、93°C）。另修好三個程式問題（MPI UCX、B3 只跑一筆、B1 NVLS 計時彙整）並以 498225 驗證；72B TP8 卡住的原因待新數據（加了診斷版本）。細節見實驗卡「Opus 補充 2」與 D058
- **第二次主量測**：498237 = H200 整台 hgpn46（03:00 開始，約 75 分鐘）；498238 = H100 排除 hgpn01（排隊，預估 11-26）。B0 新增 8 卡溫度檢查，不合格就自動中止
- **待使用者**：選題用 H200 還是 H100 數據；是否保留 498238；是否向國網中心回報 hgpn01 第 7 張卡過熱
- **dev 分區**：實際只排得到 hgpn01（其他節點與 normal 共用且 normal 優先）

## 2026-11-25 00:30 更新（部分過時）

- **M25（選題用單節點量測）主工作已送出：497945**（`dev`、8 卡、2 小時，等資源中）。驗證跑了 5 次（497760、497896、497906、497916、497929），修正：mpirun 名額（`--oversubscribe`）、移除 torchcodec、FlashInfer 全部關閉（JIT 需 nvcc ≥ 12.8，叢集最高 12.6）、vLLM 關 `fuse_allreduce_rms`。實驗卡已補記
- **送出工具**：叢集 `wallet` 壞掉，`submit.sh --skip-wallet`（D057）
- **跨機器**：多機啟動失敗（pmix_v4／pmi2／mpirun UCX wireup），M20 SHARP 與 M23 都沒有數據；4 台 M23 已取消（D056）。待修
- **選題**：M25 跑完後，依卡上判讀規則並排五個單機題目（T1–T5）給使用者選
- **觀察（待正式量測）**：驗證時 GPU 忙碌時脈中位數約 1590 MHz；vLLM eager 模式的加總 kernel 時間含等待其他 rank，分析時要分開

## 2026-11-17 12:30 更新（第一批工作已跑完；以下 05:55 段落部分過時）

- **已完成**：475459（S01 nccl-tests，09:57，hgpn01）、475444（S01 vLLM，hgpn01）、475780（M20 H100 #1，hgpn01）、475782（M20 H200 #1，hgpn44）。仍排隊：475781（M20 SHARP 2 台）、475783、475784（M23）。原始數據已 commit
- **方向 A 的關卡出現負面訊號（與 P10 的五五波預期相關，待使用者決定）**：兩台不同節點（H100 hgpn01、H200 hgpn44）的 `sharp_hello` 都失敗：`no AM service record found(SA query)` → `Failed to connect to Aggregation Manager (sharp_am)`（`results/M20_env_snapshot/*/sharp_precheck/sharp_hello.txt`）。代表網路上找不到 SHARP 管理服務，使用者端無法自行啟用。P10 正式判定仍待 475781，但跨機 SHARP 很可能不可用，除非國網中心開啟
- **P6 正面訊號**：兩台都載入 `nv_peer_mem` 與 `gdrdrv`（`snapshot/lsmod_grep.txt`）；仍需跨機 NCCL 日誌確認 GPUDirect RDMA 實際使用
- **S01 nccl-tests 冒煙通過（只有 mpirun 能啟動）**：8 卡 128 MB busbw 約 371 GB/s、`#wrong`=0（`results/S01_env_setup/20261117_095709_job475459/allreduce_mpirun.log`）；`srun --mpi=pmix_v4` rc=16、`pmi2` rc=1
- **M20 單台 NVLS 段失敗、時脈段被跳過**：mpirun 下 NVLS 小訊息正常，但建立 512 MB multicast 群組後 `transport/nvls.cc:257 Cuda failure 999 'unknown error'`（`nvls/nvls_a_default.failed_mpirun.log:678`），整個程式失敗 → 腳本判定「沒有可用的啟動方式」而跳過 NVLS 對照與時脈輪次。另 `mcast_query.cu` 編譯失敗（計算節點沒有 `cc1plus`）。是否為可重現的現象待查
- **S01 vLLM 驗證失敗（環境問題）**：venv 的 python 指向 `/usr/bin/python3.12`，計算節點上不存在（`bad interpreter`）。需改用計算節點可見的 Python 重建 venv
- **M21 暫不送**：M20 第 1 台沒有產生時脈暖機結果（P7）

## 目前狀態（2026-11-17 05:55，nano5 分支）

- **05:51 所有工作仍在排隊**：`dev` 被小工作分佔，整台工作難以排上；475459 預估 08:48、475444 預估 09:08（一路延後）。M20／M23 綁 475459，之後才會開始

- **使用者 01:15 離開約五小時，指示「做完所有剩下的任務」**。執行方式因排隊壅塞調整（D053）
- **候選方向**：S04 完成（R04_A、R04_BE＋session2 版、R04_CD＋smtax_model 版）。整合判斷見 `docs/notes_synthesis_nano5.md` 最後一節：主線 A（兩層交換器加總），C 當量測軸，B 當應用；D 不做；E 備案。**待使用者決定是否採用 A 為主線**
- **待使用者處理**：是否寄出給國網中心的詢問稿 `docs/nchc_inquiry_20261117.md`（SHARP、`nvidia_peermem`、IB 拓樸）
- **S01**：驗證工作 475444（torch／vLLM TP=8）、475459（單台 H100 nccl-tests）在 `dev` 排隊（預估 04:00 前後）。Opus 待辦：讀結果、補完並驗收 R01、把 vLLM 0.32.0 記為決策。其餘冒煙測試併入 M20／M23（D053）
- **M20**：腳本完成（S05，Sonnet），已排隊：475780（H100 第 1 台，`dev`）、475781（2 台 SHARP，`normal`）、475782（H200 第 1 台，`normal2`），以 `afterok:475459` 綁 S01 的 nccl-tests 冒煙測試（原 475733–475735 綁兩個 S01 工作，但 vLLM 測試 475444 預估要到 11-18；`scontrol update` 被叢集拒絕，01:50 取消重送）。R05 為階段性報告。**H100 第 2、3 台、H200 第 2 台尚未送**：Opus 看過 475780 輸出正常後送（`exp/M20_env_snapshot/node.sbatch`，`--exclude` 已量節點）
- **M21**：腳本完成（`node_a.sbatch` 約 70 分、`node_b.sbatch` 約 50 分，`dev`，只乾跑過）；Opus 看過 M20 第 1 台（P4、P7）後決定暖機長度再送。R06 第 5 節 16 條已由 Opus 審過（01:55）：全部同意；#5 已補跨機 NVLS＋SHARP；#12 已確認 `Bytes -> Algo` 屬 NCCL_TUNING（v2.32.6 `src/enqueue/enqueue.cc:900`）；#15 已建分析環境；#9（M23 無暖機）待 M20 P7 決定，可改執行時讀入的 `m23_lib.sh`
- **M23**：已排隊 475783（4 台 H100，`4nodes`，9 h）、475784（2 台 H200，`normal2`，3 h），以 `afterok:475459` 綁 S01（原 475773／475774 取消重送）；預估 11-19／11-26。設定清單在執行時讀入的 `m23_lib.sh`（Opus 已加入跨機 NVLS＋SHARP）
- **同節點紀錄**：叢集拿掉 `--exclusive`，量測工作改以 `exp/lib/cotenants.sh` 記錄同節點工作（D054，待使用者確認）
- **分析用 Python**：`/work/allen0117/comm_opt_env/venvs/analysis/bin/python`
- **工作環境**：H100 分兩組 hgpn01–06、hgpn17–21；H200 為 hgpn39–41、43–46。論文搜尋 MCP 可用，但 arXiv 常回 429
- **注意**：同一專案只開一個 Claude 對話（2026-11-16 深夜曾有兩個對話同時寫入，R04_BE 被覆蓋後已還原）

## 已核准的實驗

| ID | 核准日期 | 狀態 |
|---|---|---|
| M0_baseline | 2026-11-02（D004） | 完成 |
| M4_repeat | 2026-11-02（D006） | 完成 |
| M2_p2p_channels | 2026-11-02（D006） | 完成 |
| M5_e2e_comm_share | 2026-11-02（D009） | 完成 |
| M6_allreduce_bakeoff | 2026-11-03（D012） | 完成 |
| M7_where_slow | 2026-11-03（D014） | 完成 |
| M8_ll128_atomic_a100 | 2026-11-03（D016） | 完成 |
| M9_fused_baselines | 2026-11-04（D019） | 完成 |
| M10_why_fused_slow | 2026-11-04（D021） | 完成 |
| M11_before_prototype | 2026-11-04（D023） | 完成 |
| M12_push_vs_pull | 2026-11-04（D025） | 完成 |
| M13_push_retest | 2026-11-04（D027） | 完成 |
| M14_fused_prototype | 2026-11-04（D029） | 完成 |
| M15_bf16 | 2026-11-04（D031） | 完成 |
| M16_epoch_fix | 2026-11-04（D033） | 完成（修好但變慢，D035） |
| M17_epoch_ab | 2026-11-04（D036） | 完成（採用 m16nf，D037） |
| M18_incremental | 2026-11-04（D038） | 暫停（步驟 a、b 完成；D041、D042） |
| M19_small_batch | 2026-11-04（D044） | 完成（D045） |
| S01_env_setup（環境建置，非量測） | 2026-11-16（D050） | 進行中 |
| S02_lit_map（文獻地圖 T1–T5） | 2026-11-16（D051） | 進行中 |
| S03_system_code_survey | 2026-11-16（D051） | 進行中 |
| M20_env_snapshot | 2026-11-16（D051） | 已核准，待 S01 |

## 下一步

1. S01 工作完成後：讀結果、驗收 R01；把 vLLM 0.32.0 記為決策
2. M20 H100 第 1 台完成後：判讀 P4（NVLS）、P6（GPUDirect RDMA）、P7（時脈）、`sharp_hello`；決定 M21 暖機長度並送 M21
3. 讀 R05、R06，抽查；M20 全部完成後逐條判定 P1–P10
4. 使用者回來後：決定主線（A？）、是否寄國網中心詢問
