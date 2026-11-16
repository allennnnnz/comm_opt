# 目前狀態

> 每次 session 開始先讀本檔，結束時更新「目前狀態」與「下一步」。

## 目前狀態（2026-11-16 深夜，nano5 分支）

- **換環境、重新開始**（D050、D051）：Nano5（8× H100／H200 每台，Slurm，D048）；文獻優先、不受 A100 題目限制。計畫 `docs/plan_nano5.md`
- **文獻與原始碼調查完成**：R02_T1–T5（約 134 項核實）、R03（8 個系統原始碼）。Opus 整理與抽查：**`docs/notes_synthesis_nano5.md`**
- **候選方向**（D052，使用者選擇全部繼續深入）：A 兩層交換器加總（NVLS＋IB SHARP，主推）、B 跨機 TP decode 小訊息延遲、C 通訊的 SM 資源稅、D 效能模型選 collective／E PD 分離 KV 傳輸
- **S01 環境建置**：部分完成（R01）。vLLM 0.32.0+cu129、torch 2.13.0+cu129、NCCL 2.29.7、nccl-tests v2.21.1 已編譯；安裝在 `/work/allen0117/comm_opt_env`；`/home` 配額 100 GB
  - **待辦**：8 卡驗證工作 475444（torch／vLLM TP=8）、475459（單台 H100 nccl-tests）在 `dev` 排隊，預估 2026-11-17 01:42／02:02 開始（23:45 查詢） → 讀 `results/S01_env_setup/*_job475444`、`*_job475459`；確認 torch NCCL 版本＝nccl-tests 執行時版本
  - **待送**（一次一個，不要同時排太多）：2 台 H100、單台 H200、2 台 H200 的 all_reduce 冒煙測試；用 `exp/lib/nano5_env.sh` 再跑一次單台驗證
  - 之後把 R01 補完並驗收；vLLM 0.32.0 正式記為決策
- **M20**：已核准，加了 P10（SHARP，2 台 H100）；S01 通過後交給 Sonnet 執行
- **論文搜尋工具**：2026-11-16 使用者重開 VS Code 後可用（arXiv MCP、paper-search MCP）。arXiv 搜尋**要用欄位前綴**（`abs:"..."`）＋`sort_by: relevance`，否則回傳無關的最新論文
- **S04 候選方向深入調查**（2026-11-16 23:55 交給三位 Sonnet）：規格 `docs/specs/S04_direction_deep_dive.md`；報告 `R04_A_switch_reduction`、`R04_BE_crossnode_inference`、`R04_CD_smtax_model`（待交回後 Opus 讀＋抽查）

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

1. 讀 S01 驗證工作 475444、475459 的結果；依序送其餘冒煙測試；補完並驗收 R01
2. S01 通過後，把 S05（M20）交給 Sonnet；R04_A 回來後補 SHARP 設定給 `sharp.sbatch`
3. 讀 R04_A／BE／CD，抽查關鍵論點，更新候選方向判斷（`docs/notes_synthesis_nano5.md`），回報使用者
4. M20 完成後：判定 P1–P10，寫 M21、M23 的執行規格書
