# Git 歷史改寫紀錄（2026-11-04）

- **原因**：使用者要求把專案 push 到 GitHub（公開 repo `allennnnnz/comm_opt`）。GitHub 單一檔案上限 100 MB，而歷史中有 M10 由 nsys 轉出的衍生檔 `results/M10_why_fused_slow/20261104_083556/nsys_torch_h5120_r0.sqlite`（291 MB）。依使用者決定（2026-11-04），從歷史中移除它與同一實驗的 `nsys_lmd_h5120_r0.sqlite`（16 MB，同為衍生檔）。兩個檔案仍在本機硬碟；原始追蹤檔 `.nsys-rep` 保留在歷史中（sha256 見 `results/MANIFEST_large_files.txt`）
- **方法**：`git filter-branch --index-filter 'git rm --cached --ignore-unmatch ...' -- main`；48 個 commit 全部保留，每個 commit 的訊息與其他檔案內容不變，只有 commit 編號改變
- **完整備份**（改寫前）：`/home/rogerlee/comm_opt_git_backup_20261104.bundle`（360 MB，`git bundle verify` 通過；不在 repo 內）
- **對照用途**：`results/*/env.txt` 等處記錄的 `comm_opt <hash>` 是改寫前的編號，請用下表換算

| # | 改寫前 | 改寫後 | commit 訊息 |
|---|---|---|---|
| 1 | `5e904c4a1dbd` | `5e904c4a1dbd` | Set up research rules, project layout, and the M0 baseline experiment |
| 2 | `82bc9079eba7` | `82bc9079eba7` | M0 baseline: run nccl-tests sweep, analyze, record 7 candidate phenomena |
| 3 | `033c7982b521` | `033c7982b521` | Add plain-language conclusion section to experiment cards |
| 4 | `08a662895c46` | `08a662895c46` | Pre-register M4_repeat and M2_p2p_channels before running |
| 5 | `c609d32e1885` | `c609d32e1885` | M4 and M2_p2p_channels done; G1 report recommends F1 |
| 6 | `3e8bc8c92af8` | `3e8bc8c92af8` | Plain-language summaries avoid English jargon and codes |
| 7 | `23d13865221c` | `23d13865221c` | D008: record F1 as an observation; next measure end-to-end communication share |
| 8 | `f6733b8a1129` | `f6733b8a1129` | M5: pre-register end-to-end comm-share experiment and inference tooling |
| 9 | `fef7ceb67e95` | `fef7ceb67e95` | M5: training recipes/driver, inference analysis, and overlap-cost helper |
| 10 | `f1a39405e4df` | `f1a39405e4df` | M5 done: communication is 8-30% of end-to-end time on this node |
| 11 | `81b17dd54b73` | `81b17dd54b73` | M5 card: say 'framework defaults', not 'mainstream settings'; list how defaults differ from production practice |
| 12 | `719e829256d6` | `719e829256d6` | Lit note: novelty scan for TP inference communication directions |
| 13 | `a57dda2abad7` | `a57dda2abad7` | CLAUDE.md: all replies to the user in Traditional Chinese |
| 14 | `dfdd918e374f` | `dfdd918e374f` | Literature survey mapped to measured data; recommend decode all-reduce (D011) |
| 15 | `e59e9bacca6a` | `e59e9bacca6a` | M6: pre-register all-reduce bake-off (criteria before data) |
| 16 | `a4162f948106` | `a4162f948106` | M6 done: existing all-reduces are 1.5-2.1x from achievable at 64K-1M |
| 17 | `cf85f43ab255` | `cf85f43ab255` | M7: pre-register where-is-it-slow sweep; D014 approval and low-latency-nccl findings |
| 18 | `e1171a68cf1f` | `e1171a68cf1f` | M7 card: record low-latency-nccl commit |
| 19 | `289dacc40833` | `289dacc40833` | M7 done: NCCL picks too few CTAs on A100; no existing all-reduce gets one sync + two-shot volume |
| 20 | `9be947126ca0` | `9be947126ca0` | M8: pre-register LL128-atomic feasibility test on A100 (D016) |
| 21 | `0ef084664dce` | `0ef084664dce` | M8: NVLink-only writers (GPUs 1-7) and a raced-lines power counter, before the main run |
| 22 | `60aa7edab6eb` | `60aa7edab6eb` | M8 done: LL128-atomic premise likely holds on A100 with 32-bit atomics |
| 23 | `419dd0f3f634` | `419dd0f3f634` | Lit check: fused all-reduce + RMSNorm already exists on A100 (TRT-LLM, LMDeploy) |
| 24 | `1f52d7e4451c` | `1f52d7e4451c` | M9 card (proposal): measure existing fused AR+RMSNorm kernels on A100 |
| 25 | `0aa779955547` | `0aa779955547` | M9 card: record successful sm_80 build of FlashInfer's TRT-LLM fusion sources (compile only) |
| 26 | `28d12db4d232` | `28d12db4d232` | M9: torch bench, run script, LMDeploy test patch; pre-measurement corrections |
| 27 | `1299baa521b1` | `1299baa521b1` | M9 done: existing fused AR+RMSNorm kernels are not faster on A100; 2.2-3.1x from bound |
| 28 | `757526c2a672` | `757526c2a672` | M10 card: why fused kernels lose on A100 (cluster/block-count hypothesis), pre-registered |
| 29 | `fbf1fd8d8a35` | `f8176abe2147` | M10 done: missing clusters is not the main reason fused kernels lose on A100 |
| 30 | `34b15fe18324` | `9cdb6ce3eafd` | M11 card (proposal): 8-to-8 atomic reduce-scatter feasibility, vLLM time breakdown, NVLink utilization |
| 31 | `628aee9e199c` | `bbd9fed83a60` | M11: probes and run script; pre-measurement findings recorded |
| 32 | `d00abea4182d` | `1348b6c2aa87` | M11 done: atomic reduce-scatter is correct but the probe loses to vLLM; NVLink not saturated |
| 33 | `1d17ec923800` | `dc1c762da5d2` | docs: findings overview M0-M11 for planning the next method and experiment flow |
| 34 | `a0dc0c2d51b8` | `f4a274877dbe` | M12 card: LL128-style 16-byte push vs vLLM-style pull, pre-registered (D025) |
| 35 | `f4457d11a6b3` | `893d49422b1d` | M12: LL128 push/pull probe, run and analysis scripts (smoke test only) |
| 36 | `9bdc56694555` | `930a82b12788` | M12 done: LL128 push is as fast as pull for all-gather; reduce-scatter stays pull |
| 37 | `744a2ac8f02a` | `02a5e60dfad6` | M13 card: why rs_push is slow, re-test large messages in steady state (pre-registered, D027) |
| 38 | `628c58a9b8b2` | `527f74ecee00` | M13: probe (two new variants, steady mode), run and analysis scripts (smoke test only) |
| 39 | `5ed968730e42` | `f9d4ef5bd28f` | M13 done: switch by size; M12's large-message slowness was the one-shot harness |
| 40 | `52cdbe35ee24` | `d645bf3c7b3a` | M14 card: fused pull-RS + RMSNorm + push-AG prototype, pre-registered (D029) |
| 41 | `c41e7fe4c672` | `c5a377181b44` | M14: prototype kernel (PyTorch extension), benchmark, run and analysis scripts (smoke test only) |
| 42 | `89e08e2ca408` | `68340d78e605` | M14 done: fused prototype is a go (1.30/1.56/1.37x vs vLLM at 160K/320K/640K) |
| 43 | `993015adf69d` | `ade7db952fe8` | M15 card: bf16 support for the fused prototype, pre-registered (D031); start from M14 sources |
| 44 | `7d9fba6b0a11` | `f2efb41e2172` | M15: fp16/bf16 templated prototype, bench --dtype, run and analysis scripts (smoke test only) |
| 45 | `2d08e0fac7d3` | `47108ea59a00` | M15 done: prototype supports bf16 (1.33/1.60/1.39x vs vLLM bf16 at 160K/320K/640K) |
| 46 | `dbc5fb312045` | `e06439f150df` | docs: status inventory after M15; overview adds M12-M15 and the blind-spot list |
| 47 | `f003c8d4f0b7` | `0146bc6ce266` | M16 card: global epoch fix for the prototype, pre-registered (D033); start from M15 sources |
| 48 | `15a92888f1db` | `32af5fc3aeae` | M16: global-epoch kernel, interleaving test, run and analysis scripts |
