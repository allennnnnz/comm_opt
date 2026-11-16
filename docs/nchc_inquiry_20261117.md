# 給國網中心技術支援的詢問稿（草稿，2026-11-17）

> 用途：請使用者確認後自行寄出（填入姓名與聯絡方式）。內容只涉及叢集設定，不含研究細節。我們會同時在計算節點上自行測試（實驗 M20），回覆可與測試結果互相印證。

---

主旨：晶創主機（Nano5）InfiniBand SHARP 與 GPUDirect RDMA 支援詢問（計畫 MST114546）

您好，

我們在 Nano5 的 H100／H200 節點上做 GPU 間集體通訊（NCCL）的研究，想請教以下幾項叢集設定。我們沒有 root 權限，以下項目無法自行確認：

1. **InfiniBand SHARP（交換器內加總）**
   - 計算網路（Quantum-2／NDR）是否有啟用 SHARP Aggregation Manager（或 UFM 的 SHARP 功能）？
   - 一般使用者的工作是否可以使用 SHARP？例如透過 HPC-X 內的 `nccl_rdma_sharp_plugin`，設定 `NCCL_COLLNET_ENABLE=1`。
   - 若可使用，每個工作是否有 SHARP 資源配額（聚合樹數量等）的限制？低延遲樹（LLT）與串流聚合（SAT）是否都有開啟？

2. **GPUDirect RDMA**
   - 計算節點是否載入了 `nvidia_peermem` 核心模組？我們在登入節點沒有看到這個模組，不確定計算節點的狀況。
   - 若沒有載入，是否有其他建議的 GPUDirect RDMA 使用方式？

3. **InfiniBand 拓樸**
   - 各節點 8 張 400G 網卡是否為 rail-optimized 接法，也就是各節點同一編號的網卡接到同一組 leaf 交換器？
   - H100 節點 hgpn01–06 與 hgpn17–21 是否位於不同的 leaf／spine 交換器之下？跨組通訊是否會多經過一層交換器？
   - 是否有提供 NCCL 用的拓樸檔，或建議的 NCCL 環境變數（例如 `NCCL_IB_HCA`）？

4. **IPoIB**：計算節點的 InfiniBand 介面是否有設定 IPoIB？SHARP 用它分送結果。

5. **GDRCopy**：計算節點是否也有 `gdrdrv`？我們在登入節點看到 `/dev/gdrdrv`。

謝謝您的協助。

（姓名）
（聯絡方式）
