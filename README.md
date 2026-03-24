# PCB-AOI: 混合動力瑕疵檢測系統 (Hybrid Defect Detection System)

[![C++](https://img.shields.io/badge/Language-C++17-blue.svg)](https://isocpp.org/)
[![OpenVINO](https://img.shields.io/badge/Inference-OpenVINO-orange.svg)](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit.html)
[![Accuracy](https://img.shields.io/badge/Target_Accuracy-95%25-green.svg)]

## 1. 需求分析 (Requirement Analysis)
在傳統 PCB 生產線中，視覺檢測面臨**「效能與精度的矛盾」**：
* **傳統 CV 演算法：** 速度極快，但難以區分微小且複雜的瑕疵類別（如辨識斷路與毛刺之細微差異）。
* **深度學習 (DL)：** 具備極高精度，但在缺乏高階 GPU 的工控機 (IPC) 環境下，推論延遲 (Latency) 無法滿足即時產線需求。

**本專案目標：** 利用 **參考比較法 (Referential Inspection)** 邏輯，建立一套能在 Intel CPU 環境下穩定運行，且兼顧高速定位與多類別分類的混合系統。

---

## 2. 規格與邊界 (Specifications & Boundaries)
本專案模擬中速自動化產線場景，設定以下硬性指標：

| 項目 (Item) | 規格 (Specifications) | 設計意圖 (Design Rationale) |
| :--- | :--- | :--- |
| **運行節拍 (Cycle Time)** | **70ms ~ 130ms / 幀** | 確保在中速產線（~4000 PPH）具備充足緩衝餘裕。 |
| **硬體環境 (Hardware)** | **Intel CPU (Non-GPU)** | 模擬低成本、高穩定性的工業邊緣運算設備。 |
| **準確率目標 (Accuracy)** | **$\ge$ 95%** | 降低漏檢率，確保工業級的出貨良率。 |
| **作業系統 (OS)** | **Windows 10/11** | 考量工業相機 SDK 與 PLC 通訊軟體之高度成熟度。 |
| **模型量化 (Quantization)** | **INT-8** | 在 CPU 上壓榨出 2-3 倍的推論效能，且精度損失 < 1%。 |

---

## 3. 系統設計 (System Design)

### 雙架構工作流 (Hybrid Pipeline)
系統採用解耦設計，將檢測流程分為兩階段：

1. **第一階段：規則式定位 (CV Engine)**
   * 利用 `cv::absdiff` 與形態學過濾 (Morphology) 在 10ms 內鎖定疑似瑕疵的 **ROI**。
   * **價值：** 大幅減少 AI 需要掃描的無效區域，節省 80% 以上的推論算力開銷。
2. **第二階段：權重分類引擎 (DL Engine)**
   * 將 ROI 裁切後送入經 **OpenVINO INT-8** 優化的輕量化模型進行 6 類瑕疵精確分類。
   * **價值：** 確保在複雜紋理下依然具備高度辨識力，解決傳統算法誤報率高的痛點。

### 🚀 快速啟動 (Quick Start)

詳情請參閱 [環境建置手冊 (environment_setup.md)](docs/environment_setup.md)。

```cmd
# 1. 啟動環境
conda activate pcb-aoi

# 2. 已就位數據集檢查 (若無數據請參閱數據環境配置一章)
dir data\DeepPCB-master
```

### 核心技術棧 (Tech Stack)
* **語言：** C++ 17 (利用 Smart Pointers 與 Memory Pool 確保記憶體管理之決定性)。
* **推論引擎：** Intel OpenVINO Toolkit (針對 Intel AVX-512 指令集優化)。
* **訓練框架：** Python / PyTorch。
* **模型轉換：** ONNX $\rightarrow$ OpenVINO IR (Intermediate Representation)。

---

## 4. 關鍵工程決策 (Key Engineering Decisions)
* **為什麼用 C++ 17？** 產線軟體需要極高的穩定性與即時性。C++ 能精確控制記憶體配置與釋放，避免 Python 垃圾回收機制 (GC) 造成的節拍抖動。
* **為什麼用 OpenVINO？** 針對 Intel 指令集深度優化，是目前工業界在無 GPU 情況下部署 AI 的標竿解決方案。
* **一致性保障：** 系統內建 `Preprocessor` 模組，確保 C++ 端的影像歸一化 (Normalization) 參數與 Python 訓練端完全鏡像對齊。

---

## 5. 數據環境配置 (Dataset Setup)
本專案依賴 **DeepPCB** 開放數據集進行開發與驗證。為維持 Repo 整潔，數據集並不包含在版本控制中。

**首次使用請執行下列指令：**
```bash
# 自動由 GitHub 抓取 DeepPCB 並放置於 /data/DeepPCB
python scripts/setup_data.py
```

---

## 6. 專案目錄結構 (Project Structure)
```text
pcb-hybrid-aoi/
├── multi_agent.yaml           # 【核心配置】多代理人協作與 Sprint 流程定義
├── AGENTRULE.md               # 【最高憲法】通訊協議、效能門檻與品質準則
├── README.md                  # 【專案導引】當前進度與環境安裝說明
├── requirements.txt           # 【依賴管理】Python 環境必要套件清單
│
├── python_research/           # 🧪 【核心研發】算法 prototype 與 驗證中心
│   ├── cv_prototype_v1.py     # Stage 1: CV 定位器算法
│   ├── dl_inference_v1.py     # Stage 2: DL 分類器集成 (OpenVINO)
│   ├── hybrid_pipeline_v1.py  # 整合管線: 端到端自動化檢測
│   ├── test_cv_prototype.py   # CV 單元測試
│   ├── test_dl_inference.py   # DL 單元測試 (含邊緣強健性驗證)
│   ├── test_integration.py    # 整合測試 (含 130ms 門檻與視覺化產出)
│   ├── run_all_tests.py       # 自動化測試總控運行器
│   ├── train_classifier.py    # AI 模型訓練邏輯
│   ├── visualize_results.py   # 視覺化對比分析 (CV vs. Ground Truth)
│   ├── pcb_classifier_v1.onnx # 部署用部署權重 (OpenVINO 格式)
│   └── test_data/             # 📊 各類稽核報告與測試結果圖片
│
├── scripts/                   # ⚙️ 【自動化腳本】後勤、維護與數據準備
│   ├── setup_data.py          # 自動下載原始數據集 (DeepPCB)
│   ├── preprocess_patches.py  # 將原始圖切成 64x64 小塊
│   ├── evaluate_accuracy.py   # 準確率基準測試 (拼 95% 以上門檻)
│   └── export_to_onnx.py      # 模型格式轉換工具 (PTH -> ONNX)
│
├── data/                      # 📦 【數據倉儲】大型外部二進位文件
│   ├── DeepPCB-master/        # 原始數據集與標註檔
│   └── patches/               # 處理後的小圖塊 (0~6 類別分類存儲)
│
├── cpp_deployment/            # 🚀 【未來部署】C++ 高性能移植物件 (開發預留中)
│   ├── include/               # C++ 標頭檔
│   └── src/                   # C++ 核心實作
│
└── docs/                      # 📑 【文檔中心】標準協議與開發契約
    └── contracts/
        └── detection_schema.md # 定義 [x1, y1, x2, y2] 與類別映對標準
```

```