# 🛠️ AOI-Hybrid: 環境建置手冊 (V1.1)

本專案使用 **Conda** 作為標準隔離開發環境，確保 CV 演算法與 DL 分類器在多機台間的一致性。

## 1. 隔離環境建置 (Conda Flow)
請在專案根目錄開啟終端機 (cmd/powershell)，依序執行：

```bash
# A. 建立獨立環境 (Python 3.9)
conda create -n pcb-aoi python=3.9 -y

# B. 啟動環境
conda activate pcb-aoi

# C. 安裝視覺核心依賴 (含 OpenCV, NumPy, Matplotlib)
conda install -c conda-forge opencv numpy matplotlib tqdm -y

# D. 驗證 OpenCV 啟動成功
python -c "import cv2; print('✅ OpenCV Ready:', cv2.__version__)"
```

## 2. 原始數據集對接 (DeepPCB Dataset)
跑通 CV 演算法之前，必須先就位數據資產。

```bash
# 執行自動化數據下載與解壓 (Target: data/DeepPCB)
python scripts/setup_data.py

# 注意：若您是手動下載，目錄名稱可能為 data/DeepPCB-master
```

## 3. 開發者效能驗證 (Stage 1 CV Testing)
用於確認環境中的 OpenCV 與您的硬體效能是否能達成 **130ms** 的延遲指標。

```bash
# 確認 OpenCV 可用且數據集路徑正確
python -c "import cv2; import os; print('✅ OpenCV:', cv2.__version__); print('✅ Data Folder:', 'data/DeepPCB-master' if os.path.exists('data/DeepPCB-master') else 'data/DeepPCB')"
```

## 4. 目錄結構定義 (Developer Layout)
- `python_research/`: 存放所有 CV/DL 原型與效能測試腳本。
- `data/`: 存放 DeepPCB 與其他開發資產 (已加入 .gitignore)。
- `scripts/`: 通用工具集 (如數據佈署、環境檢查)。
- `docs/`: 專案設計文件與協議規範。
- `cpp_deployment/`: C++ 推論核心部署目錄。
