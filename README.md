# 機器學習課程_學習紀錄

我將機器學習課程中較有興趣的實作過程紀錄於此專案儲存庫。<br><br>


## CNN模型訓練_動物影像分類

### 📌 專案簡介<br>
這個專案紀錄了我在課堂上的 CNN 影像分類實作過程(動物分類)：包含資料前處理與資料增強、模型設計與訓練，在驗證集上進行性能評估。<br>
專案提供分類結果的可視化，包括分類報告（Precision、Recall、F1-score）以及混淆矩陣，幫助觀察各類別的預測效果。<br><br>

### 📂 專案結構
| 檔案名稱 | 用途說明 |
|:---|:---|
| CNN模型訓練_動物辨識.py| 模型訓練程式，包含資料集切分、資料增強、模型結構設計、超參數設置|
| CNN模型載入_動物辨識.py| 載入模型並在測試集驗證，輸出分類報告與混淆矩陣|
| best_model_1.zip | 訓練完成的模型（需先解壓縮再載入)|

<br>

### 1. 資料前處理與資料增強
本專案使用自訂影像資料集（存於 `X_train.npy` 與 `y_train.npy`），  
資料集共 **15,396 筆影像**，每張影像大小為 **224×224×3 (RGB)**。  
（因資料來源限制，此處不公開）

**1.1. 資料前處理**

| 步驟 | 方法 | 說明 |
|:---|:---|:---|
| 載入資料 | `X_train.npy`, `y_train.npy` | 影像 shape = (15396, 224, 224, 3)，標籤 shape = (15396,) |
| 正規化 | `X.astype(np.float32) / 255.0` | 將像素值縮放至 [0, 1] |
| 標籤轉換 | `y.astype(np.int64)` | 確保標籤為整數格式 |
| 資料切分 | `train_test_split` | 驗證集比例 20% (`test_size=0.2`)，隨機種子 42 (`random_state=42`)，分層抽樣 (`stratify=y`) |

<br>

**1.2. 資料增強**

| 資料集 | 操作 | 參數 |
|:---|:---|:---|
| **訓練集 (train_transform)** | RandomHorizontalFlip | 預設 50% 機率 |
|  | RandomRotation | ±15° |
|  | ColorJitter | brightness=0.2, contrast=0.2, saturation=0.2 |
|  | ToTensor | 轉為 PyTorch Tensor |
| **驗證集 (val_transform)** | ToTensor | 僅轉換，不做增強 |

<br><br>

### 2. 模型結構設計
訓練集圖像輸入尺寸**：本模型輸入影像為 **224×224×3**（RGB）。若資料集原始尺寸不同，請先 `Resize((224, 224))`。

### 2.1. 特徵提取（Features）
| 區塊 | 層 | 輸出張量 |
|:---|:---|:---|
| Features-1 | Conv2d(3→32, k=3, p=1) → BN → ReLU → MaxPool2d(2) | 32×112×112 |
| Features-2 | Conv2d(32→64, 3, p=1) → BN → ReLU → MaxPool2d(2) | 64×56×56 |
| Features-3 | Conv2d(64→128, 3, p=1) → BN → ReLU → MaxPool2d(2) | 128×28×28 |
| Features-4 | Conv2d(128→256, 3, p=1) → BN → ReLU → MaxPool2d(2) | 256×14×14 |
| Features-5 | Conv2d(256→512, 3, p=1) → BN → ReLU → MaxPool2d(2) | 512×7×7 |

### 2.2. 分類（Classifier）
| 區塊 | 層 | 輸出張量 |
|:---|:---|:---|
| Classifier-1 | Flatten → Linear(512×7×7 → 1024) → BN → ReLU → Dropout(0.5) | 1024 |
| Classifier-2 | Linear(1024 → 512) → BN → ReLU → Dropout(0.4) | 512 |
| Classifier-3 | Linear(512 → 256) → BN → ReLU → Dropout(0.4) | 256 |
| Classifier-4 | Linear(256 → 128) → BN → ReLU → Dropout(0.3) | 128 |

### 2.3. 輸出（Output）
| 區塊 | 層 | 輸出張量 |
|:---|:---|:---|
| Output | Linear(128 → num_classes) | num_classes |

<br><br>

### 3. 訓練前超參數設置 (Pre-training Hyperparameters)

<br>

**3.1. 資料與訓練設定**
| 參數 | 值 |
|:---|:---|
| 批次大小 (Batch size) | 32 |
| 訓練週期 (Epochs) | 20 |

<br>

**3.2. 基本優化設定**
| 類別 | 參數 | 值 |
|:---|:---|:---|
| 損失函數 (Loss) | CrossEntropyLoss |  |
| 最佳化器 (Optimizer) | Adam | lr=0.0005 |

<br>

**3.3. 學習率調整策略 (Scheduler): 使用StepLR，每隔 `step_size` epoch 調整一次**
| 參數 | 值 | 說明 |
|:---|:---|:---|
| step_size | 5 | 每 5 個 epoch 調整 |
| gamma | 0.5 | 學習率乘 0.5 |

<br>

**3.4. 早停機制 (Early Stopping)**
| 參數 | 值 | 說明 |
|:---|:---|:---|
| patience | 5 | 若連續 5 個 epoch 驗證集準確率無提升則停止訓練 |

<br><br>

### 4.訓練過程
CNN模型的結構設計。<br>
CNN模型訓練過程中，各個回合的表現。
<p align="center">
  <img src="實作過程圖片/螢幕擷取畫面 2025-08-25 005335.png" width="55%" >
</p>

<br><br>

### 5. 結果可視化
在驗證集中隨機抽出16張動物圖片進行預測，動物辨識的結果良好。
<p align="center">
  <img src="實作過程圖片/螢幕擷取畫面 2025-08-25 005407.png" width="70%">
</p>
<br>
<p align="center">
  <img src="實作過程圖片/螢幕擷取畫面 2025-08-25 013814.png" width="45%" >&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="實作過程圖片/螢幕擷取畫面 2025-08-25 013827.png" width="45%">
</p>

