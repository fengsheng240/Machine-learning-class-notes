import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import random
from torch.utils.data import TensorDataset, DataLoader

# 讀取資料 -------------------------------------------------------------------------------
X_test = np.load('X_train.npy')
y_test = np.load('y_train.npy')
print("資料形狀：", X_test.shape)
# ----------------------------------------------------------------------------------------



# CNN 結構（和訓練時完全一致）
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, len(np.unique(y_test)))
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN(num_classes=len(np.unique(y_test))).to(device)

# 載入模型----------------------------------------------------------------------------
model = torch.load("best_model_1.pt", map_location=device)
# ------------------------------------------------------------------------------------

model.eval()
print(" 模型已載入完成！")

# 前處理（直接轉 tensor，除以 255.0，並 permute）
X_test_tensor = torch.from_numpy(X_test).float() / 255.0
X_test_tensor = X_test_tensor.permute(0, 3, 1, 2)  # HWC ➜ CHW
y_test_tensor = torch.from_numpy(y_test).long()
print(" 圖像轉換完成！")

# 建立 Dataset 與 DataLoader
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, pin_memory=True)

# 批次推論
y_true, y_pred = [], []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        preds = outputs.argmax(dim=1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

# 計算準確率
acc = accuracy_score(y_true, y_pred)
print(f"\n 測試集準確率：{acc:.4f}")

# 顯示 16 張隨機圖片及其預測 / 真實標籤
idxs = random.sample(range(X_test.shape[0]), min(16, X_test.shape[0]))
plt.figure(figsize=(10, 10))
for i, idx in enumerate(idxs):
    img = X_test[idx]  # shape: (H, W, 3), range: 0~1
    img_display = (img * 255).astype(np.uint8)  # 還原顯示
    img_tensor = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
    with torch.no_grad():
        pred = model(img_tensor).argmax(1).item()
    plt.subplot(4, 4, i + 1)
    plt.imshow(img_display)
    plt.title(f"True: {y_test[idx]} | Pred: {pred}")
    plt.axis('off')
plt.tight_layout()
plt.show()

# 分類報告與混淆矩陣
print("\n Classification Report:")
print(classification_report(y_true, y_pred, digits=4))
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()