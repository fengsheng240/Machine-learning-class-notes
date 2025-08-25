import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import random
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split

# 顯示 GPU 資訊
print(torch.cuda.get_device_name(0))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
print("使用設備：", device)

# 讀取資料
X = np.load('X_train.npy')  # shape = (N, H, W, C)
y = np.load('y_train.npy')  # shape = (N,)
print("資料形狀：", X.shape)

# 預覽前 12 張圖像
plt.figure(figsize=(5, 5))
for k in range(12):
    plt.subplot(3, 4, k + 1)
    plt.imshow(X[k], cmap='gray')
plt.tight_layout()
plt.show()

# 預處理：正規化到 0~1
X = X.astype(np.float32) / 255.0
y = y.astype(np.int64)
num_classes = len(np.unique(y))

# 資料增強
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor()
])
val_transform = transforms.ToTensor()

# 自訂 Dataset 類別
class CustomImageDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = (self.X[idx] * 255).astype(np.uint8)  # 還原為 0-255
        img = Image.fromarray(img)  # numpy → PIL
        label = self.y[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

# 切分訓練 / 驗證集
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

train_dataset = CustomImageDataset(X_train, y_train, transform=train_transform)
val_dataset = CustomImageDataset(X_val, y_val, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, pin_memory=True)

print("訓練資料大小：", round(X.nbytes / (1024*1024), 2), "MB")

# CNN 模型
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
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

model = CNN(num_classes=num_classes).to(device)
print(model)

# 損失與優化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# EarlyStopping 設定
best_val_acc = 0
early_stop_counter = 0
patience = 5

# 開始訓練
print("\n🚀 開始訓練模型...\n")
for epoch in range(20):
    model.train()
    total_loss, correct = 0, 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()

    acc = correct / len(train_loader.dataset)

    model.eval()
    val_loss, val_correct = 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            val_correct += (outputs.argmax(1) == labels).sum().item()

    val_acc = val_correct / len(val_loader.dataset)
    val_loss = val_loss / len(val_loader.dataset)

    print(f"Epoch {epoch+1}/20 - Loss: {total_loss:.2f} - Acc: {acc:.4f} - Val Loss: {val_loss:.2f} - Val Acc: {val_acc:.4f}")
    scheduler.step()

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model_aug_222233.pth")
        torch.save(model, "best_model_full_2222233.pt")
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        if early_stop_counter >= patience:
            print("🛑 Early stopping triggered")
            break

# 預測視覺化
model.load_state_dict(torch.load("best_model_aug_222233.pth"))
model.eval()
idxs = random.sample(range(len(val_dataset)), 16)
plt.figure(figsize=(10, 10))
for i, idx in enumerate(idxs):
    img, true_label = val_dataset[idx]
    with torch.no_grad():
        pred = model(img.unsqueeze(0).to(device)).argmax().item()
    plt.subplot(4, 4, i + 1)
    plt.imshow(img.permute(1, 2, 0).cpu(), cmap='gray')
    plt.title(f"True: {true_label} | Pred: {pred}")
    plt.axis('off')
plt.tight_layout()
plt.show()
