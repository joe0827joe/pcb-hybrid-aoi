import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

# --- 1. 定義輕量化 CNN (針對 Intel CPU 最佳化) ---
class TinyPCBNet(nn.Module):
    def __init__(self, num_classes=7):
        super(TinyPCBNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# --- 2. 訓練主流程 ---
def train_model():
    DATA_DIR = "data/patches"
    batch_size = 32
    epochs = 15
    lr = 0.001

    # 初始化增強 (Augmentation) 對應「真實產線差異」
    transform = transforms.Compose([
        transforms.Grayscale(), # 強制轉灰階
        transforms.RandomRotation(5), # 模擬旋轉差異
        transforms.RandomAffine(0, translate=(0.05, 0.05)), # 模擬對位位移
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # 載入數據
    full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型、優化器與損失函數
    model = TinyPCBNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"--- Training Started (Real Model) ---")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{epochs}], Avg Loss: {total_loss/len(train_loader):.4f}")

    # 存檔供 OpenVINO 轉換
    torch.save(model.state_dict(), "python_research/models/pcb_classifier_v1.pth")
    print("✅ Real Model Saved to python_research/models/pcb_classifier_v1.pth")

if __name__ == "__main__":
    if os.path.exists("data/patches"):
        train_model()
    else:
        print("Dataset not ready. Please run scripts/preprocess_patches.py first.")
