import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import sys

# 取得專案目錄路徑
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PR_ROOT = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(PR_ROOT)

# 對齊 train_classifier.py 中的模型架構
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

def evaluate_accuracy():
    print("--- [ACCURACY EVALUATION] Starting Final Benchmark ---")
    MODEL_PATH = os.path.join(PR_ROOT, "models", "pcb_classifier_v1.pth")
    DATA_DIR = os.path.join(PROJECT_ROOT, "data", "patches")
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model file {MODEL_PATH} not found.")
        return
    
    if not os.path.exists(DATA_DIR):
        print(f"❌ Error: Dataset {DATA_DIR} not found.")
        return

    # 1. 準備驗證集 (與訓練時同樣的 20% Split 比重)
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
    test_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - test_size
    _, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    classes = full_dataset.classes # ['0', '1'...]

    # 2. 載入權重
    model = TinyPCBNet()
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    # 3. 測試
    correct = 0
    total = 0
    class_correct = [0] * 7
    class_total = [0] * 7
    
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 統計各類別準確度
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    accuracy = 100 * correct / total
    print(f"--- [Final Report] Summary ---")
    print(f"Overall Accuracy: {accuracy:.2f}%")
    print(f"Target Threshold: 95.00%")
    print(f"{'PASSED' if accuracy >= 95 else 'FAILED'}")
    
    print("\n--- Per-Class Accuracy ---")
    map_list = ["Background", "Open", "Short", "Mousebite", "Spur", "Copper", "Pin-hole"]
    for i in range(7):
        if class_total[i] > 0:
            print(f"[{i}] {map_list[i]:<10}: {100 * class_correct[i] / class_total[i]:.2f}% ({class_correct[i]}/{class_total[i]})")
        else:
            print(f"[{i}] {map_list[i]:<10}: N/A (No samples)")

    # 生成正式報告
    report_file = os.path.join(PR_ROOT, "results", "accuracy_report.txt")
    os.makedirs(os.path.dirname(report_file), exist_ok=True)
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("# AOI Classification Accuracy Report\n")
        f.write(f"- Status: {'PASSED' if accuracy >= 95 else 'FAILED'}\n")
        f.write(f"- Total Samples: {total}\n")
        f.write(f"- Global Accuracy: {accuracy:.2f}%\n")
        f.write("\n--- Details ---\n")
        for i in range(7):
            if class_total[i] > 0:
                f.write(f"{map_list[i]}: {100 * class_correct[i] / class_total[i]:.2f}%\n")

if __name__ == "__main__":
    evaluate_accuracy()
