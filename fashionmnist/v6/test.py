import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

BATCH_SIZE = 128
MODEL_FILE = "best_model_v6.pth"

if not os.path.exists(MODEL_FILE):
    print(f"FEHLER: '{MODEL_FILE}' nicht gefunden!")
    print("Führe zuerst 'python train.py' aus.")
    exit()


# ResNet Block (identisch zu train.py)
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class AdvancedCNN(nn.Module):
    def __init__(self):
        super(AdvancedCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.5)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.fc(out)
        return out


# GPU Support
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

model = AdvancedCNN()
model.load_state_dict(torch.load(MODEL_FILE, weights_only=True, map_location=device))
model = model.to(device)
model.eval()

print(f"Modell geladen auf: {device}\n")
print("="*70)
print("V6 Test mit Test Time Augmentation (TTA)")
print("="*70)

# Standard Test (ohne TTA)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])

test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

standard_acc = 100 * correct / total

print(f"\n1. Standard Test (ohne TTA):")
print(f"   Accuracy: {standard_acc:.2f}%")

# Test Time Augmentation (TTA)
print(f"\n2. Test mit TTA (5 Augmentationen)...")

tta_transforms = [
    transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))]),
    transforms.Compose([transforms.RandomHorizontalFlip(p=1.0), transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))]),
    transforms.Compose([transforms.RandomRotation(10), transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))]),
    transforms.Compose([transforms.RandomRotation(-10), transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))]),
    transforms.Compose([transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])
]

correct_tta = 0
total_tta = 0

with torch.no_grad():
    for aug_idx, tta_transform in enumerate(tta_transforms):
        test_dataset_tta = datasets.FashionMNIST(root='./data', train=False, download=True, transform=tta_transform)
        test_loader_tta = DataLoader(test_dataset_tta, batch_size=BATCH_SIZE, shuffle=False)

        predictions = []
        for images, labels in test_loader_tta:
            images = images.to(device)
            outputs = model(images)
            predictions.append(outputs.cpu())

        if aug_idx == 0:
            all_predictions = torch.cat(predictions, dim=0)
            labels_all = test_dataset_tta.targets
        else:
            all_predictions += torch.cat(predictions, dim=0)

# Durchschnitt über alle Augmentationen
all_predictions /= len(tta_transforms)
_, predicted_tta = torch.max(all_predictions, 1)
correct_tta = (predicted_tta == labels_all).sum().item()
total_tta = len(labels_all)

tta_acc = 100 * correct_tta / total_tta

print(f"   Accuracy mit TTA: {tta_acc:.2f}%")
print(f"   Verbesserung: +{tta_acc - standard_acc:.2f}%")

print("\n" + "="*70)
print("Zusammenfassung:")
print("="*70)
print(f"Testbilder:        {total}")
print(f"Standard Accuracy: {standard_acc:.2f}%")
print(f"TTA Accuracy:      {tta_acc:.2f}%")
print("="*70)
print(f"V4 Baseline:       93.27%")
print(f"V5:                90.70%")
print(f"V6 (Standard):     {standard_acc:.2f}%")
print(f"V6 (mit TTA):      {tta_acc:.2f}%")
print("="*70)

if tta_acc >= 98:
    print("✓✓✓ 98% ZIEL ERREICHT! ✓✓✓")
elif tta_acc >= 95:
    print(f"✓ Sehr gut! Nur noch {98 - tta_acc:.2f}% bis zum Ziel.")
else:
    print(f"Noch {98 - tta_acc:.2f}% bis zum 98% Ziel.")
