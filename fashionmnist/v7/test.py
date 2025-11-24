import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

MODEL_FILE = "best_model_v7.pth"
BATCH_SIZE = 256

if not os.path.exists(MODEL_FILE):
    print(f"FEHLER: '{MODEL_FILE}' nicht gefunden!")
    print("Führe zuerst 'python train.py' aus.")
    exit()


# SE Block
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# Efficient ResBlock
class EfficientResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_se=True):
        super(EfficientResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels) if use_se else nn.Identity()
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# Model
class EfficientCNN(nn.Module):
    def __init__(self):
        super(EfficientCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 3, stride=2)
        self.layer3 = self._make_layer(128, 256, 3, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, 10)

    def _make_layer(self, in_ch, out_ch, blocks, stride):
        layers = [EfficientResBlock(in_ch, out_ch, stride, use_se=True)]
        for _ in range(1, blocks):
            layers.append(EfficientResBlock(out_ch, out_ch, 1, use_se=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# GPU Setup
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

model = EfficientCNN()
model.load_state_dict(torch.load(MODEL_FILE, weights_only=True, map_location=device))
model = model.to(device)
model.eval()

print("="*75)
print("V7 Test - Efficient CNN mit SE-Attention")
print("="*75)
print(f"Device: {device}\n")

# Standard Test
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

accuracy = 100 * correct / total

print("="*75)
print("Ergebnisse:")
print("="*75)
print(f"Testbilder:     {total}")
print(f"Korrekt:        {correct}")
print(f"Accuracy:       {accuracy:.2f}%")
print("="*75)
print("\nVersion Vergleich:")
print(f"  V5:  90.70%")
print(f"  V6:  94.39%")
print(f"  V7:  {accuracy:.2f}% {'✓ BESTE VERSION!' if accuracy > 94.39 else ''}")
print("="*75)

if accuracy >= 98:
    print("\n✓✓✓ 98% ZIEL ERREICHT! ✓✓✓")
elif accuracy >= 96:
    print(f"\n✓✓ Sehr nah! Nur noch {98 - accuracy:.2f}% bis 98%")
elif accuracy >= 95:
    print(f"\n✓ Sehr gut! Noch {98 - accuracy:.2f}% bis 98%")
else:
    print(f"\nNoch {98 - accuracy:.2f}% bis zum 98% Ziel")

print("\nTipp: Mit mehr Epochen (25-30) könnten wir noch höher kommen!")
