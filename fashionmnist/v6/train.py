import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Konfiguration für 98% Accuracy
BATCH_SIZE = 128
LEARNING_RATE = 0.05
NUM_EPOCHS = 25
WEIGHT_DECAY = 5e-4

# Aggressive Data Augmentation
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])

# Daten laden
train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=train_transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# ResNet Block mit Skip Connection
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip Connection - wenn Dimensionen nicht passen, anpassen
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        # Hauptweg
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # Skip Connection addieren!
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# ResNet-ähnliches CNN für 98% Accuracy
class AdvancedCNN(nn.Module):
    def __init__(self):
        super(AdvancedCNN, self).__init__()

        # Initial Layer
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # ResNet Blocks - jeder Block hat 2 ResidualBlocks
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

        # Global Average Pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier mit Dropout
        self.fc = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.5)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        # Erster Block mit stride
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        # Rest mit stride=1
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


# CutMix Augmentation - mischt Bild-Patches
def cutmix_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    # Zufälliges Rechteck ausschneiden
    W = x.size()[2]
    H = x.size()[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    # Patch austauschen
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]

    # Adjust lambda
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam


# Label Smoothing Loss
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes=10, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.classes = classes
        self.confidence = 1.0 - smoothing

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))


model = AdvancedCNN()

# GPU Support - MPS für Apple Silicon
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"Training auf: CUDA GPU - {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    device = torch.device('mps')
    print(f"Training auf: Apple Silicon GPU (MPS)")
else:
    device = torch.device('cpu')
    print(f"Training auf: CPU")

model = model.to(device)

criterion = LabelSmoothingLoss(classes=10, smoothing=0.1)
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=WEIGHT_DECAY, nesterov=True)

# Cosine Annealing mit Warm Restarts
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1)

print("\n" + "="*70)
print("V6 - Advanced ResNet CNN für 98% Accuracy")
print("="*70)
print(f"Modell Parameter: {sum(p.numel() for p in model.parameters()):,}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Epochen: {NUM_EPOCHS}")
print(f"Device: {device}")
print("="*70)
print("Neue Features:")
print("✓ ResNet mit Skip Connections (8 Residual Blocks)")
print("✓ Tiefes Netzwerk (64→128→256→512 Filter)")
print("✓ CutMix Augmentation")
print("✓ Label Smoothing")
print("✓ Cosine Annealing mit Warm Restarts")
print("✓ Nesterov Momentum")
print("="*70 + "\n")

train_losses = []
test_accuracies = []
learning_rates = []
best_acc = 0

for epoch in range(NUM_EPOCHS):
    # Training
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # CutMix mit 50% Wahrscheinlichkeit
        if np.random.rand() < 0.5:
            images, labels_a, labels_b, lam = cutmix_data(images, labels, alpha=1.0)
            outputs = model(images)
            loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        running_loss += loss.item()

    scheduler.step()

    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    learning_rates.append(optimizer.param_groups[0]['lr'])

    # Testing
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_acc = 100 * correct / total
    test_accuracies.append(test_acc)

    # Bestes Modell speichern
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), "best_model_v6.pth")
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {avg_train_loss:.4f}, Acc: {test_acc:.2f}%, LR: {learning_rates[-1]:.5f} ✓ BEST')
    else:
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {avg_train_loss:.4f}, Acc: {test_acc:.2f}%, LR: {learning_rates[-1]:.5f}')

print("\n" + "="*70)
print(f"Training abgeschlossen!")
print(f"Beste Accuracy: {best_acc:.2f}%")
print(f"Verbesserung vs V5 (90.70%): {best_acc - 90.70:+.2f}%")
print(f"Ziel (98%): {'✓ ERREICHT!' if best_acc >= 98 else f'Noch {98 - best_acc:.2f}% zu gehen'}")
print("="*70 + "\n")

torch.save(model.state_dict(), "final_model_v6.pth")

# Visualisierung
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Training Loss
axes[0].plot(train_losses, linewidth=2, color='blue')
axes[0].set_title('Training Loss', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].grid(True, alpha=0.3)

# Test Accuracy
axes[1].plot(test_accuracies, linewidth=2, color='green')
axes[1].axhline(y=90.70, color='orange', linestyle='--', label='V5: 90.70%', alpha=0.7)
axes[1].axhline(y=93.27, color='gray', linestyle='--', label='V4: 93.27%', alpha=0.5)
axes[1].axhline(y=98, color='red', linestyle='--', label='Ziel: 98%', alpha=0.7)
axes[1].axhline(y=best_acc, color='darkgreen', linestyle='-', linewidth=2, label=f'V6 Best: {best_acc:.2f}%')
axes[1].set_title('Test Accuracy', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy (%)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Learning Rate
axes[2].plot(learning_rates, linewidth=2, color='purple')
axes[2].set_title('Learning Rate (Cosine Annealing)', fontsize=14, fontweight='bold')
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('Learning Rate')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('v6_results.png', dpi=100)
print("Grafiken gespeichert: v6_results.png")
plt.show()
