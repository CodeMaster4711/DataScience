import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import time

# Konfiguration - Schneller & Genauer
BATCH_SIZE = 256  # Größer für schnelleres Training
LEARNING_RATE = 0.1
NUM_EPOCHS = 15  # Weniger als V6!
WEIGHT_DECAY = 1e-4

# RandAugment - Bessere Augmentation als V6
class RandAugment:
    def __init__(self, n=2, m=9):
        self.n = n  # Anzahl Augmentationen
        self.m = m  # Magnitude

    def __call__(self, img):
        ops = [
            lambda img: transforms.functional.rotate(img, self.m * 3),
            lambda img: transforms.functional.affine(img, 0, (self.m/10, 0), 1, 0),
            lambda img: transforms.functional.affine(img, 0, (0, self.m/10), 1, 0),
        ]
        for _ in range(self.n):
            op = np.random.choice(ops)
            img = op(img)
        return img

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    RandAugment(n=2, m=9),
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])

train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=train_transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# Squeeze-and-Excitation Block (Attention Mechanism)
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
        # Squeeze: Global Information
        y = self.squeeze(x).view(b, c)
        # Excitation: Adaptive Recalibration
        y = self.excitation(y).view(b, c, 1, 1)
        # Scale
        return x * y.expand_as(x)


# Efficient ResBlock mit SE-Attention
class EfficientResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_se=True):
        super(EfficientResBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # SE-Attention
        self.se = SEBlock(out_channels) if use_se else nn.Identity()

        # Shortcut
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)  # Attention!
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# V7: Efficient + Accurate CNN
class EfficientCNN(nn.Module):
    def __init__(self):
        super(EfficientCNN, self).__init__()

        # Stem
        self.conv1 = nn.Conv2d(1, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Efficient Blocks mit SE
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 3, stride=2)  # 3 blocks
        self.layer3 = self._make_layer(128, 256, 3, stride=2)  # 3 blocks
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

        # Head
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


model = EfficientCNN()

# GPU Setup
if torch.cuda.is_available():
    device = torch.device('cuda')
    print("Training auf: CUDA GPU")
elif torch.backends.mps.is_available():
    device = torch.device('mps')
    print("Training auf: Apple Silicon GPU (MPS)")
else:
    device = torch.device('cpu')
    print("Training auf: CPU")

model = model.to(device)

# Mixed Precision für 2x Speedup
use_amp = device.type in ['cuda', 'mps']
scaler = torch.amp.GradScaler('cuda' if device.type == 'cuda' else 'cpu') if use_amp else None

# Optimierte Loss mit Label Smoothing
class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / 9)
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))

criterion = LabelSmoothingLoss(smoothing=0.1)
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9,
                      weight_decay=WEIGHT_DECAY, nesterov=True)

# OneCycle für schnellste Konvergenz
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=LEARNING_RATE, epochs=NUM_EPOCHS,
    steps_per_epoch=len(train_loader), pct_start=0.3
)

print("\n" + "="*75)
print("V7 - SCHNELLER & GENAUER (Ziel: 96%+ in 15 Epochen)")
print("="*75)
print(f"Modell Parameter: {sum(p.numel() for p in model.parameters()):,}")
print(f"Batch Size: {BATCH_SIZE} (größer = schneller)")
print(f"Epochen: {NUM_EPOCHS} (weniger als V6!)")
print(f"Device: {device}")
print(f"Mixed Precision: {'✓ AN (2x schneller)' if use_amp else '✗ AUS'}")
print("="*75)
print("Neue Features:")
print("✓ Squeeze-Excitation Attention (lernt wichtige Features)")
print("✓ RandAugment (bessere Augmentation)")
print("✓ Mixed Precision Training (2x Speed)")
print("✓ Größere Batches (schneller)")
print("✓ Optimierte Architektur (10 statt 8 Blocks)")
print("✓ OneCycle LR (schnellste Konvergenz)")
print("="*75 + "\n")

train_losses = []
test_accuracies = []
epoch_times = []
best_acc = 0

start_total = time.time()

for epoch in range(NUM_EPOCHS):
    epoch_start = time.time()

    # Training
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        # Mixed Precision Training
        if use_amp and scaler:
            with torch.amp.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)

    # Testing
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            if use_amp:
                with torch.amp.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
                    outputs = model(images)
            else:
                outputs = model(images)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_acc = 100 * correct / total
    test_accuracies.append(test_acc)

    epoch_time = time.time() - epoch_start
    epoch_times.append(epoch_time)

    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), "best_model_v7.pth")
        print(f'[{epoch+1}/{NUM_EPOCHS}] Loss: {avg_loss:.4f}, Acc: {test_acc:.2f}%, Time: {epoch_time:.1f}s ✓ BEST')
    else:
        print(f'[{epoch+1}/{NUM_EPOCHS}] Loss: {avg_loss:.4f}, Acc: {test_acc:.2f}%, Time: {epoch_time:.1f}s')

total_time = time.time() - start_total

print("\n" + "="*75)
print(f"Training abgeschlossen in {total_time/60:.1f} Minuten!")
print(f"Durchschnittliche Zeit pro Epoche: {np.mean(epoch_times):.1f}s")
print(f"Beste Accuracy: {best_acc:.2f}%")
print(f"Verbesserung vs V6 (94.39%): {best_acc - 94.39:+.2f}%")
print(f"Fortschritt zu 98%: {best_acc:.2f}% / 98% ({(best_acc/98)*100:.1f}%)")
print("="*75 + "\n")

torch.save(model.state_dict(), "final_model_v7.pth")

# Visualisierung
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Loss
axes[0].plot(train_losses, 'b-', linewidth=2)
axes[0].set_title('Training Loss', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].grid(True, alpha=0.3)

# Accuracy
axes[1].plot(test_accuracies, 'g-', linewidth=2, label=f'V7: {best_acc:.2f}%')
axes[1].axhline(y=94.39, color='purple', linestyle='--', label='V6: 94.39%', alpha=0.7)
axes[1].axhline(y=90.70, color='orange', linestyle='--', label='V5: 90.70%', alpha=0.5)
axes[1].axhline(y=98, color='r', linestyle='--', label='Ziel: 98%', alpha=0.7)
axes[1].set_title('Test Accuracy', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy (%)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Training Speed
axes[2].bar(range(len(epoch_times)), epoch_times, color='teal', alpha=0.7)
axes[2].axhline(y=np.mean(epoch_times), color='r', linestyle='--',
                label=f'Avg: {np.mean(epoch_times):.1f}s')
axes[2].set_title('Training Speed', fontsize=14, fontweight='bold')
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('Zeit (Sekunden)')
axes[2].legend()
axes[2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('v7_results.png', dpi=100)
print("Grafiken: v7_results.png")
plt.show()
