"""
V11 - Modernes, vereinfachtes CNN für Fashion-MNIST
Fokus: Einfachheit + Moderne PyTorch Best Practices + Höhere Accuracy

Neue Features:
- Depthwise Separable Convolutions (effizienter)
- Squeeze-and-Excitation Blocks (besser)
- Residual Connections (tieferes Training)
- Label Smoothing (bessere Generalisierung)
- Cosine Annealing mit Warmup (optimale LR)
- Mixed Precision Training (schneller)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np


# ============================================================================
# SETUP: Reproduzierbarkeit
# ============================================================================
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================================
# CONFIG
# ============================================================================
class Config:
    # Data
    batch_size = 64
    num_workers = 2  # Jetzt funktioniert Multiprocessing!

    # Model
    dropout = 0.3

    # Training
    epochs = 60
    lr = 0.001
    weight_decay = 1e-4
    label_smoothing = 0.1
    warmup_epochs = 5

    # Mixed Precision (wird später je nach Device gesetzt)
    use_amp = False


# ============================================================================
# MODERN CNN ARCHITECTURE
# ============================================================================

class SqueezeExcitation(nn.Module):
    """SE-Block: Adaptive Feature Recalibration"""
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        b, c, _, _ = x.size()
        squeeze = F.adaptive_avg_pool2d(x, 1).view(b, c)
        excitation = self.fc1(squeeze)
        excitation = F.relu(excitation)
        excitation = self.fc2(excitation)
        excitation = torch.sigmoid(excitation).view(b, c, 1, 1)
        return x * excitation


class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Convolution - effizienter"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size,
                                   stride, padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return F.relu(x)


class ResidualBlock(nn.Module):
    """Residual Block mit SE"""
    def __init__(self, channels, dropout):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv(channels, channels)
        self.conv2 = DepthwiseSeparableConv(channels, channels)
        self.se = SqueezeExcitation(channels)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.se(out)
        out = out + residual
        return F.relu(out)


class ModernCNN(nn.Module):
    """Modernes CNN mit Best Practices"""
    def __init__(self, num_classes=10, dropout=0.3):
        super().__init__()

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        # Stages
        self.stage1 = nn.Sequential(
            DepthwiseSeparableConv(64, 64),
            ResidualBlock(64, dropout),
            nn.MaxPool2d(2),
        )

        self.stage2 = nn.Sequential(
            DepthwiseSeparableConv(64, 128),
            ResidualBlock(128, dropout),
            nn.MaxPool2d(2),
        )

        self.stage3 = nn.Sequential(
            DepthwiseSeparableConv(128, 256),
            ResidualBlock(256, dropout),
        )

        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================
def train_epoch(model, loader, optimizer, criterion, scaler, device, cfg, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    # Warmup
    if epoch < cfg.warmup_epochs:
        warmup_lr = cfg.lr * (epoch + 1) / cfg.warmup_epochs
        for param_group in optimizer.param_groups:
            param_group['lr'] = warmup_lr

    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        # Mixed Precision Training
        with torch.amp.autocast(device_type=device.type, enabled=cfg.use_amp):
            output = model(data)
            loss = criterion(output, target)

        scaler.scale(loss).backward()

        # Gradient Clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

    return total_loss / len(loader), 100. * correct / total


@torch.no_grad()
def test_epoch(model, loader, criterion, device, cfg):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    for data, target in loader:
        data, target = data.to(device), target.to(device)

        output = model(data)
        loss = criterion(output, target)

        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

    return total_loss / len(loader), 100. * correct / total


def main():
    # Setup
    set_seed(42)
    cfg = Config()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else
                         "mps" if torch.backends.mps.is_available() else "cpu")

    # Mixed Precision nur für CUDA
    if device.type == 'cuda':
        cfg.use_amp = True

    print(f"Using device: {device}")
    print(f"Mixed Precision: {cfg.use_amp}")

    # Data
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])

    train_dataset = datasets.FashionMNIST('./data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.FashionMNIST('./data', train=False, transform=test_transform)

    # pin_memory nur für CUDA
    use_pin_memory = device.type == 'cuda'

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=use_pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=cfg.num_workers, pin_memory=use_pin_memory)

    # Model
    model = ModernCNN(dropout=cfg.dropout).to(device)
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs - cfg.warmup_epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    # Scaler für Mixed Precision
    if device.type == 'cuda' and cfg.use_amp:
        scaler = torch.cuda.amp.GradScaler()
    else:
        # Für MPS/CPU: dummy scaler der nichts macht
        scaler = torch.cuda.amp.GradScaler(enabled=False)

    print(f"\nModel Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training
    print("\n" + "="*70)
    print("Starting Training - V11 Modern CNN")
    print("="*70)

    train_losses, train_accs = [], []
    test_losses, test_accs = [], []
    best_acc = 0

    for epoch in range(cfg.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion,
                                           scaler, device, cfg, epoch)
        test_loss, test_acc = test_epoch(model, test_loader, criterion, device, cfg)

        # Scheduler step
        if epoch >= cfg.warmup_epochs:
            scheduler.step()

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        # Logging
        lr = optimizer.param_groups[0]['lr']
        status = ""
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'best_model_v11.pth')
            status = "✓ BEST"

        print(f'Epoch {epoch+1:2d}/{cfg.epochs} | '
              f'Train: {train_acc:.2f}% Loss: {train_loss:.4f} | '
              f'Test: {test_acc:.2f}% Loss: {test_loss:.4f} | '
              f'LR: {lr:.6f} {status}')

    print("\n" + "="*70)
    print(f"Training Complete!")
    print(f"Best Test Accuracy: {best_acc:.2f}%")
    print("="*70 + "\n")

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Loss curves
    axes[0, 0].plot(train_losses, label='Train', linewidth=2)
    axes[0, 0].plot(test_losses, label='Test', linewidth=2)
    axes[0, 0].set_title('Loss Curves', fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # Accuracy curves
    axes[0, 1].plot(train_accs, label='Train', linewidth=2)
    axes[0, 1].plot(test_accs, label='Test', linewidth=2)
    axes[0, 1].axhline(y=best_acc, color='r', linestyle='--', label=f'Best: {best_acc:.2f}%')
    axes[0, 1].set_title('Accuracy Curves', fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # Learning Rate Schedule
    lrs = []
    temp_optimizer = AdamW(model.parameters(), lr=cfg.lr)
    temp_scheduler = CosineAnnealingLR(temp_optimizer, T_max=cfg.epochs - cfg.warmup_epochs)
    for e in range(cfg.epochs):
        if e < cfg.warmup_epochs:
            lrs.append(cfg.lr * (e + 1) / cfg.warmup_epochs)
        else:
            lrs.append(temp_optimizer.param_groups[0]['lr'])
            temp_scheduler.step()

    axes[1, 0].plot(lrs, linewidth=2, color='purple')
    axes[1, 0].set_title('Learning Rate Schedule', fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].grid(alpha=0.3)

    # Train-Test Gap
    gaps = [train_accs[i] - test_accs[i] for i in range(len(train_accs))]
    axes[1, 1].plot(gaps, linewidth=2, color='orange')
    axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[1, 1].set_title('Train-Test Gap (Overfitting Indicator)', fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Gap (%)')
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('v11_training_results.png', dpi=150)
    print("Plot saved: v11_training_results.png\n")
    plt.show()

    # Final Results
    model.load_state_dict(torch.load('best_model_v11.pth'))
    _, final_acc = test_epoch(model, test_loader, criterion, device, cfg)

    print("="*70)
    print("FINAL RESULTS - V11")
    print("="*70)
    print(f"Final Test Accuracy: {final_acc:.2f}%")
    print(f"Best Test Accuracy:  {best_acc:.2f}%")
    print(f"Model Parameters:    {sum(p.numel() for p in model.parameters()):,}")
    print("\nModern Features Used:")
    print("  ✓ Depthwise Separable Convolutions")
    print("  ✓ Squeeze-and-Excitation Blocks")
    print("  ✓ Residual Connections")
    print("  ✓ Global Average Pooling")
    print("  ✓ Label Smoothing")
    print("  ✓ Cosine Annealing + Warmup")
    print("  ✓ Gradient Clipping")
    if cfg.use_amp:
        print("  ✓ Mixed Precision Training")
    print("="*70)


if __name__ == '__main__':
    main()
