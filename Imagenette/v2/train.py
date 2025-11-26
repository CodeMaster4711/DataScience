# -*- coding: utf-8 -*-
"""
V2 - Optimized Modern CNN for Imagenette
Based on V12 Fashion-MNIST architecture, adapted for Imagenette

Key features:
- Modern CNN architecture with SE blocks and Depthwise Separable Convs
- 25% of training data
- Optimized for RGB images (224x224x3)
- Label Smoothing and Dropout regularization
- Cosine Annealing LR schedule with warmup
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets
from torch.utils.data import DataLoader, Subset
from fastai.data.external import URLs, untar_data
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
from tqdm import tqdm

from normalize import get_train_transforms, get_val_transforms


# ============================================================================
# SETUP
# ============================================================================
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Globaler Daten-Ordner
script_dir = Path(__file__).parent.parent
data_dir = script_dir / "data"
data_dir.mkdir(exist_ok=True)
os.environ['FASTAI_HOME'] = str(data_dir)


# ============================================================================
# CONFIG
# ============================================================================
class Config:
    # Data
    batch_size = 32  # Smaller for larger images
    num_workers = 2
    data_percentage = 0.25  # 25% of data

    # Model
    dropout = 0.2

    # Training
    epochs = 40
    lr = 0.001
    weight_decay = 1e-4
    label_smoothing = 0.05
    warmup_epochs = 5

    use_amp = False


# ============================================================================
# ARCHITECTURE
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
    """Depthwise Separable Convolution"""
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
    """Residual Block with SE"""
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
    """
    Modern CNN for Imagenette (224x224x3 → 10 classes)
    Architecture: 64 → 128 → 256 → 512
    """
    def __init__(self, num_classes=10, dropout=0.2):
        super().__init__()

        # Stem - 3 input channels for RGB
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),  # 224→112
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),  # 112→56
        )

        # Stage 1: 56x56
        self.stage1 = nn.Sequential(
            DepthwiseSeparableConv(64, 64),
            ResidualBlock(64, dropout),
            nn.MaxPool2d(2),  # 56→28
        )

        # Stage 2: 28x28
        self.stage2 = nn.Sequential(
            DepthwiseSeparableConv(64, 128),
            ResidualBlock(128, dropout),
            nn.MaxPool2d(2),  # 28→14
        )

        # Stage 3: 14x14
        self.stage3 = nn.Sequential(
            DepthwiseSeparableConv(128, 256),
            ResidualBlock(256, dropout),
            nn.MaxPool2d(2),  # 14→7
        )

        # Stage 4: 7x7
        self.stage4 = nn.Sequential(
            DepthwiseSeparableConv(256, 512),
            ResidualBlock(512, dropout),
        )

        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
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
        x = self.stage4(x)
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

    pbar = tqdm(loader, desc=f"Training Epoch {epoch+1}")
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        # Mixed Precision
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

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{total_loss/(batch_idx+1):.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })

    return total_loss / len(loader), 100. * correct / total


@torch.no_grad()
def test_epoch(model, loader, criterion, device, cfg, desc="Validation"):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=desc)
    for data, target in pbar:
        data, target = data.to(device), target.to(device)

        output = model(data)
        loss = criterion(output, target)

        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

        pbar.set_postfix({
            'loss': f'{total_loss/(pbar.n+1):.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })

    return total_loss / len(loader), 100. * correct / total


def main():
    set_seed(42)
    cfg = Config()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else
                         "mps" if torch.backends.mps.is_available() else "cpu")

    if device.type == 'cuda':
        cfg.use_amp = True

    print(f"Using device: {device}")
    print(f"Mixed Precision: {cfg.use_amp}")

    # Dataset laden
    print("\nLoading Imagenette dataset...")
    path = untar_data(URLs.IMAGENETTE)

    # Datasets mit Transformationen
    train_dataset_full = datasets.ImageFolder(
        root=f"{path}/train",
        transform=get_train_transforms()
    )

    val_dataset = datasets.ImageFolder(
        root=f"{path}/val",
        transform=get_val_transforms()
    )

    # 25% der Trainingsdaten auswählen
    num_train = len(train_dataset_full)
    num_samples = int(num_train * cfg.data_percentage)
    indices = np.random.choice(num_train, num_samples, replace=False)
    train_dataset = Subset(train_dataset_full, indices)

    print(f"\nDataset Info:")
    print(f"Total Training Images: {num_train}")
    print(f"Using Images (25%): {len(train_dataset)}")
    print(f"Validation Images: {len(val_dataset)}")
    print(f"Classes: {train_dataset_full.classes}")

    use_pin_memory = device.type == 'cuda'

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=use_pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=use_pin_memory)

    # Model
    model = ModernCNN(dropout=cfg.dropout).to(device)
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # LR Schedule
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs - cfg.warmup_epochs)

    # Loss with Label Smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    # Scaler for mixed precision
    if device.type == 'cuda' and cfg.use_amp:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=False)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel Parameters: {num_params:,}")

    # Training
    print("\n" + "="*70)
    print("V2 - Modern CNN for Imagenette")
    print("="*70)
    print("Architecture:")
    print("  • Input: 224x224x3 RGB images")
    print("  • Stages: 64 → 128 → 256 → 512 channels")
    print("  • SE Blocks + Depthwise Separable Convolutions")
    print("  • Global Average Pooling → 10 classes")
    print(f"  • Parameters: {num_params:,}")
    print("\nTraining Setup:")
    print(f"  • Data: 25% ({len(train_dataset)} images)")
    print(f"  • Epochs: {cfg.epochs}")
    print(f"  • Batch Size: {cfg.batch_size}")
    print(f"  • Learning Rate: {cfg.lr} (Cosine Annealing + Warmup)")
    print(f"  • Regularization: Dropout {cfg.dropout}, Label Smoothing {cfg.label_smoothing}")
    print("="*70 + "\n")

    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    best_acc = 0

    for epoch in range(cfg.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion,
                                           scaler, device, cfg, epoch)
        val_loss, val_acc = test_epoch(model, val_loader, criterion, device, cfg)

        if epoch >= cfg.warmup_epochs:
            scheduler.step()

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        lr = optimizer.param_groups[0]['lr']
        status = ""
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f'{script_dir}/v2/best_model.pth')
            status = "✓ BEST"

        print(f'\nEpoch {epoch+1:2d}/{cfg.epochs} Summary:')
        print(f'  Train: {train_acc:.2f}% Loss: {train_loss:.4f}')
        print(f'  Val:   {val_acc:.2f}% Loss: {val_loss:.4f}')
        print(f'  LR: {lr:.6f} {status}\n')

    print("\n" + "="*70)
    print(f"Training Complete!")
    print(f"Best Validation Accuracy: {best_acc:.2f}%")
    print("="*70 + "\n")

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Loss curves
    axes[0, 0].plot(train_losses, label='Train', linewidth=2)
    axes[0, 0].plot(val_losses, label='Val', linewidth=2)
    axes[0, 0].set_title('Loss Curves', fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # Accuracy curves
    axes[0, 1].plot(train_accs, label='Train', linewidth=2)
    axes[0, 1].plot(val_accs, label='Val', linewidth=2)
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

    # Train-Val Gap
    gaps = [train_accs[i] - val_accs[i] for i in range(len(train_accs))]
    axes[1, 1].plot(gaps, linewidth=2, color='orange')
    axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[1, 1].set_title('Train-Val Gap (Overfitting Indicator)', fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Gap (%)')
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{script_dir}/v2/training_results.png', dpi=150)
    print("Plot saved: v2/training_results.png\n")
    plt.show()

    # Final Results
    model.load_state_dict(torch.load(f'{script_dir}/v2/best_model.pth'))
    _, final_acc = test_epoch(model, val_loader, criterion, device, cfg, desc="Final Evaluation")

    print("\n" + "="*70)
    print("FINAL RESULTS - V2")
    print("="*70)
    print(f"Final Val Accuracy:  {final_acc:.2f}%")
    print(f"Best Val Accuracy:   {best_acc:.2f}%")
    print(f"Model Parameters:    {num_params:,}")
    print(f"Training Data Used:  {len(train_dataset)} images (25%)")
    print("\nModel Details:")
    print(f"  • Architecture: Modern CNN with SE blocks")
    print(f"  • Input Size: 224x224x3")
    print(f"  • Output Classes: 10 (Imagenette)")
    print(f"  • Regularization: Dropout {cfg.dropout}, Label Smoothing {cfg.label_smoothing}")
    print(f"  • Data Augmentation: Flip, Rotation, ColorJitter, RandomErasing")
    print("="*70)


if __name__ == '__main__':
    main()
