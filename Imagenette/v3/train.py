# -*- coding: utf-8 -*-
"""
V3 - Anti-Overfitting CNN for Imagenette
Fixes für das Overfitting-Problem

Hauptverbesserungen:
1. ✅ VALIDATION SET (nicht Test Set während Training!)
2. ✅ Train + Val Accuracy tracking (Train-Val Gap sichtbar)
3. ✅ STARKE Regularisierung gegen Overfitting:
   - Höherer Dropout (0.3 → 0.4)
   - Weight Decay erhöht
   - Label Smoothing erhöht
   - STARKE Data Augmentation
4. ✅ Early Stopping
5. ✅ Learning Rate Scheduler mit ReduceLROnPlateau
6. ✅ Kleineres Modell (weniger Parameter = weniger Overfitting)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
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


script_dir = Path(__file__).parent.parent
data_dir = script_dir / "data"
data_dir.mkdir(exist_ok=True)
os.environ['FASTAI_HOME'] = str(data_dir)


# ============================================================================
# CONFIG - Anti-Overfitting
# ============================================================================
class Config:
    # Data
    batch_size = 32
    num_workers = 2
    data_percentage = 0.40  # 40% der Daten

    # Model - KLEINERES Modell gegen Overfitting
    dropout = 0.4  # ERHÖHT von 0.2 → 0.4

    # Training - STÄRKERE Regularisierung
    epochs = 50
    lr = 0.0005  # Kleinere LR
    weight_decay = 5e-4  # ERHÖHT von 1e-4 → 5e-4
    label_smoothing = 0.15  # ERHÖHT von 0.05 → 0.15

    # Early Stopping
    patience = 10  # Stoppt nach 10 Epochen ohne Verbesserung

    use_amp = False


# ============================================================================
# KLEINERES MODELL (weniger Parameter)
# ============================================================================

class SqueezeExcitation(nn.Module):
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


class SmallerCNN(nn.Module):
    """
    KLEINERES Modell gegen Overfitting
    V2: 64 → 128 → 256 → 512
    V3: 48 → 96  → 192 → 384 (ca. 50% weniger Parameter)
    """
    def __init__(self, num_classes=10, dropout=0.4):
        super().__init__()

        # Stem - KLEINER
        self.stem = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=7, stride=2, padding=3, bias=False),  # 64→48
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
        )

        # Stages - KLEINER
        self.stage1 = nn.Sequential(
            DepthwiseSeparableConv(48, 48),
            ResidualBlock(48, dropout),
            nn.MaxPool2d(2),
        )

        self.stage2 = nn.Sequential(
            DepthwiseSeparableConv(48, 96),  # 128→96
            ResidualBlock(96, dropout),
            nn.MaxPool2d(2),
        )

        self.stage3 = nn.Sequential(
            DepthwiseSeparableConv(96, 192),  # 256→192
            ResidualBlock(192, dropout),
            nn.MaxPool2d(2),
        )

        self.stage4 = nn.Sequential(
            DepthwiseSeparableConv(192, 384),  # 512→384
            ResidualBlock(384, dropout),
        )

        self.gap = nn.AdaptiveAvgPool2d(1)

        # Classifier mit MEHR Dropout
        self.classifier = nn.Sequential(
            nn.Dropout(dropout * 1.5),  # Noch stärker!
            nn.Linear(384, num_classes)
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
# TRAINING FUNCTIONS mit Train+Val Tracking
# ============================================================================
def train_epoch(model, loader, optimizer, criterion, scaler, device, cfg):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Training")
    for data, target in pbar:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        with torch.amp.autocast(device_type=device.type, enabled=cfg.use_amp):
            output = model(data)
            loss = criterion(output, target)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

        pbar.set_postfix({
            'loss': f'{total_loss/(pbar.n+1):.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })

    return total_loss / len(loader), 100. * correct / total


@torch.no_grad()
def val_epoch(model, loader, criterion, device, cfg, desc="Validation"):
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
    set_seed(42)
    cfg = Config()

    device = torch.device("cuda" if torch.cuda.is_available() else
                         "mps" if torch.backends.mps.is_available() else "cpu")

    if device.type == 'cuda':
        cfg.use_amp = True

    print(f"Using device: {device}")
    print(f"Mixed Precision: {cfg.use_amp}")

    # Dataset laden
    print("\nLoading Imagenette dataset...")
    path = untar_data(URLs.IMAGENETTE)

    # Lade komplettes Training Set (ohne Transform)
    full_dataset = datasets.ImageFolder(root=f"{path}/train")
    num_total = len(full_dataset)

    # 40% der Daten auswählen
    num_samples = int(num_total * cfg.data_percentage)
    all_indices = np.random.choice(num_total, num_samples, replace=False)

    # Train-Val Split (80-20)
    num_train = int(0.8 * len(all_indices))

    train_indices = all_indices[:num_train]
    val_indices = all_indices[num_train:]

    # Train Dataset mit Augmentation
    train_dataset_with_aug = datasets.ImageFolder(
        root=f"{path}/train",
        transform=get_train_transforms()
    )
    train_dataset = Subset(train_dataset_with_aug, train_indices)

    # Val Dataset ohne Augmentation
    val_dataset_no_aug = datasets.ImageFolder(
        root=f"{path}/train",
        transform=get_val_transforms()
    )
    val_dataset = Subset(val_dataset_no_aug, val_indices)

    # Test Set (nur für finale Evaluation)
    test_dataset = datasets.ImageFolder(
        root=f"{path}/val",
        transform=get_val_transforms()
    )

    print(f"\n📊 Dataset Split (KORREKT!):")
    print(f"Total Available: {num_total:,}")
    print(f"Using (40%): {num_samples:,}")
    print(f"  ├─ Train: {len(train_dataset):,} (80%)")
    print(f"  └─ Val:   {len(val_dataset):,} (20%)")
    print(f"Test (finale Evaluation): {len(test_dataset):,}")
    print(f"Classes: {full_dataset.classes}")

    use_pin_memory = device.type == 'cuda'

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=use_pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=use_pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=cfg.num_workers, pin_memory=use_pin_memory)

    # KLEINERES Modell
    model = SmallerCNN(dropout=cfg.dropout).to(device)
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # ReduceLROnPlateau statt CosineAnnealing
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    # STÄRKERES Label Smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp) if device.type == 'cuda' else torch.cuda.amp.GradScaler(enabled=False)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"\n🏗️  Model: {num_params:,} parameters (kleiner gegen Overfitting!)")

    print("\n" + "="*70)
    print("V3 - Anti-Overfitting CNN")
    print("="*70)
    print("🛡️  Anti-Overfitting Maßnahmen:")
    print("  1. ✅ Validation Set (80-20 Split)")
    print("  2. ✅ Train + Val Accuracy Tracking")
    print(f"  3. ✅ Höherer Dropout: {cfg.dropout}")
    print(f"  4. ✅ Mehr Weight Decay: {cfg.weight_decay}")
    print(f"  5. ✅ Mehr Label Smoothing: {cfg.label_smoothing}")
    print("  6. ✅ Starke Data Augmentation")
    print(f"  7. ✅ Kleineres Modell: {num_params:,} params")
    print(f"  8. ✅ Early Stopping: patience={cfg.patience}")
    print("  9. ✅ ReduceLROnPlateau Scheduler")
    print("="*70 + "\n")

    # Training
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    train_val_gaps = []
    lrs = []

    best_val_acc = 0
    epochs_without_improvement = 0

    for epoch in range(cfg.epochs):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{cfg.epochs}")
        print('='*70)

        # Training
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion,
                                           scaler, device, cfg)

        # Validation
        val_loss, val_acc = val_epoch(model, val_loader, criterion, device, cfg)

        # Scheduler step
        scheduler.step(val_acc)

        # Tracking
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        gap = train_acc - val_acc
        train_val_gaps.append(gap)
        lrs.append(optimizer.param_groups[0]['lr'])

        status = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            torch.save(model.state_dict(), f'{script_dir}/v3/best_model.pth')
            status = "✓ BEST"
        else:
            epochs_without_improvement += 1

        print(f'\n📊 Epoch {epoch+1} Results:')
        print(f'  Train: Acc={train_acc:.2f}% Loss={train_loss:.4f}')
        print(f'  Val:   Acc={val_acc:.2f}% Loss={val_loss:.4f}')
        print(f'  Gap:   {gap:.2f}% {"⚠️  OVERFITTING!" if gap > 10 else "✅ OK"}')
        print(f'  LR:    {optimizer.param_groups[0]["lr"]:.6f}')
        print(f'  {status}')

        # Early Stopping Check
        if epochs_without_improvement >= cfg.patience:
            print(f"\n🛑 Early Stopping nach {epoch+1} Epochen!")
            print(f"   Keine Verbesserung seit {cfg.patience} Epochen")
            break

    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    print(f"Best Val Accuracy: {best_val_acc:.2f}%")
    print("="*70 + "\n")

    # Finale Test Evaluation
    print("📝 Finale Test Evaluation...")
    model.load_state_dict(torch.load(f'{script_dir}/v3/best_model.pth'))
    test_loss, test_acc = val_epoch(model, test_loader, criterion, device, cfg, desc="Test")

    print(f"\n🎯 FINALE ERGEBNISSE:")
    print(f"  Best Val Acc:  {best_val_acc:.2f}%")
    print(f"  Test Acc:      {test_acc:.2f}%")
    print(f"  Model Params:  {num_params:,}")

    # Visualisierung
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('V3 - Anti-Overfitting Training Analysis', fontsize=16, fontweight='bold')

    # Loss
    axes[0, 0].plot(train_losses, label='Train', linewidth=2)
    axes[0, 0].plot(val_losses, label='Val', linewidth=2)
    axes[0, 0].set_title('Loss Curves')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # Accuracy
    axes[0, 1].plot(train_accs, label='Train', linewidth=2)
    axes[0, 1].plot(val_accs, label='Val', linewidth=2)
    axes[0, 1].axhline(y=best_val_acc, color='r', linestyle='--', label=f'Best Val: {best_val_acc:.2f}%')
    axes[0, 1].set_title('Accuracy Curves')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # Train-Val Gap (WICHTIG!)
    axes[0, 2].plot(train_val_gaps, linewidth=2, color='orange')
    axes[0, 2].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[0, 2].axhline(y=10, color='red', linestyle='--', alpha=0.5, label='Overfitting Threshold (10%)')
    axes[0, 2].set_title('Train-Val Gap (Overfitting Indicator)', fontweight='bold')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Gap (%)')
    axes[0, 2].legend()
    axes[0, 2].grid(alpha=0.3)

    # Learning Rate
    axes[1, 0].plot(lrs, linewidth=2, color='purple')
    axes[1, 0].set_title('Learning Rate Schedule')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(alpha=0.3)

    # Overfitting Status
    epochs_list = list(range(1, len(train_val_gaps) + 1))
    colors = ['green' if gap < 10 else 'red' for gap in train_val_gaps]
    axes[1, 1].bar(epochs_list, train_val_gaps, color=colors, alpha=0.6)
    axes[1, 1].axhline(y=10, color='red', linestyle='--', label='Threshold: 10%')
    axes[1, 1].set_title('Overfitting Status per Epoch')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Train-Val Gap (%)')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    # Summary Stats
    axes[1, 2].axis('off')
    summary_text = f"""
📊 TRAINING SUMMARY

Best Val Accuracy: {best_val_acc:.2f}%
Test Accuracy: {test_acc:.2f}%

Model Parameters: {num_params:,}

Training Setup:
• Data: {len(train_dataset)} train, {len(val_dataset)} val
• Epochs Trained: {len(train_accs)}
• Early Stopping: {'Yes' if len(train_accs) < cfg.epochs else 'No'}

Anti-Overfitting:
• Dropout: {cfg.dropout}
• Weight Decay: {cfg.weight_decay}
• Label Smoothing: {cfg.label_smoothing}
• Data Augmentation: Strong
• Model Size: Smaller

Final Gap: {train_val_gaps[-1]:.2f}%
Status: {"✅ OK" if train_val_gaps[-1] < 10 else "⚠️  Overfitting"}
"""
    axes[1, 2].text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                   verticalalignment='center')

    plt.tight_layout()
    plt.savefig(f'{script_dir}/v3/training_analysis.png', dpi=150)
    print(f"\n💾 Plot saved: v3/training_analysis.png")
    plt.show()

    print("\n" + "="*70)
    print("✅ V3 - Anti-Overfitting Maßnahmen erfolgreich implementiert!")
    print("="*70)


if __name__ == '__main__':
    main()
