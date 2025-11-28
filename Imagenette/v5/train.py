# -*- coding: utf-8 -*-
"""
V5 - Optimized Training Schedule & Hyperparameters
Verbesserung von V4: Smooth Learning Rate Curve + Fine-tuning

Probleme in V4:
- ❌ Learning Rate: "Treppen-Drops" statt glatte Kurve (ReduceLROnPlateau)
- ❌ Plateau nach Epoch 40 (beide Kurven stagnieren)
- ⚠️  Leichtes Overfitting möglich (zu aggressive Regularisierung)

V5 Strategie:
1. ✅ OneCycleLR - GLATTE Learning Rate Kurve (kein "komisches" Treppen-Verhalten)
2. ✅ Längeres Training (100 Epochs statt 60)
3. ✅ Dropout reduziert (0.25 statt 0.3) - Model hat Kapazität für weniger Regularisierung
4. ✅ Early Stopping Patience erhöht (20 statt 15)
5. ✅ Optional: MixUp für bessere Generalisierung

Erwartete Verbesserung: 82.65% → 85-87%
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR  # V5: SMOOTH Learning Rate Curve!
from torchvision import datasets
from torch.utils.data import DataLoader, Subset
from fastai.data.external import URLs, untar_data
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
from tqdm import tqdm

from normalize import get_train_transforms, get_val_transforms, mixup_data, mixup_criterion


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
# CONFIG - V5 Optimized
# ============================================================================
class Config:
    # Data
    batch_size = 32
    num_workers = 2
    data_percentage = 0.40  # BLEIBT bei 40%

    # Model - V5: Weniger aggressive Regularisierung
    dropout = 0.25  # V5: Reduziert von 0.3 (kaum Overfitting in V4!)

    # Training - V5: Längeres Training mit OneCycleLR
    epochs = 100  # V5: Erhöht von 60 (mit smooth LR schedule)
    lr = 0.001  # Max LR für OneCycleLR
    weight_decay = 2e-4  # Bleibt
    label_smoothing = 0.1  # Bleibt

    # OneCycleLR Settings
    pct_start = 0.3  # Warmup für 30% der Epochs
    div_factor = 25.0  # Initial LR = max_lr / 25
    final_div_factor = 1e4  # Final LR = max_lr / 1e4

    # MixUp - V5: Für bessere Generalisierung
    use_mixup = True  # MixUp aktivieren
    mixup_alpha = 0.2  # Beta distribution parameter

    # Early Stopping - V5: Mehr Geduld
    patience = 20  # V5: Erhöht von 15

    use_amp = False


# ============================================================================
# DEEPER ARCHITECTURE - 6 Stages statt 4
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


class DeeperCNN(nn.Module):
    """
    TIEFERES CNN mit 6 Stages
    V3: 4 Stages - 48→96→192→384 (633K params)
    V4: 6 Stages - 48→96→192→256→384→512 (~1M params)
    """
    def __init__(self, num_classes=10, dropout=0.3):
        super().__init__()

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=7, stride=2, padding=3, bias=False),  # 224→112
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),  # 112→56
        )

        # Stage 1: 56x56 - 48 channels + 2 Residual Blocks
        self.stage1 = nn.Sequential(
            DepthwiseSeparableConv(48, 48),
            ResidualBlock(48, dropout),
            ResidualBlock(48, dropout),  # EXTRA Block!
            nn.MaxPool2d(2),  # 56→28
        )

        # Stage 2: 28x28 - 96 channels + 2 Residual Blocks
        self.stage2 = nn.Sequential(
            DepthwiseSeparableConv(48, 96),
            ResidualBlock(96, dropout),
            ResidualBlock(96, dropout),  # EXTRA Block!
            nn.MaxPool2d(2),  # 28→14
        )

        # Stage 3: 14x14 - 192 channels + 2 Residual Blocks
        self.stage3 = nn.Sequential(
            DepthwiseSeparableConv(96, 192),
            ResidualBlock(192, dropout),
            ResidualBlock(192, dropout),  # EXTRA Block!
            nn.MaxPool2d(2),  # 14→7
        )

        # Stage 4: 7x7 - 256 channels (NEU!) + 2 Residual Blocks
        self.stage4 = nn.Sequential(
            DepthwiseSeparableConv(192, 256),
            ResidualBlock(256, dropout),
            ResidualBlock(256, dropout),
        )

        # Stage 5: 7x7 - 384 channels (NEU!) + 2 Residual Blocks
        self.stage5 = nn.Sequential(
            DepthwiseSeparableConv(256, 384),
            ResidualBlock(384, dropout),
            ResidualBlock(384, dropout),
        )

        # Stage 6: 7x7 - 512 channels (NEU!) + 2 Residual Blocks
        self.stage6 = nn.Sequential(
            DepthwiseSeparableConv(384, 512),
            ResidualBlock(512, dropout),
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
        x = self.stage5(x)
        x = self.stage6(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================
def train_epoch(model, loader, optimizer, criterion, scaler, device, cfg, scheduler=None):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Training")
    for data, target in pbar:
        data, target = data.to(device), target.to(device)

        # V5: MixUp Augmentation (optional)
        if cfg.use_mixup:
            data, target_a, target_b, lam = mixup_data(data, target, alpha=cfg.mixup_alpha)

        optimizer.zero_grad()

        with torch.amp.autocast(device_type=device.type, enabled=cfg.use_amp):
            output = model(data)

            # V5: MixUp Loss
            if cfg.use_mixup:
                loss = mixup_criterion(criterion, output, target_a, target_b, lam)
            else:
                loss = criterion(output, target)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        # V5: OneCycleLR step nach jedem Batch (nicht nach Epoch!)
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1)

        # Accuracy Berechnung (bei MixUp approximiert)
        if cfg.use_mixup:
            correct += (lam * pred.eq(target_a).sum().item() +
                       (1 - lam) * pred.eq(target_b).sum().item())
        else:
            correct += pred.eq(target).sum().item()

        total += target_a.size(0) if cfg.use_mixup else target.size(0)

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

    full_dataset = datasets.ImageFolder(root=f"{path}/train")
    num_total = len(full_dataset)

    # 40% der Daten
    num_samples = int(num_total * cfg.data_percentage)
    all_indices = np.random.choice(num_total, num_samples, replace=False)

    # Train-Val Split (80-20)
    num_train = int(0.8 * len(all_indices))

    train_indices = all_indices[:num_train]
    val_indices = all_indices[num_train:]

    # Datasets
    train_dataset_with_aug = datasets.ImageFolder(
        root=f"{path}/train",
        transform=get_train_transforms()
    )
    train_dataset = Subset(train_dataset_with_aug, train_indices)

    val_dataset_no_aug = datasets.ImageFolder(
        root=f"{path}/train",
        transform=get_val_transforms()
    )
    val_dataset = Subset(val_dataset_no_aug, val_indices)

    test_dataset = datasets.ImageFolder(
        root=f"{path}/val",
        transform=get_val_transforms()
    )

    print(f"\n📊 Dataset Split:")
    print(f"Total Available: {num_total:,}")
    print(f"Using (40%): {num_samples:,}")
    print(f"  ├─ Train: {len(train_dataset):,} (80%)")
    print(f"  └─ Val:   {len(val_dataset):,} (20%)")
    print(f"Test: {len(test_dataset):,}")
    print(f"Classes: {full_dataset.classes}")

    use_pin_memory = device.type == 'cuda'

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=use_pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=use_pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=cfg.num_workers, pin_memory=use_pin_memory)

    # TIEFERES Modell (gleich wie V4)
    model = DeeperCNN(dropout=cfg.dropout).to(device)
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # V5: OneCycleLR für SMOOTH Learning Rate Curve (keine "komischen" Treppen mehr!)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=cfg.lr,
        epochs=cfg.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=cfg.pct_start,
        div_factor=cfg.div_factor,
        final_div_factor=cfg.final_div_factor,
        anneal_strategy='cos'  # Cosine annealing für smooth curve
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp) if device.type == 'cuda' else torch.cuda.amp.GradScaler(enabled=False)

    num_params = sum(p.numel() for p in model.parameters())
    v3_params = 633_318
    increase = num_params - v3_params

    print(f"\n🏗️  Model Architecture:")
    print(f"  V3: 4 Stages, {v3_params:,} params → 75% accuracy")
    print(f"  V4: 6 Stages, {num_params:,} params (+{increase:,}, +{100*increase/v3_params:.1f}%)")

    print("\n" + "="*70)
    print("V5 - Optimized Training Schedule & Hyperparameters")
    print("="*70)
    print("🎯 Ziel: Smooth LR Curve + Bessere Convergence (>85%)")
    print("\n📐 Architecture:")
    print("  • Same as V4: 6 Stages, 12 Residual Blocks")
    print("  • Progressive: 48→96→192→256→384→512")
    print(f"  • Parameters: {num_params:,}")
    print("\n⚙️  Training Setup (V5 Changes):")
    print(f"  • Data: 40% ({len(train_dataset):,} images)")
    print(f"  • ✨ LR Scheduler: OneCycleLR (SMOOTH curve!)")
    print(f"  • ✨ Dropout: {cfg.dropout} (von 0.3 reduziert)")
    print(f"  • Weight Decay: {cfg.weight_decay}")
    print(f"  • Label Smoothing: {cfg.label_smoothing}")
    print(f"  • ✨ MixUp: {'Enabled' if cfg.use_mixup else 'Disabled'} (alpha={cfg.mixup_alpha if cfg.use_mixup else 'N/A'})")
    print(f"  • ✨ Epochs: {cfg.epochs} (von 60 erhöht)")
    print(f"  • ✨ Early Stopping: patience={cfg.patience} (von 15 erhöht)")
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

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion,
                                           scaler, device, cfg, scheduler)

        val_loss, val_acc = val_epoch(model, val_loader, criterion, device, cfg)

        # V5: OneCycleLR step ist bereits in train_epoch (nach jedem Batch)

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
            torch.save(model.state_dict(), f'{script_dir}/v5/best_model.pth')
            status = "✓ BEST"
        else:
            epochs_without_improvement += 1

        print(f'\n📊 Epoch {epoch+1} Results:')
        print(f'  Train: Acc={train_acc:.2f}% Loss={train_loss:.4f}')
        print(f'  Val:   Acc={val_acc:.2f}% Loss={val_loss:.4f}')
        print(f'  Gap:   {gap:.2f}% {"⚠️  OVERFITTING!" if gap > 10 else "✅ OK"}')
        print(f'  LR:    {optimizer.param_groups[0]["lr"]:.6f}')
        print(f'  {status}')

        if epochs_without_improvement >= cfg.patience:
            print(f"\n🛑 Early Stopping nach {epoch+1} Epochen!")
            break

    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    print(f"V4 Best: 82.65%")
    print(f"V5 Best: {best_val_acc:.2f}%")
    print(f"Improvement: {best_val_acc - 82.65:+.2f}%")
    print("="*70 + "\n")

    # Finale Test Evaluation
    print("📝 Finale Test Evaluation...")
    model.load_state_dict(torch.load(f'{script_dir}/v5/best_model.pth'))
    test_loss, test_acc = val_epoch(model, test_loader, criterion, device, cfg, desc="Test")

    print(f"\n🎯 FINALE ERGEBNISSE:")
    print(f"  V4 Val:  82.65%")
    print(f"  V5 Val:  {best_val_acc:.2f}% ({best_val_acc-82.65:+.2f}%)")
    print(f"  V5 Test: {test_acc:.2f}%")

    # Visualisierung
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('V5 - Optimized Training Schedule Analysis', fontsize=16, fontweight='bold')

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
    axes[0, 1].axhline(y=82.65, color='gray', linestyle='--', label='V4: 82.65%', alpha=0.5)
    axes[0, 1].axhline(y=best_val_acc, color='r', linestyle='--', label=f'V5: {best_val_acc:.2f}%')
    axes[0, 1].set_title('Accuracy Curves')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # Train-Val Gap
    axes[0, 2].plot(train_val_gaps, linewidth=2, color='orange')
    axes[0, 2].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[0, 2].axhline(y=10, color='red', linestyle='--', alpha=0.5, label='Threshold: 10%')
    axes[0, 2].set_title('Train-Val Gap')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Gap (%)')
    axes[0, 2].legend()
    axes[0, 2].grid(alpha=0.3)

    # Learning Rate - V5: Jetzt SMOOTH CURVE! 🎉
    axes[1, 0].plot(lrs, linewidth=2, color='purple')
    axes[1, 0].set_title('Learning Rate Schedule (V5: OneCycleLR - SMOOTH!)')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(alpha=0.3)

    # Overfitting Status
    epochs_list = list(range(1, len(train_val_gaps) + 1))
    colors = ['green' if gap < 10 else 'red' for gap in train_val_gaps]
    axes[1, 1].bar(epochs_list, train_val_gaps, color=colors, alpha=0.6)
    axes[1, 1].axhline(y=10, color='red', linestyle='--', label='Threshold: 10%')
    axes[1, 1].set_title('Overfitting Status')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Gap (%)')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    # Comparison
    axes[1, 2].axis('off')
    comparison_text = f"""
📊 V4 vs V5 COMPARISON

Architecture:
  V4 & V5: Same (6 Stages, {num_params:,} params)

Key Changes (V5):
  ✅ OneCycleLR (smooth LR curve!)
  ✅ Dropout: 0.3 → 0.25
  ✅ Epochs: 60 → 100
  ✅ Patience: 15 → 20

Performance:
  V4 Val Acc: 82.65%
  V5 Val Acc: {best_val_acc:.2f}%
  Improvement: {best_val_acc-82.65:+.2f}%

  V5 Test Acc: {test_acc:.2f}%

Training:
  Data: Same (40%)
  Epochs: {len(train_accs)}
  Early Stop: {'Yes' if len(train_accs) < cfg.epochs else 'No'}

Final Gap: {train_val_gaps[-1]:.2f}%
Status: {"✅ OK" if train_val_gaps[-1] < 10 else "⚠️  Overfitting"}

Result: {"🎉 SUCCESS!" if best_val_acc > 82.65 else "⚠️  No improvement"}
"""
    axes[1, 2].text(0.1, 0.5, comparison_text, fontsize=11, family='monospace',
                   verticalalignment='center')

    plt.tight_layout()
    plt.savefig(f'{script_dir}/v5/training_analysis.png', dpi=150)
    print(f"\n💾 Plot saved: v5/training_analysis.png")
    plt.show()

    print("\n" + "="*70)
    print("V5 - Optimized Training Schedule")
    print("="*70)
    print(f"✅ OneCycleLR: SMOOTH Learning Rate Curve (kein Treppen-Verhalten!)")
    print(f"✅ Dropout: 0.3 → 0.25 (weniger aggressive Regularisierung)")
    print(f"✅ Training: 60 → 100 Epochs mit Patience 20")
    print(f"✅ Best Val Accuracy: {best_val_acc:.2f}% ({best_val_acc-82.65:+.2f}% vs V4)")
    print(f"✅ Test Accuracy: {test_acc:.2f}%")
    print("="*70)


if __name__ == '__main__':
    main()
