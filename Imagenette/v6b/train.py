# -*- coding: utf-8 -*-
"""
V6b - DEFAULT Initialization + Weights & Biases Integration (BASELINE)
Vergleich zu V6a: Gleiche Architektur OHNE spezielle Initialisierung

V6b Strategie (OHNE korrekte Initialisierung - Baseline):
1. ❌ DEFAULT Final Layer Init (wie V5)
2. ✅ Initialization Sanity Check (verify loss @ init für Vergleich)
3. ✅ Weights & Biases vollständiges Tracking
4. ✅ Gradient Visualisierung (wandb.watch)
5. ✅ Per-Batch und Per-Epoch Logging
6. ✅ Same architecture as V5 (DeeperCNN, OneCycleLR, MixUp)

Zweck: Baseline zum Vergleich mit V6a - zeigt Impact der korrekten Init
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torchvision import datasets
from torch.utils.data import DataLoader, Subset
from fastai.data.external import URLs, untar_data
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
from tqdm import tqdm
import json
import wandb  # V6: Weights & Biases Integration!

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

# W&B: Connect to LOCAL server instead of wandb.ai
os.environ['WANDB_BASE_URL'] = 'http://localhost:8080'


# ============================================================================
# CONFIG - V6b (BASELINE)
# ============================================================================
class Config:
    # Data
    batch_size = 32
    num_workers = 2
    data_percentage = 0.40

    # Model - Same as V5
    dropout = 0.25
    num_classes = 10

    # Training - Same as V5
    epochs = 100
    lr = 0.001
    weight_decay = 2e-4
    label_smoothing = 0.1

    # OneCycleLR - Same as V5
    pct_start = 0.3
    div_factor = 25.0
    final_div_factor = 1e4

    # MixUp - Same as V5
    use_mixup = True
    mixup_alpha = 0.2

    # Early Stopping - Same as V5
    patience = 20

    # System
    use_amp = False
    seed = 42

    # V6b Specific
    correct_init = False  # v6b: FALSE (default initialization - BASELINE)
    version = "v6b"

    # W&B - Local Server
    wandb_project = "imagenette-training"
    wandb_entity = "codemaster4711"  # Your local W&B server username


# ============================================================================
# ARCHITECTURE - Same as V5
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
    V6a: Same architecture as V5 but with CORRECT INITIALIZATION
    6 Stages - 48→96→192→256→384→512
    """
    def __init__(self, num_classes=10, dropout=0.25):
        super().__init__()

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
        )

        # Stage 1: 56x56 - 48 channels
        self.stage1 = nn.Sequential(
            DepthwiseSeparableConv(48, 48),
            ResidualBlock(48, dropout),
            ResidualBlock(48, dropout),
            nn.MaxPool2d(2),
        )

        # Stage 2: 28x28 - 96 channels
        self.stage2 = nn.Sequential(
            DepthwiseSeparableConv(48, 96),
            ResidualBlock(96, dropout),
            ResidualBlock(96, dropout),
            nn.MaxPool2d(2),
        )

        # Stage 3: 14x14 - 192 channels
        self.stage3 = nn.Sequential(
            DepthwiseSeparableConv(96, 192),
            ResidualBlock(192, dropout),
            ResidualBlock(192, dropout),
            nn.MaxPool2d(2),
        )

        # Stage 4: 7x7 - 256 channels
        self.stage4 = nn.Sequential(
            DepthwiseSeparableConv(192, 256),
            ResidualBlock(256, dropout),
            ResidualBlock(256, dropout),
        )

        # Stage 5: 7x7 - 384 channels
        self.stage5 = nn.Sequential(
            DepthwiseSeparableConv(256, 384),
            ResidualBlock(384, dropout),
            ResidualBlock(384, dropout),
        )

        # Stage 6: 7x7 - 512 channels
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
        """
        V6b: DEFAULT INITIALIZATION (Baseline - suboptimal for initial loss)

        This uses standard initialization WITHOUT optimization for initial loss:
        - Weights: Normal with std=0.1 (10x larger than v6a)
        - Bias: Random small values instead of zero
        - Result: Initial loss will be HIGHER than theoretical -log(1/10)

        Purpose: Baseline to compare against v6a's correct initialization
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # V6b: DEFAULT INIT - Larger weights, random bias (suboptimal!)
                nn.init.normal_(m.weight, 0, 0.1)   # Larger std → worse initial loss
                nn.init.normal_(m.bias, 0, 0.01)    # Random bias → not uniform output

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
# INITIALIZATION SANITY CHECK - V6a NEW!
# ============================================================================
def check_initialization(model, loader, criterion, device, num_classes=10):
    """
    Verify that initialization gives correct initial loss.

    For balanced classification with n classes:
    - Expected loss: -log(1/n) = log(n)
    - Expected accuracy: 100/n %
    """
    print("\n" + "="*70)
    print("INITIALIZATION SANITY CHECK")
    print("="*70)

    model.eval()
    with torch.no_grad():
        # Evaluate on first batch
        data, target = next(iter(loader))
        data, target = data.to(device), target.to(device)

        output = model(data)
        init_loss = criterion(output, target).item()
        init_pred = output.argmax(dim=1)
        init_acc = 100. * init_pred.eq(target).sum().item() / target.size(0)

        # Check softmax probabilities
        probs = F.softmax(output, dim=1)
        avg_probs = probs.mean(dim=0)

    expected_loss = np.log(num_classes)
    expected_acc = 100.0 / num_classes
    expected_prob = 1.0 / num_classes

    print(f"Number of classes: {num_classes}")
    print(f"\nLoss:")
    print(f"  Measured:  {init_loss:.4f}")
    print(f"  Expected:  {expected_loss:.4f} (-log(1/{num_classes}))")
    print(f"  Difference: {abs(init_loss - expected_loss):.4f}")

    print(f"\nAccuracy:")
    print(f"  Measured:  {init_acc:.2f}%")
    print(f"  Expected:  ~{expected_acc:.2f}%")

    print(f"\nAverage Probabilities per Class:")
    for i, prob in enumerate(avg_probs):
        print(f"  Class {i}: {prob.item():.4f} (expected: {expected_prob:.4f})")

    # Verdict
    loss_ok = abs(init_loss - expected_loss) < 0.5
    prob_ok = all(abs(p.item() - expected_prob) < 0.05 for p in avg_probs)

    print(f"\n{'='*70}")
    if loss_ok and prob_ok:
        print("✅ INITIALIZATION CORRECT!")
        print("   Model is ready for training.")
    else:
        print("⚠️  WARNING: Initialization may be suboptimal!")
        if not loss_ok:
            print("   - Initial loss differs from theoretical value")
        if not prob_ok:
            print("   - Softmax outputs are not uniform")
        print("   This may indicate a bug in the model or loss function.")
    print("="*70 + "\n")

    model.train()

    return {
        "init_loss": init_loss,
        "init_accuracy": init_acc,
        "expected_loss": expected_loss,
        "expected_accuracy": expected_acc,
        "avg_probs": avg_probs.cpu().numpy().tolist()
    }


# ============================================================================
# TRAINING FUNCTIONS WITH W&B LOGGING
# ============================================================================
def train_epoch(model, loader, optimizer, criterion, scaler, device, cfg,
                scheduler=None, epoch=0, global_step=0):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Training")
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)

        # MixUp
        if cfg.use_mixup:
            data, target_a, target_b, lam = mixup_data(data, target, alpha=cfg.mixup_alpha)

        optimizer.zero_grad()

        with torch.amp.autocast(device_type=device.type, enabled=cfg.use_amp):
            output = model(data)
            if cfg.use_mixup:
                loss = mixup_criterion(criterion, output, target_a, target_b, lam)
            else:
                loss = criterion(output, target)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        # Calculate gradient norm
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

        # Step scheduler
        if scheduler is not None:
            current_lr = scheduler.get_last_lr()[0]
            scheduler.step()
        else:
            current_lr = optimizer.param_groups[0]['lr']

        total_loss += loss.item()
        pred = output.argmax(dim=1)

        if cfg.use_mixup:
            correct += (lam * pred.eq(target_a).sum().item() +
                       (1 - lam) * pred.eq(target_b).sum().item())
            total += target_a.size(0)
        else:
            correct += pred.eq(target).sum().item()
            total += target.size(0)

        # V6: Log per-batch metrics to W&B
        if batch_idx % 10 == 0:
            wandb.log({
                "train/loss_batch": loss.item(),
                "learning_rate": current_lr,
                "train/grad_norm": grad_norm.item(),
                "batch": global_step + batch_idx,
            })

        pbar.set_postfix({
            'loss': f'{total_loss/(pbar.n+1):.4f}',
            'acc': f'{100.*correct/total:.2f}%',
            'lr': f'{current_lr:.6f}'
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

    # V6: Initialize W&B - Connected to LOCAL server (http://localhost:8080)
    run_name = f"{cfg.version}-default-init-baseline"
    wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        name=run_name,
        config={
            "version": cfg.version,
            "architecture": "DeeperCNN-6stages",
            "dataset": "Imagenette",
            "data_percentage": cfg.data_percentage,
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "optimizer": "AdamW",
            "lr": cfg.lr,
            "weight_decay": cfg.weight_decay,
            "dropout": cfg.dropout,
            "label_smoothing": cfg.label_smoothing,
            "scheduler": "OneCycleLR",
            "pct_start": cfg.pct_start,
            "div_factor": cfg.div_factor,
            "final_div_factor": cfg.final_div_factor,
            "mixup_alpha": cfg.mixup_alpha,
            "use_mixup": cfg.use_mixup,
            "patience": cfg.patience,
            "correct_init": cfg.correct_init,
            "seed": cfg.seed,
        }
    )

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

    # Model
    model = DeeperCNN(num_classes=cfg.num_classes, dropout=cfg.dropout).to(device)
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    scheduler = OneCycleLR(
        optimizer,
        max_lr=cfg.lr,
        epochs=cfg.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=cfg.pct_start,
        div_factor=cfg.div_factor,
        final_div_factor=cfg.final_div_factor,
        anneal_strategy='cos'
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp) if device.type == 'cuda' else torch.cuda.amp.GradScaler(enabled=False)

    num_params = sum(p.numel() for p in model.parameters())

    # Update W&B config
    wandb.config.update({"num_params": num_params})

    # V6: Watch model for gradient tracking
    wandb.watch(model, log="all", log_freq=100, log_graph=True)

    print(f"\n🏗️  Model Architecture:")
    print(f"  Parameters: {num_params:,}")

    print("\n" + "="*70)
    print("V6b - DEFAULT Initialization + W&B Tracking (BASELINE)")
    print("="*70)
    print("🎯 Ziel: Baseline zum Vergleich mit V6a (ohne optimierte Init)")
    print("\n📐 Architecture:")
    print("  • Same as V5: 6 Stages, 12 Residual Blocks")
    print("  • Progressive: 48→96→192→256→384→512")
    print(f"  • Parameters: {num_params:,}")
    print("\n⚙️  V6b Features:")
    print(f"  • ❌ DEFAULT Init: Larger weights (std=0.1), random bias")
    print(f"  • ⚠️  Expected Init Loss: HIGHER than -log(1/10)")
    print(f"  • ✅ W&B Tracking: All metrics, gradients, plots")
    print(f"  • ✅ Gradient visualization via wandb.watch()")
    print("\n⚙️  Training Setup:")
    print(f"  • Data: 40% ({len(train_dataset):,} images)")
    print(f"  • LR Scheduler: OneCycleLR (SMOOTH curve!)")
    print(f"  • Dropout: {cfg.dropout}")
    print(f"  • MixUp: {'Enabled' if cfg.use_mixup else 'Disabled'} (alpha={cfg.mixup_alpha if cfg.use_mixup else 'N/A'})")
    print(f"  • Epochs: {cfg.epochs}")
    print(f"  • Early Stopping: patience={cfg.patience}")
    print("="*70 + "\n")

    # V6: INITIALIZATION SANITY CHECK
    init_stats = check_initialization(model, train_loader, criterion, device, cfg.num_classes)

    # Log init stats to W&B
    wandb.log({
        "init/loss": init_stats["init_loss"],
        "init/accuracy": init_stats["init_accuracy"],
        "init/expected_loss": init_stats["expected_loss"],
        "init/expected_accuracy": init_stats["expected_accuracy"],
    })

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

        global_step = epoch * len(train_loader)
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion,
                                           scaler, device, cfg, scheduler, epoch, global_step)

        val_loss, val_acc = val_epoch(model, val_loader, criterion, device, cfg)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        gap = train_acc - val_acc
        train_val_gaps.append(gap)
        lrs.append(optimizer.param_groups[0]['lr'])

        # V6: Log epoch metrics to W&B
        wandb.log({
            "train/loss": train_loss,
            "train/accuracy": train_acc,
            "val/loss": val_loss,
            "val/accuracy": val_acc,
            "train_val_gap": gap,
            "epoch": epoch + 1,
        })

        status = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            torch.save(model.state_dict(), f'{script_dir}/v6b/best_model.pth')
            wandb.save(f'{script_dir}/v6b/best_model.pth')
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

    # Finale Test Evaluation
    print("📝 Finale Test Evaluation...")
    model.load_state_dict(torch.load(f'{script_dir}/v6b/best_model.pth'))
    test_loss, test_acc = val_epoch(model, test_loader, criterion, device, cfg, desc="Test")

    print(f"\n🎯 FINALE ERGEBNISSE:")
    print(f"  V5:  82.65%")
    print(f"  V6b: {best_val_acc:.2f}% ({best_val_acc-82.65:+.2f}%)")
    print(f"  Test: {test_acc:.2f}%")

    # V6: Log final metrics to W&B
    wandb.log({
        "test/loss": test_loss,
        "test/accuracy": test_acc,
        "final/best_val_accuracy": best_val_acc,
        "final/train_val_gap": train_val_gaps[-1],
        "final/epochs_trained": len(train_accs)
    })

    # Visualisierung
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('V6b - Default Init (BASELINE) + W&B Tracking Analysis', fontsize=16, fontweight='bold')

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
    axes[0, 1].axhline(y=82.65, color='gray', linestyle='--', label='V5: 82.65%', alpha=0.5)
    axes[0, 1].axhline(y=best_val_acc, color='r', linestyle='--', label=f'V6b: {best_val_acc:.2f}%')
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

    # Learning Rate - SMOOTH CURVE!
    axes[1, 0].plot(lrs, linewidth=2, color='purple')
    axes[1, 0].set_title('Learning Rate Schedule (OneCycleLR - SMOOTH!)')
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
📊 V6b - DEFAULT Init (BASELINE)

Initialization:
  Init Loss: {init_stats['init_loss']:.4f}
  Expected:  {init_stats['expected_loss']:.4f}
  Difference: {abs(init_stats['init_loss'] - init_stats['expected_loss']):.4f}
  Status: {"⚠️  HIGHER" if init_stats['init_loss'] > init_stats['expected_loss'] else "✅"}

Performance:
  V5 Val Acc: 82.65%
  V6b Val Acc: {best_val_acc:.2f}%
  Improvement: {best_val_acc-82.65:+.2f}%

  Test Acc: {test_acc:.2f}%

Training:
  Epochs: {len(train_accs)}
  Early Stop: {'Yes' if len(train_accs) < cfg.epochs else 'No'}
  Final Gap: {train_val_gaps[-1]:.2f}%

W&B Tracking: ✅ ENABLED
Gradient Viz: ✅ ENABLED
Purpose: BASELINE for v6a comparison
"""
    axes[1, 2].text(0.1, 0.5, comparison_text, fontsize=11, family='monospace',
                   verticalalignment='center')

    plt.tight_layout()
    plt.savefig(f'{script_dir}/v6b/training_analysis.png', dpi=150)
    print(f"\n💾 Plot saved: v6b/training_analysis.png")

    # V6: Log plot to W&B
    wandb.log({"training_analysis": wandb.Image(f'{script_dir}/v6b/training_analysis.png')})

    plt.show()

    # Save config to JSON
    config_dict = {k: v for k, v in vars(cfg).items() if not k.startswith('_')}
    config_dict['num_params'] = num_params
    config_dict['best_val_acc'] = best_val_acc
    config_dict['test_acc'] = test_acc
    with open(f'{script_dir}/v6b/config.json', 'w') as f:
        json.dump(config_dict, f, indent=2)

    print("\n" + "="*70)
    print("V6b - DEFAULT Initialization (BASELINE) Complete!")
    print("="*70)
    print(f"⚠️  Init Loss: {init_stats['init_loss']:.4f} (Expected: {init_stats['expected_loss']:.4f})")
    print(f"   Difference: {abs(init_stats['init_loss'] - init_stats['expected_loss']):.4f} (higher than v6a)")
    print(f"✅ Best Val Accuracy: {best_val_acc:.2f}%")
    print(f"✅ Test Accuracy: {test_acc:.2f}%")
    print(f"✅ W&B Logs (OFFLINE): {script_dir}/v6b/wandb/")
    print(f"   View with: cd {script_dir}/v6b && wandb offline sync wandb/latest-run")
    print("="*70)

    # V6: Finish W&B run
    wandb.finish()


if __name__ == '__main__':
    main()
