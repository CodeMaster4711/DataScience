# -*- coding: utf-8 -*-
"""
V7a - ResNet18 + Weights & Biases Integration
Basierend auf v6a (beste Version) mit ResNet18 Architektur

V7a Features:
1. ✅ ResNet18 Architecture (11M params)
2. ✅ Korrekte Final Layer Init → Loss @ init = -log(1/10) ≈ 2.3026
3. ✅ Initialization Sanity Check (verify loss @ init)
4. ✅ Weights & Biases vollständiges Tracking
5. ✅ Gradient Visualisierung (wandb.watch)
6. ✅ Per-Batch und Per-Epoch Logging
7. ✅ Same training as V6a (OneCycleLR, MixUp)

Erwartung: ResNet18 sollte ähnlich oder besser als v6a (82.90% test acc) sein
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
import sys
from tqdm import tqdm
import json
import wandb

# Add parent directory to path for model imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.resnet import resnet18, check_initialization

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


# v7a liegt in Imagenette/v7/v7a/, also 3 Ebenen tief
# data liegt in Imagenette/data/
script_dir = Path(__file__).parent.parent.parent  # Imagenette/
data_dir = script_dir / "data"
data_dir.mkdir(exist_ok=True)
os.environ['FASTAI_HOME'] = str(data_dir)

# W&B: Connect to LOCAL server instead of wandb.ai
os.environ['WANDB_BASE_URL'] = 'http://localhost:8080'


# ============================================================================
# CONFIG - V7a (ResNet18)
# ============================================================================
class Config:
    # Data
    batch_size = 64  # ResNet benefits from larger batches (vs 32 in v6a)
    num_workers = 2
    data_percentage = 0.40

    # Model - ResNet18
    model_name = "ResNet18"
    dropout = 0.0  # ResNet with BN doesn't need much dropout (vs 0.25 in v6a)
    num_classes = 10

    # Training - Optimized for ResNet
    epochs = 100
    lr = 0.001  # Start conservative, can increase if needed
    weight_decay = 1e-4  # Lower than v6a (2e-4) because BN helps regularization
    label_smoothing = 0.1

    # OneCycleLR - Same as V6a
    pct_start = 0.3
    div_factor = 25.0
    final_div_factor = 1e4

    # MixUp - Same as V6a
    use_mixup = True
    mixup_alpha = 0.2

    # Early Stopping - Same as V6a
    patience = 20

    # System
    use_amp = False
    seed = 42

    # V7a Specific
    correct_init = True  # Use correct initialization (like v6a)
    version = "v7a"

    # W&B - Local Server
    wandb_project = "imagenette-training"
    wandb_entity = "codemaster4711"  # Your local W&B server username


# ============================================================================
# ARCHITECTURE - ResNet18
# ============================================================================
# Architecture is imported from ../models/resnet.py
# See import at top: from models.resnet import resnet18, check_initialization
#
# ResNet18 specs:
# - 18 layers (BasicBlock)
# - ~11M parameters
# - Proper skip connections
# - Batch Normalization
# - Correct initialization (Kaiming He + final layer -log(1/10))


# ============================================================================
# INITIALIZATION SANITY CHECK
# ============================================================================
# check_initialization() is imported from ../models/resnet.py
# See import at top: from models.resnet import resnet18, check_initialization


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
    run_name = f"{cfg.version}-correct-init"
    wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        name=run_name,
        config={
            "version": cfg.version,
            "architecture": "ResNet18",
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

    # Model - ResNet18
    model = resnet18(num_classes=cfg.num_classes, dropout=cfg.dropout,
                     correct_init=cfg.correct_init).to(device)
    print(f"Model: ResNet18")
    print(f"Parameters: {model.get_num_params():,}")

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
    print("V6a - Correct Initialization + W&B Tracking")
    print("="*70)
    print("🎯 Ziel: Korrekte Init Loss + Vollständiges Experiment Tracking")
    print("\n📐 Architecture:")
    print("  • Same as V5: 6 Stages, 12 Residual Blocks")
    print("  • Progressive: 48→96→192→256→384→512")
    print(f"  • Parameters: {num_params:,}")
    print("\n⚙️  V6a NEW Features:")
    print(f"  • ✨ Correct Init: Final layer → uniform softmax output")
    print(f"  • ✨ Expected Init Loss: -log(1/10) ≈ 2.3026")
    print(f"  • ✨ W&B Tracking: All metrics, gradients, plots")
    print(f"  • ✨ Gradient visualization via wandb.watch()")
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
            torch.save(model.state_dict(), f'{script_dir}/v6a/best_model.pth')
            wandb.save(f'{script_dir}/v6a/best_model.pth')
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
    model.load_state_dict(torch.load(f'{script_dir}/v6a/best_model.pth'))
    test_loss, test_acc = val_epoch(model, test_loader, criterion, device, cfg, desc="Test")

    print(f"\n🎯 FINALE ERGEBNISSE:")
    print(f"  V5:  82.65%")
    print(f"  V6a: {best_val_acc:.2f}% ({best_val_acc-82.65:+.2f}%)")
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
    fig.suptitle('V6a - Correct Init + W&B Tracking Analysis', fontsize=16, fontweight='bold')

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
    axes[0, 1].axhline(y=best_val_acc, color='r', linestyle='--', label=f'V6a: {best_val_acc:.2f}%')
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
📊 V6a - Correct Initialization

Initialization:
  Init Loss: {init_stats['init_loss']:.4f}
  Expected:  {init_stats['expected_loss']:.4f}
  Difference: {abs(init_stats['init_loss'] - init_stats['expected_loss']):.4f}
  Status: {"✅ CORRECT" if abs(init_stats['init_loss'] - init_stats['expected_loss']) < 0.5 else "⚠️  CHECK"}

Performance:
  V5 Val Acc: 82.65%
  V6a Val Acc: {best_val_acc:.2f}%
  Improvement: {best_val_acc-82.65:+.2f}%

  Test Acc: {test_acc:.2f}%

Training:
  Epochs: {len(train_accs)}
  Early Stop: {'Yes' if len(train_accs) < cfg.epochs else 'No'}
  Final Gap: {train_val_gaps[-1]:.2f}%

W&B Tracking: ✅ ENABLED
Gradient Viz: ✅ ENABLED
"""
    axes[1, 2].text(0.1, 0.5, comparison_text, fontsize=11, family='monospace',
                   verticalalignment='center')

    plt.tight_layout()
    plt.savefig(f'{script_dir}/v6a/training_analysis.png', dpi=150)
    print(f"\n💾 Plot saved: v6a/training_analysis.png")

    # V6: Log plot to W&B
    wandb.log({"training_analysis": wandb.Image(f'{script_dir}/v6a/training_analysis.png')})

    plt.show()

    # Save config to JSON
    config_dict = {k: v for k, v in vars(cfg).items() if not k.startswith('_')}
    config_dict['num_params'] = num_params
    config_dict['best_val_acc'] = best_val_acc
    config_dict['test_acc'] = test_acc
    with open(f'{script_dir}/v6a/config.json', 'w') as f:
        json.dump(config_dict, f, indent=2)

    print("\n" + "="*70)
    print("V6a - Correct Initialization SUCCESS!")
    print("="*70)
    print(f"✅ Init Loss: {init_stats['init_loss']:.4f} (Expected: {init_stats['expected_loss']:.4f})")
    print(f"✅ Best Val Accuracy: {best_val_acc:.2f}%")
    print(f"✅ Test Accuracy: {test_acc:.2f}%")
    print(f"✅ W&B Logs (OFFLINE): {script_dir}/v6a/wandb/")
    print(f"   View with: cd {script_dir}/v6a && wandb offline sync wandb/latest-run")
    print("="*70)

    # V6: Finish W&B run
    wandb.finish()


if __name__ == '__main__':
    main()
