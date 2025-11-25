"""
V10-Step2: Fashion-MNIST mit erweiterten Debugging & Monitoring Features

Implementiert alle Schritte für systematisches ML-Debugging:
- Setup: Fixed Seed, Init Well, Verify Loss@Init
- Baselines: Human, Input-Independent, Overfit One Batch
- Monitoring: Batch-Level Loss, Per-Class Metrics, Prediction Dynamics
- Debugging: Backprop Dependencies, Input Visualization
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import random

# Importiere unsere Module
from debug_tools import (
    human_baseline_test,
    test_input_independent_baseline,
    overfit_one_batch,
    verify_batch_independence
)
from metrics import (
    compute_detailed_metrics,
    print_detailed_metrics,
    plot_confusion_matrix,
    compare_train_test_metrics,
    analyze_misclassifications
)
from visualization import (
    visualize_input_before_net,
    PredictionDynamicsTracker,
    plot_training_curves,
    plot_batch_level_loss
)


# ============================================================================
# SETUP: Fixed Random Seed
# ============================================================================
def set_seed(seed=42):
    """Setze alle Random Seeds für vollständige Reproduzierbarkeit"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)


# ============================================================================
# KONFIGURATION
# ============================================================================
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 50

# Debug Flags - setze auf True um Tests zu aktivieren
RUN_HUMAN_BASELINE = True
RUN_INPUT_INDEPENDENT_BASELINE = True
RUN_OVERFIT_TEST = True
RUN_BATCH_INDEPENDENCE_TEST = True


# ============================================================================
# DATEN LADEN - Simplify (keine Augmentation)
# ============================================================================
transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# ============================================================================
# MODELL DEFINITION - MyNN (wie in v9)
# ============================================================================
class MyNN(nn.Module):
    def __init__(self, input_channels):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(p=0.4),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=0.4),

            nn.Linear(64, 10)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Init Well: He/Kaiming initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

        # Letzte Schicht: kleine Gewichte für Loss@Init ≈ 2.302
        final_layer = self.classifier[-1]
        nn.init.normal_(final_layer.weight, mean=0, std=0.01)
        nn.init.constant_(final_layer.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ============================================================================
# DEVICE SETUP
# ============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.backends.mps.is_available():
    device = torch.device('mps')

print("\n" + "="*70)
print("V10-Step2 - Advanced Debugging & Monitoring")
print("="*70)
print(f"Using device: {device}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Learning Rate: {LEARNING_RATE}")
print(f"Epochs: {NUM_EPOCHS}")
print("="*70 + "\n")


# ============================================================================
# DEBUGGING TESTS (vor dem Training!)
# ============================================================================
print("\n" + "🔍"*35)
print("RUNNING DEBUGGING TESTS")
print("🔍"*35 + "\n")

# 1. Human Baseline Test
if RUN_HUMAN_BASELINE:
    human_baseline_test(test_dataset, num_samples=16)

# 2. Input-Independent Baseline
if RUN_INPUT_INDEPENDENT_BASELINE:
    baseline_acc = test_input_independent_baseline(train_loader, test_loader, device, epochs=10)

# 3. Visualize Input Before Net
visualize_input_before_net(train_loader, num_samples=16)


# ============================================================================
# MODELL ERSTELLEN & VERIFY LOSS@INIT
# ============================================================================
model = MyNN(1).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("\n" + "="*70)
print("VERIFY LOSS @ INIT")
print("="*70)
print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Prüfe initialen Loss
model.eval()
with torch.no_grad():
    for batch_features, batch_labels in train_loader:
        batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
        outputs = model(batch_features)
        init_loss = criterion(outputs, batch_labels).item()

        probs = torch.softmax(outputs, dim=1)
        avg_probs = probs.mean(dim=0)

        print(f"Initial Loss: {init_loss:.4f} (expected: ~2.302 = -log(1/10))")
        print(f"Deviation from expected: {abs(init_loss - 2.302):.4f}")

        if abs(init_loss - 2.302) < 0.5:
            print("✓ Loss is in expected range!")
        else:
            print("⚠ Loss deviates significantly from expected value")

        print(f"\nInitial class probabilities (should be ~0.1 each):")
        for i, prob in enumerate(avg_probs[:5]):  # Zeige nur erste 5
            print(f"  Class {i}: {prob:.4f}", end="  ")
        print("...")
        break

print("="*70 + "\n")


# 4. Overfit One Batch Test
if RUN_OVERFIT_TEST:
    overfit_loss, overfit_acc = overfit_one_batch(MyNN, device, train_dataset, batch_size=8)


# ============================================================================
# SETUP FÜR PREDICTION DYNAMICS TRACKING
# ============================================================================
# Erstelle festen Batch für Tracking
fixed_batch_size = 16
fixed_indices = np.random.choice(len(test_dataset), fixed_batch_size, replace=False)
fixed_batch_x = []
fixed_batch_y = []

for idx in fixed_indices:
    img, label = test_dataset[idx]
    fixed_batch_x.append(img)
    fixed_batch_y.append(label)

fixed_batch_x = torch.stack(fixed_batch_x)
fixed_batch_y = torch.tensor(fixed_batch_y)

pred_tracker = PredictionDynamicsTracker(fixed_batch_x, fixed_batch_y, device)


# ============================================================================
# TRAINING LOOP
# ============================================================================
print("\n" + "🚀"*35)
print("STARTING TRAINING")
print("🚀"*35 + "\n")

train_losses = []
test_accuracies = []
batch_losses = []  # Für batch-level monitoring
best_acc = 0

for epoch in range(NUM_EPOCHS):
    # Training
    model.train()
    running_loss = 0.0
    epoch_batch_losses = []

    for batch_features, batch_labels in train_loader:
        batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)

        outputs = model(batch_features)
        loss = criterion(outputs, batch_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        batch_losses.append(loss.item())
        epoch_batch_losses.append(loss.item())

    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Evaluation
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            outputs = model(batch_features)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == batch_labels).sum().item()
            total += batch_labels.size(0)

    test_acc = 100 * correct / total
    test_accuracies.append(test_acc)

    # Track Prediction Dynamics alle 5 Epochen
    if (epoch + 1) % 5 == 0 or epoch == 0:
        pred_tracker.update(model, epoch + 1)

    # Save best model
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), "outputs/best_model_v10.pth")
        print(f'Epoch [{epoch+1:2d}/{NUM_EPOCHS}] Loss: {avg_train_loss:.4f} | Acc: {test_acc:.2f}% ✓ BEST')
    else:
        print(f'Epoch [{epoch+1:2d}/{NUM_EPOCHS}] Loss: {avg_train_loss:.4f} | Acc: {test_acc:.2f}%')

print("\n" + "="*70)
print(f"TRAINING COMPLETE!")
print(f"Best Test Accuracy: {best_acc:.2f}%")
print("="*70 + "\n")


# ============================================================================
# POST-TRAINING ANALYSIS
# ============================================================================
print("\n" + "📊"*35)
print("POST-TRAINING ANALYSIS")
print("📊"*35 + "\n")

# Lade bestes Modell
model.load_state_dict(torch.load("outputs/best_model_v10.pth"))

# Compute Detailed Metrics
print("Computing detailed metrics on full datasets...")
train_metrics = compute_detailed_metrics(model, train_loader, device, "Train")
test_metrics = compute_detailed_metrics(model, test_loader, device, "Test")

# Print Metrics
print_detailed_metrics(train_metrics, "Train")
print_detailed_metrics(test_metrics, "Test")

# Compare Train vs Test
compare_train_test_metrics(train_metrics, test_metrics)

# Confusion Matrix
plot_confusion_matrix(test_metrics)

# Analyze Misclassifications
analyze_misclassifications(test_metrics, test_dataset)

# Training Curves
plot_training_curves(train_losses, test_accuracies)

# Batch-Level Loss
plot_batch_level_loss(batch_losses)

# Prediction Dynamics
pred_tracker.plot()


# ============================================================================
# FINAL DEBUGGING TEST: Batch Independence
# ============================================================================
if RUN_BATCH_INDEPENDENCE_TEST:
    verify_batch_independence(model, device, test_loader)


# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*70)
print("FINAL SUMMARY - V10-Step2")
print("="*70)
print(f"\n📈 Performance:")
print(f"   Best Test Accuracy: {best_acc:.2f}%")
print(f"   Final Train Accuracy: {train_metrics['accuracy']:.2f}%")
print(f"   Train-Test Gap: {train_metrics['accuracy'] - test_metrics['accuracy']:.2f}%")

if RUN_INPUT_INDEPENDENT_BASELINE:
    print(f"\n🎯 Baseline Comparisons:")
    print(f"   Input-Independent Baseline: {baseline_acc:.2f}%")
    print(f"   Our Model: {best_acc:.2f}%")
    print(f"   Improvement: {best_acc - baseline_acc:.2f}%")

print(f"\n✓ All debugging tests completed!")
print(f"✓ Visualizations saved to outputs/ directory")
print(f"✓ Best model saved to outputs/best_model_v10.pth")

print("\n" + "="*70)
print("Check the outputs/ folder for detailed analysis plots!")
print("="*70 + "\n")
