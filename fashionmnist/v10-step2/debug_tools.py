"""
Debug Tools für Fashion-MNIST Training
Enthält Baseline-Tests und Debugging-Funktionen
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms


# Fashion-MNIST Klassennamen
CLASS_NAMES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def human_baseline_test(test_dataset, num_samples=16, save_path='outputs/human_baseline_samples.png'):
    """
    1. HUMAN BASELINE TEST
    Zeigt zufällige Bilder aus dem Test-Set für menschliche Bewertung.
    Menschen sollten ~95%+ Accuracy erreichen.
    """
    print("\n" + "="*70)
    print("1. HUMAN BASELINE TEST")
    print("="*70)
    print("Displaying 16 random test samples for human evaluation...")
    print("Expected human accuracy: ~95%+")
    print(f"Samples saved to: {save_path}")

    # Wähle zufällige Samples
    indices = np.random.choice(len(test_dataset), num_samples, replace=False)

    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    fig.suptitle('Human Baseline Test - Can you classify these?', fontsize=16, fontweight='bold')

    for idx, ax in enumerate(axes.flat):
        if idx < num_samples:
            image, label = test_dataset[indices[idx]]
            # Konvertiere von Tensor zu numpy
            img_np = image.squeeze().numpy()

            ax.imshow(img_np, cmap='gray')
            ax.set_title(f'Label: {CLASS_NAMES[label]}', fontsize=10)
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Human baseline samples saved!")
    print("="*70 + "\n")
    plt.close()


class InputIndependentBaseline(nn.Module):
    """
    2. INPUT-INDEPENDENT BASELINE
    Modell das nur Bias lernt (Input wird ignoriert/auf 0 gesetzt).
    Sollte DEUTLICH schlechter sein als echtes Modell.
    Falls gleich gut → Bug: Modell extrahiert keine Features!
    """
    def __init__(self):
        super().__init__()
        # Nur ein Linear Layer - keine Feature Extraction
        self.fc = nn.Linear(28 * 28, 10)

    def forward(self, x):
        # Setze Input auf konstante Nullen - ignoriere echte Bilder!
        x = torch.zeros_like(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def test_input_independent_baseline(train_loader, test_loader, device, epochs=10):
    """
    Trainiert und testet Input-Independent Baseline.
    """
    print("\n" + "="*70)
    print("2. INPUT-INDEPENDENT BASELINE TEST")
    print("="*70)
    print("Training model that IGNORES input (all inputs = 0)...")
    print("This should perform WORSE than real model!")

    model = InputIndependentBaseline().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training
    for epoch in range(epochs):
        model.train()
        for batch_features, batch_labels in train_loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)

            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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

    baseline_acc = 100 * correct / total

    print(f"\nInput-Independent Baseline Accuracy: {baseline_acc:.2f}%")
    print(f"Expected: ~10% (random guessing)")

    if baseline_acc > 15:
        print("⚠ WARNING: Baseline is too good! Should be ~10%")
    else:
        print("✓ Baseline is appropriately bad (close to random guessing)")

    print("\n⚠ IMPORTANT: Your real model MUST beat this baseline!")
    print("   If not, your model is not learning from the images!")
    print("="*70 + "\n")

    return baseline_acc


def overfit_one_batch(model_class, device, train_dataset, batch_size=8, max_iterations=1000):
    """
    3. OVERFIT ONE BATCH
    Versucht einen einzelnen Batch perfekt zu lernen (Loss → 0, Accuracy = 100%).
    Falls nicht möglich → Bug im Modell oder Training!
    """
    print("\n" + "="*70)
    print("3. OVERFIT ONE BATCH TEST")
    print("="*70)
    print(f"Attempting to overfit {batch_size} samples to 100% accuracy...")
    print("This tests if model has enough capacity to memorize.")

    # Erstelle Mini-Dataset mit nur einem Batch
    indices = np.random.choice(len(train_dataset), batch_size, replace=False)
    mini_batch_x = []
    mini_batch_y = []

    for idx in indices:
        img, label = train_dataset[idx]
        mini_batch_x.append(img)
        mini_batch_y.append(label)

    mini_batch_x = torch.stack(mini_batch_x).to(device)
    mini_batch_y = torch.tensor(mini_batch_y).to(device)

    # Modell erstellen (höhere Kapazität für Overfitting)
    model = model_class(1).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    losses = []
    accuracies = []

    print(f"\nTraining on {batch_size} samples...")
    for iteration in range(max_iterations):
        model.train()

        # Forward pass
        outputs = model(mini_batch_x)
        loss = criterion(outputs, mini_batch_y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Berechne Accuracy
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == mini_batch_y).sum().item()
        acc = 100 * correct / batch_size

        losses.append(loss.item())
        accuracies.append(acc)

        # Print Fortschritt
        if (iteration + 1) % 100 == 0 or iteration == 0:
            print(f"  Iteration {iteration+1:4d}: Loss = {loss.item():.4f}, Acc = {acc:.1f}%")

        # Früher Stop bei perfekter Accuracy
        if acc == 100.0 and loss.item() < 0.01:
            print(f"\n✓ SUCCESS: Perfect overfitting achieved at iteration {iteration+1}!")
            break

    # Finale Evaluation
    final_loss = losses[-1]
    final_acc = accuracies[-1]

    print(f"\nFinal Results after {len(losses)} iterations:")
    print(f"  Loss: {final_loss:.4f}")
    print(f"  Accuracy: {final_acc:.1f}%")

    if final_acc == 100.0:
        print("\n✓ Model can perfectly memorize data - capacity is sufficient!")
    else:
        print(f"\n⚠ WARNING: Could not achieve 100% accuracy (only {final_acc:.1f}%)")
        print("   This suggests a bug in model or training loop!")

    # Visualisierung
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Loss curve
    axes[0].plot(losses, linewidth=2, color='blue')
    axes[0].set_title('Overfit Loss Curve', fontweight='bold')
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True, alpha=0.3)

    # Accuracy curve
    axes[1].plot(accuracies, linewidth=2, color='green')
    axes[1].axhline(y=100, color='red', linestyle='--', label='Target: 100%')
    axes[1].set_title('Overfit Accuracy Curve', fontweight='bold')
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Zeige die trainierten Bilder
    model.eval()
    with torch.no_grad():
        outputs = model(mini_batch_x)
        _, predictions = torch.max(outputs, 1)

    # Visualisiere Samples
    num_show = min(8, batch_size)
    for i in range(num_show):
        row = i // 4
        col = i % 4
        if i < 4:
            ax_idx = 2

    # Zeige erste 8 Bilder vom Batch
    axes[2].axis('off')
    axes[2].set_title('Overfitted Samples', fontweight='bold')

    plt.tight_layout()
    plt.savefig('outputs/overfit_one_batch.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: outputs/overfit_one_batch.png")
    print("="*70 + "\n")
    plt.close()

    return final_loss, final_acc


def verify_batch_independence(model, device, test_loader):
    """
    9. USE BACKPROP TO CHART DEPENDENCIES
    Verifiziert dass keine ungewollten Batch-Abhängigkeiten existieren.
    Loss nur für ein Bild i → Gradient sollte nur für i ≠ 0 sein.
    """
    print("\n" + "="*70)
    print("9. BATCH INDEPENDENCE TEST (Backprop Dependencies)")
    print("="*70)
    print("Testing for unintended cross-batch dependencies...")

    model.eval()

    # Hole einen Batch
    for batch_features, batch_labels in test_loader:
        batch_features = batch_features.to(device)
        batch_labels = batch_labels.to(device)

        batch_size = batch_features.size(0)

        # Test für erstes Bild (i=0)
        test_idx = 0

        # Forward pass
        batch_features.requires_grad = True
        outputs = model(batch_features)

        # Loss NUR für Bild i (nicht für gesamten Batch!)
        loss_single = outputs[test_idx].sum()

        # Backward pass
        loss_single.backward()

        # Prüfe Gradienten
        gradients = batch_features.grad

        # Gradient für Bild i sollte != 0 sein
        grad_norm_target = gradients[test_idx].abs().sum().item()

        # Gradienten für alle anderen Bilder sollten = 0 sein
        other_grads = []
        for i in range(batch_size):
            if i != test_idx:
                grad_norm = gradients[i].abs().sum().item()
                other_grads.append(grad_norm)

        max_other_grad = max(other_grads) if other_grads else 0

        print(f"\nResults for image index {test_idx}:")
        print(f"  Gradient norm for target image {test_idx}: {grad_norm_target:.6f}")
        print(f"  Max gradient norm for other images: {max_other_grad:.6f}")

        if max_other_grad < 1e-6:
            print("\n✓ SUCCESS: No cross-batch dependencies detected!")
            print("   Gradients are correctly isolated to single image.")
        else:
            print(f"\n⚠ WARNING: Cross-batch dependencies detected!")
            print(f"   Other images have non-zero gradients (max: {max_other_grad:.6f})")
            print("   This suggests a bug in your vectorized implementation!")

        print("="*70 + "\n")
        break  # Nur einen Batch testen

    return max_other_grad < 1e-6
