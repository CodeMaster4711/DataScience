import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Konfiguration - Exakt wie im Medium-Artikel
torch.manual_seed(42)
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 50

# Einfache Transformationen (keine Augmentation wie im Artikel)
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Daten laden
train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# MyNN - Exakt wie im Medium-Artikel
# Nur 2 Conv-Layer (32 -> 64) mit padding="same"
class MyNN(nn.Module):
    def __init__(self, input_channels):
        super().__init__()

        # Feature Extraction - 2 Conv Blocks
        self.features = nn.Sequential(
            # Conv Block 1: 1x28x28 -> 32x28x28 -> 32x14x14
            nn.Conv2d(input_channels, 32, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Block 2: 32x14x14 -> 64x14x14 -> 64x7x7
            nn.Conv2d(32, 64, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Classifier - FC Layers mit Dropout
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(p=0.4),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=0.4),

            nn.Linear(64, 10)  # 10 classes for Fashion-MNIST
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


model = MyNN(1)

# Device Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.backends.mps.is_available():
    device = torch.device('mps')

model = model.to(device)

# Loss und Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("\n" + "="*70)
print("V8 - Simple CNN from Scratch (Medium Article Implementation)")
print("="*70)
print(f"Using device: {device}")
print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Epochs: {NUM_EPOCHS}")
print(f"Learning Rate: {LEARNING_RATE}")
print("="*70)
print("Architecture (MyNN):")
print("  Features:")
print("    Conv1: 1 -> 32 (3x3, padding=same) + ReLU + BN + MaxPool")
print("    Conv2: 32 -> 64 (3x3, padding=same) + ReLU + BN + MaxPool")
print("  Classifier:")
print("    FC1: 3136 -> 128 + ReLU + Dropout(0.4)")
print("    FC2: 128 -> 64 + ReLU + Dropout(0.4)")
print("    FC3: 64 -> 10 (Output)")
print("="*70)
print("Target: ~92% Test Accuracy")
print("="*70 + "\n")

train_losses = []
test_accuracies = []
best_acc = 0

# Training Loop - Exakt wie im Artikel
for epoch in range(NUM_EPOCHS):
    # Training
    model.train()
    running_loss = 0.0

    for batch_features, batch_labels in train_loader:
        batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)

        # Forward pass
        outputs = model(batch_features)
        loss = criterion(outputs, batch_labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

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

    # Save best model
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), "best_model_v8.pth")
        print(f'Epoch [{epoch+1:2d}/{NUM_EPOCHS}], Loss: {avg_train_loss:.4f}, Acc: {test_acc:.2f}% ✓ BEST')
    else:
        print(f'Epoch [{epoch+1:2d}/{NUM_EPOCHS}], Loss: {avg_train_loss:.4f}, Acc: {test_acc:.2f}%')

print("\n" + "="*70)
print(f"Training Complete!")
print(f"Best Test Accuracy: {best_acc:.2f}%")
print(f"Target (92%): {'✓ ACHIEVED!' if best_acc >= 92 else f'Gap: {92 - best_acc:.2f}%'}")
print("="*70 + "\n")

# Final evaluation function
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch_features, batch_labels in loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            outputs = model(batch_features)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == batch_labels).sum().item()
            total += batch_labels.size(0)
    return correct / total

train_acc = evaluate(model, train_loader)
test_acc = evaluate(model, test_loader)

print(f"Train Accuracy: {train_acc:.2%}")
print(f"Test Accuracy: {test_acc:.2%}")

torch.save(model.state_dict(), "final_model_v8.pth")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Training Loss
axes[0].plot(train_losses, linewidth=2, color='blue')
axes[0].set_title('Training Loss', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].grid(True, alpha=0.3)

# Test Accuracy
axes[1].plot(test_accuracies, linewidth=2, color='green')
axes[1].axhline(y=92, color='red', linestyle='--', label='Target: 92%', alpha=0.7)
axes[1].axhline(y=best_acc, color='darkgreen', linestyle='-', linewidth=2, label=f'V8 Best: {best_acc:.2f}%')
axes[1].set_title('Test Accuracy', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy (%)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('v8_results.png', dpi=100)
print("\nPlot saved: v8_results.png")
plt.show()
