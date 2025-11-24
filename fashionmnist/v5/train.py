import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Konfiguration
BATCH_SIZE = 128
LEARNING_RATE = 0.01
NUM_EPOCHS = 10

# Data Augmentation
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Daten laden
train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=train_transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Einfaches CNN Modell
model = nn.Sequential(
    # Block 1
    nn.Conv2d(1, 32, kernel_size=3, padding=1),
    nn.BatchNorm2d(32),
    nn.ReLU(),
    nn.Conv2d(32, 32, kernel_size=3, padding=1),
    nn.BatchNorm2d(32),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Dropout2d(0.25),

    # Block 2
    nn.Conv2d(32, 64, kernel_size=3, padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.Conv2d(64, 64, kernel_size=3, padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Dropout2d(0.25),

    # Fully Connected
    nn.Flatten(),
    nn.Linear(64 * 7 * 7, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 10)
)

# GPU Support - CUDA (NVIDIA) oder MPS (Apple Silicon)
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"Training auf: CUDA GPU - {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    device = torch.device('mps')
    print(f"Training auf: Apple Silicon GPU (MPS)")
else:
    device = torch.device('cpu')
    print(f"Training auf: CPU (keine GPU gefunden)")

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=5e-4)

# OneCycleLR - Große Steps am Anfang, kleine am Ende!
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=LEARNING_RATE,
    epochs=NUM_EPOCHS,
    steps_per_epoch=len(train_loader),
    pct_start=0.3,
    anneal_strategy='cos'
)

print("\n" + "="*60)
print("V5 - Optimiertes CNN Training")
print("="*60)
print(f"Modell Parameter: {sum(p.numel() for p in model.parameters()):,}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Epochen: {NUM_EPOCHS}")
print(f"Device: {device}")
print("="*60 + "\n")

train_losses = []
test_accuracies = []
learning_rates = []
best_acc = 0

for epoch in range(NUM_EPOCHS):
    # Training
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Forward Pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward Pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()  # LR update nach jedem Batch

        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    learning_rates.append(optimizer.param_groups[0]['lr'])

    # Testing
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_acc = 100 * correct / total
    test_accuracies.append(test_acc)

    # Bestes Modell speichern
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), "best_model_v5.pth")
        print(f'Epoche [{epoch+1}/{NUM_EPOCHS}], Loss: {avg_train_loss:.4f}, Acc: {test_acc:.2f}%, LR: {learning_rates[-1]:.6f} ✓ BEST')
    else:
        print(f'Epoche [{epoch+1}/{NUM_EPOCHS}], Loss: {avg_train_loss:.4f}, Acc: {test_acc:.2f}%, LR: {learning_rates[-1]:.6f}')

print("\n" + "="*60)
print(f"Training abgeschlossen!")
print(f"Beste Accuracy: {best_acc:.2f}%")
print(f"Verbesserung vs V4 (93.27%): {best_acc - 93.27:+.2f}%")
print("="*60 + "\n")

torch.save(model.state_dict(), "final_model_v5.pth")

# Visualisierung
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Training Loss
axes[0].plot(train_losses, linewidth=2, color='blue')
axes[0].set_title('Training Loss')
axes[0].set_xlabel('Epoche')
axes[0].set_ylabel('Loss')
axes[0].grid(True, alpha=0.3)

# Test Accuracy
axes[1].plot(test_accuracies, linewidth=2, color='green')
axes[1].axhline(y=93.27, color='orange', linestyle='--', label='V4: 93.27%', alpha=0.7)
axes[1].axhline(y=best_acc, color='r', linestyle='--', label=f'V5 Best: {best_acc:.2f}%')
axes[1].set_title('Test Accuracy')
axes[1].set_xlabel('Epoche')
axes[1].set_ylabel('Accuracy (%)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Learning Rate (OneCycleLR)
axes[2].plot(learning_rates, linewidth=2, color='purple')
axes[2].set_title('Learning Rate (OneCycleLR)')
axes[2].set_xlabel('Epoche')
axes[2].set_ylabel('Learning Rate')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('v5_results.png', dpi=100)
print("Grafiken gespeichert: v5_results.png")
plt.show()
