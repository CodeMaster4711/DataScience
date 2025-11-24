import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

BATCH_SIZE = 256  # Noch größere Batch Size für schnelleres Training
LEARNING_RATE = 0.003  # Höhere LR für schnellere Konvergenz
NUM_EPOCHS = 15  # Weniger Epochen, reicht völlig

# Data Augmentation für bessere Generalisierung
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # Zufälliges Spiegeln
    transforms.RandomRotation(10),  # Leichte Rotation
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Leichte Verschiebung
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=train_transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=test_transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# Tieferes CNN mit Batch Normalization
class ImprovedCNN(nn.Module):
    def __init__(self):
        super(ImprovedCNN, self).__init__()

        # Block 1
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout2d(0.25)

        # Block 2
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2)
        self.dropout2 = nn.Dropout2d(0.25)

        # Block 3
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2)
        self.dropout3 = nn.Dropout2d(0.3)

        # Fully Connected Layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 3 * 3, 512)
        self.bn7 = nn.BatchNorm1d(512)
        self.dropout4 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.bn8 = nn.BatchNorm1d(256)
        self.dropout5 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        # Block 1
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout1(x)

        # Block 2
        x = torch.relu(self.bn3(self.conv3(x)))
        x = torch.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.dropout2(x)

        # Block 3
        x = torch.relu(self.bn5(self.conv5(x)))
        x = torch.relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)
        x = self.dropout3(x)

        # Fully Connected
        x = self.flatten(x)
        x = torch.relu(self.bn7(self.fc1(x)))
        x = self.dropout4(x)
        x = torch.relu(self.bn8(self.fc2(x)))
        x = self.dropout5(x)
        x = self.fc3(x)

        return x

model = ImprovedCNN()

# GPU Support
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(f"Training auf: {device}")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

# Learning Rate Scheduler - CosineAnnealing für bessere Konvergenz
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

print("Starte Training mit verbessertem CNN...")
print(f"Modell hat {sum(p.numel() for p in model.parameters())} Parameter")

train_losses = []
test_accuracies = []
best_acc = 0

for epoch in range(NUM_EPOCHS):
    # Training
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Testing
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_acc = 100 * correct / total
    test_accuracies.append(test_acc)

    # Learning Rate anpassen
    scheduler.step()

    # Bestes Modell speichern
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), "best_model.pth")
        print(f'✓ Neues bestes Modell gespeichert!')

    current_lr = optimizer.param_groups[0]['lr']
    print(f'Epoche [{epoch+1}/{NUM_EPOCHS}], Loss: {avg_train_loss:.4f}, Test Acc: {test_acc:.2f}%, LR: {current_lr:.6f}')

print("\nTraining abgeschlossen!")
print(f"Beste Test Accuracy: {best_acc:.2f}%")

torch.save(model.state_dict(), "final_model.pth")
print("Finales Modell wurde als 'final_model.pth' gespeichert.")

# Visualisierung
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(train_losses)
plt.title('Training Loss')
plt.xlabel('Epoche')
plt.ylabel('Loss')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(test_accuracies)
plt.axhline(y=best_acc, color='r', linestyle='--', label=f'Best: {best_acc:.2f}%')
plt.title('Test Accuracy')
plt.xlabel('Epoche')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_results.png')
print("Diagramm als 'training_results.png' gespeichert.")
plt.show()
