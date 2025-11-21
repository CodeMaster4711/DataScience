import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

print("=== VERSION 1: Einfaches Modell ===")
print("Architektur: 784 -> 128 -> 64 -> 10")

# Modell Definition
input_dim = 28 * 28

model = nn.Sequential(
    nn.Linear(784, 128),    # Hidden Layer 1: 128 Knoten
    nn.ReLU(),
    nn.Linear(128, 64),     # Hidden Layer 2: 64 Knoten
    nn.ReLU(),
    nn.Linear(64, 10)       # Output Layer: 10 Knoten
)

print(f"Anzahl Parameter: {sum(p.numel() for p in model.parameters()):,}")

# Lade Daten
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Training Setup
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
num_epochs = 5

# Training Loop
print("\n=== Training startet ===")
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(-1, 28*28)

        # Forward + Backward Pass
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()         # Backpropagation
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}')

# Speichere Modell
torch.save(model.state_dict(), 'model_v1.pth')
print("\nModell gespeichert als 'model_v1.pth'")
