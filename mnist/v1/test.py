import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

print("=== VERSION 1: Test ===")

# Modell Definition (identisch zu train.py)
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
)

# Lade das trainierte Modell
model.load_state_dict(torch.load('model_v1.pth'))
model.eval()

# Lade Test-Daten
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_dataset = datasets.MNIST('../data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Teste das Modell
correct = 0
total = 0

print("Starte Testing...")

with torch.no_grad():
    for data, target in test_loader:
        data = data.view(-1, 28*28)
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

# Berechne Genauigkeit
accuracy = 100 * correct / total

print(f"\n--- Test-Ergebnisse V1 ---")
print(f"Getestete Bilder: {total}")
print(f"Richtig erkannt: {correct}")
print(f"Genauigkeit: {accuracy:.2f}%")
