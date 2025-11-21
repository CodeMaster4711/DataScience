import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Modell-Architektur (muss identisch zum Training sein!)
input_dim = 28 * 28
input_layer = nn.Linear(input_dim, 128)
output_layer = nn.Linear(64, 10)

model = nn.Sequential(
    input_layer,
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    output_layer
)

# Lade das trainierte Modell
model.load_state_dict(torch.load('mnist_model.pth'))
model.eval()  # Setze Modell in Evaluierungsmodus

# Lade Test-Daten
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Teste das Modell
correct = 0
total = 0

print("Starte Testing...")

with torch.no_grad():  # Kein Gradient-Berechnung beim Testen
    for data, target in test_loader:
        data = data.view(-1, 28*28)  # Flatten: 28x28 � 784
        output = model(data)

        # Hole die vorhergesagte Klasse (Index mit h�chstem Wert)
        _, predicted = torch.max(output.data, 1)

        total += target.size(0)
        correct += (predicted == target).sum().item()

# Berechne Genauigkeit
accuracy = 100 * correct / total

print(f"\n--- Test-Ergebnisse ---")
print(f"Getestete Bilder: {total}")
print(f"Richtig erkannt: {correct}")
print(f"Genauigkeit: {accuracy:.2f}%")
