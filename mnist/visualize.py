import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Modell-Architektur (identisch zum Training)
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
model.eval()

print("=== NETZWERK-ARCHITEKTUR ===")
print(model)
print("\nLayer-Details:")
print(f"Input Layer:  784 Knoten (28x28 Pixel)")
print(f"Hidden Layer 1: 128 Knoten")
print(f"Activation: ReLU")
print(f"Hidden Layer 2: 64 Knoten")
print(f"Activation: ReLU")
print(f"Output Layer: 10 Knoten (Ziffern 0-9)")

# Lade Test-Daten
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)

# Hole einen Batch von Test-Bildern
data, targets = next(iter(test_loader))

# Mache Vorhersagen
with torch.no_grad():
    outputs = model(data.view(-1, 28*28))
    _, predictions = torch.max(outputs, 1)

# Visualisiere 16 Bilder mit Vorhersagen
fig, axes = plt.subplots(4, 4, figsize=(12, 12))
fig.suptitle('MNIST Test-Vorhersagen', fontsize=16, fontweight='bold')

for i, ax in enumerate(axes.flat):
    # Zeige das Bild
    img = data[i].squeeze()  # Entferne Channel-Dimension
    ax.imshow(img, cmap='gray')

    # Echte Label und Vorhersage
    true_label = targets[i].item()
    pred_label = predictions[i].item()

    # Farbe: Gruen wenn richtig, Rot wenn falsch
    color = 'green' if true_label == pred_label else 'red'

    # Titel mit echter Label und Vorhersage
    ax.set_title(f'Echt: {true_label}\nVorhersage: {pred_label}',
                 color=color, fontweight='bold', fontsize=10)
    ax.axis('off')

plt.tight_layout()
plt.savefig('mnist_predictions.png', dpi=150, bbox_inches='tight')
print("\n=== VISUALISIERUNG ===")
print("Visualisierung gespeichert als 'mnist_predictions.png'")
print("Gruene Titel = Richtig erkannt")
print("Rote Titel = Falsch erkannt")
plt.show()

# Zusaetzliche Statistiken
correct = (predictions == targets).sum().item()
total = len(targets)
accuracy = 100 * correct / total

print(f"\n=== STATISTIK (dieser Batch) ===")
print(f"Richtig: {correct}/{total}")
print(f"Genauigkeit: {accuracy:.1f}%")
