import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

BATCH_SIZE = 128
MODEL_FILE = "best_model_v5.pth"

if not os.path.exists(MODEL_FILE):
    print(f"FEHLER: '{MODEL_FILE}' nicht gefunden!")
    print("Führe zuerst 'python train.py' aus.")
    exit()

# Test-Daten
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Modell-Architektur (muss identisch zu train.py sein!)
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

# GPU Support - CUDA oder MPS (Apple Silicon)
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

model.load_state_dict(torch.load(MODEL_FILE, weights_only=True, map_location=device))
model = model.to(device)
model.eval()

print(f"Modell geladen auf: {device}\n")

# Testen
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total

print("="*60)
print(f"V5 CNN Test-Ergebnisse:")
print("="*60)
print(f"Testbilder:  {total}")
print(f"Korrekt:     {correct}")
print(f"Genauigkeit: {accuracy:.2f}%")
print("="*60)

if accuracy > 93.27:
    print(f"✓ Verbesserung vs V4: +{accuracy - 93.27:.2f}%")
elif accuracy > 90:
    print(f"✓ Sehr gut! (V4: 93.27%)")
else:
    print("⚠ Unter Erwartung")
