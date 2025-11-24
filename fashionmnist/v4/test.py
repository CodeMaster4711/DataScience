
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

# 1. Konfiguration
BATCH_SIZE = 64
MODEL_FILE = "mein_modell.pth"

# Überprüfen, ob das Modell überhaupt existiert
if not os.path.exists(MODEL_FILE):
    print(f"FEHLER: Die Datei '{MODEL_FILE}' wurde nicht gefunden.")
    print("Bitte führe zuerst das Training aus und speichere das Modell mit torch.save()!")
    exit()

# 2. Test-Daten laden
# Wichtig: train=False für die Testdaten
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 3. Modell-Architektur neu erstellen
# Wir müssen exakt dieselbe Struktur wie im Training definieren
model = nn.Sequential(
    nn.Conv2d(1, 32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Dropout2d(0.25),

    nn.Conv2d(32, 64, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Dropout2d(0.25),

    nn.Flatten(),
    nn.Linear(64 * 7 * 7, 128),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(128, 10)
)

# 4. Gespeichertes "Wissen" laden
# Wir laden die Gewichte in die leere Architektur
model.load_state_dict(torch.load(MODEL_FILE, weights_only=True))
model.eval() # Wichtig: Schaltet das Modell in den Test-Modus

print("Modell geladen. Starte Berechnung der Genauigkeit...")

# 5. Testen
correct = 0
total = 0

with torch.no_grad(): # Keine Gradientenberechnung nötig (spart Speicher)
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# 6. Ergebnis berechnen und ausgeben
accuracy = 100 * correct / total

print("-" * 30)
print(f"Anzahl Testbilder: {total}")
print(f"Korrekt erkannt:   {correct}")
print(f"Genauigkeit:       {accuracy:.2f}%")
print("-" * 30)

# Kurze Bewertung des Ergebnisses
if accuracy < 15:
    print("-> Das Ergebnis ist sehr schlecht (Zufall). Wurde das Modell trainiert?")
elif accuracy < 80:
    print("-> Das Ergebnis ist okay, aber verbesserungswürdig.")
else:
    print("-> Das Ergebnis ist gut für ein lineares Modell!")