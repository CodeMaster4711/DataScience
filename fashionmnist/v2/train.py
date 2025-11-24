import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

input_dim = 28 * 28
output_dim = 10

BATCH_SIZE = 64
LEARNING_RATE = 0.01
NUM_EPOCHS = 10

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)) 
])

train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

model = nn.Sequential(
    nn.Flatten(),                    # Schritt 1: Aus 28x28 Matrix einen flachen Vektor machen (784 Pixel)
    nn.Linear(784, 128),             # Schritt 2: Erste versteckte Schicht (784 -> 128)
    nn.ReLU(),                       # Aktivierung für Nichtlinearität
    nn.Linear(128, 64),              # Schritt 3: Zweite versteckte Schicht (128 -> 64)
    nn.ReLU(),                       # Aktivierung für Nichtlinearität
    nn.Linear(64, 10)                # Schritt 4: Output-Schicht (64 -> 10 Klassen)
)

criterion = nn.CrossEntropyLoss() # Standard für Klassifizierung
optimizer = optim.SGD(model.parameters(), lr=0.01) # Stochastischer Gradientenabstieg

print("Starte Training mit simpler Input->Output Architektur...")

num_epochs = 3
gradients_per_layer = {name: [] for name, _ in model.named_parameters() if 'weight' in name}

for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        
        # A. Forward Pass (Vorhersage berechnen)
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # B. Backward Pass (Lernen)
        optimizer.zero_grad() # Alte Gradienten löschen
        loss.backward()       # Gradienten berechnen
        optimizer.step()      # Gewichte anpassen
        
        running_loss += loss.item()

        if i % 100 == 0:  # Reduziere Häufigkeit, um Speicher zu sparen
                for name, param in model.named_parameters():
                    if param.grad is not None and 'weight' in name:
                        gradients_per_layer[name].append(param.grad.clone().detach().cpu().numpy().flatten())        
    print(f'Epoche [{epoch+1}/{num_epochs}], Verlust: {running_loss / len(train_loader):.4f}')

print("Training abgeschlossen.")

# Gradienten visualisieren (nach Training)
fig, axes = plt.subplots(len(gradients_per_layer), 1, figsize=(10, 8))
for idx, (layer_name, grads) in enumerate(gradients_per_layer.items()):
    if grads:
        # Flatten alle Gradienten für Histogramm
        all_grads = [g for batch_grads in grads for g in batch_grads]
        axes[idx].hist(all_grads, bins=50, alpha=0.7)
        axes[idx].set_title(f'Gradienten-Verteilung für {layer_name}')
        axes[idx].set_xlabel('Gradient-Wert')
        axes[idx].set_ylabel('Häufigkeit')
plt.tight_layout()
plt.show()


torch.save(model.state_dict(), "mein_modell.pth")
print("Modell wurde als 'mein_modell.pth' gespeichert.")