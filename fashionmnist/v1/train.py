import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

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
    nn.Flatten(),       # Schritt 1: Macht aus der 28x28 Matrix einen flachen Vektor (784 Pixel)
    nn.Linear(784, 10)  # Schritt 2: Input (784) direkt auf Output (10 Klassen)
)

criterion = nn.CrossEntropyLoss() # Standard für Klassifizierung
optimizer = optim.SGD(model.parameters(), lr=0.01) # Stochastischer Gradientenabstieg

print("Starte Training mit simpler Input->Output Architektur...")

num_epochs = 3

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
        
    print(f'Epoche [{epoch+1}/{num_epochs}], Verlust: {running_loss / len(train_loader):.4f}')

print("Training abgeschlossen.")

torch.save(model.state_dict(), "mein_modell.pth")
print("Modell wurde als 'mein_modell.pth' gespeichert.")