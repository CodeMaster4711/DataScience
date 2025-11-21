import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

input_dim = 28 * 28 

input_layer = nn.Linear(input_dim, 128)


output_layer = nn.Linear(64, 10)  # 64 → 10 (muss zu vorheriger Layer passen!)


model = nn.Sequential(
    input_layer,
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    output_layer
)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(-1, 28*28)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')
torch.save(model.state_dict(), 'mnist_model.pth')
print("Modelltraining abgeschlossen und gespeichert als 'mnist_model.pth'")

