# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from fastai.data.external import URLs, untar_data
from pathlib import Path
import os
import numpy as np
from tqdm import tqdm

from normalize import get_train_transforms, get_val_transforms

# Globaler Daten-Ordner
script_dir = Path(__file__).parent.parent
data_dir = script_dir / "data"
data_dir.mkdir(exist_ok=True)
os.environ['FASTAI_HOME'] = str(data_dir)

# Hyperparameter
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 10
DATA_PERCENTAGE = 0.05  # 5% der Daten
IMG_SIZE = 224
NUM_CLASSES = 10  # Imagenette hat 10 Klassen

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Dataset laden
path = untar_data(URLs.IMAGENETTE)

# Datasets mit Transformationen
train_dataset_full = datasets.ImageFolder(
    root=f"{path}/train",
    transform=get_train_transforms()
)

val_dataset = datasets.ImageFolder(
    root=f"{path}/val",
    transform=get_val_transforms()
)

# 5% der Trainingsdaten ausw�hlen
num_train = len(train_dataset_full)
num_samples = int(num_train * DATA_PERCENTAGE)
indices = np.random.choice(num_train, num_samples, replace=False)
train_dataset = Subset(train_dataset_full, indices)

print(f"\nDataset Info:")
print(f"Total Training Bilder: {num_train}")
print(f"Verwendete Bilder (5%): {len(train_dataset)}")
print(f"Validation Bilder: {len(val_dataset)}")
print(f"Klassen: {train_dataset_full.classes}")

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)


# Einfaches Netz mit nur EINEM Layer (Linear Classifier)
class SimpleNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleNet, self).__init__()
        # Input: 3 Kan�le � 224 � 224 = 150528 Features
        # Output: 10 Klassen

        # Flatten Layer
        self.flatten = nn.Flatten()

        # Ein einziger Linear Layer
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        x = self.flatten(x)  # (batch, 3, 224, 224) � (batch, 150528)
        x = self.fc(x)       # (batch, 150528) � (batch, 10)
        return x


# Model initialisieren
input_size = 3 * IMG_SIZE * IMG_SIZE  # 3 � 224 � 224 = 150528
model = SimpleNet(input_size, NUM_CLASSES).to(device)

print(f"\nModel Architektur:")
print(model)
print(f"\nAnzahl Parameter: {sum(p.numel() for p in model.parameters()):,}")

# Loss und Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


def train_epoch(model, loader, criterion, optimizer, device):
    """Trainiert eine Epoche"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Training")
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistiken
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Progress bar update
        pbar.set_postfix({
            'loss': running_loss / (pbar.n + 1),
            'acc': 100. * correct / total
        })

    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device):
    """Validiert das Model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(loader, desc="Validation")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({
                'loss': running_loss / (pbar.n + 1),
                'acc': 100. * correct / total
            })

    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


# Training Loop
print(f"\n{'='*60}")
print(f"TRAINING STARTET")
print(f"{'='*60}")

best_val_acc = 0.0

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print("-" * 60)

    # Training
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)

    # Validation
    val_loss, val_acc = validate(model, val_loader, criterion, device)

    print(f"\nEpoch {epoch+1} Zusammenfassung:")
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")

    # Best model speichern
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), f"{script_dir}/v1/best_model.pth")
        print(f" Neues bestes Model gespeichert! (Val Acc: {val_acc:.2f}%)")

print(f"\n{'='*60}")
print(f"TRAINING ABGESCHLOSSEN")
print(f"{'='*60}")
print(f"Beste Validation Accuracy: {best_val_acc:.2f}%")
print(f"Model gespeichert unter: {script_dir}/v1/best_model.pth")
