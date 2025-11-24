# V9 - Optimized CNN with Better Initialization

## Übersicht

Version 9 basiert auf V8, fügt aber wichtige Optimierungen hinzu, um bessere und reproduzierbare Ergebnisse zu erzielen.

## Implementierte Optimierungen

### 1. ✓ Fixed Random Seed (Reproduzierbarkeit)

```python
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

**Warum?** Garantiert, dass das Experiment bei jedem Lauf exakt die gleichen Ergebnisse liefert.

### 2. ✓ Simplify - Keine Data Augmentation

```python
transform = transforms.Compose([
    transforms.ToTensor(),  # Nur Tensor-Konvertierung, keine Augmentation
])
```

**Warum?** Baseline ohne Augmentation, um andere Optimierungen isoliert zu testen.

### 3. ✓ Verify Loss @ Init

```python
# Erwarteter Initial Loss für 10 Klassen: -log(1/10) ≈ 2.302
init_loss = criterion(outputs, batch_labels).item()
print(f"Initial Loss: {init_loss:.4f} (expected: ~2.302)")
```

**Warum?**
- Überprüft, ob die Initialisierung korrekt ist
- Bei 10 Klassen und gleichmäßiger Verteilung: Loss ≈ 2.302
- Weicht der Wert stark ab, ist etwas mit der Initialisierung falsch

### 4. ✓ Init Well - Optimierte Gewichtsinitialisierung

```python
def _initialize_weights(self):
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            # He/Kaiming initialization für ReLU
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)

    # Letzte Schicht: kleine Gewichte für gleichmäßige Vorhersagen
    final_layer = self.classifier[-1]
    nn.init.normal_(final_layer.weight, mean=0, std=0.01)
    nn.init.constant_(final_layer.bias, 0)
```

**Warum?**
- **He/Kaiming Init**: Optimal für ReLU-Aktivierungen, verhindert vanishing/exploding gradients
- **Kleine Gewichte in letzter Schicht**: Startet mit gleichmäßigen Vorhersagen (~10% pro Klasse)
- **Bias auf 0**: Fashion-MNIST ist ausgewogen (6.000 Bilder pro Klasse)

## Architektur (unverändert von V8)

```
Input: 1x28x28

Features:
  Conv1: 1 -> 32 (3x3, padding=same)
  ReLU + BatchNorm + MaxPool2d(2x2)
  -> 32x14x14

  Conv2: 32 -> 64 (3x3, padding=same)
  ReLU + BatchNorm + MaxPool2d(2x2)
  -> 64x7x7

Classifier:
  Flatten -> 3136
  FC1: 3136 -> 128 + ReLU + Dropout(0.4)
  FC2: 128 -> 64 + ReLU + Dropout(0.4)
  FC3: 64 -> 10 (Output)
```

## Hyperparameter

- **Batch Size**: 32
- **Learning Rate**: 0.001
- **Optimizer**: Adam
- **Epochs**: 50
- **Dropout**: 0.4
- **Seed**: 42

## Erwartete Verbesserungen

| Metrik | V8 | V9 (Ziel) |
|--------|----|-----------|
| Test Accuracy | ~92% | ~92-93% |
| Initial Loss | variabel | ~2.30 |
| Reproduzierbarkeit | ❌ | ✓ |
| Initialisierung | Standard | Optimiert |

## Training starten

```bash
cd fashionmnist/v9
python train.py
```

## Was kommt als Nächstes?

Weitere Optimierungen für höhere Accuracy:
- Learning Rate Scheduling
- Data Augmentation (nach Baseline-Tests)
- Tieferes Netzwerk oder mehr Filter
- Regularisierung (L2, Label Smoothing)
