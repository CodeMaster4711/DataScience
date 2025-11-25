# V11 - Modernes, Vereinfachtes CNN

## Was ist neu?

**Einfachheit**: Nur 1 Datei (`train.py`) statt komplexer Module

**Moderne PyTorch Best Practices** für höhere Accuracy:

### 1. Bessere CNN-Architektur

**Depthwise Separable Convolutions**
```python
# Normaler Conv: 3x3x64x128 = 73,728 Parameter
# Depthwise Separable: (3x3x64) + (1x1x64x128) = 8,768 Parameter
# ~9x weniger Parameter, ähnliche Performance!
```
- Effizienter (weniger Parameter)
- Schneller zu trainieren
- Verwendet in MobileNet, EfficientNet

**Squeeze-and-Excitation (SE) Blocks**
```python
class SqueezeExcitation(nn.Module):
    # Lernt: Welche Feature-Channels sind wichtig?
    # Boost wichtige Features, supprimiere unwichtige
```
- Adaptive Feature Recalibration
- +1-2% Accuracy fast "kostenlos"
- Verwendet in SENet, EfficientNet

**Residual Connections**
```python
out = conv(x) + x  # Skip connection
```
- Ermöglicht tieferes Training
- Besserer Gradientenfluss
- Aus ResNet

**Global Average Pooling**
```python
# Alt: Flatten + FC (viele Parameter)
# Neu: GAP + FC (weniger Parameter, weniger Overfitting)
x = F.adaptive_avg_pool2d(x, 1)
```

### 2. Bessere Training-Techniken

**Label Smoothing**
```python
# Statt: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
# Jetzt: [0.01, 0.01, 0.01, 0.91, 0.01, ...]
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```
- Weniger overconfident
- Bessere Generalisierung
- +0.5-1% Accuracy

**AdamW statt Adam**
```python
optimizer = AdamW(...)  # Bessere Weight Decay
```
- Sauberere Weight Regularisierung
- Bessere Konvergenz

**Cosine Annealing + Warmup**
```python
# Warmup: LR steigt 0 -> max (erste 5 Epochen)
# Cosine: LR sinkt smooth max -> 0 (restliche Epochen)
```
- Bessere Konvergenz
- Findet bessere Minima

**Gradient Clipping**
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```
- Verhindert explodierende Gradienten
- Stabileres Training

**Mixed Precision Training**
```python
with torch.cuda.amp.autocast():
    output = model(data)
```
- ~2x schneller auf GPU
- Weniger Memory
- Gleiche Accuracy

### 3. Bessere Data Augmentation

```python
transforms.RandomHorizontalFlip(p=0.5)     # Spiegeln
transforms.RandomRotation(15)               # Rotation ±15°
transforms.RandomAffine(..., translate=0.1) # Verschieben
```

## Architektur

```
Input: 1x28x28

Stem:
  Conv 1->64 + BN + ReLU

Stage 1 (28x28 -> 14x14):
  DepthwiseSeparableConv 64->64
  ResidualBlock(64) + SE
  MaxPool

Stage 2 (14x14 -> 7x7):
  DepthwiseSeparableConv 64->128
  ResidualBlock(128) + SE
  MaxPool

Stage 3 (7x7):
  DepthwiseSeparableConv 128->256
  ResidualBlock(256) + SE

Head:
  GlobalAveragePooling
  Dropout(0.3)
  Linear 256->10

Parameters: ~400K (effizient!)
```

## Training

```bash
cd fashionmnist/v11
python train.py
```

### Config anpassen

In `train.py`:
```python
class Config:
    batch_size = 64
    epochs = 60
    lr = 0.001
    dropout = 0.3
    label_smoothing = 0.1
    warmup_epochs = 5
    use_amp = True  # Mixed Precision
```

## Erwartete Ergebnisse

| Metrik | Wert |
|--------|------|
| Test Accuracy | **~94-95%** |
| Training Zeit | ~5-10 min (GPU) |
| Parameters | ~400K |
| Memory | ~1GB |

## Vergleich

| Version | Code | Features | Accuracy | Parameter |
|---------|------|----------|----------|-----------|
| v1 | Einfach | Basic CNN | ~88% | ~100K |
| v4 | Mittel | Deep CNN | ~93% | ~1M |
| v10-step2 | **Komplex** | Full Debug | ~93% | ~420K |
| **v11** | **Einfach** | **Modern** | **~94-95%** | **~400K** |

## Was macht v11 besser?

### Code
- ✓ Nur 1 File (nicht 4-5)
- ✓ ~300 Zeilen (nicht 1000+)
- ✓ Klare Struktur
- ✓ Gut kommentiert

### Performance
- ✓ Höhere Accuracy (+1-2%)
- ✓ Weniger Parameter
- ✓ Schnelleres Training (Mixed Precision)
- ✓ Bessere Generalisierung

### Moderne Praktiken
- ✓ Depthwise Separable Conv
- ✓ SE-Blocks
- ✓ Residual Connections
- ✓ Global Average Pooling
- ✓ Label Smoothing
- ✓ Cosine Annealing
- ✓ Warmup
- ✓ Gradient Clipping
- ✓ Mixed Precision

## Weitere Verbesserungen möglich

Falls du noch höher willst (~95-96%):

1. **Test-Time Augmentation (TTA)**
```python
# Mehrfache Vorhersagen mit leichten Augmentationen
# Durchschnitt bilden
```

2. **Model Ensembling**
```python
# Trainiere 3-5 Modelle
# Durchschnitt der Vorhersagen
```

3. **Cutout/CutMix**
```python
# Fortgeschrittene Augmentation
# Schneide Teile aus Bildern aus
```

4. **EfficientNet-Style Compound Scaling**
```python
# Skaliere depth, width, resolution zusammen
```

5. **Knowledge Distillation**
```python
# Trainiere großes Modell
# Destilliere in kleines Modell
```

## Output

Training zeigt:
- Epoch-by-Epoch: Train/Test Loss & Accuracy
- Learning Rate Schedule
- Best Model Speicherung
- 4 Plots:
  - Loss Curves
  - Accuracy Curves
  - LR Schedule
  - Train-Test Gap (Overfitting)

## Key Insights

**Warum ist v11 besser als v10?**

| Aspekt | v10-step2 | v11 |
|--------|-----------|-----|
| **Code** | 4 Files, 1000+ Zeilen | 1 File, ~300 Zeilen |
| **Fokus** | Debugging | Performance |
| **Lernkurve** | Steil | Flach |
| **Wartung** | Schwer | Einfach |
| **Accuracy** | ~93% | ~94-95% |

**Wann v10, wann v11?**
- v10: Wenn du Bugs suchst, System debuggen willst
- v11: Wenn du beste Performance mit sauberem Code willst

## Moderne PyTorch Patterns

### 1. Depthwise Separable Conv
```python
# Effizienter als normale Conv
depthwise = nn.Conv2d(C, C, 3, groups=C)  # Separate Channels
pointwise = nn.Conv2d(C, C_out, 1)        # Mix Channels
```

### 2. SE-Block
```python
# Channel Attention
gap = F.adaptive_avg_pool2d(x, 1)  # Squeeze
weights = sigmoid(fc(gap))          # Excitation
x = x * weights                     # Scale
```

### 3. Residual Connection
```python
# Skip Connection
out = f(x) + x
```

### 4. Global Average Pooling
```python
# Statt Flatten + große FC
x = F.adaptive_avg_pool2d(x, 1)  # -> Bx C x1x1
x = x.view(B, C)                  # -> BxC
```

## Lizenz

Educational purposes.
