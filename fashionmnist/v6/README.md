# FashionMNIST V6 - Ziel: 98% Accuracy

## Ausgangslage
- **V4**: 93.27%
- **V5**: 90.70%
- **V6 Ziel**: 98%

## Hauptverbesserungen für 98%

### 1. ResNet mit Skip Connections
```python
out = self.conv2(out)
out += self.shortcut(x)  # Skip Connection!
```
**Warum?** Gradienten können direkt durchfließen → besseres Training tiefer Netzwerke

### 2. Tieferes Netzwerk
- V5: 2 Blöcke (32→64 Filter)
- V6: 8 ResBlocks (64→128→256→512 Filter)
- **4x mehr Kapazität** zum Lernen komplexer Features

### 3. CutMix Augmentation
```python
# Mischt zwei Bilder durch Ausschneiden eines Rechtecks
x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
```
**Warum?** Modell lernt robustere Features, versteht Kontext besser

### 4. Label Smoothing
- Statt 100% sicher → 90% sicher
- Verhindert Overconfidence
- Bessere Generalisierung

### 5. Cosine Annealing mit Warm Restarts
```python
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10)
```
**Warum?** LR steigt und fällt periodisch → entkommt lokalen Minima

### 6. Test Time Augmentation (TTA)
- Testet Bild 5x mit verschiedenen Augmentationen
- Mittelt Vorhersagen
- **+0.5-1% Accuracy boost**

### 7. Nesterov Momentum
```python
optimizer = SGD(..., momentum=0.9, nesterov=True)
```
**Warum?** "Lookahead" Momentum → schnellere Konvergenz

### 8. Mehr Epochen
- V5: 10 Epochen
- V6: 25 Epochen
- Mehr Zeit zum Lernen der komplexen Features

## Architektur-Vergleich

| Feature | V5 | V6 |
|---------|----|----|
| Netzwerktyp | Simple CNN | ResNet |
| Tiefe | 2 Blocks | 8 ResBlocks |
| Filter | 32→64 | 64→128→256→512 |
| Parameter | 871K | ~11M |
| Skip Connections | ❌ | ✅ |
| Augmentation | Basic | Basic + CutMix |
| Label Smoothing | ❌ | ✅ |
| TTA | ❌ | ✅ |
| Epochen | 10 | 25 |

## Training starten

```bash
cd fashionmnist/v6
python train.py
```

**Dauer**: ~10-15 Minuten auf Apple Silicon GPU

## Testen (mit TTA)

```bash
python test.py
```

Dies testet das Modell:
1. Standard (wie trainiert)
2. Mit TTA (5 Augmentationen)

## Erwartete Ergebnisse

- **Standard**: 94-96%
- **Mit TTA**: 95-98%

## Warum ist 98% schwierig?

FashionMNIST hat einige schwierige Klassen:
- Shirt vs T-Shirt vs Pullover (sehr ähnlich)
- Sandalen vs Sneaker (Überlappung)
- Coat vs Pullover (manchmal schwer zu unterscheiden)

Die letzten 2-3% benötigen:
- Sehr tiefe Netzwerke
- Viel Training
- Fortgeschrittene Techniken (die wir jetzt haben!)

## Wenn 98% nicht erreicht wird

Weitere Möglichkeiten:
1. **Mehr Epochen** (50+)
2. **Größeres Modell** (mehr ResBlocks)
3. **Ensemble** (mehrere Modelle kombinieren)
4. **AutoAugment** (learned augmentation policies)
5. **Knowledge Distillation**

Aber mit V6 sollten wir sehr nah an 98% kommen!
