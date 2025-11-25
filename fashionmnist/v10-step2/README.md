# V10-Step2: Advanced Debugging & Monitoring für Fashion-MNIST

## Überblick

Version 10-Step2 implementiert systematisches ML-Debugging und umfassendes Monitoring für bessere und zuverlässigere Ergebnisse. Diese Version baut auf v9-step1 auf und fügt **9 wichtige Debugging- und Monitoring-Schritte** hinzu.

## Warum diese Version?

Die meisten ML-Projekte scheitern nicht an fehlender Komplexität, sondern an subtilen Bugs:
- Preprocessing-Fehler
- Falsche Initialisierung
- Batch-Abhängigkeiten
- Overfitting ohne es zu bemerken
- Modell lernt nichts von den Daten

**V10-Step2 findet diese Probleme BEVOR sie zu schlechten Resultaten führen!**

## Dateistruktur

```
v10-step2/
├── train.py              # Haupt-Training mit allen Features
├── debug_tools.py        # Baseline-Tests & Debugging
├── metrics.py            # Erweiterte Metriken
├── visualization.py      # Visualisierungsfunktionen
├── README.md            # Diese Datei
└── outputs/             # Generierte Plots und Modelle
    ├── human_baseline_samples.png
    ├── input_visualization.png
    ├── overfit_one_batch.png
    ├── confusion_matrix.png
    ├── train_vs_test_comparison.png
    ├── misclassifications.png
    ├── prediction_dynamics.png
    ├── training_curves.png
    ├── batch_level_loss.png
    └── best_model_v10.pth
```

## Die 9 Debugging-Schritte

### 🛠️ Setup und Initialisierung (aus v9-step1)

**✓ Fixed Random Seed**
- Alle Seeds gesetzt: Python, NumPy, PyTorch, CUDA
- Garantiert reproduzierbare Ergebnisse
- Wichtig für Bug-Fixing und Experimente

**✓ Simplify (Keine Data Augmentation)**
- Nur `ToTensor()` - keine Augmentation
- Reduziert potenzielle Fehlerquellen
- Baseline ohne Tricks

**✓ Verify Loss @ Init**
- Prüft initialen Loss: sollte ~2.302 sein für 10 Klassen
- Formel: `-log(1/10) ≈ 2.302`
- Weicht Wert stark ab → Initialisierung falsch!

**✓ Init Well (Gute Initialisierung)**
- He/Kaiming Init für Conv & FC-Layer (optimal für ReLU)
- Letzte Schicht: kleine Gewichte (std=0.01)
- Bias auf 0 (ausgewogener Datensatz)

### 🔍 Baselines und Debugging (NEU in v10-step2)

**1. Human Baseline Test**
```python
human_baseline_test(test_dataset, num_samples=16)
```
- Zeigt 16 zufällige Test-Bilder
- Dokumentiert erwartete menschliche Accuracy (~95%+)
- Visualisierung: `outputs/human_baseline_samples.png`
- **Warum?** Model sollte besser als Zufallsraten sein, aber muss nicht besser als Menschen sein

**2. Input-Independent Baseline**
```python
baseline_acc = test_input_independent_baseline(train_loader, test_loader, device)
```
- Trainiert Modell das Input IGNORIERT (alles auf 0 gesetzt)
- Sollte nur ~10% Accuracy erreichen (random guessing)
- **Kritischer Test:** Falls echtes Modell NICHT besser → Bug! Modell lernt nichts von Bildern!

**3. Overfit One Batch**
```python
overfit_loss, overfit_acc = overfit_one_batch(MyNN, device, train_dataset, batch_size=8)
```
- Versucht 8 Bilder perfekt zu memorieren
- Ziel: 100% Accuracy, Loss → 0
- **Falls nicht möglich:** Bug im Modell oder Training-Loop!
- Visualisierung: `outputs/overfit_one_batch.png`

### 📈 Training und Monitoring (NEU in v10-step2)

**4. Verify Decreasing Training Loss**
```python
plot_batch_level_loss(batch_losses)
```
- Trackt Loss für JEDEN Batch (nicht nur Epoch-Average)
- Zeigt Instabilitäten, Spikes, Plateaus früh
- Moving Average für smoothen Trend
- Visualisierung: `outputs/batch_level_loss.png`

**5. Add Significant Digits to Your Eval**
```python
metrics = compute_detailed_metrics(model, test_loader, device)
```
- Evaluation über KOMPLETTES Test-Set (10.000 Bilder)
- Confusion Matrix für alle 10 Klassen
- Per-Class Accuracy
- **Identifiziert:** Welche Klassen sind am schwersten?
- Visualisierung: `outputs/confusion_matrix.png`

**6. Human Interpretable Metrics**
```python
print_detailed_metrics(metrics, "Test")
compare_train_test_metrics(train_metrics, test_metrics)
```
- Accuracy, Precision, Recall, F1-Score
- Top-3 und Top-5 Accuracy
- Train vs Test Vergleich (Overfitting Detection)
- Per-Class Breakdown mit Visual Bars
- Visualisierung: `outputs/train_vs_test_comparison.png`

**7. Visualize Just Before the Net**
```python
visualize_input_before_net(train_loader)
```
- Zeigt Bilder EXAKT wie sie ins Netzwerk gehen
- Prüft Shape, Dtype, Min/Max, Mean, Std
- **Findet:** Preprocessing-Fehler, falsche Normalisierung
- Visualisierung: `outputs/input_visualization.png`

**8. Visualize Prediction Dynamics**
```python
pred_tracker = PredictionDynamicsTracker(fixed_batch_x, fixed_batch_y, device)
pred_tracker.update(model, epoch)
pred_tracker.plot()
```
- Trackt Vorhersagen für festen Batch über Training
- Zeigt Confidence-Entwicklung
- Heatmap: Wie ändern sich Vorhersagen?
- **Insights:** Stabilität des Lernprozesses
- Visualisierung: `outputs/prediction_dynamics.png`

**9. Use Backprop to Chart Dependencies**
```python
verify_batch_independence(model, device, test_loader)
```
- Loss nur für Bild `i` berechnen
- Backprop durchführen
- **Prüft:** Gradient nur für Bild `i` ≠ 0, alle anderen = 0
- **Findet:** Ungewollte Batch-Abhängigkeiten (Vektorisierungs-Bugs)

## Training starten

```bash
cd fashionmnist/v10-step2
python train.py
```

### Debug Flags anpassen

In [train.py](train.py) kannst du einzelne Tests deaktivieren:

```python
RUN_HUMAN_BASELINE = True              # Test 1
RUN_INPUT_INDEPENDENT_BASELINE = True  # Test 2
RUN_OVERFIT_TEST = True                # Test 3
RUN_BATCH_INDEPENDENCE_TEST = True     # Test 9
```

## Was wird ausgegeben?

### Console Output

1. **Setup Info:** Device, Config, Model Parameters
2. **Debugging Tests:**
   - Human Baseline Samples
   - Input-Independent Baseline Accuracy (~10%)
   - Input Visualization Stats
   - Loss@Init Verification (~2.302)
   - Overfit One Batch Results (should reach 100%)
3. **Training Progress:** Epoch-by-Epoch Loss & Accuracy
4. **Post-Training Analysis:**
   - Detailed Metrics (Train & Test)
   - Per-Class Accuracy
   - Train-Test Comparison
   - Batch Independence Test
5. **Final Summary:** Best Accuracy, Improvements, Saved Files

### Visualizations (in `outputs/`)

| Datei | Beschreibung |
|-------|--------------|
| `human_baseline_samples.png` | 16 Test-Bilder für menschliche Bewertung |
| `input_visualization.png` | Bilder direkt vor dem Netzwerk |
| `overfit_one_batch.png` | Overfitting-Fortschritt auf 8 Bildern |
| `confusion_matrix.png` | Absolute & normalisierte Confusion Matrix |
| `train_vs_test_comparison.png` | Side-by-side Metriken-Vergleich |
| `misclassifications.png` | Top-16 "sicherste" Fehler |
| `prediction_dynamics.png` | Vorhersage-Entwicklung über Training |
| `training_curves.png` | Loss & Accuracy Kurven |
| `batch_level_loss.png` | Batch-Level Loss (detailliert) |
| `best_model_v10.pth` | Bestes gespeichertes Modell |

## Erwartete Ergebnisse

| Metrik | Ziel | Bedeutung |
|--------|------|-----------|
| Test Accuracy | ~93-94% | Hauptziel |
| Loss @ Init | ~2.30 ± 0.3 | Korrekte Initialisierung |
| Input-Independent Baseline | ~10% | Modell lernt von Daten |
| Overfit One Batch | 100% | Modell-Kapazität ausreichend |
| Train-Test Gap | < 5% | Kein starkes Overfitting |
| Batch Independence | Pass | Keine Cross-Batch Bugs |

## Debugging-Workflow

Wenn etwas schiefgeht, folge diesem Workflow:

1. **Loss @ Init falsch?** (≠ ~2.302)
   - → Prüfe Initialisierung in `_initialize_weights()`
   - → Prüfe letzte Schicht

2. **Input-Independent Baseline zu gut?** (> 15%)
   - → Bug: Modell nutzt nicht die Bilder!
   - → Prüfe Forward-Pass

3. **Overfit One Batch schlägt fehl?** (< 100%)
   - → Bug im Modell oder Training-Loop
   - → Prüfe Loss-Berechnung, Optimizer

4. **Batch Independence fehlgeschlagen?**
   - → Cross-Batch Abhängigkeiten
   - → Prüfe BatchNorm, Custom Layers

5. **Input Visualization sieht falsch aus?**
   - → Preprocessing-Fehler
   - → Prüfe Transforms, Normalisierung

6. **Train-Test Gap > 10%?**
   - → Starkes Overfitting
   - → Füge Regularisierung hinzu (Dropout, Weight Decay)

7. **Per-Class Accuracy sehr ungleich?**
   - → Einige Klassen sind schwierig
   - → Prüfe Class Imbalance, sammle mehr Daten

## Next Steps

Nach v10-step2 kannst du optimieren:

- **Data Augmentation:** RandomHorizontalFlip, Rotation
- **Learning Rate Scheduling:** CosineAnnealing, ReduceLROnPlateau
- **Tieferes Netzwerk:** Mehr Conv-Layer, ResNet Blocks
- **Regularisierung:** Label Smoothing, Mixup, CutMix
- **Optimizer-Tuning:** SGD + Momentum, AdamW

## Architektur

Aktuelles Modell (MyNN):
```
Input: 1x28x28

Features:
  Conv1: 1 -> 32 (3x3, padding=same) + ReLU + BN + MaxPool
  Conv2: 32 -> 64 (3x3, padding=same) + ReLU + BN + MaxPool

Classifier:
  Flatten: 64x7x7 = 3136
  FC1: 3136 -> 128 + ReLU + Dropout(0.4)
  FC2: 128 -> 64 + ReLU + Dropout(0.4)
  FC3: 64 -> 10 (Output)

Parameters: ~420K
```

## Vergleich mit vorherigen Versionen

| Version | Features | Test Accuracy | Debugging |
|---------|----------|---------------|-----------|
| v1 | Basic CNN | ~88% | ❌ |
| v4 | Deep CNN + BN | ~93% | ❌ |
| v6 | ResNet + Advanced | ~94% | ❌ |
| v9-step1 | Fixed Seed + Init | ~92% | ⚠️ Basis |
| **v10-step2** | **Full Debugging** | **~93-94%** | **✓ Vollständig** |

## Wichtigste Erkenntnisse

1. **Debugging > Architektur:** Die meisten Bugs sind subtil und werden durch bessere Architektur NICHT behoben
2. **Baselines sind kritisch:** Ohne Baselines weißt du nicht ob dein Modell wirklich lernt
3. **Visualisierung rettet dich:** Die meisten Bugs sind durch Visualisierung sofort sichtbar
4. **Reproduzierbarkeit ist Pflicht:** Ohne Fixed Seed kannst du keine Bugs finden
5. **Test auf allen Ebenen:** Batch-Level, Epoch-Level, Per-Class - jede Perspektive hilft

## Häufige Fehler (die v10-step2 findet!)

- ❌ Initialisierung falsch → Loss@Init ≠ 2.302
- ❌ Preprocessing-Fehler → Input Visualization zeigt es
- ❌ Modell zu klein → Overfit One Batch schlägt fehl
- ❌ Batch-Abhängigkeiten → Backprop Test findet sie
- ❌ Modell lernt nicht → Input-Independent Baseline gleich gut
- ❌ Overfitting → Train-Test Vergleich zeigt großen Gap

## Lizenz

Educational purposes only.

## Kontakt

Bei Fragen oder Bugs öffne ein Issue im Repository.

---

**Happy Debugging! 🐛🔍**
