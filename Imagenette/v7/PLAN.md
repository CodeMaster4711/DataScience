# V7 Plan: ResNet Implementation + W&B Tracking

## 🎯 Ziel

Implementiere **ResNet Architektur** (ResNet18/34/50) mit vollständigem W&B Tracking und vergleiche gegen v6a (82.90% test acc, DeeperCNN).

**Kernfrage**: Ist die bewährte ResNet Architektur besser als unsere custom DeeperCNN?

---

## 📋 Implementation (✅ COMPLETED)

### 1. ResNet Architektur ✅
**Erstellt**: `v7/models/resnet.py`

**Features**:
- ✅ BasicBlock für ResNet18/34
- ✅ Bottleneck für ResNet50
- ✅ Proper Skip Connections mit Projection Layers
- ✅ Batch Normalization
- ✅ Korrekte Initialisierung (Kaiming He + final layer -log(1/10))
- ✅ Angepasst für Imagenette (160x160 input, 10 classes)

**Parameter Counts**:
- ResNet18: **11,181,642** params
- ResNet34: **21,289,802** params
- ResNet50: **23,528,522** params

vs. v6a DeeperCNN: **3,080,674** params

### 2. Training Scripts ✅
**Erstellt**:
- `v7/v7a/train.py` - ResNet18 (11M params)
- `v7/v7b/train.py` - ResNet34 (21M params)
- `v7/v7c/train.py` - ResNet50 (25M params)

**Basis**: v6a (beste Version mit korrekter Init)

**Änderungen vs v6a**:
| Hyperparameter | v6a | v7 | Begründung |
|---------------|-----|-----|-------------|
| Batch Size | 32 | **64** | ResNet profitiert von größeren Batches |
| Dropout | 0.25 | **0.0** | BN ersetzt teilweise Dropout |
| Weight Decay | 2e-4 | **1e-4** | BN hilft bei Regularization |
| Learning Rate | 0.001 | 0.001 | Conservative start |
| Architecture | DeeperCNN | **ResNet** | Proven architecture |

**Beibehalten**:
- ✅ OneCycleLR Scheduler (smooth LR curve)
- ✅ MixUp Augmentation (α=0.2)
- ✅ Label Smoothing (0.1)
- ✅ 100 Epochs + Early Stopping (patience=20)
- ✅ Korrekte Initialisierung (-log(1/10))
- ✅ 40% Dataset

---

## 📊 W&B Integration

### Dashboard Metriken

**Alle v6a Metriken beibehalten**:
- train/loss, train/accuracy
- val/loss, val/accuracy
- learning_rate, grad_norm
- train_val_gap
- init/loss, init/accuracy
- test/accuracy (final)

**Zusätzlich für ResNet**:
- Modell: `num_params` (Parameter Count)
- Architecture: `"ResNet18"` / `"ResNet34"` / `"ResNet50"`
- Version: `"v7a"` / `"v7b"` / `"v7c"`

### W&B Dashboard URL
```
http://localhost:8080/codemaster4711/imagenette-training
```

**Runs**:
- v6a-correct-init (Baseline: 82.90% test acc)
- v7a-correct-init (ResNet18)
- v7b-correct-init (ResNet34)
- v7c-correct-init (ResNet50)

---

## 🔬 Geplante Experimente

### Experiment 1: Architecture Comparison
**Frage**: Welche ResNet Variante ist am besten?

**Setup**:
```bash
# Train all 3 in parallel
cd v7a && python3 train.py &
cd v7b && python3 train.py &
cd v7c && python3 train.py &
wait
```

**Erwartete Ergebnisse**:
- **ResNet18** (11M): ~83-84% (ähnlich v6a, mehr params)
- **ResNet34** (21M): ~84-85% (**beste Balance**, +1-2% vs v6a)
- **ResNet50** (25M): ~82-84% (evtl. Overfitting wegen 25M params bei nur 40% data)

**Metrics zu vergleichen**:
- Test Accuracy (wichtigst!)
- Train-Val Gap (Overfitting)
- Training Time
- Parameters / Performance Trade-off

### Experiment 2: ResNet vs. Custom Architecture
**Frage**: ResNet vs. DeeperCNN?

**Setup**:
- Beste ResNet Variante (wahrscheinlich v7b)
- vs. v6a (DeeperCNN, 82.90% test acc)

**Comparison Metriken**:
| Metric | v6a (DeeperCNN) | v7b (ResNet34) | Winner |
|--------|-----------------|----------------|--------|
| Test Accuracy | 82.90% | ? | ? |
| Parameters | 3.1M | 21.3M | v6a |
| Training Time | ~23min | ? | ? |
| Train-Val Gap | -5.68% | ? | ? |
| Convergence Speed | ~100 epochs | ? | ? |

### Experiment 3: Hyperparameter Optimization (Optional)
Falls ResNet nicht überzeugt:

**W&B Sweep Config**:
```yaml
program: train.py
method: bayes
metric:
  name: val/accuracy
  goal: maximize
parameters:
  lr:
    min: 0.0005
    max: 0.003
  weight_decay:
    min: 5e-5
    max: 5e-4
  batch_size:
    values: [32, 64, 96]
  dropout:
    values: [0.0, 0.05, 0.1]
```

---

## 🚀 Training Anleitung

### Prerequisites

1. **W&B Server läuft**:
   ```bash
   docker ps | grep wandb
   # Sollte wandb-local Container zeigen
   ```

2. **Logged in**:
   ```bash
   wandb whoami
   # Sollte: codemaster4711 zeigen
   ```

### Training starten

**Option 1: Einzeln (sequential)**:
```bash
# v7a (ResNet18)
cd /Users/cedricstillecke/Documents/CloudExplain/DataScienceTutorial/Imagenette/v7/v7a
python3 train.py

# v7b (ResNet34)
cd ../v7b
python3 train.py

# v7c (ResNet50)
cd ../v7c
python3 train.py
```

**Option 2: Parallel** (empfohlen, wenn genug RAM/GPU):
```bash
cd /Users/cedricstillecke/Documents/CloudExplain/DataScienceTutorial/Imagenette/v7

# Start all 3 in background
(cd v7a && python3 train.py) &
(cd v7b && python3 train.py) &
(cd v7c && python3 train.py) &

# Wait for all to finish
wait

echo "✅ All trainings complete!"
```

**Monitor**:
```bash
# W&B Dashboard
open http://localhost:8080/codemaster4711/imagenette-training

# Or watch logs
tail -f v7a/wandb/latest-run/logs/debug.log
```

---

## 📁 Ordnerstruktur

```
v7/
├── PLAN.md                    # Dieser Plan
├── models/
│   ├── __init__.py
│   └── resnet.py              # ResNet Implementation
├── v7a/                       # ResNet18
│   ├── train.py
│   └── normalize.py
├── v7b/                       # ResNet34
│   ├── train.py
│   └── normalize.py
├── v7c/                       # ResNet50
│   ├── train.py
│   └── normalize.py
└── FAZIT.md                   # (nach Experimenten)
```

---

## 💡 Erwartete Erkenntnisse

### Warum könnte ResNet besser sein?

1. **Skip Connections**:
   - Besserer Gradient Flow
   - Tiefere Architekturen möglich
   - Weniger Vanishing Gradients

2. **Batch Normalization**:
   - Stabileres Training
   - Höhere Learning Rates möglich
   - Weniger Overfitting

3. **Proven Architecture**:
   - ImageNet SOTA (2015)
   - Gut erforscht
   - Viele Best Practices verfügbar

### Warum könnte DeeperCNN besser sein?

1. **Effizienz**:
   - Nur 3M params (vs 11-25M)
   - Schnelleres Training
   - Weniger Overfitting bei kleinem Dataset (40%)

2. **Moderne Features**:
   - Depthwise Separable Convs (effizienter)
   - Squeeze-Excitation Blocks (attention)
   - Speziell für Imagenette optimiert

### 🎯 Meine Vorhersage

| Modell | Test Acc | Params | Train Time | Winner |
|--------|----------|--------|------------|--------|
| v6a (DeeperCNN) | 82.90% | 3.1M | ~23min | Effizienz ✅ |
| v7a (ResNet18) | ~83.5% | 11M | ~30min | - |
| v7b (ResNet34) | ~84.5% | 21M | ~45min | **Performance ✅** |
| v7c (ResNet50) | ~83.0% | 25M | ~60min | Overfitting ⚠️ |

**Erwartung**: ResNet34 (v7b) wird am besten sein (+1-2% vs v6a), aber mit deutlich mehr Parametern und längerer Trainingszeit.

**Trade-off**: Performance vs. Efficiency

---

## 📝 Next Steps (nach Training)

1. **W&B Dashboard analysieren**:
   - Run Comparison Table
   - Learning Curves
   - Train-Val Gap
   - Gradient Flow

2. **FAZIT.md schreiben**:
   - Ergebnisse dokumentieren
   - v6a vs v7a/b/c Vergleich
   - Lessons Learned
   - Empfehlung für v8

3. **Entscheidung**:
   - Wenn ResNet besser: v8 mit ResNet + Optimizations
   - Wenn DeeperCNN besser: v8 mit DeeperCNN + neue Features
   - Oder: Ensemble (v6a + beste ResNet)

---

## 🎓 Theoretischer Hintergrund

### ResNet Key Innovation

**Problem**: Vanishing Gradients bei tiefen Netzen

**Lösung**: Skip Connections
```
out = F.relu(conv(x) + x)  # Residual learning
```

**Vorteil**:
- Gradient kann direkt durch Skip Connection fließen
- Netzwerk lernt nur "Residuals" (Abweichungen von Identity)
- Identity Mapping ist trivial zu lernen (Weight=0)

### Batch Normalization Benefits

1. **Internal Covariate Shift Reduction**
2. **Higher Learning Rates** möglich
3. **Less Sensitive to Initialization**
4. **Regularization Effect** (ähnlich Dropout)

---

**Version**: v7 (v7a + v7b + v7c)
**Datum**: 2025-11-28
**Basierend auf**: v6a (82.90% test acc)
**Status**: ✅ Implementation Complete - Ready for Training!
