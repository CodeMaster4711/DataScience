# Diffusion Models - Project Plan

## Übersicht

Dieses Projekt implementiert **Class-Conditional Diffusion Models** für Fashion MNIST, um Bilder basierend auf Klassen-Labels zu generieren (z.B. "Pullover", "T-Shirt", "Schuhe").

### Was sind Diffusion Models?

Diffusion Models sind generative Modelle, die lernen, Bilder zu generieren, indem sie einen Denoising-Prozess lernen:

1. **Forward Process (Training)**:
   - Echtes Bild → schrittweise Noise hinzufügen → Pures Noise
   - Über T=1000 Zeitschritte

2. **Reverse Process (Inference)**:
   - Pures Noise → schrittweise Noise entfernen → Generiertes Bild
   - Das Modell lernt, bei jedem Schritt den Noise vorherzusagen

3. **Class-Conditional**:
   - Das Modell wird mit Klassen-Labels konditioniert
   - Ermöglicht gezielte Generierung: "Zeige mir einen Pullover"

## Version 1 (v1) - Baseline DDPM

### Architektur

**U-Net mit Conditioning:**
- **Encoder**: 28×28 → 14×14 → 7×7
- **Decoder**: 7×7 → 14×14 → 28×28
- **Channels**: 64 → 128 → 256 → 512
- **ResNet Blocks**: 2 pro Resolution
- **Self-Attention**: Bei 7×7 Resolution
- **Skip Connections**: Encoder zu Decoder
- **Conditioning**: Time Embedding + Class Embedding via Adaptive Group Norm

**Komponenten:**
- `model.py`: U-Net Implementation (~15M Parameter)
- `diffusion.py`: DDPM Scheduler, Beta Schedule, Sampling
- `train.py`: Training Loop mit WandB Integration
- `sample.py`: Inference Script für Bildgenerierung

### Hyperparameter

**AKTUELLE VERSION (Fast - ~30 Min auf Mac):**
```python
# Model
base_channels = 32  # Schneller
channel_mults = [1, 2, 4]
num_res_blocks = 1  # Schneller
attention_resolutions = []  # Attention entfernt (sehr teuer!)
dropout = 0.1

# Diffusion
timesteps = 1000
beta_schedule = "cosine"

# Training
epochs = 50  # Für schnelles Training
batch_size = 256
lr = 2e-4
ema_decay = 0.9999
guidance_scale = 3.0
p_uncond = 0.1
```

**Alternative: Hohe Qualität (~25-30h auf Mac):**
```python
base_channels = 64
num_res_blocks = 2
attention_resolutions = [7]
epochs = 300
batch_size = 128
```

### Dataset

**Fashion MNIST:**
- 60,000 Training-Bilder
- 10,000 Test-Bilder
- 10 Klassen: T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot
- Größe: 28×28 Grayscale
- Normalisierung: [-1, 1] (wichtig für Diffusion!)

### Training-Anleitung

1. **Vorbereitung:**
   ```bash
   cd deffusion/v1
   ```

2. **WandB starten** (falls nicht läuft):
   ```bash
   cd ../..
   docker-compose up -d wandb
   # oder manuell: siehe WANDB_DOCKER.md
   ```

3. **Training starten:**
   ```bash
   python train.py
   ```

4. **Monitoring:**
   - Terminal: Progress Bar mit Loss
   - WandB Dashboard: http://localhost:8080/codemaster4711/diffusion-training
   - Samples werden alle 10 Epochs generiert

5. **Trainingszeit (Fast Version):**
   - **MPS (M-Serie Mac)**: ~10-15 Sek/Epoch → ~10-15 Minuten für 50 Epochs
   - **GPU (CUDA)**: ~5-8 Sek/Epoch → ~5-8 Minuten für 50 Epochs
   - **CPU**: ~1-2 Min/Epoch → ~1-2 Stunden für 50 Epochs

   **Alternative (Hohe Qualität):**
   - **MPS (M-Serie Mac)**: ~5-6 Min/Epoch → ~25-30 Stunden für 300 Epochs
   - **GPU (CUDA)**: ~15-30 Sek/Epoch → ~2-3 Stunden für 300 Epochs

### Inference-Anleitung

Nach dem Training kannst du Bilder generieren:

**1. Bestimmte Klasse generieren:**
```bash
python sample.py --class pullover --n_samples 16 --guidance_scale 3.0
```

**2. Alle Klassen generieren:**
```bash
python sample.py --all_classes --guidance_scale 5.0
```

**3. Schnelles Sampling mit DDIM:**
```bash
python sample.py --class sneaker --use_ddim --ddim_steps 50 --n_samples 32
```

**4. Einzelne Bilder speichern:**
```bash
python sample.py --class dress --n_samples 20 --save_individual
```

**Verfügbare Klassen:**
- `t-shirt_top` (0)
- `trouser` (1)
- `pullover` (2)
- `dress` (3)
- `coat` (4)
- `sandal` (5)
- `shirt` (6)
- `sneaker` (7)
- `bag` (8)
- `ankle_boot` (9)

### Wichtige Features

**1. Exponential Moving Average (EMA)**
- Kritisch für stabile, hochqualitative Samples
- Decay: 0.9999
- EMA-Weights werden für Sampling verwendet

**2. Classifier-Free Guidance**
- 10% der Trainings-Samples sind "unconditional"
- Guidance Scale = 3.0 bei Inference
- Verbessert die Konditionierung deutlich

**3. Adaptive Group Normalization**
- Konditionierung via Scale & Shift Parameter
- Kombiniert Time + Class Embeddings
- Eingefügt in jedem ResNet Block

**4. Self-Attention**
- Nur bei 7×7 Resolution (nicht bei 28×28, zu teuer)
- Verbessert Bildqualität
- Multi-Head Attention mit 4 Heads

### Erwartete Ergebnisse

**Nach 100 Epochs:**
- Erkennbare Formen
- Noch etwas noisy
- Klassen sind unterscheidbar

**Nach 200 Epochs:**
- Gute Bildqualität
- Klare Klassen-Konditionierung
- Wenig Noise

**Nach 300 Epochs:**
- Sehr gute Bildqualität
- Starke Klassen-Konditionierung
- Realistische Fashion-Items

**Metriken:**
- Loss sollte kontinuierlich fallen
- Visuelle Inspektion ist wichtiger als Loss-Wert
- FID (Fréchet Inception Distance): 20-40 ist gut

### Output-Files

Nach dem Training findest du:
- `best_model.pth`: Bestes Modell (nach Loss)
- `final_model.pth`: Finales Modell nach Training
- `checkpoint_epoch_XXX.pth`: Periodische Checkpoints (alle 50 Epochs)
- `samples_epoch_XXX.png`: Generierte Samples während Training
- `wandb/`: WandB Logs (offline)

Nach Inference:
- `samples/samples_grid.png`: Grid mit allen Samples
- `samples/individual/`: Einzelne Bilder (optional)

## Version 2 (v2) - Geplante Verbesserungen

### Ideen für v2:

1. **Höhere Auflösung**
   - 64×64 oder 128×128 (CIFAR-10, CelebA)
   - Mehr Channels und Attention-Layer

2. **Besseres Sampling**
   - DDIM standardmäßig
   - Weniger Steps (50 statt 1000)
   - DPM-Solver++ für noch schnelleres Sampling

3. **Conditioning Verbesserungen**
   - Text Conditioning (statt nur Klassen)
   - CLIP Embeddings
   - Cross-Attention statt AdaGN

4. **Architektur-Verbesserungen**
   - U-Net XL (mehr Parameter)
   - Transformer Blocks
   - Flash Attention

5. **Training-Optimierungen**
   - Mixed Precision Training (FP16)
   - Gradient Checkpointing
   - Multi-GPU Training

## Version 3 (v3) - Latent Diffusion

### Idee: Stable Diffusion Style

Statt im Pixel-Space zu arbeiten, trainiere im Latent-Space:

1. **VAE Encoder/Decoder**
   - Komprimiere Bilder in kleineren Latent-Space
   - 256×256 → 32×32 Latents

2. **Diffusion im Latent-Space**
   - Viel schneller zu trainieren
   - Bessere Skalierung zu hohen Auflösungen

3. **Text-to-Image**
   - CLIP Text Encoder
   - Cross-Attention Conditioning
   - Prompt: "A red pullover on white background"

## Technische Details

### Wie funktioniert DDPM?

**Training:**
```python
# 1. Nimm echtes Bild x_0
x_0 = real_image

# 2. Sample random timestep t ∈ [0, T-1]
t = random.randint(0, 1000)

# 3. Sample noise
noise = torch.randn_like(x_0)

# 4. Forward diffusion: füge Noise hinzu
x_t = sqrt(α_t) * x_0 + sqrt(1 - α_t) * noise

# 5. Predict noise mit Model
predicted_noise = model(x_t, t, class_label)

# 6. MSE Loss
loss = MSE(predicted_noise, noise)

# 7. Backprop & Update
```

**Sampling:**
```python
# 1. Start mit pure Noise
x = torch.randn(1, 1, 28, 28)

# 2. Iteriere von t=T bis t=0
for t in reversed(range(1000)):
    # Predict noise
    predicted_noise = model(x, t, class_label)

    # Remove noise
    x = denoise(x, predicted_noise, t)

    # Add small noise (außer bei t=0)
    if t > 0:
        x = x + small_noise

# 3. x ist jetzt das generierte Bild
```

### Beta Schedule

**Cosine Schedule (empfohlen):**
```python
alphas_cumprod = cos²(π/2 * (t/T + s)/(1 + s))
betas = 1 - alphas_cumprod[t] / alphas_cumprod[t-1]
```

Vorteile:
- Bessere Noise-Verteilung
- Weniger Noise am Anfang/Ende
- Bessere Bildqualität

**Linear Schedule (einfacher):**
```python
betas = linspace(0.0001, 0.02, T)
```

### Classifier-Free Guidance

**Idee:** Verbessere Conditioning durch "guided diffusion"

**Training:**
- 10% der Zeit: Trainiere ohne Klassen-Label (unconditional)
- 90% der Zeit: Trainiere mit Klassen-Label (conditional)

**Sampling:**
```python
# Predict mit und ohne Conditioning
noise_cond = model(x, t, class_label)
noise_uncond = model(x, t, uncond_token)

# Guided prediction
noise = noise_uncond + scale * (noise_cond - noise_uncond)
```

- Scale = 0: Nur unconditional (ignoriert Label)
- Scale = 1: Normal conditional
- Scale > 1: Überbetonung des Labels (bessere Konditionierung)
- Typisch: Scale = 3.0 bis 7.0

## Troubleshooting

### Problem: Samples sind nur Noise

**Lösung:**
- Trainiere länger (mindestens 100 Epochs)
- Checke Normalisierung: Bilder müssen in [-1, 1] sein
- Checke EMA: Nutze EMA-Weights für Sampling
- Guidance Scale erhöhen (z.B. 5.0)

### Problem: Training ist sehr langsam

**Lösung:**
- Nutze GPU statt CPU
- Reduziere Batch Size wenn OOM
- Nutze Mixed Precision Training (FP16)
- Reduziere num_res_blocks auf 1

### Problem: Samples sehen alle gleich aus

**Lösung:**
- Guidance Scale reduzieren (z.B. 1.0 oder 2.0)
- p_uncond erhöhen (mehr unconditional training)
- Dropout erhöhen (mehr Diversity)

### Problem: Klassen-Konditionierung funktioniert nicht

**Lösung:**
- Guidance Scale erhöhen (z.B. 5.0)
- p_uncond checken (sollte 0.1 sein)
- Checke Class Embedding Layer
- Checke ob uncond_token korrekt ist (= num_classes)

## Ressourcen

### Papers

1. **DDPM (Original):**
   - "Denoising Diffusion Probabilistic Models" (Ho et al., 2020)
   - https://arxiv.org/abs/2006.11239

2. **Improved DDPM:**
   - "Improved Denoising Diffusion Probabilistic Models" (Nichol & Dhariwal, 2021)
   - https://arxiv.org/abs/2102.09672
   - Cosine Beta Schedule

3. **DDIM (Faster Sampling):**
   - "Denoising Diffusion Implicit Models" (Song et al., 2021)
   - https://arxiv.org/abs/2010.02502

4. **Classifier-Free Guidance:**
   - "Classifier-Free Diffusion Guidance" (Ho & Salimans, 2022)
   - https://arxiv.org/abs/2207.12598

### Code

1. **lucidrains/denoising-diffusion-pytorch:**
   - Saubere, minimale Implementation
   - https://github.com/lucidrains/denoising-diffusion-pytorch

2. **openai/improved-diffusion:**
   - Offizielle OpenAI Implementation
   - https://github.com/openai/improved-diffusion

3. **huggingface/diffusers:**
   - Production-ready Library
   - https://github.com/huggingface/diffusers

## Status

- [x] v1 Implementation
  - [x] U-Net Model
  - [x] DDPM Scheduler
  - [x] Training Script
  - [x] Sampling Script
  - [x] WandB Integration
  - [ ] Training durchführen (300 Epochs)
- [ ] v2 Verbesserungen
- [ ] v3 Latent Diffusion

## Notizen

- Fashion MNIST ist ein guter Startpunkt (klein, schnell)
- Nach erfolgreicher v1 Implementation: Skalieren zu CIFAR-10 oder CelebA
- EMA ist kritisch - ohne EMA sind Samples sehr noisy
- Classifier-Free Guidance verbessert Konditionierung massiv
- Geduld: Diffusion Models brauchen lange zum Trainieren (200-300 Epochs)

---

**Erstellt:** 2025-12-03
**Letztes Update:** 2025-12-03
**Author:** Claude Code + Cedric
