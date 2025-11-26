import matplotlib.pyplot as plt
from fastai.data.external import URLs, untar_data
from torchvision import datasets
import random
from pathlib import Path
import os
import numpy as np
from normalize import get_val_transforms, get_basic_transforms, denormalize, IMAGENET_MEAN, IMAGENET_STD

# Globaler Daten-Ordner im Imagenette Verzeichnis
script_dir = Path(__file__).parent.parent
data_dir = script_dir / "data"
data_dir.mkdir(exist_ok=True)

# Fastai Home-Verzeichnis setzen
os.environ['FASTAI_HOME'] = str(data_dir)

# Dataset laden
path = untar_data(URLs.IMAGENETTE)

# Dataset mit und ohne Normalisierung
dataset_raw = datasets.ImageFolder(root=f"{path}/train", transform=get_basic_transforms())
dataset_normalized = datasets.ImageFolder(root=f"{path}/train", transform=get_val_transforms())

# 12 zufällige Bilder auswählen
num_images = 12
indices = random.sample(range(len(dataset_raw)), num_images)

# Plot erstellen (4 Zeilen x 6 Spalten = 12 Bilder mit Vorher/Nachher)
fig, axes = plt.subplots(4, 6, figsize=(18, 12))
fig.suptitle('Bildnormalisierung: Vorher vs. Nachher', fontsize=16, fontweight='bold')

for idx, (row_idx, col_start) in enumerate([(i//3, (i%3)*2) for i in range(num_images)]):
    # Original (ohne Normalisierung)
    img_raw, label = dataset_raw[indices[idx]]

    # Normalisiert
    img_norm, _ = dataset_normalized[indices[idx]]

    # Denormalisieren für Visualisierung
    img_denorm = denormalize(img_norm)

    # Tensor zu NumPy Array (C, H, W) → (H, W, C)
    img_raw_np = img_raw.permute(1, 2, 0).numpy()
    img_denorm_np = img_denorm.permute(1, 2, 0).numpy()

    # Klassenname
    class_name = dataset_raw.classes[label]

    # Original anzeigen
    ax_original = axes[row_idx, col_start]
    ax_original.imshow(img_raw_np)

    # Bildstatistiken berechnen
    height, width = img_raw_np.shape[:2]
    pixels = height * width

    ax_original.set_title(f'Original\n{class_name}\n{width}x{height} ({pixels:,} px)',
                         fontsize=8)
    ax_original.axis('off')

    # Normalisiert anzeigen
    ax_norm = axes[row_idx, col_start + 1]
    ax_norm.imshow(img_denorm_np)

    # Statistiken des normalisierten Tensors
    mean_vals = img_norm.mean(dim=[1, 2]).numpy()
    std_vals = img_norm.std(dim=[1, 2]).numpy()

    ax_norm.set_title(f'Normalisiert\nMean: [{mean_vals[0]:.2f}, {mean_vals[1]:.2f}, {mean_vals[2]:.2f}]\nStd: [{std_vals[0]:.2f}, {std_vals[1]:.2f}, {std_vals[2]:.2f}]',
                     fontsize=8)
    ax_norm.axis('off')

plt.tight_layout()
plt.savefig(f"{script_dir}/v1/normalization_comparison.png", dpi=150, bbox_inches='tight')
print(f"Visualisierung gespeichert unter: {script_dir}/v1/normalization_comparison.png")
plt.show()

# Detaillierte Informationen ausgeben
print("\n" + "="*70)
print("NORMALISIERUNGS-DETAILS")
print("="*70)
print(f"\n1. GRÖSSEN-NORMALISIERUNG:")
print(f"   - Alle Bilder: Resize auf 256x256 → Center Crop auf 224x224")
print(f"   - Finale Größe: 224x224 Pixel")

print(f"\n2. FARB-NORMALISIERUNG:")
print(f"   - Alle Bilder werden zu RGB (3 Kanäle) konvertiert")
print(f"   - Format: (Kanäle, Höhe, Breite) = (3, 224, 224)")

print(f"\n3. PIXEL-WERT-NORMALISIERUNG:")
print(f"   - Schritt 1: 0-255 → 0.0-1.0 (Division durch 255)")
print(f"   - Schritt 2: Standardisierung mit ImageNet-Werten")
print(f"   - Mean pro Kanal: R={IMAGENET_MEAN[0]}, G={IMAGENET_MEAN[1]}, B={IMAGENET_MEAN[2]}")
print(f"   - Std pro Kanal:  R={IMAGENET_STD[0]}, G={IMAGENET_STD[1]}, B={IMAGENET_STD[2]}")

print(f"\n4. FINALES TENSOR-FORMAT:")
print(f"   - Form: torch.Size([3, 224, 224])")
print(f"   - Dtype: torch.float32")
print(f"   - Wertebereich: ~[-2.5, 2.5] (typisch nach Normalisierung)")

print(f"\n5. BATCH-FORMAT FÜR NEURONALES NETZ:")
print(f"   - Form: torch.Size([batch_size, 3, 224, 224])")
print(f"   - Beispiel mit Batch=32: torch.Size([32, 3, 224, 224])")

print("\n" + "="*70)
print(f"\nDataset-Informationen:")
print(f"Anzahl Bilder im Trainingsset: {len(dataset_raw)}")
print(f"Anzahl Klassen: {len(dataset_raw.classes)}")
print(f"Klassen: {', '.join(dataset_raw.classes)}")
print("="*70)
