import matplotlib.pyplot as plt
from fastai.data.external import URLs, untar_data
from torchvision import datasets
import random
from pathlib import Path
import os

# Globaler Daten-Ordner im Imagenette Verzeichnis
script_dir = Path(__file__).parent.parent
data_dir = script_dir / "data"
data_dir.mkdir(exist_ok=True)

# Fastai Home-Verzeichnis setzen
os.environ['FASTAI_HOME'] = str(data_dir)

# Dataset laden (wird in data_dir gespeichert)
path = untar_data(URLs.IMAGENETTE)
dataset = datasets.ImageFolder(root=f"{path}/train")

# 20 zufällige Bilder auswählen
num_images = 20
indices = random.sample(range(len(dataset)), num_images)

# Plot erstellen
fig, axes = plt.subplots(4, 5, figsize=(15, 12))
fig.suptitle('Imagenette Dataset - Sample Bilder', fontsize=16, fontweight='bold')

for idx, ax in enumerate(axes.flat):
    if idx < num_images:
        # Bild und Label laden
        img, label = dataset[indices[idx]]

        # Pixel-Dimensionen ermitteln
        width, height = img.size
        num_pixels = width * height

        # Klassennamen
        class_name = dataset.classes[label]

        # Bild anzeigen
        ax.imshow(img)
        ax.set_title(f'{class_name}\n{width}x{height} px\n({num_pixels:,} Pixel)',
                    fontsize=9)
        ax.axis('off')
    else:
        ax.axis('off')

plt.tight_layout()
plt.savefig(f"{path}/imagenette_visualization.png", dpi=150, bbox_inches='tight')
print(f"Visualisierung gespeichert unter: {path}/imagenette_visualization.png")
plt.show()

# Statistiken ausgeben
print(f"\nDataset Informationen:")
print(f"Anzahl Bilder im Trainingsset: {len(dataset)}")
print(f"Anzahl Klassen: {len(dataset.classes)}")
print(f"Klassen: {', '.join(dataset.classes)}")
