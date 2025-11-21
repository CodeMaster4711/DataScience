"""
Fashion-MNIST Pixel-Analyse und Visualisierung
Zeigt die Pixelanzahl und rendert mehrere Bilder aus dem Test-Datensatz
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms

# Fashion-MNIST Label-Namen
LABEL_NAMES = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}


def load_fashion_mnist():
    """Lädt den Fashion-MNIST Datensatz mit PyTorch"""
    print("Lade Fashion-MNIST Datensatz mit PyTorch...")

    # Transform um Tensoren in NumPy Arrays zu konvertieren
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Lade Train und Test Datensätze
    train_dataset = datasets.FashionMNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.FashionMNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    # Konvertiere zu NumPy Arrays für einfachere Handhabung
    x_train = train_dataset.data.numpy()
    y_train = train_dataset.targets.numpy()
    x_test = test_dataset.data.numpy()
    y_test = test_dataset.targets.numpy()

    print(f"✓ Datensatz geladen")
    print(f"  Training: {x_train.shape[0]} Bilder")
    print(f"  Test: {x_test.shape[0]} Bilder")
    print(f"  PyTorch Version: {torch.__version__}")
    return (x_train, y_train), (x_test, y_test)


def analyze_image_pixels(image):
    """Analysiert die Pixelanzahl eines Bildes"""
    height, width = image.shape
    total_pixels = height * width

    print("\n" + "="*50)
    print("PIXEL-ANALYSE")
    print("="*50)
    print(f"Bildhöhe:      {height} Pixel")
    print(f"Bildbreite:    {width} Pixel")
    print(f"Gesamtpixel:   {total_pixels} Pixel")
    print(f"Bildform:      {image.shape}")
    print(f"Datentyp:      {image.dtype}")
    print(f"Wertebereich:  {image.min()} - {image.max()}")
    print("="*50)

    return total_pixels


def plot_image_grid(images, labels, num_rows=5, num_cols=5, title="Fashion-MNIST Beispiele"):
    """
    Rendert ein Grid von Bildern mit ihren Labels

    Args:
        images: Array von Bildern
        labels: Array von Labels
        num_rows: Anzahl der Zeilen
        num_cols: Anzahl der Spalten
        title: Titel der Visualisierung
    """
    num_images = num_rows * num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 12))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    for i in range(num_images):
        row = i // num_cols
        col = i % num_cols

        # Bild anzeigen
        axes[row, col].imshow(images[i], cmap='gray')
        axes[row, col].axis('off')

        # Label als Titel
        label_name = LABEL_NAMES[labels[i]]
        axes[row, col].set_title(f"{label_name}\n(Label: {labels[i]})",
                                  fontsize=9, pad=5)

    plt.tight_layout()
    return fig


def plot_label_distribution(labels, dataset_name="Test"):
    """Zeigt die Verteilung der Labels"""
    fig, ax = plt.subplots(figsize=(12, 6))

    unique, counts = np.unique(labels, return_counts=True)
    label_names = [LABEL_NAMES[i] for i in unique]

    bars = ax.bar(label_names, counts, color='skyblue', edgecolor='navy')
    ax.set_xlabel('Kategorie', fontsize=12)
    ax.set_ylabel('Anzahl', fontsize=12)
    ax.set_title(f'Label-Verteilung im {dataset_name}-Datensatz', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Werte auf den Balken anzeigen
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig


def show_examples_per_category(images, labels, samples_per_category=5):
    """Zeigt Beispiele für jede Kategorie"""
    fig, axes = plt.subplots(10, samples_per_category, figsize=(15, 20))
    fig.suptitle('Beispiele für jede Kategorie', fontsize=16, fontweight='bold')

    for category in range(10):
        # Finde Indizes für diese Kategorie
        indices = np.where(labels == category)[0][:samples_per_category]

        for i, idx in enumerate(indices):
            axes[category, i].imshow(images[idx], cmap='gray')
            axes[category, i].axis('off')

            if i == 0:
                axes[category, i].set_ylabel(LABEL_NAMES[category],
                                             fontsize=11, fontweight='bold', rotation=0,
                                             ha='right', va='center')

    plt.tight_layout()
    return fig


def main():
    """Hauptfunktion"""
    # Datensatz laden
    (x_train, y_train), (x_test, y_test) = load_fashion_mnist()

    # Pixel-Analyse für ein einzelnes Bild
    print("\n>>> Analysiere erstes Test-Bild...")
    analyze_image_pixels(x_test[0])

    # Zufällige Bilder aus dem Test-Set auswählen
    num_samples = 25
    random_indices = np.random.choice(len(x_test), num_samples, replace=False)
    sample_images = x_test[random_indices]
    sample_labels = y_test[random_indices]

    print(f"\n>>> Erstelle Visualisierungen...")

    # 1. Grid mit zufälligen Beispielen
    plot_image_grid(sample_images, sample_labels,
                    title="Fashion-MNIST: 25 zufällige Test-Bilder")

    # 2. Label-Verteilung
    plot_label_distribution(y_test, "Test")

    # 3. Beispiele pro Kategorie
    show_examples_per_category(x_test, y_test, samples_per_category=5)

    print("✓ Visualisierungen erstellt")
    print("\n>>> Fenster schließen zum Beenden...")
    plt.show()


if __name__ == "__main__":
    main()
