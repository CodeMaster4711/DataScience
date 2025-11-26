import torch
from torchvision import transforms
from PIL import Image

# ImageNet Normalisierungswerte (Standard für vortrainierte Modelle)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Standard Bildgröße
IMG_SIZE = 224

def get_train_transforms():
    """
    Training Transformationen mit Data Augmentation
    """
    return transforms.Compose([
        # 1. RGB Konvertierung (falls nötig)
        transforms.Lambda(lambda x: x.convert('RGB') if x.mode != 'RGB' else x),

        # 2. Resize auf etwas größere Größe
        transforms.Resize(256),

        # 3. Random Crop für Data Augmentation
        transforms.RandomCrop(IMG_SIZE),

        # 4. Optional: Random Horizontal Flip
        transforms.RandomHorizontalFlip(p=0.5),

        # 5. Zu Tensor konvertieren (0-255 → 0-1, HWC → CHW)
        transforms.ToTensor(),

        # 6. Normalisierung mit ImageNet mean/std
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def get_val_transforms():
    """
    Validation/Test Transformationen ohne Data Augmentation
    """
    return transforms.Compose([
        # 1. RGB Konvertierung (falls nötig)
        transforms.Lambda(lambda x: x.convert('RGB') if x.mode != 'RGB' else x),

        # 2. Resize auf 256x256
        transforms.Resize(256),

        # 3. Center Crop auf 224x224
        transforms.CenterCrop(IMG_SIZE),

        # 4. Zu Tensor konvertieren (0-255 → 0-1, HWC → CHW)
        transforms.ToTensor(),

        # 5. Normalisierung mit ImageNet mean/std
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def get_basic_transforms():
    """
    Basis Transformationen ohne Normalisierung (für Visualisierung)
    """
    return transforms.Compose([
        transforms.Lambda(lambda x: x.convert('RGB') if x.mode != 'RGB' else x),
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor()
    ])

def denormalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """
    Rückgängig machen der Normalisierung für Visualisierung

    Args:
        tensor: Normalisierter Tensor (C, H, W)
        mean: Mean-Werte für Denormalisierung
        std: Std-Werte für Denormalisierung

    Returns:
        Denormalisierter Tensor im Bereich [0, 1]
    """
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)

    # Denormalisierung: x_original = x_normalized * std + mean
    denorm_tensor = tensor * std + mean

    # Auf [0, 1] clippen
    denorm_tensor = torch.clamp(denorm_tensor, 0, 1)

    return denorm_tensor

def normalize_image(image_path, transform_type='val'):
    """
    Lädt und normalisiert ein einzelnes Bild

    Args:
        image_path: Pfad zum Bild
        transform_type: 'train', 'val', oder 'basic'

    Returns:
        Original PIL Image, Normalisierter Tensor
    """
    # Bild laden
    img = Image.open(image_path)

    # Transformation wählen
    if transform_type == 'train':
        transform = get_train_transforms()
    elif transform_type == 'val':
        transform = get_val_transforms()
    else:
        transform = get_basic_transforms()

    # Transformieren
    normalized = transform(img)

    return img, normalized

if __name__ == "__main__":
    print("Normalisierungs-Konfiguration:")
    print(f"Bildgröße: {IMG_SIZE}x{IMG_SIZE}")
    print(f"ImageNet Mean: {IMAGENET_MEAN}")
    print(f"ImageNet Std: {IMAGENET_STD}")
    print(f"\nErwartete Tensor-Form: (3, {IMG_SIZE}, {IMG_SIZE})")
    print(f"Erwarteter Wertebereich nach Normalisierung: ~[-2, 2]")
