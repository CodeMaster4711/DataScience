# -*- coding: utf-8 -*-
import torch
from torchvision import transforms

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

        # 4. Data Augmentation
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),

        # 5. Zu Tensor konvertieren (0-255 → 0-1, HWC → CHW)
        transforms.ToTensor(),

        # 6. Normalisierung mit ImageNet mean/std
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),

        # 7. Random Erasing (Cutout-like)
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
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
