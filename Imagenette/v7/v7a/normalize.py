# -*- coding: utf-8 -*-
import torch
import numpy as np
from torchvision import transforms

# ImageNet Normalisierungswerte
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

IMG_SIZE = 224

def get_train_transforms():
    """Starke Data Augmentation gegen Overfitting"""
    return transforms.Compose([
        transforms.Lambda(lambda x: x.convert('RGB') if x.mode != 'RGB' else x),
        transforms.Resize(256),
        transforms.RandomCrop(IMG_SIZE),

        # STARKE Augmentation gegen Overfitting
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(20),  # Erhöht von 15
        transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.85, 1.15)),  # Stärker
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),  # Stärker
        transforms.RandomGrayscale(p=0.1),  # NEU

        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),

        transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),  # Erhöht
    ])

def get_val_transforms():
    """Validation ohne Augmentation"""
    return transforms.Compose([
        transforms.Lambda(lambda x: x.convert('RGB') if x.mode != 'RGB' else x),
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def denormalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """Denormalisierung für Visualisierung"""
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    denorm_tensor = tensor * std + mean
    denorm_tensor = torch.clamp(denorm_tensor, 0, 1)
    return denorm_tensor


# ============================================================================
# MIXUP AUGMENTATION - V5
# ============================================================================
def mixup_data(x, y, alpha=0.2):
    """
    MixUp Augmentation für bessere Generalisierung

    Paper: "mixup: Beyond Empirical Risk Minimization" (Zhang et al., 2018)

    Args:
        x: Input batch [B, C, H, W]
        y: Target labels [B]
        alpha: Beta distribution parameter (default: 0.2)

    Returns:
        mixed_x: Mixed input
        y_a, y_b: Original labels for both samples
        lam: Mixing coefficient
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    MixUp Loss Funktion

    Kombiniert die Loss für beide gemischten Labels

    Args:
        criterion: Loss function (z.B. CrossEntropyLoss)
        pred: Model predictions
        y_a, y_b: Original labels
        lam: Mixing coefficient

    Returns:
        Mixed loss
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
