# -*- coding: utf-8 -*-
import torch
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
