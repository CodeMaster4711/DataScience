# -*- coding: utf-8 -*-
"""
ResNet Implementation for Imagenette
Based on "Deep Residual Learning for Image Recognition" (He et al., 2015)

Features:
- Basic Block (for ResNet18/34)
- Bottleneck Block (for ResNet50/101/152)
- Correct Initialization (Kaiming He + final layer -log(1/10))
- Adapted for Imagenette (10 classes, 160x160 input)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BasicBlock(nn.Module):
    """
    Basic ResNet Block (for ResNet18/34)

    Structure:
        Conv(3x3) → BN → ReLU → Conv(3x3) → BN → (+identity) → ReLU

    Expansion: 1 (output channels = input channels)
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        # First conv block
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # Second conv block
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        # First conv
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # Second conv
        out = self.conv2(out)
        out = self.bn2(out)

        # Skip connection (with projection if needed)
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """
    Bottleneck ResNet Block (for ResNet50/101/152)

    Structure:
        Conv(1x1) → BN → ReLU → Conv(3x3) → BN → ReLU → Conv(1x1) → BN → (+identity) → ReLU

    Expansion: 4 (output channels = 4 * base channels)
    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        # 1x1 conv (compress)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # 3x3 conv (spatial processing)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 1x1 conv (expand)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        # 1x1 conv
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 3x3 conv
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        # 1x1 conv
        out = self.conv3(out)
        out = self.bn3(out)

        # Skip connection (with projection if needed)
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """
    ResNet Architecture

    Args:
        block: BasicBlock or Bottleneck
        layers: List of number of blocks per stage [stage1, stage2, stage3, stage4]
        num_classes: Number of output classes (10 for Imagenette)
        dropout: Dropout probability before final FC layer
        correct_init: Use correct final layer initialization (-log(1/n))
    """

    def __init__(self, block, layers, num_classes=10, dropout=0.0, correct_init=True):
        super(ResNet, self).__init__()

        self.in_channels = 64
        self.num_classes = num_classes
        self.correct_init = correct_init

        # Initial conv layer (stem)
        # For Imagenette (smaller images), we use slightly smaller kernel
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet stages
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Global Average Pooling + Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Initialize weights
        self._init_weights()

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """Create a ResNet stage with multiple blocks"""
        downsample = None

        # If stride != 1 or channels change, we need projection for skip connection
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )

        layers = []
        # First block (potentially with downsample)
        layers.append(block(self.in_channels, out_channels, stride, downsample))

        # Update in_channels for next blocks
        self.in_channels = out_channels * block.expansion

        # Remaining blocks
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def _init_weights(self):
        """
        Initialize weights following best practices:
        - Kaiming He init for Conv layers (good for ReLU)
        - Constant 1 for BatchNorm weights, 0 for biases
        - Final layer: small std + zero bias for correct init loss
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Kaiming He initialization (good for ReLU)
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

            elif isinstance(m, nn.BatchNorm2d):
                # BN: weight=1, bias=0
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                if self.correct_init:
                    # V6a style: Correct initialization
                    # Small std → logits ≈ 0 → softmax ≈ 1/n → loss ≈ -log(1/n)
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)
                else:
                    # Default PyTorch init
                    nn.init.normal_(m.weight, 0, 0.1)
                    nn.init.normal_(m.bias, 0, 0.01)

    def forward(self, x):
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # ResNet stages
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Classifier
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x

    def get_num_params(self):
        """Count total number of parameters"""
        return sum(p.numel() for p in self.parameters())


# ============================================================================
# Factory Functions
# ============================================================================

def resnet18(num_classes=10, dropout=0.0, correct_init=True):
    """
    ResNet-18

    Config: [2, 2, 2, 2] blocks per stage
    Total layers: 18
    Parameters: ~11M (for ImageNet, slightly less for Imagenette)
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes, dropout, correct_init)
    return model


def resnet34(num_classes=10, dropout=0.0, correct_init=True):
    """
    ResNet-34

    Config: [3, 4, 6, 3] blocks per stage
    Total layers: 34
    Parameters: ~21M
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes, dropout, correct_init)
    return model


def resnet50(num_classes=10, dropout=0.0, correct_init=True):
    """
    ResNet-50

    Config: [3, 4, 6, 3] Bottleneck blocks per stage
    Total layers: 50
    Parameters: ~25M
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes, dropout, correct_init)
    return model


# ============================================================================
# Initialization Sanity Check (from v6a)
# ============================================================================

def check_initialization(model, loader, criterion, device, num_classes=10):
    """
    Verify initialization sanity check
    Expected: Loss @ init ≈ -log(1/num_classes) ≈ 2.3026 for 10 classes

    Returns dict with init metrics
    """
    model.eval()

    with torch.no_grad():
        # Get first batch
        data, target = next(iter(loader))
        data, target = data.to(device), target.to(device)

        # Forward pass
        output = model(data)
        init_loss = criterion(output, target).item()

        # Calculate accuracy
        pred = output.argmax(dim=1)
        correct = pred.eq(target).sum().item()
        init_accuracy = 100. * correct / target.size(0)

        # Check probability distribution
        probs = F.softmax(output, dim=1)
        avg_probs = probs.mean(dim=0)

        # Expected values
        expected_loss = np.log(num_classes)  # -log(1/10) = log(10) ≈ 2.3026
        expected_prob = 1.0 / num_classes
        expected_accuracy = 100.0 / num_classes

        # Check if close to expected
        loss_ok = abs(init_loss - expected_loss) < 0.5
        prob_ok = all(abs(p.item() - expected_prob) < 0.05 for p in avg_probs)

        init_check = {
            "init_loss": init_loss,
            "init_accuracy": init_accuracy,
            "expected_loss": expected_loss,
            "expected_accuracy": expected_accuracy,
            "loss_ok": loss_ok,
            "avg_prob_per_class": avg_probs.cpu().numpy().tolist(),
            "expected_prob": expected_prob,
            "prob_ok": prob_ok,
            "status": "✅ PASS" if (loss_ok and prob_ok) else "⚠️  WARNING"
        }

    model.train()
    return init_check


# ============================================================================
# Test/Debug
# ============================================================================

if __name__ == '__main__':
    # Test all 3 ResNet variants
    print("="*70)
    print("ResNet Implementation Test")
    print("="*70)

    # Create dummy input (batch_size=2, channels=3, height=160, width=160)
    x = torch.randn(2, 3, 160, 160)

    models = {
        "ResNet18": resnet18(correct_init=True),
        "ResNet34": resnet34(correct_init=True),
        "ResNet50": resnet50(correct_init=True),
    }

    for name, model in models.items():
        print(f"\n{name}:")
        print(f"  Parameters: {model.get_num_params():,}")

        # Forward pass
        output = model(x)
        print(f"  Output shape: {output.shape}")
        print(f"  Expected: torch.Size([2, 10])")

        # Check probabilities
        probs = F.softmax(output, dim=1)
        avg_prob = probs.mean(dim=0).mean().item()
        print(f"  Avg prob per class: {avg_prob:.4f} (expected: 0.1000)")

        assert output.shape == (2, 10), f"Wrong output shape: {output.shape}"
        print(f"  ✅ Test passed!")

    print("\n" + "="*70)
    print("All tests passed! ✅")
    print("="*70)
