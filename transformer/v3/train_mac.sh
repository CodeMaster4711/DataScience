#!/bin/bash
# Mac M3 Extreme Memory Mode Training Script
# This allows MPS to use swap memory if needed

echo "🚀 Starting Transformer v3 Training - Mac M3 Extreme Mode"
echo "=================================================="
echo ""
echo "⚠️  WARNING: This will use swap memory if RAM runs out"
echo "    Your Mac might slow down temporarily"
echo ""

# Set MPS environment variables for extreme memory mode
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0  # Allow unlimited MPS memory
export PYTORCH_MPS_LOW_WATERMARK_RATIO=0.0   # No low watermark

# Reduce memory fragmentation
export PYTORCH_MPS_PREFER_METAL=1

# Clear any existing MPS cache
echo "Clearing system memory cache..."
sudo purge 2>/dev/null || echo "Note: Run 'sudo purge' manually if training fails"

echo ""
echo "Environment variables set:"
echo "  PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 (unlimited)"
echo "  PYTORCH_MPS_LOW_WATERMARK_RATIO=0.0"
echo ""

# Run training
python train.py

echo ""
echo "Training completed or interrupted"
