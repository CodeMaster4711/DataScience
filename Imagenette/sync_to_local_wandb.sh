#!/bin/bash
#
# Synct W&B offline runs zum lokalen W&B Server
#

echo "======================================================================"
echo "W&B Offline → Local Server Sync"
echo "======================================================================"

# Setze lokalen W&B Server als Base URL
export WANDB_BASE_URL="http://localhost:8080"

echo ""
echo "🔗 W&B Base URL: $WANDB_BASE_URL"
echo ""

# Sync v6a
echo "📤 Syncing v6a offline run..."
cd v6a
wandb sync wandb/offline-run-*
cd ..

echo ""

# Sync v6b
echo "📤 Syncing v6b offline run..."
cd v6b
wandb sync wandb/offline-run-*
cd ..

echo ""
echo "======================================================================"
echo "✅ Sync Complete!"
echo "======================================================================"
echo ""
echo "🌐 Öffne im Browser: http://localhost:8080"
echo ""
echo "Dort siehst du das volle W&B Dashboard mit:"
echo "  • Interaktive Plots"
echo "  • Run Vergleiche"
echo "  • Metriken"
echo "  • Gradients"
echo "  • Hyperparameters"
echo ""
echo "======================================================================"
