#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
W&B Local Server Demo - Komplett lokal!
"""

import random
import wandb
import os

# WICHTIG: Verbinde zum LOKALEN W&B Server statt wandb.ai
os.environ['WANDB_BASE_URL'] = 'http://localhost:8080'

print("="*70)
print("W&B Local Server Demo")
print("="*70)
print(f"Server URL: {os.environ['WANDB_BASE_URL']}")
print("="*70)

# Start a new wandb run - geht zum LOKALEN Server!
run = wandb.init(
    # WICHTIG: Diese Entity/Project werden auf deinem LOKALEN Server erstellt
    entity="local-user",  # Dein lokaler Username
    project="demo-project",
    config={
        "learning_rate": 0.02,
        "architecture": "CNN",
        "dataset": "CIFAR-100",
        "epochs": 10,
    },
)

print("\n✅ Run started!")
print(f"Run ID: {run.id}")
print(f"Run Name: {run.name}")
print(f"View at: http://localhost:8080/{run.entity}/{run.project}/runs/{run.id}")

# Simulate training
epochs = 10
offset = random.random() / 5

for epoch in range(2, epochs):
    acc = 1 - 2**-epoch - random.random() / epoch - offset
    loss = 2**-epoch + random.random() / epoch + offset

    # Log metrics to wandb - geht zum lokalen Server!
    run.log({"acc": acc, "loss": loss, "epoch": epoch})

    print(f"Epoch {epoch}: acc={acc:.4f}, loss={loss:.4f}")

# Finish the run
run.finish()

print("\n" + "="*70)
print("✅ Demo Complete!")
print("="*70)
print(f"Öffne Dashboard: http://localhost:8080")
print("="*70)
