#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Einfacher W&B Offline Viewer - 100% Lokal
Zeigt nur die gespeicherten Plots an - KEIN wandb Import nötig!
"""

from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt


def view_wandb_offline():
    """Zeigt W&B offline runs lokal an"""

    base_dir = Path(__file__).parent

    print("="*70)
    print("🔬 W&B Offline Viewer - 100% Lokal")
    print("="*70)

    # Finde offline runs
    v6a_runs = list((base_dir / "v6a" / "wandb").glob("offline-run-*"))
    v6b_runs = list((base_dir / "v6b" / "wandb").glob("offline-run-*"))

    print(f"\n📁 Gefundene Runs:")
    print(f"  v6a: {len(v6a_runs)} run(s)")
    print(f"  v6b: {len(v6b_runs)} run(s)")

    if not v6a_runs and not v6b_runs:
        print("\n⚠️  Keine offline runs gefunden!")
        return

    # Zeige beide Runs
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('W&B Offline Runs - v6a vs v6b', fontsize=14, fontweight='bold')

    for idx, (runs, name, ax) in enumerate([(v6a_runs, 'v6a', axes[0]),
                                              (v6b_runs, 'v6b', axes[1])]):
        if runs:
            latest_run = sorted(runs, key=lambda p: p.stat().st_mtime)[-1]

            # Suche nach gespeicherten Plots in media/images
            plot_files = list((latest_run / "files" / "media" / "images").glob("*.png"))

            if plot_files:
                # Zeige den ersten Plot
                img = Image.open(plot_files[0])
                ax.imshow(img)
                ax.axis('off')
                ax.set_title(f'{name}: {latest_run.name}', fontsize=10)
                print(f"\n✅ {name}: {latest_run.name}")
                print(f"   Plot: {plot_files[0].name}")
            else:
                ax.text(0.5, 0.5, f'{name}\nKeine Plots gefunden',
                       ha='center', va='center')
                ax.axis('off')
                print(f"\n⚠️  {name}: Keine Plots in {latest_run.name}")
        else:
            ax.text(0.5, 0.5, f'{name}\nKein Run gefunden',
                   ha='center', va='center')
            ax.axis('off')

    plt.tight_layout()

    # Zeige auch die direkten Training Plots
    print("\n" + "="*70)
    print("💡 Weitere Plots verfügbar:")
    print("="*70)

    if (base_dir / "v6a" / "training_analysis.png").exists():
        print("✅ v6a/training_analysis.png")
    if (base_dir / "v6b" / "training_analysis.png").exists():
        print("✅ v6b/training_analysis.png")

    print("\n" + "="*70)
    plt.savefig('wandb_offline_overview.png', dpi=150, bbox_inches='tight')
    print("💾 Overview gespeichert: wandb_offline_overview.png")
    print("="*70)

    plt.show()


if __name__ == '__main__':
    view_wandb_offline()
