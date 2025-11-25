"""
Visualisierungsfunktionen für Fashion-MNIST Training
Input Visualization, Prediction Dynamics, Training Curves, etc.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np


# Fashion-MNIST Klassennamen
CLASS_NAMES = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def visualize_input_before_net(data_loader, num_samples=16, save_path='outputs/input_visualization.png'):
    """
    7. VISUALIZE JUST BEFORE THE NET
    Zeigt Bilder GENAU so wie sie in das Netzwerk gehen.
    Kritisch um Preprocessing-Fehler zu finden!
    """
    print("\n" + "="*70)
    print("7. INPUT VISUALIZATION (Just Before the Net)")
    print("="*70)
    print("Displaying images EXACTLY as they enter the network...")
    print("Check for preprocessing errors (normalization, shape, etc.)")

    # Hole ersten Batch
    for batch_features, batch_labels in data_loader:
        batch_size = batch_features.size(0)
        num_show = min(num_samples, batch_size)

        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        fig.suptitle('Input Visualization - Just Before Network', fontsize=16, fontweight='bold')

        for i, ax in enumerate(axes.flat):
            if i < num_show:
                # Bild so wie es ins Netz geht
                img = batch_features[i].squeeze().numpy()
                label = batch_labels[i].item()

                ax.imshow(img, cmap='gray')
                ax.set_title(f'{CLASS_NAMES[label]}\nShape: {batch_features[i].shape}\n'
                           f'Min: {img.min():.2f}, Max: {img.max():.2f}',
                           fontsize=9)
                ax.axis('off')
            else:
                ax.axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Input visualization saved to: {save_path}")

        # Statistiken
        print(f"\nInput Statistics:")
        print(f"  Shape: {batch_features[0].shape}")
        print(f"  Dtype: {batch_features.dtype}")
        print(f"  Min value: {batch_features.min():.4f}")
        print(f"  Max value: {batch_features.max():.4f}")
        print(f"  Mean: {batch_features.mean():.4f}")
        print(f"  Std: {batch_features.std():.4f}")

        if batch_features.min() < -2 or batch_features.max() > 2:
            print("\n⚠ WARNING: Unusual value range! Check normalization.")
        else:
            print("\n✓ Value range looks normal.")

        print("="*70 + "\n")
        break  # Nur ersten Batch


class PredictionDynamicsTracker:
    """
    8. VISUALIZE PREDICTION DYNAMICS
    Trackt Vorhersagen für einen festen Batch über das Training hinweg.
    Zeigt wie sich Confidence und Vorhersagen entwickeln.
    """
    def __init__(self, fixed_batch_x, fixed_batch_y, device):
        self.fixed_batch_x = fixed_batch_x.to(device)
        self.fixed_batch_y = fixed_batch_y
        self.device = device

        self.epochs = []
        self.predictions_history = []
        self.confidences_history = []
        self.correct_history = []

    def update(self, model, epoch):
        """
        Speichert Vorhersagen für aktuellen Epoch.
        """
        model.eval()
        with torch.no_grad():
            outputs = model(self.fixed_batch_x)
            probs = torch.softmax(outputs, dim=1)

            # Predictions und Confidence
            confidences, predictions = torch.max(probs, dim=1)

            # Korrekt oder nicht?
            correct = (predictions.cpu() == self.fixed_batch_y).numpy()

            self.epochs.append(epoch)
            self.predictions_history.append(predictions.cpu().numpy())
            self.confidences_history.append(confidences.cpu().numpy())
            self.correct_history.append(correct)

    def plot(self, save_path='outputs/prediction_dynamics.png'):
        """
        Visualisiert die Entwicklung der Vorhersagen.
        """
        if len(self.epochs) == 0:
            print("No prediction history to plot!")
            return

        num_samples = len(self.fixed_batch_y)
        num_show = min(8, num_samples)

        fig = plt.figure(figsize=(16, 10))

        # 1. Confidence über Zeit für jedes Bild
        ax1 = plt.subplot(2, 2, 1)
        for i in range(num_show):
            confidences = [conf[i] for conf in self.confidences_history]
            color = 'green' if self.correct_history[-1][i] else 'red'
            ax1.plot(self.epochs, confidences, marker='o', label=f'Sample {i}',
                    color=color, alpha=0.7)

        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Prediction Confidence')
        ax1.set_title('Confidence Evolution (Green=Correct, Red=Wrong)', fontweight='bold')
        ax1.legend(ncol=2, fontsize=8)
        ax1.grid(True, alpha=0.3)

        # 2. Anzahl korrekter Vorhersagen über Zeit
        ax2 = plt.subplot(2, 2, 2)
        num_correct = [correct.sum() for correct in self.correct_history]
        ax2.plot(self.epochs, num_correct, marker='o', linewidth=2, color='blue')
        ax2.axhline(y=num_samples, color='green', linestyle='--',
                   label=f'Perfect ({num_samples}/{num_samples})')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Number Correct')
        ax2.set_title('Correct Predictions Over Time', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Heatmap: Vorhersagen über Zeit
        ax3 = plt.subplot(2, 2, 3)
        pred_matrix = np.array(self.predictions_history).T[:num_show]

        im = ax3.imshow(pred_matrix, aspect='auto', cmap='tab10', interpolation='nearest')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Sample Index')
        ax3.set_title('Prediction Changes Over Time', fontweight='bold')
        ax3.set_xticks(range(0, len(self.epochs), max(1, len(self.epochs)//10)))
        ax3.set_xticklabels([self.epochs[i] for i in range(0, len(self.epochs),
                            max(1, len(self.epochs)//10))])
        ax3.set_yticks(range(num_show))

        # Colorbar mit Klassennamen
        cbar = plt.colorbar(im, ax=ax3)
        cbar.set_label('Predicted Class')

        # 4. Zeige die festen Bilder
        ax4 = plt.subplot(2, 2, 4)
        ax4.axis('off')

        # Mini-Grid der Samples
        mini_fig_size = 2
        for i in range(num_show):
            mini_ax = plt.gcf().add_subplot(2, 2, 4,
                                           frameon=False)
            # Positioniere Sub-Images
            img = self.fixed_batch_x[i].cpu().squeeze().numpy()
            true_label = self.fixed_batch_y[i].item()
            final_pred = self.predictions_history[-1][i]
            final_conf = self.confidences_history[-1][i]

            # Kompakte Darstellung
            col = i % 4
            row = i // 4

        ax4.set_title('Fixed Test Samples', fontweight='bold')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Prediction dynamics saved to: {save_path}")
        plt.close()

        # Print Summary
        print("\n" + "="*70)
        print("PREDICTION DYNAMICS SUMMARY")
        print("="*70)
        print(f"Tracked {num_samples} samples over {len(self.epochs)} epochs")
        print(f"\nFinal Results:")
        final_correct = self.correct_history[-1].sum()
        print(f"  Correct: {final_correct}/{num_samples} ({100*final_correct/num_samples:.1f}%)")
        print(f"  Avg Confidence: {self.confidences_history[-1].mean():.2%}")
        print("="*70 + "\n")


def plot_training_curves(train_losses, test_accuracies, save_path='outputs/training_curves.png'):
    """
    Plottet Training Loss und Test Accuracy Kurven.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Training Loss
    axes[0].plot(train_losses, linewidth=2, color='blue', label='Training Loss')
    axes[0].set_xlabel('Epoch', fontweight='bold')
    axes[0].set_ylabel('Loss', fontweight='bold')
    axes[0].set_title('Training Loss Curve', fontweight='bold', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Test Accuracy
    axes[1].plot(test_accuracies, linewidth=2, color='green', label='Test Accuracy')
    axes[1].axhline(y=max(test_accuracies), color='red', linestyle='--',
                   label=f'Best: {max(test_accuracies):.2f}%', alpha=0.7)
    axes[1].set_xlabel('Epoch', fontweight='bold')
    axes[1].set_ylabel('Accuracy (%)', fontweight='bold')
    axes[1].set_title('Test Accuracy Curve', fontweight='bold', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Training curves saved to: {save_path}")
    plt.close()


def plot_batch_level_loss(batch_losses, save_path='outputs/batch_level_loss.png'):
    """
    4. VERIFY DECREASING TRAINING LOSS
    Plottet Loss auf Batch-Level (nicht nur Epoch-Average).
    Hilft Instabilitäten früh zu erkennen.
    """
    print("\n" + "="*70)
    print("4. BATCH-LEVEL LOSS MONITORING")
    print("="*70)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Full batch loss curve
    axes[0].plot(batch_losses, linewidth=1, alpha=0.7, color='blue')
    axes[0].set_xlabel('Batch Iteration', fontweight='bold')
    axes[0].set_ylabel('Loss', fontweight='bold')
    axes[0].set_title('Batch-Level Loss (All Batches)', fontweight='bold', fontsize=14)
    axes[0].grid(True, alpha=0.3)

    # Moving average (smoother)
    window_size = min(100, len(batch_losses) // 10)
    if window_size > 1:
        moving_avg = np.convolve(batch_losses, np.ones(window_size)/window_size, mode='valid')
        axes[1].plot(moving_avg, linewidth=2, color='darkblue',
                    label=f'Moving Avg (window={window_size})')
        axes[1].set_xlabel('Batch Iteration', fontweight='bold')
        axes[1].set_ylabel('Loss (Smoothed)', fontweight='bold')
        axes[1].set_title('Smoothed Batch Loss', fontweight='bold', fontsize=14)
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Batch-level loss plot saved to: {save_path}")

    # Analyse
    if len(batch_losses) > 100:
        recent_losses = batch_losses[-100:]
        early_losses = batch_losses[:100]

        recent_avg = np.mean(recent_losses)
        early_avg = np.mean(early_losses)
        improvement = early_avg - recent_avg

        print(f"\nLoss Analysis:")
        print(f"  Early batches (first 100): {early_avg:.4f}")
        print(f"  Recent batches (last 100): {recent_avg:.4f}")
        print(f"  Improvement: {improvement:.4f}")

        if improvement > 0:
            print(f"  ✓ Loss is decreasing as expected!")
        else:
            print(f"  ⚠ WARNING: Loss not decreasing! Check learning rate or model.")

    print("="*70 + "\n")
    plt.close()
