"""
Erweiterte Metriken für Fashion-MNIST Training
Confusion Matrix, Per-Class Accuracy, Top-K Accuracy, etc.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score


# Fashion-MNIST Klassennamen
CLASS_NAMES = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def compute_detailed_metrics(model, data_loader, device, dataset_name="Test"):
    """
    5. ADD SIGNIFICANT DIGITS TO YOUR EVAL
    Berechnet umfassende Metriken über das GESAMTE Dataset (nicht nur Batches).
    """
    model.eval()

    all_predictions = []
    all_labels = []
    all_outputs = []

    with torch.no_grad():
        for batch_features, batch_labels in data_loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)

            outputs = model(batch_features)
            _, predicted = torch.max(outputs, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
            all_outputs.append(outputs.cpu())

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_outputs = torch.cat(all_outputs, dim=0)

    # Gesamtgenauigkeit
    accuracy = 100 * (all_predictions == all_labels).sum() / len(all_labels)

    # Top-3 und Top-5 Accuracy
    top3_acc = top_k_accuracy(all_outputs, torch.tensor(all_labels), k=3)
    top5_acc = top_k_accuracy(all_outputs, torch.tensor(all_labels), k=5)

    # Per-Class Metrics
    per_class_acc = {}
    for i, class_name in enumerate(CLASS_NAMES):
        class_mask = all_labels == i
        if class_mask.sum() > 0:
            class_correct = (all_predictions[class_mask] == all_labels[class_mask]).sum()
            per_class_acc[class_name] = 100 * class_correct / class_mask.sum()

    # Precision, Recall, F1
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')

    metrics = {
        'accuracy': accuracy,
        'top3_accuracy': top3_acc,
        'top5_accuracy': top5_acc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'per_class_accuracy': per_class_acc,
        'predictions': all_predictions,
        'labels': all_labels,
        'outputs': all_outputs
    }

    return metrics


def top_k_accuracy(outputs, labels, k=3):
    """
    Berechnet Top-K Accuracy.
    Model ist korrekt wenn wahres Label in Top-K Vorhersagen ist.
    """
    _, top_k_pred = outputs.topk(k, dim=1, largest=True, sorted=True)
    correct = top_k_pred.eq(labels.view(-1, 1).expand_as(top_k_pred))
    top_k_acc = 100 * correct.sum().item() / labels.size(0)
    return top_k_acc


def print_detailed_metrics(metrics, dataset_name="Test"):
    """
    6. HUMAN INTERPRETABLE METRICS
    Zeigt alle Metriken in lesbarem Format.
    """
    print("\n" + "="*70)
    print(f"{dataset_name.upper()} SET - DETAILED METRICS")
    print("="*70)

    print(f"\n📊 Overall Metrics:")
    print(f"  Accuracy:      {metrics['accuracy']:.2f}%")
    print(f"  Top-3 Acc:     {metrics['top3_accuracy']:.2f}%")
    print(f"  Top-5 Acc:     {metrics['top5_accuracy']:.2f}%")
    print(f"  Precision:     {metrics['precision']:.4f}")
    print(f"  Recall:        {metrics['recall']:.4f}")
    print(f"  F1-Score:      {metrics['f1_score']:.4f}")

    print(f"\n📈 Per-Class Accuracy:")
    per_class = metrics['per_class_accuracy']

    # Sortiere nach Accuracy (schlechteste zuerst)
    sorted_classes = sorted(per_class.items(), key=lambda x: x[1])

    for class_name, acc in sorted_classes:
        bar = "█" * int(acc / 2)  # Visual bar
        print(f"  {class_name:12s}: {acc:5.2f}% {bar}")

    # Identifiziere schwierigste Klassen
    worst_3 = sorted_classes[:3]
    best_3 = sorted_classes[-3:]

    print(f"\n⚠ Hardest Classes (lowest accuracy):")
    for class_name, acc in worst_3:
        print(f"  • {class_name}: {acc:.2f}%")

    print(f"\n✓ Best Classes (highest accuracy):")
    for class_name, acc in best_3:
        print(f"  • {class_name}: {acc:.2f}%")

    print("="*70 + "\n")


def plot_confusion_matrix(metrics, save_path='outputs/confusion_matrix.png'):
    """
    Erstellt und speichert Confusion Matrix.
    """
    predictions = metrics['predictions']
    labels = metrics['labels']

    # Berechne Confusion Matrix
    cm = confusion_matrix(labels, predictions)

    # Normalisiere auf [0, 1]
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Absolute Zahlen
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                ax=axes[0], cbar_kws={'label': 'Count'})
    axes[0].set_title('Confusion Matrix (Counts)', fontweight='bold', fontsize=14)
    axes[0].set_xlabel('Predicted Label')
    axes[0].set_ylabel('True Label')

    # Normalisiert (Prozent)
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                ax=axes[1], cbar_kws={'label': 'Proportion'})
    axes[1].set_title('Confusion Matrix (Normalized)', fontweight='bold', fontsize=14)
    axes[1].set_xlabel('Predicted Label')
    axes[1].set_ylabel('True Label')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Confusion matrix saved to: {save_path}")
    plt.close()


def compare_train_test_metrics(train_metrics, test_metrics, save_path='outputs/train_vs_test_comparison.png'):
    """
    Vergleicht Train und Test Metriken side-by-side.
    Hilft Overfitting zu identifizieren.
    """
    print("\n" + "="*70)
    print("TRAIN vs TEST COMPARISON")
    print("="*70)

    metrics_names = ['Accuracy', 'Top-3 Acc', 'Top-5 Acc', 'Precision', 'Recall', 'F1-Score']
    train_values = [
        train_metrics['accuracy'],
        train_metrics['top3_accuracy'],
        train_metrics['top5_accuracy'],
        train_metrics['precision'] * 100,
        train_metrics['recall'] * 100,
        train_metrics['f1_score'] * 100
    ]
    test_values = [
        test_metrics['accuracy'],
        test_metrics['top3_accuracy'],
        test_metrics['top5_accuracy'],
        test_metrics['precision'] * 100,
        test_metrics['recall'] * 100,
        test_metrics['f1_score'] * 100
    ]

    # Print Tabelle
    print(f"\n{'Metric':<15} {'Train':<10} {'Test':<10} {'Gap':<10}")
    print("-" * 50)
    for name, train_val, test_val in zip(metrics_names, train_values, test_values):
        gap = train_val - test_val
        gap_str = f"{gap:+.2f}%"
        print(f"{name:<15} {train_val:>7.2f}%  {test_val:>7.2f}%  {gap_str:>9}")

    # Overfitting Check
    acc_gap = train_values[0] - test_values[0]
    if acc_gap > 5:
        print(f"\n⚠ WARNING: Large train-test gap ({acc_gap:.2f}%) - possible overfitting!")
    elif acc_gap < 0:
        print(f"\n⚠ UNUSUAL: Test accuracy higher than train - check for bugs!")
    else:
        print(f"\n✓ Train-test gap is reasonable ({acc_gap:.2f}%)")

    print("="*70 + "\n")

    # Visualisierung
    x = np.arange(len(metrics_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, train_values, width, label='Train', color='skyblue')
    bars2 = ax.bar(x + width/2, test_values, width, label='Test', color='lightcoral')

    ax.set_ylabel('Score (%)', fontweight='bold')
    ax.set_title('Train vs Test Metrics Comparison', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Füge Werte über Balken hinzu
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Comparison plot saved to: {save_path}")
    plt.close()


def analyze_misclassifications(metrics, dataset, num_examples=16, save_path='outputs/misclassifications.png'):
    """
    Zeigt am häufigsten falsch klassifizierte Beispiele.
    """
    predictions = metrics['predictions']
    labels = metrics['labels']
    outputs = metrics['outputs']

    # Finde falsch klassifizierte Beispiele
    misclassified_mask = predictions != labels
    misclassified_indices = np.where(misclassified_mask)[0]

    if len(misclassified_indices) == 0:
        print("✓ No misclassifications found!")
        return

    # Berechne Confidence für jede Vorhersage
    probs = torch.softmax(outputs, dim=1)
    confidences = []
    for idx in misclassified_indices:
        pred_class = predictions[idx]
        confidence = probs[idx, pred_class].item()
        confidences.append((idx, confidence))

    # Sortiere nach höchster Confidence (am "sichersten" falsch)
    confidences.sort(key=lambda x: x[1], reverse=True)

    # Zeige Top-N "sichersten" Fehler
    num_show = min(num_examples, len(confidences))

    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    fig.suptitle('Most Confident Misclassifications', fontsize=16, fontweight='bold')

    for i, ax in enumerate(axes.flat):
        if i < num_show:
            dataset_idx = confidences[i][0]
            confidence = confidences[i][1]

            image, true_label = dataset[dataset_idx]
            pred_label = predictions[dataset_idx]

            img_np = image.squeeze().numpy()

            ax.imshow(img_np, cmap='gray')
            title = f'True: {CLASS_NAMES[true_label]}\nPred: {CLASS_NAMES[pred_label]}\nConf: {confidence:.2%}'
            ax.set_title(title, fontsize=9, color='red')
            ax.axis('off')
        else:
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Misclassification analysis saved to: {save_path}")
    plt.close()
