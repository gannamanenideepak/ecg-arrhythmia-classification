"""
evaluate.py
-----------
Evaluates the trained 1D CNN and Random Forest baseline models on the
untouched DS2 test set (22 independent patients, 49,697 beats).

Outputs:
1. Per-class Precision, Recall, and F1-score for all 5 AAMI classes (N, S, V, F, Q).
2. Macro and weighted averages.
3. Overall classification accuracy.
4. Confusion matrix heatmaps saved as PNG images (reports/confusion_matrix_cnn.png, etc.).
5. Full evaluation metrics saved as JSON files (reports/metrics_cnn.json, etc.).
6. Formatted side-by-side comparison printed to the console.
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support
)

# Import the model architecture defined in train.py
from train import ECGBeatCNN

CLASS_NAMES = ['N', 'S', 'V', 'F', 'Q']
CLASS_DESCRIPTIONS = [
    'N: Normal / Non-ectopic',
    'S: Supraventricular ectopic',
    'V: Ventricular ectopic',
    'F: Fusion of ventricular & normal',
    'Q: Unknown / Paced / Unclassifiable'
]


def plot_and_save_confusion_matrix(cm: np.ndarray, model_name: str, output_path: str):
    """
    Plots a styled confusion matrix with raw counts and normalized recall percentages,
    saving the figure to the specified path.
    """
    # Calculate row-normalized matrix (recalls)
    row_sums = cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.divide(cm.astype('float'), row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)
    
    # Create combined cell annotations: "Count\n(Recall %)"
    annot = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]:,}\n({cm_norm[i, j] * 100:.1f}%)"
            
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        cm_norm,
        annot=annot,
        fmt="",
        cmap='Blues',
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        cbar_kws={'label': 'Normalized Recall'},
        ax=ax
    )
    ax.set_title(f"AAMI EC57 Confusion Matrix: {model_name}\nEvaluated on Inter-Patient DS2 Test Set", fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel("Predicted Class", fontsize=11, fontweight='semibold')
    ax.set_ylabel("True Class", fontsize=11, fontweight='semibold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix plot to: {output_path}")


def compute_and_save_metrics(y_true: np.ndarray, y_pred: np.ndarray, model_name: str, json_path: str, png_path: str):
    """
    Computes accuracy, per-class precision, recall, F1, and saves results to JSON and PNG.
    """
    acc = float(accuracy_score(y_true, y_pred))
    precisions, recalls, f1s, supports = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(CLASS_NAMES))), zero_division=0
    )
    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    weighted_prec, weighted_rec, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    
    metrics_dict = {
        'model_name': model_name,
        'overall_accuracy': acc,
        'macro_averages': {
            'precision': float(macro_prec),
            'recall': float(macro_rec),
            'f1_score': float(macro_f1)
        },
        'weighted_averages': {
            'precision': float(weighted_prec),
            'recall': float(weighted_rec),
            'f1_score': float(weighted_f1)
        },
        'per_class_metrics': {
            CLASS_NAMES[i]: {
                'description': CLASS_DESCRIPTIONS[i],
                'precision': float(precisions[i]),
                'recall': float(recalls[i]),
                'f1_score': float(f1s[i]),
                'support': int(supports[i])
            }
            for i in range(len(CLASS_NAMES))
        },
        'confusion_matrix': cm.tolist()
    }
    
    # Save metrics JSON
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"Saved evaluation metrics to: {json_path}")
    
    # Save Confusion Matrix Plot
    plot_and_save_confusion_matrix(cm, model_name, png_path)
    
    return metrics_dict


def evaluate_cnn(data_dir: str, models_dir: str, reports_dir: str):
    """
    Loads best 1D CNN model checkpoint and evaluates on DS2 test set.
    """
    model_path = os.path.join(models_dir, 'best_ecg_cnn.pt')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"CNN model checkpoint not found at: {model_path}. Please run train.py first.")
        
    test_data = np.load(os.path.join(data_dir, 'test.npz'))
    X_wave = torch.tensor(test_data['waveform'][:, np.newaxis, :], dtype=torch.float32)
    X_rr = torch.tensor(test_data['rr'], dtype=torch.float32)
    y_true = test_data['labels']
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ECGBeatCNN(num_classes=5).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    test_loader = DataLoader(TensorDataset(X_wave, X_rr), batch_size=256, shuffle=False)
    all_preds = []
    
    with torch.no_grad():
        for b_wave, b_rr in test_loader:
            b_wave = b_wave.to(device)
            b_rr = b_rr.to(device)
            logits = model(b_wave, b_rr)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            
    y_pred = np.array(all_preds)
    
    json_path = os.path.join(reports_dir, 'metrics_cnn.json')
    png_path = os.path.join(reports_dir, 'confusion_matrix_cnn.png')
    
    return compute_and_save_metrics(y_true, y_pred, "1D CNN with RR Fusion", json_path, png_path)


def evaluate_rf(data_dir: str, models_dir: str, reports_dir: str):
    """
    Loads Random Forest baseline model and evaluates on DS2 test set.
    """
    model_path = os.path.join(models_dir, 'rf_baseline.joblib')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Random Forest model not found at: {model_path}. Please run train.py first.")
        
    test_data = np.load(os.path.join(data_dir, 'test.npz'))
    X_simple = test_data['simple']
    y_true = test_data['labels']
    
    rf = joblib.load(model_path)
    y_pred = rf.predict(X_simple)
    
    json_path = os.path.join(reports_dir, 'metrics_rf.json')
    png_path = os.path.join(reports_dir, 'confusion_matrix_rf.png')
    
    return compute_and_save_metrics(y_true, y_pred, "Random Forest Baseline", json_path, png_path)


def print_comparison_table(metrics_cnn: dict, metrics_rf: dict):
    """
    Prints a clean, formatted comparison table of DS2 test results to standard output.
    """
    print("\n" + "=" * 80)
    print("                     DS2 TEST SET EVALUATION RESULTS (INTER-PATIENT)")
    print("=" * 80)
    print(f"{'Class':<6} {'Description':<35} {'Support':<8} | {'1D CNN (Precision / Recall / F1)':<32} | {'Random Forest (P / R / F1)'}")
    print("-" * 80)
    
    for cls in CLASS_NAMES:
        c_m = metrics_cnn['per_class_metrics'][cls]
        r_m = metrics_rf['per_class_metrics'][cls]
        desc = c_m['description'][:33]
        supp = c_m['support']
        
        c_str = f"{c_m['precision']*100:5.1f}% / {c_m['recall']*100:5.1f}% / {c_m['f1_score']*100:5.1f}%"
        r_str = f"{r_m['precision']*100:5.1f}% / {r_m['recall']*100:5.1f}% / {r_m['f1_score']*100:5.1f}%"
        
        print(f"{cls:<6} {desc:<35} {supp:<8} | {c_str:<32} | {r_str}")
        
    print("-" * 80)
    print(f"{'Overall Accuracy':<42} | CNN: {metrics_cnn['overall_accuracy']*100:5.2f}%{' ':19} | RF: {metrics_rf['overall_accuracy']*100:5.2f}%")
    print(f"{'Macro Average F1':<42} | CNN: {metrics_cnn['macro_averages']['f1_score']*100:5.2f}%{' ':19} | RF: {metrics_rf['macro_averages']['f1_score']*100:5.2f}%")
    print(f"{'Weighted Average F1':<42} | CNN: {metrics_cnn['weighted_averages']['f1_score']*100:5.2f}%{' ':19} | RF: {metrics_rf['weighted_averages']['f1_score']*100:5.2f}%")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate ECG beat classification models on DS2 test set.")
    parser.add_argument('--data_dir', type=str, default='data/processed', help="Path to processed npz directory")
    parser.add_argument('--models_dir', type=str, default='models', help="Directory containing trained models")
    parser.add_argument('--reports_dir', type=str, default='reports', help="Directory to save figures and metrics")
    args = parser.parse_args()
    
    print("\nEvaluating 1D CNN model on DS2 test set...")
    metrics_cnn = evaluate_cnn(args.data_dir, args.models_dir, args.reports_dir)
    
    print("\nEvaluating Random Forest baseline model on DS2 test set...")
    metrics_rf = evaluate_rf(args.data_dir, args.models_dir, args.reports_dir)
    
    print_comparison_table(metrics_cnn, metrics_rf)


if __name__ == '__main__':
    main()

