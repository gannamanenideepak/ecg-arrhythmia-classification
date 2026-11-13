"""
train.py
--------
Trains:
1. A small, CPU-friendly 1D Convolutional Neural Network (PyTorch) taking both
   the normalized 200-sample beat waveform and 4 RR timing features.
2. A Random Forest baseline classifier (scikit-learn) taking handcrafted simple features.

Key features:
- Strictly reproducible: fixed random seeds for python, numpy, and torch.
- Class imbalance handled on training data using smoothed inverse class frequency weights.
- DS2 test set is NEVER touched during training or hyperparameter tuning.
- Saves the best CNN model checkpoint based on validation Macro F1 score.
- Saves the trained Random Forest model.
"""

import os
import random
import argparse
import numpy as np
import joblib
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, classification_report


# ---------------------------------------------------------
# 1. Reproducibility Configuration
# ---------------------------------------------------------
def set_seed(seed: int = 42):
    """
    Sets fixed random seeds across all libraries for deterministic execution.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed} (reproducibility enabled).")


# ---------------------------------------------------------
# 2. 1D CNN Architecture with Dual-Input Fusion
# ---------------------------------------------------------
class ECGBeatCNN(nn.Module):
    """
    Lightweight 1D CNN for ECG beat classification.
    Takes two inputs:
      1. wave: Normalized beat waveform of shape (batch, 1, 200)
      2. rr: Standardized RR timing features of shape (batch, 4)
    """
    def __init__(self, num_classes: int = 5):
        super().__init__()
        
        # Waveform feature extractor: 3 Conv1D blocks
        # Conv Block 1: receptive field 7, downsamples 200 -> 100
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        # Conv Block 2: receptive field 5, downsamples 100 -> 50
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        # Conv Block 3: receptive field 3, global pooling -> 64-dim embedding
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        
        # RR features projection: 4 features -> 16-dim embedding
        self.rr_fc = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU()
        )
        
        # Fusion classifier: (64 waveform + 16 RR) = 80 dims -> 5 classes
        self.classifier = nn.Sequential(
            nn.Linear(64 + 16, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )

    def forward(self, wave: torch.Tensor, rr: torch.Tensor) -> torch.Tensor:
        # wave: (B, 1, 200) -> (B, 64)
        feat_wave = self.conv3(self.conv2(self.conv1(wave)))
        # rr: (B, 4) -> (B, 16)
        feat_rr = self.rr_fc(rr)
        # Concatenate multi-modal features
        fused = torch.cat([feat_wave, feat_rr], dim=1)
        # Class logits
        logits = self.classifier(fused)
        return logits


# ---------------------------------------------------------
# 3. Training & Validation Procedures
# ---------------------------------------------------------
def compute_class_weights(y_train: np.ndarray, num_classes: int = 5) -> torch.Tensor:
    """
    Computes smoothed inverse class frequency weights for training.
    Using square-root smoothing prevents runaway gradients on rare classes (e.g. Q)
    while giving minority classes (S, V, F) strong loss penalties.
    """
    class_counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
    # Avoid zero division
    class_counts = np.maximum(class_counts, 1.0)
    total_samples = np.sum(class_counts)
    
    # Square-root smoothed inverse frequency
    weights = (total_samples / (num_classes * class_counts)) ** 0.5
    # Normalize so mean weight is 1.0
    weights = weights / np.mean(weights)
    return torch.tensor(weights, dtype=torch.float32)


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device):
    """
    Evaluates PyTorch model on a DataLoader and returns accuracy, macro F1, and predictions.
    """
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for b_wave, b_rr, b_y in loader:
            b_wave = b_wave.to(device)
            b_rr = b_rr.to(device)
            logits = model(b_wave, b_rr)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(b_y.numpy())
            
    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    return acc, macro_f1, np.array(all_preds), np.array(all_targets)


def train_cnn(data_dir: str, models_dir: str, epochs: int = 12, batch_size: int = 128, lr: float = 1e-3):
    """
    Trains the 1D CNN using PyTorch and saves the best model checkpoint.
    """
    print("\n==========================================")
    print("1. Training 1D CNN with RR Feature Fusion")
    print("==========================================")
    
    # 1. Load train and val data (test set is NOT touched)
    train_data = np.load(os.path.join(data_dir, 'train.npz'))
    val_data = np.load(os.path.join(data_dir, 'val.npz'))
    
    X_wave_train = torch.tensor(train_data['waveform'][:, np.newaxis, :], dtype=torch.float32)
    X_rr_train = torch.tensor(train_data['rr'], dtype=torch.float32)
    y_train = torch.tensor(train_data['labels'], dtype=torch.long)
    
    X_wave_val = torch.tensor(val_data['waveform'][:, np.newaxis, :], dtype=torch.float32)
    X_rr_val = torch.tensor(val_data['rr'], dtype=torch.float32)
    y_val = torch.tensor(val_data['labels'], dtype=torch.long)
    
    train_loader = DataLoader(TensorDataset(X_wave_train, X_rr_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_wave_val, X_rr_val, y_val), batch_size=batch_size, shuffle=False)
    
    # 2. Setup Device, Model, Class Weights & Loss
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    class_weights = compute_class_weights(train_data['labels']).to(device)
    print("Class weights applied in training loss:")
    for idx, (name, w) in enumerate(zip(['N', 'S', 'V', 'F', 'Q'], class_weights.cpu().numpy())):
        print(f"  Class {name} (idx {idx}): weight = {w:.3f}")
        
    model = ECGBeatCNN(num_classes=5).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,} (CPU-friendly)")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    best_val_f1 = 0.0
    best_model_path = os.path.join(models_dir, 'best_ecg_cnn.pt')
    
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for b_wave, b_rr, b_y in train_loader:
            b_wave = b_wave.to(device)
            b_rr = b_rr.to(device)
            b_y = b_y.to(device)
            
            optimizer.zero_grad()
            logits = model(b_wave, b_rr)
            loss = criterion(logits, b_y)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * len(b_y)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == b_y).sum().item()
            total += len(b_y)
            
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        
        # Evaluate on validation split
        val_acc, val_f1, _, _ = evaluate_model(model, val_loader, device)
        scheduler.step(val_f1)
        
        is_best = val_f1 > best_val_f1
        if is_best:
            best_val_f1 = val_f1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'val_acc': val_acc
            }, best_model_path)
            
        star = " (*New Best Saved*)" if is_best else ""
        print(f"Epoch [{epoch:02d}/{epochs:02d}] Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f} | "
              f"Val Acc: {val_acc:.4f} | Val Macro F1: {val_f1:.4f}{star}")
        
    print(f"\nCNN training completed. Best Validation Macro F1: {best_val_f1:.4f}")
    print(f"Best model saved to: {best_model_path}")


# ---------------------------------------------------------
# 4. Random Forest Baseline Model
# ---------------------------------------------------------
def train_rf_baseline(data_dir: str, models_dir: str):
    """
    Trains a Random Forest baseline on handcrafted simple features.
    """
    print("\n==========================================")
    print("2. Training Random Forest Baseline")
    print("==========================================")
    
    train_data = np.load(os.path.join(data_dir, 'train.npz'))
    val_data = np.load(os.path.join(data_dir, 'val.npz'))
    
    # Handcrafted simple features (4 RR + 8 statistical + 20 downsampled waveform)
    X_train = train_data['simple']
    y_train = train_data['labels']
    
    X_val = val_data['simple']
    y_val = val_data['labels']
    
    print(f"Training Random Forest on {X_train.shape[0]} samples with {X_train.shape[1]} features...")
    # class_weight='balanced' handles the severe class imbalance
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    
    val_preds = rf.predict(X_val)
    val_acc = accuracy_score(y_val, val_preds)
    val_f1 = f1_score(y_val, val_preds, average='macro', zero_division=0)
    
    print(f"Random Forest Validation Accuracy: {val_acc:.4f} | Validation Macro F1: {val_f1:.4f}")
    
    rf_model_path = os.path.join(models_dir, 'rf_baseline.joblib')
    joblib.dump(rf, rf_model_path)
    print(f"Random Forest model saved to: {rf_model_path}")


# ---------------------------------------------------------
# 5. Main Execution Entrypoint
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train ECG beat classification models.")
    parser.add_argument('--data_dir', type=str, default='data/processed', help="Path to processed npz directory")
    parser.add_argument('--models_dir', type=str, default='models', help="Directory to save trained models")
    parser.add_argument('--epochs', type=int, default=12, help="Number of training epochs for CNN")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size for training")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate for Adam optimizer")
    parser.add_argument('--seed', type=int, default=42, help="Fixed random seed for reproducibility")
    args = parser.parse_args()
    
    set_seed(args.seed)
    os.makedirs(args.models_dir, exist_ok=True)
    
    # 1. Train PyTorch 1D CNN
    train_cnn(args.data_dir, args.models_dir, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    
    # 2. Train Random Forest Baseline
    train_rf_baseline(args.data_dir, args.models_dir)


if __name__ == '__main__':
    main()

