import os
import random
import pickle
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = Path("../data/train")

# Hyperparameters
LSTM_HIDDEN_SIZE = 256
LSTM_NUM_LAYERS = 2
SEQ_LEN = 100
FUTURE_K = 20
BATCH_SIZE = 128  # Increased batch size for better GPU utilization
NUM_EPOCHS = 100
NUM_WORKERS = min(8, os.cpu_count() // 2)  # Adaptive worker count
LR = 1e-3
GRAD_CLIP = 1.0
DROPOUT = 0.2

SCALER_TYPE = "local-scaler"
TARGET_COLUMN = "steer-filtered"
NUMBER_SCENES_IN_DATASET = 20000

MODEL_NUMBER_VERSION = 4
MODEL_VERSION = f"v{MODEL_NUMBER_VERSION}_lstm-{LSTM_NUM_LAYERS}-{LSTM_HIDDEN_SIZE}_{NUMBER_SCENES_IN_DATASET}-dataset_lr-{LR}_epochs-{NUM_EPOCHS}_seq-{SEQ_LEN}_{SCALER_TYPE}_futurek-{FUTURE_K}_target-{TARGET_COLUMN}"
MODEL_SAVE_PATH = Path(f"../models/{MODEL_VERSION}")
MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)

SAVE_COLUMNS = ['roll', 'aEgo', 'vEgo', 'latAccelSteeringAngle', 'steerFiltered']
RENAME_COLUMNS = {
    'latAccelSteeringAngle': 'targetLateralAcceleration',
    'steerFiltered': 'steerCommand'
}

def read_and_preprocess_file(path):
    """Read and preprocess a single file"""
    try:
        df = pd.read_csv(path, usecols=SAVE_COLUMNS)
        df = df.rename(columns=RENAME_COLUMNS)
        return df
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None

def fit_scalers(file_paths, sample_size=5000):
    """Efficient scaler fitting with parallel processing"""
    sample_paths = random.sample(file_paths, min(len(file_paths), sample_size))
    
    # Parallel file reading
    dfs = []
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        future_to_path = {executor.submit(read_and_preprocess_file, p): p for p in sample_paths}
        for future in tqdm(as_completed(future_to_path), total=len(sample_paths), desc="Reading files"):
            df = future.result()
            if df is not None and len(df) > 0:
                dfs.append(df)
    
    if not dfs:
        raise ValueError("No valid data found for scaling")
    
    df = pd.concat(dfs, ignore_index=True)
    
    scalers = {
        'roll': StandardScaler().fit(df[['roll']].values),
        'aEgo': StandardScaler().fit(df[['aEgo']].values),
        'vEgo': StandardScaler().fit(df[['vEgo']].values),
        'targetLateralAcceleration': StandardScaler().fit(df[['targetLateralAcceleration']].values),
    }
    
    # Clean scaler attributes for pickling
    for scaler in scalers.values():
        if hasattr(scaler, 'feature_names_in_'):
            delattr(scaler, 'feature_names_in_')
        if hasattr(scaler, 'n_features_in_'):
            delattr(scaler, 'n_features_in_')
    
    return scalers

class PreprocessedDrivingDataset(Dataset):
    """Optimized dataset with preprocessed data caching"""
    def __init__(self, file_paths, seq_len=SEQ_LEN, future_k=FUTURE_K, scalers=None):
        self.file_paths = file_paths
        self.seq_len = seq_len
        self.future_k = future_k
        self.scalers = scalers
        self.valid_indices = self._preprocess_and_filter_files()
        
    def _preprocess_and_filter_files(self):
        """Preprocess files and filter out short ones"""
        valid_indices = []
        self.preprocessed_data = []
        
        for idx, path in enumerate(tqdm(self.file_paths, desc="Preprocessing files")):
            df = pd.read_csv(path, usecols=SAVE_COLUMNS)
            df = df.rename(columns=RENAME_COLUMNS)
            
            if len(df) < self.seq_len + self.future_k:
                continue
                
            # Apply scaling
            for col, scaler in self.scalers.items():
                df[col] = scaler.transform(df[[col]].values)
            
            # Local steer scaling
            first_100 = df.iloc[:min(100, len(df))]
            steer_scaler = RobustScaler().fit(first_100[['steerCommand']].values)
            df['steerCommand'] = steer_scaler.transform(df[['steerCommand']].values)
            
            self.preprocessed_data.append(df.values)
            valid_indices.append(idx)
            
        return valid_indices
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        arr = self.preprocessed_data[idx]
        
        physics_input = arr[:self.seq_len, :3]
        control_input = arr[:self.seq_len, 3:5]
        
        # Future plan
        fut_raw = arr[1:1 + self.future_k, 3]
        if len(fut_raw) < self.future_k:
            pad_val = fut_raw[-1] if len(fut_raw) > 0 else 0.0
            fut_raw = np.pad(fut_raw, (0, self.future_k - len(fut_raw)), constant_values=pad_val)
        
        future_plan_matrix = np.repeat(fut_raw.reshape(1, -1), self.seq_len, axis=0)
        control_aug = np.concatenate([control_input, future_plan_matrix], axis=1)
        
        y = arr[1:self.seq_len + 1, 4].reshape(-1, 1)
        
        return (
            torch.tensor(physics_input, dtype=torch.float32),
            torch.tensor(control_aug, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32)
        )

class EfficientLSTM(nn.Module):
    """More efficient LSTM architecture with layer normalization"""
    def __init__(self, physics_input_size, control_input_size, hidden_size, num_layers, dropout=0.2):
        super().__init__()
        
        self.physics_encoder = nn.LSTM(
            physics_input_size, hidden_size, num_layers, 
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        
        self.control_encoder = nn.LSTM(
            control_input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        
        self.decoder = nn.LSTM(
            control_input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        
        # Add layer normalization for better training stability
        self.ln_physics = nn.LayerNorm(hidden_size)
        self.ln_control = nn.LayerNorm(hidden_size)
        self.ln_output = nn.LayerNorm(hidden_size)
        
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_physics, input_control_sequence):
        # Physics encoding
        _, (hidden_phsc, cell_phsc) = self.physics_encoder(input_physics)
        hidden_phsc = self.ln_physics(hidden_phsc)
        
        # Control encoding
        _, (hidden_ctrl, cell_ctrl) = self.control_encoder(input_control_sequence)
        hidden_ctrl = self.ln_control(hidden_ctrl)
        
        # Fusion
        hidden = (hidden_phsc + hidden_ctrl) / 2
        cell = (cell_phsc + cell_ctrl) / 2
        
        # Decoding
        decoder_output, _ = self.decoder(input_control_sequence, (hidden, cell))
        decoder_output = self.ln_output(decoder_output)
        decoder_output = self.dropout(decoder_output)
        
        return self.fc_out(decoder_output)

def train_val_split(file_paths, val_ratio=0.2, seed=SEED):
    """Stratified split maintaining temporal distribution"""
    random.seed(seed)
    shuffled = file_paths.copy()
    random.shuffle(shuffled)
    split_idx = int(len(shuffled) * (1 - val_ratio))
    return shuffled[:split_idx], shuffled[split_idx:]

def train_model(model, model_save_path, train_loader, val_loader, num_epochs=NUM_EPOCHS, lr=LR):
    """Enhanced training with learning rate scheduling and early stopping"""
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    criterion = nn.HuberLoss(delta=1.0)  # More robust than MSE
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 10
    
    history = {
        'epoch': [], 'train_loss': [], 'train_mae': [], 'train_rmse': [], 'train_r2': [],
        'val_loss': [], 'val_mae': [], 'val_rmse': [], 'val_r2': [], 'lr': []
    }

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_preds, train_targets = [], []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Train")
        for physics, control, y in pbar:
            physics, control, y = physics.to(DEVICE), control.to(DEVICE), y.to(DEVICE)
            
            optimizer.zero_grad(set_to_none=True)  # Faster zero_grad
            pred = model(physics, control)
            loss = criterion(pred, y)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            
            train_loss += loss.item()
            train_preds.append(pred.detach().cpu().numpy())
            train_targets.append(y.detach().cpu().numpy())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # Validation
        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []
        
        with torch.no_grad():
            for physics, control, y in tqdm(val_loader, desc="Validating"):
                physics, control, y = physics.to(DEVICE), control.to(DEVICE), y.to(DEVICE)
                pred = model(physics, control)
                loss = criterion(pred, y)
                val_loss += loss.item()
                
                val_preds.append(pred.cpu().numpy())
                val_targets.append(y.cpu().numpy())

        # Calculate metrics
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        train_preds = np.vstack(train_preds).reshape(-1)
        train_targets = np.vstack(train_targets).reshape(-1)
        val_preds = np.vstack(val_preds).reshape(-1)
        val_targets = np.vstack(val_targets).reshape(-1)

        train_mae = mean_absolute_error(train_targets, train_preds)
        train_rmse = mean_squared_error(train_targets, train_preds, squared=False)
        train_r2 = r2_score(train_targets, train_preds)
        
        val_mae = mean_absolute_error(val_targets, val_preds)
        val_rmse = mean_squared_error(val_targets, val_preds, squared=False)
        val_r2 = r2_score(val_targets, val_preds)

        # Update learning rate
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Update history
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(avg_train_loss)
        history['train_mae'].append(train_mae)
        history['train_rmse'].append(train_rmse)
        history['train_r2'].append(train_r2)
        history['val_loss'].append(avg_val_loss)
        history['val_mae'].append(val_mae)
        history['val_rmse'].append(val_rmse)
        history['val_r2'].append(val_r2)
        history['lr'].append(current_lr)

        print(f"\nEpoch {epoch+1}/{num_epochs} | LR: {current_lr:.2e}")
        print(f"Train - Loss: {avg_train_loss:.4f}, MAE: {train_mae:.4f}, RMSE: {train_rmse:.4f}, R²: {train_r2:.4f}")
        print(f"Val   - Loss: {avg_val_loss:.4f}, MAE: {val_mae:.4f}, RMSE: {val_rmse:.4f}, R²: {val_r2:.4f}")
        print("-" * 80)

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': best_val_loss,
                'metrics': {
                    'train_mae': train_mae, 'train_rmse': train_rmse, 'train_r2': train_r2,
                    'val_mae': val_mae, 'val_rmse': val_rmse, 'val_r2': val_r2
                }
            }, model_save_path / 'lstm_future_best_state_dict.pt')

            torch.save(model.state_dict(), model_save_path / 'lstm_future_best.pt')
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

    # Save final model and history
    torch.save(model.state_dict(), model_save_path / 'lstm_future_final.pt')
    
    # Save history
    history_df = pd.DataFrame(history)
    history_df.to_csv(model_save_path / 'training_history.csv', index=False)
    with open(model_save_path / 'history.pkl', 'wb') as f:
        pickle.dump(history, f)
    
    # Plot training curves
    plot_training_history(history, model_save_path)
    
    return model

def plot_training_history(history, save_path):
    """Plot training history metrics"""
    plt.figure(figsize=(15, 10))
    
    # Loss
    plt.subplot(2, 2, 1)
    plt.plot(history['epoch'], history['train_loss'], label='Train Loss')
    plt.plot(history['epoch'], history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    # MAE
    plt.subplot(2, 2, 2)
    plt.plot(history['epoch'], history['train_mae'], label='Train MAE')
    plt.plot(history['epoch'], history['val_mae'], label='Val MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.title('MAE')
    
    # R²
    plt.subplot(2, 2, 3)
    plt.plot(history['epoch'], history['train_r2'], label='Train R²')
    plt.plot(history['epoch'], history['val_r2'], label='Val R²')
    plt.xlabel('Epoch')
    plt.ylabel('R² Score')
    plt.legend()
    plt.title('R² Score')
    
    # Learning Rate
    plt.subplot(2, 2, 4)
    plt.plot(history['epoch'], history['lr'], label='Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('LR')
    plt.yscale('log')
    plt.legend()
    plt.title('Learning Rate Schedule')
    
    plt.tight_layout()
    plt.savefig(save_path / 'training_history.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Load files
    all_files = list(DATA_DIR.glob('**/*.csv'))
    all_files = all_files[:NUMBER_SCENES_IN_DATASET]
    print(f"Found {len(all_files)} files")
    
    # Split data
    train_files, val_files = train_val_split(all_files, val_ratio=0.2)
    print(f"Train: {len(train_files)}, Val: {len(val_files)}")
    
    # Fit scalers
    print("Fitting scalers...")
    scalers = fit_scalers(train_files)
    with open(MODEL_SAVE_PATH / 'scalers.pkl', 'wb') as f:
        pickle.dump(scalers, f)
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = PreprocessedDrivingDataset(train_files, scalers=scalers)
    val_dataset = PreprocessedDrivingDataset(val_files, scalers=scalers)
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    
    # Model
    model = EfficientLSTM(
        physics_input_size=3,
        control_input_size=2 + FUTURE_K,
        hidden_size=LSTM_HIDDEN_SIZE,
        num_layers=LSTM_NUM_LAYERS,
        dropout=DROPOUT
    ).to(DEVICE)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    train_model(model, MODEL_SAVE_PATH, train_loader, val_loader)

if __name__ == '__main__':
    main()