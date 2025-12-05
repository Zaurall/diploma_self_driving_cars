import os
import random
import pickle
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = Path("../data/train")

LSTM_HIDDEN_SIZE = 256
LSTM_NUM_LAYERS = 2
SEQ_LEN = 100
FUTURE_K = 20
BATCH_SIZE = 64
NUM_EPOCHS = 20
NUM_WORKERS = 4
LR = 1e-3

SCALER_TYPE = "local-scaler"
TARGET_COLUMN = "steer-filtered"
NUMBER_SCENES_IN_DATASET = 10000

MODEL_NUMBER_VERSION = 52
MODEL_VERSION = f"v{MODEL_NUMBER_VERSION}_lstm-{LSTM_NUM_LAYERS}-{LSTM_HIDDEN_SIZE}_{NUMBER_SCENES_IN_DATASET}-dataset_lr-{LR}_epochs-{NUM_EPOCHS}_seq-{SEQ_LEN}_{SCALER_TYPE}_futurek-{FUTURE_K}_target-{TARGET_COLUMN}"
MODEL_SAVE_PATH = Path(f"../models/{MODEL_VERSION}")
MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)

SAVE_COLUMNS = ['roll', 'aEgo', 'vEgo', 'latAccelSteeringAngle', 'steerFiltered']
RENAME_COLUMNS = {
    'latAccelSteeringAngle': 'targetLateralAcceleration',
    'steerFiltered': 'steerCommand'
}

class PreprocessedDrivingDataset(Dataset):
    """
    Optimized dataset class that preprocesses all data upfront
    Inspired by PyTorch's custom dataset examples :cite[3]:cite[7]
    """
    def __init__(self, file_paths, seq_len=SEQ_LEN, future_k=FUTURE_K, scalers=None, preprocess_all=True):
        self.file_paths = file_paths
        self.seq_len = seq_len
        self.future_k = future_k
        self.scalers = scalers
        self.preprocess_all = preprocess_all
        
        if preprocess_all:
            self.preprocessed_data = self._preprocess_all_files()
        else:
            self.preprocessed_data = None

    def _preprocess_all_files(self):
        """Preprocess all files upfront for faster training"""
        processed_data = []
        
        for path in tqdm(self.file_paths, desc="Preprocessing files"):
            df = pd.read_csv(path)
            df = df[SAVE_COLUMNS].rename(columns=RENAME_COLUMNS)
            
            # Skip short episodes
            if len(df) < self.seq_len + self.future_k:
                continue
                
            # Apply standard scalers
            for col, scaler in self.scalers.items():
                df[col] = scaler.transform(df[[col]].values)
            
            # Per-file robust scaling for steerCommand
            first_100 = df.iloc[:min(100, len(df))]
            steer_scaler = RobustScaler().fit(first_100[['steerCommand']].values)
            df['steerCommand'] = steer_scaler.transform(df[['steerCommand']].values)
            
            # Store preprocessed data
            processed_data.append(df.values)
        
        return processed_data

    def __len__(self):
        if self.preprocess_all:
            return len(self.preprocessed_data)
        return len(self.file_paths)

    def __getitem__(self, idx):
        if self.preprocess_all:
            arr = self.preprocessed_data[idx]
        else:
            path = self.file_paths[idx]
            df = pd.read_csv(path)
            df = df[SAVE_COLUMNS].rename(columns=RENAME_COLUMNS)
            
            if len(df) < self.seq_len + self.future_k:
                return self[random.randint(0, len(self) - 1)]
                
            for col, scaler in self.scalers.items():
                df[col] = scaler.transform(df[[col]].values)
            
            first_100 = df.iloc[:min(100, len(df))]
            steer_scaler = RobustScaler().fit(first_100[['steerCommand']].values)
            df['steerCommand'] = steer_scaler.transform(df[['steerCommand']].values)
            
            arr = df.values

        # Input features: physics + targetLateralAcceleration + future plan
        input_features = arr[:self.seq_len, :4]  # roll, aEgo, vEgo, targetLateralAcceleration
        
        # Future plan
        fut_raw = arr[1:1 + self.future_k, 3]
        if len(fut_raw) < self.future_k:
            pad_val = fut_raw[-1] if len(fut_raw) > 0 else 0.0
            fut_raw = np.pad(fut_raw, (0, self.future_k - len(fut_raw)), constant_values=pad_val)
        
        future_plan_matrix = np.repeat(fut_raw.reshape(1, -1), self.seq_len, axis=0)
        input_aug = np.concatenate([input_features, future_plan_matrix], axis=1)
        
        # Labels: predict next-step steerCommand
        y = arr[1:self.seq_len + 1, 4].reshape(-1, 1)

        return (
            torch.tensor(input_aug, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32)
        )

class EfficientLSTM(nn.Module):
    """
    Simplified LSTM model that takes all features as input
    and predicts only steerCommand
    """
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        self.fc_out = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc_out(lstm_out)

def fit_scalers(file_paths, sample_size=5000):
    """Efficient scaler fitting with sampling"""
    sample_paths = random.sample(file_paths, min(len(file_paths), sample_size))
    
    # Use list comprehension for faster reading
    dfs = [pd.read_csv(p)[SAVE_COLUMNS].rename(columns=RENAME_COLUMNS) for p in sample_paths]
    df = pd.concat(dfs, ignore_index=True)
    
    scalers = {
        'roll': StandardScaler().fit(df[['roll']].values),
        'aEgo': StandardScaler().fit(df[['aEgo']].values),
        'vEgo': StandardScaler().fit(df[['vEgo']].values),
        'targetLateralAcceleration': StandardScaler().fit(df[['targetLateralAcceleration']].values),
    }
    return scalers

def train_val_split(file_paths, val_ratio=0.2, seed=SEED):
    """Stratified split to ensure distribution consistency"""
    random.seed(seed)
    shuffled = list(file_paths)
    random.shuffle(shuffled)
    split_idx = int(len(shuffled) * (1 - val_ratio))
    return shuffled[:split_idx], shuffled[split_idx:]

def train_model(model, model_save_path, train_loader, val_loader, num_epochs=NUM_EPOCHS, lr=LR):
    """Enhanced training function with better monitoring"""
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_val_loss = float('inf')
    history = {
        'epoch': [], 'train_loss': [], 'train_mae': [], 'train_rmse': [], 'train_r2': [],
        'val_loss': [], 'val_mae': [], 'val_rmse': [], 'val_r2': [], 'learning_rate': []
    }

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_preds, train_targets = [], []

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Train"):
            x, y = x.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            train_preds.append(pred.detach().cpu().numpy())
            train_targets.append(y.detach().cpu().numpy())

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []

        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Val"):
                x, y = x.to(DEVICE), y.to(DEVICE)
                pred = model(x)
                loss = criterion(pred, y)
                val_loss += loss.item()
                
                val_preds.append(pred.detach().cpu().numpy())
                val_targets.append(y.detach().cpu().numpy())

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
        history['learning_rate'].append(current_lr)

        print(f"Epoch {epoch+1}/{num_epochs} | LR: {current_lr:.2e}")
        print(f"Train - Loss: {avg_train_loss:.4f}, MAE: {train_mae:.4f}, RMSE: {train_rmse:.4f}, R2: {train_r2:.4f}")
        print(f"Val   - Loss: {avg_val_loss:.4f}, MAE: {val_mae:.4f}, RMSE: {val_rmse:.4f}, R2: {val_r2:.4f}")
        print("-" * 80)

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }, model_save_path / 'lstm_future_best.pt')

    # Save final model and history
    torch.save(model.state_dict(), model_save_path / 'lstm_future_final.pt')
    history_df = pd.DataFrame(history)
    history_df.to_csv(model_save_path / 'training_history.csv', index=False)
    
    with open(model_save_path / 'history.pkl', 'wb') as f:
        pickle.dump(history, f)

    return model

def main():
    # Load and prepare data
    all_files = list(DATA_DIR.glob('**/*.csv'))
    all_files = all_files[:NUMBER_SCENES_IN_DATASET]
    train_files, val_files = train_val_split(all_files, val_ratio=0.2)

    # Fit scalers
    scalers = fit_scalers(train_files)
    for s in scalers.values():
        if hasattr(s, 'feature_names_in_'):
            s.feature_names_in_ = None
    
    with open(MODEL_SAVE_PATH / 'scalers.pkl', 'wb') as f:
        pickle.dump(scalers, f)

    # Create datasets with optional preprocessing
    train_dataset = PreprocessedDrivingDataset(
        train_files, seq_len=SEQ_LEN, future_k=FUTURE_K, 
        scalers=scalers, preprocess_all=True  # Set to False for large datasets
    )
    
    val_dataset = PreprocessedDrivingDataset(
        val_files, seq_len=SEQ_LEN, future_k=FUTURE_K, 
        scalers=scalers, preprocess_all=False
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    # Initialize model
    input_size = 4 + FUTURE_K  # roll, aEgo, vEgo, targetLateralAcceleration + future plan
    model = EfficientLSTM(
        input_size=input_size,
        hidden_size=LSTM_HIDDEN_SIZE,
        num_layers=LSTM_NUM_LAYERS,
        dropout=0.2
    ).to(DEVICE)

    # Train model
    train_model(model, MODEL_SAVE_PATH, train_loader, val_loader,
                num_epochs=NUM_EPOCHS, lr=LR)

if __name__ == '__main__':
    main()