import os
import random
import pickle
from pathlib import Path
from collections import defaultdict

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
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

from scipy.signal import butter, filtfilt


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = Path("../data/train")

# Transformer-specific parameters
D_MODEL = 256
NHEAD = 8
NUM_ENCODER_LAYERS = 2
NUM_DECODER_LAYERS = 2
DIM_FEEDFORWARD = 512
DROPOUT = 0.1

SEQ_LEN = 100
FUTURE_K = 20        # number of future target lataccel points (Option A)
BATCH_SIZE = 64
NUM_EPOCHS = 20
NUM_WORKERS = 4
LR = 1e-3


SCALER_TYPE = "local-scaler"
# SCALER_TYPE = "global-scaler"
TARGET_COLUMN = "steer-filtered"

NUMBER_SCENES_IN_DATASET = 10000

MODEL_NUMBER_VERSION = 1  # Changed to indicate Transformer version
MODEL_VERSION = f"v{MODEL_NUMBER_VERSION}_transformer-{NUM_ENCODER_LAYERS}-{NUM_DECODER_LAYERS}-{D_MODEL}_{NUMBER_SCENES_IN_DATASET}-dataset_lr-{LR}_epochs-{NUM_EPOCHS}_seq-{SEQ_LEN}_{SCALER_TYPE}_futurek-{FUTURE_K}_target-{TARGET_COLUMN}"
MODEL_SAVE_PATH = Path(f"../models/{MODEL_VERSION}")
MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)

SAVE_COLUMNS = ['roll', 'aEgo', 'vEgo', 'latAccelSteeringAngle', 'steerFiltered']
RENAME_COLUMNS = {
    'latAccelSteeringAngle': 'targetLateralAcceleration',
    'steerFiltered': 'steerCommand'
}


def lowpass_filter(data, cutoff=2.0, fs=20.0, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)


def fit_scalers(file_paths, sample_size=5000):
    sample_paths = random.sample(file_paths, min(len(file_paths), sample_size))
    dfs = [pd.read_csv(p)[SAVE_COLUMNS].rename(columns=RENAME_COLUMNS) for p in sample_paths]
    df = pd.concat(dfs)

    scalers = {
        'roll': StandardScaler().fit(df[['roll']].values),
        'aEgo': StandardScaler().fit(df[['aEgo']].values),
        'vEgo': StandardScaler().fit(df[['vEgo']].values),
        'targetLateralAcceleration': StandardScaler().fit(df[['targetLateralAcceleration']].values),
    }
    return scalers


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerEncoderDecoder(nn.Module):
    def __init__(self, physics_input_size, control_input_size, d_model, nhead, 
                 num_encoder_layers, num_decoder_layers, dim_feedforward, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # Input projections
        self.physics_projection = nn.Linear(physics_input_size, d_model)
        self.control_projection = nn.Linear(control_input_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        # Output layer
        self.fc_out = nn.Linear(d_model, 1)
        
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input_physics, input_control_sequence):
        # Project inputs to d_model dimension
        physics_embedded = self.physics_projection(input_physics) * np.sqrt(self.d_model)
        control_embedded = self.control_projection(input_control_sequence) * np.sqrt(self.d_model)
        
        # Add positional encoding
        physics_embedded = self.pos_encoder(physics_embedded)
        control_embedded = self.pos_encoder(control_embedded)
        
        # Create masks
        seq_len = input_physics.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(DEVICE)
        
        # Transformer forward pass
        # Use physics as encoder input and control as decoder input
        output = self.transformer(
            src=physics_embedded,
            tgt=control_embedded,
            tgt_mask=tgt_mask
        )
        
        return self.fc_out(output)


class DrivingDataset(Dataset):
    def __init__(self, file_paths, seq_len=SEQ_LEN, future_k=FUTURE_K, scalers=None):
        self.file_paths = file_paths
        self.seq_len = seq_len
        self.future_k = future_k
        self.scalers = scalers

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        df = pd.read_csv(path)
        df = df[SAVE_COLUMNS].rename(columns=RENAME_COLUMNS)

        min_needed = self.seq_len + self.future_k
        if len(df) < min_needed:
            return self[random.randint(0, len(self) - 1)]

        for col, scaler in self.scalers.items():
            df[col] = scaler.transform(df[[col]].values)

        first_100 = df.iloc[:min(100, len(df))]
        steer_scaler = RobustScaler().fit(first_100[['steerCommand']].values)
        df['steerCommand'] = steer_scaler.transform(df[['steerCommand']].values)

        arr = df[['roll', 'aEgo', 'vEgo', 'targetLateralAcceleration', 'steerCommand']].values
        physics_input = arr[:self.seq_len, :3]
        control_input = arr[:self.seq_len, 3:5]
        
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


def train_val_split(file_paths, val_ratio=0.2, seed=SEED):
    random.seed(seed)
    shuffled = list(file_paths)
    random.shuffle(shuffled)
    split_idx = int(len(shuffled) * (1 - val_ratio))
    return shuffled[:split_idx], shuffled[split_idx:]


def train_model(model, model_save_path, train_loader, val_loader, num_epochs=NUM_EPOCHS, lr=LR):
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_val_loss = float('inf')
    history = {
        'epoch': [],
        'train_loss': [], 'train_mae': [], 'train_rmse': [], 'train_r2': [],
        'val_loss': [], 'val_mae': [], 'val_rmse': [], 'val_r2': []
    }

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_preds_all, train_tgts_all = [], []

        for physics, control, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Train"):
            physics, control, y = physics.to(DEVICE), control.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            pred = model(physics, control)
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            train_preds_all.append(pred.detach().cpu().numpy())
            train_tgts_all.append(y.detach().cpu().numpy())

        avg_train_loss = train_loss / len(train_loader)
        train_preds = np.vstack(train_preds_all).reshape(-1)
        train_targets = np.vstack(train_tgts_all).reshape(-1)
        train_mae = mean_absolute_error(train_targets, train_preds)
        train_rmse = mean_squared_error(train_targets, train_preds, squared=False)
        train_r2 = r2_score(train_targets, train_preds)

        model.eval()
        val_loss = 0.0
        val_preds_all, val_tgts_all = [], []

        with torch.no_grad():
            for physics, control, y in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Val"):
                physics, control, y = physics.to(DEVICE), control.to(DEVICE), y.to(DEVICE)
                pred = model(physics, control)
                loss = criterion(pred, y)
                val_loss += loss.item()

                val_preds_all.append(pred.detach().cpu().numpy())
                val_tgts_all.append(y.detach().cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_preds = np.vstack(val_preds_all).reshape(-1)
        val_targets = np.vstack(val_tgts_all).reshape(-1)
        val_mae = mean_absolute_error(val_targets, val_preds)
        val_rmse = mean_squared_error(val_targets, val_preds, squared=False)
        val_r2 = r2_score(val_targets, val_preds)

        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        print(
            f"Epoch {epoch+1}/{num_epochs} | LR: {current_lr:.2e} | "
            f"Train Loss: {avg_train_loss:.4f}, MAE: {train_mae:.4f}, RMSE: {train_rmse:.4f}, R2: {train_r2:.4f} | "
            f"Val Loss: {avg_val_loss:.4f}, MAE: {val_mae:.4f}, RMSE: {val_rmse:.4f}, R2: {val_r2:.4f}"
        )
        
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(avg_train_loss)
        history['train_mae'].append(train_mae)
        history['train_rmse'].append(train_rmse)
        history['train_r2'].append(train_r2)
        history['val_loss'].append(avg_val_loss)
        history['val_mae'].append(val_mae)
        history['val_rmse'].append(val_rmse)
        history['val_r2'].append(val_r2)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_save_path / 'transformer_future_best.pt')

    torch.save(model.state_dict(), model_save_path / 'transformer_future_final.pt')            

    history_df = pd.DataFrame(history)
    history_df.to_csv(f"{model_save_path}/history.csv", index=False)

    with open(model_save_path / 'history.pkl', 'wb') as f:
        pickle.dump(dict(history), f)

    return model


def main():
    all_files = list(DATA_DIR.glob('**/*.csv'))
    all_files = all_files[:NUMBER_SCENES_IN_DATASET]
    train_files, val_files = train_val_split(all_files, val_ratio=0.2)

    scalers = fit_scalers(train_files)
    for s in scalers.values():
        if hasattr(s, 'feature_names_in_'):
            s.feature_names_in_ = None
    with open(MODEL_SAVE_PATH / 'scalers.pkl', 'wb') as f:
        pickle.dump(scalers, f)

    train_dataset = DrivingDataset(train_files, seq_len=SEQ_LEN, future_k=FUTURE_K, scalers=scalers)
    val_dataset = DrivingDataset(val_files, seq_len=SEQ_LEN, future_k=FUTURE_K, scalers=scalers)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

    physics_input_size = 3
    control_input_size = 2 + FUTURE_K

    model = TransformerEncoderDecoder(
        physics_input_size=physics_input_size,
        control_input_size=control_input_size,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT
    ).to(DEVICE)

    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")

    train_model(model, MODEL_SAVE_PATH, train_loader, val_loader,
                num_epochs=NUM_EPOCHS, lr=LR)


if __name__ == '__main__':
    main()