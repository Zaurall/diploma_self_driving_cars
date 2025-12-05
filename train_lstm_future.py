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


# -------------------- Reproducibility --------------------
SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)


# -------------------- Configuration --------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = Path("../data/train")  # adjust if needed
SEQ_LEN = 100
FUTURE_K = 20        # number of future target lataccel points (Option A)
BATCH_SIZE = 64
NUM_EPOCHS = 20
NUM_WORKERS = 4
LR = 1e-3

MODEL_VERSION = f"lstm_futureA_seq{SEQ_LEN}_h{FUTURE_K}_epochs{NUM_EPOCHS}_lr{LR}"
MODEL_SAVE_PATH = Path(f"../models/{MODEL_VERSION}")
MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)

SAVE_COLUMNS = ['roll', 'aEgo', 'vEgo', 'latAccelSteeringAngle', 'steerFiltered']
RENAME_COLUMNS = {
    'latAccelSteeringAngle': 'targetLateralAcceleration',
    'steerFiltered': 'steerCommand'
}


# -------------------- Utilities --------------------
def train_val_split(file_paths, val_ratio=0.2, seed=SEED):
    random.seed(seed)
    shuffled = list(file_paths)
    random.shuffle(shuffled)
    split_idx = int(len(shuffled) * (1 - val_ratio))
    return shuffled[:split_idx], shuffled[split_idx:]


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


# -------------------- Dataset --------------------
class DrivingDataset(Dataset):
    """Returns sequences plus a repeated future plan vector (Option A).

    For each episode we take:
      - physics_input: (SEQ_LEN, 3) -> roll, aEgo, vEgo
      - control_input: (SEQ_LEN, 2) -> targetLateralAcceleration, steerCommand
      - future plan: first FUTURE_K future targetLateralAcceleration values after the sequence start
                     (shape (FUTURE_K,)) -> repeated and concatenated to each timestep of control_input
      - label y: next-step steerCommand for each timestep (SEQ_LEN, 1)
    Final control_aug shape: (SEQ_LEN, 2 + FUTURE_K)
    """
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

        # Need at least seq_len + future horizon + 1 rows
        min_needed = self.seq_len + self.future_k
        if len(df) < min_needed:
            # fallback to another random sample
            return self[random.randint(0, len(self) - 1)]

        # Apply fitted scalers
        for col, scaler in self.scalers.items():
            df[col] = scaler.transform(df[[col]].values)

        # Per-file robust scaling for steerCommand using first 100 rows (or entire if shorter)
        first_100 = df.iloc[:min(100, len(df))]
        steer_scaler = RobustScaler().fit(first_100[['steerCommand']].values)
        df['steerCommand'] = steer_scaler.transform(df[['steerCommand']].values)

        arr = df[['roll', 'aEgo', 'vEgo', 'targetLateralAcceleration', 'steerCommand']].values

        # Core sequences (take first seq_len rows)
        physics_input = arr[:self.seq_len, :3]
        control_input = arr[:self.seq_len, 3:5]  # targetLatAcc, steerCommand

        # Future plan: first FUTURE_K future target lat acc values immediately after first row
        fut_raw = arr[1:1 + self.future_k, 3]  # column index 3 = targetLateralAcceleration
        # Pad if needed (should not usually happen due to min_needed) but be safe
        if len(fut_raw) < self.future_k:
            pad_val = fut_raw[-1] if len(fut_raw) > 0 else 0.0
            fut_raw = np.pad(fut_raw, (0, self.future_k - len(fut_raw)), constant_values=pad_val)

        # Repeat future plan for each timestep and concatenate to control inputs
        future_plan_matrix = np.repeat(fut_raw.reshape(1, -1), self.seq_len, axis=0)  # (seq_len, FUTURE_K)
        control_aug = np.concatenate([control_input, future_plan_matrix], axis=1)      # (seq_len, 2 + FUTURE_K)

        # Labels: predict next-step steerCommand for each timestep
        y = arr[1:self.seq_len + 1, 4].reshape(-1, 1)

        return (
            torch.tensor(physics_input, dtype=torch.float32),
            torch.tensor(control_aug, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32)
        )


# -------------------- Model --------------------
class LstmEncoderDecoder(nn.Module):
    def __init__(self, physics_input_size, control_input_size, hidden_size, num_layers, dropout=0.2):
        super().__init__()
        self.physics_encoder = nn.LSTM(physics_input_size, hidden_size, num_layers,
                                       batch_first=True, dropout=dropout)
        self.control_encoder = nn.LSTM(control_input_size, hidden_size, num_layers,
                                       batch_first=True, dropout=dropout)
        # Decoder consumes the same augmented control input sequence
        self.decoder = nn.LSTM(control_input_size, hidden_size, num_layers,
                               batch_first=True, dropout=dropout)
        self.fc_out = nn.Linear(hidden_size, 1)

    def forward(self, input_physics, input_control_sequence):
        _, (hidden_phsc, cell_phsc) = self.physics_encoder(input_physics)
        _, (hidden_ctrl, cell_ctrl) = self.control_encoder(input_control_sequence)
        hidden = (hidden_phsc + hidden_ctrl) / 2.0
        cell = (cell_phsc + cell_ctrl) / 2.0
        dec_out, _ = self.decoder(input_control_sequence, (hidden, cell))
        return self.fc_out(dec_out)


# -------------------- Training Loop --------------------
def train_model(model, model_save_path, train_loader, val_loader, num_epochs=NUM_EPOCHS, lr=LR):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best_val_loss = float('inf')
    history = defaultdict(list)

    for epoch in range(num_epochs):
        # ---- Train ----
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

        # ---- Validation ----
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

        print(
            f"Epoch {epoch+1}/{num_epochs} | "
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
            torch.save(model.state_dict(), model_save_path / 'lstm_future_best.pt')

    torch.save(model.state_dict(), model_save_path / 'lstm_future_final.pt')
    with open(model_save_path / 'history.pkl', 'wb') as f:
        pickle.dump(dict(history), f)
    return model


def main():
    all_files = list(DATA_DIR.glob('**/*.csv'))
    # Optionally subsample for quicker experiments: all_files = all_files[:10000]
    train_files, val_files = train_val_split(all_files, val_ratio=0.2)

    scalers = fit_scalers(train_files)
    # Remove feature_names_in_ attribute (sometimes causes pickling or reuse issues)
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
    control_input_size = 2 + FUTURE_K  # augmented with future plan vector
    hidden_size = 256
    num_layers = 2

    model = LstmEncoderDecoder(
        physics_input_size=physics_input_size,
        control_input_size=control_input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=0.2
    ).to(DEVICE)

    train_model(model, MODEL_SAVE_PATH, train_loader, val_loader,
                num_epochs=NUM_EPOCHS, lr=LR)


if __name__ == '__main__':
    main()
