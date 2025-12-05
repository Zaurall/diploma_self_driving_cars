import os
import glob
import random
import pickle
import warnings
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, RobustScaler

# Suppress sklearn feature names warnings
warnings.filterwarnings("ignore", message="X has feature names, but.*was fitted without feature names")


class CommaDataset(Dataset):
    """
    Memory-efficient per-file sequential dataset.
    Supports both next-step (sequence-based) and single-step modes.
    """
    def __init__(self, file_paths, config, scalers=None):
        """
        Args:
            file_paths: list of CSV file paths
            seq_len: number of time steps in each sample
            future_k: number of future target acceleration steps to include
            scalers: dictionary of sklearn scalers (fitted externally)
            mode: 'sequence' (RNN) or 'frame' (tabular regression)
        """
        data_cfg = config["data"]
        
        self.file_paths = file_paths
        self.seq_len = data_cfg["seq_len"]
        self.future_k = data_cfg["future_k"]
        self.scalers = scalers or {}
        self.mode = data_cfg["mode"]

        self.save_columns = data_cfg["features"] + [data_cfg["target_column"]]
        self.rename_columns = data_cfg["rename_columns"]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        df = pd.read_csv(path)

        # Filter & rename columns if needed
        df = df[self.save_columns].rename(columns=self.rename_columns).copy()

        # # Need at least seq_len + future horizon + 1 rows
        # min_needed = self.seq_len + self.future_k
        # if len(df) < min_needed:
        #     return self[random.randint(0, len(self) - 1)]  # Skip short episodes

        # Apply global scalers (pre-fitted) - use .values for consistency
        for col, scaler in self.scalers.items():
            if col in df.columns:
                df.loc[:, col] = scaler.transform(df[[col]].values).flatten()

        # Robust scaling per-file for steerCommand using first seq_len rows (or entire if shorter)
        first_rows = df.head(min(self.seq_len, len(df)))[["steerCommand"]]
        steer_scaler = RobustScaler().fit(first_rows.values)
        df.loc[:, "steerCommand"] = steer_scaler.transform(df[["steerCommand"]].values).flatten()
        # df['steerCommand'] = lowpass_filter(df['steerCommand'].values)
        # df['steerCommand'] = df['steerCommand'].rolling(5, min_periods=1, center=True).mean()
        # df['steerCommand'] = self.scalers['steerCommand'].transform(df[['steerCommand']].values)

        
        if self.mode == "frame":
            # Single timestep regression
            x = df[["roll", "aEgo", "vEgo", "targetLateralAcceleration"]].values
            y = df["steerCommand"].values
            x = torch.tensor(x, dtype=torch.float32)
            y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
            return x, y

        # --- Sequence mode ---
        # TODO peace of shit
        min_needed = self.seq_len + self.future_k
        if len(df) < min_needed:
            return self[random.randint(0, len(self) - 1)]

        arr = df.values
        input_features = arr[:self.seq_len, :4]
        # Future plan: first FUTURE_K future target lat acc values immediately after first row
        # column index 3 = targetLateralAcceleration
        fut_raw = arr[1:1 + self.future_k, 3] if self.future_k > 0 else np.array([])
        if len(fut_raw) < self.future_k:
            # pad_val = fut_raw[-1] if len(fut_raw) > 0 else 0.0
            pad_val = 0
            fut_raw = np.pad(fut_raw, (0, self.future_k - len(fut_raw)), constant_values=pad_val)

        # Append future plan if needed
        if self.future_k > 0:
            # Repeat future plan for each timestep and concatenate to control inputs
            future_matrix = np.repeat(fut_raw.reshape(1, -1), self.seq_len, axis=0) # (seq_len, FUTURE_K)
            x = np.concatenate([input_features, future_matrix], axis=1)    # (seq_len, 2 + FUTURE_K)
        else:
            x = input_features

        # Labels: predict next-step steerCommand for each timestep
        y = arr[1:self.seq_len + 1, 4].reshape(-1, 1)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


class CommaDataModule(pl.LightningDataModule):
    """
    Hybrid of CommaData + DrivingDataset:
    - Handles multiple platforms
    - Memory efficient (lazy file loading)
    - Sequence or frame-level mode
    - Optional future horizon
    """
    def __init__(
        self,
        config,
        model_save_path
    ):
        super().__init__()
        self.config = config
        platforms = config["data"]["platforms"]
        if isinstance(platforms, str):
            platforms = [platforms]
        self.platforms = platforms
        self.model_save_path = model_save_path
        self.rename_columns = config["data"]["rename_columns"]
        self.save_columns = config["data"]["features"] + [config["data"]["target_column"]]
        self.scalers = {}
        self.file_paths_train = []
        self.file_paths_val = []


    def _save_scalers(self):
        # Save scalers without removing feature_names_in_ to maintain consistency
        with open(self.model_save_path / 'scalers.pkl', 'wb') as f:
            pickle.dump(self.scalers, f)

    def _fit_scalers(self):
        # TODO скейлить каждую платформу отдельно
        # TODO отдельный выбор скейлера
        # локальный и глобальный скейлер
        sample_paths = random.sample(self.file_paths_train, min(len(self.file_paths_train), self.config["data"]["scalers_fit_sample_size"]))
        dfs = [pd.read_csv(p)[self.save_columns].rename(columns=self.rename_columns) for p in sample_paths]
        # dfs = process_map(pd.read_csv, files, max_workers=24, chunksize=100)
        df = pd.concat(dfs) # , ignore_index=True)
    
        # Use .values to fit scalers with numpy arrays (no feature names)
        self.scalers = {
            'roll': StandardScaler().fit(df[['roll']].values),
            'aEgo': StandardScaler().fit(df[['aEgo']].values),
            'vEgo': StandardScaler().fit(df[['vEgo']].values),
            'targetLateralAcceleration': StandardScaler().fit(df[['targetLateralAcceleration']].values), # RobustScaler() 
            # 'steerCommand': RobustScaler().fit(df[['steerCommand']].values)
        }

        self._save_scalers()
             
    def setup(self, stage=None):
        random.seed(self.config["seed"])
        
        all_files = []
        for p in self.platforms:
            files = glob.glob(os.path.join(self.config["data"]["path"], p, "*.csv"))
            all_files.extend(files)
        assert len(all_files) > 0, "No CSV files found in the provided platforms"

        # TODO должен брать равномерно из каждой платформы, а берет вообще случайные
        random.shuffle(all_files)
        n_train = int(len(all_files) * self.config["data"]["train_frac"])
        self.file_paths_train = all_files[:n_train]
        self.file_paths_val = all_files[n_train:]

        self._fit_scalers()

        print(f"Loaded {len(self.file_paths_train)} train and {len(self.file_paths_val)} val files.")

    def train_dataloader(self):
        ds_train = CommaDataset(
            self.file_paths_train,
            config=self.config,
            scalers=self.scalers,
        )
        return DataLoader(ds_train, batch_size=self.config["data"]["batch_size"], shuffle=True, num_workers=self.config["data"]["num_workers"]) #, pin_memory=True

    def val_dataloader(self):
        ds_val = CommaDataset(
            self.file_paths_val,
            config=self.config,
            scalers=self.scalers,
        )
        # TODO зачем умножаем на 4?
        return DataLoader(ds_val, batch_size=self.config["data"]["batch_size"] * 4, shuffle=False, num_workers=self.config["data"]["num_workers"]) #, pin_memory=True
