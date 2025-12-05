from . import BaseController
import torch
import torch.nn as nn
import numpy as np
import pickle
import os
from typing import List
import warnings


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


class Controller(BaseController):
    """Efficient LSTM controller with layer normalization"""

    def __init__(
        self,
        model_path="../models/v4_lstm-2-256_20000-dataset_lr-0.001_epochs-100_seq-100_local-scaler_futurek-20_target-steer-filtered/lstm_future_best.pt",
        scaler_path="../models/v4_lstm-2-256_20000-dataset_lr-0.001_epochs-100_seq-100_local-scaler_futurek-20_target-steer-filtered/scalers.pkl",
        seq_len: int = 100,
        future_k: int = 20,
        hidden_size: int = 256,
        num_layers: int = 2,
    ):
        self.seq_len = seq_len
        self.future_k = future_k
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Histories
        self.physics_history: List[List[float]] = []  # [roll, aEgo, vEgo]
        self.control_history: List[List[float]] = []  # [targetLatAcc, steerCommand] + future_k horizon

        control_input_size = 2 + future_k

        self.model = EfficientLSTM(
            physics_input_size=3,
            control_input_size=control_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=0.2
        ).to(self.device)

        # Resolve paths relative to this file if not absolute
        model_abs_path = os.path.join(os.path.dirname(__file__), model_path) if not os.path.isabs(model_path) else model_path
        scaler_abs_path = os.path.join(os.path.dirname(__file__), scaler_path) if not os.path.isabs(scaler_path) else scaler_path

        if os.path.isfile(model_abs_path):
            # Load just the model state_dict (not the full checkpoint)
            model_state_dict = torch.load(model_abs_path, map_location=self.device)
            self.model.load_state_dict(model_state_dict)
            self.model.eval()
            print("Loaded EfficientLSTM model state_dict")
        else:
            raise FileNotFoundError(f"Model weights not found: {model_abs_path}")

        with open(scaler_abs_path, 'rb') as f:
            self.scalers = pickle.load(f)

        # Steer scaler built online for first 100 steps
        self.steer_scaler = None
        self.first_steers: List[float] = []

    # ... rest of the controller methods remain the same ...
    def _scale_basic(self, roll, a_ego, v_ego, target_lataccel):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            roll_s = self.scalers['roll'].transform([[roll]])[0][0]
            a_s = self.scalers['aEgo'].transform([[a_ego]])[0][0]
            v_s = self.scalers['vEgo'].transform([[v_ego]])[0][0]
            tgt_s = self.scalers['targetLateralAcceleration'].transform([[target_lataccel]])[0][0]
        return roll_s, a_s, v_s, tgt_s

    def _scale_steer(self, steer_raw: float) -> float:
        from sklearn.preprocessing import RobustScaler
        if self.steer_scaler is not None:
            return self.steer_scaler.transform([[steer_raw]])[0][0]
        if len(self.first_steers) < 100:
            self.first_steers.append(steer_raw)
            if len(self.first_steers) == 100:
                self.steer_scaler = RobustScaler().fit(np.array(self.first_steers).reshape(-1, 1))
            return steer_raw
        return steer_raw

    def _build_future_vector(self, future_plan):
        arr = future_plan.lataccel[: self.future_k]
        if len(arr) < self.future_k:
            pad_val = arr[-1] if len(arr) > 0 else 0.0
            arr = arr + [pad_val] * (self.future_k - len(arr))
        scaled = [self.scalers['targetLateralAcceleration'].transform([[v]])[0][0] for v in arr]
        return scaled

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        roll = state.roll_lataccel
        a_ego = state.a_ego
        v_ego = state.v_ego

        roll_s, a_s, v_s, tgt_s = self._scale_basic(roll, a_ego, v_ego, target_lataccel)
        steer_s = self._scale_steer(0.0)
        fut_vec = self._build_future_vector(future_plan)

        self.physics_history.append([roll_s, a_s, v_s])
        self.control_history.append([tgt_s, steer_s] + fut_vec)

        if len(self.physics_history) > self.seq_len:
            self.physics_history = self.physics_history[-self.seq_len:]
            self.control_history = self.control_history[-self.seq_len:]

        if len(self.physics_history) < self.seq_len:
            error = (target_lataccel - current_lataccel)
            return 0.3 * error

        physics_tensor = torch.tensor([self.physics_history], dtype=torch.float32, device=self.device)
        control_tensor = torch.tensor([self.control_history], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            out = self.model(physics_tensor, control_tensor)
            pred_scaled = out[0, -1, 0].item()

        if self.steer_scaler is not None:
            pred = self.steer_scaler.inverse_transform([[pred_scaled]])[0][0]
        else:
            pred = pred_scaled

        return float(pred)