from . import BaseController
import torch
import torch.nn as nn
import numpy as np
import pickle
import os
from typing import List
import warnings


class LstmEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc_out = nn.Linear(hidden_size, 1)  # Predict steerCommand

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc_out(lstm_out)


class Controller(BaseController):
    """LSTM controller that consumes physics, targetLateralAcceleration, and future plan.
    
    Expects a model trained with inputs:
        input: (seq_len, 4 + FUTURE_K) -> [roll, aEgo, vEgo, targetLateralAcceleration] + FUTURE_K future targetLatAcc values
    Output: steerCommand prediction
    """

    def __init__(
        self,
        model_path="../models/v5_lstm-2-256_10000-dataset_lr-0.001_epochs-20_seq-100_local-scaler_futurek-20_target-steer-filtered/lstm_future_best.pt",
        scaler_path="../models/v5_lstm-2-256_10000-dataset_lr-0.001_epochs-20_seq-100_local-scaler_futurek-20_target-steer-filtered/scalers.pkl",
        seq_len: int = 100,
        future_k: int = 20,
        hidden_size: int = 256,
        num_layers: int = 2,
    ):
        self.seq_len = seq_len
        self.future_k = future_k
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # History of combined input features: [roll, aEgo, vEgo, targetLateralAcceleration] + future_k horizon
        self.input_history: List[List[float]] = []

        input_size = 4 + future_k  # roll, aEgo, vEgo, targetLateralAcceleration + future plan

        self.model = LstmEncoder(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
        ).to(self.device)

        # Resolve paths relative to this file if not absolute
        model_abs_path = os.path.join(os.path.dirname(__file__), model_path) if not os.path.isabs(model_path) else model_path
        scaler_abs_path = os.path.join(os.path.dirname(__file__), scaler_path) if not os.path.isabs(scaler_path) else scaler_path

        if os.path.isfile(model_abs_path):
            self.model.load_state_dict(torch.load(model_abs_path, map_location=self.device))
            self.model.eval()
        else:
            raise FileNotFoundError(f"Model weights not found: {model_abs_path}")

        with open(scaler_abs_path, 'rb') as f:
            self.scalers = pickle.load(f)

        # Steer scaler built online for first 100 steps (mirrors training per-file local robust scaling)
        self.steer_scaler = None
        self.first_steers: List[float] = []

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
        # Collect first 100 then fit
        if len(self.first_steers) < 100:
            self.first_steers.append(steer_raw)
            if len(self.first_steers) == 100:
                self.steer_scaler = RobustScaler().fit(np.array(self.first_steers).reshape(-1, 1))
            return steer_raw  # unscaled during warmup
        return steer_raw

    def _build_future_vector(self, future_plan):
        # future_plan.lataccel is a list of raw future target lateral accel values
        arr = future_plan.lataccel[: self.future_k]
        if len(arr) < self.future_k:
            # pad with last value or zero
            pad_val = arr[-1] if len(arr) > 0 else 0.0
            arr = arr + [pad_val] * (self.future_k - len(arr))
        # Scale each using targetLatAcc scaler
        scaled = [self.scalers['targetLateralAcceleration'].transform([[v]])[0][0] for v in arr]
        return scaled  # length future_k

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        roll = state.roll_lataccel
        a_ego = state.a_ego
        v_ego = state.v_ego

        # Scale input features
        roll_s, a_s, v_s, tgt_s = self._scale_basic(roll, a_ego, v_ego, target_lataccel)
        
        # Build future vector
        fut_vec = self._build_future_vector(future_plan)

        # Create combined input: [roll, aEgo, vEgo, targetLateralAcceleration] + future plan
        combined_input = [roll_s, a_s, v_s, tgt_s] + fut_vec

        # Add to history
        self.input_history.append(combined_input)

        # Trim history
        if len(self.input_history) > self.seq_len:
            self.input_history = self.input_history[-self.seq_len:]

        # Not enough context -> simple proportional fallback
        if len(self.input_history) < self.seq_len:
            error = (target_lataccel - current_lataccel)
            return 0.3 * error

        # Convert to tensor
        input_tensor = torch.tensor([self.input_history], dtype=torch.float32, device=self.device)

        # Predict
        with torch.no_grad():
            out = self.model(input_tensor)
            pred_scaled = out[0, -1, 0].item()  # Get last timestep prediction

        # Inverse scaling if steer scaler fitted
        if self.steer_scaler is not None:
            pred = self.steer_scaler.inverse_transform([[pred_scaled]])[0][0]
        else:
            pred = pred_scaled  # during warmup still in raw space

        return float(pred)