from . import BaseController
import torch
import torch.nn as nn
import numpy as np
import pickle
import os
from typing import List
import warnings


class LstmEncoderDecoder(nn.Module):
    def __init__(self, physics_input_size, control_input_size, hidden_size, num_layers, dropout=0.2):
        super().__init__()
        self.physics_encoder = nn.LSTM(physics_input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.control_encoder = nn.LSTM(control_input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.decoder = nn.LSTM(control_input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc_out = nn.Linear(hidden_size, 1)  # Predict steerCommand

    def forward(self, input_physics, input_control_sequence):
        _, (hidden_phsc, cell_phsc) = self.physics_encoder(input_physics)
        _, (hidden_ctrl, cell_ctrl) = self.control_encoder(input_control_sequence)
        hidden_enc = (hidden_phsc + hidden_ctrl) / 2
        cell_enc = (cell_phsc + cell_ctrl) / 2
        decoder_output, _ = self.decoder(input_control_sequence, (hidden_enc, cell_enc))
        return self.fc_out(decoder_output)


class Controller(BaseController):
    """LSTM controller that consumes a fixed-length future plan vector (Option A).

    Expects a model trained with inputs:
        physics: (seq_len, 3) -> [roll, aEgo, vEgo]
        control: (seq_len, 2 + FUTURE_K) -> [targetLatAcc, steerCommand] + FUTURE_K future targetLatAcc values (repeated per timestep)

    At inference we build each timestep's augmented control vector using the *current* future plan horizon.
    This means across the sequence the future vector can change over time (slight distribution shift vs training
    if training used a constant horizon for the whole sample).
    """

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

        self.model = LstmEncoderDecoder(
            physics_input_size=3,
            control_input_size=control_input_size,
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
                # Retroactively rescale history
                for i in range(len(self.control_history)):
                    steer_scaled = self.steer_scaler.transform([[self.control_history[i][1]]])[0][0]
                    self.control_history[i][1] = steer_scaled
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

        roll_s, a_s, v_s, tgt_s = self._scale_basic(roll, a_ego, v_ego, target_lataccel)
        steer_s = self._scale_steer(0.0)  # assume we don't know true steer yet; model predicts it
        fut_vec = self._build_future_vector(future_plan)

        self.physics_history.append([roll_s, a_s, v_s])
        self.control_history.append([tgt_s, steer_s] + fut_vec)

        # Trim
        if len(self.physics_history) > self.seq_len:
            self.physics_history = self.physics_history[-self.seq_len:]
            self.control_history = self.control_history[-self.seq_len:]

        # Not enough context -> simple proportional fallback
        if len(self.physics_history) < self.seq_len:
            error = (target_lataccel - current_lataccel)
            return 0.3 * error

        physics_tensor = torch.tensor([self.physics_history], dtype=torch.float32, device=self.device)
        control_tensor = torch.tensor([self.control_history], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            out = self.model(physics_tensor, control_tensor)
            pred_scaled = out[0, -1, 0].item()

        # Inverse scaling if steer scaler fitted
        if self.steer_scaler is not None:
            pred = self.steer_scaler.inverse_transform([[pred_scaled]])[0][0]
        else:
            pred = pred_scaled  # during warmup still in raw space

        return float(pred)
