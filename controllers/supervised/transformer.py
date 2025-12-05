from . import BaseController
import torch
import torch.nn as nn
import numpy as np
import pickle
import os
from typing import List

import warnings
warnings.filterwarnings("ignore", message="Converting mask without torch.bool dtype to bool")

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
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(physics_embedded.device)
        
        # Transformer forward pass
        # Use physics as encoder input and control as decoder input
        output = self.transformer(
            src=physics_embedded,
            tgt=control_embedded,
            tgt_mask=tgt_mask
        )
        
        return self.fc_out(output)


class Controller(BaseController):
    """Transformer controller that consumes physics, targetLateralAcceleration, and future plan.
    
    Expects a model trained with inputs:
        physics: [roll, aEgo, vEgo]
        control: [targetLateralAcceleration, steerCommand] + FUTURE_K future targetLatAcc values
    Output: steerCommand prediction
    """

    def __init__(
        self,
        model_path="../models/v1_transformer-2-2-256_10000-dataset_lr-0.001_epochs-20_seq-100_local-scaler_futurek-20_target-steer-filtered/transformer_future_best.pt",
        scaler_path="../models/v1_transformer-2-2-256_10000-dataset_lr-0.001_epochs-20_seq-100_local-scaler_futurek-20_target-steer-filtered/scalers.pkl",
        seq_len: int = 100,
        future_k: int = 20,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 512,
    ):
        self.seq_len = seq_len
        self.future_k = future_k
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Separate histories for physics and control inputs
        self.physics_history: List[List[float]] = []  # [roll, aEgo, vEgo]
        self.control_history: List[List[float]] = []  # [targetLateralAcceleration, steerCommand] + future plan

        physics_input_size = 3  # roll, aEgo, vEgo
        control_input_size = 2 + future_k  # targetLateralAcceleration, steerCommand + future plan

        self.model = TransformerEncoderDecoder(
            physics_input_size=physics_input_size,
            control_input_size=control_input_size,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=0.1
        ).to(self.device)

        # Resolve paths relative to this file if not absolute
        model_abs_path = os.path.join(os.path.dirname(__file__), model_path) if not os.path.isabs(model_path) else model_path
        scaler_abs_path = os.path.join(os.path.dirname(__file__), scaler_path) if not os.path.isabs(scaler_path) else scaler_path

        if os.path.isfile(model_abs_path):
            self.model.load_state_dict(torch.load(model_abs_path, map_location=self.device))
            self.model.eval()
            print(f"Loaded Transformer encoder-decoder model from {model_abs_path}")
        else:
            raise FileNotFoundError(f"Transformer model weights not found: {model_abs_path}")

        with open(scaler_abs_path, 'rb') as f:
            self.scalers = pickle.load(f)

        # Steer scaler built online for first 100 steps
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
                print("Steer scaler fitted after collecting 100 samples")
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

        # Create physics input: [roll, aEgo, vEgo]
        physics_input = [roll_s, a_s, v_s]
        
        # Create control input: [targetLateralAcceleration, steerCommand] + future plan
        # For current steerCommand, we need to use the previous prediction or a default
        if len(self.control_history) > 0:
            prev_steer = self.control_history[-1][1]  # Previous steerCommand
        else:
            prev_steer = 0.0  # Initial value
        
        control_input = [tgt_s, prev_steer] + fut_vec

        # Add to histories
        self.physics_history.append(physics_input)
        self.control_history.append(control_input)

        # Trim histories
        if len(self.physics_history) > self.seq_len:
            self.physics_history = self.physics_history[-self.seq_len:]
            self.control_history = self.control_history[-self.seq_len:]

        # Not enough context -> simple proportional fallback
        if len(self.physics_history) < self.seq_len:
            error = (target_lataccel - current_lataccel)
            return 0.3 * error

        # Convert to tensors
        physics_tensor = torch.tensor([self.physics_history], dtype=torch.float32, device=self.device)
        control_tensor = torch.tensor([self.control_history], dtype=torch.float32, device=self.device)

        # Predict
        with torch.no_grad():
            out = self.model(physics_tensor, control_tensor)
            pred_scaled = out[0, -1, 0].item()  # Get last timestep prediction

        # Update the control history with the new prediction
        if len(self.control_history) > 0:
            self.control_history[-1][1] = pred_scaled  # Update steerCommand in the latest control input

        # Inverse scaling if steer scaler fitted
        if self.steer_scaler is not None:
            pred = self.steer_scaler.inverse_transform([[pred_scaled]])[0][0]
        else:
            pred = pred_scaled  # during warmup still in raw space

        return float(pred)