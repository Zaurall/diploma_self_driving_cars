from . import BaseController
import torch
import torch.nn as nn
import numpy as np
import pickle
import os
import json
import warnings
from sklearn.preprocessing import RobustScaler
from sklearn.exceptions import DataConversionWarning

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers, output_dim, dropout=0.0):
        super().__init__()
        dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out

class Controller(BaseController):
    """
    LSTM-based controller using the pretrained model
    """
    def __init__(self):
        current_dir = os.path.dirname(__file__)
        model_dir = os.path.abspath(os.path.join(current_dir, "..", "models", "CHEVROLET_VOLT_PREMIER_2017_Oct_13_single"))
        
        with open(os.path.join(model_dir, "config.json"), "r") as f:
            config = json.load(f)

        # Use seq_len from config, not hardcoded
        self.seq_len = 20  # Should be 100 from your config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize history buffer for sequence input
        self.history = []  # Will store [roll, aEgo, vEgo, targetLateralAcceleration]
        
        # Load model
        self.model = LSTMModel(
            input_dim=len(config["data"]["features"]) + config["data"].get("future_k", 0),
            hidden_size=config["model"]["lstm"]["hidden_size"],
            num_layers=config["model"]["lstm"]["num_layers"],
            output_dim=1,
            dropout=config["model"]["lstm"].get("dropout", 0.0)
        ).to(self.device)
        
        # Load model weights
        model_abs_path = os.path.join(model_dir, "best_model.pt") 
        scaler_abs_path = os.path.join(model_dir, "scalers.pkl")
        
        self.model.load_state_dict(torch.load(model_abs_path, map_location=self.device))
        self.model.eval()
        
        # Load global scalers
        with open(scaler_abs_path, 'rb') as f:
            self.scalers = pickle.load(f)
            
        # Initialize per-episode steer scaler (like in your dataset)
        self.steer_scaler = None
        self.first_steps_steers = []
        self.step_count = 0

    def preprocess_inputs(self, roll, a_ego, v_ego, target_lataccel):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            
            # Scale the inputs using global scalers (same as training)
            roll_scaled = self.scalers['roll'].transform([[roll]])[0][0]
            a_ego_scaled = self.scalers['aEgo'].transform([[a_ego]])[0][0]
            v_ego_scaled = self.scalers['vEgo'].transform([[v_ego]])[0][0]
            target_lataccel_scaled = self.scalers['targetLateralAcceleration'].transform([[target_lataccel]])[0][0]
            
            return [roll_scaled, a_ego_scaled, v_ego_scaled, target_lataccel_scaled]

    def update_steer_scaler(self, steer_command):
        """Update the per-episode steer scaler (matching training behavior)"""
        if self.steer_scaler is None:
            self.first_steps_steers.append(steer_command)
            
            # Create scaler when we have enough samples (use seq_len like in training)
            if len(self.first_steps_steers) >= min(self.seq_len, 100):
                self.steer_scaler = RobustScaler()
                self.steer_scaler.fit(np.array(self.first_steps_steers).reshape(-1, 1))

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        self.step_count += 1
        
        # Extract state variables
        roll = state.roll_lataccel
        a_ego = state.a_ego  
        v_ego = state.v_ego
        
        # Preprocess inputs (global scaling)
        scaled_features = self.preprocess_inputs(roll, a_ego, v_ego, target_lataccel)
        
        # Update history buffer
        self.history.append(scaled_features)
        
        # Keep only the last seq_len entries
        if len(self.history) > self.seq_len:
            self.history = self.history[-self.seq_len:]
        
        # If we don't have enough history, use simple PID fallback
        if len(self.history) < self.seq_len:
            error = target_lataccel - current_lataccel
            steer_output = 0.3 * error  # Simple P controller
            
            # Update steer scaler for future use
            self.update_steer_scaler(steer_output)
            return steer_output
            
        # Prepare sequence input for model (pad if necessary)
        sequence = np.array(self.history)
        if len(sequence) < self.seq_len:
            # Pad with zeros or repeat first row
            padding = np.zeros((self.seq_len - len(sequence), sequence.shape[1]))
            sequence = np.vstack([padding, sequence])
        
        # Convert to tensor and add batch dimension
        model_input = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Get prediction from model
        with torch.no_grad():
            output = self.model(model_input)
            # Get the last timestep prediction (most recent)
            prediction = output[0, -1, 0].cpu().item()
        
        # Convert prediction back to original scale if we have the steer scaler
        if self.steer_scaler is not None:
            prediction = self.steer_scaler.inverse_transform([[prediction]])[0][0]
        
        # Update steer scaler with current prediction
        self.update_steer_scaler(prediction)
        
        return prediction