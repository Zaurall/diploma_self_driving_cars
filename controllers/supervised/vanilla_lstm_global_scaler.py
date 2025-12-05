from . import BaseController
import torch
import torch.nn as nn
import numpy as np
import pickle
import os

import warnings
from sklearn.exceptions import DataConversionWarning

class LstmEncoderDecoder(nn.Module):
    def __init__(self, physics_input_size, control_input_size, hidden_size, num_layers, dropout=0.2):
        super().__init__()
        self.physics_encoder = nn.LSTM(physics_input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.control_encoder = nn.LSTM(control_input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.decoder = nn.LSTM(control_input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc_out = nn.Linear(hidden_size, 1)  # Predict steerCommand

    def forward(self, input_physics, input_control_sequence, output_control_sequence=None):
        _, (hidden_phsc, cell_phsc) = self.physics_encoder(input_physics)
        _, (hidden_ctrl, cell_ctrl) = self.control_encoder(input_control_sequence)
        
        hidden_enc = (hidden_phsc + hidden_ctrl) / 2
        cell_enc = (cell_phsc + cell_ctrl) / 2
        
        # Use input_control_sequence as output_control_sequence if not provided
        decoder_input = input_control_sequence if output_control_sequence is None else output_control_sequence
        
        decoder_output, _ = self.decoder(decoder_input, (hidden_enc, cell_enc))
        return self.fc_out(decoder_output)

class Controller(BaseController):
    """
    LSTM-based controller using the pretrained model
    """
    def __init__(self, model_path="../models/lstm_v3_1000-dataset_lr-3_epochs-20_seq-len-100_global-scaler/lstm_best_model.pt", scaler_path="../models/lstm_v3_1000-dataset_lr-3_epochs-20_seq-len-100_global-scaler/scalers.pkl"):
        # model_path = "../models/base_v1_full_dataset_10_epochs/lstm_best_model.pt"
        # model_path = "../models/base_v1_full_dataset_10_epochs/lstm_best_model_2_epochs_6280885_test-.pt"
        self.seq_len = 20
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize history buffers
        self.physics_history = []  # [roll, aEgo, vEgo]
        self.control_history = []  # [targetLateralAcceleration, steerCommand]
        
        # Load model
        self.model = LstmEncoderDecoder(
            physics_input_size=3,
            control_input_size=2,
            hidden_size=128,
            num_layers=4
        ).to(self.device)
        
        # Check if model path is absolute or relative
        model_abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), model_path))
        scaler_abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), scaler_path))
        
        self.model.load_state_dict(torch.load(model_abs_path, map_location=self.device))
        self.model.eval()
        
        # Load scalers
        with open(scaler_abs_path, 'rb') as f:
            self.scalers = pickle.load(f)
            
        required_scalers = ['roll', 'aEgo', 'vEgo', 'targetLateralAcceleration', 'steerCommand']            
        for scaler in required_scalers:            
            if scaler not in self.scalers:            
                raise ValueError(f"Missing required scaler: {scaler}")

    def preprocess_inputs(self, roll, a_ego, v_ego, target_lataccel, steer_command):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            
            scaled_values = [            
                self.scalers['roll'].transform([[roll]])[0][0],            
                self.scalers['aEgo'].transform([[a_ego]])[0][0],            
                self.scalers['vEgo'].transform([[v_ego]])[0][0],            
                self.scalers['targetLateralAcceleration'].transform([[target_lataccel]])[0][0],            
                self.scalers['steerCommand'].transform([[steer_command]])[0][0]            
            ]
            
            return tuple(scaled_values)

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        # Extract state variables
        roll = state.roll_lataccel
        a_ego = state.a_ego  
        v_ego = state.v_ego


        current_steer = 0.0 if not self.control_history else self.control_history[-1][1]
        scaled_inputs = self.preprocess_inputs(
            roll, a_ego, v_ego, target_lataccel, current_steer
        )
        roll_scaled, a_ego_scaled, v_ego_scaled, target_lataccel_scaled, steer_command_scaled = scaled_inputs
        
        
        # Update history buffers
        self.physics_history.append([roll_scaled, a_ego_scaled, v_ego_scaled])
        self.control_history.append([target_lataccel_scaled, steer_command_scaled])
        
        # Keep only the last seq_len entries
        if len(self.physics_history) > self.seq_len:
            self.physics_history = self.physics_history[-self.seq_len:]
            self.control_history = self.control_history[-self.seq_len:]
        
        # If we don't have enough history, use PID as fallback
        if len(self.physics_history) < self.seq_len:
            # Simple PID implementation
            error = (target_lataccel - current_lataccel)
            return 0.3 * error  # Simple P controller
            
        # Prepare inputs for the model
        physics_input = torch.tensor([self.physics_history], dtype=torch.float32).to(self.device)
        control_input = torch.tensor([self.control_history], dtype=torch.float32).to(self.device)
        
        # Get prediction from model
        with torch.no_grad():
            output = self.model(physics_input, control_input)
            prediction_scaled = output[0, -1, 0].cpu().item()  # Last timestep prediction
        
        prediction = self.scalers['steerCommand'].inverse_transform([[prediction_scaled]])[0][0]
            
        return prediction