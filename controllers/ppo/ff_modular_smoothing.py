from controllers import BaseController
import numpy as np
import torch
from scipy.signal import butter, filtfilt
from tinyphysics import CONTEXT_LENGTH, CONTROL_START_IDX
import torch
from torch import nn
import numpy as np

class FFNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_sizes, activation='relu', dropout_rate=0.0):
        super(FFNN, self).__init__()

        if activation == 'relu':
            self.activation = nn.ReLU
        elif activation == 'tanh':
            self.activation = nn.Tanh
        elif activation == 'elu':
            self.activation = nn.ELU
        else:
            raise ValueError(f"Unknown activation: {activation}")

        prev_size = input_dim
        layers = []
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(self.activation())
            if dropout_rate > 0:
                layers.append(nn.Dropout(p=dropout_rate))
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float)
        return self.model(x)

class FFPPOPolicy(nn.Module):
    def __init__(self, input_dim, hidden_dim=[64, 32], activation='tanh', dropout_rate=0.0):
        super().__init__()

        self.actor = FFNN(
            input_dim=input_dim, 
            output_dim=1,  # mean output for steering
            hidden_sizes=hidden_dim, 
            activation=activation
        )

        self.critic = FFNN(
            input_dim=input_dim, 
            output_dim=1, 
            hidden_sizes=hidden_dim, 
            activation=activation
        )

        self.log_std = nn.Parameter(torch.zeros(1))  # trainable log std

    def forward(self, x):
        # last time step's output
        mean = self.actor(x)
        std = self.log_std.exp()
        value = self.critic(x)
        return mean, std, value  


class Controller(BaseController):
    """
    A simple PID controller
    """
    def __init__(self,):
        self.p = 0.1
        self.i = 0.9
        self.d = -0.003
        self.alpha = 0.61
        self.alpha_ramp_steps = 10
        self.error_integral = 0
        self.prev_error = 0
        self.prev_action = 0.0          # <-- for EMA smoothing

        self.policy = FFPPOPolicy(
            input_dim=(10 * 10),
            hidden_dim=[256, 128, 64],
            activation='tanh', 
            # dropout_rate=0.2
        )
        checkpoint_path = './models/ppo_ff_tanh_256-128-64_dropout-0_100-rollouts/best_model_weights.pth'
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
        self.value = []
        self.mean = []
        self.pid_action=[]
    
    def lowpass_filter(self,data, cutoff=0.5, fs=10, order=2):
        nyq = 0.5 * fs
        norm_cutoff = cutoff / nyq
        b, a = butter(order, norm_cutoff, btype='low', analog=False)
        return filtfilt(b, a, data)  # Zero-phase filtering
        
    def update(self, target_lataccel_history, current_lataccel_history, state_history, action_history, future_plan ):
        self.state_history = state_history
        self.current_lataccel_history = current_lataccel_history
        
        self.action_history = action_history
        self.target_lataccel_history = target_lataccel_history
        self.futureplan = future_plan
            
        a_ego=[s.a_ego for s in self.state_history[-CONTEXT_LENGTH:]]
        v_ego=[s.v_ego for s in self.state_history[-CONTEXT_LENGTH:]]
        roll_lataccel=[s.roll_lataccel for s in self.state_history[-CONTEXT_LENGTH:]]
        if len(self.futureplan.lataccel) < CONTEXT_LENGTH:
            # Pad future plan with zeros if not enough data
            pad_length = CONTEXT_LENGTH - len(self.futureplan.lataccel)
            last_val = self.futureplan.lataccel[-1] if self.futureplan.lataccel else 0.0
            self.futureplan.lataccel.extend([last_val] * pad_length)
            last_val = self.futureplan.roll_lataccel[-1] if self.futureplan.roll_lataccel else 0.0
            self.futureplan.roll_lataccel.extend([last_val] * pad_length)
            last_val = self.futureplan.v_ego[-1] if self.futureplan.v_ego else 0.0
            self.futureplan.v_ego.extend([last_val] * pad_length)
            last_val = self.futureplan.a_ego[-1] if self.futureplan.a_ego else 0.0
            self.futureplan.a_ego.extend([last_val] * pad_length)
            
        input = np.column_stack((
            self.action_history[-int(CONTEXT_LENGTH/2):],
            roll_lataccel[-int(CONTEXT_LENGTH/2):],
            v_ego[-int(CONTEXT_LENGTH/2):],
            a_ego[-int(CONTEXT_LENGTH/2):],
            self.current_lataccel_history[-int(CONTEXT_LENGTH/2):],
            self.target_lataccel_history[-int(CONTEXT_LENGTH/2):],
            self.futureplan.lataccel[:int(CONTEXT_LENGTH/2)],
            self.futureplan.a_ego[:int(CONTEXT_LENGTH/2)],
            self.futureplan.roll_lataccel[:int(CONTEXT_LENGTH/2)],
            self.futureplan.v_ego[:int(CONTEXT_LENGTH/2)]
        ))
        input_tensor = torch.tensor(input, dtype=torch.float32).flatten().unsqueeze(0)
        mean, std, value = self.policy(input_tensor)
        self.value.append(value.item())
        self.mean.append(mean.item())

        self.target_lataccel_history = self.lowpass_filter(
            self.target_lataccel_history, cutoff=0.5, fs=10, order=2
        ).tolist()
        target_lataccel = self.target_lataccel_history[-1]
        if len(future_plan.lataccel) >= 5:
            target_lataccel = np.average(
                [target_lataccel] + future_plan.lataccel[0:5],
                weights=[4, 3, 2, 2, 2, 1]
            )
        error = (target_lataccel - self.current_lataccel_history[-1])
        if len(self.target_lataccel_history) < CONTROL_START_IDX:
            self.error_integral = 0
        else:
            self.error_integral += error * 0.1
        error_diff = (error - self.prev_error) / 0.1
        self.prev_error = error
        pid_term = self.p * error + self.i * self.error_integral + self.d * error_diff
        self.pid_action.append(pid_term)

        # --- fade‑in PPO influence ---
        steps_since_control_start = max(0, len(self.target_lataccel_history) - CONTROL_START_IDX)
        ramp = max(0.0, min(1.0, steps_since_control_start / float(self.alpha_ramp_steps)))
        effective_alpha = self.alpha

        # raw (unsmoothed) action
        raw_action = effective_alpha * mean.item() + pid_term

        # --- EMA smoothing on action ---
        ema_beta = 0.8  # smaller => smoother; try 0.2–0.5
        smoothed_action = ema_beta * raw_action + (1.0 - ema_beta) * self.prev_action
        self.prev_action = smoothed_action

        self.action_history = self.action_history + [smoothed_action]

        # additional low‑pass on whole history
        self.action_history = self.lowpass_filter(
            np.array(self.action_history), cutoff=0.6, fs=10, order=2
        ).tolist()
        return self.action_history[-1]