from controllers import BaseController
import numpy as np
import torch
from scipy.signal import butter, filtfilt
from tinyphysics import CONTEXT_LENGTH, CONTROL_START_IDX
from torch import nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 128):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, T, d_model]
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, d_model]
        T = x.size(1)
        return x + self.pe[:, :T]


class TransformerActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        seq_len: int = 1,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        discrete_action: bool = False,
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.seq_len = seq_len
        self.discrete_action = discrete_action

        self.obs_embedding = nn.Linear(obs_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.actor_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, action_dim),
        )

        self.critic_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, obs: torch.Tensor):
        # obs: [B, obs_dim] or [B, T, obs_dim]
        if obs.ndim == 2:
            obs = obs.unsqueeze(1)  # [B, obs_dim] -> [B, 1, obs_dim]

        x = self.obs_embedding(obs)
        x = self.pos_encoder(x)
        x = self.transformer(x)

        seq_repr = x[:, -1, :]  # Use last token

        policy_output = self.actor_head(seq_repr)
        value = self.critic_head(seq_repr)

        return policy_output, value


class TransformerPolicy(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        discrete_action: bool = False,
        seq_len: int = 1,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        log_std_init: float = -0.5,
        device: str = "cpu",
    ):
        super().__init__()

        self.discrete_action = discrete_action
        self.device = torch.device(device)

        self.net = TransformerActorCritic(
            obs_dim=obs_dim,
            action_dim=action_dim,
            seq_len=seq_len,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            discrete_action=discrete_action,
        ).to(self.device)

        if not discrete_action:
            self.log_std = nn.Parameter(
                torch.ones(action_dim, device=self.device) * log_std_init
            )

    @torch.no_grad()
    def forward(self, obs: torch.Tensor):
        obs = obs.to(self.device)
        policy_out, value = self.net(obs)

        if self.discrete_action:
            raise NotImplementedError("Discrete transformer policy not wired into trainer.")
        else:
            mean = policy_out
            std = self.log_std.exp().expand_as(mean)
            return mean, std, value


class Controller(BaseController):
    """
    Transformer-based PPO controller with PID fallback (optimized for inference speed)
    """
    def __init__(self):
        # PID params
        self.p = 0.1
        self.i = 0.9
        self.d = -0.003
        self.alpha = 0.6
        self.error_integral = 0.0
        self.prev_error = 0.0

        # Choose device once
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize transformer policy
        self.policy = TransformerPolicy(
            obs_dim=100,
            action_dim=1,
            discrete_action=False,
            seq_len=1,
            d_model=128,
            nhead=4,
            num_layers=2,
            dim_feedforward=256,
            dropout=0.1,
            log_std_init=-0.5,
            device=str(self.device),
        )

        # Load trained weights to device once
        checkpoint_path = "./models/ppo_transformer_20-rollouts/final_model_weights.pth"
        state_dict = torch.load(checkpoint_path, map_location=self.device)

        missing, unexpected = self.policy.load_state_dict(state_dict, strict=False)
        if unexpected:
            print("Warning: unexpected keys in checkpoint (ignored):", unexpected)
        if missing:
            print("Warning: missing keys in checkpoint (using init values):", missing)

        self.policy.eval()

        # Fixed low-pass filter coefficients (for CPU-side filtering)
        self._lp_b_actions, self._lp_a_actions = butter(2, 0.6 / (0.5 * 10), btype='low', analog=False)
        self._lp_b_target, self._lp_a_target = butter(2, 0.5 / (0.5 * 10), btype='low', analog=False)

        self.value = []
        self.mean = []
        self.pid_action = []

        # Keep histories as numpy arrays for faster slicing
        self.action_history = []
        self.current_lataccel_history = []
        self.target_lataccel_history = []
        self.state_history = None
        self.futureplan = None

    def _lowpass_filter_cached(self, data: np.ndarray, b, a):
        if data.size < 3:
            return data
        return filtfilt(b, a, data)

    def update(self, target_lataccel_history, current_lataccel_history, state_history, action_history, future_plan):
        # Keep references (these may be Python lists; we convert minimally)
        self.state_history = state_history
        self.current_lataccel_history = current_lataccel_history
        self.action_history = action_history
        self.target_lataccel_history = target_lataccel_history
        self.futureplan = future_plan

        # Extract last CONTEXT_LENGTH states
        states = state_history[-CONTEXT_LENGTH:]
        a_ego = np.array([s.a_ego for s in states], dtype=np.float32)
        v_ego = np.array([s.v_ego for s in states], dtype=np.float32)
        roll_lataccel = np.array([s.roll_lataccel for s in states], dtype=np.float32)

        # Pad future plan (in-place, but done once per step)
        if len(self.futureplan.lataccel) < CONTEXT_LENGTH:
            pad_length = CONTEXT_LENGTH - len(self.futureplan.lataccel)

            def pad_list(lst, last_default=0.0):
                last_val = lst[-1] if lst else last_default
                lst.extend([last_val] * pad_length)

            pad_list(self.futureplan.lataccel, 0.0)
            pad_list(self.futureplan.roll_lataccel, 0.0)
            pad_list(self.futureplan.v_ego, 0.0)
            pad_list(self.futureplan.a_ego, 0.0)

        half_ctx = CONTEXT_LENGTH // 2

        # Convert histories to numpy arrays once
        action_hist_np = np.asarray(self.action_history[-half_ctx:], dtype=np.float32)
        curr_lataccel_np = np.asarray(self.current_lataccel_history[-half_ctx:], dtype=np.float32)
        target_lataccel_np = np.asarray(self.target_lataccel_history[-half_ctx:], dtype=np.float32)

        future_lataccel_np = np.asarray(self.futureplan.lataccel[:half_ctx], dtype=np.float32)
        future_a_ego_np = np.asarray(self.futureplan.a_ego[:half_ctx], dtype=np.float32)
        future_roll_lataccel_np = np.asarray(self.futureplan.roll_lataccel[:half_ctx], dtype=np.float32)
        future_v_ego_np = np.asarray(self.futureplan.v_ego[:half_ctx], dtype=np.float32)

        roll_lataccel_half = roll_lataccel[-half_ctx:]
        v_ego_half = v_ego[-half_ctx:]
        a_ego_half = a_ego[-half_ctx:]

        # Build feature matrix [half_ctx, 10], then flatten to [1, 100]
        input_data = np.column_stack(
            (
                action_hist_np,
                roll_lataccel_half,
                v_ego_half,
                a_ego_half,
                curr_lataccel_np,
                target_lataccel_np,
                future_lataccel_np,
                future_a_ego_np,
                future_roll_lataccel_np,
                future_v_ego_np,
            )
        ).astype(np.float32)

        input_tensor = torch.from_numpy(input_data.reshape(1, -1))  # [1, 100]

        # Policy forward on GPU, no grad
        mean, std, value = self.policy(input_tensor)

        mean_item = float(mean.item())
        self.value.append(float(value.item()))
        self.mean.append(mean_item)

        # Low-pass filter target lataccel history
        target_lataccel_np_full = np.asarray(self.target_lataccel_history, dtype=np.float32)
        target_lataccel_filtered = self._lowpass_filter_cached(
            target_lataccel_np_full, self._lp_b_target, self._lp_a_target
        )
        self.target_lataccel_history = target_lataccel_filtered.tolist()
        target_lataccel = self.target_lataccel_history[-1]

        # Weighted average with future plan
        if len(future_plan.lataccel) >= 5:
            weights = np.array([4, 3, 2, 2, 2, 1], dtype=np.float32)
            vals = np.array(
                [target_lataccel] + future_plan.lataccel[0:5],
                dtype=np.float32,
            )
            target_lataccel = float(np.average(vals, weights=weights))

        # PID controller (pure Python math)
        current_lataccel_last = self.current_lataccel_history[-1]
        error = target_lataccel - current_lataccel_last

        if len(self.target_lataccel_history) < CONTROL_START_IDX:
            self.error_integral = 0.0
        else:
            self.error_integral += error * 0.1

        error_diff = (error - self.prev_error) / 0.1
        self.prev_error = error

        pid_action = self.p * error + self.i * self.error_integral + self.d * error_diff
        self.pid_action.append(pid_action)

        # Blend transformer output with PID
        action = self.alpha * mean_item + pid_action
        self.action_history = self.action_history + [action]

        # Low-pass filter actions
        actions_np = np.asarray(self.action_history, dtype=np.float32)
        actions_filtered = self._lowpass_filter_cached(
            actions_np, self._lp_b_actions, self._lp_a_actions
        )
        self.action_history = actions_filtered.tolist()

        return self.action_history[-1]