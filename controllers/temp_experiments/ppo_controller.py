import torch
import torch.nn as nn
import numpy as np
from scipy.signal import butter, filtfilt
from torch.distributions.normal import Normal

# ==========================================
#  CleanRL-style Layer Initialization
# ==========================================
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Orthogonal layer initialization."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


# ==========================================
#  PPO-style Agent Class
# ==========================================
class Agent(nn.Module):
    def __init__(self, n_obs, n_act, device=None):
        super().__init__()
        self.device = device or torch.device("cpu")

        # Critic network (value function)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(n_obs, 64, device=self.device)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 32, device=self.device)),
            nn.Tanh(),
            layer_init(nn.Linear(32, 1, device=self.device), std=1.0),
        )

        # Actor network (mean)
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(n_obs, 64, device=self.device)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 32, device=self.device)),
            nn.Tanh(),
            layer_init(nn.Linear(32, n_act, device=self.device), std=0.01),
        )

        # Trainable log standard deviation
        self.actor_logstd = nn.Parameter(torch.zeros(1, n_act, device=self.device))

    def get_value(self, x):
        """Returns value prediction from critic."""
        return self.critic(x)

    def get_action_and_value(self, obs, action=None):
        """Samples or evaluates an action and returns (action, log_prob, entropy, value)."""
        action_mean = self.actor_mean(obs)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)

        dist = Normal(action_mean, action_std)
        if action is None:
            # Reparameterized sample: mean + std * noise
            action = action_mean + action_std * torch.randn_like(action_mean)

        log_prob = dist.log_prob(action).sum(1)
        entropy = dist.entropy().sum(1)
        value = self.critic(obs)

        return action, log_prob, entropy, value


# ==========================================
#  PID + Neural Policy Controller
# ==========================================
class Controller:
    """
    PID + PPO Policy Blended Controller
    """

    def __init__(self):
        # PID parameters
        self.p = 0.1
        self.i = 0.9
        self.d = -0.003
        self.alpha = 0.6
        self.error_integral = 0.0
        self.prev_error = 0.0

        # PPO Policy setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = Agent(n_obs=(10 * 10), n_act=1, device=self.device)
        checkpoint_path = "./models/PPO_FF-network_20-rollouts/final_model_weights.pth"
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=self.device))

        # Logging
        self.value_log = []
        self.mean_log = []
        self.pid_action_log = []

    # ------------------------------------------
    # Low-pass filter for smooth control signals
    # ------------------------------------------
    def lowpass_filter(self, data, cutoff=0.5, fs=10, order=2):
        nyq = 0.5 * fs
        norm_cutoff = cutoff / nyq
        b, a = butter(order, norm_cutoff, btype="low", analog=False)
        return filtfilt(b, a, data)

    # ------------------------------------------
    # Main Update Function
    # ------------------------------------------
    def update(self, target_lataccel_history, current_lataccel_history, state_history, action_history, future_plan):
        self.state_history = state_history
        self.current_lataccel_history = current_lataccel_history
        self.action_history = action_history
        self.target_lataccel_history = target_lataccel_history
        self.futureplan = future_plan

        # Extract state components
        a_ego = [s.a_ego for s in self.state_history[-CONTEXT_LENGTH:]]
        v_ego = [s.v_ego for s in self.state_history[-CONTEXT_LENGTH:]]
        roll_lataccel = [s.roll_lataccel for s in self.state_history[-CONTEXT_LENGTH:]]

        # Pad future plan if necessary
        if len(self.futureplan.lataccel) < CONTEXT_LENGTH:
            pad_length = CONTEXT_LENGTH - len(self.futureplan.lataccel)
            for attr in ["lataccel", "roll_lataccel", "v_ego", "a_ego"]:
                seq = getattr(self.futureplan, attr)
                seq.extend([seq[-1] if seq else 0.0] * pad_length)

        # Build input feature vector
        input_np = np.column_stack((
            self.action_history[-int(CONTEXT_LENGTH / 2):],
            roll_lataccel[-int(CONTEXT_LENGTH / 2):],
            v_ego[-int(CONTEXT_LENGTH / 2):],
            a_ego[-int(CONTEXT_LENGTH / 2):],
            self.current_lataccel_history[-int(CONTEXT_LENGTH / 2):],
            self.target_lataccel_history[-int(CONTEXT_LENGTH / 2):],
            self.futureplan.lataccel[:int(CONTEXT_LENGTH / 2)],
            self.futureplan.a_ego[:int(CONTEXT_LENGTH / 2)],
            self.futureplan.roll_lataccel[:int(CONTEXT_LENGTH / 2)],
            self.futureplan.v_ego[:int(CONTEXT_LENGTH / 2)],
        ))

        # Convert to tensor
        input_tensor = torch.tensor(input_np, dtype=torch.float32, device=self.device).flatten().unsqueeze(0)

        # Neural policy inference
        with torch.no_grad():
            action_tensor, log_prob, entropy, value = self.policy.get_action_and_value(input_tensor)

        mean_action = action_tensor.item()
        self.value_log.append(value.item())
        self.mean_log.append(mean_action)

        # Apply low-pass filter to target lateral acceleration
        self.target_lataccel_history = self.lowpass_filter(self.target_lataccel_history, cutoff=0.5, fs=10, order=2).tolist()
        target_lataccel = self.target_lataccel_history[-1]

        # Future blending
        if len(future_plan.lataccel) >= 5:
            target_lataccel = np.average(
                [target_lataccel] + future_plan.lataccel[0:5],
                weights=[4, 3, 2, 2, 2, 1],
            )

        # PID error computation
        error = (target_lataccel - self.current_lataccel_history[-1])
        if len(self.target_lataccel_history) < CONTROL_START_IDX:
            self.error_integral = 0.0
        else:
            self.error_integral += error * 0.1

        error_diff = (error - self.prev_error) / 0.1
        self.prev_error = error

        # PID control
        pid_output = self.p * error + self.i * self.error_integral + self.d * error_diff
        self.pid_action_log.append(pid_output)

        # Blended action: neural + PID
        blended_action = self.alpha * mean_action + (1 - self.alpha) * pid_output
        self.action_history.append(blended_action)

        # Smooth output
        self.action_history = self.lowpass_filter(np.array(self.action_history), cutoff=0.6, fs=10, order=2).tolist()

        return self.action_history[-1]
