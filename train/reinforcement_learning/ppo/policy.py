import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

from train.reinforcement_learning.ppo.model import FFNN, TransformerActorCritic


class TransformerPPOPolicy(nn.Module):
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
    ) -> None:
        super().__init__()

        self.discrete_action = discrete_action

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
        )

        if not discrete_action:
            # log_std is a learned parameter for continuous actions
            self.log_std = nn.Parameter(
                torch.ones(action_dim) * log_std_init
            )

    def forward(self, obs: torch.Tensor):
        # policy_out: [B, action_dim], value: [B, 1]
        policy_out, value = self.net(obs)

        if self.discrete_action:
            # For discrete actions we output logits; trainer would need Categorical
            raise NotImplementedError("Discrete transformer policy not wired into trainer.")
        else:
            mean = policy_out                       # [B, action_dim]
            std = self.log_std.exp().expand_as(mean)
            return mean, std, value                 # value: [B, 1]

    @torch.no_grad()
    def act(self, obs: torch.Tensor):
        mean, std, value = self.forward(obs)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob, value.squeeze(-1)

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        mean, std, value = self.forward(obs)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy, value.squeeze(-1)

    



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


class FFComplexPPOPolicy(nn.Module):
    def __init__(self, input_dim, hidden_dim=[256, 128, 64], activation='relu', use_batch_norm=True, dropout_rate=0.2):
        super().__init__()

        # Shared feature extraction network
        self.shared_net = FFNN(
            input_dim=input_dim,
            output_dim=hidden_dim[-1],  # Output of the last shared hidden layer
            hidden_sizes=hidden_dim[:-1],
            activation=activation,
            use_batch_norm=use_batch_norm,
            dropout_rate=dropout_rate
        )

        # Actor head
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim[-1], hidden_dim[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dim[-1], 1)  # Mean output for steering
        )

        # Critic head
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim[-1], hidden_dim[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dim[-1], 1)
        )

        self.log_std = nn.Parameter(torch.zeros(1))  # Trainable log std

    def forward(self, x):
        # Pass input through the shared network
        shared_features = self.shared_net(x)

        # Get actor and critic outputs from their respective heads
        mean = self.actor(shared_features)
        value = self.critic(shared_features)
        std = self.log_std.exp()

        return mean, std, value