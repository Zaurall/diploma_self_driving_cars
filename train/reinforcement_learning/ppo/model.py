import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
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
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, d_model]
        """
        T = x.size(1)
        return x + self.pe[:, :T]


class TransformerActorCritic(nn.Module):
    """
    Simple Transformer-based actor-critic for PPO.

    Assumptions:
      - Observations are flat vectors of length obs_dim.
      - We treat a window of observations as a sequence of length seq_len.
      - If you only have single-step obs, set seq_len=1 in the policy wrapper.
    """

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
    ) -> None:
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.seq_len = seq_len
        self.discrete_action = discrete_action

        # Project raw observation vector to model dimension
        self.obs_embedding = nn.Linear(obs_dim, d_model)

        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # [B, T, D]
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # We pool over the sequence using the last token representation.
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

        # Optional: initialize a bit more carefully
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        obs: [B, obs_dim] or [B, T, obs_dim]
        returns:
          - policy_logits or mean: [B, action_dim]
          - value: [B, 1]
        """
        if obs.ndim == 2:
            # [B, obs_dim] -> [B, 1, obs_dim]
            obs = obs.unsqueeze(1)

        # [B, T, obs_dim] -> [B, T, d_model]
        x = self.obs_embedding(obs)
        x = self.pos_encoder(x)

        # [B, T, d_model]
        x = self.transformer(x)

        # Use last token as sequence representation
        seq_repr = x[:, -1, :]  # [B, d_model]

        policy_output = self.actor_head(seq_repr)
        value = self.critic_head(seq_repr)

        return policy_output, value