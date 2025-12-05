import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np


class ActorCritic(nn.Module):
    """
    Actor-Critic network for continuous action space (lateral control)
    """
    
    def __init__(self, obs_dim, action_dim, hidden_sizes=[256, 256], activation='tanh'):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Choose activation function
        if activation == 'relu':
            self.activation = nn.ReLU
        elif activation == 'tanh':
            self.activation = nn.Tanh
        elif activation == 'elu':
            self.activation = nn.ELU
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Shared feature extractor
        self.feature_net = self._build_mlp(obs_dim, hidden_sizes)
        
        # Actor head (policy network)
        self.actor_mean = nn.Linear(hidden_sizes[-1], action_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
        
        # Critic head (value network)
        self.critic = nn.Linear(hidden_sizes[-1], 1)
        
        # Initialize weights
        self._init_weights()
    
    def _build_mlp(self, input_dim, hidden_sizes):
        """Build MLP layers"""
        layers = []
        prev_size = input_dim
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(self.activation())
            prev_size = hidden_size
        
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        
        # Final layers with smaller initialization
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.constant_(self.actor_mean.bias, 0.0)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.constant_(self.critic.bias, 0.0)
    
    def forward(self, obs):
        """Forward pass through both actor and critic"""
        features = self.feature_net(obs)
        
        # Actor output
        action_mean = self.actor_mean(features)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        
        # Critic output
        value = self.critic(features)
        
        return action_mean, action_std, value
    
    def get_action(self, obs, deterministic=False):
        """Sample action from policy"""
        action_mean, action_std, value = self.forward(obs)
        
        if deterministic:
            return action_mean, value
        else:
            # Sample from normal distribution
            dist = Normal(action_mean, action_std)
            action = dist.sample()
            return action, value
    
    def evaluate_actions(self, obs, actions):
        """Evaluate actions for PPO update"""
        action_mean, action_std, value = self.forward(obs)
        
        # Compute log probabilities
        dist = Normal(action_mean, action_std)
        log_probs = dist.log_prob(actions).sum(dim=-1, keepdim=True)
        
        # Compute entropy
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        
        return value, log_probs, entropy
