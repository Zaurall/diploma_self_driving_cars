import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from .actor_critic import ActorCritic


class PPOAgent:
    """
    Proximal Policy Optimization Agent
    """
    
    def __init__(
        self,
        obs_dim,
        action_dim,
        hidden_sizes=[256, 256],
        activation='tanh',
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        device='cuda'
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        # Actor-Critic network
        self.ac = ActorCritic(
            obs_dim, 
            action_dim, 
            hidden_sizes, 
            activation
        ).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.ac.parameters(), lr=learning_rate)
        
    def get_action(self, obs, deterministic=False):
        """Get action from policy"""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).to(self.device)
            action, value = self.ac.get_action(obs_tensor, deterministic)
            return action.cpu().numpy(), value.cpu().numpy()
    
    def compute_gae(self, rewards, values, dones, next_values):
        """
        Compute Generalized Advantage Estimation (GAE)
        """
        advantages = np.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = next_values
            else:
                next_value = values[t + 1]
            
            # TD error
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            
            # GAE
            last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
            advantages[t] = last_gae
        
        # Returns = advantages + values
        returns = advantages + values
        
        return advantages, returns
    
    def update(self, rollout_buffer, num_epochs=10, batch_size=256):
        """
        Update policy using PPO
        
        Args:
            rollout_buffer: dict containing rollout data
            num_epochs: number of PPO epochs
            batch_size: mini-batch size
        """
        # Extract rollout data
        obs = torch.FloatTensor(rollout_buffer['obs']).to(self.device)
        actions = torch.FloatTensor(rollout_buffer['actions']).to(self.device)
        old_log_probs = torch.FloatTensor(rollout_buffer['log_probs']).to(self.device)
        advantages = torch.FloatTensor(rollout_buffer['advantages']).to(self.device)
        returns = torch.FloatTensor(rollout_buffer['returns']).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Training statistics
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        num_updates = 0
        
        # Multiple epochs of SGD
        for epoch in range(num_epochs):
            # Generate random mini-batches
            indices = np.random.permutation(len(obs))
            
            for start in range(0, len(obs), batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]
                
                # Mini-batch data
                batch_obs = obs[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Evaluate actions
                values, log_probs, entropy = self.ac.evaluate_actions(batch_obs, batch_actions)
                
                # Policy loss (clipped surrogate objective)
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss (clipped value function)
                value_loss = F.mse_loss(values, batch_returns)
                
                # Entropy bonus
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                # Gradient descent
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Track statistics
                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                num_updates += 1
        
        # Return training statistics
        stats = {
            'loss': total_loss / num_updates,
            'policy_loss': total_policy_loss / num_updates,
            'value_loss': total_value_loss / num_updates,
            'entropy': total_entropy / num_updates
        }
        
        return stats
    
    def save(self, path):
        """Save model weights"""
        torch.save({
            'ac_state_dict': self.ac.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)
    
    def load(self, path):
        """Load model weights"""
        checkpoint = torch.load(path, map_location=self.device)
        self.ac.load_state_dict(checkpoint['ac_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
