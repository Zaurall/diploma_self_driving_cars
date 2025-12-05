import numpy as np


class RolloutBuffer:
    """
    Buffer for storing rollout experiences for PPO
    """
    
    def __init__(self, num_envs, num_steps, obs_dim, action_dim):
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Pre-allocate arrays
        self.observations = np.zeros((num_steps, num_envs, obs_dim), dtype=np.float32)
        self.actions = np.zeros((num_steps, num_envs, action_dim), dtype=np.float32)
        self.rewards = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.values = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.log_probs = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.dones = np.zeros((num_steps, num_envs), dtype=np.float32)
        
        self.ptr = 0
        self.full = False
    
    def add(self, obs, action, reward, value, log_prob, done):
        """Add a transition to the buffer"""
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = done
        
        self.ptr += 1
        
        if self.ptr == self.num_steps:
            self.full = True
    
    def get(self, next_values, advantages=None, returns=None):
        """
        Get all data from buffer
        
        If advantages and returns are provided, use them.
        Otherwise, they should be computed externally.
        """
        assert self.full, "Buffer is not full"
        
        # Flatten batch dimensions
        obs = self.observations.reshape(-1, self.obs_dim)
        actions = self.actions.reshape(-1, self.action_dim)
        log_probs = self.log_probs.reshape(-1)
        
        # If advantages/returns provided, flatten them
        if advantages is not None:
            advantages = advantages.reshape(-1)
        if returns is not None:
            returns = returns.reshape(-1)
        
        data = {
            'obs': obs,
            'actions': actions,
            'log_probs': log_probs,
            'advantages': advantages,
            'returns': returns
        }
        
        return data
    
    def reset(self):
        """Reset the buffer"""
        self.ptr = 0
        self.full = False
