import gym
from gym import spaces
import numpy as np
import pandas as pd
import os
import sys

# Add parent directory to path to import tinyphysics
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator, CONTROL_START_IDX


class LateralControlEnv(gym.Env):
    """
    Gym environment for lateral control using tinyphysics simulator
    """
    
    def __init__(self, data_file, model_path='../models/tinyphysics.onnx', max_steps=1000):
        super().__init__()
        
        self.data_file = data_file
        self.model_path = model_path
        self.max_steps = max_steps
        
        # Load data and initialize simulator
        self.data = pd.read_csv(data_file)
        self.model = TinyPhysicsModel(model_path, debug=False)
        self.sim = TinyPhysicsSimulator(self.model, self.data, controller=None, debug=False)
        
        # Action space: continuous steering command [-1, 1]
        self.action_space = spaces.Box(low=-2.0, high=2.0, shape=(1,), dtype=np.float32)
        
        # Observation space: [roll, v_ego, a_ego, target_lataccel, current_lataccel, lataccel_error]
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(6,), 
            dtype=np.float32
        )
        
        self.current_step = 0
        self.total_cost = 0.0
        
    def reset(self):
        """Reset the environment"""
        # Reload data and reset simulator
        self.data = pd.read_csv(self.data_file)
        self.model = TinyPhysicsModel(self.model_path, debug=False)
        self.sim = TinyPhysicsSimulator(self.model, self.data, controller=None, debug=False)
        
        self.current_step = 0
        self.total_cost = 0.0
        
        # Get initial observation
        state = self._get_state()
        return state
    
    def _get_state(self):
        """Get current state observation"""
        idx = self.current_step + CONTROL_START_IDX
        
        if idx >= len(self.data):
            # Return zero state if out of bounds
            return np.zeros(6, dtype=np.float32)
        
        row = self.data.iloc[idx]
        
        # State vector
        roll = row['roll']
        v_ego = row['vEgo']
        a_ego = row['aEgo']
        target_lataccel = row['targetLateralAcceleration']
        
        # Get current lateral acceleration from simulator
        current_lataccel = self.sim.current_lataccel if hasattr(self.sim, 'current_lataccel') else 0.0
        
        # Compute error
        lataccel_error = target_lataccel - current_lataccel
        
        state = np.array([
            roll,
            v_ego,
            a_ego,
            target_lataccel,
            current_lataccel,
            lataccel_error
        ], dtype=np.float32)
        
        return state
    
    def step(self, action):
        """Execute one step in the environment"""
        # Extract steering command from action
        steer_command = float(action[0])
        
        # Clip steering command to reasonable range
        steer_command = np.clip(steer_command, -2.0, 2.0)
        
        # Get current target lateral acceleration
        idx = self.current_step + CONTROL_START_IDX
        
        if idx >= len(self.data):
            # Episode done
            return self._get_state(), 0.0, True, {}
        
        target_lataccel = self.data.iloc[idx]['targetLateralAcceleration']
        
        # Step simulator with steering command
        self.sim.step(steer_command)
        current_lataccel = self.sim.current_lataccel
        
        # Compute instantaneous cost
        cost = self._compute_cost(target_lataccel, current_lataccel, steer_command)
        
        # Reward is negative cost
        reward = -cost
        
        # Update counters
        self.current_step += 1
        self.total_cost += cost
        
        # Check if episode is done
        done = (self.current_step >= self.max_steps) or (idx + 1 >= len(self.data))
        
        # Get next state
        next_state = self._get_state()
        
        # Info dict
        info = {
            'cost': cost,
            'total_cost': self.total_cost,
            'target_lataccel': target_lataccel,
            'current_lataccel': current_lataccel,
            'steer_command': steer_command
        }
        
        return next_state, reward, done, info
    
    def _compute_cost(self, target_lataccel, current_lataccel, steer_command):
        """
        Compute instantaneous cost
        Cost = w1 * lataccel_error^2 + w2 * steer_command^2 + w3 * jerk^2
        """
        # Lateral acceleration tracking error
        lataccel_error = target_lataccel - current_lataccel
        lataccel_cost = lataccel_error ** 2
        
        # Steering effort (penalize large steering commands)
        steer_cost = 0.01 * (steer_command ** 2)
        
        # Jerk penalty (penalize rapid steering changes)
        if hasattr(self, 'prev_steer_command'):
            jerk = steer_command - self.prev_steer_command
            jerk_cost = 0.001 * (jerk ** 2)
        else:
            jerk_cost = 0.0
        
        self.prev_steer_command = steer_command
        
        total_cost = lataccel_cost + steer_cost + jerk_cost
        
        return total_cost
    
    def render(self, mode='human'):
        """Render the environment (optional)"""
        pass
    
    def close(self):
        """Close the environment"""
        pass


class VectorizedLateralControlEnv:
    """
    Vectorized environment wrapper for parallel rollouts
    """
    
    def __init__(self, data_files, model_path='../models/tinyphysics.onnx', max_steps=1000):
        self.num_envs = len(data_files)
        self.envs = [
            LateralControlEnv(data_file, model_path, max_steps)
            for data_file in data_files
        ]
        
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
    
    def reset(self):
        """Reset all environments"""
        return np.array([env.reset() for env in self.envs])
    
    def step(self, actions):
        """Step all environments"""
        results = [env.step(action) for env, action in zip(self.envs, actions)]
        
        states = np.array([r[0] for r in results])
        rewards = np.array([r[1] for r in results])
        dones = np.array([r[2] for r in results])
        infos = [r[3] for r in results]
        
        # Auto-reset done environments
        for i, done in enumerate(dones):
            if done:
                states[i] = self.envs[i].reset()
        
        return states, rewards, dones, infos
    
    def close(self):
        """Close all environments"""
        for env in self.envs:
            env.close()
