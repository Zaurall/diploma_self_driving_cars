import torch
import numpy as np
from pathlib import Path
import sys
import os

project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import matplotlib.pyplot as plt
from time import time
from tqdm import tqdm
import random
from typing import List
from torch.utils.data import DataLoader, TensorDataset
from tinyphysics import TinyPhysicsModel, CONTROL_START_IDX
from train.reinforcement_learning.ppo.policy import TransformerPPOPolicy, FFPPOPolicy
from train.reinforcement_learning.ppo.environment import PPOEnv

from tinyphysics import CONTEXT_LENGTH

class PPOTrainer:
    def __init__(
        self, 
        model: TinyPhysicsModel,
        policy, # Add typing
        data_path: str, 
        gamma=0.99, 
        lam=0.95, 
        clip_eps=0.2, 
        epochs=10, 
        batch_size=64,
        minibatch_size=256,
        lr=3e-4,
        debug: bool = False
    ) -> None:
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.policy = policy.to(self.device)
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size
        self.minibatch_size = minibatch_size
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.env_list = self.create_env(data_path)
        
        self.policy_logs = {
            "mean": [],
            "std": [],
            "value": [],
            "entropy": [],
            "log_prob": [],
            "reward": [],
            "policy_loss": [],
            "value_loss": [],
            "total_loss": []
        }

    def create_env(self, data_path: str) -> List[str]:
        data_path = Path(data_path)
        if data_path.is_file():
            return [str(data_path)]
        if data_path.is_dir():
            files = sorted(data_path.glob('*.csv'))[:5000]
        return [str(f) for f in files]

    def sample_env_batch(self, batch_size=4):
        return random.sample(self.env_list, min(batch_size, len(self.env_list)))

    def compute_gae_vectorized(self, rewards, values, dones):
        """
        Compute Generalized Advantage Estimation in vectorized manner
        
        Args:
            rewards: list of rewards
            values: list of values
            dones: list of done flags
        
        Returns:
            advantages: torch.Tensor
            returns: torch.Tensor
        """
        rewards = np.array(rewards)
        values = np.array(values)
        dones = np.array(dones)
        
        advantages = np.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            last_gae = delta + self.gamma * self.lam * (1 - dones[t]) * last_gae
            advantages[t] = last_gae
        
        advantages = torch.FloatTensor(advantages)
        returns = advantages + torch.FloatTensor(values)
        
        return advantages, returns

    def run_rollouts_batch(self, env_paths):
        """
        Run multiple environments and collect all data efficiently
        
        Args:
            env_paths: list of paths to environment data files
            
        Returns:
            dict containing batched rollout data
        """
        all_data = {
            'obs': [],
            'actions': [],
            'log_probs': [],
            'values': [],
            'rewards': [],
            'dones': []
        }
        
        all_rewards = []
        
        # Collect data from all environments
        for env_path in env_paths:
            try:
                env = PPOEnv(self.model, env_path, self.policy, debug=False)
                env.reset()
                env.rollout_buffer = []
                
                done = False
                while not done:
                    _, _, _, done = env.step(evaluation=False)
                
                # Extract data from rollout buffer
                if len(env.rollout_buffer) == 0:
                    continue
                
                env_rewards = []
                for exp in env.rollout_buffer:
                    all_data['obs'].append(exp['obs'])
                    all_data['actions'].append(torch.tensor([[exp['action']]], dtype=torch.float32))
                    all_data['log_probs'].append(torch.tensor([[exp['log_prob']]], dtype=torch.float32))
                    all_data['values'].append(exp['value'])
                    all_data['rewards'].append(exp['reward'])
                    all_data['dones'].append(exp['done'])
                    env_rewards.append(exp['reward'])
                
                all_rewards.append(np.mean(env_rewards))
                
            except Exception as e:
                print(f"Error in rollout for {env_path}: {e}")
                continue
        
        return all_data, all_rewards

    def update_policy_efficient(self, rollout_data):
        """
        Efficient policy update using DataLoader for mini-batch training
        
        Args:
            rollout_data: dict containing all rollout data
            
        Returns:
            bool: True if training should stop, False otherwise
        """
        if len(rollout_data['obs']) == 0:
            print("No data to update policy")
            return False
        
        # Stack all observations
        obs = torch.cat(rollout_data['obs'], dim=0).to(self.device)
        actions = torch.cat(rollout_data['actions'], dim=0).to(self.device)
        old_log_probs = torch.cat(rollout_data['log_probs'], dim=0).to(self.device)
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae_vectorized(
            rollout_data['rewards'],
            rollout_data['values'],
            rollout_data['dones']
        )
        
        advantages = advantages.to(self.device)
        returns = returns.to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Create dataset and dataloader
        dataset = TensorDataset(obs, actions, old_log_probs, returns, advantages)
        dataloader = DataLoader(
            dataset, 
            batch_size=self.minibatch_size, 
            shuffle=True,
            drop_last=False
        )
        
        # Multiple epochs of SGD
        for epoch in range(self.epochs):
            epoch_policy_loss = 0
            epoch_value_loss = 0
            epoch_total_loss = 0
            epoch_mean = 0
            epoch_std = 0
            epoch_value = 0
            epoch_entropy = 0
            epoch_log_prob = 0
            num_batches = 0
            
            for batch_obs, batch_actions, batch_old_log_probs, batch_returns, batch_advantages in dataloader:
                # Forward pass
                mean, std, values = self.policy(batch_obs)
                
                # Early stopping check
                if std.mean().item() < 0.3:
                    print("Training stopped due to low std deviation in policy output.")
                    return True
                
                # Create distribution
                dist = torch.distributions.Normal(mean, std)
                log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                # PPO policy loss
                ratio = (log_probs - batch_old_log_probs).exp()
                surr1 = ratio * batch_advantages.unsqueeze(-1)
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * batch_advantages.unsqueeze(-1)
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = ((batch_returns.unsqueeze(-1) - values) ** 2).mean()
                
                # Total loss
                total_loss = 100 * policy_loss + 0.5 * value_loss - 0.02 * entropy
                
                # Optimization step
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                self.optimizer.step()
                
                # Accumulate statistics
                epoch_policy_loss += policy_loss.item()
                epoch_value_loss += value_loss.item()
                epoch_total_loss += total_loss.item()
                epoch_mean += mean.mean().item()
                epoch_std += std.mean().item()
                epoch_value += values.mean().item()
                epoch_entropy += entropy.item()
                epoch_log_prob += log_probs.mean().item()
                num_batches += 1
            
            # Log epoch statistics (only last epoch)
            if epoch == self.epochs - 1:
                self.policy_logs["mean"].append(epoch_mean / num_batches)
                self.policy_logs["std"].append(epoch_std / num_batches)
                self.policy_logs["value"].append(epoch_value / num_batches)
                self.policy_logs["entropy"].append(epoch_entropy / num_batches)
                self.policy_logs["log_prob"].append(epoch_log_prob / num_batches)
                self.policy_logs["policy_loss"].append(epoch_policy_loss / num_batches)
                self.policy_logs["value_loss"].append(epoch_value_loss / num_batches)
                self.policy_logs["total_loss"].append(epoch_total_loss / num_batches)
        
        return False

    def evaluate_policy(self, env_path: str, render: bool = True):
        """Evaluate policy on a single environment"""
        env = PPOEnv(self.model, env_path, self.policy, debug=False)
        env.reset()
        env.rollout_buffer = []
        
        done = False
        rewards = []
        
        while not done:
            cost, _, _, done = env.step(evaluation=True)
            rewards.append(cost)
        
        if render:
            print(f"Evaluation Total Reward: {np.mean(rewards):.2f}")
            steps = list(range(CONTROL_START_IDX, len(env.current_lataccel_history)))
            
            plt.figure(figsize=(20, 5))
            plt.plot(steps, env.current_lataccel_history[CONTROL_START_IDX:], label="Current LatAccel")
            plt.plot(steps, env.target_lataccel_history[CONTROL_START_IDX:], label="Target LatAccel", linestyle="--")
            plt.xlabel("Step")
            plt.ylabel("Lateral Acceleration")
            plt.title("Policy Behavior Evaluation")
            plt.legend()
            plt.grid()
            plt.show()
        
        return (
            env.current_lataccel_history[CONTROL_START_IDX:],
            env.target_lataccel_history[CONTROL_START_IDX:],
            np.mean(rewards)
        )

    def plot_training_dynamics(self):
        """Plot training metrics"""
        keys = list(self.policy_logs.keys())
        plt.figure(figsize=(20, 12))
        
        for i, key in enumerate(keys):
            plt.subplot(3, 3, i + 1)
            plt.plot(self.policy_logs[key])
            plt.title(key.capitalize())
            plt.grid()
        
        plt.tight_layout()
        plt.show()

    def train(self, model_path, num_rollouts=1000):
        """
        Main training loop with vectorized batch processing
        
        Args:
            num_rollouts: number of rollout iterations
        """
        pbar = tqdm(range(num_rollouts), desc='Training PPO')
        all_accel = []
        target = []
        
        for rollout_idx in pbar:
            start_time = time()
            
            # Sample batch of environments
            env_paths = self.sample_env_batch(batch_size=self.batch_size)
            
            # Run all rollouts and collect data
            rollout_data, rollout_rewards = self.run_rollouts_batch(env_paths)
            
            if len(rollout_rewards) == 0:
                print("No successful rollouts in this iteration, skipping update...")
                continue
            
            # Calculate average reward
            avg_reward = np.mean(rollout_rewards)
            self.policy_logs["reward"].append(avg_reward)
            
            # Update policy with all collected data
            try:
                if self.update_policy_efficient(rollout_data):
                    print("Training stopped due to low std deviation.")
                    break
            except Exception as e:
                print(f"Error in policy update: {e}")
                import traceback
                traceback.print_exc()
                continue
            
            # Save checkpoint
            if rollout_idx % 10 == 0:
                torch.save(self.policy.state_dict(), f"{model_path}/model_weights_rollout_{rollout_idx}.pth")
                try:
                    accel, targ, _ = self.evaluate_policy(self.env_list[0], render=False)
                    all_accel.append(accel)
                    target = targ
                except Exception as e:
                    print(f"Error in evaluation: {e}")
            
            time_taken = time() - start_time
            pbar.set_postfix({
                'time': f'{time_taken:.2f}s',
                'envs': len(rollout_rewards),
                'avg_reward': f'{avg_reward:.4f}',
                'value_loss': f'{self.policy_logs["value_loss"][-1]:.4f}',
                'std': f'{self.policy_logs["std"][-1]:.4f}'
            })
        
        # Plot evaluation trajectories
        if len(all_accel) > 0:
            plt.figure(figsize=(20, 5))
            base_alpha = 0.1
            max_alpha = 1.0
            
            for i in range(len(all_accel)):
                N = len(all_accel) - 1
                alpha = base_alpha * ((max_alpha / base_alpha) ** (i / N)) if N > 0 else max_alpha
                plt.plot(all_accel[i], color='tab:blue', alpha=alpha)
            
            if len(target) > 0:
                plt.plot(target, label="Target LatAccel", color='tab:orange', linestyle="--")
            plt.xlabel("Step")
            plt.ylabel("Lateral Acceleration")
            plt.title("Policy Behavior During Training")
            plt.legend()
            plt.grid()
            plt.show()


def main():
    # Enable cuDNN benchmark for faster training
    torch.backends.cudnn.benchmark = True
    
    # Initialize model and policy
    model = TinyPhysicsModel("./models/tinyphysics.onnx", debug=False)
    data_dir = Path("./data/test")

    obs_dim = int(CONTEXT_LENGTH / 2) * 10
    action_dim = 1
    discrete_action = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy = TransformerPPOPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        discrete_action=discrete_action,
        seq_len=1,          # increase if you build an observation history
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1,
    ).to(device)
    
    # policy = FFPPOPolicy(input_dim=(10 * 10))

    
    
    # Create trainer
    trainer = PPOTrainer(
        model=model,
        policy=policy,
        data_path=data_dir,
        gamma=0.99,
        lam=0.95,
        clip_eps=0.2,
        epochs=4,              # Reduced from 10
        batch_size=32,          # Number of environments per rollout
        minibatch_size=512,     # Mini-batch size for SGD
        lr=3e-4,
        debug=False
    )

    model_path = r'./models/ppo_transformer_20-rollouts'
    os.makedirs(model_path, exist_ok=True)
    
    # Train
    trainer.train(model_path, num_rollouts=20)
    
    # Final evaluation
    print("\nEvaluating on 100 test environments...")
    total_rewards = []
    for env_path in tqdm(trainer.env_list[:100]):
        _, _, reward = trainer.evaluate_policy(env_path, render=False)
        total_rewards.append(reward)
    
    print(f"Average Reward over 100 environments: {np.mean(total_rewards):.2f}")
    
    # Visualize final policy
    trainer.evaluate_policy(r'./data/test/00000.csv', render=True)
    trainer.plot_training_dynamics()
    
    # Save final model
    torch.save(trainer.policy.state_dict(), f"{model_path}/final_model_weights.pth")


if __name__ == '__main__':
    main()