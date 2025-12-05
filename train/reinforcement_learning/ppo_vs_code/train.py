import os
import yaml
import glob
import random
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

from environment import LateralControlEnv, VectorizedLateralControlEnv
from policy import PPOAgent
from utils import RolloutBuffer, Logger


def set_seed(seed):
    """Set seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def collect_rollouts(envs, agent, buffer, num_steps):
    """
    Collect rollouts from vectorized environments
    
    Args:
        envs: Vectorized environment
        agent: PPO agent
        buffer: Rollout buffer
        num_steps: Number of steps to collect
    
    Returns:
        Episode statistics
    """
    obs = envs.reset()
    
    episode_rewards = []
    episode_costs = []
    episode_lengths = []
    
    current_episode_rewards = np.zeros(envs.num_envs)
    current_episode_costs = np.zeros(envs.num_envs)
    current_episode_lengths = np.zeros(envs.num_envs)
    
    for step in range(num_steps):
        # Get action from policy
        actions, values = agent.get_action(obs)
        
        # Compute log probabilities (for PPO update)
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).to(agent.device)
            action_tensor = torch.FloatTensor(actions).to(agent.device)
            _, log_probs, _ = agent.ac.evaluate_actions(obs_tensor, action_tensor)
            log_probs = log_probs.cpu().numpy().flatten()
        
        # Step environments
        next_obs, rewards, dones, infos = envs.step(actions)
        
        # Store transition in buffer
        buffer.add(obs, actions, rewards, values.flatten(), log_probs, dones)
        
        # Update episode statistics
        current_episode_rewards += rewards
        current_episode_lengths += 1
        
        for i, info in enumerate(infos):
            current_episode_costs[i] += info.get('cost', 0.0)
        
        # Check for done episodes
        for i, done in enumerate(dones):
            if done:
                episode_rewards.append(current_episode_rewards[i])
                episode_costs.append(current_episode_costs[i])
                episode_lengths.append(current_episode_lengths[i])
                
                current_episode_rewards[i] = 0
                current_episode_costs[i] = 0
                current_episode_lengths[i] = 0
        
        obs = next_obs
    
    # Compute next values for GAE
    with torch.no_grad():
        _, next_values = agent.get_action(obs)
    
    # Compute advantages and returns using GAE
    advantages = np.zeros((num_steps, envs.num_envs), dtype=np.float32)
    returns = np.zeros((num_steps, envs.num_envs), dtype=np.float32)
    
    for env_idx in range(envs.num_envs):
        env_rewards = buffer.rewards[:, env_idx]
        env_values = buffer.values[:, env_idx]
        env_dones = buffer.dones[:, env_idx]
        env_next_value = next_values[env_idx]
        
        env_advantages, env_returns = agent.compute_gae(
            env_rewards, env_values, env_dones, env_next_value
        )
        
        advantages[:, env_idx] = env_advantages
        returns[:, env_idx] = env_returns
    
    # Compute episode statistics
    stats = {
        'mean_reward': np.mean(episode_rewards) if episode_rewards else 0.0,
        'mean_cost': np.mean(episode_costs) if episode_costs else 0.0,
        'mean_length': np.mean(episode_lengths) if episode_lengths else 0.0,
        'num_episodes': len(episode_rewards)
    }
    
    return advantages, returns, stats


def train_ppo(config_path='config.yaml'):
    """
    Main PPO training loop
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set seed
    set_seed(config['seed'])
    
    # Create directories
    os.makedirs(config['training']['model_save_path'], exist_ok=True)
    os.makedirs(config['training']['log_dir'], exist_ok=True)
    
    # Get training data files
    data_files = []
    platform_files = glob.glob(os.path.join(config['env']['data_path'], '*.csv'))
    data_files.extend(platform_files)
    # for platform in config['env']['platforms']:
    #     platform_files = glob.glob(os.path.join(config['env']['data_path'], platform, '*.csv'))
    #     data_files.extend(platform_files)
    
    print(f"Found {len(data_files)} training files")
    
    # Sample files for parallel environments
    num_envs = config['env']['num_envs']
    sampled_files = random.sample(data_files, min(num_envs, len(data_files)))
    
    # Create vectorized environment
    model_path = '../../models/tinyphysics.onnx'
    envs = VectorizedLateralControlEnv(
        sampled_files,
        model_path=model_path,
        max_steps=config['env']['max_steps']
    )
    
    # Create PPO agent
    obs_dim = envs.observation_space.shape[0]
    action_dim = envs.action_space.shape[0]
    
    agent = PPOAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=config['ppo']['hidden_sizes'],
        activation=config['ppo']['activation'],
        learning_rate=config['ppo']['learning_rate'],
        gamma=config['ppo']['gamma'],
        gae_lambda=config['ppo']['gae_lambda'],
        clip_ratio=config['ppo']['clip_ratio'],
        value_coef=config['ppo']['value_coef'],
        entropy_coef=config['ppo']['entropy_coef'],
        max_grad_norm=config['ppo']['max_grad_norm'],
        device=config['device']
    )
    
    # Create rollout buffer
    buffer = RolloutBuffer(
        num_envs=num_envs,
        num_steps=config['ppo']['num_steps'],
        obs_dim=obs_dim,
        action_dim=action_dim
    )
    
    # Create logger
    logger = Logger(
        log_dir=config['training']['log_dir'],
        experiment_name='ppo_lateral_control'
    )
    
    # Training loop
    total_timesteps = config['training']['total_timesteps']
    num_steps = config['ppo']['num_steps']
    num_updates = total_timesteps // (num_steps * num_envs)
    
    print(f"Starting PPO training for {num_updates} updates ({total_timesteps} total timesteps)")
    
    global_step = 0
    
    for update in tqdm(range(num_updates), desc="Training"):
        # Collect rollouts
        advantages, returns, episode_stats = collect_rollouts(
            envs, agent, buffer, num_steps
        )
        
        # Get rollout data
        rollout_data = buffer.get(next_values=None, advantages=advantages, returns=returns)
        
        # Update policy
        train_stats = agent.update(
            rollout_data,
            num_epochs=config['ppo']['num_epochs'],
            batch_size=config['ppo']['batch_size']
        )
        
        # Update global step
        global_step += num_steps * num_envs
        
        # Log statistics
        if update % (config['training']['log_freq'] // num_steps) == 0:
            logger.log_training_stats(train_stats, global_step)
            logger.log_episode_stats(episode_stats, global_step)
            
            print(f"\nUpdate {update} | Step {global_step}")
            print(f"  Mean Reward: {episode_stats['mean_reward']:.2f}")
            print(f"  Mean Cost: {episode_stats['mean_cost']:.4f}")
            print(f"  Mean Length: {episode_stats['mean_length']:.1f}")
            print(f"  Policy Loss: {train_stats['policy_loss']:.4f}")
            print(f"  Value Loss: {train_stats['value_loss']:.4f}")
        
        # Save checkpoint
        if update % (config['training']['save_freq'] // (num_steps * num_envs)) == 0:
            save_path = os.path.join(
                config['training']['model_save_path'],
                f"checkpoint_{global_step}.pt"
            )
            agent.save(save_path)
            print(f"\nSaved checkpoint to {save_path}")
        
        # Reset buffer
        buffer.reset()
    
    # Save final model
    final_path = os.path.join(config['training']['model_save_path'], "final_model.pt")
    agent.save(final_path)
    print(f"\nTraining complete! Final model saved to {final_path}")
    
    # Close logger and environments
    logger.close()
    envs.close()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train PPO for lateral control')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    args = parser.parse_args()
    
    train_ppo(args.config)
