from torch.utils.tensorboard import SummaryWriter
import os


class Logger:
    """
    Logger for training metrics using Tensorboard
    """
    
    def __init__(self, log_dir='./runs', experiment_name='ppo_lateral_control'):
        self.log_dir = os.path.join(log_dir, experiment_name)
        self.writer = SummaryWriter(self.log_dir)
        self.step = 0
    
    def log_scalar(self, tag, value, step=None):
        """Log a scalar value"""
        if step is None:
            step = self.step
        self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, main_tag, tag_value_dict, step=None):
        """Log multiple scalar values"""
        if step is None:
            step = self.step
        self.writer.add_scalars(main_tag, tag_value_dict, step)
    
    def log_training_stats(self, stats, step=None):
        """Log training statistics"""
        if step is None:
            step = self.step
        
        for key, value in stats.items():
            self.log_scalar(f'train/{key}', value, step)
    
    def log_episode_stats(self, stats, step=None):
        """Log episode statistics"""
        if step is None:
            step = self.step
        
        for key, value in stats.items():
            self.log_scalar(f'episode/{key}', value, step)
    
    def increment_step(self):
        """Increment global step counter"""
        self.step += 1
    
    def close(self):
        """Close the writer"""
        self.writer.close()
