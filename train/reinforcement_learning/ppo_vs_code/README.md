# PPO Lateral Control Training

Simplified PPO (Proximal Policy Optimization) training for lateral control on a single GPU server.

## Overview

This is a simplified version adapted from the AWS Fargate-based setup, designed to run on a single server with GPU access in JupyterLab.

**Key Changes:**
- Removed AWS Fargate dependencies
- Removed GRPC communication (single-process training)
- Added JupyterLab-friendly notebook interface
- Optimized for single GPU training

## Directory Structure

```
ppo_training/
├── README.md                 # This file
├── config.yaml              # Training configuration
├── environment/             # Custom environment wrapper
│   ├── __init__.py
│   └── lateral_env.py       # Lateral control environment
├── policy/                  # PPO policy and networks
│   ├── __init__.py
│   ├── actor_critic.py      # Actor-Critic networks
│   └── ppo_agent.py         # PPO algorithm implementation
├── utils/                   # Utility functions
│   ├── __init__.py
│   ├── replay_buffer.py     # Experience buffer
│   └── logger.py            # Training logger
├── train.py                 # Main training script
├── train.ipynb              # Jupyter notebook for training
├── eval.py                  # Evaluation script
└── requirements.txt         # Python dependencies
```

## Features

- **Single GPU Training**: Optimized for A100 GPU
- **Vectorized Environments**: Parallel environment execution using multiprocessing
- **PPO Algorithm**: Clipped surrogate objective with GAE
- **Tensorboard Logging**: Track training progress
- **Checkpoint System**: Save/load model weights
- **JupyterLab Compatible**: Run training in notebooks

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training in Jupyter Notebook

Open `train.ipynb` and run the cells.

### Training from Command Line

```bash
python train.py --config config.yaml
```

### Evaluation

```bash
python eval.py --model_path models/best_model.pt --data_path ../data/test
```

## Configuration

Edit `config.yaml` to adjust hyperparameters:
- Learning rate
- Number of parallel environments
- PPO clip ratio
- GAE lambda
- Training epochs
- etc.

## Notes

- Requires ~8GB GPU memory for training
- Training typically takes 2-4 hours for convergence
- Checkpoints saved every N episodes
- Tensorboard logs saved to `runs/`
