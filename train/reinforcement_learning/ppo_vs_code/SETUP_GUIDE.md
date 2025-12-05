# PPO Training Setup - Complete Structure

## ✅ Created Directory Structure

```
ppo_training/
├── README.md                    ✓ Created
├── config.yaml                  ✓ Created  
├── requirements.txt             ✓ Created
├── train.py                     ✓ Created (CLI training script)
├── train.ipynb                  ✓ Created (Jupyter notebook)
├── eval.py                      ⚠️ To be created
│
├── environment/                 ✓ Created
│   ├── __init__.py             ✓
│   └── lateral_env.py          ✓ (LateralControlEnv + VectorizedEnv)
│
├── policy/                      ✓ Created
│   ├── __init__.py             ✓
│   ├── actor_critic.py         ✓ (Actor-Critic network)
│   └── ppo_agent.py            ✓ (PPO algorithm)
│
├── utils/                       ✓ Created
│   ├── __init__.py             ✓
│   ├── replay_buffer.py        ✓ (RolloutBuffer for PPO)
│   └── logger.py               ✓ (Tensorboard logger)
│
├── models/                      📁 (Created during training)
│   ├── checkpoint_*.pt         (Training checkpoints)
│   ├── final_model.pt          (Final trained model)
│   └── training_stats.npz      (Training statistics)
│
└── runs/                        📁 (Created during training)
    └── ppo_lateral_control/    (Tensorboard logs)
```

## 🚀 Quick Start Guide

### 1. Installation

```bash
cd ppo_training
pip install -r requirements.txt
```

### 2. Training Options

#### Option A: Jupyter Notebook (Recommended for JupyterLab)
```bash
jupyter lab train.ipynb
```
Then run all cells sequentially.

#### Option B: Command Line
```bash
python train.py --config config.yaml
```

### 3. Monitor Training
```bash
tensorboard --logdir=./runs
```

### 4. Evaluation
```bash
python eval.py --model_path models/final_model.pt
```

## 📊 Key Features

### vs Original AWS Fargate Setup

| Feature | Original | Simplified Version |
|---------|----------|-------------------|
| **Infrastructure** | AWS Fargate (cloud) | Single GPU server (local) |
| **Communication** | GRPC protocol | Direct Python calls |
| **Parallelization** | Multiple containers | Vectorized environments |
| **Scaling** | CI/CD pipeline | Fixed parallel envs |
| **Permissions** | Root required | No root needed |
| **Interface** | CLI only | CLI + Jupyter notebook |

### Advantages of This Setup

✅ **Single GPU optimized** - Designed for A100  
✅ **JupyterLab friendly** - Interactive training  
✅ **No cloud costs** - Runs locally  
✅ **No root access needed** - User-level installation  
✅ **Simplified architecture** - Easier to debug  
✅ **Faster iteration** - No container build times  

## ⚙️ Configuration

Edit `config.yaml` to customize:

- **Learning rate**: `3e-4` (default)
- **Num parallel envs**: `8` (adjust based on GPU memory)
- **Batch size**: `256`
- **Total timesteps**: `1,000,000`
- **PPO epochs**: `10`
- **Clip ratio**: `0.2`

## 📈 Expected Results

- Training time: **2-4 hours** on A100
- GPU memory usage: **~6-8 GB**
- Convergence: **~500k-800k timesteps**

## 🔧 Troubleshooting

### Out of Memory (OOM)
- Reduce `num_envs` in config.yaml
- Reduce `batch_size`
- Reduce `hidden_sizes`

### Slow Training
- Increase `num_envs` if memory allows
- Increase `batch_size`
- Use mixed precision training (fp16)

### Poor Performance
- Adjust reward function in `environment/lateral_env.py`
- Tune PPO hyperparameters
- Increase network capacity

## 📝 Next Steps

1. **Run training** - Start with default config
2. **Monitor progress** - Use Tensorboard
3. **Tune hyperparameters** - Based on results
4. **Evaluate** - Test on held-out data
5. **Deploy** - Create controller for inference

## 🎯 Differences from AWS Setup

### Removed Components:
- ❌ Dockerfile
- ❌ GRPC protocol (`rollout.proto`)
- ❌ AWS deployment scripts
- ❌ CI/CD pipeline
- ❌ Client-server architecture

### Added Components:
- ✅ Jupyter notebook interface
- ✅ Simplified training loop
- ✅ Local tensorboard logging
- ✅ Direct environment interaction

## 📚 File Descriptions

### Core Training Files

- **`train.py`** - Main CLI training script with full PPO loop
- **`train.ipynb`** - Interactive Jupyter notebook for training
- **`config.yaml`** - All hyperparameters and settings
- **`eval.py`** - Evaluation script (to be created)

### Environment

- **`lateral_env.py`** - Gym-compatible lateral control environment
  - `LateralControlEnv` - Single environment
  - `VectorizedLateralControlEnv` - Parallel environments

### Policy

- **`actor_critic.py`** - Neural network architecture
  - Shared feature extractor
  - Separate actor (policy) and critic (value) heads
  
- **`ppo_agent.py`** - PPO algorithm implementation
  - GAE computation
  - Policy update with clipping
  - Value function training

### Utils

- **`replay_buffer.py`** - Experience storage for PPO
- **`logger.py`** - Tensorboard logging utilities

## 💡 Tips for JupyterLab

1. **Run cells sequentially** - Don't skip cells
2. **Monitor GPU** - Use `nvidia-smi` in terminal
3. **Save frequently** - Checkpoints every N steps
4. **Visualize early** - Plot metrics during training
5. **Adjust on-the-fly** - Modify hyperparameters between runs

## 🔗 Related Files

Make sure you have these from your main project:
- `../tinyphysics.py` - Physics simulator
- `../models/tinyphysics.onnx` - Dynamics model
- `../data/train/` - Training data

## 📧 Support

Issues with:
- **CUDA errors** - Check CUDA version compatibility
- **Import errors** - Verify all files are created
- **Path errors** - Use absolute paths if needed
