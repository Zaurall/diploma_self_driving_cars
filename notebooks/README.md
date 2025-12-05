# Optuna-based LSTM Training Pipeline

This directory contains a flexible training pipeline for LSTM models using Optuna hyperparameter optimization.

## Files

- `train.py` - Main training script with Optuna integration
- `model.py` - Model definitions (LSTM, GRU, MLP)  
- `new_flexible_dataset.py` - PyTorch dataset and data module
- `config.yaml` - Configuration file
- `test_train.py` - Test script to verify setup

## Features

✅ **Reproducible Training**: Set seeds across all libraries (torch, numpy, random, os)  
✅ **Optuna Integration**: Hyperparameter optimization with configurable number of trials  
✅ **Flexible Model Naming**: `{platform}_{date}_{trial_info}` format  
✅ **Complete Config Saving**: All parameters saved to model folder  
✅ **Multi-optimizer Support**: Adam, SGD, RMSprop, AdamW with trial-based selection  
✅ **Future Plan Support**: Option A integration (future_k parameter)  

## Quick Start

1. **Install Dependencies**:
```bash
pip install torch optuna pandas numpy scikit-learn pyyaml tqdm
```

2. **Configure Training**:
Edit `config.yaml`:
- Set `data.path` to your CSV data directory
- Set `data.platforms` to your target platform(s)  
- Set `meta.number_trials` to 1 (single run) or >1 (Optuna optimization)
- Configure `data.future_k` for future plan horizon

3. **Test Setup**:
```bash
python test_train.py
```

4. **Run Training**:
```bash
python train.py
```

## Configuration

### Single Training Run
```yaml
meta:
  number_trials: 1  # Single run with config values
```

### Optuna Optimization
```yaml
meta:
  number_trials: 10  # Run 10 optimization trials
```

### Model Types
```yaml
meta:
  model_type: "lstm"  # or "gru", "mlp"
```

### Future Plan Integration  
```yaml
data:
  future_k: 20  # Number of future target acceleration points (0 to disable)
```

## Output Structure

Each training run creates a folder:
```
../models/{platform}_{Mon_DD}_{trial_info}/
├── config.json          # Complete configuration used
├── best_model.pt         # Best model weights  
├── final_model.pt        # Final model weights
├── history.csv           # Training metrics
├── history.pkl           # Training history
└── scalers.pkl          # Data scalers
```

Optuna studies create an additional summary:
```
../models/{platform}_{Mon_DD}_optuna_study/
└── optuna_results.json   # All trial results and best parameters
```

## Model Architecture

The LSTM model supports:
- **Input**: Physics features (roll, aEgo, vEgo, targetLateralAcceleration) + optional future plan
- **Output**: Next-step steering command prediction
- **Sequence Mode**: Predicts steering for each timestep in sequence
- **Future Plan**: Repeats future horizon across all timesteps (Option A)

## Reproducibility

All random seeds are set consistently:
- `torch.manual_seed()` 
- `torch.cuda.manual_seed_all()`
- `numpy.random.seed()`
- `random.seed()`  
- `os.environ['PYTHONHASHSEED']`
- `torch.backends.cudnn.deterministic = True`

Each Optuna trial gets a unique seed: `base_seed + trial_number`

## Hyperparameter Search Space

When using Optuna (`number_trials > 1`):

**Optimizer Selection**:
- Learning rate: log-uniform [1e-6, 1e-1]  
- Weight decay: uniform [0.0, 1e-2]
- Optimizer type: categorical [adam, sgd, rmsprop, adamw]

**Model Architecture** (LSTM):
- Hidden size: categorical [128, 256, 512]
- Number of layers: integer [1, 3]  
- Dropout: uniform [0.0, 0.5]

**Optimizer-specific**:
- Adam: amsgrad (True/False)
- SGD: momentum [0-1], nesterov (True/False), dampening [0-1]  
- RMSprop: alpha [0-1], momentum [0-1], centered (True/False)

## Troubleshooting

**No CSV files found**: Check `data.path` and `data.platforms` in config.yaml  
**Import errors**: Install required packages  
**CUDA out of memory**: Reduce `batch_size` in config.yaml  
**Low performance**: Increase `number_trials` for better hyperparameter search