import yaml
import random
import pickle
import json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm
import optuna

import torch
import torch.nn as nn

from model import build_model
from new_flexible_dataset import CommaDataModule

# TODO написать выбор loss функции

def set_seed(seed):
    """Set seeds for reproducibility across all libraries"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # Make CuDNN deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def configure_optimizer(config, model, trial=None):
    """Configure optimizer either from config or via Optuna trial suggestions"""
    opt_args = {}
    
    if trial is not None:
        # Use Optuna trial suggestions
        lr = trial.suggest_float("lr", 1e-6, 1e-1, log=True)
        weight_decay = trial.suggest_float("weight_decay", 0.0, 1e-2)
        optimizer_type = trial.suggest_categorical("optimizer", ["adam", "sgd", "rmsprop", "adamw"])
    else:
        # Use config values
        lr = config["train"]["learning_rate"]
        weight_decay = config["train"]["weight_decay"]
        optimizer_type = config["meta"]["optimizer"]

    opt_args["lr"] = lr
    opt_args["weight_decay"] = weight_decay

    # ---------------------------
    # Optimizer-specific args
    # ---------------------------
    if optimizer_type == "adam":
        if trial is not None:
            amsgrad = trial.suggest_categorical("adam_amsgrad", [True, False])
        else:
            amsgrad = config["optimizer"][optimizer_type]["amsgrad"]
        opt_args["amsgrad"] = amsgrad
        optimizer = torch.optim.Adam(model.parameters(), **opt_args)

    elif optimizer_type == "sgd":
        if trial is not None:
            momentum = trial.suggest_float("sgd_momentum", 0.0, 1.0)
            nesterov = trial.suggest_categorical("sgd_nesterov", [True, False])
            if not nesterov:
                dampening = trial.suggest_float("sgd_dampening", 0.0, 1.0)
            else:
                dampening = 0
        else:
            momentum = config["optimizer"][optimizer_type]["momentum"]
            nesterov = config["optimizer"][optimizer_type]["nesterov"]
            dampening = config["optimizer"][optimizer_type].get("dampening", 0)
        
        opt_args["momentum"] = momentum
        opt_args["nesterov"] = nesterov
        if not nesterov:
            opt_args["dampening"] = dampening
        optimizer = torch.optim.SGD(model.parameters(), **opt_args)

    elif optimizer_type == "rmsprop":
        if trial is not None:
            alpha = trial.suggest_float("rmsprop_alpha", 0.0, 1.0)
            momentum = trial.suggest_float("rmsprop_momentum", 0.0, 1.0)
            centered = trial.suggest_categorical("rmsprop_centered", [True, False])
        else:
            alpha = config["optimizer"][optimizer_type]["alpha"]
            momentum = config["optimizer"][optimizer_type]["momentum"]
            centered = config["optimizer"][optimizer_type]["centered"]
        
        opt_args["alpha"] = alpha
        opt_args["momentum"] = momentum
        opt_args["centered"] = centered
        optimizer = torch.optim.RMSprop(model.parameters(), **opt_args)

    elif optimizer_type == "adamw":
        if trial is not None:
            amsgrad = trial.suggest_categorical("adamw_amsgrad", [True, False])
        else:
            amsgrad = config["optimizer"][optimizer_type]["amsgrad"]
        opt_args["amsgrad"] = amsgrad
        optimizer = torch.optim.AdamW(model.parameters(), **opt_args)

    else:
        raise ValueError(f"Invalid optimizer: {optimizer_type}")

    return optimizer, opt_args

def train_model(config, train_loader, val_loader, model, device, save_path, trial=None):
    # -------------------------------
    #  Optimizer & Loss
    # -------------------------------
    optimizer, optimizer_config = configure_optimizer(config, model, trial)
    criterion = torch.nn.MSELoss()
    
    # -------------------------------
    #  Training Loop
    # -------------------------------
    num_epochs = config["train"]["num_epochs"]
    best_val_loss = float("inf")

    history = defaultdict(list)
    history = {
        'epoch': [],
        'train_loss': [], 'train_mae': [], 'train_rmse': [], 'train_r2': [],
        'val_loss': [], 'val_mae': [], 'val_rmse': [], 'val_r2': []
    }

    for epoch in range(num_epochs):
        # -------------------------------
        # Train Phase
        # -------------------------------
        model.train()
        train_loss = 0.0
        train_preds_all, train_tgts_all = [], []

        for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Train"):
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            train_preds_all.append(pred.detach().cpu().numpy())
            train_tgts_all.append(yb.detach().cpu().numpy())

        avg_train_loss = train_loss / len(train_loader)
        train_preds = np.concatenate(train_preds_all, axis=0).flatten()
        train_targets = np.concatenate(train_tgts_all, axis=0).flatten()
        train_mae = mean_absolute_error(train_targets, train_preds)
        train_rmse = np.sqrt(mean_squared_error(train_targets, train_preds))
        train_r2 = r2_score(train_targets, train_preds)

        # -------------------------------
        # Validation Phase
        # -------------------------------
        model.eval()
        val_loss = 0.0
        val_preds_all, val_tgts_all = [], []

        with torch.no_grad():
            for xb, yb in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Val"):
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_loss += loss.item()
                val_preds_all.append(pred.detach().cpu().numpy())
                val_tgts_all.append(yb.detach().cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_preds = np.concatenate(val_preds_all, axis=0).flatten()
        val_targets = np.concatenate(val_tgts_all, axis=0).flatten()
        val_mae = mean_absolute_error(val_targets, val_preds)
        val_rmse = np.sqrt(mean_squared_error(val_targets, val_preds))
        val_r2 = r2_score(val_targets, val_preds)

        # -------------------------------
        # Logging & Saving
        # -------------------------------
        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {avg_train_loss:.4f}, MAE: {train_mae:.4f}, RMSE: {train_rmse:.4f}, R2: {train_r2:.4f} | "
            f"Val Loss: {avg_val_loss:.4f}, MAE: {val_mae:.4f}, RMSE: {val_rmse:.4f}, R2: {val_r2:.4f}"
        )

        # Save history
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(avg_train_loss)
        history['train_mae'].append(train_mae)
        history['train_rmse'].append(train_rmse)
        history['train_r2'].append(train_r2)
        history['val_loss'].append(avg_val_loss)
        history['val_mae'].append(val_mae)
        history['val_rmse'].append(val_rmse)
        history['val_r2'].append(val_r2)

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path / "best_model.pt")
            print(f"✅ Saved new best model with val_loss={best_val_loss:.5f}")

    # -------------------------------
    # Final Save
    # -------------------------------
    torch.save(model.state_dict(), save_path / "final_model.pt")

    # Save history
    history_df = pd.DataFrame(history)
    history_df.to_csv(save_path / "history.csv", index=False)
    with open(save_path / "history.pkl", "wb") as f:
        pickle.dump(dict(history), f)

    print(f"Training complete. Best val loss: {best_val_loss:.5f}")
    
    # Save optimizer config to training config
    config["used_optimizer"] = optimizer_config
    config["used_optimizer"]["type"] = optimizer.__class__.__name__
    
    return best_val_loss

def objective(trial, config):
    """Optuna objective function for hyperparameter optimization"""
    
    # Set reproducible seed for this trial
    trial_seed = config["seed"] + trial.number
    set_seed(trial_seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Suggest model hyperparameters via Optuna
    if config["meta"]["model_type"] == "lstm":
        hidden_size = trial.suggest_categorical("hidden_size", [128, 256, 512])
        num_layers = trial.suggest_int("num_layers", 1, 3)
        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        
        # Update config with trial suggestions
        config["model"]["lstm"]["hidden_size"] = hidden_size
        config["model"]["lstm"]["num_layers"] = num_layers
        config["model"]["lstm"]["dropout"] = dropout

    # -------------------------------
    #  Model
    # -------------------------------        
    model = build_model(config)
    model.to(device)
    
    # -------------------------------
    #  Model Save Path
    # -------------------------------
    platform = config["data"]["platforms"][0] if isinstance(config["data"]["platforms"], list) else config["data"]["platforms"]
    date_str = datetime.today().strftime("%b_%d")
    version = f"{platform}_{date_str}_trial_{trial.number}"
    save_path = Path(f"../models/{version}")
    save_path.mkdir(parents=True, exist_ok=True)
    
    # -------------------------------
    #  Dataset
    # -------------------------------
    dm = CommaDataModule(
        config=config,
        model_save_path=save_path,
    )
    dm.setup()
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    # -------------------------------
    #  Training 
    # -------------------------------
    best_val_loss = train_model(config, train_loader, val_loader, model, device, save_path, trial)
    
    # Save complete config with trial parameters
    trial_config = config.copy()
    trial_config["trial_number"] = trial.number
    trial_config["trial_seed"] = trial_seed
    trial_config["best_val_loss"] = best_val_loss
    
    with open(save_path / "config.json", "w") as f:
        json.dump(trial_config, f, indent=4)
    
    return best_val_loss

def train_pipeline():
    """Main training pipeline using Optuna optimization"""
    
    # Load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Set global seed
    set_seed(config["seed"])
    
    # Single trial mode (no optimization)
    if config["meta"].get("number_trials", 1) == 1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # -------------------------------
        #  Model
        # -------------------------------        
        model = build_model(config)
        model.to(device)
        
        # -------------------------------
        #  Model Save Path
        # -------------------------------
        platform = config["data"]["platforms"][0] if isinstance(config["data"]["platforms"], list) else config["data"]["platforms"]
        date_str = datetime.today().strftime("%b_%d")
        version = f"{platform}_{date_str}_single"
        save_path = Path(f"../models/{version}")
        save_path.mkdir(parents=True, exist_ok=True)
        
        # -------------------------------
        #  Dataset
        # -------------------------------
        dm = CommaDataModule(
            config=config,
            model_save_path=save_path,
        )
        dm.setup()
        train_loader = dm.train_dataloader()
        val_loader = dm.val_dataloader()

        # -------------------------------
        #  Training 
        # -------------------------------
        best_val_loss = train_model(config, train_loader, val_loader, model, device, save_path)
        
        # Save config
        config["best_val_loss"] = best_val_loss
        with open(save_path / "config.json", "w") as f:
            json.dump(config, f, indent=4)
            
        print(f"Single training complete. Best val loss: {best_val_loss:.5f}")
        
    else:
        # Multiple trials with Optuna optimization
        n_trials = config["meta"]["number_trials"]
        
        study = optuna.create_study(
            direction="minimize",
            study_name=f"lstm_optimization_{datetime.today().strftime('%Y%m%d_%H%M%S')}",
        )
        
        print(f"Starting Optuna optimization with {n_trials} trials")
        study.optimize(lambda trial: objective(trial, config.copy()), n_trials=n_trials)
        
        print("Optuna optimization completed!")
        print(f"Best trial: {study.best_trial.number}")
        print(f"Best value: {study.best_value:.5f}")
        print(f"Best params: {study.best_trial.params}")
        
        # Save study results
        platform = config["data"]["platforms"][0] if isinstance(config["data"]["platforms"], list) else config["data"]["platforms"]
        date_str = datetime.today().strftime("%b_%d")
        study_path = Path(f"../models/{platform}_{date_str}_optuna_study")
        study_path.mkdir(parents=True, exist_ok=True)
        
        study_results = {
            "best_trial": study.best_trial.number,
            "best_value": study.best_value,
            "best_params": study.best_trial.params,
            "all_trials": [
                {
                    "number": trial.number,
                    "value": trial.value,
                    "params": trial.params
                }
                for trial in study.trials
            ]
        }
        
        with open(study_path / "optuna_results.json", "w") as f:
            json.dump(study_results, f, indent=4)


if __name__ == '__main__':
    train_pipeline()