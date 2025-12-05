#!/usr/bin/env python3
"""
Test script to verify the training pipeline works correctly.
Run this before the full training to catch any issues early.
"""

import yaml
import sys
from pathlib import Path

def test_imports():
    """Test that all required modules can be imported"""
    try:
        import torch
        import optuna
        import numpy as np
        import pandas as pd
        from sklearn.preprocessing import StandardScaler
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_config():
    """Test that config file is valid"""
    try:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        required_keys = ["seed", "data", "model", "train", "meta"]
        for key in required_keys:
            if key not in config:
                print(f"❌ Missing config key: {key}")
                return False
        
        print("✅ Config file is valid")
        return True
    except Exception as e:
        print(f"❌ Config error: {e}")
        return False

def test_model_building():
    """Test that model can be built from config"""
    try:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        from model import build_model
        model, model_info = build_model(config)
        print(f"✅ Model built successfully: {model_info}")
        return True
    except Exception as e:
        print(f"❌ Model building error: {e}")
        return False

def test_data_path():
    """Test that data path exists"""
    try:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        data_path = Path(config["data"]["path"])
        if not data_path.exists():
            print(f"❌ Data path does not exist: {data_path}")
            return False
        
        # Check for CSV files
        csv_files = list(data_path.rglob("*.csv"))
        if len(csv_files) == 0:
            print(f"❌ No CSV files found in: {data_path}")
            return False
        
        print(f"✅ Data path exists with {len(csv_files)} CSV files")
        return True
    except Exception as e:
        print(f"❌ Data path error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing training pipeline components...")
    
    tests = [
        test_imports,
        test_config,
        test_model_building,
        test_data_path,
    ]
    
    results = []
    for test in tests:
        print(f"\n🔍 Running {test.__name__}...")
        results.append(test())
    
    print(f"\n📊 Test Results: {sum(results)}/{len(results)} passed")
    
    if all(results):
        print("🎉 All tests passed! Ready for training.")
        return 0
    else:
        print("❌ Some tests failed. Please fix issues before training.")
        return 1

if __name__ == "__main__":
    sys.exit(main())