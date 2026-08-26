import os
import yaml
from pathlib import Path

DEFAULT_CONFIG = {
    'pipeline': {
        'mode': 'resume',
        'output_dir': 'report',
        'data_dir': 'data'
    },
    'optimization': {
        'target_mae': 1.0,
        'epochs_per_iter': 5,
        'max_iters': 3
    },
    'deep_learning': {
        'units': 64,
        'learning_rate': 0.001,
        'batch_size': 64,
        'patience': 10,
        'dropout': 0.2
    },
    'machine_learning': {
        'ridge_alpha': 1.0,
        'gbdt_max_iter': 40,
        'gbdt_learning_rate': 0.1,
        'gbdt_random_state': 42
    },
    'data_quality': {
        'temp_min': -30.0,
        'temp_max': 50.0,
        'dew_min': -40.0,
        'dew_max': 40.0,
        'sample_test_size': 200
    }
}

def load_config(config_path=None):
    """
    Loads YAML configuration file and merges it with default settings.
    
    Parameters:
    -----------
    config_path : str or Path, optional
        Path to YAML file. If None, checks for default 'config.yaml' in current directory.
        
    Returns:
    --------
    dict
        Merged configuration dictionary.
    """
    config = dict(DEFAULT_CONFIG)
    
    if config_path is None:
        default_path = Path("config.yaml")
        if default_path.exists():
            config_path = default_path
            
    if config_path and Path(config_path).exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                user_cfg = yaml.safe_load(f)
            if user_cfg and isinstance(user_cfg, dict):
                for section, vals in user_cfg.items():
                    if section in config and isinstance(vals, dict):
                        config[section].update(vals)
                    else:
                        config[section] = vals
                print(f"✓ Successfully loaded configuration from {config_path}", flush=True)
        except Exception as e:
            print(f"Warning: Could not parse config file {config_path} ({e}). Using defaults.", flush=True)
            
    return config
