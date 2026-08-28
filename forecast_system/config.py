import os
import yaml
import subprocess
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
        'max_iters': 3,
        'lr_decay_enabled': True,
        'lr_decay_policy': 'plateau',
        'lr_decay_factor': 0.5,
        'lr_decay_patience': 2,
        'lr_min': 1e-6,
        'lr_decay_threshold': 1e-4,
        'lr_cooldown': 0
    },
    'persistence': {
        'enabled': True
    },
    'climatology': {
        'enabled': True
    },
    'ridge': {
        'enabled': True,
        'alpha': 1.0,
        'learning_rate': 0.01,
        'lr_decay_enabled': True,
        'lr_decay_policy': 'plateau',
        'lr_decay_factor': 0.5,
        'lr_decay_patience': 2,
        'lr_min': 1e-6,
        'lr_decay_threshold': 1e-4
    },
    'gbdt': {
        'enabled': True,
        'max_iter': 40,
        'learning_rate': 0.1,
        'random_state': 42,
        'lr_decay_enabled': True,
        'lr_decay_policy': 'plateau',
        'lr_decay_factor': 0.5,
        'lr_decay_patience': 2,
        'lr_min': 1e-5,
        'lr_decay_threshold': 1e-4
    },
    'lstm': {
        'enabled': True,
        'units': 64,
        'learning_rate': 0.001,
        'batch_size': 64,
        'patience': 10,
        'dropout': 0.2,
        'lr_decay_enabled': True,
        'lr_decay_policy': 'plateau',
        'lr_decay_factor': 0.5,
        'lr_decay_patience': 2,
        'lr_min': 1e-6,
        'lr_decay_threshold': 1e-4
    },
    'cnn': {
        'enabled': True,
        'filters': 64,
        'kernel_size': 3,
        'learning_rate': 0.001,
        'batch_size': 64,
        'patience': 10,
        'dropout': 0.2,
        'lr_decay_enabled': True,
        'lr_decay_policy': 'plateau',
        'lr_decay_factor': 0.5,
        'lr_decay_patience': 2,
        'lr_min': 1e-6,
        'lr_decay_threshold': 1e-4
    },
    'dense': {
        'enabled': True,
        'hidden_units': [128, 64],
        'learning_rate': 0.001,
        'batch_size': 64,
        'patience': 10,
        'dropout': 0.2,
        'lr_decay_enabled': True,
        'lr_decay_policy': 'plateau',
        'lr_decay_factor': 0.5,
        'lr_decay_patience': 2,
        'lr_min': 1e-6,
        'lr_decay_threshold': 1e-4
    },
    'linear': {
        'enabled': True,
        'learning_rate': 0.001,
        'batch_size': 64,
        'patience': 10,
        'lr_decay_enabled': True,
        'lr_decay_policy': 'plateau',
        'lr_decay_factor': 0.5,
        'lr_decay_patience': 2,
        'lr_min': 1e-6,
        'lr_decay_threshold': 1e-4
    },
    'deep_learning': {
        'enabled': True,
        'units': 64,
        'learning_rate': 0.001,
        'batch_size': 64,
        'patience': 10,
        'dropout': 0.2,
        'lr_decay_enabled': True,
        'lr_decay_policy': 'plateau',
        'lr_decay_factor': 0.5,
        'lr_decay_patience': 2,
        'lr_min': 1e-6,
        'lr_decay_threshold': 1e-4
    },
    'machine_learning': {
        'ridge_alpha': 1.0,
        'gbdt_max_iter': 40,
        'gbdt_learning_rate': 0.1,
        'gbdt_random_state': 42,
        'lr_decay_enabled': True,
        'lr_decay_policy': 'plateau',
        'lr_decay_factor': 0.5,
        'lr_decay_patience': 2,
        'lr_min': 1e-6,
        'lr_decay_threshold': 1e-4
    },
    'data_quality': {
        'temp_min': -30.0,
        'temp_max': 50.0,
        'dew_min': -40.0,
        'dew_max': 40.0,
        'sample_test_size': 200
    }
}


def get_git_repo_root(start_path=None):
    """
    Dynamically finds the root directory of the Git repository starting from start_path.
    Does not use hardcoded paths.
    """
    if start_path is None:
        p = Path.cwd().resolve()
    else:
        p = Path(start_path).resolve()
    curr = p if p.is_dir() else p.parent

    temp = curr
    while True:
        if (temp / ".git").exists():
            return temp
        if temp.parent == temp:
            break
        temp = temp.parent

    try:
        res = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=str(curr),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False
        )
        if res.returncode == 0 and res.stdout.strip():
            return Path(res.stdout.strip()).resolve()
    except Exception:
        pass

    return None


def is_path_in_git_repo(target_path):
    """
    Returns (True, git_root) if target_path is located inside a Git repository.
    """
    target = Path(target_path).resolve()
    git_root = get_git_repo_root(target)
    if git_root is None:
        git_root = get_git_repo_root(Path.cwd())
    if git_root is None:
        return False, None
    try:
        if target == git_root or git_root in target.parents:
            return True, git_root
    except Exception:
        pass
    return False, git_root


def load_config(config_path=None):
    """
    Loads YAML configuration file and merges it with default settings.
    Refuses to run if config_path is inside the Git repository.

    Parameters:
    -----------
    config_path : str or Path, optional
        Path to YAML file. If None, checks for default 'config.yaml'.

    Returns:
    --------
    dict
        Merged configuration dictionary with metadata.
    """
    config = {k: (dict(v) if isinstance(v, dict) else v) for k, v in DEFAULT_CONFIG.items()}

    if config_path is None:
        config_path = Path("config.yaml")

    resolved_path = Path(config_path).resolve()

    if not resolved_path.exists():
        raise FileNotFoundError(f"Configuration file not found at '{resolved_path}'")

    in_repo, git_root = is_path_in_git_repo(resolved_path)
    if in_repo:
        raise ValueError(
            f"Refusing to run: Configuration file '{resolved_path}' is inside the Git repository at '{git_root}'. "
            f"Please place configuration files outside the Git repository."
        )

    try:
        with open(resolved_path, 'r', encoding='utf-8') as f:
            user_cfg = yaml.safe_load(f)
        if user_cfg and isinstance(user_cfg, dict):
            for section, vals in user_cfg.items():
                if section in config and isinstance(vals, dict):
                    config[section].update(vals)
                else:
                    config[section] = vals
            print(f"✓ Successfully loaded configuration from {resolved_path}", flush=True)
    except Exception as e:
        if isinstance(e, ValueError):
            raise
        print(f"Warning: Could not parse config file {resolved_path} ({e}). Using defaults.", flush=True)

    config['_config_path'] = str(resolved_path)
    config['_config_dir'] = str(resolved_path.parent)
    return config


