"""Configuration management utilities."""

import os
import yaml
from datetime import datetime


def load_config(config_path):
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str): Path to YAML configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config, save_path):
    """
    Save configuration to YAML file.
    
    Args:
        config (dict): Configuration dictionary
        save_path (str): Path to save the configuration
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def merge_configs(base_config, cli_config):
    """
    Merge CLI arguments with base configuration.
    CLI arguments take precedence over config file values.
    
    Args:
        base_config (dict): Base configuration from file
        cli_config (dict): Configuration from CLI arguments
        
    Returns:
        dict: Merged configuration
    """
    merged = base_config.copy()
    
    # Update nested dictionaries recursively
    for key, value in cli_config.items():
        if value is not None:
            if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
                merged[key].update(value)
            else:
                merged[key] = value
    
    return merged


def create_exp_dir(base_dir='runs', exp_name=None):
    """
    Create experiment directory with timestamp.
    Format: runs/exp_YYYYMMDD_HHMMSS or runs/exp_name_YYYYMMDD_HHMMSS
    
    Args:
        base_dir (str): Base directory for experiments
        exp_name (str): Optional experiment name prefix
        
    Returns:
        str: Path to created experiment directory
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if exp_name:
        dir_name = f"{exp_name}_{timestamp}"
    else:
        dir_name = f"exp_{timestamp}"
    
    exp_dir = os.path.join(base_dir, dir_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # Create subdirectories
    os.makedirs(os.path.join(exp_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'plots'), exist_ok=True)
    
    return exp_dir


def validate_config(config):
    """
    Validate configuration dictionary for required fields and sensible values.
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        dict: Validated configuration
        
    Raises:
        ValueError: If configuration is invalid
    """
    required_sections = ['model', 'data', 'noise', 'training', 'testing']
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")
    
    # Validate noise configuration
    if config['noise']['sigma_min'] < 0:
        raise ValueError("sigma_min must be non-negative")
    
    if config['noise']['sigma_max'] <= config['noise']['sigma_min']:
        raise ValueError("sigma_max must be greater than sigma_min")
    
    # Validate training configuration
    if config['training']['epochs'] <= 0:
        raise ValueError("epochs must be positive")
    
    if config['training']['learning_rate'] <= 0:
        raise ValueError("learning_rate must be positive")
    
    # Validate data configuration
    if config['data']['batch_size_train'] <= 0:
        raise ValueError("batch_size_train must be positive")
    
    if config['data']['batch_size_test'] <= 0:
        raise ValueError("batch_size_test must be positive")
    
    return config

