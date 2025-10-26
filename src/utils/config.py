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


def update_config_with_cli(config, cli_args):
    """
    Update config dictionary with CLI arguments.
    Handles nested dictionary updates for specific parameters.
    
    Args:
        config (dict): Base configuration
        cli_args (dict): CLI arguments
        
    Returns:
        dict: Updated configuration
    """
    # Map CLI arguments to config keys
    if cli_args.get('batch_size') is not None:
        config['data']['batch_size_train'] = cli_args['batch_size']
    
    if cli_args.get('batch_size_test') is not None:
        config['data']['batch_size_test'] = cli_args['batch_size_test']
    
    if cli_args.get('epochs') is not None:
        config['training']['epochs'] = cli_args['epochs']
    
    if cli_args.get('lr') is not None:
        config['training']['learning_rate'] = cli_args['lr']
    
    if cli_args.get('device') is not None:
        config['device'] = cli_args['device']
    
    if cli_args.get('seed') is not None:
        config['seed'] = cli_args['seed']
    
    if cli_args.get('sigma_min') is not None:
        config['noise']['sigma_min'] = cli_args['sigma_min']
    
    if cli_args.get('sigma_max') is not None:
        config['noise']['sigma_max'] = cli_args['sigma_max']
    
    if cli_args.get('resume') is not None:
        config['training']['resume'] = cli_args['resume']
    
    if cli_args.get('resume_path') is not None:
        config['training']['resume_path'] = cli_args['resume_path']
    
    if cli_args.get('checkpoint_path') is not None:
        config['testing']['checkpoint_path'] = cli_args['checkpoint_path']
    
    return config

