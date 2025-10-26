"""
Testing/Evaluation script for CIFAR-10 Noise Estimation Model.

This script evaluates a trained ResNet model on the test set and generates
detailed metrics and visualizations.

Usage:
    python test.py --checkpoint runs/exp_20240101_120000/checkpoints/best_model.pth
    python test.py --checkpoint path/to/checkpoint.pth --config config/custom.yaml
"""

import os
import sys
import click
import torch
import random
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.models.resnet import ResNet18
from src.data.dataset import get_dataloaders
from src.training.evaluator import Evaluator
from src.utils.config import (
    load_config,
    save_config,
    create_exp_dir,
    update_config_with_cli
)
from src.utils.logger import Logger


def set_seed(seed):
    """
    Set random seed for reproducibility.
    
    Args:
        seed (int): Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@click.command()
@click.option(
    '--checkpoint',
    type=click.Path(exists=True),
    required=True,
    help='Path to model checkpoint file'
)
@click.option(
    '--config',
    type=click.Path(exists=True),
    default='config/default.yaml',
    help='Path to configuration YAML file'
)
@click.option(
    '--exp-name',
    type=str,
    default='test',
    help='Experiment name for saving results'
)
@click.option(
    '--batch-size',
    type=int,
    default=None,
    help='Test batch size'
)
@click.option(
    '--device',
    type=str,
    default=None,
    help='Device to use (cuda or cpu)'
)
@click.option(
    '--seed',
    type=int,
    default=None,
    help='Random seed for reproducibility'
)
@click.option(
    '--sigma-min',
    type=float,
    default=None,
    help='Minimum noise standard deviation for testing'
)
@click.option(
    '--sigma-max',
    type=float,
    default=None,
    help='Maximum noise standard deviation for testing'
)
@click.option(
    '--save-visualizations/--no-visualizations',
    default=True,
    help='Whether to save visualization plots'
)
@click.option(
    '--evaluate-by-noise-level',
    is_flag=True,
    default=False,
    help='Evaluate performance at different noise levels'
)
def main(checkpoint, config, exp_name, batch_size, device, seed, 
         sigma_min, sigma_max, save_visualizations, evaluate_by_noise_level):
    """
    Evaluate CIFAR-10 noise estimation model.
    
    This script loads a trained model and evaluates its performance on the
    test set. It generates detailed metrics and visualization plots.
    """
    # Load configuration
    print(f"Loading configuration from: {config}")
    base_config = load_config(config)
    
    # Update config with CLI arguments
    cli_args = {
        'batch_size_test': batch_size,
        'device': device,
        'seed': seed,
        'sigma_min': sigma_min,
        'sigma_max': sigma_max,
        'checkpoint_path': checkpoint
    }
    
    config = update_config_with_cli(base_config, cli_args)
    config['testing']['checkpoint_path'] = checkpoint
    
    # Set random seed
    set_seed(config['seed'])
    print(f"Random seed set to: {config['seed']}")
    
    # Create experiment directory for test results
    exp_dir = create_exp_dir('runs', exp_name)
    print(f"Test results directory: {exp_dir}")
    
    # Save configuration
    config_save_path = os.path.join(exp_dir, 'config.yaml')
    save_config(config, config_save_path)
    print(f"Configuration saved to: {config_save_path}")
    
    # Setup logger
    log_dir = os.path.join(exp_dir, 'logs')
    logger = Logger(log_dir, name='test')
    logger.info("=" * 60)
    logger.info("CIFAR-10 Noise Estimation Evaluation")
    logger.info("=" * 60)
    logger.info(f"Test results directory: {exp_dir}")
    logger.info(f"Checkpoint: {checkpoint}")
    
    # Log configuration
    logger.info("\nConfiguration:")
    logger.info(f"  Batch size: {config['data']['batch_size_test']}")
    logger.info(f"  Noise range: [{config['noise']['sigma_min']}, {config['noise']['sigma_max']}]")
    logger.info(f"  Device: {config['device']}")
    logger.info(f"  Seed: {config['seed']}")
    
    # Load data
    logger.info("\nLoading CIFAR-10 test dataset...")
    _, test_loader, _, _ = get_dataloaders(config)
    logger.info(f"Test samples: {len(test_loader.dataset)}")
    logger.info(f"Test batches: {len(test_loader)}")
    
    # Create model
    logger.info("\nBuilding model...")
    model = ResNet18(num_classes=config['model']['num_classes'])
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {config['model']['name']}")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Create evaluator
    evaluator = Evaluator(
        model=model,
        test_loader=test_loader,
        config=config,
        logger=logger,
        exp_dir=exp_dir
    )
    
    # Load checkpoint
    evaluator.load_checkpoint(checkpoint)
    
    # Evaluate
    logger.info("\n" + "=" * 60)
    metrics = evaluator.evaluate(save_visualizations=save_visualizations)
    logger.info("=" * 60)
    
    # Evaluate by noise level if requested
    if evaluate_by_noise_level:
        logger.info("\n" + "=" * 60)
        evaluator.evaluate_by_noise_level(num_bins=10)
        logger.info("=" * 60)
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Evaluation completed successfully!")
    logger.info("=" * 60)
    logger.info(f"Results saved to: {exp_dir}")
    logger.info(f"Metrics: {os.path.join(exp_dir, 'logs', 'test_metrics.json')}")
    if save_visualizations:
        logger.info(f"Plots: {os.path.join(exp_dir, 'plots')}")
    
    # Print key metrics
    print("\n" + "=" * 60)
    print("KEY METRICS:")
    print("=" * 60)
    print(f"MSE:  {metrics['mse']:.6f}")
    print(f"RMSE: {metrics['rmse']:.6f}")
    print(f"MAE:  {metrics['mae']:.6f}")
    print(f"R²:   {metrics['r2_score']:.6f}")
    print("=" * 60)


if __name__ == '__main__':
    main()

