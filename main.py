"""
Main script for training and evaluating CIFAR-10 noise estimation model.

This script includes two main modes:
- train: Train ResNet model for noise level estimation
- test: Evaluate trained model on test set

Usage:
    python main.py train --config config/default.yaml
    python main.py train --config config/default.yaml --epochs 200 --lr 0.001
    python main.py test --checkpoint runs/exp_20240101_120000/checkpoints/best_model.pth
    python main.py test --checkpoint path/to/checkpoint.pth --config config/custom.yaml
"""

import os
import sys
import click
import torch
import random
import numpy as np

# Add src path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.models.resnet import ResNet18
from src.data.dataset import get_dataloaders
from src.training.trainer import Trainer
from src.training.evaluator import Evaluator
from src.utils.config import (
    load_config,
    save_config,
    create_exp_dir,
    update_config_with_cli
)
from src.utils.logger import Logger
from src.utils.visualization import create_noise_estimation_table


def set_seed(seed):
    """
    Set random seed for reproducibility.
    
    Args:
        seed (int): Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_experiment(config_path, exp_name, cli_args, mode='train'):
    """
    Setup experiment including loading config, creating directory and logger.
    
    Args:
        config_path (str): Path to config file
        exp_name (str): Experiment name
        cli_args (dict): Command line arguments
        mode (str): Execution mode ('train' or 'test')
    
    Returns:
        tuple: (config, exp_dir, logger)
    """
    # Load base config
    print(f"Loading config from: {config_path}")
    base_config = load_config(config_path)
    
    # Update config with CLI arguments
    config = update_config_with_cli(base_config, cli_args)
    
    # Set random seed
    set_seed(config['seed'])
    print(f"Random seed set: {config['seed']}")
    
    # Create experiment directory
    exp_dir = create_exp_dir('runs', exp_name)
    print(f"Experiment directory: {exp_dir}")
    
    # Save config
    config_save_path = os.path.join(exp_dir, 'config.yaml')
    save_config(config, config_save_path)
    print(f"Config saved to: {config_save_path}")
    
    # Setup logger
    log_dir = os.path.join(exp_dir, 'logs')
    logger = Logger(log_dir, name=mode)
    
    # Log title
    title = "Training CIFAR-10 Noise Estimation Model" if mode == 'train' else "Evaluating CIFAR-10 Noise Estimation Model"
    logger.info("=" * 60)
    logger.info(title)
    logger.info("=" * 60)
    logger.info(f"Experiment directory: {exp_dir}")
    logger.info(f"Config file: {config_path}")
    
    return config, exp_dir, logger


def create_model_and_log_info(config, logger):
    """
    Create model and log its information.
    
    Args:
        config (dict): Experiment config
        logger (Logger): Logger
    
    Returns:
        torch.nn.Module: Created model
    """
    logger.info("\nCreating model...")
    model = ResNet18(num_classes=config['model']['num_classes'])
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {config['model']['name']}")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    return model


@click.group()
@click.version_option(version='1.0.0')
def main():
    """
    Comprehensive tool for training and evaluating CIFAR-10 noise estimation model.
    
    This tool includes two main modes:
    - train: Train new model
    - test: Evaluate trained model
    """
    pass


@main.command()
@click.option(
    '--config',
    type=click.Path(exists=True),
    default='config/default.yaml',
    help='Path to YAML config file'
)
@click.option(
    '--exp-name',
    type=str,
    default=None,
    help='Experiment name (timestamp will be added)'
)
@click.option(
    '--epochs',
    type=int,
    default=None,
    help='Number of training epochs'
)
@click.option(
    '--batch-size',
    type=int,
    default=None,
    help='Training batch size'
)
@click.option(
    '--batch-size-test',
    type=int,
    default=None,
    help='Test batch size'
)
@click.option(
    '--lr',
    type=float,
    default=None,
    help='Learning rate'
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
    help='Minimum noise standard deviation'
)
@click.option(
    '--sigma-max',
    type=float,
    default=None,
    help='Maximum noise standard deviation'
)
@click.option(
    '--resume',
    is_flag=True,
    default=False,
    help='Resume training from checkpoint'
)
@click.option(
    '--resume-path',
    type=click.Path(exists=True),
    default=None,
    help='Path to checkpoint for resuming training'
)
def train(config, exp_name, epochs, batch_size, batch_size_test, lr, device,
          seed, sigma_min, sigma_max, resume, resume_path):
    """
    Train CIFAR-10 noise estimation model.
    
    This command trains a ResNet model to predict the noise level (sigma) added
    to CIFAR-10 images using the formula x_noise = x + eps * sigma
    where eps ~ N(0, I).
    """
    # Prepare CLI arguments
    cli_args = {
        'epochs': epochs,
        'batch_size': batch_size,
        'batch_size_test': batch_size_test,
        'lr': lr,
        'device': device,
        'seed': seed,
        'sigma_min': sigma_min,
        'sigma_max': sigma_max,
        'resume': resume,
        'resume_path': resume_path
    }
    
    # Setup experiment
    config, exp_dir, logger = setup_experiment(config, exp_name, cli_args, 'train')
    
    # Log config
    logger.info("\nConfig:")
    logger.info(f"  Epochs: {config['training']['epochs']}")
    logger.info(f"  Batch size (train): {config['data']['batch_size_train']}")
    logger.info(f"  Batch size (test): {config['data']['batch_size_test']}")
    logger.info(f"  Learning rate: {config['training']['learning_rate']}")
    logger.info(f"  Optimizer: {config['training']['optimizer']}")
    logger.info(f"  Scheduler: {config['training']['scheduler']}")
    logger.info(f"  Noise range: [{config['noise']['sigma_min']}, {config['noise']['sigma_max']}]")
    logger.info(f"  Device: {config['device']}")
    logger.info(f"  Seed: {config['seed']}")
    
    # Load data
    logger.info("\nLoading CIFAR-10 dataset...")
    train_loader, val_loader, _, _ = get_dataloaders(config)
    logger.info(f"Training samples: {len(train_loader.dataset)}")
    logger.info(f"Validation samples: {len(val_loader.dataset)}")
    logger.info(f"Training batches: {len(train_loader)}")
    logger.info(f"Validation batches: {len(val_loader)}")
    
    # Create model
    model = create_model_and_log_info(config, logger)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        logger=logger,
        exp_dir=exp_dir
    )
    
    # Train
    try:
        trainer.train()
    except KeyboardInterrupt:
        logger.warning("\nTraining interrupted by user!")
        logger.info("Saving checkpoint...")
        trainer.save_checkpoint(trainer.start_epoch + len(trainer.train_losses) - 1)
        logger.info("Checkpoint saved. Exiting...")
        sys.exit(0)
    
    logger.info("\n" + "=" * 60)
    logger.info("Training completed successfully!")
    logger.info("=" * 60)
    logger.info(f"Results saved to: {exp_dir}")
    logger.info(f"Best model: {os.path.join(exp_dir, 'checkpoints', 'best_model.pth')}")
    logger.info(f"Loss curve: {os.path.join(exp_dir, 'plots', 'loss_curve.png')}")
    
    # Generate noise estimation visualization table
    logger.info("\nGenerating noise estimation visualization...")
    try:
        # Load test data for visualization
        _, _, _, test_loader = get_dataloaders(config)
        
        # Create plots directory if it doesn't exist
        plots_dir = os.path.join(exp_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Generate visualization
        viz_path = os.path.join(plots_dir, 'noise_estimation_table.png')
        device = config['device'] if torch.cuda.is_available() else 'cpu'
        
        visualization_results = create_noise_estimation_table(
            model=trainer.model,
            test_loader=test_loader,
            config=config,
            save_path=viz_path,
            num_samples=16,
            device=device
        )
        
        logger.info(f"Noise estimation table saved: {viz_path}")
        
    except Exception as e:
        logger.warning(f"Failed to generate noise estimation visualization: {str(e)}")


@main.command()
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
    help='Path to YAML config file'
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
def test(checkpoint, config, exp_name, batch_size, device, seed,
         sigma_min, sigma_max, save_visualizations, evaluate_by_noise_level):
    """
    Evaluate CIFAR-10 noise estimation model.
    
    This command loads a trained model and evaluates its performance on the
    test set. It also produces accuracy metrics and visualization plots.
    """
    # Prepare CLI arguments
    cli_args = {
        'batch_size_test': batch_size,
        'device': device,
        'seed': seed,
        'sigma_min': sigma_min,
        'sigma_max': sigma_max,
        'checkpoint_path': checkpoint
    }
    
    # Setup experiment
    config, exp_dir, logger = setup_experiment(config, exp_name, cli_args, 'test')
    config['testing']['checkpoint_path'] = checkpoint
    
    # Log information
    logger.info(f"Checkpoint: {checkpoint}")
    
    # Log config
    logger.info("\nConfig:")
    logger.info(f"  Batch size: {config['data']['batch_size_test']}")
    logger.info(f"  Noise range: [{config['noise']['sigma_min']}, {config['noise']['sigma_max']}]")
    logger.info(f"  Device: {config['device']}")
    logger.info(f"  Seed: {config['seed']}")
    
    # Load data
    logger.info("\nLoading CIFAR-10 test set...")
    _, test_loader, _, _ = get_dataloaders(config)
    logger.info(f"Test samples: {len(test_loader.dataset)}")
    logger.info(f"Test batches: {len(test_loader)}")
    
    # Create model
    model = create_model_and_log_info(config, logger)
    
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
    
    # Generate noise estimation visualization table
    logger.info("\nGenerating noise estimation visualization...")
    try:
        # Create plots directory if it doesn't exist
        plots_dir = os.path.join(exp_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Generate visualization
        viz_path = os.path.join(plots_dir, 'noise_estimation_table.png')
        device = config['device'] if torch.cuda.is_available() else 'cpu'
        
        visualization_results = create_noise_estimation_table(
            model=evaluator.model,
            test_loader=test_loader,
            config=config,
            save_path=viz_path,
            num_samples=16,
            device=device
        )
        
        logger.info(f"Noise estimation table saved: {viz_path}")
        
    except Exception as e:
        logger.warning(f"Failed to generate noise estimation visualization: {str(e)}")
    
    # Print key metrics
    print("\n" + "=" * 60)
    print("Key Metrics:")
    print("=" * 60)
    print(f"MSE:  {metrics['mse']:.6f}")
    print(f"RMSE: {metrics['rmse']:.6f}")
    print(f"MAE:  {metrics['mae']:.6f}")
    print(f"R²:   {metrics['r2_score']:.6f}")
    print("=" * 60)


if __name__ == '__main__':
    main()
