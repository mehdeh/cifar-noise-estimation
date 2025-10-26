"""
Training script for CIFAR-10 Noise Estimation Model.

This script trains a ResNet model to estimate the noise level added to CIFAR-10 images.
Usage:
    python train.py --config config/default.yaml
    python train.py --config config/default.yaml --epochs 200 --lr 0.001
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
from src.training.trainer import Trainer
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
    '--config',
    type=click.Path(exists=True),
    default='config/default.yaml',
    help='Path to configuration YAML file'
)
@click.option(
    '--exp-name',
    type=str,
    default=None,
    help='Experiment name (will be prepended to timestamp)'
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
    help='Path to checkpoint to resume from'
)
def main(config, exp_name, epochs, batch_size, batch_size_test, lr, device, 
         seed, sigma_min, sigma_max, resume, resume_path):
    """
    Train CIFAR-10 noise estimation model.
    
    This script trains a ResNet model to predict the noise level (sigma) 
    added to CIFAR-10 images using the formula: x_noise = x + eps * sigma
    where eps ~ N(0, I).
    """
    # Load base configuration
    print(f"Loading configuration from: {config}")
    base_config = load_config(config)
    
    # Update config with CLI arguments
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
    
    config = update_config_with_cli(base_config, cli_args)
    
    # Set random seed
    set_seed(config['seed'])
    print(f"Random seed set to: {config['seed']}")
    
    # Create experiment directory
    exp_dir = create_exp_dir('runs', exp_name)
    print(f"Experiment directory: {exp_dir}")
    
    # Save configuration to experiment directory
    config_save_path = os.path.join(exp_dir, 'config.yaml')
    save_config(config, config_save_path)
    print(f"Configuration saved to: {config_save_path}")
    
    # Setup logger
    log_dir = os.path.join(exp_dir, 'logs')
    logger = Logger(log_dir, name='train')
    logger.info("=" * 60)
    logger.info("CIFAR-10 Noise Estimation Training")
    logger.info("=" * 60)
    logger.info(f"Experiment directory: {exp_dir}")
    logger.info(f"Configuration file: {config}")
    
    # Log configuration
    logger.info("\nConfiguration:")
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
    logger.info("\nBuilding model...")
    model = ResNet18(num_classes=config['model']['num_classes'])
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {config['model']['name']}")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
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
    logger.info(f"Loss plot: {os.path.join(exp_dir, 'plots', 'loss_curve.png')}")


if __name__ == '__main__':
    main()

