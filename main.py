"""
CIFAR-10 Noise Estimation - Clean CLI Interface

This script provides a unified interface for training and evaluating noise estimation models.
The design follows clean code principles with minimal CLI parameters and config-first approach.

Usage:
    python main.py train [--config CONFIG] [--exp-name NAME] [--device DEVICE] [--resume]
    python main.py test --checkpoint CHECKPOINT [--config CONFIG] [--exp-name NAME] [--device DEVICE]

Examples:
    # Train with default config
    python main.py train
    
    # Train with custom config and experiment name
    python main.py train --config config/custom.yaml --exp-name my_experiment
    
    # Resume training from checkpoint
    python main.py train --resume --resume-from path/to/checkpoint.pth
    
    # Evaluate trained model
    python main.py test --checkpoint runs/exp_20240101_120000/checkpoints/best_model.pth
"""

import os
import sys
import argparse
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
    validate_config
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


def setup_experiment(config_path, exp_name, mode, device_override=None):
    """
    Setup experiment: load config, create directories, setup logging.
    
    Args:
        config_path (str): Path to config file
        exp_name (str): Experiment name (optional)
        mode (str): 'train' or 'test'
        device_override (str): Device override (optional)
    
    Returns:
        tuple: (config, exp_dir, logger)
    """
    # Load and validate configuration
    print(f"Loading config from: {config_path}")
    config = load_config(config_path)
    config = validate_config(config)
    
    # Apply device override if specified
    if device_override:
        config['device'] = device_override
    
    # Set random seed for reproducibility
    set_seed(config['seed'])
    print(f"Random seed set: {config['seed']}")
    
    # Create experiment directory
    exp_dir = create_exp_dir('runs', exp_name)
    print(f"Experiment directory: {exp_dir}")
    
    # Save config to experiment directory
    config_save_path = os.path.join(exp_dir, 'config.yaml')
    save_config(config, config_save_path)
    print(f"Config saved to: {config_save_path}")
    
    # Setup logger
    log_dir = os.path.join(exp_dir, 'logs')
    logger = Logger(log_dir, name=mode)
    
    # Log header
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


def run_training(config_path, exp_name, device_override, resume, resume_from):
    """
    Run training pipeline with clean separation of concerns.
    
    Args:
        config_path (str): Path to config file
        exp_name (str): Experiment name (optional)
        device_override (str): Device override (optional)
        resume (bool): Whether to resume training
        resume_from (str): Checkpoint path to resume from (optional)
    """
    # Setup experiment
    config, exp_dir, logger = setup_experiment(config_path, exp_name, 'train', device_override)
    
    # Apply resume settings
    if resume:
        config['training']['resume'] = True
    if resume_from:
        config['training']['resume_path'] = resume_from
    
    # Log configuration
    logger.info("\nTraining Configuration:")
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
    
    # Run training with keyboard interrupt handling
    try:
        trainer.train()
    except KeyboardInterrupt:
        logger.warning("\nTraining interrupted by user!")
        logger.info("Saving checkpoint...")
        trainer.save_checkpoint(trainer.start_epoch + len(trainer.train_losses) - 1)
        logger.info("Checkpoint saved. Exiting...")
        sys.exit(0)
    
    # Log completion
    logger.info("\n" + "=" * 60)
    logger.info("Training completed successfully!")
    logger.info("=" * 60)
    logger.info(f"Results saved to: {exp_dir}")
    logger.info(f"Best model: {os.path.join(exp_dir, 'checkpoints', 'best_model.pth')}")
    logger.info(f"Loss curve: {os.path.join(exp_dir, 'plots', 'loss_curve.png')}")
    
    # Generate visualization
    generate_noise_estimation_visualization(trainer.model, config, exp_dir, logger)


def run_evaluation(checkpoint_path, config_path, exp_name, device_override):
    """
    Run evaluation pipeline with clean separation of concerns.
    
    Args:
        checkpoint_path (str): Path to model checkpoint
        config_path (str): Path to config file
        exp_name (str): Experiment name (optional)
        device_override (str): Device override (optional)
    """
    # Setup experiment
    config, exp_dir, logger = setup_experiment(config_path, exp_name, 'test', device_override)
    config['testing']['checkpoint_path'] = checkpoint_path
    
    # Log information
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info("\nEvaluation Configuration:")
    logger.info(f"  Batch size: {config['data']['batch_size_test']}")
    logger.info(f"  Noise range: [{config['noise']['sigma_min']}, {config['noise']['sigma_max']}]")
    logger.info(f"  Device: {config['device']}")
    logger.info(f"  Seed: {config['seed']}")
    
    # Load data
    logger.info("\nLoading CIFAR-10 test set...")
    _, _, _, test_loader = get_dataloaders(config)
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
    evaluator.load_checkpoint(checkpoint_path)
    
    # Run evaluation
    logger.info("\n" + "=" * 60)
    metrics = evaluator.evaluate(save_visualizations=True)
    logger.info("=" * 60)
    
    # Evaluate by noise level
    logger.info("\n" + "=" * 60)
    evaluator.evaluate_by_noise_level(num_bins=10)
    logger.info("=" * 60)
    
    # Log completion
    logger.info("\n" + "=" * 60)
    logger.info("Evaluation completed successfully!")
    logger.info("=" * 60)
    logger.info(f"Results saved to: {exp_dir}")
    logger.info(f"Metrics: {os.path.join(exp_dir, 'logs', 'test_metrics.json')}")
    logger.info(f"Plots: {os.path.join(exp_dir, 'plots')}")
    
    # Generate visualization
    generate_noise_estimation_visualization(evaluator.model, config, exp_dir, logger)
    
    # Print key metrics
    print("\n" + "=" * 60)
    print("Key Metrics:")
    print("=" * 60)
    print(f"MSE:  {metrics['mse']:.6f}")
    print(f"RMSE: {metrics['rmse']:.6f}")
    print(f"MAE:  {metrics['mae']:.6f}")
    print(f"R²:   {metrics['r2_score']:.6f}")
    print("=" * 60)


def generate_noise_estimation_visualization(model, config, exp_dir, logger):
    """
    Generate noise estimation visualization table.
    
    Args:
        model: Trained model
        config (dict): Configuration
        exp_dir (str): Experiment directory
        logger: Logger instance
    """
    logger.info("\nGenerating noise estimation visualization...")
    try:
        # Load test data for visualization
        _, _, _, test_loader = get_dataloaders(config)
        
        # Create plots directory
        plots_dir = os.path.join(exp_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Generate visualization
        viz_path = os.path.join(plots_dir, 'noise_estimation_table.png')
        device = config['device'] if torch.cuda.is_available() else 'cpu'
        
        visualization_results = create_noise_estimation_table(
            model=model,
            test_loader=test_loader,
            config=config,
            save_path=viz_path,
            num_samples=32,
            device=device
        )
        
        logger.info(f"Noise estimation table saved: {viz_path}")
        
    except Exception as e:
        logger.warning(f"Failed to generate noise estimation visualization: {str(e)}")


def create_argument_parser():
    """
    Create and configure argument parser with clean structure.
    
    Returns:
        argparse.ArgumentParser: Configured parser
    """
    parser = argparse.ArgumentParser(
        prog='CIFAR-10 Noise Estimation',
        description='Train and evaluate noise estimation models on CIFAR-10 dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default configuration
  python main.py train
  
  # Train with custom config and experiment name
  python main.py train --config config/custom.yaml --exp-name my_experiment
  
  # Resume training from checkpoint
  python main.py train --resume --resume-from path/to/checkpoint.pth
  
  # Evaluate trained model
  python main.py test --checkpoint runs/exp_20240101_120000/checkpoints/best_model.pth
        """
    )
    
    # Add subcommands
    subparsers = parser.add_subparsers(dest='mode', help='Operation mode', required=True)
    
    # Training parser
    train_parser = subparsers.add_parser('train', help='Train noise estimation model')
    train_parser.add_argument(
        '--config', 
        type=str, 
        default='config/default.yaml',
        help='Path to YAML config file (default: config/default.yaml)'
    )
    train_parser.add_argument(
        '--exp-name', 
        type=str, 
        default=None,
        help='Experiment name prefix (timestamp will be added)'
    )
    train_parser.add_argument(
        '--device', 
        type=str, 
        choices=['cuda', 'cpu'], 
        default=None,
        help='Device to use (overrides config setting)'
    )
    train_parser.add_argument(
        '--resume', 
        action='store_true',
        help='Resume training from latest checkpoint'
    )
    train_parser.add_argument(
        '--resume-from', 
        type=str, 
        default=None,
        help='Resume training from specific checkpoint path'
    )
    
    # Testing parser  
    test_parser = subparsers.add_parser('test', help='Evaluate trained model')
    test_parser.add_argument(
        '--checkpoint', 
        type=str, 
        required=True,
        help='Path to model checkpoint file'
    )
    test_parser.add_argument(
        '--config', 
        type=str, 
        default='config/default.yaml',
        help='Path to YAML config file (default: config/default.yaml)'
    )
    test_parser.add_argument(
        '--exp-name', 
        type=str, 
        default='test',
        help='Experiment name for saving results (default: test)'
    )
    test_parser.add_argument(
        '--device', 
        type=str, 
        choices=['cuda', 'cpu'], 
        default=None,
        help='Device to use (overrides config setting)'
    )
    
    return parser


def main():
    """
    Main entry point with clean argument parsing and execution flow.
    """
    parser = create_argument_parser()
    args = parser.parse_args()
    
    try:
        if args.mode == 'train':
            run_training(
                config_path=args.config,
                exp_name=args.exp_name,
                device_override=args.device,
                resume=args.resume,
                resume_from=args.resume_from
            )
        
        elif args.mode == 'test':
            run_evaluation(
                checkpoint_path=args.checkpoint,
                config_path=args.config,
                exp_name=args.exp_name,
                device_override=args.device
            )
        
        else:
            parser.print_help()
            sys.exit(1)
    
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
