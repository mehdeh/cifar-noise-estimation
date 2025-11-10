#!/usr/bin/env python3
"""
Optimized training script with advanced PyTorch performance features
This script includes:
- Mixed precision training (AMP)
- Compiled model optimization
- Better memory management
- Performance monitoring
"""

import os
import sys
import argparse
import torch
import time
from contextlib import nullcontext

# Add src path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.models.resnet import ResNet18
from src.data.dataset import get_dataloaders, add_noise
from src.utils.config import load_config, validate_config
from src.utils.logger import Logger
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


class OptimizedTrainer:
    """Optimized trainer with advanced PyTorch features."""
    
    def __init__(self, config, exp_dir):
        self.config = config
        self.exp_dir = exp_dir
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Enable optimizations
        if torch.cuda.is_available():
            # Enable cudNN benchmarking for consistent input sizes
            torch.backends.cudnn.benchmark = True
            print("Enabled cudNN benchmark mode")
            
        # Setup mixed precision training
        self.use_amp = torch.cuda.is_available()
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None
        if self.use_amp:
            print("Enabled Automatic Mixed Precision (AMP)")
        
        # Create model
        self.model = ResNet18(num_classes=1).to(self.device)
        
        # Try to compile model for PyTorch 2.0+ optimization
        try:
            self.model = torch.compile(self.model)
            print("Model compiled for optimization")
        except Exception as e:
            print(f"Model compilation not available: {e}")
        
        # Setup loss and optimizer
        self.criterion = nn.MSELoss()
        
        if config['training']['optimizer'] == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=config['training']['learning_rate'],
                weight_decay=config['training']['weight_decay'],
                fused=True if torch.cuda.is_available() else False  # Use fused Adam on GPU
            )
        else:
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=config['training']['learning_rate'],
                momentum=config['training']['momentum'],
                weight_decay=config['training']['weight_decay']
            )
        
        # Setup scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['training']['epochs']
        )
        
        # Noise parameters
        self.sigma_min = config['noise']['sigma_min']
        self.sigma_max = config['noise']['sigma_max']
        
        # Training state
        self.epoch = 0
        self.best_loss = float('inf')
        
    def train_epoch(self, train_loader):
        """Train for one epoch with optimizations."""
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)
        
        # Use non_blocking transfer for better GPU utilization
        with tqdm(train_loader, desc=f"Epoch {self.epoch}") as pbar:
            for batch_idx, (images, _) in enumerate(pbar):
                # Transfer to GPU with non_blocking
                images = images.to(self.device, non_blocking=True)
                
                # Use autocast for mixed precision
                autocast_context = torch.amp.autocast('cuda') if self.use_amp else nullcontext()
                
                with autocast_context:
                    # Add noise
                    noisy_images, sigma_targets = add_noise(images, self.sigma_min, self.sigma_max)
                    
                    # Forward pass
                    predictions = self.model(noisy_images).squeeze()
                    loss = self.criterion(predictions, sigma_targets)
                
                # Backward pass with mixed precision
                self.optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
                
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()
                
                # Statistics
                total_loss += loss.item()
                avg_loss = total_loss / (batch_idx + 1)
                pbar.set_postfix({'loss': f'{avg_loss:.6f}'})
        
        return total_loss / num_batches
    
    def validate(self, val_loader):
        """Validate with optimizations."""
        self.model.eval()
        total_loss = 0.0
        num_batches = len(val_loader)
        
        with torch.no_grad():
            for images, _ in tqdm(val_loader, desc="Validation"):
                images = images.to(self.device, non_blocking=True)
                
                # Use autocast for validation too (saves memory)
                autocast_context = torch.amp.autocast('cuda') if self.use_amp else nullcontext()
                with autocast_context:
                    noisy_images, sigma_targets = add_noise(images, self.sigma_min, self.sigma_max)
                    predictions = self.model(noisy_images).squeeze()
                    loss = self.criterion(predictions, sigma_targets)
                
                total_loss += loss.item()
        
        return total_loss / num_batches
    
    def train(self, train_loader, val_loader):
        """Main training loop."""
        print(f"Starting optimized training...")
        print(f"Batch size: {self.config['data']['batch_size_train']}")
        print(f"Mixed precision: {self.use_amp}")
        print(f"Total batches per epoch: {len(train_loader)}")
        
        start_time = time.time()
        
        for epoch in range(self.config['training']['epochs']):
            self.epoch = epoch
            
            epoch_start = time.time()
            
            # Training
            train_loss = self.train_epoch(train_loader)
            
            # Validation
            val_loss = self.validate(val_loader)
            
            # Update scheduler
            self.scheduler.step()
            
            epoch_time = time.time() - epoch_start
            
            print(f"Epoch {epoch}: Train Loss: {train_loss:.6f}, "
                  f"Val Loss: {val_loss:.6f}, Time: {epoch_time:.2f}s")
            
            # Save best model
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                checkpoint_path = os.path.join(self.exp_dir, 'best_model_optimized.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': val_loss,
                }, checkpoint_path)
                print(f"Saved best model: {checkpoint_path}")
        
        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.2f}s")
        print(f"Average time per epoch: {total_time/self.config['training']['epochs']:.2f}s")


def main():
    parser = argparse.ArgumentParser(description='Optimized CIFAR-10 Noise Estimation Training')
    parser.add_argument('--config', type=str, default='config/optimized.yaml',
                        help='Configuration file path')
    parser.add_argument('--exp-name', type=str, default='optimized_training',
                        help='Experiment name')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    config = validate_config(config)
    
    # Create experiment directory
    from src.utils.config import create_exp_dir
    exp_dir = create_exp_dir('runs', args.exp_name)
    print(f"Experiment directory: {exp_dir}")
    
    # Load data
    print("Loading CIFAR-10 dataset...")
    train_loader, val_loader, _, _ = get_dataloaders(config)
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Create trainer and start training
    trainer = OptimizedTrainer(config, exp_dir)
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()

