"""Training module for noise estimation model."""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from ..data.dataset import add_noise
from ..utils.visualization import plot_losses, save_sample_images


class Trainer:
    """
    Trainer class for noise estimation model.
    Handles training loop, validation, checkpointing, and logging.
    """
    
    def __init__(self, model, train_loader, val_loader, config, logger, exp_dir):
        """
        Initialize trainer.
        
        Args:
            model (nn.Module): Neural network model
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader): Validation data loader
            config (dict): Configuration dictionary
            logger (Logger): Logger instance
            exp_dir (str): Experiment directory path
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.logger = logger
        self.exp_dir = exp_dir
        
        # Setup device
        self.device = torch.device(
            config['device'] if torch.cuda.is_available() else 'cpu'
        )
        self.model = self.model.to(self.device)
        
        # Setup loss function
        self.criterion = self._get_criterion(config['training']['loss'])
        
        # Setup optimizer
        self.optimizer = self._get_optimizer(config['training'])
        
        # Setup learning rate scheduler
        self.scheduler = self._get_scheduler(config['training'])
        
        # Training state
        self.start_epoch = 0
        self.best_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
        # Noise configuration
        self.sigma_min = config['noise']['sigma_min']
        self.sigma_max = config['noise']['sigma_max']
        
        # Resume from checkpoint if specified
        if config['training']['resume'] and config['training']['resume_path']:
            self.load_checkpoint(config['training']['resume_path'])
    
    def _get_criterion(self, loss_name):
        """Get loss function."""
        if loss_name == 'MSELoss':
            return nn.MSELoss()
        elif loss_name == 'L1Loss':
            return nn.L1Loss()
        else:
            raise ValueError(f"Unknown loss function: {loss_name}")
    
    def _get_optimizer(self, training_config):
        """Get optimizer."""
        if training_config['optimizer'] == 'SGD':
            return optim.SGD(
                self.model.parameters(),
                lr=training_config['learning_rate'],
                momentum=training_config['momentum'],
                weight_decay=training_config['weight_decay']
            )
        elif training_config['optimizer'] == 'Adam':
            return optim.Adam(
                self.model.parameters(),
                lr=training_config['learning_rate'],
                weight_decay=training_config['weight_decay']
            )
        else:
            raise ValueError(f"Unknown optimizer: {training_config['optimizer']}")
    
    def _get_scheduler(self, training_config):
        """Get learning rate scheduler."""
        if training_config['scheduler'] == 'CosineAnnealingLR':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=training_config['scheduler_params']['T_max']
            )
        elif training_config['scheduler'] == 'StepLR':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=training_config['scheduler_params'].get('step_size', 30),
                gamma=training_config['scheduler_params'].get('gamma', 0.1)
            )
        elif training_config['scheduler'] == 'None':
            return None
        else:
            return None
    
    def train_epoch(self, epoch):
        """
        Train for one epoch.
        
        Args:
            epoch (int): Current epoch number
            
        Returns:
            float: Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        with tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]") as pbar:
            for batch_idx, (images, _) in enumerate(pbar):
                images = images.to(self.device)
                
                # Add noise dynamically (different noise each iteration)
                noisy_images, sigma_targets = add_noise(
                    images, self.sigma_min, self.sigma_max
                )
                
                # Forward pass
                self.optimizer.zero_grad()
                predictions = self.model(noisy_images).squeeze()  # (B,)
                
                # Compute loss
                loss = self.criterion(predictions, sigma_targets)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Update statistics
                total_loss += loss.item()
                avg_loss = total_loss / (batch_idx + 1)
                
                # Update progress bar
                pbar.set_postfix({'loss': f'{avg_loss:.6f}'})
                
                # Log batch if needed
                if batch_idx % 50 == 0:
                    self.logger.log_batch(epoch, batch_idx, num_batches, loss.item(), 'train')
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self, epoch):
        """
        Validate model on validation set.
        
        Args:
            epoch (int): Current epoch number
            
        Returns:
            float: Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = len(self.val_loader)
        
        # For visualization (save first batch)
        save_visualizations = (epoch % self.config['logging']['plot_freq'] == 0)
        first_batch_saved = False
        
        with torch.no_grad():
            with tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]") as pbar:
                for batch_idx, (images, _) in enumerate(pbar):
                    images = images.to(self.device)
                    
                    # Add noise (consistent for validation)
                    noisy_images, sigma_targets = add_noise(
                        images, self.sigma_min, self.sigma_max
                    )
                    
                    # Forward pass
                    predictions = self.model(noisy_images).squeeze()  # (B,)
                    
                    # Compute loss
                    loss = self.criterion(predictions, sigma_targets)
                    total_loss += loss.item()
                    
                    # Update progress bar
                    avg_loss = total_loss / (batch_idx + 1)
                    pbar.set_postfix({'loss': f'{avg_loss:.6f}'})
                    
                    # Save sample images from first batch
                    if save_visualizations and not first_batch_saved:
                        save_path = os.path.join(
                            self.exp_dir, 'plots', f'samples_epoch_{epoch}.png'
                        )
                        save_sample_images(images, noisy_images, save_path, num_samples=8)
                        first_batch_saved = True
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def train(self):
        """
        Main training loop.
        """
        self.logger.info("Starting training...")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Total epochs: {self.config['training']['epochs']}")
        self.logger.info(f"Training samples: {len(self.train_loader.dataset)}")
        self.logger.info(f"Validation samples: {len(self.val_loader.dataset)}")
        
        num_epochs = self.config['training']['epochs']
        
        for epoch in range(self.start_epoch, num_epochs):
            # Train
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate(epoch)
            self.val_losses.append(val_loss)
            
            # Log epoch results
            self.logger.log_epoch(epoch, train_loss, val_loss)
            
            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
                self.logger.info(f"Learning rate: {current_lr:.6f}")
            
            # Save checkpoint
            is_best = val_loss < self.best_loss
            if is_best:
                self.best_loss = val_loss
            
            if (epoch % self.config['training']['save_freq'] == 0) or (epoch == num_epochs - 1) or is_best:
                self.save_checkpoint(epoch, is_best)
            
            # Plot losses
            if self.config['logging']['save_plots']:
                plot_path = os.path.join(self.exp_dir, 'plots', 'loss_curve.png')
                plot_losses(self.train_losses, self.val_losses, plot_path)
        
        self.logger.info("Training completed!")
        self.logger.info(f"Best validation loss: {self.best_loss:.6f}")
        
        # Save final metrics
        self.logger.save_metrics()
    
    def save_checkpoint(self, epoch, is_best=False):
        """
        Save model checkpoint.
        
        Args:
            epoch (int): Current epoch
            is_best (bool): Whether this is the best model so far
        """
        checkpoint_dir = os.path.join(self.exp_dir, 'checkpoints')
        
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_loss': self.best_loss,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(state, checkpoint_path)
        self.logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        # Save best model
        if is_best:
            best_path = os.path.join(checkpoint_dir, 'best_model.pth')
            torch.save(state, best_path)
            self.logger.info(f"Saved best model: {best_path}")
        
        # Save latest model (for easy resuming)
        latest_path = os.path.join(checkpoint_dir, 'latest.pth')
        torch.save(state, latest_path)
    
    def load_checkpoint(self, checkpoint_path):
        """
        Load checkpoint to resume training.
        
        Args:
            checkpoint_path (str): Path to checkpoint file
        """
        self.logger.info(f"Loading checkpoint from: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.start_epoch = checkpoint['epoch'] + 1
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        
        self.logger.info(f"Resumed from epoch {checkpoint['epoch']}")

