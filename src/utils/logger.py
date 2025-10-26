"""Logging utilities for training and evaluation."""

import os
import sys
import logging
from datetime import datetime


def setup_logger(log_dir, name='train'):
    """
    Setup logger that writes to both file and console.
    
    Args:
        log_dir (str): Directory to save log files
        name (str): Logger name
        
    Returns:
        logging.Logger: Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler
    log_file = os.path.join(log_dir, f'{name}.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


class Logger:
    """
    Training/evaluation logger that tracks metrics and saves logs.
    """
    
    def __init__(self, log_dir, name='train'):
        """
        Initialize logger.
        
        Args:
            log_dir (str): Directory to save logs
            name (str): Logger name
        """
        self.logger = setup_logger(log_dir, name)
        self.log_dir = log_dir
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'epochs': []
        }
    
    def info(self, message):
        """Log info message."""
        self.logger.info(message)
    
    def warning(self, message):
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message):
        """Log error message."""
        self.logger.error(message)
    
    def log_epoch(self, epoch, train_loss, val_loss):
        """
        Log epoch metrics.
        
        Args:
            epoch (int): Current epoch number
            train_loss (float): Training loss
            val_loss (float): Validation loss
        """
        self.metrics['epochs'].append(epoch)
        self.metrics['train_loss'].append(train_loss)
        self.metrics['val_loss'].append(val_loss)
        
        self.info(
            f"Epoch [{epoch}] - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
        )
    
    def log_batch(self, epoch, batch_idx, total_batches, loss, phase='train'):
        """
        Log batch information.
        
        Args:
            epoch (int): Current epoch
            batch_idx (int): Current batch index
            total_batches (int): Total number of batches
            loss (float): Batch loss
            phase (str): 'train' or 'val'
        """
        if batch_idx % 50 == 0:  # Log every 50 batches
            self.info(
                f"{phase.capitalize()} Epoch [{epoch}] "
                f"[{batch_idx}/{total_batches}] - Loss: {loss:.6f}"
            )
    
    def save_metrics(self):
        """Save metrics to file."""
        import json
        metrics_file = os.path.join(self.log_dir, 'metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=4)
    
    def get_metrics(self):
        """
        Get logged metrics.
        
        Returns:
            dict: Dictionary containing all logged metrics
        """
        return self.metrics


class ProgressMeter:
    """Simple progress meter for tracking training progress."""
    
    def __init__(self, num_batches, meters, prefix=""):
        """
        Initialize progress meter.
        
        Args:
            num_batches (int): Total number of batches
            meters (dict): Dictionary of meter names and values
            prefix (str): Prefix for display
        """
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
    
    def display(self, batch):
        """Display current progress."""
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [f"{name}: {val:.6f}" for name, val in self.meters.items()]
        print('\t'.join(entries))
    
    def _get_batch_fmtstr(self, num_batches):
        """Get format string for batch number."""
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

