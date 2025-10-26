"""Evaluation module for noise estimation model."""

import os
import torch
import numpy as np
from tqdm import tqdm

from ..data.dataset import add_noise
from ..utils.visualization import (
    plot_scatter_predictions,
    plot_noise_distribution,
    save_sample_images
)


class Evaluator:
    """
    Evaluator class for noise estimation model.
    Handles model evaluation and generates detailed reports and visualizations.
    """
    
    def __init__(self, model, test_loader, config, logger, exp_dir):
        """
        Initialize evaluator.
        
        Args:
            model (nn.Module): Neural network model
            test_loader (DataLoader): Test data loader
            config (dict): Configuration dictionary
            logger (Logger): Logger instance
            exp_dir (str): Experiment directory path
        """
        self.model = model
        self.test_loader = test_loader
        self.config = config
        self.logger = logger
        self.exp_dir = exp_dir
        
        # Setup device
        self.device = torch.device(
            config['device'] if torch.cuda.is_available() else 'cpu'
        )
        self.model = self.model.to(self.device)
        
        # Noise configuration
        self.sigma_min = config['noise']['sigma_min']
        self.sigma_max = config['noise']['sigma_max']
    
    def load_checkpoint(self, checkpoint_path):
        """
        Load model checkpoint for evaluation.
        
        Args:
            checkpoint_path (str): Path to checkpoint file
        """
        self.logger.info(f"Loading checkpoint from: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        epoch = checkpoint.get('epoch', 'unknown')
        self.logger.info(f"Loaded model from epoch {epoch}")
    
    def evaluate(self, save_visualizations=True):
        """
        Evaluate model on test set.
        
        Args:
            save_visualizations (bool): Whether to save visualization plots
            
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        self.logger.info("Starting evaluation...")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Test samples: {len(self.test_loader.dataset)}")
        
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        all_errors = []
        total_loss = 0.0
        
        # For saving sample images
        first_batch_images = None
        first_batch_noisy = None
        
        with torch.no_grad():
            with tqdm(self.test_loader, desc="Evaluating") as pbar:
                for batch_idx, (images, _) in enumerate(pbar):
                    images = images.to(self.device)
                    
                    # Add noise
                    noisy_images, sigma_targets = add_noise(
                        images, self.sigma_min, self.sigma_max
                    )
                    
                    # Forward pass
                    predictions = self.model(noisy_images).squeeze()  # (B,)
                    
                    # Compute loss
                    loss = torch.mean((predictions - sigma_targets) ** 2)
                    total_loss += loss.item()
                    
                    # Store predictions and targets
                    all_predictions.append(predictions.cpu())
                    all_targets.append(sigma_targets.cpu())
                    all_errors.append((predictions - sigma_targets).cpu())
                    
                    # Save first batch for visualization
                    if batch_idx == 0 and save_visualizations:
                        first_batch_images = images
                        first_batch_noisy = noisy_images
                    
                    # Update progress bar
                    avg_loss = total_loss / (batch_idx + 1)
                    pbar.set_postfix({'loss': f'{avg_loss:.6f}'})
        
        # Concatenate all results
        all_predictions = torch.cat(all_predictions)
        all_targets = torch.cat(all_targets)
        all_errors = torch.cat(all_errors)
        
        # Calculate metrics
        metrics = self.calculate_metrics(all_predictions, all_targets, all_errors)
        metrics['average_loss'] = total_loss / len(self.test_loader)
        
        # Log metrics
        self.log_metrics(metrics)
        
        # Save visualizations
        if save_visualizations:
            self.save_visualizations(
                all_predictions, all_targets, all_errors,
                first_batch_images, first_batch_noisy
            )
        
        # Save metrics to file
        self.save_metrics_to_file(metrics)
        
        self.logger.info("Evaluation completed!")
        
        return metrics
    
    def calculate_metrics(self, predictions, targets, errors):
        """
        Calculate evaluation metrics.
        
        Args:
            predictions (torch.Tensor): Predicted noise levels
            targets (torch.Tensor): True noise levels
            errors (torch.Tensor): Prediction errors
            
        Returns:
            dict: Dictionary of metrics
        """
        predictions_np = predictions.numpy()
        targets_np = targets.numpy()
        errors_np = errors.numpy()
        
        metrics = {
            'mse': np.mean(errors_np ** 2),
            'rmse': np.sqrt(np.mean(errors_np ** 2)),
            'mae': np.mean(np.abs(errors_np)),
            'max_error': np.max(np.abs(errors_np)),
            'std_error': np.std(errors_np),
            'mean_error': np.mean(errors_np),
            'median_error': np.median(np.abs(errors_np)),
            'r2_score': self.calculate_r2(predictions_np, targets_np),
            'pearson_correlation': np.corrcoef(predictions_np, targets_np)[0, 1]
        }
        
        return metrics
    
    def calculate_r2(self, predictions, targets):
        """
        Calculate R-squared score.
        
        Args:
            predictions (np.ndarray): Predicted values
            targets (np.ndarray): True values
            
        Returns:
            float: R-squared score
        """
        ss_res = np.sum((targets - predictions) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        return r2
    
    def log_metrics(self, metrics):
        """
        Log metrics to logger.
        
        Args:
            metrics (dict): Dictionary of metrics
        """
        self.logger.info("\n" + "=" * 60)
        self.logger.info("EVALUATION METRICS")
        self.logger.info("=" * 60)
        
        self.logger.info(f"MSE (Mean Squared Error):     {metrics['mse']:.6f}")
        self.logger.info(f"RMSE (Root Mean Squared Error): {metrics['rmse']:.6f}")
        self.logger.info(f"MAE (Mean Absolute Error):    {metrics['mae']:.6f}")
        self.logger.info(f"Median Absolute Error:        {metrics['median_error']:.6f}")
        self.logger.info(f"Max Absolute Error:           {metrics['max_error']:.6f}")
        self.logger.info(f"Mean Error (Bias):            {metrics['mean_error']:.6f}")
        self.logger.info(f"Std Error:                    {metrics['std_error']:.6f}")
        self.logger.info(f"R² Score:                     {metrics['r2_score']:.6f}")
        self.logger.info(f"Pearson Correlation:          {metrics['pearson_correlation']:.6f}")
        
        self.logger.info("=" * 60 + "\n")
    
    def save_metrics_to_file(self, metrics):
        """
        Save metrics to JSON file.
        
        Args:
            metrics (dict): Dictionary of metrics
        """
        import json
        
        metrics_file = os.path.join(self.exp_dir, 'logs', 'test_metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        self.logger.info(f"Saved metrics to: {metrics_file}")
    
    def save_visualizations(self, predictions, targets, errors, 
                           sample_images, sample_noisy):
        """
        Generate and save visualization plots.
        
        Args:
            predictions (torch.Tensor): All predictions
            targets (torch.Tensor): All targets
            errors (torch.Tensor): All errors
            sample_images (torch.Tensor): Sample clean images
            sample_noisy (torch.Tensor): Sample noisy images
        """
        plots_dir = os.path.join(self.exp_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        self.logger.info("Generating visualizations...")
        
        # 1. Scatter plot: True vs Predicted
        scatter_path = os.path.join(plots_dir, 'test_scatter_predictions.png')
        plot_scatter_predictions(targets, predictions, scatter_path)
        self.logger.info(f"Saved scatter plot: {scatter_path}")
        
        # 2. Noise distribution plots
        dist_path = os.path.join(plots_dir, 'test_noise_distribution.png')
        plot_noise_distribution(targets, predictions, dist_path)
        self.logger.info(f"Saved distribution plots: {dist_path}")
        
        # 3. Sample images (clean vs noisy)
        if sample_images is not None and sample_noisy is not None:
            samples_path = os.path.join(plots_dir, 'test_sample_images.png')
            save_sample_images(sample_images, sample_noisy, samples_path, num_samples=8)
            self.logger.info(f"Saved sample images: {samples_path}")
    
    def evaluate_by_noise_level(self, num_bins=10):
        """
        Evaluate model performance at different noise levels.
        
        Args:
            num_bins (int): Number of bins to divide noise range
            
        Returns:
            dict: Performance metrics per noise level bin
        """
        self.logger.info(f"Evaluating by noise level ({num_bins} bins)...")
        
        self.model.eval()
        
        # Create bins
        sigma_bins = np.linspace(self.sigma_min, self.sigma_max, num_bins + 1)
        bin_predictions = [[] for _ in range(num_bins)]
        bin_targets = [[] for _ in range(num_bins)]
        
        with torch.no_grad():
            for images, _ in tqdm(self.test_loader, desc="Evaluating by noise level"):
                images = images.to(self.device)
                
                # Add noise
                noisy_images, sigma_targets = add_noise(
                    images, self.sigma_min, self.sigma_max
                )
                
                # Forward pass
                predictions = self.model(noisy_images).squeeze()
                
                # Assign to bins
                for pred, target in zip(predictions.cpu(), sigma_targets.cpu()):
                    bin_idx = min(
                        np.digitize(target.item(), sigma_bins) - 1,
                        num_bins - 1
                    )
                    bin_predictions[bin_idx].append(pred.item())
                    bin_targets[bin_idx].append(target.item())
        
        # Calculate metrics per bin
        results = {}
        for i in range(num_bins):
            if len(bin_predictions[i]) > 0:
                preds = np.array(bin_predictions[i])
                tgts = np.array(bin_targets[i])
                errors = preds - tgts
                
                bin_range = f"[{sigma_bins[i]:.3f}, {sigma_bins[i+1]:.3f}]"
                results[bin_range] = {
                    'count': len(preds),
                    'mae': np.mean(np.abs(errors)),
                    'mse': np.mean(errors ** 2),
                    'mean_error': np.mean(errors)
                }
        
        # Log results
        self.logger.info("\nPerformance by Noise Level:")
        self.logger.info("-" * 60)
        for bin_range, metrics in results.items():
            self.logger.info(
                f"σ {bin_range}: Count={metrics['count']}, "
                f"MAE={metrics['mae']:.6f}, MSE={metrics['mse']:.6f}"
            )
        
        return results

