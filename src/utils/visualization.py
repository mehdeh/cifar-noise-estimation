"""Visualization utilities for plotting and saving results."""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision.utils import make_grid


def plot_losses(train_losses, val_losses, save_path=None):
    """
    Plot training and validation losses.
    
    Args:
        train_losses (list): List of training losses per epoch
        val_losses (list): List of validation losses per epoch
        save_path (str): Path to save the plot. If None, only display
    """
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_predictions(images, targets, predictions, save_path=None, num_samples=16):
    """
    Plot clean images, noisy images, and model predictions.
    
    Args:
        images (torch.Tensor): Clean images (B, C, H, W)
        targets (torch.Tensor): Target noise levels (B,)
        predictions (torch.Tensor): Predicted noise levels (B,)
        save_path (str): Path to save the plot
        num_samples (int): Number of samples to display
    """
    num_samples = min(num_samples, images.shape[0])
    
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.flatten()
    
    for i in range(num_samples):
        ax = axes[i]
        
        # Denormalize image for display
        img = images[i].cpu()
        img = denormalize_image(img)
        img = img.permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        
        ax.imshow(img)
        ax.set_title(
            f'Target: {targets[i]:.3f}\nPred: {predictions[i]:.3f}',
            fontsize=9
        )
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_noise_distribution(sigma_values, predictions, save_path=None):
    """
    Plot distribution of noise levels and prediction errors.
    
    Args:
        sigma_values (torch.Tensor): True noise levels
        predictions (torch.Tensor): Predicted noise levels
        save_path (str): Path to save the plot
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # True noise distribution
    axes[0].hist(sigma_values.cpu().numpy(), bins=50, alpha=0.7, color='blue')
    axes[0].set_xlabel('Noise Level (σ)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('True Noise Distribution')
    axes[0].grid(True, alpha=0.3)
    
    # Predicted noise distribution
    axes[1].hist(predictions.cpu().numpy(), bins=50, alpha=0.7, color='green')
    axes[1].set_xlabel('Noise Level (σ)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Predicted Noise Distribution')
    axes[1].grid(True, alpha=0.3)
    
    # Prediction error distribution
    errors = (predictions - sigma_values).cpu().numpy()
    axes[2].hist(errors, bins=50, alpha=0.7, color='red')
    axes[2].set_xlabel('Prediction Error')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Prediction Error Distribution')
    axes[2].axvline(x=0, color='black', linestyle='--', linewidth=2)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def save_sample_images(clean_images, noisy_images, save_path, num_samples=8):
    """
    Save a grid of clean and noisy images side by side.
    
    Args:
        clean_images (torch.Tensor): Clean images (B, C, H, W)
        noisy_images (torch.Tensor): Noisy images (B, C, H, W)
        save_path (str): Path to save the image grid
        num_samples (int): Number of samples to save
    """
    num_samples = min(num_samples, clean_images.shape[0])
    
    # Select samples
    clean_samples = clean_images[:num_samples]
    noisy_samples = noisy_images[:num_samples]
    
    # Denormalize for visualization
    clean_samples = denormalize_batch(clean_samples)
    noisy_samples = denormalize_batch(noisy_samples)
    
    # Create grids
    clean_grid = make_grid(clean_samples, nrow=4, padding=2, normalize=False)
    noisy_grid = make_grid(noisy_samples, nrow=4, padding=2, normalize=False)
    
    # Combine grids vertically
    combined_grid = torch.cat([clean_grid, noisy_grid], dim=1)
    
    # Save
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(combined_grid.permute(1, 2, 0).cpu().numpy())
    ax.set_title('Top: Clean Images | Bottom: Noisy Images', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def denormalize_image(img, mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)):
    """
    Denormalize a single image tensor.
    
    Args:
        img (torch.Tensor): Normalized image (C, H, W)
        mean (tuple): Mean used for normalization
        std (tuple): Std used for normalization
        
    Returns:
        torch.Tensor: Denormalized image
    """
    # Ensure tensors are on the same device as input image
    device = img.device
    mean = torch.tensor(mean, device=device, dtype=img.dtype).view(3, 1, 1)
    std = torch.tensor(std, device=device, dtype=img.dtype).view(3, 1, 1)
    return img * std + mean


def denormalize_batch(imgs, mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)):
    """
    Denormalize a batch of images.
    
    Args:
        imgs (torch.Tensor): Normalized images (B, C, H, W)
        mean (tuple): Mean used for normalization
        std (tuple): Std used for normalization
        
    Returns:
        torch.Tensor: Denormalized images
    """
    # Ensure tensors are on the same device as input images
    device = imgs.device
    mean = torch.tensor(mean, device=device, dtype=imgs.dtype).view(1, 3, 1, 1)
    std = torch.tensor(std, device=device, dtype=imgs.dtype).view(1, 3, 1, 1)
    return imgs * std + mean


def create_noise_estimation_table(model, test_loader, config, save_path, num_samples=32, device='cuda'):
    """
    Create a comprehensive visualization table showing:
    1. Original random images
    2. Sigma values (noise levels)
    3. Noisy images (original + noise)
    4. Estimated sigma from trained model
    5. Estimation error (absolute difference)
    
    Args:
        model (torch.nn.Module): Trained noise estimation model
        test_loader (DataLoader): Test data loader
        config (dict): Configuration dictionary
        save_path (str): Path to save the visualization
        num_samples (int): Number of samples to display
        device (str): Device to run inference on
    """
    from ..data.dataset import add_noise
    from argparse import Namespace
    
    # Force CPU computation to avoid device mismatch issues
    device = 'cpu'
    
    model.eval()
    
    # Ensure model is on the correct device and in eval mode
    model = model.to(device)
    
    # Get random batch from test loader
    for images, _ in test_loader:
        if images.shape[0] >= num_samples:
            # Select random samples
            indices = torch.randperm(images.shape[0])[:num_samples]
            images = images[indices].to(device)
            break
    
    # Generate random sigma values for visualization
    sigma_min = config['noise']['sigma_min']
    sigma_max = config['noise']['sigma_max']
    sigma_values = torch.rand(num_samples, device=device) * (sigma_max - sigma_min) + sigma_min
    
    # Add noise with specific sigma values
    eps = torch.randn_like(images)
    sigma_expanded = sigma_values.view(num_samples, 1, 1, 1)
    noisy_images = images + eps * sigma_expanded
    
    # Get model predictions
    with torch.no_grad():
        sigma_estimated = model(noisy_images).squeeze()
    
    # Calculate estimation error (absolute difference)
    estimation_error = torch.abs(sigma_estimated - sigma_values)
    
    # Create the visualization using improved version of user's function
    def min_max_norm_batch_0_1(x):
        """Min-max normalization for each image in batch."""
        # Ensure tensor is on CPU for visualization
        x = x.detach().cpu()
        batch_size = x.shape[0]
        x_reshaped = x.view(batch_size, -1)
        min_vals = x_reshaped.min(dim=1, keepdim=True)[0].view(batch_size, 1, 1, 1)
        max_vals = x_reshaped.max(dim=1, keepdim=True)[0].view(batch_size, 1, 1, 1)
        return (x - min_vals) / (max_vals - min_vals + 1e-8)
    
    def show_batch_image(batchlist, ncol=4, save_path=None):
        """Enhanced version of user's visualization function."""
        count = len(batchlist)
        fig, axes = plt.subplots(1, count, figsize=(20, 25))
        fig.dpi = 300
        
        for ax, batch in zip(axes, batchlist):
            ax.set_title(batch.name, fontsize=12, fontweight='bold', pad=5)
            
            if batch.showtype == 'grid':
                # Denormalize images for proper display
                if batch.name in ['Original Images', 'Noisy Images']:
                    # Denormalize images for proper display
                    display_images = denormalize_batch(batch.value)
                    display_images = torch.clamp(display_images, 0, 1)
                else:
                    display_images = min_max_norm_batch_0_1(batch.value)
                
                # Create grid for visualization (tensors already on CPU)
                # For 32 samples in 4 columns × 8 rows layout
                grid_img_batch = make_grid(display_images, nrow=ncol, padding=2)
                ax.imshow(grid_img_batch.permute(1, 2, 0).detach().numpy())
                
            elif batch.showtype == 'table':
                # Calculate nrow based on the current batch size and ncol
                batch_size = len(batch.value)
                nrow = (batch_size + ncol - 1) // ncol  # Ceiling division (8 rows for 32 samples with 4 cols)
                
                # Pad the batch if necessary to fill the grid
                padded_size = nrow * ncol
                if batch_size < padded_size:
                    padding = torch.zeros(padded_size - batch_size, dtype=batch.value.dtype, device=batch.value.device)
                    padded_batch = torch.cat([batch.value, padding])
                else:
                    padded_batch = batch.value
                
                matbatch = padded_batch.reshape([nrow, ncol])
                im = ax.matshow(matbatch.detach().numpy(), cmap=plt.cm.Blues, aspect=0.99)
                
                # Add colorbar for table visualization
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                
                for i in range(nrow):
                    for j in range(ncol):
                        if i * ncol + j < batch_size:  # Only show values for actual data
                            ax.text(j, i, "{:.3f}".format(matbatch[i, j].item()), 
                                   va='center', ha='center', fontsize=8, fontweight='bold')
            
            ax.axis('off')
        
        # Reduce whitespace and improve layout - similar to original notebook
        plt.suptitle('Noise Estimation Visualization', fontsize=16, fontweight='bold', y=0.95)
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
            plt.close()
        else:
            plt.show()
    
    # Prepare data for visualization
    batchlist = [
        Namespace(name='Original Images', value=images, showtype='grid'),
        Namespace(name='True Sigma (σ)', value=sigma_values, showtype='table'),
        Namespace(name='Noisy Images', value=noisy_images, showtype='grid'),
        Namespace(name='Estimated Sigma (σ)', value=sigma_estimated, showtype='table'),
        Namespace(name='Estimation Error |σ_true - σ_est|', value=estimation_error, showtype='table')
    ]
    
    # Create and save visualization
    show_batch_image(batchlist, ncol=4, save_path=save_path)
    
    return {
        'original_images': images,
        'sigma_true': sigma_values,
        'noisy_images': noisy_images,
        'sigma_estimated': sigma_estimated,
        'estimation_error': estimation_error
    }


def plot_scatter_predictions(targets, predictions, save_path=None):
    """
    Create scatter plot of true vs predicted noise levels.
    
    Args:
        targets (torch.Tensor): True noise levels
        predictions (torch.Tensor): Predicted noise levels
        save_path (str): Path to save the plot
    """
    plt.figure(figsize=(8, 8))
    
    targets_np = targets.cpu().numpy()
    predictions_np = predictions.cpu().numpy()
    
    plt.scatter(targets_np, predictions_np, alpha=0.5, s=10)
    
    # Plot ideal line (y=x)
    min_val = min(targets_np.min(), predictions_np.min())
    max_val = max(targets_np.max(), predictions_np.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Ideal')
    
    plt.xlabel('True Noise Level (σ)', fontsize=12)
    plt.ylabel('Predicted Noise Level (σ)', fontsize=12)
    plt.title('Noise Level Prediction: True vs Predicted', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # Calculate and display metrics
    mse = np.mean((targets_np - predictions_np) ** 2)
    mae = np.mean(np.abs(targets_np - predictions_np))
    plt.text(
        0.05, 0.95, 
        f'MSE: {mse:.6f}\nMAE: {mae:.6f}',
        transform=plt.gca().transAxes,
        fontsize=11,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

