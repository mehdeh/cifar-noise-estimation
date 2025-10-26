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
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
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
    mean = torch.tensor(mean, device=imgs.device).view(1, 3, 1, 1)
    std = torch.tensor(std, device=imgs.device).view(1, 3, 1, 1)
    return imgs * std + mean


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

