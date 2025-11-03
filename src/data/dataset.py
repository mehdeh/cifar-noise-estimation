"""CIFAR-10 dataset loading utilities."""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset


def get_dataloaders(config):
    """
    Create train and test dataloaders for CIFAR-10.
    
    Note: Noise is NOT added in the dataset. It will be added dynamically 
    during training to ensure different noise in each epoch.
    
    Args:
        config (dict): Configuration dictionary containing data settings
        
    Returns:
        tuple: (train_loader, test_loader, train_dataset, test_dataset)
    """
    data_config = config['data']
    
    # Define transformations for training
    train_transforms = []
    
    # Data augmentation
    if data_config['augmentation']['random_crop']:
        train_transforms.append(
            transforms.RandomCrop(
                data_config['augmentation']['crop_size'],
                padding=data_config['augmentation']['padding']
            )
        )
    
    if data_config['augmentation']['random_horizontal_flip']:
        train_transforms.append(transforms.RandomHorizontalFlip())
    
    train_transforms.append(transforms.ToTensor())
    
    # Normalization
    if data_config['normalize']:
        train_transforms.append(
            transforms.Normalize(
                mean=tuple(data_config['mean']),
                std=tuple(data_config['std'])
            )
        )
    
    transform_train = transforms.Compose(train_transforms)
    
    # Define transformations for testing (no augmentation)
    test_transforms = [transforms.ToTensor()]
    
    if data_config['normalize']:
        test_transforms.append(
            transforms.Normalize(
                mean=tuple(data_config['mean']),
                std=tuple(data_config['std'])
            )
        )
    
    transform_test = transforms.Compose(test_transforms)
    
    # Load CIFAR-10 dataset
    trainset = torchvision.datasets.CIFAR10(
        root=data_config['data_dir'],
        train=True,
        download=True,
        transform=transform_train
    )
    
    testset = torchvision.datasets.CIFAR10(
        root=data_config['data_dir'],
        train=False,
        download=True,
        transform=transform_test
    )
    
    # Apply test_samples limit if specified
    test_samples = config.get('testing', {}).get('test_samples', None)
    if test_samples is not None and test_samples > 0 and test_samples < len(testset):
        # Create a subset of the test set
        indices = list(range(test_samples))
        testset_for_loader = Subset(testset, indices)
    else:
        testset_for_loader = testset
    
    # Determine if pin_memory should be used (only when CUDA is available)
    pin_memory = torch.cuda.is_available()
    
    # Create dataloaders
    train_loader = DataLoader(
        trainset,
        batch_size=data_config['batch_size_train'],
        shuffle=True,
        num_workers=data_config['num_workers'],
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        testset_for_loader,
        batch_size=data_config['batch_size_test'],
        shuffle=False,
        num_workers=data_config['num_workers'],
        pin_memory=pin_memory
    )
    
    return train_loader, test_loader, trainset, testset


def add_noise(images, sigma_min=0.0, sigma_max=1.0):
    """
    Add Gaussian noise to images: x_noise = x + eps * sigma
    where eps ~ N(0, I) and sigma is uniformly sampled from [sigma_min, sigma_max]
    
    This function should be called during training/testing for each batch
    to ensure different noise levels in each iteration.
    
    Args:
        images (torch.Tensor): Clean images of shape (B, C, H, W)
        sigma_min (float): Minimum noise standard deviation
        sigma_max (float): Maximum noise standard deviation
        
    Returns:
        tuple: (noisy_images, sigma_values) where
            - noisy_images: Images with added noise (B, C, H, W)
            - sigma_values: Noise levels used for each image (B,)
    """
    batch_size = images.shape[0]
    device = images.device
    
    # Sample random noise levels uniformly from [sigma_min, sigma_max]
    sigma_values = torch.rand(batch_size, device=device) * (sigma_max - sigma_min) + sigma_min
    
    # Generate Gaussian noise: eps ~ N(0, I)
    eps = torch.randn_like(images)
    
    # Add noise: x_noise = x + eps * sigma
    # Reshape sigma for broadcasting: (B,) -> (B, 1, 1, 1)
    sigma_expanded = sigma_values.view(batch_size, 1, 1, 1)
    noisy_images = images + eps * sigma_expanded
    
    return noisy_images, sigma_values


def get_cifar10_classes():
    """
    Get CIFAR-10 class names.
    
    Returns:
        tuple: Class names
    """
    return (
        'plane', 'car', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    )

