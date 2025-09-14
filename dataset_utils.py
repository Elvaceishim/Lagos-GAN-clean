#!/usr/bin/env python3
"""
Lagos-GAN PyTorch Dataset Classes
Optimized dataset classes for GAN training with proper normalization and augmentation.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from pathlib import Path
from PIL import Image
import random
import numpy as np
from typing import Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)

class LagosGANDataset(Dataset):
    """
    Base dataset class for Lagos-GAN with proper preprocessing and augmentation.
    Normalizes images to [-1, 1] range as expected by GANs.
    """
    
    def __init__(
        self, 
        data_path: str,
        image_size: int = 256,
        split: str = 'train',
        augment: bool = True,
        normalize_range: str = 'tanh'  # 'tanh' for [-1,1] or 'sigmoid' for [0,1]
    ):
        """
        Initialize dataset.
        
        Args:
            data_path (str): Path to dataset directory
            image_size (int): Target image size (256 or 512)
            split (str): Dataset split ('train', 'val', 'test')
            augment (bool): Whether to apply data augmentation
            normalize_range (str): Normalization range ('tanh' or 'sigmoid')
        """
        self.data_path = Path(data_path)
        self.image_size = image_size
        self.split = split
        self.augment = augment and (split == 'train')
        self.normalize_range = normalize_range
        
        # Find all image files
        self.image_files = self._find_image_files()
        
        # Setup transforms
        self.transform = self._setup_transforms()
        
        logger.info(f"Loaded {len(self.image_files)} images from {self.data_path / split}")
    
    def _find_image_files(self) -> List[Path]:
        """Find all image files in the dataset directory."""
        split_dir = self.data_path / self.split
        
        if not split_dir.exists():
            raise ValueError(f"Split directory not found: {split_dir}")
        
        extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = [
            f for f in split_dir.rglob('*')
            if f.is_file() and f.suffix.lower() in extensions
        ]
        
        if not image_files:
            raise ValueError(f"No image files found in {split_dir}")
        
        return sorted(image_files)
    
    def _setup_transforms(self) -> transforms.Compose:
        """Setup image transforms based on split and settings."""
        transform_list = []
        
        # Resize
        transform_list.append(
            transforms.Resize((self.image_size, self.image_size), 
                            transforms.InterpolationMode.LANCZOS)
        )
        
        # Data augmentation for training
        if self.augment:
            transform_list.extend([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05))
            ])
        
        # Convert to tensor
        transform_list.append(transforms.ToTensor())
        
        # Normalize
        if self.normalize_range == 'tanh':
            # Normalize to [-1, 1] for GAN training
            transform_list.append(
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            )
        else:
            # Keep in [0, 1] range
            pass
        
        return transforms.Compose(transform_list)
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get a single image tensor."""
        image_path = self.image_files[idx]
        
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Apply transforms
            image_tensor = self.transform(image)
            
            return image_tensor
            
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            # Return a blank image if loading fails
            blank = torch.zeros(3, self.image_size, self.image_size)
            if self.normalize_range == 'tanh':
                blank = blank * 2 - 1  # Convert to [-1, 1]
            return blank


class Lagos2DuplexDataset(Dataset):
    """
    Paired dataset for Lagos2Duplex CycleGAN training.
    Loads traditional Lagos houses and modern duplex houses.
    """
    
    def __init__(
        self,
        data_path: str,
        image_size: int = 256,
        split: str = 'train',
        augment: bool = True
    ):
        """
        Initialize paired dataset for CycleGAN.
        
        Args:
            data_path (str): Root path to lagos2duplex dataset
            image_size (int): Target image size
            split (str): Dataset split
            augment (bool): Whether to apply augmentation
        """
        self.data_path = Path(data_path)
        self.image_size = image_size
        self.split = split
        self.augment = augment and (split == 'train')
        
        # Setup datasets for both domains
        self.lagos_dataset = LagosGANDataset(
            data_path=str(self.data_path / 'lagos'),
            image_size=image_size,
            split=split,
            augment=augment,
            normalize_range='tanh'
        )
        
        self.duplex_dataset = LagosGANDataset(
            data_path=str(self.data_path / 'duplex'),
            image_size=image_size,
            split=split,
            augment=augment,
            normalize_range='tanh'
        )
        
        # Use the larger dataset size for unpaired training
        self.length = max(len(self.lagos_dataset), len(self.duplex_dataset))
        
        logger.info(f"Lagos2Duplex Dataset - Lagos: {len(self.lagos_dataset)}, "
                   f"Duplex: {len(self.duplex_dataset)}, Total pairs: {self.length}")
    
    def __len__(self) -> int:
        return self.length
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a pair of images (lagos, duplex).
        
        Returns:
            Tuple of (lagos_image, duplex_image) tensors
        """
        # Get Lagos image (cycle through if needed)
        lagos_idx = idx % len(self.lagos_dataset)
        lagos_image = self.lagos_dataset[lagos_idx]
        
        # Get Duplex image (cycle through if needed)
        duplex_idx = idx % len(self.duplex_dataset)
        duplex_image = self.duplex_dataset[duplex_idx]
        
        return lagos_image, duplex_image


class AfroCoverDataset(LagosGANDataset):
    """Specialized dataset for AfroCover album generation."""
    
    def __init__(self, data_path: str, **kwargs):
        # Set default path for afrocover
        if not data_path.endswith('afrocover'):
            data_path = str(Path(data_path) / 'afrocover')
        super().__init__(data_path, **kwargs)


def create_dataloaders(
    data_path: str,
    dataset_type: str = 'lagos2duplex',
    batch_size: int = 4,
    image_size: int = 256,
    num_workers: int = 4,
    augment: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        data_path (str): Path to dataset root
        dataset_type (str): Type of dataset ('lagos2duplex', 'afrocover')
        batch_size (int): Batch size
        image_size (int): Image resolution
        num_workers (int): Number of worker processes
        augment (bool): Whether to apply augmentation
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    if dataset_type == 'lagos2duplex':
        # Create CycleGAN datasets
        train_dataset = Lagos2DuplexDataset(
            data_path=data_path,
            image_size=image_size,
            split='train',
            augment=augment
        )
        
        val_dataset = Lagos2DuplexDataset(
            data_path=data_path,
            image_size=image_size,
            split='val',
            augment=False
        )
        
        test_dataset = Lagos2DuplexDataset(
            data_path=data_path,
            image_size=image_size,
            split='test',
            augment=False
        )
        
    elif dataset_type == 'afrocover':
        # Create single-domain datasets
        train_dataset = AfroCoverDataset(
            data_path=data_path,
            image_size=image_size,
            split='train',
            augment=augment
        )
        
        val_dataset = AfroCoverDataset(
            data_path=data_path,
            image_size=image_size,
            split='val',
            augment=False
        )
        
        test_dataset = AfroCoverDataset(
            data_path=data_path,
            image_size=image_size,
            split='test',
            augment=False
        )
    
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader, test_loader


def denormalize_tensor(tensor: torch.Tensor, range_type: str = 'tanh') -> torch.Tensor:
    """
    Denormalize tensor for visualization.
    
    Args:
        tensor: Input tensor in [-1, 1] or [0, 1] range
        range_type: 'tanh' for [-1, 1] or 'sigmoid' for [0, 1]
    
    Returns:
        Tensor in [0, 1] range suitable for visualization
    """
    if range_type == 'tanh':
        # Convert from [-1, 1] to [0, 1]
        return (tensor + 1) / 2
    else:
        # Already in [0, 1]
        return tensor


if __name__ == "__main__":
    # Test the datasets
    import matplotlib.pyplot as plt
    
    print("Testing Lagos2Duplex dataset...")
    
    try:
        # Test dataset loading
        dataset = Lagos2DuplexDataset(
            data_path="data/lagos2duplex",
            image_size=256,
            split='train'
        )
        
        print(f"Dataset length: {len(dataset)}")
        
        # Test single item
        lagos_img, duplex_img = dataset[0]
        print(f"Lagos image shape: {lagos_img.shape}, range: [{lagos_img.min():.3f}, {lagos_img.max():.3f}]")
        print(f"Duplex image shape: {duplex_img.shape}, range: [{duplex_img.min():.3f}, {duplex_img.max():.3f}]")
        
        # Test dataloader
        train_loader, val_loader, test_loader = create_dataloaders(
            data_path="data/lagos2duplex",
            dataset_type="lagos2duplex",
            batch_size=2
        )
        
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        print(f"Test batches: {len(test_loader)}")
        
        print("✅ Dataset testing successful!")
        
    except Exception as e:
        print(f"❌ Dataset testing failed: {e}")
