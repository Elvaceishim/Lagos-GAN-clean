"""
Lagos2Duplex Dataset Module

Handles loading and preprocessing of Lagos house and duplex images for CycleGAN training.
"""

import os
import json
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random
import numpy as np


class Lagos2DuplexDataset(Dataset):
    """Dataset for Lagos house to duplex transformation using CycleGAN"""
    
    def __init__(self, data_path, image_size=256, augmentations=True, split='train'):
        """
        Args:
            data_path (str): Path to dataset directory
            image_size (int): Target image size for training
            augmentations (bool): Whether to apply data augmentations
            split (str): Dataset split ('train', 'val', 'test')
        """
        self.data_path = Path(data_path)
        self.image_size = image_size
        self.split = split
        
        # Load image paths for both domains
        self.lagos_paths = self._load_domain_paths('lagos')  # Domain A
        self.duplex_paths = self._load_domain_paths('duplex')  # Domain B
        
        # For CycleGAN, we need equal access to both domains
        self.max_size = max(len(self.lagos_paths), len(self.duplex_paths))
        
        # Load metadata
        self.metadata = self._load_metadata()
        
        # Setup transforms
        self.transforms = self._setup_transforms(augmentations)
        
        print(f"Loaded {len(self.lagos_paths)} Lagos houses and {len(self.duplex_paths)} duplexes for {split} split")
        print(f"Dataset size (max): {self.max_size}")
    
    def _load_domain_paths(self, domain):
        """Load image paths for a specific domain (lagos or duplex)"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_paths = []
        
        # For our processed dataset structure: data_processed/{domain_dataset}/{domain}/{split}/
        if domain == 'lagos':
            # Load Lagos images from afrocover dataset
            domain_dir = self.data_path / 'afrocover' / self.split
        elif domain == 'duplex':
            # Load duplex images from lagos2duplex dataset
            domain_dir = self.data_path / 'lagos2duplex' / 'duplex' / self.split
        else:
            # Fallback for other domain names
            domain_dir = self.data_path / domain / self.split
        
        if domain_dir.exists():
            for ext in image_extensions:
                image_paths.extend(domain_dir.glob(f'*{ext}'))
                image_paths.extend(domain_dir.glob(f'*{ext.upper()}'))
        else:
            print(f"Warning: Domain directory {domain_dir} not found")
        
        return sorted(image_paths)
    
    def _load_metadata(self):
        """Load dataset metadata including licensing information"""
        metadata_path = self.data_path / 'metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _setup_transforms(self, augmentations=True):
        """Setup image preprocessing and augmentation pipeline"""
        if augmentations and self.split == 'train':
            # Training transforms with augmentations
            transform = A.Compose([
                A.Resize(self.image_size + 30, self.image_size + 30),  # Slightly larger for random crop
                A.RandomCrop(self.image_size, self.image_size),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(
                    brightness=0.1,
                    contrast=0.1,
                    saturation=0.1,
                    hue=0.05,
                    p=0.3
                ),
                A.RandomRotate90(p=0.1),
                A.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5]
                ),
                ToTensorV2()
            ])
        else:
            # Validation/test transforms without augmentations
            transform = A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5]
                ),
                ToTensorV2()
            ])
        
        return transform
    
    def _load_image(self, image_path):
        """Load and preprocess a single image"""
        try:
            image = Image.open(image_path).convert('RGB')
            image_array = np.array(image)
            transformed = self.transforms(image=image_array)
            return transformed['image']
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a black image as fallback
            return torch.zeros(3, self.image_size, self.image_size)
    
    def __len__(self):
        return self.max_size
    
    def __getitem__(self, idx):
        """Get a pair of images from both domains"""
        # For CycleGAN, we need unpaired data
        # Randomly sample from each domain
        lagos_idx = idx % len(self.lagos_paths)
        duplex_idx = random.randint(0, len(self.duplex_paths) - 1)
        
        # Load images from both domains
        lagos_image = self._load_image(self.lagos_paths[lagos_idx])
        duplex_image = self._load_image(self.duplex_paths[duplex_idx])
        
        # Get metadata
        lagos_name = self.lagos_paths[lagos_idx].name
        duplex_name = self.duplex_paths[duplex_idx].name
        
        return {
            'A': lagos_image,           # Domain A (Lagos houses)
            'B': duplex_image,          # Domain B (Modern duplexes)
            'A_path': str(self.lagos_paths[lagos_idx]),
            'B_path': str(self.duplex_paths[duplex_idx]),
            'A_metadata': self.metadata.get(lagos_name, {}),
            'B_metadata': self.metadata.get(duplex_name, {})
        }
    
    def get_sample_pairs(self, num_samples=4):
        """Get sample pairs for visualization"""
        indices = torch.randperm(len(self))[:num_samples]
        samples_A = []
        samples_B = []
        
        for idx in indices:
            item = self[idx]
            samples_A.append(item['A'])
            samples_B.append(item['B'])
        
        return torch.stack(samples_A), torch.stack(samples_B)


class PairedLagos2DuplexDataset(Dataset):
    """Alternative dataset for paired Lagos-to-duplex data (if available)"""
    
    def __init__(self, data_path, image_size=256, augmentations=True, split='train'):
        """
        For cases where we have paired before/after images of the same properties
        """
        self.data_path = Path(data_path)
        self.image_size = image_size
        self.split = split
        
        # Load paired image paths
        self.paired_paths = self._load_paired_paths()
        self.transforms = self._setup_transforms(augmentations)
        
        print(f"Loaded {len(self.paired_paths)} paired images for {split} split")
    
    def _load_paired_paths(self):
        """Load paths for paired images"""
        # Expected structure: data_path/pairs/{split}/
        # Each pair should have matching filenames in lagos/ and duplex/ subdirs
        pairs_dir = self.data_path / 'pairs' / self.split
        paired_paths = []
        
        if pairs_dir.exists():
            lagos_dir = pairs_dir / 'lagos'
            duplex_dir = pairs_dir / 'duplex'
            
            if lagos_dir.exists() and duplex_dir.exists():
                for lagos_file in lagos_dir.glob('*'):
                    if lagos_file.is_file():
                        # Look for matching duplex file
                        duplex_file = duplex_dir / lagos_file.name
                        if duplex_file.exists():
                            paired_paths.append((lagos_file, duplex_file))
        
        return paired_paths
    
    def _setup_transforms(self, augmentations=True):
        """Setup transforms for paired images (same transform applied to both)"""
        if augmentations and self.split == 'train':
            return A.Compose([
                A.Resize(self.image_size + 30, self.image_size + 30),
                A.RandomCrop(self.image_size, self.image_size),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.3),
                A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ToTensorV2()
            ], additional_targets={'image_B': 'image'})
        else:
            return A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ToTensorV2()
            ], additional_targets={'image_B': 'image'})
    
    def __len__(self):
        return len(self.paired_paths)
    
    def __getitem__(self, idx):
        lagos_path, duplex_path = self.paired_paths[idx]
        
        # Load both images
        lagos_image = Image.open(lagos_path).convert('RGB')
        duplex_image = Image.open(duplex_path).convert('RGB')
        
        # Apply same transform to both images
        transformed = self.transforms(
            image=np.array(lagos_image),
            image_B=np.array(duplex_image)
        )
        
        return {
            'A': transformed['image'],
            'B': transformed['image_B'],
            'A_path': str(lagos_path),
            'B_path': str(duplex_path)
        }


def create_dataloaders(data_path, batch_size=1, image_size=256, num_workers=4, paired=False):
    """Create train and validation dataloaders"""
    
    DatasetClass = PairedLagos2DuplexDataset if paired else Lagos2DuplexDataset
    
    # Create datasets
    train_dataset = DatasetClass(
        data_path=data_path,
        image_size=image_size,
        augmentations=True,
        split='train'
    )
    
    val_dataset = DatasetClass(
        data_path=data_path,
        image_size=image_size,
        augmentations=False,
        split='val'
    )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test the dataset
    import numpy as np
    
    # Create a test dataset (modify path as needed)
    dataset = Lagos2DuplexDataset(
        data_path="./data/lagos2duplex",
        image_size=256,
        augmentations=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        # Test loading a sample
        sample = dataset[0]
        print(f"Lagos image shape: {sample['A'].shape}")
        print(f"Duplex image shape: {sample['B'].shape}")
        print(f"Lagos path: {sample['A_path']}")
        print(f"Duplex path: {sample['B_path']}")
        
        # Test dataloader
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)
        batch = next(iter(dataloader))
        print(f"Batch A shape: {batch['A'].shape}")
        print(f"Batch B shape: {batch['B'].shape}")
    else:
        print("No images found in dataset directory")
