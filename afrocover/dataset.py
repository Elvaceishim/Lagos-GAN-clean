"""
AfroCover Dataset Module

Handles loading and preprocessing of African-inspired album cover dataset.
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
import numpy as np


class AfrocoverDataset(Dataset):
    """Dataset for African-inspired album covers"""

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

        # Load image paths and metadata
        self.image_paths = self._load_image_paths()
        self.metadata = self._load_metadata()

        # Setup transforms
        self.transforms = self._setup_transforms(augmentations)

        print(f"Loaded {len(self.image_paths)} images for {split} split")

    def _load_image_paths(self):
        """Load all image paths from the dataset directory"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_paths = []

        # Look for images in data_path/{split}/ directory
        split_dir = self.data_path / self.split
        if split_dir.exists():
            for ext in image_extensions:
                image_paths.extend(split_dir.glob(f'*{ext}'))
                image_paths.extend(split_dir.glob(f'*{ext.upper()}'))
        else:
            # Fallback: look in data_path/ directly
            for ext in image_extensions:
                image_paths.extend(self.data_path.glob(f'*{ext}'))
                image_paths.extend(self.data_path.glob(f'*{ext.upper()}'))

        return sorted(image_paths)

    def _load_metadata(self):
        """Load dataset metadata including licensing information"""
        metadata_path = self.data_path / 'metadata.json'
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _setup_transforms(self, augmentations=True):
        """Setup image preprocessing and augmentation pipeline"""
        if augmentations and self.split == 'train':
            # Training transforms with augmentations
            transform = A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.2),
                A.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1,
                    p=0.5
                ),
                A.RandomCrop(
                    height=int(self.image_size * 0.9),
                    width=int(self.image_size * 0.9),
                    p=0.3
                ),
                A.Resize(self.image_size, self.image_size),
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

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """Get a single sample from the dataset"""
        img_path = self.image_paths[idx]

        try:
            # Load image
            image = Image.open(img_path).convert('RGB')

            # Apply transforms
            image_array = np.array(image)
            transformed = self.transforms(image=image_array)
            image_tensor = transformed['image']

            # Prepare metadata
            metadata = self.metadata.get(img_path.name, {})

            return {
                'image': image_tensor,
                'metadata': metadata,
                'path': str(img_path)
            }

        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            fallback = torch.zeros(3, self.image_size, self.image_size)
            return {
                'image': fallback,
                'metadata': {'filename': 'error', 'path': str(img_path), 'index': idx},
                'path': str(img_path)
            }

    def get_sample_images(self, num_samples=8):
        """Get a batch of sample images for visualization"""
        sample_indices = torch.randperm(len(self))[:num_samples]
        samples = []

        for idx in sample_indices:
            sample = self[idx.item()]
            samples.append(sample['image'])

        return torch.stack(samples)


def create_dataloaders(data_path, batch_size=16, image_size=256, num_workers=4):
    """Create data loaders for training, validation, and testing

    Returns:
        dataloaders (dict): {'train': loader, 'val': loader, 'test': loader}
        datasets (dict): dataset objects for each available split
    """

    datasets = {}
    dataloaders = {}

    for split in ['train', 'val', 'test']:
        split_path = Path(data_path) / split
        if split_path.exists() and any(split_path.iterdir()):
            ds = AfrocoverDataset(
                data_path=data_path,
                image_size=image_size,
                augmentations=(split == 'train'),
                split=split
            )
            datasets[split] = ds
            dataloaders[split] = torch.utils.data.DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=(split == 'train'),
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
                drop_last=(split == 'train')
            )

    return dataloaders, datasets


def analyze_dataset(data_path):
    """Analyze dataset statistics and distribution"""
    stats = {}

    for split in ['train', 'val', 'test']:
        split_path = Path(data_path) / split
        if split_path.exists():
            image_files = []
            for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                image_files.extend(split_path.glob(f'*{ext}'))
                image_files.extend(split_path.glob(f'*{ext.upper()}'))

            stats[split] = {
                'count': len(image_files),
                'files': [f.name for f in image_files[:10]]  # example file names
            }

    return stats


if __name__ == "__main__":
    # Quick test/harness
    data_dir = "./data/afrocover"
    print("Analyzing dataset:", data_dir)
    stats = analyze_dataset(data_dir)
    print(stats)

    dataloaders, datasets = create_dataloaders(data_dir, batch_size=2, image_size=256)
    if 'train' in dataloaders:
        loader = dataloaders['train']
        batch = next(iter(loader))
        print("Batch keys:", batch.keys())
        print("Image batch shape:", batch['image'].shape)
    else:
        print("No train split detected")
