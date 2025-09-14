"""
Lagos2Duplex Module - CycleGAN for House Transformation

This module contains the complete pipeline for transforming old Lagos houses
into modern duplex designs using CycleGAN architecture.
"""

from .train import main as train_lagos2duplex
from .models import CycleGANGenerator, CycleGANDiscriminator, ImagePool, define_G, define_D
from .dataset import Lagos2DuplexDataset, PairedLagos2DuplexDataset, create_dataloaders
from .losses import CycleGANLoss, GANLoss, CycleLoss, IdentityLoss
from .config import CycleGANConfig, get_default_config, get_quick_test_config

__version__ = "1.0.0"

__all__ = [
    'train_lagos2duplex',
    'CycleGANGenerator',
    'CycleGANDiscriminator', 
    'ImagePool',
    'define_G',
    'define_D',
    'Lagos2DuplexDataset',
    'PairedLagos2DuplexDataset',
    'create_dataloaders',
    'CycleGANLoss',
    'GANLoss',
    'CycleLoss',
    'IdentityLoss',
    'CycleGANConfig',
    'get_default_config',
    'get_quick_test_config'
]
