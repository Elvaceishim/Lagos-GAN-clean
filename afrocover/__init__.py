"""
AfroCover Module - StyleGAN2 for African Album Cover Generation

This module contains the complete pipeline for generating African-inspired album covers
using StyleGAN2-ADA architecture.
"""

from .train import main as train_afrocover
from .models import StyleGAN2Generator, StyleGAN2Discriminator
from .dataset import AfrocoverDataset, create_dataloaders

__all__ = [
    'train_afrocover',
    'StyleGAN2Generator', 
    'StyleGAN2Discriminator',
    'AfrocoverDataset',
    'create_dataloaders'
]
