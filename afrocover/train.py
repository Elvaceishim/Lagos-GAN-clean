"""
AfroCover StyleGAN2 Training Script

This module handles training of StyleGAN2-ADA for generating African-inspired album covers.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import argparse
import os
import time
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm

from .models import StyleGAN2Generator, StyleGAN2Discriminator
from .dataset import AfrocoverDataset, create_dataloaders, analyze_dataset


def parse_args():
    parser = argparse.ArgumentParser(description='Train AfroCover StyleGAN2')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to AfroCover dataset')
    parser.add_argument('--output_dir', type=str, default='./checkpoints/afrocover',
                        help='Output directory for checkpoints')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.002,
                        help='Learning rate')
    parser.add_argument('--image_size', type=int, default=256,
                        help='Image resolution')
    parser.add_argument('--z_dim', type=int, default=512,
                        help='Latent dimension')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='Save checkpoint frequency (epochs)')
    parser.add_argument('--sample_freq', type=int, default=5,
                        help='Generate samples frequency (epochs)')
    parser.add_argument('--channel_multiplier', type=float, default=1.0,
                        help='Scales the number of feature channels (use <1.0 to trim the network)')
    
    return parser.parse_args()


def setup_training(args):
    """Setup training environment and directories"""
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(f"{args.output_dir}/samples", exist_ok=True)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    return device


def load_dataset(args):
    """Load and prepare the AfroCover dataset"""
    print(f"Loading dataset from: {args.data_path}")
    
    # Analyze dataset
    stats = analyze_dataset(args.data_path)
    print("Dataset statistics:")
    for split, info in stats.items():
        print(f"  {split}: {info['count']} images")
    
    # Create dataloaders
    dataloaders, datasets = create_dataloaders(
        data_path=args.data_path,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=4
    )
    
    return dataloaders, datasets


def create_models(args, device):
    """Create StyleGAN2 generator and discriminator"""
    print("Creating StyleGAN2 models...")
    
    generator = StyleGAN2Generator(
        z_dim=args.z_dim,
        w_dim=args.z_dim,
        img_resolution=args.image_size,
        img_channels=3,
        channel_multiplier=args.channel_multiplier,
    ).to(device)
    
    discriminator = StyleGAN2Discriminator(
        img_resolution=args.image_size,
        img_channels=3,
        channel_multiplier=args.channel_multiplier,
    ).to(device)
    
    return generator, discriminator


def train_epoch(generator, discriminator, dataloader, optimizers, device, epoch, args):
    """Train for one epoch"""
    generator.train()
    discriminator.train()
    
    g_optimizer, d_optimizer = optimizers
    criterion = nn.BCEWithLogitsLoss()
    
    g_losses = []
    d_losses = []
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for batch_idx, batch in enumerate(pbar):
        real_images = batch['image'].to(device)
        batch_size = real_images.size(0)
        
        # Real and fake labels
        real_labels = torch.ones(batch_size, 1, device=device)
        fake_labels = torch.zeros(batch_size, 1, device=device)
        
        # Train Discriminator
        d_optimizer.zero_grad()
        
        # Real images
        d_real = discriminator(real_images)
        d_loss_real = criterion(d_real, real_labels)
        
        # Fake images
        z = torch.randn(batch_size, args.z_dim, device=device)
        fake_images = generator(z)
        d_fake = discriminator(fake_images.detach())
        d_loss_fake = criterion(d_fake, fake_labels)
        
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        d_optimizer.step()
        
        # Train Generator
        g_optimizer.zero_grad()
        
        d_fake = discriminator(fake_images)
        g_loss = criterion(d_fake, real_labels)
        g_loss.backward()
        g_optimizer.step()
        
        g_losses.append(g_loss.item())
        d_losses.append(d_loss.item())
        
        # Update progress bar
        pbar.set_postfix({
            'G_loss': f'{g_loss.item():.4f}',
            'D_loss': f'{d_loss.item():.4f}'
        })
    
    return np.mean(g_losses), np.mean(d_losses)


def generate_samples(generator, device, args, epoch, num_samples=16):
    """Generate sample images"""
    generator.eval()
    
    with torch.no_grad():
        z = torch.randn(num_samples, args.z_dim, device=device)
        fake_images = generator(z)
        
        # Denormalize images
        fake_images = (fake_images + 1) / 2
        fake_images = torch.clamp(fake_images, 0, 1)
        
        # Save grid
        sample_path = f"{args.output_dir}/samples/epoch_{epoch:04d}.png"
        vutils.save_image(fake_images, sample_path, nrow=4, normalize=False)
        print(f"Saved samples to {sample_path}")


def save_checkpoint(generator, discriminator, optimizers, epoch, args, g_loss, d_loss):
    """Save model checkpoint"""
    g_optimizer, d_optimizer = optimizers
    
    checkpoint = {
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'g_optimizer_state_dict': g_optimizer.state_dict(),
        'd_optimizer_state_dict': d_optimizer.state_dict(),
        'g_loss': g_loss,
        'd_loss': d_loss,
        'args': vars(args)
    }
    
    checkpoint_path = f"{args.output_dir}/checkpoint_epoch_{epoch:04d}.pt"
    torch.save(checkpoint, checkpoint_path)
    
    # Also save as latest
    latest_path = f"{args.output_dir}/latest_checkpoint.pt"
    torch.save(checkpoint, latest_path)
    
    print(f"Saved checkpoint to {checkpoint_path}")


def load_checkpoint(checkpoint_path, generator, discriminator, optimizers):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path)
    
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    
    if optimizers:
        g_optimizer, d_optimizer = optimizers
        g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
    
    return checkpoint['epoch'], checkpoint['g_loss'], checkpoint['d_loss']


def main(args=None):
    """Main training function"""
    if args is None:
        args = parse_args()
    
    print("Starting AfroCover StyleGAN2 Training")
    print(f"Configuration: {vars(args)}")
    
    # Setup
    device = setup_training(args)
    
    # Load dataset
    dataloaders, datasets = load_dataset(args)
    
    if 'train' not in dataloaders:
        print("Error: Training data not found!")
        return
    
    train_dataloader = dataloaders['train']
    print(f"Training on {len(train_dataloader.dataset)} images")
    
    # Create models
    generator, discriminator = create_models(args, device)
    
    # Setup optimizers
    g_optimizer = optim.Adam(generator.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
    optimizers = (g_optimizer, d_optimizer)
    
    # Load checkpoint if resuming
    start_epoch = 0
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        start_epoch, _, _ = load_checkpoint(args.resume, generator, discriminator, optimizers)
        start_epoch += 1
    
    # Training loop
    print("Starting training...")
    best_g_loss = float('inf')
    
    for epoch in range(start_epoch, args.num_epochs):
        epoch_start_time = time.time()
        
        # Train for one epoch
        g_loss, d_loss = train_epoch(
            generator, discriminator, train_dataloader, 
            optimizers, device, epoch, args
        )
        
        epoch_time = time.time() - epoch_start_time
        
        print(f"Epoch {epoch}/{args.num_epochs} - "
              f"G_loss: {g_loss:.4f}, D_loss: {d_loss:.4f} - "
              f"Time: {epoch_time:.2f}s")
        
        # Generate samples
        if epoch % args.sample_freq == 0:
            generate_samples(generator, device, args, epoch)
        
        # Save checkpoint
        if epoch % args.save_freq == 0 or g_loss < best_g_loss:
            save_checkpoint(generator, discriminator, optimizers, epoch, args, g_loss, d_loss)
            if g_loss < best_g_loss:
                best_g_loss = g_loss
                # Save best model
                best_path = f"{args.output_dir}/best_model.pt"
                torch.save({
                    'generator_state_dict': generator.state_dict(),
                    'epoch': epoch,
                    'g_loss': g_loss
                }, best_path)
    
    print("Training completed!")


if __name__ == "__main__":
    main()
