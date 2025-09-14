#!/usr/bin/env python3
"""
Enhanced training script for Lagos GAN projects

This script provides an easy way to start training for both:
1. Lagos2Duplex CycleGAN (default)
2. AfroCover StyleGAN2

Usage:
    python start_training.py                    # Train Lagos2Duplex with default config
    python start_training.py --project afrocover  # Train AfroCover StyleGAN2
"""

import os
import sys
import argparse
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))

from lagos2duplex.train import main as train_main
from lagos2duplex.config import get_default_config, get_quick_test_config, get_high_quality_config


def check_environment():
    """Check if the environment is set up correctly"""
    # Check if data exists
    data_path = current_dir / "data_processed"
    if not data_path.exists():
        print("❌ Error: Processed data not found!")
        print(f"Expected data at: {data_path}")
        print("Please run the preprocessing script first:")
        print("python preprocess_dataset.py --target_size 256 --output_format png")
        return False
    
    # Check if datasets exist
    lagos_path = data_path / "afrocover" / "train"
    duplex_path = data_path / "lagos2duplex" / "duplex" / "train"
    
    if not lagos_path.exists():
        print(f"❌ Error: Lagos training data not found at {lagos_path}")
        return False
    
    if not duplex_path.exists():
        print(f"❌ Error: Duplex training data not found at {duplex_path}")
        return False
    
    # Count images
    lagos_count = len(list(lagos_path.glob("*.png"))) + len(list(lagos_path.glob("*.jpg")))
    duplex_count = len(list(duplex_path.glob("*.png"))) + len(list(duplex_path.glob("*.jpg")))
    
    print(f"✅ Found {lagos_count} Lagos images and {duplex_count} duplex images")
    
    if lagos_count == 0 or duplex_count == 0:
        print("❌ Error: No training images found!")
        return False
    
    return True


def check_afrocover_environment():
    """Check if AfroCover environment is set up correctly"""
    data_path = current_dir / "data_processed" / "afrocover"
    if not data_path.exists():
        print("❌ Error: AfroCover data not found!")
        return False
    
    total_count = 0
    for split in ['train', 'val', 'test']:
        split_path = data_path / split
        if split_path.exists():
            count = len(list(split_path.glob("*.png"))) + len(list(split_path.glob("*.jpg")))
            total_count += count
    
    print(f"✅ Found {total_count} AfroCover images")
    return total_count > 0


def main():
    parser = argparse.ArgumentParser(description="Start Lagos GAN Training")
    parser.add_argument('--project', choices=['lagos2duplex', 'afrocover'], 
                       default='lagos2duplex',
                       help='Which project to train (default: lagos2duplex)')
    parser.add_argument('--config', choices=['quick', 'default', 'high_quality'], 
                       default='default',
                       help='Training configuration')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    parser.add_argument('--no_wandb', action='store_true', help='Disable Weights & Biases logging')
    parser.add_argument('--gpu', type=int, help='GPU ID to use (0, 1, etc.)')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    print(f"� Lagos GAN Training - {args.project.upper()}")
    print("=" * 50)
    
    # Check environment based on project
    if args.project == 'lagos2duplex':
        if not check_environment():
            return 1
        print("🏠 Training Lagos2Duplex CycleGAN...")
        return train_lagos2duplex(args)
    elif args.project == 'afrocover':
        if not check_afrocover_environment():
            return 1
        print("🎵 Training AfroCover StyleGAN2...")
        return train_afrocover(args)


def train_lagos2duplex(args):
    """Train Lagos2Duplex CycleGAN"""
    # Get configuration
    if args.config == 'quick':
        config = get_quick_test_config()
    elif args.config == 'high_quality':
        config = get_high_quality_config()
    else:
        config = get_default_config()
    
    # Override with command line arguments
    if args.epochs:
        config.training.num_epochs = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.learning_rate:
        config.training.g_lr = args.learning_rate
        config.training.d_lr = args.learning_rate
    
    # Set paths
    config.data.data_path = str(current_dir / "data_processed")
    config.training.output_dir = str(current_dir / "checkpoints" / "lagos2duplex")
    
    if args.no_wandb:
        config.logging.use_wandb = False
    
    print(f"� Configuration: {config.training.num_epochs} epochs, batch size {config.training.batch_size}")
    
    # Create output directory
    Path(config.training.output_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        train_main(config)
        print("\n🎉 Lagos2Duplex training completed!")
        return 0
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        return 1


def train_afrocover(args):
    """Train AfroCover StyleGAN2"""
    try:
        from afrocover.train import main as afrocover_main
        import argparse as afro_argparse
        
        # Create arguments for afrocover
        afro_args = afro_argparse.Namespace()
        afro_args.data_path = str(current_dir / "data_processed" / "afrocover")
        afro_args.output_dir = str(current_dir / "checkpoints" / "afrocover")
        afro_args.batch_size = args.batch_size or 4
        afro_args.num_epochs = args.epochs or 100
        afro_args.learning_rate = args.learning_rate or 0.002
        afro_args.image_size = 256
        afro_args.z_dim = 512
        afro_args.resume = args.resume
        afro_args.save_freq = 10
        afro_args.sample_freq = 5
        
        print(f"� Configuration: {afro_args.num_epochs} epochs, batch size {afro_args.batch_size}")
        
        # Create output directory
        Path(afro_args.output_dir).mkdir(parents=True, exist_ok=True)
        
        afrocover_main(afro_args)
        print("\n🎉 AfroCover training completed!")
        return 0
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        return 1
        sys.argv = original_argv


if __name__ == "__main__":
    exit(main())
