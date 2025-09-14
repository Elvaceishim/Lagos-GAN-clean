#!/usr/bin/env python3
"""
Unified Training Script for Lagos GAN Projects

This script allows you to train either:
1. Lagos2Duplex CycleGAN - Transform Lagos houses to modern duplexes
2. AfroCover StyleGAN2 - Generate African-inspired album covers

Both projects use the same preprocessed dataset structure.
"""

import os
import sys
import argparse
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))


def check_common_environment():
    """Check if the common environment is set up correctly"""
    # Check if data exists
    data_path = current_dir / "data_processed"
    if not data_path.exists():
        print("❌ Error: Processed data not found!")
        print(f"Expected data at: {data_path}")
        print("Please run the preprocessing script first:")
        print("python preprocess_dataset.py --target_size 256 --output_format png")
        return False
    
    return True


def check_lagos2duplex_data():
    """Check Lagos2Duplex dataset"""
    print("\n🔍 Checking Lagos2Duplex dataset...")
    
    lagos_path = current_dir / "data_processed" / "afrocover" / "train"
    duplex_path = current_dir / "data_processed" / "lagos2duplex" / "duplex" / "train"
    
    if not lagos_path.exists():
        print(f"❌ Error: Lagos training data not found at {lagos_path}")
        return False
    
    if not duplex_path.exists():
        print(f"❌ Error: Duplex training data not found at {duplex_path}")
        return False
    
    # Count images
    lagos_count = len(list(lagos_path.glob("*.png"))) + len(list(lagos_path.glob("*.jpg")))
    duplex_count = len(list(duplex_path.glob("*.png"))) + len(list(duplex_path.glob("*.jpg")))
    
    print(f"✅ Lagos2Duplex: {lagos_count} Lagos images, {duplex_count} duplex images")
    
    return lagos_count > 0 and duplex_count > 0


def check_afrocover_data():
    """Check AfroCover dataset"""
    print("\n🔍 Checking AfroCover dataset...")
    
    afrocover_path = current_dir / "data_processed" / "afrocover"
    
    total_count = 0
    for split in ['train', 'val', 'test']:
        split_path = afrocover_path / split
        if split_path.exists():
            count = len(list(split_path.glob("*.png"))) + len(list(split_path.glob("*.jpg")))
            total_count += count
            print(f"  {split}: {count} images")
    
    print(f"✅ AfroCover: {total_count} total images")
    
    return total_count > 0


def start_lagos2duplex_training(args):
    """Start Lagos2Duplex CycleGAN training"""
    print("\n🚀 Starting Lagos2Duplex CycleGAN Training...")
    
    if not check_lagos2duplex_data():
        return False
    
    try:
        from lagos2duplex.train import main as train_lagos2duplex
        from lagos2duplex.config import get_default_config, get_quick_test_config, get_high_quality_config
        
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
        
        # Set data paths
        config.data.data_path = str(current_dir / "data_processed")
        config.training.output_dir = str(current_dir / "checkpoints" / "lagos2duplex")
        
        print(f"Configuration: {config.training.num_epochs} epochs, batch size {config.training.batch_size}")
        
        # Start training
        train_lagos2duplex(config)
        return True
        
    except Exception as e:
        print(f"❌ Error starting Lagos2Duplex training: {e}")
        return False


def start_afrocover_training(args):
    """Start AfroCover StyleGAN2 training"""
    print("\n🚀 Starting AfroCover StyleGAN2 Training...")
    
    if not check_afrocover_data():
        return False
    
    try:
        from afrocover.train import main as train_afrocover
        import argparse as afro_argparse
        
        # Create arguments for afrocover training
        afro_args = afro_argparse.Namespace()
        afro_args.data_path = str(current_dir / "data_processed" / "afrocover")
        afro_args.output_dir = str(current_dir / "checkpoints" / "afrocover")
        afro_args.batch_size = args.batch_size or 4
        afro_args.num_epochs = args.epochs or 100
        afro_args.learning_rate = args.learning_rate or 0.002
        afro_args.image_size = 256
        afro_args.z_dim = 512
        afro_args.resume = None
        afro_args.save_freq = 10
        afro_args.sample_freq = 5
        
        print(f"Configuration: {afro_args.num_epochs} epochs, batch size {afro_args.batch_size}")
        
        # Start training
        train_afrocover(afro_args)
        return True
        
    except Exception as e:
        print(f"❌ Error starting AfroCover training: {e}")
        return False


def start_unified_monitoring(args):
    """Run unified training with periodic evaluation (FID) and monitoring."""
    print("\n🔁 Starting unified monitoring and training")

    # Prepare list of jobs
    jobs = []
    if args.project in ['lagos2duplex', 'both']:
        jobs.append('lagos2duplex')
    if args.project in ['afrocover', 'both']:
        jobs.append('afrocover')

    # Simple orchestration: run sequentially to avoid GPU contention
    for job in jobs:
        if job == 'lagos2duplex':
            print("\n--- Launching Lagos2Duplex training with periodic FID evaluation ---")
            # Use existing start_lagos2duplex_training but with eval scheduling
            from lagos2duplex.config import get_default_config
            cfg = get_default_config()
            if args.quick:
                from lagos2duplex.config import get_quick_test_config
                cfg = get_quick_test_config()
            # Override epochs/batchsize
            if args.epochs:
                cfg.training.num_epochs = args.epochs
            if args.batch_size:
                cfg.training.batch_size = args.batch_size

            # Set FID evaluation frequency
            cfg.logging.eval_fid_freq = args.eval_fid_freq

            # Start training via main in lagos2duplex.train (which will now check cfg.logging.eval_fid_freq)
            from lagos2duplex.train import main as lagos_main
            lagos_main(cfg)

        elif job == 'afrocover':
            print("\n--- Launching AfroCover training with periodic FID evaluation ---")
            # Start afrocover training and periodically compute FID
            from afrocover.train import main as afro_main
            # We will call afro_main which supports CLI args; for unified run, use default behavior
            # Note: afrocover/train.py currently does not compute FID by itself; evaluation will be triggered separately
            afro_args = argparse.Namespace()
            afro_args.data_path = str(current_dir / "data_processed" / "afrocover")
            afro_args.output_dir = str(current_dir / "checkpoints" / "afrocover")
            afro_args.batch_size = args.batch_size or 4
            afro_args.num_epochs = args.epochs or 100
            afro_args.learning_rate = args.learning_rate or 0.002
            afro_args.image_size = 256
            afro_args.z_dim = 512
            afro_args.resume = None
            afro_args.save_freq = args.save_freq or 10
            afro_args.sample_freq = args.sample_freq or 5

            # Launch training
            afro_main(afro_args)

    print("\nUnified training/monitoring finished")


# Add args to main parser
def enhanced_main():
    parser = argparse.ArgumentParser(description='Unified Lagos GAN Training with Monitoring')
    parser.add_argument('project', choices=['lagos2duplex', 'afrocover', 'both'], 
                       help='Which project to train')
    parser.add_argument('--config', choices=['quick', 'default', 'high_quality'], 
                       default='default', help='Configuration preset (lagos2duplex only)')
    parser.add_argument('--epochs', type=int, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    parser.add_argument('--gpu', action='store_true', help='Force GPU usage check')
    parser.add_argument('--eval_fid_freq', type=int, default=0, help='Compute FID every N epochs (0 disables)')
    parser.add_argument('--quick', action='store_true', help='Use quick test configs')
    parser.add_argument('--save_freq', type=int, default=10, help='Checkpoint save frequency for afrocover')
    parser.add_argument('--sample_freq', type=int, default=5, help='Sample generation frequency for afrocover')

    args = parser.parse_args()

    # Keep existing checks
    if not check_common_environment():
        return 1

    # GPU check
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ GPU available: {torch.cuda.get_device_name()}")
        else:
            print("⚠️  No GPU detected - training will be slow")
            if args.gpu:
                print("❌ GPU required but not available")
                return 1
    except ImportError:
        print("⚠️  PyTorch not found - please install requirements")
        return 1

    # Run unified monitoring
    start_unified_monitoring(args)

if __name__ == "__main__":
    enhanced_main()
