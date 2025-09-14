#!/usr/bin/env python3
"""
Script to help add and organize training data for Lagos-GAN project.
This script will help you:
1. Copy images from source folders to the correct training directories
2. Automatically split data into train/val/test sets
3. Validate image formats and sizes
4. Generate data statistics

Usage:
    python add_training_data.py --help
"""

import os
import shutil
import argparse
from pathlib import Path
import random
from PIL import Image
import sys

def validate_image(image_path):
    """Check if image is valid and meets minimum requirements."""
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            if width < 256 or height < 256:
                print(f"Warning: {image_path} is too small ({width}x{height}). Minimum: 256x256")
                return False
            return True
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return False

def split_data(source_dir, dest_base_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """Split images from source directory into train/val/test folders."""
    if not os.path.exists(source_dir):
        print(f"Source directory {source_dir} does not exist!")
        return
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    all_images = []
    
    for ext in image_extensions:
        all_images.extend(Path(source_dir).glob(f"*{ext}"))
        all_images.extend(Path(source_dir).glob(f"*{ext.upper()}"))
    
    if not all_images:
        print(f"No images found in {source_dir}")
        return
    
    # Shuffle for random split
    random.shuffle(all_images)
    
    total = len(all_images)
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    
    train_images = all_images[:train_count]
    val_images = all_images[train_count:train_count + val_count]
    test_images = all_images[train_count + val_count:]
    
    # Create destination directories
    train_dir = os.path.join(dest_base_dir, 'train')
    val_dir = os.path.join(dest_base_dir, 'val')
    test_dir = os.path.join(dest_base_dir, 'test')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Copy files
    valid_count = 0
    
    def copy_images(image_list, dest_dir, split_name):
        nonlocal valid_count
        copied = 0
        for img_path in image_list:
            if validate_image(img_path):
                dest_path = os.path.join(dest_dir, img_path.name)
                shutil.copy2(img_path, dest_path)
                copied += 1
                valid_count += 1
            else:
                print(f"Skipping invalid image: {img_path}")
        print(f"Copied {copied} images to {split_name}")
        return copied
    
    train_copied = copy_images(train_images, train_dir, "train")
    val_copied = copy_images(val_images, val_dir, "val")
    test_copied = copy_images(test_images, test_dir, "test")
    
    print(f"\n=== Data Split Summary ===")
    print(f"Total valid images: {valid_count}")
    print(f"Train: {train_copied} ({train_copied/valid_count*100:.1f}%)")
    print(f"Val: {val_copied} ({val_copied/valid_count*100:.1f}%)")
    print(f"Test: {test_copied} ({test_copied/valid_count*100:.1f}%)")

def count_existing_data():
    """Count existing data in all folders."""
    base_dir = Path("data")
    
    print("\n=== Current Data Count ===")
    
    # AfroCover data
    afrocover_dir = base_dir / "afrocover"
    if afrocover_dir.exists():
        print("\nAfroCover Dataset:")
        for split in ['train', 'val', 'test']:
            split_dir = afrocover_dir / split
            if split_dir.exists():
                count = len(list(split_dir.glob("*.jpg")) + list(split_dir.glob("*.png")) + list(split_dir.glob("*.jpeg")))
                print(f"  {split}: {count} images")
    
    # Lagos2Duplex data
    lagos2duplex_dir = base_dir / "lagos2duplex"
    if lagos2duplex_dir.exists():
        print("\nLagos2Duplex Dataset:")
        for style in ['lagos', 'duplex']:
            style_dir = lagos2duplex_dir / style
            if style_dir.exists():
                print(f"  {style.title()} style:")
                for split in ['train', 'val', 'test']:
                    split_dir = style_dir / split
                    if split_dir.exists():
                        count = len(list(split_dir.glob("*.jpg")) + list(split_dir.glob("*.png")) + list(split_dir.glob("*.jpeg")))
                        print(f"    {split}: {count} images")

def main():
    parser = argparse.ArgumentParser(description="Add training data to Lagos-GAN project")
    parser.add_argument("--source", help="Source directory containing images to add")
    parser.add_argument("--dataset", choices=['afrocover', 'lagos', 'duplex'], 
                        help="Which dataset to add images to")
    parser.add_argument("--count", action="store_true", help="Count existing data")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Training data ratio")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation data ratio")
    
    args = parser.parse_args()
    
    if args.count:
        count_existing_data()
        return
    
    if not args.source or not args.dataset:
        print("Usage examples:")
        print("  # Count existing data")
        print("  python add_training_data.py --count")
        print()
        print("  # Add album covers")
        print("  python add_training_data.py --source ~/Downloads/album_covers --dataset afrocover")
        print()
        print("  # Add Lagos house images")
        print("  python add_training_data.py --source ~/Downloads/lagos_houses --dataset lagos")
        print()
        print("  # Add Duplex house images")
        print("  python add_training_data.py --source ~/Downloads/duplex_houses --dataset duplex")
        return
    
    # Determine destination directory
    if args.dataset == 'afrocover':
        dest_dir = "data/afrocover"
    elif args.dataset == 'lagos':
        dest_dir = "data/lagos2duplex/lagos"
    elif args.dataset == 'duplex':
        dest_dir = "data/lagos2duplex/duplex"
    else:
        print(f"Error: Unknown dataset '{args.dataset}'.")
        return
    
    print(f"Adding images from {args.source} to {args.dataset} dataset...")
    split_data(args.source, dest_dir, args.train_ratio, args.val_ratio, 1.0 - args.train_ratio - args.val_ratio)
    
    print("\nDone! Use --count to see updated statistics.")

if __name__ == "__main__":
    main()
