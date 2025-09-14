#!/usr/bin/env python3
"""
Data Management Script for Lagos-GAN

This script helps organize and prepare training data for both AfroCover and Lagos2Duplex models.
"""

import os
import shutil
import random
from pathlib import Path
from PIL import Image
import argparse


class DataManager:
    """Manages training data organization and validation"""
    
    def __init__(self, data_root="/Users/mac/Lagos-GAN/data"):
        self.data_root = Path(data_root)
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
    def organize_afrocover_data(self, source_dir, train_ratio=0.8, val_ratio=0.1):
        """
        Organize album cover images into train/val/test splits
        
        Args:
            source_dir: Directory containing album cover images
            train_ratio: Fraction for training (default 0.8 = 80%)
            val_ratio: Fraction for validation (default 0.1 = 10%)
        """
        source_path = Path(source_dir)
        if not source_path.exists():
            print(f"❌ Source directory not found: {source_dir}")
            return
            
        # Get all image files
        image_files = []
        for ext in self.supported_formats:
            image_files.extend(source_path.glob(f"*{ext}"))
            image_files.extend(source_path.glob(f"*{ext.upper()}"))
            
        if not image_files:
            print(f"❌ No images found in {source_dir}")
            return
            
        print(f"📊 Found {len(image_files)} images")
        
        # Shuffle and split
        random.shuffle(image_files)
        total = len(image_files)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
        
        train_files = image_files[:train_end]
        val_files = image_files[train_end:val_end]
        test_files = image_files[val_end:]
        
        # Create destination directories
        afrocover_path = self.data_root / "afrocover"
        train_dir = afrocover_path / "train"
        val_dir = afrocover_path / "val"
        test_dir = afrocover_path / "test"
        
        for dir_path in [train_dir, val_dir, test_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        # Copy files
        print("📁 Organizing AfroCover data...")
        self._copy_files(train_files, train_dir, "train")
        self._copy_files(val_files, val_dir, "val")
        self._copy_files(test_files, test_dir, "test")
        
        print("✅ AfroCover data organized successfully!")
        
    def organize_house_data(self, lagos_dir, duplex_dir, train_ratio=0.8, val_ratio=0.1):
        """
        Organize house images into domain-specific splits
        
        Args:
            lagos_dir: Directory containing simple Lagos house images
            duplex_dir: Directory containing duplex house images
        """
        self._organize_domain_data(lagos_dir, "lagos", train_ratio, val_ratio)
        self._organize_domain_data(duplex_dir, "duplex", train_ratio, val_ratio)
        
    def _organize_domain_data(self, source_dir, domain_name, train_ratio, val_ratio):
        """Organize data for a specific domain (lagos or duplex)"""
        source_path = Path(source_dir)
        if not source_path.exists():
            print(f"❌ Source directory not found: {source_dir}")
            return
            
        # Get all image files
        image_files = []
        for ext in self.supported_formats:
            image_files.extend(source_path.glob(f"*{ext}"))
            image_files.extend(source_path.glob(f"*{ext.upper()}"))
            
        if not image_files:
            print(f"❌ No images found in {source_dir}")
            return
            
        print(f"📊 Found {len(image_files)} {domain_name} images")
        
        # Shuffle and split
        random.shuffle(image_files)
        total = len(image_files)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
        
        train_files = image_files[:train_end]
        val_files = image_files[train_end:val_end]
        test_files = image_files[val_end:]
        
        # Create destination directories
        domain_path = self.data_root / "lagos2duplex" / domain_name
        train_dir = domain_path / "train"
        val_dir = domain_path / "val"
        test_dir = domain_path / "test"
        
        for dir_path in [train_dir, val_dir, test_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        # Copy files
        print(f"📁 Organizing {domain_name} data...")
        self._copy_files(train_files, train_dir, f"{domain_name} train")
        self._copy_files(val_files, val_dir, f"{domain_name} val")
        self._copy_files(test_files, test_dir, f"{domain_name} test")
        
        print(f"✅ {domain_name.title()} data organized successfully!")
        
    def _copy_files(self, file_list, dest_dir, split_name):
        """Copy files to destination with validation"""
        print(f"   📋 Copying {len(file_list)} images to {split_name}...")
        
        for i, src_file in enumerate(file_list):
            try:
                # Validate image
                img = Image.open(src_file)
                img.verify()
                
                # Copy with sequential naming
                dest_file = dest_dir / f"{split_name}_{i:04d}{src_file.suffix}"
                shutil.copy2(src_file, dest_file)
                
            except Exception as e:
                print(f"⚠️  Skipping {src_file.name}: {e}")
                
    def validate_data(self):
        """Validate the organized dataset"""
        print("🔍 Validating dataset...")
        
        # Check AfroCover data
        afrocover_path = self.data_root / "afrocover"
        for split in ["train", "val", "test"]:
            split_dir = afrocover_path / split
            if split_dir.exists():
                count = len(list(split_dir.glob("*")))
                print(f"   AfroCover {split}: {count} images")
            else:
                print(f"   ❌ AfroCover {split}: directory missing")
                
        # Check Lagos2Duplex data
        lagos2duplex_path = self.data_root / "lagos2duplex"
        for domain in ["lagos", "duplex"]:
            for split in ["train", "val", "test"]:
                split_dir = lagos2duplex_path / domain / split
                if split_dir.exists():
                    count = len(list(split_dir.glob("*")))
                    print(f"   {domain.title()} {split}: {count} images")
                else:
                    print(f"   ❌ {domain.title()} {split}: directory missing")
                    
    def create_sample_data(self, num_samples=10):
        """Create sample placeholder images for testing"""
        print(f"🎨 Creating {num_samples} sample images for testing...")
        
        from PIL import Image, ImageDraw, ImageFont
        import random
        
        # Create sample album covers
        afrocover_train = self.data_root / "afrocover" / "train"
        afrocover_train.mkdir(parents=True, exist_ok=True)
        
        for i in range(num_samples):
            # Create colorful sample album cover
            img = Image.new('RGB', (256, 256), 
                          (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
            draw = ImageDraw.Draw(img)
            
            # Add some text
            try:
                draw.text((50, 120), f"Album {i+1}", fill='white')
            except:
                pass  # If font not available
                
            img.save(afrocover_train / f"sample_album_{i:03d}.jpg")
            
        # Create sample house images
        for domain, color in [("lagos", (139, 69, 19)), ("duplex", (70, 130, 180))]:
            domain_train = self.data_root / "lagos2duplex" / domain / "train"
            domain_train.mkdir(parents=True, exist_ok=True)
            
            for i in range(num_samples):
                # Create simple house shape
                img = Image.new('RGB', (256, 256), (135, 206, 235))  # Sky blue background
                draw = ImageDraw.Draw(img)
                
                # Draw simple house
                draw.rectangle([80, 150, 176, 220], fill=color)  # House body
                draw.polygon([(60, 150), (128, 100), (196, 150)], fill=(139, 0, 0))  # Roof
                draw.rectangle([100, 180, 120, 220], fill=(101, 67, 33))  # Door
                draw.rectangle([140, 170, 160, 190], fill=(173, 216, 230))  # Window
                
                img.save(domain_train / f"sample_{domain}_{i:03d}.jpg")
                
        print("✅ Sample data created successfully!")


def main():
    parser = argparse.ArgumentParser(description="Manage Lagos-GAN training data")
    parser.add_argument("--create-samples", action="store_true", 
                       help="Create sample data for testing")
    parser.add_argument("--organize-afrocover", type=str,
                       help="Organize AfroCover data from source directory")
    parser.add_argument("--organize-houses", nargs=2, metavar=("LAGOS_DIR", "DUPLEX_DIR"),
                       help="Organize house data from two source directories")
    parser.add_argument("--validate", action="store_true",
                       help="Validate existing dataset")
    
    args = parser.parse_args()
    
    manager = DataManager()
    
    if args.create_samples:
        manager.create_sample_data()
        
    if args.organize_afrocover:
        manager.organize_afrocover_data(args.organize_afrocover)
        
    if args.organize_houses:
        lagos_dir, duplex_dir = args.organize_houses
        manager.organize_house_data(lagos_dir, duplex_dir)
        
    if args.validate:
        manager.validate_data()
        
    if not any([args.create_samples, args.organize_afrocover, args.organize_houses, args.validate]):
        print("🔧 Lagos-GAN Data Manager")
        print("=" * 40)
        print("Usage examples:")
        print("  Create sample data:     python manage_data.py --create-samples")
        print("  Organize album covers:  python manage_data.py --organize-afrocover ~/Downloads/album_covers")
        print("  Organize house data:    python manage_data.py --organize-houses ~/Downloads/lagos_houses ~/Downloads/duplex_houses")
        print("  Validate dataset:       python manage_data.py --validate")


if __name__ == "__main__":
    main()
