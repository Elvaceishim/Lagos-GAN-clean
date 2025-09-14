"""
Data Preprocessing Scripts for LagosGAN

This module contains scripts for preparing and augmenting datasets for both 
AfroCover and Lagos2Duplex models.
"""

import os
import json
import shutil
from pathlib import Path
from PIL import Image, ImageOps
import numpy as np
from typing import List, Dict, Tuple
import argparse


class DatasetPreprocessor:
    """Base class for dataset preprocessing"""
    
    def __init__(self, input_dir: str, output_dir: str, image_size: int = 256):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.image_size = image_size
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Supported image formats
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
        # Metadata storage
        self.metadata = {}
    
    def validate_image(self, image_path: Path) -> bool:
        """Validate if image is processable"""
        try:
            # Check file extension
            if image_path.suffix.lower() not in self.supported_formats:
                return False
            
            # Try to open and validate image
            with Image.open(image_path) as img:
                # Check if image is valid
                img.verify()
                
                # Re-open for size check (verify() closes the image)
                with Image.open(image_path) as img:
                    width, height = img.size
                    
                    # Check minimum size requirements
                    if min(width, height) < 64:
                        print(f"Image too small: {image_path} ({width}x{height})")
                        return False
                    
                    return True
                    
        except Exception as e:
            print(f"Invalid image {image_path}: {e}")
            return False
    
    def preprocess_image(self, image_path: Path, output_path: Path) -> bool:
        """Preprocess a single image"""
        try:
            with Image.open(image_path) as img:
                # Convert to RGB
                img = img.convert('RGB')
                
                # Apply preprocessing steps
                img = self.resize_and_crop(img)
                img = self.enhance_image(img)
                
                # Save processed image
                img.save(output_path, 'JPEG', quality=95)
                
                return True
                
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return False
    
    def resize_and_crop(self, img: Image.Image) -> Image.Image:
        """Resize and center crop image to target size"""
        # Calculate crop size (square crop from center)
        width, height = img.size
        crop_size = min(width, height)
        
        # Center crop to square
        left = (width - crop_size) // 2
        top = (height - crop_size) // 2
        right = left + crop_size
        bottom = top + crop_size
        
        img = img.crop((left, top, right, bottom))
        
        # Resize to target size
        img = img.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        
        return img
    
    def enhance_image(self, img: Image.Image) -> Image.Image:
        """Apply basic image enhancements"""
        # Auto-contrast (subtle enhancement)
        img = ImageOps.autocontrast(img, cutoff=1)
        
        # Could add more enhancements here:
        # - Color balance
        # - Sharpening
        # - Noise reduction
        
        return img
    
    def save_metadata(self):
        """Save metadata to JSON file"""
        metadata_path = self.output_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        print(f"Metadata saved to: {metadata_path}")


class AfroCoverPreprocessor(DatasetPreprocessor):
    """Preprocessor for AfroCover album cover dataset"""
    
    def __init__(self, input_dir: str, output_dir: str, image_size: int = 256):
        super().__init__(input_dir, output_dir, image_size)
        
        # Create split directories
        for split in ['train', 'val', 'test']:
            (self.output_dir / split).mkdir(exist_ok=True)
    
    def process_dataset(self, train_ratio: float = 0.8, val_ratio: float = 0.1):
        """Process the entire AfroCover dataset"""
        print(f"Processing AfroCover dataset from: {self.input_dir}")
        
        # Find all images
        image_paths = []
        for ext in self.supported_formats:
            image_paths.extend(self.input_dir.glob(f"**/*{ext}"))
            image_paths.extend(self.input_dir.glob(f"**/*{ext.upper()}"))
        
        # Validate images
        valid_images = [path for path in image_paths if self.validate_image(path)]
        print(f"Found {len(valid_images)} valid images out of {len(image_paths)} total")
        
        # Shuffle and split
        np.random.shuffle(valid_images)
        
        n_train = int(len(valid_images) * train_ratio)
        n_val = int(len(valid_images) * val_ratio)
        
        splits = {
            'train': valid_images[:n_train],
            'val': valid_images[n_train:n_train + n_val],
            'test': valid_images[n_train + n_val:]
        }
        
        # Process each split
        processed_counts = {}
        for split_name, split_images in splits.items():
            print(f"\nProcessing {split_name} split ({len(split_images)} images)...")
            
            split_dir = self.output_dir / split_name
            processed_count = 0
            
            for i, image_path in enumerate(split_images):
                output_filename = f"afrocover_{split_name}_{i:05d}.jpg"
                output_path = split_dir / output_filename
                
                if self.preprocess_image(image_path, output_path):
                    # Store metadata
                    self.metadata[output_filename] = {
                        'original_path': str(image_path),
                        'split': split_name,
                        'license': 'CC-BY',  # Update based on actual licensing
                        'source': 'curated_african_album_covers'
                    }
                    processed_count += 1
                
                if (i + 1) % 100 == 0:
                    print(f"  Processed {i + 1}/{len(split_images)} images")
            
            processed_counts[split_name] = processed_count
            print(f"  {split_name}: {processed_count} images processed")
        
        # Save metadata
        self.save_metadata()
        
        print(f"\nAfroCover dataset processing complete:")
        for split_name, count in processed_counts.items():
            print(f"  {split_name}: {count} images")


class Lagos2DuplexPreprocessor(DatasetPreprocessor):
    """Preprocessor for Lagos2Duplex dataset"""
    
    def __init__(self, input_dir: str, output_dir: str, image_size: int = 256):
        super().__init__(input_dir, output_dir, image_size)
        
        # Create domain directories
        for domain in ['lagos', 'duplex']:
            for split in ['train', 'val', 'test']:
                (self.output_dir / domain / split).mkdir(parents=True, exist_ok=True)
    
    def process_dataset(self, train_ratio: float = 0.8, val_ratio: float = 0.1):
        """Process the Lagos2Duplex dataset"""
        print(f"Processing Lagos2Duplex dataset from: {self.input_dir}")
        
        # Process each domain separately
        domains = ['lagos', 'duplex']  # Domain A and Domain B
        
        for domain in domains:
            domain_input = self.input_dir / domain
            if not domain_input.exists():
                print(f"Warning: Domain directory {domain_input} not found")
                continue
            
            print(f"\nProcessing {domain} domain...")
            
            # Find all images in domain
            image_paths = []
            for ext in self.supported_formats:
                image_paths.extend(domain_input.glob(f"**/*{ext}"))
                image_paths.extend(domain_input.glob(f"**/*{ext.upper()}"))
            
            # Validate images
            valid_images = [path for path in image_paths if self.validate_image(path)]
            print(f"  Found {len(valid_images)} valid images out of {len(image_paths)} total")
            
            # Shuffle and split
            np.random.shuffle(valid_images)
            
            n_train = int(len(valid_images) * train_ratio)
            n_val = int(len(valid_images) * val_ratio)
            
            splits = {
                'train': valid_images[:n_train],
                'val': valid_images[n_train:n_train + n_val],
                'test': valid_images[n_train + n_val:]
            }
            
            # Process each split
            for split_name, split_images in splits.items():
                print(f"  Processing {split_name} split ({len(split_images)} images)...")
                
                split_dir = self.output_dir / domain / split_name
                processed_count = 0
                
                for i, image_path in enumerate(split_images):
                    output_filename = f"{domain}_{split_name}_{i:05d}.jpg"
                    output_path = split_dir / output_filename
                    
                    if self.preprocess_image(image_path, output_path):
                        # Store metadata
                        self.metadata[output_filename] = {
                            'original_path': str(image_path),
                            'domain': domain,
                            'split': split_name,
                            'license': 'CC-BY',  # Update based on actual licensing
                            'source': f'{domain}_houses_dataset'
                        }
                        processed_count += 1
                    
                    if (i + 1) % 50 == 0:
                        print(f"    Processed {i + 1}/{len(split_images)} images")
                
                print(f"    {split_name}: {processed_count} images processed")
        
        # Save metadata
        self.save_metadata()
        print(f"\nLagos2Duplex dataset processing complete")


def create_augmented_dataset(input_dir: str, output_dir: str, num_augmentations: int = 3):
    """Create augmented versions of the dataset for training"""
    print(f"Creating augmented dataset with {num_augmentations} augmentations per image...")
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Define augmentation functions
    def augment_image(img: Image.Image, aug_type: int) -> Image.Image:
        """Apply different augmentation based on type"""
        if aug_type == 0:
            # Horizontal flip
            return img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        elif aug_type == 1:
            # Slight rotation (-10 to +10 degrees)
            angle = np.random.uniform(-10, 10)
            return img.rotate(angle, fillcolor=(255, 255, 255))
        elif aug_type == 2:
            # Color jitter
            from PIL import ImageEnhance
            img = ImageEnhance.Brightness(img).enhance(np.random.uniform(0.8, 1.2))
            img = ImageEnhance.Contrast(img).enhance(np.random.uniform(0.8, 1.2))
            img = ImageEnhance.Color(img).enhance(np.random.uniform(0.8, 1.2))
            return img
        else:
            return img
    
    # Process all images
    for image_path in input_path.glob("**/*.jpg"):
        if image_path.is_file():
            try:
                with Image.open(image_path) as img:
                    img = img.convert('RGB')
                    
                    # Save original
                    relative_path = image_path.relative_to(input_path)
                    output_original = output_path / relative_path
                    output_original.parent.mkdir(parents=True, exist_ok=True)
                    img.save(output_original, 'JPEG', quality=95)
                    
                    # Create augmented versions
                    for aug_idx in range(num_augmentations):
                        augmented_img = augment_image(img, aug_idx)
                        
                        # Save with augmented filename
                        stem = image_path.stem
                        suffix = image_path.suffix
                        aug_filename = f"{stem}_aug{aug_idx}{suffix}"
                        output_aug = output_original.parent / aug_filename
                        
                        augmented_img.save(output_aug, 'JPEG', quality=95)
                        
            except Exception as e:
                print(f"Error augmenting {image_path}: {e}")


def main():
    """Main function for dataset preprocessing"""
    parser = argparse.ArgumentParser(description='LagosGAN Dataset Preprocessing')
    parser.add_argument('--task', choices=['afrocover', 'lagos2duplex', 'augment'], 
                        required=True, help='Preprocessing task')
    parser.add_argument('--input_dir', type=str, required=True, 
                        help='Input directory containing raw images')
    parser.add_argument('--output_dir', type=str, required=True, 
                        help='Output directory for processed images')
    parser.add_argument('--image_size', type=int, default=256, 
                        help='Target image size')
    parser.add_argument('--train_ratio', type=float, default=0.8, 
                        help='Training set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.1, 
                        help='Validation set ratio')
    parser.add_argument('--num_augmentations', type=int, default=3, 
                        help='Number of augmentations per image (for augment task)')
    
    args = parser.parse_args()
    
    print(f"Starting {args.task} preprocessing...")
    
    if args.task == 'afrocover':
        preprocessor = AfroCoverPreprocessor(
            args.input_dir, 
            args.output_dir, 
            args.image_size
        )
        preprocessor.process_dataset(args.train_ratio, args.val_ratio)
        
    elif args.task == 'lagos2duplex':
        preprocessor = Lagos2DuplexPreprocessor(
            args.input_dir, 
            args.output_dir, 
            args.image_size
        )
        preprocessor.process_dataset(args.train_ratio, args.val_ratio)
        
    elif args.task == 'augment':
        create_augmented_dataset(
            args.input_dir, 
            args.output_dir, 
            args.num_augmentations
        )
    
    print("Preprocessing complete!")


if __name__ == "__main__":
    main()
