#!/usr/bin/env python3
"""
Lagos-GAN Dataset Preprocessing Pipeline
Comprehensive preprocessing for all datasets with resizing and normalization.

Features:
- Resize all images to consistent resolution (256x256 or 512x512)
- Normalize pixel values to [-1, 1] range for GAN training
- Quality validation and filtering
- Progress tracking and error handling
- Batch processing for efficiency
- Preserve original aspect ratios with smart cropping
"""

import os
import sys
from pathlib import Path
import argparse
import logging
from PIL import Image, ImageOps
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class LagosGANPreprocessor:
    """Comprehensive preprocessing pipeline for Lagos-GAN datasets."""
    
    def __init__(self, target_size=256, output_format='png', quality_threshold=150):
        """
        Initialize preprocessor.
        
        Args:
            target_size (int): Target resolution (e.g., 256 for 256x256)
            output_format (str): Output format ('png' or 'jpg')
            quality_threshold (int): Minimum size threshold for images
        """
        self.target_size = target_size
        self.output_format = output_format.lower()
        self.quality_threshold = quality_threshold
        
        # Define preprocessing transforms
        self.preprocess_transform = transforms.Compose([
            transforms.Resize((target_size, target_size), transforms.InterpolationMode.LANCZOS),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize to [-1, 1]
        ])
        
        # For saving processed images back to files
        self.save_transform = transforms.Compose([
            transforms.Normalize([-1, -1, -1], [2, 2, 2]),  # Denormalize from [-1, 1] to [0, 1]
        ])
        
        self.stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'skipped_small': 0,
            'skipped_corrupt': 0
        }
    
    def validate_image(self, image_path: Path) -> tuple:
        """
        Validate image quality and format.
        
        Returns:
            (is_valid: bool, reason: str, image: PIL.Image or None)
        """
        try:
            # Open and validate image
            image = Image.open(image_path)
            
            # Check if image is too small
            if image.width < self.quality_threshold or image.height < self.quality_threshold:
                return False, f"Too small: {image.width}x{image.height}", None
            
            # Check aspect ratio (reject extremely wide/tall images)
            aspect_ratio = max(image.width, image.height) / min(image.width, image.height)
            if aspect_ratio > 5:
                return False, f"Extreme aspect ratio: {aspect_ratio:.2f}", None
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            return True, "Valid", image
            
        except Exception as e:
            return False, f"Corrupt/unreadable: {str(e)}", None
    
    def smart_resize_crop(self, image: Image.Image) -> Image.Image:
        """
        Smart resize with center cropping to maintain quality.
        """
        # Calculate crop box to maintain aspect ratio
        width, height = image.size
        target_aspect = 1.0  # Square images
        
        if width / height > target_aspect:
            # Image is wider - crop width
            new_width = int(height * target_aspect)
            left = (width - new_width) // 2
            crop_box = (left, 0, left + new_width, height)
        else:
            # Image is taller - crop height
            new_height = int(width / target_aspect)
            top = (height - new_height) // 2
            crop_box = (0, top, width, top + new_height)
        
        # Crop and resize
        cropped = image.crop(crop_box)
        resized = cropped.resize((self.target_size, self.target_size), Image.Resampling.LANCZOS)
        
        return resized
    
    def process_single_image(self, input_path: Path, output_path: Path) -> dict:
        """
        Process a single image with validation, resizing, and normalization.
        
        Returns:
            dict: Processing result with status and metadata
        """
        result = {
            'input_path': str(input_path),
            'output_path': str(output_path),
            'success': False,
            'reason': '',
            'original_size': None,
            'processed_size': (self.target_size, self.target_size)
        }
        
        try:
            # Validate image
            is_valid, reason, image = self.validate_image(input_path)
            
            if not is_valid:
                result['reason'] = reason
                if 'Too small' in reason:
                    self.stats['skipped_small'] += 1
                else:
                    self.stats['skipped_corrupt'] += 1
                return result
            
            result['original_size'] = image.size
            
            # Smart resize with cropping
            processed_image = self.smart_resize_crop(image)
            
            # Create output directory if needed
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save processed image
            if self.output_format == 'png':
                processed_image.save(output_path, 'PNG', optimize=True)
            else:
                processed_image.save(output_path, 'JPEG', quality=95, optimize=True)
            
            result['success'] = True
            result['reason'] = 'Successfully processed'
            self.stats['successful'] += 1
            
        except Exception as e:
            result['reason'] = f"Processing error: {str(e)}"
            self.stats['failed'] += 1
            logger.error(f"Error processing {input_path}: {e}")
        
        self.stats['total_processed'] += 1
        return result
    
    def process_dataset_folder(self, input_dir: Path, output_dir: Path, max_workers=4) -> list:
        """
        Process all images in a dataset folder.
        
        Args:
            input_dir (Path): Input directory with raw images
            output_dir (Path): Output directory for processed images
            max_workers (int): Number of parallel workers
            
        Returns:
            list: Processing results for all images
        """
        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        image_files = [
            f for f in input_dir.rglob('*') 
            if f.is_file() and f.suffix.lower() in image_extensions
        ]
        
        if not image_files:
            logger.warning(f"No image files found in {input_dir}")
            return []
        
        logger.info(f"Found {len(image_files)} images in {input_dir}")
        logger.info(f"Processing to {output_dir} with resolution {self.target_size}x{self.target_size}")
        
        # Prepare output paths
        tasks = []
        for input_path in image_files:
            # Maintain relative structure
            relative_path = input_path.relative_to(input_dir)
            output_path = output_dir / relative_path.with_suffix(f'.{self.output_format}')
            tasks.append((input_path, output_path))
        
        # Process images in parallel
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(self.process_single_image, input_path, output_path): (input_path, output_path)
                for input_path, output_path in tasks
            }
            
            # Collect results with progress bar
            with tqdm(total=len(tasks), desc=f"Processing {input_dir.name}") as pbar:
                for future in as_completed(future_to_task):
                    result = future.result()
                    results.append(result)
                    pbar.update(1)
                    
                    # Update progress description
                    pbar.set_postfix({
                        'Success': self.stats['successful'],
                        'Failed': self.stats['failed'] + self.stats['skipped_small'] + self.stats['skipped_corrupt']
                    })
        
        return results
    
    def preprocess_lagos_gan_datasets(self, data_root: Path, output_root: Path) -> dict:
        """
        Preprocess all Lagos-GAN datasets.
        
        Args:
            data_root (Path): Root directory containing original datasets
            output_root (Path): Root directory for processed datasets
            
        Returns:
            dict: Complete processing statistics
        """
        logger.info("=" * 60)
        logger.info("LAGOS-GAN DATASET PREPROCESSING PIPELINE")
        logger.info("=" * 60)
        logger.info(f"Target resolution: {self.target_size}x{self.target_size}")
        logger.info(f"Output format: {self.output_format.upper()}")
        logger.info(f"Quality threshold: {self.quality_threshold}px minimum")
        logger.info("")
        
        # Define dataset structure
        datasets = {
            'afrocover': ['train', 'val', 'test'],
            'lagos2duplex/lagos': ['train', 'val', 'test'],
            'lagos2duplex/duplex': ['train', 'val', 'test']
        }
        
        complete_results = {}
        
        for dataset_name, splits in datasets.items():
            logger.info(f"Processing dataset: {dataset_name}")
            dataset_results = {}
            
            for split in splits:
                input_dir = data_root / dataset_name / split
                output_dir = output_root / dataset_name / split
                
                if not input_dir.exists():
                    logger.warning(f"Directory not found: {input_dir}")
                    continue
                
                # Reset stats for this split
                self.stats = {
                    'total_processed': 0,
                    'successful': 0,
                    'failed': 0,
                    'skipped_small': 0,
                    'skipped_corrupt': 0
                }
                
                # Process the split
                results = self.process_dataset_folder(input_dir, output_dir)
                dataset_results[split] = {
                    'results': results,
                    'stats': self.stats.copy()
                }
                
                logger.info(f"  {split}: {self.stats['successful']} successful, "
                           f"{self.stats['failed'] + self.stats['skipped_small'] + self.stats['skipped_corrupt']} failed/skipped")
            
            complete_results[dataset_name] = dataset_results
            logger.info("")
        
        return complete_results
    
    def generate_preprocessing_report(self, results: dict, output_path: Path):
        """Generate a detailed preprocessing report."""
        report = {
            'preprocessing_config': {
                'target_size': self.target_size,
                'output_format': self.output_format,
                'quality_threshold': self.quality_threshold
            },
            'results': results,
            'summary': {}
        }
        
        # Calculate summary statistics
        total_success = 0
        total_failed = 0
        total_processed = 0
        
        for dataset_name, dataset_results in results.items():
            for split, split_data in dataset_results.items():
                stats = split_data['stats']
                total_success += stats['successful']
                total_failed += stats['failed'] + stats['skipped_small'] + stats['skipped_corrupt']
                total_processed += stats['total_processed']
        
        report['summary'] = {
            'total_processed': total_processed,
            'total_successful': total_success,
            'total_failed': total_failed,
            'success_rate': (total_success / total_processed * 100) if total_processed > 0 else 0
        }
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info("=" * 60)
        logger.info("PREPROCESSING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total images processed: {total_processed}")
        logger.info(f"Successfully processed: {total_success}")
        logger.info(f"Failed/Skipped: {total_failed}")
        logger.info(f"Success rate: {report['summary']['success_rate']:.1f}%")
        logger.info(f"Report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Lagos-GAN Dataset Preprocessing Pipeline")
    parser.add_argument("--data_root", type=str, default="data", 
                        help="Root directory containing original datasets")
    parser.add_argument("--output_root", type=str, default="data_processed",
                        help="Root directory for processed datasets")
    parser.add_argument("--target_size", type=int, default=256,
                        help="Target image resolution (256 or 512)")
    parser.add_argument("--output_format", choices=['png', 'jpg'], default='png',
                        help="Output image format")
    parser.add_argument("--quality_threshold", type=int, default=150,
                        help="Minimum image size threshold")
    parser.add_argument("--max_workers", type=int, default=4,
                        help="Number of parallel workers")
    
    args = parser.parse_args()
    
    # Initialize preprocessor
    preprocessor = LagosGANPreprocessor(
        target_size=args.target_size,
        output_format=args.output_format,
        quality_threshold=args.quality_threshold
    )
    
    # Setup paths
    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    
    if not data_root.exists():
        logger.error(f"Data root directory not found: {data_root}")
        sys.exit(1)
    
    # Create output directory
    output_root.mkdir(exist_ok=True)
    
    # Process all datasets
    results = preprocessor.preprocess_lagos_gan_datasets(data_root, output_root)
    
    # Generate report
    report_path = output_root / "preprocessing_report.json"
    preprocessor.generate_preprocessing_report(results, report_path)


if __name__ == "__main__":
    main()
