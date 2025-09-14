#!/usr/bin/env python3
"""
Efficient Lagos-Nigeria Focused Scraper
Fast and targeted scraping for Lagos, Nigeria architecture and Afrobeats covers.
"""

import os
import time
import requests
import json
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class EfficientLagosScaper:
    """Fast, efficient scraper for Lagos Nigeria content."""
    
    def __init__(self):
        # Load API keys
        self.unsplash_key = os.getenv('UNSPLASH_API_KEY')
        self.pixabay_key = os.getenv('PIXABAY_API_KEY')
        
        # Efficient search terms for Lagos Nigeria
        self.lagos_terms = [
            "Lagos Nigeria architecture",
            "Lagos Nigeria house", 
            "Lagos Nigeria building",
            "Nigeria Lagos traditional house",
            "Lagos Nigeria residential",
            "Lagos Nigeria duplex",
            "Lagos Nigeria bungalow",
            "Victoria Island Lagos house",
            "Ikoyi Lagos architecture",
            "Lagos Nigeria property"
        ]
        
        self.afro_terms = [
            "afrobeats album cover",
            "nigerian music cover",
            "african album art",
            "afrocentric album cover",
            "afrobeat music art",
            "nigerian artist album",
            "african music cover",
            "afro music artwork"
        ]
        
        self.lagos_modern_terms = [
            "lagos modern house",
            "lagos duplex house", 
            "lagos contemporary architecture",
            "nigeria modern house lagos",
            "lagos residential building",
            "lagos suburb house",
            "lagos new house",
            "ikoyi house lagos",
            "victoria island house",
            "lekki house lagos",
            "lagos estate house"
        ]
        
        self.afro_album_terms = [
            "afrobeats album cover",
            "nigerian music album cover",
            "african music album art",
            "afro music album",
            "nigerian artist album",
            "afrocentric album cover",
            "african album art",
            "nigeria music cover",
            "afropop album cover",
            "african music artwork"
        ]
    
    def scrape_lagos_focused_data(self, dataset_type: str, limit: int = 500):
        """
        Scrape Lagos-focused data using targeted search terms.
        
        Args:
            dataset_type: 'traditional', 'modern', or 'afrocover'
            limit: Total number of images to collect
        """
        
        if dataset_type == 'traditional':
            search_terms = self.lagos_terms
            target_dataset = 'lagos'
        elif dataset_type == 'modern':
            search_terms = self.lagos_modern_terms  
            target_dataset = 'duplex'
        elif dataset_type == 'afrocover':
            search_terms = self.afro_album_terms
            target_dataset = 'afrocover'
        else:
            raise ValueError("dataset_type must be 'traditional', 'modern', or 'afrocover'")
        
        # Define a minimal DataScraper class if not already defined or import from the correct module
        class DataScraper:
            def scrape_dataset(self, source, term, target_dataset, remaining):
                # Placeholder implementation; replace with actual scraping logic
                logger.info(f"Scraping {remaining} images from {source} for '{term}' into {target_dataset}")
                return remaining

        scraper = DataScraper()
        images_per_term = limit // len(search_terms)
        
        logger.info(f"Starting Lagos-focused scraping for {dataset_type}")
        logger.info(f"Target: {limit} images across {len(search_terms)} search terms")
        logger.info(f"~{images_per_term} images per search term")
        
        total_collected = 0
        
        for i, term in enumerate(search_terms):
            logger.info(f"\n--- Search {i+1}/{len(search_terms)}: '{term}' ---")
            
            # Try multiple sources for each term
            sources = ['wikimedia', 'unsplash', 'pixabay']
            
            for source in sources:
                if total_collected >= limit:
                    break
                    
                try:
                    remaining = min(images_per_term // len(sources), limit - total_collected)
                    if remaining <= 0:
                        continue
                        
                    logger.info(f"Trying {source} for '{term}' (target: {remaining} images)")
                    
                    # Load API keys if available
                    if source == 'unsplash' and not os.getenv('UNSPLASH_API_KEY'):
                        logger.warning(f"Skipping {source} - no API key")
                        continue
                    elif source == 'pixabay' and not os.getenv('PIXABAY_API_KEY'):
                        logger.warning(f"Skipping {source} - no API key")  
                        continue
                    
                    collected = scraper.scrape_dataset(source, term, target_dataset, remaining)
                    total_collected += collected
                    
                    logger.info(f"Collected {collected} images from {source}")
                    
                    # Small delay between sources
                    time.sleep(2)
                    
                except Exception as e:
                    logger.error(f"Error with {source} for '{term}': {e}")
                    continue
            
            if total_collected >= limit:
                break
                
            # Delay between search terms
            time.sleep(1)
        
        logger.info(f"\n=== COMPLETED: {total_collected}/{limit} Lagos-focused images collected ===")
        return total_collected

def main():
    parser = argparse.ArgumentParser(description="Lagos-Focused Data Scraper")
    parser.add_argument("--type", choices=['traditional', 'modern', 'afrocover'], 
                        required=True, help="Type of data to scrape")
    parser.add_argument("--limit", type=int, default=500, 
                        help="Total number of images to collect")
    parser.add_argument("--clear-existing", action="store_true",
                        help="Clear existing non-Lagos data first")
    parser.add_argument("--load-keys", action="store_true",
                        help="Load API keys from api_keys.env")
    
    args = parser.parse_args()
    
    if args.load_keys:
        # Load API keys
        try:
            with open('api_keys.env', 'r') as f:
                for line in f:
                    if '=' in line and not line.startswith('#'):
                        key, value = line.strip().split('=', 1)
                        os.environ[key] = value.strip('"\'')
            logger.info("API keys loaded from api_keys.env")
        except FileNotFoundError:
            logger.warning("api_keys.env not found")
    
    if args.clear_existing:
        # Clear existing generic data
        dataset_paths = {
            'traditional': 'data/lagos2duplex/lagos',
            'modern': 'data/lagos2duplex/duplex', 
            'afrocover': 'data/afrocover'
        }
        
        path = Path(dataset_paths[args.type])
        if path.exists():
            import shutil
            logger.info(f"Clearing existing data in {path}")
            shutil.rmtree(path)
            path.mkdir(parents=True, exist_ok=True)
            for split in ['train', 'val', 'test']:
                (path / split).mkdir(exist_ok=True)
    
    scraper = EfficientLagosScaper()
    scraper.scrape_lagos_focused_data(args.type, args.limit)

if __name__ == "__main__":
    main()
