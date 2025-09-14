#!/usr/bin/env python3
"""
Fast Lagos Nigeria Scraper
Efficient scraper specifically for Lagos, Nigeria architecture and Afrobeats covers.
"""

import os
import time
import requests
import json
from pathlib import Path
import logging
from PIL import Image
import hashlib
import io
import argparse
from urllib.parse import urlparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class FastLagosNigeriaScraper:
    """Efficient scraper for Lagos Nigeria specific content."""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
        
        # Load API keys
        self.unsplash_key = os.getenv('UNSPLASH_API_KEY')
        self.pixabay_key = os.getenv('PIXABAY_API_KEY')
        
        # High-priority Lagos Nigeria terms
        self.lagos_nigeria_terms = [
            "Lagos Nigeria house",
            "Lagos Nigeria architecture", 
            "Lagos Nigeria building",
            "Lagos Nigeria duplex",
            "Victoria Island Lagos",
            "Ikoyi Lagos house",
            "Lagos Island Nigeria",
            "Lagos Nigeria residential",
            "Nigerian house Lagos",
            "Lagos Nigeria property"
        ]
        
        # Afrobeats/Nigerian music terms
        self.afro_music_terms = [
            "afrobeats album cover",
            "nigerian music album",
            "afrobeat album art",
            "nigerian artist cover",
            "african music cover",
            "afrocentric album",
            "nigerian album artwork",
            "afro music art"
        ]
    
    def search_unsplash_fast(self, query, limit=50):
        """Fast Unsplash search with Nigeria focus."""
        images = []
        
        if not self.unsplash_key:
            logger.warning("No Unsplash API key found")
            return images
            
        url = "https://api.unsplash.com/search/photos"
        headers = {"Authorization": f"Client-ID {self.unsplash_key}"}
        
        # Add Nigeria to query for specificity
        if "Nigeria" not in query:
            query = f"{query} Nigeria"
            
        params = {
            "query": query,
            "per_page": min(30, limit),
            "orientation": "all"
        }
        
        try:
            response = self.session.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            results = data.get('results', [])
            
            for photo in results[:limit]:
                try:
                    # Get regular size image
                    img_url = photo['urls']['regular']
                    
                    # Create descriptive filename
                    description = photo.get('description', '') or photo.get('alt_description', '')
                    if description:
                        filename = f"{description[:50]}_{photo['id']}.jpg"
                    else:
                        filename = f"unsplash_{photo['id']}.jpg"
                    
                    # Clean filename
                    filename = "".join(c for c in filename if c.isalnum() or c in '._- ').strip()
                    filename = filename.replace(' ', '_')
                    
                    images.append({
                        'url': img_url,
                        'filename': filename,
                        'source': 'unsplash',
                        'query': query
                    })
                    
                except Exception as e:
                    logger.error(f"Error processing Unsplash photo: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Unsplash search error for '{query}': {e}")
            
        return images
    
    def search_pixabay_fast(self, query, limit=50):
        """Fast Pixabay search with Nigeria focus."""
        images = []
        
        if not self.pixabay_key or self.pixabay_key == "your_pixabay_api_key_here":
            logger.warning("No valid Pixabay API key found")
            return images
            
        url = "https://pixabay.com/api/"
        
        # Add Nigeria for specificity
        if "Nigeria" not in query:
            query = f"{query} Nigeria"
            
        params = {
            "key": self.pixabay_key,
            "q": query,
            "image_type": "photo",
            "min_width": 256,
            "min_height": 256,
            "per_page": min(100, limit)
        }
        
        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            hits = data.get('hits', [])
            
            for hit in hits[:limit]:
                try:
                    img_url = hit['webformatURL']
                    tags = hit.get('tags', '').replace(', ', '_')
                    filename = f"pixabay_{tags[:30]}_{hit['id']}.jpg"
                    filename = "".join(c for c in filename if c.isalnum() or c in '._-').strip()
                    
                    images.append({
                        'url': img_url,
                        'filename': filename,
                        'source': 'pixabay',
                        'query': query
                    })
                    
                except Exception as e:
                    logger.error(f"Error processing Pixabay image: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Pixabay search error for '{query}': {e}")
            
        return images
    
    def download_image(self, image_info, output_dir):
        """Download and save image."""
        try:
            response = self.session.get(image_info['url'], timeout=15)
            response.raise_for_status()
            
            # Validate image
            img = Image.open(io.BytesIO(response.content))
            if img.size[0] < 256 or img.size[1] < 256:
                logger.warning(f"Image too small: {img.size}")
                return False
                
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
                
            # Save image
            filepath = output_dir / image_info['filename']
            img.save(filepath, 'JPEG', quality=95)
            
            logger.info(f"Saved: {image_info['filename']}")
            return True
            
        except Exception as e:
            logger.error(f"Download failed for {image_info['url']}: {e}")
            return False
    
    def scrape_lagos_architecture(self, dataset_type, limit=100):
        """Scrape Lagos Nigeria architecture images."""
        
        if dataset_type == "lagos":
            search_terms = self.lagos_nigeria_terms
            output_base = Path("data/lagos2duplex/lagos")
        elif dataset_type == "duplex":
            # Add duplex-specific terms
            search_terms = [f"{term} duplex" for term in self.lagos_nigeria_terms[:5]]
            output_base = Path("data/lagos2duplex/duplex")
        else:
            logger.error(f"Unknown dataset type: {dataset_type}")
            return
            
        # Create directories
        for split in ['train', 'val', 'test']:
            (output_base / split).mkdir(parents=True, exist_ok=True)
            
        all_images = []
        
        # Search across multiple terms
        images_per_term = max(1, limit // len(search_terms))
        
        for term in search_terms:
            logger.info(f"Searching for: {term}")
            
            # Try Unsplash first
            unsplash_images = self.search_unsplash_fast(term, images_per_term)
            all_images.extend(unsplash_images)
            
            # Then Pixabay
            pixabay_images = self.search_pixabay_fast(term, images_per_term)
            all_images.extend(pixabay_images)
            
            # Don't exceed limit
            if len(all_images) >= limit:
                all_images = all_images[:limit]
                break
                
            time.sleep(0.5)  # Brief pause between searches
            
        logger.info(f"Found {len(all_images)} images to download")
        
        # Download images
        successful = 0
        for i, img_info in enumerate(all_images, 1):
            # Determine split (80/10/10)
            if i <= len(all_images) * 0.8:
                split_dir = output_base / "train"
            elif i <= len(all_images) * 0.9:
                split_dir = output_base / "val"
            else:
                split_dir = output_base / "test"
                
            if self.download_image(img_info, split_dir):
                successful += 1
                
            if i % 10 == 0:
                logger.info(f"Progress: {i}/{len(all_images)} ({successful} successful)")
                
            time.sleep(0.5)  # Respectful delay
            
        logger.info(f"Downloaded {successful}/{len(all_images)} images for {dataset_type}")
    
    def scrape_afrobeats_covers(self, limit=100):
        """Scrape Afrobeats/Nigerian album covers."""
        
        output_base = Path("data/afrocover")
        
        # Create directories
        for split in ['train', 'val', 'test']:
            (output_base / split).mkdir(parents=True, exist_ok=True)
            
        all_images = []
        images_per_term = max(1, limit // len(self.afro_music_terms))
        
        for term in self.afro_music_terms:
            logger.info(f"Searching for: {term}")
            
            # Search both APIs
            unsplash_images = self.search_unsplash_fast(term, images_per_term)
            all_images.extend(unsplash_images)
            
            pixabay_images = self.search_pixabay_fast(term, images_per_term)
            all_images.extend(pixabay_images)
            
            if len(all_images) >= limit:
                all_images = all_images[:limit]
                break
                
            time.sleep(0.5)
            
        logger.info(f"Found {len(all_images)} album cover images to download")
        
        # Download
        successful = 0
        for i, img_info in enumerate(all_images, 1):
            # Determine split
            if i <= len(all_images) * 0.8:
                split_dir = output_base / "train"
            elif i <= len(all_images) * 0.9:
                split_dir = output_base / "val"
            else:
                split_dir = output_base / "test"
                
            if self.download_image(img_info, split_dir):
                successful += 1
                
            if i % 10 == 0:
                logger.info(f"Progress: {i}/{len(all_images)} ({successful} successful)")
                
            time.sleep(0.5)
            
        logger.info(f"Downloaded {successful}/{len(all_images)} album covers")

def main():
    parser = argparse.ArgumentParser(description="Fast Lagos Nigeria Scraper")
    parser.add_argument("--type", choices=['lagos', 'duplex', 'afrocover'], 
                        required=True, help="Type of data to scrape")
    parser.add_argument("--limit", type=int, default=200, 
                        help="Number of images to collect")
    
    args = parser.parse_args()
    
    # Load environment variables
    if os.path.exists('api_keys.env'):
        with open('api_keys.env', 'r') as f:
            for line in f:
                if '=' in line and not line.strip().startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value
    
    scraper = FastLagosNigeriaScraper()
    
    if args.type == 'afrocover':
        scraper.scrape_afrobeats_covers(args.limit)
    else:
        scraper.scrape_lagos_architecture(args.type, args.limit)

if __name__ == "__main__":
    main()
