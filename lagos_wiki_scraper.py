#!/usr/bin/env python3
"""
Lagos Nigeria Focused Scraper
Efficient scraper using Wikimedia Commons and other free sources for Lagos-specific content.
"""

import os
import time
import requests
import json
from pathlib import Path
import logging
from PIL import Image
import io
import argparse
from urllib.parse import quote, urljoin

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class LagosNigeriaWikiScraper:
    """Lagos Nigeria focused scraper using Wikimedia Commons."""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Lagos-GAN-Research/1.0 (Educational Research Project)'
        })
        
        # Lagos Nigeria specific terms
        self.lagos_terms = [
            "Lagos Nigeria",
            "Lagos Nigeria architecture", 
            "Lagos Nigeria house",
            "Lagos Nigeria building",
            "Victoria Island Lagos",
            "Ikoyi Lagos", 
            "Lagos Island Nigeria",
            "Lagos Nigeria residential",
            "Lagos Nigeria property",
            "Nigerian architecture Lagos",
            "Lagos Nigeria duplex",
            "Lagos Nigeria bungalow"
        ]
        
        # Traditional/heritage focused terms
        self.traditional_terms = [
            "Lagos traditional architecture",
            "Nigerian traditional house",
            "Yoruba architecture Lagos",
            "Lagos colonial architecture", 
            "Lagos heritage building",
            "Lagos indigenous architecture",
            "Nigerian vernacular architecture",
            "Lagos historical building"
        ]
        
        # Modern/duplex terms
        self.modern_terms = [
            "Lagos modern house",
            "Lagos contemporary house",
            "Lagos duplex house",
            "Lagos Nigeria modern architecture",
            "Lagos residential development",
            "Lagos modern building",
            "Lagos apartment building",
            "Lagos condominium"
        ]
        
        # Afrobeats/Nigerian music terms
        self.music_terms = [
            "Nigerian album cover",
            "Afrobeats album",
            "Nigerian music",
            "Afrobeat album cover",
            "Nigerian artist",
            "African music album",
            "Afrocentric album",
            "Nigerian musician"
        ]
    
    def search_wikimedia_lagos(self, query, limit=20):
        """Search Wikimedia Commons for Lagos Nigeria content."""
        images = []
        
        # Wikimedia API endpoint
        api_url = "https://commons.wikimedia.org/w/api.php"
        
        params = {
            'action': 'query',
            'format': 'json',
            'list': 'search',
            'srsearch': f'filetype:bitmap {query}',
            'srnamespace': 6,  # File namespace
            'srlimit': limit,
            'srinfo': 'size',
            'srprop': 'size'
        }
        
        try:
            response = self.session.get(api_url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            if 'query' not in data or 'search' not in data['query']:
                return images
                
            for result in data['query']['search']:
                try:
                    title = result['title']
                    if not title.startswith('File:'):
                        continue
                        
                    # Get file info
                    file_params = {
                        'action': 'query',
                        'format': 'json',
                        'titles': title,
                        'prop': 'imageinfo',
                        'iiprop': 'url|size|metadata',
                        'iiurlwidth': 800
                    }
                    
                    file_response = self.session.get(api_url, params=file_params, timeout=10)
                    file_data = file_response.json()
                    
                    pages = file_data.get('query', {}).get('pages', {})
                    for page_id, page_data in pages.items():
                        imageinfo = page_data.get('imageinfo', [])
                        if imageinfo:
                            info = imageinfo[0]
                            
                            # Check image size
                            width = info.get('width', 0)
                            height = info.get('height', 0)
                            if width < 256 or height < 256:
                                continue
                                
                            # Get download URL
                            img_url = info.get('thumburl') or info.get('url')
                            if not img_url:
                                continue
                                
                            # Create filename
                            clean_title = title.replace('File:', '').replace(' ', '_')
                            filename = f"lagos_{clean_title[:50]}_{abs(hash(img_url)) % 10000}.jpg"
                            filename = "".join(c for c in filename if c.isalnum() or c in '._-')
                            
                            images.append({
                                'url': img_url,
                                'filename': filename,
                                'source': 'wikimedia',
                                'title': title,
                                'query': query
                            })
                            
                except Exception as e:
                    logger.error(f"Error processing result: {e}")
                    continue
                    
                time.sleep(0.1)  # Brief pause between file requests
                
        except Exception as e:
            logger.error(f"Wikimedia search error for '{query}': {e}")
            
        return images
    
    def download_image(self, image_info, output_dir):
        """Download and save image."""
        try:
            response = self.session.get(image_info['url'], timeout=20)
            response.raise_for_status()
            
            # Validate image
            img = Image.open(io.BytesIO(response.content))
            if img.size[0] < 256 or img.size[1] < 256:
                return False
                
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
                
            # Save image
            filepath = output_dir / image_info['filename']
            img.save(filepath, 'JPEG', quality=90)
            
            logger.info(f"✓ {image_info['filename']}")
            return True
            
        except Exception as e:
            logger.error(f"Download failed for {image_info['url']}: {e}")
            return False
    
    def scrape_lagos_focused(self, dataset_type, limit=100):
        """Scrape Lagos Nigeria focused images."""
        
        if dataset_type == "lagos":
            # Traditional Lagos architecture
            search_terms = self.traditional_terms + self.lagos_terms[:6]
            output_base = Path("data/lagos2duplex/lagos")
        elif dataset_type == "duplex":
            # Modern/duplex Lagos architecture  
            search_terms = self.modern_terms + [f"{term} duplex" for term in self.lagos_terms[:4]]
            output_base = Path("data/lagos2duplex/duplex")
        elif dataset_type == "afrocover":
            # Nigerian/Afrobeats music
            search_terms = self.music_terms
            output_base = Path("data/afrocover")
        else:
            logger.error(f"Unknown dataset type: {dataset_type}")
            return
            
        # Create directories
        for split in ['train', 'val', 'test']:
            (output_base / split).mkdir(parents=True, exist_ok=True)
            
        all_images = []
        images_per_term = max(1, limit // len(search_terms))
        
        logger.info(f"Searching for {dataset_type} images with {len(search_terms)} terms...")
        
        for i, term in enumerate(search_terms, 1):
            logger.info(f"[{i}/{len(search_terms)}] Searching: {term}")
            
            images = self.search_wikimedia_lagos(term, images_per_term)
            all_images.extend(images)
            
            logger.info(f"  Found {len(images)} images for '{term}'")
            
            if len(all_images) >= limit:
                all_images = all_images[:limit]
                break
                
            time.sleep(1.0)  # Respectful delay
            
        # Remove duplicates by URL
        seen_urls = set()
        unique_images = []
        for img in all_images:
            if img['url'] not in seen_urls:
                seen_urls.add(img['url'])
                unique_images.append(img)
        
        logger.info(f"Found {len(unique_images)} unique {dataset_type} images to download")
        
        if not unique_images:
            logger.warning(f"No images found for {dataset_type}. Try different search terms.")
            return
            
        # Download images
        successful = 0
        for i, img_info in enumerate(unique_images, 1):
            # Determine split (80/10/10)
            if i <= len(unique_images) * 0.8:
                split_dir = output_base / "train"
            elif i <= len(unique_images) * 0.9:
                split_dir = output_base / "val"
            else:
                split_dir = output_base / "test"
                
            if self.download_image(img_info, split_dir):
                successful += 1
                
            if i % 5 == 0:
                logger.info(f"Progress: {i}/{len(unique_images)} ({successful} successful)")
                
            time.sleep(0.8)  # Respectful delay
            
        logger.info(f"✅ Downloaded {successful}/{len(unique_images)} {dataset_type} images")
        
        # Show final counts
        train_count = len(list((output_base / "train").glob("*.jpg")))
        val_count = len(list((output_base / "val").glob("*.jpg")))
        test_count = len(list((output_base / "test").glob("*.jpg")))
        
        logger.info(f"Final {dataset_type} dataset: {train_count} train, {val_count} val, {test_count} test")

def main():
    parser = argparse.ArgumentParser(description="Lagos Nigeria Focused Scraper")
    parser.add_argument("--type", choices=['lagos', 'duplex', 'afrocover'], 
                        required=True, help="Type of data to scrape")
    parser.add_argument("--limit", type=int, default=100, 
                        help="Number of images to collect")
    
    args = parser.parse_args()
    
    scraper = LagosNigeriaWikiScraper()
    scraper.scrape_lagos_focused(args.type, args.limit)

if __name__ == "__main__":
    main()
