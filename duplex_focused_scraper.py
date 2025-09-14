#!/usr/bin/env python3
"""
Duplex-Focused Scraper for Lagos Nigeria
Specialized scraper targeting Lagos Nigeria modern/duplex houses specifically.
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

class DuplexFocusedScraper:
    """Specialized scraper for Lagos Nigeria duplex/modern houses."""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Lagos-GAN-Research/1.0 (Educational Research Project)'
        })
        
        # Highly specific duplex/modern house terms for Lagos Nigeria
        self.duplex_terms = [
            "Lagos Nigeria duplex house",
            "Lagos Nigeria modern house", 
            "Lagos Nigeria contemporary house",
            "Lagos Nigeria luxury house",
            "Lagos Nigeria mansion",
            "Lagos Nigeria two-story house",
            "Lagos Nigeria multi-story house",
            "Lagos Nigeria residential building",
            "Lagos Nigeria estate house",
            "Lagos Nigeria villa",
            "Victoria Island Lagos duplex",
            "Ikoyi Lagos modern house", 
            "Lekki Lagos duplex",
            "Lagos Nigeria penthouse",
            "Lagos Nigeria townhouse",
            "Lagos Nigeria apartment building",
            "Lagos Nigeria condominium",
            "Banana Island Lagos house",
            "Lagos Nigeria real estate",
            "Lagos Nigeria property development",
            "Lagos Nigeria architectural design",
            "Lagos Nigeria modern architecture",
            "Lagos Nigeria residential architecture",
            "Lagos Nigeria housing development",
            "Lagos Nigeria gated community",
            "Lagos Nigeria luxury property"
        ]
        
    def search_wikimedia(self, query: str, limit: int = 50) -> list:
        """Search Wikimedia Commons for duplex-related images."""
        images = []
        
        try:
            # Search for files
            search_url = "https://commons.wikimedia.org/w/api.php"
            params = {
                'action': 'query',
                'format': 'json',
                'list': 'search',
                'srsearch': f'filetype:bitmap {query}',
                'srnamespace': 6,  # File namespace
                'srlimit': min(limit, 50),
                'srprop': 'title|snippet'
            }
            
            response = self.session.get(search_url, params=params)
            data = response.json()
            
            if 'query' in data and 'search' in data['query']:
                for result in data['query']['search']:
                    title = result.get('title', '')
                    if title.startswith('File:'):
                        # Get image info
                        info_params = {
                            'action': 'query',
                            'format': 'json',
                            'titles': title,
                            'prop': 'imageinfo',
                            'iiprop': 'url|size',
                            'iiurlwidth': 800
                        }
                        
                        info_response = self.session.get(search_url, params=info_params)
                        info_data = info_response.json()
                        
                        if 'query' in info_data and 'pages' in info_data['query']:
                            for page in info_data['query']['pages'].values():
                                if 'imageinfo' in page:
                                    img_info = page['imageinfo'][0]
                                    img_url = img_info.get('thumburl') or img_info.get('url')
                                    if img_url:
                                        images.append({
                                            'url': img_url,
                                            'title': title,
                                            'width': img_info.get('width', 0),
                                            'height': img_info.get('height', 0)
                                        })
                        
                        # Rate limiting
                        time.sleep(0.3)
                        
        except Exception as e:
            logger.error(f"Error searching Wikimedia for '{query}': {e}")
            
        return images
    
    def download_image(self, img_data: dict, output_dir: Path) -> bool:
        """Download and save image."""
        try:
            response = self.session.get(img_data['url'], timeout=30)
            response.raise_for_status()
            
            # Validate image
            image = Image.open(io.BytesIO(response.content))
            
            # Filter out small images
            if image.width < 200 or image.height < 200:
                return False
                
            # Filter out very long/thin images (likely banners)
            aspect_ratio = max(image.width, image.height) / min(image.width, image.height)
            if aspect_ratio > 3:
                return False
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Generate filename
            safe_title = "".join(c for c in img_data['title'] if c.isalnum() or c in (' ', '-', '_')).rstrip()
            filename = f"duplex_{safe_title}_{hash(img_data['url']) % 10000}.jpg"
            filepath = output_dir / filename
            
            # Save image
            image.save(filepath, 'JPEG', quality=90)
            logger.info(f"✓ {filename}")
            return True
            
        except Exception as e:
            logger.error(f"✗ Failed to download {img_data.get('title', 'unknown')}: {e}")
            return False
    
    def scrape_duplex_images(self, limit: int = 200) -> int:
        """Main scraping function for duplex images."""
        output_dir = Path("data/lagos2duplex/duplex/train")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting duplex scraping with {len(self.duplex_terms)} search terms...")
        
        all_images = []
        
        # Search with each term
        for i, term in enumerate(self.duplex_terms):
            logger.info(f"[{i+1}/{len(self.duplex_terms)}] Searching: {term}")
            
            images = self.search_wikimedia(term, limit=30)
            logger.info(f"  Found {len(images)} images for '{term}'")
            
            all_images.extend(images)
            time.sleep(0.5)  # Rate limiting between searches
        
        # Remove duplicates
        unique_images = {}
        for img in all_images:
            url = img['url']
            if url not in unique_images:
                unique_images[url] = img
        
        all_images = list(unique_images.values())
        logger.info(f"Found {len(all_images)} unique duplex images to download")
        
        # Limit to requested amount
        all_images = all_images[:limit]
        
        # Download images
        successful = 0
        for i, img_data in enumerate(all_images):
            if self.download_image(img_data, output_dir):
                successful += 1
            
            if (i + 1) % 10 == 0:
                logger.info(f"Progress: {i+1}/{len(all_images)} ({successful} successful)")
            
            time.sleep(0.5)  # Rate limiting between downloads
        
        logger.info(f"Duplex scraping complete: {successful}/{len(all_images)} images downloaded")
        return successful

def main():
    parser = argparse.ArgumentParser(description="Duplex-focused scraper for Lagos Nigeria")
    parser.add_argument("--limit", type=int, default=200, help="Maximum images to download")
    
    args = parser.parse_args()
    
    scraper = DuplexFocusedScraper()
    scraper.scrape_duplex_images(args.limit)

if __name__ == "__main__":
    main()
