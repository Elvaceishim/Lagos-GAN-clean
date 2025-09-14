#!/usr/bin/env python3
"""
Enhanced Duplex Scraper - Extended Search Terms
Even more comprehensive search for Lagos Nigeria modern/duplex architecture.
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

class EnhancedDuplexScraper:
    """Enhanced scraper with extensive search terms for Lagos duplex/modern houses."""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Lagos-GAN-Research/1.0 (Educational Research Project)'
        })
        
        # Extensive duplex/modern house terms
        self.extended_terms = [
            # Basic duplex terms
            "Lagos duplex house",
            "Lagos modern house",
            "Lagos contemporary house",
            "Lagos luxury house",
            "Lagos mansion",
            "Lagos villa",
            "Lagos penthouse",
            "Lagos townhouse",
            
            # Area-specific terms
            "Victoria Island house",
            "Ikoyi modern house", 
            "Lekki duplex",
            "Banana Island house",
            "Parkview Estate Lagos",
            "Magodo Lagos house",
            "GRA Ikeja house",
            "Surulere Lagos house",
            "Gbagada Lagos house",
            "Ajah Lagos house",
            
            # Architectural terms
            "Lagos residential architecture",
            "Lagos modern architecture",
            "Lagos contemporary architecture", 
            "Lagos residential building",
            "Lagos apartment building",
            "Lagos condominium",
            "Lagos real estate",
            "Lagos property",
            "Lagos housing",
            "Lagos residential development",
            
            # Nigerian context
            "Nigerian modern house",
            "Nigerian contemporary house",
            "Nigerian duplex",
            "Nigerian luxury house",
            "Nigerian mansion",
            "Nigerian villa",
            "Nigerian residential",
            "Nigerian real estate",
            "Nigerian property",
            "Nigerian housing",
            
            # West African context
            "West African modern house",
            "West African architecture",
            "African modern house",
            "African contemporary house",
            "African residential",
            
            # Specific architectural features
            "Lagos two-story house",
            "Lagos multi-story house",
            "Lagos high-rise residential",
            "Lagos gated community",
            "Lagos estate house",
            "Lagos luxury property",
            "Lagos modern building",
            "Lagos contemporary building",
            "Lagos residential complex",
            "Lagos housing project"
        ]
        
    def search_wikimedia(self, query: str, limit: int = 30) -> list:
        """Search Wikimedia Commons for images."""
        images = []
        
        try:
            search_url = "https://commons.wikimedia.org/w/api.php"
            params = {
                'action': 'query',
                'format': 'json',
                'list': 'search',
                'srsearch': f'filetype:bitmap {query}',
                'srnamespace': 6,
                'srlimit': min(limit, 50),
                'srprop': 'title|snippet'
            }
            
            response = self.session.get(search_url, params=params)
            data = response.json()
            
            if 'query' in data and 'search' in data['query']:
                for result in data['query']['search']:
                    title = result.get('title', '')
                    if title.startswith('File:'):
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
                        
                        time.sleep(0.2)
                        
        except Exception as e:
            logger.error(f"Error searching for '{query}': {e}")
            
        return images
    
    def download_image(self, img_data: dict, output_dir: Path) -> bool:
        """Download and validate image."""
        try:
            response = self.session.get(img_data['url'], timeout=30)
            response.raise_for_status()
            
            image = Image.open(io.BytesIO(response.content))
            
            # Quality filters
            if image.width < 150 or image.height < 150:
                return False
                
            aspect_ratio = max(image.width, image.height) / min(image.width, image.height)
            if aspect_ratio > 4:
                return False
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Generate filename
            safe_title = "".join(c for c in img_data['title'] if c.isalnum() or c in (' ', '-', '_')).rstrip()
            filename = f"enhanced_duplex_{safe_title}_{hash(img_data['url']) % 10000}.jpg"
            filepath = output_dir / filename
            
            image.save(filepath, 'JPEG', quality=85)
            logger.info(f"✓ {filename}")
            return True
            
        except Exception as e:
            logger.error(f"✗ Failed: {img_data.get('title', 'unknown')}: {e}")
            return False
    
    def scrape_enhanced_duplex(self, limit: int = 300) -> int:
        """Enhanced duplex scraping with extensive search terms."""
        output_dir = Path("data/lagos2duplex/duplex/train")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Enhanced duplex scraping with {len(self.extended_terms)} search terms...")
        
        all_images = []
        
        for i, term in enumerate(self.extended_terms):
            logger.info(f"[{i+1}/{len(self.extended_terms)}] Searching: {term}")
            
            images = self.search_wikimedia(term, limit=20)
            logger.info(f"  Found {len(images)} images for '{term}'")
            
            all_images.extend(images)
            time.sleep(0.4)
        
        # Remove duplicates
        unique_images = {}
        for img in all_images:
            url = img['url']
            if url not in unique_images:
                unique_images[url] = img
        
        all_images = list(unique_images.values())
        logger.info(f"Found {len(all_images)} unique enhanced duplex images")
        
        # Limit to requested amount
        all_images = all_images[:limit]
        
        # Download images
        successful = 0
        for i, img_data in enumerate(all_images):
            if self.download_image(img_data, output_dir):
                successful += 1
            
            if (i + 1) % 15 == 0:
                logger.info(f"Progress: {i+1}/{len(all_images)} ({successful} successful)")
            
            time.sleep(0.3)
        
        logger.info(f"Enhanced duplex scraping complete: {successful}/{len(all_images)} images downloaded")
        return successful

def main():
    parser = argparse.ArgumentParser(description="Enhanced duplex scraper")
    parser.add_argument("--limit", type=int, default=300, help="Maximum images to download")
    
    args = parser.parse_args()
    
    scraper = EnhancedDuplexScraper()
    scraper.scrape_enhanced_duplex(args.limit)

if __name__ == "__main__":
    main()
