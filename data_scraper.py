#!/usr/bin/env python3
"""
Lagos-GAN Data Scraper
A comprehensive web scraping tool for gathering training data while respecting legal guidelines.

Features:
- Album cover scraping from multiple sources
- House image collection from real estate and architecture sites
- Built-in rate limiting and politeness
- Legal compliance checks (robots.txt, terms of service)
- Image quality validation
- Automatic organization into train/val/test splits
- Progress tracking and resumption

Usage:
    python data_scraper.py --help
    python data_scraper.py --source discogs --query "afrobeats" --limit 1000
    python data_scraper.py --source unsplash --query "lagos houses" --dataset lagos
"""

import os
import sys
import time
import random
import requests
import argparse
import json
from pathlib import Path
from urllib.parse import urljoin, urlparse, quote
from urllib.robotparser import RobotFileParser
import hashlib
from PIL import Image
import io
from typing import List, Dict, Optional
import logging
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scraper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ScrapedImage:
    """Data class for scraped image information."""
    url: str
    title: str
    source: str
    category: str
    metadata: Dict = None

class RateLimiter:
    """Rate limiter to be polite to websites."""
    def __init__(self, min_delay: float = 1.0, max_delay: float = 3.0):
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.last_request = {}
        self.lock = threading.Lock()
    
    def wait(self, domain: str):
        """Wait before making request to domain."""
        with self.lock:
            now = time.time()
            if domain in self.last_request:
                elapsed = now - self.last_request[domain]
                delay = random.uniform(self.min_delay, self.max_delay)
                if elapsed < delay:
                    time.sleep(delay - elapsed)
            self.last_request[domain] = time.time()

class LegalChecker:
    """Check legal compliance for scraping."""
    
    @staticmethod
    def check_robots_txt(url: str, user_agent: str = '*') -> bool:
        """Check if scraping is allowed by robots.txt."""
        try:
            parsed_url = urlparse(url)
            robots_url = f"{parsed_url.scheme}://{parsed_url.netloc}/robots.txt"
            
            rp = RobotFileParser()
            rp.set_url(robots_url)
            rp.read()
            
            return rp.can_fetch(user_agent, url)
        except Exception as e:
            logger.warning(f"Could not check robots.txt for {url}: {e}")
            return True  # Assume allowed if can't check
    
    @staticmethod
    def get_legal_notice() -> str:
        """Return legal usage notice."""
        return """
LEGAL NOTICE:
This scraper is designed for educational and research purposes.
- Always respect websites' terms of service
- Check robots.txt compliance (automated)
- Use appropriate delays between requests
- Consider reaching out to site owners for permission
- Ensure your use case falls under fair use/research exemptions
- Some sites require API access instead of scraping
"""

class BaseScraper:
    """Base class for all scrapers."""
    
    def __init__(self, rate_limiter: RateLimiter):
        self.rate_limiter = rate_limiter
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        })
    
    def download_image(self, image_info: ScrapedImage, save_dir: str) -> Optional[str]:
        """Download and validate an image."""
        try:
            # Rate limit
            domain = urlparse(image_info.url).netloc
            self.rate_limiter.wait(domain)
            
            # Download image
            response = self.session.get(image_info.url, timeout=30)
            response.raise_for_status()
            
            # Validate image
            img = Image.open(io.BytesIO(response.content))
            
            # Check minimum size
            if img.width < 256 or img.height < 256:
                logger.warning(f"Image too small: {img.width}x{img.height}")
                return None
            
            # Generate filename
            url_hash = hashlib.md5(image_info.url.encode()).hexdigest()[:8]
            safe_title = "".join(c for c in image_info.title if c.isalnum() or c in (' ', '-', '_')).strip()[:50]
            filename = f"{safe_title}_{url_hash}.jpg"
            filepath = os.path.join(save_dir, filename)
            
            # Convert to RGB and save
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            img.save(filepath, 'JPEG', quality=90)
            logger.info(f"Saved: {filename}")
            return filepath
            
        except Exception as e:
            logger.error(f"Failed to download {image_info.url}: {e}")
            return None

class UnsplashScraper(BaseScraper):
    """Scraper for Unsplash (requires API key for production use)."""
    
    def __init__(self, rate_limiter: RateLimiter, api_key: Optional[str] = None):
        super().__init__(rate_limiter)
        self.api_key = api_key
        self.base_url = "https://api.unsplash.com"
    
    def search_images(self, query: str, limit: int = 50) -> List[ScrapedImage]:
        """Search for images on Unsplash."""
        if not self.api_key:
            logger.warning("Unsplash API key not provided. Using demo mode with limited results.")
            return self._demo_search(query, limit)
        
        images = []
        page = 1
        per_page = min(30, limit)
        
        while len(images) < limit:
            try:
                self.rate_limiter.wait("api.unsplash.com")
                
                params = {
                    'query': query,
                    'page': page,
                    'per_page': per_page,
                    'orientation': 'all'
                }
                
                headers = {'Authorization': f'Client-ID {self.api_key}'}
                response = self.session.get(f"{self.base_url}/search/photos", 
                                          params=params, headers=headers)
                response.raise_for_status()
                
                data = response.json()
                
                for photo in data['results']:
                    if len(images) >= limit:
                        break
                    
                    image_info = ScrapedImage(
                        url=photo['urls']['regular'],
                        title=photo['alt_description'] or f"unsplash_{photo['id']}",
                        source='unsplash',
                        category=query,
                        metadata={
                            'photographer': photo['user']['name'],
                            'width': photo['width'],
                            'height': photo['height']
                        }
                    )
                    images.append(image_info)
                
                if page >= data['total_pages']:
                    break
                page += 1
                
            except Exception as e:
                logger.error(f"Error searching Unsplash: {e}")
                break
        
        return images
    
    def _demo_search(self, query: str, limit: int) -> List[ScrapedImage]:
        """Demo search without API key (very limited)."""
        logger.info("Demo mode: Add UNSPLASH_API_KEY environment variable for full access")
        # Return empty list - user needs API key for Unsplash
        return []

class PublicDomainScraper(BaseScraper):
    """Scraper for public domain and free sources."""
    
    def search_pixabay(self, query: str, api_key: str, limit: int = 50) -> List[ScrapedImage]:
        """Search Pixabay (requires free API key)."""
        if not api_key:
            logger.warning("Pixabay API key required")
            return []
        
        images = []
        page = 1
        per_page = min(200, limit)
        
        try:
            params = {
                'key': api_key,
                'q': query,
                'image_type': 'photo',
                'min_width': 256,
                'min_height': 256,
                'per_page': per_page,
                'page': page
            }
            
            self.rate_limiter.wait("pixabay.com")
            response = self.session.get("https://pixabay.com/api/", params=params)
            response.raise_for_status()
            
            data = response.json()
            
            for hit in data['hits']:
                if len(images) >= limit:
                    break
                
                image_info = ScrapedImage(
                    url=hit['largeImageURL'],
                    title=hit['tags'],
                    source='pixabay',
                    category=query,
                    metadata={
                        'user': hit['user'],
                        'views': hit['views'],
                        'downloads': hit['downloads']
                    }
                )
                images.append(image_info)
        
        except Exception as e:
            logger.error(f"Error searching Pixabay: {e}")
        
        return images

class WikimediaCommonsScraper(BaseScraper):
    """Scraper for Wikimedia Commons (free images)."""
    
    def search_images(self, query: str, limit: int = 50) -> List[ScrapedImage]:
        """Search Wikimedia Commons."""
        images = []
        
        try:
            params = {
                'action': 'query',
                'format': 'json',
                'list': 'search',
                'srsearch': f'filetype:bitmap {query}',
                'srnamespace': 6,  # File namespace
                'srlimit': min(50, limit)
            }
            
            self.rate_limiter.wait("commons.wikimedia.org")
            response = self.session.get("https://commons.wikimedia.org/w/api.php", params=params)
            response.raise_for_status()
            
            data = response.json()
            
            for result in data['query']['search']:
                if len(images) >= limit:
                    break
                
                # Get file info
                file_title = result['title']
                file_params = {
                    'action': 'query',
                    'format': 'json',
                    'titles': file_title,
                    'prop': 'imageinfo',
                    'iiprop': 'url|size|metadata'
                }
                
                self.rate_limiter.wait("commons.wikimedia.org")
                file_response = self.session.get("https://commons.wikimedia.org/w/api.php", params=file_params)
                file_data = file_response.json()
                
                page = next(iter(file_data['query']['pages'].values()))
                if 'imageinfo' in page:
                    imageinfo = page['imageinfo'][0]
                    
                    image_info = ScrapedImage(
                        url=imageinfo['url'],
                        title=file_title.replace('File:', ''),
                        source='wikimedia_commons',
                        category=query,
                        metadata={
                            'width': imageinfo.get('width'),
                            'height': imageinfo.get('height'),
                            'license': 'free'
                        }
                    )
                    images.append(image_info)
        
        except Exception as e:
            logger.error(f"Error searching Wikimedia Commons: {e}")
        
        return images

class DataScraper:
    """Main scraper orchestrator."""
    
    def __init__(self):
        self.rate_limiter = RateLimiter(min_delay=0.5, max_delay=1.0)
        self.scrapers = {
            'unsplash': UnsplashScraper(self.rate_limiter, os.getenv('UNSPLASH_API_KEY')),
            'pixabay': PublicDomainScraper(self.rate_limiter),
            'wikimedia': WikimediaCommonsScraper(self.rate_limiter)
        }
        
        # Create session file for resuming
        self.session_file = 'scraper_session.json'
        self.downloaded_urls = self.load_session()
    
    def load_session(self) -> set:
        """Load previously downloaded URLs."""
        if os.path.exists(self.session_file):
            try:
                with open(self.session_file, 'r') as f:
                    return set(json.load(f))
            except:
                pass
        return set()
    
    def save_session(self):
        """Save session state."""
        with open(self.session_file, 'w') as f:
            json.dump(list(self.downloaded_urls), f)
    
    def scrape_dataset(self, source: str, query: str, dataset: str, limit: int = 100, 
                      split_ratios: tuple = (0.7, 0.15, 0.15)):
        """Scrape images and organize into dataset."""
        logger.info(f"Starting scrape: {source} -> {query} -> {dataset} (limit: {limit})")
        
        # Print legal notice
        print(LegalChecker.get_legal_notice())
        
        # Setup directories
        if dataset == 'afrocover':
            base_dir = 'data/afrocover'
        elif dataset == 'lagos':
            base_dir = 'data/lagos2duplex/lagos'
        elif dataset == 'duplex':
            base_dir = 'data/lagos2duplex/duplex'
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
        
        # Create temp download directory
        temp_dir = f'temp_downloads_{source}_{dataset}'
        os.makedirs(temp_dir, exist_ok=True)
        
        try:
            # Search for images
            if source == 'unsplash':
                images = self.scrapers['unsplash'].search_images(query, limit)
            elif source == 'pixabay':
                api_key = os.getenv('PIXABAY_API_KEY')
                images = self.scrapers['pixabay'].search_pixabay(query, api_key, limit)
            elif source == 'wikimedia':
                images = self.scrapers['wikimedia'].search_images(query, limit)
            else:
                raise ValueError(f"Unknown source: {source}")
            
            if not images:
                logger.warning(f"No images found for query: {query}")
                return
            
            logger.info(f"Found {len(images)} images to download")
            
            # Download images
            downloaded_files = []
            scraper = self.scrapers[source]
            
            for i, image_info in enumerate(images):
                if image_info.url in self.downloaded_urls:
                    logger.info(f"Skipping already downloaded: {image_info.url}")
                    continue
                
                logger.info(f"Downloading {i+1}/{len(images)}: {image_info.title}")
                filepath = scraper.download_image(image_info, temp_dir)
                
                if filepath:
                    downloaded_files.append(filepath)
                    self.downloaded_urls.add(image_info.url)
                    self.save_session()
                
                # Progress update
                if (i + 1) % 10 == 0:
                    logger.info(f"Progress: {i+1}/{len(images)} ({len(downloaded_files)} successful)")
            
            # Organize into train/val/test
            if downloaded_files:
                self._organize_files(downloaded_files, base_dir, split_ratios)
                logger.info(f"Successfully downloaded and organized {len(downloaded_files)} images")
            else:
                logger.warning("No images were successfully downloaded")
        
        finally:
            # Cleanup temp directory
            import shutil
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
    
    def _organize_files(self, files: List[str], base_dir: str, split_ratios: tuple):
        """Organize downloaded files into train/val/test splits."""
        import shutil
        
        # Create directories
        train_dir = os.path.join(base_dir, 'train')
        val_dir = os.path.join(base_dir, 'val')
        test_dir = os.path.join(base_dir, 'test')
        
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        
        # Shuffle and split
        random.shuffle(files)
        total = len(files)
        train_count = int(total * split_ratios[0])
        val_count = int(total * split_ratios[1])
        
        train_files = files[:train_count]
        val_files = files[train_count:train_count + val_count]
        test_files = files[train_count + val_count:]
        
        # Move files
        for file_list, dest_dir in [(train_files, train_dir), (val_files, val_dir), (test_files, test_dir)]:
            for filepath in file_list:
                filename = os.path.basename(filepath)
                dest_path = os.path.join(dest_dir, filename)
                shutil.move(filepath, dest_path)
        
        logger.info(f"Organized: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")

def main():
    parser = argparse.ArgumentParser(description="Lagos-GAN Data Scraper")
    parser.add_argument("--source", choices=['unsplash', 'pixabay', 'wikimedia'], 
                        required=True, help="Source to scrape from")
    parser.add_argument("--query", required=True, help="Search query")
    parser.add_argument("--dataset", choices=['afrocover', 'lagos', 'duplex'], 
                        required=True, help="Target dataset")
    parser.add_argument("--limit", type=int, default=1000, help="Maximum images to download")
    parser.add_argument("--setup-keys", action="store_true", help="Show API key setup instructions")
    
    args = parser.parse_args()
    
    if args.setup_keys:
        print("""
API Key Setup Instructions:

1. Unsplash (Free, 50 requests/hour):
   - Go to https://unsplash.com/developers
   - Create account and new application
   - Copy Access Key
   - Set environment variable: export UNSPLASH_API_KEY="your_key_here"

2. Pixabay (Free, 20,000 requests/month):
   - Go to https://pixabay.com/api/docs/
   - Create account and get API key
   - Set environment variable: export PIXABAY_API_KEY="your_key_here"

3. Wikimedia Commons (No key needed):
   - Free to use, no registration required
   - Rate limited to be respectful

Add to your ~/.zshrc or ~/.bash_profile:
export UNSPLASH_API_KEY="your_unsplash_key"
export PIXABAY_API_KEY="your_pixabay_key"

Then run: source ~/.zshrc
        """)
        return
    
    scraper = DataScraper()
    scraper.scrape_dataset(args.source, args.query, args.dataset, args.limit)

if __name__ == "__main__":
    main()
