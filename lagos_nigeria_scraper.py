#!/usr/bin/env python3
"""
Lagos, Nigeria Specific Scraper
Focused scraper for Lagos, Nigeria architecture and Nigerian/African album covers.

This scraper ensures we get:
1. Lagos, Nigeria traditional houses (not Portugal Lagos)
2. Lagos, Nigeria modern duplex houses 
3. Nigerian/African/Afrobeats album covers
4. West African architectural styles

Search strategies:
- Always include "Nigeria" in architecture searches
- Use Nigerian area names (Victoria Island, Ikoyi, Lekki, etc.)
- Focus on Afrobeats, Nigerian music, West African artists
- Use Nigerian architectural terms
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
import logging
from data_scraper import DataScraper, BaseScraper, ScrapedImage, RateLimiter
from typing import List, Dict

logger = logging.getLogger(__name__)

class LagosNigeriaArchitectureScraper:
    """Specialized scraper for Lagos, Nigeria architecture"""
    
    def __init__(self):
        self.base_scraper = DataScraper()
        self.lagos_specific_queries = [
            # Traditional Lagos, Nigeria houses
            "Lagos Nigeria traditional house",
            "Lagos Nigeria vernacular architecture", 
            "Nigerian traditional architecture Lagos",
            "Lagos Nigeria compound house",
            "Lagos Nigeria residential architecture",
            "Yoruba architecture Lagos Nigeria",
            "Lagos Nigeria colonial architecture",
            "Lagos Nigeria bungalow house",
            "traditional Nigerian house Lagos state",
            "Lagos Nigeria indigenous architecture",
            
            # Lagos neighborhoods and areas
            "Victoria Island Lagos Nigeria house",
            "Ikoyi Lagos Nigeria architecture",
            "Surulere Lagos Nigeria residential",
            "Ikeja Lagos Nigeria building",
            "Lekki Lagos Nigeria house",
            "Lagos Island Nigeria architecture",
            "Yaba Lagos Nigeria residential",
            "Mushin Lagos Nigeria house",
            "Apapa Lagos Nigeria building",
            "Mainland Lagos Nigeria architecture",
            
            # Modern Lagos, Nigeria duplexes
            "Lagos Nigeria modern duplex house",
            "Lagos Nigeria contemporary duplex",
            "Nigerian duplex house Lagos",
            "Lagos Nigeria residential duplex",
            "modern house Lagos Nigeria duplex",
            "Lagos Nigeria luxury duplex",
            "Nigerian modern architecture Lagos duplex",
            "Lagos Nigeria estate duplex house",
            "contemporary Nigerian duplex Lagos",
            "Lagos Nigeria suburban duplex"
        ]
        
        self.afro_album_queries = [
            # Nigerian music specifically
            "Nigerian album cover",
            "Afrobeats album cover",
            "Nigerian music album art",
            "Afrobeats album artwork",
            "Nigerian artist album cover",
            "West African album cover",
            "Lagos music album cover",
            "Nigerian hip hop album cover",
            "Afrofusion album artwork",
            "Nigerian R&B album cover",
            
            # Specific Nigerian artists (if available)
            "Burna Boy album cover",
            "Wizkid album cover", 
            "Davido album cover",
            "Tiwa Savage album cover",
            "Yemi Alade album cover",
            "Mr Eazi album cover",
            "Tekno album cover",
            "Runtown album cover",
            "Nigerian Afrobeats compilation",
            "West African music compilation"
        ]
    
    def scrape_lagos_traditional(self, limit: int = 500):
        """Scrape traditional Lagos, Nigeria houses"""
        print("🏠 Scraping Lagos, Nigeria Traditional Houses...")
        
        total_collected = 0
        for query in self.lagos_specific_queries[:10]:  # Traditional queries
            if total_collected >= limit:
                break
                
            remaining = min(100, limit - total_collected)
            print(f"  🔍 Query: '{query}' (limit: {remaining})")
            
            try:
                # Try multiple sources
                for source in ['wikimedia', 'unsplash']:
                    if total_collected >= limit:
                        break
                    
                    batch_limit = min(50, remaining)
                    success = self._run_scraper(source, query, 'lagos', batch_limit)
                    if success:
                        total_collected += batch_limit
                        time.sleep(2)  # Brief pause between sources
                        
            except Exception as e:
                logger.error(f"Error with query '{query}': {e}")
                continue
        
        print(f"✅ Collected {total_collected} Lagos, Nigeria traditional house images")
        return total_collected
    
    def scrape_lagos_duplex(self, limit: int = 500):
        """Scrape modern Lagos, Nigeria duplex houses"""
        print("🏢 Scraping Lagos, Nigeria Modern Duplexes...")
        
        total_collected = 0
        for query in self.lagos_specific_queries[10:]:  # Modern duplex queries
            if total_collected >= limit:
                break
                
            remaining = min(100, limit - total_collected)
            print(f"  🔍 Query: '{query}' (limit: {remaining})")
            
            try:
                for source in ['wikimedia', 'unsplash']:
                    if total_collected >= limit:
                        break
                    
                    batch_limit = min(50, remaining)
                    success = self._run_scraper(source, query, 'duplex', batch_limit)
                    if success:
                        total_collected += batch_limit
                        time.sleep(2)
                        
            except Exception as e:
                logger.error(f"Error with query '{query}': {e}")
                continue
        
        print(f"✅ Collected {total_collected} Lagos, Nigeria duplex images")
        return total_collected
    
    def scrape_afro_albums(self, limit: int = 500):
        """Scrape Afro-centric and Nigerian album covers"""
        print("🎵 Scraping Nigerian/Afrobeats Album Covers...")
        
        total_collected = 0
        for query in self.afro_album_queries:
            if total_collected >= limit:
                break
                
            remaining = min(100, limit - total_collected)
            print(f"  🔍 Query: '{query}' (limit: {remaining})")
            
            try:
                for source in ['wikimedia', 'unsplash']:
                    if total_collected >= limit:
                        break
                    
                    batch_limit = min(50, remaining)
                    success = self._run_scraper(source, query, 'afrocover', batch_limit)
                    if success:
                        total_collected += batch_limit
                        time.sleep(2)
                        
            except Exception as e:
                logger.error(f"Error with query '{query}': {e}")
                continue
        
        print(f"✅ Collected {total_collected} Nigerian/Afrobeats album covers")
        return total_collected
    
    def _run_scraper(self, source: str, query: str, dataset: str, limit: int) -> bool:
        """Run the base scraper with specific parameters"""
        try:
            import subprocess
            import os
            
            # Load environment variables
            env = os.environ.copy()
            if os.path.exists('api_keys.env'):
                with open('api_keys.env', 'r') as f:
                    for line in f:
                        if '=' in line and not line.startswith('#'):
                            key, value = line.strip().split('=', 1)
                            env[key] = value
            
            cmd = [
                'python', 'data_scraper.py',
                '--source', source,
                '--query', query,
                '--dataset', dataset,
                '--limit', str(limit)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, env=env)
            
            if result.returncode == 0:
                # Check if we actually got images
                if "Successfully downloaded and organized" in result.stdout:
                    return True
                else:
                    print(f"    ⚠️  No images found for '{query}' on {source}")
                    return False
            else:
                print(f"    ❌ Error scraping '{query}' on {source}: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"    ❌ Exception scraping '{query}': {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description="Lagos, Nigeria Specific Scraper")
    parser.add_argument("--dataset", choices=['traditional', 'duplex', 'afrocover', 'all'], 
                        default='all', help="Which dataset to scrape")
    parser.add_argument("--limit", type=int, default=500, help="Images per dataset")
    parser.add_argument("--clear-generic", action="store_true", 
                        help="Clear existing generic (non-Lagos) data first")
    
    args = parser.parse_args()
    
    if args.clear_generic:
        print("🧹 Clearing existing generic data...")
        import shutil
        
        # Backup current data
        backup_dir = f"data_backup_{int(time.time())}"
        if os.path.exists('data'):
            shutil.copytree('data', backup_dir)
            print(f"   📦 Backed up to {backup_dir}")
        
        # Clear and recreate
        if os.path.exists('data'):
            shutil.rmtree('data')
        
        # Recreate structure
        for dataset in ['afrocover', 'lagos2duplex/lagos', 'lagos2duplex/duplex']:
            for split in ['train', 'val', 'test']:
                os.makedirs(f'data/{dataset}/{split}', exist_ok=True)
        
        print("   ✅ Cleared generic data, ready for Lagos-specific scraping")
    
    scraper = LagosNigeriaArchitectureScraper()
    
    total_images = 0
    
    if args.dataset in ['traditional', 'all']:
        total_images += scraper.scrape_lagos_traditional(args.limit)
    
    if args.dataset in ['duplex', 'all']:
        total_images += scraper.scrape_lagos_duplex(args.limit)
        
    if args.dataset in ['afrocover', 'all']:
        total_images += scraper.scrape_afro_albums(args.limit)
    
    print(f"\n🎉 TOTAL COLLECTED: {total_images} Lagos, Nigeria specific images!")
    print("\n📊 Final dataset status:")
    
    # Count final images
    for dataset, path in [
        ("AfroCover", "data/afrocover"),
        ("Lagos Traditional", "data/lagos2duplex/lagos"), 
        ("Lagos Duplex", "data/lagos2duplex/duplex")
    ]:
        if os.path.exists(path):
            count = sum(len([f for f in os.listdir(f"{path}/{split}") if f.endswith('.jpg')]) 
                       for split in ['train', 'val', 'test'] if os.path.exists(f"{path}/{split}"))
            print(f"   {dataset}: {count} images")

if __name__ == "__main__":
    main()
