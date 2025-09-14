#!/usr/bin/env python3
"""
Nigerian Sources Scraper
Specialized scraper for Nigerian music sites and real estate platforms.

IMPORTANT LEGAL NOTICE:
This scraper is for educational/research purposes only.
- Always check robots.txt and terms of service
- Some sites may require explicit permission
- Consider contacting site owners for research use
- Respect rate limits and don't overload servers
- Use responsibly and ethically

Sources covered:
- NotJustOk (Nigerian music)
- 360Nobs (Nigerian entertainment)
- PropertyPro.ng (Real estate)
- Jiji.ng (Real estate section)
"""

import os
import time
import random
import requests
import json
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from pathlib import Path
import logging
from data_scraper import BaseScraper, ScrapedImage, RateLimiter, LegalChecker
import argparse
from typing import List
import re

logger = logging.getLogger(__name__)

class NigerianMusicScraper(BaseScraper):
    """Scraper for Nigerian music websites."""
    
    def __init__(self, rate_limiter: RateLimiter):
        super().__init__(rate_limiter)
        # Optimized rate limiting for faster collection
        self.rate_limiter.min_delay = 0.5
        self.rate_limiter.max_delay = 1.5
    
    def scrape_notjustok_demo(self, limit: int = 50) -> List[ScrapedImage]:
        """
        Demo scraper for NotJustOk structure.
        NOTE: This is educational - actual implementation would need permission.
        """
        logger.warning("DEMO MODE: This shows the structure for educational purposes")
        logger.warning("For actual use, contact NotJustOk for permission or use their API if available")
        
        # This is just to show the structure - don't actually scrape without permission
        demo_images = []
        
        # Simulated album cover data that would come from NotJustOk
        demo_albums = [
            {
                'title': 'Davido - Timeless',
                'image_url': 'https://example.com/davido-timeless.jpg',
                'artist': 'Davido',
                'genre': 'Afrobeats'
            },
            {
                'title': 'Burna Boy - Love Damini',
                'image_url': 'https://example.com/burna-love-damini.jpg',
                'artist': 'Burna Boy',
                'genre': 'Afrobeats'
            }
        ]
        
        for album in demo_albums[:limit]:
            image_info = ScrapedImage(
                url=album['image_url'],
                title=album['title'],
                source='notjustok_demo',
                category='nigerian_music',
                metadata={
                    'artist': album['artist'],
                    'genre': album['genre']
                }
            )
            demo_images.append(image_info)
        
        logger.info(f"Generated {len(demo_images)} demo entries")
        return demo_images
    
    def get_implementation_guide(self) -> str:
        """Return implementation guide for Nigerian music sites."""
        return """
IMPLEMENTATION GUIDE FOR NIGERIAN MUSIC SITES:

1. NotJustOk (notjustok.com):
   - Contact: Use their contact form for research permission
   - API: Check if they have a public API
   - Structure: Look for album art in article pages and music reviews
   - Politeness: Use 3-5 second delays between requests

2. 360Nobs (360nobs.com):
   - Contact: Reach out for educational use permission
   - Focus: Entertainment news with album covers
   - Structure: Article pages with embedded album art

3. Alternative Approaches:
   - Spotify Web API (requires registration)
   - Apple Music API (requires developer account)
   - Last.fm API (free, good for African music)
   - MusicBrainz API (open source music database)

RECOMMENDED APPROACH:
Use official APIs instead of scraping:

# Spotify Web API example
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

client_credentials_manager = SpotifyClientCredentials(
    client_id="your_client_id",
    client_secret="your_client_secret"
)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# Search for Afrobeats albums
results = sp.search(q='genre:afrobeats', type='album', limit=50)
for album in results['albums']['items']:
    album_art_url = album['images'][0]['url']
    # Download album art legally
"""

class NigerianRealEstateScraper(BaseScraper):
    """Scraper for Nigerian real estate websites."""
    
    def __init__(self, rate_limiter: RateLimiter):
        super().__init__(rate_limiter)
        self.rate_limiter.min_delay = 3.0
        self.rate_limiter.max_delay = 7.0
    
    def scrape_propertypro_demo(self, property_type: str = "duplex", limit: int = 50) -> List[ScrapedImage]:
        """
        Demo scraper for PropertyPro.ng structure.
        NOTE: Educational demonstration only.
        """
        logger.warning("DEMO MODE: Educational structure demonstration")
        logger.warning("For actual use, contact PropertyPro.ng for research permission")
        
        # Demo structure showing what PropertyPro listings might contain
        demo_properties = [
            {
                'title': 'Modern 4-Bedroom Duplex in Victoria Island',
                'image_url': 'https://example.com/vi-duplex-1.jpg',
                'location': 'Victoria Island, Lagos',
                'type': 'duplex',
                'price': '₦250,000,000'
            },
            {
                'title': 'Luxury 5-Bedroom Duplex in Ikoyi',
                'image_url': 'https://example.com/ikoyi-duplex-2.jpg',
                'location': 'Ikoyi, Lagos',
                'type': 'duplex',
                'price': '₦180,000,000'
            },
            {
                'title': 'Traditional Bungalow in Surulere',
                'image_url': 'https://example.com/surulere-bungalow.jpg',
                'location': 'Surulere, Lagos',
                'type': 'bungalow',
                'price': '₦45,000,000'
            }
        ]
        
        filtered_properties = [p for p in demo_properties if property_type.lower() in p['type'].lower()]
        
        demo_images = []
        for prop in filtered_properties[:limit]:
            image_info = ScrapedImage(
                url=prop['image_url'],
                title=prop['title'],
                source='propertypro_demo',
                category=f'nigerian_{property_type}',
                metadata={
                    'location': prop['location'],
                    'type': prop['type'],
                    'price': prop['price']
                }
            )
            demo_images.append(image_info)
        
        logger.info(f"Generated {len(demo_images)} demo property entries")
        return demo_images
    
    def get_real_estate_guide(self) -> str:
        """Return guide for real estate scraping."""
        return """
NIGERIAN REAL ESTATE SCRAPING GUIDE:

1. PropertyPro.ng:
   - Contact: research@propertypro.ng for academic use
   - API: Check if they offer API access
   - Focus: Duplex and modern homes in Lagos
   - Rate limit: 3-5 seconds between requests

2. Jiji.ng (Real Estate Section):
   - Contact: Contact form for research permission
   - Structure: Property listings with multiple images
   - Focus: Both traditional and modern homes

3. Other Nigerian Real Estate Sites:
   - Nigeria-Property.com
   - PrivateProperty.com.ng
   - ToLet.com.ng

LEGAL ALTERNATIVE APPROACHES:

1. Google Images API (Paid):
   - Search: "Lagos duplex houses", "Nigerian bungalow"
   - Legal: Proper attribution and usage rights

2. Flickr API (Free):
   - Search: Creative Commons licensed photos
   - Tags: "Lagos architecture", "Nigerian houses"

3. Unsplash/Pixabay:
   - Search: "African architecture", "Lagos buildings"
   - License: Free to use

4. Government/Academic Sources:
   - Lagos State Ministry of Housing
   - Urban planning research papers
   - Architectural firm portfolios (with permission)

RECOMMENDED IMPLEMENTATION:
1. Contact sites directly for research permission
2. Use official APIs where available
3. Focus on Creative Commons/open licensed content
4. Consider partnering with local photographers
"""

class LegalMusicAPIScraper:
    """Legal music API scraper using official APIs."""
    
    def __init__(self):
        self.session = requests.Session()
    
    def setup_spotify_api(self, client_id: str, client_secret: str):
        """Setup Spotify API access."""
        try:
            import spotipy
            from spotipy.oauth2 import SpotifyClientCredentials
            
            client_credentials_manager = SpotifyClientCredentials(
                client_id=client_id,
                client_secret=client_secret
            )
            self.spotify = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
            return True
        except ImportError:
            logger.error("Install spotipy: pip install spotipy")
            return False
        except Exception as e:
            logger.error(f"Spotify API setup failed: {e}")
            return False
    
    def search_spotify_albums(self, query: str, limit: int = 50) -> List[ScrapedImage]:
        """Search Spotify for album covers."""
        if not hasattr(self, 'spotify'):
            logger.error("Spotify API not set up. Call setup_spotify_api() first.")
            return []
        
        try:
            results = self.spotify.search(q=query, type='album', limit=limit, market='NG')
            images = []
            
            for album in results['albums']['items']:
                if album['images']:
                    # Get highest quality image
                    image_url = album['images'][0]['url']
                    
                    image_info = ScrapedImage(
                        url=image_url,
                        title=f"{album['artists'][0]['name']} - {album['name']}",
                        source='spotify_api',
                        category='album_cover',
                        metadata={
                            'artist': album['artists'][0]['name'],
                            'album': album['name'],
                            'release_date': album['release_date'],
                            'total_tracks': album['total_tracks']
                        }
                    )
                    images.append(image_info)
            
            return images
            
        except Exception as e:
            logger.error(f"Spotify search failed: {e}")
            return []
    
    def search_lastfm_albums(self, query: str, api_key: str, limit: int = 50) -> List[ScrapedImage]:
        """Search Last.fm for album covers."""
        try:
            params = {
                'method': 'album.search',
                'album': query,
                'api_key': api_key,
                'format': 'json',
                'limit': limit
            }
            
            response = self.session.get('http://ws.audioscrobbler.com/2.0/', params=params)
            response.raise_for_status()
            data = response.json()
            
            images = []
            
            if 'results' in data and 'albummatches' in data['results']:
                for album in data['results']['albummatches']['album']:
                    # Get largest image
                    image_url = None
                    for img in album['image']:
                        if img['size'] == 'extralarge' or img['size'] == 'large':
                            image_url = img['#text']
                            break
                    
                    if image_url:
                        image_info = ScrapedImage(
                            url=image_url,
                            title=f"{album['artist']} - {album['name']}",
                            source='lastfm_api',
                            category='album_cover',
                            metadata={
                                'artist': album['artist'],
                                'album': album['name'],
                                'mbid': album.get('mbid', '')
                            }
                        )
                        images.append(image_info)
            
            return images
            
        except Exception as e:
            logger.error(f"Last.fm search failed: {e}")
            return []

def main():
    parser = argparse.ArgumentParser(description="Nigerian Sources Scraper (Educational)")
    parser.add_argument("--source", choices=['demo_music', 'demo_realestate', 'spotify', 'lastfm'], 
                        required=True, help="Source to demonstrate")
    parser.add_argument("--query", help="Search query")
    parser.add_argument("--limit", type=int, default=1000, help="Limit results")
    parser.add_argument("--show-guides", action="store_true", help="Show implementation guides")
    parser.add_argument("--setup-apis", action="store_true", help="Show API setup instructions")
    
    args = parser.parse_args()
    
    if args.show_guides:
        rate_limiter = RateLimiter()
        music_scraper = NigerianMusicScraper(rate_limiter)
        realestate_scraper = NigerianRealEstateScraper(rate_limiter)
        
        print("=== NIGERIAN MUSIC SCRAPING GUIDE ===")
        print(music_scraper.get_implementation_guide())
        print("\n=== NIGERIAN REAL ESTATE SCRAPING GUIDE ===")
        print(realestate_scraper.get_real_estate_guide())
        return
    
    if args.setup_apis:
        print("""
API SETUP FOR LEGAL MUSIC SCRAPING:

1. SPOTIFY WEB API (Recommended):
   - Go to https://developer.spotify.com/dashboard
   - Create new app
   - Get Client ID and Client Secret
   - Set environment variables:
     export SPOTIFY_CLIENT_ID="your_client_id"
     export SPOTIFY_CLIENT_SECRET="your_client_secret"
   - Install: pip install spotipy

2. LAST.FM API (Free):
   - Go to https://www.last.fm/api/account/create
   - Get API key
   - Set environment variable:
     export LASTFM_API_KEY="your_api_key"

3. MUSICBRAINZ (No key needed):
   - Open source music database
   - Free to use with rate limiting
   - Good for album metadata

Usage examples:
python nigerian_scraper.py --source spotify --query "afrobeats"
python nigerian_scraper.py --source lastfm --query "davido"
        """)
        return
    
    # Demo implementations
    if args.source == 'demo_music':
        rate_limiter = RateLimiter()
        scraper = NigerianMusicScraper(rate_limiter)
        images = scraper.scrape_notjustok_demo(args.limit)
        print(f"Demo generated {len(images)} music entries")
        
    elif args.source == 'demo_realestate':
        rate_limiter = RateLimiter()
        scraper = NigerianRealEstateScraper(rate_limiter)
        images = scraper.scrape_propertypro_demo("duplex", args.limit)
        print(f"Demo generated {len(images)} real estate entries")
        
    elif args.source == 'spotify':
        if not args.query:
            print("Please provide --query for Spotify search")
            return
            
        client_id = os.getenv('SPOTIFY_CLIENT_ID')
        client_secret = os.getenv('SPOTIFY_CLIENT_SECRET')
        
        if not client_id or not client_secret:
            print("Please set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET environment variables")
            return
        
        scraper = LegalMusicAPIScraper()
        if scraper.setup_spotify_api(client_id, client_secret):
            images = scraper.search_spotify_albums(args.query, args.limit)
            print(f"Found {len(images)} albums on Spotify")
        
    elif args.source == 'lastfm':
        if not args.query:
            print("Please provide --query for Last.fm search")
            return
            
        api_key = os.getenv('LASTFM_API_KEY')
        if not api_key:
            print("Please set LASTFM_API_KEY environment variable")
            return
        
        scraper = LegalMusicAPIScraper()
        images = scraper.search_lastfm_albums(args.query, api_key, args.limit)
        print(f"Found {len(images)} albums on Last.fm")

if __name__ == "__main__":
    main()
