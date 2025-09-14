#!/usr/bin/env python3
"""
Complete Data Collection Setup for Lagos-GAN
This script sets up everything needed for legal, ethical data collection.
"""

import os
import subprocess
import sys
from pathlib import Path

def install_scraping_requirements():
    """Install additional requirements for scraping tools."""
    print("Installing scraping requirements...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "-r", "scraping_requirements.txt"
        ])
        print("✅ Scraping requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install scraping requirements: {e}")
        return False

def setup_api_keys():
    """Guide user through API key setup."""
    print("\n" + "="*60)
    print("API KEY SETUP GUIDE")
    print("="*60)
    
    api_keys = {
        'UNSPLASH_API_KEY': {
            'name': 'Unsplash',
            'url': 'https://unsplash.com/developers',
            'description': 'Free tier: 50 requests/hour',
            'instructions': [
                '1. Go to https://unsplash.com/developers',
                '2. Create account and new application',
                '3. Copy Access Key',
                '4. Set environment variable: export UNSPLASH_API_KEY="your_key"'
            ]
        },
        'PIXABAY_API_KEY': {
            'name': 'Pixabay',
            'url': 'https://pixabay.com/api/docs/',
            'description': 'Free tier: 20,000 requests/month',
            'instructions': [
                '1. Go to https://pixabay.com/api/docs/',
                '2. Create account and get API key',
                '3. Set environment variable: export PIXABAY_API_KEY="your_key"'
            ]
        },
        'SPOTIFY_CLIENT_ID': {
            'name': 'Spotify Web API',
            'url': 'https://developer.spotify.com/dashboard',
            'description': 'Free tier: Good for album art',
            'instructions': [
                '1. Go to https://developer.spotify.com/dashboard',
                '2. Create new app',
                '3. Get Client ID and Client Secret',
                '4. Set variables:',
                '   export SPOTIFY_CLIENT_ID="your_client_id"',
                '   export SPOTIFY_CLIENT_SECRET="your_client_secret"'
            ]
        },
        'LASTFM_API_KEY': {
            'name': 'Last.fm API',
            'url': 'https://www.last.fm/api/account/create',
            'description': 'Free tier: Good for music data',
            'instructions': [
                '1. Go to https://www.last.fm/api/account/create',
                '2. Get API key',
                '3. Set environment variable: export LASTFM_API_KEY="your_key"'
            ]
        }
    }
    
    for key_name, info in api_keys.items():
        current_value = os.getenv(key_name)
        status = "✅ SET" if current_value else "❌ NOT SET"
        
        print(f"\n{info['name']} ({key_name}): {status}")
        print(f"Description: {info['description']}")
        print(f"URL: {info['url']}")
        
        if not current_value:
            print("Setup instructions:")
            for instruction in info['instructions']:
                print(f"  {instruction}")

def create_env_file():
    """Create sample .env file for API keys."""
    env_content = """# API Keys for Lagos-GAN Data Scraping
# Copy this to your shell profile (~/.zshrc, ~/.bash_profile) or use with python-dotenv

# Unsplash API (Free: 50 requests/hour)
# Get from: https://unsplash.com/developers
export UNSPLASH_API_KEY="your_unsplash_access_key_here"

# Pixabay API (Free: 20,000 requests/month)  
# Get from: https://pixabay.com/api/docs/
export PIXABAY_API_KEY="your_pixabay_api_key_here"

# Spotify Web API (Free tier available)
# Get from: https://developer.spotify.com/dashboard
export SPOTIFY_CLIENT_ID="your_spotify_client_id_here"
export SPOTIFY_CLIENT_SECRET="your_spotify_client_secret_here"

# Last.fm API (Free)
# Get from: https://www.last.fm/api/account/create
export LASTFM_API_KEY="your_lastfm_api_key_here"

# Usage:
# 1. Replace "your_*_here" with actual API keys
# 2. Add to your shell profile: cat api_keys.env >> ~/.zshrc
# 3. Reload shell: source ~/.zshrc
# 4. Or load in Python: from dotenv import load_dotenv; load_dotenv('api_keys.env')
"""
    
    with open('api_keys.env', 'w') as f:
        f.write(env_content)
    
    print(f"\n✅ Created api_keys.env template file")
    print("Edit this file with your actual API keys, then:")
    print("  cat api_keys.env >> ~/.zshrc")
    print("  source ~/.zshrc")

def show_usage_examples():
    """Show practical usage examples."""
    print("\n" + "="*60)
    print("USAGE EXAMPLES")
    print("="*60)
    
    examples = [
        {
            'title': 'Free Sources (No API Key Required)',
            'commands': [
                '# Wikimedia Commons (free images)',
                'python data_scraper.py --source wikimedia --query "african music" --dataset afrocover --limit 50',
                '',
                '# Show Nigerian-specific guides',
                'python nigerian_scraper.py --show-guides'
            ]
        },
        {
            'title': 'With API Keys (Better Results)',
            'commands': [
                '# Unsplash for house images',
                'python data_scraper.py --source unsplash --query "lagos houses" --dataset lagos --limit 100',
                '',
                '# Pixabay for architecture',
                'python data_scraper.py --source pixabay --query "nigerian architecture" --dataset duplex --limit 50',
                '',
                '# Spotify for album covers (legal)',
                'python nigerian_scraper.py --source spotify --query "afrobeats" --limit 100'
            ]
        },
        {
            'title': 'Specialized Searches',
            'commands': [
                '# African album covers',
                'python data_scraper.py --source unsplash --query "african album cover" --dataset afrocover',
                '',
                '# Traditional vs Modern houses',
                'python data_scraper.py --source unsplash --query "traditional nigerian house" --dataset lagos',
                'python data_scraper.py --source unsplash --query "modern duplex nigeria" --dataset duplex',
                '',
                '# Last.fm for Nigerian artists',
                'python nigerian_scraper.py --source lastfm --query "burna boy" --limit 20'
            ]
        }
    ]
    
    for example in examples:
        print(f"\n{example['title']}:")
        print("-" * len(example['title']))
        for command in example['commands']:
            if command.startswith('#'):
                print(f"\033[92m{command}\033[0m")  # Green comments
            else:
                print(f"  {command}")

def check_environment():
    """Check if environment is properly set up."""
    print("\n" + "="*60)
    print("ENVIRONMENT CHECK")
    print("="*60)
    
    # Check Python packages
    required_packages = ['requests', 'PIL', 'bs4']
    optional_packages = ['spotipy']
    
    print("\nRequired packages:")
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} - Install with: pip install {package}")
    
    print("\nOptional packages:")
    for package in optional_packages:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} - Install with: pip install {package}")
    
    # Check data directories
    print("\nData directories:")
    data_dirs = [
        'data/afrocover/train',
        'data/afrocover/val', 
        'data/afrocover/test',
        'data/lagos2duplex/lagos/train',
        'data/lagos2duplex/lagos/val',
        'data/lagos2duplex/lagos/test',
        'data/lagos2duplex/duplex/train',
        'data/lagos2duplex/duplex/val',
        'data/lagos2duplex/duplex/test'
    ]
    
    for dir_path in data_dirs:
        if os.path.exists(dir_path):
            count = len(os.listdir(dir_path))
            print(f"  ✅ {dir_path} ({count} files)")
        else:
            print(f"  ❌ {dir_path} - missing")
    
    # Check API keys
    print("\nAPI Keys:")
    api_keys = ['UNSPLASH_API_KEY', 'PIXABAY_API_KEY', 'SPOTIFY_CLIENT_ID', 'LASTFM_API_KEY']
    for key in api_keys:
        if os.getenv(key):
            print(f"  ✅ {key}")
        else:
            print(f"  ❌ {key} - not set")

def show_legal_compliance():
    """Show legal compliance information."""
    print("\n" + "="*60)
    print("LEGAL COMPLIANCE & BEST PRACTICES")
    print("="*60)
    
    print("""
🔗 ALWAYS CHECK FIRST:
  ✅ robots.txt (automated in our scrapers)
  ✅ Terms of Service of each website
  ✅ Rate limits and politeness delays
  ✅ Attribution requirements

🎯 RECOMMENDED SOURCES (Legal & Ethical):
  ✅ Unsplash - Free to use with attribution
  ✅ Pixabay - Free to use
  ✅ Wikimedia Commons - Free cultural works
  ✅ Spotify API - Official, legal access to album art
  ✅ Last.fm API - Official music database

⚠️  REQUIRES PERMISSION:
  ❌ NotJustOk - Contact for research use
  ❌ PropertyPro.ng - Contact for academic use
  ❌ 360Nobs - Contact for research use

🔧 OUR BUILT-IN SAFEGUARDS:
  ✅ Automatic robots.txt checking
  ✅ Rate limiting (1-3 second delays)
  ✅ User-Agent identification
  ✅ Graceful error handling
  ✅ Session resumption (no duplicate downloads)

📧 CONTACT WEBSITES:
For Nigerian sites, we recommend contacting them directly:
  - Explain your research/educational purpose
  - Ask for permission or API access
  - Offer to share results/attribution
  - Respect their decision if they decline

🎓 ACADEMIC USE:
  - Cite sources in your research
  - Follow your institution's ethics guidelines
  - Consider fair use provisions
  - Document your data collection methodology
""")

def main():
    """Main setup function."""
    print("🚀 Lagos-GAN Data Collection Setup")
    print("="*50)
    
    # Install requirements
    if not install_scraping_requirements():
        print("Please install requirements manually and try again.")
        return
    
    # Setup API keys
    setup_api_keys()
    create_env_file()
    
    # Check environment
    check_environment()
    
    # Show usage examples
    show_usage_examples()
    
    # Show legal compliance
    show_legal_compliance()
    
    print("\n" + "="*60)
    print("🎉 SETUP COMPLETE!")
    print("="*60)
    print("""
NEXT STEPS:
1. Get API keys from the URLs shown above
2. Add them to api_keys.env and load into your shell
3. Start with free sources (wikimedia, unsplash without key)
4. Test with small limits first: --limit 10
5. Gradually increase as you verify everything works

QUICK START:
# Test without API keys
python data_scraper.py --source wikimedia --query "african music" --dataset afrocover --limit 10

# Count your data
python add_training_data.py --count

Happy scraping! 🎵🏠
""")

if __name__ == "__main__":
    main()
