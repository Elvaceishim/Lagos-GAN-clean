#!/usr/bin/env python3
"""
Example: How to Add Your Own Training Data

This script shows practical examples of adding real training data to Lagos-GAN.
"""

import os
import shutil
from pathlib import Path


def example_add_album_covers():
    """Example: Add album covers from your collection"""
    
    # Let's say you have album covers in Downloads
    source_folder = "~/Downloads/album_covers"  # Your folder
    target_folder = "/Users/mac/Lagos-GAN/data/afrocover/train"
    
    print("🎨 Adding Album Covers:")
    print(f"   Source: {source_folder}")
    print(f"   Target: {target_folder}")
    print()
    print("Commands to run:")
    print(f"   # Copy all JPG files")
    print(f"   cp {source_folder}/*.jpg {target_folder}/")
    print(f"   cp {source_folder}/*.jpeg {target_folder}/")
    print(f"   cp {source_folder}/*.png {target_folder}/")
    print()
    print("   # Or organize automatically:")
    print(f"   ./.venv/bin/python manage_data.py --organize-afrocover {source_folder}")
    

def example_add_house_photos():
    """Example: Add house photos from real estate sites"""
    
    lagos_folder = "~/Downloads/lagos_houses"    # Simple houses
    duplex_folder = "~/Downloads/duplex_houses"  # Modern duplexes
    
    print("🏠 Adding House Photos:")
    print(f"   Lagos houses: {lagos_folder}")
    print(f"   Duplex houses: {duplex_folder}")
    print()
    print("Commands to run:")
    print(f"   # Manual copy")
    print(f"   cp {lagos_folder}/*.jpg /Users/mac/Lagos-GAN/data/lagos2duplex/lagos/train/")
    print(f"   cp {duplex_folder}/*.jpg /Users/mac/Lagos-GAN/data/lagos2duplex/duplex/train/")
    print()
    print("   # Or organize automatically:")
    print(f"   ./.venv/bin/python manage_data.py --organize-houses {lagos_folder} {duplex_folder}")


def real_world_workflow():
    """Complete workflow for real data"""
    
    print("🔄 Complete Real-World Workflow:")
    print("=" * 50)
    print()
    
    print("1️⃣ Collect Data:")
    print("   • Download 500+ album covers from Creative Commons sources")
    print("   • Screenshot real estate listings (Lagos & modern houses)")
    print("   • Ensure you have usage rights")
    print()
    
    print("2️⃣ Organize Data:")
    print("   mkdir -p ~/training_data/album_covers")
    print("   mkdir -p ~/training_data/lagos_houses")
    print("   mkdir -p ~/training_data/duplex_houses")
    print("   # Move your collected images to these folders")
    print()
    
    print("3️⃣ Process Data:")
    print("   cd /Users/mac/Lagos-GAN")
    print("   ./.venv/bin/python manage_data.py --organize-afrocover ~/training_data/album_covers")
    print("   ./.venv/bin/python manage_data.py --organize-houses ~/training_data/lagos_houses ~/training_data/duplex_houses")
    print()
    
    print("4️⃣ Validate Data:")
    print("   ./.venv/bin/python manage_data.py --validate")
    print()
    
    print("5️⃣ Start Training:")
    print("   ./.venv/bin/python afrocover/train.py --data_path data/afrocover --num_epochs 100")
    print("   ./.venv/bin/python lagos2duplex/train.py --data_path data/lagos2duplex --num_epochs 200")


def quick_test_setup():
    """Quick setup for immediate testing"""
    
    print("⚡ Quick Test Setup (Using Sample Data):")
    print("=" * 50)
    print()
    
    print("Already done - sample data created! ✅")
    print()
    print("What you can do right now:")
    print("   ./.venv/bin/python manage_data.py --validate")
    print("   ./.venv/bin/python afrocover/train.py --data_path data/afrocover --num_epochs 5")
    print()
    print("This will train on sample data to test the pipeline.")


def data_sources():
    """Suggest data sources"""
    
    print("📊 Recommended Data Sources:")
    print("=" * 50)
    print()
    
    print("🎨 Album Cover Sources:")
    print("   • Spotify API (for metadata)")
    print("   • Last.fm datasets")
    print("   • MusicBrainz (open music database)")
    print("   • Creative Commons music archives")
    print("   • Bandcamp (independent artists)")
    print("   • Freepik/Unsplash (stock music graphics)")
    print()
    
    print("🏠 House Photo Sources:")
    print("   • Nigerian real estate sites:")
    print("     - PropertyPro.ng")
    print("     - Lamudi.com.ng")
    print("     - Jumia House")
    print("   • Google Street View (Lagos neighborhoods)")
    print("   • Architecture firms' portfolios")
    print("   • Construction company websites")
    print("   • Social media hashtags:")
    print("     - #lagoshouses #nigerianarchitecture")
    print("     - #duplexdesign #modernhomes")


if __name__ == "__main__":
    print("🎯 Lagos-GAN Training Data Examples")
    print("=" * 50)
    print()
    
    print("Choose what you want to learn about:")
    print("1. Add album covers")
    print("2. Add house photos") 
    print("3. Complete workflow")
    print("4. Quick test setup")
    print("5. Data sources")
    print()
    
    choice = input("Enter choice (1-5): ").strip()
    print()
    
    if choice == "1":
        example_add_album_covers()
    elif choice == "2":
        example_add_house_photos()
    elif choice == "3":
        real_world_workflow()
    elif choice == "4":
        quick_test_setup()
    elif choice == "5":
        data_sources()
    else:
        print("Running all examples...")
        print()
        example_add_album_covers()
        print("\n" + "="*50 + "\n")
        example_add_house_photos()
        print("\n" + "="*50 + "\n")
        real_world_workflow()
