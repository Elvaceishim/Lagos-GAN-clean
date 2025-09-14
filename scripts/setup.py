"""
Quick setup script for LagosGAN project
"""

import subprocess
import sys
from pathlib import Path


def create_directories():
    """Create necessary data and checkpoint directories"""
    directories = [
        'data/afrocover/train',
        'data/afrocover/val', 
        'data/afrocover/test',
        'data/lagos2duplex/lagos/train',
        'data/lagos2duplex/lagos/val',
        'data/lagos2duplex/lagos/test',
        'data/lagos2duplex/duplex/train',
        'data/lagos2duplex/duplex/val',
        'data/lagos2duplex/duplex/test',
        'checkpoints/afrocover',
        'checkpoints/lagos2duplex',
        'results/images',
        'results/metrics',
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created: {directory}")


def main():
    print("Setting up LagosGAN project structure...")
    create_directories()
    print("\nSetup complete! You can now:")
    print("1. Add your datasets to the data/ directories")
    print("2. Install dependencies: pip install -r requirements.txt")
    print("3. Start training: python afrocover/train.py --data_path data/afrocover")
    print("4. Run the demo: python demo/app.py")


if __name__ == "__main__":
    main()
