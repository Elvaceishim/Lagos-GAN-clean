# Training Data Guide for Lagos-GAN

## Quick Start: Adding Your Training Data

### 1. Count Current Data

```bash
# Check what data you already have
python add_training_data.py --count
```

### 2. Add Album Covers (AfroCover Dataset)

```bash
# Example: Add album covers from your Downloads folder
python add_training_data.py --source ~/Downloads/album_covers --dataset afrocover

# Or from any other folder
python add_training_data.py --source /path/to/your/album/covers --dataset afrocover
```

### 3. Add House Images (Lagos2Duplex Dataset)

```bash
# Add Lagos-style house images
python add_training_data.py --source ~/Downloads/lagos_houses --dataset lagos

# Add Duplex-style house images
python add_training_data.py --source ~/Downloads/duplex_houses --dataset duplex
```

## Data Requirements

### AfroCover (Album Covers)

- **Type**: Album cover images
- **Best sources**: African/Afrobeats album covers
- **Format**: JPG, PNG (256x256 minimum)
- **Quantity needed**: 1000+ for good results
- **Where to find**:
  - Spotify album art (via web scraping tools)
  - Apple Music artwork
  - Bandcamp, SoundCloud covers
  - Music blogs and websites
  - Nigerian music label websites

### Lagos2Duplex (House Architecture)

#### Lagos-style houses (Traditional/Local)

- **Type**: Traditional Lagos residential buildings
- **Characteristics**:
  - Bungalows and story buildings
  - Traditional Nigerian architecture
  - Urban Lagos compounds
  - Older residential styles
- **Format**: JPG, PNG (256x256 minimum)
- **Quantity needed**: 500+ images

#### Duplex-style houses (Modern)

- **Type**: Modern duplex buildings
- **Characteristics**:
  - Two-story residential homes
  - Contemporary architecture
  - Modern design elements
  - Clean, geometric lines
- **Format**: JPG, PNG (256x256 minimum)
- **Quantity needed**: 500+ images

## Data Collection Tips

### 1. Legal and Ethical Considerations

- Use only images you have rights to use
- Consider Creative Commons licensed images
- Respect copyright and attribution requirements
- For personal/research use, many sources are acceptable

### 2. Image Quality Guidelines

- **Resolution**: Higher is better (512x512, 1024x1024)
- **Clarity**: Sharp, well-lit images
- **Consistency**: Similar lighting and angles when possible
- **Variety**: Different styles, colors, compositions

### 3. Good Sources for Data

#### Album Covers:

- **Discogs**: Large database of album art
- **MusicBrainz**: Open music database
- **Spotify Web API**: (requires proper permissions)
- **Last.fm**: Music discovery platform
- **Nigerian music websites**: NotJustOk, 360Nobs, Pulse Nigeria

#### House Images:

- **Real Estate Websites**:
  - PropertyPro.ng (Nigeria)
  - Jiji.ng real estate section
  - Private Property Nigeria
- **Architecture Websites**:
  - ArchDaily
  - Dezeen (filter for Nigerian projects)
- **Photography Platforms**:
  - Unsplash (search "Lagos houses", "Nigerian architecture")
  - Pixabay, Pexels
- **Google Images**: (with proper usage rights filters)

### 4. Web Scraping Tools (Use Responsibly)

```bash
# Install useful tools
pip install beautifulsoup4 requests pillow

# Example scraper (create your own based on website terms)
# Always check robots.txt and terms of service first
```

## Manual Data Organization

If you prefer to organize manually:

### AfroCover:

```bash
# Copy your album covers to these folders:
cp /path/to/your/album/covers/* data/afrocover/train/     # 70%
cp /path/to/more/covers/* data/afrocover/val/            # 15%
cp /path/to/test/covers/* data/afrocover/test/           # 15%
```

### Lagos2Duplex:

```bash
# Lagos-style houses
cp /path/to/lagos/houses/* data/lagos2duplex/lagos/train/
cp /path/to/lagos/val/* data/lagos2duplex/lagos/val/
cp /path/to/lagos/test/* data/lagos2duplex/lagos/test/

# Duplex-style houses
cp /path/to/duplex/houses/* data/lagos2duplex/duplex/train/
cp /path/to/duplex/val/* data/lagos2duplex/duplex/val/
cp /path/to/duplex/test/* data/lagos2duplex/duplex/test/
```

## Validation

After adding data, always validate:

```bash
# Check data counts
python add_training_data.py --count

# Test dataset loading
python -c "
from afrocover.dataset import AfroCoverDataset
from lagos2duplex.dataset import Lagos2DuplexDataset
import torch

# Test AfroCover dataset
afro_train = AfroCoverDataset('data/afrocover/train')
print(f'AfroCover train: {len(afro_train)} images')

# Test Lagos2Duplex dataset
lagos_train = Lagos2DuplexDataset('data/lagos2duplex', split='train')
print(f'Lagos2Duplex train: {len(lagos_train)} pairs')
"
```

## Ready to Train?

Once you have sufficient data:

```bash
# Train AfroCover model
python afrocover/train.py

# Train Lagos2Duplex model
python lagos2duplex/train.py
```

The models will automatically use your training data and save checkpoints as they train!
