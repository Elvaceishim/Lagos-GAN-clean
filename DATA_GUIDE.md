# 📁 Lagos-GAN Data Management Guide

## 🎯 **Quick Start - Test with Sample Data**

```bash
# 1. Create sample data for immediate testing
cd /Users/mac/Lagos-GAN
./.venv/bin/python manage_data.py --create-samples

# 2. Validate the data
./.venv/bin/python manage_data.py --validate

# 3. Test training with sample data
./.venv/bin/python afrocover/train.py --data_path data/afrocover --num_epochs 5
```

## 📊 **Real Data Requirements**

### **🎨 AfroCover (Album Cover Generation)**

**What you need:**
- **1,000-10,000+ album cover images**
- African-inspired, colorful, artistic designs
- High quality, diverse styles

**Where to get data:**
1. **Spotify/Apple Music screenshots** (personal use only)
2. **Creative Commons music artwork**
3. **Stock photo sites** (Unsplash, Pexels)
4. **African art databases**
5. **Music label websites** (with permission)

### **🏠 Lagos2Duplex (House Transformation)**

**What you need:**
- **500-5,000+ house images** (each type)
- Lagos houses: Simple, traditional, single-story
- Duplex houses: Modern, multi-story, contemporary

**Where to get data:**
1. **Real estate websites** (Jumia House, Property24)
2. **Google Street View** (Lagos neighborhoods)
3. **Architecture websites**
4. **Construction company portfolios**
5. **Social media** (Instagram #lagoshouses)

## 💻 **Adding Your Own Data**

### **Method 1: Organize Existing Collections**

```bash
# If you have album covers in a folder
./.venv/bin/python manage_data.py --organize-afrocover ~/Downloads/album_covers

# If you have house photos in separate folders
./.venv/bin/python manage_data.py --organize-houses ~/Downloads/lagos_houses ~/Downloads/duplex_houses
```

### **Method 2: Manual Organization**

```bash
# Copy your images to the appropriate folders
cp ~/your_album_covers/*.jpg data/afrocover/train/
cp ~/your_lagos_houses/*.jpg data/lagos2duplex/lagos/train/
cp ~/your_duplex_houses/*.jpg data/lagos2duplex/duplex/train/

# Then validate
./.venv/bin/python manage_data.py --validate
```

### **Method 3: Download Script (Advanced)**

Create a download script for web scraping:

```python
# Example download script (create as download_data.py)
import requests
from pathlib import Path

def download_sample_data():
    # This is just an example - replace with actual data sources
    urls = [
        "https://example.com/album1.jpg",
        "https://example.com/house1.jpg",
        # Add your URLs here
    ]
    
    for i, url in enumerate(urls):
        response = requests.get(url)
        if response.status_code == 200:
            with open(f"data/afrocover/train/downloaded_{i}.jpg", "wb") as f:
                f.write(response.content)
```

## 📋 **Data Quality Guidelines**

### **Image Requirements:**
- **Format**: JPG, PNG preferred
- **Size**: Minimum 256x256 pixels
- **Quality**: Clear, not blurry or pixelated
- **Variety**: Diverse styles, colors, compositions

### **Album Cover Specifics:**
- ✅ Colorful, artistic designs
- ✅ Text/typography elements
- ✅ African cultural elements
- ✅ Music-related imagery
- ❌ Avoid copyrighted material
- ❌ No NSFW content

### **House Image Specifics:**
- ✅ Front-facing views
- ✅ Clear building structure
- ✅ Good lighting
- ✅ Minimal obstructions
- ❌ Avoid indoor shots
- ❌ No people in foreground

## 🔍 **Data Validation & Troubleshooting**

```bash
# Check dataset statistics
./.venv/bin/python manage_data.py --validate

# Test dataset loading
./.venv/bin/python -c "
from afrocover.dataset import AfrocoverDataset
dataset = AfrocoverDataset('data/afrocover', split='train')
print(f'Dataset size: {len(dataset)}')
if len(dataset) > 0:
    sample = dataset[0]
    print(f'Image shape: {sample["image"].shape}')
"
```

## 📈 **Recommended Dataset Sizes**

### **Minimum (Testing):**
- AfroCover: 100+ images
- Lagos houses: 50+ images each type
- Training time: 2-4 hours

### **Good Results:**
- AfroCover: 1,000+ images
- Lagos houses: 500+ images each type
- Training time: 8-12 hours

### **Professional Quality:**
- AfroCover: 5,000+ images
- Lagos houses: 2,000+ images each type
- Training time: 24-48 hours

## 🚀 **Ready to Train?**

Once you have data organized:

```bash
# Train AfroCover model
./.venv/bin/python afrocover/train.py \
    --data_path data/afrocover \
    --batch_size 4 \
    --num_epochs 100 \
    --output_dir checkpoints/afrocover

# Train Lagos2Duplex model
./.venv/bin/python lagos2duplex/train.py \
    --data_path data/lagos2duplex \
    --batch_size 4 \
    --num_epochs 200 \
    --output_dir checkpoints/lagos2duplex
```

## 💡 **Pro Tips**

1. **Start Small**: Begin with sample data to test the pipeline
2. **Quality > Quantity**: 500 good images > 2000 poor images
3. **Data Augmentation**: The code automatically applies transformations
4. **Monitor Training**: Use TensorBoard to track progress
5. **Backup Data**: Keep original images separate from processed data

## ⚖️ **Legal Considerations**

- Use only images you have rights to
- Consider Creative Commons licensed content
- For commercial use, ensure proper licensing
- Respect copyright and artist rights
- Consider creating original content or commissioning artists

---

**Need help?** Run the data manager for interactive guidance:
```bash
./.venv/bin/python manage_data.py
```
