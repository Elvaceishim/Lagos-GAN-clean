# Lagos2Duplex CycleGAN Training Guide

This guide explains how to train a CycleGAN model to transform old Lagos houses into modern duplexes using your preprocessed dataset.

## 📁 Project Structure

```
Lagos-GAN/
├── data_processed/                 # Preprocessed dataset (256x256 images)
│   ├── afrocover/                 # Lagos houses (Domain A)
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── lagos2duplex/              # Duplex houses (Domain B)
│       └── duplex/
│           ├── train/
│           ├── val/
│           └── test/
├── lagos2duplex/                  # CycleGAN implementation
│   ├── __init__.py
│   ├── config.py                  # Training configuration
│   ├── dataset.py                 # PyTorch dataset classes
│   ├── losses.py                  # Loss functions
│   ├── models.py                  # Generator and discriminator models
│   └── train.py                   # Main training script
├── start_training.py              # Easy-to-use training launcher
└── checkpoints/                   # Saved model checkpoints
    └── lagos2duplex/
```

## 🚀 Quick Start

### 1. Verify Data Preprocessing

Ensure your dataset has been preprocessed to 256x256 resolution:

```bash
python preprocess_dataset.py --target_size 256 --output_format png
```

### 2. Start Training (Easy Method)

Use the simple training launcher:

```bash
# Quick test (5 epochs, fast iteration)
python start_training.py --mode quick

# Standard training (200 epochs)
python start_training.py --mode standard

# High-quality training (300 epochs, enhanced parameters)
python start_training.py --mode high-quality
```

### 3. Advanced Training Options

```bash
# Custom number of epochs
python start_training.py --mode standard --epochs 150

# Custom batch size (if you have memory issues, try batch_size 1)
python start_training.py --mode standard --batch_size 2

# Use specific GPU
python start_training.py --mode standard --gpu 0

# Disable Weights & Biases logging
python start_training.py --mode standard --no_wandb

# Resume from checkpoint
python start_training.py --mode standard --resume checkpoints/lagos2duplex/checkpoint_epoch_50.pt
```

## 🔧 Manual Training (Advanced)

For full control, use the training script directly:

```bash
cd lagos2duplex
python train.py \
    --data_path ../data_processed \
    --output_dir ../checkpoints/lagos2duplex \
    --num_epochs 200 \
    --batch_size 1 \
    --learning_rate 0.0002 \
    --lambda_cycle 10.0 \
    --lambda_identity 0.5 \
    --image_size 256
```

## 📊 Training Configuration

### Model Architecture

- **Generator**: ResNet-based with 9 residual blocks
- **Discriminator**: PatchGAN discriminator
- **Image Size**: 256x256 pixels
- **Normalization**: Instance normalization
- **Activation**: ReLU for generator, LeakyReLU for discriminator

### Loss Functions

- **Adversarial Loss**: LSGAN (Least Squares GAN)
- **Cycle Consistency Loss**: L1 loss with weight 10.0
- **Identity Loss**: L1 loss with weight 0.5 (optional)
- **Perceptual Loss**: VGG-based perceptual loss (optional)

### Training Parameters

- **Optimizer**: Adam with β₁=0.5, β₂=0.999
- **Learning Rate**: 0.0002 with linear decay
- **Batch Size**: 1 (can be increased with more GPU memory)
- **Epochs**: 200 (standard), 300 (high-quality)

## 📈 Monitoring Training

### Weights & Biases (Recommended)

The training automatically logs to Weights & Biases:

1. Install: `pip install wandb`
2. Login: `wandb login`
3. Monitor at: https://wandb.ai/

Logged metrics include:

- Generator and discriminator losses
- Cycle consistency losses
- Generated image samples
- Learning rate schedules

### TensorBoard (Alternative)

Enable TensorBoard logging in config:

```python
config.logging.use_tensorboard = True
```

View logs:

```bash
tensorboard --logdir logs/lagos2duplex/tensorboard
```

### Console Output

Training progress is displayed in the console with:

- Loss values for each component
- Progress bars for epochs and batches
- Sample image generation timestamps

## 💾 Checkpoints and Results

### Automatic Saving

- **Latest checkpoint**: `checkpoints/lagos2duplex/latest.pt`
- **Best model**: `checkpoints/lagos2duplex/best.pt` (based on validation loss)
- **Periodic saves**: Every 20 epochs by default
- **Final checkpoint**: `checkpoints/lagos2duplex/final_checkpoint.pt`

### Generated Images

Sample images are saved in:

- `results/lagos2duplex/train_epoch_X_batch_Y.png`
- `results/lagos2duplex/val_epoch_X_batch_0.png`

## 🔧 Troubleshooting

### Common Issues

#### 1. Out of Memory Error

```bash
# Reduce batch size
python start_training.py --mode standard --batch_size 1

# Or use gradient accumulation in config
```

#### 2. Slow Training

```bash
# Increase number of data loading workers
# Edit config.data.num_workers = 8

# Use mixed precision (if supported)
# Edit config.system.mixed_precision = True
```

#### 3. Poor Results

```bash
# Try high-quality mode
python start_training.py --mode high-quality

# Increase cycle loss weight
python lagos2duplex/train.py --lambda_cycle 15.0

# Add perceptual loss
# Edit config.training.lambda_perceptual = 1.0
```

#### 4. Dataset Not Found

```bash
# Verify data structure
ls -la data_processed/afrocover/train/
ls -la data_processed/lagos2duplex/duplex/train/

# Re-run preprocessing if needed
python preprocess_dataset.py --target_size 256 --output_format png
```

## 📝 Configuration Files

### Loading Custom Config

```bash
# Save custom configuration
python -c "
from lagos2duplex.config import CycleGANConfig
config = CycleGANConfig()
config.training.num_epochs = 300
config.training.lambda_cycle = 15.0
config.save('my_config.json')
"

# Use custom configuration
python lagos2duplex/train.py --config my_config.json
```

### Key Configuration Options

- `model.netG`: Generator architecture ('resnet_9blocks', 'resnet_6blocks')
- `training.gan_mode`: GAN loss type ('lsgan', 'vanilla', 'wgangp')
- `training.lambda_cycle`: Cycle consistency weight (default: 10.0)
- `training.lambda_identity`: Identity loss weight (default: 0.5)
- `data.augmentations`: Enable data augmentation (default: True)
- `system.mixed_precision`: Enable mixed precision training

## 🎯 Expected Results

### Training Timeline

- **Epochs 1-50**: Basic structure learning
- **Epochs 50-100**: Feature refinement
- **Epochs 100-200**: Fine-tuning and stabilization
- **Epochs 200+**: Quality improvements

### Evaluation Metrics

The model automatically tracks:

- **Generator Loss**: Should decrease and stabilize
- **Discriminator Loss**: Should stabilize around 0.5
- **Cycle Loss**: Should decrease consistently
- **Validation Loss**: Should follow training loss

### Visual Quality

Look for:

- Realistic architectural transformations
- Preserved structural elements
- Consistent style transfer
- Minimal artifacts

## 🔄 Resume Training

To resume interrupted training:

```bash
# Resume from latest checkpoint
python start_training.py --mode standard --resume checkpoints/lagos2duplex/latest.pt

# Resume from specific epoch
python start_training.py --mode standard --resume checkpoints/lagos2duplex/checkpoint_epoch_100.pt
```

## 🏁 Next Steps

After training completes:

1. **Evaluate Results**: Check generated samples in `results/` directory
2. **Test on New Images**: Use the trained model for inference
3. **Fine-tune Parameters**: Adjust configuration for better results
4. **Create Demos**: Build interactive applications using the trained model

## 📞 Support

If you encounter issues:

1. Check the console output for error messages
2. Verify your data structure matches the expected format
3. Ensure all dependencies are installed
4. Check GPU memory usage and reduce batch size if needed

Happy training! 🏠➡️🏠
