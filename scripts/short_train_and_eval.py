"""Short training run (5 epochs) with checkpoints and evaluation.
- Uses get_quick_test_config() with smaller image size and batch.
- Saves a checkpoint after each epoch to config.paths.checkpoints_dir/latest.pt and epoch files.
- After training, runs scripts/generate_and_eval.py to produce generation+evaluation outputs.

Designed to be run from project root.
"""
import os, sys, traceback
sys.path.insert(0, os.getcwd())
from datetime import datetime
print('Short train+eval run at:', datetime.now().isoformat())
print('CWD:', os.getcwd())
# Bootstrap: ensure test_results log gets an immediate marker so external pollers can detect activity
try:
    os.makedirs('test_results', exist_ok=True)
    with open('test_results/train_resume_epochs_to_7_2025-09-11.txt', 'a') as f:
        f.write(f"[BOOTSTRAP] short_train_and_eval.py started at {datetime.now().isoformat()}\n")
except Exception:
    pass

import torch

# Import project modules
from lagos2duplex.config import get_quick_test_config
from lagos2duplex.dataset import create_dataloaders
from lagos2duplex.train import (
    create_models, setup_optimizers, setup_schedulers, CycleGANLoss,
    ImagePool, save_checkpoint, train_epoch, setup_logging, validate_model
)

# Detect data path similar to quick_test_updated
candidates = [
    './data', './data_processed', './data/processed', './datasets',
    './data/lagos2duplex', './data/afrocover'
]
data_path = None
for p in candidates:
    if os.path.exists(p):
        data_path = os.path.abspath(p)
        break
if data_path is None:
    data_path = os.path.abspath('./data')
    print('Warning: no common data_path found; using fallback:', data_path)
else:
    print('Using detected data_path:', data_path)

# Prepare config
cfg = get_quick_test_config()
# small/fast options
cfg.data.image_size = 64
cfg.training.batch_size = 2
cfg.training.num_epochs = 5
cfg.data.data_path = data_path
cfg.logging.use_wandb = False
cfg.logging.print_freq = 10
cfg.logging.display_freq = 50

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# Create dataloaders
try:
    train_loader, val_loader = create_dataloaders(data_path=cfg.data.data_path,
                                                  batch_size=cfg.training.batch_size,
                                                  image_size=cfg.data.image_size,
                                                  num_workers=0)
    print('Train batches:', len(train_loader), 'Val batches:', len(val_loader))
except Exception as e:
    print('Failed to create dataloaders:', e)
    traceback.print_exc()
    raise

# Create models
G_AB, G_BA, D_A, D_B = create_models(cfg, device)
models = (G_AB, G_BA, D_A, D_B)
print('Models created')

# Setup optimizers and schedulers
optimizers = setup_optimizers(G_AB, G_BA, D_A, D_B, cfg)
schedulers = setup_schedulers(optimizers, cfg)
print('Optimizers and schedulers ready')

# Criterion
criterion = CycleGANLoss(
    gan_mode=cfg.training.gan_mode,
    lambda_cycle=cfg.training.lambda_cycle,
    lambda_identity=cfg.training.lambda_identity,
    lambda_perceptual=cfg.training.lambda_perceptual,
    cycle_loss_type=cfg.training.cycle_loss_type,
    identity_loss_type=cfg.training.identity_loss_type
)
# Move components to device where applicable
try:
    criterion.gan_loss = criterion.gan_loss.to(device)
    criterion.cycle_loss = criterion.cycle_loss.to(device)
    criterion.identity_loss = criterion.identity_loss.to(device)
    if getattr(criterion, 'perceptual_loss', None) is not None:
        criterion.perceptual_loss = criterion.perceptual_loss.to(device)
except Exception:
    pass

# Image pools
fake_A_pool = ImagePool(cfg.training.pool_size)
fake_B_pool = ImagePool(cfg.training.pool_size)
fake_pools = (fake_A_pool, fake_B_pool)

# Logging
loggers = setup_logging(cfg)

# Training loop with checkpoints
start_epoch = 0
for epoch in range(start_epoch, cfg.training.num_epochs):
    print(f"\n=== Epoch {epoch+1}/{cfg.training.num_epochs} ===")
    try:
        epoch_losses = train_epoch(models=models, optimizers=optimizers, schedulers=schedulers,
                                   criterion=criterion, fake_pools=fake_pools, dataloader=train_loader,
                                   device=device, epoch=epoch, config=cfg, loggers=loggers)
        print('Epoch losses:', epoch_losses)
    except Exception as e:
        print('train_epoch failed at epoch', epoch, ':', e)
        traceback.print_exc()
        break

    # Validate
    try:
        val_loss = validate_model(models, val_loader, criterion, device, epoch, cfg, loggers)
        print('Validation loss:', val_loss)
    except Exception as e:
        print('Validation failed:', e)

    # Save checkpoint
    try:
        ckpt_path = save_checkpoint(models, optimizers, schedulers, epoch, cfg)
        print('Saved checkpoint:', ckpt_path)
    except Exception as e:
        print('Failed to save checkpoint:', e)

print('\nTraining run complete')

# After training, run generation+evaluation script
print('\nRunning generation and evaluation...')
try:
    # Use the same python interpreter
    python_exec = sys.executable
    cmd = f"{python_exec} scripts/generate_and_eval.py"
    print('Executing:', cmd)
    rc = os.system(cmd)
    print('generate_and_eval exit code:', rc)
except Exception as e:
    print('Failed to run generate_and_eval.py:', e)

print('Short train+eval finished.')
