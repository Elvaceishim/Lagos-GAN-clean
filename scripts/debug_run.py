"""Debug runner: import key functions and run minimal steps to confirm dataset, models, and training loop can start."""
import os, sys, traceback
sys.path.insert(0, os.getcwd())
from datetime import datetime
print('Debug run at:', datetime.now().isoformat())
print('CWD:', os.getcwd())
print('Python:', sys.executable)

try:
    from lagos2duplex.config import get_quick_test_config
    from lagos2duplex.dataset import create_dataloaders
    from lagos2duplex.train import create_models, setup_optimizers, setup_schedulers, CycleGANLoss, ImagePool, train_epoch
    print('Imported lagos2duplex modules')
except Exception as e:
    print('Import failure:', e)
    traceback.print_exc()
    raise

cfg = get_quick_test_config()
cfg.data.image_size = 64
cfg.training.batch_size = 2
# detect data path
candidates = ['./data','./data/lagos2duplex','./data/afrocover']
for p in candidates:
    if os.path.exists(p):
        cfg.data.data_path = os.path.abspath(p)
        break
print('Using data_path:', cfg.data.data_path)

# Use a proper torch.device object so create_models can inspect device.type
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

try:
    train_loader, val_loader = create_dataloaders(data_path=cfg.data.data_path, batch_size=cfg.training.batch_size, image_size=cfg.data.image_size, num_workers=0)
    print('Got dataloaders - train batches:', len(train_loader), 'val batches:', len(val_loader))
    batch = next(iter(train_loader))
    print('Batch keys:', list(batch.keys()))
    a = batch['A']
    b = batch['B']
    print('A shape:', getattr(a,'shape',None), 'B shape:', getattr(b,'shape',None))
except Exception as e:
    print('Dataloader error:', e)
    traceback.print_exc()
    raise

# Create models
try:
    G_AB, G_BA, D_A, D_B = create_models(cfg, device=device)
    print('Models created')
except Exception as e:
    print('Create models failed:', e)
    traceback.print_exc()
    raise

# Build optimizers and criterion
try:
    optimizers = setup_optimizers(G_AB, G_BA, D_A, D_B, cfg)
    schedulers = setup_schedulers(optimizers, cfg)
    criterion = CycleGANLoss(gan_mode=cfg.training.gan_mode, lambda_cycle=cfg.training.lambda_cycle, lambda_identity=cfg.training.lambda_identity, lambda_perceptual=cfg.training.lambda_perceptual)
    print('Optimizers and criterion ready')
except Exception as e:
    print('Optimizer/criterion setup failed:', e)
    traceback.print_exc()
    raise

# Try a single training epoch but limit dataloader to small subset
try:
    from torch.utils.data import Subset
    subset = Subset(train_loader.dataset, list(range(min(4, len(train_loader.dataset)))))
    from torch.utils.data import DataLoader
    small_dl = DataLoader(subset, batch_size=cfg.training.batch_size, shuffle=True, num_workers=0)
    fake_A_pool = ImagePool(cfg.training.pool_size)
    fake_B_pool = ImagePool(cfg.training.pool_size)
    print('Starting one train_epoch with small dataloader...')
    epoch_losses = train_epoch(models=(G_AB,G_BA,D_A,D_B), optimizers=optimizers, schedulers=schedulers, criterion=criterion, fake_pools=(fake_A_pool,fake_B_pool), dataloader=small_dl, device='cpu', epoch=0, config=cfg, loggers={})
    print('Epoch losses:', epoch_losses)
except Exception as e:
    print('train_epoch run failed:', e)
    traceback.print_exc()
    raise

print('Debug run completed successfully')
