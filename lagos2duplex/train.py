"""
Lagos2Duplex CycleGAN Training Script

This module handles training of CycleGAN for transforming old Lagos houses into modern duplexes.
"""

import torch
import torch.nn as nn
import argparse
import os
import json
import time
from pathlib import Path
import itertools
from tqdm import tqdm
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime

# Import CycleGAN components
from .models import define_G, define_D, ImagePool, init_net
from .dataset import Lagos2DuplexDataset, create_dataloaders
from .losses import CycleGANLoss
from .config import CycleGANConfig, get_default_config, get_quick_test_config

# Optional imports for logging
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with: pip install wandb")

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

def log_checkpoint_to_wandb(*paths):
    if not (WANDB_AVAILABLE and wandb.run is not None):
        return
    artifact = wandb.Artifact("lagos2duplex-checkpoint", type="model")
    for path in paths:
        artifact.add_file(path)
    wandb.run.log_artifact(artifact, aliases=["latest"])

def parse_args():
    parser = argparse.ArgumentParser(description='Train Lagos2Duplex CycleGAN')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to configuration file')
    parser.add_argument('--data_path', type=str, default='data_processed',
                        help='Path to processed dataset')
    parser.add_argument('--output_dir', type=str, default='./checkpoints/lagos2duplex',
                        help='Output directory for checkpoints')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0002,
                        help='Learning rate')
    parser.add_argument('--image_size', type=int, default=256,
                        help='Image resolution')
    parser.add_argument('--lambda_cycle', type=float, default=10.0,
                        help='Weight for cycle consistency loss')
    parser.add_argument('--lambda_identity', type=float, default=0.5,
                        help='Weight for identity loss')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--wandb_project', type=str, default='lagosgan-lagos2duplex',
                        help='Weights & Biases project name')
    parser.add_argument('--quick_test', action='store_true',
                        help='Run quick test with reduced parameters')
    parser.add_argument('--no_wandb', action='store_true',
                        help='Disable Weights & Biases logging')
    
    return parser.parse_args()


def setup_training(config):
    """Setup training environment and directories"""
    # Create directories
    os.makedirs(config.paths.checkpoints_dir, exist_ok=True)
    os.makedirs(config.paths.results_dir, exist_ok=True)
    os.makedirs(config.paths.logs_dir, exist_ok=True)
    
    # Setup device
    device = torch.device(config.system.device)
    print(f"Using device: {device}")
    
    # Set random seeds for reproducibility
    if config.system.seed is not None:
        torch.manual_seed(config.system.seed)
        np.random.seed(config.system.seed)
        if device.type == 'cuda':
            torch.cuda.manual_seed(config.system.seed)
            torch.cuda.manual_seed_all(config.system.seed)
    
    # Set deterministic mode
    if config.system.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    return device


def setup_logging(config):
    """Setup logging services (W&B, TensorBoard)"""
    loggers = {}
    
    # Setup Weights & Biases
    if config.logging.use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project=config.logging.wandb_project,
            name=config.logging.wandb_run_name,
            config=config.to_dict()
        )
        loggers['wandb'] = wandb
        print("Weights & Biases logging enabled")
    
    # Setup TensorBoard
    if config.logging.use_tensorboard and TENSORBOARD_AVAILABLE:
        log_dir = os.path.join(config.paths.logs_dir, 'tensorboard')
        loggers['tensorboard'] = SummaryWriter(log_dir)
        print(f"TensorBoard logging enabled: {log_dir}")
    
    return loggers


def load_dataset(config):
    """Load and prepare the Lagos2Duplex dataset"""
    print(f"Loading dataset from: {config.data.data_path}")
    
    train_loader, val_loader = create_dataloaders(
        data_path=config.data.data_path,
        batch_size=config.training.batch_size,
        image_size=config.data.image_size,
        num_workers=config.data.num_workers,
        paired=config.data.paired
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    return train_loader, val_loader


def create_models(config, device):
    """Create CycleGAN generators and discriminators"""
    print("Creating CycleGAN models...")
    
    # Generators: A -> B (lagos -> duplex), B -> A (duplex -> lagos)
    G_AB = define_G(
        input_nc=config.model.input_nc,
        output_nc=config.model.output_nc,
        ngf=config.model.ngf,
        netG=config.model.netG,
        norm=config.model.norm,
        use_dropout=config.model.use_dropout,
        init_type=config.model.init_type,
        init_gain=config.model.init_gain,
        gpu_ids=config.system.gpu_ids if device.type == 'cuda' else []
    )
    
    G_BA = define_G(
        input_nc=config.model.input_nc,
        output_nc=config.model.output_nc,
        ngf=config.model.ngf,
        netG=config.model.netG,
        norm=config.model.norm,
        use_dropout=config.model.use_dropout,
        init_type=config.model.init_type,
        init_gain=config.model.init_gain,
        gpu_ids=config.system.gpu_ids if device.type == 'cuda' else []
    )
    
    # Discriminators: for domain A (lagos) and domain B (duplex)
    D_A = define_D(
        input_nc=config.model.input_nc,
        ndf=config.model.ndf,
        netD=config.model.netD,
        n_layers_D=config.model.n_layers_D,
        norm=config.model.norm,
        init_type=config.model.init_type,
        init_gain=config.model.init_gain,
        gpu_ids=config.system.gpu_ids if device.type == 'cuda' else []
    )
    
    D_B = define_D(
        input_nc=config.model.input_nc,
        ndf=config.model.ndf,
        netD=config.model.netD,
        n_layers_D=config.model.n_layers_D,
        norm=config.model.norm,
        init_type=config.model.init_type,
        init_gain=config.model.init_gain,
        gpu_ids=config.system.gpu_ids if device.type == 'cuda' else []
    )
    
    # Print model information
    total_params_G = sum(p.numel() for p in G_AB.parameters()) + sum(p.numel() for p in G_BA.parameters())
    total_params_D = sum(p.numel() for p in D_A.parameters()) + sum(p.numel() for p in D_B.parameters())
    print(f"Total Generator parameters: {total_params_G:,}")
    print(f"Total Discriminator parameters: {total_params_D:,}")
    
    return G_AB, G_BA, D_A, D_B


def setup_optimizers(G_AB, G_BA, D_A, D_B, config):
    """Setup optimizers for generators and discriminators"""
    # Generator optimizer (shared for both generators)
    optimizer_G = torch.optim.Adam(
        itertools.chain(G_AB.parameters(), G_BA.parameters()),
        lr=config.training.learning_rate,
        betas=(config.training.beta1, config.training.beta2)
    )
    
    # Discriminator optimizers
    optimizer_D_A = torch.optim.Adam(
        D_A.parameters(),
        lr=config.training.learning_rate,
        betas=(config.training.beta1, config.training.beta2)
    )
    
    optimizer_D_B = torch.optim.Adam(
        D_B.parameters(),
        lr=config.training.learning_rate,
        betas=(config.training.beta1, config.training.beta2)
    )
    
    return optimizer_G, optimizer_D_A, optimizer_D_B


def setup_schedulers(optimizers, config):
    """Setup learning rate schedulers"""
    optimizer_G, optimizer_D_A, optimizer_D_B = optimizers
    schedulers = []
    
    if config.training.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 - config.training.n_epochs_decay) / float(config.training.n_epochs_decay + 1)
            return lr_l
        
        scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lambda_rule)
        scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=lambda_rule)
        scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=lambda_rule)
        
        schedulers = [scheduler_G, scheduler_D_A, scheduler_D_B]
    
    return schedulers


def train_epoch(models, optimizers, schedulers, criterion, fake_pools, dataloader, 
                device, epoch, config, loggers):
    """Train for one epoch"""
    G_AB, G_BA, D_A, D_B = models
    optimizer_G, optimizer_D_A, optimizer_D_B = optimizers
    fake_A_pool, fake_B_pool = fake_pools
    
    # Set models to training mode
    G_AB.train()
    G_BA.train()
    D_A.train()
    D_B.train()
    
    epoch_losses = {
        'G_total': [], 'G_A': [], 'G_B': [], 
        'cycle_A': [], 'cycle_B': [], 'identity_A': [], 'identity_B': [],
        'D_A': [], 'D_B': []
    }
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.training.num_epochs}")
    
    for batch_idx, batch in enumerate(progress_bar):
        real_A = batch['A'].to(device)  # Lagos houses
        real_B = batch['B'].to(device)  # Modern duplexes
        
        # =====================
        # Train Generators
        # =====================
        optimizer_G.zero_grad()
        
        # Forward cycle: A -> B -> A
        fake_B = G_AB(real_A)
        rec_A = G_BA(fake_B)
        
        # Backward cycle: B -> A -> B
        fake_A = G_BA(real_B)
        rec_B = G_AB(fake_A)
        
        # Identity loss (optional)
        idt_A = idt_B = None
        if config.training.lambda_identity > 0:
            idt_A = G_BA(real_A)
            idt_B = G_AB(real_B)
        
        # Discriminator outputs for generator loss
        D_A_fake = D_A(fake_A)
        D_B_fake = D_B(fake_B)
        
        # Compute generator loss
        loss_G, loss_dict_G = criterion.compute_generator_loss(
            real_A, real_B, fake_A, fake_B, rec_A, rec_B, idt_A, idt_B, D_A_fake, D_B_fake
        )
        
        loss_G.backward()
        
        # Gradient clipping
        if config.training.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                itertools.chain(G_AB.parameters(), G_BA.parameters()),
                config.training.max_grad_norm
            )
        
        optimizer_G.step()
        
        # =====================
        # Train Discriminator A
        # =====================
        optimizer_D_A.zero_grad()
        
        # Real images
        D_A_real = D_A(real_A)
        
        # Fake images from pool
        fake_A_pooled = fake_A_pool.query(fake_A)
        D_A_fake = D_A(fake_A_pooled.detach())
        
        # Compute discriminator A loss
        loss_D_A, loss_dict_D_A = criterion.compute_discriminator_loss(D_A_real, D_A_fake)
        loss_D_A.backward()
        optimizer_D_A.step()
        
        # =====================
        # Train Discriminator B
        # =====================
        optimizer_D_B.zero_grad()
        
        # Real images
        D_B_real = D_B(real_B)
        
        # Fake images from pool
        fake_B_pooled = fake_B_pool.query(fake_B)
        D_B_fake = D_B(fake_B_pooled.detach())
        
        # Compute discriminator B loss
        loss_D_B, loss_dict_D_B = criterion.compute_discriminator_loss(D_B_real, D_B_fake)
        loss_D_B.backward()
        optimizer_D_B.step()
        
        # =====================
        # Logging and statistics
        # =====================
        # Store losses
        epoch_losses['G_total'].append(loss_G.item())
        epoch_losses['G_A'].append(loss_dict_G['G_A'].item())
        epoch_losses['G_B'].append(loss_dict_G['G_B'].item())
        epoch_losses['cycle_A'].append(loss_dict_G['cycle_A'].item())
        epoch_losses['cycle_B'].append(loss_dict_G['cycle_B'].item())
        epoch_losses['D_A'].append(loss_D_A.item())
        epoch_losses['D_B'].append(loss_D_B.item())
        
        if 'identity_A' in loss_dict_G:
            epoch_losses['identity_A'].append(loss_dict_G['identity_A'].item())
            epoch_losses['identity_B'].append(loss_dict_G['identity_B'].item())
        
        # Update progress bar
        progress_bar.set_postfix({
            'G': f"{loss_G.item():.3f}",
            'D_A': f"{loss_D_A.item():.3f}",
            'D_B': f"{loss_D_B.item():.3f}"
        })
        
        # Log to external services
        global_step = epoch * len(dataloader) + batch_idx
        
        if batch_idx % config.logging.print_freq == 0:
            log_dict = {
                'train/loss_G': loss_G.item(),
                'train/loss_D_A': loss_D_A.item(),
                'train/loss_D_B': loss_D_B.item(),
                'train/loss_cycle_A': loss_dict_G['cycle_A'].item(),
                'train/loss_cycle_B': loss_dict_G['cycle_B'].item(),
                'epoch': epoch
            }
            
            if 'wandb' in loggers:
                loggers['wandb'].log(log_dict)
            
            if 'tensorboard' in loggers:
                for key, value in log_dict.items():
                    if key != 'epoch':
                        loggers['tensorboard'].add_scalar(key, value, global_step)
        
        # Display images
        if batch_idx % config.logging.display_freq == 0:
            display_images(real_A, real_B, fake_A, fake_B, rec_A, rec_B, 
                         epoch, batch_idx, config, loggers)
    
    # Update learning rate schedulers
    if schedulers:
        for scheduler in schedulers:
            scheduler.step()
    
    # Calculate epoch averages
    epoch_avg_losses = {key: np.mean(losses) for key, losses in epoch_losses.items()}

    # Append to history file
    history_path = os.path.join(config.paths.results_dir, 'loss_history.json')
    try:
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                history = json.load(f)
        else:
            history = {'G_total': [], 'D_A': [], 'D_B': []}

        history['G_total'].append(float(epoch_avg_losses['G_total']))
        history['D_A'].append(float(epoch_avg_losses['D_A']))
        history['D_B'].append(float(epoch_avg_losses['D_B']))

        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        print(f"Warning: could not write loss history: {e}")

    # Plot loss curves and save
    try:
        import matplotlib.pyplot as plt
        epochs = list(range(1, len(history['G_total']) + 1))
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, history['G_total'], label='Generator')
        plt.plot(epochs, history['D_A'], label='Discriminator A')
        plt.plot(epochs, history['D_B'], label='Discriminator B')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Curves')
        plt.legend()
        plt.grid(True)
        plot_path = os.path.join(config.paths.results_dir, 'loss_curves.png')
        plt.savefig(plot_path, dpi=150)
        plt.close()

        # Log to wandb and tensorboard
        if 'wandb' in loggers:
            try:
                loggers['wandb'].log({
                    'loss_curves': wandb.Image(plot_path)
                }, step=epoch)
            except Exception:
                pass
        if 'tensorboard' in loggers:
            try:
                img = Image.open(plot_path)
                img_tensor = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
                # TensorBoard has add_image; instead we log scalar histories
                for i, val in enumerate(history['G_total']):
                    loggers['tensorboard'].add_scalar('train/G_total', val, i+1)
                for i, val in enumerate(history['D_A']):
                    loggers['tensorboard'].add_scalar('train/D_A', val, i+1)
                for i, val in enumerate(history['D_B']):
                    loggers['tensorboard'].add_scalar('train/D_B', val, i+1)
            except Exception:
                pass
    except Exception as e:
        print(f"Warning: could not plot loss curves: {e}")

    return epoch_avg_losses


def display_images(real_A, real_B, fake_A, fake_B, rec_A, rec_B, 
                  epoch, batch_idx, config, loggers, prefix='train'):
    """Display generated images"""
    with torch.no_grad():
        # Take first image from batch
        images = {
            'real_A': real_A[0],
            'fake_B': fake_B[0],
            'rec_A': rec_A[0],
            'real_B': real_B[0],
            'fake_A': fake_A[0],
            'rec_B': rec_B[0]
        }
        
        # Convert to numpy and denormalize
        def tensor_to_image(tensor):
            image = tensor.cpu().detach()
            image = (image + 1.0) / 2.0  # Denormalize from [-1, 1] to [0, 1]
            image = torch.clamp(image, 0, 1)
            return image.permute(1, 2, 0).numpy()
        
        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        titles = [
            ['Real Lagos', 'Generated Duplex', 'Reconstructed Lagos'],
            ['Real Duplex', 'Generated Lagos', 'Reconstructed Duplex']
        ]
        image_keys = [
            ['real_A', 'fake_B', 'rec_A'],
            ['real_B', 'fake_A', 'rec_B']
        ]
        
        for i in range(2):
            for j in range(3):
                key = image_keys[i][j]
                axes[i, j].imshow(tensor_to_image(images[key]))
                axes[i, j].set_title(titles[i][j])
                axes[i, j].axis('off')
        
        plt.tight_layout()
        plt.suptitle(f'Epoch {epoch+1}, Batch {batch_idx}', y=1.02)
        
        # Save figure
        save_path = os.path.join(config.paths.results_dir, f'{prefix}_epoch_{epoch+1}_batch_{batch_idx}.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        
        # Log to external services
        if 'wandb' in loggers:
            loggers['wandb'].log({
                f"{prefix}/generated_images": wandb.Image(save_path)
            })
        
        if 'tensorboard' in loggers:
            # Create grid for tensorboard
            img_grid = torch.cat([
                torch.cat([images['real_A'], images['fake_B'], images['rec_A']], dim=2),
                torch.cat([images['real_B'], images['fake_A'], images['rec_B']], dim=2)
            ], dim=1)
            loggers['tensorboard'].add_image(f'{prefix}/generated_images', img_grid, epoch)
        
        plt.close()


def validate_model(models, val_loader, criterion, device, epoch, config, loggers):
    """Validate the model"""
    G_AB, G_BA, D_A, D_B = models
    
    # Set models to evaluation mode
    G_AB.eval()
    G_BA.eval()
    D_A.eval()
    D_B.eval()
    
    val_losses = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            real_A = batch['A'].to(device)
            real_B = batch['B'].to(device)
            
            # Forward pass
            fake_B = G_AB(real_A)
            fake_A = G_BA(real_B)
            rec_A = G_BA(fake_B)
            rec_B = G_AB(fake_A)
            
            # Compute cycle loss only for validation
            cycle_loss_A = criterion.cycle_loss(rec_A, real_A)
            cycle_loss_B = criterion.cycle_loss(rec_B, real_B)
            total_cycle_loss = cycle_loss_A + cycle_loss_B
            
            val_losses.append(total_cycle_loss.item())
            
            # Generate sample images for first batch
            if batch_idx == 0:
                display_images(real_A, real_B, fake_A, fake_B, rec_A, rec_B,
                             epoch, 0, config, loggers, prefix='val')
    
    avg_val_loss = np.mean(val_losses)
    
    # Log validation results
    if 'wandb' in loggers:
        loggers['wandb'].log({'val/cycle_loss': avg_val_loss, 'epoch': epoch})
    
    if 'tensorboard' in loggers:
        loggers['tensorboard'].add_scalar('val/cycle_loss', avg_val_loss, epoch)
    
    return avg_val_loss


def compute_fid_cyclegan(G_AB, real_B_dir, val_loader, device, num_images=500, batch_size=16):
    """Compute FID for CycleGAN generated images (A->B) against real B images.
    Uses pytorch-fid if available; otherwise returns None.
    """
    try:
        try:
            from pytorch_fid import fid_score
            use_pytorch_fid = True
        except Exception:
            print("Please install 'pytorch-fid' to compute FID (pip install pytorch-fid)")
            return None

        import tempfile
        import shutil
        from PIL import Image
        to_pil = lambda t: Image.fromarray(((t.cpu().numpy().transpose(1,2,0) * 127.5) + 127.5).astype('uint8'))

        temp_dir = Path(tempfile.mkdtemp(prefix='cyclegan_fid_'))
        generated_count = 0

        G_AB.eval()
        with torch.no_grad():
            for batch in val_loader:
                real_A = batch['A'].to(device)
                cur_bs = real_A.shape[0]
                fake_B = G_AB(real_A)

                # Convert fake_B from [-1,1] to [0,255]
                if isinstance(fake_B, torch.Tensor):
                    if fake_B.min() < -0.5:
                        fake_B = (fake_B + 1.0) / 2.0
                    fake_B = torch.clamp(fake_B, 0.0, 1.0)

                    for b in range(fake_B.shape[0]):
                        img = to_pil(fake_B[b])
                        img.save(temp_dir / f"gen_{generated_count:06d}.png")
                        generated_count += 1
                        if generated_count >= num_images:
                            break
                if generated_count >= num_images:
                    break

        if generated_count == 0:
            shutil.rmtree(temp_dir)
            return None

        # Compute FID
        fid_value = fid_score.calculate_fid_given_paths(
            [str(real_B_dir), str(temp_dir)],
            batch_size=batch_size,
            device=device,
            dims=2048
        )

        # Cleanup
        shutil.rmtree(temp_dir)

        return float(fid_value)

    except Exception as e:
        print(f"Error computing FID for CycleGAN: {e}")
        return None


def save_checkpoint(models, optimizers, schedulers, epoch, config, 
                   filename=None, is_best=False):
    """Save model checkpoint"""
    G_AB, G_BA, D_A, D_B = models
    optimizer_G, optimizer_D_A, optimizer_D_B = optimizers
    
    if filename is None:
        filename = f'checkpoint_epoch_{epoch+1}.pt'
    
    checkpoint_path = os.path.join(config.paths.checkpoints_dir, filename)
    
    # Prepare checkpoint data
    checkpoint = {
        'epoch': epoch + 1,
        'config': config.to_dict(),
        'G_AB_state_dict': G_AB.state_dict(),
        'G_BA_state_dict': G_BA.state_dict(),
        'D_A_state_dict': D_A.state_dict(),
        'D_B_state_dict': D_B.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),
        'optimizer_D_A_state_dict': optimizer_D_A.state_dict(),
        'optimizer_D_B_state_dict': optimizer_D_B.state_dict(),
    }
    
    # Add scheduler states if available
    if schedulers:
        checkpoint['scheduler_states'] = [s.state_dict() for s in schedulers]
    
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")
    
    # Save as latest
    latest_path = os.path.join(config.paths.checkpoints_dir, 'latest.pt')
    torch.save(checkpoint, latest_path)
    
    # Save as best if specified
    if is_best:
        best_path = os.path.join(config.paths.checkpoints_dir, 'best.pt')
        torch.save(checkpoint, best_path)
        print("Best model saved!")
    
    return checkpoint_path

log_checkpoint_to_wandb(checkpoint_path, config_path)


def load_checkpoint(checkpoint_path, models, optimizers, schedulers, device):
    """Load model checkpoint"""
    print(f"Loading checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    G_AB, G_BA, D_A, D_B = models
    optimizer_G, optimizer_D_A, optimizer_D_B = optimizers
    
    def _unwrap(module):
        return module.module if isinstance(module, torch.nn.DataParallel) else module

    # Load model states (handle DataParallel/non-DataParallel transparently)
    G_AB_module = _unwrap(G_AB)
    G_BA_module = _unwrap(G_BA)
    D_A_module = _unwrap(D_A)
    D_B_module = _unwrap(D_B)

    G_AB_module.load_state_dict(checkpoint['G_AB_state_dict'])
    G_BA_module.load_state_dict(checkpoint['G_BA_state_dict'])
    D_A_module.load_state_dict(checkpoint['D_A_state_dict'])
    D_B_module.load_state_dict(checkpoint['D_B_state_dict'])
    
    # Load optimizer states
    optimizer_G_module = _unwrap(optimizer_G)
    optimizer_D_A_module = _unwrap(optimizer_D_A)
    optimizer_D_B_module = _unwrap(optimizer_D_B)

    optimizer_G_module.load_state_dict(checkpoint['optimizer_G_state_dict'])
    optimizer_D_A_module.load_state_dict(checkpoint['optimizer_D_A_state_dict'])
    optimizer_D_B_module.load_state_dict(checkpoint['optimizer_D_B_state_dict'])
    
    # Load scheduler states if available
    if schedulers and 'scheduler_states' in checkpoint:
        for scheduler, state in zip(schedulers, checkpoint['scheduler_states']):
            _unwrap(scheduler).load_state_dict(state)
    
    start_epoch = checkpoint['epoch']
    print(f"Resumed from epoch {start_epoch}")
    
    return start_epoch


def create_config_from_args(args):
    """Create configuration from command line arguments"""
    if args.config:
        # Load from config file
        config = CycleGANConfig.load(args.config)
    elif args.quick_test:
        # Use quick test configuration
        config = get_quick_test_config()
    else:
        # Use default configuration
        config = get_default_config()
    
    # Override with command line arguments
    if args.data_path:
        config.data.data_path = args.data_path
    if args.output_dir:
        config.paths.checkpoints_dir = args.output_dir
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.num_epochs:
        config.training.num_epochs = args.num_epochs
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate
    if args.image_size:
        config.data.image_size = args.image_size
    if args.lambda_cycle:
        config.training.lambda_cycle = args.lambda_cycle
    if args.lambda_identity:
        config.training.lambda_identity = args.lambda_identity
    if args.wandb_project:
        config.logging.wandb_project = args.wandb_project
    if args.no_wandb:
        config.logging.use_wandb = False
    
    # Set resume checkpoint
    config.paths.resume_checkpoint = args.resume
    
    return config


def main():
    args = parse_args()
    
    # Create configuration
    config = create_config_from_args(args)
    config.validate()
    
    # Setup
    device = setup_training(config)
    loggers = setup_logging(config)
    train_loader, val_loader = load_dataset(config)
    
    # Create models
    G_AB, G_BA, D_A, D_B = create_models(config, device)
    models = (G_AB, G_BA, D_A, D_B)
    
    # Setup optimizers and schedulers
    optimizers = setup_optimizers(G_AB, G_BA, D_A, D_B, config)
    schedulers = setup_schedulers(optimizers, config)
    
    # Setup loss function
    criterion = CycleGANLoss(
        gan_mode=config.training.gan_mode,
        lambda_cycle=config.training.lambda_cycle,
        lambda_identity=config.training.lambda_identity,
        lambda_perceptual=config.training.lambda_perceptual,
        cycle_loss_type=config.training.cycle_loss_type,
        identity_loss_type=config.training.identity_loss_type
    )
    
    # Move loss functions to device
    criterion.gan_loss = criterion.gan_loss.to(device)
    criterion.cycle_loss = criterion.cycle_loss.to(device)
    criterion.identity_loss = criterion.identity_loss.to(device)
    if criterion.perceptual_loss:
        criterion.perceptual_loss = criterion.perceptual_loss.to(device)
    
    # Setup image pools
    fake_A_pool = ImagePool(config.training.pool_size)
    fake_B_pool = ImagePool(config.training.pool_size)
    fake_pools = (fake_A_pool, fake_B_pool)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if config.paths.resume_checkpoint and os.path.exists(config.paths.resume_checkpoint):
        start_epoch = load_checkpoint(config.paths.resume_checkpoint, models, optimizers, schedulers, device)
    
    # Save configuration
    config_save_path = os.path.join(config.paths.checkpoints_dir, 'config.json')
    config.save(config_save_path)
    
    print("Starting Lagos2Duplex CycleGAN training...")
    # Heartbeat: ensure a training heartbeat log is written immediately so external pollers can detect activity
    try:
        os.makedirs(config.paths.logs_dir, exist_ok=True)
        heartbeat_path = os.path.join(config.paths.logs_dir, 'training_heartbeat.log')
        with open(heartbeat_path, 'a') as _hf:
            _hf.write(f"Training started at {datetime.now().isoformat()}\n")
    except Exception:
        pass
    print(f"Image size: {config.data.image_size}x{config.data.image_size}")
    print(f"Batch size: {config.training.batch_size}")
    print(f"Epochs: {config.training.num_epochs}")
    print(f"Lambda cycle: {config.training.lambda_cycle}")
    print(f"Lambda identity: {config.training.lambda_identity}")
    print(f"Device: {device}")
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, config.training.num_epochs):
        # write epoch heartbeat immediately
        try:
            with open(heartbeat_path, 'a') as _hf:
                _hf.write(f"Epoch {epoch+1} start at {datetime.now().isoformat()}\n")
        except Exception:
            pass
        # Train for one epoch
        epoch_losses = train_epoch(
            models, optimizers, schedulers, criterion, fake_pools,
            train_loader, device, epoch, config, loggers
        )
        
        # Validate model
        if len(val_loader) > 0:
            val_loss = validate_model(models, val_loader, criterion, device, epoch, config, loggers)
            
            # Check if this is the best model
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
        else:
            is_best = False
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{config.training.num_epochs} completed:")
        print(f"  Generator Loss: {epoch_losses['G_total']:.4f}")
        print(f"  Discriminator A Loss: {epoch_losses['D_A']:.4f}")
        print(f"  Discriminator B Loss: {epoch_losses['D_B']:.4f}")
        print(f"  Cycle Loss A: {epoch_losses['cycle_A']:.4f}")
        print(f"  Cycle Loss B: {epoch_losses['cycle_B']:.4f}")
        if len(val_loader) > 0:
            print(f"  Validation Loss: {val_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % config.logging.save_epoch_freq == 0:
            save_checkpoint(models, optimizers, schedulers, epoch, config, is_best=is_best)
        
        # Evaluate FID if configured
        if config.logging.eval_fid_freq > 0 and (epoch + 1) % config.logging.eval_fid_freq == 0:
            real_B_dir = getattr(config.data, 'val_B_dir', None)

            if not real_B_dir:
                # Fall back to conventional dataset layout if explicit path is missing
                inferred_dir = Path(config.data.data_path) / 'lagos2duplex' / 'duplex' / 'val'
                if inferred_dir.exists():
                    real_B_dir = str(inferred_dir)

            if real_B_dir and os.path.isdir(real_B_dir):
                fid_value = compute_fid_cyclegan(G_AB, real_B_dir, val_loader, device)
                if fid_value is not None:
                    print(f"  FID: {fid_value:.4f}")
                    # Log FID to wandb and tensorboard
                    if 'wandb' in loggers:
                        loggers['wandb'].log({'eval/fid': fid_value, 'epoch': epoch})
                    if 'tensorboard' in loggers:
                        loggers['tensorboard'].add_scalar('eval/fid', fid_value, epoch)
            else:
                print("  Skipping FID evaluation: validation duplex directory not configured or missing.")
    
    # Save final checkpoint
    save_checkpoint(models, optimizers, schedulers, config.training.num_epochs - 1, config, 
                   filename='final_checkpoint.pt')
    
    # Close loggers
    if 'wandb' in loggers:
        loggers['wandb'].finish()
    if 'tensorboard' in loggers:
        loggers['tensorboard'].close()
    
    print("Training completed successfully!")


if __name__ == "__main__":
    main()
