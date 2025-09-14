"""Generate images from trained CycleGAN (G_AB) and evaluate diversity and realism against the val set.
Saves comparison images and a small summary report.

Notes:
- Uses simple L2-based diversity and nearest-neighbor L2 as a realism proxy.
- If `lpips` is installed, uses LPIPS as a perceptual realism metric.
- Warns about small Duplex dataset size and possible overfitting.
"""
import os, sys, traceback
sys.path.insert(0, os.getcwd())
from datetime import datetime
print('Generate+Eval run at:', datetime.now().isoformat())
print('CWD:', os.getcwd())

import torch
import numpy as np
from PIL import Image
import torchvision.utils as vutils

# Helpers
def save_side_by_side(real, generated, save_path):
    # real, generated are tensors in [C,H,W] with values in [-1,1] or [0,1]
    def to_pil(t):
        t = t.detach().cpu()
        # If in [-1,1]
        if t.min() < -0.5:
            t = (t + 1.0) / 2.0
        t = torch.clamp(t, 0, 1)
        arr = (t.numpy().transpose(1,2,0) * 255).astype('uint8')
        return Image.fromarray(arr)
    r = to_pil(real)
    g = to_pil(generated)
    # Concatenate horizontally
    new = Image.new('RGB', (r.width + g.width, r.height))
    new.paste(r, (0, 0))
    new.paste(g, (r.width, 0))
    new.save(save_path)

# Try to import LPIPS
have_lpips = False
try:
    import lpips
    lpips_alex = lpips.LPIPS(net='alex')
    have_lpips = True
    print('LPIPS available: using perceptual metric')
except Exception:
    print('LPIPS not available; falling back to L2/SSIM proxies')

# Load project modules and config
try:
    from lagos2duplex.config import get_quick_test_config, CycleGANConfig
    from lagos2duplex.dataset import create_dataloaders
    from lagos2duplex.train import create_models
except Exception as e:
    print('Failed to import project modules:', e)
    traceback.print_exc()
    raise

# Prepare config and device
cfg = get_quick_test_config()
# Use small sizes for fast generation
cfg.data.image_size = 64
cfg.training.batch_size = 2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# Create models
try:
    G_AB, G_BA, D_A, D_B = create_models(cfg, device)
    print('Models instantiated')
except Exception as e:
    print('Failed to create models:', e)
    traceback.print_exc()
    raise

# Try to load latest checkpoint if available
ckpt_path = os.path.join(cfg.paths.checkpoints_dir, 'latest.pt')
if os.path.exists(ckpt_path):
    try:
        ckpt = torch.load(ckpt_path, map_location=device)
        print('Loaded checkpoint:', ckpt_path)
        # Load states if available
        if 'G_AB_state_dict' in ckpt:
            G_AB.load_state_dict(ckpt['G_AB_state_dict'])
        if 'G_BA_state_dict' in ckpt:
            G_BA.load_state_dict(ckpt['G_BA_state_dict'])
        print('Model weights loaded from checkpoint')
    except Exception as e:
        print('Failed to load checkpoint weights:', e)
        traceback.print_exc()
else:
    print('No checkpoint found at', ckpt_path, '- generating with current weights (likely random)')

G_AB.eval()

# Create dataloaders
data_path = cfg.data.data_path
train_loader, val_loader = create_dataloaders(data_path=data_path, batch_size=cfg.training.batch_size, image_size=cfg.data.image_size, num_workers=0)

# Prepare results dirs
out_dir = os.path.join(cfg.paths.results_dir, 'gen_vs_val')
os.makedirs(out_dir, exist_ok=True)
print('Saving generated comparisons to:', out_dir)

# Collect generated images and real val images (A domain -> generate B)
num_samples = min(50, len(val_loader) * cfg.training.batch_size)
print('Will process up to', num_samples, 'validation samples')

generated_list = []
real_list = []
filenames = []
count = 0
with torch.no_grad():
    for batch in val_loader:
        real_A = batch['A'].to(device)
        bs = real_A.shape[0]
        fake_B = G_AB(real_A)
        for i in range(bs):
            if count >= num_samples:
                break
            gen = fake_B[i].cpu()
            real = batch['B'][i].cpu()
            # save side-by-side
            save_path = os.path.join(out_dir, f'pair_{count:04d}.png')
            try:
                save_side_by_side(real, gen, save_path)
            except Exception:
                # fallback: use torchvision save_image grid
                vutils.save_image(torch.stack([real, gen]), save_path, normalize=True, value_range=None)
            generated_list.append(np.array(( (gen.numpy().transpose(1,2,0) + 1)/2.0 )))
            real_list.append(np.array(( (real.numpy().transpose(1,2,0) + 1)/2.0 )))
            filenames.append(save_path)
            count += 1
        if count >= num_samples:
            break

if count == 0:
    print('No validation images processed; exiting')
    sys.exit(0)

# Convert lists to arrays in float [0,1]
generated_arr = np.stack(generated_list)
real_arr = np.stack(real_list)
print('Generated array shape:', generated_arr.shape)

# Compute diversity: mean pairwise L2 between generated images
try:
    n = generated_arr.shape[0]
    flat = generated_arr.reshape(n, -1)
    diffs = []
    for i in range(n):
        for j in range(i+1, n):
            diffs.append(np.linalg.norm(flat[i] - flat[j]))
    mean_pairwise_l2 = float(np.mean(diffs)) if len(diffs) > 0 else 0.0
    print('Diversity (mean pairwise L2) over generated set:', mean_pairwise_l2)
except Exception as e:
    print('Failed to compute diversity:', e)
    mean_pairwise_l2 = None

# Compute realism proxy: nearest-neighbor L2 to validation real set
try:
    from sklearn.metrics import pairwise_distances
    dists = pairwise_distances(flat, real_arr.reshape(real_arr.shape[0], -1), metric='euclidean')
    nn_dists = dists.min(axis=1)
    mean_nn_l2 = float(nn_dists.mean())
    print('Realism proxy (mean nearest-neighbor L2 to real val images):', mean_nn_l2)
except Exception as e:
    print('sklearn not available or failed; fallback to simple numpy NN compute')
    try:
        real_flat = real_arr.reshape(real_arr.shape[0], -1)
        dists = np.sqrt(((flat[:,None,:] - real_flat[None,:,:])**2).sum(axis=2))
        nn_dists = dists.min(axis=1)
        mean_nn_l2 = float(nn_dists.mean())
        print('Realism proxy (mean NN L2):', mean_nn_l2)
    except Exception as e2:
        print('Failed to compute NN realism proxy:', e2)
        mean_nn_l2 = None

# LPIPS if available: mean LPIPS to nearest real
mean_lpips = None
if have_lpips:
    try:
        import torch
        gens = torch.tensor((generated_arr * 2.0 - 1.0).transpose(0,3,1,2)).float().to(device)
        reals = torch.tensor((real_arr * 2.0 - 1.0).transpose(0,3,1,2)).float().to(device)
        # compute pairwise LPIPS (n x m)
        n = gens.shape[0]
        m = reals.shape[0]
        lp_vals = np.zeros((n,m))
        for i in range(n):
            for j in range(m):
                val = lpips_alex(gens[i:i+1], reals[j:j+1])
                lp_vals[i,j] = float(val.cpu().item())
        nn_lp = lp_vals.min(axis=1)
        mean_lpips = float(nn_lp.mean())
        print('Mean LPIPS to nearest real:', mean_lpips)
    except Exception as e:
        print('LPIPS computation failed:', e)
        mean_lpips = None

# Compute KID (Kernel Inception Distance) using ResNet50 features as a proxy
mean_kid = None
try:
    import torchvision.models as models
    from torch import nn
    device_feat = device
    # Load pretrained ResNet50 as feature extractor (remove final fc)
    resnet = models.resnet50(pretrained=True).to(device_feat)
    feat_extractor = nn.Sequential(*list(resnet.children())[:-1]).eval()

    # Helper to extract features in batches from numpy images in [0,1]
    def extract_features(arr):
        bsize = 16
        feats = []
        with torch.no_grad():
            for i in range(0, arr.shape[0], bsize):
                batch = arr[i:i+bsize]
                tensor = torch.tensor((batch * 2.0 - 1.0).transpose(0,3,1,2)).float().to(device_feat)
                # Convert from [-1,1] to [0,1]
                tensor = (tensor + 1.0) / 2.0
                # ImageNet normalization
                mean = torch.tensor([0.485, 0.456, 0.406], device=device_feat).view(1,3,1,1)
                std = torch.tensor([0.229, 0.224, 0.225], device=device_feat).view(1,3,1,1)
                tensor = (tensor - mean) / std
                feat = feat_extractor(tensor)
                feat = feat.view(feat.size(0), -1).cpu().numpy()
                feats.append(feat)
        if len(feats) == 0:
            return np.zeros((0, feat_extractor(torch.zeros(1,3,cfg.data.image_size,cfg.data.image_size).to(device_feat)).view(1,-1).shape[1]))
        return np.vstack(feats)

    gen_feats = extract_features(generated_arr)
    real_feats = extract_features(real_arr)

    # Polynomial kernel MMD^2 estimator (degree=3, c=1)
    def poly_kernel(x, y, degree=3, c=1.0):
        d = x.shape[1]
        return ((x.dot(y.T) / float(d)) + c) ** degree

    if gen_feats.shape[0] > 1 and real_feats.shape[0] > 1:
        XX = poly_kernel(gen_feats, gen_feats)
        YY = poly_kernel(real_feats, real_feats)
        XY = poly_kernel(gen_feats, real_feats)
        n = gen_feats.shape[0]
        m = real_feats.shape[0]
        sum_xx = (np.sum(XX) - np.trace(XX)) / (n * (n - 1))
        sum_yy = (np.sum(YY) - np.trace(YY)) / (m * (m - 1))
        sum_xy = np.sum(XY) / (n * m)
        mean_kid = float(sum_xx + sum_yy - 2 * sum_xy)
        print('KID (ResNet proxy) computed:', mean_kid)
    else:
        mean_kid = None
except Exception as e:
    print('KID computation failed or torchvision not available:', e)
    mean_kid = None

# Save summary
summary_path = os.path.join(out_dir, 'summary.txt')
with open(summary_path, 'w') as f:
    f.write(f'Generate+Eval run at: {datetime.now().isoformat()}\n')
    f.write(f'Num processed: {count}\n')
    f.write(f'Diversity_mean_pairwise_l2: {mean_pairwise_l2}\n')
    f.write(f'Realism_mean_nn_l2: {mean_nn_l2}\n')
    f.write(f'Mean_LPIPS_to_nn: {mean_lpips}\n')
    f.write(f'Mean_KID: {mean_kid}\n')

print('\nSummary:')
print('  Samples processed:', count)
print('  Diversity (mean pairwise L2):', mean_pairwise_l2)
print('  Realism proxy (mean NN L2):', mean_nn_l2)
print('  Mean LPIPS to NN (if available):', mean_lpips)
print('  Mean KID (if available):', mean_kid)
print('\nSaved comparison images and summary at:', out_dir)

# Caution about Duplex dataset size
print('\nCaution: Duplex dataset is smaller and may lead to overfitting. Interpret diversity/realism metrics accordingly.')
