"""Quick test script (updated)
- Passes a detected data_path to create_dataloaders
- Passes device when calling create_models if required
- Adapts latent/vector size for AfroCover generator by inspecting mapping layer

Saves human-readable prints to stdout (use tee to capture to a file)
"""
import os, sys, traceback
# Ensure local project packages (lagos2duplex, afrocover) are importable
sys.path.insert(0, os.getcwd())
from datetime import datetime
print('Quick tests run at:', datetime.now().isoformat())
print('CWD:', os.getcwd())

# Torch/device
try:
    import torch
    print('PyTorch version:', torch.__version__)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
except Exception as e:
    print('Failed to import torch:', e)
    traceback.print_exc()
    raise

# 1) Lagos2Duplex dataset load (with correct data_path)
print('\n--- Lagos2Duplex dataset quick load test ---')
try:
    from lagos2duplex.dataset import create_dataloaders

    # Auto-detect a reasonable data_path
    candidates = [
        './data',
        './data_processed',
        './data/processed',
        './datasets',
        './data/lagos2duplex',
        './data/afrocover',
    ]
    data_path = None
    for p in candidates:
        if os.path.exists(p):
            data_path = p
            break
    if data_path is None:
        data_path = './data'  # fallback
        print('No common data_path found; using fallback:', data_path)
    else:
        print('Using detected data_path:', data_path)

    # Create small dataloaders for quick test
    try:
        train_loader, val_loader = create_dataloaders(data_path=data_path, batch_size=2, image_size=64, num_workers=0)
    except TypeError:
        # Some codebases return (train,val) or dicts; try positional
        loaders = create_dataloaders(data_path, 2, 64, 0)
        if isinstance(loaders, (list, tuple)):
            train_loader = loaders[0]
            val_loader = loaders[1] if len(loaders) > 1 else None
        else:
            train_loader = loaders
            val_loader = None

    # Attempt to fetch a batch
    try:
        batch = next(iter(train_loader))
        print('Fetched one train batch. Type:', type(batch))
        if isinstance(batch, dict):
            for k,v in batch.items():
                try:
                    if hasattr(v, 'shape'):
                        print(f"  {k} shape: {v.shape}")
                    else:
                        print(f"  {k} type: {type(v)}")
                except Exception:
                    pass
        elif hasattr(batch, '__len__'):
            print('  len(batch)=', len(batch))
        else:
            print('  batch repr:', repr(batch)[:200])
        print('Lagos2Duplex dataset quick load: SUCCESS')
    except Exception as e:
        print('Failed to get a train batch:', e)
        traceback.print_exc()
        raise
except Exception as e:
    print('Lagos2Duplex dataset quick load: FAILED')
    traceback.print_exc()

# 2) CycleGAN one-epoch dry run (include device when calling create_models)
print('\n--- CycleGAN one-epoch dry run (quick) ---')
try:
    import inspect
    from lagos2duplex import train as lagos_train
    # Prepare a minimal cfg-like object if available
    cfg = None
    try:
        from lagos2duplex.config import get_quick_test_config
        cfg = get_quick_test_config()
        if hasattr(cfg, 'image_size'):
            cfg.image_size = 64
        if hasattr(cfg, 'batch_size'):
            cfg.batch_size = 2
    except Exception:
        # create a simple namespace
        from types import SimpleNamespace
        cfg = SimpleNamespace(image_size=64, batch_size=2)

    # Call create_models with device if required
    if hasattr(lagos_train, 'create_models'):
        create_models = lagos_train.create_models
        sig = inspect.signature(create_models)
        kwargs = {}
        # pass cfg if name matches
        if 'cfg' in sig.parameters:
            kwargs['cfg'] = cfg
        # pass device if accepted
        if 'device' in sig.parameters:
            kwargs['device'] = device
        # try to call with kwargs, else try positional
        try:
            models = create_models(**kwargs)
            print('create_models called with kwargs:', kwargs.keys())
        except Exception as e:
            print('create_models(**kwargs) failed, trying positional calls:', e)
            try:
                # common fallback signatures: create_models(cfg, device)
                models = create_models(cfg, device)
                print('create_models(cfg, device) succeeded')
            except Exception as e2:
                print('Failed to instantiate models:', e2)
                raise

        # If there's a train_epoch helper, try to run a minimal step
        if hasattr(lagos_train, 'train_epoch'):
            try:
                # Build optimizers using the expected signature: setup_optimizers(G_AB, G_BA, D_A, D_B, cfg)
                optimizers = None
                try:
                    # models is a tuple (G_AB, G_BA, D_A, D_B)
                    optimizers = lagos_train.setup_optimizers(*models, cfg)
                    print('Optimizers created')
                except Exception as e:
                    print('setup_optimizers failed:', e)
                    optimizers = None

                # Build schedulers
                try:
                    schedulers = lagos_train.setup_schedulers(optimizers, cfg) if optimizers is not None else None
                except Exception:
                    schedulers = None

                # Create loss/criterion and move sub-modules to device
                try:
                    criterion = lagos_train.CycleGANLoss(
                        gan_mode=cfg.training.gan_mode if hasattr(cfg.training, 'gan_mode') else 'lsgan',
                        lambda_cycle=cfg.training.lambda_cycle if hasattr(cfg.training, 'lambda_cycle') else 10.0,
                        lambda_identity=cfg.training.lambda_identity if hasattr(cfg.training, 'lambda_identity') else 0.5,
                        lambda_perceptual=getattr(cfg.training, 'lambda_perceptual', 0.0),
                        cycle_loss_type=getattr(cfg.training, 'cycle_loss_type', 'l1'),
                        identity_loss_type=getattr(cfg.training, 'identity_loss_type', 'l1')
                    )
                    # Move loss components to device if present
                    try:
                        criterion.gan_loss = criterion.gan_loss.to(device)
                        criterion.cycle_loss = criterion.cycle_loss.to(device)
                        criterion.identity_loss = criterion.identity_loss.to(device)
                        if getattr(criterion, 'perceptual_loss', None) is not None:
                            criterion.perceptual_loss = criterion.perceptual_loss.to(device)
                    except Exception:
                        pass
                except Exception as e:
                    print('Failed to create criterion:', e)
                    raise

                # Setup image pools
                try:
                    fake_A_pool = lagos_train.ImagePool(cfg.training.pool_size)
                    fake_B_pool = lagos_train.ImagePool(cfg.training.pool_size)
                    fake_pools = (fake_A_pool, fake_B_pool)
                except Exception:
                    fake_pools = (None, None)

                # Prepare a very small dataloader (subset) so the epoch is quick
                try:
                    import torch.utils.data as data
                    dataset_obj = None
                    try:
                        dataset_obj = train_loader.dataset
                    except Exception:
                        # try to create dataloaders directly
                        dataset_obj = None
                    if dataset_obj is not None:
                        subset_size = min(4, len(dataset_obj))
                        indices = list(range(subset_size))
                        small_dataset = data.Subset(dataset_obj, indices)
                        small_dataloader = data.DataLoader(small_dataset, batch_size=cfg.training.batch_size if hasattr(cfg.training, 'batch_size') else 2, shuffle=True, num_workers=0)
                        print('Small dataloader prepared with', subset_size, 'samples')
                    else:
                        small_dataloader = train_loader
                except Exception as e:
                    print('Failed to prepare small dataloader:', e)
                    small_dataloader = train_loader

                # Setup minimal loggers (disable external services for quick test)
                try:
                    loggers = lagos_train.setup_logging(cfg)
                except Exception:
                    loggers = {}

                # Finally call train_epoch with constructed arguments
                try:
                    lagos_train.train_epoch(models=models, optimizers=optimizers, schedulers=schedulers,
                                             criterion=criterion, fake_pools=fake_pools, dataloader=small_dataloader,
                                             device=device, epoch=0, config=cfg, loggers=loggers)
                    print('CycleGAN dry run: SUCCESS (train_epoch)')
                except TypeError as e:
                    # train_epoch may expect positional args; try positional call
                    print('train_epoch positional attempt due to TypeError:', e)
                    try:
                        lagos_train.train_epoch(models, optimizers, schedulers, criterion, fake_pools, small_dataloader, device, 0, cfg, loggers)
                        print('CycleGAN dry run: SUCCESS (train_epoch positional)')
                    except Exception:
                        print('train_epoch positional call failed')
                        raise
            except Exception as e:
                print('train_epoch call failed:', e)
                traceback.print_exc()
                # do fallback forward/backward test if possible
                if isinstance(models, (list, tuple)):
                    G_AB, G_BA, D_A, D_B = models
                    # try a simple forward/backward on G_AB and D_A
                    try:
                        optimG = torch.optim.Adam(list(G_AB.parameters()) + list(G_BA.parameters()), lr=1e-4)
                        optimD = torch.optim.Adam(list(D_A.parameters()) + list(D_B.parameters()), lr=1e-4)
                        real = torch.randn(2, 3, 64, 64, device=device)
                        fake = G_AB(real)
                        outD_real = D_A(real)
                        outD_fake = D_A(fake.detach())
                        lossD = outD_real.mean() - outD_fake.mean()
                        lossD.backward()
                        optimD.step()
                        optimD.zero_grad()
                        lossG = -outD_fake.mean()
                        lossG.backward()
                        optimG.step()
                        print('CycleGAN fallback forward/backward: SUCCESS')
                    except Exception:
                        print('CycleGAN fallback forward/backward failed')
        else:
            print('No train_epoch helper found; create_models succeeded')
    else:
        raise RuntimeError('lagos2duplex.train.create_models not found')
except Exception as e:
    print('CycleGAN one-epoch dry run: FAILED')
    traceback.print_exc()

# 3) AfroCover StyleGAN2 forward pass (adapt latent sizes)
print('\n--- AfroCover StyleGAN2 forward pass test ---')
try:
    from afrocover import models as afro_models
    G = None
    D = None
    # Prefer known class names
    if hasattr(afro_models, 'StyleGAN2Generator'):
        gen_cls = afro_models.StyleGAN2Generator
    else:
        # find a name with Style or Generator
        gen_cls = None
        for name in dir(afro_models):
            if 'Generator' in name and 'Style' in name:
                gen_cls = getattr(afro_models, name)
                break
    if gen_cls is None:
        raise RuntimeError('No generator class found in afrocover.models')

    # Try sensible constructor signatures
    try:
        G = gen_cls(image_size=64, channel_multiplier=1).to(device)
    except Exception:
        try:
            G = gen_cls(64, 3, 512).to(device)
        except Exception:
            try:
                G = gen_cls().to(device)
            except Exception:
                raise
    print('Instantiated generator:', type(G))

    # Determine expected latent size by inspecting mapping network weights
    latent_dim = None
    try:
        for name, p in G.named_parameters():
            if 'mapping' in name and p.ndim == 2 and 'weight' in name:
                latent_dim = p.shape[1]
                print('Detected mapping in_features from', name, '->', latent_dim)
                break
    except Exception:
        pass
    # Fallback checks
    if latent_dim is None:
        for attr in ('z_dim', 'latent_dim', 'w_dim', 'mapping_size'):
            if hasattr(G, attr):
                latent_dim = getattr(G, attr)
                print('Detected latent dim from attribute', attr, '->', latent_dim)
                break
    if latent_dim is None:
        latent_dim = 512
        print('Falling back to latent_dim =', latent_dim)

    z = torch.randn(2, latent_dim, device=device)
    # Try different forward call styles
    imgs = None
    try:
        imgs = G(z)
    except TypeError:
        try:
            imgs = G(z, None)
        except TypeError:
            try:
                imgs = G(z, truncation_psi=1.0)
            except Exception as e:
                print('Generator forward failed with several calling conventions:', e)
                raise
    print('Generator output type:', type(imgs))
    try:
        print('Generator output shape:', imgs.shape)
    except Exception:
        pass

    # Instantiate discriminator
    if hasattr(afro_models, 'StyleGAN2Discriminator'):
        disc_cls = afro_models.StyleGAN2Discriminator
    else:
        disc_cls = None
        for name in dir(afro_models):
            if 'Discriminator' in name and 'Style' in name:
                disc_cls = getattr(afro_models, name)
                break
    if disc_cls is None:
        raise RuntimeError('No discriminator class found in afrocover.models')
    try:
        D = disc_cls(image_size=imgs.shape[-1], channel_multiplier=1).to(device)
    except Exception:
        try:
            D = disc_cls(imgs.shape[-1], 3).to(device)
        except Exception:
            D = disc_cls().to(device)
    print('Instantiated discriminator:', type(D))

    # Run discriminator on generated images
    logits = D(imgs)
    print('Discriminator output type:', type(logits))
    try:
        print('Discriminator output shape:', logits.shape)
    except Exception:
        pass

    print('AfroCover StyleGAN2 forward pass: SUCCESS')
except Exception as e:
    print('AfroCover StyleGAN2 test: FAILED')
    traceback.print_exc()

print('\nAll tests attempted.')
