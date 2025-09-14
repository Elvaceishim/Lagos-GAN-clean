import sys
import os
# ensure repo root is on sys.path so local package imports work when running script directly
sys.path.insert(0, os.getcwd())

import argparse
import torch
import os
import pprint
import importlib

def find_generator_state(ck):
    for k in ("G_AB_state_dict","G_A_state_dict","G_state_dict","generator_state_dict","netG_A","netG"):
        if k in ck:
            return ck[k], k
    for k,v in ck.items():
        if isinstance(v, dict):
            return v, k
    return None, None

def build_generator_from_config(config):
    """
    Try to instantiate the generator using the project's CycleGAN implementation.
    Falls back to define_G if available.
    """
    input_nc = config.get("input_nc", 3)
    output_nc = config.get("output_nc", 3)
    ngf = config.get("ngf", config.get("ndf", 64))
    netG_name = config.get("netG", "resnet_9blocks")
    norm = config.get("norm", "instance")
    use_dropout = config.get("use_dropout", False)
    init_type = config.get("init_type", "normal")
    init_gain = config.get("init_gain", 0.02)

    # Prefer direct imports of known classes/functions from the package
    m = importlib.import_module('lagos2duplex.models')
    define_G = getattr(m, 'define_G', None)
    CycleGANGenerator = getattr(m, 'CycleGANGenerator', None)
    get_norm_layer = getattr(m, 'get_norm_layer', None)

    # If CycleGANGenerator is available, instantiate with proper args
    if CycleGANGenerator is not None and get_norm_layer is not None:
        norm_layer = get_norm_layer(norm_type=norm)
        n_blocks = 9 if netG_name == 'resnet_9blocks' else 6
        # CycleGANGenerator signature: (input_nc, output_nc, ngf, norm_layer=..., use_dropout=False, n_blocks=9)
        return CycleGANGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=n_blocks)

    # Fallback to define_G if present
    if define_G is not None:
        return define_G(input_nc, output_nc, ngf, netG_name, norm=norm, use_dropout=use_dropout, init_type=init_type, init_gain=init_gain, gpu_ids=[])

    raise RuntimeError(
        "Could not instantiate generator automatically. Edit scripts/export_torchscript.py to construct the generator."
    )

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--img_size", type=int, default=64)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    ck = torch.load(args.checkpoint, map_location=device)
    if not isinstance(ck, dict):
        raise RuntimeError("Checkpoint must be a dict. Got: " + str(type(ck)))

    state, found_key = find_generator_state(ck)
    if state is None:
        print("Could not find a generator state dict in checkpoint. Top-level keys:")
        pprint.pprint(list(ck.keys())[:200])
        raise RuntimeError("No generator state found. Edit the script.")

    print("Found generator state under key:", found_key)

    config = {}
    if "config" in ck and isinstance(ck["config"], dict):
        config = ck["config"].get("model", {}) if isinstance(ck["config"].get("model", {}), dict) else ck["config"]

    print("Using model config:", config)

    netG = build_generator_from_config(config)

    try:
        netG.load_state_dict(state)
    except Exception as e:
        print("First load_state_dict failed:", e)
        new_state = {k.replace("module.", ""): v for k,v in state.items()}
        try:
            netG.load_state_dict(new_state)
            print("Loaded state after stripping 'module.' prefix.")
        except Exception as e2:
            print("Second load attempt failed. Sample keys:")
            pprint.pprint(list(state.keys())[:40])
            raise RuntimeError("Failed to load state into generator: " + str(e2))

    netG.to(device)
    netG.eval()

    dummy = torch.randn(1, 3, args.img_size, args.img_size, device=device)
    with torch.no_grad():
        traced = torch.jit.trace(netG, dummy, strict=False)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    traced.save(args.out)
    print("Saved TorchScript model to:", args.out)

if __name__ == "__main__":
    main()