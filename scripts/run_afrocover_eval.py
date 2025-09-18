#!/usr/bin/env python3
"""Compute AfroCover StyleGAN metrics and store results."""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import torch

# Ensure repo root on path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from afrocover.models import StyleGAN2Generator  # noqa: E402
from eval.evaluate import AfroCoverEvaluator  # noqa: E402

CHECKPOINT_PATH = REPO_ROOT / "checkpoints" / "afrocover" / "latest.pt"
RAW_ROOT = REPO_ROOT / "data" / "afrocover"
PROCESSED_ROOT = REPO_ROOT / "data_processed" / "afrocover"
REAL_IMAGES_PATH = (
    PROCESSED_ROOT / "test" if (PROCESSED_ROOT / "test").exists() else RAW_ROOT / "test"
)
OUTPUT_JSON = REPO_ROOT / "eval_results.json"

DEFAULT_NUM_GENERATED = 200
DEFAULT_BATCH_SIZE = 20


def load_generator(device: torch.device) -> StyleGAN2Generator:
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    state = torch.load(CHECKPOINT_PATH, map_location=device)

    # Pull training config saved in checkpoint, with sane fallbacks
    cfg = state.get("args", {})
    image_size = cfg.get("image_size", 256)
    channel_multiplier = cfg.get("channel_multiplier", 1.0)
    z_dim = cfg.get("z_dim", 512)

    generator = StyleGAN2Generator(
        z_dim=z_dim,
        w_dim=z_dim,
        img_resolution=image_size,
        img_channels=3,
        channel_multiplier=channel_multiplier,
    ).to(device)

    if "generator_state_dict" in state:
        generator.load_state_dict(state["generator_state_dict"])
    elif "generator" in state:
        generator.load_state_dict(state["generator"])
    else:
        generator.load_state_dict(state)

    generator.eval()
    return generator, image_size, channel_multiplier, z_dim


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator, image_size, channel_multiplier, z_dim = load_generator(device)

    evaluator = AfroCoverEvaluator(generator, device=device.type)
    fid_score = evaluator.calculate_fid(
        real_images_path=REAL_IMAGES_PATH,
        num_generated=DEFAULT_NUM_GENERATED,
        batch_size=DEFAULT_BATCH_SIZE,
    )

    results = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "device": device.type,
        "afrocover_fid": fid_score,
        "num_generated": DEFAULT_NUM_GENERATED,
        "batch_size": DEFAULT_BATCH_SIZE,
        "checkpoint": str(CHECKPOINT_PATH),
        "real_images_path": str(REAL_IMAGES_PATH),
        "image_size": image_size,
        "channel_multiplier": channel_multiplier,
        "z_dim": z_dim,
    }

    print(json.dumps(results, indent=2))
    with open(OUTPUT_JSON, "w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2)


if __name__ == "__main__":
    main()
