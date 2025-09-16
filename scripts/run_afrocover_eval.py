#!/usr/bin/env python3
"""Compute AfroCover StyleGAN metrics and store results."""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import torch

# Ensure repo root is on path so `afrocover` imports resolve
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from afrocover.models import StyleGAN2Generator
from eval.evaluate import AfroCoverEvaluator

CHECKPOINT_PATH = REPO_ROOT / "checkpoints" / "afrocover" / "latest.pt"
DATA_ROOT = REPO_ROOT / "data" / "afrocover"
PROCESSED_ROOT = REPO_ROOT / "data_processed" / "afrocover"

# Prefer processed images (uniform 256x256) if available; otherwise fall back to raw data.
if (PROCESSED_ROOT / "test").exists():
    REAL_IMAGES_PATH = PROCESSED_ROOT / "test"
else:
    REAL_IMAGES_PATH = DATA_ROOT / "test"
OUTPUT_JSON = REPO_ROOT / "eval_results.json"

Z_DIM = 512
IMAGE_RESOLUTION = 256
NUM_GENERATED = 10  # keep CPU runs manageable
BATCH_SIZE = 5


def load_generator(device: torch.device) -> StyleGAN2Generator:
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    generator = StyleGAN2Generator(
        z_dim=Z_DIM,
        w_dim=Z_DIM,
        img_resolution=IMAGE_RESOLUTION,
        img_channels=3,
    ).to(device)

    state = torch.load(CHECKPOINT_PATH, map_location=device)
    if "generator_state_dict" in state:
        generator.load_state_dict(state["generator_state_dict"])
    elif "generator" in state:
        generator.load_state_dict(state["generator"])
    else:
        generator.load_state_dict(state)

    generator.eval()
    return generator


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = "cuda" if torch.cuda.is_available() else "cpu"

    generator = load_generator(device)

    evaluator = AfroCoverEvaluator(generator, device=device_name)
    fid_score = evaluator.calculate_fid(
        real_images_path=REAL_IMAGES_PATH,
        num_generated=NUM_GENERATED,
        batch_size=BATCH_SIZE,
    )

    results = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "device": device_name,
        "afrocover_fid": fid_score,
        "num_generated": NUM_GENERATED,
        "batch_size": BATCH_SIZE,
        "checkpoint": str(CHECKPOINT_PATH),
        "real_images_path": str(REAL_IMAGES_PATH),
    }

    print(json.dumps(results, indent=2))

    with open(OUTPUT_JSON, "w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2)


if __name__ == "__main__":
    main()
