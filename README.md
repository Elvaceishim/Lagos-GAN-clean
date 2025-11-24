---
title: Lagos Gan Demo
sdk: gradio
sdk_version: 5.46.0
app_file: app.py
short_description: Experimental ML project showcasing how GANs work
---

# LagosGAN

LagosGAN is an experimental ML project showcasing how GANs can power African-centered synthetic creativity. It combines two GAN applications in one system:

- **AfroCover** (StyleGAN2): generates African-inspired music/album cover art
- **Lagos2Duplex** (CycleGAN): translates old Lagos house photos into modern duplex mockups

## Objectives

- Build two GAN pipelines (StyleGAN2-ADA and CycleGAN/CUT)
- Provide a live demo (Gradio) for public use
- Deliver a working demo and an article narrating my experience and what I've learnt

## Project Structure

```
lagosgan/
├── afrocover/         # StyleGAN2 pipeline for album covers
├── lagos2duplex/      # CycleGAN pipeline for house transformation
├── demo/              # Gradio demo app
├── eval/              # FID/KID/LPIPS evaluation scripts
├── scripts/           # preprocessing and augmentation
├── docs/              # model & dataset cards
├── data/              # datasets (gitignored)
├── checkpoints/       # model weights (gitignored)
└── requirements.txt   # dependencies
```

## Success Metrics

- FID < 60 for AfroCover at 256²
- User study: ≥70% participants prefer generated duplex vs baseline
- Article: 2k+ words with charts, metrics, and visuals

## Updated AfroCover Workflow (CPU-friendly)

Recent experiments introduced a lighter StyleGAN configuration to make CPU-only iterations feasible.

- Trim channels with `--channel_multiplier` and lower resolution via `--image_size`:
  ```bash
  python afrocover/train.py \
      --data_path data_processed/afrocover \
      --output_dir checkpoints/afrocover \
      --image_size 128 \
      --channel_multiplier 0.5 \
      --batch_size 2 \
      --num_epochs 20
  ```
- Copy the latest checkpoint for evaluation:
  ```bash
  cp checkpoints/afrocover/latest_checkpoint.pt checkpoints/afrocover/latest.pt
  ```
- Run the updated evaluator (auto-detects saved hyperparameters):
  ```bash
  PYTHONPATH=$(pwd) python scripts/run_afrocover_eval.py
  ```

Training on CPU still yields an AfroCover FID of ~464 (200 generated samples). Achieving the original <60 target will require extended GPU sessions; see `docs/model_cards.md` for current numbers and next steps.

## Links

- [Live Demo](https://huggingface.co/spaces/theelvace/lagos-gan-demo)
- [Article](https://medium.com/@theelvace/my-introduction-to-generative-adversarial-networks-gans-90f63ebe88f0)
- [Model Cards](docs/model_cards.md)
- [Dataset Cards](docs/dataset_cards.md)

## License

This project is for educational and research purposes. All datasets used are properly licensed and attributed.
