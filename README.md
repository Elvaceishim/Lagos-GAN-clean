# LagosGAN

LagosGAN is an experimental ML project showcasing how GANs can power African-centered synthetic creativity. It combines two GAN applications in one system:

- **AfroCover** (StyleGAN2): generates African-inspired music/album cover art
- **Lagos2Duplex** (CycleGAN): translates old Lagos house photos into modern duplex mockups

## Objectives

- Build two GAN pipelines (StyleGAN2-ADA and CycleGAN/CUT)
- Provide a live demo (Gradio/Streamlit) for public use
- Deliver working demo, code repo, and thought-leadership article within one week

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

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare datasets:**
   ```bash
   python scripts/prepare_afrocover_data.py
   python scripts/prepare_lagos_data.py
   ```

3. **Train models:**
   ```bash
   # AfroCover (StyleGAN2)
   python afrocover/train.py

   # Lagos2Duplex (CycleGAN)
   python lagos2duplex/train.py
   ```

4. **Run demo:**
   ```bash
   PYTHONPATH=$(pwd) python demo/app.py
   ```
   The app exposes both GAN experiences through a Gradio UI. Keep `share=True` inside `demo/app.py` if you need the temporary public link.

### Note on Gradio 5.x

Gradio 5 introduced a regression in its API schema helper that crashes when boolean `additionalProperties` entries appear in a component schema. Until the upstream fix ships, the demo applies a small monkey patch near the top of `demo/app.py` that coerces those boolean nodes into safe defaults. If you change Gradio versions or run in a Hugging Face Space, leave that shim in place (or remove it once a patched release lands).

## Success Metrics

- At least 12 high-quality samples per task
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

- [Live Demo](#) (Coming soon)
- [Article](#) (Coming soon)
- [Model Cards](docs/model_cards.md)
- [Dataset Cards](docs/dataset_cards.md)

## License

This project is for educational and research purposes. All datasets used are properly licensed and attributed.
