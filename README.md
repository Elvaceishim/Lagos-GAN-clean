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
   python demo/app.py
   ```

## Success Metrics

- At least 12 high-quality samples per task
- FID < 60 for AfroCover at 256²
- User study: ≥70% participants prefer generated duplex vs baseline
- Article: 2k+ words with charts, metrics, and visuals

## Links

- [Live Demo](#) (Coming soon)
- [Article](#) (Coming soon)
- [Model Cards](docs/model_cards.md)
- [Dataset Cards](docs/dataset_cards.md)

## License

This project is for educational and research purposes. All datasets used are properly licensed and attributed.
