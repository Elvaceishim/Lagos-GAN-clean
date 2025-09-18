# Checkpoint Status (September 2025)

## AfroCover
- Latest checkpoint: `checkpoints/afrocover/latest_checkpoint.pt`
- Resolution / channel multiplier: 128×128, `channel_multiplier=0.5`
- Last training run: CPU-only, 20 epochs total (across multiple sessions)
- Latest evaluation: `scripts/run_afrocover_eval.py` with 200 generated samples → FID ≈ 464.3
- Next steps: resume training on GPU to reach FID < 60 or fine-tune at 256²

## Lagos2Duplex
- Latest checkpoint: `checkpoints/lagos2duplex/latest.pt`
- Status: Stable; evaluation scripts unchanged in this cycle

## Notes
- Copy `latest_checkpoint.pt` to `latest.pt` before running the evaluator.
- Archive large raw datasets externally before running long trainings to keep the repo light.
