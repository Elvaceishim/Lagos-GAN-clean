import sys
import os
import argparse
import glob
import subprocess
import csv
from datetime import datetime

# ...existing code...

def run_checkpoint_eval(python_exec, ckpt_path, out_dir, num_samples, metrics, seed, log_dir):
    name = os.path.splitext(os.path.basename(ckpt_path))[0]
    run_out = os.path.join(out_dir, name)
    os.makedirs(run_out, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{name}.txt")

    cmd = [
        python_exec,
        os.path.join("scripts", "generate_and_eval.py"),
        "--checkpoint", ckpt_path,
        "--out_dir", run_out,
        "--num_samples", str(num_samples),
        "--seed", str(seed),
        "--metric", " ".join(metrics)
    ]
    # flatten flags if metrics are multiple (generate_and_eval.py should accept repeated --metric or space-separated)
    # ensure the command is robust: use shell=False
    with open(log_file, "wb") as lf:
        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
    return log_file, name

def parse_metrics_from_log(log_file):
    kid = "NA"
    lpips = "NA"
    with open(log_file, "r", errors="ignore") as f:
        for line in f:
            l = line.strip()
            if not l:
                continue
            # adjust patterns to match generate_and_eval.py outputs
            if "KID" in l.upper():
                try:
                    kid = l.split()[-1]
                except Exception:
                    pass
            if "LPIPS" in l.upper():
                try:
                    lpips = l.split()[-1]
                except Exception:
                    pass
    return kid, lpips

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints_dir", default="checkpoints/lagos2duplex", help="Directory with .pt checkpoints")
    p.add_argument("--out_root", default="results/lagos2duplex/gen_vs_val", help="Where to store generated images per-checkpoint")
    p.add_argument("--log_dir", default="test_results/eval_runs", help="Where to store per-check logs")
    p.add_argument("--num_samples", type=int, default=200, help="Number of generated samples per checkpoint")
    p.add_argument("--metrics", nargs="+", default=["kid","lpips"], help="Metrics to compute (kid lpips ...)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--csv_out", default="test_results/eval_summary.csv")
    args = p.parse_args()

    os.makedirs(args.out_root, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.csv_out) or ".", exist_ok=True)

    python_exec = sys.executable  # use same python that runs this script

    ckpt_paths = sorted(glob.glob(os.path.join(args.checkpoints_dir, "*.pt")))
    if not ckpt_paths:
        print("No checkpoints found in", args.checkpoints_dir)
        sys.exit(1)

    rows = []
    for ck in ckpt_paths:
        print(f"[{datetime.now().isoformat()}] Evaluating {ck} ...")
        log_file, name = run_checkpoint_eval(python_exec, ck, args.out_root, args.num_samples, args.metrics, args.seed, args.log_dir)
        kid, lpips = parse_metrics_from_log(log_file)
        rows.append((name, ck, kid, lpips, log_file))

    # write CSV
    with open(args.csv_out, "w", newline="") as csvf:
        writer = csv.writer(csvf)
        writer.writerow(["checkpoint_name","checkpoint_path","kid","lpips","log_file"])
        for r in rows:
            writer.writerow(r)

    print("Done. Summary written to", args.csv_out)
    print("Per-checkpoint logs:", args.log_dir)
    print("Generated outputs:", args.out_root)

if __name__ == "__main__":
    main()