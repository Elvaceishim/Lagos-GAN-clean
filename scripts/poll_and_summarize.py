"""Poll a training log file every 60s and summarize training progress once it contains output.
Writes a short summary to test_results/train_resume_summary_YYYY-MM-DD.txt and prints to stdout.
"""
import time
import re
from datetime import datetime
from pathlib import Path

LOG_PATH = Path('test_results/train_resume_epochs_to_7_2025-09-11.txt')
SUMMARY_PATH = Path(f'test_results/train_resume_summary_{datetime.now().date()}.txt')
POLL_INTERVAL = 60  # seconds

print('Polling for log at:', LOG_PATH)
print('Will write summary to:', SUMMARY_PATH)

while True:
    if LOG_PATH.exists() and LOG_PATH.stat().st_size > 0:
        print('Log file detected and non-empty. Reading contents...')
        text = LOG_PATH.read_text(errors='ignore')
        lines = text.splitlines()
        # Find epoch completions and losses
        epochs = []  # list of dicts: {'epoch':int, 'num_epochs':int, 'gen':float, 'd_a':float, 'd_b':float}
        checkpoints = []
        last_epoch_info = {}
        for i, line in enumerate(lines):
            # Epoch completion
            m = re.search(r"Epoch\s+(\d+)/(\d+).*completed", line)
            if m:
                epoch = int(m.group(1))
                total = int(m.group(2))
                info = {'epoch': epoch, 'total': total}
                # Look ahead for the three loss lines
                for j in range(i+1, min(i+8, len(lines))):
                    l = lines[j].strip()
                    mg = re.match(r"Generator Loss:\s*([0-9.]+)", l)
                    if mg:
                        info['gen'] = float(mg.group(1))
                        continue
                    md = re.match(r"Discriminator A Loss:\s*([0-9.]+)", l)
                    if md:
                        info['d_a'] = float(md.group(1))
                        continue
                    md2 = re.match(r"Discriminator B Loss:\s*([0-9.]+)", l)
                    if md2:
                        info['d_b'] = float(md2.group(1))
                        continue
                    mc = re.match(r"Validation Loss:\s*([0-9.]+)", l)
                    if mc:
                        info['val'] = float(mc.group(1))
                epochs.append(info)
                last_epoch_info = info
            # Checkpoint saved
            if 'Checkpoint saved' in line or 'Checkpoint saved:' in line or 'Checkpoint saved:' in line:
                checkpoints.append(line.strip())
            # alternate message printed in save_checkpoint
            mck = re.search(r"Checkpoint saved: (.+)$", line)
            if mck:
                checkpoints.append(mck.group(1).strip())
        # Also catch lines like "Checkpoint saved: <path>"
        # Build summary text
        summary_lines = []
        summary_lines.append(f'Poll summary run at: {datetime.now().isoformat()}')
        if epochs:
            summary_lines.append(f'Epochs completed (count): {len(epochs)}')
            summary_lines.append(f'Last reported epoch: {last_epoch_info.get("epoch")} / {last_epoch_info.get("total")}')
            if 'gen' in last_epoch_info:
                summary_lines.append(f'  Generator Loss (last): {last_epoch_info.get("gen"):.4f}')
            if 'd_a' in last_epoch_info:
                summary_lines.append(f'  Discriminator A Loss (last): {last_epoch_info.get("d_a"):.4f}')
            if 'd_b' in last_epoch_info:
                summary_lines.append(f'  Discriminator B Loss (last): {last_epoch_info.get("d_b"):.4f}')
            if 'val' in last_epoch_info:
                summary_lines.append(f'  Validation Loss (last): {last_epoch_info.get("val"):.4f}')
        else:
            summary_lines.append('No full epoch completion lines parsed yet; partial logs found.')
            # Try to find training progress lines (tqdm) or loss prints
            # Scan last 200 lines for occurrences of 'G:' or 'loss_G' or similar
            tail = lines[-200:]
            gen_vals = []
            for l in tail:
                m = re.search(r"loss_G\W*:\W*([0-9.]+)", l)
                if m:
                    try:
                        gen_vals.append(float(m.group(1)))
                    except:
                        pass
            if gen_vals:
                summary_lines.append(f'  Recent generator loss samples: {gen_vals[-3:]}')
        if checkpoints:
            summary_lines.append('Checkpoint events:')
            for c in checkpoints[-5:]:
                summary_lines.append('  ' + c)
        else:
            summary_lines.append('No checkpoint save messages found yet.')

        # Write summary file
        SUMMARY_PATH.write_text('\n'.join(summary_lines) + '\n')
        print('\n'.join(summary_lines))
        print('Wrote summary to', SUMMARY_PATH)
        break
    else:
        print(f'Log not ready yet (will check again in {POLL_INTERVAL}s)...')
        time.sleep(POLL_INTERVAL)

print('Poller exiting.')
