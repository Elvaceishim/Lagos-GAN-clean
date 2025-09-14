"""Wrapper to run a target Python script and tee its stdout/stderr to a log file.
Usage: python3 scripts/run_and_log.py <target_script> <log_path>
"""
import sys, runpy, os
from datetime import datetime

class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, data):
        for f in self.files:
            try:
                f.write(data)
                f.flush()
            except Exception:
                pass
    def flush(self):
        for f in self.files:
            try:
                f.flush()
            except Exception:
                pass


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else 'scripts/short_train_and_eval.py'
    logpath = sys.argv[2] if len(sys.argv) > 2 else 'test_results/run_and_log.txt'
    os.makedirs(os.path.dirname(logpath), exist_ok=True)
    with open(logpath, 'a') as lf:
        tee = Tee(sys.stdout, lf)
        sys.stdout = tee
        sys.stderr = tee
        print('\n--- RUN START ---', datetime.now().isoformat())
        try:
            runpy.run_path(target, run_name='__main__')
        except SystemExit as e:
            print('Script exited with SystemExit:', e)
        except Exception as e:
            import traceback
            print('Unhandled exception while running target:', e)
            traceback.print_exc()
        print('--- RUN END ---', datetime.now().isoformat())

if __name__ == '__main__':
    main()
