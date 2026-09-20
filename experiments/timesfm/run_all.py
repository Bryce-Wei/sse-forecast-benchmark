"""Run or resume the complete 64-date comparison using this Python interpreter."""
import argparse
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parent


def main(limit=None):
    stages = [('download_model.py', []), ('run_timesfm.py', []), ('run_lstm.py', [])]
    for script, args in stages:
        if limit is not None and script.startswith('run_') and '--smoke' not in args:
            args = args + ['--limit', str(limit)]
        subprocess.run([sys.executable, str(ROOT / script), *args], check=True)
    if limit is None:
        subprocess.run([sys.executable, str(ROOT / 'aggregate.py')], check=True)
    else:
        print('Limited validation finished; aggregation requires all 64 dates.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit', type=int, help='First N dates only; retains full training per date')
    main(parser.parse_args().limit)
