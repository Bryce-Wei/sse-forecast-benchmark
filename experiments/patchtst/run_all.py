"""Run this paired comparison with the independently reusable LSTM20."""
import argparse
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parent
SHARED_LSTM = ROOT.parent / 'lstm' / 'run_all.py'


def main(limit=None, language='zh'):
    if limit is not None and limit < 1:
        raise ValueError('--limit must be positive')
    stages = [(ROOT / 'run_patchtst.py', ['--smoke']), (ROOT / 'run_patchtst.py', [])]
    for script, args in stages:
        if limit is not None and script.name.startswith('run_') and '--smoke' not in args:
            args = args + ['--limit', str(limit)]
        subprocess.run([sys.executable, str(script), *args], check=True)
    shared_args = ['--history', '20', '--language', language]
    if limit is not None:
        shared_args += ['--limit', str(limit)]
    subprocess.run([sys.executable, str(SHARED_LSTM), *shared_args], check=True)
    if limit is None:
        subprocess.run([sys.executable, str(ROOT / 'aggregate.py'), '--language', language], check=True)
    else:
        print('Limited validation finished; aggregation requires all 64 dates.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit', type=int, help='First N dates only; retain full training per date')
    parser.add_argument('--language', choices=['zh', 'en'], default='zh')
    main(**vars(parser.parse_args()))
