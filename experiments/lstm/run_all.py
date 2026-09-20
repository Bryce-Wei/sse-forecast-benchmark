"""Independent LSTM rolling experiment; shared by both paired comparisons."""
import argparse
import multiprocessing
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE))
from experiments.lstm import train
from experiments.lstm.protocol import checked_history


def main(history, limit=None, workers=2, reload_only=False, smoke=False, language='zh'):
    checked_history(history)
    if limit is not None and limit < 1:
        raise ValueError('--limit must be positive')
    if smoke:
        train.smoke(history)
        return
    train.main(history, workers=max(1, min(2, workers)), reload_only=reload_only, limit=limit)
    if limit is None:
        from experiments.lstm.aggregate import main as aggregate
        aggregate(history, language=language)
    else:
        print('Limited LSTM validation finished; overall metrics require all 64 dates.')


def cli(fixed_history=None):
    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history', type=int, choices=[fixed_history] if fixed_history else [20, 32],
                        required=fixed_history is None, default=fixed_history)
    parser.add_argument('--limit', type=int, help='Only first N dates; preserve full training for each date')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--reload-only', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--language', choices=['zh', 'en'], default='zh')
    args = parser.parse_args()
    main(**vars(args))


if __name__ == '__main__':
    cli()
