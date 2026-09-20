"""Draw standalone LSTM history/test and test-detail panels, without fitting."""
import argparse
import json
from pathlib import Path
import sys

import pandas as pd

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE))
from experiments.lstm.protocol import output_dir, read
from benchmark_support import load_prices, data_fingerprint
from forecast_plotting import plot_forecast, GREEN, GRAY


def main(history, language='zh', reference=False):
    source = BASE/'results/reference/lstm'/f'history_{history}' if reference else output_dir(history)
    out = BASE/'runs/lstm_reference'/f'history_{history}' if reference else source
    rows = read(source/'predictions.json')
    metrics = read(source/'metrics.json')
    if len(rows) != 64 or metrics['protocol']['history'] != history:
        raise ValueError('Expected a completed matching 20/32-day LSTM run.')
    if reference:
        expected = read(BASE/'results/reference/manifest.json')['canonical_prices_sha256']
        if data_fingerprint() != expected:
            raise ValueError('Local historical prices differ from the archived experiment.')
        observed = load_prices().close
    else:
        observed = pd.DataFrame(read(source/'observations.json'))
        observed['date'] = pd.to_datetime(observed['date'])
        observed = observed.set_index('date').close
    first = rows[0]
    known = observed.loc[first['train_start']:first['train_end']]
    if len(known) != first['train_observations']:
        raise ValueError('Initial historical window differs from the saved LSTM run.')
    zh = language == 'zh'
    audit = plot_forecast(
        known, rows, [('lstm_close', f'LSTM（{history}日）' if zh else f'LSTM ({history} days)', GREEN, '-'),
                      ('previous_close', '前日基准' if zh else 'Previous close', GRAY, '--')],
        out/'comparison.png', '首次训练历史' if zh else 'Initial training history', language)
    (out/'plot_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    print(out/'comparison.png')
    return audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history', type=int, choices=[20,32], required=True)
    parser.add_argument('--language', choices=['zh','en'], default='zh')
    parser.add_argument('--reference', action='store_true', help='Draw the archived LSTM series without training')
    main(**vars(parser.parse_args()))
