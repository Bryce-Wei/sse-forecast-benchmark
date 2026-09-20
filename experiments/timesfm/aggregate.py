"""Score every saved next-day prediction against the matching held-out actual."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from common import ROOT, OUT, BASE, RAW_HASH, PROTOCOL, DIGEST, HISTORY, EPOCHS, frame, read, dump, metric, LSTM_OUT, LSTM_DIGEST, load_lstm_records


def main(language='zh'):
    f = frame()
    review = f.loc['2026-06-15':]
    assert len(review) == 64 and HISTORY == 32
    records = {}
    for name in ['timesfm', 'lstm']:
        if name == 'lstm':
            rows = load_lstm_records()
        else:
            files = sorted((OUT / 'per_day' / name).glob('*.json'))
            rows = [read(path) for path in files]
        assert [r['date'] for r in rows] == review.index.strftime('%Y-%m-%d').tolist()
        expected_digest = LSTM_DIGEST if name == 'lstm' else DIGEST
        assert all(r['protocol_digest'] == expected_digest and 'actual_close' not in r for r in rows)
        records[name] = rows
    joined = []
    for i, day in enumerate(review.index):
        t, l = records['timesfm'][i], records['lstm'][i]
        past = f.loc[f.index < day].tail(HISTORY)
        assert t['input_dates'] == l['prediction_input_dates'] == past.index.strftime('%Y-%m-%d').tolist()
        np.testing.assert_array_equal(t['input_values'], past.close.to_numpy())
        assert t['previous_close'] == l['previous_close'] == float(past.close.iloc[-1])
        assert l['model_config']['input_shape'] == [32, 1]
        assert t['requested_horizon'] == 1 and t['frozen'] and not t['local_training']
        joined.append({'date': str(day.date()), 'actual_close': float(f.loc[day, 'close']),
                       'previous_close': t['previous_close'], 'timesfm_close': t['predicted_close'],
                       'lstm_close': l['predicted_close'], 'lstm_epoch': l['selected_epoch'],
                       'input_start': t['input_dates'][0], 'input_end': t['input_dates'][-1],
                       'train_start': l['train_start'], 'train_end': l['train_end'],
                       'train_observations': l['train_observations'], 'train_windows': l['train_windows']})
    df = pd.DataFrame(joined)
    cols = {'timesfm': 'timesfm_close', 'lstm': 'lstm_close', 'baseline': 'previous_close'}
    overall = {name: metric(df.actual_close, df[col]) for name, col in cols.items()}
    monthly = [{'month': month, 'count': len(part), **{name: metric(part.actual_close, part[col]) for name, col in cols.items()}}
               for month, part in df.groupby(df.date.str[:7], sort=True)]
    directions = {name: float(np.mean(np.sign(df[col] - df.previous_close) == np.sign(df.actual_close - df.previous_close)))
                  for name, col in cols.items() if name != 'baseline'}
    actual_returns = np.log(df.actual_close / df.previous_close)
    return_errors = {name: metric(actual_returns, np.log(df[col] / df.previous_close)) for name, col in cols.items()}
    runtime = read(OUT / 'timesfm_runtime.json')
    assert runtime['complete_count'] == 64, 'Run TimesFM for the full review before aggregation'
    assert runtime['all_parameters_frozen'] and runtime['state_hash_before'] == runtime['state_hash_after']
    provenance = read(ROOT / 'checkpoint_provenance.json')
    epochs = [r['selected_epoch'] for r in records['lstm']]
    reloads = {'timesfm': read(OUT / 'reload_timesfm.json'),
               'lstm': read(LSTM_OUT / 'reload_lstm.json')}
    assert all(len(v) == 3 and all(r['passed'] for r in v) for v in reloads.values())
    data = {'symbol': '000001.SS', 'review_start': joined[0]['date'], 'review_end': joined[-1]['date'],
            'review_count': len(joined), 'initial_train_start': joined[0]['train_start'],
            'initial_train_end': joined[0]['train_end'], 'last_train_start': joined[-1]['train_start'],
            'last_train_end': joined[-1]['train_end'], 'input_history': HISTORY}
    source = {'model_id': provenance['model_id'], 'model_revision': provenance['model_revision'],
              'code_revision': provenance['code_revision'], 'code_distribution': 'timesfm==2.0.1',
              'reference_payload_sha256': RAW_HASH, 'data_fingerprint': PROTOCOL['data_fingerprint'], 'checkpoint_sha256': runtime['checkpoint_sha256'],
              'model_last_modified': provenance['last_modified']}
    metrics = {'data': data, 'overall': overall, 'monthly': monthly, 'direction_accuracy': directions,
               'actual_up_days': int((df.actual_close > df.previous_close).sum()),
               'return_errors': return_errors, 'source': source, 'protocol': PROTOCOL, 'shared_lstm': {'output': LSTM_OUT.relative_to(BASE).as_posix(), 'protocol_digest': LSTM_DIGEST},
               'reload_checks': reloads,
               'models': {'timesfm': runtime,
                          'lstm': {'parameters': records['lstm'][0]['parameter_count'],
                                   'config': records['lstm'][0]['model_config'],
                                   'epochs_min': min(epochs), 'epochs_max': max(epochs),
                                   'epochs_median': float(np.median(epochs)),
                                   'epochs_at_cap': sum(e == EPOCHS for e in epochs),
                                   'train_observations_min': int(df.train_observations.min()),
                                   'train_observations_max': int(df.train_observations.max())}}}
    dump(OUT / 'predictions.json', joined)
    dump(OUT / 'metrics.json', metrics)
    display = f.loc[f.index >= pd.Timestamp(data['review_start']) - pd.DateOffset(years=2)]
    dump(OUT / 'observations.json', [{'date': str(d.date()), 'close': float(row.close)} for d, row in display.iterrows()])
    dump(OUT / 'checkpoint_provenance.json', provenance)
    print(json.dumps({'complete': True, 'count': len(joined), 'overall': overall, 'direction_accuracy': directions,
                      'lstm_epochs_at_cap': metrics['models']['lstm']['epochs_at_cap']}, indent=2))
    from render_comparison import main as render
    render(language=language)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--language', choices=['zh', 'en'], default='zh')
    main(**vars(parser.parse_args()))
