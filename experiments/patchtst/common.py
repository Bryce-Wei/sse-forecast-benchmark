"""Shared, past-only protocol for the SSE PatchTST/LSTM comparison."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

import sys
ROOT = Path(__file__).resolve().parent
BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE))
from benchmark_support import load_prices, data_fingerprint
OUT = BASE / 'runs' / 'patchtst'
RAW = BASE / 'data' / 'yahoo_sse.json'
RAW_HASH = 'aa52cac6e0320a14f944f976007b886aece26390aff6db388f3e5de7e7ed19dc'
HISTORY, YEARS, VALID, EPOCHS, BATCH, SEED = 20, 2, 40, 40, 32, 88
PROTOCOL = {
    'symbol': '000001.SS', 'train_years': YEARS, 'history': HISTORY, 'horizon': 1,
    'features': ['close'], 'target': 'next trading observation close',
    'review_start': '2026-06-15', 'review_end': '2026-09-11', 'review_count': 64,
    'max_epochs': EPOCHS, 'validation_targets': VALID, 'batch_size': BATCH, 'seed': SEED,
    'loss': 'MAE', 'optimizer': 'Adam', 'learning_rate': 0.001, 'gradient_clip_value': 1.0,
    'selection': 'Every target day: final40 historical target dates select epoch, then fresh full-window refit',
    'scaler': 'Historical mean/population std; validation fit excludes validation observations; final fit uses all past2y',
    'window': '[target date minus 2 calendar years, target date); all sample inputs and labels inside window',
    'sampling': '20 consecutive observations, stride1; full shuffled epochs, no repetition to arbitrary update count',
    'initialization': 'New model and optimizer for both validation fitting and final fitting each day, seed88',
    'reference_payload_sha256': RAW_HASH,
    'data_fingerprint': data_fingerprint(),
    'comparison_note': 'Matched information and training-budget rules; PatchTST includes its official RevIN, LSTM follows classroom layers. Model-configuration comparison, not isolated attention effect.',
    'historical_note': 'These64 dates were already examined in earlier experiments; retrospective review, not untouched final test.'
}
DIGEST = hashlib.sha256(json.dumps(PROTOCOL, sort_keys=True).encode()).hexdigest()


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def frame():
    f = load_prices().loc[:'2026-09-11']
    assert len(f.loc['2026-06-15':]) == 64
    return f


def get_jobs(model, limit=None):
    f = frame()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'protocol.json'
    if path.exists():
        assert read(path) == PROTOCOL
    else:
        dump(path, PROTOCOL)
    jobs = []
    targets = f.loc['2026-06-15':].index
    if limit is not None:
        if limit < 1:
            raise ValueError('--limit must be positive')
        targets = targets[:limit]
    for target in targets:
        lower = target - pd.DateOffset(years=YEARS)
        history = f.loc[(f.index >= lower) & (f.index < target)]
        assert f.index.min() < lower
        key = str(target.date())
        job = {'date': key, 'lower': str(lower.date()), 'dates': [str(x.date()) for x in history.index],
               'values': history.close.to_list(), 'protocol_digest': DIGEST, 'model': model}
        saved = OUT / 'per_day' / model / f'{key}.json'
        if saved.exists():
            r = read(saved)
            assert r['protocol_digest'] == DIGEST and r['training_dates'] == job['dates']
            assert (OUT / r['model_file']).is_file()
        else:
            jobs.append(job)
    return jobs


def arrays(job):
    dates = pd.to_datetime(job['dates'])
    target = pd.Timestamp(job['date'])
    lower = target - pd.DateOffset(years=YEARS)
    assert job['protocol_digest'] == DIGEST and str(lower.date()) == job['lower']
    assert dates.min() >= lower and dates.max() < target and dates.is_unique
    values = np.asarray(job['values'], dtype=float)
    n = len(values)
    cutoff = n - VALID
    assert cutoff > HISTORY and len(dates) == n
    out = {'values': values, 'dates': dates, 'cutoff': cutoff}
    for name, fit_end in [('validation', cutoff), ('final', n)]:
        mu = float(values[:fit_end].mean())
        sd = float(values[:fit_end].std(ddof=0))
        assert sd > 0
        z = ((values - mu) / sd).astype('float32')
        x = np.stack([z[i-HISTORY:i, None] for i in range(HISTORY, n)])
        y = z[HISTORY:, None]
        out[name] = {'x': x, 'y': y, 'mean': mu, 'std': sd,
                     'next_x': z[-HISTORY:][None, :, None], 'scaler_fit_rows': fit_end}
    out['validation_train_count'] = cutoff - HISTORY
    return out


def audit_fields(job, a):
    n = len(job['dates']); cut = a['cutoff']
    return {'date': job['date'], 'model': job['model'], 'protocol_digest': DIGEST,
            'train_lower_bound': job['lower'], 'train_start': job['dates'][0], 'train_end': job['dates'][-1],
            'train_observations': n, 'train_windows': n-HISTORY, 'training_dates': job['dates'],
            'training_target_dates': job['dates'][HISTORY:], 'prediction_input_dates': job['dates'][-HISTORY:],
            'validation_train_end': job['dates'][cut-1], 'validation_target_dates': job['dates'][cut:],
            'validation_train_windows': cut-HISTORY, 'previous_close': float(a['values'][-1]),
            'scalers': {stage: {k: a[stage][k] for k in ['mean', 'std', 'scaler_fit_rows']}
                       for stage in ['validation', 'final']}}


def metric(actual, pred):
    d = np.asarray(actual, dtype=float) - np.asarray(pred, dtype=float)
    assert np.isfinite(d).all()
    return {'rmse': float(np.sqrt(np.mean(d*d))), 'mae': float(np.mean(np.abs(d)))}
