"""Past-only LSTM data protocol and validated access to shared daily records."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark_support import load_prices, data_fingerprint

BASE = Path(__file__).resolve().parents[2]
YEARS, VALID, EPOCHS, BATCH, SEED = 2, 40, 40, 32, 88
REVIEW_START, REVIEW_END = '2026-06-15', '2026-09-11'


def checked_history(history):
    if history not in (20, 32):
        raise ValueError('LSTM history must be 20 or 32 observations')
    return history


def output_dir(history):
    return BASE / 'runs' / 'lstm' / f'history_{checked_history(history)}'


def model_config(history):
    return {
        'architecture': 'LSTM(32, return_sequences=True) -> LSTM(16, activation=relu) -> Dense(1)',
        'input_shape': [checked_history(history), 1], 'features': ['close'],
        'optimizer': 'Adam', 'learning_rate': 0.001, 'clipvalue': 1.0,
        'loss': 'MAE', 'batch_size': BATCH, 'max_selection_epochs': EPOCHS,
        'validation_targets': VALID, 'seed': SEED, 'revin': False,
        'selection_rule': 'Minimum MAE on final40 historical target dates; earliest epoch wins ties',
        'refit_rule': 'Fresh seed88 model and optimizer, all past2y samples, selected number of full epochs',
    }


def make_protocol(history):
    return {
        'schema': 'shared-lstm-v1', 'symbol': '000001.SS',
        'history': checked_history(history), 'train_years': YEARS, 'horizon': 1,
        'features': ['close'], 'target': 'next trading observation close',
        'review_start': REVIEW_START, 'review_end': REVIEW_END, 'review_count': 64,
        'window': '[target date minus 2 calendar years, target date)',
        'scaler': 'Historical mean/population std; validation scaler excludes validation observations; refit scaler uses all historical rows',
        'sampling': 'Consecutive observations, stride1; all input/target rows inside the historical window',
        'model_config': model_config(history), 'data_fingerprint': data_fingerprint(),
        'historical_note': 'Previously examined dates: retrospective review, not an untouched final test.',
    }


def digest(protocol):
    return hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def frame():
    f = load_prices().loc[:REVIEW_END]
    assert len(f.loc[REVIEW_START:]) == 64
    return f


def make_job(target, f, protocol):
    lower = target - pd.DateOffset(years=YEARS)
    historical = f.loc[(f.index >= lower) & (f.index < target)]
    assert f.index.min() < lower
    return {'date': str(target.date()), 'lower': str(lower.date()),
            'dates': [str(d.date()) for d in historical.index],
            'values': historical.close.to_list(), 'model': 'lstm',
            'history': protocol['history'], 'protocol': protocol,
            'protocol_digest': digest(protocol)}


def arrays(job):
    history = checked_history(job['history'])
    assert job['protocol']['history'] == history
    assert job['protocol_digest'] == digest(job['protocol'])
    dates = pd.to_datetime(job['dates'])
    target = pd.Timestamp(job['date'])
    lower = target - pd.DateOffset(years=YEARS)
    assert str(lower.date()) == job['lower']
    assert dates.min() >= lower and dates.max() < target and dates.is_unique
    assert dates.is_monotonic_increasing
    values = np.asarray(job['values'], dtype=float)
    assert np.isfinite(values).all() and (values > 0).all()
    n = len(values)
    cutoff = n - VALID
    assert cutoff > history and len(dates) == n
    result = {'values': values, 'dates': dates, 'cutoff': cutoff,
              'validation_train_count': cutoff - history}
    for stage, fit_end in [('validation', cutoff), ('final', n)]:
        mu = float(values[:fit_end].mean())
        sd = float(values[:fit_end].std(ddof=0))
        assert sd > 0
        z = ((values - mu) / sd).astype('float32')
        result[stage] = {
            'x': np.stack([z[i-history:i, None] for i in range(history, n)]),
            'y': z[history:, None], 'mean': mu, 'std': sd,
            'next_x': z[-history:][None, :, None], 'scaler_fit_rows': fit_end,
        }
    return result


def audit_fields(job, a):
    n = len(job['dates']); cut = a['cutoff']; history = job['history']
    return {'date': job['date'], 'model': 'lstm', 'history': history,
            'protocol_digest': job['protocol_digest'],
            'train_lower_bound': job['lower'], 'train_start': job['dates'][0],
            'train_end': job['dates'][-1], 'train_observations': n,
            'train_windows': n-history, 'training_dates': job['dates'],
            'training_target_dates': job['dates'][history:],
            'prediction_input_dates': job['dates'][-history:],
            'validation_train_end': job['dates'][cut-1],
            'validation_target_dates': job['dates'][cut:],
            'validation_train_windows': cut-history,
            'previous_close': float(a['values'][-1]),
            'scalers': {stage: {key: a[stage][key] for key in ['mean', 'std', 'scaler_fit_rows']}
                        for stage in ['validation', 'final']}}


def validate_record(row, job, require_model=True):
    """Reject cache rows from another lookback, data snapshot, or training rule."""
    history = job['history']
    assert row['model'] == 'lstm' and row['history'] == history
    assert row['protocol_digest'] == job['protocol_digest'], 'LSTM cache protocol mismatch'
    assert row['model_config'] == model_config(history), 'LSTM cache model configuration mismatch'
    assert row['date'] == job['date'] and 'actual_close' not in row
    expected = audit_fields(job, arrays(job))
    for key, value in expected.items():
        assert row[key] == value, f'LSTM cache field mismatch: {key}'
    validation = row['validation_history']
    assert [r['epoch'] for r in validation] == list(range(1, EPOCHS+1))
    assert all(np.isfinite(r['mae_points']) for r in validation)
    assert row['selected_epoch'] == min(validation, key=lambda r: (r['mae_points'], r['epoch']))['epoch']
    assert row['initial_weights_hash'] == row['refit_initial_weights_hash']
    n = len(job['dates'])
    assert row['selection_optimizer_initial_iterations'] == row['refit_optimizer_initial_iterations'] == 0
    assert row['selection_optimizer_final_iterations'] == EPOCHS * math.ceil((n-history-VALID)/BATCH)
    assert row['refit_optimizer_final_iterations'] == row['selected_epoch'] * math.ceil((n-history)/BATCH)
    assert np.isfinite(row['predicted_close']) and row['predicted_close'] > 0
    out = output_dir(history).resolve()
    model_path = (out / row['model_file']).resolve()
    assert model_path.is_relative_to(out), 'Model path must remain inside this history output'
    if require_model:
        assert model_path.is_file(), 'Missing shared LSTM saved model'


def get_jobs(history, limit=None):
    checked_history(history)
    if limit is not None and limit < 1:
        raise ValueError('--limit must be positive')
    f = frame(); protocol = make_protocol(history); out = output_dir(history)
    saved_protocol = out / 'protocol.json'
    if saved_protocol.exists():
        assert read(saved_protocol) == protocol, 'Existing LSTM output has a different data/configuration protocol'
    else:
        dump(saved_protocol, protocol)
    targets = f.loc[REVIEW_START:].index
    if limit is not None:
        targets = targets[:limit]
    jobs = []
    for target in targets:
        job = make_job(target, f, protocol)
        saved = out / 'per_day' / 'lstm' / f"{job['date']}.json"
        if saved.exists():
            validate_record(read(saved), job)
        else:
            jobs.append(job)
    return jobs


def load_records(history, require_complete=True, expected_fingerprint=None, dates=None):
    """Read the shared cache, validate every selected record, and return date order.

    Comparison callers supply their input fingerprint. `dates` optionally selects
    specific dates for limited validation; it is never allowed with a full-run
    assertion. This function neither imports TensorFlow nor trains a model.
    """
    protocol = make_protocol(history); out = output_dir(history); f = frame()
    assert read(out / 'protocol.json') == protocol, 'LSTM output protocol mismatch'
    if expected_fingerprint is not None:
        assert protocol['data_fingerprint'] == expected_fingerprint, 'Comparison data snapshot mismatch'
    files = sorted((out / 'per_day' / 'lstm').glob('*.json'))
    all_dates = f.loc[REVIEW_START:].index.strftime('%Y-%m-%d').tolist()
    if dates is not None:
        if require_complete:
            raise ValueError('dates selection is only supported for a partial validation')
        requested = sorted(set(dates))
        assert all(date in all_dates for date in requested)
        files = [out / 'per_day' / 'lstm' / f'{date}.json' for date in requested]
    assert files, 'No saved shared LSTM records'
    rows = [read(path) for path in files]
    observed_dates = [row['date'] for row in rows]
    assert observed_dates == sorted(set(observed_dates))
    if require_complete:
        assert observed_dates == all_dates, 'Shared LSTM review is not complete'
    else:
        assert all(date in all_dates for date in observed_dates)
    for row in rows:
        validate_record(row, make_job(pd.Timestamp(row['date']), f, protocol))
    assert len({row['initial_weights_hash'] for row in rows}) == 1
    return rows


def metric(actual, predicted):
    difference = np.asarray(actual, dtype=float) - np.asarray(predicted, dtype=float)
    assert np.isfinite(difference).all()
    return {'rmse': float(np.sqrt(np.mean(difference*difference))),
            'mae': float(np.mean(np.abs(difference)))}
