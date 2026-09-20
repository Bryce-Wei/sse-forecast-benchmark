"""Shared, past-only protocol for the SSE TimesFM/LSTM comparison."""
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
from experiments.lstm.protocol import output_dir as lstm_output_dir, make_protocol as lstm_protocol, digest as lstm_digest, load_records as read_shared_lstm
OUT = BASE / 'runs' / 'timesfm'
RAW = BASE / 'data' / 'yahoo_sse.json'
RAW_HASH = 'aa52cac6e0320a14f944f976007b886aece26390aff6db388f3e5de7e7ed19dc'
HISTORY, YEARS, VALID, EPOCHS, BATCH, SEED = 32, 2, 40, 40, 32, 88
PROTOCOL = {
    'symbol': '000001.SS', 'train_years': YEARS, 'history': HISTORY, 'horizon': 1,
    'features': ['close'], 'target': 'next trading observation close',
    'review_start': '2026-06-15', 'review_end': '2026-09-11', 'review_count': 64,
    'max_epochs': EPOCHS, 'validation_targets': VALID, 'batch_size': BATCH, 'seed': SEED,
    'loss': 'MAE', 'optimizer': 'Adam', 'learning_rate': 0.001, 'gradient_clip_value': 1.0,
    'selection': 'LSTM only: every target day final40 historical target dates select epoch, then fresh full-window refit; TimesFM no fitting or test-driven parameter selection',
    'scaler': 'Historical mean/population std; validation fit excludes validation observations; final fit uses all past2y',
    'window': '[target date minus 2 calendar years, target date); all sample inputs and labels inside window',
    'sampling': '32 consecutive observations, stride1; full shuffled epochs, no repetition to arbitrary update count',
    'initialization': 'LSTM: new model and optimizer for both validation fitting and final fitting each day, seed88; TimesFM: frozen pretrained checkpoint',
    'reference_payload_sha256': RAW_HASH,
    'data_fingerprint': data_fingerprint(),
    'comparison_note': 'Same dates, Close target, and latest32 real observations. LSTM freshly retrained daily on preceding2 calendar years; TimesFM2.5 frozen external pretrained weights, zero-shot with no local fitting. Training information and compute are not matched.',
    'historical_note': 'These64 dates were already examined in earlier experiments; retrospective review, not untouched final test.'
}
DIGEST = hashlib.sha256(json.dumps(PROTOCOL, sort_keys=True).encode()).hexdigest()
LSTM_OUT = lstm_output_dir(HISTORY)
LSTM_PROTOCOL = lstm_protocol(HISTORY)
LSTM_DIGEST = lstm_digest(LSTM_PROTOCOL)


def load_lstm_records(require_complete=True, dates=None):
    return read_shared_lstm(HISTORY, require_complete=require_complete,
                            expected_fingerprint=PROTOCOL['data_fingerprint'], dates=dates)



def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def frame():
    f = load_prices().loc[:'2026-09-11']
    assert len(f.loc['2026-06-15':]) == 64
    return f


def metric(actual, pred):
    d = np.asarray(actual, dtype=float) - np.asarray(pred, dtype=float)
    assert np.isfinite(d).all()
    return {'rmse': float(np.sqrt(np.mean(d*d))), 'mae': float(np.mean(np.abs(d)))}
