"""Past-only, fresh daily classroom LSTM control for the TimesFM experiment."""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import math
import multiprocessing
from pathlib import Path
import sys
import time
import numpy as np
from common import OUT, HISTORY, VALID, EPOCHS, BATCH, SEED, DIGEST, get_jobs, arrays, audit_fields, dump, read, frame

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
_TF_READY = False
MODEL_CONFIG = {
    'architecture': 'LSTM(32, return_sequences=True) -> LSTM(16, activation=relu) -> Dense(1)',
    'input_shape': [HISTORY, 1], 'features': ['close'],
    'optimizer': 'Adam', 'learning_rate': 0.001, 'clipvalue': 1.0,
    'loss': 'MAE', 'batch_size': BATCH, 'max_selection_epochs': EPOCHS,
    'validation_targets': VALID, 'seed': SEED, 'revin': False,
    'selection_rule': 'Minimum MAE on final40 historical target dates; earliest epoch wins ties',
    'refit_rule': 'Fresh seed88 model and optimizer, all past2y samples, selected number of full epochs',
}


def tensorflow():
    global _TF_READY
    import tensorflow as tf
    if not _TF_READY:
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(2)
        tf.config.experimental.enable_op_determinism()
        tf.get_logger().setLevel('ERROR')
        _TF_READY = True
    return tf


def fresh_model(tf):
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEED)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(HISTORY, 1)),
        tf.keras.layers.LSTM(32, return_sequences=True),
        tf.keras.layers.LSTM(16, activation='relu'),
        tf.keras.layers.Dense(1),
    ], name='sse_lstm_two_year_32_day')
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipvalue=1.0), loss='mae')
    assert int(model.optimizer.iterations.numpy()) == 0
    init_hash = hashlib.sha256(b''.join(w.tobytes() for w in model.get_weights())).hexdigest()
    return model, init_hash


def dataset(tf, x, y):
    options = tf.data.Options()
    options.threading.private_threadpool_size = 1
    options.threading.max_intra_op_parallelism = 1
    return (tf.data.Dataset.from_tensor_slices((x, y))
            .shuffle(len(x), seed=SEED, reshuffle_each_iteration=True)
            .batch(BATCH, drop_remainder=False).with_options(options))


def fit_day(job):
    started = time.monotonic()
    tf = tensorflow()
    a = arrays(job)
    v, f = a['validation'], a['final']
    split = a['validation_train_count']
    train_x, train_y = v['x'][:split], v['y'][:split]
    val_x, val_y = v['x'][split:], v['y'][split:]
    assert len(val_x) == VALID and train_x.shape[1:] == (HISTORY, 1)
    assert job['dates'][a['cutoff'] - 1] < job['dates'][a['cutoff']]
    model, initial_hash = fresh_model(tf)
    count = int(model.count_params())
    val_history = []

    class ValidationLog(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            predicted = self.model(val_x, training=False).numpy()
            error = float(np.mean(np.abs(predicted - val_y))) * v['std']
            assert np.isfinite(error)
            val_history.append({'epoch': epoch + 1, 'mae_points': error})

    model.fit(dataset(tf, train_x, train_y), epochs=EPOCHS, callbacks=[ValidationLog()], shuffle=False, verbose=0)
    selection_iterations = int(model.optimizer.iterations.numpy())
    assert selection_iterations == EPOCHS * math.ceil(len(train_x) / BATCH)
    assert len(val_history) == EPOCHS
    selected_epoch = min(val_history, key=lambda row: (row['mae_points'], row['epoch']))['epoch']
    del model
    model, refit_hash = fresh_model(tf)
    assert refit_hash == initial_hash and model.count_params() == count
    model.fit(dataset(tf, f['x'], f['y']), epochs=selected_epoch, shuffle=False, verbose=0)
    refit_iterations = int(model.optimizer.iterations.numpy())
    assert refit_iterations == selected_epoch * math.ceil(len(f['x']) / BATCH)
    prediction = float(model(f['next_x'], training=False).numpy()[0, 0]) * f['std'] + f['mean']
    assert np.isfinite(prediction) and prediction > 0
    model_path = OUT / 'models' / 'lstm' / f'{job["date"]}.keras'
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    result = audit_fields(job, a)
    result.update({
        'predicted_close': prediction, 'selected_epoch': selected_epoch,
        'validation_history': val_history, 'initial_weights_hash': initial_hash,
        'refit_initial_weights_hash': refit_hash, 'model_file': model_path.relative_to(OUT).as_posix(),
        'parameter_count': count, 'elapsed_seconds': time.monotonic() - started,
        'model_config': MODEL_CONFIG, 'tensorflow_version': tf.__version__,
        'selection_optimizer_initial_iterations': 0,
        'selection_optimizer_final_iterations': selection_iterations,
        'refit_optimizer_initial_iterations': 0,
        'refit_optimizer_final_iterations': refit_iterations,
    })
    # Only the historical job is visible here; no evaluation actual is joined.
    dump(OUT / 'per_day' / 'lstm' / f'{job["date"]}.json', result)
    return result


def verify_reload(require_complete=True):
    tf = tensorflow()
    f = frame()
    files = sorted((OUT / 'per_day' / 'lstm').glob('*.json'))
    assert files, 'No saved daily models'
    if require_complete:
        assert len(files) == 64
    records = [read(p) for p in files]
    assert len({r['initial_weights_hash'] for r in records}) == 1
    assert all(r['refit_initial_weights_hash'] == r['initial_weights_hash'] for r in records)
    assert all(r['model_config'] == MODEL_CONFIG and r['protocol_digest'] == DIGEST for r in records)
    checks = []
    for i in sorted({0, len(records) // 2, len(records) - 1}):
        r = records[i]
        historical = f.loc[(f.index >= r['train_lower_bound']) & (f.index < r['date'])]
        job = {'date': r['date'], 'lower': r['train_lower_bound'],
               'dates': [str(d.date()) for d in historical.index], 'values': historical.close.to_list(),
               'protocol_digest': DIGEST, 'model': 'lstm'}
        a = arrays(job)
        assert job['dates'] == r['training_dates']
        for stage in ['validation', 'final']:
            for k in ['mean', 'std', 'scaler_fit_rows']:
                assert a[stage][k] == r['scalers'][stage][k]
        model = tf.keras.models.load_model(OUT / r['model_file'], compile=False)
        reproduced = float(model(a['final']['next_x'], training=False).numpy()[0, 0]) * a['final']['std'] + a['final']['mean']
        difference = abs(reproduced - r['predicted_close'])
        assert difference < 0.001
        checks.append({'date': r['date'], 'absolute_difference_points': difference, 'passed': True})
        del model
        tf.keras.backend.clear_session()
    dump(OUT / 'reload_lstm.json', checks)
    return checks


def main(workers, reload_only=False, limit=None):
    import json
    started = time.monotonic()
    if not reload_only:
        jobs = get_jobs('lstm', limit)
        complete = len(list((OUT / 'per_day' / 'lstm').glob('*.json')))
        print(json.dumps({'model': 'lstm', 'cached': complete, 'to_fit': len(jobs), 'workers': workers}), flush=True)
        if jobs:
            with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
                futures = [pool.submit(fit_day, job) for job in jobs]
                for future in as_completed(futures):
                    row = future.result()
                    complete += 1
                    print(json.dumps({'model': 'lstm', 'complete': complete, 'total': 64, 'date': row['date'],
                                      'selected_epoch': row['selected_epoch'],
                                      'seconds': round(row['elapsed_seconds'], 1),
                                      'elapsed_seconds': round(time.monotonic() - started, 1)}), flush=True)
    checks = verify_reload(require_complete=limit is None)
    completed = len(list((OUT / 'per_day' / 'lstm').glob('*.json')))
    print(json.dumps({'model': 'lstm', 'complete': completed == 64, 'completed_dates': completed, 'reload_checks': checks,
                      'elapsed_seconds': round(time.monotonic() - started, 1)}), flush=True)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--reload-only', action='store_true')
    parser.add_argument('--limit', type=int, help='Only first N dates; keeps the full 40-epoch selection protocol')
    parser.add_argument('--smoke', action='store_true', help='Check model forward/backward without training the review')
    args = parser.parse_args()
    if args.smoke:
        tf = tensorflow()
        model, _ = fresh_model(tf)
        x = np.arange(2 * HISTORY, dtype='float32').reshape(2, HISTORY, 1) / (2 * HISTORY)
        assert model(x, training=False).shape == (2, 1)
        assert np.isfinite(model.train_on_batch(x, np.zeros((2, 1), dtype='float32')))
        print({'smoke_passed': True, 'history': HISTORY, 'parameters': model.count_params()})
    else:
        main(max(1, min(2, args.workers)), args.reload_only, args.limit)
