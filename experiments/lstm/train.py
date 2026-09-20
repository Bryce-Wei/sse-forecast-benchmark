"""One shared training implementation for independent and paired LSTM runs."""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'

from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import math
import multiprocessing
import sys
import time
import numpy as np
import pandas as pd
from experiments.lstm.protocol import (
    VALID, EPOCHS, BATCH, SEED, get_jobs, arrays, audit_fields, dump,
    frame, model_config, make_protocol, make_job, output_dir, load_records,
)

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
_TF_READY = False



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


def fresh_model(tf, history):
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEED)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(history, 1)),
        tf.keras.layers.LSTM(32, return_sequences=True),
        tf.keras.layers.LSTM(16, activation='relu'),
        tf.keras.layers.Dense(1),
    ], name=f'sse_lstm_two_year_{history}_day')
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
    history = job['history']
    out = output_dir(history)
    assert job['protocol'] == make_protocol(history)
    a = arrays(job)
    v, f = a['validation'], a['final']
    split = a['validation_train_count']
    train_x, train_y = v['x'][:split], v['y'][:split]
    val_x, val_y = v['x'][split:], v['y'][split:]
    assert len(val_x) == VALID and train_x.shape[1:] == (history, 1)
    assert job['dates'][a['cutoff'] - 1] < job['dates'][a['cutoff']]
    model, initial_hash = fresh_model(tf, history)
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
    model, refit_hash = fresh_model(tf, history)
    assert refit_hash == initial_hash and model.count_params() == count
    model.fit(dataset(tf, f['x'], f['y']), epochs=selected_epoch, shuffle=False, verbose=0)
    refit_iterations = int(model.optimizer.iterations.numpy())
    assert refit_iterations == selected_epoch * math.ceil(len(f['x']) / BATCH)
    prediction = float(model(f['next_x'], training=False).numpy()[0, 0]) * f['std'] + f['mean']
    assert np.isfinite(prediction) and prediction > 0
    model_path = out / 'models' / 'lstm' / f'{job["date"]}.keras'
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    result = audit_fields(job, a)
    result.update({
        'predicted_close': prediction, 'selected_epoch': selected_epoch,
        'validation_history': val_history, 'initial_weights_hash': initial_hash,
        'refit_initial_weights_hash': refit_hash, 'model_file': model_path.relative_to(out).as_posix(),
        'parameter_count': count, 'elapsed_seconds': time.monotonic() - started,
        'model_config': model_config(history), 'tensorflow_version': tf.__version__,
        'selection_optimizer_initial_iterations': 0,
        'selection_optimizer_final_iterations': selection_iterations,
        'refit_optimizer_initial_iterations': 0,
        'refit_optimizer_final_iterations': refit_iterations,
    })
    # Only the historical job is visible here; no evaluation actual is joined.
    dump(out / 'per_day' / 'lstm' / f'{job["date"]}.json', result)
    return result


def verify_reload(history, require_complete=True):
    tf = tensorflow()
    f = frame(); out = output_dir(history); protocol = make_protocol(history)
    records = load_records(history, require_complete=require_complete)
    checks = []
    for i in sorted({0, len(records) // 2, len(records) - 1}):
        r = records[i]
        job = make_job(pd.Timestamp(r['date']), f, protocol)
        a = arrays(job)
        model = tf.keras.models.load_model(out / r['model_file'], compile=False)
        reproduced = float(model(a['final']['next_x'], training=False).numpy()[0, 0]) * a['final']['std'] + a['final']['mean']
        difference = abs(reproduced - r['predicted_close'])
        assert difference < 0.001
        checks.append({'date': r['date'], 'history': history,
                       'absolute_difference_points': difference, 'passed': True})
        del model
        tf.keras.backend.clear_session()
    dump(out / 'reload_lstm.json', checks)
    return checks


def main(history, workers=2, reload_only=False, limit=None):
    import json
    started = time.monotonic()
    out = output_dir(history)
    if not reload_only:
        jobs = get_jobs(history, limit)
        complete = len(list((out / 'per_day' / 'lstm').glob('*.json')))
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
    checks = verify_reload(history, require_complete=limit is None)
    completed = len(list((out / 'per_day' / 'lstm').glob('*.json')))
    print(json.dumps({'model': 'lstm', 'complete': completed == 64, 'completed_dates': completed, 'reload_checks': checks,
                      'elapsed_seconds': round(time.monotonic() - started, 1)}), flush=True)


def smoke(history):
    tf = tensorflow()
    model, _ = fresh_model(tf, history)
    x = np.arange(2 * history, dtype='float32').reshape(2, history, 1) / (2 * history)
    assert model(x, training=False).shape == (2, 1)
    assert np.isfinite(model.train_on_batch(x, np.zeros((2, 1), dtype='float32')))
    print({'smoke_passed': True, 'history': history, 'parameters': model.count_params()})
