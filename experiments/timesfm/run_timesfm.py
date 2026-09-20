"""Fixed official TimesFM 2.5, 32 real closes -> next close, with no local fitting."""
from __future__ import annotations
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parent
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['HF_HOME'] = str(ROOT.parents[1] / '.cache' / 'huggingface')
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import time
import numpy as np
import torch
import timesfm
from common import BASE, OUT, DIGEST, PROTOCOL, HISTORY, SEED, frame, read, dump
from download_model import EXPECTED_SHA, EXPECTED_SIZE, REVISION, MODEL_ID, DEST, sha, verify

CONFIG = dict(max_context=32, max_horizon=128, per_core_batch_size=1,
              normalize_inputs=True, use_continuous_quantile_head=True,
              force_flip_invariance=True, infer_is_positive=True,
              fix_quantile_crossing=True, return_backcast=False)


def load():
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        str(DEST), local_files_only=True, torch_compile=False)
    model.model.requires_grad_(False)
    model.model.eval()
    model.compile(timesfm.ForecastConfig(**CONFIG))
    assert not model.model.training
    assert not any(p.requires_grad for p in model.model.parameters())
    assert dataclasses.asdict(model.forecast_config) == {**CONFIG, 'window_size': 0}
    return model


def model_hash(model):
    h = hashlib.sha256()
    for name, tensor in model.model.state_dict().items():
        h.update(name.encode())
        h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def main(limit=None, inputs_only=False):
    start = time.monotonic()
    assert HISTORY == 32
    f = frame()
    jobs = []
    targets = f.loc['2026-06-15':].index
    if limit is not None:
        if limit < 1:
            raise ValueError('--limit must be positive')
        targets = targets[:limit]
    for target in targets:
        past = f.loc[f.index < target].tail(HISTORY)
        assert len(past) == HISTORY and past.index[-1] < target
        jobs.append({'date': str(target.date()), 'input_dates': [str(d.date()) for d in past.index],
                     'input_values': past.close.to_list()})
    # The inference inputs contain no target-day actuals and no future observations.
    dump(OUT / 'timesfm_inputs.json', jobs)
    if (OUT / 'protocol.json').exists():
        assert read(OUT / 'protocol.json') == PROTOCOL
    else:
        dump(OUT / 'protocol.json', PROTOCOL)
    if inputs_only:
        print(json.dumps({'inputs_validated': len(jobs), 'history': HISTORY, 'inference_executed': False}))
        return
    assert importlib.metadata.version('timesfm') == '2.0.1'
    verify()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    model = load()
    before = model_hash(model)
    parameters = sum(p.numel() for p in model.model.parameters())
    print(json.dumps({'loaded': MODEL_ID, 'parameters': parameters, 'device': str(model.model.device), 'history': HISTORY}), flush=True)
    for i, job in enumerate(jobs):
        t0 = time.monotonic()
        values = np.asarray(job['input_values'], dtype=np.float32)
        with torch.inference_mode():
            point, quantiles = model.forecast(horizon=1, inputs=[values.copy()])
        assert point.shape == (1, 1) and quantiles.shape == (1, 1, 10)
        assert np.isfinite(quantiles).all() and np.isfinite(point).all() and point[0, 0] > 0
        assert point[0, 0] == quantiles[0, 0, 5]
        assert np.all(np.diff(quantiles[0, 0, 1:]) >= 0)
        row = {**job, 'predicted_close': float(point[0, 0]),
               'previous_close': job['input_values'][-1], 'protocol_digest': DIGEST,
               'model_id': MODEL_ID, 'model_revision': REVISION,
               'forecast_config': CONFIG, 'requested_horizon': 1,
               'input_float32_sha256': hashlib.sha256(values.tobytes()).hexdigest(),
               'point_definition': 'official point_forecast = quantiles[...,5], q0.5 median',
               'mean_channel': float(quantiles[0, 0, 0]),
               'quantile_levels': [.1, .2, .3, .4, .5, .6, .7, .8, .9],
               'quantile_values': [float(x) for x in quantiles[0, 0, 1:]],
               'local_training': False, 'frozen': True,
               'elapsed_seconds': time.monotonic() - t0}
        dump(OUT / 'per_day' / 'timesfm' / f"{job['date']}.json", row)
        if (i + 1) % 8 == 0:
            print(json.dumps({'model': 'timesfm', 'complete': i + 1, 'total': len(jobs), 'seconds': round(time.monotonic() - start, 1)}), flush=True)
    after = model_hash(model)
    assert before == after
    del model
    # Independent fresh load: verify first, middle, and last single-day predictions.
    model = load()
    checks = []
    for i in sorted({0, len(jobs) // 2, len(jobs) - 1}):
        job = jobs[i]
        row = read(OUT / 'per_day' / 'timesfm' / f"{job['date']}.json")
        with torch.inference_mode():
            reproduced, _ = model.forecast(horizon=1, inputs=[np.asarray(job['input_values'], dtype=np.float32)])
        difference = abs(float(reproduced[0, 0]) - row['predicted_close'])
        assert difference < .001
        checks.append({'date': job['date'], 'absolute_difference_points': difference, 'passed': True})
    dump(OUT / 'reload_timesfm.json', checks)
    sources = {}
    package = Path(timesfm.__file__).parent
    for path in package.rglob('*.py'):
        sources[str(path.relative_to(package)).replace('\\', '/')] = sha(path)
    runtime = {'model_id': MODEL_ID, 'model_revision': REVISION,
               'checkpoint_sha256': EXPECTED_SHA, 'parameters': parameters,
               'state_hash_before': before, 'state_hash_after': after,
               'all_parameters_frozen': not any(p.requires_grad for p in model.model.parameters()),
               'forecast_config': CONFIG, 'source_file_hashes': sources,
               'packages': {n: importlib.metadata.version(n) for n in ['timesfm', 'torch', 'numpy', 'huggingface_hub', 'safetensors']},
               'device': str(model.model.device), 'torch_compile': False,
               'local_training': False, 'complete_count': len(jobs),
               'elapsed_seconds': time.monotonic() - start}
    dump(OUT / 'timesfm_runtime.json', runtime)
    print(json.dumps({'complete': len(jobs) == 64, 'completed_dates': len(jobs), 'reload_checks': checks, 'seconds': runtime['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit', type=int, help='Infer only first N review dates')
    parser.add_argument('--smoke', action='store_true', help='Real one-date inference and fresh-load reproduction')
    parser.add_argument('--inputs-only', action='store_true', help='Validate past-only inputs without loading weights')
    args = parser.parse_args()
    main(1 if args.smoke else args.limit, args.inputs_only)
