"""Use the author's unmodified supervised PatchTST for daily SSE forecasts."""
import os
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace
import numpy as np
from common import ROOT, BASE, OUT, HISTORY, BATCH, EPOCHS, SEED, PROTOCOL, get_jobs, arrays, audit_fields, dump, read, frame

VENDOR = BASE / 'third_party' / 'PatchTST'
sys.path.insert(0, str(VENDOR / 'PatchTST_supervised'))
CONFIG = dict(enc_in=1, seq_len=HISTORY, pred_len=1, e_layers=2, n_heads=4,
              d_model=32, d_ff=64, dropout=0.1, fc_dropout=0.0, head_dropout=0.0,
              individual=False, patch_len=5, stride=3, padding_patch=None,
              revin=True, affine=True, subtract_last=False, decomposition=False, kernel_size=25)
MODEL_CONFIG = {'config': CONFIG, 'normalization': 'LayerNorm', 'patch_count': 6,
                'implementation': 'Official supervised Model; unchanged source; no pretrained weights'}
CONFIG_DIGEST = hashlib.sha256(json.dumps(MODEL_CONFIG, sort_keys=True).encode()).hexdigest()
_READY = False


def setup():
    global _READY
    import torch
    if not _READY:
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        _READY = True
    manifest = read(ROOT / 'installation.json')
    for rel, expected in manifest['files'].items():
        assert hashlib.sha256((VENDOR / rel).read_bytes()).hexdigest() == expected


def new_model():
    import torch
    from models.PatchTST import Model
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    model = Model(SimpleNamespace(**CONFIG), norm='LayerNorm')
    initial = hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
    return model, initial


def fit_day(job):
    import torch
    setup()
    started = time.monotonic()
    a = arrays(job)
    v = a['validation']; cut = a['validation_train_count']
    x = torch.from_numpy(v['x'][:cut]); y = torch.from_numpy(v['y'][:cut])
    xv = torch.from_numpy(v['x'][cut:]); yv = torch.from_numpy(v['y'][cut:])
    model, initial = new_model()
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    assert not optimizer.state
    rng = np.random.default_rng(SEED)
    history = []
    validation_updates = 0

    def epoch(net, opt, xs, ys, generator):
        net.train()
        order = generator.permutation(len(xs))
        updates = 0
        for start in range(0, len(xs), BATCH):
            ix = order[start:start+BATCH]
            opt.zero_grad(set_to_none=True)
            result = net(xs[ix])[:, 0, :]
            loss = torch.nn.functional.l1_loss(result, ys[ix])
            assert torch.isfinite(loss)
            loss.backward()
            torch.nn.utils.clip_grad_value_(net.parameters(), 1.0)
            opt.step()
            updates += 1
        return updates

    for number in range(1, EPOCHS+1):
        validation_updates += epoch(model, optimizer, x, y, rng)
        model.eval()
        with torch.no_grad():
            mae = float(torch.mean(torch.abs(model(xv)[:, 0, :] - yv))) * v['std']
        assert np.isfinite(mae)
        history.append({'epoch': number, 'mae_points': mae})
    selected = min(history, key=lambda r: r['mae_points'])['epoch']
    # Discard validation model, optimizer and scaler; refit on all known rows.
    model, refit_initial = new_model()
    assert initial == refit_initial
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    assert not optimizer.state
    f = a['final']; xf = torch.from_numpy(f['x']); yf = torch.from_numpy(f['y'])
    rng = np.random.default_rng(SEED)
    final_updates = 0
    for _ in range(selected):
        final_updates += epoch(model, optimizer, xf, yf, rng)
    assert final_updates == selected * int(np.ceil(len(xf)/BATCH))
    model.eval()
    with torch.no_grad():
        prediction = float(model(torch.from_numpy(f['next_x']))[0, 0, 0]) * f['std'] + f['mean']
    assert np.isfinite(prediction) and prediction > 0
    model_file = OUT / 'models' / 'patchtst' / f'{job["date"]}.pt'
    model_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_file)
    result = audit_fields(job, a)
    result.update(predicted_close=prediction, selected_epoch=selected, validation_history=history,
                  initial_weights_hash=initial, refit_initial_weights_hash=refit_initial,
                  optimizer_initial_iterations=0, validation_optimizer_updates=validation_updates,
                  final_optimizer_updates=final_updates, parameter_count=params, model_file=model_file.relative_to(OUT).as_posix(),
                  model_config=MODEL_CONFIG, config_digest=CONFIG_DIGEST, torch_version=str(torch.__version__),
                  elapsed_seconds=time.monotonic()-started)
    dump(OUT / 'per_day' / 'patchtst' / f'{job["date"]}.json', result)
    return result


def reload_checks(require_complete=True):
    import torch
    import pandas as pd
    setup()
    f = frame()
    dates = pd.to_datetime([p.stem for p in sorted((OUT / 'per_day' / 'patchtst').glob('*.json'))])
    assert len(dates) > 0
    if require_complete:
        assert len(dates) == 64
    checks = []
    for day in dates[sorted({0, len(dates)//2, len(dates)-1})]:
        r = read(OUT / 'per_day' / 'patchtst' / f'{day.date()}.json')
        historical = f.loc[(f.index >= day-pd.DateOffset(years=2)) & (f.index < day)].close.to_numpy()
        mean = historical.mean(); std = historical.std()
        np.testing.assert_allclose([mean, std], [r['scalers']['final']['mean'], r['scalers']['final']['std']])
        x = ((historical[-20:]-mean)/std).astype('float32')[None, :, None]
        model, _ = new_model()
        model.load_state_dict(torch.load(OUT / r['model_file'], weights_only=True, map_location='cpu'))
        model.eval()
        with torch.no_grad():
            value = float(model(torch.from_numpy(x))[0, 0, 0])*std+mean
        error = abs(value-r['predicted_close'])
        assert error < 0.001
        checks.append({'date': str(day.date()), 'absolute_error_points': error, 'passed': True})
    dump(OUT / 'reload_patchtst.json', checks)


def smoke():
    import torch
    setup(); m, initial = new_model()
    test_input = torch.arange(40, dtype=torch.float32).reshape(2, 20, 1) / 40
    m.train(); out = m(test_input)
    assert tuple(out.shape) == (2, 1, 1) and torch.isfinite(out).all()
    out.sum().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
    m.eval()
    with torch.no_grad():
        first = m(test_input).clone()
        altered = test_input.clone(); altered[1] += 100
        second = m(altered)
    torch.testing.assert_close(first[0], second[0], rtol=1e-5, atol=1e-5)
    count = sum(p.numel() for p in m.parameters() if p.requires_grad)
    dump(OUT / 'installation_smoke.json', {'passed': True, 'torch': str(torch.__version__),
         'output_shape': list(out.shape), 'backward_finite': True, 'batch_independence': True,
         'parameter_count': count, 'initial_hash': initial, 'model_config': MODEL_CONFIG})
    print(json.dumps({'smoke_passed': True, 'parameters': count, 'torch': str(torch.__version__)}), flush=True)


def main(workers, limit=None):
    jobs = get_jobs('patchtst', limit)
    for path in (OUT/'per_day'/'patchtst').glob('*.json'):
        assert read(path)['config_digest'] == CONFIG_DIGEST
    done = len(list((OUT / 'per_day' / 'patchtst').glob('*.json')))
    print(json.dumps({'model': 'patchtst', 'cached': done, 'to_fit': len(jobs), 'workers': workers}), flush=True)
    if jobs:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            futures = [pool.submit(fit_day, j) for j in jobs]
            for future in as_completed(futures):
                r = future.result(); done += 1
                print(json.dumps({'model': 'patchtst', 'completed': done, 'date': r['date'],
                      'selected_epoch': r['selected_epoch'], 'seconds': round(r['elapsed_seconds'], 1)}), flush=True)
    completed = len(list((OUT / 'per_day' / 'patchtst').glob('*.json')))
    reload_checks(require_complete=limit is None)
    print(json.dumps({'completed_dates': completed, 'complete': completed == 64, 'reload_passed': True}), flush=True)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--reload-only', action='store_true')
    parser.add_argument('--limit', type=int, help='Only first N dates; full training protocol is preserved')
    args = parser.parse_args()
    if args.smoke: smoke()
    elif args.reload_only: reload_checks()
    else: main(max(1, min(args.workers, 2)), args.limit)
