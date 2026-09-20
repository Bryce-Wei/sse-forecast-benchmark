"""Join actual closes only after both sets of saved forecasts are complete."""
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from common import ROOT, OUT, BASE, RAW_HASH, PROTOCOL, DIGEST, HISTORY, EPOCHS, BATCH, VALID, frame, read, dump, metric


def main():
    f = frame(); review = f.loc['2026-06-15':]
    records = {}
    for model in ['patchtst', 'lstm']:
        paths = sorted((OUT/'per_day'/model).glob('*.json'))
        assert len(paths) == len(review) == 64, f'{model} is not complete'
        rows = [read(p) for p in paths]
        assert [r['date'] for r in rows] == [str(d.date()) for d in review.index]
        assert len({r['initial_weights_hash'] for r in rows}) == 1
        assert all(r['initial_weights_hash'] == r['refit_initial_weights_hash'] for r in rows)
        for r in rows:
            target = pd.Timestamp(r['date']); lower = target-pd.DateOffset(years=2)
            historical = f.loc[(f.index >= lower) & (f.index < target)]
            dates = [str(d.date()) for d in historical.index]
            assert r['protocol_digest'] == DIGEST and r['training_dates'] == dates
            assert r['training_target_dates'] == dates[20:] and r['prediction_input_dates'] == dates[-20:]
            assert r['validation_target_dates'] == dates[-40:]
            assert r['validation_train_end'] == dates[-41]
            assert r['train_windows'] == len(dates)-20
            assert r['previous_close'] == float(historical.close.iloc[-1])
            for stage, n in [('validation', len(dates)-40), ('final', len(dates))]:
                values = historical.close.to_numpy()[:n]
                s = r['scalers'][stage]
                assert s['scaler_fit_rows'] == n
                np.testing.assert_allclose([s['mean'],s['std']], [values.mean(), values.std()], rtol=1e-12)
            assert [e['epoch'] for e in r['validation_history']] == list(range(1, EPOCHS+1))
            assert min(r['validation_history'], key=lambda e: e['mae_points'])['epoch'] == r['selected_epoch']
            expected_select = EPOCHS*math.ceil((len(dates)-HISTORY-VALID)/BATCH)
            expected_final = r['selected_epoch']*math.ceil((len(dates)-HISTORY)/BATCH)
            if model == 'patchtst':
                assert r['optimizer_initial_iterations'] == 0
                assert r['validation_optimizer_updates'] == expected_select and r['final_optimizer_updates'] == expected_final
            else:
                assert r['selection_optimizer_initial_iterations'] == r['refit_optimizer_initial_iterations'] == 0
                assert r['selection_optimizer_final_iterations'] == expected_select and r['refit_optimizer_final_iterations'] == expected_final
            assert (OUT / r['model_file']).is_file() and 'actual_close' not in r
        records[model] = rows
    joined = []
    for i, day in enumerate(review.index):
        a, b = records['patchtst'][i], records['lstm'][i]
        assert a['training_dates'] == b['training_dates']
        joined.append({'date': str(day.date()), 'actual_close': float(f.loc[day, 'close']),
                       'previous_close': a['previous_close'], 'patchtst_close': a['predicted_close'],
                       'lstm_close': b['predicted_close'], 'patchtst_epoch': a['selected_epoch'],
                       'lstm_epoch': b['selected_epoch'], 'train_start': a['train_start'],
                       'train_end': a['train_end'], 'train_observations': a['train_observations'],
                       'train_windows': a['train_windows']})
    df = pd.DataFrame(joined)
    names = {'patchtst':'patchtst_close','lstm':'lstm_close','baseline':'previous_close'}
    overall = {name:metric(df.actual_close, df[col]) for name,col in names.items()}
    monthly = []
    for month, part in df.groupby(df.date.str[:7],sort=True):
        monthly.append({'month':month,'count':len(part),**{name:metric(part.actual_close,part[col]) for name,col in names.items()}})
    installation = read(ROOT/'installation.json')
    smoke = read(OUT/'installation_smoke.json')
    checks = {name:read(OUT/f'reload_{name}.json') for name in ['patchtst','lstm']}
    assert all(len(rows)==3 and all(r['passed'] for r in rows) for rows in checks.values())
    model_details = {}
    for name, rows in records.items():
        epochs = [r['selected_epoch'] for r in rows]
        model_details[name] = {'parameters':rows[0]['parameter_count'],'epochs_min':min(epochs),
            'epochs_max':max(epochs),'epochs_median':float(np.median(epochs)), 'epochs_at_cap':sum(e==EPOCHS for e in epochs),
            'training_seconds_sum':float(sum(r['elapsed_seconds'] for r in rows)),
            'model_config':rows[0]['model_config'], 'initial_weights_hash':rows[0]['initial_weights_hash']}
    actual_return = np.log(df.actual_close/df.previous_close)
    returns = {name:metric(actual_return,np.log(df[col]/df.previous_close)) for name,col in names.items()}
    directions = {name:float(np.mean(np.sign(df[col]-df.previous_close)==np.sign(df.actual_close-df.previous_close)))
                  for name,col in names.items() if name!='baseline'}
    results = {'data':{'symbol':'000001.SS','review_start':joined[0]['date'],'review_end':joined[-1]['date'],
        'review_count':len(joined),'initial_train_start':joined[0]['train_start'],'initial_train_end':joined[0]['train_end'],
        'last_train_start':joined[-1]['train_start'],'last_train_end':joined[-1]['train_end']},
        'protocol':PROTOCOL,'overall':overall,'monthly':monthly,'return_errors':returns,'direction_accuracy':directions,
        'actual_up_days':int(np.sum(df.actual_close>df.previous_close)),
        'training_observations':{'min':int(df.train_observations.min()),'max':int(df.train_observations.max())},
        'training_windows':{'min':int(df.train_windows.min()),'max':int(df.train_windows.max())},
        'models':model_details,'source':{'repository':installation['repository'],'commit':installation['commit'],'reference_payload_sha256':RAW_HASH, 'data_fingerprint':PROTOCOL['data_fingerprint']},
        'reload_checks':checks,'installation_smoke':smoke,
        'runtime':{name:importlib.metadata.version(name) for name in ['numpy','pandas','matplotlib','torch','tensorflow']}}
    dump(OUT/'predictions.json',joined)
    dump(OUT/'metrics.json',results)
    dump(OUT/'installation.json',installation)
    display = f.loc[f.index >= pd.Timestamp(joined[0]['date'])-pd.DateOffset(years=2)]
    dump(OUT/'observations.json',[{'date':str(d.date()),'close':float(r.close)} for d,r in display.iterrows()])
    print(json.dumps({'complete':True,'overall':overall,'directions':directions,'models':{k:{x:v[x] for x in ['parameters','epochs_min','epochs_max','epochs_median']} for k,v in model_details.items()}},indent=2))
    from render_comparison import main as render
    render()


if __name__ == '__main__':
    main()
