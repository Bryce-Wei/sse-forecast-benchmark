"""Project implementation inspired by the classroom ARIMA workflow.

Each test-date return is appended only AFTER that date's forecast is saved.
Forecasts are one trading observation ahead, never six months ahead at once.
"""
from pathlib import Path
import argparse
import json
import sys
import time
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels
from statsmodels.tsa.arima.model import ARIMA

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmark_support import load_prices, data_fingerprint

OUT = ROOT / 'runs' / 'arima_original'
TRAIN_START = pd.Timestamp('2023-03-14')
TEST_CUTOFF = pd.Timestamp('2026-03-14')
LAST_COMPLETED_DAY = pd.Timestamp('2026-09-11')
ORDER = (5, 1, 0)
SCALE = 100.0  # Fit in percentage-point units for numerical conditioning.


def metric(actual, predicted):
    delta=np.asarray(predicted)-np.asarray(actual)
    return {'mae':float(np.mean(np.abs(delta))), 'rmse':float(np.sqrt(np.mean(delta**2))), 'mse':float(np.mean(delta**2))}


def fit_one(history):
    attempts=[]
    # Predefined retry rule, not selected using realized test outcomes.
    for optimizer, iterations in [('lbfgs', 200), ('lbfgs', 1000), ('powell', 1000)]:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            fitted=ARIMA(np.asarray(history)*SCALE, order=ORDER).fit(method_kwargs={'method':optimizer, 'maxiter':iterations})
            forecast=float(np.asarray(fitted.forecast(steps=1))[0])/SCALE
        converged=bool(fitted.mle_retvals.get('converged',False))
        attempt={'optimizer':optimizer,'maxiter':iterations,'converged':converged,'warnings':list(dict.fromkeys(str(w.message) for w in caught))}
        attempts.append(attempt)
        if converged and np.isfinite(forecast):
            return forecast, attempts
    raise RuntimeError(f'ARIMA failed to converge; no substitute forecast was inserted: {attempts}')


def prepare_data():
    prices=load_prices()['close'].sort_index()
    assert not prices.index.duplicated().any(), 'Duplicate dates in provider response'
    prices=prices.loc[:LAST_COMPLETED_DAY]
    assert np.isfinite(prices.to_numpy()).all() and (prices>0).all()
    returns=np.log(prices/prices.shift(1)).dropna().loc[TRAIN_START:]
    train=returns.loc[returns.index<TEST_CUTOFF]
    test=returns.loc[(returns.index>=TEST_CUTOFF)&(returns.index<=LAST_COMPLETED_DAY)]
    assert len(train)>=500 and len(test)==125, 'The fixed protocol requires 125 review dates'
    assert train.index.max()<test.index.min()
    assert test.index.max()==LAST_COMPLETED_DAY, 'Latest complete day is missing'
    return prices, returns, train, test, data_fingerprint()


def plot_outputs(prices, returns, train, test, result, report, language='en'):
    from plotting import render
    records = result.reset_index().to_dict('records')
    for row in records:
        row['date'] = pd.Timestamp(row['date']).strftime('%Y-%m-%d')
    render(prices, train, records, OUT/'sse_arima_index_points.png',
           OUT/'sse_arima_returns.png', 'ARIMA(5,1,0)', language)


def main(smoke=False, language='en'):
    OUT.mkdir(parents=True, exist_ok=True)
    prices,returns,train,test,digest=prepare_data()
    if smoke:
        test=test.iloc[:2]
    history=list(train.to_numpy())
    last_observed_date=train.index[-1]
    rows=[]
    print(json.dumps({'training_start':str(train.index.min().date()),'training_end':str(train.index.max().date()),'training_observations':len(train),'test_start':str(test.index.min().date()),'test_end':str(test.index.max().date()),'test_observations':len(test)},indent=2),flush=True)
    begun=time.monotonic()
    for number,(date,actual_return) in enumerate(test.items(),1):
        assert last_observed_date<date
        assert len(history)==len(train)+number-1
        predicted,attempts=fit_one(history)
        previous_close=float(prices.loc[prices.index<date].iloc[-1])
        historic_mean=float(np.mean(history))
        # Save the prediction before appending this test date's observed return.
        row={'date':date,'trained_through':last_observed_date,'history_observations':len(history),'predicted_log_return':predicted,'history_mean_prediction':historic_mean,'previous_close':previous_close,'predicted_close':previous_close*float(np.exp(predicted)),'actual_log_return':float(actual_return),'actual_close':float(prices.loc[date]),'fit_attempts':attempts}
        rows.append(row)
        history.append(float(actual_return))
        last_observed_date=date
        if number==1 or number%10==0 or number==len(test):
            print(f'{number}/{len(test)} | {date:%Y-%m-%d} | elapsed={time.monotonic()-begun:.1f}s | attempts={len(attempts)}',flush=True)
    result=pd.DataFrame(rows).set_index('date')
    assert np.isfinite(result[['predicted_log_return','predicted_close','actual_close']].to_numpy()).all()
    assert (result['trained_through'].to_numpy()<result.index.to_numpy()).all()
    assert np.allclose(np.log(result['actual_close']/result['previous_close']),result['actual_log_return'])
    assert np.allclose(result['predicted_close'],result['previous_close']*np.exp(result['predicted_log_return']))
    report={
        'symbol':'000001.SS','name':'SSE Composite Index','source':'Yahoo Finance chart API','source_url':'https://finance.yahoo.com/quote/000001.SS/history/',
        'last_permitted_observation':'2026-09-11','data_sha256':digest,
        'fingerprint_method':'SHA-256 of canonical date/close rows, excluding mutable provider metadata',
        'smoke_check_only':smoke,
        'initial_training_start':str(train.index.min().date()),'initial_training_end':str(train.index.max().date()),'initial_training_observations':len(train),
        'test_start':str(test.index.min().date()),'test_end':str(test.index.max().date()),'test_observations':len(test),
        'order':list(ORDER),'fit_scale':SCALE,'method':'Expanding-window refit, one trading observation ahead; append actual after prediction.',
        'index_conversion':'previous actual close * exp(predicted log return), no mean bias correction',
        'log_return':{'arima':metric(result['actual_log_return'],result['predicted_log_return']),'zero_return':metric(result['actual_log_return'],np.zeros(len(result))),'historical_mean':metric(result['actual_log_return'],result['history_mean_prediction'])},
        'index_points':{'arima':metric(result['actual_close'],result['predicted_close']),'unchanged_price':metric(result['actual_close'],result['previous_close'])},
        'direction_accuracy':float(np.mean(np.sign(result['predicted_log_return'])==np.sign(result['actual_log_return']))),
        'test_total_index_change':float(result['actual_close'].iloc[-1]/result['previous_close'].iloc[0]-1),
        'fit_retry_dates':[str(r['date'].date()) for r in rows if len(r['fit_attempts'])>1],
        'nonconverged_final_fits':0,'total_runtime_seconds':time.monotonic()-begun,
        'versions':{'numpy':np.__version__,'pandas':pd.__version__,'statsmodels':statsmodels.__version__,'matplotlib':matplotlib.__version__},
    }
    (OUT/'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    result.reset_index().to_json(OUT/'predictions.json',orient='records',date_format='iso',indent=2)
    prices.rename_axis('date').reset_index().to_json(OUT/'historical_closes.json',orient='records',date_format='iso',indent=2)
    plot_outputs(prices,returns,train,test,result,report,language)
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--language', choices=['zh','en'], default='en')
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--smoke',action='store_true',help='Fit only the first two review dates; write separate smoke outputs.')
    mode.add_argument('--plot-only',action='store_true',help='Render existing full-run predictions without fitting.')
    args=parser.parse_args()
    if args.smoke:
        OUT=ROOT/'runs'/'arima_smoke'/'original'
    if args.plot_only:
        prices,returns,train,test,_=prepare_data()
        result=pd.read_json(OUT/'predictions.json',orient='records',convert_dates=['date','trained_through']).set_index('date')
        report=json.loads((OUT/'metrics.json').read_text(encoding='utf-8'))
        plot_outputs(prices,returns,train,test,result,report,args.language)
        print('Re-rendered both plots from saved forecasts; no refitting.')
    else:
        main(smoke=args.smoke, language=args.language)
