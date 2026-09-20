"""Small, prespecified ARIMA comparison on frozen SSE data. No network use.

Selection: 2025-03-14 <= target date < 2026-03-14, pooled point RMSE.
Historical review: 2026-03-16 through 2026-09-11, already seen in prior run.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import argparse
import json
import math
import sys
import time
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, kpss, acf
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from benchmark_support import load_prices, data_fingerprint

BASE=ROOT/'runs'/'arima_original'
OUT=ROOT/'runs'/'arima_improved'
CACHE=OUT/'cache'
TRAIN_START='2023-03-14'
VALID_START='2025-03-14'
VALID_MID='2025-09-14'
REVIEW_START='2026-03-14'
REVIEW_MID='2026-06-14'
REVIEW_END='2026-09-11'
SPECS=[
    {'name':'Original ARIMA(5,1,0)','order':[5,1,0],'trend':'n'},
    {'name':'AR(5), no extra differencing','order':[5,0,0],'trend':'n'},
    {'name':'Historical mean','order':[0,0,0],'trend':'c'},
    {'name':'AR(1)','order':[1,0,0],'trend':'c'},
    {'name':'MA(1)','order':[0,0,1],'trend':'c'},
    {'name':'ARMA(1,1)','order':[1,0,1],'trend':'c'},
]
WINDOWS=[126,252,None]
CONFIGS=[dict(spec,window=window,id=f'{spec["order"][0]}{spec["order"][1]}{spec["order"][2]}_{spec["trend"]}_{window or "expanding"}') for spec in SPECS for window in WINDOWS]


def dump(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def load_data(require_original=True):
    raw_hash=data_fingerprint()
    if require_original:
        prior_path=BASE/'metrics.json'
        if not prior_path.exists():
            raise FileNotFoundError('Run python experiments/arima/run_forecast.py first, or use run_all.py.')
        prior=json.loads(prior_path.read_text(encoding='utf-8'))
        assert raw_hash==prior['data_sha256'], 'Input changed from original experiment'
        assert not prior.get('smoke_check_only'), 'Smoke results cannot serve as the full baseline'
    prices=load_prices()['close'].sort_index().loc[:REVIEW_END]
    assert not prices.index.duplicated().any()
    assert np.isfinite(prices).all() and (prices>0).all()
    all_returns=np.log(prices/prices.shift(1)).dropna().loc[TRAIN_START:]
    assert prices.index[-1]==pd.Timestamp(REVIEW_END), 'Final review day is missing'
    assert len(all_returns.loc[VALID_START:pd.Timestamp(REVIEW_START)-pd.Timedelta(days=1)])==242
    assert len(all_returns.loc[REVIEW_START:REVIEW_END])==125
    return prices,all_returns,raw_hash


def config_label(c):
    return f'{c["name"]} / {c["window"] or "expanding"}'


def fit_return(x,c):
    retries=0
    all_warnings=[]
    for solver,maxiter in [('lbfgs',200),('lbfgs',1000),('powell',1000)]:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            result=ARIMA(x*100,order=tuple(c['order']),trend=c['trend']).fit(method_kwargs={'method':solver,'maxiter':maxiter})
            prediction=float(np.asarray(result.forecast(1))[0])/100
        all_warnings.extend(str(w.message) for w in caught)
        if result.mle_retvals.get('converged',False) and np.isfinite(prediction):
            return prediction,retries,list(dict.fromkeys(all_warnings))
        retries+=1
    raise RuntimeError(f'No converged finite prediction: {c["id"]}')


def walk(c,stage):
    if stage not in ('validation','review'):
        raise ValueError(stage)
    prices,returns,digest=load_data()
    cache_path=CACHE/f'{stage}_{c["id"]}.json'
    if cache_path.exists():
        cached=json.loads(cache_path.read_text(encoding='utf-8'))
        if cached.get('data_sha256')==digest and cached.get('config')==c and cached.get('stage')==stage:
            return cached
    mask=((returns.index>=VALID_START)&(returns.index<REVIEW_START)) if stage=='validation' else ((returns.index>=REVIEW_START)&(returns.index<=REVIEW_END))
    target_positions=np.flatnonzero(mask)
    rows=[]
    start=time.monotonic()
    with threadpool_limits(limits=1):
        for pos in target_positions:
            date=returns.index[pos]
            lower=0 if c['window'] is None else max(0,pos-c['window'])
            history=returns.iloc[lower:pos]
            assert len(history)>=126 and history.index[-1]<date
            if c['window'] is not None: assert len(history)==c['window']
            try:
                predicted,retries,warning_texts=fit_return(history.to_numpy(),c)
            except Exception as exc:
                out={'data_sha256':digest,'id':c['id'],'config':c,'stage':stage,'eligible':False,'failure_date':str(date.date()),'failure':str(exc),'rows':[]}
                dump(cache_path,out)
                return out
            previous=float(prices.loc[prices.index<date].iloc[-1])
            rows.append({'date':str(date.date()),'trained_from':str(history.index[0].date()),'trained_through':str(history.index[-1].date()),'history_count':len(history),'predicted_log_return':predicted,'predicted_close':previous*math.exp(predicted),'previous_close':previous,'actual_log_return':float(returns.iloc[pos]),'actual_close':float(prices.loc[date]),'retries':retries,'warnings':warning_texts})
    out={'data_sha256':digest,'id':c['id'],'config':c,'stage':stage,'eligible':True,'rows':rows,'runtime_seconds':time.monotonic()-start}
    dump(cache_path,out)
    return out


def scores(rows):
    frame=pd.DataFrame(rows)
    point_errors=frame['predicted_close']-frame['actual_close']
    return_errors=frame['predicted_log_return']-frame['actual_log_return']
    active=frame['predicted_log_return']!=0
    direction_accuracy=float(np.mean(np.sign(frame.loc[active,'predicted_log_return'])==np.sign(frame.loc[active,'actual_log_return']))) if active.any() else None
    return {'n':len(frame),'point_rmse':float(np.sqrt(np.mean(point_errors**2))),'point_mae':float(np.mean(np.abs(point_errors))),'return_rmse':float(np.sqrt(np.mean(return_errors**2))),'return_mae':float(np.mean(np.abs(return_errors))),'direction_accuracy':direction_accuracy,'direction_coverage':float(active.mean())}


def subsets(rows,cut):
    return {'first_half':scores([r for r in rows if r['date']<cut]),'second_half':scores([r for r in rows if r['date']>=cut])}


def zero_rows(rows):
    return [dict(r,predicted_log_return=0.0,predicted_close=r['previous_close']) for r in rows]


def diagnose(prices,returns):
    diagnostics=[]
    for cutoff in [VALID_START,VALID_MID,REVIEW_START]:
        known_r=returns.loc[returns.index<cutoff]
        known_p=prices.loc[(prices.index>=TRAIN_START)&(prices.index<cutoff)]
        for name,series in [('log_index_level',np.log(known_p)),('log_returns',known_r),('differenced_log_returns',known_r.diff().dropna())]:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                a=adfuller(series,regression='c',autolag='AIC')
                k=kpss(series,regression='c',nlags='auto')
            diagnostics.append({'known_before':cutoff,'series':name,'n':len(series),'adf_stat':float(a[0]),'adf_p':float(a[1]),'adf_lags':int(a[2]),'kpss_stat':float(k[0]),'kpss_p_reported_bound':float(k[1]),'kpss_lags':int(k[2]),'acf_lag1':float(acf(series,nlags=1,fft=False)[1]),'warnings':[str(w.message) for w in caught]})
    return diagnostics


def block_comparison(model_rows,reference_rows,block_length):
    model=pd.DataFrame(model_rows).set_index('date').sort_index()
    ref=pd.DataFrame(reference_rows).set_index('date').sort_index()
    assert model.index.equals(ref.index)
    loss=(model['predicted_close']-model['actual_close']).to_numpy()**2
    ref_loss=(ref['predicted_close']-ref['actual_close']).to_numpy()**2
    n=len(loss)
    rng=np.random.default_rng(5545+block_length)
    starts=rng.integers(0,n,size=(5000,int(np.ceil(n/block_length))))
    indices=((starts[:,:,None]+np.arange(block_length))%n).reshape(5000,-1)[:,:n]
    gains=100*(1-np.sqrt(loss[indices].mean(axis=1)/ref_loss[indices].mean(axis=1)))
    return {'block_length':block_length,'replications':5000,'seed':5545+block_length,'rmse_reduction_percent':float(100*(1-np.sqrt(loss.mean()/ref_loss.mean()))),'ci95_percent':[float(v) for v in np.quantile(gains,[0.025,0.975])],'method':'Paired circular moving-block bootstrap; conditional on fixed forecasts, not corrected for model search.'}


def smoke():
    """Exercise all 18 configurations once, without model selection or headline metrics."""
    prices,returns,digest=load_data(require_original=False)
    target=returns.index[returns.index>=VALID_START][0]
    known=returns.loc[returns.index<target]
    observations=[]
    with threadpool_limits(limits=1):
        for config in CONFIGS:
            history=known if config['window'] is None else known.iloc[-config['window']:]
            predicted,retries,caught=fit_return(history.to_numpy(),config)
            assert history.index[-1]<target and np.isfinite(predicted)
            if config['window'] is not None:
                assert len(history)==config['window']
            observations.append({'id':config['id'],'date':str(target.date()),'history_count':len(history),'predicted_log_return':predicted,'retries':retries,'warnings':caught})
            print('Smoke fit passed: '+config['id'],flush=True)
    folder=ROOT/'runs'/'arima_smoke'
    folder.mkdir(parents=True,exist_ok=True)
    dump(folder/'comparison_smoke.json',{'smoke_check_only':True,'data_sha256':digest,'configurations_checked':len(observations),'rows':observations,'note':'One forecast per candidate only. No selection, accuracy comparison, or full reproduction claimed.'})


def main(workers=3):
    OUT.mkdir(parents=True,exist_ok=True)
    CACHE.mkdir(parents=True,exist_ok=True)
    prices,returns,digest=load_data()
    protocol={'data_sha256':digest,'training_start':TRAIN_START,'validation_start':VALID_START,'validation_mid':VALID_MID,'review_start_cutoff':REVIEW_START,'review_end':REVIEW_END,'primary_selection_metric':'pooled validation index-point RMSE; numerical tie by candidate id','configurations':CONFIGS,'baseline':'zero return / previous observed close','selection_rule':'Choose best eligible candidate on validation only; retain baseline as default if candidate does not beat it on validation. Freeze one winner per window for secondary comparison.','historical_review_status':'Already seen under original model; historical re-evaluation, not a virgin test set.','retries':'lbfgs200, lbfgs1000, powell1000; any remaining failure disqualifies entire configuration, never drop dates.','bootstrap_blocks':[5,10],'bootstrap_replications':5000,'fit_scale':100,'stability_rule':'Report both temporal halves, all seven calendar-month segments (March and September partial), and paired block-bootstrap CIs vs original and zero-return baseline. No post-review re-selection.'}
    protocol_path=OUT/'protocol.json'
    if protocol_path.exists():
        assert json.loads(protocol_path.read_text(encoding='utf-8'))==protocol,'Protocol changed after execution'
    else: dump(protocol_path,protocol)
    dump(OUT/'differencing_diagnostics.json',diagnose(prices,returns))
    print('Frozen protocol: 18 configs; selecting on earlier validation POINT RMSE.',flush=True)
    validation=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs={pool.submit(walk,c,'validation'):c for c in CONFIGS}
        for done,future in enumerate(as_completed(jobs),1):
            item=future.result()
            validation.append(item)
            text=f'{done}/18 validation {item["id"]}: '
            print(text+(f'RMSE={scores(item["rows"])["point_rmse"]:.4f}' if item['eligible'] else item['failure']),flush=True)
    eligible=[v for v in validation if v['eligible']]
    assert eligible
    expected_dates=[str(d.date()) for d in returns.index[(returns.index>=VALID_START)&(returns.index<REVIEW_START)]]
    for v in eligible: assert [r['date'] for r in v['rows']]==expected_dates
    ranked=sorted(eligible,key=lambda v:(scores(v['rows'])['point_rmse'],v['id']))
    winner=ranked[0]
    window_winners=[min([v for v in eligible if v['config']['window']==w],key=lambda v:(scores(v['rows'])['point_rmse'],v['id'])) for w in WINDOWS]
    baseline_validation=zero_rows(winner['rows'])
    selected={'primary_candidate_id':winner['id'],'primary_candidate':winner['config'],'window_winner_ids':[v['id'] for v in window_winners],'validation_baseline_score':scores(baseline_validation),'candidate_beats_baseline_on_validation':scores(winner['rows'])['point_rmse']<scores(baseline_validation)['point_rmse'],'selection_frozen_before_review':True}
    dump(OUT/'selection.json',selected)
    print('Selection frozen: '+json.dumps(selected,ensure_ascii=False),flush=True)
    review=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(walk,v['config'],'review') for v in window_winners]
        for future in as_completed(futures):
            item=future.result()
            assert item['eligible'],item
            review.append(item)
            print(f'Review {item["id"]}: point RMSE={scores(item["rows"])["point_rmse"]:.4f}',flush=True)
    primary=next(v for v in review if v['id']==winner['id'])
    prior=pd.read_json(BASE/'predictions.json',convert_dates=['date','trained_through'])
    prior['date']=prior['date'].dt.strftime('%Y-%m-%d')
    original_rows=prior[['date','predicted_log_return','predicted_close','previous_close','actual_log_return','actual_close']].to_dict('records')
    assert [r['date'] for r in primary['rows']]==[r['date'] for r in original_rows]
    assert np.allclose([r['actual_close'] for r in primary['rows']],[r['actual_close'] for r in original_rows])
    baseline_review=zero_rows(primary['rows'])
    monthly=[]
    for month in sorted({r['date'][:7] for r in primary['rows']}):
        monthly.append({'month':month,'partial_month':month in ['2026-03','2026-09'],'selected':scores([r for r in primary['rows'] if r['date'].startswith(month)]),'original':scores([r for r in original_rows if r['date'].startswith(month)]),'baseline':scores([r for r in baseline_review if r['date'].startswith(month)])})
    ranking=[{'id':v['id'],'config':v['config'],'overall':scores(v['rows']),**subsets(v['rows'],VALID_MID),'retries':sum(r['retries'] for r in v['rows']),'dates_with_warnings':sum(bool(r['warnings']) for r in v['rows'])} for v in ranked]
    report={'source_url':'https://finance.yahoo.com/quote/000001.SS/history/','data_sha256':digest,'selection':selected,'validation_dates':[expected_dates[0],expected_dates[-1]],'validation_observations':len(expected_dates),'review_dates':[primary['rows'][0]['date'],primary['rows'][-1]['date']],'review_observations':len(primary['rows']),'validation_ranking':ranking,'ineligible_configs':[v for v in validation if not v['eligible']],'review_selected':scores(primary['rows']),'review_original':scores(original_rows),'review_baseline':scores(baseline_review),'review_halves':{'selected':subsets(primary['rows'],REVIEW_MID),'original':subsets(original_rows,REVIEW_MID),'baseline':subsets(baseline_review,REVIEW_MID)},'review_window_winners':[{'id':v['id'],'config':v['config'],'score':scores(v['rows'])} for v in sorted(review,key=lambda x:str(x['config']['window']))],'monthly':monthly,'bootstrap_vs_original':[block_comparison(primary['rows'],original_rows,b) for b in [5,10]],'bootstrap_vs_baseline':[block_comparison(primary['rows'],baseline_review,b) for b in [5,10]],'review_retries':sum(r['retries'] for v in review for r in v['rows']),'review_dates_with_warnings':sum(bool(r['warnings']) for v in review for r in v['rows']),'historical_review_limitation':protocol['historical_review_status']}
    dump(OUT/'comparison_results.json',report)
    dump(OUT/'selected_predictions.json',primary)
    dump(OUT/'validation_scores.json',ranking)
    dump(OUT/'window_review_predictions.json',review)
    print(json.dumps({k:report[k] for k in ['review_selected','review_original','review_baseline','bootstrap_vs_original','bootstrap_vs_baseline']},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke',action='store_true',help='Fit each of the 18 configurations on one validation date only.')
    parser.add_argument('--workers',type=int,default=3,help='Parallel processes for the full comparison (default: 3).')
    args=parser.parse_args()
    if args.workers<1:
        parser.error('--workers must be at least 1')
    if args.smoke:
        smoke()
    else:
        main(workers=args.workers)
