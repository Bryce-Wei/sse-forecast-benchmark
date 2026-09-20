"""Render local full-run ARIMA results; no fitting or model selection."""
import argparse
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from compare_models import OUT, load_data, config_label


def main(language='en'):
    result=json.loads((OUT/'comparison_results.json').read_text(encoding='utf-8'))
    selected=json.loads((OUT/'selected_predictions.json').read_text(encoding='utf-8'))
    pred=pd.DataFrame(selected['rows'])
    pred['date']=pd.to_datetime(pred['date'])
    pred=pred.set_index('date')
    prices,returns,digest=load_data()
    assert digest==result['data_sha256'], 'Data changed since predictions were produced'
    assert len(pred)==125 and pred.index.is_unique
    config=selected['config']
    label=config_label(config)
    train=returns.loc[returns.index<pred.index[0]]
    if config['window'] is not None:
        train=train.tail(config['window'])
    dates=f'{pred.index[0]:%Y-%m-%d} to {pred.index[-1]:%Y-%m-%d}'
    from plotting import render
    render(prices, train, selected['rows'], OUT/'sse_arima_improved_points.png',
           OUT/'sse_arima_improved_returns.png', label, language)
    # Error bars are a metric comparison, distinct from the two-panel forecasts.
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
                         'axes.edgecolor':'#bfbfbf','axes.linewidth':0.8,
                         'grid.color':'#dddddd','grid.linewidth':0.7,
                         'savefig.facecolor':'white'})

    fig,(ax,bx)=plt.subplots(1,2,figsize=(12,5.8),gridspec_kw={'width_ratios':[1,2]})
    fig.subplots_adjust(left=.08,right=.98,bottom=.19,top=.82,wspace=.26)
    colors=['#3572bd','#14965e','#777777']
    values=[result[key]['point_rmse'] for key in ['review_original','review_selected','review_baseline']]
    bars=ax.bar(np.arange(3),values,color=colors,width=.6,zorder=3)
    ax.bar_label(bars,fmt='%.2f',padding=5)
    ax.set_xticks(np.arange(3),['Original\nARIMA(5,1,0)','Selected\ncandidate','Previous\nclose'],fontsize=9)
    ax.set_ylim(0,max(values)*1.25)
    ax.set(ylabel='Index-point RMSE (lower is better)',title='Full review period')
    ax.grid(axis='y',zorder=0)
    monthly=result['monthly']
    x=np.arange(len(monthly))
    series=[(-.24,'original','Original',colors[0]),(0,'selected','Selected',colors[1]),(.24,'baseline','Previous close',colors[2])]
    for shift,key,name,color in series:
        bx.bar(x+shift,[m[key]['point_rmse'] for m in monthly],width=.23,color=color,label=name,zorder=3)
    bx.set_xticks(x,[m['month'][5:]+('*' if m['partial_month'] else '') for m in monthly])
    bx.set(xlabel='Month in 2026 (* partial month)',title='Monthly error comparison')
    bx.set_ylim(0,max(m[key]['point_rmse'] for m in monthly for key in ['original','selected','baseline'])*1.3)
    bx.legend(loc='upper right',fontsize=9)
    bx.grid(axis='y',zorder=0)
    fig.suptitle('ARIMA comparison on the same historical review dates',fontsize=15,y=.96)
    fig.text(.5,.90,f'{dates} | {len(pred)} sessions | Candidate selection uses the preceding validation year',ha='center',fontsize=10)
    fig.text(.5,.05,'The review period was previously inspected. These results are a historical re-evaluation.',ha='center',fontsize=9)
    fig.savefig(OUT/'model_comparison.png',dpi=180)
    plt.close(fig)

    lines=[
        '# ARIMA comparison', '',
        f'Validation-selected candidate: **{label}**. Review: {dates} ({len(pred)} sessions).', '',
        '| Model | Point RMSE | Point MAE |', '|---|---:|---:|',
    ]
    for key,name in [('review_original','Original ARIMA(5,1,0)'),('review_selected','Validation-selected candidate'),('review_baseline','Previous-close baseline')]:
        score=result[key]
        lines.append(f'| {name} | {score["point_rmse"]:.4f} | {score["point_mae"]:.4f} |')
    lines.extend(['','## Paired circular block bootstrap','',
                  'Positive reductions indicate lower selected-model RMSE. Conditional on the saved forecasts; no correction for model search.',
                  '', '| Reference | Block days | RMSE reduction | 95% interval |','|---|---:|---:|---:|'])
    for key,name in [('bootstrap_vs_original','Original'),('bootstrap_vs_baseline','Previous close')]:
        for item in result[key]:
            low,high=item['ci95_percent']
            lines.append(f'| {name} | {item["block_length"]} | {item["rmse_reduction_percent"]:.2f}% | [{low:.2f}%, {high:.2f}%] |')
    lines.extend(['','## Interpretation','',
                  '- Every prediction uses strictly earlier returns and the previous observed close.',
                  '- The candidate and the winners for each window are selected on validation data before review evaluation.',
                  '- Check the previous-close baseline before inferring forecasting skill from visually close price curves.',
                  '- This historical review interval was inspected during earlier experiments; it is not a new untouched test set.',
                  '- If the candidate fails to beat the baseline on validation, the stated selection protocol retains the baseline as the default.',
                  '', '## Outputs','',
                  '- [Return forecasts](sse_arima_improved_returns.png)',
                  '- [Index forecasts](sse_arima_improved_points.png)',
                  '- [Error comparison](model_comparison.png)',
                  '- [Metrics](comparison_results.json)',
                  '- [Protocol](protocol.json)',
                  '- [Predictions](selected_predictions.json)',
                  '', f'Data fingerprint: `{digest}`.'])
    (OUT/'comparison_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Saved three figures and comparison_report.md in runs/arima_improved.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--language', choices=['zh','en'], default='en')
    main(parser.parse_args().language)
