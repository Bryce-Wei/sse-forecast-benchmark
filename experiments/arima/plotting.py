"""ARIMA plots: real initial history, a held-out test region, then test detail."""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from forecast_plotting import plot_forecast, configure_style, GREEN, GRAY


def price_history(prices, train_returns):
    # N log returns depend on N+1 observed closes; include the initial lagged close.
    begin = prices.index.get_loc(train_returns.index[0])
    if begin < 1:
        raise ValueError('The initial return requires one earlier observed close.')
    history = prices.iloc[begin-1:].loc[:train_returns.index[-1]]
    assert len(history) == len(train_returns) + 1
    return history


def plot_returns(train, rows, output, language='en'):
    configure_style(language)
    frame = pd.DataFrame(rows)
    dates = pd.to_datetime(frame['date'])
    if train.index[-1] >= dates.iloc[0]:
        raise ValueError('Training returns overlap the forecast dates.')
    if not np.isfinite(frame[['actual_log_return', 'predicted_log_return']].to_numpy()).all():
        raise ValueError('Non-finite return forecast.')
    split = train.index[-1] + (dates.iloc[0] - train.index[-1]) / 2
    fig, axes = plt.subplots(2, 1, figsize=(6.7, 3.9))
    fig.subplots_adjust(left=.12, right=.985, top=.96, bottom=.12, hspace=.73)
    zh = language == 'zh'
    actual_label = '真实收益率' if zh else 'Actual return'
    pred_label = 'ARIMA预测' if zh else 'ARIMA forecast'
    axes[0].plot(train.index, train.to_numpy()*100, color='#29659B', lw=.8)
    for ax in axes:
        ax.plot(dates, frame.actual_log_return*100, color='#C9363E', lw=1.0, label=actual_label)
        ax.plot(dates, frame.predicted_log_return*100, color=GREEN, lw=1.0, label=pred_label)
        ax.axhline(0, color=GRAY, lw=.65, ls='--')
        ax.set_ylabel('对数收益率（%）' if zh else 'Log return (%)')
        ax.grid(axis='y', alpha=.7)
        ax.set_axisbelow(True)
        ax.margins(x=.012)
    axes[0].axvspan(split, dates.iloc[-1], color='#238247', alpha=.04)
    axes[0].axvline(split, color=GRAY, ls='--', lw=.9)
    axes[0].text(.01, .98, '首次训练历史' if zh else 'Initial training history',
                 color='#29659B', va='top', transform=axes[0].transAxes)
    axes[0].text(.99, .98, '测试' if zh else 'Test', ha='right', va='top', transform=axes[0].transAxes)
    axes[0].xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=6))
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axes[1].xaxis.set_major_locator(mdates.MonthLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    axes[1].legend(loc='lower center', bbox_to_anchor=(.5, 1.0), ncol=2, frameon=False)
    axes[1].set_xlabel('测试期放大 · 交易日期' if zh else 'Test-period detail - trading date')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def render(prices, train_returns, rows, points_path, returns_path, model_label='ARIMA', language='en'):
    history = price_history(prices, train_returns)
    audit = plot_forecast(
        history, rows,
        [('predicted_close', model_label, GREEN, '-'),
         ('previous_close', '前日基准' if language=='zh' else 'Previous close', GRAY, '--')],
        points_path, '首次训练历史' if language=='zh' else 'Initial training history', language)
    plot_returns(train_returns, rows, returns_path, language)
    audit_path = Path(points_path).with_suffix('.audit.json')
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    return audit
