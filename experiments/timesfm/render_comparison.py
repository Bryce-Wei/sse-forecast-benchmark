"""Render saved, completed forecasts; this module does not train any model."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from common import OUT, HISTORY, read

MODEL = 'timesfm'
DISPLAY = 'TimesFM 2.5'


def main():
    predicted = pd.DataFrame(read(OUT / 'predictions.json'))
    history = pd.DataFrame(read(OUT / 'observations.json'))
    metrics = read(OUT / 'metrics.json')
    predicted['date'] = pd.to_datetime(predicted['date'])
    history['date'] = pd.to_datetime(history['date'])
    assert len(predicted) == 64
    cutoff = predicted.date.iloc[0]
    known = history.loc[history.date < cutoff]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.3), gridspec_kw={'width_ratios': [1.05, 1.25]})
    ax = axes[0]
    ax.plot(known.date, known.close, color='#2875bb', lw=1.2)
    ax.set_title('Historical closes before the first test day')
    ax.set_ylabel('SSE Composite close (points)')
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax = axes[1]
    ax.plot(predicted.date, predicted.actual_close, color='#d43b43', lw=1.5, label='Actual')
    ax.plot(predicted.date, predicted[MODEL + '_close'], color='#118c6b', lw=1.3, label=DISPLAY)
    ax.plot(predicted.date, predicted.lstm_close, color='#8767ad', lw=1.1, label='LSTM')
    ax.plot(predicted.date, predicted.previous_close, color='#777777', lw=1, ls='--', label='Previous close')
    ax.set_title('Rolling next-day predictions: 64 test dates')
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=.22)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(axis='x', labelrotation=25)
    score = metrics['overall']
    line = '   |   '.join(f"{label}: RMSE {score[key]['rmse']:.2f}, MAE {score[key]['mae']:.2f}" for key, label in [(MODEL, DISPLAY), ('lstm', 'LSTM'), ('baseline', 'Baseline')])
    fig.suptitle(f'SSE Composite: {DISPLAY} vs LSTM | {HISTORY} input observations', fontsize=14)
    fig.text(.5, .095, line, ha='center', fontsize=9)
    note = 'Each curve joins daily one-step forecasts; this is not a single three-month forecast.'
    if MODEL == 'timesfm':
        note += '\nLeft panel is LSTM history; TimesFM uses frozen external pretrained weights.'
    fig.text(.5, .035, note, ha='center', fontsize=8, color='#555555')
    fig.tight_layout(rect=(0, .13, 1, .93))
    fig.savefig(OUT / 'comparison.png', dpi=180)
    plt.close(fig)
    print(OUT / 'comparison.png')


if __name__ == '__main__':
    main()
