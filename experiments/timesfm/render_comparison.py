"""Render completed paired forecasts using the common history/test layout."""
import argparse
import json
import pandas as pd
from common import OUT, HISTORY, read
from forecast_plotting import plot_forecast, GREEN, DARKGREEN, GRAY

MODEL = 'timesfm'
DISPLAY = 'TimesFM 2.5'


def main(language='zh'):
    rows = read(OUT / 'predictions.json')
    observed = pd.DataFrame(read(OUT / 'observations.json'))
    observed['date'] = pd.to_datetime(observed['date'])
    observed = observed.set_index('date').close
    if len(rows) != 64:
        raise ValueError('Only a complete 64-date experiment can produce the comparison figure.')
    first = rows[0]
    known = observed.loc[first['train_start']:first['train_end']]
    if len(known) != first['train_observations']:
        raise ValueError('Historical plot does not match the initial training window.')
    zh = language == 'zh'
    history_label = ('历史参照（LSTM训练期）' if zh else 'Reference history (LSTM training)') if MODEL == 'timesfm' else ('首次训练历史' if zh else 'Initial training history')
    audit = plot_forecast(
        known, rows, [(MODEL+'_close', DISPLAY, GREEN, '-'),
                      ('lstm_close', f'LSTM（{HISTORY}日）' if zh else f'LSTM ({HISTORY} days)', DARKGREEN, ':'),
                      ('previous_close', '前日基准' if zh else 'Previous close', GRAY, '--')],
        OUT/'comparison.png', history_label, language)
    (OUT/'plot_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    print(OUT/'comparison.png')
    return audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--language', choices=['zh','en'], default='zh')
    main(parser.parse_args().language)
