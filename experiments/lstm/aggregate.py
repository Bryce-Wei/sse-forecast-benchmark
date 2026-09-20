"""Score the standalone shared LSTM only after the full review is complete."""
import argparse
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE))
import numpy as np
import pandas as pd
from experiments.lstm.protocol import (
    EPOCHS, frame, load_records, make_protocol, output_dir, read, dump, metric,
)


def main(history, language='zh'):
    out = output_dir(history); f = frame()
    records = load_records(history, require_complete=True)
    reloads = read(out / 'reload_lstm.json')
    assert len(reloads) == 3 and all(r['passed'] and r['history'] == history for r in reloads)
    joined = []
    for row in records:
        day = pd.Timestamp(row['date'])
        joined.append({'date': row['date'], 'actual_close': float(f.loc[day, 'close']),
                       'previous_close': row['previous_close'], 'lstm_close': row['predicted_close'],
                       'lstm_epoch': row['selected_epoch'], 'train_start': row['train_start'],
                       'train_end': row['train_end'], 'train_observations': row['train_observations'],
                       'train_windows': row['train_windows']})
    df = pd.DataFrame(joined)
    columns = {'lstm': 'lstm_close', 'baseline': 'previous_close'}
    overall = {key: metric(df.actual_close, df[col]) for key, col in columns.items()}
    monthly = [{'month': month, 'count': len(part),
                **{key: metric(part.actual_close, part[col]) for key, col in columns.items()}}
               for month, part in df.groupby(df.date.str[:7], sort=True)]
    direction = float(np.mean(np.sign(df.lstm_close-df.previous_close) ==
                              np.sign(df.actual_close-df.previous_close)))
    returns = np.log(df.actual_close/df.previous_close)
    return_errors = {key: metric(returns, np.log(df[col]/df.previous_close))
                     for key, col in columns.items()}
    epochs = [r['selected_epoch'] for r in records]
    metrics = {
        'data': {'symbol': '000001.SS', 'review_start': joined[0]['date'],
                 'review_end': joined[-1]['date'], 'review_count': len(joined), 'input_history': history,
                 'initial_train_start': joined[0]['train_start'], 'initial_train_end': joined[0]['train_end'],
                 'last_train_start': joined[-1]['train_start'], 'last_train_end': joined[-1]['train_end']},
        'protocol': make_protocol(history), 'overall': overall, 'monthly': monthly,
        'direction_accuracy': {'lstm': direction}, 'actual_up_days': int((df.actual_close > df.previous_close).sum()),
        'return_errors': return_errors, 'reload_checks': {'lstm': reloads},
        'models': {'lstm': {'parameters': records[0]['parameter_count'], 'config': records[0]['model_config'],
                            'epochs_min': min(epochs), 'epochs_max': max(epochs),
                            'epochs_median': float(np.median(epochs)), 'epochs_at_cap': sum(e == EPOCHS for e in epochs)}},
    }
    dump(out / 'predictions.json', joined)
    dump(out / 'metrics.json', metrics)
    display = f.loc[f.index >= pd.Timestamp(joined[0]['date'])-pd.DateOffset(years=2)]
    dump(out / 'observations.json', [{'date': str(day.date()), 'close': float(row.close)}
                                    for day, row in display.iterrows()])
    print({'complete': True, 'history': history, 'count': len(joined), 'overall': overall})
    from experiments.lstm.render_comparison import main as render
    render(history, language=language)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history', type=int, choices=[20, 32], required=True)
    parser.add_argument('--language', choices=['zh', 'en'], default='zh')
    main(**vars(parser.parse_args()))
