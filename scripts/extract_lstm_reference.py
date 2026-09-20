"""Expose the already-published LSTM records as standalone 20/32-day views.

This copies values out of archived paired experiments; it never fits a model.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT/'results/reference'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def main():
    for history, group in [(20, 'patchtst'), (32, 'timesfm')]:
        source = REF/group/'predictions.json'
        paired = read(source)
        old_metrics = read(REF/group/'metrics.json')
        rows = [{key: row[key] for key in ['date','actual_close','previous_close','lstm_close',
                    'lstm_epoch','train_start','train_end','train_observations','train_windows']}
                for row in paired]
        out = REF/'lstm'/f'history_{history}'
        metadata = {
            'provenance': 'Exact LSTM-column extraction from the archived paired experiment; no new training.',
            'source': source.relative_to(ROOT).as_posix(),
            'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'data': old_metrics['data'],
            'protocol': {'history':history,'train_years':2,'horizon':1,'review_count':64},
            'overall': {key:old_metrics['overall'][key] for key in ['lstm','baseline']},
            'monthly':[{key:part[key] for key in ['month','count','lstm','baseline']}
                       for part in old_metrics['monthly']],
        }
        dump(out/'predictions.json', rows)
        dump(out/'metrics.json', metadata)
        print(f'Extracted LSTM {history}: {len(rows)} unchanged archived predictions.')


if __name__=='__main__':
    main()
