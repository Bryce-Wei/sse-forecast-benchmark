"""Small regression checks for leakage and incompatible shared cache records."""
import copy
import math
import unittest

import numpy as np
import pandas as pd

from experiments.lstm.protocol import (
    BATCH, EPOCHS, VALID, arrays, audit_fields, make_job, model_config,
    output_dir, validate_record,
)


class SharedProtocolTests(unittest.TestCase):
    def fixture(self, history=20):
        dates = pd.bdate_range('2023-01-02', '2026-09-11')
        values = 100 + np.arange(len(dates)) * .1 + np.sin(np.arange(len(dates)) / 20)
        f = pd.DataFrame({'close': values}, index=dates)
        protocol = {'history': history, 'data_fingerprint': 'synthetic-unit-test'}
        job = make_job(pd.Timestamp('2026-06-15'), f, protocol)
        a = arrays(job)
        row = audit_fields(job, a)
        n = len(job['dates'])
        row.update(model_config=model_config(history),
                   validation_history=[{'epoch': i, 'mae_points': EPOCHS+1-i} for i in range(1, EPOCHS+1)],
                   selected_epoch=EPOCHS, initial_weights_hash='test', refit_initial_weights_hash='test',
                   selection_optimizer_initial_iterations=0, refit_optimizer_initial_iterations=0,
                   selection_optimizer_final_iterations=EPOCHS*math.ceil((n-history-VALID)/BATCH),
                   refit_optimizer_final_iterations=EPOCHS*math.ceil((n-history)/BATCH),
                   predicted_close=200.0, model_file='models/lstm/2026-06-15.keras')
        return job, a, row

    def test_distinct_histories_and_past_only_scalers(self):
        self.assertNotEqual(output_dir(20), output_dir(32))
        for history in (20, 32):
            job, a, row = self.fixture(history)
            self.assertLess(max(job['dates']), job['date'])
            self.assertEqual(a['final']['next_x'].shape, (1, history, 1))
            self.assertEqual(a['validation']['scaler_fit_rows'], len(job['values'])-VALID)
            self.assertAlmostEqual(a['validation']['mean'], np.mean(job['values'][:-VALID]))
            validate_record(row, job, require_model=False)

    def test_changed_lookback_configuration_and_digest_are_rejected(self):
        job, _, row = self.fixture()
        mutations = [lambda r: r.update(history=32),
                     lambda r: r['model_config'].update(input_shape=[32, 1]),
                     lambda r: r.update(protocol_digest='wrong-snapshot'),
                     lambda r: r.update(selected_epoch=1)]
        for mutate in mutations:
            changed = copy.deepcopy(row)
            mutate(changed)
            with self.assertRaises(AssertionError):
                validate_record(changed, job, require_model=False)

    def test_target_day_cannot_enter_training_inputs(self):
        job, _, _ = self.fixture()
        job['dates'][-1] = job['date']
        with self.assertRaises(AssertionError):
            arrays(job)

    def test_cross_history_model_path_is_rejected(self):
        job, _, row = self.fixture()
        row['model_file'] = '../history_32/models/lstm/2026-06-15.keras'
        with self.assertRaises(AssertionError):
            validate_record(row, job, require_model=False)


if __name__ == '__main__':
    unittest.main()
