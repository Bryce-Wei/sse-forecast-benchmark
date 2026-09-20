"""Guard against misleading historical/test boundaries in every forecast plot."""
import unittest
import pandas as pd

from forecast_plotting import validate_forecast


class ForecastBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.history = pd.Series([100.0, 101.0], index=pd.to_datetime(['2026-06-11','2026-06-12']))
        self.rows = [
            {'date':'2026-06-15','actual_close':102.0,'previous_close':101.0,'predicted_close':101.5},
            {'date':'2026-06-16','actual_close':103.0,'previous_close':102.0,'predicted_close':102.5}]
        self.curves = [('predicted_close','Model','#238247','-')]

    def test_valid_boundary_keeps_history_separate(self):
        history, frame = validate_forecast(self.history, self.rows, self.curves)
        self.assertLess(history.index.max(), frame.date.min())
        self.assertEqual(len(history), 2)
        self.assertEqual(len(frame), 2)

    def test_history_cannot_include_test_day(self):
        history = pd.concat([self.history, pd.Series([102.0], index=pd.to_datetime(['2026-06-15']))])
        with self.assertRaisesRegex(ValueError, 'strictly before'):
            validate_forecast(history, self.rows, self.curves)

    def test_wrong_history_boundary_close_rejected(self):
        history = self.history.copy()
        history.iloc[-1] = 99
        with self.assertRaisesRegex(ValueError, 'previous_close'):
            validate_forecast(history, self.rows, self.curves)

    def test_missing_intermediate_test_day_detected_by_previous_close(self):
        self.rows[1]['previous_close'] = 101
        with self.assertRaisesRegex(ValueError, 'contiguous'):
            validate_forecast(self.history, self.rows, self.curves)

    def test_reversed_test_rows_rejected(self):
        with self.assertRaisesRegex(ValueError, 'ordered'):
            validate_forecast(self.history, self.rows[::-1], self.curves)


if __name__=='__main__':
    unittest.main()
