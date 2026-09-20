"""Data-integrity tests use synthetic fixtures, never model performance claims."""
from copy import deepcopy
from datetime import datetime, timezone
import unittest

from benchmark_support import frame_fingerprint, parse_prices


def fixture():
    timestamps = [int(datetime(2026, 9, day, tzinfo=timezone.utc).timestamp()) for day in [10, 11, 14]]
    return {"chart": {"error": None, "result": [{"meta": {"symbol": "000001.SS", "instrumentType": "INDEX"},
            "timestamp": timestamps, "indicators": {"quote": [{"close": [100.0, 101.0, 999.0]}]}}]}}


class DataIntegrityTests(unittest.TestCase):
    def test_future_values_do_not_enter_historical_fingerprint(self):
        before = fixture()
        after = deepcopy(before)
        after["chart"]["result"][0]["indicators"]["quote"][0]["close"][-1] = 5000
        self.assertEqual(frame_fingerprint(parse_prices(before)), frame_fingerprint(parse_prices(after)))
        self.assertEqual(len(parse_prices(before)), 2)

    def test_provider_metadata_does_not_change_prices_fingerprint(self):
        before = fixture()
        after = deepcopy(before)
        after["chart"]["result"][0]["meta"]["regularMarketPrice"] = 200
        self.assertEqual(frame_fingerprint(parse_prices(before)), frame_fingerprint(parse_prices(after)))

    def test_duplicate_dates_rejected(self):
        payload = fixture()
        payload["chart"]["result"][0]["timestamp"][1] = payload["chart"]["result"][0]["timestamp"][0]
        with self.assertRaises(ValueError):
            parse_prices(payload)

    def test_missing_close_rejected(self):
        payload = fixture()
        payload["chart"]["result"][0]["indicators"]["quote"][0]["close"][0] = None
        with self.assertRaises(ValueError):
            parse_prices(payload)

    def test_wrong_instrument_rejected(self):
        payload = fixture()
        payload["chart"]["result"][0]["meta"]["instrumentType"] = "EQUITY"
        with self.assertRaises(ValueError):
            parse_prices(payload)


if __name__ == "__main__":
    unittest.main()
