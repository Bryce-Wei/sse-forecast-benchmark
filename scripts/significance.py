"""Paired block-bootstrap RMSE comparisons from archived forecasts, entirely offline.

Reuses the bootstrap of the ARIMA comparison and first reproduces its archived
intervals, so the 64-day comparisons use exactly the same method and seeds.
Use --check to compare against the committed results instead of writing them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from experiments.arima.compare_models import block_comparison

OUTPUT = REPO / "results" / "analysis" / "significance.json"
BLOCKS = [5, 10]


def read(relative, sources):
    path = REPO / relative
    sources[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text(encoding="utf-8"))


def rows(records, column):
    return [{"date": row["date"][:10], "predicted_close": row[column], "actual_close": row["actual_close"]}
            for row in records]


def close(left, right):
    """Structural equality with a float tolerance for cross-platform summation order."""
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(close(left[k], right[k]) for k in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(close(a, b) for a, b in zip(left, right))
    if isinstance(left, float) or isinstance(right, float):
        return math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-9)
    return left == right


def analyse():
    sources = {}
    archived = read("results/reference/arima/comparison_results.json", sources)
    selected = read("results/reference/arima/selected_predictions.json", sources)["rows"]
    original = read("results/reference/arima/original_predictions.json", sources)
    arima = [
        ("arima", "Selected ARIMA (252-return mean)", "Original ARIMA(5,1,0)",
         rows(selected, "predicted_close"), rows(original, "predicted_close"), "bootstrap_vs_original"),
        ("arima", "Selected ARIMA (252-return mean)", "Previous-close baseline",
         rows(selected, "predicted_close"), rows(selected, "previous_close"), "bootstrap_vs_baseline"),
    ]
    comparisons = []
    for experiment, model, reference, model_rows, reference_rows, key in arima:
        result = [block_comparison(model_rows, reference_rows, block) for block in BLOCKS]
        if not close(result, archived[key]):
            raise SystemExit(f"Bootstrap does not reproduce archived ARIMA {key}; method or data differ.")
        comparisons.append({"experiment": experiment, "model": model, "reference": reference,
                            "n": len(model_rows), "bootstrap": result})
    for experiment, column, model, history in [("patchtst", "patchtst_close", "PatchTST", 20),
                                               ("timesfm", "timesfm_close", "TimesFM 2.5", 32)]:
        records = read(f"results/reference/{experiment}/predictions.json", sources)
        lstm = f"LSTM ({history}-day input)"
        for name, model_column, reference, reference_column in [
                (lstm, "lstm_close", "Previous-close baseline", "previous_close"),
                (model, column, lstm, "lstm_close"),
                (model, column, "Previous-close baseline", "previous_close")]:
            comparisons.append({"experiment": experiment, "model": name, "reference": reference,
                                "n": len(records),
                                "bootstrap": [block_comparison(rows(records, model_column),
                                                               rows(records, reference_column), block)
                                              for block in BLOCKS]})
    return {
        "method": "Paired circular moving-block bootstrap of the RMSE reduction, reused from "
                  "experiments/arima/compare_models.py (5000 replications, seed 5545 + block length). "
                  "Positive values mean the model's RMSE is lower than the reference's.",
        "scope": "Post-hoc analysis of archived forecasts; conditional on fixed forecasts, "
                 "not corrected for model search or multiple comparisons.",
        "arima_archive_reproduced": True,
        "source_sha256": sources,
        "comparisons": comparisons,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail if the committed results differ; write nothing")
    args = parser.parse_args()
    result = analyse()
    for item in result["comparisons"]:
        intervals = "  ".join(f"b={b['block_length']}: [{b['ci95_percent'][0]:6.2f}, {b['ci95_percent'][1]:6.2f}]"
                              for b in item["bootstrap"])
        print(f"{item['model']:>32} vs {item['reference']:<24} "
              f"{item['bootstrap'][0]['rmse_reduction_percent']:7.2f}%  {intervals}")
    if args.check:
        committed = json.loads(OUTPUT.read_text(encoding="utf-8"))
        if not close(result, committed):
            raise SystemExit(f"{OUTPUT.relative_to(REPO)} differs from the recomputed results.")
        print(f"PASS: {OUTPUT.relative_to(REPO)} matches; archived ARIMA intervals reproduced.")
        return
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Archived ARIMA intervals reproduced. Wrote {OUTPUT.relative_to(REPO)}.")


if __name__ == "__main__":
    main()
