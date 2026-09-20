"""Recompute archived metrics and check dates and artifact hashes, entirely offline."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REF = REPO / "results" / "reference"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def metrics(rows, prediction):
    errors = [float(row[prediction]) - float(row["actual_close"]) for row in rows]
    assert errors and all(math.isfinite(error) for error in errors)
    return {"rmse": math.sqrt(math.fsum(error * error for error in errors) / len(errors)),
            "mae": math.fsum(abs(error) for error in errors) / len(errors)}


def assert_metrics(rows, column, expected, prefix=""):
    actual = metrics(rows, column)
    for metric, value in actual.items():
        assert math.isclose(value, expected[prefix + metric], abs_tol=1e-7, rel_tol=1e-10), (
            column, metric, value, expected[prefix + metric])
    return actual


def check_dates(rows, count, start, end):
    dates = [row["date"][:10] for row in rows]
    assert len(dates) == count and dates == sorted(set(dates))
    assert (dates[0], dates[-1]) == (start, end)
    for row in rows:
        day = row["date"][:10]
        for field in ["train_end", "trained_through", "input_end"]:
            if field in row:
                assert row[field][:10] < day, (day, field)
    for previous, current in zip(rows, rows[1:]):
        assert math.isclose(previous["actual_close"], current["previous_close"], abs_tol=1e-7)


def main():
    manifest = read(REF / "manifest.json")
    for entry in manifest["reference_files"]:
        assert hashlib.sha256((REPO / entry["file"]).read_bytes()).hexdigest() == entry["published_sha256"]
    artifacts = {**manifest["figures"], manifest["report_pdf"]["path"]: manifest["report_pdf"]["sha256"]}
    for path, expected in artifacts.items():
        assert hashlib.sha256((REPO / path).read_bytes()).hexdigest() == expected, path

    comparison = read(REF / "arima/comparison_results.json")
    selected = read(REF / "arima/selected_predictions.json")["rows"]
    original = read(REF / "arima/original_predictions.json")
    check_dates(selected, 125, "2026-03-16", "2026-09-11")
    check_dates(original, 125, "2026-03-16", "2026-09-11")
    assert [r["date"][:10] for r in selected] == [r["date"][:10] for r in original]
    output = {"arima_selected": assert_metrics(selected, "predicted_close", comparison["review_selected"], "point_"),
              "arima_original": assert_metrics(original, "predicted_close", comparison["review_original"], "point_"),
              "arima_baseline": assert_metrics(selected, "previous_close", comparison["review_baseline"], "point_")}
    for row in selected:
        assert math.isclose(row["previous_close"] * math.exp(row["predicted_log_return"]),
                            row["predicted_close"], abs_tol=1e-7)
    paired_rows = []
    for name in ["patchtst", "timesfm"]:
        rows = read(REF / name / "predictions.json")
        published = read(REF / name / "metrics.json")
        check_dates(rows, 64, "2026-06-15", "2026-09-11")
        for model, column in [(name, name + "_close"), ("lstm", "lstm_close"), ("baseline", "previous_close")]:
            output[f"{name}_{model}"] = assert_metrics(rows, column, published["overall"][model])
        paired_rows.append(rows)
    for left, right in zip(*paired_rows):
        assert all(left[key] == right[key] for key in ["date", "actual_close", "previous_close"])
    print(json.dumps(output, indent=2))
    print("PASS: 18 MAE/RMSE values, chronological dates, paired baselines, 14 reference files, 6 figures and PDF.")


if __name__ == "__main__":
    main()
