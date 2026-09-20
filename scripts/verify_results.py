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


def check_figures(manifest):
    """Check generated images and provenance without requiring an installed data cache."""
    generated = read(REPO / "reports/figures/figures_manifest.json")
    assert generated["layout"] == "standard"
    assert generated["canonical_prices_sha256"] == manifest["canonical_prices_sha256"]
    assert set(generated["figures"]) == {Path(path).name for path in manifest["figures"]}
    for group in ["source_sha256", "code_sha256"]:
        for relative, expected in generated[group].items():
            path = REPO / relative
            # The historical cache is intentionally not distributed with the repository.
            if group == "source_sha256" and relative not in {
                "results/reference/manifest.json", *[e["file"] for e in manifest["reference_files"]]
            }:
                continue
            assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, relative
    regenerated = 0
    for relative, published_hash in manifest["figures"].items():
        actual = hashlib.sha256((REPO / relative).read_bytes()).hexdigest()
        assert actual == generated["figures"][Path(relative).name]["sha256"], relative
        regenerated += actual != published_hash
    pdf = manifest["report_pdf"]
    assert hashlib.sha256((REPO / pdf["path"]).read_bytes()).hexdigest() == pdf["sha256"]
    return regenerated


def main():
    manifest = read(REF / "manifest.json")
    for entry in manifest["reference_files"]:
        assert hashlib.sha256((REPO / entry["file"]).read_bytes()).hexdigest() == entry["published_sha256"]
    regenerated = check_figures(manifest)

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
        history = 20 if name == "patchtst" else 32
        standalone = read(REF / f"lstm/history_{history}/predictions.json")
        standalone_metrics = read(REF / f"lstm/history_{history}/metrics.json")
        check_dates(standalone, 64, "2026-06-15", "2026-09-11")
        assert standalone_metrics["protocol"] == {
            "history": history, "train_years": 2, "horizon": 1, "review_count": 64}
        assert standalone_metrics["source"] == f"results/reference/{name}/predictions.json"
        assert standalone_metrics["source_sha256"] == hashlib.sha256((REF / name / "predictions.json").read_bytes()).hexdigest()
        fields = ["date", "actual_close", "previous_close", "lstm_close", "lstm_epoch",
                  "train_start", "train_end", "train_observations", "train_windows"]
        assert standalone == [{key: row[key] for key in fields} for row in rows]
        assert standalone_metrics["data"] == published["data"]
        assert standalone_metrics["overall"] == {key: published["overall"][key] for key in ["lstm", "baseline"]}
        assert standalone_metrics["monthly"] == [
            {key: part[key] for key in ["month", "count", "lstm", "baseline"]}
            for part in published["monthly"]]
        paired_rows.append(rows)
    for left, right in zip(*paired_rows):
        assert all(left[key] == right[key] for key in ["date", "actual_close", "previous_close"])
    print(json.dumps(output, indent=2))
    print(f"PASS: {len(output) * 2} MAE/RMSE values, chronological dates, paired baselines, "
          f"128 matching standalone LSTM records, {len(manifest['reference_files'])} reference files, "
          f"{len(manifest['figures'])} figures and PDF. Regenerated figures: {regenerated}.")


if __name__ == "__main__":
    main()
