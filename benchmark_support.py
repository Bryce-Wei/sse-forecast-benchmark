"""Shared local data loading for the published historical experiments.

Raw provider responses are downloaded by the reader, never embedded in this
repository. A canonical date/close fingerprint ignores mutable provider metadata.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent
DATA_PATH = Path(os.environ.get("SSE_DATA_PATH", REPO / "data" / "yahoo_sse.json"))
DATA_END = pd.Timestamp("2026-09-11")


def parse_prices(payload: dict) -> pd.DataFrame:
    """Validate Yahoo chart data and select the published observation horizon."""
    try:
        chart = payload["chart"]
        if chart.get("error"):
            raise ValueError(f"Provider error: {chart['error']}")
        result = chart["result"][0]
        meta = result["meta"]
        if meta.get("symbol") != "000001.SS" or meta.get("instrumentType") != "INDEX":
            raise ValueError("Expected Shanghai Composite index 000001.SS, not an equity.")
        timestamps = result["timestamp"]
        closes = result["indicators"]["quote"][0]["close"]
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError("The file is not a supported Yahoo Finance chart response.") from exc
    if len(timestamps) != len(closes) or not timestamps:
        raise ValueError("Mismatched or empty timestamp/close arrays.")
    dates = (pd.to_datetime(timestamps, unit="s", utc=True)
             .tz_convert("Asia/Shanghai").normalize().tz_localize(None))
    frame = pd.DataFrame({"close": closes}, index=dates).sort_index()
    if not frame.index.is_unique:
        raise ValueError("Duplicate trading dates in the provider response.")
    frame = frame.loc[:DATA_END].astype(float)
    if frame.empty or not np.isfinite(frame.to_numpy()).all() or (frame.close <= 0).any():
        raise ValueError("Missing/non-finite/non-positive closes; no forward-fill is performed.")
    return frame


def load_prices(path: str | Path | None = None) -> pd.DataFrame:
    source = Path(path) if path is not None else DATA_PATH
    if not source.is_file():
        raise FileNotFoundError(
            f"Missing {source}. Run: python scripts/fetch_data.py "
            "(or set SSE_DATA_PATH to your saved Yahoo chart JSON)."
        )
    return parse_prices(json.loads(source.read_text(encoding="utf-8")))


def frame_fingerprint(frame: pd.DataFrame) -> str:
    rows = [[date.strftime("%Y-%m-%d"), float(value)]
            for date, value in frame.close.items()]
    canonical = json.dumps(rows, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def data_fingerprint(path: str | Path | None = None) -> str:
    return frame_fingerprint(load_prices(path))


def configure_plot_font():
    """Use an installed CJK font if present; keep plots usable on other systems."""
    import matplotlib
    from matplotlib import font_manager
    available = {font.name for font in font_manager.fontManager.ttflist}
    fonts = [name for name in ["Microsoft YaHei", "Noto Sans CJK SC", "Noto Sans CJK JP",
                               "WenQuanYi Zen Hei", "PingFang SC", "SimHei"]
             if name in available]
    matplotlib.rcParams["font.family"] = fonts + ["DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
