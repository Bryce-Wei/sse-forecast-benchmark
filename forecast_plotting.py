"""Shared, model-free plotting of saved one-step forecasts and real history.

Only pandas, NumPy and Matplotlib are required. This module never fetches data,
loads model weights, or substitutes model predictions for historical closes.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RED = "#C9363E"
GREEN = "#238247"
BLUE = "#29659B"
GRAY = "#707880"
DARKGREEN = "#6B9D76"


def configure_style(language: str = "zh") -> str:
    """Choose an installed font; fail clearly rather than emit missing glyphs."""
    if language not in {"zh", "en"}:
        raise ValueError("language must be 'zh' or 'en'.")
    installed = {font.name for font in font_manager.fontManager.ttflist}
    candidates = ["Microsoft YaHei", "Noto Sans CJK SC", "Noto Sans CJK JP",
                  "Source Han Sans SC", "WenQuanYi Zen Hei", "PingFang SC", "SimHei"]
    font = next((name for name in candidates if name in installed), None)
    if language == "zh" and font is None:
        raise RuntimeError(
            "No supported Chinese font is installed. Install Noto Sans CJK SC "
            "(or Microsoft YaHei/PingFang SC), then rerun; alternatively use "
            "--language en (plot_forecast(..., language='en') in Python)."
        )
    chosen = font if language == "zh" else "DejaVu Sans"
    plt.rcParams.update({
        "font.family": chosen, "font.size": 9.5,
        "axes.titlesize": 10, "axes.labelsize": 9.5,
        "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
        "legend.fontsize": 9.5, "axes.unicode_minus": False,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#AAB2B8", "axes.linewidth": .65,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "grid.color": "#E5E9EC",
        "grid.linewidth": .5,
    })
    return chosen


def style_axis(ax, ylabel: str = ""):
    ax.set_ylabel(ylabel, labelpad=6)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.tick_params(length=3, width=.6, pad=3)
    ax.margins(x=.012)


def date_axis(ax, full_history: bool = False):
    if full_history:
        locator = mdates.AutoDateLocator(minticks=3, maxticks=6)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    else:
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))


def save_figure(fig, output: str | Path, dpi: int = 220) -> dict:
    """Save a fixed physical canvas and return portable artifact metadata."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    for text in fig.findobj(matplotlib.text.Text):
        if text.get_text() and text.get_visible() and text.get_fontsize() < 9.5:
            raise ValueError("Report text must be at least 9.5 points.")
    inches = [float(x) for x in fig.get_size_inches()]
    fig.savefig(output, dpi=dpi)
    plt.close(fig)
    return {"sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "inches": inches, "pixels": [int(round(x * dpi)) for x in inches],
            "dpi": dpi}


def validate_forecast(history, rows, predictions) -> tuple[pd.Series, pd.DataFrame]:
    """Require genuine ordered history and internally consistent saved forecasts.

    ``history`` must end immediately before the first forecast in the supplied
    saved series. Pass a date-indexed Series/DataFrame or a date/close DataFrame.
    The caller remains responsible for matching observations against its source.
    """
    if isinstance(history, pd.Series):
        historical = history.astype(float).copy()
    elif isinstance(history, pd.DataFrame):
        if "close" not in history:
            raise ValueError("Historical DataFrame must contain 'close'.")
        historical = history.set_index("date")["close"] if "date" in history else history["close"]
        historical = historical.astype(float).copy()
    else:
        raise TypeError("history must be a pandas Series or DataFrame of real closes.")
    historical.index = pd.DatetimeIndex(pd.to_datetime(historical.index))
    if historical.index.tz is not None:
        historical.index = historical.index.tz_localize(None)
    historical.index = historical.index.normalize()
    frame = pd.DataFrame(rows).copy()
    if frame.empty or historical.empty:
        raise ValueError("Historical and test observations must both be nonempty.")
    required = {"date", "actual_close", "previous_close"} | {item[0] for item in predictions}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing saved forecast fields: {sorted(missing)}")
    if not predictions:
        raise ValueError("At least one prediction curve is required.")
    frame["date"] = pd.to_datetime(frame["date"])
    if frame["date"].dt.tz is not None:
        frame["date"] = frame["date"].dt.tz_localize(None)
    frame["date"] = frame["date"].dt.normalize()
    if (not historical.index.is_unique or not historical.index.is_monotonic_increasing
            or not frame.date.is_unique or not frame.date.is_monotonic_increasing
            or historical.index.hasnans or frame.date.isna().any()):
        raise ValueError("Historical and test dates must be unique, ordered, and valid.")
    if historical.index[-1] >= frame.date.iloc[0]:
        raise ValueError("History must end strictly before the first test date.")
    numeric = frame[sorted(required - {"date"})].to_numpy(dtype=float)
    if (not np.isfinite(numeric).all() or not np.isfinite(historical.to_numpy()).all()
            or (historical <= 0).any() or (frame.actual_close <= 0).any()
            or (frame.previous_close <= 0).any()):
        raise ValueError("Close observations must be finite and strictly positive.")
    if not np.isclose(historical.iloc[-1], frame.previous_close.iloc[0], rtol=0, atol=1e-7):
        raise ValueError("The final historical close does not match the first previous_close.")
    if len(frame) > 1 and not np.allclose(frame.previous_close.iloc[1:], frame.actual_close.iloc[:-1], rtol=0, atol=1e-7):
        raise ValueError("Saved test observations do not form a contiguous one-step series.")
    return historical, frame


def plot_forecast(history, rows, predictions, output, history_label,
                  language="zh", *, figsize=(6.7, 3.5), dpi=220) -> dict:
    """Plot real pre-test history above, followed by a test-period close-up.

    ``predictions`` is a sequence of ``(field, label, color, linestyle)`` tuples.
    The first curve is the principal forecast shown with actuals in the top
    panel. The bottom panel includes all supplied comparison/baseline curves.
    ``history_label`` must describe what the real observations represent; for
    externally pretrained models call them reference history, not local training.
    """
    font = configure_style(language)
    historical, frame = validate_forecast(history, rows, predictions)
    zh = language == "zh"
    d = frame.date
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    fig.subplots_adjust(left=.115, right=.985, bottom=.135, top=.96, hspace=.83)
    ax = axes[0]
    ax.plot(historical.index, historical, color=BLUE, lw=.85, zorder=3)
    field, label, color, linestyle = predictions[0]
    ax.plot(d, frame.actual_close, color=RED, lw=1.0, zorder=4)
    ax.plot(d, frame[field], color=color, lw=1.0, ls=linestyle, zorder=4)
    cutoff = d.iloc[0] - pd.Timedelta(hours=12)
    ax.axvspan(cutoff, d.iloc[-1] + pd.Timedelta(days=1), color="#EEF5E9", zorder=0)
    ax.axvline(cutoff, color="#4D5963", lw=1.25, ls="--", zorder=5)
    bbox = {"facecolor": "white", "edgecolor": "none", "pad": 1.4, "alpha": .94}
    ax.text(.018, .96, history_label, transform=ax.transAxes, color=BLUE,
            va="top", bbox=bbox)
    ax.text(.99, .96, "测试期" if zh else "Test", transform=ax.transAxes,
            color="#48535D", va="top", ha="right", bbox=bbox)
    style_axis(ax, "指数点位" if zh else "Index points")
    date_axis(ax, full_history=True)
    # Keep the top labels above all data, including peaks in the test region.
    ax.margins(y=.22)
    ax = axes[1]
    ax.plot(d, frame.actual_close, color=RED, lw=1.25,
            label="真实收盘" if zh else "Actual", zorder=5)
    for field, label, color, linestyle in predictions:
        ax.plot(d, frame[field], color=color, lw=1.05, ls=linestyle, label=label, zorder=4)
    count = len(predictions) + 1
    columns = count if zh or count <= 3 else 2
    ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.0), ncol=columns,
              frameon=False, handlelength=1.55, columnspacing=.75,
              handletextpad=.4, borderaxespad=.3)
    style_axis(ax, "指数点位" if zh else "Index points")
    date_axis(ax)
    years = sorted(set(frame.date.dt.year))
    year_label = str(years[0]) if len(years) == 1 else f"{years[0]}–{years[-1]}"
    ax.set_xlabel(f"测试期放大 · {year_label}年交易日期" if zh else f"Test close-up · trading date ({year_label})", labelpad=3)
    metadata = save_figure(fig, output, dpi=dpi)
    metadata.update({
        "layout": "history_and_test_above_test_closeup_below",
        "font_family": font, "language": language,
        "history": {"label": history_label, "count": len(historical),
                    "start": historical.index[0].strftime("%Y-%m-%d"),
                    "end": historical.index[-1].strftime("%Y-%m-%d")},
        "test": {"count": len(frame), "start": d.iloc[0].strftime("%Y-%m-%d"),
                 "end": d.iloc[-1].strftime("%Y-%m-%d")},
        "rmse_points": {field: float(np.sqrt(np.mean((frame[field] - frame.actual_close) ** 2)))
                        for field, *_ in predictions},
    })
    return metadata
