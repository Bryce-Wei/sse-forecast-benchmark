"""Recreate report figures from archived predictions, without training/downloads.

Default: consistent history/test + test-close-up layouts for all four models.
Use --layout original to recreate the six layouts embedded in the original PDF.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchmark_support import DATA_PATH, frame_fingerprint, load_prices
from forecast_plotting import (BLUE, DARKGREEN, GRAY, GREEN, RED, configure_style,
                               plot_forecast, save_figure, style_axis,
                               validate_forecast)


def values(rows, key):
    return np.array([row[key] for row in rows], dtype=float)


def dates(rows):
    return pd.to_datetime([row["date"][:10] for row in rows])


def date_axis(ax, full=False):
    # Original PDF tick convention, retained for --layout original.
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[6, 12]) if full else mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m" if full else "%m-%d"))


class ReportFigures:
    def __init__(self, output, layout, language, data_path):
        self.output = Path(output)
        self.layout = layout
        self.language = language
        self.data_path = Path(data_path)
        self.sources = {}
        self.figures = {}
        self.verified = {}
        self.font = configure_style(language)

    def text(self, zh, en):
        return zh if self.language == "zh" else en

    def read(self, relative):
        path = REPO / relative
        self.sources[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text(encoding="utf-8"))

    def rmse(self, rows, field, expected, key, actual="actual_close"):
        result = float(np.sqrt(np.mean((values(rows, field) - values(rows, actual)) ** 2)))
        if not np.isclose(result, expected, atol=1e-7, rtol=1e-10):
            raise ValueError(f"Archived RMSE mismatch for {key}: {result} != {expected}")
        self.verified[key] = {"rmse": result, "metadata_rmse": expected, "count": len(rows)}
        return result

    def save(self, fig, name):
        self.figures[name] = save_figure(fig, self.output / name)

    def verify_source(self, prices, rows, expected_count, group):
        """Reject revised/misaligned raw prices before combining with forecasts."""
        wanted = dates(rows)
        observed = prices.loc[wanted.min():wanted.max()]
        if len(rows) != expected_count or not observed.index.equals(wanted):
            raise ValueError(f"{group}: source/test dates or observation counts differ from the archive.")
        if not np.allclose(observed.close, values(rows, "actual_close"), atol=1e-7, rtol=0):
            raise ValueError(f"{group}: source/test actual_close differs from the archive.")
        previous = prices.close.shift(1).reindex(wanted)
        if not np.allclose(previous, values(rows, "previous_close"), atol=1e-7, rtol=0):
            raise ValueError(f"{group}: source previous_close differs from the archive.")

    def overview(self, arima_scores, scores):
        t = self.text
        fig, axes = plt.subplots(1, 3, figsize=(6.7, 2.65))
        fig.subplots_adjust(left=.079, right=.99, bottom=.25, top=.79, wspace=.35)
        panels = [
            (t("ARIMA · 半年 / 125日", "ARIMA · 125 test days"),
             [t("原模型", "Orig."), t("选定模型", "Chosen"), t("前日基准", "Baseline")], arima_scores),
            (t("PatchTST · 三个月 / 64日", "PatchTST · 64 test days"),
             ["PatchTST", t("LSTM\n20日", "LSTM\n20 days"), t("前日基准", "Baseline")], scores["patchtst"]),
            (t("TimesFM · 三个月 / 64日", "TimesFM · 64 test days"),
             ["TimesFM", t("LSTM\n32日", "LSTM\n32 days"), t("前日基准", "Baseline")], scores["timesfm"]),
        ]
        for ax, (title, labels, nums) in zip(axes, panels):
            ax.set_title(title, pad=21, fontsize=9.5)
            bars = ax.bar(np.arange(3), nums, color=[GREEN, DARKGREEN, GRAY], width=.66, zorder=3)
            ax.set_xticks(np.arange(3), labels)
            ax.tick_params(axis="x", length=0, pad=5)
            ax.set_ylim(0, 65)
            ax.set_yticks([0, 20, 40, 60])
            ax.grid(axis="y")
            ax.set_axisbelow(True)
            for bar, num in zip(bars, nums):
                ax.text(bar.get_x() + bar.get_width()/2, num + 1.8, f"{num:.4f}", ha="center", va="bottom", fontsize=9.5)
        axes[0].set_ylabel(t("RMSE（点）", "RMSE (points)"))
        fig.text(.5, .025, t("各面板对应独立比较设置；误差越低越好。", "Separate comparison settings in each panel; lower error is better."), ha="center", color="#48535D")
        self.save(fig, "overview.png")

    def arima_returns(self, rows, name):
        t = self.text
        fig, axes = plt.subplots(2, 1, figsize=(6.7, 3.6), sharex=True, gridspec_kw={"height_ratios": [1.4, 1]})
        fig.subplots_adjust(left=.115, right=.985, bottom=.12, top=.91, hspace=.25)
        d = dates(rows)
        axes[0].plot(d, values(rows, "actual_close"), color=RED, lw=1.2, label=t("真实收盘", "Actual close"))
        axes[0].plot(d, values(rows, "predicted_close"), color=GREEN, lw=1.05, label=t("选定ARIMA", "Selected ARIMA"))
        axes[0].legend(loc="lower center", bbox_to_anchor=(.5, 1.0), ncol=2, frameon=False, borderaxespad=.3)
        axes[1].plot(d, values(rows, "actual_log_return") * 100, color=RED, lw=1.0)
        axes[1].plot(d, values(rows, "predicted_log_return") * 100, color=GREEN, lw=1.0)
        axes[1].axhline(0, color=GRAY, lw=.65, ls="--", zorder=0)
        style_axis(axes[0], t("指数点位", "Index points"))
        style_axis(axes[1], t("对数收益率（%）", "Log return (%)"))
        date_axis(axes[1])
        axes[1].set_xlabel(t("2026年 · 交易日期", "Trading date · 2026"), labelpad=3)
        self.save(fig, name)

    def original_lstm(self, rows):
        t = self.text
        fig, ax = plt.subplots(figsize=(6.7, 2.8))
        fig.subplots_adjust(left=.115, right=.985, bottom=.19, top=.82)
        d = dates(rows)
        ax.plot(d, values(rows, "previous_close"), color=GRAY, lw=1., ls="--", label=t("前日基准", "Previous close"))
        ax.plot(d, values(rows, "actual_close"), color=RED, lw=1.25, label=t("真实收盘", "Actual close"))
        ax.plot(d, values(rows, "lstm_close"), color=GREEN, lw=1.2, label=t("LSTM（32日）", "LSTM (32 days)"))
        handles, labels = ax.get_legend_handles_labels()
        ax.legend([handles[i] for i in [1, 2, 0]], [labels[i] for i in [1, 2, 0]], loc="lower center", bbox_to_anchor=(.5, 1.01), ncol=3, frameon=False)
        style_axis(ax, t("指数点位", "Index points")); date_axis(ax)
        ax.set_xlabel(t("2026年 · 交易日期", "Trading date · 2026"), labelpad=4)
        self.save(fig, "lstm.png")

    def original_model(self, model, display, rows, history, lookback):
        t = self.text
        fig, axes = plt.subplots(2, 1, figsize=(6.7, 3.5))
        fig.subplots_adjust(left=.115, right=.985, bottom=.12, top=.97, hspace=.76)
        d = dates(rows)
        hist_label = t("LSTM训练历史", "LSTM training history") if model == "timesfm" else t("首次训练历史", "Initial training history")
        axes[0].plot(history.index, history.close, color=BLUE, lw=.8)
        axes[0].plot(d, values(rows, "actual_close"), color=RED, lw=.9)
        axes[0].plot(d, values(rows, f"{model}_close"), color=GREEN, lw=.9)
        axes[0].axvline(d[0], color=GRAY, lw=.6, ls=":")
        bbox = {"facecolor": "white", "edgecolor": "none", "pad": 1}
        axes[0].text(.02, .93, hist_label, transform=axes[0].transAxes, color=BLUE, va="top", bbox=bbox)
        axes[0].text(.99, .93, t("测试", "Test"), transform=axes[0].transAxes, color="#48535D", va="top", ha="right", bbox=bbox)
        axes[0].set_yticks([3000, 3500, 4000])
        style_axis(axes[0], t("指数点位", "Index points")); date_axis(axes[0], full=True)
        ax = axes[1]
        ax.plot(d, values(rows, "previous_close"), color=GRAY, lw=.9, ls="--", label=t("前日基准", "Baseline"))
        ax.plot(d, values(rows, "actual_close"), color=RED, lw=1.15, label=t("真实收盘", "Actual"))
        ax.plot(d, values(rows, f"{model}_close"), color=GREEN, lw=1.1, label=display)
        ax.plot(d, values(rows, "lstm_close"), color=DARKGREEN, lw=1., ls=":", label=t(f"LSTM（{lookback}日）", f"LSTM ({lookback}d)"))
        handles, labels = ax.get_legend_handles_labels()
        ax.legend([handles[i] for i in [1, 2, 3, 0]], [labels[i] for i in [1, 2, 3, 0]], loc="lower center", bbox_to_anchor=(.5, 1.0), ncol=4, frameon=False, handlelength=1.6, columnspacing=.8, handletextpad=.4, borderaxespad=.3)
        style_axis(ax, t("指数点位", "Index points")); date_axis(ax)
        ax.set_xlabel(t("测试期放大 · 2026年交易日期", "Test close-up · trading date (2026)"), labelpad=3)
        self.save(fig, f"{model}.png")

    def monthly(self, groups):
        t = self.text
        fig, axes = plt.subplots(1, 2, figsize=(6.7, 3.1), sharey=True)
        fig.subplots_adjust(left=.092, right=.985, bottom=.25, top=.79, wspace=.15)
        for ax, (model, display, rows, meta, lookback) in zip(axes, groups):
            months = sorted(set(row["date"][:7] for row in rows))
            series = {model: [], "lstm": [], "baseline": []}
            counts = []
            for month in months:
                subset = [row for row in rows if row["date"].startswith(month)]
                month_meta = next(item for item in meta["monthly"] if item["month"] == month)
                if len(subset) != month_meta["count"]:
                    raise ValueError(f"{model}: monthly observation counts differ.")
                counts.append(len(subset))
                for label, field in [(model, f"{model}_close"), ("lstm", "lstm_close"), ("baseline", "previous_close")]:
                    series[label].append(self.rmse(subset, field, month_meta[label]["rmse"], f"{model}_{month}_{label}"))
            x = np.arange(len(months))
            ax.plot(x, series[model], color=GREEN, marker="o", ms=4, lw=1.2, label=display)
            ax.plot(x, series["lstm"], color=DARKGREEN, marker="s", ms=4, lw=1.2, ls=":", label=t(f"LSTM（{lookback}日）", f"LSTM ({lookback}d)"))
            ax.plot(x, series["baseline"], color=GRAY, marker="^", ms=4, lw=1., ls="--", label=t("前日基准", "Baseline"))
            ax.set_title(t(f"{display}组", f"{display} comparison"), pad=30)
            ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.), ncol=3, frameon=False, handlelength=1., columnspacing=.45, handletextpad=.3, borderaxespad=.3)
            labels = [t(f"{int(m[-2:])}月", pd.Timestamp(m + "-01").strftime("%b")) + ("*" if i in [0, 3] else "") + f"\n(n={n})" for i, (m, n) in enumerate(zip(months, counts))]
            ax.set_xticks(x, labels)
            ax.set_ylim(0, 85); ax.set_yticks([0, 20, 40, 60, 80])
            style_axis(ax, t("RMSE（点）", "RMSE (points)") if ax is axes[0] else "")
            ax.set_xlim(-.22, 3.22)
        fig.text(.5, .025, t("* 仅覆盖部分月份：6月15日起、9月11日止。", "* Partial months: from June 15, through September 11."), ha="center", color="#48535D")
        self.save(fig, "monthly.png")

    def render(self):
        t = self.text
        archive = self.read("results/reference/manifest.json")
        prices = load_prices(self.data_path)
        canonical_hash = frame_fingerprint(prices)
        if canonical_hash != archive["canonical_prices_sha256"]:
            raise ValueError("Raw dates/closes do not match the archived canonical fingerprint. Restore the matching saved data; revised provider history cannot be mixed with these predictions.")
        # Paths in the manifest are relative to the repository, including custom data paths.
        self.sources[Path(os.path.relpath(self.data_path.resolve(), REPO)).as_posix()] = hashlib.sha256(self.data_path.read_bytes()).hexdigest()
        arima_doc = self.read("results/reference/arima/selected_predictions.json")
        arima = arima_doc["rows"]
        original = self.read("results/reference/arima/original_predictions.json")
        arima_meta = self.read("results/reference/arima/comparison_results.json")
        original_meta = self.read("results/reference/arima/original_metrics.json")
        groups = []
        histories = {}
        scores = {}
        self.verify_source(prices, arima, 125, "ARIMA selected")
        self.verify_source(prices, original, 125, "ARIMA original")
        if (len(arima) != arima_meta["review_observations"]
                or [arima[0]["date"], arima[-1]["date"]] != arima_meta["review_dates"]
                or len(original) != original_meta["test_observations"]
                or original[0]["date"][:10] != original_meta["test_start"]
                or original[-1]["date"][:10] != original_meta["test_end"]):
            raise ValueError("ARIMA: archived test dates/counts differ from metrics.")
        for model, display, lookback in [("patchtst", "PatchTST", 20), ("timesfm", "TimesFM", 32)]:
            rows = self.read(f"results/reference/{model}/predictions.json")
            meta = self.read(f"results/reference/{model}/metrics.json")
            self.verify_source(prices, rows, meta["data"]["review_count"], model)
            if rows[0]["date"] != meta["data"]["review_start"] or rows[-1]["date"] != meta["data"]["review_end"]:
                raise ValueError(f"{model}: archived test range differs from metrics.")
            history = prices.loc[meta["data"]["initial_train_start"]:meta["data"]["initial_train_end"]]
            if (len(history) != rows[0]["train_observations"]
                    or history.index[0].strftime("%Y-%m-%d") != rows[0]["train_start"]
                    or history.index[-1].strftime("%Y-%m-%d") != rows[0]["train_end"]):
                raise ValueError(f"{model}: archived initial history count/dates differ.")
            validate_forecast(history, rows, [(f"{model}_close", display, GREEN, "-")])
            histories[model] = history
            groups.append((model, display, rows, meta, lookback))
            scores[model] = [self.rmse(rows, field, meta["overall"][label]["rmse"], f"{model}_{label}")
                             for field, label in [(f"{model}_close", model), ("lstm_close", "lstm"), ("previous_close", "baseline")]]
        first = arima[0]
        start_index = prices.index.get_loc(pd.Timestamp(first["trained_from"]))
        if start_index == 0:
            raise ValueError("ARIMA requires the close before the first training return.")
        arima_history = prices.loc[prices.index[start_index - 1]:first["trained_through"]]
        if len(arima_history) != first["history_count"] + 1:
            raise ValueError("ARIMA history must contain 253 closes for 252 log returns.")
        validate_forecast(arima_history, arima, [("predicted_close", "ARIMA", GREEN, "-")])
        histories["arima"] = arima_history
        arima_scores = [
            self.rmse(original, "predicted_close", original_meta["index_points"]["arima"]["rmse"], "arima_original"),
            self.rmse(arima, "predicted_close", arima_meta["review_selected"]["point_rmse"], "arima_selected"),
            self.rmse(arima, "previous_close", arima_meta["review_baseline"]["point_rmse"], "arima_baseline"),
        ]
        self.rmse(arima, "predicted_log_return", arima_meta["review_selected"]["return_rmse"], "arima_log_return", actual="actual_log_return")
        self.output.mkdir(parents=True, exist_ok=True)
        self.overview(arima_scores, scores)
        self.monthly(groups)
        times = groups[1][2]
        if self.layout == "original":
            self.arima_returns(arima, "arima.png")
            self.original_lstm(times)
            for model, display, rows, _, lookback in groups:
                self.original_model(model, display, rows, histories[model], lookback)
        else:
            baseline = ("previous_close", t("前日基准", "Previous close"), GRAY, "--")
            jobs = [
                ("arima", arima, arima_history, [("predicted_close", t("选定ARIMA", "Selected ARIMA"), GREEN, "-"), baseline], t("首次训练历史（253个收盘）", "Initial history (253 closes)")),
                ("lstm", times, histories["timesfm"], [("lstm_close", t("LSTM（32日）", "LSTM (32d)"), GREEN, "-"), baseline], t("LSTM首次训练历史", "Initial LSTM training history")),
            ]
            for model, display, rows, _, lookback in groups:
                predictions = [(f"{model}_close", display, GREEN, "-"), ("lstm_close", t(f"LSTM（{lookback}日）", f"LSTM ({lookback}d)"), DARKGREEN, ":"), baseline]
                history_label = t("历史参照（非TimesFM本地训练）", "Reference history (no local TimesFM training)") if model == "timesfm" else t("首次训练历史", "Initial training history")
                jobs.append((model, rows, histories[model], predictions, history_label))
            for model, rows, history, predictions, label in jobs:
                self.figures[f"{model}.png"] = plot_forecast(history, rows, predictions, self.output / f"{model}.png", label, language=self.language)
            self.arima_returns(arima, "arima_returns.png")
        checks = {model: {"count": len(history), "start": history.index[0].strftime("%Y-%m-%d"), "end": history.index[-1].strftime("%Y-%m-%d")}
                  for model, history in histories.items()}
        checks["lstm"] = checks["timesfm"].copy()
        manifest = {
            "method": "Saved archived one-step forecasts; no fitting, downloading, imputation, or editing reference results.",
            "layout": self.layout, "language": self.language, "font_family": self.font,
            "minimum_font_points": 9.5, "canonical_prices_sha256": canonical_hash,
            "source_sha256": self.sources, "history_checks": checks,
            "verified_rmse": self.verified, "figures": self.figures,
            "code_sha256": {relative: hashlib.sha256((REPO / relative).read_bytes()).hexdigest()
                            for relative in ["scripts/render_report_figures.py", "forecast_plotting.py"]},
        }
        if self.layout == "original":
            expected_figures = archive.get("original_pdf_figures", archive["figures"])
            comparisons = {}
            for path, expected_hash in expected_figures.items():
                name = Path(path).name
                if name in self.figures:
                    comparisons[name] = {"expected_sha256": expected_hash,
                                         "actual_sha256": self.figures[name]["sha256"],
                                         "byte_identical": self.figures[name]["sha256"] == expected_hash}
            manifest["original_pdf_reproduction"] = {
                "comparison": comparisons,
                "all_byte_identical": len(comparisons) == 6 and all(item["byte_identical"] for item in comparisons.values()),
                "note": "Same saved values and original layouts. PNG byte identity also depends on language, installed font and Matplotlib version; differing hashes alone are not a data mismatch.",
            }
        (self.output / "figures_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", choices=["standard", "original"], default="standard", help="standard: common two-panel model plots; original: six original PDF layouts")
    parser.add_argument("--output", type=Path, help="Output directory; defaults to reports/figures (standard) or runs/pdf-original-figures (original)")
    parser.add_argument("--language", choices=["zh", "en"], default="zh", help="Chinese requires an installed supported CJK font")
    parser.add_argument("--data", type=Path, default=DATA_PATH, help="Existing matching Yahoo JSON; never downloaded by this command")
    args = parser.parse_args()
    output = args.output or REPO / ("reports/figures" if args.layout == "standard" else "runs/pdf-original-figures")
    try:
        manifest = ReportFigures(output, args.layout, args.language, args.data).render()
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        parser.exit(2, f"Cannot render report figures: {exc}\n")
    print(f"Rendered {len(manifest['figures'])} figures; {len(manifest['verified_rmse'])} archived RMSE checks passed. Output: {output}")
    if "original_pdf_reproduction" in manifest:
        comparison = manifest["original_pdf_reproduction"]
        matched = sum(item["byte_identical"] for item in comparison["comparison"].values())
        print(f"Original PDF PNG byte matches: {matched}/6. Font/language/Matplotlib changes may alter pixels; archived values were checked separately.")


if __name__ == "__main__":
    main()
