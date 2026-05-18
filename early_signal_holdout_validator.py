from pathlib import Path

import numpy as np
import pandas as pd

from bar_entry_trade_simulator import ENTRY_BARS, format_table, grouped_summary, simulate_all, summarize
from early_trade_simulator import load_predictions, make_signals, summarize_signals


ARTIFACT_DIR = Path("model_artifacts")
HOLDOUT_SIGNAL_FILE = ARTIFACT_DIR / "early_holdout_trade_signals.csv"
HOLDOUT_BAR_RESULT_FILE = ARTIFACT_DIR / "early_holdout_bar_entry_results.csv"
REPORT_FILE = ARTIFACT_DIR / "early_signal_holdout_validator_report.md"


def chronological_split(df, calibration_fraction=0.50):
    ordered_dates = sorted(pd.to_datetime(df["date"]).dt.date.unique())
    split_index = max(1, int(round(len(ordered_dates) * calibration_fraction)))
    split_date = ordered_dates[split_index - 1]

    calibration = df[pd.to_datetime(df["date"]).dt.date <= split_date].copy()
    holdout = df[pd.to_datetime(df["date"]).dt.date > split_date].copy()
    return calibration, holdout, split_date


def threshold_rows(df, min_signals=15):
    rows = []
    for pattern_threshold in [0.65, 0.70, 0.75, 0.80, 0.85]:
        for edge_threshold in [0.20, 0.30, 0.40, 0.50, 0.60]:
            for direction_probability_threshold in [0.45, 0.50, 0.55, 0.60, 0.65]:
                signals = make_signals(
                    df,
                    pattern_threshold=pattern_threshold,
                    edge_threshold=edge_threshold,
                    direction_probability_threshold=direction_probability_threshold,
                )
                row = summarize_signals(signals)
                if row["signals"] < min_signals:
                    continue
                row.update(
                    {
                        "pattern_threshold": pattern_threshold,
                        "edge_threshold": edge_threshold,
                        "direction_probability_threshold": direction_probability_threshold,
                    }
                )
                rows.append(row)

    sweep = pd.DataFrame(rows)
    if sweep.empty:
        return sweep

    sweep["selection_score"] = (
        sweep["avg_trade_return_proxy"].fillna(-999)
        + sweep["hit_rate_proxy"].fillna(0) * 0.35
        + np.log1p(sweep["signals"]) * 0.04
    )
    return sweep.sort_values("selection_score", ascending=False)


def apply_thresholds(df, row):
    return make_signals(
        df,
        pattern_threshold=float(row["pattern_threshold"]),
        edge_threshold=float(row["edge_threshold"]),
        direction_probability_threshold=float(row["direction_probability_threshold"]),
    )


def evaluate_selected_thresholds(calibration, holdout, selection):
    calibration_signals = apply_thresholds(calibration, selection)
    holdout_signals = apply_thresholds(holdout, selection)
    return summarize_signals(calibration_signals), summarize_signals(holdout_signals), holdout_signals


def main():
    df = load_predictions()
    calibration, holdout, split_date = chronological_split(df)

    sweep = threshold_rows(calibration)
    if sweep.empty:
        raise SystemExit("No calibration threshold combination produced enough signals.")

    selected = sweep.iloc[0]
    calibration_summary, holdout_summary, holdout_signals = evaluate_selected_thresholds(
        calibration, holdout, selected
    )

    if holdout_signals.empty:
        raise SystemExit("Selected calibration thresholds produced no holdout signals.")

    holdout_signals = holdout_signals.sort_values("trade_return_proxy", ascending=False)
    holdout_signals.to_csv(HOLDOUT_SIGNAL_FILE, index=False)

    holdout_bar_results = simulate_all(holdout_signals)
    if not holdout_bar_results.empty:
        holdout_bar_results.to_csv(HOLDOUT_BAR_RESULT_FILE, index=False)

    bar_overall = summarize(holdout_bar_results) if not holdout_bar_results.empty else pd.Series(dtype=float)
    by_entry = grouped_summary(holdout_bar_results, "entry_bar") if not holdout_bar_results.empty else pd.DataFrame()
    by_side = grouped_summary(holdout_bar_results, "trade_side") if not holdout_bar_results.empty else pd.DataFrame()
    by_asset = grouped_summary(holdout_bar_results, "asset_class") if not holdout_bar_results.empty else pd.DataFrame()

    display_cols = [
        "ticker",
        "date",
        "asset_class",
        "trade_side",
        "trade_return_proxy",
        "avg_return",
        "early_pattern_probability",
        "early_up_probability",
        "early_down_probability",
        "early_direction_edge",
    ]

    bar_cols = [
        "ticker",
        "date",
        "asset_class",
        "trade_side",
        "entry_bar",
        "close_return_pct",
        "mfe_pct",
        "mae_pct",
        "barrier_result",
        "early_pattern_probability",
        "early_direction_edge",
    ]

    lines = [
        "# Early Signal Holdout Validator",
        "",
        "This report chooses thresholds on the earlier half of the existing walk-forward prediction rows, then evaluates those thresholds on the later half. It is still not a full retrain-by-period validation, but it is stricter than selecting thresholds on the same rows being reported.",
        "",
        "## Split",
        "",
        "```text",
        f"calibration_rows={len(calibration)}",
        f"holdout_rows={len(holdout)}",
        f"split_date={split_date}",
        "```",
        "",
        "## Selected Thresholds",
        "",
        "```text",
        selected[
            [
                "pattern_threshold",
                "edge_threshold",
                "direction_probability_threshold",
                "signals",
                "avg_trade_return_proxy",
                "median_trade_return_proxy",
                "hit_rate_proxy",
                "selection_score",
            ]
        ].to_string(),
        "```",
        "",
        "## Calibration Proxy Summary",
        "",
        "```text",
        pd.Series(calibration_summary).to_string(),
        "```",
        "",
        "## Holdout Proxy Summary",
        "",
        "```text",
        pd.Series(holdout_summary).to_string(),
        "```",
        "",
        "## Holdout Bar Entry Summary",
        "",
        "```text",
        bar_overall.to_string() if not bar_overall.empty else "No bar rows.",
        "```",
        "",
        "## Holdout Bar By Entry",
        "",
        "```text",
        format_table(by_entry),
        "```",
        "",
        "## Holdout Bar By Side",
        "",
        "```text",
        format_table(by_side),
        "```",
        "",
        "## Holdout Bar By Asset",
        "",
        "```text",
        format_table(by_asset),
        "```",
        "",
        "## Top Holdout Signals",
        "",
        "```text",
        holdout_signals[display_cols].head(30).to_string(index=False),
        "```",
        "",
        "## Worst Holdout Signals",
        "",
        "```text",
        holdout_signals[display_cols].tail(30).to_string(index=False),
        "```",
    ]

    if not holdout_bar_results.empty:
        lines.extend(
            [
                "",
                "## Worst Holdout Bar Entries",
                "",
                "```text",
                holdout_bar_results.sort_values("close_return_pct", ascending=True)[bar_cols]
                .head(30)
                .to_string(index=False),
                "```",
            ]
        )

    REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(REPORT_FILE.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
