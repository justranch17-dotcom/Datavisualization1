from pathlib import Path

import numpy as np
import pandas as pd


ARTIFACT_DIR = Path("model_artifacts")
EARLY_PATTERN_FILE = ARTIFACT_DIR / "early_pattern_predictions.csv"
EARLY_DIRECTIONAL_FILE = ARTIFACT_DIR / "early_directional_predictions.csv"
SIGNAL_FILE = ARTIFACT_DIR / "early_trade_signals.csv"
REPORT_FILE = ARTIFACT_DIR / "early_trade_simulator_report.md"


ETF_TICKERS = {
    "DIA",
    "EEM",
    "EFA",
    "EWJ",
    "EWZ",
    "FXI",
    "GLD",
    "HYG",
    "INDA",
    "IWM",
    "MCHI",
    "QQQ",
    "SLV",
    "SMH",
    "SPY",
    "TLT",
    "VTI",
    "VOO",
    "XLC",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLU",
    "XLV",
    "XLY",
}


def asset_class(ticker):
    if ticker.endswith("_USD"):
        return "crypto"
    if ticker.endswith("_F"):
        return "futures"
    if ticker.endswith("_X"):
        return "forex"
    if ticker in ETF_TICKERS:
        return "etf"
    return "stock"


def load_predictions():
    if not EARLY_PATTERN_FILE.exists():
        raise SystemExit("Run early_pattern_predictor.py first.")
    if not EARLY_DIRECTIONAL_FILE.exists():
        raise SystemExit("Run early_directional_predictor.py first.")

    pattern = pd.read_csv(EARLY_PATTERN_FILE)
    directional = pd.read_csv(EARLY_DIRECTIONAL_FILE)[
        [
            "ticker",
            "date",
            "early_up_probability",
            "early_down_probability",
            "early_direction_edge",
            "early_direction_signal",
        ]
    ]
    merged = pattern.merge(directional, on=["ticker", "date"], how="inner")
    merged["asset_class"] = merged["ticker"].map(asset_class)
    return merged


def make_signals(df, pattern_threshold, edge_threshold, direction_probability_threshold):
    signals = df[df["early_pattern_probability"] >= pattern_threshold].copy()
    signals = signals[signals["early_direction_edge"].abs() >= edge_threshold]

    long_mask = (
        (signals["early_direction_edge"] > 0)
        & (signals["early_up_probability"] >= direction_probability_threshold)
    )
    short_mask = (
        (signals["early_direction_edge"] < 0)
        & (signals["early_down_probability"] >= direction_probability_threshold)
    )

    signals["trade_side"] = np.select([long_mask, short_mask], ["long", "short"], default="none")
    signals = signals[signals["trade_side"] != "none"].copy()
    signals["trade_return_proxy"] = np.where(
        signals["trade_side"] == "long",
        signals["avg_return"],
        -signals["avg_return"],
    )
    signals["trade_won_proxy"] = signals["trade_return_proxy"] > 0
    signals["abs_return"] = signals["avg_return"].abs()
    signals["pattern_threshold"] = pattern_threshold
    signals["edge_threshold"] = edge_threshold
    signals["direction_probability_threshold"] = direction_probability_threshold
    return signals


def summarize_signals(signals):
    if signals.empty:
        return {
            "signals": 0,
            "avg_trade_return_proxy": np.nan,
            "median_trade_return_proxy": np.nan,
            "hit_rate_proxy": np.nan,
            "avg_abs_return": np.nan,
            "long_signals": 0,
            "short_signals": 0,
        }

    return {
        "signals": len(signals),
        "avg_trade_return_proxy": signals["trade_return_proxy"].mean(),
        "median_trade_return_proxy": signals["trade_return_proxy"].median(),
        "hit_rate_proxy": signals["trade_won_proxy"].mean(),
        "avg_abs_return": signals["abs_return"].mean(),
        "long_signals": int((signals["trade_side"] == "long").sum()),
        "short_signals": int((signals["trade_side"] == "short").sum()),
    }


def build_threshold_sweep(df):
    rows = []
    for pattern_threshold in [0.70, 0.75, 0.80, 0.85]:
        for edge_threshold in [0.20, 0.30, 0.40, 0.50]:
            for direction_probability_threshold in [0.45, 0.50, 0.55, 0.60]:
                signals = make_signals(
                    df,
                    pattern_threshold=pattern_threshold,
                    edge_threshold=edge_threshold,
                    direction_probability_threshold=direction_probability_threshold,
                )
                row = summarize_signals(signals)
                row.update(
                    {
                        "pattern_threshold": pattern_threshold,
                        "edge_threshold": edge_threshold,
                        "direction_probability_threshold": direction_probability_threshold,
                    }
                )
                rows.append(row)

    sweep = pd.DataFrame(rows)
    sweep = sweep[sweep["signals"] >= 25].copy()
    if sweep.empty:
        return sweep

    sweep["score"] = (
        sweep["avg_trade_return_proxy"].fillna(-999)
        + sweep["hit_rate_proxy"].fillna(0) * 0.50
        + np.log1p(sweep["signals"]) * 0.03
    )
    return sweep.sort_values("score", ascending=False)


def side_summary(signals):
    if signals.empty:
        return pd.DataFrame()
    return (
        signals.groupby("trade_side", as_index=False)
        .agg(
            signals=("ticker", "size"),
            avg_trade_return_proxy=("trade_return_proxy", "mean"),
            median_trade_return_proxy=("trade_return_proxy", "median"),
            hit_rate_proxy=("trade_won_proxy", "mean"),
            avg_abs_return=("abs_return", "mean"),
        )
        .sort_values("avg_trade_return_proxy", ascending=False)
    )


def asset_summary(signals):
    if signals.empty:
        return pd.DataFrame()
    return (
        signals.groupby("asset_class", as_index=False)
        .agg(
            signals=("ticker", "size"),
            avg_trade_return_proxy=("trade_return_proxy", "mean"),
            median_trade_return_proxy=("trade_return_proxy", "median"),
            hit_rate_proxy=("trade_won_proxy", "mean"),
            avg_abs_return=("abs_return", "mean"),
        )
        .sort_values("avg_trade_return_proxy", ascending=False)
    )


def main():
    df = load_predictions()
    sweep = build_threshold_sweep(df)
    if sweep.empty:
        raise SystemExit("No threshold combination produced at least 25 signals.")

    best = sweep.iloc[0]
    signals = make_signals(
        df,
        pattern_threshold=float(best["pattern_threshold"]),
        edge_threshold=float(best["edge_threshold"]),
        direction_probability_threshold=float(best["direction_probability_threshold"]),
    ).sort_values("trade_return_proxy", ascending=False)
    signals.to_csv(SIGNAL_FILE, index=False)

    baseline = df.copy()
    baseline["signed_early_move_return_proxy"] = np.where(
        baseline["avg_early_move"] >= 0,
        baseline["avg_return"],
        -baseline["avg_return"],
    )

    top_columns = [
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
        "useful_pattern_ensemble_score",
    ]

    lines = [
        "# Early Trade Simulator",
        "",
        "This is a signed-return proxy, not an execution backtest. It uses full-day `avg_return` as the outcome after choosing a side from early signals.",
        "",
        "The threshold sweep is a discovery pass selected on the same walk-forward test rows. Treat the best threshold set as a candidate for the next real bar-entry simulation, not as validated trading performance.",
        "",
        "## Best Threshold Set",
        "",
        "```text",
        best[
            [
                "pattern_threshold",
                "edge_threshold",
                "direction_probability_threshold",
                "signals",
                "avg_trade_return_proxy",
                "median_trade_return_proxy",
                "hit_rate_proxy",
                "avg_abs_return",
                "long_signals",
                "short_signals",
            ]
        ].to_string(),
        "```",
        "",
        "## Baseline",
        "",
        "```text",
        f"all_test_rows={len(df)}",
        f"avg_abs_return={df['avg_return'].abs().mean():.4f}",
        f"early-move side avg return proxy={baseline['signed_early_move_return_proxy'].mean():.4f}",
        f"early-move side hit rate proxy={(baseline['signed_early_move_return_proxy'] > 0).mean():.4f}",
        "```",
        "",
        "## Top Threshold Sweeps",
        "",
        "```text",
        sweep.head(20).to_string(index=False),
        "```",
        "",
        "## Side Summary",
        "",
        "```text",
        side_summary(signals).to_string(index=False),
        "```",
        "",
        "## Asset Class Summary",
        "",
        "```text",
        asset_summary(signals).to_string(index=False),
        "```",
        "",
        "## Best Proxy Trades",
        "",
        "```text",
        signals[top_columns].head(30).to_string(index=False),
        "```",
        "",
        "## Worst Proxy Trades",
        "",
        "```text",
        signals[top_columns].tail(30).to_string(index=False),
        "```",
    ]

    REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(REPORT_FILE.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
