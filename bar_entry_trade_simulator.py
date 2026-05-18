from pathlib import Path

import numpy as np
import pandas as pd

from local_market_data import load_downloaded_data


ARTIFACT_DIR = Path("model_artifacts")
SIGNAL_FILE = ARTIFACT_DIR / "early_trade_signals.csv"
RESULT_FILE = ARTIFACT_DIR / "bar_entry_trade_results.csv"
REPORT_FILE = ARTIFACT_DIR / "bar_entry_trade_simulator_report.md"

SESSION_START = "09:30"
SESSION_END = "16:00"
ENTRY_BARS = [20, 25, 28]
INTERVAL = "5m"


def signed_return(side, entry_price, exit_price):
    if entry_price == 0:
        return np.nan
    raw_return = (exit_price - entry_price) / entry_price * 100.0
    return raw_return if side == "long" else -raw_return


def side_high_low_returns(side, entry_price, highs, lows):
    if entry_price == 0:
        return np.nan, np.nan

    if side == "long":
        mfe = (highs.max() - entry_price) / entry_price * 100.0
        mae = (lows.min() - entry_price) / entry_price * 100.0
    else:
        mfe = (entry_price - lows.min()) / entry_price * 100.0
        mae = (entry_price - highs.max()) / entry_price * 100.0
    return float(mfe), float(mae)


def first_barrier_hit(side, future, entry_price, target_pct, stop_pct):
    target_price = entry_price * (1 + target_pct / 100.0) if side == "long" else entry_price * (1 - target_pct / 100.0)
    stop_price = entry_price * (1 - stop_pct / 100.0) if side == "long" else entry_price * (1 + stop_pct / 100.0)

    for offset, (_, row) in enumerate(future.iterrows(), start=1):
        high = float(row["High"])
        low = float(row["Low"])

        if side == "long":
            hit_target = high >= target_price
            hit_stop = low <= stop_price
        else:
            hit_target = low <= target_price
            hit_stop = high >= stop_price

        if hit_target and hit_stop:
            return "both_same_bar", offset
        if hit_target:
            return "target", offset
        if hit_stop:
            return "stop", offset

    return "neither", np.nan


def session_for_signal(ticker, date):
    df = load_downloaded_data(ticker, INTERVAL)
    if df.empty:
        return pd.DataFrame()

    session = df.between_time(SESSION_START, SESSION_END)
    if session.empty:
        return pd.DataFrame()

    target_date = pd.to_datetime(date).date()
    return session[session.index.date == target_date].copy()


def simulate_signal(signal, entry_bar):
    day = session_for_signal(signal["ticker"], signal["date"])
    if day.empty or len(day) <= entry_bar:
        return None

    side = signal["trade_side"]
    entry = day.iloc[entry_bar]
    future = day.iloc[entry_bar + 1 :]
    if future.empty:
        return None

    entry_price = float(entry["Close"])
    close_price = float(day["Close"].iloc[-1])
    close_return = signed_return(side, entry_price, close_price)

    early_window = day.iloc[: entry_bar + 1]
    early_range_pct = (
        (float(early_window["High"].max()) - float(early_window["Low"].min())) / entry_price * 100.0
        if entry_price
        else np.nan
    )
    target_pct = max(0.25, early_range_pct * 0.50)
    stop_pct = max(0.25, early_range_pct * 0.50)

    mfe_pct, mae_pct = side_high_low_returns(
        side,
        entry_price,
        future["High"].astype(float),
        future["Low"].astype(float),
    )
    barrier_result, barrier_bars = first_barrier_hit(side, future, entry_price, target_pct, stop_pct)

    return {
        "ticker": signal["ticker"],
        "date": signal["date"],
        "asset_class": signal.get("asset_class", ""),
        "trade_side": side,
        "entry_bar": entry_bar,
        "entry_time": entry.name,
        "bars_in_session": len(day),
        "bars_after_entry": len(future),
        "entry_price": entry_price,
        "close_price": close_price,
        "close_return_pct": close_return,
        "won_to_close": close_return > 0,
        "mfe_pct": mfe_pct,
        "mae_pct": mae_pct,
        "early_range_pct": early_range_pct,
        "target_pct": target_pct,
        "stop_pct": stop_pct,
        "barrier_result": barrier_result,
        "barrier_bars": barrier_bars,
        "early_pattern_probability": signal["early_pattern_probability"],
        "early_up_probability": signal["early_up_probability"],
        "early_down_probability": signal["early_down_probability"],
        "early_direction_edge": signal["early_direction_edge"],
        "useful_pattern_ensemble_score": signal["useful_pattern_ensemble_score"],
        "full_day_avg_return_pct": signal["avg_return"],
        "proxy_trade_return_pct": signal["trade_return_proxy"],
    }


def simulate_all(signals):
    rows = []
    for _, signal in signals.iterrows():
        for entry_bar in ENTRY_BARS:
            row = simulate_signal(signal, entry_bar)
            if row is not None:
                rows.append(row)
    return pd.DataFrame(rows)


def summarize(group):
    if group.empty:
        return pd.Series(dtype=float)

    target_hits = group["barrier_result"].isin(["target", "both_same_bar"]).mean()
    stop_hits = group["barrier_result"].isin(["stop", "both_same_bar"]).mean()
    return pd.Series(
        {
            "trades": len(group),
            "avg_close_return_pct": group["close_return_pct"].mean(),
            "median_close_return_pct": group["close_return_pct"].median(),
            "hit_rate_to_close": group["won_to_close"].mean(),
            "avg_mfe_pct": group["mfe_pct"].mean(),
            "avg_mae_pct": group["mae_pct"].mean(),
            "target_hit_rate": target_hits,
            "stop_hit_rate": stop_hits,
            "avg_barrier_bars": group["barrier_bars"].mean(),
        }
    )


def grouped_summary(results, by):
    if results.empty:
        return pd.DataFrame()
    return results.groupby(by).apply(summarize, include_groups=False).reset_index()


def format_table(df):
    if df.empty:
        return "No rows."
    return df.to_string(index=False)


def main():
    if not SIGNAL_FILE.exists():
        raise SystemExit("Run early_trade_simulator.py first.")

    signals = pd.read_csv(SIGNAL_FILE)
    results = simulate_all(signals)
    if results.empty:
        raise SystemExit("No bar-entry results could be simulated.")

    results = results.sort_values(["entry_bar", "close_return_pct"], ascending=[True, False])
    results.to_csv(RESULT_FILE, index=False)

    overall = summarize(results)
    by_entry = grouped_summary(results, "entry_bar")
    by_side = grouped_summary(results, "trade_side")
    by_asset = grouped_summary(results, "asset_class")
    by_entry_side = grouped_summary(results, ["entry_bar", "trade_side"])

    best_rows = results.sort_values("close_return_pct", ascending=False).head(25)
    worst_rows = results.sort_values("close_return_pct", ascending=True).head(25)
    display_cols = [
        "ticker",
        "date",
        "asset_class",
        "trade_side",
        "entry_bar",
        "entry_time",
        "close_return_pct",
        "mfe_pct",
        "mae_pct",
        "barrier_result",
        "barrier_bars",
        "early_pattern_probability",
        "early_direction_edge",
    ]

    lines = [
        "# Bar Entry Trade Simulator",
        "",
        "This is the first raw-OHLCV check after the early signal files. It uses the already-selected early trade signals and enters at fixed 5m bar indexes inside the regular session.",
        "",
        "Important: this still inherits the discovery-selected signal threshold from `early_trade_simulator.py`. It is more realistic than the full-day proxy, but it is not yet an out-of-sample strategy validation.",
        "",
        "## Settings",
        "",
        "```text",
        f"signals={len(signals)}",
        f"entry_bars={ENTRY_BARS}",
        f"session={SESSION_START}-{SESSION_END}",
        "target_pct=max(0.25, 0.50 * early_range_pct)",
        "stop_pct=max(0.25, 0.50 * early_range_pct)",
        "```",
        "",
        "## Overall",
        "",
        "```text",
        overall.to_string(),
        "```",
        "",
        "## By Entry Bar",
        "",
        "```text",
        format_table(by_entry),
        "```",
        "",
        "## By Side",
        "",
        "```text",
        format_table(by_side),
        "```",
        "",
        "## By Entry Bar And Side",
        "",
        "```text",
        format_table(by_entry_side),
        "```",
        "",
        "## By Asset Class",
        "",
        "```text",
        format_table(by_asset),
        "```",
        "",
        "## Best Bar Entries",
        "",
        "```text",
        best_rows[display_cols].to_string(index=False),
        "```",
        "",
        "## Worst Bar Entries",
        "",
        "```text",
        worst_rows[display_cols].to_string(index=False),
        "```",
    ]

    REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(REPORT_FILE.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
