"""
Small strategy-testing helpers for the phone Streamlit app.

The first supported setup is the box strategy:
- previous regular-session day high/low define the box
- the first 15 regular-session minutes must sweep outside the box, or open outside it
- price must close back inside the box before the continuation trigger is valid
- high-side continuation is a sell; low-side continuation is a buy
"""

import pandas as pd
import plotly.graph_objects as go

from project_paths import PROJECT_ROOT

DATA_DIR = PROJECT_ROOT / "downloaded_historical_data"
SUPPORTED_TIMEFRAMES = ("5m", "1m")
REGULAR_OPEN = "09:30"
REGULAR_CLOSE = "16:00"
OPENING_WINDOW_MINUTES = 15
TARGET_BOX_FRACTION = 0.25


def list_available_tickers():
    """Return tickers that have top-level downloaded CSV files."""
    tickers = set()
    if not DATA_DIR.exists():
        return []

    for path in DATA_DIR.glob("*.csv"):
        stem = path.stem
        for timeframe in SUPPORTED_TIMEFRAMES:
            suffix = f"_{timeframe}"
            if stem.endswith(suffix):
                tickers.add(stem[: -len(suffix)])

    return sorted(tickers)


def get_data_path(ticker, timeframe):
    return DATA_DIR / f"{ticker.upper()}_{timeframe}.csv"


def load_market_data(ticker, timeframe):
    """Load and normalize a local OHLCV file."""
    path = get_data_path(ticker, timeframe)
    if not path.exists():
        raise FileNotFoundError(f"Missing data file: {path}")

    df = pd.read_csv(path)
    required = {"DateTime", "Open", "High", "Low", "Close"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {', '.join(sorted(missing))}")

    df = df.copy()
    df["DateTime"] = pd.to_datetime(df["DateTime"], errors="coerce")
    for column in ["Open", "High", "Low", "Close", "Volume"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["DateTime", "Open", "High", "Low", "Close"])
    df = df.sort_values("DateTime").reset_index(drop=True)
    df["SessionDate"] = df["DateTime"].dt.date
    return df


def regular_session_only(df):
    times = df["DateTime"].dt.strftime("%H:%M")
    return df[(times >= REGULAR_OPEN) & (times <= REGULAR_CLOSE)].copy()


def add_previous_day_box_levels(df):
    """Attach previous regular-session high/low to each intraday bar."""
    regular = regular_session_only(df)
    daily = (
        regular.groupby("SessionDate", as_index=False)
        .agg(CurrentDayHigh=("High", "max"), CurrentDayLow=("Low", "min"))
        .sort_values("SessionDate")
    )
    daily["PrevDayHigh"] = daily["CurrentDayHigh"].shift(1)
    daily["PrevDayLow"] = daily["CurrentDayLow"].shift(1)

    levels = daily[["SessionDate", "PrevDayHigh", "PrevDayLow"]]
    return df.merge(levels, on="SessionDate", how="left")


def run_box_strategy(df, include_extended_hours=False):
    """Return chart bars, qualified signals, and target-based result rows."""
    with_levels = add_previous_day_box_levels(df)
    chart_df = with_levels if include_extended_hours else regular_session_only(with_levels)
    chart_df = chart_df.dropna(subset=["PrevDayHigh", "PrevDayLow"]).copy()
    signal_df = regular_session_only(with_levels).dropna(subset=["PrevDayHigh", "PrevDayLow"]).copy()

    signal_rows = []
    result_rows = []
    for session_date, day in signal_df.groupby("SessionDate", sort=True):
        prev_high = float(day["PrevDayHigh"].iloc[0])
        prev_low = float(day["PrevDayLow"].iloc[0])
        box_range = prev_high - prev_low
        if box_range <= 0:
            continue

        first_time = day["DateTime"].iloc[0]
        opening_window_end = first_time + pd.Timedelta(minutes=OPENING_WINDOW_MINUTES)
        opening_window = day[day["DateTime"] < opening_window_end]

        setups = []
        if _opening_window_swept_high(opening_window, prev_high):
            setups.append(
                {
                    "signal": "Sell",
                    "entry": prev_high,
                    "target": prev_high - (box_range * TARGET_BOX_FRACTION),
                    "direction": "high",
                }
            )

        if _opening_window_swept_low(opening_window, prev_low):
            setups.append(
                {
                    "signal": "Buy",
                    "entry": prev_low,
                    "target": prev_low + (box_range * TARGET_BOX_FRACTION),
                    "direction": "low",
                }
            )

        for setup in setups:
            signal_row = _find_reentry_continuation_signal(day, setup, prev_high, prev_low)
            if signal_row is None:
                continue

            result = _build_result_from_target(
                session_date=session_date,
                signal=setup["signal"],
                entry_price=setup["entry"],
                target_price=setup["target"],
                day_after_entry=day[day["DateTime"] >= signal_row["DateTime"]],
            )
            signal_rows.append(
                _build_signal(
                    signal_row,
                    setup["signal"],
                    setup["entry"],
                    setup["target"],
                    result["Outcome"],
                )
            )
            result_rows.append(result)

    signals = pd.DataFrame(signal_rows)
    results = pd.DataFrame(result_rows)
    if not results.empty:
        results = results.sort_values(["SessionDate", "Signal", "EntryTime"]).reset_index(drop=True)
        results["CumulativePL"] = results["PL"].cumsum()

    return chart_df, signals, results


def summarize_results(results):
    if results.empty:
        return {
            "trades": 0,
            "wins": 0,
            "win_rate": 0.0,
            "total_pl": 0.0,
            "avg_pl": 0.0,
        }

    wins = int((results["Outcome"] == "Target hit").sum())
    trades = int(len(results))
    return {
        "trades": trades,
        "wins": wins,
        "win_rate": wins / trades,
        "total_pl": float(results["PL"].sum()),
        "avg_pl": float(results["PL"].mean()),
    }


def make_box_strategy_figure(day_df, day_signals, ticker, box_session_date=None):
    """Build a candlestick chart with previous-day box levels and signals."""
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=day_df["DateTime"],
            open=day_df["Open"],
            high=day_df["High"],
            low=day_df["Low"],
            close=day_df["Close"],
            name=ticker.upper(),
        )
    )

    level_source = day_df
    title_date = day_df["SessionDate"].iloc[0]
    if box_session_date is not None:
        selected_session_rows = day_df[day_df["SessionDate"] == box_session_date]
        if not selected_session_rows.empty:
            level_source = selected_session_rows
            title_date = box_session_date

    prev_high = float(level_source["PrevDayHigh"].iloc[0])
    prev_low = float(level_source["PrevDayLow"].iloc[0])
    x0 = day_df["DateTime"].iloc[0]
    x1 = day_df["DateTime"].iloc[-1]

    fig.add_trace(
        go.Scatter(
            x=[x0, x1],
            y=[prev_high, prev_high],
            mode="lines",
            line=dict(color="#c43b47", width=2, dash="dash"),
            name="Prev day high",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[x0, x1],
            y=[prev_low, prev_low],
            mode="lines",
            line=dict(color="#15803d", width=2, dash="dash"),
            name="Prev day low",
        )
    )

    if not day_signals.empty:
        targets = day_signals[["Signal", "Target"]].drop_duplicates()
        for _, target_row in targets.iterrows():
            color = "#15803d" if target_row["Signal"] == "Buy" else "#c43b47"
            label = "Buy target" if target_row["Signal"] == "Buy" else "Sell target"
            fig.add_trace(
                go.Scatter(
                    x=[x0, x1],
                    y=[target_row["Target"], target_row["Target"]],
                    mode="lines",
                    line=dict(color=color, width=1, dash="dot"),
                    name=label,
                )
            )

        buys = day_signals[day_signals["Signal"] == "Buy"]
        sells = day_signals[day_signals["Signal"] == "Sell"]
        if not buys.empty:
            fig.add_trace(
                go.Scatter(
                    x=buys["DateTime"],
                    y=buys["Price"],
                    mode="markers+text",
                    marker=dict(symbol="triangle-up", size=14, color="#15803d"),
                    text=buys["Outcome"],
                    textposition="bottom center",
                    name="Buy signals",
                )
            )
        if not sells.empty:
            fig.add_trace(
                go.Scatter(
                    x=sells["DateTime"],
                    y=sells["Price"],
                    mode="markers+text",
                    marker=dict(symbol="triangle-down", size=14, color="#c43b47"),
                    text=sells["Outcome"],
                    textposition="top center",
                    name="Sell signals",
                )
            )

    fig.update_layout(
        height=560,
        template="plotly_white",
        margin=dict(l=24, r=24, t=44, b=24),
        title=f"{ticker.upper()} box strategy - {title_date}",
        xaxis_title="Time",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def _opening_window_swept_high(opening_window, prev_high):
    if opening_window.empty:
        return False
    return bool((opening_window["Open"] > prev_high).any() or (opening_window["High"] > prev_high).any())


def _opening_window_swept_low(opening_window, prev_low):
    if opening_window.empty:
        return False
    return bool((opening_window["Open"] < prev_low).any() or (opening_window["Low"] < prev_low).any())


def _find_reentry_continuation_signal(day, setup, prev_high, prev_low):
    reentered_box = False

    for _, row in day.iterrows():
        close_inside_box = prev_low <= float(row["Close"]) <= prev_high
        if close_inside_box:
            reentered_box = True
            continue

        if not reentered_box:
            continue

        if setup["direction"] == "high" and float(row["High"]) > prev_high:
            return row

        if setup["direction"] == "low" and float(row["Low"]) < prev_low:
            return row

    return None


def _build_signal(row, signal, price, target, outcome):
    return {
        "SessionDate": row["SessionDate"],
        "DateTime": row["DateTime"],
        "Signal": signal,
        "Price": float(price),
        "Target": float(target),
        "Close": float(row["Close"]),
        "Outcome": outcome,
    }


def _build_result_from_target(session_date, signal, entry_price, target_price, day_after_entry):
    entry_time = day_after_entry["DateTime"].iloc[0]
    close_price = float(day_after_entry["Close"].iloc[-1])
    exit_time = day_after_entry["DateTime"].iloc[-1]
    outcome = "No target"

    if signal == "Buy":
        target_hits = day_after_entry[day_after_entry["High"] >= target_price]
        if not target_hits.empty:
            close_price = float(target_price)
            exit_time = target_hits["DateTime"].iloc[0]
            outcome = "Target hit"
    else:
        target_hits = day_after_entry[day_after_entry["Low"] <= target_price]
        if not target_hits.empty:
            close_price = float(target_price)
            exit_time = target_hits["DateTime"].iloc[0]
            outcome = "Target hit"

    if signal == "Buy":
        pl = close_price - entry_price
    else:
        pl = entry_price - close_price

    return {
        "SessionDate": session_date,
        "EntryTime": entry_time,
        "Signal": signal,
        "Entry": float(entry_price),
        "Target": float(target_price),
        "ExitTime": exit_time,
        "Exit": float(close_price),
        "PL": float(pl),
        "Outcome": outcome,
        "ExitRule": "25% box target, else same-day close",
    }
