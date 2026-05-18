from pathlib import Path

import pandas as pd


DATA_FOLDER = Path(__file__).with_name("downloaded_historical_data")


def safe_ticker_name(ticker):
    return (
        ticker.upper()
        .replace("=", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace(".", "_")
        .replace("^", "")
        .replace(" ", "_")
    )


def downloaded_data_path(ticker, interval="5m"):
    return DATA_FOLDER / f"{safe_ticker_name(ticker)}_{interval}.csv"


def list_downloaded_tickers(interval="5m"):
    if not DATA_FOLDER.exists():
        return []

    suffix = f"_{interval}.csv"
    tickers = []

    for path in DATA_FOLDER.glob(f"*{suffix}"):
        ticker = path.name[: -len(suffix)]
        if ticker and not ticker.endswith("_FMP"):
            tickers.append(ticker)

    return sorted(set(tickers))


def load_downloaded_data(ticker, interval="5m"):
    path = downloaded_data_path(ticker, interval)

    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)

    if "DateTime" not in df.columns:
        return pd.DataFrame()

    df["DateTime"] = pd.to_datetime(df["DateTime"])
    df = df.set_index("DateTime").sort_index()

    normalized_ticker = safe_ticker_name(ticker)

    if df.index.tz is None and normalized_ticker.endswith("_USD"):
        df.index = df.index.tz_localize("UTC").tz_convert("America/New_York")
    elif df.index.tz is None:
        df.index = df.index.tz_localize("America/New_York")
    else:
        df.index = df.index.tz_convert("America/New_York")

    rename_map = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    }
    df = df.rename(columns=rename_map)

    keep_columns = [column for column in ["Open", "High", "Low", "Close", "Volume"] if column in df.columns]
    return df[keep_columns]
