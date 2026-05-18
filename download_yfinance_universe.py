import argparse
from pathlib import Path

import pandas as pd
import yfinance as yf

from local_market_data import downloaded_data_path, safe_ticker_name


DEFAULT_SYMBOLS = [
    "EURUSD=X",
    "GBPUSD=X",
    "USDJPY=X",
    "USDCHF=X",
    "USDCAD=X",
    "AUDUSD=X",
    "NZDUSD=X",
    "EURJPY=X",
    "GBPJPY=X",
    "ES=F",
    "NQ=F",
    "YM=F",
    "RTY=F",
    "GC=F",
    "SI=F",
    "CL=F",
    "NG=F",
    "ZB=F",
    "ZN=F",
    "6E=F",
    "6J=F",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Download extra yfinance symbols for structural learning.")
    parser.add_argument("--symbols", nargs="*", default=DEFAULT_SYMBOLS, help="Symbols to download.")
    parser.add_argument("--period", default="60d", help="YFinance period. Intraday 5m usually supports about 60 days.")
    parser.add_argument("--interval", default="5m", help="YFinance interval. Default: 5m")
    return parser.parse_args()


def normalize_download(df, symbol):
    if df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if df.index.tz is not None:
        df.index = df.index.tz_convert("America/New_York")
    else:
        df.index = df.index.tz_localize("UTC").tz_convert("America/New_York")

    df = df.rename(
        columns={
            "Open": "Open",
            "High": "High",
            "Low": "Low",
            "Close": "Close",
            "Volume": "Volume",
        }
    )
    keep = [column for column in ["Open", "High", "Low", "Close", "Volume"] if column in df.columns]
    df = df[keep].dropna(subset=["Open", "High", "Low", "Close"]).copy()
    df["DateTime"] = df.index.tz_localize(None)
    df["Ticker"] = symbol
    df["DataFeed"] = "yfinance"
    return df[["DateTime", "Open", "High", "Low", "Close", "Volume", "Ticker", "DataFeed"]]


def main():
    args = parse_args()

    for symbol in args.symbols:
        print(f"Downloading {symbol}...")
        df = yf.download(
            symbol,
            period=args.period,
            interval=args.interval,
            progress=False,
            auto_adjust=False,
        )
        normalized = normalize_download(df, symbol)

        if normalized.empty:
            print(f"  no data returned for {symbol}")
            continue

        output = downloaded_data_path(symbol, args.interval)
        output.parent.mkdir(exist_ok=True)
        normalized.to_csv(output, index=False)
        print(f"  saved {len(normalized)} rows to {output}")

    print("Done.")


if __name__ == "__main__":
    main()
