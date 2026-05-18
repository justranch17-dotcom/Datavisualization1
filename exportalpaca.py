import argparse
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from alpaca.data.enums import DataFeed
from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from local_market_data import DATA_FOLDER, downloaded_data_path, safe_ticker_name


KEY_NAMES = ["APCA_API_KEY_ID", "ALPACA_API_KEY", "ALPACA_KEY_ID"]
SECRET_NAMES = ["APCA_API_SECRET_KEY", "ALPACA_API_SECRET", "ALPACA_SECRET_KEY"]


def parse_args():
    parser = argparse.ArgumentParser(description="Download Alpaca stock bars for the local Streamlit apps.")
    parser.add_argument("--ticker", default="AAPL", help="Ticker to download. Default: AAPL")
    parser.add_argument("--interval", default="5m", choices=["1m", "5m", "15m", "30m", "1h"], help="Bar interval. Default: 5m")
    parser.add_argument("--days", type=int, default=730, help="Calendar days to request. Default: 730")
    parser.add_argument("--chunk-days", type=int, default=30, help="Days per Alpaca request. Default: 30")
    parser.add_argument("--feed", default="sip", choices=[feed.value for feed in DataFeed], help="Alpaca stock data feed. Default: sip")
    parser.add_argument("--asset-class", default="auto", choices=["auto", "stock", "crypto"], help="Asset class. Default: auto")
    parser.add_argument("--env-file", default="alpacakeys.env", help="Env file containing Alpaca keys. Default: alpacakeys.env")
    parser.add_argument("--end-lag-days", type=int, default=1, help="Stop this many days before now. Useful for SIP restrictions. Default: 1")
    parser.add_argument("--pause-seconds", type=float, default=0.5, help="Pause between requests. Default: 0.5")
    parser.add_argument("--max-retries", type=int, default=5, help="Retries per chunk for temporary connection/API errors. Default: 5")
    parser.add_argument("--compare-fmp", action="store_true", help="Compare saved Alpaca file against the FMP file for the same ticker.")
    return parser.parse_args()


def read_env_file(path):
    values = {}
    env_path = Path(path)

    if not env_path.exists():
        return values

    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()

        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")

    return values


def get_first_value(names, file_values):
    for name in names:
        value = os.getenv(name) or file_values.get(name)

        if value:
            return value

    return None


def get_alpaca_keys(env_file):
    file_values = read_env_file(env_file)
    api_key = get_first_value(KEY_NAMES, file_values)
    secret_key = get_first_value(SECRET_NAMES, file_values)

    if not api_key or not secret_key:
        expected = ", ".join(KEY_NAMES[:2]) + " and " + ", ".join(SECRET_NAMES[:2])
        raise SystemExit(f"Missing Alpaca keys. Add them to {env_file} using names like {expected}.")

    return api_key, secret_key


def timeframe_from_interval(interval):
    if interval.endswith("m"):
        return TimeFrame(int(interval[:-1]), TimeFrameUnit.Minute)

    if interval.endswith("h"):
        return TimeFrame(int(interval[:-1]), TimeFrameUnit.Hour)

    raise ValueError(f"Unsupported interval: {interval}")


def normalize_bars_df(bars_df, ticker):
    if bars_df.empty:
        return pd.DataFrame()

    df = bars_df.reset_index()

    if "timestamp" not in df.columns:
        return pd.DataFrame()

    if "symbol" in df.columns:
        df = df[df["symbol"].astype(str).str.upper() == ticker.upper()]

    df["DateTime"] = pd.to_datetime(df["timestamp"])

    if df["DateTime"].dt.tz is None:
        df["DateTime"] = df["DateTime"].dt.tz_localize("UTC")

    df["DateTime"] = df["DateTime"].dt.tz_convert("America/New_York")
    df = df.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    )
    keep_columns = [column for column in ["DateTime", "Open", "High", "Low", "Close", "Volume"] if column in df.columns]
    df = df[keep_columns].sort_values("DateTime")
    return df


def is_crypto_symbol(ticker):
    return "/" in ticker


def download_alpaca(ticker, interval, days, chunk_days, feed, env_file, pause_seconds, asset_class="auto", end_lag_days=1, max_retries=5):
    api_key, secret_key = get_alpaca_keys(env_file)

    if asset_class == "auto":
        asset_class = "crypto" if is_crypto_symbol(ticker) else "stock"

    if asset_class == "crypto":
        client = CryptoHistoricalDataClient(api_key, secret_key)
    else:
        client = StockHistoricalDataClient(api_key, secret_key)

    timeframe = timeframe_from_interval(interval)
    ny_tz = ZoneInfo("America/New_York")
    end = datetime.now(tz=ny_tz) - timedelta(days=end_lag_days)
    start = end - timedelta(days=days)
    chunk_start = start
    chunks = []

    while chunk_start < end:
        chunk_end = min(chunk_start + timedelta(days=chunk_days), end)
        cache_path = alpaca_chunk_cache_path(ticker, interval, chunk_start, chunk_end, asset_class, feed)
        print(f"Downloading {ticker} {interval} from Alpaca: {chunk_start.date()} to {chunk_end.date()}")

        if cache_path.exists():
            chunk = pd.read_csv(cache_path)
            print("  Loaded cached chunk.")
        else:
            for attempt in range(max_retries + 1):
                try:
                    if asset_class == "crypto":
                        request = CryptoBarsRequest(
                            symbol_or_symbols=[ticker],
                            timeframe=timeframe,
                            start=chunk_start,
                            end=chunk_end,
                        )
                        bars = client.get_crypto_bars(request)
                    else:
                        request = StockBarsRequest(
                            symbol_or_symbols=[ticker],
                            timeframe=timeframe,
                            start=chunk_start,
                            end=chunk_end,
                            feed=DataFeed(feed),
                        )
                        bars = client.get_stock_bars(request)

                    break
                except Exception:
                    if attempt >= max_retries:
                        raise

                    wait_seconds = min(60, 5 * (attempt + 1))
                    print(f"  Temporary request failure. Waiting {wait_seconds} seconds, then retrying...")
                    time.sleep(wait_seconds)

            chunk = normalize_bars_df(bars.df, ticker)

            if not chunk.empty:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                chunk.to_csv(cache_path, index=False)

        if chunk.empty:
            print("  No rows returned.")
        else:
            print(f"  Rows: {len(chunk)}")
            chunks.append(chunk)

        chunk_start = chunk_end

        if pause_seconds > 0:
            time.sleep(pause_seconds)

    if not chunks:
        raise SystemExit("No Alpaca data was downloaded.")

    df = pd.concat(chunks, ignore_index=True)
    df["DateTime"] = pd.to_datetime(df["DateTime"], utc=True)
    df = df.drop_duplicates(subset=["DateTime"]).sort_values("DateTime")

    if asset_class == "crypto":
        df["DateTime"] = df["DateTime"].dt.tz_localize(None)
    else:
        df["DateTime"] = df["DateTime"].dt.tz_convert("America/New_York").dt.tz_localize(None)

    df["Ticker"] = ticker.upper()
    df["DataFeed"] = "crypto" if asset_class == "crypto" else feed

    DATA_FOLDER.mkdir(exist_ok=True)
    output_path = downloaded_data_path(ticker, interval)
    df.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")
    print(f"Rows saved: {len(df)}")
    print(f"Trading days saved: {df['DateTime'].dt.date.nunique()}")
    print(f"Date range: {df['DateTime'].min()} to {df['DateTime'].max()}")
    return output_path


def alpaca_chunk_cache_path(ticker, interval, chunk_start, chunk_end, asset_class, feed):
    feed_name = "crypto" if asset_class == "crypto" else feed
    folder = DATA_FOLDER / "_alpaca_chunks" / asset_class / feed_name / safe_ticker_name(ticker) / interval
    return folder / f"{safe_ticker_name(ticker)}_{feed_name}_{interval}_{chunk_start.date()}_{chunk_end.date()}.csv"


def compare_with_fmp(ticker, interval, alpaca_path):
    fmp_path = downloaded_data_path(ticker, interval)

    if not fmp_path.exists():
        print(f"No FMP file found to compare: {fmp_path}")
        return

    alpaca = pd.read_csv(alpaca_path)
    fmp = pd.read_csv(fmp_path)
    alpaca["DateTime"] = pd.to_datetime(alpaca["DateTime"])
    fmp["DateTime"] = pd.to_datetime(fmp["DateTime"])
    alpaca = alpaca[(alpaca["DateTime"].dt.time >= pd.Timestamp("09:30").time()) & (alpaca["DateTime"].dt.time <= pd.Timestamp("15:55").time())]
    fmp = fmp[(fmp["DateTime"].dt.time >= pd.Timestamp("09:30").time()) & (fmp["DateTime"].dt.time <= pd.Timestamp("15:55").time())]

    alpaca_counts = alpaca.groupby(alpaca["DateTime"].dt.date).size()
    fmp_counts = fmp.groupby(fmp["DateTime"].dt.date).size()
    common_times = sorted(set(alpaca["DateTime"]).intersection(set(fmp["DateTime"])))

    print("Comparison with FMP:")
    print(f"  Alpaca rows: {len(alpaca)} | days: {alpaca_counts.size}")
    print(f"  FMP rows: {len(fmp)} | days: {fmp_counts.size}")
    print(f"  Shared timestamps: {len(common_times)}")

    if common_times:
        merged = alpaca.merge(fmp, on="DateTime", suffixes=("_Alpaca", "_FMP"))
        close_diff = (merged["Close_Alpaca"] - merged["Close_FMP"]).abs()
        print(f"  Avg close difference on shared timestamps: {close_diff.mean():.4f}")
        print(f"  Max close difference on shared timestamps: {close_diff.max():.4f}")


def main():
    args = parse_args()
    output_path = download_alpaca(
        ticker=args.ticker,
        interval=args.interval,
        days=args.days,
        chunk_days=args.chunk_days,
        feed=args.feed,
        env_file=args.env_file,
        pause_seconds=args.pause_seconds,
        asset_class=args.asset_class,
        end_lag_days=args.end_lag_days,
        max_retries=args.max_retries,
    )

    if args.compare_fmp:
        compare_with_fmp(args.ticker, args.interval, output_path)


if __name__ == "__main__":
    main()
