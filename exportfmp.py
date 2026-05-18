import argparse
import os
import time
from datetime import timedelta
from getpass import getpass

import pandas as pd
import requests

from local_market_data import DATA_FOLDER, downloaded_data_path


FMP_INTERVALS = {
    "1m": "1min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1hour",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Download FMP intraday data in chunks for the Streamlit apps.")
    parser.add_argument("--ticker", default="AAPL", help="Ticker to download. Default: AAPL")
    parser.add_argument("--interval", default="5m", choices=sorted(FMP_INTERVALS), help="Bar interval. Default: 5m")
    parser.add_argument("--days", type=int, default=160, help="Total calendar days to request. Default: 160")
    parser.add_argument("--chunk-days", type=int, default=7, help="Days per FMP request. Default: 7")
    parser.add_argument("--pause-seconds", type=float, default=0.5, help="Pause between requests. Default: 0.5")
    parser.add_argument("--max-retries", type=int, default=8, help="Retries for rate limits or temporary errors. Default: 8")
    parser.add_argument("--api-key", default=os.getenv("FMP_API_KEY"), help="FMP API key. Can also use FMP_API_KEY env var.")
    return parser.parse_args()


def get_api_key(api_key):
    if api_key:
        return api_key

    return getpass("FMP API key: ").strip()


def download_chunk(ticker, interval, start_date, end_date, api_key, max_retries):
    url = f"https://financialmodelingprep.com/stable/historical-chart/{FMP_INTERVALS[interval]}"
    params = {
        "symbol": ticker,
        "from": start_date.isoformat(),
        "to": end_date.isoformat(),
        "apikey": api_key,
    }

    for attempt in range(max_retries + 1):
        response = requests.get(url, params=params, timeout=45)

        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            wait_seconds = int(retry_after) if retry_after and retry_after.isdigit() else min(90, 10 * (attempt + 1))
            print(f"  Rate limited. Waiting {wait_seconds} seconds, then retrying...")
            time.sleep(wait_seconds)
            continue

        if response.status_code >= 500:
            wait_seconds = min(60, 5 * (attempt + 1))
            print(f"  FMP server error {response.status_code}. Waiting {wait_seconds} seconds, then retrying...")
            time.sleep(wait_seconds)
            continue

        response.raise_for_status()
        break
    else:
        response.raise_for_status()

    data = response.json()

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)

    if "date" not in df.columns:
        return pd.DataFrame()

    df["DateTime"] = pd.to_datetime(df["date"])
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
    return df[keep_columns]


def chunk_cache_path(ticker, interval, start_date, end_date):
    cache_folder = DATA_FOLDER / "_chunks" / ticker.replace("^", "CARET_").replace("/", "_")
    cache_folder.mkdir(parents=True, exist_ok=True)
    return cache_folder / f"{ticker.replace('^', 'CARET_')}_{interval}_{start_date}_{end_date}.csv"


def download_ticker(ticker, interval, days, chunk_days, pause_seconds, max_retries, api_key):
    ticker = ticker.upper().strip()
    today = pd.Timestamp.today(tz="America/New_York").date()
    start_date = today - timedelta(days=days)
    chunks = []
    chunk_start = start_date

    while chunk_start < today:
        chunk_end = min(chunk_start + timedelta(days=chunk_days - 1), today)
        cache_path = chunk_cache_path(ticker, interval, chunk_start, chunk_end)

        if cache_path.exists():
            print(f"Using cached {ticker} {interval}: {chunk_start} to {chunk_end}")
            chunk = pd.read_csv(cache_path)
        else:
            print(f"Downloading {ticker} {interval}: {chunk_start} to {chunk_end}")
            chunk = download_chunk(ticker, interval, chunk_start, chunk_end, api_key, max_retries)

        if chunk.empty:
            print("  No rows returned.")
        else:
            print(f"  Rows: {len(chunk)}")
            chunks.append(chunk)
            chunk.to_csv(cache_path, index=False)

        chunk_start = chunk_end + timedelta(days=1)

        if pause_seconds > 0:
            time.sleep(pause_seconds)

    if not chunks:
        print("No data was downloaded.")
        return None

    df = pd.concat(chunks, ignore_index=True)
    df["DateTime"] = pd.to_datetime(df["DateTime"])
    df = df.drop_duplicates(subset=["DateTime"]).sort_values("DateTime")
    df["Ticker"] = ticker
    df = df[df["DateTime"].dt.date >= start_date]

    DATA_FOLDER.mkdir(exist_ok=True)
    output_path = downloaded_data_path(ticker, interval)
    df.to_csv(output_path, index=False)

    rows_by_day = df.groupby(df["DateTime"].dt.date).size()
    expected_rows = {
        "1m": 390,
        "5m": 78,
        "15m": 26,
        "30m": 13,
        "1h": 7,
    }.get(interval)
    partial_days = rows_by_day[rows_by_day < expected_rows] if expected_rows else pd.Series(dtype=int)

    print(f"Saved: {output_path}")
    print(f"Rows saved: {len(df)}")
    print(f"Trading days saved: {rows_by_day.size}")
    print(f"Date range: {df['DateTime'].min()} to {df['DateTime'].max()}")

    if expected_rows:
        print(f"Expected regular-session rows per full day: {expected_rows}")

    if not partial_days.empty:
        print("Partial days found:")
        for day, row_count in partial_days.items():
            print(f"  {day}: {row_count} rows")

    return output_path


def main():
    args = parse_args()
    api_key = get_api_key(args.api_key)

    if not api_key:
        raise SystemExit("No FMP API key provided.")

    download_ticker(
        ticker=args.ticker,
        interval=args.interval,
        days=args.days,
        chunk_days=args.chunk_days,
        pause_seconds=args.pause_seconds,
        max_retries=args.max_retries,
        api_key=api_key,
    )


if __name__ == "__main__":
    main()
