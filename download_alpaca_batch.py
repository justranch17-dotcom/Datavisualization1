import argparse
import time
from pathlib import Path

import pandas as pd

from exportalpaca import download_alpaca
from local_market_data import downloaded_data_path, safe_ticker_name


REPORT_FILE = Path(__file__).with_name("downloaded_historical_data") / "download_quality_report.csv"


def parse_args():
    parser = argparse.ArgumentParser(description="Download many Alpaca tickers with pauses and quality checks.")
    parser.add_argument("--ticker-file", default="Nexttickers.txt", help="Ticker file, one ticker per line.")
    parser.add_argument("--interval", default="5m", choices=["1m", "5m", "15m", "30m", "1h"])
    parser.add_argument("--days", type=int, default=730)
    parser.add_argument("--chunk-days", type=int, default=30)
    parser.add_argument("--feed", default="sip")
    parser.add_argument("--env-file", default="alpacakeys.env")
    parser.add_argument("--end-lag-days", type=int, default=1)
    parser.add_argument("--pause-seconds", type=float, default=1.0, help="Pause between API chunks.")
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--ticker-pause-seconds", type=float, default=8.0, help="Pause after each ticker.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip tickers that already have saved data.")
    parser.add_argument("--max-tickers", type=int, default=None, help="Limit ticker count for testing.")
    return parser.parse_args()


def read_tickers(path):
    tickers = []
    seen = set()

    for raw_line in Path(path).read_text().splitlines():
        line = raw_line.strip()

        if not line or line.startswith("#"):
            continue

        ticker = line.split("#", 1)[0].strip()
        key = ticker.upper()

        if key in seen:
            continue

        seen.add(key)
        tickers.append(ticker)

    return tickers


def quality_check(ticker, interval):
    path = downloaded_data_path(ticker, interval)

    if not path.exists():
        return {
            "Ticker": ticker,
            "Interval": interval,
            "Status": "missing_file",
            "Rows": 0,
            "Days": 0,
            "Partial Days": 0,
            "Duplicate Timestamps": 0,
            "Null OHLC Rows": 0,
            "Flat Close Streak Max": 0,
            "First Timestamp": "",
            "Last Timestamp": "",
        }

    df = pd.read_csv(path)

    if df.empty or "DateTime" not in df.columns:
        return {
            "Ticker": ticker,
            "Interval": interval,
            "Status": "empty_or_bad_file",
            "Rows": len(df),
            "Days": 0,
            "Partial Days": 0,
            "Duplicate Timestamps": 0,
            "Null OHLC Rows": 0,
            "Flat Close Streak Max": 0,
            "First Timestamp": "",
            "Last Timestamp": "",
        }

    df["DateTime"] = pd.to_datetime(df["DateTime"])
    data_feed = str(df["DataFeed"].dropna().iloc[0]) if "DataFeed" in df.columns and df["DataFeed"].notna().any() else ""
    duplicate_timestamps = int(df["DateTime"].duplicated().sum())
    null_ohlc_rows = int(df[[column for column in ["Open", "High", "Low", "Close"] if column in df.columns]].isna().any(axis=1).sum())
    close_changed = df["Close"].ne(df["Close"].shift()).astype(int) if "Close" in df.columns else pd.Series(dtype=int)
    streak_ids = close_changed.cumsum() if not close_changed.empty else pd.Series(dtype=int)
    flat_close_streak = int(df.groupby(streak_ids).size().max()) if not streak_ids.empty else 0

    if "/" in ticker:
        day_counts = df.groupby(df["DateTime"].dt.date).size()
        partial_days = 0
    else:
        session_end = pd.Timestamp("15:59").time() if interval == "1m" else pd.Timestamp("15:55").time()
        session = df[
            (df["DateTime"].dt.time >= pd.Timestamp("09:30").time())
            & (df["DateTime"].dt.time <= session_end)
        ]
        day_counts = session.groupby(session["DateTime"].dt.date).size()
        expected_rows = {"1m": 390, "5m": 78, "15m": 26, "30m": 13, "1h": 7}.get(interval)
        partial_days = int((day_counts < expected_rows).sum()) if expected_rows else 0

    status = "ok"

    if duplicate_timestamps:
        status = "duplicate_timestamps"
    elif null_ohlc_rows:
        status = "null_ohlc"
    elif flat_close_streak >= 25:
        status = "possible_jammed_flat_close"
    elif partial_days > max(6, int(max(1, len(day_counts)) * 0.05)) and "/" not in ticker:
        status = "many_partial_days"

    return {
        "Ticker": safe_ticker_name(ticker),
        "Interval": interval,
        "Status": status,
        "Rows": int(len(df)),
        "Days": int(day_counts.size),
        "Partial Days": partial_days,
        "Duplicate Timestamps": duplicate_timestamps,
        "Null OHLC Rows": null_ohlc_rows,
        "Flat Close Streak Max": flat_close_streak,
        "Data Feed": data_feed,
        "First Timestamp": str(df["DateTime"].min()),
        "Last Timestamp": str(df["DateTime"].max()),
    }


def write_report_row(row):
    REPORT_FILE.parent.mkdir(exist_ok=True)

    if REPORT_FILE.exists():
        report = pd.read_csv(REPORT_FILE)
        report = report[
            ~(
                (report["Ticker"].astype(str) == str(row["Ticker"]))
                & (report["Interval"].astype(str) == str(row["Interval"]))
            )
        ]
        report = pd.concat([report, pd.DataFrame([row])], ignore_index=True)
    else:
        report = pd.DataFrame([row])

    report = report.sort_values(["Interval", "Ticker"])
    report.to_csv(REPORT_FILE, index=False)


def main():
    args = parse_args()
    tickers = read_tickers(args.ticker_file)

    if args.max_tickers:
        tickers = tickers[: args.max_tickers]

    print(f"Tickers queued: {len(tickers)}")

    for index, ticker in enumerate(tickers, start=1):
        path = downloaded_data_path(ticker, args.interval)

        if args.skip_existing and path.exists():
            print(f"[{index}/{len(tickers)}] Skipping existing {ticker}: {path}")
            row = quality_check(ticker, args.interval)
            write_report_row(row)
            continue

        print(f"[{index}/{len(tickers)}] Starting {ticker}")

        try:
            download_alpaca(
                ticker=ticker,
                interval=args.interval,
                days=args.days,
                chunk_days=args.chunk_days,
                feed=args.feed,
                env_file=args.env_file,
                pause_seconds=args.pause_seconds,
                end_lag_days=args.end_lag_days,
                max_retries=args.max_retries,
            )
            row = quality_check(ticker, args.interval)
            write_report_row(row)
            print(f"Quality: {row['Status']} | rows={row['Rows']} | days={row['Days']} | partial_days={row['Partial Days']}")
        except Exception as exc:
            row = {
                "Ticker": safe_ticker_name(ticker),
                "Interval": args.interval,
                "Status": f"failed: {type(exc).__name__}: {exc}",
                "Rows": 0,
                "Days": 0,
                "Partial Days": 0,
                "Duplicate Timestamps": 0,
                "Null OHLC Rows": 0,
                "Flat Close Streak Max": 0,
                "Data Feed": args.feed,
                "First Timestamp": "",
                "Last Timestamp": "",
            }
            write_report_row(row)
            print(row["Status"])

        if index < len(tickers) and args.ticker_pause_seconds > 0:
            print(f"Waiting {args.ticker_pause_seconds} seconds before next ticker...")
            time.sleep(args.ticker_pause_seconds)

    print(f"Batch complete. Report: {REPORT_FILE}")


if __name__ == "__main__":
    main()
