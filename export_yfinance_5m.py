import yfinance as yf
import pandas as pd
from pathlib import Path

# ======================
# SETTINGS
# ======================

TICKERS = [
    "QQQ",
    "SPY",
    "NQ=F",
]

PERIOD = "60d"
INTERVAL = "5m"

SESSION_START = "09:30"
SESSION_END = "16:00"

START_VALUE = 100.0

OUTPUT_FOLDER = Path("powerbi_exports")
OUTPUT_FOLDER.mkdir(exist_ok=True)

OUTPUT_FILE = OUTPUT_FOLDER / "market_5m_powerbi.csv"


# ======================
# HELPERS
# ======================

def safe_ticker_name(ticker):
    return (
        ticker
        .replace("=", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace(" ", "_")
    )


def download_ticker_data(ticker):
    print(f"Downloading {ticker}...")

    df = yf.download(
        ticker,
        period=PERIOD,
        interval=INTERVAL,
        progress=False,
        auto_adjust=False
    )

    if df.empty:
        print(f"No data returned for {ticker}.")
        return pd.DataFrame()

    # Fix possible MultiIndex columns from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Convert timezone to New York time
    if df.index.tz is not None:
        df.index = df.index.tz_convert("America/New_York")
    else:
        df.index = df.index.tz_localize("UTC").tz_convert("America/New_York")

    # Keep regular trading session
    df = df.between_time(SESSION_START, SESSION_END)

    if df.empty:
        print(f"No session data found for {ticker}.")
        return pd.DataFrame()

    # Add ticker
    df["Ticker"] = ticker

    # Add datetime pieces
    df["DateTime"] = df.index
    df["Date"] = df.index.date
    df["Time"] = df.index.time

    # Add bar number inside each day
    df["BarNumber"] = df.groupby("Date").cumcount() + 1

    # Add first close of each day
    df["DayFirstClose"] = df.groupby("Date")["Close"].transform("first")

    # Add normalized values
    df["NormalizedOpen"] = START_VALUE * (df["Open"] / df["DayFirstClose"])
    df["NormalizedHigh"] = START_VALUE * (df["High"] / df["DayFirstClose"])
    df["NormalizedLow"] = START_VALUE * (df["Low"] / df["DayFirstClose"])
    df["NormalizedClose"] = START_VALUE * (df["Close"] / df["DayFirstClose"])

    # Reorder columns for Power BI
    columns = [
        "Ticker",
        "Date",
        "DateTime",
        "Time",
        "BarNumber",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "DayFirstClose",
        "NormalizedOpen",
        "NormalizedHigh",
        "NormalizedLow",
        "NormalizedClose",
    ]

    df = df[columns]

    return df


# ======================
# MAIN EXPORT
# ======================

all_data = []

for ticker in TICKERS:
    ticker_df = download_ticker_data(ticker)

    if not ticker_df.empty:
        all_data.append(ticker_df)

if not all_data:
    print("No data was downloaded.")
else:
    final_df = pd.concat(all_data, ignore_index=True)

    # Make DateTime clean for CSV / Power BI
    final_df["DateTime"] = pd.to_datetime(final_df["DateTime"]).dt.tz_localize(None)

    final_df.to_csv(OUTPUT_FILE, index=False)

    print(f"Export complete: {OUTPUT_FILE}")
    print(f"Rows exported: {len(final_df)}")
    print(f"Tickers exported: {final_df['Ticker'].unique().tolist()}")
    print(f"Date range: {final_df['Date'].min()} to {final_df['Date'].max()}")