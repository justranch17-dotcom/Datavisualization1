import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests
import numpy as np
from local_market_data import list_downloaded_tickers, load_downloaded_data
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.volatility import BollingerBands
from ta.volume import VolumeWeightedAveragePrice
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Advanced Market Pattern Analyzer", layout="wide", page_icon="📈")

st.title("📈 Advanced Market Pattern Analyzer")
st.markdown("*Find patterns in high-volatility, high-liquidity markets*")


# ======================
# TECHNICAL ANALYSIS HELPERS
# ======================

def add_technical_indicators(df):
    """Add technical indicators to the dataframe"""
    if df.empty:
        return df

    # RSI
    rsi_indicator = RSIIndicator(close=df['Close'], window=14)
    df['RSI'] = rsi_indicator.rsi()

    # MACD
    macd_indicator = MACD(close=df['Close'])
    df['MACD'] = macd_indicator.macd()
    df['MACD_Signal'] = macd_indicator.macd_signal()
    df['MACD_Hist'] = macd_indicator.macd_diff()

    # Bollinger Bands
    bb_indicator = BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['BB_Upper'] = bb_indicator.bollinger_hband()
    df['BB_Lower'] = bb_indicator.bollinger_lband()
    df['BB_Middle'] = bb_indicator.bollinger_mavg()

    # Moving Averages
    sma_20 = SMAIndicator(close=df['Close'], window=20)
    sma_50 = SMAIndicator(close=df['Close'], window=50)
    ema_12 = EMAIndicator(close=df['Close'], window=12)
    ema_26 = EMAIndicator(close=df['Close'], window=26)

    df['SMA_20'] = sma_20.sma_indicator()
    df['SMA_50'] = sma_50.sma_indicator()
    df['EMA_12'] = ema_12.ema_indicator()
    df['EMA_26'] = ema_26.ema_indicator()

    # Volume indicators
    df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()

    # Volatility (ATR-like)
    df['TR'] = np.maximum(
        df['High'] - df['Low'],
        np.maximum(
            abs(df['High'] - df['Close'].shift(1)),
            abs(df['Low'] - df['Close'].shift(1))
        )
    )
    df['ATR'] = df['TR'].rolling(window=14).mean()

    # Price change percentages
    df['Price_Change_Pct'] = df['Close'].pct_change() * 100
    df['Volume_Change_Pct'] = df['Volume'].pct_change() * 100

    return df

def calculate_pattern_similarity(day1_data, day2_data, weights=None):
    """Calculate similarity between two days' price patterns"""
    if weights is None:
        weights = {'close': 0.7, 'high': 0.15, 'low': 0.15}

    try:
        # Normalize both days to same length by interpolating
        len1, len2 = len(day1_data['close']), len(day2_data['close'])
        max_len = max(len1, len2)

        if len1 != len2:
            # Interpolate shorter series to match longer one
            x_long = np.linspace(0, 1, max_len)
            if len1 < len2:
                x_short = np.linspace(0, 1, len1)
                day1_interp = {}
                for key in ['close', 'high', 'low']:
                    day1_interp[key] = np.interp(x_long, x_short, day1_data[key])
                day1_data = day1_interp
            else:
                x_short = np.linspace(0, 1, len2)
                day2_interp = {}
                for key in ['close', 'high', 'low']:
                    day2_interp[key] = np.interp(x_long, x_short, day2_data[key])
                day2_data = day2_interp

        # Calculate weighted Euclidean distance
        distance = 0
        total_weight = 0
        for key, weight in weights.items():
            if key in day1_data and key in day2_data:
                dist = euclidean(day1_data[key], day2_data[key])
                distance += weight * dist
                total_weight += weight

        if total_weight > 0:
            distance /= total_weight

        # Convert distance to similarity score (0-100)
        similarity = max(0, 100 - distance * 10)
        return similarity
    except:
        return 0

def cluster_similar_days(matching_days, n_clusters=3):
    """Cluster similar days using K-means on normalized price data"""
    if len(matching_days) < n_clusters:
        return {f"Cluster_{i}": [day] for i, day in enumerate(matching_days)}

    # Extract normalized close prices as features
    features = []
    valid_days = []

    for day in matching_days:
        close_prices = day['normalized']['close']
        if len(close_prices) > 10:  # Only use days with sufficient data
            # Normalize to 0-1 range and resample to fixed length
            prices_norm = (close_prices - np.min(close_prices)) / (np.max(close_prices) - np.min(close_prices))
            prices_resampled = np.interp(np.linspace(0, 1, 50), np.linspace(0, 1, len(prices_norm)), prices_norm)
            features.append(prices_resampled)
            valid_days.append(day)

    if len(features) < n_clusters:
        return {f"Cluster_{i}": [day] for i, day in enumerate(valid_days)}

    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(features)

    # Group days by cluster
    clustered_days = {}
    for i, cluster_id in enumerate(clusters):
        cluster_name = f"Cluster_{cluster_id}"
        if cluster_name not in clustered_days:
            clustered_days[cluster_name] = []
        clustered_days[cluster_name].append(valid_days[i])

    return clustered_days

@st.cache_data(show_spinner=True)
def download_data(ticker, period, interval, from_date, to_date, data_source, fmp_api_key):
    if data_source == "Downloaded historical data":
        return load_downloaded_data(ticker, interval)

    if data_source == "Financial Modeling Prep":
        return download_data_fmp(
            ticker=ticker,
            interval=interval,
            from_date=from_date,
            to_date=to_date,
            api_key=fmp_api_key
        )

    return download_data_yahoo(
        ticker=ticker,
        period=period,
        interval=interval
    )


@st.cache_data(show_spinner=True)
def download_data_yahoo(ticker, period, interval):
    df = yf.download(
        ticker,
        period=period,
        interval=interval,
        progress=False,
        auto_adjust=False
    )

    if df.empty:
        return df

    # Fix possible MultiIndex columns from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Convert timezone to New York time
    if df.index.tz is not None:
        df.index = df.index.tz_convert("America/New_York")
    else:
        df.index = df.index.tz_localize("UTC").tz_convert("America/New_York")

    return df


@st.cache_data(show_spinner=True)
def download_data_fmp(ticker, interval, from_date, to_date, api_key):
    if not api_key:
        st.error("Enter your Financial Modeling Prep API key in the sidebar.")
        return pd.DataFrame()

    interval_map = {
        "1m": "1min",
        "5m": "5min",
        "15m": "15min",
        "30m": "30min",
        "1h": "1hour",
    }

    if interval not in interval_map:
        st.error("Financial Modeling Prep does not support this interval in this app.")
        return pd.DataFrame()

    fmp_interval = interval_map[interval]

    url = f"https://financialmodelingprep.com/stable/historical-chart/{fmp_interval}"

    params = {
        "symbol": ticker,
        "from": str(from_date),
        "to": str(to_date),
        "apikey": api_key,
    }

    try:
        response = requests.get(url, params=params, timeout=30)
    except requests.exceptions.RequestException as e:
        st.error(f"Financial Modeling Prep request failed: {e}")
        return pd.DataFrame()

    if response.status_code != 200:
        st.error(f"Financial Modeling Prep request failed: {response.status_code}")
        st.write(response.text)
        return pd.DataFrame()

    try:
        data = response.json()
    except ValueError:
        st.error("Financial Modeling Prep did not return valid JSON.")
        st.write(response.text)
        return pd.DataFrame()

    if not data:
        st.warning(
            "Financial Modeling Prep returned no data. "
            "Check the ticker, API key, plan limits, or date range."
        )
        return pd.DataFrame()

    df = pd.DataFrame(data)

    if "date" not in df.columns:
        st.error("FMP response did not include a date column.")
        st.write(df.head())
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    df = df.sort_index()

    # FMP returns lowercase column names.
    df = df.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    )

    required_columns = {"Open", "High", "Low", "Close"}

    if not required_columns.issubset(set(df.columns)):
        st.error(f"FMP data is missing required columns. Found: {list(df.columns)}")
        return pd.DataFrame()

    keep_columns = ["Open", "High", "Low", "Close"]

    if "Volume" in df.columns:
        keep_columns.append("Volume")

    df = df[keep_columns]

    # FMP timestamps usually come back timezone-naive.
    # For this scanner, treat them as New York market time.
    if df.index.tz is None:
        df.index = df.index.tz_localize("America/New_York")
    else:
        df.index = df.index.tz_convert("America/New_York")

    return df


# ======================
# ENHANCED PATTERN / FILTER HELPERS
# ======================

def normalize_day_enhanced(day_data, start_value, include_indicators=True):
    """Enhanced normalization including technical indicators"""
    first_price = day_data["Close"].iloc[0]

    normalized = {
        "close": start_value * (day_data["Close"] / first_price),
        "high": start_value * (day_data["High"] / first_price),
        "low": start_value * (day_data["Low"] / first_price),
        "open": start_value * (day_data["Open"] / first_price),
    }

    if include_indicators and "RSI" in day_data.columns:
        # Normalize indicators to 0-100 scale where applicable
        normalized["rsi"] = day_data["RSI"].values
        normalized["macd"] = day_data["MACD"].values
        normalized["volume"] = day_data["Volume"].values / day_data["Volume"].max() * 100 if day_data["Volume"].max() > 0 else np.zeros(len(day_data))

    return normalized

def filter_hit_enhanced(normalized, filt):
    """
    Enhanced filter checking with technical indicators
    """
    start_bar = int(filt["start_bar"])
    end_bar = int(filt["end_bar"])
    threshold = float(filt["threshold"])
    direction = filt["direction"]
    price_source = filt["price_source"]
    indicator_type = filt.get("indicator_type", "Price")

    # Convert UI bars to Python index positions
    start_i = start_bar - 1
    end_i = end_bar - 1

    # Not enough bars to evaluate this filter
    if len(normalized["close"]) <= end_i:
        return False

    # Choose which data series to test
    if indicator_type == "Price":
        if price_source == "Close only":
            series = normalized["close"]
        else:
            if direction == "Hit or go ABOVE threshold":
                series = normalized["high"]
            else:
                series = normalized["low"]
    elif indicator_type == "RSI":
        series = normalized.get("rsi", np.full(len(normalized["close"]), 50))
    elif indicator_type == "MACD":
        series = normalized.get("macd", np.zeros(len(normalized["close"])))
    elif indicator_type == "Volume":
        series = normalized.get("volume", np.zeros(len(normalized["close"])))
    else:
        series = normalized["close"]

    window = series.iloc[start_i:end_i + 1] if hasattr(series, 'iloc') else series[start_i:end_i + 1]

    if direction == "Hit or go ABOVE threshold":
        return (window >= threshold).any()
    elif direction == "Hit or go BELOW threshold":
        return (window <= threshold).any()
    elif direction == "Cross ABOVE threshold":
        return (window.iloc[0] < threshold and window.iloc[-1] > threshold) if len(window) > 1 else False
    elif direction == "Cross BELOW threshold":
        return (window.iloc[0] > threshold and window.iloc[-1] < threshold) if len(window) > 1 else False
    else:
        return False


def day_passes_filters_enhanced(normalized, filters, logic_mode):
    """Enhanced filter checking with detailed results"""
    active_filters = [f for f in filters if f["enabled"]]

    # If no filters are enabled, show all days
    if not active_filters:
        return True, []

    results = []

    for filt in active_filters:
        passed = filter_hit_enhanced(normalized, filt)
        results.append({
            "name": filt["name"],
            "passed": passed,
            "threshold": filt["threshold"],
            "start_bar": filt["start_bar"],
            "end_bar": filt["end_bar"],
            "direction": filt["direction"],
            "indicator_type": filt.get("indicator_type", "Price"),
            "price_source": filt.get("price_source", "Close only"),
        })

    if logic_mode == "ALL filters must pass":
        final_pass = all(r["passed"] for r in results)
    else:
        final_pass = any(r["passed"] for r in results)

    return final_pass, results


def build_filter_group_enhanced(chart_name, default_filters=2):
    """Enhanced filter builder with technical indicators"""
    st.sidebar.subheader(f"🎯 {chart_name} Filters")

    num_filters = st.sidebar.number_input(
        f"{chart_name} - Number of filters",
        min_value=0,
        max_value=20,
        value=default_filters,
        step=1,
        key=f"{chart_name}_num_filters"
    )

    filters = []

    for i in range(int(num_filters)):
        with st.sidebar.expander(f"🔍 {chart_name} - Filter {i + 1}", expanded=False):
            enabled = st.checkbox(
                "Enable this filter",
                value=True,
                key=f"{chart_name}_enabled_{i}"
            )

            name = st.text_input(
                "Filter name",
                value=f"{chart_name} Filter {i + 1}",
                key=f"{chart_name}_name_{i}"
            )

            indicator_type = st.selectbox(
                "Indicator Type",
                ["Price", "RSI", "MACD", "Volume"],
                index=0,
                key=f"{chart_name}_indicator_{i}"
            )

            direction = st.selectbox(
                "Condition",
                ["Hit or go ABOVE threshold", "Hit or go BELOW threshold", "Cross ABOVE threshold", "Cross BELOW threshold"],
                index=0 if i == 1 else 1,
                key=f"{chart_name}_direction_{i}"
            )

            # Dynamic threshold based on indicator
            if indicator_type == "Price":
                threshold_default = 100.25 if i == 1 else 99.80
                threshold_step = 0.05
            elif indicator_type == "RSI":
                threshold_default = 70 if i == 1 else 30
                threshold_step = 1.0
            elif indicator_type == "MACD":
                threshold_default = 0.0
                threshold_step = 0.01
            else:  # Volume
                threshold_default = 50.0
                threshold_step = 1.0

            threshold = st.number_input(
                "Threshold",
                value=threshold_default,
                step=threshold_step,
                key=f"{chart_name}_threshold_{i}"
            )

            start_bar = st.number_input(
                "Start bar",
                min_value=1,
                value=10 if i == 1 else 1,
                step=1,
                key=f"{chart_name}_start_bar_{i}"
            )

            end_bar = st.number_input(
                "End bar",
                min_value=1,
                value=50 if i == 1 else 10,
                step=1,
                key=f"{chart_name}_end_bar_{i}"
            )

            # Price source only relevant for Price indicators
            price_source = "Close only"
            if indicator_type == "Price":
                price_source = st.selectbox(
                    "Price source",
                    ["Wicks: High for above, Low for below", "Close only"],
                    index=0,
                    key=f"{chart_name}_price_source_{i}"
                )

            if end_bar < start_bar:
                st.warning("End bar should be greater than or equal to start bar.")

            filters.append({
                "enabled": enabled,
                "name": name,
                "indicator_type": indicator_type,
                "direction": direction,
                "threshold": threshold,
                "start_bar": start_bar,
                "end_bar": end_bar,
                "price_source": price_source,
            })

    return filters


def analyze_chart_group_enhanced(session, start_value, filters, logic_mode, include_indicators=True):
    """
    
    """
    matching_days = []
    non_matching_days = []
    filter_result_rows = []
    max_len = 0

    for day, day_data in session.groupby(session.index.date):
        if day_data.empty:
            continue

        # Add technical indicators
        day_data_with_indicators = add_technical_indicators(day_data.copy())

        normalized = normalize_day_enhanced(day_data_with_indicators, start_value, include_indicators)

        passed, results = day_passes_filters_enhanced(
            normalized=normalized,
            filters=filters,
            logic_mode=logic_mode
        )

        day_record = {
            "day": str(day),
            "data": day_data_with_indicators,
            "normalized": normalized,
            "results": results,
        }

        max_len = max(max_len, len(normalized["close"]))

        if passed:
            matching_days.append(day_record)
        else:
            non_matching_days.append(day_record)

        for result in results:
            filter_result_rows.append({
                "Date": str(day),
                "Filter": result["name"],
                "Passed": result["passed"],
                "Threshold": result["threshold"],
                "Start Bar": result["start_bar"],
                "End Bar": result["end_bar"],
                "Direction": result["direction"],
                "Indicator": result["indicator_type"],
            })

    return {
        "matching_days": matching_days,
        "non_matching_days": non_matching_days,
        "filter_result_rows": filter_result_rows,
        "max_len": max_len,
    }


def make_pattern_chart_enhanced(
    chart_name,
    ticker,
    matching_days,
    non_matching_days,
    filters,
    start_value,
    max_len,
    show_non_matches,
    show_individual_legend,
    overlap_map,
    only_show_overlaps,
    show_indicators=True
):
    fig = go.Figure()

    colors = px.colors.qualitative.Dark24

    # Optional background non-matches
    if show_non_matches and not only_show_overlaps:
        for item in non_matching_days:
            y = item["normalized"]["close"]
            x = list(range(1, len(y) + 1))

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    line=dict(width=1),
                    opacity=0.08,
                    name="Non-match",
                    legendgroup="Non-matches",
                    showlegend=False,
                    hovertemplate=(
                        "Date: %{customdata}<br>"
                        "Bar: %{x}<br>"
                        "Normalized Close: %{y:.2f}"
                        "<extra></extra>"
                    ),
                    customdata=[item["day"]] * len(x),
                )
            )

    visible_match_count = 0

    # Matching days
    for idx, item in enumerate(matching_days):
        day = item["day"]
        overlap_info = overlap_map.get(day, [])

        if only_show_overlaps and len(overlap_info) == 0:
            continue

        y = item["normalized"]["close"]
        x = list(range(1, len(y) + 1))

        color = colors[idx % len(colors)]

        if len(overlap_info) == 0:
            line_width = 2
            line_dash = "solid"
            opacity = 0.75
            overlap_label = "Only in this chart"
        elif len(overlap_info) == 1:
            line_width = 4
            line_dash = "dash"
            opacity = 0.95
            overlap_label = f"Also in {overlap_info[0]}"
        else:
            line_width = 5
            line_dash = "solid"
            opacity = 1.0
            overlap_label = "In all 3 charts"

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                line=dict(
                    width=line_width,
                    color=color,
                    dash=line_dash
                ),
                opacity=opacity,
                name=(
                    f"{day} | {overlap_label}"
                    if show_individual_legend
                    else "Matching days"
                ),
                legendgroup="Matching days",
                showlegend=True if show_individual_legend else visible_match_count == 0,
                hovertemplate=(
                    "Date: %{customdata[0]}<br>"
                    "Bar: %{x}<br>"
                    "Normalized Close: %{y:.2f}<br>"
                    "Overlap: %{customdata[1]}"
                    "<extra></extra>"
                ),
                customdata=[[day, overlap_label]] * len(x),
            )
        )

        # Add RSI subplot if enabled
        if show_indicators and "rsi" in item["normalized"]:
            rsi_y = item["normalized"]["rsi"]
            if len(rsi_y) == len(x):
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=rsi_y,
                        mode="lines",
                        line=dict(width=1, color=color, dash="dot"),
                        opacity=0.6,
                        name=f"{day} RSI",
                        legendgroup="RSI",
                        showlegend=False,
                        yaxis="y2",
                        hovertemplate=(
                            "Date: %{customdata}<br>"
                            "Bar: %{x}<br>"
                            "RSI: %{y:.1f}"
                            "<extra></extra>"
                        ),
                        customdata=[day] * len(x),
                    )
                )

        visible_match_count += 1

    # Baseline
    if max_len > 0:
        fig.add_shape(
            type="line",
            x0=1,
            x1=max_len,
            y0=start_value,
            y1=start_value,
            line=dict(width=1, dash="dash"),
        )

        fig.add_annotation(
            x=max_len,
            y=start_value,
            text=f"Start {start_value}",
            showarrow=False,
            xanchor="left",
            yanchor="middle",
        )

    # Filter line segments
    active_filters = [f for f in filters if f["enabled"]]

    for i, filt in enumerate(active_filters):
        color = colors[i % len(colors)]

        # Adjust Y position based on indicator type
        y_pos = float(filt["threshold"])
        if filt["indicator_type"] == "RSI":
            y_pos = y_pos  # RSI is already 0-100
        elif filt["indicator_type"] == "MACD":
            y_pos = y_pos  # MACD can be any value
        elif filt["indicator_type"] == "Volume":
            y_pos = y_pos  # Volume is normalized 0-100

        fig.add_shape(
            type="line",
            x0=int(filt["start_bar"]),
            x1=int(filt["end_bar"]),
            y0=y_pos,
            y1=y_pos,
            line=dict(
                color=color,
                width=3,
                dash="dash"
            ),
        )

        fig.add_annotation(
            x=int(filt["end_bar"]),
            y=y_pos,
            text=f'{filt["name"]}: {filt["threshold"]}',
            showarrow=False,
            xanchor="left",
            yanchor="middle",
            font=dict(color=color),
        )

    title_suffix = "Overlaps Only" if only_show_overlaps else "All Matches"

    fig.update_layout(
        title=f"📊 {chart_name} — {ticker} — {title_suffix}",
        xaxis_title="Bar Number Inside Session",
        yaxis_title=f"Normalized Price Start = {start_value}",
        height=600,
        hovermode="closest",
        legend_title="Days",
        template="plotly_white",
        margin=dict(l=40, r=40, t=70, b=40),
    )

    # Add secondary Y-axis for RSI if indicators are shown
    if show_indicators:
        fig.update_layout(
            yaxis2=dict(
                title="RSI",
                overlaying="y",
                side="right",
                range=[0, 100],
                showgrid=False,
            )
        )

    fig.update_xaxes(
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        showline=True,
    )

    fig.update_yaxes(
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        showline=True,
    )

    return fig, visible_match_count


# ======================
# SIDEBAR SETTINGS
# ======================

st.sidebar.header("📊 Data Source")

data_source = st.sidebar.selectbox(
    "Data Source",
    ["Downloaded historical data", "Financial Modeling Prep", "Yahoo Finance"],
    index=0
)

# Dates for FMP
today = pd.Timestamp.today().date()
one_year_ago = (pd.Timestamp.today() - pd.DateOffset(years=1)).date()

fmp_api_key = ""
period = None

if data_source == "Financial Modeling Prep":
    fmp_api_key = st.sidebar.text_input(
        "FMP API Key",
        type="password"
    )

    from_date = one_year_ago
    to_date = today

    st.sidebar.write(f"FMP Date Range: **{from_date} to {to_date}**")

elif data_source == "Yahoo Finance":
    from_date = None
    to_date = None

    period = st.sidebar.selectbox(
        "Yahoo Period",
        ["5d", "1mo", "2mo", "60d", "6mo", "1y", "2y"],
        index=3
    )
else:
    from_date = None
    to_date = None
    downloaded_tickers = list_downloaded_tickers("5m")

    if downloaded_tickers:
        st.sidebar.write(f"Downloaded tickers: **{len(downloaded_tickers)}**")
    else:
        st.sidebar.warning("No downloaded 5m data found. Run exportfmp.py first.")


st.sidebar.header("🎨 Chart Settings")

if data_source == "Downloaded historical data":
    ticker = st.sidebar.selectbox(
        "Downloaded ticker",
        downloaded_tickers if downloaded_tickers else ["AAPL"],
        index=0,
    )
    interval = "5m"
    st.sidebar.write("Interval: **5m**")
else:
    ticker = st.sidebar.text_input("Ticker", value="QQQ")

    interval = st.sidebar.selectbox(
        "Interval",
        ["1m", "2m", "5m", "15m", "30m", "1h"],
        index=2
    )

start_time = st.sidebar.text_input("Session Start Time", value="09:30")
end_time = st.sidebar.text_input("Session End Time", value="13:30")

start_value = st.sidebar.number_input(
    "Normalized Start Value",
    value=100.0,
    step=0.25
)

show_indicators = st.sidebar.checkbox(
    "Show Technical Indicators",
    value=True
)

st.sidebar.header("🔍 Filter Logic")

logic_mode = st.sidebar.radio(
    "How should filters inside each chart work?",
    ["ALL filters must pass", "ANY filter can pass"],
    index=0
)

show_non_matches = st.sidebar.checkbox(
    "Show non-matching days faded in background",
    value=False
)

show_individual_legend = st.sidebar.checkbox(
    "Show each matching day in legend",
    value=False
)

only_show_overlaps = st.sidebar.checkbox(
    "Only show days that overlap with another chart",
    value=False
)

enable_clustering = st.sidebar.checkbox(
    "Enable Pattern Clustering",
    value=True
)

n_clusters = st.sidebar.slider(
    "Number of Pattern Clusters",
    min_value=2,
    max_value=5,
    value=3
) if enable_clustering else 3


st.sidebar.header("🎯 Chart Filter Groups")

chart_1_filters = build_filter_group_enhanced("Chart 1", default_filters=2)
chart_2_filters = build_filter_group_enhanced("Chart 2", default_filters=2)
chart_3_filters = build_filter_group_enhanced("Chart 3", default_filters=2)

chart_filter_groups = {
    "Chart 1": chart_1_filters,
    "Chart 2": chart_2_filters,
    "Chart 3": chart_3_filters,
}

run_button = st.sidebar.button("🚀 Run Analysis")


# ======================
# MAIN LOGIC
# ======================

if run_button:

    df = download_data(
        ticker=ticker,
        period=period,
        interval=interval,
        from_date=from_date,
        to_date=to_date,
        data_source=data_source,
        fmp_api_key=fmp_api_key
    )

    if df.empty:
        st.error("No data downloaded. Try a different ticker, interval, data source, or API key.")
        st.stop()

    required_columns = {"Open", "High", "Low", "Close"}

    if not required_columns.issubset(set(df.columns)):
        st.error(f"Missing required columns. Found: {list(df.columns)}")
        st.stop()

    session = df.between_time(start_time, end_time)

    if session.empty:
        st.error("No data found inside that session time window.")
        st.stop()

    # Add technical indicators
    if show_indicators:
        session = add_technical_indicators(session.copy())

    # ======================
    # ANALYZE EACH CHART GROUP
    # ======================

    chart_results = {}

    for chart_name, group_filters in chart_filter_groups.items():
        chart_results[chart_name] = analyze_chart_group_enhanced(
            session=session,
            start_value=start_value,
            filters=group_filters,
            logic_mode=logic_mode,
            include_indicators=show_indicators
        )

    # ======================
    # FIND OVERLAPS BETWEEN CHARTS
    # ======================

    matching_sets = {}

    for chart_name, result in chart_results.items():
        matching_sets[chart_name] = set(
            item["day"] for item in result["matching_days"]
        )

    overlap_maps = {}

    for chart_name in chart_results.keys():
        overlap_maps[chart_name] = {}

        current_days = matching_sets[chart_name]

        for day in current_days:
            also_in = []

            for other_chart_name, other_days in matching_sets.items():
                if other_chart_name == chart_name:
                    continue

                if day in other_days:
                    also_in.append(other_chart_name)

            overlap_maps[chart_name][day] = also_in

    # ======================
    # DISPLAY SUMMARY
    # ======================

    st.subheader("Chart Match Summary")

    chart_1_days = matching_sets["Chart 1"]
    chart_2_days = matching_sets["Chart 2"]
    chart_3_days = matching_sets["Chart 3"]

    all_three = sorted(chart_1_days & chart_2_days & chart_3_days)
    chart_1_and_2 = sorted((chart_1_days & chart_2_days) - set(all_three))
    chart_1_and_3 = sorted((chart_1_days & chart_3_days) - set(all_three))
    chart_2_and_3 = sorted((chart_2_days & chart_3_days) - set(all_three))

    summary_rows = []

    for chart_name, result in chart_results.items():
        overlap_count = sum(
            1 for item in result["matching_days"]
            if len(overlap_maps[chart_name].get(item["day"], [])) > 0
        )

        summary_rows.append(
            {
                "Chart": chart_name,
                "Total Matching Days": len(result["matching_days"]),
                "Overlapping Matching Days": overlap_count,
                "Non-Matching Days": len(result["non_matching_days"]),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    st.dataframe(summary_df, use_container_width=True)

    st.write(f"Data Source: **{data_source}**")

    if data_source == "Financial Modeling Prep":
        st.write(f"FMP Date Range Used: **{from_date} to {to_date}**")

    st.write(f"Filter logic inside each chart: **{logic_mode}**")

    if only_show_overlaps:
        st.info("Showing only days that appear in at least two charts.")

    # ======================
    # PATTERN CLUSTERING
    # ======================

    if enable_clustering:
        st.subheader("🎯 Pattern Clustering Analysis")

        all_matching_days = []
        for result in chart_results.values():
            all_matching_days.extend(result["matching_days"])

        if len(all_matching_days) >= n_clusters:
            clustered_days = cluster_similar_days(all_matching_days, n_clusters)

            col1, col2 = st.columns(2)

            with col1:
                st.write("**Pattern Clusters:**")
                for cluster_name, days in clustered_days.items():
                    with st.expander(f"{cluster_name} ({len(days)} days)"):
                        for day in days:
                            st.write(f"• {day['day']}")

            with col2:
                # Show similarity matrix for first few days
                if len(all_matching_days) > 1:
                    st.write("**Pattern Similarities:**")
                    similarity_data = []
                    sample_days = all_matching_days[:min(5, len(all_matching_days))]

                    for i, day1 in enumerate(sample_days):
                        for j, day2 in enumerate(sample_days):
                            if i < j:
                                similarity = calculate_pattern_similarity(
                                    day1['normalized'], day2['normalized']
                                )
                                similarity_data.append({
                                    "Day 1": day1['day'],
                                    "Day 2": day2['day'],
                                    "Similarity": f"{similarity:.1f}%"
                                })

                    if similarity_data:
                        st.dataframe(pd.DataFrame(similarity_data), use_container_width=True)

    # ======================
    # DISPLAY THREE CHARTS
    # ======================

    chart_config = {
        "scrollZoom": True,
        "displaylogo": False,
        "modeBarButtonsToAdd": [
            "drawline",
            "drawopenpath",
            "eraseshape",
        ],
    }

    col1, col2, col3 = st.columns(3)

    chart_columns = {
        "Chart 1": col1,
        "Chart 2": col2,
        "Chart 3": col3,
    }

    visible_counts = {}

    for chart_name, col in chart_columns.items():
        result = chart_results[chart_name]

        fig, visible_count = make_pattern_chart_enhanced(
            chart_name=chart_name,
            ticker=ticker,
            matching_days=result["matching_days"],
            non_matching_days=result["non_matching_days"],
            filters=chart_filter_groups[chart_name],
            start_value=start_value,
            max_len=result["max_len"],
            show_non_matches=show_non_matches,
            show_individual_legend=show_individual_legend,
            overlap_map=overlap_maps[chart_name],
            only_show_overlaps=only_show_overlaps,
            show_indicators=show_indicators,
        )

        visible_counts[chart_name] = visible_count

        with col:
            st.plotly_chart(fig, use_container_width=True, config=chart_config)

            st.write(f"**{visible_count} visible matching days**")
            st.write(f"**{len(result['matching_days'])} total matching days**")

            if result["matching_days"]:
                if only_show_overlaps:
                    visible_days = [
                        item["day"]
                        for item in result["matching_days"]
                        if len(overlap_maps[chart_name].get(item["day"], [])) > 0
                    ]
                else:
                    visible_days = [item["day"] for item in result["matching_days"]]

                with st.expander(f"{chart_name} Visible Dates"):
                    if visible_days:
                        st.write(visible_days)
                    else:
                        st.write("No visible dates based on current settings.")

    # ======================
    # STATISTICAL ANALYSIS
    # ======================

    st.subheader("📈 Statistical Analysis")

    all_matching_days = [day for result in chart_results.values() for day in result["matching_days"]]
    all_non_matching_days = [day for result in chart_results.values() for day in result["non_matching_days"]]

    if all_matching_days:
        # Calculate returns for matching vs non-matching days
        matching_returns = []
        non_matching_returns = []

        for day in all_matching_days:
            if len(day["data"]) > 1:
                day_return = (day["data"]["Close"].iloc[-1] - day["data"]["Close"].iloc[0]) / day["data"]["Close"].iloc[0] * 100
                matching_returns.append(day_return)

        for day in all_non_matching_days:
            if len(day["data"]) > 1:
                day_return = (day["data"]["Close"].iloc[-1] - day["data"]["Close"].iloc[0]) / day["data"]["Close"].iloc[0] * 100
                non_matching_returns.append(day_return)

        if matching_returns and non_matching_returns:
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Matching Days Avg Return",
                    f"{np.mean(matching_returns):.2f}%",
                    f"{np.mean(matching_returns) - np.mean(non_matching_returns):.2f}% vs non-matching"
                )

            with col2:
                st.metric(
                    "Matching Days Win Rate",
                    f"{(np.array(matching_returns) > 0).mean() * 100:.1f}%",
                    f"{((np.array(matching_returns) > 0).mean() - (np.array(non_matching_returns) > 0).mean()) * 100:.1f}% vs non-matching"
                )

            with col3:
                st.metric(
                    "Matching Days Volatility",
                    f"{np.std(matching_returns):.2f}%",
                    f"{np.std(matching_returns) - np.std(non_matching_returns):.2f}% vs non-matching"
                )

    # ======================
    # OVERLAP DETAILS
    # ======================

    st.subheader("Overlapping Days")

    overlap_detail = {
        "In All 3 Charts": all_three,
        "In Chart 1 and Chart 2 Only": chart_1_and_2,
        "In Chart 1 and Chart 3 Only": chart_1_and_3,
        "In Chart 2 and Chart 3 Only": chart_2_and_3,
    }

    for label, days in overlap_detail.items():
        with st.expander(f"{label} ({len(days)})"):
            if days:
                st.write(days)
            else:
                st.write("None")

    # ======================
    # EXPORT FUNCTIONALITY
    # ======================

    st.subheader("💾 Export Results")

    if st.button("Export Pattern Analysis to CSV"):
        export_data = []

        for chart_name, result in chart_results.items():
            for item in result["matching_days"]:
                export_data.append({
                    "Chart": chart_name,
                    "Date": item["day"],
                    "Matches_Filters": True,
                    "Overlaps_With": ", ".join(overlap_maps[chart_name].get(item["day"], [])),
                })

        if export_data:
            export_df = pd.DataFrame(export_data)
            csv = export_df.to_csv(index=False)

            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"pattern_analysis_{ticker}_{pd.Timestamp.today().date()}.csv",
                mime="text/csv"
            )
            st.success("Export data prepared!")

    # ======================
    # FILTER PASS / FAIL DETAILS
    # ======================

    st.subheader("Filter Pass / Fail Detail")

    for chart_name, result in chart_results.items():
        with st.expander(f"{chart_name} Filter Details"):
            if result["filter_result_rows"]:
                result_df = pd.DataFrame(result["filter_result_rows"])
                st.dataframe(result_df, use_container_width=True)
            else:
                st.write("No filter details available.")

else:
    st.info("🎯 Choose your settings on the left, then click '🚀 Run Analysis' to find patterns in high-volatility markets!")

    # Show some helpful tips
    with st.expander("💡 Tips for Finding Patterns"):
        st.markdown("""
        - Start with Yahoo Finance if you do not have an FMP API key yet.
        - Use a shorter interval like 5m or 15m when you want intraday patterns.
        - If no days match, loosen the filters or switch from ALL filters to ANY filter.
        - Use RSI, MACD, and Volume filters after the price filters are working.
        """)
