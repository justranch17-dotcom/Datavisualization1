import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests
from local_market_data import list_downloaded_tickers, load_downloaded_data
from plotly.subplots import make_subplots

st.set_page_config(page_title="Market Pattern Scanner", layout="wide")

st.title("Market Pattern Scanner")


# ======================
# DATA DOWNLOAD HELPERS
# ======================

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

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

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

    if df.index.tz is None:
        df.index = df.index.tz_localize("America/New_York")
    else:
        df.index = df.index.tz_convert("America/New_York")

    return df


# ======================
# PATTERN HELPERS
# ======================

def normalize_day(day_data, start_value):
    first_price = day_data["Close"].iloc[0]

    return {
        "open": start_value * (day_data["Open"] / first_price),
        "high": start_value * (day_data["High"] / first_price),
        "low": start_value * (day_data["Low"] / first_price),
        "close": start_value * (day_data["Close"] / first_price),
    }


def get_interval_minutes(interval):
    mapping = {
        "1m": 1,
        "2m": 2,
        "5m": 5,
        "15m": 15,
        "30m": 30,
        "1h": 60,
    }

    return mapping.get(interval, 5)


def analyze_morning_patterns(
    session,
    start_value,
    morning_bars,
    direction,
    move_percent,
    use_volume_filter,
    volume_ratio_min,
    volume_ratio_max
):
    """
    Finds days where the early-session movement matches the selected pattern.

    Example:
    Direction = Up
    Morning Bars = 3
    Move Percent = 0.25

    Means:
    In bars 1 through 3, did NormalizedHigh hit 100.25 or higher?
    """

    all_days = []
    morning_volume_values = []

    for day, day_data in session.groupby(session.index.date):

        if day_data.empty:
            continue

        if len(day_data) < morning_bars:
            continue

        normalized = normalize_day(day_data, start_value)

        morning_window = day_data.iloc[:morning_bars]

        if "Volume" in day_data.columns:
            morning_volume = float(morning_window["Volume"].sum())
        else:
            morning_volume = None

        if morning_volume is not None:
            morning_volume_values.append(morning_volume)

        all_days.append(
            {
                "day": str(day),
                "date_obj": day,
                "data": day_data,
                "normalized": normalized,
                "morning_volume": morning_volume,
            }
        )

    avg_morning_volume = None

    if morning_volume_values:
        avg_morning_volume = sum(morning_volume_values) / len(morning_volume_values)

    matching_days = []
    non_matching_days = []

    if direction == "Up":
        target_level = start_value * (1 + move_percent / 100)
    else:
        target_level = start_value * (1 - move_percent / 100)

    for item in all_days:
        normalized = item["normalized"]

        morning_high_window = normalized["high"].iloc[:morning_bars]
        morning_low_window = normalized["low"].iloc[:morning_bars]
        morning_close_window = normalized["close"].iloc[:morning_bars]

        max_morning_high = float(morning_high_window.max())
        min_morning_low = float(morning_low_window.min())
        last_morning_close = float(morning_close_window.iloc[-1])

        if direction == "Up":
            price_passed = max_morning_high >= target_level
        else:
            price_passed = min_morning_low <= target_level

        volume_ratio = None
        volume_passed = True

        if use_volume_filter:
            if item["morning_volume"] is None or avg_morning_volume is None or avg_morning_volume == 0:
                volume_passed = False
            else:
                volume_ratio = item["morning_volume"] / avg_morning_volume
                volume_passed = volume_ratio_min <= volume_ratio <= volume_ratio_max
        else:
            if item["morning_volume"] is not None and avg_morning_volume not in [None, 0]:
                volume_ratio = item["morning_volume"] / avg_morning_volume

        passed = price_passed and volume_passed

        item["stats"] = {
            "target_level": target_level,
            "max_morning_high": max_morning_high,
            "min_morning_low": min_morning_low,
            "last_morning_close": last_morning_close,
            "morning_volume": item["morning_volume"],
            "avg_morning_volume": avg_morning_volume,
            "volume_ratio": volume_ratio,
            "price_passed": price_passed,
            "volume_passed": volume_passed,
            "passed": passed,
        }

        if passed:
            matching_days.append(item)
        else:
            non_matching_days.append(item)

    return {
        "matching_days": matching_days,
        "non_matching_days": non_matching_days,
        "all_days": all_days,
        "avg_morning_volume": avg_morning_volume,
        "target_level": target_level,
    }


def make_overview_chart(
    ticker,
    matching_days,
    non_matching_days,
    start_value,
    target_level,
    morning_bars,
    show_non_matches,
    show_individual_legend
):
    fig = go.Figure()

    colors = px.colors.qualitative.Dark24

    max_len = 0

    for item in matching_days + non_matching_days:
        max_len = max(max_len, len(item["normalized"]["close"]))

    if show_non_matches:
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

    for idx, item in enumerate(matching_days):
        y = item["normalized"]["close"]
        x = list(range(1, len(y) + 1))

        color = colors[idx % len(colors)]

        vol_ratio = item["stats"]["volume_ratio"]

        if vol_ratio is None:
            volume_text = "N/A"
        else:
            volume_text = f"{vol_ratio:.2f}x"

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                line=dict(width=2, color=color),
                opacity=0.85,
                name=item["day"] if show_individual_legend else "Matching days",
                legendgroup="Matching days",
                showlegend=True if show_individual_legend else idx == 0,
                hovertemplate=(
                    "Date: %{customdata[0]}<br>"
                    "Bar: %{x}<br>"
                    "Normalized Close: %{y:.2f}<br>"
                    "Morning Volume Ratio: %{customdata[1]}"
                    "<extra></extra>"
                ),
                customdata=[[item["day"], volume_text]] * len(x),
            )
        )

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

        fig.add_shape(
            type="line",
            x0=1,
            x1=morning_bars,
            y0=target_level,
            y1=target_level,
            line=dict(width=3, dash="dash"),
        )

        fig.add_annotation(
            x=morning_bars,
            y=target_level,
            text=f"Morning target {target_level:.2f}",
            showarrow=False,
            xanchor="left",
            yanchor="middle",
        )

        fig.add_vrect(
            x0=1,
            x1=morning_bars,
            opacity=0.08,
            line_width=0,
            annotation_text="Morning pattern window",
            annotation_position="top left",
        )

    fig.update_layout(
        title=f"{ticker} Morning Pattern Matches",
        xaxis_title="Bar Number Inside Session",
        yaxis_title=f"Normalized Price Start = {start_value}",
        height=650,
        hovermode="closest",
        legend_title="Days",
        template="plotly_white",
    )

    fig.update_xaxes(
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        showline=True,
        rangeslider=dict(visible=True),
    )

    fig.update_yaxes(
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        showline=True,
        fixedrange=False,
    )

    return fig


def make_detail_chart(selected_item, ticker, start_value, morning_bars):
    day = selected_item["day"]
    day_data = selected_item["data"]
    normalized = selected_item["normalized"]

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.50, 0.30, 0.20],
        subplot_titles=(
            f"{ticker} {day} Candles",
            "Normalized Movement",
            "Volume"
        )
    )

    fig.add_trace(
        go.Candlestick(
            x=day_data.index,
            open=day_data["Open"],
            high=day_data["High"],
            low=day_data["Low"],
            close=day_data["Close"],
            name="Candles",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=day_data.index,
            y=normalized["close"],
            mode="lines",
            name="Normalized Close",
            line=dict(width=2),
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=day_data.index,
            y=normalized["high"],
            mode="lines",
            name="Normalized High",
            line=dict(width=1, dash="dot"),
            opacity=0.7,
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=day_data.index,
            y=normalized["low"],
            mode="lines",
            name="Normalized Low",
            line=dict(width=1, dash="dot"),
            opacity=0.7,
        ),
        row=2,
        col=1,
    )

    if "Volume" in day_data.columns:
        fig.add_trace(
            go.Bar(
                x=day_data.index,
                y=day_data["Volume"],
                name="Volume",
                opacity=0.7,
            ),
            row=3,
            col=1,
        )

    if len(day_data) >= morning_bars:
        morning_end_time = day_data.index[morning_bars - 1]

        for row in [1, 2, 3]:
            fig.add_vrect(
                x0=day_data.index[0],
                x1=morning_end_time,
                opacity=0.08,
                line_width=0,
                row=row,
                col=1,
            )

    fig.update_layout(
        title=f"Detailed View — {ticker} — {day}",
        height=900,
        template="plotly_white",
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
    )

    fig.update_xaxes(
        rangeslider_visible=False,
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
    )

    fig.update_yaxes(fixedrange=False)

    return fig


def build_results_table(matching_days):
    rows = []

    for item in matching_days:
        stats = item["stats"]

        rows.append(
            {
                "Date": item["day"],
                "Target Level": round(stats["target_level"], 4),
                "Max Morning High": round(stats["max_morning_high"], 4),
                "Min Morning Low": round(stats["min_morning_low"], 4),
                "Last Morning Close": round(stats["last_morning_close"], 4),
                "Morning Volume": stats["morning_volume"],
                "Avg Morning Volume": stats["avg_morning_volume"],
                "Volume Ratio": stats["volume_ratio"],
                "Price Passed": stats["price_passed"],
                "Volume Passed": stats["volume_passed"],
            }
        )

    if not rows:
        return pd.DataFrame()

    result_df = pd.DataFrame(rows)

    if "Volume Ratio" in result_df.columns:
        result_df["Volume Ratio"] = result_df["Volume Ratio"].round(3)

    return result_df


# ======================
# SIDEBAR SETTINGS
# ======================

st.sidebar.header("Data Source")

data_source = st.sidebar.selectbox(
    "Data Source",
    ["Downloaded historical data", "Financial Modeling Prep", "Yahoo Finance"],
    index=0
)

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
        ["5d", "1mo", "2mo", "60d"],
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


st.sidebar.header("Chart Settings")

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

interval_minutes = get_interval_minutes(interval)

start_time = st.sidebar.text_input("Session Start Time", value="09:30")
end_time = st.sidebar.text_input("Session End Time", value="16:00")

start_value = st.sidebar.number_input(
    "Normalized Start Value",
    value=100.0,
    step=0.25
)


st.sidebar.header("Morning Pattern Scanner")

morning_minutes = st.sidebar.number_input(
    "Morning window minutes",
    min_value=1,
    max_value=240,
    value=15,
    step=1
)

morning_bars = max(1, int(round(morning_minutes / interval_minutes)))

st.sidebar.write(f"Using first **{morning_bars} bars** for the morning window.")

direction = st.sidebar.selectbox(
    "Morning direction",
    ["Up", "Down"],
    index=0
)

move_percent = st.sidebar.number_input(
    "Required move percent",
    min_value=0.01,
    max_value=10.0,
    value=0.25,
    step=0.05
)

use_volume_filter = st.sidebar.checkbox(
    "Use volume spike filter",
    value=True
)

volume_ratio_min = st.sidebar.number_input(
    "Minimum volume ratio",
    min_value=0.0,
    max_value=10.0,
    value=1.05,
    step=0.05
)

volume_ratio_max = st.sidebar.number_input(
    "Maximum volume ratio",
    min_value=0.0,
    max_value=10.0,
    value=1.50,
    step=0.05
)

st.sidebar.caption(
    "Volume ratio compares that day's opening-window volume to the average opening-window volume across all days."
)

show_non_matches = st.sidebar.checkbox(
    "Show non-matching days faded in overview",
    value=False
)

show_individual_legend = st.sidebar.checkbox(
    "Show each matching day in legend",
    value=False
)

run_button = st.sidebar.button("Run Scanner")

if "scanner_has_run" not in st.session_state:
    st.session_state.scanner_has_run = False

if "scanner_results" not in st.session_state:
    st.session_state.scanner_results = None

if "selected_match_index" not in st.session_state:
    st.session_state.selected_match_index = 0

def select_matching_day(matching_days):
    """
    Lets you switch through matching days using Previous / Next buttons.
    Stores the selected index in Streamlit session state.
    """

    matching_dates = [item["day"] for item in matching_days]

    if "selected_match_index" not in st.session_state:
        st.session_state.selected_match_index = 0

    if st.session_state.selected_match_index >= len(matching_dates):
        st.session_state.selected_match_index = 0

    if st.session_state.selected_match_index < 0:
        st.session_state.selected_match_index = len(matching_dates) - 1

    col_prev, col_select, col_next = st.columns([1, 4, 1])

    with col_prev:
        if st.button("⬅ Previous", key="previous_matching_day"):
            st.session_state.selected_match_index -= 1

            if st.session_state.selected_match_index < 0:
                st.session_state.selected_match_index = len(matching_dates) - 1

            st.rerun()

    with col_next:
        if st.button("Next ➡", key="next_matching_day"):
            st.session_state.selected_match_index += 1

            if st.session_state.selected_match_index >= len(matching_dates):
                st.session_state.selected_match_index = 0

            st.rerun()

    with col_select:
        selected_day = st.selectbox(
            "Choose a matching day to inspect",
            matching_dates,
            index=st.session_state.selected_match_index,
            key="selected_matching_day_dropdown"
        )

    st.session_state.selected_match_index = matching_dates.index(selected_day)

    selected_item = matching_days[st.session_state.selected_match_index]

    return selected_day, selected_item


# ======================
# MAIN LOGIC
# ======================

if run_button:
    st.session_state.scanner_has_run = True
    st.session_state.selected_match_index = 0

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

    if use_volume_filter and "Volume" not in df.columns:
        st.error("Volume filter is turned on, but this data source did not return a Volume column.")
        st.stop()

    session = df.between_time(start_time, end_time)

    if session.empty:
        st.error("No data found inside that session time window.")
        st.stop()

    results = analyze_morning_patterns(
        session=session,
        start_value=start_value,
        morning_bars=morning_bars,
        direction=direction,
        move_percent=move_percent,
        use_volume_filter=use_volume_filter,
        volume_ratio_min=volume_ratio_min,
        volume_ratio_max=volume_ratio_max,
    )
    st.session_state.scanner_results = {
        "results": results,
        "ticker": ticker,
        "data_source": data_source,
        "from_date": from_date,
        "to_date": to_date,
        "start_value": start_value,
        "target_level": results["target_level"],
        "morning_bars": morning_bars,
        "morning_minutes": morning_minutes,
        "direction": direction,
        "move_percent": move_percent,
        "use_volume_filter": use_volume_filter,
        "volume_ratio_min": volume_ratio_min,
        "volume_ratio_max": volume_ratio_max,
    }
if st.session_state.scanner_results is not None:
    saved = st.session_state.scanner_results

    results = saved["results"]
    ticker = saved["ticker"]
    data_source = saved["data_source"]
    from_date = saved["from_date"]
    to_date = saved["to_date"]
    start_value = saved["start_value"]
    target_level = saved["target_level"]
    morning_bars = saved["morning_bars"]
    morning_minutes = saved["morning_minutes"]
    direction = saved["direction"]
    move_percent = saved["move_percent"]
    use_volume_filter = saved["use_volume_filter"]
    volume_ratio_min = saved["volume_ratio_min"]
    volume_ratio_max = saved["volume_ratio_max"]

    matching_days = results["matching_days"]
    non_matching_days = results["non_matching_days"]
    all_days = results["all_days"]
    avg_morning_volume = results["avg_morning_volume"]
    matching_days = results["matching_days"]
    non_matching_days = results["non_matching_days"]
    all_days = results["all_days"]
    avg_morning_volume = results["avg_morning_volume"]
    target_level = results["target_level"]

    total_days = len(all_days)

    st.subheader("Morning Pattern Summary")

    col_a, col_b, col_c, col_d = st.columns(4)

    with col_a:
        st.metric("Matching Days", len(matching_days))

    with col_b:
        st.metric("Total Days Checked", total_days)

    with col_c:
        if total_days > 0:
            match_rate = len(matching_days) / total_days * 100
        else:
            match_rate = 0
        st.metric("Match Rate", f"{match_rate:.1f}%")

    with col_d:
        if avg_morning_volume is not None:
            st.metric("Avg Opening Volume", f"{avg_morning_volume:,.0f}")
        else:
            st.metric("Avg Opening Volume", "N/A")

    st.write(f"Data Source: **{data_source}**")

    if data_source == "Financial Modeling Prep":
        st.write(f"FMP Date Range Used: **{from_date} to {to_date}**")

    st.write(
        f"Pattern: **{direction} {move_percent:.2f}% within first {morning_minutes} minutes "
        f"({morning_bars} bars)**"
    )

    if use_volume_filter:
        st.write(
            f"Volume filter: opening-window volume ratio between "
            f"**{volume_ratio_min:.2f}x** and **{volume_ratio_max:.2f}x**"
        )
    else:
        st.write("Volume filter: **Off**")

    st.subheader("Overview: Matching Days")

    overview_fig = make_overview_chart(
        ticker=ticker,
        matching_days=matching_days,
        non_matching_days=non_matching_days,
        start_value=start_value,
        target_level=target_level,
        morning_bars=morning_bars,
        show_non_matches=show_non_matches,
        show_individual_legend=show_individual_legend,
    )

    chart_config = {
        "scrollZoom": True,
        "displaylogo": False,
        "modeBarButtonsToAdd": [
            "drawline",
            "drawopenpath",
            "eraseshape",
        ],
    }

    st.plotly_chart(overview_fig, use_container_width=True, config=chart_config)

    results_table = build_results_table(matching_days)

    if not results_table.empty:
        st.subheader("Matching Days Table")
        st.dataframe(results_table, use_container_width=True)
    else:
        st.warning("No matching days found with the current settings.")

    st.subheader("Detailed Day View")

    if matching_days:
        selected_day, selected_item = select_matching_day(matching_days)

    

        selected_stats = selected_item["stats"]

        detail_col_1, detail_col_2, detail_col_3, detail_col_4 = st.columns(4)

        with detail_col_1:
            st.metric("Max Morning High", f"{selected_stats['max_morning_high']:.2f}")

        with detail_col_2:
            st.metric("Min Morning Low", f"{selected_stats['min_morning_low']:.2f}")

        with detail_col_3:
            if selected_stats["volume_ratio"] is not None:
                st.metric("Volume Ratio", f"{selected_stats['volume_ratio']:.2f}x")
            else:
                st.metric("Volume Ratio", "N/A")

        with detail_col_4:
            st.metric("Target Level", f"{selected_stats['target_level']:.2f}")

        detail_fig = make_detail_chart(
            selected_item=selected_item,
            ticker=ticker,
            start_value=start_value,
            morning_bars=morning_bars,
        )

        st.plotly_chart(detail_fig, use_container_width=True, config=chart_config)

    else:
        st.info("No matching days to inspect. Loosen the move or volume settings.")

else:
    st.info("Choose your settings on the left, then click Run Scanner.")
    
