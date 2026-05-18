import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from local_market_data import list_downloaded_tickers, load_downloaded_data
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from ta.momentum import RSIIndicator
from ta.trend import MACD

warnings.filterwarnings("ignore")

FEEDBACK_FILE = Path(__file__).with_name("mach2_group_feedback.csv")

st.set_page_config(page_title="Mach 2 Market Pattern Lab", layout="wide")

st.title("Mach 2 Market Pattern Lab")
st.caption("A simpler workspace for testing market patterns across real intraday days.")


# ======================
# DATA
# ======================


@st.cache_data(show_spinner=True)
def download_data(ticker, period, interval, from_date, to_date, data_source, fmp_api_key):
    if data_source == "Downloaded historical data":
        return load_downloaded_data(ticker, interval)

    if data_source == "Financial Modeling Prep":
        return download_data_fmp(ticker, interval, from_date, to_date, fmp_api_key)

    return download_data_yahoo(ticker, period, interval)


@st.cache_data(show_spinner=True)
def download_data_yahoo(ticker, period, interval):
    df = yf.download(
        ticker,
        period=period,
        interval=interval,
        progress=False,
        auto_adjust=False,
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

    url = f"https://financialmodelingprep.com/stable/historical-chart/{interval_map[interval]}"
    params = {
        "symbol": ticker,
        "from": str(from_date),
        "to": str(to_date),
        "apikey": api_key,
    }

    try:
        response = requests.get(url, params=params, timeout=30)
    except requests.exceptions.RequestException as exc:
        st.error(f"Financial Modeling Prep request failed: {exc}")
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
        st.warning("Financial Modeling Prep returned no data.")
        return pd.DataFrame()

    df = pd.DataFrame(data)

    if "date" not in df.columns:
        st.error("FMP response did not include a date column.")
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
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
# PATTERN ENGINE
# ======================


def get_interval_minutes(interval):
    return {
        "1m": 1,
        "2m": 2,
        "5m": 5,
        "15m": 15,
        "30m": 30,
        "1h": 60,
    }.get(interval, 5)


def add_indicators(day_data):
    df = day_data.copy()

    if len(df) >= 14:
        df["RSI"] = RSIIndicator(close=df["Close"], window=14).rsi()
    else:
        df["RSI"] = np.nan

    if len(df) >= 26:
        macd = MACD(close=df["Close"])
        df["MACD"] = macd.macd()
        df["MACD_Signal"] = macd.macd_signal()
    else:
        df["MACD"] = np.nan
        df["MACD_Signal"] = np.nan

    if "Volume" in df.columns and df["Volume"].max() > 0:
        df["Volume_Pct_Day_Max"] = df["Volume"] / df["Volume"].max() * 100
    else:
        df["Volume_Pct_Day_Max"] = np.nan

    return df


def normalize_day(day_data, start_value):
    first_price = day_data["Close"].iloc[0]
    normalized = {
        "Normalized Close": start_value * (day_data["Close"] / first_price),
        "Normalized High": start_value * (day_data["High"] / first_price),
        "Normalized Low": start_value * (day_data["Low"] / first_price),
        "Normalized Open": start_value * (day_data["Open"] / first_price),
        "RSI": day_data["RSI"],
        "MACD": day_data["MACD"],
        "Volume % of Day Max": day_data["Volume_Pct_Day_Max"],
    }
    return normalized


def strategy_defaults(strategy_name, start_value, interval_minutes):
    first_15 = max(1, round(15 / interval_minutes))
    first_30 = max(1, round(30 / interval_minutes))

    if strategy_name == "Opening Drive Up":
        return [
            {
                "name": "Early push up",
                "enabled": True,
                "series": "Normalized High",
                "condition": "Hit or go ABOVE",
                "threshold": start_value + 0.30,
                "start_bar": 1,
                "end_bar": first_15,
            }
        ]

    if strategy_name == "Opening Drive Down":
        return [
            {
                "name": "Early push down",
                "enabled": True,
                "series": "Normalized Low",
                "condition": "Hit or go BELOW",
                "threshold": start_value - 0.30,
                "start_bar": 1,
                "end_bar": first_15,
            }
        ]

    if strategy_name == "Reversal Watch":
        return [
            {
                "name": "Early selloff",
                "enabled": True,
                "series": "Normalized Low",
                "condition": "Hit or go BELOW",
                "threshold": start_value - 0.25,
                "start_bar": 1,
                "end_bar": first_15,
            },
            {
                "name": "Recovery above start",
                "enabled": True,
                "series": "Normalized Close",
                "condition": "Hit or go ABOVE",
                "threshold": start_value,
                "start_bar": first_15,
                "end_bar": first_30,
            },
        ]

    return [
        {
            "name": "Price condition",
            "enabled": True,
            "series": "Normalized High",
            "condition": "Hit or go ABOVE",
            "threshold": start_value + 0.25,
            "start_bar": 1,
            "end_bar": first_15,
        }
    ]


def build_filter(chart_name, filter_index, default_filter):
    with st.sidebar.expander(f"{chart_name} filter {filter_index + 1}", expanded=False):
        enabled = st.checkbox(
            "Use this filter",
            value=default_filter["enabled"],
            key=f"{chart_name}_filter_{filter_index}_enabled",
        )

        name = st.text_input(
            "Name",
            value=default_filter["name"],
            key=f"{chart_name}_filter_{filter_index}_name",
        )

        series = st.selectbox(
            "What to test",
            [
                "Normalized High",
                "Normalized Low",
                "Normalized Close",
                "RSI",
                "MACD",
                "Volume % of Day Max",
            ],
            index=[
                "Normalized High",
                "Normalized Low",
                "Normalized Close",
                "RSI",
                "MACD",
                "Volume % of Day Max",
            ].index(default_filter["series"]),
            key=f"{chart_name}_filter_{filter_index}_series",
        )

        condition = st.selectbox(
            "Condition",
            [
                "Hit or go ABOVE",
                "Hit or go BELOW",
                "Start BELOW and finish ABOVE",
                "Start ABOVE and finish BELOW",
            ],
            index=[
                "Hit or go ABOVE",
                "Hit or go BELOW",
                "Start BELOW and finish ABOVE",
                "Start ABOVE and finish BELOW",
            ].index(default_filter["condition"]),
            key=f"{chart_name}_filter_{filter_index}_condition",
        )

        threshold = st.number_input(
            "Threshold",
            value=float(default_filter["threshold"]),
            step=0.05 if series.startswith("Normalized") else 1.0,
            key=f"{chart_name}_filter_{filter_index}_threshold",
        )

        col_a, col_b = st.columns(2)

        with col_a:
            start_bar = st.number_input(
                "Start bar",
                min_value=1,
                value=int(default_filter["start_bar"]),
                step=1,
                key=f"{chart_name}_filter_{filter_index}_start",
            )

        with col_b:
            end_bar = st.number_input(
                "End bar",
                min_value=1,
                value=int(default_filter["end_bar"]),
                step=1,
                key=f"{chart_name}_filter_{filter_index}_end",
            )

        if end_bar < start_bar:
            st.warning("End bar should be greater than or equal to start bar.")

    return {
        "enabled": enabled,
        "name": name,
        "series": series,
        "condition": condition,
        "threshold": threshold,
        "start_bar": start_bar,
        "end_bar": end_bar,
    }


def build_chart_group(chart_index, strategy_name, start_value, interval_minutes):
    chart_name = st.sidebar.text_input(
        f"Chart {chart_index + 1} name",
        value=f"Chart {chart_index + 1}",
        key=f"chart_{chart_index}_name",
    )

    defaults = strategy_defaults(strategy_name, start_value, interval_minutes)

    num_filters = st.sidebar.number_input(
        f"{chart_name} filters",
        min_value=0,
        max_value=8,
        value=len(defaults),
        step=1,
        key=f"chart_{chart_index}_num_filters",
    )

    filters = []

    for filter_index in range(int(num_filters)):
        default_filter = defaults[filter_index] if filter_index < len(defaults) else defaults[-1]
        filters.append(build_filter(chart_name, filter_index, default_filter))

    return {"name": chart_name, "filters": filters}


def filter_passes(normalized, filt):
    if not filt["enabled"]:
        return True

    start_i = int(filt["start_bar"]) - 1
    end_i = int(filt["end_bar"]) - 1
    series = normalized[filt["series"]]

    if len(series) <= end_i:
        return False

    window = series.iloc[start_i : end_i + 1].dropna()

    if window.empty:
        return False

    threshold = float(filt["threshold"])
    condition = filt["condition"]

    if condition == "Hit or go ABOVE":
        return bool((window >= threshold).any())

    if condition == "Hit or go BELOW":
        return bool((window <= threshold).any())

    if condition == "Start BELOW and finish ABOVE":
        return bool(window.iloc[0] < threshold and window.iloc[-1] > threshold)

    if condition == "Start ABOVE and finish BELOW":
        return bool(window.iloc[0] > threshold and window.iloc[-1] < threshold)

    return False


def analyze_days(session, start_value, chart_groups, logic_mode):
    all_days = []
    chart_results = {
        group["name"]: {"matching_days": [], "filter_rows": [], "max_len": 0}
        for group in chart_groups
    }

    for day, day_data in session.groupby(session.index.date):
        if day_data.empty:
            continue

        day_data = add_indicators(day_data)
        normalized = normalize_day(day_data, start_value)

        day_record = {
            "day": str(day),
            "date": day,
            "data": day_data,
            "normalized": normalized,
        }

        all_days.append(day_record)

        for group in chart_groups:
            active_filters = [f for f in group["filters"] if f["enabled"]]
            filter_results = []

            for filt in active_filters:
                passed = filter_passes(normalized, filt)
                filter_results.append({"filter": filt, "passed": passed})
                chart_results[group["name"]]["filter_rows"].append(
                    {
                        "Date": str(day),
                        "Filter": filt["name"],
                        "Series": filt["series"],
                        "Condition": filt["condition"],
                        "Threshold": filt["threshold"],
                        "Start Bar": filt["start_bar"],
                        "End Bar": filt["end_bar"],
                        "Passed": passed,
                    }
                )

            if not active_filters:
                group_passed = True
            elif logic_mode == "ALL filters must pass":
                group_passed = all(item["passed"] for item in filter_results)
            else:
                group_passed = any(item["passed"] for item in filter_results)

            if group_passed:
                chart_results[group["name"]]["matching_days"].append(day_record)

            chart_results[group["name"]]["max_len"] = max(
                chart_results[group["name"]]["max_len"],
                len(normalized["Normalized Close"]),
            )

    matched_any = {
        item["day"]
        for result in chart_results.values()
        for item in result["matching_days"]
    }
    no_filter_match_days = [item for item in all_days if item["day"] not in matched_any]

    return {
        "all_days": all_days,
        "chart_results": chart_results,
        "no_filter_match_days": no_filter_match_days,
    }


def prepare_all_days(session, start_value):
    all_days = []

    for day, day_data in session.groupby(session.index.date):
        if day_data.empty:
            continue

        day_data = add_indicators(day_data)
        normalized = normalize_day(day_data, start_value)
        all_days.append(
            {
                "day": str(day),
                "date": day,
                "data": day_data,
                "normalized": normalized,
            }
        )

    return all_days


def calculate_outcome(item, lookahead_bars):
    close = item["data"]["Close"]

    if len(close) < 2:
        return None

    end_i = min(len(close) - 1, lookahead_bars - 1)
    return (close.iloc[end_i] - close.iloc[0]) / close.iloc[0] * 100


def get_momentum_fingerprint(item, resample_points, include_volume_shape, focus_momentum_shifts):
    data = item["data"]
    close = data["Close"].astype(float)
    high = data["High"].astype(float)
    low = data["Low"].astype(float)

    if len(close) < 6:
        return None

    base = close.iloc[0]

    if base == 0:
        return None

    normalized_close = (close / base - 1) * 100
    returns = close.pct_change().fillna(0) * 100
    momentum_fast = returns.rolling(3, min_periods=1).mean()
    momentum_slow = returns.rolling(8, min_periods=1).mean()
    momentum_spread = momentum_fast - momentum_slow
    acceleration = momentum_spread.diff().fillna(0)
    typical_price = (high + low + close) / 3
    average_price = typical_price.expanding().mean()
    average_price_distance = (close / average_price - 1) * 100
    range_pct = (high - low) / close * 100

    if "Volume" in data.columns and data["Volume"].max() > 0:
        volume = data["Volume"].astype(float)
        volume_pressure = volume / volume.rolling(20, min_periods=1).median().replace(0, np.nan)
        volume_pressure = volume_pressure.replace([np.inf, -np.inf], np.nan).fillna(1.0)

        if volume.sum() > 0:
            vwap = (typical_price * volume).cumsum() / volume.cumsum().replace(0, np.nan)
            vwap_distance = (close / vwap - 1).replace([np.inf, -np.inf], np.nan).fillna(0) * 100
        else:
            vwap_distance = average_price_distance * 0
    else:
        volume_pressure = pd.Series(np.ones(len(close)), index=close.index)
        vwap_distance = average_price_distance * 0

    old_x = np.linspace(0, 1, len(close))
    new_x = np.linspace(0, 1, resample_points)

    def sample(series):
        clean = pd.Series(series).replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)
        return np.interp(new_x, old_x, clean.to_numpy())

    price_shape = sample(normalized_close)
    price_shape = price_shape - price_shape[0]
    momentum_shape = sample(momentum_spread)
    acceleration_shape = sample(acceleration)
    range_shape = sample(range_pct)
    average_distance_shape = sample(average_price_distance)
    vwap_distance_shape = sample(vwap_distance)
    volume_shape = sample(volume_pressure)

    shift_score = (acceleration.abs() * (1 + volume_pressure.clip(0, 5) / 5)).replace([np.inf, -np.inf], np.nan).fillna(0)
    shift_indices = shift_score.sort_values(ascending=False).head(5).index
    shift_positions = []
    shift_directions = []
    shift_strengths = []

    for shift_index in shift_indices:
        position = data.index.get_loc(shift_index)
        shift_positions.append(position / max(1, len(data) - 1))
        shift_directions.append(np.sign(momentum_spread.iloc[position]))
        shift_strengths.append(float(shift_score.iloc[position]))

    while len(shift_positions) < 5:
        shift_positions.append(0.0)
        shift_directions.append(0.0)
        shift_strengths.append(0.0)

    summary_features = np.array(
        [
            float((close.iloc[-1] / close.iloc[0] - 1) * 100),
            float((high.max() - low.min()) / close.iloc[0] * 100),
            float(momentum_spread.max()),
            float(momentum_spread.min()),
            float(momentum_spread.iloc[-1]),
            float(acceleration.abs().mean()),
            float(volume_pressure.iloc[: max(1, len(volume_pressure) // 3)].mean()),
            float(vwap_distance.iloc[-1]),
            float(average_price_distance.iloc[-1]),
        ]
    )

    if focus_momentum_shifts:
        features = [
            momentum_shape * 2.0,
            acceleration_shape * 2.0,
            average_distance_shape,
            vwap_distance_shape,
            np.array(shift_positions) * 1.5,
            np.array(shift_directions),
            np.array(shift_strengths),
            summary_features,
        ]
    else:
        features = [
            price_shape,
            momentum_shape,
            acceleration_shape,
            range_shape,
            average_distance_shape,
            vwap_distance_shape,
            summary_features,
        ]

    if include_volume_shape:
        features.append(volume_shape)

    return {
        "features": np.concatenate(features),
        "momentum_spread": momentum_spread,
        "acceleration": acceleration,
        "volume_pressure": volume_pressure,
        "average_price_distance": average_price_distance,
        "vwap_distance": vwap_distance,
        "shift_positions": shift_positions,
        "shift_strengths": shift_strengths,
        "shift_directions": shift_directions,
    }


def group_structural_days(
    all_days,
    n_groups,
    resample_points,
    include_volume_shape,
    focus_momentum_shifts,
    separate_outliers=False,
):
    feature_rows = []
    valid_days = []

    for item in all_days:
        fingerprint = get_momentum_fingerprint(
            item=item,
            resample_points=resample_points,
            include_volume_shape=include_volume_shape,
            focus_momentum_shifts=focus_momentum_shifts,
        )

        if fingerprint is None:
            continue

        item["momentum_fingerprint"] = fingerprint
        feature_rows.append(fingerprint["features"])
        valid_days.append(item)

    if len(valid_days) < 2:
        return {"groups": [], "summary": pd.DataFrame(), "all_grouped_days": valid_days}

    scaled_features = StandardScaler().fit_transform(np.vstack(feature_rows))
    outlier_mask = np.zeros(len(valid_days), dtype=bool)

    if separate_outliers and len(valid_days) >= 12:
        center = np.median(scaled_features, axis=0)
        distances = np.linalg.norm(scaled_features - center, axis=1)
        q1, q3 = np.percentile(distances, [25, 75])
        iqr = q3 - q1
        distance_cutoff = q3 + 1.5 * iqr
        percentile_cutoff = np.percentile(distances, 90)
        cutoff = max(distance_cutoff, percentile_cutoff)
        outlier_mask = distances > cutoff

        max_outliers = max(1, int(round(len(valid_days) * 0.15)))

        if outlier_mask.sum() > max_outliers:
            outlier_indices = np.argsort(distances)[-max_outliers:]
            outlier_mask = np.zeros(len(valid_days), dtype=bool)
            outlier_mask[outlier_indices] = True

        if outlier_mask.sum() >= len(valid_days) - 2:
            outlier_mask = np.zeros(len(valid_days), dtype=bool)

    cluster_items = [item for item, is_outlier in zip(valid_days, outlier_mask) if not is_outlier]
    cluster_features = scaled_features[~outlier_mask]

    if len(cluster_items) < 2:
        cluster_items = valid_days
        cluster_features = scaled_features
        outlier_mask = np.zeros(len(valid_days), dtype=bool)

    max_balanced_groups = max(2, len(cluster_items) // 3)
    cluster_count = min(int(n_groups), max_balanced_groups, len(cluster_items))
    labels = KMeans(n_clusters=cluster_count, random_state=42, n_init=10).fit_predict(cluster_features)

    for item, label in zip(cluster_items, labels):
        item["structure_group"] = int(label) + 1

    outlier_members = []

    if separate_outliers and outlier_mask.any():
        outlier_members = [item for item, is_outlier in zip(valid_days, outlier_mask) if is_outlier]

        for item in outlier_members:
            item["structure_group"] = 0

    if separate_outliers and len(cluster_items) >= 12:
        group_sizes = pd.Series(labels).value_counts()
        tiny_labels = set(group_sizes[group_sizes < 2].index)

        if tiny_labels and len(tiny_labels) < len(group_sizes):
            moved_members = [item for item, label in zip(cluster_items, labels) if label in tiny_labels]
            outlier_members.extend(moved_members)
            cluster_items = [item for item, label in zip(cluster_items, labels) if label not in tiny_labels]
            labels = np.array([label for label in labels if label not in tiny_labels])

            for item in moved_members:
                item["structure_group"] = 0

    groups = []

    for label in sorted(set(labels)):
        members = [item for item in cluster_items if item["structure_group"] == int(label) + 1]
        groups.append(make_structural_group(int(label) + 1, members))

    assigned_days = set()
    exclusive_groups = []

    for group in sorted(groups, key=lambda item: item["count"], reverse=True):
        exclusive_members = []

        for member in group["members"]:
            if member["day"] in assigned_days:
                continue

            assigned_days.add(member["day"])
            exclusive_members.append(member)

        if not exclusive_members:
            continue

        exclusive_groups.append(make_structural_group(group["id"], exclusive_members))

    if outlier_members:
        exclusive_groups.append(make_structural_group(0, outlier_members, is_outlier=True))

    groups = sorted(exclusive_groups, key=lambda item: item["id"])

    summary = pd.DataFrame(
        [
            {
                "Group": group["name"],
                "Days": group["count"],
                "Avg Session Return %": round(group["avg_return"], 3),
                "Win Rate %": round(group["win_rate"], 1),
                "Avg Full-Day Range %": round(group["avg_range"], 3),
                "Avg First-Third Move %": round(group["avg_early_move"], 3),
                "Avg Momentum Shift Strength": round(group["avg_shift_strength"], 3),
                "Avg Shift Count": round(group["avg_shift_count"], 1),
            }
            for group in groups
        ]
    )

    return {"groups": groups, "summary": summary, "all_grouped_days": valid_days}


def make_structural_group(group_id, members, is_outlier=False):
    returns = np.array(
        [
            (item["data"]["Close"].iloc[-1] - item["data"]["Close"].iloc[0]) / item["data"]["Close"].iloc[0] * 100
            for item in members
        ]
    )
    ranges = np.array(
        [
            (item["data"]["High"].max() - item["data"]["Low"].min()) / item["data"]["Close"].iloc[0] * 100
            for item in members
        ]
    )
    first_third_returns = []
    momentum_shift_counts = []
    avg_largest_shift_strengths = []

    for item in members:
        close = item["data"]["Close"]
        end_i = max(1, min(len(close) - 1, len(close) // 3))
        first_third_returns.append((close.iloc[end_i] - close.iloc[0]) / close.iloc[0] * 100)
        fingerprint = item.get("momentum_fingerprint", {})
        shift_strengths = fingerprint.get("shift_strengths", [])
        avg_largest_shift_strengths.append(max(shift_strengths) if shift_strengths else 0)
        momentum_shift_counts.append(sum(1 for value in shift_strengths if value > 0))

    avg_return = float(np.mean(returns))
    avg_early_move = float(np.mean(first_third_returns))
    avg_shift_strength = float(np.mean(avg_largest_shift_strengths))

    if is_outlier:
        name = "Outliers"
    elif avg_return > 0.05 and avg_early_move < 0:
        name = f"Structure Group {group_id}: reversal up"
    elif avg_return < -0.05 and avg_early_move > 0:
        name = f"Structure Group {group_id}: reversal down"
    elif avg_return > 0.05:
        name = f"Structure Group {group_id}: momentum up"
    elif avg_return < -0.05:
        name = f"Structure Group {group_id}: momentum down"
    else:
        name = f"Structure Group {group_id}: momentum chop"

    return {
        "id": group_id,
        "name": name,
        "members": members,
        "count": len(members),
        "avg_return": avg_return,
        "win_rate": float((returns > 0).mean() * 100),
        "avg_range": float(np.mean(ranges)),
        "avg_early_move": avg_early_move,
        "avg_shift_strength": avg_shift_strength,
        "avg_shift_count": float(np.mean(momentum_shift_counts)),
        "is_outlier": is_outlier,
    }


def make_structure_outcome_chart(structural_groups):
    rows = []

    for group in structural_groups:
        for item in group["members"]:
            day_return = (item["data"]["Close"].iloc[-1] - item["data"]["Close"].iloc[0]) / item["data"]["Close"].iloc[0] * 100
            rows.append({"Group": group["name"], "Full-Day Return %": day_return, "Date": item["day"]})

    if not rows:
        return go.Figure()

    fig = px.box(
        pd.DataFrame(rows),
        x="Group",
        y="Full-Day Return %",
        points="all",
        hover_data=["Date"],
        title="How Similar Structure Groups Finished",
    )
    fig.add_hline(y=0, line_dash="dash", line_color="#6b7280")
    fig.update_layout(height=500, template="plotly_white", margin=dict(l=40, r=30, t=70, b=120))
    fig.update_xaxes(tickangle=-20)

    return fig


def make_structure_similarity_chart(structural_groups, resample_points):
    rows = []
    labels = []

    for group in structural_groups:
        for item in group["members"]:
            close = item["normalized"]["Normalized Close"].dropna()

            if len(close) < 4:
                continue

            old_x = np.linspace(0, 1, len(close))
            new_x = np.linspace(0, 1, resample_points)
            path = np.interp(new_x, old_x, close.to_numpy())
            rows.append(path - path[0])
            labels.append(f"{group['name']} | {item['day']}")

    fig = go.Figure()

    if not rows:
        fig.update_layout(title="Grouped Day Shape Heatmap", height=420, template="plotly_white")
        return fig

    fig.add_trace(
        go.Heatmap(
            z=np.vstack(rows),
            x=np.arange(1, resample_points + 1),
            y=labels,
            colorscale="RdBu",
            zmid=0,
            colorbar=dict(title="Move"),
            hovertemplate="%{y}<br>Bar: %{x}<br>Move: %{z:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Grouped Day Shape Heatmap",
        xaxis_title="Resampled session path",
        yaxis_title="Grouped days",
        height=max(500, min(1000, 24 * len(labels) + 140)),
        template="plotly_white",
        margin=dict(l=190, r=30, t=70, b=40),
    )

    return fig


def make_group_momentum_chart(group, resample_points):
    rows = []

    for item in group["members"]:
        fingerprint = item.get("momentum_fingerprint")

        if not fingerprint:
            continue

        series = fingerprint["momentum_spread"].replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)
        old_x = np.linspace(0, 1, len(series))
        new_x = np.linspace(0, 1, resample_points)
        rows.append(np.interp(new_x, old_x, series.to_numpy()))

    fig = go.Figure()

    if not rows:
        fig.update_layout(title=f"{group['name']} momentum", height=420, template="plotly_white")
        return fig

    matrix = np.vstack(rows)
    mean = matrix.mean(axis=0)
    p25 = np.percentile(matrix, 25, axis=0)
    p75 = np.percentile(matrix, 75, axis=0)
    x = np.arange(1, resample_points + 1)

    fig.add_trace(
        go.Scatter(
            x=x,
            y=p75,
            mode="lines",
            line=dict(width=0, color="rgba(14, 165, 233, 0)"),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=p25,
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(14, 165, 233, 0.18)",
            line=dict(width=0, color="rgba(14, 165, 233, 0)"),
            name="Middle 50%",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=mean,
            mode="lines",
            line=dict(width=3, color="#0284c7"),
            name="Average momentum",
            hovertemplate="Session position: %{x}<br>Momentum: %{y:.3f}<extra></extra>",
        )
    )
    fig.add_hline(y=0, line_dash="dash", line_color="#6b7280")
    fig.update_layout(
        title=f"{group['name']} momentum pressure",
        xaxis_title="Resampled session path",
        yaxis_title="Fast momentum minus slow momentum",
        height=460,
        template="plotly_white",
        hovermode="x unified",
    )

    return fig


def make_momentum_shift_map(structural_groups):
    rows = []

    for group in structural_groups:
        for item in group["members"]:
            fingerprint = item.get("momentum_fingerprint")

            if not fingerprint:
                continue

            for position, strength, direction in zip(
                fingerprint["shift_positions"],
                fingerprint["shift_strengths"],
                fingerprint["shift_directions"],
            ):
                if strength <= 0:
                    continue

                rows.append(
                    {
                        "Group": group["name"],
                        "Date": item["day"],
                        "Session Position %": position * 100,
                        "Shift Strength": strength,
                        "Direction": "bullish" if direction > 0 else "bearish" if direction < 0 else "neutral",
                    }
                )

    if not rows:
        return go.Figure()

    fig = px.scatter(
        pd.DataFrame(rows),
        x="Session Position %",
        y="Shift Strength",
        color="Group",
        symbol="Direction",
        hover_data=["Date", "Direction"],
        title="Where Momentum Shifts Tend To Appear",
    )
    fig.update_layout(height=520, template="plotly_white", margin=dict(l=40, r=30, t=70, b=40))

    return fig


def group_feedback_signature(ticker, interval, start_time, end_time, group):
    return "|".join(
        [
            str(ticker).upper(),
            str(interval),
            str(start_time),
            str(end_time),
            str(round(group.get("avg_return", 0), 2)),
            str(round(group.get("avg_range", 0), 2)),
            str(round(group.get("avg_early_move", 0), 2)),
            str(round(group.get("avg_shift_strength", 0), 2)),
            str(round(group.get("avg_shift_count", 0), 1)),
        ]
    )


def load_group_feedback():
    if not FEEDBACK_FILE.exists():
        return pd.DataFrame(
            columns=[
                "timestamp",
                "signature",
                "ticker",
                "interval",
                "session",
                "group_name",
                "rating",
                "note",
            ]
        )

    try:
        return pd.read_csv(FEEDBACK_FILE)
    except (OSError, pd.errors.ParserError):
        return pd.DataFrame()


def save_group_feedback(signature, ticker, interval, start_time, end_time, group_name, rating, note):
    feedback = load_group_feedback()
    new_row = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp.now().isoformat(timespec="seconds"),
                "signature": signature,
                "ticker": ticker,
                "interval": interval,
                "session": f"{start_time}-{end_time}",
                "group_name": group_name,
                "rating": rating,
                "note": note,
            }
        ]
    )
    feedback = pd.concat([feedback, new_row], ignore_index=True)
    feedback.to_csv(FEEDBACK_FILE, index=False)


def feedback_score_for_signature(feedback, signature):
    if feedback.empty or "signature" not in feedback.columns:
        return 0

    matches = feedback[feedback["signature"] == signature]

    if matches.empty or "rating" not in matches.columns:
        return 0

    rating_map = {
        "Good pattern": 1,
        "Not useful": -1,
    }
    return int(matches["rating"].map(rating_map).fillna(0).sum())


def cosine_similarity_score(left, right):
    left = np.asarray(left)
    right = np.asarray(right)
    denom = np.linalg.norm(left) * np.linalg.norm(right)

    if denom == 0:
        return 0.0

    return float(np.dot(left, right) / denom)


def build_previous_day_context(group, grouped_days, resample_points):
    sorted_days = sorted(grouped_days, key=lambda item: item["date"])
    day_by_date = {item["date"]: item for item in sorted_days}
    contexts = []

    for item in sorted(group["members"], key=lambda item: item["date"]):
        prior_candidates = [day for day in sorted_days if day["date"] < item["date"]]

        if not prior_candidates:
            continue

        previous_item = prior_candidates[-1]
        previous_fingerprint = previous_item.get("momentum_fingerprint")
        current_fingerprint = item.get("momentum_fingerprint")

        if previous_fingerprint is None:
            previous_fingerprint = get_momentum_fingerprint(
                previous_item,
                resample_points=resample_points,
                include_volume_shape=True,
                focus_momentum_shifts=True,
            )
            previous_item["momentum_fingerprint"] = previous_fingerprint

        if current_fingerprint is None:
            current_fingerprint = get_momentum_fingerprint(
                item,
                resample_points=resample_points,
                include_volume_shape=True,
                focus_momentum_shifts=True,
            )
            item["momentum_fingerprint"] = current_fingerprint

        similarity = None

        if previous_fingerprint is not None and current_fingerprint is not None:
            similarity = cosine_similarity_score(
                previous_fingerprint["features"],
                current_fingerprint["features"],
            )

        previous_return = (previous_item["data"]["Close"].iloc[-1] - previous_item["data"]["Close"].iloc[0]) / previous_item["data"]["Close"].iloc[0] * 100
        current_return = (item["data"]["Close"].iloc[-1] - item["data"]["Close"].iloc[0]) / item["data"]["Close"].iloc[0] * 100

        contexts.append(
            {
                "current": item,
                "previous": previous_item,
                "similarity": similarity,
                "previous_return": previous_return,
                "current_return": current_return,
                "previous_group": previous_item.get("structure_group"),
            }
        )

    return contexts


def make_previous_day_context_table(contexts):
    rows = []

    for context in contexts:
        rows.append(
            {
                "Group Day": context["current"]["day"],
                "Previous Day": context["previous"]["day"],
                "Previous Group": context["previous_group"],
                "Prior vs Group-Day Similarity": round(context["similarity"], 3) if context["similarity"] is not None else None,
                "Previous Day Return %": round(context["previous_return"], 3),
                "Group Day Return %": round(context["current_return"], 3),
            }
        )

    return pd.DataFrame(rows)


def make_previous_day_overlay_chart(group, contexts, start_value, include_group_days=False):
    previous_days = [context["previous"] for context in contexts]
    fig = go.Figure()
    colors = px.colors.qualitative.Dark24

    for idx, item in enumerate(previous_days):
        y = item["normalized"]["Normalized Close"]
        x = list(range(1, len(y) + 1))
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                line=dict(width=2, color=colors[idx % len(colors)]),
                opacity=0.82,
                name=f"Prev {item['day']}",
                hovertemplate="Previous day: %{customdata}<br>Bar: %{x}<br>Close: %{y:.2f}<extra></extra>",
                customdata=[item["day"]] * len(x),
            )
        )

    if include_group_days:
        for idx, context in enumerate(contexts):
            item = context["current"]
            y = item["normalized"]["Normalized Close"]
            x = list(range(1, len(y) + 1))
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    line=dict(width=2, color=colors[idx % len(colors)], dash="dash"),
                    opacity=0.55,
                    name=f"Group {item['day']}",
                    legendgroup="Group days",
                    hovertemplate="Group day: %{customdata}<br>Bar: %{x}<br>Close: %{y:.2f}<extra></extra>",
                    customdata=[item["day"]] * len(x),
                )
            )

    if previous_days:
        max_len = max(len(item["normalized"]["Normalized Close"]) for item in previous_days)
        fig.add_shape(
            type="line",
            x0=1,
            x1=max_len,
            y0=start_value,
            y1=start_value,
            line=dict(width=1, dash="dash", color="#6b7280"),
        )

    fig.update_layout(
        title=f"Previous-day overlays before {group['name']}",
        xaxis_title="Bar inside previous session",
        yaxis_title=f"Normalized price, start = {start_value}",
        height=560,
        template="plotly_white",
        hovermode="closest",
    )

    return fig


def make_previous_to_current_scatter(contexts):
    rows = []

    for context in contexts:
        similarity = context["similarity"] if context["similarity"] is not None else 0
        rows.append(
            {
                "Previous Day Return %": context["previous_return"],
                "Group Day Return %": context["current_return"],
                "Similarity": similarity,
                "Marker Size": max(0.05, similarity + 1),
                "Group Day": context["current"]["day"],
                "Previous Day": context["previous"]["day"],
            }
        )

    if not rows:
        return go.Figure()

    fig = px.scatter(
        pd.DataFrame(rows),
        x="Previous Day Return %",
        y="Group Day Return %",
        size="Marker Size",
        color="Similarity",
        hover_data=["Previous Day", "Group Day", "Similarity"],
        title="Did The Previous Day Lead Into This Group?",
        color_continuous_scale="Viridis",
    )
    fig.add_hline(y=0, line_dash="dash", line_color="#6b7280")
    fig.add_vline(x=0, line_dash="dash", line_color="#6b7280")
    fig.update_layout(height=500, template="plotly_white")

    return fig


def discover_segment_patterns(
    all_days,
    setup_bars,
    confirmation_bars,
    scan_start_bar,
    scan_end_bar,
    min_volume_pct,
    min_abs_setup_move,
    n_clusters,
    samples_per_segment=16,
):
    candidates = []
    feature_rows = []

    for item in all_days:
        data = item["data"]

        if len(data) < setup_bars + confirmation_bars + 1:
            continue

        day_volume_median = data["Volume"].median() if "Volume" in data.columns else np.nan
        last_start = min(scan_end_bar - setup_bars, len(data) - setup_bars - confirmation_bars)
        first_start = max(0, scan_start_bar - 1)

        if last_start < first_start:
            continue

        for start_i in range(first_start, last_start + 1):
            setup = data.iloc[start_i : start_i + setup_bars]
            future = data.iloc[start_i + setup_bars : start_i + setup_bars + confirmation_bars]

            if setup.empty or future.empty:
                continue

            base_price = setup["Close"].iloc[0]

            if base_price == 0:
                continue

            close_norm = (setup["Close"] / base_price - 1) * 100
            high_norm = (setup["High"] / base_price - 1) * 100
            low_norm = (setup["Low"] / base_price - 1) * 100
            setup_move = float(close_norm.iloc[-1] - close_norm.iloc[0])
            setup_range = float(high_norm.max() - low_norm.min())
            max_up = float(high_norm.max())
            max_down = float(low_norm.min())
            confirmation_return = float((future["Close"].iloc[-1] - setup["Close"].iloc[-1]) / setup["Close"].iloc[-1] * 100)
            confirmation_max_up = float((future["High"].max() - setup["Close"].iloc[-1]) / setup["Close"].iloc[-1] * 100)
            confirmation_max_down = float((future["Low"].min() - setup["Close"].iloc[-1]) / setup["Close"].iloc[-1] * 100)

            if "Volume" in setup.columns and pd.notna(day_volume_median) and day_volume_median > 0:
                volume_ratio = float(setup["Volume"].mean() / day_volume_median)
            else:
                volume_ratio = 0.0

            if volume_ratio < min_volume_pct / 100:
                continue

            if abs(setup_move) < min_abs_setup_move:
                continue

            old_x = np.linspace(0, 1, len(close_norm))
            new_x = np.linspace(0, 1, samples_per_segment)
            shape = np.interp(new_x, old_x, close_norm.to_numpy())
            shape = shape - shape[0]

            slope = np.polyfit(np.arange(len(close_norm)), close_norm.to_numpy(), 1)[0] if len(close_norm) > 1 else 0
            rsi_mean = float(setup["RSI"].mean()) if "RSI" in setup.columns and setup["RSI"].notna().any() else 50.0

            candidate = {
                "day": item["day"],
                "item": item,
                "start_i": start_i,
                "end_i": start_i + setup_bars - 1,
                "future_end_i": start_i + setup_bars + confirmation_bars - 1,
                "start_bar": start_i + 1,
                "end_bar": start_i + setup_bars,
                "future_end_bar": start_i + setup_bars + confirmation_bars,
                "setup_move": setup_move,
                "setup_range": setup_range,
                "max_up": max_up,
                "max_down": max_down,
                "volume_ratio": volume_ratio,
                "confirmation_return": confirmation_return,
                "confirmation_max_up": confirmation_max_up,
                "confirmation_max_down": confirmation_max_down,
                "rsi_mean": rsi_mean,
            }
            candidates.append(candidate)
            feature_rows.append(
                np.concatenate(
                    [
                        shape,
                        np.array(
                            [
                                setup_move,
                                setup_range,
                                max_up,
                                max_down,
                                volume_ratio,
                                slope,
                                rsi_mean / 100,
                            ]
                        ),
                    ]
                )
            )

    if len(candidates) < 2:
        return {"candidates": candidates, "clusters": [], "summary": pd.DataFrame()}

    cluster_count = min(int(n_clusters), len(candidates))
    scaled_features = StandardScaler().fit_transform(np.vstack(feature_rows))
    labels = KMeans(n_clusters=cluster_count, random_state=42, n_init=10).fit_predict(scaled_features)

    for candidate, label in zip(candidates, labels):
        candidate["cluster"] = int(label) + 1

    clusters = []

    for cluster_id in sorted(set(labels)):
        members = [candidate for candidate in candidates if candidate["cluster"] == int(cluster_id) + 1]
        confirmation_values = np.array([member["confirmation_return"] for member in members])
        setup_values = np.array([member["setup_move"] for member in members])
        volume_values = np.array([member["volume_ratio"] for member in members])
        unique_days = sorted(set(member["day"] for member in members))

        direction = "up" if np.mean(setup_values) >= 0 else "down"
        confirmation_direction = "up" if np.mean(confirmation_values) >= 0 else "down"

        clusters.append(
            {
                "id": int(cluster_id) + 1,
                "name": f"Pattern {int(cluster_id) + 1}: setup {direction}, confirms {confirmation_direction}",
                "members": members,
                "unique_days": unique_days,
                "count": len(members),
                "unique_day_count": len(unique_days),
                "avg_setup_move": float(np.mean(setup_values)),
                "avg_volume_ratio": float(np.mean(volume_values)),
                "avg_confirmation_return": float(np.mean(confirmation_values)),
                "confirmation_win_rate": float((confirmation_values > 0).mean() * 100),
                "avg_range": float(np.mean([member["setup_range"] for member in members])),
            }
        )

    summary = pd.DataFrame(
        [
            {
                "Pattern": cluster["name"],
                "Segments": cluster["count"],
                "Unique Days": cluster["unique_day_count"],
                "Avg Setup Move %": round(cluster["avg_setup_move"], 3),
                "Avg Volume Ratio": round(cluster["avg_volume_ratio"], 2),
                "Avg Confirmation %": round(cluster["avg_confirmation_return"], 3),
                "Confirmation Win Rate %": round(cluster["confirmation_win_rate"], 1),
                "Avg Setup Range %": round(cluster["avg_range"], 3),
            }
            for cluster in clusters
        ]
    )

    return {"candidates": candidates, "clusters": clusters, "summary": summary}


def make_segment_signature_chart(cluster, setup_bars, confirmation_bars):
    fig = go.Figure()
    setup_rows = []
    future_rows = []

    for member in cluster["members"]:
        data = member["item"]["data"]
        setup = data.iloc[member["start_i"] : member["end_i"] + 1]
        future = data.iloc[member["end_i"] + 1 : member["future_end_i"] + 1]

        if setup.empty or future.empty:
            continue

        base = setup["Close"].iloc[0]
        setup_path = (setup["Close"] / base - 1) * 100
        future_path = (future["Close"] / setup["Close"].iloc[-1] - 1) * 100
        setup_rows.append(setup_path.to_numpy())
        future_rows.append(future_path.to_numpy())

    if not setup_rows:
        fig.update_layout(title=cluster["name"], height=520, template="plotly_white")
        return fig

    setup_matrix = np.vstack(setup_rows)
    future_matrix = np.vstack(future_rows) if future_rows else np.empty((0, 0))
    setup_x = np.arange(1, setup_bars + 1)
    future_x = np.arange(setup_bars + 1, setup_bars + confirmation_bars + 1)
    setup_mean = np.mean(setup_matrix, axis=0)
    setup_p25 = np.percentile(setup_matrix, 25, axis=0)
    setup_p75 = np.percentile(setup_matrix, 75, axis=0)

    fig.add_trace(
        go.Scatter(
            x=setup_x,
            y=setup_p75,
            mode="lines",
            line=dict(width=0, color="rgba(37, 99, 235, 0)"),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=setup_x,
            y=setup_p25,
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(37, 99, 235, 0.18)",
            line=dict(width=0, color="rgba(37, 99, 235, 0)"),
            name="Setup middle 50%",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=setup_x,
            y=setup_mean,
            mode="lines",
            line=dict(width=3, color="#2563eb"),
            name="Average setup",
            hovertemplate="Setup bar: %{x}<br>Move: %{y:.2f}%<extra></extra>",
        )
    )

    if future_matrix.size:
        future_mean = np.mean(future_matrix, axis=0) + setup_mean[-1]
        future_p25 = np.percentile(future_matrix, 25, axis=0) + setup_mean[-1]
        future_p75 = np.percentile(future_matrix, 75, axis=0) + setup_mean[-1]
        fig.add_trace(
            go.Scatter(
                x=future_x,
                y=future_p75,
                mode="lines",
                line=dict(width=0, color="rgba(22, 163, 74, 0)"),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=future_x,
                y=future_p25,
                mode="lines",
                fill="tonexty",
                fillcolor="rgba(22, 163, 74, 0.18)",
                line=dict(width=0, color="rgba(22, 163, 74, 0)"),
                name="Confirmation middle 50%",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=future_x,
                y=future_mean,
                mode="lines",
                line=dict(width=3, color="#16a34a"),
                name="Average confirmation",
                hovertemplate="Future bar: %{x}<br>Move: %{y:.2f}%<extra></extra>",
            )
        )

    fig.add_vrect(
        x0=setup_bars + 0.5,
        x1=setup_bars + confirmation_bars + 0.5,
        fillcolor="#16a34a",
        opacity=0.06,
        line_width=0,
    )
    fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="#6b7280")
    fig.update_layout(
        title=cluster["name"],
        xaxis_title="Bars inside detected segment",
        yaxis_title="Move from setup start (%)",
        height=540,
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=40, r=30, t=70, b=40),
    )

    return fig


def make_segment_heatmap(cluster, setup_bars, confirmation_bars):
    rows = []
    labels = []
    valid_members = []

    for member in cluster["members"]:
        data = member["item"]["data"]
        segment = data.iloc[member["start_i"] : member["future_end_i"] + 1]

        if len(segment) < setup_bars + confirmation_bars:
            continue

        base = segment["Close"].iloc[0]
        rows.append(((segment["Close"] / base - 1) * 100).to_numpy())
        labels.append(f"{member['day']} bar {member['start_bar']}")
        valid_members.append(member)

    fig = go.Figure()

    if not rows:
        fig.update_layout(title=f"{cluster['name']} heatmap", height=420, template="plotly_white")
        return fig

    matrix = np.vstack(rows)
    order = np.argsort([member["confirmation_return"] for member in valid_members])[::-1]
    matrix = matrix[order]
    labels = [labels[index] for index in order]

    fig.add_trace(
        go.Heatmap(
            z=matrix,
            x=np.arange(1, setup_bars + confirmation_bars + 1),
            y=labels,
            colorscale="RdBu",
            zmid=0,
            colorbar=dict(title="Move %"),
            hovertemplate="%{y}<br>Bar: %{x}<br>Move: %{z:.2f}%<extra></extra>",
        )
    )
    fig.add_vline(x=setup_bars + 0.5, line_dash="dash", line_color="#111827")
    fig.update_layout(
        title=f"{cluster['name']} member segments",
        xaxis_title="Setup bars, then confirmation bars",
        yaxis_title="Detected windows",
        height=max(420, min(900, 28 * len(labels) + 130)),
        template="plotly_white",
        margin=dict(l=120, r=30, t=70, b=40),
    )

    return fig


def make_segment_location_chart(clusters):
    rows = []

    for cluster in clusters:
        for member in cluster["members"]:
            rows.append(
                {
                    "Pattern": cluster["name"],
                    "Start Bar": member["start_bar"],
                    "Confirmation Return %": member["confirmation_return"],
                    "Volume Ratio": member["volume_ratio"],
                    "Day": member["day"],
                }
            )

    if not rows:
        return go.Figure()

    df = pd.DataFrame(rows)
    fig = px.scatter(
        df,
        x="Start Bar",
        y="Confirmation Return %",
        color="Pattern",
        size="Volume Ratio",
        hover_data=["Day", "Volume Ratio"],
        title="Where Automatic Patterns Appear",
    )
    fig.add_hline(y=0, line_dash="dash", line_color="#6b7280")
    fig.update_layout(height=500, template="plotly_white", margin=dict(l=40, r=30, t=70, b=40))

    return fig


def make_detected_segment_detail_chart(member, ticker):
    data = member["item"]["data"]
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.55, 0.25, 0.20],
        subplot_titles=(
            f"{ticker} {member['day']} detected setup",
            "Close move from setup start",
            "Volume",
        ),
    )
    segment = data.iloc[member["start_i"] : member["future_end_i"] + 1]
    setup = data.iloc[member["start_i"] : member["end_i"] + 1]
    confirmation = data.iloc[member["end_i"] + 1 : member["future_end_i"] + 1]
    base = setup["Close"].iloc[0]

    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"],
            name="Candles",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=segment.index,
            y=(segment["Close"] / base - 1) * 100,
            mode="lines+markers",
            line=dict(width=2, color="#2563eb"),
            name="Segment move",
        ),
        row=2,
        col=1,
    )

    if "Volume" in data.columns:
        fig.add_trace(
            go.Bar(x=data.index, y=data["Volume"], name="Volume", opacity=0.7),
            row=3,
            col=1,
        )

    for row in [1, 2, 3]:
        fig.add_vrect(
            x0=setup.index[0],
            x1=setup.index[-1],
            fillcolor="#2563eb",
            opacity=0.10,
            line_width=0,
            row=row,
            col=1,
        )

        if not confirmation.empty:
            fig.add_vrect(
                x0=confirmation.index[0],
                x1=confirmation.index[-1],
                fillcolor="#16a34a",
                opacity=0.10,
                line_width=0,
                row=row,
                col=1,
            )

    fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="#6b7280", row=2, col=1)
    fig.update_layout(
        title=(
            f"Setup bars {member['start_bar']}-{member['end_bar']} | "
            f"confirmation {member['confirmation_return']:.2f}%"
        ),
        height=860,
        template="plotly_white",
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
    )
    fig.update_xaxes(rangeslider_visible=False, showspikes=True, spikemode="across")

    return fig


def make_pattern_chart(
    chart_name,
    ticker,
    matching_days,
    background_days,
    filters,
    start_value,
    show_background,
    show_individual_legend,
):
    fig = go.Figure()
    colors = px.colors.qualitative.Dark24

    max_len = 0

    for item in matching_days + background_days:
        max_len = max(max_len, len(item["normalized"]["Normalized Close"]))

    if show_background:
        for item in background_days:
            y = item["normalized"]["Normalized Close"]
            x = list(range(1, len(y) + 1))
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    line=dict(width=1, color="#8a8f98"),
                    opacity=0.10,
                    name="Background days",
                    legendgroup="Background",
                    showlegend=False,
                    hovertemplate="Date: %{customdata}<br>Bar: %{x}<br>Close: %{y:.2f}<extra></extra>",
                    customdata=[item["day"]] * len(x),
                )
            )

    for idx, item in enumerate(matching_days):
        y = item["normalized"]["Normalized Close"]
        x = list(range(1, len(y) + 1))
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                line=dict(width=2, color=colors[idx % len(colors)]),
                opacity=0.88,
                name=item["day"] if show_individual_legend else "Matching days",
                legendgroup="Matching days",
                showlegend=show_individual_legend or idx == 0,
                hovertemplate="Date: %{customdata}<br>Bar: %{x}<br>Close: %{y:.2f}<extra></extra>",
                customdata=[item["day"]] * len(x),
            )
        )

    if max_len > 0:
        fig.add_shape(
            type="line",
            x0=1,
            x1=max_len,
            y0=start_value,
            y1=start_value,
            line=dict(width=1, dash="dash", color="#4b5563"),
        )

    for idx, filt in enumerate([f for f in filters if f["enabled"]]):
        if filt["series"].startswith("Normalized"):
            fig.add_shape(
                type="line",
                x0=int(filt["start_bar"]),
                x1=int(filt["end_bar"]),
                y0=float(filt["threshold"]),
                y1=float(filt["threshold"]),
                line=dict(width=2, dash="dot", color=colors[idx % len(colors)]),
            )

    fig.update_layout(
        title=f"{chart_name} - {ticker}",
        xaxis_title="Bar inside session",
        yaxis_title=f"Normalized price, start = {start_value}",
        height=560,
        template="plotly_white",
        hovermode="closest",
        margin=dict(l=40, r=30, t=70, b=40),
    )
    fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor")
    fig.update_yaxes(showspikes=True, spikemode="across", spikesnap="cursor")

    return fig


def resample_normalized_close(days, points=80):
    rows = []
    labels = []

    for item in days:
        series = item["normalized"]["Normalized Close"].dropna()

        if len(series) < 2:
            continue

        old_x = np.linspace(0, 1, len(series))
        new_x = np.linspace(0, 1, points)
        rows.append(np.interp(new_x, old_x, series.to_numpy()))
        labels.append(item["day"])

    if not rows:
        return np.array([]), [], np.array([])

    return np.arange(1, points + 1), labels, np.vstack(rows)


def make_signature_chart(chart_name, ticker, matching_days, compare_days, start_value):
    x, _, matrix = resample_normalized_close(matching_days)
    compare_x, _, compare_matrix = resample_normalized_close(compare_days)
    fig = go.Figure()

    if matrix.size == 0:
        fig.update_layout(
            title=f"{chart_name} - {ticker}",
            height=520,
            template="plotly_white",
        )
        return fig

    p25 = np.percentile(matrix, 25, axis=0)
    p50 = np.percentile(matrix, 50, axis=0)
    p75 = np.percentile(matrix, 75, axis=0)
    mean = np.mean(matrix, axis=0)

    if compare_matrix.size:
        compare_mean = np.mean(compare_matrix, axis=0)
        fig.add_trace(
            go.Scatter(
                x=compare_x,
                y=compare_mean,
                mode="lines",
                line=dict(width=2, color="#9ca3af", dash="dash"),
                name="All other days average",
                hovertemplate="Bar: %{x}<br>Average: %{y:.2f}<extra></extra>",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=p75,
            mode="lines",
            line=dict(width=0, color="rgba(37, 99, 235, 0)"),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=p25,
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(37, 99, 235, 0.18)",
            line=dict(width=0, color="rgba(37, 99, 235, 0)"),
            name="Middle 50%",
            hovertemplate="Bar: %{x}<br>25th pct: %{y:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=mean,
            mode="lines",
            line=dict(width=3, color="#2563eb"),
            name="Matching days average",
            hovertemplate="Bar: %{x}<br>Average: %{y:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=p50,
            mode="lines",
            line=dict(width=2, color="#111827", dash="dot"),
            name="Matching days median",
            hovertemplate="Bar: %{x}<br>Median: %{y:.2f}<extra></extra>",
        )
    )
    fig.add_shape(
        type="line",
        x0=1,
        x1=int(x[-1]),
        y0=start_value,
        y1=start_value,
        line=dict(width=1, dash="dash", color="#4b5563"),
    )
    fig.update_layout(
        title=f"{chart_name} signature - {ticker}",
        xaxis_title="Percent-of-session bars, resampled",
        yaxis_title=f"Normalized price, start = {start_value}",
        height=560,
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=40, r=30, t=70, b=40),
    )

    return fig


def make_heatmap_chart(chart_name, ticker, days, lookahead_bars):
    x, labels, matrix = resample_normalized_close(days)
    fig = go.Figure()

    if matrix.size == 0:
        fig.update_layout(
            title=f"{chart_name} heatmap - {ticker}",
            height=520,
            template="plotly_white",
        )
        return fig

    outcomes = [calculate_outcome(item, lookahead_bars) for item in days if item["day"] in labels]
    order = np.argsort([value if value is not None else 0 for value in outcomes])[::-1]
    ordered_matrix = matrix[order] - matrix[order][:, [0]]
    ordered_labels = [labels[i] for i in order]

    fig.add_trace(
        go.Heatmap(
            z=ordered_matrix,
            x=x,
            y=ordered_labels,
            colorscale="RdBu",
            zmid=0,
            colorbar=dict(title="Move from start"),
            hovertemplate="Date: %{y}<br>Bar: %{x}<br>Move: %{z:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"{chart_name} heatmap - {ticker}",
        xaxis_title="Percent-of-session bars, resampled",
        yaxis_title="Days, sorted by outcome",
        height=max(420, min(900, 28 * len(ordered_labels) + 120)),
        template="plotly_white",
        margin=dict(l=90, r=30, t=70, b=40),
    )

    return fig


def make_small_multiples_chart(chart_name, ticker, days, start_value, lookahead_bars, max_days):
    selected_days = days[:max_days]
    cols = 3
    rows = max(1, int(np.ceil(len(selected_days) / cols)))
    titles = []

    for item in selected_days:
        outcome = calculate_outcome(item, lookahead_bars)
        outcome_text = f"{outcome:.2f}%" if outcome is not None else "N/A"
        titles.append(f"{item['day']} | {outcome_text}")

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=titles,
        vertical_spacing=0.08,
        horizontal_spacing=0.05,
    )

    for idx, item in enumerate(selected_days):
        row = idx // cols + 1
        col = idx % cols + 1
        y = item["normalized"]["Normalized Close"]
        x = list(range(1, len(y) + 1))
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                line=dict(width=2, color="#2563eb"),
                showlegend=False,
                hovertemplate="Bar: %{x}<br>Close: %{y:.2f}<extra></extra>",
            ),
            row=row,
            col=col,
        )
        fig.add_hline(
            y=start_value,
            line_width=1,
            line_dash="dash",
            line_color="#9ca3af",
            row=row,
            col=col,
        )

    fig.update_layout(
        title=f"{chart_name} small multiples - {ticker}",
        height=max(420, rows * 260),
        template="plotly_white",
        margin=dict(l=40, r=30, t=80, b=40),
    )
    fig.update_xaxes(showticklabels=False)

    return fig


def make_outcome_distribution_chart(chart_results, no_filter_match_days, lookahead_bars):
    rows = []

    for chart_name, result in chart_results.items():
        for item in result["matching_days"]:
            outcome = calculate_outcome(item, lookahead_bars)

            if outcome is not None:
                rows.append({"Group": chart_name, "Outcome Return %": outcome})

    for item in no_filter_match_days:
        outcome = calculate_outcome(item, lookahead_bars)

        if outcome is not None:
            rows.append({"Group": "No-filter-match days", "Outcome Return %": outcome})

    if not rows:
        return go.Figure()

    outcome_df = pd.DataFrame(rows)
    fig = px.box(
        outcome_df,
        x="Group",
        y="Outcome Return %",
        points="all",
        title="Outcome Distribution by Pattern Group",
    )
    fig.update_layout(
        height=500,
        template="plotly_white",
        margin=dict(l=40, r=30, t=70, b=110),
    )
    fig.update_xaxes(tickangle=-20)

    return fig


def make_detail_chart(item, ticker, start_value):
    day_data = item["data"]
    normalized = item["normalized"]

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.52, 0.30, 0.18],
        subplot_titles=(
            f"{ticker} {item['day']} candles",
            "Normalized movement",
            "Volume",
        ),
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
            y=normalized["Normalized Close"],
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
            y=normalized["Normalized High"],
            mode="lines",
            name="Normalized High",
            line=dict(width=1, dash="dot"),
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=day_data.index,
            y=normalized["Normalized Low"],
            mode="lines",
            name="Normalized Low",
            line=dict(width=1, dash="dot"),
        ),
        row=2,
        col=1,
    )

    if "Volume" in day_data.columns:
        fig.add_trace(
            go.Bar(x=day_data.index, y=day_data["Volume"], name="Volume", opacity=0.7),
            row=3,
            col=1,
        )

    fig.update_layout(
        title=f"Individual Day View - {ticker} - {item['day']}",
        height=850,
        template="plotly_white",
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
    )
    fig.update_xaxes(rangeslider_visible=False, showspikes=True, spikemode="across")
    fig.update_yaxes(fixedrange=False)

    return fig


def make_summary_table(chart_results, no_filter_match_days, all_days, lookahead_bars):
    rows = []

    for chart_name, result in chart_results.items():
        outcomes = [
            calculate_outcome(item, lookahead_bars)
            for item in result["matching_days"]
        ]
        outcomes = [value for value in outcomes if value is not None]
        rows.append(
            {
                "Chart": chart_name,
                "Matching Days": len(result["matching_days"]),
                "Percent of Days": round(len(result["matching_days"]) / max(1, len(all_days)) * 100, 1),
                "Avg Session Move %": round(np.mean(outcomes), 3) if outcomes else None,
                "Win Rate %": round((np.array(outcomes) > 0).mean() * 100, 1) if outcomes else None,
            }
        )

    no_match_outcomes = [
        calculate_outcome(item, lookahead_bars)
        for item in no_filter_match_days
    ]
    no_match_outcomes = [value for value in no_match_outcomes if value is not None]
    rows.append(
        {
            "Chart": "No-filter-match days",
            "Matching Days": len(no_filter_match_days),
            "Percent of Days": round(len(no_filter_match_days) / max(1, len(all_days)) * 100, 1),
            "Avg Session Move %": round(np.mean(no_match_outcomes), 3) if no_match_outcomes else None,
            "Win Rate %": round((np.array(no_match_outcomes) > 0).mean() * 100, 1) if no_match_outcomes else None,
        }
    )

    return pd.DataFrame(rows)


def pick_day(label, days):
    if not days:
        st.info("No days in this group yet.")
        return None

    dates = [item["day"] for item in days]
    selected_date = st.selectbox(label, dates)
    return days[dates.index(selected_date)]


# ======================
# SIDEBAR
# ======================


st.sidebar.header("Data")

data_source = st.sidebar.selectbox(
    "Data source",
    ["Downloaded historical data", "Yahoo Finance", "Financial Modeling Prep"],
    index=0,
)

today = pd.Timestamp.today().date()
one_year_ago = (pd.Timestamp.today() - pd.DateOffset(years=1)).date()

fmp_api_key = ""
period = None

if data_source == "Financial Modeling Prep":
    fmp_api_key = st.sidebar.text_input("FMP API key", type="password")
    from_date = one_year_ago
    to_date = today
    st.sidebar.write(f"FMP date range: **{from_date} to {to_date}**")
elif data_source == "Yahoo Finance":
    from_date = None
    to_date = None
    period = st.sidebar.selectbox(
        "Yahoo period",
        ["5d", "1mo", "2mo", "60d", "6mo", "1y", "2y"],
        index=3,
    )
else:
    from_date = None
    to_date = None
    downloaded_tickers = list_downloaded_tickers("5m")

    if downloaded_tickers:
        st.sidebar.write(f"Downloaded tickers: **{len(downloaded_tickers)}**")
    else:
        st.sidebar.warning("No downloaded 5m data found. Run exportfmp.py first.")

st.sidebar.header("Market")

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
        index=2,
    )
interval_minutes = get_interval_minutes(interval)

start_time = st.sidebar.text_input("Session start", value="09:30")
end_time = st.sidebar.text_input("Session end", value="16:00")
start_value = st.sidebar.number_input("Normalized start", value=100.0, step=0.25)

if "structural_day_group_mode" not in st.session_state:
    st.session_state.structural_day_group_mode = False

if st.sidebar.button("Group Similar Days Automatically"):
    st.session_state.structural_day_group_mode = True

if st.session_state.structural_day_group_mode:
    st.sidebar.success("Structural grouping mode is on.")

    if st.sidebar.button("Show manual chart filters"):
        st.session_state.structural_day_group_mode = False
        st.rerun()

structural_day_group_mode = st.session_state.structural_day_group_mode

chart_groups = []
logic_mode = "ALL filters must pass"
show_background_days = False
show_individual_legend = False
show_no_filter_chart = False
pattern_view = "Signature band"
lookahead_minutes = 120
lookahead_bars = max(1, int(round(lookahead_minutes / interval_minutes)))
structural_groups_count = 5
structural_resample_points = 80
structural_include_volume = True
structural_focus_momentum = True
structural_separate_outliers = True

if structural_day_group_mode:
    st.sidebar.header("Structural Day Grouping")
    structural_groups_count = st.sidebar.slider(
        "Number of similar-day groups",
        min_value=4,
        max_value=12,
        value=8,
    )
    structural_resample_points = st.sidebar.slider(
        "Shape detail",
        min_value=30,
        max_value=140,
        value=80,
        step=10,
    )
    structural_include_volume = st.sidebar.checkbox(
        "Include volume shape in grouping",
        value=True,
    )
    structural_focus_momentum = st.sidebar.checkbox(
        "Focus grouping on momentum shifts",
        value=True,
    )
    structural_separate_outliers = st.sidebar.checkbox(
        "Put unusual days in Outliers group",
        value=True,
    )
    run_button = st.sidebar.button("Run Similar-Day Grouping")
else:
    st.sidebar.header("Strategy")

    strategy_name = st.sidebar.selectbox(
        "Starter strategy",
        ["Custom", "Opening Drive Up", "Opening Drive Down", "Reversal Watch"],
        index=1,
    )

    logic_mode = st.sidebar.radio(
        "Filters inside each chart",
        ["ALL filters must pass", "ANY filter can pass"],
        index=0,
    )

    num_charts = st.sidebar.number_input(
        "Number of pattern charts",
        min_value=1,
        max_value=6,
        value=2,
        step=1,
    )

    for chart_index in range(int(num_charts)):
        st.sidebar.divider()
        chart_groups.append(
            build_chart_group(chart_index, strategy_name, start_value, interval_minutes)
        )

    st.sidebar.header("Display")

    show_background_days = st.sidebar.checkbox(
        "Show non-matching days faded behind each chart",
        value=False,
    )
    show_individual_legend = st.sidebar.checkbox(
        "Show each matching day in legends",
        value=False,
    )
    show_no_filter_chart = st.sidebar.checkbox(
        "Add chart for days that match no filters",
        value=True,
    )
    pattern_view = st.sidebar.selectbox(
        "Pattern visualization",
        ["Signature band", "Heatmap", "Overlay lines"],
        index=0,
    )
    lookahead_minutes = st.sidebar.number_input(
        "Outcome window minutes",
        min_value=1,
        max_value=390,
        value=120,
        step=5,
    )
    lookahead_bars = max(1, int(round(lookahead_minutes / interval_minutes)))

    run_button = st.sidebar.button("Run Pattern Scan")


# ======================
# RUN
# ======================


if "mach2_results" not in st.session_state:
    st.session_state.mach2_results = None

if run_button:
    df = download_data(
        ticker=ticker,
        period=period,
        interval=interval,
        from_date=from_date,
        to_date=to_date,
        data_source=data_source,
        fmp_api_key=fmp_api_key,
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

    structural_results = None

    if structural_day_group_mode:
        all_days = prepare_all_days(session, start_value)
        structural_results = group_structural_days(
            all_days=all_days,
            n_groups=structural_groups_count,
            resample_points=structural_resample_points,
            include_volume_shape=structural_include_volume,
            focus_momentum_shifts=structural_focus_momentum,
            separate_outliers=structural_separate_outliers,
        )
        results = {
            "all_days": all_days,
            "chart_results": {},
            "no_filter_match_days": [],
        }
    else:
        results = analyze_days(
            session=session,
            start_value=start_value,
            chart_groups=chart_groups,
            logic_mode=logic_mode,
        )

    st.session_state.mach2_results = {
        "results": results,
        "auto_patterns": None,
        "structural_results": structural_results,
        "structural_day_group_mode": structural_day_group_mode,
        "ticker": ticker,
        "interval": interval,
        "start_time": start_time,
        "end_time": end_time,
        "data_source": data_source,
        "from_date": from_date,
        "to_date": to_date,
        "start_value": start_value,
        "chart_groups": chart_groups,
        "logic_mode": logic_mode,
        "lookahead_bars": lookahead_bars,
        "lookahead_minutes": lookahead_minutes,
        "show_background_days": show_background_days,
        "show_individual_legend": show_individual_legend,
        "show_no_filter_chart": show_no_filter_chart,
        "pattern_view": pattern_view,
        "structural_groups_count": structural_groups_count,
        "structural_resample_points": structural_resample_points,
        "structural_include_volume": structural_include_volume,
        "structural_focus_momentum": structural_focus_momentum,
        "structural_separate_outliers": structural_separate_outliers,
    }


if st.session_state.mach2_results is None:
    st.info("Choose a market, select or customize a strategy, then run the pattern scan.")
    st.markdown(
        """
        This version is built around a simpler question: when a day matches a strategy setup, what did similar days actually do next?
        """
    )
    st.stop()

saved = st.session_state.mach2_results
results = saved["results"]
structural_results = saved.get("structural_results")
saved_structural_mode = saved.get("structural_day_group_mode", False)
ticker = saved["ticker"]
saved_interval = saved.get("interval", interval)
saved_start_time = saved.get("start_time", start_time)
saved_end_time = saved.get("end_time", end_time)
start_value = saved["start_value"]
chart_groups = saved["chart_groups"]
lookahead_bars = saved["lookahead_bars"]
lookahead_minutes = saved["lookahead_minutes"]
show_background_days = saved["show_background_days"]
show_individual_legend = saved["show_individual_legend"]
show_no_filter_chart = saved["show_no_filter_chart"]
pattern_view = saved.get("pattern_view", "Signature band")
structural_resample_points = saved.get("structural_resample_points", 80)

all_days = results["all_days"]
chart_results = results["chart_results"]
no_filter_match_days = results["no_filter_match_days"]

chart_config = {
    "scrollZoom": True,
    "displaylogo": False,
    "modeBarButtonsToAdd": ["drawline", "drawopenpath", "eraseshape"],
}

if saved_structural_mode:
    st.subheader("Similar Day Structure Groups")

    if not structural_results or not structural_results["groups"]:
        st.info("Not enough complete days were available to create structural groups.")
        st.stop()

    structural_groups = structural_results["groups"]
    grouped_days = structural_results["all_grouped_days"]
    group_feedback = load_group_feedback()

    for group in structural_groups:
        signature = group_feedback_signature(
            ticker=ticker,
            interval=saved_interval,
            start_time=saved_start_time,
            end_time=saved_end_time,
            group=group,
        )
        group["feedback_signature"] = signature
        group["feedback_score"] = feedback_score_for_signature(group_feedback, signature)

    structural_groups = sorted(
        structural_groups,
        key=lambda group: (group["feedback_score"], group["count"]),
        reverse=True,
    )

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Days Grouped", len(grouped_days))
    with col_b:
        st.metric("Structure Groups", len(structural_groups))
    with col_c:
        st.metric("Shape Detail", structural_resample_points)

    feedback_summary = pd.DataFrame(
        [
            {
                "Group": group["name"],
                "Days": group["count"],
                "Avg Session Return %": round(group["avg_return"], 3),
                "Win Rate %": round(group["win_rate"], 1),
                "Avg Full-Day Range %": round(group["avg_range"], 3),
                "Avg First-Third Move %": round(group["avg_early_move"], 3),
                "Avg Momentum Shift Strength": round(group["avg_shift_strength"], 3),
                "Avg Shift Count": round(group["avg_shift_count"], 1),
                "Feedback Score": group.get("feedback_score", 0),
            }
            for group in structural_groups
        ]
    )
    st.dataframe(feedback_summary, width="stretch")
    st.plotly_chart(
        make_structure_outcome_chart(structural_groups),
        width="stretch",
        config=chart_config,
    )
    st.plotly_chart(
        make_structure_similarity_chart(structural_groups, structural_resample_points),
        width="stretch",
        config=chart_config,
    )
    st.plotly_chart(
        make_momentum_shift_map(structural_groups),
        width="stretch",
        config=chart_config,
    )

    st.subheader("Group Visuals")

    for group in structural_groups:
        other_days = [
            item
            for item in grouped_days
            if item["day"] not in {member["day"] for member in group["members"]}
        ]

        with st.expander(f"{group['name']} ({group['count']} days)", expanded=True):
            feedback_score = group.get("feedback_score", 0)
            feedback_label = "Liked before" if feedback_score > 0 else "Marked not useful" if feedback_score < 0 else "No feedback yet"
            st.caption(f"Feedback score: {feedback_score} | {feedback_label}")

            feedback_col_1, feedback_col_2, feedback_col_3 = st.columns([1, 1, 3])

            with feedback_col_1:
                if st.button("Good pattern", key=f"good_pattern_{group['id']}_{group['feedback_signature']}"):
                    save_group_feedback(
                        signature=group["feedback_signature"],
                        ticker=ticker,
                        interval=saved_interval,
                        start_time=saved_start_time,
                        end_time=saved_end_time,
                        group_name=group["name"],
                        rating="Good pattern",
                        note=st.session_state.get(f"feedback_note_{group['id']}_{group['feedback_signature']}", ""),
                    )
                    st.success("Saved as a good pattern.")
                    st.rerun()

            with feedback_col_2:
                if st.button("Not useful", key=f"bad_pattern_{group['id']}_{group['feedback_signature']}"):
                    save_group_feedback(
                        signature=group["feedback_signature"],
                        ticker=ticker,
                        interval=saved_interval,
                        start_time=saved_start_time,
                        end_time=saved_end_time,
                        group_name=group["name"],
                        rating="Not useful",
                        note=st.session_state.get(f"feedback_note_{group['id']}_{group['feedback_signature']}", ""),
                    )
                    st.warning("Saved as not useful.")
                    st.rerun()

            with feedback_col_3:
                st.text_input(
                    "Pattern note",
                    key=f"feedback_note_{group['id']}_{group['feedback_signature']}",
                    placeholder="What makes this group useful?",
                )

            tab_signature, tab_overlay, tab_previous_overlay, tab_previous, tab_momentum, tab_heatmap, tab_table = st.tabs(
                ["Signature", "Overlay", "Previous Overlay", "Previous Stats", "Momentum", "Heatmap", "Days"]
            )

            with tab_signature:
                st.plotly_chart(
                    make_signature_chart(
                        chart_name=group["name"],
                        ticker=ticker,
                        matching_days=group["members"],
                        compare_days=other_days,
                        start_value=start_value,
                    ),
                    width="stretch",
                    config=chart_config,
                )

            with tab_overlay:
                st.plotly_chart(
                    make_pattern_chart(
                        chart_name=group["name"],
                        ticker=ticker,
                        matching_days=group["members"],
                        background_days=[],
                        filters=[],
                        start_value=start_value,
                        show_background=False,
                        show_individual_legend=True,
                    ),
                    width="stretch",
                    config=chart_config,
                )

            previous_contexts = build_previous_day_context(
                group=group,
                grouped_days=grouped_days,
                resample_points=structural_resample_points,
            )

            with tab_previous_overlay:
                if not previous_contexts:
                    st.info("No previous days were available for this group.")
                else:
                    include_group_days = st.checkbox(
                        "Also show the matching group days as dashed lines",
                        value=False,
                        key=f"include_group_days_previous_overlay_{group['id']}",
                    )
                    st.plotly_chart(
                        make_previous_day_overlay_chart(
                            group=group,
                            contexts=previous_contexts,
                            start_value=start_value,
                            include_group_days=include_group_days,
                        ),
                        width="stretch",
                        config=chart_config,
                        key=f"previous_overlay_chart_{group['id']}",
                    )

            with tab_previous:
                if not previous_contexts:
                    st.info("No previous-day matches were available for this group.")
                else:
                    context_table = make_previous_day_context_table(previous_contexts)
                    avg_similarity = context_table["Prior vs Group-Day Similarity"].dropna().mean()
                    avg_previous_return = context_table["Previous Day Return %"].mean()
                    avg_group_return = context_table["Group Day Return %"].mean()

                    metric_a, metric_b, metric_c = st.columns(3)

                    with metric_a:
                        st.metric(
                            "Avg Prior-Day Similarity",
                            f"{avg_similarity:.3f}" if pd.notna(avg_similarity) else "N/A",
                        )
                    with metric_b:
                        st.metric("Avg Previous Return", f"{avg_previous_return:.2f}%")
                    with metric_c:
                        st.metric("Avg Group-Day Return", f"{avg_group_return:.2f}%")

                    st.dataframe(context_table, width="stretch")
                    st.plotly_chart(
                        make_previous_day_overlay_chart(
                            group=group,
                            contexts=previous_contexts,
                            start_value=start_value,
                            include_group_days=False,
                        ),
                        width="stretch",
                        config=chart_config,
                        key=f"previous_stats_overlay_chart_{group['id']}",
                    )
                    st.plotly_chart(
                        make_previous_to_current_scatter(previous_contexts),
                        width="stretch",
                        config=chart_config,
                        key=f"previous_to_current_scatter_{group['id']}",
                    )

            with tab_momentum:
                st.plotly_chart(
                    make_group_momentum_chart(group, structural_resample_points),
                    width="stretch",
                    config=chart_config,
                )

            with tab_heatmap:
                st.plotly_chart(
                    make_heatmap_chart(
                        chart_name=group["name"],
                        ticker=ticker,
                        days=group["members"],
                        lookahead_bars=lookahead_bars,
                    ),
                    width="stretch",
                    config=chart_config,
                )

            with tab_table:
                rows = []
                for item in group["members"]:
                    day_return = (item["data"]["Close"].iloc[-1] - item["data"]["Close"].iloc[0]) / item["data"]["Close"].iloc[0] * 100
                    day_range = (item["data"]["High"].max() - item["data"]["Low"].min()) / item["data"]["Close"].iloc[0] * 100
                    fingerprint = item.get("momentum_fingerprint", {})
                    shift_strengths = fingerprint.get("shift_strengths", [])
                    largest_shift = max(shift_strengths) if shift_strengths else 0
                    rows.append(
                        {
                            "Date": item["day"],
                            "Session Return %": round(day_return, 3),
                            "Full-Day Range %": round(day_range, 3),
                            "Largest Momentum Shift": round(largest_shift, 3),
                            "Bars": len(item["data"]),
                        }
                    )
                st.dataframe(pd.DataFrame(rows), width="stretch")

    st.subheader("Inspect Similar Days")
    group_options = [group["name"] for group in structural_groups]
    selected_group_name = st.selectbox("Choose a structure group", group_options)
    selected_group = structural_groups[group_options.index(selected_group_name)]
    selected_item = pick_day("Choose a day inside this group", selected_group["members"])

    if selected_item is not None:
        detail_fig = make_detail_chart(selected_item, ticker, start_value)
        st.plotly_chart(detail_fig, width="stretch", config=chart_config)

    st.stop()

summary_df = make_summary_table(
    chart_results=chart_results,
    no_filter_match_days=no_filter_match_days,
    all_days=all_days,
    lookahead_bars=lookahead_bars,
)

st.subheader("Pattern Summary")

col_a, col_b, col_c, col_d = st.columns(4)
with col_a:
    st.metric("Days Checked", len(all_days))
with col_b:
    st.metric("Pattern Charts", len(chart_groups))
with col_c:
    st.metric("No-Filter-Match Days", len(no_filter_match_days))
with col_d:
    st.metric("Outcome Window", f"{lookahead_minutes} min")

st.dataframe(summary_df, width="stretch")

st.plotly_chart(
    make_outcome_distribution_chart(chart_results, no_filter_match_days, lookahead_bars),
    width="stretch",
    config=chart_config,
)

st.subheader("Pattern Charts")

for group in chart_groups:
    chart_name = group["name"]
    result = chart_results[chart_name]
    matching_days = result["matching_days"]
    background_days = [item for item in all_days if item["day"] not in {d["day"] for d in matching_days}]

    if pattern_view == "Signature band":
        fig = make_signature_chart(
            chart_name=chart_name,
            ticker=ticker,
            matching_days=matching_days,
            compare_days=background_days,
            start_value=start_value,
        )
    elif pattern_view == "Heatmap":
        fig = make_heatmap_chart(
            chart_name=chart_name,
            ticker=ticker,
            days=matching_days,
            lookahead_bars=lookahead_bars,
        )
    else:
        fig = make_pattern_chart(
            chart_name=chart_name,
            ticker=ticker,
            matching_days=matching_days,
            background_days=background_days,
            filters=group["filters"],
            start_value=start_value,
            show_background=show_background_days,
            show_individual_legend=show_individual_legend,
        )

    with st.expander(f"{chart_name} ({len(matching_days)} days)", expanded=True):
        st.plotly_chart(fig, width="stretch", config=chart_config)

        if result["filter_rows"]:
            st.dataframe(pd.DataFrame(result["filter_rows"]), width="stretch")
        else:
            st.write("No active filters in this chart.")

if show_no_filter_chart:
    no_match_background = [item for item in all_days if item["day"] not in {d["day"] for d in no_filter_match_days}]
    if pattern_view == "Signature band":
        fig = make_signature_chart(
            chart_name="No-filter-match days",
            ticker=ticker,
            matching_days=no_filter_match_days,
            compare_days=no_match_background,
            start_value=start_value,
        )
    elif pattern_view == "Heatmap":
        fig = make_heatmap_chart(
            chart_name="No-filter-match days",
            ticker=ticker,
            days=no_filter_match_days,
            lookahead_bars=lookahead_bars,
        )
    else:
        fig = make_pattern_chart(
            chart_name="No-filter-match days",
            ticker=ticker,
            matching_days=no_filter_match_days,
            background_days=no_match_background,
            filters=[],
            start_value=start_value,
            show_background=show_background_days,
            show_individual_legend=show_individual_legend,
        )

    with st.expander(f"No-filter-match days ({len(no_filter_match_days)} days)", expanded=True):
        st.plotly_chart(fig, width="stretch", config=chart_config)

st.subheader("Individual Day Viewer")

viewer_groups = {"All days": all_days, "No-filter-match days": no_filter_match_days}

for chart_name, result in chart_results.items():
    viewer_groups[chart_name] = result["matching_days"]

viewer_choice = st.selectbox("Choose a group to inspect", list(viewer_groups.keys()))
selected_item = pick_day("Choose a day", viewer_groups[viewer_choice])

if selected_item is not None:
    day_return = calculate_outcome(selected_item, lookahead_bars)
    day_data = selected_item["data"]
    col_1, col_2, col_3, col_4 = st.columns(4)

    with col_1:
        st.metric("Open", f"{day_data['Open'].iloc[0]:.2f}")
    with col_2:
        st.metric("Close at Window", f"{day_data['Close'].iloc[min(len(day_data)-1, lookahead_bars-1)]:.2f}")
    with col_3:
        st.metric("Window Return", f"{day_return:.2f}%" if day_return is not None else "N/A")
    with col_4:
        st.metric("Bars in Day", len(day_data))

    detail_fig = make_detail_chart(selected_item, ticker, start_value)
    st.plotly_chart(detail_fig, width="stretch", config=chart_config)

st.subheader("Export")

export_rows = []

for chart_name, result in chart_results.items():
    for item in result["matching_days"]:
        export_rows.append(
            {
                "Group": chart_name,
                "Date": item["day"],
                "Outcome Window Minutes": lookahead_minutes,
                "Outcome Return %": calculate_outcome(item, lookahead_bars),
            }
        )

for item in no_filter_match_days:
    export_rows.append(
        {
            "Group": "No-filter-match days",
            "Date": item["day"],
            "Outcome Window Minutes": lookahead_minutes,
            "Outcome Return %": calculate_outcome(item, lookahead_bars),
        }
    )

if export_rows:
    export_df = pd.DataFrame(export_rows)
    st.download_button(
        "Download scan results",
        data=export_df.to_csv(index=False),
        file_name=f"mach2_pattern_scan_{ticker}_{pd.Timestamp.today().date()}.csv",
        mime="text/csv",
    )
