from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from local_market_data import load_downloaded_data


ARTIFACT_DIR = Path("model_artifacts")
PREDICTION_FILE = ARTIFACT_DIR / "bar20_entry_quality_predictions.csv"
REPORT_FILE = ARTIFACT_DIR / "bar20_entry_quality_model_report.txt"
TIMING_SUMMARY_FILE = ARTIFACT_DIR / "entry_bar_timing_sweep_summary.csv"
TIMING_PREDICTION_FILE = ARTIFACT_DIR / "entry_bar_timing_predictions.csv"
TIMING_REPORT_FILE = ARTIFACT_DIR / "entry_bar_timing_sweep_report.md"
COMBINED_RANKING_FILE = ARTIFACT_DIR / "combined_live_signal_rankings.csv"
COMBINED_REPORT_FILE = ARTIFACT_DIR / "combined_live_signal_ranker_report.md"
LIVE_REPLAY_FILE = ARTIFACT_DIR / "live_replay_scanner_results.csv"
LIVE_REPLAY_REPORT_FILE = ARTIFACT_DIR / "live_replay_scanner_report.md"
RAW_STRUCTURE_PREDICTION_FILE = ARTIFACT_DIR / "raw_early_structure_predictions.csv"
RAW_STRUCTURE_SUMMARY_FILE = ARTIFACT_DIR / "raw_early_structure_summary.csv"
RAW_STRUCTURE_REPORT_FILE = ARTIFACT_DIR / "raw_early_structure_predictor_report.md"
STAGED_RANKING_FILE = ARTIFACT_DIR / "staged_live_signal_rankings.csv"
STAGED_SUMMARY_FILE = ARTIFACT_DIR / "staged_live_signal_summary.csv"
STAGED_REPORT_FILE = ARTIFACT_DIR / "staged_live_signal_ranker_report.md"
PATTERN_GROUPING_FILE = ARTIFACT_DIR / "pattern_grouping_candidates.csv"
PATTERN_GROUPING_REPORT_FILE = ARTIFACT_DIR / "pattern_grouping_lab_report.md"
SESSION_START = "09:30"
SESSION_END = "16:00"


st.set_page_config(page_title="Pattern Model Dashboard", layout="wide")


@st.cache_data
def load_predictions():
    if not PREDICTION_FILE.exists():
        return pd.DataFrame()
    df = pd.read_csv(PREDICTION_FILE)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["rank"] = df["bar20_entry_quality_probability"].rank(ascending=False, method="first")
    df["quality_bucket"] = pd.qcut(
        df["bar20_entry_quality_probability"].rank(method="first"),
        q=10,
        labels=[f"D{i}" for i in range(1, 11)],
    )
    return df.sort_values("bar20_entry_quality_probability", ascending=False)


@st.cache_data
def load_day(ticker, date):
    df = load_downloaded_data(ticker, "5m")
    if df.empty:
        return df
    session = df.between_time(SESSION_START, SESSION_END)
    return session[session.index.date == date].copy()


@st.cache_data
def load_timing_summary():
    if not TIMING_SUMMARY_FILE.exists():
        return pd.DataFrame()
    return pd.read_csv(TIMING_SUMMARY_FILE)


@st.cache_data
def load_timing_predictions():
    if not TIMING_PREDICTION_FILE.exists():
        return pd.DataFrame()
    timing = pd.read_csv(TIMING_PREDICTION_FILE)
    timing["date"] = pd.to_datetime(timing["date"]).dt.date
    return timing.sort_values(["entry_bar", "entry_quality_probability"], ascending=[True, False])


@st.cache_data
def load_combined_rankings():
    if not COMBINED_RANKING_FILE.exists():
        return pd.DataFrame()
    combined = pd.read_csv(COMBINED_RANKING_FILE)
    combined["date"] = pd.to_datetime(combined["date"]).dt.date
    return combined.sort_values("combined_live_score", ascending=False)


@st.cache_data
def load_live_replay():
    if not LIVE_REPLAY_FILE.exists():
        return pd.DataFrame()
    replay = pd.read_csv(LIVE_REPLAY_FILE)
    replay["date"] = pd.to_datetime(replay["date"]).dt.date
    return replay.sort_values("research_context_score", ascending=False)


@st.cache_data
def load_raw_structure_summary():
    if not RAW_STRUCTURE_SUMMARY_FILE.exists():
        return pd.DataFrame()
    return pd.read_csv(RAW_STRUCTURE_SUMMARY_FILE)


@st.cache_data
def load_raw_structure_predictions():
    if not RAW_STRUCTURE_PREDICTION_FILE.exists():
        return pd.DataFrame()
    raw_structure = pd.read_csv(RAW_STRUCTURE_PREDICTION_FILE)
    raw_structure["date"] = pd.to_datetime(raw_structure["date"]).dt.date
    return raw_structure.sort_values(
        ["entry_bar", "raw_early_structure_probability"],
        ascending=[True, False],
    )


@st.cache_data
def load_staged_summary():
    if not STAGED_SUMMARY_FILE.exists():
        return pd.DataFrame()
    return pd.read_csv(STAGED_SUMMARY_FILE)


@st.cache_data
def load_staged_rankings():
    if not STAGED_RANKING_FILE.exists():
        return pd.DataFrame()
    staged = pd.read_csv(STAGED_RANKING_FILE)
    staged["date"] = pd.to_datetime(staged["date"]).dt.date
    return staged.sort_values(["variant", "staged_live_score"], ascending=[True, False])


@st.cache_data
def load_pattern_groups():
    if not PATTERN_GROUPING_FILE.exists():
        return pd.DataFrame()
    groups = pd.read_csv(PATTERN_GROUPING_FILE)
    return groups.sort_values("tightness_score", ascending=False)


def bucket_summary(df):
    return (
        df.groupby("quality_bucket", observed=True)
        .agg(
            rows=("ticker", "size"),
            avg_probability=("bar20_entry_quality_probability", "mean"),
            avg_signed_return=("signed_bar20_close_return_pct", "mean"),
            median_signed_return=("signed_bar20_close_return_pct", "median"),
            hit_rate=("target_bar20_followthrough", "mean"),
            strong_hit_rate=("target_bar20_strong_followthrough", "mean"),
        )
        .reset_index()
    )


def make_day_chart(day, row):
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=day.index,
            open=day["Open"],
            high=day["High"],
            low=day["Low"],
            close=day["Close"],
            name="5m",
        )
    )

    entry_time = pd.to_datetime(row["entry_time"])
    if entry_time.tzinfo is None and day.index.tz is not None:
        entry_time = entry_time.tz_localize(day.index.tz)
    elif entry_time.tzinfo is not None and day.index.tz is not None:
        entry_time = entry_time.tz_convert(day.index.tz)

    fig.add_vline(x=entry_time, line_width=2, line_dash="dash", line_color="#111827")
    fig.add_annotation(
        x=entry_time,
        y=float(day["High"].max()),
        text=f"bar 20 {row['implied_side']}",
        showarrow=False,
        yanchor="bottom",
    )

    fig.update_layout(
        height=520,
        margin=dict(l=10, r=10, t=35, b=10),
        xaxis_rangeslider_visible=False,
        title=f"{row['ticker']} {row['date']} | quality {row['bar20_entry_quality_probability']:.3f} | signed return {row['signed_bar20_close_return_pct']:.2f}%",
    )
    return fig


df = load_predictions()

st.title("Pattern Learning Model Dashboard")
st.caption("Visual view of structural pattern families, ranking scores, and realized outcomes.")

if df.empty:
    st.error("Missing model_artifacts/bar20_entry_quality_predictions.csv. Run bar20_entry_quality_model.py first.")
    st.stop()

with st.sidebar:
    st.header("Filters")
    asset_options = sorted(df["asset_class"].dropna().unique())
    side_options = sorted(df["implied_side"].dropna().unique())
    selected_assets = st.multiselect("Asset class", asset_options, default=asset_options)
    selected_sides = st.multiselect("Side", side_options, default=side_options)
    min_quality = st.slider("Minimum entry quality", 0.0, 1.0, 0.50, 0.01)
    min_early_pattern = st.slider("Minimum early pattern", 0.0, 1.0, 0.00, 0.01)
    top_n = st.slider("Top rows", 25, 500, 100, 25)

filtered = df[
    df["asset_class"].isin(selected_assets)
    & df["implied_side"].isin(selected_sides)
    & (df["bar20_entry_quality_probability"] >= min_quality)
    & (df["early_pattern_probability"] >= min_early_pattern)
].copy()

top = filtered.head(top_n).copy()

metric_cols = st.columns(5)
metric_cols[0].metric("Rows", f"{len(filtered):,}")
metric_cols[1].metric("Avg Signed Return", f"{filtered['signed_bar20_close_return_pct'].mean():.2f}%")
metric_cols[2].metric("Hit Rate", f"{filtered['target_bar20_followthrough'].mean():.1%}")
metric_cols[3].metric("Strong Hit Rate", f"{filtered['target_bar20_strong_followthrough'].mean():.1%}")
metric_cols[4].metric("Avg Quality", f"{filtered['bar20_entry_quality_probability'].mean():.3f}")

tab_overview, tab_patterns, tab_timing, tab_combined, tab_replay, tab_raw_structure, tab_staged, tab_ranked, tab_chart, tab_report = st.tabs(
    [
        "Overview",
        "Pattern Groups",
        "Entry Timing",
        "Combined Ranker",
        "Live Replay",
        "Raw Structure",
        "Staged Ranker",
        "Ranked Rows",
        "Ticker Chart",
        "Model Report",
    ]
)

with tab_overview:
    left, right = st.columns(2)

    summary = bucket_summary(df)
    with left:
        st.subheader("Quality Deciles")
        fig = px.bar(
            summary,
            x="quality_bucket",
            y="avg_signed_return",
            color="hit_rate",
            hover_data=["rows", "median_signed_return", "strong_hit_rate"],
            labels={
                "quality_bucket": "Quality bucket, D10 is highest",
                "avg_signed_return": "Avg signed bar20-to-close return %",
                "hit_rate": "Hit rate",
            },
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Quality Vs Outcome")
        sample = filtered.head(1500)
        fig = px.scatter(
            sample,
            x="bar20_entry_quality_probability",
            y="signed_bar20_close_return_pct",
            color="implied_side",
            symbol="asset_class",
            hover_data=["ticker", "date", "early_pattern_probability", "early_direction_edge"],
            labels={
                "bar20_entry_quality_probability": "Entry quality probability",
                "signed_bar20_close_return_pct": "Signed bar20-to-close return %",
            },
        )
        fig.add_hline(y=0, line_dash="dash", line_color="#6b7280")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Side And Asset Behavior")
    side_asset = (
        filtered.groupby(["asset_class", "implied_side"], as_index=False)
        .agg(
            rows=("ticker", "size"),
            avg_quality=("bar20_entry_quality_probability", "mean"),
            avg_signed_return=("signed_bar20_close_return_pct", "mean"),
            hit_rate=("target_bar20_followthrough", "mean"),
        )
        .sort_values("avg_signed_return", ascending=False)
    )
    st.dataframe(side_asset, use_container_width=True, hide_index=True)

with tab_patterns:
    st.subheader("Structural Pattern Families")
    groups = load_pattern_groups()

    if groups.empty:
        st.info("Run pattern_grouping_lab.py to populate this view.")
    else:
        pattern_types = sorted(groups["pattern_name"].str.split(": ").str[-1].dropna().unique())
        tickers = sorted(groups["ticker"].dropna().unique())

        col_a, col_b, col_c, col_d = st.columns(4)
        selected_group_counts = col_a.multiselect(
            "Group counts",
            sorted(groups["group_count_setting"].dropna().unique()),
            default=sorted(groups["group_count_setting"].dropna().unique()),
        )
        selected_pattern_types = col_b.multiselect("Pattern type", pattern_types, default=pattern_types)
        selected_tickers = col_c.multiselect("Ticker", tickers, default=[])
        min_tightness = col_d.slider("Min tightness", 0.0, 1.0, 0.75, 0.01)

        pattern_view = groups[
            groups["group_count_setting"].isin(selected_group_counts)
            & groups["pattern_name"].str.split(": ").str[-1].isin(selected_pattern_types)
            & (groups["tightness_score"] >= min_tightness)
        ].copy()
        if selected_tickers:
            pattern_view = pattern_view[pattern_view["ticker"].isin(selected_tickers)]

        summary = (
            pattern_view.groupby("group_count_setting", as_index=False)
            .agg(
                groups=("ticker", "size"),
                avg_tightness=("tightness_score", "mean"),
                avg_corr=("avg_shape_corr", "mean"),
                avg_min_corr=("min_shape_corr", "mean"),
                avg_rmse=("signature_rmse", "mean"),
                avg_band=("signature_band_width", "mean"),
                avg_days=("days_in_group", "mean"),
            )
            .sort_values("group_count_setting")
        )

        metric_cols = st.columns(5)
        metric_cols[0].metric("Pattern Families", f"{len(pattern_view):,}")
        metric_cols[1].metric("Avg Tightness", f"{pattern_view['tightness_score'].mean():.3f}")
        metric_cols[2].metric("Avg Corr", f"{pattern_view['avg_shape_corr'].mean():.3f}")
        metric_cols[3].metric("Avg Min Corr", f"{pattern_view['min_shape_corr'].mean():.3f}")
        metric_cols[4].metric("Avg Days", f"{pattern_view['days_in_group'].mean():.1f}")

        left, right = st.columns(2)
        with left:
            fig = px.bar(
                summary,
                x="group_count_setting",
                y="groups",
                color="avg_tightness",
                hover_data=["avg_corr", "avg_min_corr", "avg_days"],
                labels={
                    "group_count_setting": "Automatic group count",
                    "groups": "Pattern families",
                    "avg_tightness": "Avg tightness",
                },
            )
            st.plotly_chart(fig, use_container_width=True)

        with right:
            fig = px.scatter(
                pattern_view.head(1500),
                x="avg_shape_corr",
                y="signature_rmse",
                color="group_count_setting",
                size="days_in_group",
                hover_data=["ticker", "pattern_name", "representative_dates", "tightness_score"],
                labels={
                    "avg_shape_corr": "Average shape correlation",
                    "signature_rmse": "Signature RMSE",
                    "group_count_setting": "Groups",
                },
            )
            st.plotly_chart(fig, use_container_width=True)

        display_cols = [
            "ticker",
            "pattern_name",
            "group_count_setting",
            "days_in_group",
            "tightness_score",
            "avg_shape_corr",
            "min_shape_corr",
            "signature_rmse",
            "signature_band_width",
            "same_direction_rate",
            "median_abs_return",
            "avg_return",
            "representative_dates",
        ]
        st.dataframe(pattern_view.head(top_n)[display_cols], use_container_width=True, hide_index=True)

with tab_timing:
    st.subheader("Which Entry Bar Works Best?")
    timing_summary = load_timing_summary()
    timing_predictions = load_timing_predictions()

    if timing_summary.empty:
        st.info("Run entry_bar_timing_sweep.py to populate timing charts.")
    else:
        timing_cols = st.columns(3)
        best = timing_summary.sort_values(["top5_avg_signed_return", "top5_hit_rate"], ascending=False).iloc[0]
        timing_cols[0].metric("Best Entry Bar", int(best["entry_bar"]))
        timing_cols[1].metric("Top 5% Avg Return", f"{best['top5_avg_signed_return']:.2f}%")
        timing_cols[2].metric("Top 5% Hit Rate", f"{best['top5_hit_rate']:.1%}")

        left, right = st.columns(2)
        with left:
            fig = px.line(
                timing_summary.sort_values("entry_bar"),
                x="entry_bar",
                y=["top5_avg_signed_return", "top10_avg_signed_return", "bottom50_avg_signed_return"],
                markers=True,
                labels={"entry_bar": "5m entry bar", "value": "Avg signed return %"},
            )
            fig.add_hline(y=0, line_dash="dash", line_color="#6b7280")
            st.plotly_chart(fig, use_container_width=True)

        with right:
            fig = px.line(
                timing_summary.sort_values("entry_bar"),
                x="entry_bar",
                y=["top5_hit_rate", "top10_hit_rate", "bottom50_hit_rate"],
                markers=True,
                labels={"entry_bar": "5m entry bar", "value": "Hit rate"},
            )
            st.plotly_chart(fig, use_container_width=True)

        st.dataframe(timing_summary, use_container_width=True, hide_index=True)

        if not timing_predictions.empty:
            selected_bar = st.selectbox(
                "Entry bar predictions",
                sorted(timing_predictions["entry_bar"].unique()),
                index=0,
            )
            bar_rows = timing_predictions[timing_predictions["entry_bar"] == selected_bar].head(top_n)
            timing_display = [
                "ticker",
                "date",
                "asset_class",
                "implied_side",
                "entry_bar",
                "entry_quality_probability",
                "signed_entry_to_close_return_pct",
                "target_followthrough",
                "early_pattern_probability",
                "early_direction_edge",
                "useful_pattern_ensemble_score",
                "entry_close_position",
                "entry_last5_return_pct",
                "entry_early_range_pct",
            ]
            st.dataframe(bar_rows[timing_display], use_container_width=True, hide_index=True)

with tab_combined:
    st.subheader("Combined Live-Style Ranking")
    combined = load_combined_rankings()

    if combined.empty:
        st.info("Run combined_live_signal_ranker.py to populate this view.")
    else:
        guarded_only = st.checkbox("Guarded candidates only", value=False)
        combined_view = combined[combined["passes_exploratory_guardrail"]] if guarded_only else combined

        cols = st.columns(4)
        top5_count = max(1, int(round(len(combined_view) * 0.05)))
        top5_combined = combined_view.head(top5_count)
        cols[0].metric("Rows", f"{len(combined_view):,}")
        cols[1].metric("Top 5% Avg Return", f"{top5_combined['signed_entry_to_close_return_pct'].mean():.2f}%")
        cols[2].metric("Top 5% Hit Rate", f"{top5_combined['target_followthrough'].mean():.1%}")
        cols[3].metric("Guardrail Pass", f"{combined_view['passes_exploratory_guardrail'].mean():.1%}")

        combined_display = [
            "ticker",
            "date",
            "asset_class",
            "implied_side",
            "entry_bar",
            "combined_live_score",
            "entry_quality_probability",
            "signed_entry_to_close_return_pct",
            "target_followthrough",
            "early_pattern_probability",
            "early_direction_edge",
            "useful_pattern_ensemble_score",
            "passes_exploratory_guardrail",
            "entry_close_position",
            "entry_last5_return_pct",
            "entry_early_range_pct",
        ]
        st.dataframe(combined_view.head(top_n)[combined_display], use_container_width=True, hide_index=True)

        fig = px.scatter(
            combined_view.head(1500),
            x="combined_live_score",
            y="signed_entry_to_close_return_pct",
            color="implied_side",
            symbol="asset_class",
            hover_data=["ticker", "date", "entry_quality_probability", "early_pattern_probability"],
            labels={
                "combined_live_score": "Combined live score",
                "signed_entry_to_close_return_pct": "Signed entry-to-close return %",
            },
        )
        fig.add_hline(y=0, line_dash="dash", line_color="#6b7280")
        st.plotly_chart(fig, use_container_width=True)

with tab_replay:
    st.subheader("Bar-8 Live Replay Scanner")
    replay = load_live_replay()

    if replay.empty:
        st.info("Run live_replay_scanner.py to populate this view.")
    else:
        replay_score = st.radio(
            "Replay score",
            ["research_context_score", "raw_replay_probability"],
            horizontal=True,
        )
        guarded_only = st.checkbox("Replay guarded candidates only", value=False)
        replay_view = replay[replay["passes_exploratory_guardrail"]] if guarded_only else replay
        replay_view = replay_view.sort_values(replay_score, ascending=False)

        replay_top5_count = max(1, int(round(len(replay_view) * 0.05)))
        replay_top5 = replay_view.head(replay_top5_count)
        cols = st.columns(5)
        cols[0].metric("Rows", f"{len(replay_view):,}")
        cols[1].metric("Top 5% Avg Return", f"{replay_top5['signed_entry_to_close_return_pct'].mean():.2f}%")
        cols[2].metric("Top 5% Hit Rate", f"{replay_top5['target_followthrough'].mean():.1%}")
        cols[3].metric("Top 5% Strong Hit", f"{replay_top5['target_strong_followthrough'].mean():.1%}")
        cols[4].metric("Short Rate", f"{replay_top5['implied_side'].eq('short').mean():.1%}")

        replay_display = [
            "ticker",
            "date",
            "asset_class",
            "implied_side",
            "entry_bar",
            "raw_replay_probability",
            "research_context_score",
            "signed_entry_to_close_return_pct",
            "target_followthrough",
            "early_pattern_probability",
            "early_direction_edge",
            "useful_pattern_ensemble_score",
            "passes_exploratory_guardrail",
            "entry_close_position",
            "entry_last5_return_pct",
            "entry_early_range_pct",
        ]
        st.dataframe(replay_view.head(top_n)[replay_display], use_container_width=True, hide_index=True)

        fig = px.scatter(
            replay_view.head(1500),
            x=replay_score,
            y="signed_entry_to_close_return_pct",
            color="implied_side",
            symbol="asset_class",
            hover_data=["ticker", "date", "raw_replay_probability", "research_context_score"],
            labels={
                replay_score: replay_score.replace("_", " ").title(),
                "signed_entry_to_close_return_pct": "Signed entry-to-close return %",
            },
        )
        fig.add_hline(y=0, line_dash="dash", line_color="#6b7280")
        st.plotly_chart(fig, use_container_width=True)

with tab_raw_structure:
    st.subheader("Raw Early-Structure Predictor")
    raw_summary = load_raw_structure_summary()
    raw_predictions = load_raw_structure_predictions()

    if raw_summary.empty or raw_predictions.empty:
        st.info("Run raw_early_structure_predictor.py to populate this view.")
    else:
        best_raw = raw_summary.sort_values(["top5_target_rate", "top5_avg_abs_return"], ascending=False).iloc[0]
        cols = st.columns(4)
        cols[0].metric("Best Bar", int(best_raw["entry_bar"]))
        cols[1].metric("Top 5% Target", f"{best_raw['top5_target_rate']:.1%}")
        cols[2].metric("Top 5% Abs Return", f"{best_raw['top5_avg_abs_return']:.2f}%")
        cols[3].metric("AUC", f"{best_raw['roc_auc']:.3f}")

        left, right = st.columns(2)
        with left:
            fig = px.line(
                raw_summary.sort_values("entry_bar"),
                x="entry_bar",
                y=["top5_target_rate", "top10_target_rate", "bottom50_target_rate"],
                markers=True,
                labels={"entry_bar": "5m checkpoint bar", "value": "High-structure target rate"},
            )
            st.plotly_chart(fig, use_container_width=True)

        with right:
            fig = px.line(
                raw_summary.sort_values("entry_bar"),
                x="entry_bar",
                y=["roc_auc", "average_precision"],
                markers=True,
                labels={"entry_bar": "5m checkpoint bar", "value": "Score"},
            )
            st.plotly_chart(fig, use_container_width=True)

        st.dataframe(raw_summary, use_container_width=True, hide_index=True)

        selected_structure_bar = st.selectbox(
            "Raw structure checkpoint",
            sorted(raw_predictions["entry_bar"].unique()),
            index=len(sorted(raw_predictions["entry_bar"].unique())) - 1,
        )
        structure_view = raw_predictions[raw_predictions["entry_bar"] == selected_structure_bar].sort_values(
            "raw_early_structure_probability",
            ascending=False,
        )
        raw_structure_display = [
            "ticker",
            "date",
            "asset_class",
            "entry_bar",
            "raw_early_structure_probability",
            "target_high_structure",
            "useful_pattern_ensemble_score",
            "avg_abs_return",
            "session_open_to_entry_pct",
            "entry_early_range_pct",
            "entry_close_position",
            "entry_last5_return_pct",
            "entry_volume_late_early_ratio",
        ]
        st.dataframe(structure_view.head(top_n)[raw_structure_display], use_container_width=True, hide_index=True)

with tab_staged:
    st.subheader("Staged Live Signal Ranker")
    staged_summary = load_staged_summary()
    staged_rankings = load_staged_rankings()

    if staged_summary.empty or staged_rankings.empty:
        st.info("Run staged_live_signal_ranker.py to populate this view.")
    else:
        staged_score_rows = staged_summary[
            (staged_summary["score_col"] == "staged_live_score") & (staged_summary["slice"] == "top_5pct")
        ].sort_values("avg_signed_return", ascending=False)

        st.dataframe(staged_score_rows, use_container_width=True, hide_index=True)

        left, right = st.columns(2)
        with left:
            fig = px.bar(
                staged_score_rows,
                x="variant",
                y="avg_signed_return",
                color="hit_rate",
                hover_data=["rows", "strong_hit_rate", "target_high_structure_rate"],
                labels={"avg_signed_return": "Top 5% avg signed return %", "variant": "Variant"},
            )
            st.plotly_chart(fig, use_container_width=True)

        with right:
            fig = px.bar(
                staged_score_rows,
                x="variant",
                y="target_high_structure_rate",
                color="avg_raw_structure_probability",
                hover_data=["avg_signed_return", "hit_rate", "short_rate"],
                labels={"target_high_structure_rate": "Top 5% high-structure rate", "variant": "Variant"},
            )
            st.plotly_chart(fig, use_container_width=True)

        selected_variant = st.selectbox(
            "Staged variant",
            sorted(staged_rankings["variant"].unique()),
        )
        staged_view = staged_rankings[staged_rankings["variant"] == selected_variant].sort_values(
            "staged_live_score",
            ascending=False,
        )
        staged_display = [
            "ticker",
            "date",
            "asset_class",
            "variant",
            "implied_side",
            "entry_bar",
            "structure_bar",
            "staged_live_score",
            "entry_quality_probability",
            "raw_structure_probability",
            "signed_entry_to_close_return_pct",
            "target_followthrough",
            "target_high_structure",
            "early_direction_edge",
            "passes_exploratory_guardrail",
            "structure_confirmed",
        ]
        st.dataframe(staged_view.head(top_n)[staged_display], use_container_width=True, hide_index=True)

with tab_ranked:
    st.subheader("Top Ranked Model Rows")
    display_cols = [
        "ticker",
        "date",
        "asset_class",
        "implied_side",
        "bar20_entry_quality_probability",
        "signed_bar20_close_return_pct",
        "target_bar20_followthrough",
        "early_pattern_probability",
        "early_direction_edge",
        "useful_pattern_ensemble_score",
        "bar20_close_position",
        "bar20_last5_return_pct",
        "bar20_early_range_pct",
    ]
    st.dataframe(top[display_cols], use_container_width=True, hide_index=True)

    ranked = top.sort_values("bar20_entry_quality_probability", ascending=False).copy()
    ranked["cum_avg_signed_return"] = ranked["signed_bar20_close_return_pct"].expanding().mean()
    ranked["cum_hit_rate"] = ranked["target_bar20_followthrough"].expanding().mean()
    fig = px.line(
        ranked,
        x="rank",
        y=["cum_avg_signed_return", "cum_hit_rate"],
        labels={"rank": "Model rank"},
    )
    st.plotly_chart(fig, use_container_width=True)

with tab_chart:
    st.subheader("Inspect One Prediction On Its 5m Chart")
    choices = top.assign(
        label=lambda x: x["ticker"].astype(str)
        + " | "
        + x["date"].astype(str)
        + " | "
        + x["implied_side"].astype(str)
        + " | q="
        + x["bar20_entry_quality_probability"].round(3).astype(str)
        + " | ret="
        + x["signed_bar20_close_return_pct"].round(2).astype(str)
        + "%"
    )
    selected_label = st.selectbox("Prediction", choices["label"].tolist())
    row = choices[choices["label"] == selected_label].iloc[0]
    day = load_day(row["ticker"], row["date"])

    if day.empty:
        st.warning("No raw intraday data found for this ticker/date.")
    else:
        st.plotly_chart(make_day_chart(day, row), use_container_width=True)
        st.dataframe(row[display_cols].to_frame("value"), use_container_width=True)

with tab_report:
    if REPORT_FILE.exists():
        st.text(REPORT_FILE.read_text(encoding="utf-8"))
    else:
        st.info("No report file found.")

    if TIMING_REPORT_FILE.exists():
        st.divider()
        st.text(TIMING_REPORT_FILE.read_text(encoding="utf-8"))

    if COMBINED_REPORT_FILE.exists():
        st.divider()
        st.text(COMBINED_REPORT_FILE.read_text(encoding="utf-8"))

    if LIVE_REPLAY_REPORT_FILE.exists():
        st.divider()
        st.text(LIVE_REPLAY_REPORT_FILE.read_text(encoding="utf-8"))

    if RAW_STRUCTURE_REPORT_FILE.exists():
        st.divider()
        st.text(RAW_STRUCTURE_REPORT_FILE.read_text(encoding="utf-8"))

    if STAGED_REPORT_FILE.exists():
        st.divider()
        st.text(STAGED_REPORT_FILE.read_text(encoding="utf-8"))
