import pandas as pd
import streamlit as st

from scanner import get_patterns
from strategy_tester import (
    REGULAR_CLOSE,
    REGULAR_OPEN,
    SUPPORTED_TIMEFRAMES,
    list_available_tickers,
    load_market_data,
    make_box_strategy_figure,
    run_box_strategy,
    summarize_results,
)
from feedback_db import init_db, save_feedback, get_recent_feedback, get_feedback_summary
from codex_tasks import (
    TASK_FILE,
    SUMMARY_FILE,
    add_codex_task,
    build_task_from_feedback,
    save_feedback_summary,
)
from project_paths import PROJECT_ROOT

init_db()

st.set_page_config(
    page_title="Pattern Scanner Phone Control",
    layout="wide"
)

st.title("Pattern Scanner Phone Control")
st.caption("View scanner results, rate patterns, save notes, and generate Codex tasks.")
st.caption(f"Connected project: {PROJECT_ROOT}")

tab_scanner, tab_strategy, tab_feedback, tab_ai, tab_codex = st.tabs([
    "Scanner",
    "Strategy Tester",
    "Pattern Feedback",
    "Ask / Tell Codex",
    "Codex Files"
])


with tab_scanner:
    st.header("Scanner Results")

    patterns = get_patterns()

    if not patterns:
        st.info("No scanner results found yet.")
    else:
        for pattern in patterns:
            with st.container(border=True):
                st.subheader(pattern["setup_name"])
                st.write(pattern["summary"])

                st.caption(f"Pattern ID: {pattern['pattern_id']}")
                st.caption(f"Ticker: {pattern['ticker']} | Date: {pattern['pattern_date']}")

                stats = pattern.get("stats", {})
                if stats:
                    with st.expander("Pattern stats"):
                        st.json(stats)

                rating = st.radio(
                    "How would you rate this pattern?",
                    ["Good", "Bad", "Needs Review"],
                    horizontal=True,
                    key=f"rating_{pattern['pattern_id']}"
                )

                notes = st.text_area(
                    "Notes",
                    key=f"notes_{pattern['pattern_id']}",
                    placeholder="Example: This only worked because of one giant candle."
                )

                col1, col2 = st.columns(2)

                with col1:
                    if st.button("Save Feedback", key=f"save_{pattern['pattern_id']}"):
                        save_feedback(
                            pattern_id=pattern["pattern_id"],
                            ticker=pattern["ticker"],
                            pattern_date=pattern["pattern_date"],
                            setup_name=pattern["setup_name"],
                            rating=rating,
                            notes=notes
                        )
                        st.success("Feedback saved.")

                with col2:
                    if st.button("Create Codex Task", key=f"codex_{pattern['pattern_id']}"):
                        task = build_task_from_feedback(
                            pattern_id=pattern["pattern_id"],
                            rating=rating,
                            notes=notes
                        )
                        add_codex_task(task)
                        st.success(f"Codex task added to {TASK_FILE.name}.")


with tab_strategy:
    st.header("Strategy Tester")

    tickers = list_available_tickers()
    if not tickers:
        st.info("No local market data files found in downloaded_historical_data.")
    else:
        col1, col2, col3, col4 = st.columns([1.25, 1, 1, 1])

        with col1:
            strategy_name = st.selectbox(
                "Strategy",
                ["Box Strategy"],
                help="More strategies can be added here later.",
            )

        with col2:
            default_ticker_index = tickers.index("SPY") if "SPY" in tickers else 0
            ticker = st.selectbox("Ticker", tickers, index=default_ticker_index)

        with col3:
            timeframe = st.selectbox("Timeframe", SUPPORTED_TIMEFRAMES)

        with col4:
            days_to_test = st.number_input("Days", min_value=2, max_value=60, value=10, step=1)

        include_extended_hours = st.checkbox(
            "Show extended hours",
            value=False,
            help="Previous-day high and low are still calculated from regular market hours.",
        )

        if strategy_name == "Box Strategy":
            try:
                market_data = load_market_data(ticker, timeframe)
                chart_df, signals, results = run_box_strategy(
                    market_data,
                    include_extended_hours=include_extended_hours,
                )
            except (FileNotFoundError, ValueError) as exc:
                st.error(str(exc))
            else:
                available_days = list(chart_df["SessionDate"].drop_duplicates())
                selected_days = available_days[-int(days_to_test) :]
                strategy_days_df = chart_df[chart_df["SessionDate"].isin(selected_days)]
                if not signals.empty:
                    signals = signals[signals["SessionDate"].isin(selected_days)]
                if not results.empty:
                    results = results[results["SessionDate"].isin(selected_days)].reset_index(drop=True)
                    results["CumulativePL"] = results["PL"].cumsum()

                summary = summarize_results(results)
                metric1, metric2, metric3, metric4 = st.columns(4)
                metric1.metric("Signals", summary["trades"])
                metric2.metric("Wins", summary["wins"])
                metric3.metric("Win Rate", f"{summary['win_rate']:.0%}")
                metric4.metric("Total P/L", f"{summary['total_pl']:.2f}")

                st.caption(
                    "Test rule: the first 15 New York session minutes must sweep or open outside "
                    "the previous-day box. Price must then close back inside the box before a "
                    "continuation through that same side can trigger. Buys target 25% of the box "
                    "above the low; sells target 25% of the box below the high. Trade testing "
                    "starts at 09:30 New York time; extended hours are chart context only."
                )

                if results.empty:
                    st.info("No box strategy signals found for the selected days.")
                else:
                    st.subheader("Test Results")
                    st.dataframe(
                        results[
                            [
                                "SessionDate",
                                "EntryTime",
                                "Signal",
                                "Entry",
                                "Target",
                                "ExitTime",
                                "Exit",
                                "PL",
                                "Outcome",
                                "CumulativePL",
                                "ExitRule",
                            ]
                        ],
                        use_container_width=True,
                        hide_index=True,
                    )

                if strategy_days_df.empty:
                    st.info("Not enough data to draw strategy days yet.")
                else:
                    day_options = [str(day) for day in selected_days]
                    selected_day = st.selectbox("Day to Visualize", day_options, index=len(day_options) - 1)
                    selected_day_value = pd.to_datetime(selected_day).date()
                    if include_extended_hours:
                        prior_days = [
                            day for day in chart_df["SessionDate"].drop_duplicates() if day < selected_day_value
                        ]
                        if prior_days:
                            prior_day = prior_days[-1]
                            window_start = pd.Timestamp(f"{prior_day} {REGULAR_OPEN}")
                            window_end = pd.Timestamp(f"{selected_day_value} {REGULAR_CLOSE}")
                            day_df = chart_df[
                                (chart_df["DateTime"] >= window_start)
                                & (chart_df["DateTime"] <= window_end)
                            ]
                            st.caption(
                                f"Extended-hours view: showing {prior_day} {REGULAR_OPEN} "
                                f"through {selected_day_value} {REGULAR_CLOSE}."
                            )
                        else:
                            day_df = chart_df[chart_df["SessionDate"] == selected_day_value]
                    else:
                        day_df = strategy_days_df[strategy_days_df["SessionDate"] == selected_day_value]
                    day_signals = (
                        signals[signals["SessionDate"] == selected_day_value]
                        if not signals.empty
                        else signals
                    )

                    fig = make_box_strategy_figure(
                        day_df,
                        day_signals,
                        ticker,
                        box_session_date=selected_day_value,
                    )
                    st.plotly_chart(fig, use_container_width=True)


with tab_feedback:
    st.header("Recent Pattern Feedback")

    rows = get_recent_feedback(limit=50)

    if not rows:
        st.info("No feedback saved yet.")
    else:
        for row in rows:
            pattern_id, ticker, pattern_date, setup_name, rating, notes, created_at = row

            with st.container(border=True):
                st.subheader(f"{rating}: {setup_name}")
                st.write(notes if notes else "No notes added.")
                st.caption(f"{ticker} | {pattern_date} | {pattern_id}")
                st.caption(f"Saved: {created_at}")


with tab_ai:
    st.header("Ask / Tell Codex")

    st.write(
        "Use this to write instructions for Codex based on what you are seeing in the scanner."
    )

    user_message = st.text_area(
        "What do you want Codex to look at or change?",
        placeholder=(
            "Example: The scanner keeps finding patterns that only work because of one giant candle. "
            "Add a filter to reject those."
        )
    )

    if st.button("Save as Codex Task"):
        if user_message.strip():
            task = f"""
The user wrote this instruction from the phone Streamlit app:

{user_message}

Please review the scanner code and suggest or make the appropriate improvement.

Rules:
- Keep the existing Streamlit app working.
- Do not delete existing filters.
- Add comments explaining any new scanner logic.
- Prefer small, testable changes.
"""
            add_codex_task(task)
            st.success(f"Saved to {TASK_FILE.name}.")
        else:
            st.warning("Type an instruction first.")


with tab_codex:
    st.header("Codex Files")

    st.write(
        "This section creates files that Codex can read when you want it to improve your scanner."
    )

    if st.button("Create Feedback Summary for Codex"):
        summary = get_feedback_summary(limit=100)
        save_feedback_summary(summary)
        st.success(f"Saved to {SUMMARY_FILE.name}.")

    st.divider()

    st.subheader("What to tell Codex")
    st.code(
        "Read codex_tasks.md and codex_feedback_summary.md. "
        "Then improve my scanner based on the saved feedback. "
        "Keep the Streamlit app working and make small, testable changes.",
        language="text"
    )

    st.caption(f"Task file: {TASK_FILE}")
    st.caption(f"Feedback summary file: {SUMMARY_FILE}")
