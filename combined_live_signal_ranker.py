from pathlib import Path

import pandas as pd


ARTIFACT_DIR = Path("model_artifacts")
TIMING_PREDICTION_FILE = ARTIFACT_DIR / "entry_bar_timing_predictions.csv"
OUTPUT_FILE = ARTIFACT_DIR / "combined_live_signal_rankings.csv"
REPORT_FILE = ARTIFACT_DIR / "combined_live_signal_ranker_report.md"

DEFAULT_ENTRY_BAR = 8
COMPARISON_ENTRY_BAR = 20


def percentile_rank(series):
    return series.rank(method="average", pct=True)


def load_entry_predictions(entry_bar):
    if not TIMING_PREDICTION_FILE.exists():
        raise SystemExit("Run entry_bar_timing_sweep.py first.")
    df = pd.read_csv(TIMING_PREDICTION_FILE)
    df = df[df["entry_bar"] == entry_bar].copy()
    if df.empty:
        raise SystemExit(f"No timing predictions found for entry bar {entry_bar}.")
    return df


def add_live_scores(df):
    df = df.copy()
    df["entry_quality_rank"] = percentile_rank(df["entry_quality_probability"])
    df["early_pattern_rank"] = percentile_rank(df["early_pattern_probability"])
    df["direction_edge_rank"] = percentile_rank(df["early_direction_edge"].abs())
    df["ensemble_rank"] = percentile_rank(df["useful_pattern_ensemble_score"])

    df["combined_live_score"] = (
        df["entry_quality_rank"] * 0.45
        + df["early_pattern_rank"] * 0.20
        + df["direction_edge_rank"] * 0.20
        + df["ensemble_rank"] * 0.15
    )

    df["guardrail_short_clean"] = df["implied_side"] == "short"
    df["guardrail_long_structure"] = (
        (df["implied_side"] == "long")
        & (df["useful_pattern_ensemble_score"] >= 0.90)
        & (df["early_direction_edge"] <= 0.75)
    )
    df["passes_exploratory_guardrail"] = df["guardrail_short_clean"] | df["guardrail_long_structure"]
    return df.sort_values("combined_live_score", ascending=False)


def summarize_ranked(df, score_col):
    rows = []
    for label, quantile in [("top_1pct", 0.99), ("top_5pct", 0.95), ("top_10pct", 0.90), ("bottom_50pct", 0.50)]:
        if label == "bottom_50pct":
            part = df[df[score_col] <= df[score_col].quantile(quantile)]
        else:
            part = df[df[score_col] >= df[score_col].quantile(quantile)]
        rows.append(
            {
                "slice": label,
                "rows": len(part),
                "avg_signed_return": part["signed_entry_to_close_return_pct"].mean(),
                "hit_rate": part["target_followthrough"].mean(),
                "strong_hit_rate": part["target_strong_followthrough"].mean(),
                "guardrail_pass_rate": part["passes_exploratory_guardrail"].mean(),
                "short_rate": (part["implied_side"] == "short").mean(),
            }
        )
    return pd.DataFrame(rows)


def compare_entry_bars():
    rows = []
    for entry_bar in [DEFAULT_ENTRY_BAR, COMPARISON_ENTRY_BAR]:
        df = add_live_scores(load_entry_predictions(entry_bar))
        summary = summarize_ranked(df, "combined_live_score")
        top5 = summary[summary["slice"] == "top_5pct"].iloc[0]
        top10 = summary[summary["slice"] == "top_10pct"].iloc[0]
        rows.append(
            {
                "entry_bar": entry_bar,
                "rows": len(df),
                "top5_avg_signed_return": top5["avg_signed_return"],
                "top5_hit_rate": top5["hit_rate"],
                "top5_guardrail_pass_rate": top5["guardrail_pass_rate"],
                "top10_avg_signed_return": top10["avg_signed_return"],
                "top10_hit_rate": top10["hit_rate"],
            }
        )
    return pd.DataFrame(rows)


def main():
    ranked = add_live_scores(load_entry_predictions(DEFAULT_ENTRY_BAR))
    ranked.to_csv(OUTPUT_FILE, index=False)

    summary = summarize_ranked(ranked, "combined_live_score")
    guarded = ranked[ranked["passes_exploratory_guardrail"]].copy()
    guarded_summary = summarize_ranked(guarded, "combined_live_score") if not guarded.empty else pd.DataFrame()
    comparison = compare_entry_bars()

    display_cols = [
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

    top_guarded = guarded.sort_values("combined_live_score", ascending=False)

    lines = [
        "# Combined Live Signal Ranker",
        "",
        "This ranks candidates using the current best entry timing result. It is a live-style research ranking, not an execution system.",
        "",
        "## Score",
        "",
        "```text",
        "combined_live_score =",
        "  0.45 * entry_quality_rank",
        "  0.20 * early_pattern_rank",
        "  0.20 * abs(direction_edge)_rank",
        "  0.15 * ensemble_rank",
        "```",
        "",
        "## Entry Bar Comparison",
        "",
        "```text",
        comparison.to_string(index=False),
        "```",
        "",
        f"## Default Entry Bar {DEFAULT_ENTRY_BAR} Summary",
        "",
        "```text",
        summary.to_string(index=False),
        "```",
        "",
        "## Guarded Candidate Summary",
        "",
        "```text",
        guarded_summary.to_string(index=False) if not guarded_summary.empty else "No guarded candidates.",
        "```",
        "",
        "## Top Combined Candidates",
        "",
        "```text",
        ranked[display_cols].head(40).to_string(index=False),
        "```",
        "",
        "## Top Guarded Candidates",
        "",
        "```text",
        top_guarded[display_cols].head(40).to_string(index=False),
        "```",
    ]

    REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(REPORT_FILE.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
