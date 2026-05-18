from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from entry_bar_timing_sweep import (
    ENTRY_BARS,
    FEATURE_FILE as TIMING_FEATURE_FILE,
    build_feature_rows,
    chronological_split,
)
from early_trade_simulator import load_predictions


ARTIFACT_DIR = Path("model_artifacts")
OUTPUT_FILE = ARTIFACT_DIR / "live_replay_scanner_results.csv"
MODEL_FILE = ARTIFACT_DIR / "live_replay_scanner_model.joblib"
REPORT_FILE = ARTIFACT_DIR / "live_replay_scanner_report.md"

DEFAULT_ENTRY_BAR = 8


RAW_REPLAY_FEATURES = [
    "session_open_to_entry_pct",
    "entry_early_range_pct",
    "entry_close_position",
    "entry_distance_from_high_pct",
    "entry_distance_from_low_pct",
    "entry_last5_return_pct",
    "entry_last5_range_pct",
    "entry_last5_body_pct",
    "entry_volume_late_early_ratio",
    "entry_last5_volume_share",
]


def percentile_rank(series):
    return series.rank(method="average", pct=True)


def load_or_build_timing_features():
    if TIMING_FEATURE_FILE.exists():
        return pd.read_csv(TIMING_FEATURE_FILE)

    predictions = load_predictions()
    features = build_feature_rows(predictions)
    if features.empty:
        raise SystemExit("No replay features could be built.")
    features.to_csv(TIMING_FEATURE_FILE, index=False)
    return features


def clean_matrix(df, columns):
    return df[columns].replace([np.inf, -np.inf], np.nan).fillna(0)


def train_raw_replay_model(train, test):
    columns = [column for column in RAW_REPLAY_FEATURES if column in train.columns]
    if len(columns) != len(RAW_REPLAY_FEATURES):
        missing = sorted(set(RAW_REPLAY_FEATURES) - set(columns))
        raise SystemExit(f"Missing replay feature columns: {missing}")

    x_train = clean_matrix(train, columns)
    y_train = train["target_followthrough"]
    x_test = clean_matrix(test, columns)
    y_test = test["target_followthrough"]

    model = RandomForestClassifier(
        n_estimators=600,
        min_samples_leaf=8,
        class_weight="balanced",
        random_state=808,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)

    probabilities = model.predict_proba(x_test)[:, 1]
    auc = roc_auc_score(y_test, probabilities) if y_test.nunique() == 2 else np.nan
    ap = average_precision_score(y_test, probabilities)
    return model, columns, probabilities, auc, ap


def add_scores(test):
    test = test.copy()
    test["raw_replay_rank"] = percentile_rank(test["raw_replay_probability"])
    test["early_pattern_rank"] = percentile_rank(test["early_pattern_probability"])
    test["direction_edge_rank"] = percentile_rank(test["early_direction_edge"].abs())
    test["ensemble_rank"] = percentile_rank(test["useful_pattern_ensemble_score"])

    test["research_context_score"] = (
        test["raw_replay_rank"] * 0.60
        + test["early_pattern_rank"] * 0.15
        + test["direction_edge_rank"] * 0.15
        + test["ensemble_rank"] * 0.10
    )

    test["guardrail_short_clean"] = test["implied_side"] == "short"
    test["guardrail_long_structure"] = (
        (test["implied_side"] == "long")
        & (test["useful_pattern_ensemble_score"] >= 0.90)
        & (test["early_direction_edge"] <= 0.75)
    )
    test["passes_exploratory_guardrail"] = test["guardrail_short_clean"] | test["guardrail_long_structure"]
    return test


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
                "median_signed_return": part["signed_entry_to_close_return_pct"].median(),
                "hit_rate": part["target_followthrough"].mean(),
                "strong_hit_rate": part["target_strong_followthrough"].mean(),
                "guardrail_pass_rate": part["passes_exploratory_guardrail"].mean(),
                "short_rate": (part["implied_side"] == "short").mean(),
            }
        )
    return pd.DataFrame(rows)


def side_summary(df, score_col):
    top = df[df[score_col] >= df[score_col].quantile(0.95)].copy()
    if top.empty:
        return pd.DataFrame()
    return (
        top.groupby(["asset_class", "implied_side"], as_index=False)
        .agg(
            rows=("ticker", "size"),
            avg_score=(score_col, "mean"),
            avg_signed_return=("signed_entry_to_close_return_pct", "mean"),
            median_signed_return=("signed_entry_to_close_return_pct", "median"),
            hit_rate=("target_followthrough", "mean"),
            strong_hit_rate=("target_strong_followthrough", "mean"),
        )
        .sort_values(["avg_signed_return", "hit_rate"], ascending=[False, False])
    )


def report_lines(features, train, test, split_date, auc, ap):
    raw_summary = summarize_ranked(test, "raw_replay_probability")
    context_summary = summarize_ranked(test, "research_context_score")
    guarded = test[test["passes_exploratory_guardrail"]].copy()
    guarded_summary = (
        summarize_ranked(guarded, "raw_replay_probability") if not guarded.empty else pd.DataFrame()
    )
    side = side_summary(test, "raw_replay_probability")

    display_cols = [
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
        "entry_time",
        "entry_price",
        "entry_close_position",
        "entry_last5_return_pct",
        "entry_early_range_pct",
    ]

    top_raw = test.sort_values("raw_replay_probability", ascending=False)
    top_context = test.sort_values("research_context_score", ascending=False)

    return [
        "# Live Replay Scanner",
        "",
        "This is the first pass that scores entry quality from raw bars known by the entry checkpoint. The primary `raw_replay_probability` uses only OHLCV-derived features available by bar 8.",
        "",
        "The optional `research_context_score` blends the raw replay rank with the older structural context columns. That context is useful for research comparison, but the raw replay score is the cleaner live-style signal.",
        "",
        "## Split",
        "",
        "```text",
        f"feature_rows={len(features)}",
        f"entry_bar={DEFAULT_ENTRY_BAR}",
        f"train_rows={len(train)}",
        f"test_rows={len(test)}",
        f"split_date={split_date}",
        f"roc_auc={auc:.6f}",
        f"average_precision={ap:.6f}",
        "```",
        "",
        "## Raw Bar-8 Replay Summary",
        "",
        "```text",
        raw_summary.to_string(index=False),
        "```",
        "",
        "## Research Context Summary",
        "",
        "```text",
        context_summary.to_string(index=False),
        "```",
        "",
        "## Guarded Raw Replay Summary",
        "",
        "```text",
        guarded_summary.to_string(index=False) if not guarded_summary.empty else "No guarded rows.",
        "```",
        "",
        "## Top 5% Raw Replay By Side",
        "",
        "```text",
        side.to_string(index=False) if not side.empty else "No side summary.",
        "```",
        "",
        "## Top Raw Replay Candidates",
        "",
        "```text",
        top_raw[display_cols].head(40).to_string(index=False),
        "```",
        "",
        "## Top Research Context Candidates",
        "",
        "```text",
        top_context[display_cols].head(40).to_string(index=False),
        "```",
    ]


def main():
    features = load_or_build_timing_features()
    features = features[features["entry_bar"] == DEFAULT_ENTRY_BAR].copy()
    if features.empty:
        raise SystemExit(f"No features found for entry bar {DEFAULT_ENTRY_BAR}. Expected one of {ENTRY_BARS}.")

    train, test, split_date = chronological_split(features)
    model, columns, probabilities, auc, ap = train_raw_replay_model(train, test)

    test = test.copy()
    test["raw_replay_probability"] = probabilities
    test = add_scores(test).sort_values("raw_replay_probability", ascending=False)
    test.to_csv(OUTPUT_FILE, index=False)
    joblib.dump(
        {
            "model": model,
            "feature_columns": columns,
            "entry_bar": DEFAULT_ENTRY_BAR,
            "primary_score": "raw_replay_probability",
        },
        MODEL_FILE,
    )

    REPORT_FILE.write_text("\n".join(report_lines(features, train, test, split_date, auc, ap)), encoding="utf-8")
    print(REPORT_FILE.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
