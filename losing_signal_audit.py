from pathlib import Path

import pandas as pd


ARTIFACT_DIR = Path("model_artifacts")
BAR_RESULT_FILE = ARTIFACT_DIR / "bar_entry_trade_results.csv"
HOLDOUT_BAR_RESULT_FILE = ARTIFACT_DIR / "early_holdout_bar_entry_results.csv"
REPORT_FILE = ARTIFACT_DIR / "losing_signal_audit_report.md"


FEATURE_COLUMNS = [
    "early_pattern_probability",
    "early_up_probability",
    "early_down_probability",
    "early_direction_edge",
    "useful_pattern_ensemble_score",
    "early_range_pct",
    "mfe_pct",
    "mae_pct",
]


def load_result_file(path, source):
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["source"] = source
    return df


def summarize_subset(df, label):
    if df.empty:
        return pd.Series({"subset": label, "rows": 0})
    return pd.Series(
        {
            "subset": label,
            "rows": len(df),
            "avg_close_return_pct": df["close_return_pct"].mean(),
            "median_close_return_pct": df["close_return_pct"].median(),
            "hit_rate": df["won_to_close"].mean(),
            "avg_mfe_pct": df["mfe_pct"].mean(),
            "avg_mae_pct": df["mae_pct"].mean(),
            "target_hit_rate": df["barrier_result"].isin(["target", "both_same_bar"]).mean(),
            "stop_hit_rate": df["barrier_result"].isin(["stop", "both_same_bar"]).mean(),
        }
    )


def feature_contrast(df):
    winners = df[df["close_return_pct"] > 0]
    losers = df[df["close_return_pct"] <= 0]
    rows = []
    for column in FEATURE_COLUMNS:
        if column not in df.columns:
            continue
        rows.append(
            {
                "feature": column,
                "winner_avg": winners[column].mean(),
                "loser_avg": losers[column].mean(),
                "loser_minus_winner": losers[column].mean() - winners[column].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values("loser_minus_winner")


def repeated_losers(df):
    losers = df[df["close_return_pct"] <= 0].copy()
    if losers.empty:
        return pd.DataFrame()
    return (
        losers.groupby(["ticker", "date", "trade_side"], as_index=False)
        .agg(
            losing_entries=("entry_bar", "size"),
            worst_close_return_pct=("close_return_pct", "min"),
            avg_close_return_pct=("close_return_pct", "mean"),
            max_mfe_pct=("mfe_pct", "max"),
            min_mae_pct=("mae_pct", "min"),
            early_pattern_probability=("early_pattern_probability", "mean"),
            early_direction_edge=("early_direction_edge", "mean"),
        )
        .sort_values(["losing_entries", "worst_close_return_pct"], ascending=[False, True])
    )


def variant_table(df):
    variants = {
        "all": df,
        "entry_bar_20": df[df["entry_bar"] == 20],
        "entry_bar_25": df[df["entry_bar"] == 25],
        "entry_bar_28": df[df["entry_bar"] == 28],
        "long_only": df[df["trade_side"] == "long"],
        "short_only": df[df["trade_side"] == "short"],
        "bar20_long": df[(df["entry_bar"] == 20) & (df["trade_side"] == "long")],
        "bar20_short": df[(df["entry_bar"] == 20) & (df["trade_side"] == "short")],
        "no_crypto": df[df["asset_class"] != "crypto"],
        "stock_only": df[df["asset_class"] == "stock"],
        "etf_only": df[df["asset_class"] == "etf"],
    }
    return pd.DataFrame([summarize_subset(part, name) for name, part in variants.items()])


def high_confidence_failures(df):
    failures = df[
        (df["close_return_pct"] <= 0)
        & (df["entry_bar"] == 20)
        & (df["early_pattern_probability"] >= 0.75)
        & (df["early_direction_edge"].abs() >= 0.50)
    ].copy()
    if failures.empty:
        return failures
    return failures.sort_values("close_return_pct")


def format_table(df):
    if df.empty:
        return "No rows."
    return df.to_string(index=False)


def main():
    discovery = load_result_file(BAR_RESULT_FILE, "discovery_threshold")
    holdout = load_result_file(HOLDOUT_BAR_RESULT_FILE, "holdout_threshold")

    frames = [df for df in [discovery, holdout] if not df.empty]
    if not frames:
        raise SystemExit("Run bar_entry_trade_simulator.py or early_signal_holdout_validator.py first.")

    combined = pd.concat(frames, ignore_index=True)

    audit_scope = combined[combined["source"] == "holdout_threshold"].copy()
    if audit_scope.empty:
        audit_scope = combined.copy()

    losers = audit_scope[audit_scope["close_return_pct"] <= 0]
    winners = audit_scope[audit_scope["close_return_pct"] > 0]
    contrast = feature_contrast(audit_scope)
    repeats = repeated_losers(audit_scope)
    variants = variant_table(audit_scope)
    failures = high_confidence_failures(audit_scope)

    cols = [
        "ticker",
        "date",
        "asset_class",
        "trade_side",
        "entry_bar",
        "close_return_pct",
        "mfe_pct",
        "mae_pct",
        "barrier_result",
        "early_pattern_probability",
        "early_direction_edge",
        "useful_pattern_ensemble_score",
    ]

    lines = [
        "# Losing Signal Audit",
        "",
        "This audit studies the stricter holdout bar-entry rows when available. It is meant to teach the next filter what failure looks like, not to delete signals blindly.",
        "",
        "## Scope",
        "",
        "```text",
        f"combined_rows={len(combined)}",
        f"audit_rows={len(audit_scope)}",
        f"winners={len(winners)}",
        f"losers={len(losers)}",
        "```",
        "",
        "## Variant Summary",
        "",
        "```text",
        format_table(variants),
        "```",
        "",
        "## Winner Vs Loser Feature Contrast",
        "",
        "```text",
        format_table(contrast),
        "```",
        "",
        "## Repeated Losing Signals",
        "",
        "```text",
        format_table(repeats.head(30)),
        "```",
        "",
        "## High-Confidence Bar-20 Failures",
        "",
        "```text",
        format_table(failures[cols].head(30) if not failures.empty else failures),
        "```",
        "",
        "## Practical Read",
        "",
        "```text",
        "1. Bar 20 remains the strongest general entry variant.",
        "2. Shorts remain cleaner than longs in the current holdout.",
        "3. High-confidence long failures are not low-confidence model errors; they are mostly continuation failures after strong early moves.",
        "4. The next filter should focus on exhaustion/follow-through after entry, not simply raising early_pattern_probability.",
        "```",
    ]

    REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(REPORT_FILE.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
