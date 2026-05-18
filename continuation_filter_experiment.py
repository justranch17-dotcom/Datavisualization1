from pathlib import Path

import pandas as pd

from losing_signal_audit import summarize_subset


ARTIFACT_DIR = Path("model_artifacts")
HOLDOUT_BAR_RESULT_FILE = ARTIFACT_DIR / "early_holdout_bar_entry_results.csv"
REPORT_FILE = ARTIFACT_DIR / "continuation_filter_experiment_report.md"


def variant_masks(df):
    bar20 = df[df["entry_bar"] == 20].copy()
    return {
        "bar20_all": bar20,
        "bar20_short_only": bar20[bar20["trade_side"] == "short"],
        "bar20_long_only": bar20[bar20["trade_side"] == "long"],
        "bar20_ensemble_gte_080": bar20[bar20["useful_pattern_ensemble_score"] >= 0.80],
        "bar20_ensemble_gte_085": bar20[bar20["useful_pattern_ensemble_score"] >= 0.85],
        "bar20_ensemble_gte_090": bar20[bar20["useful_pattern_ensemble_score"] >= 0.90],
        "bar20_long_ensemble_gte_085_or_short": bar20[
            (bar20["trade_side"] == "short")
            | (
                (bar20["trade_side"] == "long")
                & (bar20["useful_pattern_ensemble_score"] >= 0.85)
            )
        ],
        "bar20_long_ensemble_gte_090_or_short": bar20[
            (bar20["trade_side"] == "short")
            | (
                (bar20["trade_side"] == "long")
                & (bar20["useful_pattern_ensemble_score"] >= 0.90)
            )
        ],
        "bar20_long_edge_lte_075_or_short": bar20[
            (bar20["trade_side"] == "short")
            | (
                (bar20["trade_side"] == "long")
                & (bar20["early_direction_edge"] <= 0.75)
            )
        ],
        "bar20_long_edge_lte_070_or_short": bar20[
            (bar20["trade_side"] == "short")
            | (
                (bar20["trade_side"] == "long")
                & (bar20["early_direction_edge"] <= 0.70)
            )
        ],
        "bar20_long_ensemble_gte_085_edge_lte_075_or_short": bar20[
            (bar20["trade_side"] == "short")
            | (
                (bar20["trade_side"] == "long")
                & (bar20["useful_pattern_ensemble_score"] >= 0.85)
                & (bar20["early_direction_edge"] <= 0.75)
            )
        ],
        "bar20_long_ensemble_gte_090_edge_lte_075_or_short": bar20[
            (bar20["trade_side"] == "short")
            | (
                (bar20["trade_side"] == "long")
                & (bar20["useful_pattern_ensemble_score"] >= 0.90)
                & (bar20["early_direction_edge"] <= 0.75)
            )
        ],
    }


def summarize_variants(df):
    rows = []
    for name, part in variant_masks(df).items():
        summary = summarize_subset(part, name)
        rows.append(summary)
    out = pd.DataFrame(rows)
    out = out[out["rows"] > 0].copy()
    return out.sort_values(["avg_close_return_pct", "hit_rate"], ascending=[False, False])


def format_table(df):
    if df.empty:
        return "No rows."
    return df.to_string(index=False)


def main():
    if not HOLDOUT_BAR_RESULT_FILE.exists():
        raise SystemExit("Run early_signal_holdout_validator.py first.")

    results = pd.read_csv(HOLDOUT_BAR_RESULT_FILE)
    variants = summarize_variants(results)

    top_variant_name = variants.iloc[0]["subset"]
    top_rows = variant_masks(results)[top_variant_name].sort_values("close_return_pct", ascending=True)

    display_cols = [
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
        "# Continuation Filter Experiment",
        "",
        "This is an exploratory holdout-only filter experiment. It tests simple bar-20 guardrails suggested by the losing-signal audit. Treat it as a clue generator, not as final strategy validation.",
        "",
        "## Variant Results",
        "",
        "```text",
        format_table(variants),
        "```",
        "",
        "## Best Variant Worst Rows",
        "",
        "```text",
        top_rows[display_cols].head(30).to_string(index=False),
        "```",
        "",
        "## Practical Read",
        "",
        "```text",
        "The current best exploratory variants favor bar-20 entries and either shorts or longs with stronger ensemble confirmation.",
        "Raising early_pattern_probability alone is not the right next filter; the failure mode is long-side exhaustion.",
        "The next model feature should describe post-entry continuation pressure from raw bars, not just the early normalized shape.",
        "```",
    ]

    REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(REPORT_FILE.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
