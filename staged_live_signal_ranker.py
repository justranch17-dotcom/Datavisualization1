from pathlib import Path

import numpy as np
import pandas as pd


ARTIFACT_DIR = Path("model_artifacts")
ENTRY_PREDICTION_FILE = ARTIFACT_DIR / "entry_bar_timing_predictions.csv"
RAW_STRUCTURE_FILE = ARTIFACT_DIR / "raw_early_structure_predictions.csv"
OUTPUT_FILE = ARTIFACT_DIR / "staged_live_signal_rankings.csv"
SUMMARY_FILE = ARTIFACT_DIR / "staged_live_signal_summary.csv"
REPORT_FILE = ARTIFACT_DIR / "staged_live_signal_ranker_report.md"

VARIANTS = {
    "early_aggressive_bar8": {"entry_bar": 8, "structure_bar": 8},
    "confirmed_bar20": {"entry_bar": 20, "structure_bar": 20},
    "confirmed_bar28": {"entry_bar": 28, "structure_bar": 28},
}


def percentile_rank(series):
    return series.rank(method="average", pct=True)


def load_inputs():
    if not ENTRY_PREDICTION_FILE.exists():
        raise SystemExit("Run entry_bar_timing_sweep.py first.")
    if not RAW_STRUCTURE_FILE.exists():
        raise SystemExit("Run raw_early_structure_predictor.py first.")

    entry = pd.read_csv(ENTRY_PREDICTION_FILE)
    structure = pd.read_csv(RAW_STRUCTURE_FILE)[
        [
            "ticker",
            "date",
            "entry_bar",
            "raw_early_structure_probability",
            "target_high_structure",
        ]
    ].rename(
        columns={
            "entry_bar": "structure_bar",
            "raw_early_structure_probability": "raw_structure_probability",
        }
    )
    return entry, structure


def guardrails(df):
    short_clean = df["implied_side"] == "short"
    long_structure = (
        (df["implied_side"] == "long")
        & (df["useful_pattern_ensemble_score"] >= 0.90)
        & (df["early_direction_edge"] <= 0.75)
    )
    return short_clean | long_structure


def build_variant(entry, structure, name, entry_bar, structure_bar):
    entry_rows = entry[entry["entry_bar"] == entry_bar].copy()
    structure_rows = structure[structure["structure_bar"] == structure_bar].copy()
    merged = entry_rows.merge(structure_rows, on=["ticker", "date"], how="inner")
    if merged.empty:
        return merged

    merged["variant"] = name
    merged["entry_quality_rank"] = percentile_rank(merged["entry_quality_probability"])
    merged["raw_structure_rank"] = percentile_rank(merged["raw_structure_probability"])
    merged["direction_edge_rank"] = percentile_rank(merged["early_direction_edge"].abs())

    merged["staged_live_score"] = (
        merged["entry_quality_rank"] * 0.45
        + merged["raw_structure_rank"] * 0.30
        + merged["direction_edge_rank"] * 0.25
    )
    merged["passes_exploratory_guardrail"] = guardrails(merged)
    merged["structure_confirmed"] = merged["raw_structure_probability"] >= merged[
        "raw_structure_probability"
    ].quantile(0.70)
    return merged


def summarize(df):
    rows = []
    for variant, variant_df in df.groupby("variant"):
        for score_col in ["staged_live_score", "entry_quality_probability"]:
            for label, quantile in [
                ("top_1pct", 0.99),
                ("top_5pct", 0.95),
                ("top_10pct", 0.90),
                ("bottom_50pct", 0.50),
            ]:
                if label == "bottom_50pct":
                    part = variant_df[variant_df[score_col] <= variant_df[score_col].quantile(quantile)]
                else:
                    part = variant_df[variant_df[score_col] >= variant_df[score_col].quantile(quantile)]
                rows.append(
                    {
                        "variant": variant,
                        "score_col": score_col,
                        "slice": label,
                        "rows": len(part),
                        "avg_signed_return": part["signed_entry_to_close_return_pct"].mean(),
                        "median_signed_return": part["signed_entry_to_close_return_pct"].median(),
                        "hit_rate": part["target_followthrough"].mean(),
                        "strong_hit_rate": part["target_strong_followthrough"].mean(),
                        "target_high_structure_rate": part["target_high_structure"].mean(),
                        "avg_raw_structure_probability": part["raw_structure_probability"].mean(),
                        "guardrail_pass_rate": part["passes_exploratory_guardrail"].mean(),
                        "short_rate": (part["implied_side"] == "short").mean(),
                    }
                )
    return pd.DataFrame(rows)


def false_positive_audit(df):
    rows = []
    early = df[df["variant"] == "early_aggressive_bar8"].copy()
    for confirmed_variant in ["confirmed_bar20", "confirmed_bar28"]:
        confirmed = df[df["variant"] == confirmed_variant][
            ["ticker", "date", "raw_structure_probability", "structure_confirmed"]
        ].rename(
            columns={
                "raw_structure_probability": f"{confirmed_variant}_raw_structure_probability",
                "structure_confirmed": f"{confirmed_variant}_structure_confirmed",
            }
        )
        joined = early.merge(confirmed, on=["ticker", "date"], how="inner")
        if joined.empty:
            continue

        early_top = joined[joined["staged_live_score"] >= joined["staged_live_score"].quantile(0.95)]
        losers = early_top[early_top["target_followthrough"] == 0]
        confirm_col = f"{confirmed_variant}_structure_confirmed"
        rows.append(
            {
                "confirmed_variant": confirmed_variant,
                "early_top5_rows": len(early_top),
                "early_top5_losers": len(losers),
                "early_top5_loss_rate": len(losers) / len(early_top) if len(early_top) else np.nan,
                "losers_confirmed_rate": losers[confirm_col].mean() if len(losers) else np.nan,
                "winners_confirmed_rate": early_top[early_top["target_followthrough"] == 1][confirm_col].mean(),
            }
        )
    return pd.DataFrame(rows)


def main():
    entry, structure = load_inputs()
    parts = []
    for name, config in VARIANTS.items():
        part = build_variant(entry, structure, name, config["entry_bar"], config["structure_bar"])
        if not part.empty:
            parts.append(part)

    if not parts:
        raise SystemExit("No staged variants could be built.")

    ranked = pd.concat(parts, ignore_index=True).sort_values(
        ["variant", "staged_live_score"], ascending=[True, False]
    )
    summary = summarize(ranked)
    audit = false_positive_audit(ranked)

    ranked.to_csv(OUTPUT_FILE, index=False)
    summary.to_csv(SUMMARY_FILE, index=False)

    display_cols = [
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

    top_lines = []
    for variant in VARIANTS:
        variant_rows = ranked[ranked["variant"] == variant].sort_values("staged_live_score", ascending=False)
        if variant_rows.empty:
            continue
        top_lines.extend(
            [
                f"### {variant}",
                "",
                "```text",
                variant_rows[display_cols].head(25).to_string(index=False),
                "```",
                "",
            ]
        )

    lines = [
        "# Staged Live Signal Ranker",
        "",
        "This compares an aggressive bar-8 entry workflow against bar-20 and bar-28 confirmation workflows using live-safe raw early-structure probabilities.",
        "",
        "## Score",
        "",
        "```text",
        "staged_live_score =",
        "  0.45 * entry_quality_rank",
        "  0.30 * raw_structure_rank",
        "  0.25 * abs(direction_edge)_rank",
        "```",
        "",
        "## Summary",
        "",
        "```text",
        summary.to_string(index=False),
        "```",
        "",
        "## Bar-8 False Positive Audit",
        "",
        "```text",
        audit.to_string(index=False) if not audit.empty else "No audit rows.",
        "```",
        "",
        "## Top Staged Candidates",
        "",
        *top_lines,
    ]

    REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(REPORT_FILE.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
