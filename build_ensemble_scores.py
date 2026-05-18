from pathlib import Path

import pandas as pd


ARTIFACT_DIR = Path("model_artifacts")
SUMMARY_SCORE_FILE = ARTIFACT_DIR / "structural_day_scores.csv"
RICH_SCORE_FILE = ARTIFACT_DIR / "structural_day_rich_scores.csv"
ENSEMBLE_SCORE_FILE = ARTIFACT_DIR / "structural_day_ensemble_scores.csv"


def percentile_rank(series):
    return series.rank(method="average", pct=True)


def main():
    if not SUMMARY_SCORE_FILE.exists() or not RICH_SCORE_FILE.exists():
        raise SystemExit("Run structural_pattern_learner.py --model-mode both first.")

    summary = pd.read_csv(SUMMARY_SCORE_FILE)
    rich = pd.read_csv(RICH_SCORE_FILE)[["ticker", "date", "useful_pattern_rich_score"]]
    merged = summary.merge(rich, on=["ticker", "date"], how="left")

    merged["summary_rank"] = percentile_rank(merged["useful_pattern_score"])
    merged["rich_rank"] = percentile_rank(merged["useful_pattern_rich_score"])

    # The summary model is the better classifier; the rich model is the better
    # shape/movement ranker. Weight summary slightly higher for stability.
    merged["useful_pattern_ensemble_score"] = (
        merged["summary_rank"] * 0.60 + merged["rich_rank"] * 0.40
    )

    merged = merged.sort_values("useful_pattern_ensemble_score", ascending=False)
    merged.to_csv(ENSEMBLE_SCORE_FILE, index=False)

    print(f"Saved {ENSEMBLE_SCORE_FILE}")
    print(
        merged[
            [
                "ticker",
                "date",
                "useful_pattern_ensemble_score",
                "useful_pattern_score",
                "useful_pattern_rich_score",
                "avg_return",
                "avg_range",
                "avg_early_move",
                "avg_shift_strength",
                "avg_shift_count",
            ]
        ]
        .head(30)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
