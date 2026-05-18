from pathlib import Path

import pandas as pd


ARTIFACT_DIR = Path("model_artifacts")
SUMMARY_SCORE_FILE = ARTIFACT_DIR / "structural_day_scores.csv"
RICH_SCORE_FILE = ARTIFACT_DIR / "structural_day_rich_scores.csv"
ENSEMBLE_SCORE_FILE = ARTIFACT_DIR / "structural_day_ensemble_scores.csv"
REPORT_FILE = ARTIFACT_DIR / "score_backtest_report.md"


def evaluate_score_file(path, score_column):
    if not path.exists():
        return f"## {path.name}\n\nMissing file.\n", pd.DataFrame()

    df = pd.read_csv(path)
    if score_column not in df.columns:
        return f"## {path.name}\n\nMissing score column `{score_column}`.\n", pd.DataFrame()

    df = df.dropna(subset=[score_column, "avg_return"])
    if df.empty:
        return f"## {path.name}\n\nNo scored rows.\n", pd.DataFrame()

    df["abs_return"] = df["avg_return"].abs()
    df["score_bucket"] = pd.qcut(df[score_column].rank(method="first"), q=10, labels=False) + 1

    bucket = (
        df.groupby("score_bucket", as_index=False)
        .agg(
            rows=("ticker", "size"),
            avg_score=(score_column, "mean"),
            avg_abs_return=("abs_return", "mean"),
            median_abs_return=("abs_return", "median"),
            avg_range=("avg_range", "mean"),
            avg_shift_strength=("avg_shift_strength", "mean"),
            avg_shift_count=("avg_shift_count", "mean"),
        )
        .sort_values("score_bucket", ascending=False)
    )

    top_1 = df[df[score_column] >= df[score_column].quantile(0.99)]
    top_5 = df[df[score_column] >= df[score_column].quantile(0.95)]
    bottom_50 = df[df[score_column] <= df[score_column].quantile(0.50)]

    lift_1 = top_1["abs_return"].mean() / max(0.0001, bottom_50["abs_return"].mean())
    lift_5 = top_5["abs_return"].mean() / max(0.0001, bottom_50["abs_return"].mean())

    top_rows = df.sort_values(score_column, ascending=False).head(20)[
        ["ticker", "date", score_column, "avg_return", "avg_range", "avg_early_move", "avg_shift_strength", "avg_shift_count"]
    ]

    text = [
        f"## {path.name}",
        "",
        f"Rows scored: {len(df):,}",
        f"Top 1% avg abs return: {top_1['abs_return'].mean():.4f}",
        f"Top 5% avg abs return: {top_5['abs_return'].mean():.4f}",
        f"Bottom 50% avg abs return: {bottom_50['abs_return'].mean():.4f}",
        f"Top 1% lift vs bottom half: {lift_1:.3f}x",
        f"Top 5% lift vs bottom half: {lift_5:.3f}x",
        "",
        "### Decile Summary",
        "",
        "```text",
        bucket.to_string(index=False),
        "```",
        "",
        "### Top 20 Candidates",
        "",
        "```text",
        top_rows.to_string(index=False),
        "```",
        "",
    ]

    return "\n".join(text), bucket


def main():
    ARTIFACT_DIR.mkdir(exist_ok=True)
    sections = ["# Score Backtest Report", ""]

    summary_text, _ = evaluate_score_file(SUMMARY_SCORE_FILE, "useful_pattern_score")
    rich_text, _ = evaluate_score_file(RICH_SCORE_FILE, "useful_pattern_rich_score")
    ensemble_text, _ = evaluate_score_file(ENSEMBLE_SCORE_FILE, "useful_pattern_ensemble_score")
    sections.extend([summary_text, rich_text, ensemble_text])

    REPORT_FILE.write_text("\n\n".join(sections), encoding="utf-8")
    print(REPORT_FILE.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
