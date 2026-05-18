from pathlib import Path

import numpy as np
import pandas as pd


ARTIFACT_DIR = Path("model_artifacts")
GROUP_FILE = ARTIFACT_DIR / "pattern_grouping_candidates.csv"
MEMBER_FILE = ARTIFACT_DIR / "pattern_grouping_members.csv"
ENSEMBLE_FILE = ARTIFACT_DIR / "structural_day_ensemble_scores.csv"
QUEUE_FILE = ARTIFACT_DIR / "pattern_review_queue.csv"
REPORT_FILE = ARTIFACT_DIR / "pattern_review_queue_report.md"


def load_inputs():
    missing = [path for path in [GROUP_FILE, MEMBER_FILE, ENSEMBLE_FILE] if not path.exists()]
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise SystemExit(f"Missing required artifacts:\n{missing_text}")

    groups = pd.read_csv(GROUP_FILE)
    members = pd.read_csv(MEMBER_FILE)
    scores = pd.read_csv(ENSEMBLE_FILE)

    members["date"] = pd.to_datetime(members["date"]).dt.date
    scores["date"] = pd.to_datetime(scores["date"]).dt.date

    return groups, members, scores


def pattern_type(pattern_name):
    return str(pattern_name).split(": ")[-1]


def uncertainty_score(values):
    if values.empty:
        return 0.5
    mean_score = float(values.mean())
    # Highest when the model score is in the undecided middle, lower at extremes.
    return max(0.0, 1.0 - abs(mean_score - 0.50) / 0.50)


def build_queue(groups, members, scores):
    member_scores = members.merge(
        scores[
            [
                "ticker",
                "date",
                "useful_pattern_ensemble_score",
                "useful_pattern_score",
                "useful_pattern_rich_score",
            ]
        ],
        on=["ticker", "date"],
        how="left",
    )

    score_summary = (
        member_scores.groupby("group_key", as_index=False)
        .agg(
            member_rows=("date", "size"),
            avg_ensemble_score=("useful_pattern_ensemble_score", "mean"),
            median_ensemble_score=("useful_pattern_ensemble_score", "median"),
            avg_summary_score=("useful_pattern_score", "mean"),
            avg_rich_score=("useful_pattern_rich_score", "mean"),
            nearest_dates=("date", lambda x: ", ".join(str(value) for value in list(x)[:8])),
        )
    )
    score_summary["model_uncertainty"] = (
        member_scores.groupby("group_key")["useful_pattern_ensemble_score"]
        .apply(uncertainty_score)
        .reindex(score_summary["group_key"])
        .to_numpy()
    )

    queue = groups.merge(score_summary, on="group_key", how="left")
    queue["pattern_type"] = queue["pattern_name"].map(pattern_type)
    queue["avg_ensemble_score"] = queue["avg_ensemble_score"].fillna(0.0)
    queue["model_uncertainty"] = queue["model_uncertainty"].fillna(0.5)
    queue["member_rows"] = queue["member_rows"].fillna(queue["days_in_group"]).astype(int)

    queue["review_priority"] = (
        queue["tightness_score"].clip(0, 1) * 0.38
        + queue["model_uncertainty"].clip(0, 1) * 0.24
        + queue["avg_shape_corr"].clip(0, 1) * 0.18
        + np.minimum(queue["days_in_group"] / 12.0, 1.0) * 0.12
        + np.minimum(queue["median_abs_return"] / 2.0, 1.0) * 0.08
    )

    queue["review_reason"] = np.select(
        [
            queue["model_uncertainty"] >= 0.75,
            queue["tightness_score"] >= 0.88,
            queue["days_in_group"] >= 10,
        ],
        [
            "model unsure on a coherent family",
            "very tight signature family",
            "larger repeated family",
        ],
        default="candidate for human pattern rating",
    )

    return queue.sort_values(
        ["review_priority", "tightness_score", "days_in_group"],
        ascending=[False, False, False],
    )


def write_report(queue):
    top = queue.head(100)
    summary = (
        queue.groupby(["pattern_type", "group_count_setting"], as_index=False)
        .agg(
            candidates=("group_key", "size"),
            avg_priority=("review_priority", "mean"),
            avg_tightness=("tightness_score", "mean"),
            avg_uncertainty=("model_uncertainty", "mean"),
            avg_days=("days_in_group", "mean"),
        )
        .sort_values(["avg_priority", "candidates"], ascending=[False, False])
    )

    display_cols = [
        "ticker",
        "pattern_name",
        "group_count_setting",
        "days_in_group",
        "review_priority",
        "review_reason",
        "tightness_score",
        "model_uncertainty",
        "avg_ensemble_score",
        "avg_shape_corr",
        "median_abs_return",
        "representative_dates",
    ]

    lines = [
        "# Pattern Review Queue",
        "",
        "This queue is for human review. It does not append feedback automatically.",
        "",
        "## Files",
        "",
        "```text",
        str(QUEUE_FILE),
        str(REPORT_FILE),
        "```",
        "",
        "## Summary",
        "",
        "```text",
        summary.head(40).to_string(index=False),
        "```",
        "",
        "## Top Review Candidates",
        "",
        "```text",
        top[display_cols].to_string(index=False),
        "```",
    ]
    REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")


def main():
    ARTIFACT_DIR.mkdir(exist_ok=True)
    groups, members, scores = load_inputs()
    queue = build_queue(groups, members, scores)
    queue.to_csv(QUEUE_FILE, index=False)
    write_report(queue)
    print(REPORT_FILE.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
