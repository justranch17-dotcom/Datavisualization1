import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from structural_pattern_learner import FEATURE_FILE


ARTIFACT_DIR = Path("model_artifacts")
GROUP_FILE = ARTIFACT_DIR / "pattern_grouping_candidates.csv"
GROUP_MEMBER_FILE = ARTIFACT_DIR / "pattern_grouping_members.csv"
REPORT_FILE = ARTIFACT_DIR / "pattern_grouping_lab_report.md"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build tight structural pattern groups for visual inspection."
    )
    parser.add_argument("--group-counts", nargs="*", type=int, default=[12, 14, 16, 18, 20])
    parser.add_argument("--min-group-days", type=int, default=3)
    parser.add_argument("--max-tickers", type=int, default=0)
    parser.add_argument("--top-groups", type=int, default=250)
    return parser.parse_args()


def shape_columns(df):
    return [column for column in df.columns if column.startswith("shape_")]


def group_metrics(group, shape_cols):
    matrix = group[shape_cols].to_numpy(dtype=float)
    signature = matrix.mean(axis=0)
    residuals = matrix - signature
    rmse = float(np.sqrt(np.mean(residuals**2)))
    band_width = float(np.mean(np.percentile(matrix, 75, axis=0) - np.percentile(matrix, 25, axis=0)))

    correlations = []
    for row in matrix:
        if np.std(row) == 0 or np.std(signature) == 0:
            continue
        correlations.append(float(np.corrcoef(row, signature)[0, 1]))

    avg_corr = float(np.mean(correlations)) if correlations else 0.0
    min_corr = float(np.min(correlations)) if correlations else 0.0
    avg_return = float(group["avg_return"].mean())
    same_direction_rate = float((np.sign(group["avg_return"]) == np.sign(avg_return)).mean())
    median_abs_return = float(group["avg_return"].abs().median())

    tightness_score = (
        avg_corr * 0.42
        + min_corr * 0.18
        + max(0.0, 1.0 - rmse / 1.20) * 0.18
        + max(0.0, 1.0 - band_width / 1.20) * 0.14
        + same_direction_rate * 0.08
    )

    return {
        "days_in_group": len(group),
        "tightness_score": tightness_score,
        "avg_shape_corr": avg_corr,
        "min_shape_corr": min_corr,
        "signature_rmse": rmse,
        "signature_band_width": band_width,
        "same_direction_rate": same_direction_rate,
        "median_abs_return": median_abs_return,
        "avg_return": avg_return,
        "avg_range": float(group["avg_range"].mean()),
        "avg_early_move": float(group["avg_early_move"].mean()),
        "avg_shift_strength": float(group["avg_shift_strength"].mean()),
        "avg_shift_count": float(group["avg_shift_count"].mean()),
    }


def direction_name(avg_return, avg_early_move):
    if avg_return > 0.05 and avg_early_move < 0:
        return "reversal up"
    if avg_return < -0.05 and avg_early_move > 0:
        return "reversal down"
    if avg_return > 0.05:
        return "momentum up"
    if avg_return < -0.05:
        return "momentum down"
    return "chop"


def representative_dates(group, shape_cols, count=8):
    matrix = group[shape_cols].to_numpy(dtype=float)
    signature = matrix.mean(axis=0)
    distances = np.sqrt(np.mean((matrix - signature) ** 2, axis=1))
    reps = group.assign(signature_distance=distances).sort_values("signature_distance").head(count)
    return ", ".join(f"{row.date}" for row in reps.itertuples())


def group_member_rows(ticker, group_count, label, group, shape_cols):
    matrix = group[shape_cols].to_numpy(dtype=float)
    signature = matrix.mean(axis=0)
    distances = np.sqrt(np.mean((matrix - signature) ** 2, axis=1))
    member_rows = []

    for distance, row in zip(distances, group.itertuples(index=False)):
        member_rows.append(
            {
                "ticker": ticker,
                "group_count_setting": group_count,
                "pattern_group": int(label),
                "group_key": f"{ticker}|{group_count}|{label}",
                "date": row.date,
                "signature_distance": float(distance),
                "avg_return": float(row.avg_return),
                "avg_range": float(row.avg_range),
                "avg_early_move": float(row.avg_early_move),
                "avg_shift_strength": float(row.avg_shift_strength),
                "avg_shift_count": float(row.avg_shift_count),
            }
        )

    return member_rows


def build_groups(features, group_counts, min_group_days, max_tickers):
    shape_cols = shape_columns(features)
    rows = []
    members = []
    tickers = sorted(features["ticker"].dropna().astype(str).unique())
    if max_tickers:
        tickers = tickers[:max_tickers]

    for ticker in tickers:
        ticker_rows = features[features["ticker"].astype(str) == ticker].sort_values("date").reset_index(drop=True)
        if len(ticker_rows) < min_group_days * 4:
            continue

        scaled = StandardScaler().fit_transform(ticker_rows[shape_cols])
        for group_count in group_counts:
            if len(ticker_rows) < group_count * min_group_days:
                continue

            model = KMeans(n_clusters=group_count, random_state=1600 + group_count, n_init=15)
            labels = model.fit_predict(scaled)
            silhouette = np.nan
            if len(set(labels)) > 1 and len(ticker_rows) > group_count:
                sample_size = min(1000, len(ticker_rows))
                silhouette = float(silhouette_score(scaled, labels, sample_size=sample_size, random_state=group_count))

            labeled = ticker_rows.copy()
            labeled["pattern_group"] = labels

            for label in sorted(labeled["pattern_group"].unique()):
                group = labeled[labeled["pattern_group"] == label].copy()
                if len(group) < min_group_days:
                    continue

                metrics = group_metrics(group, shape_cols)
                group_key = f"{ticker}|{group_count}|{label}"
                rows.append(
                    {
                        "ticker": ticker,
                        "group_count_setting": group_count,
                        "pattern_group": int(label),
                        "group_key": group_key,
                        "pattern_name": f"{ticker} g{group_count}.{label}: {direction_name(metrics['avg_return'], metrics['avg_early_move'])}",
                        "cluster_silhouette": silhouette,
                        "representative_dates": representative_dates(group, shape_cols),
                        **metrics,
                    }
                )
                members.extend(group_member_rows(ticker, group_count, label, group, shape_cols))

    return pd.DataFrame(rows), pd.DataFrame(members)


def summarize(groups):
    if groups.empty:
        return pd.DataFrame()
    return (
        groups.groupby("group_count_setting", as_index=False)
        .agg(
            groups=("pattern_group", "size"),
            avg_tightness=("tightness_score", "mean"),
            avg_corr=("avg_shape_corr", "mean"),
            avg_min_corr=("min_shape_corr", "mean"),
            avg_rmse=("signature_rmse", "mean"),
            avg_band=("signature_band_width", "mean"),
            avg_silhouette=("cluster_silhouette", "mean"),
            high_quality_groups=("tightness_score", lambda x: int((x >= 0.72).sum())),
        )
        .sort_values(["avg_tightness", "high_quality_groups"], ascending=[False, False])
    )


def main():
    args = parse_args()
    ARTIFACT_DIR.mkdir(exist_ok=True)

    if not FEATURE_FILE.exists():
        raise SystemExit("Run structural_pattern_learner.py first.")

    features = pd.read_csv(FEATURE_FILE)
    groups, members = build_groups(features, args.group_counts, args.min_group_days, args.max_tickers)
    if groups.empty:
        raise SystemExit("No pattern groups were generated.")

    groups = groups.sort_values("tightness_score", ascending=False)
    groups.to_csv(GROUP_FILE, index=False)
    members = members.sort_values(["ticker", "group_count_setting", "pattern_group", "signature_distance"])
    members.to_csv(GROUP_MEMBER_FILE, index=False)

    summary = summarize(groups)
    top = groups.head(args.top_groups)
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

    lines = [
        "# Pattern Grouping Lab",
        "",
        "This report focuses on grouping days into visually coherent pattern families. It does not make trade predictions and it does not append training feedback.",
        "",
        "## Settings",
        "",
        "```text",
        f"group_counts={args.group_counts}",
        f"min_group_days={args.min_group_days}",
        f"total_groups={len(groups)}",
        "```",
        "",
        "## Group Count Summary",
        "",
        "```text",
        summary.to_string(index=False),
        "```",
        "",
        "## Output Files",
        "",
        "```text",
        str(GROUP_FILE),
        str(GROUP_MEMBER_FILE),
        "```",
        "",
        "## Top Tight Pattern Groups",
        "",
        "```text",
        top[display_cols].to_string(index=False),
        "```",
    ]

    REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(REPORT_FILE.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
