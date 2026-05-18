from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from early_trade_simulator import asset_class
from entry_bar_timing_sweep import build_entry_features
from local_market_data import load_downloaded_data


ARTIFACT_DIR = Path("model_artifacts")
ENSEMBLE_SCORE_FILE = ARTIFACT_DIR / "structural_day_ensemble_scores.csv"
FEATURE_FILE = ARTIFACT_DIR / "raw_early_structure_features.csv"
PREDICTION_FILE = ARTIFACT_DIR / "raw_early_structure_predictions.csv"
SUMMARY_FILE = ARTIFACT_DIR / "raw_early_structure_summary.csv"
MODEL_FILE = ARTIFACT_DIR / "raw_early_structure_models.joblib"
REPORT_FILE = ARTIFACT_DIR / "raw_early_structure_predictor_report.md"

SESSION_START = "09:30"
SESSION_END = "16:00"
INTERVAL = "5m"
CHECKPOINT_BARS = [8, 12, 20, 28]


BASE_FEATURES = [
    "session_open_to_entry_pct",
    "entry_abs_open_to_entry_pct",
    "entry_early_range_pct",
    "entry_close_position",
    "entry_distance_from_high_pct",
    "entry_distance_from_low_pct",
    "entry_last5_return_pct",
    "entry_abs_last5_return_pct",
    "entry_last5_range_pct",
    "entry_last5_body_pct",
    "entry_abs_last5_body_pct",
    "entry_volume_late_early_ratio",
    "entry_last5_volume_share",
    "asset_is_stock",
    "asset_is_etf",
    "asset_is_futures",
    "asset_is_crypto",
    "asset_is_forex",
]


def add_targets(df):
    df = df.copy()
    threshold = df["useful_pattern_ensemble_score"].quantile(0.90)
    df["target_high_structure"] = (df["useful_pattern_ensemble_score"] >= threshold).astype(int)
    return df, threshold


def asset_flags(kind):
    return {
        "asset_is_stock": int(kind == "stock"),
        "asset_is_etf": int(kind == "etf"),
        "asset_is_futures": int(kind == "futures"),
        "asset_is_crypto": int(kind == "crypto"),
        "asset_is_forex": int(kind == "forex"),
    }


def build_rows(scored):
    rows = []
    for ticker, ticker_scored in scored.groupby("ticker"):
        raw = load_downloaded_data(ticker, INTERVAL)
        if raw.empty:
            continue

        session = raw.between_time(SESSION_START, SESSION_END)
        if session.empty:
            continue

        days = {str(day_key): day.copy() for day_key, day in session.groupby(session.index.date)}
        kind = asset_class(ticker)
        flags = asset_flags(kind)

        for _, scored_day in ticker_scored.iterrows():
            day = days.get(str(pd.to_datetime(scored_day["date"]).date()))
            if day is None:
                continue

            for entry_bar in CHECKPOINT_BARS:
                features = build_entry_features(day, entry_bar)
                if features is None:
                    continue

                open_to_entry = features["session_open_to_entry_pct"]
                last5_return = features["entry_last5_return_pct"]
                last5_body = features["entry_last5_body_pct"]
                rows.append(
                    {
                        "ticker": ticker,
                        "date": scored_day["date"],
                        "asset_class": kind,
                        "entry_bar": entry_bar,
                        "target_high_structure": scored_day["target_high_structure"],
                        "useful_pattern_ensemble_score": scored_day["useful_pattern_ensemble_score"],
                        "avg_return": scored_day["avg_return"],
                        "avg_abs_return": abs(scored_day["avg_return"]),
                        "bars": scored_day.get("bars", np.nan),
                        "entry_abs_open_to_entry_pct": abs(open_to_entry),
                        "entry_abs_last5_return_pct": abs(last5_return),
                        "entry_abs_last5_body_pct": abs(last5_body),
                        **flags,
                        **features,
                    }
                )

    return pd.DataFrame(rows)


def walk_forward_split(df, train_fraction=0.80):
    train_parts = []
    test_parts = []
    for _, ticker_df in df.sort_values("date").groupby("ticker"):
        if len(ticker_df) < 20:
            continue
        split = max(1, int(round(len(ticker_df) * train_fraction)))
        train_parts.append(ticker_df.iloc[:split])
        test_parts.append(ticker_df.iloc[split:])

    if not train_parts or not test_parts:
        return pd.DataFrame(), pd.DataFrame()
    return pd.concat(train_parts, ignore_index=True), pd.concat(test_parts, ignore_index=True)


def clean_matrix(df, columns):
    return df[columns].replace([np.inf, -np.inf], np.nan).fillna(0)


def ranked_slices(df, score_col):
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
                "target_rate": part["target_high_structure"].mean(),
                "avg_ensemble_score": part["useful_pattern_ensemble_score"].mean(),
                "avg_abs_return": part["avg_abs_return"].mean(),
            }
        )
    return pd.DataFrame(rows)


def train_one_checkpoint(train, test, entry_bar):
    train_bar = train[train["entry_bar"] == entry_bar].copy()
    test_bar = test[test["entry_bar"] == entry_bar].copy()

    x_train = clean_matrix(train_bar, BASE_FEATURES)
    y_train = train_bar["target_high_structure"]
    x_test = clean_matrix(test_bar, BASE_FEATURES)
    y_test = test_bar["target_high_structure"]

    model = RandomForestClassifier(
        n_estimators=350,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=1200 + entry_bar,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_test)[:, 1]

    test_bar["raw_early_structure_probability"] = probabilities
    auc = roc_auc_score(y_test, probabilities) if y_test.nunique() == 2 else np.nan
    ap = average_precision_score(y_test, probabilities)
    ranked = ranked_slices(test_bar, "raw_early_structure_probability")
    top1 = ranked[ranked["slice"] == "top_1pct"].iloc[0]
    top5 = ranked[ranked["slice"] == "top_5pct"].iloc[0]
    top10 = ranked[ranked["slice"] == "top_10pct"].iloc[0]
    bottom = ranked[ranked["slice"] == "bottom_50pct"].iloc[0]

    summary = {
        "entry_bar": entry_bar,
        "train_rows": len(train_bar),
        "test_rows": len(test_bar),
        "test_positive_rate": y_test.mean(),
        "roc_auc": auc,
        "average_precision": ap,
        "top1_target_rate": top1["target_rate"],
        "top1_avg_abs_return": top1["avg_abs_return"],
        "top5_target_rate": top5["target_rate"],
        "top5_avg_abs_return": top5["avg_abs_return"],
        "top10_target_rate": top10["target_rate"],
        "top10_avg_abs_return": top10["avg_abs_return"],
        "bottom50_target_rate": bottom["target_rate"],
        "bottom50_avg_abs_return": bottom["avg_abs_return"],
    }
    return model, test_bar, summary


def report_lines(features, train, test, threshold, summary, predictions):
    best = summary.sort_values(["top5_target_rate", "top5_avg_abs_return"], ascending=[False, False]).iloc[0]
    best_bar = int(best["entry_bar"])
    best_predictions = predictions[predictions["entry_bar"] == best_bar].sort_values(
        "raw_early_structure_probability", ascending=False
    )

    display_cols = [
        "ticker",
        "date",
        "asset_class",
        "entry_bar",
        "raw_early_structure_probability",
        "target_high_structure",
        "useful_pattern_ensemble_score",
        "avg_abs_return",
        "session_open_to_entry_pct",
        "entry_early_range_pct",
        "entry_close_position",
        "entry_last5_return_pct",
        "entry_volume_late_early_ratio",
    ]

    return [
        "# Raw Early Structure Predictor",
        "",
        "This trains live-safe early-structure models from raw OHLCV prefixes. The target is the same top-decile ensemble structural label used by the earlier normalized-shape predictor.",
        "",
        "## Split",
        "",
        "```text",
        f"feature_rows={len(features)}",
        f"train_rows={len(train)}",
        f"test_rows={len(test)}",
        f"target_threshold_top_decile_ensemble_score={threshold:.6f}",
        f"checkpoint_bars={CHECKPOINT_BARS}",
        "```",
        "",
        "## Checkpoint Summary",
        "",
        "```text",
        summary.to_string(index=False),
        "```",
        "",
        f"## Best Checkpoint: Bar {best_bar}",
        "",
        "```text",
        best_predictions[display_cols].head(40).to_string(index=False),
        "```",
    ]


def main():
    if not ENSEMBLE_SCORE_FILE.exists():
        raise SystemExit("Run build_ensemble_scores.py first.")

    scored = pd.read_csv(ENSEMBLE_SCORE_FILE)
    scored, threshold = add_targets(scored)

    if FEATURE_FILE.exists():
        features = pd.read_csv(FEATURE_FILE)
    else:
        features = build_rows(scored)
        features.to_csv(FEATURE_FILE, index=False)

    if features.empty:
        raise SystemExit("No raw early-structure feature rows could be built.")

    train, test = walk_forward_split(features)
    if train.empty or test.empty:
        raise SystemExit("Not enough ticker history for raw early-structure walk-forward test.")

    models = {}
    prediction_parts = []
    summaries = []
    for entry_bar in CHECKPOINT_BARS:
        model, test_bar, summary = train_one_checkpoint(train, test, entry_bar)
        models[entry_bar] = model
        prediction_parts.append(test_bar)
        summaries.append(summary)

    predictions = pd.concat(prediction_parts, ignore_index=True).sort_values(
        ["entry_bar", "raw_early_structure_probability"], ascending=[True, False]
    )
    summary = pd.DataFrame(summaries).sort_values(
        ["top5_target_rate", "top5_avg_abs_return"], ascending=[False, False]
    )

    predictions.to_csv(PREDICTION_FILE, index=False)
    summary.to_csv(SUMMARY_FILE, index=False)
    joblib.dump(
        {
            "models": models,
            "feature_columns": BASE_FEATURES,
            "checkpoint_bars": CHECKPOINT_BARS,
            "target_threshold": threshold,
        },
        MODEL_FILE,
    )

    REPORT_FILE.write_text(
        "\n".join(report_lines(features, train, test, threshold, summary, predictions)),
        encoding="utf-8",
    )
    print(REPORT_FILE.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
