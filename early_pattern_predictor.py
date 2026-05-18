from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score


ARTIFACT_DIR = Path("model_artifacts")
ENSEMBLE_SCORE_FILE = ARTIFACT_DIR / "structural_day_ensemble_scores.csv"
MODEL_FILE = ARTIFACT_DIR / "early_pattern_predictor.joblib"
PREDICTION_FILE = ARTIFACT_DIR / "early_pattern_predictions.csv"
REPORT_FILE = ARTIFACT_DIR / "early_pattern_predictor_report.txt"


SUMMARY_COLUMNS = [
    "avg_range",
    "avg_early_move",
    "avg_shift_strength",
    "avg_shift_count",
]


def early_shape_columns(df, fraction=0.35):
    shape_cols = [column for column in df.columns if column.startswith("shape_")]
    keep_count = max(8, int(round(len(shape_cols) * fraction)))
    return shape_cols[:keep_count]


def add_targets(df):
    df = df.copy()
    threshold = df["useful_pattern_ensemble_score"].quantile(0.90)
    df["target_high_structure"] = (df["useful_pattern_ensemble_score"] >= threshold).astype(int)
    return df, threshold


def walk_forward_split(df):
    train_parts = []
    test_parts = []

    for _, ticker_df in df.sort_values("date").groupby("ticker"):
        if len(ticker_df) < 20:
            continue
        split = int(round(len(ticker_df) * 0.80))
        train_parts.append(ticker_df.iloc[:split])
        test_parts.append(ticker_df.iloc[split:])

    if not train_parts or not test_parts:
        return pd.DataFrame(), pd.DataFrame()

    return pd.concat(train_parts, ignore_index=True), pd.concat(test_parts, ignore_index=True)


def main():
    if not ENSEMBLE_SCORE_FILE.exists():
        raise SystemExit("Run build_ensemble_scores.py first.")

    df = pd.read_csv(ENSEMBLE_SCORE_FILE)
    df, threshold = add_targets(df)
    early_cols = early_shape_columns(df)
    feature_cols = SUMMARY_COLUMNS + early_cols

    train, test = walk_forward_split(df)
    if train.empty or test.empty:
        raise SystemExit("Not enough ticker history for walk-forward test.")

    x_train = train[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_train = train["target_high_structure"]
    x_test = test[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_test = test["target_high_structure"]

    model = RandomForestClassifier(
        n_estimators=500,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=91,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)

    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.50).astype(int)

    test = test.copy()
    test["early_pattern_probability"] = probabilities
    test["early_pattern_prediction"] = predictions
    test = test.sort_values("early_pattern_probability", ascending=False)
    test.to_csv(PREDICTION_FILE, index=False)
    joblib.dump(model, MODEL_FILE)

    auc = roc_auc_score(y_test, probabilities) if y_test.nunique() == 2 else np.nan
    ap = average_precision_score(y_test, probabilities)
    report = classification_report(y_test, predictions, zero_division=0)

    top_1 = test[test["early_pattern_probability"] >= test["early_pattern_probability"].quantile(0.99)]
    top_5 = test[test["early_pattern_probability"] >= test["early_pattern_probability"].quantile(0.95)]
    bottom_50 = test[test["early_pattern_probability"] <= test["early_pattern_probability"].quantile(0.50)]

    lines = [
        "Early Pattern Predictor",
        f"target_threshold_top_decile_ensemble_score={threshold:.6f}",
        f"train_rows={len(train)} test_rows={len(test)}",
        f"test_positives={int(y_test.sum())} test_negatives={int((y_test == 0).sum())}",
        f"roc_auc={auc:.3f} average_precision={ap:.3f}",
        "",
        report,
        "Score-ranked test behavior",
        f"top_1pct_avg_abs_return={top_1['avg_return'].abs().mean():.4f}",
        f"top_5pct_avg_abs_return={top_5['avg_return'].abs().mean():.4f}",
        f"bottom_50pct_avg_abs_return={bottom_50['avg_return'].abs().mean():.4f}",
        f"top_1pct_target_rate={top_1['target_high_structure'].mean():.4f}",
        f"top_5pct_target_rate={top_5['target_high_structure'].mean():.4f}",
        f"bottom_50pct_target_rate={bottom_50['target_high_structure'].mean():.4f}",
        "",
        "Top early predictions:",
        test[
            [
                "ticker",
                "date",
                "early_pattern_probability",
                "target_high_structure",
                "useful_pattern_ensemble_score",
                "avg_return",
                "avg_range",
                "avg_early_move",
            ]
        ]
        .head(30)
        .to_string(index=False),
    ]

    REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(REPORT_FILE.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
