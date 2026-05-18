from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score


ARTIFACT_DIR = Path("model_artifacts")
ENSEMBLE_SCORE_FILE = ARTIFACT_DIR / "structural_day_ensemble_scores.csv"
MODEL_FILE = ARTIFACT_DIR / "early_directional_predictor.joblib"
PREDICTION_FILE = ARTIFACT_DIR / "early_directional_predictions.csv"
REPORT_FILE = ARTIFACT_DIR / "early_directional_predictor_report.txt"


def shape_columns(df):
    return [column for column in df.columns if column.startswith("shape_")]


def early_shape_columns(df, fraction=0.35):
    columns = shape_columns(df)
    keep_count = max(8, int(round(len(columns) * fraction)))
    return columns[:keep_count]


def add_early_shape_features(df, early_cols):
    df = df.copy()
    early = df[early_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    # These features only use the early sampled path, so they are suitable for
    # a live bar-20/bar-25 style model without seeing the full day.
    df["early_path_last"] = early.iloc[:, -1]
    df["early_path_min"] = early.min(axis=1)
    df["early_path_max"] = early.max(axis=1)
    df["early_path_range"] = df["early_path_max"] - df["early_path_min"]
    df["early_path_abs_move"] = df["early_path_last"].abs()
    df["early_path_realized_vol"] = early.diff(axis=1).abs().mean(axis=1).fillna(0)
    df["early_path_reversal"] = df["early_path_last"] - early.iloc[:, : max(2, len(early_cols) // 2)].mean(axis=1)
    return df


def add_targets(df):
    df = df.copy()
    threshold = df["useful_pattern_ensemble_score"].quantile(0.90)
    high_structure = df["useful_pattern_ensemble_score"] >= threshold
    df["target_high_structure_up"] = (high_structure & (df["avg_return"] > 0)).astype(int)
    df["target_high_structure_down"] = (high_structure & (df["avg_return"] < 0)).astype(int)
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


def train_binary_model(train, test, feature_cols, target_col, random_state):
    x_train = train[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_train = train[target_col]
    x_test = test[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_test = test[target_col]

    model = RandomForestClassifier(
        n_estimators=600,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.50).astype(int)

    auc = roc_auc_score(y_test, probabilities) if y_test.nunique() == 2 else np.nan
    average_precision = average_precision_score(y_test, probabilities)
    report = classification_report(y_test, predictions, zero_division=0)
    return model, probabilities, predictions, auc, average_precision, report


def summarize_ranked_slice(df, score_col, target_col, label):
    top_1 = df[df[score_col] >= df[score_col].quantile(0.99)]
    top_5 = df[df[score_col] >= df[score_col].quantile(0.95)]
    bottom_50 = df[df[score_col] <= df[score_col].quantile(0.50)]

    return [
        f"{label} score-ranked behavior",
        f"top_1pct_target_rate={top_1[target_col].mean():.4f}",
        f"top_5pct_target_rate={top_5[target_col].mean():.4f}",
        f"bottom_50pct_target_rate={bottom_50[target_col].mean():.4f}",
        f"top_1pct_avg_return={top_1['avg_return'].mean():.4f}",
        f"top_5pct_avg_return={top_5['avg_return'].mean():.4f}",
        f"bottom_50pct_avg_return={bottom_50['avg_return'].mean():.4f}",
        "",
    ]


def main():
    if not ENSEMBLE_SCORE_FILE.exists():
        raise SystemExit("Run build_ensemble_scores.py first.")

    df = pd.read_csv(ENSEMBLE_SCORE_FILE)
    early_cols = early_shape_columns(df)
    df = add_early_shape_features(df, early_cols)
    df, threshold = add_targets(df)

    engineered_cols = [
        "avg_early_move",
        "early_path_last",
        "early_path_min",
        "early_path_max",
        "early_path_range",
        "early_path_abs_move",
        "early_path_realized_vol",
        "early_path_reversal",
    ]
    feature_cols = engineered_cols + early_cols

    train, test = walk_forward_split(df)
    if train.empty or test.empty:
        raise SystemExit("Not enough ticker history for walk-forward test.")

    up_model, up_prob, up_pred, up_auc, up_ap, up_report = train_binary_model(
        train, test, feature_cols, "target_high_structure_up", random_state=121
    )
    down_model, down_prob, down_pred, down_auc, down_ap, down_report = train_binary_model(
        train, test, feature_cols, "target_high_structure_down", random_state=122
    )

    test = test.copy()
    test["early_up_probability"] = up_prob
    test["early_down_probability"] = down_prob
    test["early_direction_edge"] = test["early_up_probability"] - test["early_down_probability"]
    test["early_direction_signal"] = np.select(
        [
            (test["early_up_probability"] >= 0.50) & (test["early_direction_edge"] > 0),
            (test["early_down_probability"] >= 0.50) & (test["early_direction_edge"] < 0),
        ],
        ["up", "down"],
        default="none",
    )
    test["early_up_prediction"] = up_pred
    test["early_down_prediction"] = down_pred
    test = test.sort_values(
        ["early_up_probability", "early_down_probability"],
        ascending=[False, False],
    )
    test.to_csv(PREDICTION_FILE, index=False)

    joblib.dump(
        {
            "up_model": up_model,
            "down_model": down_model,
            "feature_columns": feature_cols,
            "target_threshold_top_decile_ensemble_score": threshold,
        },
        MODEL_FILE,
    )

    up_ranked = test.sort_values("early_up_probability", ascending=False)
    down_ranked = test.sort_values("early_down_probability", ascending=False)
    strongest_edges = test.reindex(test["early_direction_edge"].abs().sort_values(ascending=False).index)

    lines = [
        "Early Directional Predictor",
        f"target_threshold_top_decile_ensemble_score={threshold:.6f}",
        f"train_rows={len(train)} test_rows={len(test)}",
        f"up_test_positives={int(test['target_high_structure_up'].sum())}",
        f"down_test_positives={int(test['target_high_structure_down'].sum())}",
        "",
        "Feature note: this model avoids full-day summary leakage. It uses avg_early_move, early path features, and the first 35% of shape columns.",
        "",
        f"Up model roc_auc={up_auc:.3f} average_precision={up_ap:.3f}",
        up_report,
        f"Down model roc_auc={down_auc:.3f} average_precision={down_ap:.3f}",
        down_report,
        *summarize_ranked_slice(up_ranked, "early_up_probability", "target_high_structure_up", "Up"),
        *summarize_ranked_slice(down_ranked, "early_down_probability", "target_high_structure_down", "Down"),
        "Top early-up predictions:",
        up_ranked[
            [
                "ticker",
                "date",
                "early_up_probability",
                "early_down_probability",
                "target_high_structure_up",
                "target_high_structure_down",
                "useful_pattern_ensemble_score",
                "avg_return",
                "avg_early_move",
            ]
        ]
        .head(20)
        .to_string(index=False),
        "",
        "Top early-down predictions:",
        down_ranked[
            [
                "ticker",
                "date",
                "early_up_probability",
                "early_down_probability",
                "target_high_structure_up",
                "target_high_structure_down",
                "useful_pattern_ensemble_score",
                "avg_return",
                "avg_early_move",
            ]
        ]
        .head(20)
        .to_string(index=False),
        "",
        "Strongest directional edges:",
        strongest_edges[
            [
                "ticker",
                "date",
                "early_direction_signal",
                "early_direction_edge",
                "early_up_probability",
                "early_down_probability",
                "useful_pattern_ensemble_score",
                "avg_return",
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
