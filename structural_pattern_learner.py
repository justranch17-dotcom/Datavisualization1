import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

from local_market_data import list_downloaded_tickers, load_downloaded_data


ARTIFACT_DIR = Path("model_artifacts")
FEEDBACK_FILE = Path("mach2_group_feedback.csv")
FEATURE_FILE = ARTIFACT_DIR / "structural_day_features.csv"
TRAINING_FILE = ARTIFACT_DIR / "feedback_training_rows.csv"
SCORE_FILE = ARTIFACT_DIR / "structural_day_scores.csv"
MODEL_FILE = ARTIFACT_DIR / "structural_feedback_model.joblib"
RICH_MODEL_FILE = ARTIFACT_DIR / "structural_feedback_rich_model.joblib"
RICH_SCORE_FILE = ARTIFACT_DIR / "structural_day_rich_scores.csv"
EVALUATION_FILE = ARTIFACT_DIR / "structural_model_evaluation.txt"

MODEL_FEATURES = [
    "avg_return",
    "avg_range",
    "avg_early_move",
    "avg_shift_strength",
    "avg_shift_count",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build and train a feedback-driven structural pattern learner."
    )
    parser.add_argument("--interval", default="5m", help="Downloaded interval to use. Default: 5m")
    parser.add_argument("--session-start", default="09:30", help="Session start time. Default: 09:30")
    parser.add_argument("--session-end", default="16:00", help="Session end time. Default: 16:00")
    parser.add_argument("--resample-points", type=int, default=80, help="Shape points per day. Default: 80")
    parser.add_argument("--min-bars", type=int, default=40, help="Minimum bars required per day. Default: 40")
    parser.add_argument("--max-tickers", type=int, default=0, help="Optional ticker limit for quick tests.")
    parser.add_argument(
        "--model-mode",
        default="summary",
        choices=["summary", "rich", "both"],
        help="Train summary features, rich shape features, or both. Default: summary",
    )
    return parser.parse_args()


def shape_feature_columns(df):
    return [column for column in df.columns if column.startswith("shape_")]


def normalized_close(day, points):
    close = day["Close"].astype(float)
    if close.empty or close.iloc[0] == 0:
        return None

    values = (close / close.iloc[0] - 1.0) * 100.0
    old_x = np.linspace(0, 1, len(values))
    new_x = np.linspace(0, 1, points)
    return np.interp(new_x, old_x, values.to_numpy())


def momentum_shift_features(shape):
    diffs = np.diff(shape)
    if len(diffs) < 3:
        return 0.0, 0.0

    signs = np.sign(diffs)
    changes = np.where(np.diff(signs) != 0)[0]
    strengths = [abs(diffs[i]) + abs(diffs[i + 1]) for i in changes if i + 1 < len(diffs)]
    return float(max(strengths) if strengths else 0.0), float(len(changes))


def day_feature_row(ticker, day_key, day, points, min_bars):
    if len(day) < min_bars:
        return None

    required = {"Open", "High", "Low", "Close"}
    if not required.issubset(day.columns):
        return None

    shape = normalized_close(day, points)
    if shape is None:
        return None

    first_close = float(day["Close"].iloc[0])
    if first_close == 0:
        return None

    early_end = max(1, len(day) // 3)
    avg_return = (float(day["Close"].iloc[-1]) - first_close) / first_close * 100.0
    avg_range = (float(day["High"].max()) - float(day["Low"].min())) / first_close * 100.0
    avg_early_move = (float(day["Close"].iloc[early_end - 1]) - first_close) / first_close * 100.0
    avg_shift_strength, avg_shift_count = momentum_shift_features(shape)

    row = {
        "ticker": ticker,
        "date": str(day_key),
        "bars": len(day),
        "avg_return": avg_return,
        "avg_range": avg_range,
        "avg_early_move": avg_early_move,
        "avg_shift_strength": avg_shift_strength,
        "avg_shift_count": avg_shift_count,
    }

    # Store the shape too, so later versions can train richer models without
    # recomputing every CSV.
    for idx, value in enumerate(shape):
        row[f"shape_{idx:03d}"] = value

    return row


def build_feature_universe(interval, session_start, session_end, points, min_bars, max_tickers=0):
    tickers = [ticker for ticker in list_downloaded_tickers(interval) if ticker != "download_quality_report"]
    if max_tickers:
        tickers = tickers[:max_tickers]

    rows = []
    for ticker in tickers:
        df = load_downloaded_data(ticker, interval)
        if df.empty:
            continue

        session = df.between_time(session_start, session_end)
        if session.empty:
            continue

        for day_key, day in session.groupby(session.index.date):
            row = day_feature_row(ticker, day_key, day, points, min_bars)
            if row is not None:
                rows.append(row)

    return pd.DataFrame(rows)


def load_feedback_training_rows():
    if not FEEDBACK_FILE.exists():
        return pd.DataFrame()

    feedback = pd.read_csv(FEEDBACK_FILE)
    rows = []

    for _, row in feedback.iterrows():
        signature = str(row.get("signature", ""))
        parts = signature.split("|")
        if len(parts) < 9:
            continue

        rating = str(row.get("rating", "")).strip().lower()
        if rating == "good pattern":
            label = 1
        elif rating == "not useful":
            label = 0
        else:
            continue

        try:
            features = [float(value) for value in parts[4:9]]
        except ValueError:
            continue

        rows.append(
            {
                "ticker": str(row.get("ticker", parts[0])).upper(),
                "interval": parts[1],
                "session_start": parts[2],
                "session_end": parts[3],
                "group_name": row.get("group_name", ""),
                "rating": row.get("rating", ""),
                "label": label,
                **dict(zip(MODEL_FEATURES, features)),
            }
        )

    return pd.DataFrame(rows)


def build_rich_training_rows(features, feedback):
    """
    Expand group-level feedback into day-level rich shape examples.

    Feedback signatures describe a structural group average. We attach each
    feedback row to local days from the same ticker that are close to the
    feedback summary. This lets the model learn full normalized shapes while
    still respecting the user's group-level ratings.
    """
    if features.empty or feedback.empty:
        return pd.DataFrame()

    rows = []
    shape_cols = shape_feature_columns(features)
    feature_cols = MODEL_FEATURES + shape_cols

    for _, feedback_row in feedback.iterrows():
        ticker = str(feedback_row.get("ticker", "")).upper()
        ticker_features = features[features["ticker"].astype(str).str.upper() == ticker]

        if ticker_features.empty:
            continue

        target = feedback_row[MODEL_FEATURES].astype(float).to_numpy()
        values = ticker_features[MODEL_FEATURES].astype(float).to_numpy()
        scale = np.nanstd(values, axis=0)
        scale = np.where(scale == 0, 1.0, scale)
        distances = np.linalg.norm((values - target) / scale, axis=1)

        nearest_count = min(12, max(4, len(ticker_features) // 20))
        nearest_positions = np.argsort(distances)[:nearest_count]

        for position in nearest_positions:
            source = ticker_features.iloc[position]
            row = {
                "ticker": source["ticker"],
                "date": source["date"],
                "label": int(feedback_row["label"]),
                "source_group_name": feedback_row.get("group_name", ""),
                "source_rating": feedback_row.get("rating", ""),
                "summary_distance": float(distances[position]),
            }
            for column in feature_cols:
                row[column] = source[column]
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    rich = pd.DataFrame(rows)
    rich = rich.sort_values("summary_distance").drop_duplicates(
        subset=["ticker", "date", "label"],
        keep="first",
    )
    return rich


def split_train_test(x, y, random_state=42):
    stratify = y if y.value_counts().min() >= 2 else None
    return train_test_split(
        x,
        y,
        test_size=0.30,
        random_state=random_state,
        stratify=stratify,
    )


def evaluate_predictions(y_test, probabilities, predictions):
    auc = roc_auc_score(y_test, probabilities) if y_test.nunique() == 2 else np.nan
    average_precision = average_precision_score(y_test, probabilities)
    report = classification_report(y_test, predictions, zero_division=0)
    return auc, average_precision, report


def train_feedback_model(training, feature_columns=None, random_state=42):
    if training.empty or training["label"].nunique() < 2 or len(training) < 6:
        return None, None, None

    if feature_columns is None:
        feature_columns = MODEL_FEATURES

    x = training[feature_columns]
    y = training["label"]

    x_train, x_test, y_train, y_test = split_train_test(x, y, random_state=random_state)

    model = RandomForestClassifier(
        n_estimators=450,
        min_samples_leaf=2,
        random_state=random_state,
        class_weight="balanced",
        n_jobs=-1,
    )
    model.fit(x_train, y_train)

    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    auc, average_precision, report = evaluate_predictions(y_test, probabilities, predictions)
    metrics = {
        "rows": len(training),
        "positives": int(y.sum()),
        "negatives": int((y == 0).sum()),
        "roc_auc": auc,
        "average_precision": average_precision,
        "report": report,
    }
    return model, report, metrics


def score_universe(model, features, feature_columns=None, score_column="useful_pattern_score"):
    scored = features.copy()
    if model is None or scored.empty:
        scored[score_column] = np.nan
        return scored

    if feature_columns is None:
        feature_columns = MODEL_FEATURES

    scored[score_column] = model.predict_proba(scored[feature_columns])[:, 1]
    return scored.sort_values(score_column, ascending=False)


def format_metrics(name, metrics):
    if not metrics:
        return f"{name}: not enough data.\n"

    return (
        f"{name}\n"
        f"rows={metrics['rows']} positives={metrics['positives']} negatives={metrics['negatives']}\n"
        f"roc_auc={metrics['roc_auc']:.3f} average_precision={metrics['average_precision']:.3f}\n"
        f"{metrics['report']}\n"
    )


def main():
    args = parse_args()
    ARTIFACT_DIR.mkdir(exist_ok=True)

    print("Building structural day feature universe...")
    features = build_feature_universe(
        interval=args.interval,
        session_start=args.session_start,
        session_end=args.session_end,
        points=args.resample_points,
        min_bars=args.min_bars,
        max_tickers=args.max_tickers,
    )
    features.to_csv(FEATURE_FILE, index=False)
    print(f"Saved {len(features)} day feature rows to {FEATURE_FILE}")

    training = load_feedback_training_rows()
    training.to_csv(TRAINING_FILE, index=False)
    print(f"Saved {len(training)} feedback training rows to {TRAINING_FILE}")

    evaluation_sections = []

    model = None
    if args.model_mode in ["summary", "both"]:
        model, report, metrics = train_feedback_model(training, MODEL_FEATURES, random_state=42)
        evaluation_sections.append(format_metrics("Summary model", metrics))
        if model is None:
            print("Not enough mixed Good pattern / Not useful feedback yet to train a model.")
            print("Keep rating structural groups in Mach2AImarket.py, then run this again.")
        else:
            joblib.dump(model, MODEL_FILE)
            print(f"Saved trained model to {MODEL_FILE}")
            print("Summary validation report:")
            print(report)

        scored = score_universe(model, features, MODEL_FEATURES, "useful_pattern_score")
        scored.to_csv(SCORE_FILE, index=False)
        print(f"Saved scored structural days to {SCORE_FILE}")

        if "useful_pattern_score" in scored.columns and scored["useful_pattern_score"].notna().any():
            print("Top summary scored days:")
            print(scored[["ticker", "date", "useful_pattern_score", *MODEL_FEATURES]].head(20).to_string(index=False))

    if args.model_mode in ["rich", "both"]:
        rich_training = build_rich_training_rows(features, training)
        rich_training_file = ARTIFACT_DIR / "feedback_rich_training_rows.csv"
        rich_training.to_csv(rich_training_file, index=False)
        print(f"Saved {len(rich_training)} rich feedback training rows to {rich_training_file}")

        rich_columns = MODEL_FEATURES + shape_feature_columns(features)
        rich_model, rich_report, rich_metrics = train_feedback_model(
            rich_training,
            rich_columns,
            random_state=43,
        )
        evaluation_sections.append(format_metrics("Rich shape model", rich_metrics))

        if rich_model is not None:
            joblib.dump(rich_model, RICH_MODEL_FILE)
            print(f"Saved rich trained model to {RICH_MODEL_FILE}")
            print("Rich validation report:")
            print(rich_report)

        rich_scored = score_universe(
            rich_model,
            features,
            rich_columns,
            "useful_pattern_rich_score",
        )
        rich_scored.to_csv(RICH_SCORE_FILE, index=False)
        print(f"Saved rich scored structural days to {RICH_SCORE_FILE}")

        if "useful_pattern_rich_score" in rich_scored.columns and rich_scored["useful_pattern_rich_score"].notna().any():
            print("Top rich scored days:")
            print(rich_scored[["ticker", "date", "useful_pattern_rich_score", *MODEL_FEATURES]].head(20).to_string(index=False))

    EVALUATION_FILE.write_text("\n".join(evaluation_sections), encoding="utf-8")
    print(f"Saved model evaluation to {EVALUATION_FILE}")


if __name__ == "__main__":
    main()
