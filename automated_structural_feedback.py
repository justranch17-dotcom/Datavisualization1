import argparse
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from structural_pattern_learner import (
    FEATURE_FILE,
    FEEDBACK_FILE,
    MODEL_FEATURES,
    MODEL_FILE,
    SCORE_FILE,
    TRAINING_FILE,
    load_feedback_training_rows,
    score_universe,
)


ARTIFACT_DIR = Path("model_artifacts")
AUTO_FEEDBACK_FILE = ARTIFACT_DIR / "automated_structural_feedback.csv"
BACKTEST_FILE = ARTIFACT_DIR / "automated_feedback_backtest.csv"
COMPARISON_FILE = ARTIFACT_DIR / "learner_improvement_report.txt"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate structural feedback from cohesion/backtest rules and retrain the learner."
    )
    parser.add_argument("--coverage", type=float, default=0.80, help="Fraction of symbols to process. Default: 0.80")
    parser.add_argument(
        "--group-counts",
        nargs="*",
        type=int,
        default=[12, 14, 16, 18, 20],
        help="KMeans group counts. Defaults now cluster more tightly around 16 groups.",
    )
    parser.add_argument("--min-group-days", type=int, default=3, help="Minimum days in a group to judge it.")
    parser.add_argument("--test-fraction", type=float, default=0.20, help="Latest fraction of each symbol used for backtest.")
    parser.add_argument("--min-corr", type=float, default=0.86, help="Minimum average correlation to group signature.")
    parser.add_argument("--max-rmse", type=float, default=1.20, help="Maximum RMSE from group signature.")
    parser.add_argument("--max-band", type=float, default=1.35, help="Maximum average interquartile band width.")
    parser.add_argument("--min-same-direction", type=float, default=0.62, help="Minimum same-direction rate.")
    parser.add_argument("--min-median-abs-return", type=float, default=0.25, help="Minimum median absolute return.")
    parser.add_argument("--run-name", default="", help="Optional run label written into notes and artifact names.")
    parser.add_argument("--append-feedback", action="store_true", help="Append generated feedback to mach2_group_feedback.csv.")
    return parser.parse_args()


def shape_columns(features):
    return [column for column in features.columns if column.startswith("shape_")]


def choose_tickers(features, coverage):
    tickers = sorted(features["ticker"].dropna().astype(str).unique())
    target = max(1, int(round(len(tickers) * coverage)))
    priority = ["NVDA", "ETH_USD", "BTC_USD", "QQQ", "SPY", "NQ_F", "ES_F", "EURUSD_X", "GBPUSD_X", "USDJPY_X"]
    chosen = []

    for ticker in priority:
        if ticker in tickers and ticker not in chosen:
            chosen.append(ticker)

    for ticker in tickers:
        if ticker not in chosen:
            chosen.append(ticker)
        if len(chosen) >= target:
            break

    return chosen


def group_name(group_id, avg_return, avg_early_move):
    if avg_return > 0.05 and avg_early_move < 0:
        direction = "reversal up"
    elif avg_return < -0.05 and avg_early_move > 0:
        direction = "reversal down"
    elif avg_return > 0.05:
        direction = "momentum up"
    elif avg_return < -0.05:
        direction = "momentum down"
    else:
        direction = "momentum chop"
    return f"Auto Structure Group {group_id}: {direction}"


def feedback_signature(ticker, group):
    return "|".join(
        [
            str(ticker).upper(),
            "5m",
            "09:30",
            "16:00",
            str(round(group["avg_return"], 2)),
            str(round(group["avg_range"], 2)),
            str(round(group["avg_early_move"], 2)),
            str(round(group["avg_shift_strength"], 2)),
            str(round(group["avg_shift_count"], 1)),
        ]
    )


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
    return rmse, band_width, avg_corr


def judge_group(group, shape_cols, thresholds):
    rmse, band_width, avg_corr = group_metrics(group, shape_cols)
    avg_return = float(group["avg_return"].mean())
    avg_range = float(group["avg_range"].mean())
    avg_early_move = float(group["avg_early_move"].mean())
    avg_shift_strength = float(group["avg_shift_strength"].mean())
    avg_shift_count = float(group["avg_shift_count"].mean())
    same_direction_rate = float((np.sign(group["avg_return"]) == np.sign(avg_return)).mean())
    median_abs_return = float(group["avg_return"].abs().median())

    # This rule is intentionally aligned with the user's stated preference:
    # tight days around the signature, similar structure, and a coherent group.
    cohesion_score = (
        avg_corr * 0.45
        + max(0.0, 1.0 - rmse / 1.35) * 0.30
        + max(0.0, 1.0 - band_width / 1.50) * 0.20
        + same_direction_rate * 0.05
    )

    useful = (
        len(group) >= 5
        and avg_corr >= thresholds["min_corr"]
        and rmse <= thresholds["max_rmse"]
        and band_width <= thresholds["max_band"]
        and same_direction_rate >= thresholds["min_same_direction"]
        and median_abs_return >= thresholds["min_median_abs_return"]
    )

    if avg_corr < thresholds["min_corr"] - 0.08 or rmse > thresholds["max_rmse"] + 0.45 or band_width > thresholds["max_band"] + 0.45:
        useful = False

    return {
        "rating": "Good pattern" if useful else "Not useful",
        "cohesion_score": cohesion_score,
        "signature_rmse": rmse,
        "signature_band_width": band_width,
        "avg_shape_corr": avg_corr,
        "same_direction_rate": same_direction_rate,
        "median_abs_return": median_abs_return,
        "avg_return": avg_return,
        "avg_range": avg_range,
        "avg_early_move": avg_early_move,
        "avg_shift_strength": avg_shift_strength,
        "avg_shift_count": avg_shift_count,
    }


def generate_feedback(features, tickers, group_counts, min_group_days, test_fraction, thresholds, run_name):
    shape_cols = shape_columns(features)
    feedback_rows = []
    backtest_rows = []
    now = datetime.now().isoformat(timespec="seconds")

    for ticker in tickers:
        ticker_rows = features[features["ticker"] == ticker].sort_values("date").reset_index(drop=True)
        if len(ticker_rows) < min_group_days * 2:
            continue

        split_index = max(min_group_days, int(round(len(ticker_rows) * (1 - test_fraction))))
        train_rows = ticker_rows.iloc[:split_index].copy()
        test_rows = ticker_rows.iloc[split_index:].copy()

        for group_count in group_counts:
            if len(train_rows) < group_count * min_group_days:
                continue

            scaled_train = StandardScaler().fit_transform(train_rows[shape_cols])
            labels = KMeans(n_clusters=group_count, random_state=42, n_init=10).fit_predict(scaled_train)
            train_rows["auto_group"] = labels

            scaler = StandardScaler().fit(train_rows[shape_cols])
            model = KMeans(n_clusters=group_count, random_state=42, n_init=10).fit(scaler.transform(train_rows[shape_cols]))

            if not test_rows.empty:
                test_rows = test_rows.copy()
                test_rows["auto_group"] = model.predict(scaler.transform(test_rows[shape_cols]))

            for label in sorted(train_rows["auto_group"].unique()):
                group = train_rows[train_rows["auto_group"] == label]
                if len(group) < min_group_days:
                    continue

                metrics = judge_group(group, shape_cols, thresholds)
                name = group_name(label + 1, metrics["avg_return"], metrics["avg_early_move"])
                group_payload = {key: metrics[key] for key in MODEL_FEATURES}
                signature = feedback_signature(ticker, group_payload)
                note = (
                    f"auto feedback {run_name}: judged from tightness to signature line, "
                    f"avg corr {metrics['avg_shape_corr']:.3f}, rmse {metrics['signature_rmse']:.3f}, "
                    f"band {metrics['signature_band_width']:.3f}, same direction {metrics['same_direction_rate']:.2f}"
                )

                feedback_rows.append(
                    {
                        "timestamp": now,
                        "signature": signature,
                        "ticker": ticker,
                        "interval": "5m",
                        "session": "09:30-16:00",
                        "group_name": name,
                        "rating": metrics["rating"],
                        "note": note,
                        "group_count_setting": group_count,
                        "run_name": run_name,
                        "days_in_group": len(group),
                        **metrics,
                    }
                )

                test_group = test_rows[test_rows["auto_group"] == label] if not test_rows.empty else pd.DataFrame()
                backtest_rows.append(
                    {
                        "ticker": ticker,
                        "group_count_setting": group_count,
                        "run_name": run_name,
                        "group_name": name,
                        "rating": metrics["rating"],
                        "train_days": len(group),
                        "test_days": len(test_group),
                        "train_avg_return": metrics["avg_return"],
                        "test_avg_return": float(test_group["avg_return"].mean()) if not test_group.empty else np.nan,
                        "test_median_abs_return": float(test_group["avg_return"].abs().median()) if not test_group.empty else np.nan,
                        "cohesion_score": metrics["cohesion_score"],
                        "avg_shape_corr": metrics["avg_shape_corr"],
                        "signature_rmse": metrics["signature_rmse"],
                        "signature_band_width": metrics["signature_band_width"],
                    }
                )

    return pd.DataFrame(feedback_rows), pd.DataFrame(backtest_rows)


def append_feedback(auto_feedback):
    base_columns = ["timestamp", "signature", "ticker", "interval", "session", "group_name", "rating", "note"]
    new_rows = auto_feedback[base_columns].copy()

    if FEEDBACK_FILE.exists():
        existing = pd.read_csv(FEEDBACK_FILE)
        combined = pd.concat([existing, new_rows], ignore_index=True)
        combined = combined.drop_duplicates(subset=["signature", "ticker", "group_name", "rating", "note"], keep="first")
    else:
        combined = new_rows

    combined.to_csv(FEEDBACK_FILE, index=False)
    return len(combined)


def evaluate_model(training, label):
    if training.empty or training["label"].nunique() < 2 or len(training) < 12:
        return f"{label}: not enough mixed labels to evaluate.\n", None

    x = training[MODEL_FEATURES]
    y = training["label"]
    stratify = y if y.value_counts().min() >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.30,
        random_state=7,
        stratify=stratify,
    )

    model = RandomForestClassifier(
        n_estimators=400,
        min_samples_leaf=2,
        random_state=7,
        class_weight="balanced",
    )
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    auc = roc_auc_score(y_test, probabilities) if y_test.nunique() == 2 else np.nan
    ap = average_precision_score(y_test, probabilities)
    report = classification_report(y_test, predictions, zero_division=0)

    text = (
        f"{label}\n"
        f"rows={len(training)} positives={int(y.sum())} negatives={int((y == 0).sum())}\n"
        f"roc_auc={auc:.3f} average_precision={ap:.3f}\n"
        f"{report}\n"
    )
    return text, model


def main():
    args = parse_args()
    ARTIFACT_DIR.mkdir(exist_ok=True)

    if not FEATURE_FILE.exists():
        raise SystemExit(f"Missing {FEATURE_FILE}. Run structural_pattern_learner.py first.")

    features = pd.read_csv(FEATURE_FILE)
    selected_tickers = choose_tickers(features, args.coverage)
    print(f"Processing {len(selected_tickers)} of {features['ticker'].nunique()} symbols.")

    before_training = load_feedback_training_rows()
    before_report, _ = evaluate_model(before_training, "Before automated feedback")

    run_name = args.run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    thresholds = {
        "min_corr": args.min_corr,
        "max_rmse": args.max_rmse,
        "max_band": args.max_band,
        "min_same_direction": args.min_same_direction,
        "min_median_abs_return": args.min_median_abs_return,
    }

    auto_feedback, backtest = generate_feedback(
        features=features,
        tickers=selected_tickers,
        group_counts=args.group_counts,
        min_group_days=args.min_group_days,
        test_fraction=args.test_fraction,
        thresholds=thresholds,
        run_name=run_name,
    )

    run_feedback_file = ARTIFACT_DIR / f"automated_structural_feedback_{run_name}.csv"
    run_backtest_file = ARTIFACT_DIR / f"automated_feedback_backtest_{run_name}.csv"
    auto_feedback.to_csv(AUTO_FEEDBACK_FILE, index=False)
    auto_feedback.to_csv(run_feedback_file, index=False)
    backtest.to_csv(BACKTEST_FILE, index=False)
    backtest.to_csv(run_backtest_file, index=False)
    print(f"Generated {len(auto_feedback)} automated feedback rows.")
    print(f"Saved {AUTO_FEEDBACK_FILE}")
    print(f"Saved {run_feedback_file}")
    print(f"Saved {BACKTEST_FILE}")
    print(f"Saved {run_backtest_file}")

    if args.append_feedback:
        total_rows = append_feedback(auto_feedback)
        print(f"Appended audited automated feedback. mach2_group_feedback.csv now has {total_rows} rows.")

    after_training = load_feedback_training_rows()
    after_report, model = evaluate_model(after_training, "After automated feedback")
    after_training.to_csv(TRAINING_FILE, index=False)

    if model is not None:
        joblib.dump(model, MODEL_FILE)
        scored = score_universe(model, features)
        scored.to_csv(SCORE_FILE, index=False)

    good = backtest[backtest["rating"] == "Good pattern"]
    bad = backtest[backtest["rating"] == "Not useful"]
    summary = [
        before_report,
        after_report,
        "Backtest summary",
        f"good_groups={len(good)} bad_groups={len(bad)}",
        f"good_test_avg_abs_return={good['test_avg_return'].abs().mean():.4f}" if not good.empty else "good_test_avg_abs_return=N/A",
        f"bad_test_avg_abs_return={bad['test_avg_return'].abs().mean():.4f}" if not bad.empty else "bad_test_avg_abs_return=N/A",
        f"good_avg_cohesion={good['cohesion_score'].mean():.4f}" if not good.empty else "good_avg_cohesion=N/A",
        f"bad_avg_cohesion={bad['cohesion_score'].mean():.4f}" if not bad.empty else "bad_avg_cohesion=N/A",
        "",
    ]
    COMPARISON_FILE.write_text("\n".join(summary), encoding="utf-8")
    print(COMPARISON_FILE.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
