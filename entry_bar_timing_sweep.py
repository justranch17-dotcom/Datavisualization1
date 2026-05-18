from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from early_trade_simulator import load_predictions
from local_market_data import load_downloaded_data


ARTIFACT_DIR = Path("model_artifacts")
FEATURE_FILE = ARTIFACT_DIR / "entry_bar_timing_features.csv"
SUMMARY_FILE = ARTIFACT_DIR / "entry_bar_timing_sweep_summary.csv"
PREDICTION_FILE = ARTIFACT_DIR / "entry_bar_timing_predictions.csv"
MODEL_FILE = ARTIFACT_DIR / "entry_bar_timing_models.joblib"
REPORT_FILE = ARTIFACT_DIR / "entry_bar_timing_sweep_report.md"

SESSION_START = "09:30"
SESSION_END = "16:00"
INTERVAL = "5m"
ENTRY_BARS = [8, 12, 15, 18, 20, 22, 25, 28, 32]


def implied_side(row):
    return "long" if row["early_direction_edge"] >= 0 else "short"


def signed_return(side, entry_price, exit_price):
    if entry_price == 0:
        return np.nan
    raw_return = (exit_price - entry_price) / entry_price * 100.0
    return raw_return if side == "long" else -raw_return


def build_entry_features(day, entry_bar):
    if len(day) <= entry_bar + 1:
        return None

    entry = day.iloc[entry_bar]
    future = day.iloc[entry_bar + 1 :]
    early = day.iloc[: entry_bar + 1]
    last5 = day.iloc[max(0, entry_bar - 4) : entry_bar + 1]

    entry_price = float(entry["Close"])
    session_open = float(day["Open"].iloc[0])
    early_high = float(early["High"].max())
    early_low = float(early["Low"].min())
    early_range = early_high - early_low
    close_position = (entry_price - early_low) / early_range if early_range else 0.5

    row = {
        "entry_time": entry.name,
        "entry_price": entry_price,
        "session_open_to_entry_pct": (entry_price - session_open) / session_open * 100.0 if session_open else np.nan,
        "entry_early_range_pct": early_range / entry_price * 100.0 if entry_price else np.nan,
        "entry_close_position": close_position,
        "entry_distance_from_high_pct": (entry_price - early_high) / entry_price * 100.0 if entry_price else np.nan,
        "entry_distance_from_low_pct": (entry_price - early_low) / entry_price * 100.0 if entry_price else np.nan,
        "entry_last5_return_pct": (float(last5["Close"].iloc[-1]) - float(last5["Close"].iloc[0])) / float(last5["Close"].iloc[0]) * 100.0
        if float(last5["Close"].iloc[0])
        else np.nan,
        "entry_last5_range_pct": (float(last5["High"].max()) - float(last5["Low"].min())) / entry_price * 100.0
        if entry_price
        else np.nan,
        "entry_last5_body_pct": (float(last5["Close"].iloc[-1]) - float(last5["Open"].iloc[0])) / entry_price * 100.0
        if entry_price
        else np.nan,
        "future_close_long_return_pct": (float(day["Close"].iloc[-1]) - entry_price) / entry_price * 100.0
        if entry_price
        else np.nan,
        "future_high_long_mfe_pct": (float(future["High"].max()) - entry_price) / entry_price * 100.0
        if entry_price and not future.empty
        else np.nan,
        "future_low_long_mae_pct": (float(future["Low"].min()) - entry_price) / entry_price * 100.0
        if entry_price and not future.empty
        else np.nan,
    }

    if "Volume" in early.columns and early["Volume"].notna().any():
        first_half_volume = float(early["Volume"].iloc[: max(1, len(early) // 2)].sum())
        second_half_volume = float(early["Volume"].iloc[max(1, len(early) // 2) :].sum())
        row["entry_volume_late_early_ratio"] = second_half_volume / max(1.0, first_half_volume)
        row["entry_last5_volume_share"] = float(last5["Volume"].sum()) / max(1.0, float(early["Volume"].sum()))
    else:
        row["entry_volume_late_early_ratio"] = 0.0
        row["entry_last5_volume_share"] = 0.0

    return row


def build_feature_rows(predictions):
    rows = []
    for ticker, ticker_predictions in predictions.groupby("ticker"):
        df = load_downloaded_data(ticker, INTERVAL)
        if df.empty:
            continue

        session = df.between_time(SESSION_START, SESSION_END)
        if session.empty:
            continue

        day_map = {str(day_key): day.copy() for day_key, day in session.groupby(session.index.date)}
        for _, prediction in ticker_predictions.iterrows():
            day = day_map.get(str(pd.to_datetime(prediction["date"]).date()))
            if day is None:
                continue

            side = implied_side(prediction)
            for entry_bar in ENTRY_BARS:
                features = build_entry_features(day, entry_bar)
                if features is None:
                    continue

                signed_close_return = signed_return(
                    side,
                    features["entry_price"],
                    float(day["Close"].iloc[-1]),
                )
                rows.append(
                    {
                        "ticker": ticker,
                        "date": prediction["date"],
                        "asset_class": prediction.get("asset_class", ""),
                        "entry_bar": entry_bar,
                        "implied_side": side,
                        "signed_entry_to_close_return_pct": signed_close_return,
                        "target_followthrough": int(signed_close_return > 0),
                        "target_strong_followthrough": int(signed_close_return >= 0.50),
                        "early_pattern_probability": prediction["early_pattern_probability"],
                        "early_up_probability": prediction["early_up_probability"],
                        "early_down_probability": prediction["early_down_probability"],
                        "early_direction_edge": prediction["early_direction_edge"],
                        "useful_pattern_ensemble_score": prediction["useful_pattern_ensemble_score"],
                        **features,
                    }
                )

    return pd.DataFrame(rows)


def chronological_split(df, train_fraction=0.60):
    dates = sorted(pd.to_datetime(df["date"]).dt.date.unique())
    split_index = max(1, int(round(len(dates) * train_fraction)))
    split_date = dates[split_index - 1]
    train = df[pd.to_datetime(df["date"]).dt.date <= split_date].copy()
    test = df[pd.to_datetime(df["date"]).dt.date > split_date].copy()
    return train, test, split_date


def feature_columns():
    return [
        "early_pattern_probability",
        "early_up_probability",
        "early_down_probability",
        "early_direction_edge",
        "useful_pattern_ensemble_score",
        "session_open_to_entry_pct",
        "entry_early_range_pct",
        "entry_close_position",
        "entry_distance_from_high_pct",
        "entry_distance_from_low_pct",
        "entry_last5_return_pct",
        "entry_last5_range_pct",
        "entry_last5_body_pct",
        "entry_volume_late_early_ratio",
        "entry_last5_volume_share",
    ]


def train_one_bar(train, test, entry_bar):
    train_bar = train[train["entry_bar"] == entry_bar].copy()
    test_bar = test[test["entry_bar"] == entry_bar].copy()
    columns = feature_columns()

    x_train = train_bar[columns].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_train = train_bar["target_followthrough"]
    x_test = test_bar[columns].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_test = test_bar["target_followthrough"]

    model = RandomForestClassifier(
        n_estimators=500,
        min_samples_leaf=8,
        class_weight="balanced",
        random_state=500 + entry_bar,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_test)[:, 1]
    test_bar["entry_quality_probability"] = probabilities

    auc = roc_auc_score(y_test, probabilities) if y_test.nunique() == 2 else np.nan
    ap = average_precision_score(y_test, probabilities)
    ranked = ranked_slices(test_bar, "entry_quality_probability")
    top5 = ranked[ranked["slice"] == "top_5pct"].iloc[0].to_dict()
    top10 = ranked[ranked["slice"] == "top_10pct"].iloc[0].to_dict()
    bottom = ranked[ranked["slice"] == "bottom_50pct"].iloc[0].to_dict()

    summary = {
        "entry_bar": entry_bar,
        "train_rows": len(train_bar),
        "test_rows": len(test_bar),
        "roc_auc": auc,
        "average_precision": ap,
        "top5_rows": top5["rows"],
        "top5_avg_signed_return": top5["avg_signed_return"],
        "top5_hit_rate": top5["hit_rate"],
        "top5_strong_hit_rate": top5["strong_hit_rate"],
        "top10_rows": top10["rows"],
        "top10_avg_signed_return": top10["avg_signed_return"],
        "top10_hit_rate": top10["hit_rate"],
        "bottom50_avg_signed_return": bottom["avg_signed_return"],
        "bottom50_hit_rate": bottom["hit_rate"],
    }
    return model, test_bar, summary


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
                "avg_signed_return": part["signed_entry_to_close_return_pct"].mean(),
                "hit_rate": part["target_followthrough"].mean(),
                "strong_hit_rate": part["target_strong_followthrough"].mean(),
            }
        )
    return pd.DataFrame(rows)


def main():
    predictions = load_predictions()
    features = build_feature_rows(predictions)
    if features.empty:
        raise SystemExit("No entry timing features could be built.")

    features.to_csv(FEATURE_FILE, index=False)
    train, test, split_date = chronological_split(features)

    summaries = []
    prediction_parts = []
    models = {}
    for entry_bar in ENTRY_BARS:
        model, test_bar, summary = train_one_bar(train, test, entry_bar)
        summaries.append(summary)
        prediction_parts.append(test_bar)
        models[entry_bar] = model

    summary_df = pd.DataFrame(summaries).sort_values(
        ["top5_avg_signed_return", "top5_hit_rate"], ascending=[False, False]
    )
    prediction_df = pd.concat(prediction_parts, ignore_index=True).sort_values(
        ["entry_bar", "entry_quality_probability"], ascending=[True, False]
    )

    summary_df.to_csv(SUMMARY_FILE, index=False)
    prediction_df.to_csv(PREDICTION_FILE, index=False)
    joblib.dump({"models": models, "feature_columns": feature_columns(), "entry_bars": ENTRY_BARS}, MODEL_FILE)

    best_bar = int(summary_df.iloc[0]["entry_bar"])
    best_predictions = prediction_df[prediction_df["entry_bar"] == best_bar].sort_values(
        "entry_quality_probability", ascending=False
    )

    display_cols = [
        "ticker",
        "date",
        "asset_class",
        "implied_side",
        "entry_bar",
        "entry_quality_probability",
        "signed_entry_to_close_return_pct",
        "early_pattern_probability",
        "early_direction_edge",
        "useful_pattern_ensemble_score",
        "entry_close_position",
        "entry_last5_return_pct",
        "entry_early_range_pct",
    ]

    lines = [
        "# Entry Bar Timing Sweep",
        "",
        "This trains the same raw-entry-quality model separately for several 5m entry bars. It tests whether bar 20 is genuinely the best checkpoint or just the first one we tried.",
        "",
        "## Split",
        "",
        "```text",
        f"feature_rows={len(features)}",
        f"train_rows={len(train)}",
        f"test_rows={len(test)}",
        f"split_date={split_date}",
        f"entry_bars={ENTRY_BARS}",
        "```",
        "",
        "## Timing Summary",
        "",
        "```text",
        summary_df.to_string(index=False),
        "```",
        "",
        f"## Best Entry Bar: {best_bar}",
        "",
        "```text",
        best_predictions[display_cols].head(30).to_string(index=False),
        "```",
    ]

    REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(REPORT_FILE.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
