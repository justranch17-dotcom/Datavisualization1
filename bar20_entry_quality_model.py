from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score

from early_trade_simulator import load_predictions
from local_market_data import load_downloaded_data


ARTIFACT_DIR = Path("model_artifacts")
FEATURE_FILE = ARTIFACT_DIR / "bar20_entry_quality_features.csv"
PREDICTION_FILE = ARTIFACT_DIR / "bar20_entry_quality_predictions.csv"
MODEL_FILE = ARTIFACT_DIR / "bar20_entry_quality_model.joblib"
REPORT_FILE = ARTIFACT_DIR / "bar20_entry_quality_model_report.txt"

SESSION_START = "09:30"
SESSION_END = "16:00"
ENTRY_BAR = 20
INTERVAL = "5m"


def signed_return(side, entry_price, exit_price):
    if entry_price == 0:
        return np.nan
    value = (exit_price - entry_price) / entry_price * 100.0
    return value if side == "long" else -value


def implied_side(row):
    return "long" if row["early_direction_edge"] >= 0 else "short"


def raw_bar20_features(day):
    if len(day) <= ENTRY_BAR:
        return None

    entry = day.iloc[ENTRY_BAR]
    future = day.iloc[ENTRY_BAR + 1 :]
    early = day.iloc[: ENTRY_BAR + 1]
    last5 = day.iloc[max(0, ENTRY_BAR - 4) : ENTRY_BAR + 1]

    entry_price = float(entry["Close"])
    session_open = float(day["Open"].iloc[0])
    early_high = float(early["High"].max())
    early_low = float(early["Low"].min())
    early_range = early_high - early_low
    early_range_pct = early_range / entry_price * 100.0 if entry_price else np.nan
    close_position = (entry_price - early_low) / early_range if early_range else 0.5

    features = {
        "entry_time": entry.name,
        "entry_price": entry_price,
        "session_open_to_entry_pct": (entry_price - session_open) / session_open * 100.0 if session_open else np.nan,
        "bar20_early_range_pct": early_range_pct,
        "bar20_close_position": close_position,
        "bar20_distance_from_high_pct": (entry_price - early_high) / entry_price * 100.0 if entry_price else np.nan,
        "bar20_distance_from_low_pct": (entry_price - early_low) / entry_price * 100.0 if entry_price else np.nan,
        "bar20_last5_return_pct": (float(last5["Close"].iloc[-1]) - float(last5["Close"].iloc[0])) / float(last5["Close"].iloc[0]) * 100.0
        if float(last5["Close"].iloc[0])
        else np.nan,
        "bar20_last5_range_pct": (float(last5["High"].max()) - float(last5["Low"].min())) / entry_price * 100.0
        if entry_price
        else np.nan,
        "bar20_last5_body_pct": (float(last5["Close"].iloc[-1]) - float(last5["Open"].iloc[0])) / entry_price * 100.0
        if entry_price
        else np.nan,
        "bar20_future_close_long_return_pct": (float(day["Close"].iloc[-1]) - entry_price) / entry_price * 100.0
        if entry_price
        else np.nan,
        "bar20_future_high_long_mfe_pct": (float(future["High"].max()) - entry_price) / entry_price * 100.0
        if entry_price and not future.empty
        else np.nan,
        "bar20_future_low_long_mae_pct": (float(future["Low"].min()) - entry_price) / entry_price * 100.0
        if entry_price and not future.empty
        else np.nan,
    }

    if "Volume" in early.columns and early["Volume"].notna().any():
        first_half_volume = float(early["Volume"].iloc[: max(1, len(early) // 2)].sum())
        second_half_volume = float(early["Volume"].iloc[max(1, len(early) // 2) :].sum())
        features["bar20_volume_late_early_ratio"] = second_half_volume / max(1.0, first_half_volume)
        features["bar20_last5_volume_share"] = float(last5["Volume"].sum()) / max(1.0, float(early["Volume"].sum()))
    else:
        features["bar20_volume_late_early_ratio"] = 0.0
        features["bar20_last5_volume_share"] = 0.0

    return features


def build_raw_feature_rows(predictions):
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

            features = raw_bar20_features(day)
            if features is None:
                continue

            side = implied_side(prediction)
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
                    "implied_side": side,
                    "signed_bar20_close_return_pct": signed_close_return,
                    "target_bar20_followthrough": int(signed_close_return > 0),
                    "target_bar20_strong_followthrough": int(signed_close_return >= 0.50),
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


def feature_columns(df):
    columns = [
        "early_pattern_probability",
        "early_up_probability",
        "early_down_probability",
        "early_direction_edge",
        "useful_pattern_ensemble_score",
        "session_open_to_entry_pct",
        "bar20_early_range_pct",
        "bar20_close_position",
        "bar20_distance_from_high_pct",
        "bar20_distance_from_low_pct",
        "bar20_last5_return_pct",
        "bar20_last5_range_pct",
        "bar20_last5_body_pct",
        "bar20_volume_late_early_ratio",
        "bar20_last5_volume_share",
    ]
    return [column for column in columns if column in df.columns]


def train_entry_model(train, test, columns, target):
    x_train = train[columns].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_train = train[target]
    x_test = test[columns].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_test = test[target]

    model = RandomForestClassifier(
        n_estimators=600,
        min_samples_leaf=8,
        class_weight="balanced",
        random_state=219,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.50).astype(int)

    auc = roc_auc_score(y_test, probabilities) if y_test.nunique() == 2 else np.nan
    average_precision = average_precision_score(y_test, probabilities)
    report = classification_report(y_test, predictions, zero_division=0)
    return model, probabilities, predictions, auc, average_precision, report


def top_slice_summary(df, score_col):
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
                "avg_signed_return": part["signed_bar20_close_return_pct"].mean(),
                "hit_rate": part["target_bar20_followthrough"].mean(),
                "strong_hit_rate": part["target_bar20_strong_followthrough"].mean(),
            }
        )
    return pd.DataFrame(rows)


def main():
    predictions = load_predictions()
    features = build_raw_feature_rows(predictions)
    if features.empty:
        raise SystemExit("No bar-20 raw feature rows could be built.")

    features.to_csv(FEATURE_FILE, index=False)
    train, test, split_date = chronological_split(features)
    columns = feature_columns(features)
    model, probabilities, pred, auc, ap, report = train_entry_model(
        train,
        test,
        columns,
        "target_bar20_followthrough",
    )

    test = test.copy()
    test["bar20_entry_quality_probability"] = probabilities
    test["bar20_entry_quality_prediction"] = pred
    test = test.sort_values("bar20_entry_quality_probability", ascending=False)
    test.to_csv(PREDICTION_FILE, index=False)
    joblib.dump({"model": model, "feature_columns": columns}, MODEL_FILE)

    ranked = top_slice_summary(test, "bar20_entry_quality_probability")
    top_rows = test[
        [
            "ticker",
            "date",
            "asset_class",
            "implied_side",
            "bar20_entry_quality_probability",
            "signed_bar20_close_return_pct",
            "early_pattern_probability",
            "early_direction_edge",
            "useful_pattern_ensemble_score",
            "bar20_close_position",
            "bar20_last5_return_pct",
            "bar20_early_range_pct",
        ]
    ].head(30)

    lines = [
        "Bar 20 Entry Quality Model",
        f"feature_rows={len(features)} train_rows={len(train)} test_rows={len(test)} split_date={split_date}",
        f"roc_auc={auc:.3f} average_precision={ap:.3f}",
        "",
        report,
        "Ranked test behavior",
        ranked.to_string(index=False),
        "",
        "Top entry-quality predictions:",
        top_rows.to_string(index=False),
        "",
        "Feature columns:",
        "\n".join(columns),
    ]

    REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(REPORT_FILE.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
