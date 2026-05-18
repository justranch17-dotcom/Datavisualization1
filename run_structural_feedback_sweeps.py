import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd


ARTIFACT_DIR = Path("model_artifacts")
SWEEP_SUMMARY_FILE = ARTIFACT_DIR / "structural_feedback_sweep_summary.csv"


SWEEPS = [
    {
        "name": "pattern_core_g14_16_18",
        "group_counts": [14, 16, 18],
        "min_group_days": 3,
        "min_corr": 0.88,
        "max_rmse": 1.10,
        "max_band": 1.20,
        "min_same_direction": 0.68,
        "min_median_abs_return": 0.30,
    },
    {
        "name": "pattern_tight_g16_18_20",
        "group_counts": [16, 18, 20],
        "min_group_days": 3,
        "min_corr": 0.90,
        "max_rmse": 1.00,
        "max_band": 1.10,
        "min_same_direction": 0.72,
        "min_median_abs_return": 0.35,
    },
    {
        "name": "pattern_dense_g12_16_20",
        "group_counts": [12, 16, 20],
        "min_group_days": 3,
        "min_corr": 0.87,
        "max_rmse": 1.15,
        "max_band": 1.25,
        "min_same_direction": 0.66,
        "min_median_abs_return": 0.30,
    },
]


def run_command(command):
    print(" ".join(command), flush=True)
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    print(completed.stdout)
    if completed.stderr:
        print(completed.stderr)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def summarize_run(run_name):
    backtest_path = ARTIFACT_DIR / f"automated_feedback_backtest_{run_name}.csv"
    feedback_path = ARTIFACT_DIR / f"automated_structural_feedback_{run_name}.csv"

    if not backtest_path.exists() or not feedback_path.exists():
        return {}

    backtest = pd.read_csv(backtest_path)
    feedback = pd.read_csv(feedback_path)
    good = backtest[backtest["rating"] == "Good pattern"]
    bad = backtest[backtest["rating"] == "Not useful"]

    return {
        "run_name": run_name,
        "feedback_rows": len(feedback),
        "good_feedback_rows": int((feedback["rating"] == "Good pattern").sum()),
        "bad_feedback_rows": int((feedback["rating"] == "Not useful").sum()),
        "good_groups": len(good),
        "bad_groups": len(bad),
        "good_avg_cohesion": good["cohesion_score"].mean() if not good.empty else None,
        "bad_avg_cohesion": bad["cohesion_score"].mean() if not bad.empty else None,
        "good_test_avg_abs_return": good["test_avg_return"].abs().mean() if not good.empty else None,
        "bad_test_avg_abs_return": bad["test_avg_return"].abs().mean() if not bad.empty else None,
        "good_avg_corr": good["avg_shape_corr"].mean() if not good.empty else None,
        "bad_avg_corr": bad["avg_shape_corr"].mean() if not bad.empty else None,
    }


def main():
    ARTIFACT_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summaries = []

    for sweep in SWEEPS:
        run_name = f"{timestamp}_{sweep['name']}"
        command = [
            ".\\.venv\\Scripts\\python.exe",
            ".\\automated_structural_feedback.py",
            "--coverage",
            "0.80",
            "--group-counts",
            *[str(value) for value in sweep["group_counts"]],
            "--min-group-days",
            str(sweep["min_group_days"]),
            "--min-corr",
            str(sweep["min_corr"]),
            "--max-rmse",
            str(sweep["max_rmse"]),
            "--max-band",
            str(sweep["max_band"]),
            "--min-same-direction",
            str(sweep["min_same_direction"]),
            "--min-median-abs-return",
            str(sweep["min_median_abs_return"]),
            "--run-name",
            run_name,
            "--append-feedback",
        ]
        run_command(command)
        summaries.append({**sweep, **summarize_run(run_name)})

    summary = pd.DataFrame(summaries)
    summary.to_csv(SWEEP_SUMMARY_FILE, index=False)
    print(f"Saved {SWEEP_SUMMARY_FILE}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
