"""
Phone-facing scanner adapter.

This module intentionally does not import the big Streamlit apps, because importing
them would execute their UI. Instead, it reads safe project artifacts that already
exist, like mach2_group_feedback.csv and downloaded_historical_data.
"""

import pandas as pd

from project_paths import PROJECT_ROOT

MACH2_FEEDBACK_FILE = PROJECT_ROOT / "mach2_group_feedback.csv"
DATA_DIR = PROJECT_ROOT / "downloaded_historical_data"


def _recent_mach2_feedback_patterns(limit=20):
    if not MACH2_FEEDBACK_FILE.exists():
        return []

    try:
        feedback = pd.read_csv(MACH2_FEEDBACK_FILE).tail(limit)
    except (OSError, pd.errors.ParserError):
        return []

    patterns = []
    for index, row in feedback.iloc[::-1].iterrows():
        ticker = str(row.get("ticker", "")).upper()
        group_name = str(row.get("group_name", "Mach 2 structural group"))
        rating = str(row.get("rating", "Unrated"))
        note = str(row.get("note", "") or "").strip()
        timestamp = str(row.get("timestamp", ""))
        signature = str(row.get("signature", f"row-{index}"))

        summary = f"Previously marked as {rating} in Mach 2."
        if note and note.lower() != "nan":
            summary += f" Note: {note}"

        patterns.append(
            {
                "pattern_id": f"mach2-feedback-{index}-{abs(hash(signature))}",
                "ticker": ticker or "UNKNOWN",
                "pattern_date": timestamp[:10] if timestamp else "Unknown",
                "setup_name": group_name,
                "summary": summary,
                "stats": {
                    "source": "mach2_group_feedback.csv",
                    "signature": signature,
                    "session": row.get("session", ""),
                    "original_rating": rating,
                },
            }
        )

    return patterns


def _project_status_pattern():
    csv_count = len(list(DATA_DIR.glob("*.csv"))) if DATA_DIR.exists() else 0
    feedback_count = 0

    if MACH2_FEEDBACK_FILE.exists():
        try:
            feedback_count = len(pd.read_csv(MACH2_FEEDBACK_FILE))
        except (OSError, pd.errors.ParserError):
            feedback_count = 0

    return {
        "pattern_id": "project-status",
        "ticker": "PROJECT",
        "pattern_date": pd.Timestamp.now().date().isoformat(),
        "setup_name": "Project Status",
        "summary": (
            f"Found {csv_count} top-level downloaded market CSV files and "
            f"{feedback_count} saved Mach 2 feedback rows."
        ),
        "stats": {
            "project_root": str(PROJECT_ROOT),
            "data_dir": str(DATA_DIR),
            "mach2_feedback_file": str(MACH2_FEEDBACK_FILE),
        },
    }


def get_patterns():
    """
    Return phone-safe pattern cards.

    Later we can wire this to a saved scan-results file from Mach2AImarket.py. For
    now it reads existing feedback and project status without touching the main app.
    """
    patterns = _recent_mach2_feedback_patterns()
    if patterns:
        return patterns

    return [_project_status_pattern()]
