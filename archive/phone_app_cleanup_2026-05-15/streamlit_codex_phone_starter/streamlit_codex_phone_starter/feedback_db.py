import sqlite3
from datetime import datetime

from project_paths import PROJECT_ROOT

DB_PATH = PROJECT_ROOT / "pattern_feedback.db"


def init_db():
    """Create the feedback database if it does not exist yet."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS pattern_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pattern_id TEXT,
            ticker TEXT,
            pattern_date TEXT,
            setup_name TEXT,
            rating TEXT,
            notes TEXT,
            created_at TEXT
        )
    """)

    conn.commit()
    conn.close()


def save_feedback(pattern_id, ticker, pattern_date, setup_name, rating, notes):
    """Save one rating/note from the phone app."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO pattern_feedback
        (pattern_id, ticker, pattern_date, setup_name, rating, notes, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        pattern_id,
        ticker,
        pattern_date,
        setup_name,
        rating,
        notes,
        datetime.now().isoformat(timespec="seconds")
    ))

    conn.commit()
    conn.close()


def get_recent_feedback(limit=50):
    """Return the most recent feedback rows."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        SELECT pattern_id, ticker, pattern_date, setup_name, rating, notes, created_at
        FROM pattern_feedback
        ORDER BY id DESC
        LIMIT ?
    """, (limit,))

    rows = cur.fetchall()
    conn.close()

    return rows


def get_feedback_summary(limit=100):
    """Create a plain-English summary that Codex can read."""
    rows = get_recent_feedback(limit=limit)

    if not rows:
        return "No pattern feedback has been saved yet."

    lines = ["# Recent Pattern Feedback", ""]

    for row in rows:
        pattern_id, ticker, pattern_date, setup_name, rating, notes, created_at = row
        lines.append(f"## {rating}: {setup_name}")
        lines.append(f"- Pattern ID: {pattern_id}")
        lines.append(f"- Ticker: {ticker}")
        lines.append(f"- Date: {pattern_date}")
        lines.append(f"- Saved: {created_at}")
        lines.append(f"- Notes: {notes if notes else 'No notes added.'}")
        lines.append("")

    return "\n".join(lines)
