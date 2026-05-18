from datetime import datetime

from project_paths import PROJECT_ROOT

TASK_FILE = PROJECT_ROOT / "codex_tasks.md"
SUMMARY_FILE = PROJECT_ROOT / "codex_feedback_summary.md"


def add_codex_task(task_text):
    """Append a new task to codex_tasks.md in the project root."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    entry = f"""

---

## Codex Task - {timestamp}

{task_text}

"""

    with TASK_FILE.open("a", encoding="utf-8") as f:
        f.write(entry)


def build_task_from_feedback(pattern_id, rating, notes):
    """Turn one pattern rating into a clean Codex instruction."""
    return f"""
The user reviewed a scanner pattern from the phone Streamlit app.

Pattern ID: {pattern_id}
User rating: {rating}
User notes: {notes}

Please review the scanner logic and suggest a code improvement based on this feedback.

Rules:
- Do not remove existing scanner features.
- Keep the Streamlit UI working.
- Prefer adding a clear filter, scoring adjustment, or explanation.
- If the feedback is not enough to make a code change, summarize what more data is needed.
"""


def save_feedback_summary(summary_text):
    """Save a current feedback summary for Codex to read."""
    with SUMMARY_FILE.open("w", encoding="utf-8") as f:
        f.write(summary_text)
