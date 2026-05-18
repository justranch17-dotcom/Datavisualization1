from pathlib import Path

import pandas as pd


ROOT = Path(".")
ARTIFACT_DIR = Path("model_artifacts")
MANIFEST_FILE = ARTIFACT_DIR / "artifact_manifest.csv"
REPORT_FILE = ARTIFACT_DIR / "artifact_manifest_report.md"


NEVER_PUSH = {
    ".venv": "local Python environment",
    "__pycache__": "Python cache",
    "downloaded_historical_data": "large regenerated raw market data",
}


def category_for(path):
    text = str(path).replace("\\", "/")
    name = path.name.lower()

    if any(part in NEVER_PUSH for part in path.parts):
        return "local_only"
    if name.endswith(".env") or "key" in name or "secret" in name:
        return "secret_or_credential"
    if name.endswith(".joblib"):
        return "model_binary"
    if path.suffix.lower() == ".csv" and "model_artifacts/" in text:
        return "generated_data"
    if path.suffix.lower() in {".md", ".txt", ".py", ".toml", ".ps1"}:
        return "source_or_report"
    if path.suffix.lower() in {".log", ".db"}:
        return "runtime_local"
    return "other"


def push_policy(category, size_mb):
    if category in {"secret_or_credential", "local_only", "runtime_local"}:
        return "do_not_push"
    if size_mb >= 95:
        return "do_not_push_github_limit"
    if category in {"model_binary", "generated_data"}:
        return "prefer_artifact_storage"
    return "push_ok"


def build_manifest():
    rows = []
    for path in ROOT.rglob("*"):
        if ".git" in path.parts or not path.is_file():
            continue
        size_mb = path.stat().st_size / (1024 * 1024)
        category = category_for(path)
        rows.append(
            {
                "path": str(path),
                "size_mb": round(size_mb, 3),
                "category": category,
                "push_policy": push_policy(category, size_mb),
            }
        )
    return pd.DataFrame(rows).sort_values(["push_policy", "size_mb"], ascending=[True, False])


def write_report(manifest):
    summary = (
        manifest.groupby(["push_policy", "category"], as_index=False)
        .agg(files=("path", "size"), total_mb=("size_mb", "sum"))
        .sort_values(["push_policy", "total_mb"], ascending=[True, False])
    )
    largest = manifest.sort_values("size_mb", ascending=False).head(40)
    lines = [
        "# Artifact Manifest",
        "",
        "This records local project files and the current push policy. It helps keep GitHub focused on source, docs, feedback, and reports while large regenerated data stays local.",
        "",
        "## Summary",
        "",
        "```text",
        summary.to_string(index=False),
        "```",
        "",
        "## Largest Local Files",
        "",
        "```text",
        largest.to_string(index=False),
        "```",
    ]
    REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")


def main():
    ARTIFACT_DIR.mkdir(exist_ok=True)
    manifest = build_manifest()
    manifest.to_csv(MANIFEST_FILE, index=False)
    write_report(manifest)
    print(REPORT_FILE.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
