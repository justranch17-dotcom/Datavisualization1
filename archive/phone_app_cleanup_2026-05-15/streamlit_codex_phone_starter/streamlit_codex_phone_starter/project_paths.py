from pathlib import Path


APP_DIR = Path(__file__).resolve().parent


def find_project_root():
    """Find the Datavisulaization project root from this sidecar app."""
    for parent in [APP_DIR, *APP_DIR.parents]:
        if (parent / "Mach2AImarket.py").exists() or (parent / "Marketvisual.py").exists():
            return parent
    return APP_DIR


PROJECT_ROOT = find_project_root()
