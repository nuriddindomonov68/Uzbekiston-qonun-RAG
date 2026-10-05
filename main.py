"""
AI Data Analysis Agent — Top-level entry point.

Supports two run modes:
    python main.py api       → starts FastAPI server on port 8000
    python main.py ui        → starts Streamlit dashboard on port 8501
    python main.py setup     → initialises the SQLite database only
"""
import sys
import os

# Ensure the project root is always on sys.path so that `app.*` imports work
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def run_api() -> None:
    import uvicorn
    from app.core.config import settings

    uvicorn.run(
        "app.api.server:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )


def run_ui() -> None:
    import subprocess
    dashboard = os.path.join(ROOT, "app", "ui", "dashboard.py")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", dashboard,
         "--server.port", "8501", "--server.address", "0.0.0.0"],
        check=True,
    )


def run_setup() -> None:
    from app.database.connection import init_db
    from app.services.storage import StorageService
    init_db()
    StorageService.init_directories()
    print("✅ Database and storage directories initialised successfully.")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "api"

    if mode == "api":
        run_api()
    elif mode == "ui":
        run_ui()
    elif mode == "setup":
        run_setup()
    else:
        print(f"Unknown mode: {mode!r}. Use 'api', 'ui', or 'setup'.")
        sys.exit(1)
