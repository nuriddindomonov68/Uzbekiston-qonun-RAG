import os
from app.core.config import settings
from app.utils.logging import get_logger

logger = get_logger("services.storage")


def _abs(path: str) -> str:
    """Return absolute path; resolve relative paths against the config STORAGE_DIR parent."""
    if not os.path.isabs(path):
        base = os.path.dirname(os.path.abspath(settings.STORAGE_DIR))
        path = os.path.join(base, path)
    return os.path.normpath(path)


class StorageService:
    @staticmethod
    def init_directories() -> None:
        for attr in ("STORAGE_DIR", "UPLOAD_DIR", "CHARTS_DIR", "REPORTS_DIR", "MODELS_DIR"):
            path = _abs(getattr(settings, attr))
            os.makedirs(path, exist_ok=True)
            logger.info(f"Directory ready: {path}")

    @staticmethod
    def save_uploaded_file(session_id: str, filename: str, content: bytes) -> str:
        StorageService.init_directories()
        dest_dir = os.path.join(_abs(settings.UPLOAD_DIR), session_id)
        os.makedirs(dest_dir, exist_ok=True)
        filepath = os.path.join(dest_dir, filename)
        with open(filepath, "wb") as fh:
            fh.write(content)
        logger.info(f"Saved upload → {filepath}")
        return os.path.abspath(filepath)

    @staticmethod
    def get_chart_path(session_id: str, filename: str) -> str:
        StorageService.init_directories()
        d = os.path.join(_abs(settings.CHARTS_DIR), session_id)
        os.makedirs(d, exist_ok=True)
        return os.path.abspath(os.path.join(d, filename))

    @staticmethod
    def get_report_path(session_id: str, filename: str) -> str:
        StorageService.init_directories()
        d = os.path.join(_abs(settings.REPORTS_DIR), session_id)
        os.makedirs(d, exist_ok=True)
        return os.path.abspath(os.path.join(d, filename))

    @staticmethod
    def get_model_path(session_id: str, filename: str) -> str:
        StorageService.init_directories()
        d = os.path.join(_abs(settings.MODELS_DIR), session_id)
        os.makedirs(d, exist_ok=True)
        return os.path.abspath(os.path.join(d, filename))
