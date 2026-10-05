"""Dataset yuklash va o'qish xizmati (API ham, dashboard ham shuni ishlatadi)."""
import os
import re
import uuid
from typing import Any, Dict

import pandas as pd

from app.core.config import settings
from app.database.repository import DatasetRepository, SessionRepository
from app.services.storage import StorageService
from app.tools.loader import SUPPORTED_EXTENSIONS, file_type_of, load_dataframe


class DatasetError(ValueError):
    """Foydalanuvchiga ko'rsatish mumkin bo'lgan dataset xatosi."""


def _safe_filename(filename: str) -> str:
    name = os.path.basename((filename or "").replace("\\", "/"))
    name = re.sub(r"[^\w.\- ]", "_", name).strip(". ")
    return name or "dataset.csv"


class DatasetService:
    @staticmethod
    def register_upload(session_id: str, filename: str, content: bytes) -> Dict[str, Any]:
        filename = _safe_filename(filename)
        ext = os.path.splitext(filename)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise DatasetError(
                f"'{ext or filename}' turidagi fayl qo'llab-quvvatlanmaydi. "
                f"Ruxsat etilgan: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
        if not content:
            raise DatasetError("Fayl bo'sh.")
        if len(content) > settings.MAX_UPLOAD_MB * 1024 * 1024:
            raise DatasetError(f"Fayl juda katta (maksimum {settings.MAX_UPLOAD_MB} MB).")

        SessionRepository.create(session_id, {})
        filepath = StorageService.save_uploaded_file(session_id, filename, content)
        try:
            df = load_dataframe(filepath)
        except Exception as exc:
            os.remove(filepath)
            raise DatasetError(f"Faylni o'qib bo'lmadi: {exc}") from exc
        if df.empty or df.shape[1] == 0:
            os.remove(filepath)
            raise DatasetError("Faylda ma'lumot topilmadi.")

        dataset_id = str(uuid.uuid4())
        schema = {c: str(t) for c, t in df.dtypes.items()}
        DatasetRepository.register(dataset_id, session_id, filename, filepath,
                                   file_type_of(filename), int(df.shape[0]),
                                   int(df.shape[1]), schema)
        SessionRepository.touch(session_id)
        return DatasetRepository.get(dataset_id)

    @staticmethod
    def get_dataframe(dataset_id: str) -> pd.DataFrame:
        info = DatasetRepository.get(dataset_id)
        if not info:
            raise DatasetError("Dataset topilmadi.")
        return load_dataframe(info["filepath"])
