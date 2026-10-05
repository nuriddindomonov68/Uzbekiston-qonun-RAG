"""Dataset fayllarini (CSV, TSV, Excel, JSON) DataFrame'ga yuklash."""
import os
from functools import lru_cache

import pandas as pd

SUPPORTED_EXTENSIONS = {".csv", ".tsv", ".xlsx", ".xls", ".json"}


def file_type_of(filename: str) -> str:
    return os.path.splitext(filename)[1].lower().lstrip(".")


def _read(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        try:
            return pd.read_csv(path)
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding="latin-1")
    if ext == ".tsv":
        return pd.read_csv(path, sep="\t")
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if ext == ".json":
        try:
            return pd.read_json(path)
        except ValueError:
            return pd.read_json(path, lines=True)
    raise ValueError(f"Qo'llab-quvvatlanmaydigan fayl turi: {ext!r}. "
                     f"Ruxsat etilgan: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")


@lru_cache(maxsize=16)
def _read_cached(path: str, mtime: float) -> pd.DataFrame:
    return _read(path)


def load_dataframe(path: str) -> pd.DataFrame:
    """Faylni o'qiydi (o'zgarmagan bo'lsa keshdan) va nusxasini qaytaradi."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fayl topilmadi: {path}")
    df = _read_cached(path, os.path.getmtime(path)).copy()
    # Excel/JSON ba'zan bo'sh ustun nomlari beradi
    df.columns = [str(c) for c in df.columns]
    return df
