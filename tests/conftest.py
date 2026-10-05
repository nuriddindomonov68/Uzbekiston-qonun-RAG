import numpy as np
import pandas as pd
import pytest

from app.core.config import settings
from app.database.connection import init_db


@pytest.fixture(autouse=True)
def tmp_storage(tmp_path, monkeypatch):
    """Har bir test o'zining vaqtinchalik saqlash papkasida va LLM'siz ishlaydi."""
    base = tmp_path / "storage"
    for name, sub in [("STORAGE_DIR", ""), ("UPLOAD_DIR", "uploads"), ("CHARTS_DIR", "charts"),
                      ("REPORTS_DIR", "reports"), ("MODELS_DIR", "models")]:
        monkeypatch.setattr(settings, name, str(base / sub) if sub else str(base))
    monkeypatch.setattr(settings, "LLM_ENABLED", False)
    init_db()
    return base


@pytest.fixture
def df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 200
    d = pd.DataFrame({
        "yosh": rng.integers(18, 70, n),
        "daromad": rng.normal(5000, 1500, n),
        "shahar": rng.choice(["Toshkent", "Samarqand", "Buxoro"], n),
    })
    d["sotib_oldi"] = ((d["daromad"] > 5000) & (d["yosh"] > 30)).astype(int)
    d.loc[0:4, "daromad"] = np.nan
    d.loc[5, "daromad"] = 90000  # anomaliya
    return d


@pytest.fixture
def csv_bytes(df) -> bytes:
    return df.to_csv(index=False).encode()
