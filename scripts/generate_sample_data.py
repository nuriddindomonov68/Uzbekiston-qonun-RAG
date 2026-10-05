"""Namunaviy dataset yaratadi: assets/sample_data.csv (sintetik mijozlar ma'lumoti)."""
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(n: int = 400, seed: int = 42) -> str:
    rng = np.random.default_rng(seed)
    yosh = rng.integers(18, 70, n)
    shahar = rng.choice(["Toshkent", "Samarqand", "Buxoro", "Namangan", "Andijon"], n,
                        p=[0.4, 0.2, 0.15, 0.15, 0.1])
    jins = rng.choice(["erkak", "ayol"], n)
    daromad = np.clip(rng.normal(5_000_000, 1_800_000, n) + (yosh - 40) * 30_000, 1_200_000, None)
    xaridlar = np.clip(rng.poisson(6, n) + (daromad > 5_500_000).astype(int) * 3, 0, None)
    oxirgi = rng.integers(1, 365, n)
    score = (daromad - 5e6) / 1.8e6 + xaridlar * 0.15 - oxirgi / 300 + rng.normal(0, 0.6, n)
    sotib_oldi = (score > 0.3).astype(int)

    df = pd.DataFrame({
        "mijoz_id": [f"M{i:04d}" for i in range(1, n + 1)],
        "yosh": yosh, "shahar": shahar, "jins": jins,
        "oylik_daromad": daromad.round(0), "xaridlar_soni": xaridlar,
        "oxirgi_xarid_kun": oxirgi, "sotib_oldi": sotib_oldi,
    })
    # Realistik bo'lishi uchun: yo'qolgan qiymatlar va bir nechta anomaliya
    df.loc[rng.choice(n, 12, replace=False), "oylik_daromad"] = np.nan
    df.loc[rng.choice(n, 8, replace=False), "shahar"] = np.nan
    df.loc[rng.choice(n, 4, replace=False), "oylik_daromad"] = 25_000_000

    out = os.path.join(ROOT, "assets", "sample_data.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    return out


if __name__ == "__main__":
    print("Yaratildi:", main())
