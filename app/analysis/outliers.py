import pandas as pd
import numpy as np
from typing import Dict, Any


class OutlierDetector:

    @staticmethod
    def detect_iqr(df: pd.DataFrame, col: str) -> Dict[str, Any]:
        if not pd.api.types.is_numeric_dtype(df[col]):
            return {"error": f"{col} is not numeric"}
        s  = df[col].dropna()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr  = q3 - q1
        lo   = q1 - 1.5 * iqr
        hi   = q3 + 1.5 * iqr
        mask = (s < lo) | (s > hi)
        return {
            "method":             "IQR",
            "lower_bound":        float(lo),
            "upper_bound":        float(hi),
            "outlier_count":      int(mask.sum()),
            "outlier_percentage": float(round(mask.sum() / len(df) * 100, 2)),
        }

    @staticmethod
    def detect_zscore(df: pd.DataFrame, col: str, threshold: float = 3.0) -> Dict[str, Any]:
        if not pd.api.types.is_numeric_dtype(df[col]):
            return {"error": f"{col} is not numeric"}
        s  = df[col].dropna()
        mu, sigma = s.mean(), s.std()
        if sigma == 0:
            return {"error": f"std of {col} is zero"}
        mask = np.abs((s - mu) / sigma) > threshold
        return {
            "method":             "Z-score",
            "outlier_count":      int(mask.sum()),
            "outlier_percentage": float(round(mask.sum() / len(df) * 100, 2)),
        }

    @staticmethod
    def get_all_outliers(df: pd.DataFrame, method: str = "iqr") -> Dict[str, Dict[str, Any]]:
        cols   = df.select_dtypes(include=[np.number]).columns
        detect = OutlierDetector.detect_zscore if method.lower() == "zscore" else OutlierDetector.detect_iqr
        results = {col: detect(df, col) for col in cols}
        return {col: res for col, res in results.items() if "error" not in res}
