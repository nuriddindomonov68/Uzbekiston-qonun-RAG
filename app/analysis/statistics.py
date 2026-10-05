import pandas as pd
import numpy as np
from typing import Dict, Any


class StatsAnalyzer:
    """Automated descriptive statistics for any Pandas DataFrame."""

    @staticmethod
    def get_basic_info(df: pd.DataFrame) -> Dict[str, Any]:
        return {
            "shape":        df.shape,
            "row_count":    int(df.shape[0]),
            "column_count": int(df.shape[1]),
            "columns":      list(df.columns),
        }

    @staticmethod
    def get_data_types(df: pd.DataFrame) -> Dict[str, str]:
        return {col: str(dt) for col, dt in df.dtypes.items()}

    @staticmethod
    def get_missing_values(df: pd.DataFrame) -> Dict[str, Any]:
        counts  = df.isnull().sum()
        pct     = (counts / len(df)) * 100
        return {
            col: {"count": int(counts[col]), "percentage": float(round(pct[col], 2))}
            for col in df.columns
        }

    @staticmethod
    def get_duplicate_rows(df: pd.DataFrame) -> int:
        return int(df.duplicated().sum())

    @staticmethod
    def get_numerical_summary(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        num = df.select_dtypes(include=[np.number])
        if num.empty:
            return {}
        desc = num.describe().to_dict()
        return {
            col: {k: (float(v) if pd.notna(v) else None) for k, v in stats.items()}
            for col, stats in desc.items()
        }

    @staticmethod
    def get_categorical_summary(df: pd.DataFrame) -> Dict[str, Any]:
        cat = df.select_dtypes(exclude=[np.number])
        if cat.empty:
            return {}
        result = {}
        for col in cat.columns:
            vc = cat[col].value_counts().head(10)
            result[col] = {
                "unique_count": int(cat[col].nunique()),
                "top_values":   {str(k): int(v) for k, v in vc.items()},
            }
        return result

    @staticmethod
    def get_correlation_matrix(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        num = df.select_dtypes(include=[np.number])
        if num.empty:
            return {}
        return num.corr().fillna(0).to_dict()

    @staticmethod
    def run_full_eda(df: pd.DataFrame) -> Dict[str, Any]:
        return {
            "basic_info":           StatsAnalyzer.get_basic_info(df),
            "data_types":           StatsAnalyzer.get_data_types(df),
            "missing_values":       StatsAnalyzer.get_missing_values(df),
            "duplicate_rows":       StatsAnalyzer.get_duplicate_rows(df),
            "numerical_summary":    StatsAnalyzer.get_numerical_summary(df),
            "categorical_summary":  StatsAnalyzer.get_categorical_summary(df),
            "correlation_matrix":   StatsAnalyzer.get_correlation_matrix(df),
        }
