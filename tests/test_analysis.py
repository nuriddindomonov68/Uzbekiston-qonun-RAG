from app.analysis import OutlierDetector, StatsAnalyzer


def test_full_eda_keys(df):
    eda = StatsAnalyzer.run_full_eda(df)
    assert set(eda) == {"basic_info", "data_types", "missing_values", "duplicate_rows",
                        "numerical_summary", "categorical_summary", "correlation_matrix"}
    assert eda["basic_info"]["row_count"] == 200
    assert eda["missing_values"]["daromad"]["count"] == 5


def test_outliers_detect_extreme_value(df):
    iqr = OutlierDetector.detect_iqr(df, "daromad")
    assert iqr["outlier_count"] >= 1
    z = OutlierDetector.detect_zscore(df, "daromad")
    assert z["outlier_count"] >= 1
    assert "error" in OutlierDetector.detect_iqr(df, "shahar")


def test_get_all_outliers_only_numeric(df):
    res = OutlierDetector.get_all_outliers(df)
    assert "daromad" in res and "shahar" not in res
