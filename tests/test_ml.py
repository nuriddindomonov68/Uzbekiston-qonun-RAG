import os

from app.ml import MLExplainer, MLTrainer


def test_detect_task_type(df):
    assert MLTrainer.detect_task_type(df, "sotib_oldi") == "classification"
    assert MLTrainer.detect_task_type(df, "daromad") == "regression"


def test_train_classification_and_explain(df):
    res = MLTrainer.train_supervised(df, "sotib_oldi", algorithm="random_forest", session_id="t")
    assert res["success"] and os.path.exists(res["filepath"])
    assert res["metrics"]["accuracy"] > 0.8
    imp = MLExplainer.get_feature_importance(res["filepath"], df)
    assert imp["native_importance"]


def test_train_regression(df):
    res = MLTrainer.train_supervised(df.dropna(), "daromad", algorithm="ridge", session_id="t")
    assert res["task_type"] == "regression" and "r2" in res["metrics"]


def test_clustering(df):
    res = MLTrainer.train_clustering(df, ["yosh", "daromad"], n_clusters=3, session_id="t")
    assert res["metrics"]["silhouette_score"] > 0
