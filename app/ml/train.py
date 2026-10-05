import pickle
import os
import uuid
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, LogisticRegression,
)
from sklearn.model_selection import (
    train_test_split, KFold, cross_val_score, GridSearchCV,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder, MinMaxScaler, OneHotEncoder,
    OrdinalEncoder, RobustScaler, StandardScaler,
)
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
)

try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError:
    XGBClassifier = XGBRegressor = None  # type: ignore

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except ImportError:
    LGBMClassifier = LGBMRegressor = None  # type: ignore

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
except ImportError:
    CatBoostClassifier = CatBoostRegressor = None  # type: ignore

from app.utils.logging import get_logger

logger = get_logger("ml.train")


class MLTrainer:

    @staticmethod
    def detect_task_type(df: pd.DataFrame, target_col: str) -> str:
        y = df[target_col].dropna()
        if pd.api.types.is_object_dtype(y) or isinstance(y.dtype, pd.CategoricalDtype):
            return "classification"
        if pd.api.types.is_integer_dtype(y) and y.nunique() <= 10:
            return "classification"
        return "regression"

    @staticmethod
    def _make_preprocessor(num_cols: List[str], cat_cols: List[str], scaling: str) -> ColumnTransformer:
        scaler = {"minmax": MinMaxScaler(), "robust": RobustScaler()}.get(scaling, StandardScaler())
        num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", scaler)])
        cat_pipe = Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        return ColumnTransformer([("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)])

    @staticmethod
    def _regressors() -> Dict[str, Any]:
        m: Dict[str, Any] = {
            "linear_regression": LinearRegression(),
            "ridge": Ridge(),
            "lasso": Lasso(),
            "decision_tree": DecisionTreeRegressor(max_depth=6, random_state=42),
            "random_forest": RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42),
            "gradient_boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
            "knn": KNeighborsRegressor(),
            "svm": SVR(),
        }
        if XGBRegressor:
            m["xgboost"] = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
        if LGBMRegressor:
            m["lightgbm"] = LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
        if CatBoostRegressor:
            m["catboost"] = CatBoostRegressor(n_estimators=100, random_state=42, verbose=0)
        return m

    @staticmethod
    def _classifiers() -> Dict[str, Any]:
        m: Dict[str, Any] = {
            "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
            "naive_bayes": GaussianNB(),
            "knn": KNeighborsClassifier(),
            "svm": SVC(probability=True, random_state=42),
            "decision_tree": DecisionTreeClassifier(max_depth=6, random_state=42),
            "random_forest": RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),
            "gradient_boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        }
        if XGBClassifier:
            m["xgboost"] = XGBClassifier(n_estimators=100, eval_metric="logloss", random_state=42, verbosity=0)
        if LGBMClassifier:
            m["lightgbm"] = LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
        if CatBoostClassifier:
            m["catboost"] = CatBoostClassifier(n_estimators=100, random_state=42, verbose=0)
        return m

    @staticmethod
    def train_supervised(
        df: pd.DataFrame,
        target_col: str,
        features: Optional[List[str]] = None,
        task_type: Optional[str] = None,
        algorithm: str = "random_forest",
        scaling: str = "standard",
        test_size: float = 0.2,
        cv_folds: int = 5,
        hyperparameter_tuning: bool = False,
        session_id: str = "default",
    ) -> Dict[str, Any]:
        from app.ml.evaluate import MLEvaluator
        from app.services.storage import StorageService

        if not task_type:
            task_type = MLTrainer.detect_task_type(df, target_col)
        if not features:
            features = [c for c in df.columns if c != target_col]

        train_df = df[[target_col] + features].dropna(subset=[target_col])
        X = train_df[features]
        y = train_df[target_col]

        target_encoder: Optional[LabelEncoder] = None
        if task_type == "classification" and (
            pd.api.types.is_object_dtype(y) or isinstance(y.dtype, pd.CategoricalDtype)
        ):
            target_encoder = LabelEncoder()
            y = target_encoder.fit_transform(y.astype(str))

        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
        pre = MLTrainer._make_preprocessor(num_cols, cat_cols, scaling)

        algo = algorithm.lower().replace(" ", "_")
        model_dict = MLTrainer._regressors() if task_type == "regression" else MLTrainer._classifiers()
        base_model = model_dict.get(algo, model_dict["random_forest"])

        pipeline = Pipeline([("pre", pre), ("model", base_model)])
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=42)

        if hyperparameter_tuning:
            grid: Dict[str, Any] = {}
            if "random_forest" in algo:
                grid = {"model__n_estimators": [50, 100], "model__max_depth": [4, 6]}
            elif "decision_tree" in algo:
                grid = {"model__max_depth": [3, 5, 8]}
            elif "knn" in algo:
                grid = {"model__n_neighbors": [3, 5, 7]}
            if grid:
                scoring = "r2" if task_type == "regression" else "accuracy"
                gs = GridSearchCV(pipeline, grid, cv=3, n_jobs=-1, scoring=scoring)
                gs.fit(X_tr, y_tr)
                pipeline = gs.best_estimator_
            else:
                pipeline.fit(X_tr, y_tr)
        else:
            pipeline.fit(X_tr, y_tr)

        y_pred = pipeline.predict(X_te)
        y_proba = None
        if task_type == "classification" and hasattr(pipeline.named_steps["model"], "predict_proba"):
            try:
                y_proba = pipeline.predict_proba(X_te)
            except Exception:
                pass

        if task_type == "regression":
            metrics = MLEvaluator.evaluate_regression(y_te, y_pred)
        else:
            classes = target_encoder.classes_.tolist() if target_encoder else None
            metrics = MLEvaluator.evaluate_classification(y_te, y_pred, y_proba, classes)

        try:
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            scores = cross_val_score(pipeline, X, y, cv=cv,
                                     scoring="r2" if task_type == "regression" else "accuracy")
            metrics["cv_mean_score"] = float(np.mean(scores))
        except Exception:
            pass

        model_id  = str(uuid.uuid4())
        filepath  = StorageService.get_model_path(session_id, f"{model_id}.pkl")
        with open(filepath, "wb") as fh:
            pickle.dump({
                "pipeline": pipeline, "target_encoder": target_encoder,
                "features": features, "target_col": target_col,
                "task_type": task_type, "num_cols": num_cols,
                "cat_cols": cat_cols, "algorithm": algo,
            }, fh)

        return {
            "success":       True,
            "model_id":      model_id,
            "task_type":     task_type,
            "algorithm":     algo,
            "target_column": target_col,
            "features":      features,
            "metrics":       metrics,
            "filepath":      filepath,
        }

    @staticmethod
    def train_clustering(
        df: pd.DataFrame,
        features: List[str],
        algorithm: str = "kmeans",
        n_clusters: int = 3,
        session_id: str = "default",
    ) -> Dict[str, Any]:
        from app.ml.evaluate import MLEvaluator
        from app.services.storage import StorageService

        X = df[features].dropna()
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
        pre = MLTrainer._make_preprocessor(num_cols, cat_cols, "standard")

        algo = algorithm.lower().replace(" ", "_")
        if algo == "dbscan":
            mdl = DBSCAN(eps=0.5, min_samples=5)
        elif algo in ("hierarchical", "agglomerative"):
            mdl = AgglomerativeClustering(n_clusters=n_clusters)
        else:
            mdl = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")

        pipeline = Pipeline([("pre", pre), ("model", mdl)])
        labels = pipeline.fit_predict(X)
        X_trans = pipeline.named_steps["pre"].transform(X)
        metrics = MLEvaluator.evaluate_clustering(X_trans, labels)

        model_id = str(uuid.uuid4())
        filepath = StorageService.get_model_path(session_id, f"{model_id}.pkl")
        with open(filepath, "wb") as fh:
            pickle.dump({"pipeline": pipeline, "features": features,
                         "task_type": "clustering", "algorithm": algo}, fh)

        return {
            "success":   True,
            "model_id":  model_id,
            "task_type": "clustering",
            "algorithm": algo,
            "features":  features,
            "metrics":   metrics,
            "filepath":  filepath,
        }
