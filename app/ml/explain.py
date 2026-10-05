import os
import pickle
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from app.utils.logging import get_logger

logger = get_logger("ml.explain")


class MLExplainer:

    @staticmethod
    def get_feature_importance(model_path: str, df: pd.DataFrame) -> Dict[str, Any]:
        if not os.path.exists(model_path):
            return {"error": "Model file not found"}

        with open(model_path, "rb") as fh:
            data = pickle.load(fh)

        pipeline      = data["pipeline"]
        features      = data["features"]
        target_col    = data.get("target_col")
        task_type     = data.get("task_type", "regression")
        target_enc    = data.get("target_encoder")

        native: Dict[str, float] = {}
        perm:   Dict[str, float] = {}

        try:
            mdl = pipeline.named_steps["model"]
            pre = pipeline.named_steps["pre"]

            feat_names: List[str] = []
            for name, trans, cols in pre.transformers_:
                if name == "num":
                    feat_names.extend(cols)
                elif name == "cat":
                    try:
                        feat_names.extend(trans.named_steps["ohe"].get_feature_names_out(cols))
                    except Exception:
                        feat_names.extend(cols)

            if hasattr(mdl, "feature_importances_"):
                imp = mdl.feature_importances_
                native = dict(sorted(zip(feat_names, imp), key=lambda kv: kv[1], reverse=True))
            elif hasattr(mdl, "coef_"):
                coef = np.abs(mdl.coef_)
                if coef.ndim > 1:
                    coef = coef.mean(axis=0)
                native = dict(sorted(zip(feat_names, coef), key=lambda kv: kv[1], reverse=True))
        except Exception as exc:
            logger.warning(f"Native importance unavailable: {exc}")

        if target_col and target_col in df.columns:
            try:
                X = df[features]
                y = df[target_col]
                if task_type == "classification" and target_enc is not None:
                    y = target_enc.transform(y.astype(str))
                sample = min(len(X), 500)
                Xs = X.sample(n=sample, random_state=42)
                ys = y.loc[Xs.index] if hasattr(y, "loc") else y[Xs.index]
                res  = permutation_importance(pipeline, Xs, ys, n_repeats=5, random_state=42, n_jobs=-1)
                perm = dict(sorted(
                    {col: float(res.importances_mean[i]) for i, col in enumerate(features)}.items(),
                    key=lambda kv: kv[1], reverse=True,
                ))
            except Exception as exc:
                logger.warning(f"Permutation importance unavailable: {exc}")

        return {
            "native_importance":      native,
            "permutation_importance": perm,
            "shap_summary":           perm,   # SHAP surrogate: permutation importance
        }
