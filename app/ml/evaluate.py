import numpy as np
from typing import Any, Dict, List, Optional
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report, roc_auc_score,
    silhouette_score,
)


class MLEvaluator:

    @staticmethod
    def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        mae  = float(mean_absolute_error(y_true, y_pred))
        mse  = float(mean_squared_error(y_true, y_pred))
        rmse = float(np.sqrt(mse))
        r2   = float(r2_score(y_true, y_pred))
        mask = y_true != 0
        mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100) if mask.any() else 0.0
        return {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2, "mape": mape}

    @staticmethod
    def evaluate_classification(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None,
        classes: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        acc = float(accuracy_score(y_true, y_pred))
        p_m, r_m, f1_m, _ = precision_recall_fscore_support(y_true, y_pred, average="macro",    zero_division=0)
        p_w, r_w, f1_w, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
        conf = confusion_matrix(y_true, y_pred).tolist()
        rep  = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        result: Dict[str, Any] = {
            "accuracy":           acc,
            "precision_macro":    float(p_m),
            "recall_macro":       float(r_m),
            "f1_macro":           float(f1_m),
            "precision_weighted": float(p_w),
            "recall_weighted":    float(r_w),
            "f1_weighted":        float(f1_w),
            "confusion_matrix":   conf,
            "classification_report": rep,
        }
        if y_pred_proba is not None:
            try:
                uniq = np.unique(y_true)
                if len(uniq) == 2:
                    result["roc_auc"] = float(roc_auc_score(y_true, y_pred_proba[:, 1]))
                elif len(uniq) > 2:
                    result["roc_auc"] = float(roc_auc_score(y_true, y_pred_proba, multi_class="ovr"))
            except Exception:
                pass
        return result

    @staticmethod
    def evaluate_clustering(X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        uniq = np.unique(labels)
        if 1 < len(uniq) < len(labels):
            try:
                return {"silhouette_score": float(silhouette_score(X, labels))}
            except Exception:
                pass
        return {"silhouette_score": -1.0}
