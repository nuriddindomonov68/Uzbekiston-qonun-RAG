"""
Agent vositalari (tools).

Har bir vosita `fn(ctx, **args) -> dict` ko'rinishida; natija JSON'ga aylantiriladigan lug'at.
Vositalar mavjud modullarni (analysis, visualization, ml) ishlatadi va natijalarni bazaga yozadi.
"""
import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app.analysis import OutlierDetector, StatsAnalyzer
from app.database.repository import ChartRepository, ModelRepository
from app.ml import MLExplainer, MLTrainer
from app.services.storage import StorageService
from app.utils.logging import get_logger
from app.visualization import VisualizationEngine

logger = get_logger("tools.registry")

CHART_TYPES = [
    "histogram", "bar_chart", "scatter_plot", "line_chart", "box_plot", "pie_chart",
    "heatmap", "violin_plot", "count_plot", "distribution_plot", "pair_plot",
]


def to_jsonable(obj: Any) -> Any:
    """numpy/pandas qiymatlarini oddiy Python turlariga aylantiradi (NaN -> None)."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if (np.isnan(f) or np.isinf(f)) else f
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, (pd.Timestamp, pd.Timedelta)):
        return str(obj)
    return obj


@dataclass
class ToolContext:
    df: pd.DataFrame
    session_id: str
    dataset_id: str
    dataset_path: str = ""
    previous: List[Dict[str, Any]] = field(default_factory=list)

    def last_output(self, tool: str, key: str) -> Optional[Any]:
        for out in reversed(self.previous):
            if out.get("tool") == tool and out.get("success") and key in out.get("result", {}):
                return out["result"][key]
        return None


def _need_col(df: pd.DataFrame, col: Optional[str], label: str) -> str:
    if not col:
        raise ValueError(f"'{label}' ustuni ko'rsatilmagan.")
    if col not in df.columns:
        raise ValueError(f"'{col}' ustuni topilmadi. Mavjud ustunlar: {', '.join(map(str, df.columns))}")
    return col


# ----------------------------------------------------------------------------- tools
def tool_eda(ctx: ToolContext, **_: Any) -> Dict[str, Any]:
    eda = StatsAnalyzer.run_full_eda(ctx.df)
    miss = {c: v for c, v in eda["missing_values"].items() if v["count"] > 0}
    return to_jsonable({
        "rows": eda["basic_info"]["row_count"],
        "columns": eda["basic_info"]["column_count"],
        "column_names": eda["basic_info"]["columns"],
        "data_types": eda["data_types"],
        "missing_columns": miss,
        "duplicate_rows": eda["duplicate_rows"],
        "numerical_summary": eda["numerical_summary"],
        "categorical_summary": eda["categorical_summary"],
        "correlation_matrix": eda["correlation_matrix"],
    })


def tool_outliers(ctx: ToolContext, method: str = "iqr", **_: Any) -> Dict[str, Any]:
    method = "zscore" if str(method).lower().replace("-", "") == "zscore" else "iqr"
    res = OutlierDetector.get_all_outliers(ctx.df, method=method)
    with_outliers = {c: r for c, r in res.items() if r.get("outlier_count", 0) > 0}
    return to_jsonable({"method": method, "columns": res, "columns_with_outliers": with_outliers})


def tool_chart(ctx: ToolContext, chart_type: str = "histogram", x: Optional[str] = None,
               y: Optional[str] = None, hue: Optional[str] = None,
               title: Optional[str] = None, **_: Any) -> Dict[str, Any]:
    chart_type = str(chart_type).lower().replace(" ", "_").replace("-", "_")
    alias = {"bar": "bar_chart", "scatter": "scatter_plot", "line": "line_chart", "box": "box_plot",
             "pie": "pie_chart", "violin": "violin_plot", "count": "count_plot",
             "correlation_matrix": "heatmap", "dist": "distribution_plot", "distribution": "distribution_plot"}
    chart_type = alias.get(chart_type, chart_type)
    if chart_type not in CHART_TYPES:
        raise ValueError(f"Noma'lum grafik turi: {chart_type!r}. Mavjud: {', '.join(CHART_TYPES)}")

    df = ctx.df
    if x is None and chart_type not in ("heatmap", "pair_plot"):
        num = df.select_dtypes(include=[np.number]).columns
        if len(num) == 0:
            raise ValueError("Grafik uchun ustun tanlanmagan va raqamli ustun yo'q.")
        x = num[0]
    if x:
        _need_col(df, x, "x")
    if y:
        _need_col(df, y, "y")
    if hue:
        _need_col(df, hue, "hue")
    if chart_type in ("scatter_plot", "line_chart") and not y:
        num = [c for c in df.select_dtypes(include=[np.number]).columns if c != x]
        if not num:
            raise ValueError(f"{chart_type} uchun y ustuni kerak.")
        y = num[0]

    chart_id = str(uuid.uuid4())
    path = StorageService.get_chart_path(ctx.session_id, f"{chart_id}.png")
    title = title or None
    VisualizationEngine.generate_chart(df, chart_type, x=x, y=y, hue=hue, title=title, output_path=path)
    ChartRepository.register(chart_id, ctx.session_id, ctx.dataset_id,
                             title or f"{chart_type} {x or ''} {y or ''}".strip(), chart_type, path)
    return {"chart_id": chart_id, "chart_type": chart_type, "x": x, "y": y, "hue": hue, "chart_path": path}


def tool_auto_charts(ctx: ToolContext, max_charts: int = 4, **_: Any) -> Dict[str, Any]:
    """Avtomatik: korrelyatsiya xaritasi + birinchi raqamli ustunlar gistogrammasi."""
    df = ctx.df
    num = list(df.select_dtypes(include=[np.number]).columns)
    charts: List[Dict[str, Any]] = []
    if len(num) >= 2:
        charts.append(tool_chart(ctx, chart_type="heatmap"))
    for col in num[: max(0, int(max_charts) - len(charts))]:
        charts.append(tool_chart(ctx, chart_type="histogram", x=col))
    if not charts:
        cats = df.select_dtypes(exclude=[np.number]).columns
        if len(cats):
            charts.append(tool_chart(ctx, chart_type="count_plot", x=cats[0]))
    return {"charts": charts, "chart_paths": [c["chart_path"] for c in charts]}


def _auto_features(df: pd.DataFrame, target: str, features: Optional[List[str]]) -> Tuple[List[str], List[str]]:
    if features:
        for f in features:
            _need_col(df, f, "feature")
        return [f for f in features if f != target], []
    keep, dropped = [], []
    n = len(df)
    for c in df.columns:
        if c == target:
            continue
        s = df[c]
        is_text = not pd.api.types.is_numeric_dtype(s)
        # ID yoki erkin matn ustunlari (deyarli har qator noyob) model uchun foydasiz
        if is_text and s.nunique(dropna=True) > max(20, 0.5 * n):
            dropped.append(c)
        elif pd.api.types.is_datetime64_any_dtype(s):
            dropped.append(c)
        else:
            keep.append(c)
    if not keep:
        raise ValueError("Model uchun yaroqli feature ustunlari topilmadi.")
    return keep, dropped


def tool_train_model(ctx: ToolContext, target: Optional[str] = None, algorithm: str = "random_forest",
                     features: Optional[List[str]] = None, task_type: Optional[str] = None,
                     scaling: str = "standard", hyperparameter_tuning: bool = False,
                     **_: Any) -> Dict[str, Any]:
    df = ctx.df
    target = _need_col(df, target, "target")
    if df[target].dropna().nunique() < 2:
        raise ValueError(f"'{target}' ustunida kamida 2 xil qiymat bo'lishi kerak.")
    if len(df) < 20:
        raise ValueError("Model o'qitish uchun kamida 20 ta qator kerak.")
    feats, dropped = _auto_features(df, target, features)
    res = MLTrainer.train_supervised(
        df, target_col=target, features=feats, task_type=task_type, algorithm=algorithm,
        scaling=scaling, hyperparameter_tuning=bool(hyperparameter_tuning), session_id=ctx.session_id)
    ModelRepository.register(res["model_id"], ctx.session_id, ctx.dataset_id, res["task_type"],
                             res["algorithm"], target, feats, to_jsonable(res["metrics"]), res["filepath"])
    out = dict(res)
    out["excluded_features"] = dropped
    # katta confusion_matrix / classification_report javobni shishirmasligi uchun qisqartiramiz
    m = dict(res["metrics"])
    m.pop("classification_report", None)
    out["metrics"] = m
    return to_jsonable(out)


def tool_cluster(ctx: ToolContext, features: Optional[List[str]] = None, algorithm: str = "kmeans",
                 n_clusters: int = 3, **_: Any) -> Dict[str, Any]:
    df = ctx.df
    if not features:
        features = list(df.select_dtypes(include=[np.number]).columns[:8])
    if not features:
        raise ValueError("Klasterlash uchun raqamli ustun topilmadi.")
    for f in features:
        _need_col(df, f, "feature")
    n_clusters = max(2, int(n_clusters))
    res = MLTrainer.train_clustering(df, features=features, algorithm=algorithm,
                                     n_clusters=n_clusters, session_id=ctx.session_id)
    ModelRepository.register(res["model_id"], ctx.session_id, ctx.dataset_id, "clustering",
                             res["algorithm"], None, features, to_jsonable(res["metrics"]), res["filepath"])
    return to_jsonable(res)


def tool_explain_model(ctx: ToolContext, model_id: Optional[str] = None, **_: Any) -> Dict[str, Any]:
    model_id = model_id or ctx.last_output("train_model", "model_id")
    if model_id:
        info = ModelRepository.get(model_id)
    else:
        models = ModelRepository.get_by_session(ctx.session_id)
        info = models[0] if models else None
    if not info:
        raise ValueError("Izohlash uchun model topilmadi. Avval model o'qiting.")
    imp = MLExplainer.get_feature_importance(info["filepath"], ctx.df)
    if "error" in imp:
        raise ValueError(imp["error"])
    ranked = imp["native_importance"] or imp["permutation_importance"]
    top = dict(list(ranked.items())[:10])
    return to_jsonable({"model_id": info["id"], "algorithm": info["algorithm"],
                        "top_features": top, "permutation_importance": imp["permutation_importance"]})


def tool_report(ctx: ToolContext, **_: Any) -> Dict[str, Any]:
    from app.tools.report import generate_report
    path = generate_report(ctx.df, ctx.session_id, ctx.dataset_id)
    return {"report_path": path}


TOOLS: Dict[str, Callable[..., Dict[str, Any]]] = {
    "eda": tool_eda,
    "outliers": tool_outliers,
    "chart": tool_chart,
    "auto_charts": tool_auto_charts,
    "train_model": tool_train_model,
    "cluster": tool_cluster,
    "explain_model": tool_explain_model,
    "report": tool_report,
}

TOOL_DESCRIPTIONS = {
    "eda": "Umumiy tahlil: o'lcham, turlar, yo'qolgan qiymatlar, statistika, korrelyatsiya. args: {}",
    "outliers": "Anomaliyalarni topish. args: {method: 'iqr'|'zscore'}",
    "chart": f"Grafik chizish. args: {{chart_type: {CHART_TYPES}, x, y, hue}}",
    "auto_charts": "Avtomatik bir nechta grafik (korrelyatsiya + gistogrammalar). args: {}",
    "train_model": "Bashorat modeli o'qitish (klassifikatsiya/regressiya avtomatik). args: {target, algorithm, scaling}",
    "cluster": "Klasterlash. args: {features: [..], algorithm: 'kmeans'|'dbscan'|'hierarchical', n_clusters}",
    "explain_model": "Oxirgi modelda qaysi ustunlar muhimligini ko'rsatish. args: {}",
    "report": "PDF hisobot yaratish. args: {}",
}


def run_tool(name: str, ctx: ToolContext, args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Vositani xavfsiz ishga tushiradi; xato bo'lsa ham lug'at qaytaradi."""
    fn = TOOLS.get(name)
    if fn is None:
        return {"tool": name, "success": False, "error": f"Noma'lum vosita: {name}"}
    try:
        result = fn(ctx, **(args or {}))
        return {"tool": name, "success": True, "args": args or {}, "result": result}
    except Exception as exc:  # noqa: BLE001 - agent xatoni o'zi hal qiladi
        logger.warning(f"Tool '{name}' xatosi: {exc}")
        return {"tool": name, "success": False, "args": args or {}, "error": str(exc)}


def dumps(obj: Any) -> str:
    return json.dumps(to_jsonable(obj), ensure_ascii=False)
