"""
Rejalashtiruvchi: foydalanuvchi so'rovidan vositalar ketma-ketligini (reja) tuzadi.

1) LLM (Ollama) JSON formatida reja tuzishga uringan bo'ladi.
2) LLM o'chirilgan, ishlamayotgan yoki noto'g'ri javob bergan bo'lsa -
   kalit so'zlarga asoslangan qoidali rejalashtiruvchi (heuristic_plan) ishlaydi.
   Shuning uchun agent Ollamasiz ham ishlaydi.
"""
import json
import re
from typing import Any, Dict, List, Optional

from app.core.config import settings
from app.tools.registry import CHART_TYPES, TOOL_DESCRIPTIONS, TOOLS
from app.utils.logging import get_logger

logger = get_logger("agents.planner")

MAX_STEPS = 6

_APOS = str.maketrans({"\u2018": "'", "\u2019": "'", "\u02bb": "'", "\u02bc": "'", "`": "'", "\u00b4": "'"})

ALGORITHMS = [
    (r"random\s*forest|tasodifiy o'rmon", "random_forest"),
    (r"xgboost", "xgboost"), (r"lightgbm|lgbm", "lightgbm"), (r"catboost", "catboost"),
    (r"gradient\s*boost", "gradient_boosting"),
    (r"logistic|logistik", "logistic_regression"),
    (r"linear\s*reg|chiziqli regressiya", "linear_regression"),
    (r"ridge", "ridge"), (r"lasso", "lasso"),
    (r"decision\s*tree|daraxt", "decision_tree"),
    (r"naive\s*bayes", "naive_bayes"), (r"knn", "knn"), (r"\bsvm\b", "svm"),
]

CHART_KEYWORDS = [
    (r"heatmap|issiqlik|korrelyatsiya|korelyatsiya|correlation", "heatmap"),
    (r"scatter|nuqta", "scatter_plot"),
    (r"box\s*plot|boxplot|\bbox\b|qutichali", "box_plot"),
    (r"pie|doira|dumaloq", "pie_chart"),
    (r"violin", "violin_plot"),
    (r"pair\s*plot|pairplot", "pair_plot"),
    (r"line|chiziqli grafik|trend|dinamika", "line_chart"),
    (r"count|sanoq", "count_plot"),
    (r"bar|ustunli", "bar_chart"),
    (r"taqsimot|distribution|\bkde\b", "distribution_plot"),
    (r"gistogram|histogram", "histogram"),
]


def normalize(text: str) -> str:
    return text.translate(_APOS).lower()


def mentioned_columns(message: str, columns: List[str]) -> List[str]:
    """Xabarda tilga olingan ustunlar (xabardagi tartibda)."""
    msg = normalize(message)
    found = []
    for col in columns:
        m = re.search(rf"(?<!\w){re.escape(normalize(str(col)))}(?!\w)", msg)
        if m:
            found.append((m.start(), col))
    return [c for _, c in sorted(found)]


def _pick_target(message: str, columns: List[str], mentioned: List[str]) -> Optional[str]:
    msg = normalize(message)
    low = {normalize(str(c)): c for c in columns}
    m = re.search(r"(?:target|maqsad|nishon)\w*\s*[:=]?\s*([^\s,;]+)", msg)
    if m and m.group(1).strip(".,;:!?'\"") in low:
        return low[m.group(1).strip(".,;:!?'\"")]
    m = re.search(r"([^\s,;]+)\s+(?:ni\s+|ning\s+)?(?:bashorat|predict|prognoz)", msg)
    if m and m.group(1).strip(".,;:!?'\"") in low:
        return low[m.group(1).strip(".,;:!?'\"")]
    if mentioned:
        return mentioned[0]
    return columns[-1] if columns else None


def has_tool_intent(message: str) -> bool:
    return bool(heuristic_plan(message, ["__dummy__"]))


def heuristic_plan(message: str, columns: List[str]) -> List[Dict[str, Any]]:
    msg = normalize(message)
    cols = [str(c) for c in columns]
    mentioned = mentioned_columns(message, cols)
    steps: List[Dict[str, Any]] = []

    want_report = re.search(r"hisobot|report|pdf|otchet|отч[её]т", msg)
    want_explain = re.search(r"izoh|muhim|importance|explain|qaysi ustun|ta'sir", msg)
    want_cluster = re.search(r"klaster|cluster|guruhla|segment", msg)
    want_outliers = re.search(r"anomal|outlier|chetlan|g'alati|noodatiy", msg)
    explicit_train = re.search(r"o'qit|oqit|train|yarat|qur\b|qurib|build|\bfit\b", msg)
    want_train = re.search(r"model|bashorat|predict|prognoz|klassifik|classif|regressiya|regression", msg) \
        and not want_cluster
    if want_explain and not explicit_train:
        want_train = None
    want_chart = re.search(r"grafik|chart|plot|diagramma|gistogram|histogram|scatter|heatmap|korrelyatsiya|"
                           r"korelyatsiya|correlation|chiz|vizual|pie|doira|taqsimot|distribution", msg)
    want_eda = re.search(r"tahlil|\beda\b|statistik|analy|umumiy|tavsif|describe|summary|ma'lumot haqida|"
                         r"yo'qolgan|missing|takror|duplicate", msg)

    if want_eda:
        steps.append({"tool": "eda", "args": {}})
    if want_outliers:
        steps.append({"tool": "outliers", "args": {"method": "zscore" if re.search(r"z-?score", msg) else "iqr"}})
    if want_chart:
        ctype = None
        for pat, name in CHART_KEYWORDS:
            if re.search(pat, msg):
                ctype = name
                break
        if ctype is None and not mentioned:
            steps.append({"tool": "auto_charts", "args": {}})
        else:
            args: Dict[str, Any] = {"chart_type": ctype or "histogram"}
            if ctype not in ("heatmap", "pair_plot"):
                if mentioned:
                    args["x"] = mentioned[0]
                if len(mentioned) > 1:
                    args["y"] = mentioned[1]
                if len(mentioned) > 2:
                    args["hue"] = mentioned[2]
            steps.append({"tool": "chart", "args": args})
    if want_cluster:
        args = {"algorithm": "dbscan" if "dbscan" in msg else
                ("hierarchical" if re.search(r"hierarch|ierarx", msg) else "kmeans")}
        m = re.search(r"(\d+)\s*(?:ta\s*)?(?:klaster|cluster|guruh)", msg)
        args["n_clusters"] = int(m.group(1)) if m else 3
        if len(mentioned) >= 2:
            args["features"] = mentioned
        steps.append({"tool": "cluster", "args": args})
    if want_train:
        args = {"target": _pick_target(message, cols, mentioned)}
        for pat, name in ALGORITHMS:
            if re.search(pat, msg):
                args["algorithm"] = name
                break
        steps.append({"tool": "train_model", "args": args})
    if want_explain:
        steps.append({"tool": "explain_model", "args": {}})
    if want_report:
        steps.append({"tool": "report", "args": {}})
    return steps[:MAX_STEPS]


def _llm_plan(message: str, columns: List[str], dtypes: Dict[str, str]) -> List[Dict[str, Any]]:
    from app.core.llm import get_llm

    tools = "\n".join(f"- {n}: {d}" for n, d in TOOL_DESCRIPTIONS.items())
    cols = ", ".join(f"{c} ({dtypes.get(c, '?')})" for c in columns[:60])
    system = (
        "You are the planner of a data analysis agent. Pick the tools needed to fulfil the user's request.\n"
        f"Tools:\n{tools}\n\nDataset columns: {cols}\n\n"
        "Rules: use ONLY column names from the list; at most 5 steps; order steps logically "
        "(e.g. train_model before explain_model). If the request is a general question that needs no tool, "
        'return {"steps": []}.\n'
        'Reply with JSON only: {"steps": [{"tool": "<name>", "args": {...}}]}'
    )
    resp = get_llm().invoke([("system", system), ("human", message)])
    data = json.loads(resp.content)
    raw = data.get("steps", []) if isinstance(data, dict) else []
    steps = []
    for s in raw:
        if isinstance(s, dict) and s.get("tool") in TOOLS:
            args = s.get("args") if isinstance(s.get("args"), dict) else {}
            if args.get("chart_type") and str(args["chart_type"]).lower() not in CHART_TYPES + [
                    "bar", "scatter", "line", "box", "pie", "violin", "count", "dist", "distribution"]:
                continue
            steps.append({"tool": s["tool"], "args": args})
    return steps[:MAX_STEPS]


def make_plan(message: str, columns: List[str], dtypes: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
    if settings.LLM_ENABLED:
        try:
            plan = _llm_plan(message, columns, dtypes or {})
            if plan:
                logger.info(f"LLM rejasi: {[s['tool'] for s in plan]}")
                return plan
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"LLM rejalashtirish ishlamadi, qoidali rejaga o'tildi: {exc}")
    return heuristic_plan(message, columns)
