"""Vosita natijalaridan o'zbekcha qisqa va aniq xulosa (LLM'siz) tuzadi."""
from typing import Any, Dict, List

NAMES = {
    "random_forest": "Random Forest", "xgboost": "XGBoost", "lightgbm": "LightGBM", "catboost": "CatBoost",
    "gradient_boosting": "Gradient Boosting", "logistic_regression": "Logistic Regression",
    "linear_regression": "Linear Regression", "decision_tree": "Decision Tree", "naive_bayes": "Naive Bayes",
    "knn": "KNN", "svm": "SVM", "ridge": "Ridge", "lasso": "Lasso", "kmeans": "K-Means",
    "dbscan": "DBSCAN", "hierarchical": "Hierarchical",
}


def _n(v: Any, digits: int = 3) -> str:
    if v is None:
        return "-"
    if isinstance(v, int):
        return str(v)
    return f"{v:.{digits}f}"


def _top_corr(corr: Dict[str, Dict[str, float]]):
    best, pair = 0.0, None
    cols = list(corr)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            v = corr[a].get(b)
            if v is not None and abs(v) > abs(best):
                best, pair = v, (a, b)
    return pair, best


def summarize_one(out: Dict[str, Any]) -> str:
    tool = out["tool"]
    if not out.get("success"):
        return f"⚠️ **{tool}** bajarilmadi: {out.get('error', 'noma`lum xato')}"
    r = out["result"]

    if tool == "eda":
        lines = [f"📊 **Umumiy tahlil:** {r['rows']} qator, {r['columns']} ustun."]
        miss = r["missing_columns"]
        if miss:
            top = sorted(miss.items(), key=lambda kv: kv[1]["count"], reverse=True)[:5]
            lines.append("Yo'qolgan qiymatlar: " + ", ".join(
                f"{c} ({v['count']} ta, {v['percentage']}%)" for c, v in top) + ".")
        else:
            lines.append("Yo'qolgan qiymatlar yo'q.")
        lines.append(f"Takrorlanuvchi qatorlar: {r['duplicate_rows']}.")
        num = r["numerical_summary"]
        if num:
            lines.append("Raqamli ustunlar (o'rtacha): " + ", ".join(
                f"{c}={_n(s.get('mean'), 2)}" for c, s in list(num.items())[:6]) + ".")
        pair, val = _top_corr(r["correlation_matrix"]) if r["correlation_matrix"] else (None, 0)
        if pair:
            lines.append(f"Eng kuchli korrelyatsiya: {pair[0]} va {pair[1]} (r={val:.2f}).")
        return "\n".join(lines)

    if tool == "outliers":
        cw = r["columns_with_outliers"]
        if not cw:
            return f"🚨 **Anomaliyalar ({r['method'].upper()}):** topilmadi."
        top = sorted(cw.items(), key=lambda kv: kv[1]["outlier_count"], reverse=True)[:6]
        return f"🚨 **Anomaliyalar ({r['method'].upper()}):** " + ", ".join(
            f"{c} - {v['outlier_count']} ta ({v['outlier_percentage']}%)" for c, v in top) + "."

    if tool == "chart":
        cols = " / ".join(str(v) for v in (r.get("x"), r.get("y")) if v)
        return f"📈 **Grafik yaratildi:** {r['chart_type']}" + (f" ({cols})" if cols else "") + "."

    if tool == "auto_charts":
        return f"📈 **{len(r['charts'])} ta grafik yaratildi** (korrelyatsiya va taqsimotlar)."

    if tool == "train_model":
        m = r["metrics"]
        name = NAMES.get(r["algorithm"], r["algorithm"])
        if r["task_type"] == "classification":
            main = f"accuracy={_n(m.get('accuracy'))}, f1={_n(m.get('f1_weighted'))}"
            if "roc_auc" in m:
                main += f", ROC-AUC={_n(m['roc_auc'])}"
        else:
            main = f"R²={_n(m.get('r2'))}, RMSE={_n(m.get('rmse'))}, MAE={_n(m.get('mae'))}"
        cv = f", kross-validatsiya={_n(m['cv_mean_score'])}" if "cv_mean_score" in m else ""
        ex = f" Model o'qitishda e'tiborga olinmagan ustunlar: {', '.join(r['excluded_features'])}." \
            if r.get("excluded_features") else ""
        task = "klassifikatsiya" if r["task_type"] == "classification" else "regressiya"
        return f"🧠 **Model:** {name} ({task}), maqsad: **{r['target_column']}**. {main}{cv}.{ex}"

    if tool == "cluster":
        sil = r["metrics"].get("silhouette_score")
        name = NAMES.get(r["algorithm"], r["algorithm"])
        return f"🧩 **Klasterlash:** {name}, silhouette={_n(sil)} ({len(r['features'])} ta ustun)."

    if tool == "explain_model":
        top = list(r["top_features"].items())[:5]
        return "🔍 **Eng muhim ustunlar:** " + ", ".join(f"{k} ({v:.3f})" for k, v in top) + "."

    if tool == "report":
        return "📄 **PDF hisobot tayyor.**"
    return f"✅ {tool} bajarildi."


def summarize_outputs(outputs: List[Dict[str, Any]]) -> str:
    return "\n\n".join(summarize_one(o) for o in outputs)


HELP_TEXT = (
    "Men ma'lumotlar tahlili agentiman. Avval dataset (CSV, Excel yoki JSON) yuklang, so'ng quyidagilarni so'rang:\n"
    "- **Tahlil:** \"umumiy tahlil qil\", \"yo'qolgan qiymatlar\"\n"
    "- **Anomaliya:** \"anomaliyalarni top\"\n"
    "- **Grafik:** \"korrelyatsiya xaritasini chiz\", \"yosh bo'yicha gistogramma\"\n"
    "- **Model:** \"sotib_oldi ni bashorat qiladigan model o'qit\"\n"
    "- **Klaster:** \"3 ta klasterga ajrat\"\n"
    "- **Izoh:** \"qaysi ustunlar muhim?\"\n"
    "- **Hisobot:** \"PDF hisobot yarat\""
)
