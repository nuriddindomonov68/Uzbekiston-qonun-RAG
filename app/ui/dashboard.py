"""Streamlit dashboard: dataset yuklash, EDA, grafiklar, modellar, agent bilan suhbat va hisobot."""
import os
import sys
import uuid

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from app.agents import run_agent  # noqa: E402
from app.core.config import settings  # noqa: E402
from app.database.connection import init_db  # noqa: E402
from app.database.repository import (ConversationRepository, DatasetRepository,  # noqa: E402
                                     ModelRepository, SessionRepository)
from app.services.datasets import DatasetError, DatasetService  # noqa: E402
from app.services.storage import StorageService  # noqa: E402
from app.tools.registry import CHART_TYPES, ToolContext, run_tool  # noqa: E402
from app.ml.train import MLTrainer  # noqa: E402

st.set_page_config(page_title=settings.APP_NAME, page_icon="📊", layout="wide")
init_db()
StorageService.init_directories()

ALGOS_CLS = ["random_forest", "logistic_regression", "decision_tree", "gradient_boosting",
             "knn", "svm", "naive_bayes", "xgboost", "lightgbm", "catboost"]
ALGOS_REG = ["random_forest", "linear_regression", "ridge", "lasso", "decision_tree",
             "gradient_boosting", "knn", "svm", "xgboost", "lightgbm", "catboost"]


# ------------------------------------------------------------------ sessiya
def new_session() -> None:
    sid = str(uuid.uuid4())
    SessionRepository.create(sid, {})
    st.session_state.session_id = sid
    st.session_state.pop("dataset_id", None)


if "session_id" not in st.session_state:
    new_session()
sid: str = st.session_state.session_id

# ------------------------------------------------------------------ yon panel
with st.sidebar:
    st.title("📊 AI Data Analysis Agent")
    st.caption(f"Sessiya: `{sid[:8]}`")
    if st.button("➕ Yangi sessiya", width="stretch"):
        new_session()
        st.rerun()

    st.divider()
    st.subheader("Dataset yuklash")
    up = st.file_uploader("CSV, Excel yoki JSON", type=["csv", "tsv", "xlsx", "xls", "json"])
    if up is not None:
        key = f"{sid}:{up.name}:{up.size}"
        if st.session_state.get("last_upload") != key:
            try:
                info = DatasetService.register_upload(sid, up.name, up.getvalue())
                st.session_state.last_upload = key
                st.session_state.dataset_id = info["id"]
                st.success(f"Yuklandi: {info['filename']}")
            except DatasetError as exc:
                st.error(str(exc))

    sample = os.path.join(ROOT, "assets", "sample_data.csv")
    if os.path.exists(sample) and st.button("🧪 Namunaviy ma'lumotni yuklash", width="stretch"):
        with open(sample, "rb") as fh:
            info = DatasetService.register_upload(sid, "sample_data.csv", fh.read())
        st.session_state.dataset_id = info["id"]
        st.rerun()

    datasets = DatasetRepository.get_by_session(sid)
    if datasets:
        ids = [d["id"] for d in datasets]
        current = st.session_state.get("dataset_id", ids[0])
        if current not in ids:
            current = ids[0]
        chosen = st.selectbox("Faol dataset", ids, index=ids.index(current),
                              format_func=lambda i: next(d["filename"] for d in datasets if d["id"] == i))
        st.session_state.dataset_id = chosen

    st.divider()
    llm_state = f"yoqilgan (`{settings.LLM_MODEL}`)" if settings.LLM_ENABLED else "o'chirilgan (qoidali rejim)"
    st.caption(f"LLM: {llm_state}")

dataset_id = st.session_state.get("dataset_id")
dataset = DatasetRepository.get(dataset_id) if dataset_id else None
df = DatasetService.get_dataframe(dataset_id) if dataset else None


def ctx() -> ToolContext:
    return ToolContext(df=df, session_id=sid, dataset_id=dataset_id, dataset_path=dataset["filepath"])


def need_dataset() -> bool:
    if df is None:
        st.info("⬅️ Avval chap paneldan dataset yuklang.")
        return False
    return True


tab_chat, tab_eda, tab_chart, tab_model, tab_report = st.tabs(
    ["💬 Agent", "📋 Tahlil (EDA)", "📈 Grafiklar", "🧠 Modellar", "📄 Hisobot"])

# ------------------------------------------------------------------ chat
with tab_chat:
    st.subheader("Agent bilan suhbat")
    if df is None:
        st.info("Dataset yuklang va savol bering. Masalan: *\"umumiy tahlil qil\"*, "
                "*\"korrelyatsiya xaritasini chiz\"*, *\"3 ta klasterga ajrat\"*.")
    for m in ConversationRepository.get_recent(sid, 50):
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
    for p in st.session_state.pop("new_charts", []):
        st.image(p)
    prompt = st.chat_input("Savolingizni yozing...")
    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Agent ishlayapti..."):
                res = run_agent(sid, prompt, dataset_id)
            st.markdown(res["response"])
            for p in res["chart_paths"]:
                st.image(p)
            if res.get("report_path"):
                with open(res["report_path"], "rb") as fh:
                    st.download_button("📥 PDF hisobotni yuklab olish", fh.read(),
                                       file_name="hisobot.pdf", mime="application/pdf")

# ------------------------------------------------------------------ EDA
with tab_eda:
    if need_dataset():
        c1, c2, c3 = st.columns(3)
        c1.metric("Qatorlar", f"{df.shape[0]:,}")
        c2.metric("Ustunlar", df.shape[1])
        c3.metric("Takrorlanuvchi qatorlar", int(df.duplicated().sum()))
        st.markdown("**Dastlabki qatorlar**")
        st.dataframe(df.head(20), width="stretch")

        eda = run_tool("eda", ctx())["result"]
        left, right = st.columns(2)
        with left:
            st.markdown("**Yo'qolgan qiymatlar**")
            if eda["missing_columns"]:
                st.bar_chart(pd.Series({c: v["count"] for c, v in eda["missing_columns"].items()}))
            else:
                st.success("Yo'qolgan qiymat yo'q.")
        with right:
            st.markdown("**Ma'lumot turlari**")
            st.dataframe(pd.DataFrame({"tur": eda["data_types"]}), width="stretch")
        if eda["numerical_summary"]:
            st.markdown("**Raqamli ustunlar statistikasi**")
            st.dataframe(pd.DataFrame(eda["numerical_summary"]).T, width="stretch")
        if eda["categorical_summary"]:
            st.markdown("**Kategorik ustunlar**")
            for col, info in eda["categorical_summary"].items():
                with st.expander(f"{col} ({info['unique_count']} ta noyob qiymat)"):
                    st.bar_chart(pd.Series(info["top_values"]))
        st.markdown("**Anomaliyalar**")
        method = st.radio("Usul", ["iqr", "zscore"], horizontal=True)
        out = run_tool("outliers", ctx(), {"method": method})["result"]["columns"]
        if out:
            st.dataframe(pd.DataFrame(out).T, width="stretch")

# ------------------------------------------------------------------ grafiklar
with tab_chart:
    if need_dataset():
        cols = ["(yo'q)"] + list(df.columns)
        a, b, c, d = st.columns(4)
        ctype = a.selectbox("Grafik turi", CHART_TYPES)
        x = b.selectbox("X ustun", cols)
        y = c.selectbox("Y ustun", cols)
        hue = d.selectbox("Rang (hue)", cols)
        if st.button("Grafik chizish", type="primary"):
            args = {"chart_type": ctype, "x": None if x == cols[0] else x,
                    "y": None if y == cols[0] else y, "hue": None if hue == cols[0] else hue}
            res = run_tool("chart", ctx(), args)
            if res["success"]:
                st.image(res["result"]["chart_path"])
            else:
                st.error(res["error"])
        if st.button("Avtomatik grafiklar"):
            res = run_tool("auto_charts", ctx())
            if res["success"]:
                for p in res["result"]["chart_paths"]:
                    st.image(p)
            else:
                st.error(res["error"])

# ------------------------------------------------------------------ modellar
with tab_model:
    if need_dataset():
        st.subheader("Bashorat modeli")
        target = st.selectbox("Maqsadli ustun (target)", list(df.columns), index=len(df.columns) - 1)
        task = MLTrainer.detect_task_type(df, target)
        st.caption(f"Aniqlangan masala turi: **{'klassifikatsiya' if task == 'classification' else 'regressiya'}**")
        m1, m2 = st.columns(2)
        algo = m1.selectbox("Algoritm", ALGOS_CLS if task == "classification" else ALGOS_REG)
        scaling = m2.selectbox("Masshtablash", ["standard", "minmax", "robust"])
        if st.button("Modelni o'qitish", type="primary"):
            with st.spinner("Model o'qitilmoqda..."):
                res = run_tool("train_model", ctx(), {"target": target, "algorithm": algo, "scaling": scaling})
            if res["success"]:
                st.success("Model o'qitildi.")
                st.json(res["result"]["metrics"])
                exp = run_tool("explain_model", ToolContext(df=df, session_id=sid, dataset_id=dataset_id,
                                                            previous=[res]))
                if exp["success"]:
                    st.markdown("**Muhim ustunlar**")
                    st.bar_chart(pd.Series(exp["result"]["top_features"]))
            else:
                st.error(res["error"])

        st.divider()
        st.subheader("Klasterlash")
        num_cols = list(df.select_dtypes("number").columns)
        feats = st.multiselect("Ustunlar", num_cols, default=num_cols[:3])
        k1, k2 = st.columns(2)
        calgo = k1.selectbox("Klaster algoritmi", ["kmeans", "hierarchical", "dbscan"])
        k = k2.slider("Klasterlar soni", 2, 10, 3)
        if st.button("Klasterlash"):
            res = run_tool("cluster", ctx(), {"features": feats, "algorithm": calgo, "n_clusters": k})
            st.json(res["result"]["metrics"]) if res["success"] else st.error(res["error"])

        st.divider()
        st.subheader("Saqlangan modellar")
        models = ModelRepository.get_by_session(sid)
        if models:
            st.dataframe(pd.DataFrame([{
                "algoritm": m["algorithm"], "turi": m["model_type"], "target": m["target_column"],
                "metrikalar": ", ".join(f"{k}={v:.3f}" for k, v in m["metrics"].items()
                                        if isinstance(v, float))[:90],
                "vaqt": m["trained_at"]} for m in models]), width="stretch")
        else:
            st.caption("Hali model o'qitilmagan.")

# ------------------------------------------------------------------ hisobot
with tab_report:
    if need_dataset():
        st.write("Dataset bo'yicha to'liq PDF hisobot: umumiy ma'lumot, statistika, anomaliyalar, grafiklar, modellar.")
        if st.button("📄 PDF hisobot yaratish", type="primary"):
            with st.spinner("Hisobot tayyorlanmoqda..."):
                res = run_tool("report", ctx())
            if res["success"]:
                with open(res["result"]["report_path"], "rb") as fh:
                    st.download_button("📥 Yuklab olish", fh.read(), file_name="hisobot.pdf",
                                       mime="application/pdf")
            else:
                st.error(res["error"])
