"""PDF hisobot yaratish (fpdf2)."""
import os
import re
import uuid
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd
from fpdf import FPDF

from app.analysis import OutlierDetector, StatsAnalyzer
from app.database.repository import DatasetRepository, ModelRepository
from app.services.storage import StorageService

_REPL = {"\u2018": "'", "\u2019": "'", "\u02bb": "'", "\u02bc": "'", "\u201c": '"', "\u201d": '"',
         "\u2013": "-", "\u2014": "-", "\u2026": "...", "\u2192": "->"}


def _safe(text) -> str:
    """Standart PDF shriftlari faqat Latin-1 ni qo'llaydi: o', g' kabi belgilarni moslaymiz."""
    s = str(text)
    for k, v in _REPL.items():
        s = s.replace(k, v)
    s = re.sub(r"[^\x00-\xff]", "?", s)
    return s


class _PDF(FPDF):
    def footer(self) -> None:
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(130)
        self.cell(0, 8, _safe(f"AI Data Analysis Agent  |  {self.page_no()}-bet"), align="C")


def _h1(pdf: FPDF, text: str) -> None:
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(30, 60, 120)
    pdf.cell(0, 9, _safe(text), new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0)


def _p(pdf: FPDF, text: str, size: int = 10) -> None:
    pdf.set_font("Helvetica", "", size)
    pdf.multi_cell(0, 5.5, _safe(text), new_x="LMARGIN", new_y="NEXT")


def _table(pdf: FPDF, header: List[str], rows: List[List[str]], widths: List[float]) -> None:
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(230, 236, 248)
    for h, w in zip(header, widths):
        pdf.cell(w, 7, _safe(h)[:22], border=1, fill=True)
    pdf.ln()
    pdf.set_font("Helvetica", "", 9)
    for row in rows:
        for v, w in zip(row, widths):
            pdf.cell(w, 6.5, _safe(v)[:22], border=1)
        pdf.ln()


def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "-"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return f"{float(v):.3g}" if abs(float(v)) < 1e5 else f"{float(v):.3e}"


def generate_report(df: pd.DataFrame, session_id: str, dataset_id: str) -> str:
    from app.tools.registry import ToolContext, tool_auto_charts

    info = DatasetRepository.get(dataset_id) or {"filename": dataset_id}
    eda = StatsAnalyzer.run_full_eda(df)

    pdf = _PDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(0, 12, "Ma'lumotlar tahlili hisoboti", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(110)
    pdf.cell(0, 6, _safe(f"Fayl: {info['filename']}   |   Sana: {datetime.now():%Y-%m-%d %H:%M}"),
             new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0)

    # 1. Umumiy
    _h1(pdf, "1. Umumiy ma'lumot")
    miss_total = sum(v["count"] for v in eda["missing_values"].values())
    _p(pdf, f"Qatorlar soni: {eda['basic_info']['row_count']}\n"
            f"Ustunlar soni: {eda['basic_info']['column_count']}\n"
            f"Takrorlanuvchi qatorlar: {eda['duplicate_rows']}\n"
            f"Jami yo'qolgan qiymatlar: {miss_total}")

    # 2. Ustunlar
    _h1(pdf, "2. Ustunlar va yo'qolgan qiymatlar")
    rows = [[c, eda["data_types"][c], str(eda["missing_values"][c]["count"]),
             f"{eda['missing_values'][c]['percentage']}%"] for c in df.columns[:40]]
    _table(pdf, ["Ustun", "Turi", "Yo'qolgan", "Foiz"], rows, [60, 45, 40, 40])
    if df.shape[1] > 40:
        _p(pdf, f"... va yana {df.shape[1] - 40} ta ustun.", 8)

    # 3. Statistika
    num = eda["numerical_summary"]
    if num:
        _h1(pdf, "3. Raqamli ustunlar statistikasi")
        rows = [[c, _fmt(s.get("mean")), _fmt(s.get("std")), _fmt(s.get("min")),
                 _fmt(s.get("50%")), _fmt(s.get("max"))] for c, s in list(num.items())[:25]]
        _table(pdf, ["Ustun", "O'rtacha", "Std", "Min", "Mediana", "Max"], rows, [50, 28, 28, 28, 28, 28])

    # 4. Anomaliyalar
    outs = {c: r for c, r in OutlierDetector.get_all_outliers(df, "iqr").items() if r["outlier_count"] > 0}
    _h1(pdf, "4. Anomaliyalar (IQR usuli)")
    if outs:
        rows = [[c, str(r["outlier_count"]), f"{r['outlier_percentage']}%",
                 _fmt(r["lower_bound"]), _fmt(r["upper_bound"])] for c, r in list(outs.items())[:25]]
        _table(pdf, ["Ustun", "Soni", "Foiz", "Pastki", "Yuqori"], rows, [50, 30, 30, 35, 35])
    else:
        _p(pdf, "Anomaliyalar topilmadi.")

    # 5. Grafiklar
    ctx = ToolContext(df=df, session_id=session_id, dataset_id=dataset_id)
    try:
        chart_paths = tool_auto_charts(ctx, max_charts=4)["chart_paths"]
    except Exception:  # noqa: BLE001
        chart_paths = []
    if chart_paths:
        pdf.add_page()
        _h1(pdf, "5. Grafiklar")
        for cp in chart_paths:
            if pdf.get_y() > 190:
                pdf.add_page()
            pdf.image(cp, w=150)
            pdf.ln(4)

    # 6. Modellar
    models = [m for m in ModelRepository.get_by_session(session_id) if m["dataset_id"] == dataset_id]
    if models:
        _h1(pdf, "6. O'qitilgan modellar")
        for m in models[:5]:
            keys = ("accuracy", "f1_weighted", "roc_auc", "r2", "rmse", "mae", "silhouette_score", "cv_mean_score")
            met = ", ".join(f"{k}={_fmt(m['metrics'][k])}" for k in keys if k in m["metrics"])
            _p(pdf, f"- {m['algorithm']} ({m['model_type']}), target: {m['target_column'] or '-'}\n  {met}", 9)

    out = StorageService.get_report_path(session_id, f"report_{uuid.uuid4().hex[:8]}.pdf")
    pdf.output(out)
    return os.path.abspath(out)
