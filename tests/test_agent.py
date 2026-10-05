import pytest

from app.agents import run_agent
from app.agents.planner import heuristic_plan
from app.services.datasets import DatasetService

COLS = ["yosh", "daromad", "shahar", "sotib_oldi"]


def tools(msg):
    return [s["tool"] for s in heuristic_plan(msg, COLS)]


def test_heuristic_planner():
    assert tools("umumiy tahlil qil") == ["eda"]
    assert tools("anomaliyalarni top") == ["outliers"]
    assert tools("korrelyatsiya xaritasini chiz") == ["chart"]
    assert tools("grafik chiz") == ["auto_charts"]
    assert tools("sotib_oldi ni bashorat qiladigan model o'qit") == ["train_model"]
    assert tools("model o'qit va qaysi ustunlar muhimligini ko'rsat") == ["train_model", "explain_model"]
    assert tools("3 ta klasterga ajrat") == ["cluster"]
    assert tools("PDF hisobot yarat") == ["report"]
    assert tools("salom") == []


def test_planner_extracts_arguments():
    step = heuristic_plan("yosh va daromad bo'yicha scatter chiz", COLS)[0]
    assert step["args"] == {"chart_type": "scatter_plot", "x": "yosh", "y": "daromad"}
    step = heuristic_plan("sotib_oldi ni bashorat qil xgboost bilan", COLS)[0]
    assert step["args"]["target"] == "sotib_oldi" and step["args"]["algorithm"] == "xgboost"


def test_agent_without_dataset():
    res = run_agent("a1", "umumiy tahlil qil")
    assert "yuklang" in res["response"]
    assert "yuklang" in run_agent("a1", "salom")["response"]  # yordam matni


def test_agent_full_flow(csv_bytes):
    DatasetService.register_upload("a2", "data.csv", csv_bytes)
    res = run_agent("a2", "avval umumiy tahlil qil, so'ng sotib_oldi ni bashorat qiladigan model o'qit "
                          "va qaysi ustunlar muhimligini ayt")
    assert [o["tool"] for o in res["tool_outputs"]] == ["eda", "train_model", "explain_model"]
    assert all(o["success"] for o in res["tool_outputs"])
    assert "Model" in res["response"] and "muhim" in res["response"]
    res = run_agent("a2", "korrelyatsiya xaritasini chiz")
    assert len(res["chart_paths"]) == 1 and res["chart_ids"]


def test_agent_reports_tool_error(csv_bytes):
    DatasetService.register_upload("a3", "data.csv", csv_bytes)
    res = run_agent("a3", "qaysi ustunlar muhim?")  # model yo'q
    assert "⚠️" in res["response"]
