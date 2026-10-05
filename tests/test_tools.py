import pytest

from app.services.datasets import DatasetError, DatasetService
from app.tools import ToolContext, run_tool
from app.tools.registry import TOOLS


@pytest.fixture
def ctx(df, csv_bytes):
    info = DatasetService.register_upload("s", "data.csv", csv_bytes)
    return ToolContext(df=DatasetService.get_dataframe(info["id"]), session_id="s", dataset_id=info["id"])


def test_all_tools_registered():
    assert set(TOOLS) == {"eda", "outliers", "chart", "auto_charts", "train_model", "cluster",
                          "explain_model", "report"}


def test_upload_rejects_bad_files():
    with pytest.raises(DatasetError):
        DatasetService.register_upload("s", "virus.exe", b"x")
    with pytest.raises(DatasetError):
        DatasetService.register_upload("s", "empty.csv", b"")


def test_upload_sanitizes_filename(csv_bytes):
    info = DatasetService.register_upload("s", "../../etc/passwd.csv", csv_bytes)
    assert info["filename"] == "passwd.csv" and "etc" not in info["filepath"]


def test_eda_chart_train_explain_chain(ctx):
    assert run_tool("eda", ctx)["success"]
    ch = run_tool("chart", ctx, {"chart_type": "histogram", "x": "yosh"})
    assert ch["success"]
    tr = run_tool("train_model", ctx, {"target": "sotib_oldi"})
    assert tr["success"]
    ctx.previous = [tr]
    ex = run_tool("explain_model", ctx)
    assert ex["success"] and ex["result"]["top_features"]


def test_tool_errors_are_returned_not_raised(ctx):
    assert not run_tool("chart", ctx, {"chart_type": "histogram", "x": "yoq_ustun"})["success"]
    assert not run_tool("train_model", ctx, {"target": "yoq"})["success"]
    assert not run_tool("nomalum", ctx)["success"]
    assert not run_tool("explain_model", ctx)["success"]  # model yo'q


def test_report_pdf(ctx):
    run_tool("train_model", ctx, {"target": "sotib_oldi"})
    res = run_tool("report", ctx)
    assert res["success"]
    with open(res["result"]["report_path"], "rb") as fh:
        assert fh.read(4) == b"%PDF"
