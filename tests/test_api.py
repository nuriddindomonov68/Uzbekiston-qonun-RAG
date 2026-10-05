import pytest
from fastapi.testclient import TestClient

from app.api.server import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_full_api_flow(client, csv_bytes):
    sid = client.post("/sessions").json()["session_id"]

    r = client.post("/datasets/upload", data={"session_id": sid},
                    files={"file": ("data.csv", csv_bytes, "text/csv")})
    assert r.status_code == 200, r.text
    ds = r.json()
    assert ds["row_count"] == 200 and ds["column_count"] == 4

    assert client.get(f"/datasets/{ds['dataset_id']}/eda").status_code == 200
    assert len(client.get(f"/datasets/{ds['dataset_id']}/preview?rows=3").json()["rows"]) == 3

    chat = client.post("/chat", json={"session_id": sid, "message": "korrelyatsiya xaritasini chiz"}).json()
    assert chat["chart_paths"]
    img = client.get(chat["chart_paths"][0])
    assert img.status_code == 200 and img.headers["content-type"] == "image/png"

    client.post("/chat", json={"session_id": sid, "message": "sotib_oldi ni bashorat qiladigan model o'qit"})
    models = client.get(f"/sessions/{sid}/models").json()
    assert models and models[0]["target_column"] == "sotib_oldi"

    assert len(client.get(f"/sessions/{sid}/history").json()) == 4
    assert client.get(f"/sessions/{sid}/charts").json()

    pdf = client.post(f"/datasets/{ds['dataset_id']}/report")
    assert pdf.status_code == 200 and pdf.content[:4] == b"%PDF"

    assert client.delete(f"/sessions/{sid}").status_code == 200
    assert client.get(f"/sessions/{sid}").status_code == 404


def test_api_errors(client):
    assert client.get("/sessions/yoq").status_code == 404
    assert client.get("/charts/yoq").status_code == 404
    assert client.post("/chat", json={"session_id": "yoq", "message": "salom"}).status_code == 404
    sid = client.post("/sessions").json()["session_id"]
    assert client.post("/chat", json={"session_id": sid, "message": "  "}).status_code == 400
    bad = client.post("/datasets/upload", data={"session_id": sid},
                      files={"file": ("a.exe", b"x", "application/octet-stream")})
    assert bad.status_code == 400
