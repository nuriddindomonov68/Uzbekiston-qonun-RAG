"""FastAPI server: sessiyalar, dataset yuklash, agent bilan suhbat, grafik va hisobotlar."""
import os
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.agents import run_agent
from app.analysis import StatsAnalyzer
from app.core.config import settings
from app.database.connection import init_db
from app.database.repository import (ChartRepository, ConversationRepository, DatasetRepository,
                                     ModelRepository, SessionRepository)
from app.schemas import (ChatRequest, ChatResponse, DatasetResponse, ModelResponse,
                         SessionCreate, SessionResponse)
from app.services.datasets import DatasetError, DatasetService
from app.services.storage import StorageService
from app.tools.registry import to_jsonable
from app.utils.logging import get_logger

logger = get_logger("api")


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    StorageService.init_directories()
    logger.info(f"{settings.APP_NAME} API tayyor.")
    yield


app = FastAPI(title=settings.APP_NAME, version="1.0.0",
              description="Ma'lumotlarni tahlil qiluvchi AI agent API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.CORS_ORIGINS.split(",") if o.strip()],
    allow_methods=["*"], allow_headers=["*"],
)


# ----------------------------------------------------------------- yordamchilar
def _session_or_404(session_id: str) -> Dict[str, Any]:
    s = SessionRepository.get(session_id)
    if not s:
        raise HTTPException(404, "Sessiya topilmadi.")
    return s


def _dataset_or_404(dataset_id: str) -> Dict[str, Any]:
    d = DatasetRepository.get(dataset_id)
    if not d:
        raise HTTPException(404, "Dataset topilmadi.")
    return d


def _session_resp(s: Dict[str, Any]) -> SessionResponse:
    return SessionResponse(session_id=s["id"], created_at=str(s["created_at"]),
                           updated_at=str(s["updated_at"]), settings=s["settings"])


def _dataset_resp(d: Dict[str, Any]) -> DatasetResponse:
    return DatasetResponse(dataset_id=d["id"], filename=d["filename"], filepath=d["filepath"],
                           file_type=d["file_type"], row_count=d["row_count"],
                           column_count=d["column_count"], schema_info=d["schema"],
                           uploaded_at=str(d["uploaded_at"]))


def _model_resp(m: Dict[str, Any]) -> ModelResponse:
    return ModelResponse(model_id=m["id"], session_id=m["session_id"], dataset_id=m["dataset_id"],
                         model_type=m["model_type"], algorithm=m["algorithm"],
                         target_column=m["target_column"], features=m["features"],
                         metrics=to_jsonable(m["metrics"]), trained_at=str(m["trained_at"]))


# ----------------------------------------------------------------- umumiy
@app.get("/health", tags=["system"])
def health() -> Dict[str, Any]:
    return {"status": "ok", "app": settings.APP_NAME, "llm_enabled": settings.LLM_ENABLED,
            "llm_model": settings.LLM_MODEL}


# ----------------------------------------------------------------- sessiyalar
@app.post("/sessions", response_model=SessionResponse, tags=["sessions"])
def create_session(body: SessionCreate = SessionCreate()) -> SessionResponse:
    sid = body.session_id or str(uuid.uuid4())
    SessionRepository.create(sid, body.settings or {})
    return _session_resp(_session_or_404(sid))


@app.get("/sessions", response_model=List[SessionResponse], tags=["sessions"])
def list_sessions() -> List[SessionResponse]:
    return [_session_resp(s) for s in SessionRepository.list_all()]


@app.get("/sessions/{session_id}", response_model=SessionResponse, tags=["sessions"])
def get_session(session_id: str) -> SessionResponse:
    return _session_resp(_session_or_404(session_id))


@app.delete("/sessions/{session_id}", tags=["sessions"])
def delete_session(session_id: str) -> Dict[str, str]:
    _session_or_404(session_id)
    SessionRepository.delete(session_id)
    return {"status": "deleted"}


@app.get("/sessions/{session_id}/history", tags=["sessions"])
def get_history(session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    _session_or_404(session_id)
    return ConversationRepository.get_recent(session_id, max(1, min(limit, 500)))


@app.get("/sessions/{session_id}/datasets", response_model=List[DatasetResponse], tags=["datasets"])
def session_datasets(session_id: str) -> List[DatasetResponse]:
    _session_or_404(session_id)
    return [_dataset_resp(d) for d in DatasetRepository.get_by_session(session_id)]


@app.get("/sessions/{session_id}/models", response_model=List[ModelResponse], tags=["models"])
def session_models(session_id: str) -> List[ModelResponse]:
    _session_or_404(session_id)
    return [_model_resp(m) for m in ModelRepository.get_by_session(session_id)]


@app.get("/sessions/{session_id}/charts", tags=["charts"])
def session_charts(session_id: str) -> List[Dict[str, Any]]:
    _session_or_404(session_id)
    return [{"chart_id": c["id"], "title": c["title"], "chart_type": c["chart_type"],
             "url": f"/charts/{c['id']}", "created_at": str(c["created_at"])}
            for c in ChartRepository.get_by_session(session_id)]


# ----------------------------------------------------------------- datasetlar
@app.post("/datasets/upload", response_model=DatasetResponse, tags=["datasets"])
async def upload_dataset(session_id: str = Form(...), file: UploadFile = File(...)) -> DatasetResponse:
    limit = settings.MAX_UPLOAD_MB * 1024 * 1024
    content = await file.read(limit + 1)
    try:
        info = DatasetService.register_upload(session_id, file.filename or "dataset.csv", content)
    except DatasetError as exc:
        raise HTTPException(400, str(exc))
    return _dataset_resp(info)


@app.get("/datasets/{dataset_id}", response_model=DatasetResponse, tags=["datasets"])
def get_dataset(dataset_id: str) -> DatasetResponse:
    return _dataset_resp(_dataset_or_404(dataset_id))


@app.get("/datasets/{dataset_id}/preview", tags=["datasets"])
def preview_dataset(dataset_id: str, rows: int = 10) -> Dict[str, Any]:
    _dataset_or_404(dataset_id)
    df = DatasetService.get_dataframe(dataset_id).head(max(1, min(rows, 200)))
    return to_jsonable({"columns": list(df.columns), "rows": df.where(df.notna(), None).values.tolist()})


@app.get("/datasets/{dataset_id}/eda", tags=["datasets"])
def dataset_eda(dataset_id: str) -> Dict[str, Any]:
    _dataset_or_404(dataset_id)
    return to_jsonable(StatsAnalyzer.run_full_eda(DatasetService.get_dataframe(dataset_id)))


@app.post("/datasets/{dataset_id}/report", tags=["datasets"])
def dataset_report(dataset_id: str) -> FileResponse:
    from app.tools.report import generate_report
    info = _dataset_or_404(dataset_id)
    df = DatasetService.get_dataframe(dataset_id)
    path = generate_report(df, info["session_id"], dataset_id)
    return FileResponse(path, media_type="application/pdf", filename=f"hisobot_{dataset_id[:8]}.pdf")


# ----------------------------------------------------------------- agent
@app.post("/chat", response_model=ChatResponse, tags=["agent"])
def chat(req: ChatRequest) -> ChatResponse:
    if not req.message.strip():
        raise HTTPException(400, "Xabar bo'sh bo'lmasligi kerak.")
    _session_or_404(req.session_id)
    try:
        res = run_agent(req.session_id, req.message.strip())
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Agent xatosi: {exc}")
        raise HTTPException(500, "Agent so'rovni bajara olmadi. Server jurnalini tekshiring.")
    text = res["response"]
    if res.get("report_path") and res.get("dataset_id"):
        text += f"\n\n📄 Hisobotni yuklab olish: POST /datasets/{res['dataset_id']}/report"
    return ChatResponse(response=text, session_id=req.session_id, dataset_id=res["dataset_id"],
                        chart_paths=[f"/charts/{cid}" for cid in res["chart_ids"]] or None)


# ----------------------------------------------------------------- grafiklar
@app.get("/charts/{chart_id}", tags=["charts"])
def get_chart(chart_id: str) -> FileResponse:
    chart = ChartRepository.get(chart_id)
    if not chart or not os.path.exists(chart["filepath"]):
        raise HTTPException(404, "Grafik topilmadi.")
    return FileResponse(chart["filepath"], media_type="image/png")
