"""Agent kirish nuqtasi: run_agent()."""
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage

from app.agents.graph import build_graph
from app.database.repository import DatasetRepository, SessionRepository
from app.memory import ConversationMemory
from app.utils.logging import get_logger

logger = get_logger("agents")


def _collect_charts(outputs: List[Dict[str, Any]]) -> List[str]:
    paths: List[str] = []
    for o in outputs:
        if not o.get("success"):
            continue
        r = o["result"]
        if o["tool"] == "chart":
            paths.append(r["chart_path"])
        elif o["tool"] == "auto_charts":
            paths.extend(r["chart_paths"])
    return paths


def _collect_chart_ids(outputs: List[Dict[str, Any]]) -> List[str]:
    ids: List[str] = []
    for o in outputs:
        if not o.get("success"):
            continue
        r = o["result"]
        if o["tool"] == "chart":
            ids.append(r["chart_id"])
        elif o["tool"] == "auto_charts":
            ids.extend(c["chart_id"] for c in r["charts"])
    return ids


def run_agent(session_id: str, message: str, dataset_id: Optional[str] = None) -> Dict[str, Any]:
    """Foydalanuvchi xabarini agentga beradi va natijani qaytaradi."""
    SessionRepository.create(session_id, {})
    dataset = DatasetRepository.get(dataset_id) if dataset_id else None
    if dataset is None:
        datasets = DatasetRepository.get_by_session(session_id)
        dataset = datasets[0] if datasets else None

    memory = ConversationMemory(session_id)
    history = memory.messages()
    memory.add_user(message)

    state = {
        "session_id": session_id,
        "messages": history + [HumanMessage(content=message)],
        "dataset_id": dataset["id"] if dataset else None,
        "dataset_path": dataset["filepath"] if dataset else None,
        "next_agent": "planner", "plan": [], "current_step": 0,
        "tool_outputs": [], "error_context": None, "final_response": None,
    }
    result = build_graph().invoke(state)
    response = result.get("final_response") or "Javob tayyorlab bo'lmadi."
    memory.add_assistant(response)
    SessionRepository.touch(session_id)
    outputs = result.get("tool_outputs", [])
    return {
        "response": response,
        "session_id": session_id,
        "dataset_id": state["dataset_id"],
        "chart_paths": _collect_charts(outputs),
        "chart_ids": _collect_chart_ids(outputs),
        "tool_outputs": outputs,
        "report_path": next((o["result"]["report_path"] for o in outputs
                             if o["tool"] == "report" and o.get("success")), None),
    }


__all__ = ["run_agent", "build_graph"]
