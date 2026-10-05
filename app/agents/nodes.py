"""LangGraph tugunlari: planner -> executor (sikl) -> responder."""
import json
from typing import Any, Dict, List

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.agents.planner import has_tool_intent, make_plan
from app.agents.summary import HELP_TEXT, summarize_outputs
from app.core.config import settings
from app.schemas.state import AgentState
from app.tools.loader import load_dataframe
from app.tools.registry import ToolContext, run_tool
from app.utils.logging import get_logger

logger = get_logger("agents.nodes")


def _last_user_text(state: AgentState) -> str:
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            return str(m.content)
    return ""


def planner_node(state: AgentState) -> Dict[str, Any]:
    message = _last_user_text(state)
    path = state.get("dataset_path")
    if not path:
        # Dataset yo'q: vosita kerak bo'lgan so'rovga yuklashni so'raymiz
        err = "no_dataset" if has_tool_intent(message) else None
        return {"plan": [], "current_step": 0, "next_agent": "responder", "error_context": err}
    try:
        df = load_dataframe(path)
    except Exception as exc:  # noqa: BLE001
        return {"plan": [], "current_step": 0, "next_agent": "responder",
                "error_context": f"Datasetni o'qib bo'lmadi: {exc}"}
    columns = [str(c) for c in df.columns]
    dtypes = {str(c): str(t) for c, t in df.dtypes.items()}
    steps = make_plan(message, columns, dtypes)
    return {"plan": [json.dumps(s, ensure_ascii=False) for s in steps], "current_step": 0,
            "tool_outputs": [], "next_agent": "executor" if steps else "responder", "error_context": None}


def executor_node(state: AgentState) -> Dict[str, Any]:
    idx = state["current_step"]
    step = json.loads(state["plan"][idx])
    df = load_dataframe(state["dataset_path"])
    ctx = ToolContext(df=df, session_id=state["session_id"], dataset_id=state["dataset_id"] or "",
                      dataset_path=state["dataset_path"], previous=list(state.get("tool_outputs", [])))
    out = run_tool(step["tool"], ctx, step.get("args"))
    logger.info(f"Qadam {idx + 1}/{len(state['plan'])}: {step['tool']} -> "
                f"{'OK' if out['success'] else 'XATO'}")
    outputs = list(state.get("tool_outputs", [])) + [out]
    err = state.get("error_context")
    if not out["success"]:
        err = (err + "; " if err else "") + f"{step['tool']}: {out['error']}"
    nxt = idx + 1
    return {"tool_outputs": outputs, "current_step": nxt, "error_context": err,
            "next_agent": "executor" if nxt < len(state["plan"]) else "responder"}


def _dataset_brief(path: str) -> str:
    try:
        df = load_dataframe(path)
        cols = ", ".join(f"{c} ({t})" for c, t in list(df.dtypes.astype(str).items())[:40])
        return f"Dataset: {df.shape[0]} qator, {df.shape[1]} ustun. Ustunlar: {cols}"
    except Exception:  # noqa: BLE001
        return "Dataset o'qib bo'lmadi."


def _chat_llm_reply(state: AgentState, facts: str) -> str:
    from app.core.llm import get_chat_llm

    system = ("Sen ma'lumotlar tahlili bo'yicha yordamchisan. Faqat o'zbek tilida (lotin) qisqa, aniq javob ber. "
              "Faqat berilgan faktlarga tayan, raqamlarni o'zgartirma, o'zingdan to'qima.\n\n" + facts)
    msgs = [SystemMessage(content=system)] + list(state["messages"])
    return str(get_chat_llm().invoke(msgs).content).strip()


def responder_node(state: AgentState) -> Dict[str, Any]:
    outputs: List[Dict[str, Any]] = state.get("tool_outputs", [])

    if outputs:
        summary = summarize_outputs(outputs)
        text = summary
        if settings.LLM_ENABLED:
            try:
                facts = (f"Foydalanuvchi so'rovi bo'yicha bajarilgan ishlar natijasi:\n{summary}\n\n"
                         "Shu natijani foydalanuvchiga ravon, qisqa (5-8 gap) tushuntirib ber va oxirida "
                         "bitta foydali keyingi qadam taklif qil.")
                reply = _chat_llm_reply(state, facts)
                if reply:
                    text = f"{reply}\n\n---\n{summary}"
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"LLM javobi olinmadi, tayyor xulosa ishlatildi: {exc}")
        return {"final_response": text, "messages": state["messages"] + [AIMessage(content=text)]}

    if state.get("error_context") == "no_dataset":
        text = "Avval tahlil qilinadigan faylni (CSV, Excel yoki JSON) yuklang, keyin so'rovingizni bajaraman."
    elif state.get("error_context"):
        text = f"⚠️ {state['error_context']}"
    else:
        text = HELP_TEXT
        if settings.LLM_ENABLED:
            try:
                brief = _dataset_brief(state["dataset_path"]) if state.get("dataset_path") else \
                    "Hozircha dataset yuklanmagan."
                reply = _chat_llm_reply(state, brief)
                if reply:
                    text = reply
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"LLM suhbat javobi olinmadi: {exc}")
    return {"final_response": text, "messages": state["messages"] + [AIMessage(content=text)]}
