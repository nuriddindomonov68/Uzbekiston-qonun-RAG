"""LangGraph oqimi: planner -> executor (reja tugaguncha) -> responder -> END."""
from functools import lru_cache

from langgraph.graph import END, StateGraph

from app.agents.nodes import executor_node, planner_node, responder_node
from app.schemas.state import AgentState


def _after_planner(state: AgentState) -> str:
    return "executor" if state.get("plan") else "responder"


def _after_executor(state: AgentState) -> str:
    return "executor" if state["current_step"] < len(state["plan"]) else "responder"


@lru_cache(maxsize=1)
def build_graph():
    g = StateGraph(AgentState)
    g.add_node("planner", planner_node)
    g.add_node("executor", executor_node)
    g.add_node("responder", responder_node)
    g.set_entry_point("planner")
    g.add_conditional_edges("planner", _after_planner, {"executor": "executor", "responder": "responder"})
    g.add_conditional_edges("executor", _after_executor, {"executor": "executor", "responder": "responder"})
    g.add_edge("responder", END)
    return g.compile()
