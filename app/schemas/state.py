from __future__ import annotations
from typing import List, Dict, Any, Optional
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """Single source-of-truth passed between every node in the LangGraph DAG."""
    session_id:     str
    messages:       List[BaseMessage]
    dataset_id:     Optional[str]
    dataset_path:   Optional[str]
    next_agent:     str
    plan:           List[str]          # JSON-encoded step dicts
    current_step:   int
    tool_outputs:   List[Dict[str, Any]]
    error_context:  Optional[str]
    final_response: Optional[str]
