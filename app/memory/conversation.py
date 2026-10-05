"""Suhbat xotirasi: SQLite'dagi tarixni LangChain xabarlariga aylantiradi."""
from typing import List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from app.database.repository import ConversationRepository


class ConversationMemory:
    def __init__(self, session_id: str, window: int = 12) -> None:
        self.session_id = session_id
        self.window = window

    def add_user(self, content: str) -> None:
        ConversationRepository.add_message(self.session_id, "user", content)

    def add_assistant(self, content: str) -> None:
        ConversationRepository.add_message(self.session_id, "assistant", content)

    def messages(self) -> List[BaseMessage]:
        out: List[BaseMessage] = []
        for m in ConversationRepository.get_recent(self.session_id, self.window):
            out.append(HumanMessage(content=m["content"]) if m["role"] == "user"
                       else AIMessage(content=m["content"]))
        return out

    def clear(self) -> None:
        ConversationRepository.clear_history(self.session_id)
