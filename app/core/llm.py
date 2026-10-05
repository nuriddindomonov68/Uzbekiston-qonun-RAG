from langchain_ollama import ChatOllama
from app.core.config import settings
from app.utils.logging import get_logger

logger = get_logger("core.llm")


def get_llm() -> ChatOllama:
    """Structured JSON-output LLM for agent reasoning."""
    logger.info(f"Connecting to Ollama model '{settings.LLM_MODEL}' ...")
    return ChatOllama(
        base_url=settings.OLLAMA_BASE_URL,
        model=settings.LLM_MODEL,
        temperature=0.0,
        format="json",
        num_predict=4096,
    )


def get_chat_llm() -> ChatOllama:
    """Free-form conversational LLM."""
    return ChatOllama(
        base_url=settings.OLLAMA_BASE_URL,
        model=settings.LLM_MODEL,
        temperature=0.7,
        num_predict=4096,
    )
