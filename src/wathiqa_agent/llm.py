
"""LLM provider factory.

Uses Groq as the chat model provider.
Requires GROQ_API_KEY.
"""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_groq import ChatGroq

from .config import Settings


def get_llm(settings: Settings) -> BaseChatModel:
    """Create and return a Groq chat model."""
    if settings.llm_provider.lower() != "groq":
        raise ValueError(
            f"Unsupported LLM provider: {settings.llm_provider!r}. "
            "Only 'groq' is supported."
        )

    return ChatGroq(
        model=settings.model,
        temperature=0,
        api_key=settings.groq_api_key,
    )
