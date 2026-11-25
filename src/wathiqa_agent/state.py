"""LangGraph state for the review graph."""
from __future__ import annotations

from typing import Any, Optional, TypedDict

class ReviewState(TypedDict,total=False):
    document_path: Optional[str]
    document_text: str

    doc_type: str
    extraction: dict[str,Any]
    confidence: float

    policies: list[dict[str,Any]]
    flags: list[str]

    investigation: list[str]

    requires_human: bool
    human_feedback: Optional[str]


    decision: str
    decision_reasons: list[str]
    report: str