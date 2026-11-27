"""FastAPI entrypoint.

    uvicorn wathqa.api:app --reload

    GET  /health
    POST /reviews                    {"document_text": "..."} or {"document_path": "..."}
    POST /reviews/{thread_id}/resume {"feedback": "approve" | "reject"}
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .config import get_settings
from .graph import build_graph, resume_review, start_review


class ReviewRequest(BaseModel):
    document_text: str | None = None
    document_path: str | None = None


class ResumeRequest(BaseModel):
    feedback: str


def create_app() -> FastAPI:
    settings = get_settings()
    graph = build_graph(settings)
    app = FastAPI(title="WATHQA", version="0.1.0")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "llm_provider": settings.llm_provider}

    @app.post("/reviews")
    def create_review(req: ReviewRequest) -> dict:
        if not req.document_text and not req.document_path:
            raise HTTPException(400, "provide document_text or document_path")
        try:
            result = start_review(
                document_text=req.document_text,
                document_path=req.document_path,
                settings=settings,
                graph=graph,
            )
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(400, str(exc)) from exc
        return _render(result)

    @app.post("/reviews/{thread_id}/resume")
    def resume(thread_id: str, req: ResumeRequest) -> dict:
        try:
            result = resume_review(thread_id, req.feedback, settings=settings, graph=graph)
        except Exception as exc:  # unknown thread / not paused
            raise HTTPException(404, f"cannot resume thread {thread_id}: {exc}") from exc
        return _render(result)

    return app


def _render(result: dict) -> dict:
    state = result["state"]
    body = {
        "status": result["status"],
        "thread_id": result["thread_id"],
        "doc_type": state.get("doc_type"),
        "extraction": state.get("extraction"),
        "confidence": state.get("confidence"),
        "flags": state.get("flags", []),
        "requires_human": state.get("requires_human"),
        "decision": state.get("decision"),
    }
    if result["status"] == "needs_review":
        interrupts = state.get("__interrupt__", [])
        body["review_request"] = (
            interrupts[0].value if interrupts else None
        )
    else:
        body["report"] = state.get("report")
    return body


app = create_app()
