"""LangGraph wiring for the WATHIQA review graph.

    intake -> classify -> extract -> retrieve_policies -> validate
    validate --(flags)--> investigate --> gate
    validate --(clean)--> gate
    gate --(needs human)--> human_review --> finalize
    gate --(clean & confident)--> finalize --> END
"""

from __future__ import annotations

import uuid

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from .config import Settings, get_settings
from .llm import get_llm
from .nodes import make_nodes, route_after_gate, route_after_validate
from .state import ReviewState

from langgraph.checkpoint.memory import InMemorySaver



def build_graph(settings: Settings | None = None, llm=None, checkpointer=None):
    """Compile the review graph. A checkpointer is required for human-in-the-loop."""
    settings = settings or get_settings()
    llm = llm or get_llm(settings)
    nodes = make_nodes(llm, settings)

    builder = StateGraph(ReviewState)
    for name, fn in nodes.items():
        builder.add_node(name, fn)

    builder.add_edge(START, "intake")
    builder.add_edge("intake", "classify")
    builder.add_edge("classify", "extract")
    builder.add_edge("extract", "retrieve_policies")
    builder.add_edge("retrieve_policies", "validate")
    builder.add_conditional_edges(
        "validate", route_after_validate, {"investigate": "investigate", "gate": "gate"}
    )
    builder.add_edge("investigate", "gate")
    builder.add_conditional_edges(
        "gate", route_after_gate, {"human_review": "human_review", "finalize": "finalize"}
    )
    builder.add_edge("human_review", "finalize")
    builder.add_edge("finalize", END)

    return builder.compile(checkpointer=checkpointer or InMemorySaver())


def start_review(
    document_text: str | None = None,
    document_path: str | None = None,
    settings: Settings | None = None,
    graph=None,
    thread_id: str | None = None,
) -> dict:
    """Run the graph until it finishes or pauses for human review.

    Returns a dict with status "complete" or "needs_review", the thread_id
    (needed to resume), and the resulting state.
    """
    graph = graph or build_graph(settings)
    thread_id = thread_id or uuid.uuid4().hex
    config = {"configurable": {"thread_id": thread_id}}
    initial: ReviewState = {}
    if document_text is not None:
        initial["document_text"] = document_text
    if document_path is not None:
        initial["document_path"] = document_path

    result = graph.invoke(initial, config)
    if "__interrupt__" in result:
        return {"status": "needs_review", "thread_id": thread_id, "state": result}
    return {"status": "complete", "thread_id": thread_id, "state": result}


def resume_review(thread_id: str, feedback: str, settings: Settings | None = None, graph=None) -> dict:
    """Resume a paused review with the human's feedback (e.g. 'approve' / 'reject')."""
    graph = graph or build_graph(settings)
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke(Command(resume=feedback), config)
    return {"status": "complete", "thread_id": thread_id, "state": result}
