from conftest import SAMPLE_DIR

from wathiqa_agent.graph import build_graph, resume_review, start_review


def _run(name, settings, graph=None):
    return start_review(
        document_path=str(SAMPLE_DIR / name), settings=settings, graph=graph
    )


def test_clean_invoice_auto_approves(settings):
    result = _run("invoice_clean.txt", settings)
    assert result["status"] == "complete"
    state = result["state"]
    assert state["decision"] == "auto_approved"
    assert state["flags"] == []
    assert "WATHQA Review Report" in state["report"]


def test_vat_mismatch_routes_to_human_and_investigates(settings):
    graph = build_graph(settings)
    result = _run("invoice_vat_mismatch.txt", settings, graph=graph)
    assert result["status"] == "needs_review"
    state = result["state"]
    assert any(f.startswith("vat_mismatch") for f in state["flags"])
    # the investigation loop must have called the VAT calculator
    assert any("vat_calculator" in entry for entry in state["investigation"])
    assert "140.0" in " ".join(state["investigation"])

    resumed = resume_review(result["thread_id"], "approve", settings=settings, graph=graph)
    assert resumed["state"]["decision"] == "approved_after_review"
    assert resumed["state"]["human_feedback"] == "approve"


def test_over_limit_expense_routes_to_human(settings):
    graph = build_graph(settings)
    result = _run("expense_over_limit.txt", settings, graph=graph)
    assert result["status"] == "needs_review"
    assert any(f.startswith("over_limit:meals") for f in result["state"]["flags"])

    resumed = resume_review(result["thread_id"], "reject", settings=settings, graph=graph)
    assert resumed["state"]["decision"] == "rejected_after_review"


def test_missing_fields_route_to_human(settings):
    result = _run("invoice_missing_fields.txt", settings)
    assert result["status"] == "needs_review"
    assert "missing_fields:date" in result["state"]["flags"]


def test_each_review_gets_isolated_thread(settings):
    graph = build_graph(settings)
    first = _run("invoice_clean.txt", settings, graph=graph)
    second = _run("invoice_vat_mismatch.txt", settings, graph=graph)
    assert first["thread_id"] != second["thread_id"]
    assert first["status"] == "complete"
    assert second["status"] == "needs_review"
