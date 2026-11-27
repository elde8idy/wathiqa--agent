"""Evaluation harness: run the graph over labelled documents and score it.

Gold labels live in data/gold/labels.json. Metrics:
- doc_type_accuracy   : predicted document type == gold type
- field_accuracy      : exact match per extracted field (vendor, amount,
                        currency, date, category), averaged over docs
- routing_accuracy    : graph routed to auto-approve vs human review as expected
"""

from __future__ import annotations

import json
from pathlib import Path

from .config import Settings, get_settings
from .graph import build_graph, start_review

REPO_ROOT = Path(__file__).resolve().parent.parent
GOLD_PATH = REPO_ROOT / "data" / "gold" / "labels.json"
SAMPLE_DIR = REPO_ROOT / "data" / "sample_docs"
REPORT_PATH = REPO_ROOT / "eval_report.json"

_EVALUATED_FIELDS = ("vendor", "amount", "currency", "date", "category")


def run_eval(settings: Settings | None = None, report_path: Path = REPORT_PATH) -> dict:
    settings = settings or get_settings()
    gold = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
    graph = build_graph(settings)

    per_doc = []
    for entry in gold:
        doc_path = SAMPLE_DIR / entry["file"]
        result = start_review(
            document_path=str(doc_path), settings=settings, graph=graph
        )
        state = result["state"]
        extraction = state.get("extraction", {})

        field_scores = {}
        for field in _EVALUATED_FIELDS:
            expected = entry["fields"].get(field)
            actual = extraction.get(field)
            if isinstance(expected, float) and isinstance(actual, (int, float)):
                field_scores[field] = abs(actual - expected) < 0.01
            else:
                field_scores[field] = actual == expected

        expected_route = entry["expected_route"]
        actual_route = "human" if result["status"] == "needs_review" else "auto"

        per_doc.append(
            {
                "file": entry["file"],
                "doc_type_ok": state.get("doc_type") == entry["doc_type"],
                "field_scores": field_scores,
                "field_accuracy": round(
                    sum(field_scores.values()) / len(field_scores), 4
                ),
                "route_ok": actual_route == expected_route,
                "expected_route": expected_route,
                "actual_route": actual_route,
            }
        )

    n = len(per_doc)
    metrics = {
        "documents": n,
        "doc_type_accuracy": round(sum(d["doc_type_ok"] for d in per_doc) / n, 4),
        "field_accuracy": round(sum(d["field_accuracy"] for d in per_doc) / n, 4),
        "routing_accuracy": round(sum(d["route_ok"] for d in per_doc) / n, 4),
    }
    report = {"provider": settings.llm_provider, "metrics": metrics, "per_doc": per_doc}
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
