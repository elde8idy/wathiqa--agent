import json

from conftest import SAMPLE_DIR

from wathiqa_agent.policies import category_limit, search_policies
from wathiqa_agent.tools import parse_document, policy_search, sum_line_items, vat_calculator


def test_parse_document_reads_text_file():
    text = parse_document.invoke({"path": str(SAMPLE_DIR / "invoice_clean.txt")})
    assert "TechHub Software Ltd" in text


def test_parse_document_missing_file():
    try:
        parse_document.invoke({"path": "/nonexistent/nope.txt"})
    except FileNotFoundError:
        pass
    else:  # StructuredTool may surface errors differently
        raise AssertionError("expected FileNotFoundError")


def test_vat_calculator_math():
    result = json.loads(vat_calculator.invoke({"net_amount": 1000.0, "rate": 0.14}))
    assert result["expected_vat"] == 140.0
    assert result["expected_gross"] == 1140.0


def test_sum_line_items():
    result = json.loads(
        sum_line_items.invoke({"lines": "10x Printer paper @ 100.00\n2x Pens @ 25.00"})
    )
    assert result["line_items_counted"] == 2
    assert result["line_items_total"] == 1050.0


def test_policy_search_finds_meals_limit():
    hits = search_policies("meals spending limit")
    assert hits[0]["id"] == "POL-002"


def test_policy_search_tool_returns_json():
    hits = json.loads(policy_search.invoke({"query": "required document fields"}))
    assert any(h["id"] == "POL-006" for h in hits)


def test_category_limit():
    assert category_limit("meals") == 500.0
    assert category_limit("software") == 25000.0
    assert category_limit("unknown-category") is None
