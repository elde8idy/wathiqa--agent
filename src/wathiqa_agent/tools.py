"""LangChain tools used by the graph nodes and the investigation agent loop."""

from __future__ import annotations

import json
import re
from pathlib import Path

from langchain_core.tools import tool

from .policies import search_policies


from pathlib import Path
from langchain_core.tools import tool
from pypdf import PdfReader


@tool
def parse_document(path: str) -> str:
    """Read a document from disk and return its text content.
    Supports .txt, .md, .text, and .pdf files.
    """
    p = Path(path)

    if not p.exists():
        raise FileNotFoundError(f"document not found: {path}")

    suffix = p.suffix.lower()

    if suffix in {".txt", ".md", ".text"}:
        return p.read_text(encoding="utf-8")

    if suffix == ".pdf":
        reader = PdfReader(str(p))
        return "\n".join(
            page.extract_text() or ""
            for page in reader.pages
        )

    raise ValueError(f"unsupported document type: {suffix}")

@tool
def vat_calculator(net_amount :float, rate : float)-> str:
    """Compute VAT and gross total for a net amount and rate.
    Returns JSON with net_amount, rate, expected_vat and expected_gross.
    Use it to double-check the VAT stated on a document.
    """
    expected_vat = round(net_amount*rate,2)
    return json.dumps(
        {
            "net_amount": net_amount,
            "rate": rate,
            "expected_vat": expected_vat,
            "expected_gross": round(net_amount + expected_vat, 2),
        }
    )

@tool
def sum_line_items(lines: str) -> str:
    """Sum line items formatted as '<qty>x <description> @ <unit price>'.
    Returns JSON with per-line totals and the grand total. Use it to check
    whether the stated document total matches the listed items.
    """
    total = 0.0
    counted = 0
    for line in lines.splitlines():
        match = re.search(r"(\d+(?:\.\d+)?)\s*x\s+.*@\s*(\d+(?:\.\d+)?)", line)
        if match:
            qty, price = float(match.group(1)), float(match.group(2))
            total += qty * price
            counted += 1
    return json.dumps({"line_items_counted": counted, "line_items_total": round(total, 2)})

@tool
def policy_search(query: str, top_k: int = 3) -> str:
    """Search the company expense policy for rules relevant to the query.
    Returns JSON list of matching policy rules (id, title, text).
    """
    return json.dumps(search_policies(query, top_k=top_k))
TOOL_MAP = {
    "parse_document": parse_document,
    "vat_calculator": vat_calculator,
    "sum_line_items": sum_line_items,
    "policy_search": policy_search,
}

# Tools the investigation loop is allowed to call (no file access from the loop).
INVESTIGATION_TOOLS = [vat_calculator, sum_line_items, policy_search]