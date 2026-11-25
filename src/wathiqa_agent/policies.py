"""Company expense policy corpus and a small keyword retriever.

The retriever is deliberately dependency-free (token-overlap scoring with an
IDF-like rarity bonus) so it works fully offline. Swapping in embeddings later
only means replacing `search_policies`.
"""

from __future__ import annotations

import math
import re
from collections import Counter

POLICIES: list[dict] = [
    {
        "id": "POL-001",
        "title": "Standard VAT rate",
        "text": "All taxable purchases are subject to the standard VAT rate of 14%. "
        "The VAT amount on a document must equal net amount x 0.14 (tolerance: 1 unit).",
        "category": "tax",
    },
    {
        "id": "POL-002",
        "title": "Meals spending limit",
        "text": "Meals and entertainment expenses are limited to 500 per claim. "
        "Amounts above this limit require manager approval.",
        "category": "meals",
    },
    {
        "id": "POL-003",
        "title": "Travel spending limit",
        "text": "Travel expenses (transport, lodging) are limited to 5000 per trip claim. "
        "Amounts above this limit require manager approval.",
        "category": "travel",
    },
    {
        "id": "POL-004",
        "title": "Software spending limit",
        "text": "Software and SaaS subscription invoices are limited to 25000 per invoice. "
        "Amounts above this limit require manager approval.",
        "category": "software",
    },
    {
        "id": "POL-005",
        "title": "Office supplies spending limit",
        "text": "Office supply purchases are limited to 3000 per invoice. "
        "Amounts above this limit require manager approval.",
        "category": "office_supplies",
    },
    {
        "id": "POL-006",
        "title": "Required document fields",
        "text": "Every reimbursable document must state vendor, date, currency, category "
        "and gross total. Documents missing any required field must be reviewed manually.",
        "category": "general",
    },
    {
        "id": "POL-007",
        "title": "Non-reimbursable categories",
        "text": "Alcohol and personal gifts are never reimbursable and must be rejected.",
        "category": "general",
    },
]
_TOKEN_RE = re.compile(r"[a-z0-9]+")

def _tokens(text: str)-> list[str]:
    return _TOKEN_RE.findall(text.lower())

def search_policies(query:str , top_k: int=3)->list[dict]:
    """Score policies by token overlap with an IDF-like rarity bonus."""
    query_tokens= set(_tokens(query))
    if not query_tokens:
        return []
    docs = [p["title"] + " " + p["text"] + " " + p["category"] for p in POLICIES]
    doc_tokens = [Counter(_tokens(d)) for d in docs]
    doc_freq = Counter()
    for dt in doc_tokens:
        for tok in dt:
            doc_freq[tok] +=1

    n_docs = len(docs)
    scored: list[tuple[float, dict]] = []
    for policy, dt in zip(POLICIES, doc_tokens):
        score = 0.0
        for tok in query_tokens:
            if tok in dt:
                idf = math.log((n_docs + 1) / (doc_freq[tok] + 1)) + 1.0
                score += dt[tok] * idf
        if score > 0:
            scored.append(
                (
                    round(score, 4),
                    {
                        "id": policy["id"],
                        "title": policy["title"],
                        "text": policy["text"],
                        "score": round(score, 4),
                    },
                )
            )
    scored.sort(key=lambda item: item[0], reverse=True)
    return [hit for _, hit in scored[:top_k]]
def category_limit(category: str | None) -> float | None:
    """Return the spending limit for a category, if a policy defines one."""
    if not category:
        return None
    for policy in POLICIES:
        if policy["category"] == category and "limited to" in policy["text"]:
            match = re.search(r"limited to (\d+)", policy["text"])
            if match:
                return float(match.group(1))
    return None