from fastapi.testclient import TestClient

from wathiqa_agent.api import create_app

CLEAN_INVOICE = """INVOICE
Vendor: TechHub Software Ltd
Date: 2026-10-02
Currency: EGP
Category: software
Net: 1000.00
VAT: 140.00
Total: 1140.00
"""

BAD_VAT_INVOICE = """INVOICE
Vendor: Delta Office Supplies
Date: 2026-10-05
Currency: EGP
Category: office_supplies
Net: 1000.00
VAT: 200.00
Total: 1200.00
"""


def test_health():
    client = TestClient(create_app())
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_review_clean_document_completes():
    client = TestClient(create_app())
    resp = client.post("/reviews", json={"document_text": CLEAN_INVOICE})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "complete"
    assert body["decision"] == "auto_approved"
    assert "WATHQA Review Report" in body["report"]


def test_review_flagged_document_pauses_then_resumes():
    client = TestClient(create_app())
    resp = client.post("/reviews", json={"document_text": BAD_VAT_INVOICE})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "needs_review"
    assert any(f.startswith("vat_mismatch") for f in body["flags"])

    resumed = client.post(
        f"/reviews/{body['thread_id']}/resume", json={"feedback": "approve"}
    )
    assert resumed.status_code == 200
    assert resumed.json()["decision"] == "approved_after_review"


def test_review_requires_input():
    client = TestClient(create_app())
    resp = client.post("/reviews", json={})
    assert resp.status_code == 400
