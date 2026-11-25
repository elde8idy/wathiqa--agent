from __future__ import annotations
from typing import Optional, Literal
from pydantic import BaseModel, Field

DOC_TYPES = ("invoice", "expense_claim", "receipt", "unknown")

class ExtractedFields(BaseModel):
    """The structured record the extractor produces from a raw document."""
    vendor:  Optional[str] = None
    amount: Optional[float] = Field(default=None,description="Gross total")
    currency: Optional[str] = None
    date: Optional[str] = None
    category: Optional[str]= None
    net_amount: Optional[float] = None
    vat_amount: Optional[float] = None

    def completeness(self) -> float:
        """Fraction of fields filled, rounded to 2 decimals."""
        filled = sum(v is not None for v in self.model_dump().values())
        return round(filled / len(type(self).model_fields), 2)


class PolicyHit(BaseModel):
    id:str
    title :str
    text:str
    score: float

class ReviewResult(BaseModel):
    """Final outcome of one document review."""
    decision: Literal["approve", "reject", "human_review"]    
    doc_type: Literal["invoice", "expense_claim", "receipt", "unknown"]
    extraction: ExtractedFields
    confidence: float
    flags: list[str]
    policies: list[PolicyHit]
    investigation: list[str]
    report: str


