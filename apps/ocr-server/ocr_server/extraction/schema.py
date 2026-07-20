"""InvoiceExtraction schema — Pydantic → JSON Schema → vLLM guided JSON.

Constrained decoding guarantees parseable JSON matching this shape; it does
NOT guarantee correct values — validate.py checks those.

Every field is REQUIRED but nullable (no defaults): fields with defaults
drop out of JSON-Schema `required`, and the constrained decoder then omits
the keys entirely instead of emitting null.
"""

from typing import Literal

from pydantic import BaseModel


class Party(BaseModel):
    name: str | None
    address: str | None
    vat_id: str | None       # BG ДДС № / EU VAT (with country prefix)
    company_id: str | None   # ЕИК / registration number (no prefix)
    iban: str | None


class LineItem(BaseModel):
    description: str
    quantity: float | None
    unit: str | None
    unit_price: float | None
    vat_rate: float | None   # percent
    total: float | None


class InvoiceExtraction(BaseModel):
    document_type: Literal["invoice", "receipt", "credit_note", "proforma", "other"]
    document_number: str | None
    issue_date: str | None   # ISO 8601 (YYYY-MM-DD)
    due_date: str | None
    currency: str | None     # ISO 4217
    supplier: Party
    customer: Party
    line_items: list[LineItem]
    net_total: float | None
    vat_total: float | None
    gross_total: float | None
    payment_terms: str | None
    notes: str | None


def json_schema() -> dict:
    return InvoiceExtraction.model_json_schema()
