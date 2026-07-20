"""Validation suite unit tests — IBAN / ЕИК / VAT vectors + arithmetic."""

from ocr_server.extraction.validate import (
    date_valid,
    eik_valid,
    iban_valid,
    validate_extraction,
    vat_id_format_valid,
)

# ── identifier vectors ─────────────────────────────────────────────


def test_iban_vectors():
    assert iban_valid("BG80BNBG96611020345678")          # canonical BNB example
    assert iban_valid("GB29NWBK60161331926819")          # canonical UK example
    assert iban_valid("DE89370400440532013000")          # canonical DE example
    assert iban_valid("BG80 BNBG 9661 1020 3456 78")     # spaces tolerated
    assert not iban_valid("BG80BNBG96611020345679")      # last digit altered
    assert not iban_valid("BG00BNBG96611020345678")      # wrong check digits
    assert not iban_valid("NOTANIBAN")
    assert not iban_valid("")


def test_eik_9_digit():
    # weights 1..8 over 12345678 → 204 % 11 = 6
    assert eik_valid("123456786")
    assert not eik_valid("123456785")
    # second-pass case: 01000001 → 10 % 11 = 10 → weights 3..10 → 14 % 11 = 3
    assert eik_valid("010000013")
    assert not eik_valid("010000010")
    assert not eik_valid("12345678")      # too short
    assert not eik_valid("12345678a")


def test_eik_13_digit():
    # prefix 123456786 valid; digits 9..12 = (6,0,0,1) with weights (2,7,3,5)
    # → 12+5 = 17 % 11 = 6 → check digit 6
    assert eik_valid("1234567860016")
    assert not eik_valid("1234567860015")
    assert not eik_valid("1234567850016")  # invalid 9-digit prefix


def test_vat_id_formats():
    assert vat_id_format_valid("BG123456786")
    assert vat_id_format_valid("BG1234567890")   # 10 digits allowed
    assert vat_id_format_valid("DE811907980")
    assert vat_id_format_valid("IE6388047V")
    assert vat_id_format_valid("NL123456789B01")
    assert not vat_id_format_valid("BG12345")     # BG needs 9-10 digits
    assert not vat_id_format_valid("BG12345678X")  # BG must be digits only
    assert not vat_id_format_valid("X1")
    assert not vat_id_format_valid("123456789")


def test_dates():
    assert date_valid("2026-06-15")
    assert not date_valid("15.06.2026")
    assert not date_valid("2026-13-01")


# ── document-level validation ──────────────────────────────────────


def _doc(**overrides) -> dict:
    base = {
        "document_type": "invoice",
        "document_number": "1",
        "issue_date": "2026-06-15",
        "due_date": "2026-06-29",
        "currency": "BGN",
        "supplier": {"name": "S", "address": None, "vat_id": "BG123456786",
                     "company_id": "123456786",
                     "iban": "BG80BNBG96611020345678"},
        "customer": {"name": "C", "address": None, "vat_id": "BG010000013",
                     "company_id": "010000013", "iban": None},
        "line_items": [
            {"description": "a", "quantity": 2, "unit": None, "unit_price": 250.0,
             "vat_rate": 20, "total": 500.0},
            {"description": "b", "quantity": 1, "unit": None, "unit_price": 150.0,
             "vat_rate": 20, "total": 150.0},
        ],
        "net_total": 650.0,
        "vat_total": 130.0,
        "gross_total": 780.0,
        "payment_terms": None,
        "notes": None,
    }
    base.update(overrides)
    return base


def _by_name(checks: list[dict]) -> dict[str, dict]:
    return {c["check"]: c for c in checks}


def test_consistent_document_is_valid():
    overall, checks = validate_extraction(_doc())
    assert overall == "valid", checks
    named = _by_name(checks)
    assert named["line_1_arithmetic"]["status"] == "pass"
    assert named["lines_sum_net"]["status"] == "pass"
    assert named["totals_net_vat_gross"]["status"] == "pass"
    assert named["supplier_iban"]["status"] == "pass"
    assert named["supplier_eik"]["status"] == "pass"
    assert named["customer_eik"]["status"] == "pass"
    assert named["vat_rate_plausible"]["status"] == "pass"
    assert named["currency"]["status"] == "pass"


def test_corrupted_gross_total_needs_review():
    overall, checks = validate_extraction(_doc(gross_total=830.0))
    assert overall == "needs_review"
    named = _by_name(checks)
    assert named["totals_net_vat_gross"]["status"] == "fail"
    assert "≠ gross 830" in named["totals_net_vat_gross"]["details"]


def test_bad_line_arithmetic_fails_named_line():
    doc = _doc()
    doc["line_items"][1]["total"] = 175.0
    overall, checks = validate_extraction(doc)
    assert overall == "needs_review"
    named = _by_name(checks)
    assert named["line_1_arithmetic"]["status"] == "pass"
    assert named["line_2_arithmetic"]["status"] == "fail"
    assert named["lines_sum_net"]["status"] == "fail"


def test_missing_fields_skip_not_fail():
    doc = _doc(net_total=None, vat_total=None, gross_total=None)
    doc["line_items"] = []
    doc["supplier"]["iban"] = None
    overall, checks = validate_extraction(doc)
    assert overall == "valid"
    named = _by_name(checks)
    assert named["lines_sum_net"]["status"] == "skip"
    assert named["totals_net_vat_gross"]["status"] == "skip"
    assert named["supplier_iban"]["status"] == "skip"


def test_unusual_bg_vat_rate_warns_not_fails():
    doc = _doc()
    for item in doc["line_items"]:
        item["vat_rate"] = 19  # German rate on a BGN document
        item["total"] = item["quantity"] * item["unit_price"]
    overall, checks = validate_extraction(doc)
    named = _by_name(checks)
    assert named["vat_rate_plausible"]["status"] == "warn"
    assert overall == "valid"  # warn does not trigger review


def test_non_bg_document_rates_unconstrained():
    doc = _doc(currency="EUR")
    doc["supplier"]["vat_id"] = "IE6388047V"
    doc["customer"]["vat_id"] = "DE811907980"
    doc["supplier"]["company_id"] = None
    doc["customer"]["company_id"] = None
    doc["supplier"]["iban"] = "DE89370400440532013000"
    for item in doc["line_items"]:
        item["vat_rate"] = 23
    overall, checks = validate_extraction(doc)
    named = _by_name(checks)
    assert named["vat_rate_plausible"]["status"] == "pass"


def test_bad_iban_and_eik_fail():
    doc = _doc()
    doc["supplier"]["iban"] = "BG80BNBG96611020345679"
    doc["customer"]["company_id"] = "123456785"
    overall, checks = validate_extraction(doc)
    assert overall == "needs_review"
    named = _by_name(checks)
    assert named["supplier_iban"]["status"] == "fail"
    assert named["customer_eik"]["status"] == "fail"


def test_non_eik_company_id_skips():
    doc = _doc()
    doc["supplier"]["company_id"] = "IE0334512"  # not a 9/13-digit ЕИК
    _, checks = validate_extraction(doc)
    assert _by_name(checks)["supplier_eik"]["status"] == "skip"


def test_bad_date_and_currency_fail():
    overall, checks = validate_extraction(_doc(issue_date="15.06.2026", currency="lv"))
    assert overall == "needs_review"
    named = _by_name(checks)
    assert named["issue_date"]["status"] == "fail"
    assert named["currency"]["status"] == "fail"


def test_wellformed_uncommon_currency_warns():
    _, checks = validate_extraction(_doc(currency="XPF"))
    assert _by_name(checks)["currency"]["status"] == "warn"
