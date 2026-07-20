"""Deterministic validation of extracted invoice JSON — pure functions, no LLM.

Output contract:
  validate_extraction(data) -> (overall, checks)
    overall: 'valid' | 'needs_review'   ('needs_review' iff any check fails)
    checks:  [{"check": str, "status": "pass"|"fail"|"skip"|"warn", "details": str}]

'warn' marks plausibility findings (e.g. unusual VAT rate) that must not
block a document; 'skip' means the inputs for a check were absent. Never
silently accept failing arithmetic — this feeds financial data downstream.
"""

import datetime as _dt
import re
from decimal import Decimal, InvalidOperation

TOLERANCE = Decimal("0.011")   # plan: ±0.01, padded for float round-trips

_BG_VAT_RE = re.compile(r"^BG\d{9,10}$")
_EU_VAT_RE = re.compile(r"^[A-Z]{2}[A-Z0-9]{2,12}$")
_CURRENCY_RE = re.compile(r"^[A-Z]{3}$")
_COMMON_CURRENCIES = {
    "BGN", "EUR", "USD", "GBP", "CHF", "RON", "TRY", "JPY", "CNY", "SEK",
    "NOK", "DKK", "PLN", "CZK", "HUF", "RSD", "MKD", "CAD", "AUD",
}
_BG_VAT_RATES = {Decimal("0"), Decimal("9"), Decimal("20")}


def _dec(value) -> Decimal | None:
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


def _close(a: Decimal, b: Decimal) -> bool:
    return abs(a - b) <= TOLERANCE


# ── identifier checks (each returns True/False) ────────────────────

def iban_valid(iban: str) -> bool:
    """ISO 13616 mod-97 checksum."""
    s = re.sub(r"\s+", "", iban).upper()
    if not re.match(r"^[A-Z]{2}\d{2}[A-Z0-9]{10,30}$", s):
        return False
    rearranged = s[4:] + s[:4]
    try:
        digits = "".join(str(int(c, 36)) for c in rearranged)
    except ValueError:
        return False
    return int(digits) % 97 == 1


def eik_valid(eik: str) -> bool:
    """Bulgarian ЕИК (UIC) checksum — 9 or 13 digits.

    9-digit: weights 1..8 mod 11; remainder 10 → weights 3..10 mod 11;
    still 10 → 0. 13-digit: valid 9-digit prefix, then digits 9..12 with
    weights [2,7,3,5] mod 11; remainder 10 → weights [4,9,5,7]; still 10 → 0.
    """
    s = re.sub(r"\s+", "", eik)
    if not re.match(r"^\d{9}(\d{4})?$", s):
        return False
    digits = [int(c) for c in s]

    check = sum(d * w for d, w in zip(digits[:8], range(1, 9))) % 11
    if check == 10:
        check = sum(d * w for d, w in zip(digits[:8], range(3, 11))) % 11
        if check == 10:
            check = 0
    if digits[8] != check:
        return False
    if len(s) == 9:
        return True

    check = sum(d * w for d, w in zip(digits[8:12], (2, 7, 3, 5))) % 11
    if check == 10:
        check = sum(d * w for d, w in zip(digits[8:12], (4, 9, 5, 7))) % 11
        if check == 10:
            check = 0
    return digits[12] == check


def vat_id_format_valid(vat_id: str) -> bool:
    s = re.sub(r"\s+", "", vat_id).upper()
    if s.startswith("BG"):
        return bool(_BG_VAT_RE.match(s))
    return bool(_EU_VAT_RE.match(s))


def date_valid(value: str) -> bool:
    try:
        _dt.date.fromisoformat(value)
        return True
    except ValueError:
        return False


# ── document-level validation ──────────────────────────────────────

def _is_bg_document(data: dict) -> bool:
    if (data.get("currency") or "").upper() == "BGN":
        return True
    for party in (data.get("supplier") or {}, data.get("customer") or {}):
        if (party.get("vat_id") or "").upper().startswith("BG"):
            return True
    return False


def validate_extraction(data: dict) -> tuple[str, list[dict]]:
    checks: list[dict] = []

    def add(check: str, status: str, details: str = "") -> None:
        checks.append({"check": check, "status": status, "details": details})

    items = data.get("line_items") or []
    bg_doc = _is_bg_document(data)

    # Per-line arithmetic: quantity × unit_price ≈ total
    for i, item in enumerate(items, start=1):
        qty, price, total = _dec(item.get("quantity")), _dec(item.get("unit_price")), _dec(item.get("total"))
        name = f"line_{i}_arithmetic"
        if qty is None or price is None or total is None:
            add(name, "skip", "quantity/unit_price/total not all present")
            continue
        expected = qty * price
        if _close(expected, total):
            add(name, "pass", f"{qty} × {price} = {total}")
        else:
            add(name, "fail", f"{qty} × {price} = {expected} ≠ {total}")

    # Σ line totals ≈ net_total
    net = _dec(data.get("net_total"))
    line_totals = [_dec(i.get("total")) for i in items]
    if net is None or not items or any(t is None for t in line_totals):
        add("lines_sum_net", "skip", "net_total or line totals missing")
    else:
        s = sum(line_totals, Decimal("0"))
        if _close(s, net):
            add("lines_sum_net", "pass", f"Σ lines = {s} = net")
        else:
            add("lines_sum_net", "fail", f"Σ lines = {s} ≠ net_total {net}")

    # net + vat ≈ gross
    vat, gross = _dec(data.get("vat_total")), _dec(data.get("gross_total"))
    if net is None or vat is None or gross is None:
        add("totals_net_vat_gross", "skip", "net/vat/gross not all present")
    elif _close(net + vat, gross):
        add("totals_net_vat_gross", "pass", f"{net} + {vat} = {gross}")
    else:
        add("totals_net_vat_gross", "fail",
            f"net {net} + vat {vat} = {net + vat} ≠ gross {gross}")

    # VAT rate plausibility (BG documents: {0, 9, 20}) — warn, never fail
    rates = {_dec(i.get("vat_rate")) for i in items if i.get("vat_rate") is not None}
    rates.discard(None)
    if not rates:
        add("vat_rate_plausible", "skip", "no vat rates present")
    elif bg_doc:
        odd = sorted(str(r) for r in rates if r not in _BG_VAT_RATES)
        if odd:
            add("vat_rate_plausible", "warn",
                f"unusual BG VAT rate(s): {', '.join(odd)} (expected 0/9/20)")
        else:
            add("vat_rate_plausible", "pass", "rates within BG set {0, 9, 20}")
    else:
        add("vat_rate_plausible", "pass", "non-BG document; no rate constraint")

    # Identifiers per party
    for role in ("supplier", "customer"):
        party = data.get(role) or {}
        iban = party.get("iban")
        if iban:
            add(f"{role}_iban", "pass" if iban_valid(iban) else "fail",
                f"mod-97 checksum of {iban}")
        else:
            add(f"{role}_iban", "skip", "no IBAN present")

        cid = party.get("company_id")
        if cid and re.match(r"^\d{9}(\d{4})?$", re.sub(r"\s+", "", cid)):
            add(f"{role}_eik", "pass" if eik_valid(cid) else "fail",
                f"ЕИК checksum of {cid}")
        elif cid:
            add(f"{role}_eik", "skip", f"company id {cid!r} is not a 9/13-digit ЕИК")
        else:
            add(f"{role}_eik", "skip", "no company id present")

        vat_id = party.get("vat_id")
        if vat_id:
            add(f"{role}_vat_id", "pass" if vat_id_format_valid(vat_id) else "fail",
                f"format of {vat_id}")
        else:
            add(f"{role}_vat_id", "skip", "no VAT id present")

    # Dates
    for field in ("issue_date", "due_date"):
        value = data.get(field)
        if value:
            add(field, "pass" if date_valid(value) else "fail",
                f"{value!r} must be ISO 8601 (YYYY-MM-DD)")
        else:
            add(field, "skip", "not present")

    # Currency
    currency = data.get("currency")
    if not currency:
        add("currency", "skip", "not present")
    elif not _CURRENCY_RE.match(currency):
        add("currency", "fail", f"{currency!r} is not a 3-letter ISO 4217 code")
    elif currency not in _COMMON_CURRENCIES:
        add("currency", "warn", f"{currency!r} is well-formed but uncommon — verify")
    else:
        add("currency", "pass", currency)

    overall = "needs_review" if any(c["status"] == "fail" for c in checks) else "valid"
    return overall, checks
