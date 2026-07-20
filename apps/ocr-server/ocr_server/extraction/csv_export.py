"""Render an extracted invoice (schema.InvoiceExtraction dict) to CSV.

Layout: blank-line-separated sections that open cleanly in Excel/LibreOffice —
an INVOICE METADATA key/value block, SUPPLIER and CUSTOMER key/value blocks,
then a LINE ITEMS table. Pure function (csv module handles quoting of commas
in addresses/descriptions); the route prepends a UTF-8 BOM for Cyrillic.
"""

import csv
import io

_META_FIELDS = [
    ("Document type", "document_type"),
    ("Document number", "document_number"),
    ("Issue date", "issue_date"),
    ("Due date", "due_date"),
    ("Currency", "currency"),
    ("Net total", "net_total"),
    ("VAT total", "vat_total"),
    ("Gross total", "gross_total"),
    ("Payment terms", "payment_terms"),
    ("Notes", "notes"),
]
_PARTY_FIELDS = [
    ("Name", "name"),
    ("Address", "address"),
    ("VAT ID", "vat_id"),
    ("Company ID", "company_id"),
    ("IBAN", "iban"),
]
_ITEM_COLUMNS = [
    ("#", None),
    ("Description", "description"),
    ("Quantity", "quantity"),
    ("Unit", "unit"),
    ("Unit price", "unit_price"),
    ("VAT %", "vat_rate"),
    ("Line total", "total"),
]


def _cell(value) -> str:
    return "" if value is None else str(value)


def extraction_to_csv(data: dict) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)

    w.writerow(["INVOICE METADATA"])
    w.writerow(["Field", "Value"])
    for label, key in _META_FIELDS:
        w.writerow([label, _cell(data.get(key))])

    for section, key in (("SUPPLIER", "supplier"), ("CUSTOMER", "customer")):
        party = data.get(key) or {}
        w.writerow([])
        w.writerow([section])
        w.writerow(["Field", "Value"])
        for label, pk in _PARTY_FIELDS:
            w.writerow([label, _cell(party.get(pk))])

    w.writerow([])
    w.writerow(["LINE ITEMS"])
    w.writerow([label for label, _ in _ITEM_COLUMNS])
    for i, item in enumerate(data.get("line_items") or [], start=1):
        row = [str(i)]
        for _, key in _ITEM_COLUMNS[1:]:
            row.append(_cell(item.get(key)))
        w.writerow(row)

    return buf.getvalue()
