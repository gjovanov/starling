#!/usr/bin/env python3
"""Generate the synthetic invoice fixtures committed under tests/fixtures/.

No real customer data — all parties, numbers and IBANs are invented, but
checksums (ЕИК, IBAN mod-97) are COMPUTED here so the Phase 3 validators
pass on correct extractions and fail only where a fixture is deliberately
corrupted.

Run from apps/ocr-server:  ./.venv/bin/python tests/fixtures/generate_fixtures.py
"""

from pathlib import Path

import pymupdf

OUT_DIR = Path(__file__).resolve().parent
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


# ── checksum helpers (self-verifying fixtures) ─────────────────────

def eik9(base8: str) -> str:
    """Append the valid Bulgarian ЕИК check digit to an 8-digit base."""
    assert len(base8) == 8 and base8.isdigit()
    digits = [int(c) for c in base8]
    s = sum(d * w for d, w in zip(digits, range(1, 9))) % 11
    if s == 10:
        s = sum(d * w for d, w in zip(digits, range(3, 11))) % 11
        if s == 10:
            s = 0
    return base8 + str(s)


def iban(country: str, bban: str) -> str:
    """Compute the mod-97 check digits for country+bban."""
    rearranged = bban + country + "00"
    digits = "".join(str(int(c, 36)) for c in rearranged)
    check = 98 - int(digits) % 97
    return f"{country}{check:02d}{bban}"


# ── invoice model ──────────────────────────────────────────────────

class Inv:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def bg_invoice_1() -> Inv:
    supplier_eik = eik9("12345678")
    customer_eik = eik9("01000001")
    return Inv(
        lang="bg",
        title="ФАКТУРА",
        number="0000000123",
        date="15.06.2026",
        due="29.06.2026",
        currency="BGN",
        supplier=("Софтуер ЕООД", "ул. Витоша 15, София 1000",
                  f"ЕИК: {supplier_eik}", f"ДДС №: BG{supplier_eik}",
                  f"IBAN: {iban('BG', 'BNBG96611020345678')}"),
        customer=("Клиент АД", "бул. Марица 8, Пловдив 4000",
                  f"ЕИК: {customer_eik}", f"ДДС №: BG{customer_eik}"),
        cols=("Описание", "Кол.", "Ед. цена", "ДДС %", "Сума"),
        items=[("Софтуерна лицензия", "2", "250.00", "20", "500.00"),
               ("Техническа поддръжка", "1", "150.00", "20", "150.00")],
        totals=[("Данъчна основа:", "650.00"), ("ДДС 20%:", "130.00"),
                ("Обща сума:", "780.00")],
        footer="Плащане: 14 дни по банков път. Благодарим Ви!",
    )


def bg_invoice_2() -> Inv:
    supplier_eik = eik9("20460604")
    customer_eik = eik9("17530901")
    return Inv(
        lang="bg",
        title="ФАКТУРА ОРИГИНАЛ",
        number="2026-00088",
        date="02.07.2026",
        due="16.07.2026",
        currency="BGN",
        supplier=("Консулт БГ ООД", "ул. Раковски 99, Варна 9000",
                  f"ЕИК: {supplier_eik}", f"ДДС №: BG{supplier_eik}",
                  f"IBAN: {iban('BG', 'UNCR70001522345678')}"),
        customer=("Търговия и Логистика ЕАД", "жк. Люлин 10, София 1335",
                  f"ЕИК: {customer_eik}", f"ДДС №: BG{customer_eik}"),
        cols=("Описание", "Кол.", "Ед. цена", "ДДС %", "Сума"),
        items=[("Консултантски услуги (юни)", "40", "45.00", "20", "1800.00"),
               ("Пътни разходи", "1", "120.00", "20", "120.00"),
               ("Абонамент система", "3", "60.00", "20", "180.00")],
        totals=[("Данъчна основа:", "2100.00"), ("ДДС 20%:", "420.00"),
                ("Обща сума:", "2520.00")],
        footer="Срок за плащане: 14 дни. IBAN посочен по-горе.",
    )


def en_invoice_1() -> Inv:
    return Inv(
        lang="en",
        title="INVOICE",
        number="INV-2026-0451",
        date="2026-06-20",
        due="2026-07-04",
        currency="EUR",
        supplier=("Starling Software Ltd", "12 Harbour Road, Dublin D02, Ireland",
                  "Company No: IE0334512", "VAT: IE6388047V",
                  f"IBAN: {iban('IE', 'AIBK93115212345678')}"),
        customer=("Acme Trading GmbH", "Hauptstrasse 5, 10115 Berlin, Germany",
                  "VAT: DE811907980"),
        cols=("Description", "Qty", "Unit price", "VAT %", "Amount"),
        items=[("Software development services", "80", "95.00", "23", "7600.00"),
               ("Cloud hosting (June)", "1", "400.00", "23", "400.00")],
        totals=[("Net total:", "8000.00"), ("VAT 23%:", "1840.00"),
                ("Total due:", "9840.00")],
        footer="Payment within 14 days by bank transfer. Thank you!",
    )


def en_invoice_2() -> Inv:
    return Inv(
        lang="en",
        title="RECEIPT / INVOICE",
        number="R-77812",
        date="2026-07-01",
        due="2026-07-01",
        currency="GBP",
        supplier=("Northgate Supplies PLC", "40 King Street, Manchester M2 4LQ, UK",
                  "Company No: 04351290", "VAT: GB123456789",
                  f"IBAN: {iban('GB', 'NWBK60161331926819')}"),
        customer=("Bright Ideas Studio", "7 Camden Row, London NW1 8QL, UK",
                  "VAT: GB987654321"),
        cols=("Description", "Qty", "Unit price", "VAT %", "Amount"),
        items=[("A4 paper (box)", "10", "24.50", "20", "245.00"),
               ("Toner cartridge", "2", "62.50", "20", "125.00"),
               ("Delivery", "1", "30.00", "20", "30.00")],
        totals=[("Net total:", "400.00"), ("VAT 20%:", "80.00"),
                ("Total:", "480.00")],
        footer="Paid in full. Thank you for your order.",
    )


def bad_totals_invoice() -> Inv:
    # Deliberately corrupted: net 500 + VAT 100 must be 600, prints 650.
    inv = Inv(
        lang="en",
        title="INVOICE",
        number="INV-2026-0666",
        date="2026-06-30",
        due="2026-07-14",
        currency="EUR",
        supplier=("Broken Books Ltd", "1 Error Lane, Dublin, Ireland",
                  "VAT: IE6388047V", f"IBAN: {iban('IE', 'AIBK93115212345678')}"),
        customer=("Careless Customer BV", "Kanaalweg 3, Utrecht, Netherlands",
                  "VAT: NL123456789B01"),
        cols=("Description", "Qty", "Unit price", "VAT %", "Amount"),
        items=[("Bookkeeping services", "5", "100.00", "20", "500.00")],
        totals=[("Net total:", "500.00"), ("VAT 20%:", "100.00"),
                ("Total due:", "650.00")],   # WRONG on purpose (should be 600.00)
        footer="This invoice is intentionally inconsistent (test fixture).",
    )
    return inv


# ── rendering ──────────────────────────────────────────────────────

def render(inv: Inv) -> pymupdf.Document:
    doc = pymupdf.open()
    page = doc.new_page()  # A4 595x842
    black, grey = (0, 0, 0), (0.35, 0.35, 0.35)

    def text(x, y, s, size=10, bold=False, color=black):
        page.insert_text((x, y), s, fontsize=size, color=color,
                         fontname="fx-bold" if bold else "fx",
                         fontfile=FONT_BOLD if bold else FONT)

    text(72, 80, inv.title, size=22, bold=True)
    text(400, 80, f"№ {inv.number}", size=12, bold=True)
    text(400, 98, f"{'Дата' if inv.lang == 'bg' else 'Date'}: {inv.date}", size=10)
    text(400, 114, f"{'Падеж' if inv.lang == 'bg' else 'Due'}: {inv.due}", size=10)
    page.draw_line(pymupdf.Point(72, 130), pymupdf.Point(523, 130), color=black, width=1)

    text(72, 155, "Доставчик:" if inv.lang == "bg" else "Supplier:", size=9, color=grey)
    y = 170
    for line in inv.supplier:
        text(72, y, line, size=10)
        y += 15

    text(320, 155, "Получател:" if inv.lang == "bg" else "Bill to:", size=9, color=grey)
    y = 170
    for line in inv.customer:
        text(320, y, line, size=10)
        y += 15

    ty = 280
    page.draw_rect(pymupdf.Rect(72, ty - 14, 523, ty + 4), color=black, width=0.7)
    xs = (78, 330, 375, 435, 480)
    for x, col in zip(xs, inv.cols):
        text(x, ty, col, size=9, bold=True)
    y = ty + 22
    for row in inv.items:
        for x, cell in zip(xs, row):
            text(x, y, cell, size=10)
        y += 18
    page.draw_line(pymupdf.Point(72, y - 8), pymupdf.Point(523, y - 8), color=grey, width=0.5)

    y += 12
    for label, amount in inv.totals:
        text(360, y, label, size=10, bold=True)
        text(480, y, f"{amount} {inv.currency}", size=10, bold=True)
        y += 16

    text(72, 790, inv.footer, size=9, color=grey)
    return doc


def scanned_copy(doc: pymupdf.Document, dpi: int = 150) -> pymupdf.Document:
    """Rasterize every page and rebuild an image-only PDF ('scanned').
    JPEG-compressed so the committed fixtures stay small."""
    out = pymupdf.open()
    for page in doc:
        pix = page.get_pixmap(dpi=dpi)
        jpg = pix.tobytes("jpg", jpg_quality=80)
        p = out.new_page(width=page.rect.width, height=page.rect.height)
        p.insert_image(p.rect, stream=jpg)
    return out


def main() -> None:
    fixtures = {
        "invoice_bg_1": bg_invoice_1(),
        "invoice_bg_2": bg_invoice_2(),
        "invoice_en_1": en_invoice_1(),
        "invoice_en_2": en_invoice_2(),
        "invoice_bad_totals": bad_totals_invoice(),
    }
    for name, inv in fixtures.items():
        doc = render(inv)
        doc.subset_fonts()
        doc.save(OUT_DIR / f"{name}.pdf", garbage=4, deflate=True)
        scan = scanned_copy(doc)
        scan.save(OUT_DIR / f"{name}_scan.pdf", garbage=4, deflate=True)
        scan.close()
        doc.close()
    # One PNG upload fixture (single scanned page image).
    doc = render(fixtures["invoice_bg_1"])
    pix = doc[0].get_pixmap(dpi=200)
    pix.save(OUT_DIR / "invoice_bg_1.png")
    doc.close()
    print(f"fixtures written to {OUT_DIR}")


if __name__ == "__main__":
    main()
