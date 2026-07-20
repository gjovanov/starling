#!/usr/bin/env python3
"""OCR bake-off bench: run a folder of documents through N engines against a
running ocr-server and write a CSV of per-engine timings + validation results.

Usage (from the repo root, using the app venv):
  ./apps/ocr-server/.venv/bin/python tools/ocr-bench.py \\
      --dir apps/ocr-server/tests/fixtures \\
      --engines glm-ocr,tesseract-bul \\
      --extract auto

Writes results/ocr/bench-<timestamp>.csv and prints a per-engine summary.
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import httpx

ACCEPTED = {".pdf", ".png", ".jpg", ".jpeg"}


def wait_run(client: httpx.Client, server: str, run_id: int, timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        detail = client.get(f"{server}/api/ocr/runs/{run_id}").json()["data"]
        if detail["run"]["status"] in ("done", "failed"):
            return detail
        time.sleep(1.0)
    raise TimeoutError(f"run {run_id} did not finish within {timeout}s")


def wait_extraction(client: httpx.Client, server: str, run_id: int, timeout: float) -> dict | None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        detail = client.get(f"{server}/api/ocr/runs/{run_id}").json()["data"]
        exs = detail["extractions"]
        if exs and exs[0]["status"] in ("done", "failed"):
            return exs[0]
        time.sleep(1.0)
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="folder of PDFs/PNGs/JPEGs")
    ap.add_argument("--engines", required=True,
                    help="comma-separated engine names (see GET /api/ocr/engines)")
    ap.add_argument("--server", default="http://localhost:8092")
    ap.add_argument("--extract", choices=["auto", "off"], default="off",
                    help="run stage-2 extraction per run (default: off)")
    ap.add_argument("--language-hint", default="bg,en")
    ap.add_argument("--timeout", type=float, default=600.0, help="per-run timeout (s)")
    ap.add_argument("--out", default="results/ocr", help="output directory for the CSV")
    args = ap.parse_args()

    files = sorted(p for p in Path(args.dir).iterdir()
                   if p.suffix.lower() in ACCEPTED)
    if not files:
        print(f"no documents found in {args.dir}", file=sys.stderr)
        return 1
    engines = [e.strip() for e in args.engines.split(",") if e.strip()]

    client = httpx.Client(timeout=30.0)
    registered = {e["name"] for e in
                  client.get(f"{args.server}/api/ocr/engines").json()["data"]["engines"]}
    missing = [e for e in engines if e not in registered]
    if missing:
        print(f"engines not registered on the server: {missing} "
              f"(available: {sorted(registered)})", file=sys.stderr)
        return 1

    rows: list[dict] = []
    for path in files:
        print(f"── {path.name}")
        # Upload once (sha256 dedupe reuses the document); first engine's run
        # comes from the upload itself, further engines via the re-run endpoint.
        with open(path, "rb") as fh:
            resp = client.post(
                f"{args.server}/api/ocr/documents",
                files=[("files", (path.name, fh.read()))],
                data={"engine": engines[0], "extract": args.extract,
                      "language_hint": args.language_hint})
        if resp.status_code != 200:
            print(f"   upload rejected ({resp.status_code}): {resp.text[:120]}")
            continue
        doc = resp.json()["data"]["documents"][0]
        run_ids = {engines[0]: doc["run_id"]}
        for engine in engines[1:]:
            r = client.post(f"{args.server}/api/ocr/documents/{doc['document_id']}/runs",
                            json={"engine": engine, "extract": args.extract,
                                  "language_hint": args.language_hint})
            run_ids[engine] = r.json()["data"]["id"]

        for engine, run_id in run_ids.items():
            try:
                detail = wait_run(client, args.server, run_id, args.timeout)
            except TimeoutError as exc:
                print(f"   {engine}: {exc}")
                rows.append({"file": path.name, "engine": engine, "run_id": run_id,
                             "status": "timeout"})
                continue
            run = detail["run"]
            pages = len(detail["pages"])
            md = client.get(f"{args.server}/api/ocr/runs/{run_id}/markdown").text
            ex = None
            if args.extract == "auto" and run["status"] == "done":
                ex = wait_extraction(client, args.server, run_id, args.timeout)
            ms_page = round(run["ms_total"] / pages) if run["ms_total"] and pages else None
            rows.append({
                "file": path.name,
                "engine": engine,
                "run_id": run_id,
                "status": run["status"],
                "pages": pages,
                "ms_total": run["ms_total"],
                "ms_per_page": ms_page,
                "md_chars": len(md),
                "extraction_status": ex["status"] if ex else "",
                "validation_status": (ex.get("validation_status") or "") if ex else "",
                "error": (run.get("error") or "")[:200],
            })
            print(f"   {engine}: {run['status']} {pages}p "
                  f"{ms_page or '-'}ms/page {len(md)} chars"
                  + (f" → {ex.get('validation_status')}" if ex else ""))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    out_csv = out_dir / f"bench-{ts}.csv"
    fields = ["file", "engine", "run_id", "status", "pages", "ms_total",
              "ms_per_page", "md_chars", "extraction_status", "validation_status", "error"]
    with open(out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {out_csv} ({len(rows)} rows)")

    # Per-engine summary
    print(f"\n{'engine':<16} {'docs':>5} {'ok':>4} {'avg ms/page':>12} {'valid':>6}")
    for engine in engines:
        es = [r for r in rows if r["engine"] == engine]
        ok = [r for r in es if r.get("status") == "done"]
        mss = [r["ms_per_page"] for r in ok if r.get("ms_per_page")]
        valid = sum(1 for r in ok if r.get("validation_status") == "valid")
        avg = round(sum(mss) / len(mss)) if mss else "-"
        print(f"{engine:<16} {len(es):>5} {len(ok):>4} {avg!s:>12} "
              f"{valid}/{len(ok) if args.extract == 'auto' else '-'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
