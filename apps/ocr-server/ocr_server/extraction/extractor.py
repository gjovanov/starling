"""Stage 2 — structured extraction: Markdown -> validated JSON dict.

VllmExtractor drives the lazy Qwen3-8B-FP8 vLLM subprocess (:8004) with
schema-constrained decoding (response_format json_schema → xgrammar), so the
output always parses; validate.py judges whether the values are credible.
"""

import abc
import json
import logging

import httpx

from ..engines.vllm_process import VllmProcess
from .schema import json_schema

logger = logging.getLogger(__name__)


class ExtractorError(Exception):
    """Raised when extraction fails (upstream error, invalid output)."""


class Extractor(abc.ABC):
    model: str

    @abc.abstractmethod
    async def extract(self, markdown: str, *, language_hint: str) -> dict:
        """Extract structured invoice data from the document Markdown."""


SYSTEM_PROMPT = (
    "You extract structured data from OCR'd financial documents (invoices, "
    "receipts, credit notes). Documents may be Bulgarian (Cyrillic) or English, "
    "and the OCR text may contain recognition errors. Fill the JSON schema "
    "exactly. Rules: use null when a value is absent — NEVER invent values; "
    "amounts are plain decimal numbers without currency symbols or thousands "
    "separators; dates are ISO 8601 (YYYY-MM-DD) — convert formats like "
    "'15.06.2026'; currency is the ISO 4217 code (лв. → BGN); vat_rate is the "
    "percentage as a number; company_id is the Bulgarian ЕИК / company "
    "registration number WITHOUT country prefix; vat_id is the ДДС/VAT number "
    "INCLUDING its country prefix (e.g. BG123456789)."
)


def build_request(*, model: str, markdown: str, language_hint: str,
                  max_tokens: int, temperature: float) -> dict:
    """Pure request builder (unit-tested offline)."""
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",
             "content": f"Language hint: {language_hint}\n\nDocument (Markdown):\n\n{markdown}"},
        ],
        "temperature": temperature,
        "top_p": 0.8,
        "max_tokens": max_tokens,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "invoice_extraction", "schema": json_schema()},
        },
        # Qwen3 hybrid reasoning: thinking must be off for constrained JSON.
        "chat_template_kwargs": {"enable_thinking": False},
    }


class VllmExtractor(Extractor):
    def __init__(self, process: VllmProcess, model: str, settings,
                 broadcaster=None, pre_start=None) -> None:
        self.model = model
        self._process = process
        self._settings = settings
        self._bc = broadcaster
        # Called before booting the extractor — the engine manager unloads a
        # resident heavy OCR engine here so both never coexist on 24 GiB.
        self._pre_start = pre_start

    async def _broadcast(self, status: str) -> None:
        if self._bc is not None:
            await self._bc.broadcast(
                {"type": "engine", "name": "extractor", "status": status})

    async def extract(self, markdown: str, *, language_hint: str) -> dict:
        from ..engines.vllm_process import ProcState

        if self._process.state is not ProcState.READY:
            if self._pre_start is not None:
                await self._pre_start()
            await self._broadcast("loading")
            try:
                await self._process.ensure_started()
            except Exception:
                await self._broadcast("unloaded")
                raise
            await self._broadcast("ready")
        self._process.note_activity()

        payload = build_request(
            model=self.model, markdown=markdown, language_hint=language_hint,
            max_tokens=self._settings.extract_max_tokens,
            temperature=self._settings.extract_temperature)
        try:
            async with httpx.AsyncClient(
                    timeout=self._settings.extract_timeout_secs) as client:
                resp = await client.post(
                    f"{self._process.base_url}/v1/chat/completions", json=payload)
        except httpx.HTTPError as exc:
            raise ExtractorError(f"extraction upstream unreachable: {exc}") from exc
        finally:
            self._process.note_activity()

        if resp.status_code != 200:
            raise ExtractorError(
                f"extraction upstream returned {resp.status_code}: {resp.text[:300]}")
        try:
            content = resp.json()["choices"][0]["message"]["content"]
            return json.loads(content)
        except (KeyError, IndexError, json.JSONDecodeError) as exc:
            raise ExtractorError(f"unparseable extraction response: {exc}") from exc


class MockExtractor(Extractor):
    """Deterministic extractor for tests — echoes a minimal valid payload."""

    model = "mock-extract"

    async def extract(self, markdown: str, *, language_hint: str) -> dict:
        return {
            "document_type": "other",
            "document_number": None,
            "issue_date": None,
            "due_date": None,
            "currency": None,
            "supplier": {"name": "Mock Supplier", "address": None, "vat_id": None,
                         "company_id": None, "iban": None},
            "customer": {"name": None, "address": None, "vat_id": None,
                         "company_id": None, "iban": None},
            "line_items": [],
            "net_total": None,
            "vat_total": None,
            "gross_total": None,
            "payment_terms": None,
            "notes": f"mock extraction of {len(markdown)} markdown chars "
                     f"(language_hint={language_hint})",
        }


def create_extractor(model: str, settings, *, process: VllmProcess | None = None,
                     broadcaster=None, pre_start=None) -> Extractor:
    """Resolve an extraction model name to a client.

    Only the configured settings.extract_model routes to the vLLM extractor —
    arbitrary names must fail fast instead of spawning a GPU process.
    """
    if model == MockExtractor.model:
        return MockExtractor()
    if process is not None and model == settings.extract_model:
        return VllmExtractor(process, model, settings, broadcaster=broadcaster,
                             pre_start=pre_start)
    available = [MockExtractor.model] + (
        [settings.extract_model] if process is not None else [])
    raise ExtractorError(
        f"unknown extraction model: {model!r} (available: {available})")
