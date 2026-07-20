"""Extractor unit tests — request shape + factory routing (no GPU)."""

import pytest

from ocr_server.config import settings
from ocr_server.extraction.extractor import (
    ExtractorError,
    MockExtractor,
    build_request,
    create_extractor,
)
from ocr_server.extraction.schema import InvoiceExtraction


def test_build_request_shape():
    req = build_request(model="qwen3-8b-fp8", markdown="# Invoice\n\nTotal: 5",
                        language_hint="bg,en", max_tokens=1024, temperature=0.1)
    assert req["model"] == "qwen3-8b-fp8"
    assert req["messages"][0]["role"] == "system"
    assert "NEVER invent values" in req["messages"][0]["content"]
    assert "Language hint: bg,en" in req["messages"][1]["content"]
    assert "# Invoice" in req["messages"][1]["content"]
    assert req["max_tokens"] == 1024
    assert req["temperature"] == 0.1
    # Constrained decoding + thinking disabled for Qwen3.
    assert req["response_format"]["type"] == "json_schema"
    schema = req["response_format"]["json_schema"]["schema"]
    assert "document_type" in schema["properties"]
    assert req["chat_template_kwargs"] == {"enable_thinking": False}


def test_schema_roundtrip_accepts_mock_payload():
    mock = MockExtractor()
    import asyncio

    data = asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
        mock.extract("md", language_hint="bg,en"))
    # The mock payload must satisfy the real schema.
    InvoiceExtraction.model_validate(data)


def test_factory_routes_mock():
    ex = create_extractor("mock-extract", settings)
    assert isinstance(ex, MockExtractor)


def test_factory_rejects_unknown_without_process():
    with pytest.raises(ExtractorError, match="unknown extraction model"):
        create_extractor("qwen3-8b-fp8", settings, process=None)


def test_factory_rejects_arbitrary_names_with_process(monkeypatch):
    from ocr_server.engines.vllm_process import VllmProcess

    proc = VllmProcess(name="x", start_command=["true"], base_url="http://127.0.0.1:9")
    monkeypatch.setattr(settings, "extract_model", "qwen3-8b-fp8")
    ex = create_extractor("qwen3-8b-fp8", settings, process=proc)
    assert ex.model == "qwen3-8b-fp8"
    with pytest.raises(ExtractorError, match="unknown extraction model"):
        create_extractor("totally-else", settings, process=proc)
