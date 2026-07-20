"""Pydantic API models shared by the HTTP routes."""

from typing import Any

from pydantic import BaseModel


class ApiResponse(BaseModel):
    """The {success, data, error} envelope the shared frontend unwraps."""

    success: bool = True
    data: Any | None = None
    error: str | None = None

    @classmethod
    def ok(cls, data: Any = None) -> "ApiResponse":
        return cls(success=True, data=data)

    @classmethod
    def err(cls, error: str) -> "ApiResponse":
        return cls(success=False, error=error)


class RunCreateRequest(BaseModel):
    """POST /api/ocr/documents/{id}/runs — bake-off re-run with another engine."""

    engine: str
    extract: str | None = None        # 'auto' | 'off'; None → server default
    language_hint: str | None = None


class ExtractRequest(BaseModel):
    """POST /api/ocr/runs/{id}/extract — rerun stage 2, optionally another model."""

    model: str | None = None
