"""HTTP client for the local vllm-omni TTS server.

The TTS model is hosted by a separate `vllm-omni serve` process (port 8002 by
default). We expose a thin async wrapper so the API routes can stay focused
on validation and storage.

The wrapper is stateless except for a long-lived `httpx.AsyncClient`, created
lazily on first call. Tests inject a fake by monkey-patching
`get_tts_client()`.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Final

import httpx


# vllm-omni's accepted formats — kept in sync with the spike findings.
SUPPORTED_FORMATS: Final[frozenset[str]] = frozenset(
    {"wav", "pcm", "flac", "mp3", "aac", "opus"}
)
DEFAULT_FORMAT: Final[str] = "wav"


class TtsClientError(RuntimeError):
    """Raised on any non-200 response or transport error from the TTS server."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass
class TtsResult:
    audio_bytes: bytes
    content_type: str       # e.g. "audio/wav"
    elapsed_secs: float


class TtsClient:
    """Async wrapper around `POST /v1/audio/speech`.

    Construct once per process; `aclose()` on shutdown.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        timeout_secs: float = 300.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout_secs
        self._client: httpx.AsyncClient | None = None
        self._lock = asyncio.Lock()

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            async with self._lock:
                if self._client is None:
                    self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    async def synthesize(
        self,
        *,
        text: str,
        voice: str,
        response_format: str = DEFAULT_FORMAT,
    ) -> TtsResult:
        """Send a synthesize request and return the audio bytes.

        Caller has already validated `voice` against `voices.VOICE_IDS` and
        capped `text` length. We don't re-validate here — clients of THIS
        class are server-side code only.
        """
        if response_format not in SUPPORTED_FORMATS:
            raise TtsClientError(f"unsupported response_format: {response_format}")

        client = await self._ensure_client()
        payload = {
            "model": self._model,
            "input": text,
            "voice": voice,
            "response_format": response_format,
        }

        loop = asyncio.get_running_loop()
        t0 = loop.time()
        try:
            resp = await client.post(f"{self._base_url}/audio/speech", json=payload)
        except httpx.HTTPError as exc:
            raise TtsClientError(f"transport error talking to TTS server: {exc}") from exc

        elapsed = loop.time() - t0
        if resp.status_code != 200:
            # Truncate body so we don't blow up logs.
            body_preview = resp.text[:500] if resp.text else ""
            raise TtsClientError(
                f"TTS server returned {resp.status_code}: {body_preview}",
                status_code=resp.status_code,
            )

        return TtsResult(
            audio_bytes=resp.content,
            content_type=resp.headers.get("content-type", "application/octet-stream"),
            elapsed_secs=elapsed,
        )

    async def synthesize_stream_concat(
        self,
        *,
        text_parts: list[str],
        voice: str,
        chunk_size: int | None = None,
    ) -> AsyncIterator[bytes]:
        """Stream PCM bytes for many sentences, back-to-back, in one stream.

        Opens an upstream `/v1/audio/speech?stream=true` request per sentence
        and yields its PCM bytes; closes the upstream cleanly between
        sentences. The caller (the route handler) prepends a single
        streaming WAV header before this generator runs.

        First-part errors propagate (caller can return 5xx); per-part errors
        on later sentences are logged and dropped — partial audio is better
        than tearing down the whole stream because part 7 of 12 timed out.
        """
        for idx, part in enumerate(text_parts):
            try:
                async for chunk in self.synthesize_stream(
                    text=part, voice=voice, chunk_size=chunk_size
                ):
                    yield chunk
            except TtsClientError:
                if idx == 0:
                    raise
                # Best-effort: skip this sentence, continue with the next.
                # The caller's status text won't reflect the gap, but the
                # listener gets a slightly shorter audio rather than an
                # abrupt full-stream failure.

    async def synthesize_stream(
        self,
        *,
        text: str,
        voice: str,
        chunk_size: int | None = None,
    ) -> AsyncIterator[bytes]:
        """Stream raw PCM bytes from the upstream `/v1/audio/speech?stream=true`.

        We always request `response_format=pcm` because vllm-omni 0.18's
        `response_format=wav` streaming path asserts on missing sample-rate
        metadata in the first chunk and dies before yielding anything. Callers
        that need WAV-framed bytes should prepend our streaming header (see
        `voxtral_server.tts.wav.streaming_header`) — the PCM payload itself
        is 24 kHz / mono / signed int16 LE, matching the model's native rate.

        The upstream sometimes closes the connection without a terminal
        chunk; we treat `RemoteProtocolError` after we've seen at least one
        byte as a clean end-of-stream rather than re-raising.
        """
        client = await self._ensure_client()
        payload = {
            "model": self._model,
            "input": text,
            "voice": voice,
            "response_format": "pcm",
            "stream": True,
        }

        async with client.stream(
            "POST", f"{self._base_url}/audio/speech", json=payload
        ) as resp:
            if resp.status_code != 200:
                # Drain to text for diagnostics (cap to keep logs sane).
                body = await resp.aread()
                snippet = body[:500].decode("utf-8", errors="replace")
                raise TtsClientError(
                    f"TTS server returned {resp.status_code}: {snippet}",
                    status_code=resp.status_code,
                )

            received_any = False
            try:
                async for chunk in resp.aiter_bytes(chunk_size=chunk_size):
                    if not chunk:
                        continue
                    received_any = True
                    yield chunk
            except httpx.RemoteProtocolError as exc:
                # vllm-omni 0.18 sometimes drops the terminal chunk. If we
                # already streamed real audio, treat it as EOF; otherwise
                # surface the failure so callers can show an error.
                if not received_any:
                    raise TtsClientError(
                        f"upstream closed before first chunk: {exc}"
                    ) from exc

    async def upload_voice(
        self,
        *,
        voice_name: str,
        consent: str,
        audio_bytes: bytes,
        audio_mime: str = "audio/wav",
        ref_text: str | None = None,
    ) -> dict:
        """Forward an upload to vllm-omni's `POST /v1/audio/voices`.

        The upstream stores the audio in `SPEECH_VOICE_SAMPLES` (default
        `/tmp/voice_samples`) and registers the voice in its in-memory
        speaker table. From then on, `voice="<voice_name>"` synth requests
        use the cloned voice (only when `--task-type Base` is set).

        Returns the upstream's success payload verbatim — `{voice: {...}}`
        — so the caller can correlate fields if needed.
        """
        client = await self._ensure_client()
        files = {"audio_sample": (f"{voice_name}.wav", audio_bytes, audio_mime)}
        data: dict[str, str] = {"name": voice_name, "consent": consent}
        if ref_text:
            data["ref_text"] = ref_text
        try:
            resp = await client.post(
                f"{self._base_url}/audio/voices",
                files=files,
                data=data,
            )
        except httpx.HTTPError as exc:
            raise TtsClientError(f"transport error uploading voice: {exc}") from exc
        if resp.status_code != 200:
            body = resp.text[:500] if resp.text else ""
            raise TtsClientError(
                f"upstream rejected voice upload ({resp.status_code}): {body}",
                status_code=resp.status_code,
            )
        return resp.json()

    async def delete_voice(self, voice_name: str) -> bool:
        """Best-effort upstream voice removal. Returns True on 200/404."""
        client = await self._ensure_client()
        try:
            resp = await client.delete(f"{self._base_url}/audio/voices/{voice_name}")
        except httpx.HTTPError as exc:
            raise TtsClientError(f"transport error deleting voice: {exc}") from exc
        return resp.status_code in (200, 404)

    async def health(self) -> bool:
        """True if the upstream TTS server responds to /v1/models."""
        try:
            client = await self._ensure_client()
            resp = await client.get(f"{self._base_url}/models", timeout=5.0)
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


_singleton: TtsClient | None = None


def get_tts_client() -> TtsClient:
    """Module-level singleton. Tests monkey-patch this function."""
    global _singleton
    if _singleton is None:
        from ..config import settings

        _singleton = TtsClient(
            base_url=settings.tts_vllm_url,
            model=settings.tts_model_path,
            timeout_secs=settings.tts_request_timeout_secs,
        )
    return _singleton


def reset_tts_client() -> None:
    """Test helper — drop the cached singleton without awaiting it."""
    global _singleton
    _singleton = None
