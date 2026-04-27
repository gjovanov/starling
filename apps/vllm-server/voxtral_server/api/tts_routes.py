"""Text-to-speech routes — proxies to a local vllm-omni server.

Endpoints:
  GET    /api/tts/voices             — voice catalog (20 presets)
  GET    /api/tts/config             — frontend-needed config (output dir, caps)
  POST   /api/tts/synthesize         — synth + save (Phase 1; streaming in Phase 2)
  GET    /api/tts/output             — list saved files
  GET    /api/tts/output/{name}      — download a saved file
  DELETE /api/tts/output/{name}      — remove a saved file

All non-streaming responses use the `ApiResponse` envelope, matching the rest
of the API.
"""

from __future__ import annotations

import asyncio
import io
import logging
import time
from collections.abc import AsyncIterator
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from ..config import settings
from ..models import (
    ApiResponse,
    TtsConfig,
    TtsOutputFile,
    TtsSynthesizeRequest,
    TtsSynthesizeResponse,
    VoiceRefInfo,
)
from ..tts import refs as refs_mod, storage, text as text_split, voices, wav
from ..tts.client import TtsClientError, get_tts_client
from ..tts.lifecycle import LifecycleError, get_lifecycle


logger = logging.getLogger(__name__)
router = APIRouter()


def _output_dir() -> Path:
    """Resolve and lazily create the configured TTS output directory."""
    out = Path(settings.tts_output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    return out


def _voice_refs_dir() -> Path:
    out = Path(settings.tts_voice_refs_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    return out


def _ref_to_public(info: refs_mod.VoiceRef) -> dict:
    """Drop the on-disk paths from the public payload."""
    return VoiceRefInfo(
        id=info.id,
        name=info.name,
        ref_text=info.ref_text,
        permission_confirmed=info.permission_confirmed,
        sample_rate=info.sample_rate,
        duration_secs=info.duration_secs,
        created_at=info.created_at,
        kind="cloned",
    ).model_dump()


def _wav_duration_secs(audio_bytes: bytes) -> float | None:
    """Best-effort WAV duration via the stdlib `wave` module.

    Returns None when the bytes aren't a recognizable WAV (e.g. the format
    was overridden to mp3/opus). Never raises — duration is metadata-only.
    """
    try:
        import wave  # noqa: WPS433 — stdlib, deferred import keeps cold-start fast

        with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate() or 1
            return frames / rate
    except Exception:  # noqa: BLE001 — diagnostic-only; metadata, not flow control
        return None


# ---------------------------------------------------------------------------
# Voice catalog & static config
# ---------------------------------------------------------------------------

@router.get("/api/tts/voices")
async def list_voices() -> ApiResponse:
    """Return both built-in and uploaded (cloned) voices."""
    builtin = [{**v.model_dump(), "kind": "builtin"} for v in voices.VOICES]
    cloned = [_ref_to_public(r) for r in refs_mod.list_refs(_voice_refs_dir())]
    return ApiResponse.ok(builtin + cloned)


@router.post("/api/tts/voices/upload")
async def upload_voice_ref(
    audio_sample: UploadFile = File(...),
    name: str = Form(...),
    ref_text: str = Form(...),
    permission_confirmed: bool = Form(...),
) -> ApiResponse:
    """Upload a reference audio + transcript for voice cloning.

    Saves locally (with permission audit) AND forwards to upstream
    vllm-omni's `/v1/audio/voices`. Subsequent /api/tts/synthesize calls
    with `voice_ref_id=<the returned id>` will use this voice (after
    upstream is reloaded into `--task-type Base`).
    """
    refs = refs_mod.list_refs(_voice_refs_dir())
    if len(refs) >= settings.tts_max_voice_refs:
        return ApiResponse.err(
            f"voice ref limit reached ({settings.tts_max_voice_refs}); "
            "delete an existing one first"
        )

    audio_bytes = await audio_sample.read()
    try:
        info = refs_mod.save_ref(
            audio_bytes=audio_bytes,
            audio_filename=audio_sample.filename or "upload.wav",
            name=name,
            ref_text=ref_text,
            permission_confirmed=bool(permission_confirmed),
            voice_refs_dir=_voice_refs_dir(),
            max_audio_bytes=settings.tts_max_ref_audio_bytes,
            min_duration_secs=settings.tts_min_ref_duration_secs,
            max_duration_secs=settings.tts_max_ref_duration_secs,
            require_permission=settings.tts_require_permission,
        )
    except storage.StorageError as exc:
        return ApiResponse.err(f"upload rejected: {exc}")

    # Forward to the upstream vllm-omni voice store. Failures here are
    # surfaced as warnings — local metadata persists, so the user can
    # retry the upstream registration via re-upload.
    try:
        # Read the just-encoded WAV (24 kHz mono) so we forward the
        # canonical bytes, not the raw client-supplied data.
        with open(info.audio_path, "rb") as fh:
            encoded = fh.read()
        await get_tts_client().upload_voice(
            voice_name=info.id,
            consent="user-confirmed via Starling /api/tts/voices/upload",
            audio_bytes=encoded,
            audio_mime="audio/wav",
            ref_text=info.ref_text,
        )
    except TtsClientError as exc:
        logger.warning("upstream voice upload failed (local saved OK): %s", exc)
        # Don't fail the API call — the local entry is still useful for
        # the user. The frontend will surface a "needs re-sync" hint
        # when the upstream is restarted.

    return ApiResponse.ok(_ref_to_public(info))


@router.delete("/api/tts/voices/{ref_id}")
async def delete_voice_ref(ref_id: str) -> ApiResponse:
    """Remove an uploaded voice locally + on the upstream."""
    try:
        removed = refs_mod.delete_ref(_voice_refs_dir(), ref_id)
    except storage.StorageError as exc:
        return ApiResponse.err(f"invalid ref_id: {exc}")
    # Best-effort upstream sync.
    try:
        await get_tts_client().delete_voice(ref_id)
    except TtsClientError as exc:
        logger.warning("upstream delete failed: %s", exc)
    if not removed:
        return ApiResponse.err("voice ref not found")
    return ApiResponse.ok({"deleted": ref_id})


@router.get("/api/tts/status")
async def get_tts_status() -> ApiResponse:
    """Lightweight lifecycle snapshot for the frontend status badge."""
    lifecycle = get_lifecycle()
    info = await lifecycle.status()
    return ApiResponse.ok({
        "state": info.state,
        "pid": info.pid,
        "boot_started_at": info.boot_started_at,
        "boot_elapsed_secs": info.boot_elapsed_secs,
        "boot_timeout_secs": info.boot_timeout_secs,
        "last_activity_at": info.last_activity_at,
        "inflight_synths": info.inflight_synths,
        "blocked_reason": info.blocked_reason,
        "error": info.error,
        "autostart": lifecycle.autostart,
    })


@router.get("/api/tts/config")
async def get_tts_config() -> ApiResponse:
    cfg = TtsConfig(
        output_dir=str(_output_dir()),
        max_chars=settings.tts_max_chars,
        default_voice=settings.tts_default_voice,
        sample_rate=24000,
        supported_formats=["wav"],
    )
    data = cfg.model_dump()
    # Phase 5 hint for the frontend: max long-form wall-clock + sentence cap.
    data["long_max_secs"] = settings.tts_long_max_secs
    return ApiResponse.ok(data)


# ---------------------------------------------------------------------------
# Synthesize
# ---------------------------------------------------------------------------

@router.post("/api/tts/synthesize")
async def synthesize(req: TtsSynthesizeRequest, request: Request):
    """Synthesize speech.

    - `save=True` (default): persist to `VOXTRAL_TTS_OUTPUT_DIR`, return an
      `ApiResponse` with the saved path/duration/etc. Long-form input is
      synthesized sentence-by-sentence and assembled into one WAV file.
    - `save=False`: stream a chunked `audio/wav` response straight to the
      client. Long-form input is streamed sentence-by-sentence with a
      single WAV header at the front (no inter-sentence headers).

    Both branches share the same sentence-splitting pipeline. A single
    request that has no terminators is treated as one sentence.
    """
    # ── Validate input ────────────────────────────────────────────────────
    text = req.text.strip()
    if not text:
        return _err("text is empty", req.save)
    if len(text) > settings.tts_max_chars:
        return _err(
            f"text exceeds max length ({len(text)} > {settings.tts_max_chars})",
            req.save,
        )

    # Voice resolution: cloned voices take precedence over the built-in
    # `voice` field.
    cloned_ref: refs_mod.VoiceRef | None = None
    if req.voice_ref_id:
        try:
            cloned_ref = refs_mod.get_ref(_voice_refs_dir(), req.voice_ref_id)
        except storage.StorageError as exc:
            return _err(f"invalid voice_ref_id: {exc}", req.save)
        if cloned_ref is None:
            return _err("voice_ref_id not found", req.save)
        voice = cloned_ref.id  # vllm-omni uses our uuid as the voice name
        task_type = "Base"
    else:
        voice = req.voice or settings.tts_default_voice
        if not voices.is_known_voice(voice):
            return _err(f"unknown voice: {voice!r}", req.save)
        task_type = "CustomVoice"

    # ── Sentence split (cheap; pure regex) ────────────────────────────────
    parts = text_split.split_sentences(text) or [text]

    # ── Phase 6/7: ensure the TTS subprocess is running in the right mode
    lifecycle = get_lifecycle()
    try:
        await lifecycle.ensure_started(task_type=task_type)
    except LifecycleError as exc:
        msg = f"TTS unavailable: {exc}"
        if exc.reason == "blocked":
            # Surface as 503 with a body the frontend can parse.
            if req.save:
                return ApiResponse.err(msg)
            return JSONResponse({"success": False, "error": msg}, status_code=503)
        return _err(msg, req.save)

    # ── If we just switched to Base mode, the in-memory upstream voice
    # store has been wiped. Re-upload the chosen reference so the synth
    # below can find it.
    if cloned_ref is not None:
        try:
            with open(cloned_ref.audio_path, "rb") as fh:
                encoded = fh.read()
            await get_tts_client().upload_voice(
                voice_name=cloned_ref.id,
                consent="user-confirmed via Starling /api/tts/voices/upload",
                audio_bytes=encoded,
                audio_mime="audio/wav",
                ref_text=cloned_ref.ref_text,
            )
        except (TtsClientError, OSError) as exc:
            logger.warning("re-syncing cloned voice to upstream failed: %s", exc)
            return _err(f"could not register cloned voice with upstream: {exc}", req.save)

    # ── Streaming branch (save=False) ─────────────────────────────────────
    if not req.save:
        return _stream_response(text_parts=parts, voice=voice, request=request)

    # ── Save-to-disk branch (save=True) ───────────────────────────────────
    out_dir = _output_dir()
    filename = req.save_filename or storage.auto_filename(voice)
    try:
        target = storage.safe_join(out_dir, filename)
    except storage.StorageError as exc:
        return ApiResponse.err(f"invalid filename: {exc}")

    if target.exists() and not req.overwrite:
        return ApiResponse.err(
            f"file already exists: {target.name} (pass overwrite=true to replace)"
        )

    client = get_tts_client()
    t0 = time.perf_counter()
    lifecycle.synth_started()
    try:
        if len(parts) == 1:
            # Single-sentence fast path — buffered synth (Phase 1 behaviour).
            result = await client.synthesize(text=parts[0], voice=voice, response_format="wav")
            audio_bytes = result.audio_bytes
            elapsed = result.elapsed_secs
        else:
            # Long-form save: stream sentence-by-sentence into memory, then
            # wrap the concatenated PCM in a real WAV (with proper sizes,
            # not the streaming placeholders).
            pcm_chunks: list[bytes] = []
            async for chunk in client.synthesize_stream_concat(
                text_parts=parts, voice=voice
            ):
                pcm_chunks.append(chunk)
            pcm = b"".join(pcm_chunks)
            audio_bytes = _pcm_to_wav(pcm)
            elapsed = time.perf_counter() - t0
    except TtsClientError as exc:
        logger.warning("TTS synth failed: %s", exc)
        return ApiResponse.err(f"TTS server error: {exc}")
    finally:
        lifecycle.synth_finished()

    try:
        bytes_written = storage.write_wav(
            target, audio_bytes, overwrite=req.overwrite
        )
    except storage.StorageError as exc:
        return ApiResponse.err(f"could not write file: {exc}")

    duration = _wav_duration_secs(audio_bytes)
    return ApiResponse.ok(
        TtsSynthesizeResponse(
            filename=target.name,
            path=str(target),
            bytes=bytes_written,
            voice=voice,
            sample_rate=24000,
            duration_secs=duration,
            elapsed_secs=elapsed,
        ).model_dump()
    )


def _pcm_to_wav(pcm: bytes, *, sample_rate: int = 24000, channels: int = 1) -> bytes:
    """Wrap raw mono int16 LE PCM in a real (non-streaming) WAV container.

    Used by the long-form save path to assemble one coherent file from
    sentence-by-sentence synthesis.
    """
    import struct

    bits_per_sample = 16
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    data_size = len(pcm)
    riff_size = 36 + data_size
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        riff_size,
        b"WAVE",
        b"fmt ",
        16, 1, channels, sample_rate, byte_rate, block_align, bits_per_sample,
        b"data",
        data_size,
    )
    return header + pcm


def _err(msg: str, save: bool):
    """Return an error in the shape the client expects.

    Save-mode callers parse JSON `{success:false, error}`; streaming-mode
    callers want a plain JSON 400 they can read off `response.text`.
    """
    if save:
        return ApiResponse.err(msg)
    return JSONResponse({"success": False, "error": msg}, status_code=400)


def _stream_response(
    *,
    text_parts: list[str],
    voice: str,
    request: Request,
) -> StreamingResponse:
    """Open the upstream PCM stream, prepend a streaming WAV header, return.

    Long-form input is synthesized sentence-by-sentence (one upstream call
    per part). A single WAV header is yielded at the front; subsequent
    parts contribute raw PCM only. We poll `request.is_disconnected()`
    between parts so a closed-tab stops upstream work cleanly.
    """
    deadline = time.perf_counter() + settings.tts_long_max_secs
    lifecycle = get_lifecycle()

    async def body() -> AsyncIterator[bytes]:
        lifecycle.synth_started()
        # Header first — browser parses it before any audio frames.
        yield wav.streaming_header(sample_rate=24000, channels=1, bits_per_sample=16)
        client = get_tts_client()
        try:
            for idx, part in enumerate(text_parts):
                if await request.is_disconnected():
                    logger.info("TTS stream: client disconnected at sentence %d/%d", idx, len(text_parts))
                    return
                if time.perf_counter() > deadline:
                    logger.warning(
                        "TTS stream: hit %.0fs wall-clock cap at sentence %d/%d",
                        settings.tts_long_max_secs, idx, len(text_parts),
                    )
                    return
                try:
                    async for chunk in client.synthesize_stream(text=part, voice=voice):
                        yield chunk
                except TtsClientError as exc:
                    if idx == 0:
                        # We've already yielded the header bytes, but we
                        # can still raise so Starlette closes the stream
                        # without further work. The browser sees an
                        # incomplete audio stream → harmless.
                        logger.warning("TTS upstream failed on sentence 1: %s", exc)
                        raise
                    # Skip later-sentence failures — partial is better than
                    # tearing down the whole stream.
                    logger.warning(
                        "TTS upstream failed on sentence %d/%d (skipping): %s",
                        idx + 1, len(text_parts), exc,
                    )
        except TtsClientError as exc:
            logger.warning("TTS upstream stream aborted: %s", exc)
        finally:
            lifecycle.synth_finished()

    return StreamingResponse(
        body(),
        media_type="audio/wav",
        headers={
            # Lets the browser kick off playback ASAP.
            "Cache-Control": "no-store",
            # Suggests a download name if the user opens the URL directly.
            "Content-Disposition": f'inline; filename="tts_{voice}.wav"',
        },
    )


# ---------------------------------------------------------------------------
# Output file management
# ---------------------------------------------------------------------------

@router.get("/api/tts/output")
async def list_outputs() -> ApiResponse:
    files = storage.list_outputs(_output_dir())
    return ApiResponse.ok(
        [TtsOutputFile(name=f.name, bytes=f.size_bytes, created_at=f.created_at).model_dump() for f in files]
    )


@router.get("/api/tts/output/{filename}")
async def download_output(filename: str) -> FileResponse:
    try:
        target = storage.safe_join(_output_dir(), filename)
    except storage.StorageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not target.is_file():
        raise HTTPException(status_code=404, detail="file not found")

    return FileResponse(
        path=str(target),
        media_type="audio/wav",
        filename=target.name,
    )


@router.delete("/api/tts/output/{filename}")
async def delete_output(filename: str) -> ApiResponse:
    try:
        removed = storage.delete_output(_output_dir(), filename)
    except storage.StorageError as exc:
        return ApiResponse.err(f"invalid filename: {exc}")

    if not removed:
        return ApiResponse.err("file not found")
    return ApiResponse.ok({"deleted": filename})
