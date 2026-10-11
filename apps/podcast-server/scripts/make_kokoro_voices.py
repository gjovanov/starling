"""Build the licence-clean voice prompts in voices/ from Kokoro-82M.

VibeVoice clones each speaker from a short reference clip. These clips are
synthetic (Kokoro-82M, Apache-2.0) reading neutral prose, so no real
person's voice and no CC BY-NC preset is involved.

Runs in the Kokoro venv, not podcast-server's (kokoro + espeak-ng, CPU):

    ~/kokoro-venv/bin/python apps/podcast-server/scripts/make_kokoro_voices.py

Writes voices/<name>.wav (24 kHz mono 16-bit) and voices/voices.json (the
provenance manifest the server reads).
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import warnings
import wave
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

KOKORO_REPO = "hexgrad/Kokoro-82M"
SAMPLE_RATE = 24000
VOICES_DIR = Path(__file__).resolve().parent.parent / "voices"

VOICES = {
    "michael": {
        "kokoro_voice": "am_michael",
        "lang_code": "a",
        "gender": "male",
        "accent": "American English",
        "text": (
            "The library opens at nine in the morning and closes just after six. "
            "Most visitors come in the afternoon, when the light through the tall "
            "windows is soft and warm. On the second floor there is a long table "
            "by the wall, where people sit quietly with their books and notes."
        ),
    },
    "emma": {
        "kokoro_voice": "bf_emma",
        "lang_code": "b",
        "gender": "female",
        "accent": "British English",
        "text": (
            "The village sits at the edge of a wide valley, with a small river "
            "running beside the main road. In spring the fields turn green almost "
            "overnight, and the old stone bridge is busy with walkers and cyclists. "
            "Further up the hill, a narrow path leads through the woods to a quiet "
            "clearing above the valley."
        ),
    },
}


def _kokoro_revision() -> str:
    ref = Path.home() / ".cache/huggingface/hub/models--hexgrad--Kokoro-82M/refs/main"
    return ref.read_text().strip() if ref.exists() else "unknown"


def _trim(audio: np.ndarray, keep_secs: float = 0.15, threshold: float = 1e-3) -> np.ndarray:
    """Trim leading/trailing silence down to `keep_secs` on each side."""
    loud = np.flatnonzero(np.abs(audio) > threshold)
    if loud.size == 0:
        return audio
    pad = int(keep_secs * SAMPLE_RATE)
    return audio[max(0, loud[0] - pad): min(audio.size, loud[-1] + pad)]


def _write_wav(path: Path, audio: np.ndarray) -> None:
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())


def main() -> None:
    from kokoro import KPipeline

    VOICES_DIR.mkdir(parents=True, exist_ok=True)
    revision = _kokoro_revision()
    manifest: dict = {}
    pipelines: dict[str, KPipeline] = {}
    for name, spec in VOICES.items():
        lang = spec["lang_code"]
        if lang not in pipelines:
            pipelines[lang] = KPipeline(lang_code=lang, repo_id=KOKORO_REPO)
        chunks = [np.asarray(a, dtype=np.float32)
                  for _, _, a in pipelines[lang](spec["text"], voice=spec["kokoro_voice"], speed=1.0)]
        audio = _trim(np.concatenate(chunks))
        out = VOICES_DIR / f"{name}.wav"
        _write_wav(out, audio)
        seconds = round(audio.size / SAMPLE_RATE, 2)
        manifest[name] = {
            "file": out.name,
            "gender": spec["gender"],
            "accent": spec["accent"],
            "duration_secs": seconds,
            "sample_rate": SAMPLE_RATE,
            "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
            "source": {
                "engine": "kokoro",
                "kokoro_version": importlib.metadata.version("kokoro"),
                "model": KOKORO_REPO,
                "model_revision": revision,
                "model_license": "Apache-2.0",
                "voice": spec["kokoro_voice"],
                "lang_code": spec["lang_code"],
                "speed": 1.0,
                "text": spec["text"],
            },
        }
        print(f"{name}: {seconds:5.2f}s  ({spec['kokoro_voice']})")
    (VOICES_DIR / "voices.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {VOICES_DIR / 'voices.json'}")


if __name__ == "__main__":
    main()
