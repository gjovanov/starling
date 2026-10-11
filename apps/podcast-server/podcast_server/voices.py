"""Named voice prompts (voices/<name>.wav + the voices.json manifest).

A voice is a short reference clip VibeVoice clones the speaker from. The
shipped ones are synthetic Kokoro-82M clips (scripts/make_kokoro_voices.py);
the manifest records how each was made. A WAV dropped into the directory
without a manifest entry is still usable under its file stem.
"""

from __future__ import annotations

import json
import logging
import re
import wave
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

VOICE_SAMPLE_RATE = 24000
_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


@dataclass(frozen=True)
class Voice:
    name: str
    path: Path
    duration_secs: float
    gender: str | None = None
    accent: str | None = None
    source: dict = field(default_factory=dict)

    def describe(self) -> dict:
        return {
            "name": self.name,
            "file": self.path.name,
            "duration_secs": self.duration_secs,
            "sample_rate": VOICE_SAMPLE_RATE,
            "gender": self.gender,
            "accent": self.accent,
            "source": self.source,
        }


def _wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        if w.getnchannels() != 1 or w.getsampwidth() != 2 or w.getframerate() != VOICE_SAMPLE_RATE:
            raise ValueError(f"{path.name}: voice prompts must be 24 kHz mono 16-bit PCM")
        return round(w.getnframes() / VOICE_SAMPLE_RATE, 2)


def load_voices(voices_dir: str | Path) -> dict[str, Voice]:
    """Scan the voices directory. Cheap — called per request so new voices
    appear without a restart. Unusable files are skipped with a warning
    rather than breaking every request."""
    root = Path(voices_dir)
    manifest: dict = {}
    manifest_path = root / "voices.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    voices: dict[str, Voice] = {}
    for wav in sorted(root.glob("*.wav")):
        name = wav.stem.lower()
        if not _NAME.match(name):
            logger.warning("[voices] skipping %s: name must match %s", wav.name, _NAME.pattern)
            continue
        try:
            duration = _wav_duration(wav)
        except (ValueError, wave.Error, EOFError) as exc:
            logger.warning("[voices] skipping %s: %s", wav.name, exc)
            continue
        meta = manifest.get(name, {})
        voices[name] = Voice(
            name=name,
            path=wav.resolve(),
            duration_secs=duration,
            gender=meta.get("gender"),
            accent=meta.get("accent"),
            source=meta.get("source", {}),
        )
    return voices
