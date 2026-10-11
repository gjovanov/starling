"""The AI-generation label stamped into every rendered WAV (INFO chunk).

Microsoft's model card asks that AI use be disclosed when sharing
VibeVoice output; the label travels with the file. It is metadata — it
does not touch the audio.
"""

from __future__ import annotations

from . import __version__

SOFTWARE = f"starling podcast-server {__version__} (VibeVoice via transformers)"


def comment(model_label: str, speakers: list[str] | tuple[str, ...]) -> str:
    return (f"AI-generated speech. Model: {model_label} "
            f"(VibeVoice-1.5B, Microsoft Research, MIT). "
            f"Synthetic voice prompts: {', '.join(speakers)}.")
