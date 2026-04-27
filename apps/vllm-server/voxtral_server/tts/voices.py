"""Canonical Voxtral-4B-TTS voice catalog.

Source of truth: `models/cache/tts/params.json` →
`multimodal.audio_tokenizer_args.voice` (a `name → index` map). The 20 voices
are baked into the Mistral checkpoint and don't change unless the upstream
model is replaced. We keep them here as a static list so the API layer doesn't
have to read params.json on every request.
"""

from __future__ import annotations

from typing import Final

from pydantic import BaseModel


class VoiceInfo(BaseModel):
    id: str
    display_name: str
    language: str
    language_code: str  # ISO 639-1 (or "ar"/"hi" for the multi-script ones)
    gender: str         # "male" | "female"


VOICES: Final[list[VoiceInfo]] = [
    VoiceInfo(id="casual_female",    display_name="Casual (Female)",    language="English",     language_code="en", gender="female"),
    VoiceInfo(id="casual_male",      display_name="Casual (Male)",      language="English",     language_code="en", gender="male"),
    VoiceInfo(id="cheerful_female",  display_name="Cheerful (Female)",  language="English",     language_code="en", gender="female"),
    VoiceInfo(id="neutral_female",   display_name="Neutral (Female)",   language="English",     language_code="en", gender="female"),
    VoiceInfo(id="neutral_male",     display_name="Neutral (Male)",     language="English",     language_code="en", gender="male"),
    VoiceInfo(id="de_female",        display_name="German (Female)",    language="German",      language_code="de", gender="female"),
    VoiceInfo(id="de_male",          display_name="German (Male)",      language="German",      language_code="de", gender="male"),
    VoiceInfo(id="es_female",        display_name="Spanish (Female)",   language="Spanish",     language_code="es", gender="female"),
    VoiceInfo(id="es_male",          display_name="Spanish (Male)",     language="Spanish",     language_code="es", gender="male"),
    VoiceInfo(id="fr_female",        display_name="French (Female)",    language="French",      language_code="fr", gender="female"),
    VoiceInfo(id="fr_male",          display_name="French (Male)",      language="French",      language_code="fr", gender="male"),
    VoiceInfo(id="it_female",        display_name="Italian (Female)",   language="Italian",     language_code="it", gender="female"),
    VoiceInfo(id="it_male",          display_name="Italian (Male)",     language="Italian",     language_code="it", gender="male"),
    VoiceInfo(id="nl_female",        display_name="Dutch (Female)",     language="Dutch",       language_code="nl", gender="female"),
    VoiceInfo(id="nl_male",          display_name="Dutch (Male)",       language="Dutch",       language_code="nl", gender="male"),
    VoiceInfo(id="pt_female",        display_name="Portuguese (Female)", language="Portuguese", language_code="pt", gender="female"),
    VoiceInfo(id="pt_male",          display_name="Portuguese (Male)",  language="Portuguese",  language_code="pt", gender="male"),
    VoiceInfo(id="ar_male",          display_name="Arabic (Male)",      language="Arabic",      language_code="ar", gender="male"),
    VoiceInfo(id="hi_female",        display_name="Hindi (Female)",     language="Hindi",       language_code="hi", gender="female"),
    VoiceInfo(id="hi_male",          display_name="Hindi (Male)",       language="Hindi",       language_code="hi", gender="male"),
]

VOICE_IDS: Final[frozenset[str]] = frozenset(v.id for v in VOICES)
DEFAULT_VOICE_ID: Final[str] = "casual_male"


def is_known_voice(voice_id: str) -> bool:
    return voice_id in VOICE_IDS
