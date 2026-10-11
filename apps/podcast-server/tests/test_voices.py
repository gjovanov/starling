import hashlib
import json
from pathlib import Path

from podcast_server.voices import load_voices
from tests.conftest import write_voice

SHIPPED = Path(__file__).resolve().parent.parent / "voices"


def test_shipped_voices_are_valid_prompts():
    """The committed michael/emma prompts: right format, 15–20 s, provenance intact."""
    voices = load_voices(SHIPPED)
    manifest = json.loads((SHIPPED / "voices.json").read_text())
    assert {"michael", "emma"} <= set(voices)
    for name in ("michael", "emma"):
        v = voices[name]
        assert 15.0 <= v.duration_secs <= 20.0
        assert hashlib.sha256(v.path.read_bytes()).hexdigest() == manifest[name]["sha256"]
        assert manifest[name]["source"]["engine"] == "kokoro"
        assert manifest[name]["source"]["model_license"] == "Apache-2.0"
    assert voices["michael"].gender == "male"
    assert voices["emma"].gender == "female"


def test_manifestless_wav_is_usable_under_its_stem(tmp_path):
    write_voice(tmp_path / "Zed.wav")
    v = load_voices(tmp_path)["zed"]
    assert v.duration_secs == 1.0 and v.gender is None and v.describe()["file"] == "Zed.wav"


def test_unusable_files_are_skipped(tmp_path):
    write_voice(tmp_path / "ok.wav")
    write_voice(tmp_path / "has space.wav")
    write_voice(tmp_path / "wrongrate.wav", rate=16000)
    (tmp_path / "broken.wav").write_bytes(b"not a wav")
    assert set(load_voices(tmp_path)) == {"ok"}
