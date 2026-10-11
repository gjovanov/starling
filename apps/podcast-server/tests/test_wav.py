import io
import struct
import wave

import numpy as np

from podcast_server.wav import encode_wav


def _chunks(data: bytes) -> list[bytes]:
    pos, ids = 12, []
    while pos < len(data):
        cid, size = data[pos:pos + 4], struct.unpack("<I", data[pos + 4:pos + 8])[0]
        ids.append(cid)
        pos += 8 + size + (size % 2)
    assert pos == len(data), "chunk sizes must add up to the file length"
    return ids


def test_roundtrip_through_stdlib_wave():
    x = np.linspace(-0.5, 0.5, 2400, dtype=np.float32)
    data = encode_wav(x, 24000, software="sw", comment="AI-generated speech.")
    with wave.open(io.BytesIO(data)) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()) == (1, 2, 24000, 2400)
        pcm = np.frombuffer(w.readframes(2400), dtype="<i2")
    assert np.allclose(pcm / 32767.0, x, atol=1 / 32767)


def test_riff_layout_and_info_chunk():
    data = encode_wav(np.zeros(10, np.float32), 24000, software="abc", comment="AI-generated speech.")
    assert data[:4] == b"RIFF" and data[8:12] == b"WAVE"
    assert struct.unpack("<I", data[4:8])[0] == len(data) - 8
    assert _chunks(data) == [b"fmt ", b"LIST", b"data"]
    assert b"ISFT" in data and b"ICMT" in data and b"AI-generated speech." in data


def test_odd_length_tag_is_padded():
    data = encode_wav(np.zeros(4, np.float32), 24000, comment="odd")   # 'odd\0' is even
    _chunks(data)
    data = encode_wav(np.zeros(4, np.float32), 24000, comment="even")  # 'even\0' is odd → pad
    _chunks(data)


def test_clips_out_of_range_samples():
    data = encode_wav(np.array([2.0, -2.0], np.float32), 24000)
    assert list(np.frombuffer(data[-4:], dtype="<i2")) == [32767, -32767]


def test_no_info_chunk_without_tags():
    assert _chunks(encode_wav(np.zeros(4, np.float32), 24000)) == [b"fmt ", b"data"]
