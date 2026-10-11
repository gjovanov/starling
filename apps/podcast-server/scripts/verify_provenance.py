"""Prove the served checkpoint is Microsoft's VibeVoice-1.5B, tensor for tensor.

The served weights come from vibevoice/VibeVoice-1.5B-hf, an UNOFFICIAL
community org (its members include the Hugging Face engineer who ported
VibeVoice to transformers). They are documented as the output of
transformers' own convert_vibevoice_to_hf.py run on microsoft/VibeVoice-1.5B
— a pure key rename. This script checks that claim instead of trusting it:

  1. streams each microsoft/VibeVoice-1.5B shard at MS_REFERENCE_REVISION
     straight off the Hub (nothing is written to disk),
  2. maps every tensor name through the converter's own map_old_key_to_new,
  3. compares it with the locally cached served checkpoint: identical bytes
     when the dtypes match, identical values after the converter's dtype
     cast when they do not.

    .venv/bin/python scripts/verify_provenance.py     # streams ~5.4 GB

Exit 0 only if every Microsoft tensor is present and identical and every
served tensor is accounted for.
"""

from __future__ import annotations

import hashlib
import json
import struct
import sys
import time
from pathlib import Path

import httpx
from huggingface_hub import hf_hub_url, snapshot_download

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from podcast_server.config import MS_REFERENCE_REPO, MS_REFERENCE_REVISION, settings  # noqa: E402

CHUNK = 8 << 20


class ByteStream:
    """Exact-size consumption of an httpx byte iterator."""

    def __init__(self, chunks) -> None:
        self._it = iter(chunks)
        self._buf = b""
        self.consumed = 0

    def _pull(self) -> None:
        chunk = next(self._it, None)
        if chunk is None:
            raise EOFError("stream ended early")
        self._buf += chunk

    def read(self, n: int) -> bytes:
        parts, need = [], n
        while need:
            if not self._buf:
                self._pull()
            take = self._buf[:need]
            self._buf = self._buf[len(take):]
            parts.append(take)
            need -= len(take)
        self.consumed += n
        return b"".join(parts)

    def sha256(self, n: int) -> str:
        h, need = hashlib.sha256(), n
        while need:
            if not self._buf:
                self._pull()
            take = self._buf[:need]
            self._buf = self._buf[len(take):]
            h.update(take)
            need -= len(take)
        self.consumed += n
        return h.hexdigest()


def _header(raw: bytes) -> dict:
    meta = json.loads(raw)
    meta.pop("__metadata__", None)
    return meta


def served_tensors(snapshot: Path) -> dict[str, dict]:
    """name → {dtype, shape, sha256, file, offsets} for the local checkpoint."""
    out: dict[str, dict] = {}
    for shard in sorted(snapshot.glob("*.safetensors")):
        with open(shard, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header = _header(f.read(n))
            base = 8 + n
            for name, info in header.items():
                start, end = info["data_offsets"]
                f.seek(base + start)
                h, need = hashlib.sha256(), end - start
                while need:
                    block = f.read(min(CHUNK, need))
                    h.update(block)
                    need -= len(block)
                out[name] = {"dtype": info["dtype"], "shape": info["shape"], "sha256": h.hexdigest(),
                             "file": shard, "offsets": (base + start, base + end)}
    return out


def _values_equal(ms_bytes: bytes, ms_dtype: str, served: dict) -> bool:
    """Compare after casting the Microsoft tensor to the served dtype (the
    converter loads into a model of that dtype)."""
    import torch

    dt = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "F64": torch.float64}
    cast = torch.frombuffer(bytearray(ms_bytes), dtype=dt[ms_dtype]).to(dt[served["dtype"]])
    start, end = served["offsets"]
    with open(served["file"], "rb") as f:
        f.seek(start)
        raw = f.read(end - start)
    # bit-level: exact, and NaN-safe
    return cast.contiguous().view(torch.uint8).numpy().tobytes() == raw


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from vibevoice_key_map import map_old_key_to_new   # the converter's own mapping, vendored

    snapshot = Path(snapshot_download(settings.model_id, revision=settings.model_revision,
                                      local_files_only=True))
    print(f"served:    {settings.model_id} @ {settings.model_revision}", flush=True)
    print(f"reference: {MS_REFERENCE_REPO} @ {MS_REFERENCE_REVISION}", flush=True)
    t0 = time.monotonic()
    served = served_tensors(snapshot)
    print(f"hashed {len(served)} served tensors in {time.monotonic() - t0:.0f}s", flush=True)

    with httpx.Client(follow_redirects=True, timeout=httpx.Timeout(60.0, read=300.0)) as http:
        index = http.get(hf_hub_url(MS_REFERENCE_REPO, "model.safetensors.index.json",
                                    revision=MS_REFERENCE_REVISION)).raise_for_status().json()
        shards = sorted(set(index["weight_map"].values()))
        identical = cast_identical = 0
        mismatched, missing, matched = [], [], set()
        dtypes: dict[str, int] = {}
        for shard in shards:
            url = hf_hub_url(MS_REFERENCE_REPO, shard, revision=MS_REFERENCE_REVISION)
            t1 = time.monotonic()
            with http.stream("GET", url) as resp:
                resp.raise_for_status()
                s = ByteStream(resp.iter_bytes(CHUNK))
                n = struct.unpack("<Q", s.read(8))[0]
                header = _header(s.read(n))
                pos = 0
                for name, info in sorted(header.items(), key=lambda kv: kv[1]["data_offsets"][0]):
                    start, end = info["data_offsets"]
                    if start > pos:
                        s.read(start - pos)
                    new = map_old_key_to_new(name)
                    dtypes[info["dtype"]] = dtypes.get(info["dtype"], 0) + 1
                    target = served.get(new)
                    if target is None:
                        s.sha256(end - start)
                        missing.append(f"{name} -> {new}")
                    elif info["dtype"] == target["dtype"]:
                        same = s.sha256(end - start) == target["sha256"] and info["shape"] == target["shape"]
                        identical += same
                        if not same:
                            mismatched.append(new)
                        matched.add(new)
                    else:
                        same = info["shape"] == target["shape"] and _values_equal(
                            s.read(end - start), info["dtype"], target)
                        cast_identical += same
                        if not same:
                            mismatched.append(f"{new} ({info['dtype']}->{target['dtype']})")
                        matched.add(new)
                    pos = end
            print(f"  {shard}: {len(header)} tensors, {s.consumed / 1e9:.2f} GB in "
                  f"{time.monotonic() - t1:.0f}s", flush=True)

    unaccounted = sorted(set(served) - matched)
    report = {
        "reference_dtypes": dtypes,
        "identical_bytes": identical,
        "identical_after_cast": cast_identical,
        "mismatched": mismatched,
        "missing_in_served": missing,
        "served_not_in_reference": unaccounted,
    }
    print(json.dumps(report, indent=1))
    ok = not mismatched and not missing and not unaccounted
    print("PROVENANCE OK: the served checkpoint is microsoft/VibeVoice-1.5B, renamed"
          if ok else "PROVENANCE FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
