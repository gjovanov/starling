"""Download the pinned model revision into the Hugging Face cache.

    python -m podcast_server.download

Idempotent: snapshot_download skips files already in the cache.
"""

from __future__ import annotations

import sys
from pathlib import Path

from .config import settings


def main() -> int:
    from huggingface_hub import snapshot_download

    print(f"[Download] {settings.model_id} @ {settings.model_revision}", flush=True)
    path = Path(snapshot_download(settings.model_id, revision=settings.model_revision))
    size = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    print(f"[Download] {path}  ({size / 1e9:.2f} GB)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
