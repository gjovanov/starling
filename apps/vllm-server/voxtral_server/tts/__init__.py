"""Voxtral TTS subsystem — proxies to a separate vllm-omni process.

The TTS model itself is hosted by `start-vllm-tts.sh` on a separate port (8002
by default). This package contains the FastAPI-side glue: voice catalog,
HTTP client, output-file storage, and request validation.
"""
