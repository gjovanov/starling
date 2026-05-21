# The Starling Blog Series

Four posts about building real-time speech I/O on a single laptop GPU. Two models — Voxtral-Mini-4B-Realtime for ASR, Voxtral-4B-TTS-2603 for TTS — running on two servers (Python and Rust), behind one shared frontend.

The series is best read in order: each post assumes the architecture from the one before.

| # | Topic | Long form | Medium variant | Demo video |
|---|---|---|---|---|
| 1 | **Voxtral-Mini-4B on vllm-server** — live German captions via vLLM's realtime API, FastAPI, FFmpeg, WebRTC | [blog-post.md](voxtral-asr-vllm/blog-post.md) | [blog-post-medium.md](voxtral-asr-vllm/blog-post-medium.md) | [video.mp4](voxtral-asr-vllm/video.mp4) |
| 2 | **Voxtral-Mini-4B on burn-server** — native Rust + candle, FlashAttention v2, speculative decoding, and a five-month WebGPU dead end | [blog-post.md](voxtral-asr-burn/blog-post.md) | [blog-post-medium.md](voxtral-asr-burn/blog-post-medium.md) | [video.mp4](voxtral-asr-burn/video.mp4) |
| 3 | **Voxtral-4B-TTS on vllm-server** — 20 voices, streaming WAV, GPU lifecycle, four frontend DSP fixes | [blog-post.md](voxtral-tts-vllm/blog-post.md) | [blog-post-medium.md](voxtral-tts-vllm/blog-post-medium.md) | [video.mp4](voxtral-tts-vllm/video.mp4) |
| 4 | **Voxtral-4B-TTS on burn-server** — 386 BF16 tensors ported to native Rust, bit-exact flow matching, no Python in production | [blog-post.md](voxtral-tts-burn/blog-post.md) | [blog-post-medium.md](voxtral-tts-burn/blog-post-medium.md) | [video.mp4](voxtral-tts-burn/video.mp4) |

Each video is ≤ 2 minutes. ASR demos transcribe a 90-second clip of `media/broadcast_1.wav` (Salzburg news, German); TTS demos synthesise the matching ~150-word excerpt of `media/broadcast_1.txt`. Same source content across both modalities, so the audio in and audio out can be compared side-by-side.

## Repository structure for this series

```
docs/blog/
├── README.md                     # this file
├── _shared/
│   ├── README.md                 # full guide to rendering + recording
│   ├── scripts/                  # bash wrappers + Playwright config
│   ├── e2e/                      # spec files + caption JSONs
│   └── media/                    # broadcast_1_90s.wav + excerpt.txt
└── voxtral-{asr,tts}-{vllm,burn}/
    ├── blog-post.md              # long form (inline mermaid)
    ├── blog-post-medium.md       # short form (PNG diagrams + YouTube link)
    ├── diagrams/*.mmd            # mermaid sources
    └── video.mp4                 # ≤ 2 min demo recording
```

## Rendering the diagrams

Long-form posts embed mermaid blocks inline — GitHub renders them natively. Medium variants reference pre-rendered PNGs (Medium does not support mermaid).

```bash
npm install -g @mermaid-js/mermaid-cli
docs/blog/_shared/scripts/render-mermaid.sh
```

## Recording the demo videos

See [`_shared/README.md`](_shared/README.md) for the full guide. Quick version:

```bash
cd docs/blog/_shared/scripts
bun install
bunx playwright install chromium

# Then start the right server (CLAUDE.md has the commands), and:
./record-video.sh --post voxtral-asr-vllm
# or all four interactively:
./record-all-demos.sh
```

The wrapper validates the server is up, runs the matching Playwright spec, transcodes the captured WebM to MP4 (`libx264 crf 18 + faststart`), and drops the result at `docs/blog/<post>/video.mp4`.

## Publishing checklist

1. Render diagrams to PNG (`render-mermaid.sh`).
2. Record all four demo videos (`record-all-demos.sh`).
3. Upload each MP4 to YouTube.
4. Replace the `<!-- TODO: YouTube URL -->` placeholder in each `blog-post-medium.md`.
5. Long-form posts embed the MP4 directly from the repo — no edit needed.
6. Copy `blog-post-medium.md` into Medium / dev.to / your platform of choice.

## Related projects

- [parakeet-rs](https://github.com/gjovanov/parakeet-rs) — sibling Rust ASR server using ONNX (Parakeet TDT, Canary 1B) and whisper.cpp. Same frontend, same API contract.
- [voxtral-mini-realtime-rs](https://github.com/TrevorS/voxtral-mini-realtime-rs) — reference Rust Voxtral inference with Burn/WebGPU. burn-server's ASR path is derived from this.
- [voxtral.c](https://github.com/antirez/voxtral.c) — pure-C reference implementation by antirez. candle-native's architecture matches it byte-for-byte.
