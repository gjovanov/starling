# Starling blog series — shared assets

This directory holds everything that's reused across the four blog posts:

```
_shared/
├── README.md                 # this file
├── scripts/                  # bash + playwright config
│   ├── package.json
│   ├── playwright.config.ts
│   ├── tsconfig.json
│   ├── record-video.sh       # one demo
│   ├── record-all-demos.sh   # all four (interactive)
│   └── render-mermaid.sh     # .mmd → .png for Medium variants
├── e2e/
│   ├── overlay.ts            # caption-overlay helpers (ported from tickytack)
│   ├── record-asr-demo.spec.ts
│   ├── record-tts-demo.spec.ts
│   └── transcriptions/       # per-post on-screen captions
└── media/
    ├── broadcast_1_90s.wav   # 90 s ASR demo clip (ffmpeg -t 90 of media/broadcast_1.wav)
    └── broadcast_1_excerpt.txt  # ~150-word TTS excerpt (matches the clip's content)
```

## Recording a demo video

1. **Install Playwright once**

   ```bash
   cd docs/blog/_shared/scripts
   bun install                     # or `npm install`
   bunx playwright install chromium
   ```

2. **Start the matching server** in another terminal — see `CLAUDE.md` for the
   exact commands. Quick reference:

   | Post slug              | Server         | Required init    | Port |
   |------------------------|----------------|------------------|------|
   | `voxtral-asr-vllm`     | vllm-server    | `./init.sh`      | 8090 |
   | `voxtral-asr-burn`     | burn-server    | `./init.sh`      | 8091 |
   | `voxtral-tts-vllm`     | vllm-server    | `./init.sh --tts`| 8090 |
   | `voxtral-tts-burn`     | burn-server    | `./init.sh --tts`| 8091 |

3. **Record one post**

   ```bash
   docs/blog/_shared/scripts/record-video.sh --post voxtral-asr-vllm
   ```

   The wrapper validates the server is up, runs the right Playwright spec
   (`record-asr-demo.spec.ts` or `record-tts-demo.spec.ts`) with the right
   `BACKEND` env var, finds the resulting `.webm` under `test-results/`,
   transcodes it to MP4 with `libx264 crf 18 preset slow + faststart`, and
   drops the result at `docs/blog/<post>/video.mp4`. It also `ffprobe`s the
   duration and warns if you blew past 125 s.

4. **Record all four**

   ```bash
   docs/blog/_shared/scripts/record-all-demos.sh
   ```

   Pauses between posts so you can switch which server is running. Press
   ENTER to record, `s` then ENTER to skip a post.

## Rendering diagrams

Long-form posts (`blog-post.md`) embed mermaid sources inline — GitHub
renders these natively, no preprocessing needed. Medium variants
(`blog-post-medium.md`) reference pre-rendered PNGs, because Medium does
not support mermaid.

```bash
npm install -g @mermaid-js/mermaid-cli
docs/blog/_shared/scripts/render-mermaid.sh                 # all four posts
docs/blog/_shared/scripts/render-mermaid.sh voxtral-asr-vllm  # just one
```

## Captions

Each spec reads its scene list from `e2e/transcriptions/{asr,tts}-{vllm,burn}.json`.
Each entry has `{scene, text, duration}` where `duration` is in milliseconds.
Tweak the durations if the matching part of the captured video is slower or
faster than you expected. The 7-scene ASR script is calibrated for ~90 s of
audio; the 6-scene TTS script is calibrated for ~75–100 s of synthesis.

## After recording

1. Upload each MP4 to YouTube (manual step).
2. Replace the `<!-- TODO: YouTube URL -->` placeholder in the matching
   `docs/blog/<post>/blog-post-medium.md` with the YouTube link.
3. The long-form `blog-post.md` embeds the MP4 directly from the repo, so it
   needs no follow-up edit (GitHub plays MP4 ≤ 100 MB inline).
