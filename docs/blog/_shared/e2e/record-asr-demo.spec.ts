/**
 * ASR demo recording (Posts 1 & 2 of the Starling blog series).
 *
 * Runs against a live server (`vllm-server` on :8090 or `burn-server` on :8091),
 * uploads `broadcast_1_90s.wav` via the public REST API, creates a transcription
 * session, and records the live caption pane while narrating with an on-screen
 * caption overlay. Total runtime under 120 s.
 *
 * Usage:
 *   BACKEND=vllm playwright test record-asr-demo.spec.ts
 *   BACKEND=burn playwright test record-asr-demo.spec.ts
 *
 * The wrapper script `scripts/record-video.sh --post <slug>` sets BACKEND for you.
 */
import { test, expect, type Page } from '@playwright/test';
import * as fs from 'node:fs';
import * as path from 'node:path';

import {
  injectOverlay,
  caption,
  backendBaseUrl,
  backendLabel,
  type Scene,
} from './overlay';

const AUDIO_PATH = path.resolve(
  __dirname,
  '..',
  'media',
  process.env.AUDIO_FILE ?? 'broadcast_1_90s.wav',
);

const BASE_URL = backendBaseUrl();
const BACKEND = backendLabel();

const SCENES_PATH = path.resolve(
  __dirname,
  'transcriptions',
  BACKEND === 'burn' ? 'asr-burn.json' : 'asr-vllm.json',
);

const scenes = JSON.parse(fs.readFileSync(SCENES_PATH, 'utf8')) as Scene[];

async function uploadAudioIfMissing(baseUrl: string): Promise<string> {
  // The server exposes `GET /api/media` to list known files and
  // `POST /api/media/upload` (multipart) to add a new one. Responses are
  // wrapped in `{success, data, error}` on this stack. We upload the audio
  // file if it isn't already there, then return its filename.
  const filename = path.basename(AUDIO_PATH);
  const listResp: unknown = await fetch(`${baseUrl}/api/media`).then((r) => r.json());
  const itemsRaw =
    (listResp as { data?: unknown; media?: unknown })?.data ??
    (listResp as { media?: unknown })?.media ??
    listResp;
  const items: Array<{ filename?: string; name?: string; id?: string }> = Array.isArray(itemsRaw)
    ? (itemsRaw as Array<{ filename?: string; name?: string; id?: string }>)
    : [];
  const already = items.some(
    (m) => m.filename === filename || m.name === filename || m.id === filename,
  );
  if (already) return filename;

  const bytes = fs.readFileSync(AUDIO_PATH);
  const fd = new FormData();
  fd.set('file', new File([new Uint8Array(bytes)], filename, { type: 'audio/wav' }));
  const res = await fetch(`${baseUrl}/api/media/upload`, { method: 'POST', body: fd });
  if (!res.ok) {
    throw new Error(`Upload failed: ${res.status} ${await res.text()}`);
  }
  return filename;
}

async function selectMediaFile(page: Page, filename: string): Promise<void> {
  // Wait for the media-select <select> to populate, then choose our file.
  // The shared frontend renders options as:
  //   <option value="<id>">{filename} ({duration})</option>
  // where `id` is the filename without its extension. So we match on the id
  // (stem) or substring of the visible text.
  const stem = filename.replace(/\.[^.]+$/, '');
  const select = page.locator('#media-select');
  await expect(select).toBeVisible();
  await expect
    .poll(async () => (await select.locator('option').count()) > 1, { timeout: 15_000 })
    .toBeTruthy();

  const options = await select.locator('option').all();
  for (const opt of options) {
    const txt = (await opt.textContent())?.trim() ?? '';
    const val = (await opt.getAttribute('value')) ?? '';
    if (
      val === stem ||
      val === filename ||
      txt.includes(filename) ||
      txt.includes(stem)
    ) {
      await select.selectOption({ value: val });
      return;
    }
  }
  throw new Error(
    `Could not find ${filename} in #media-select (looked for stem=${stem}).`,
  );
}

test.describe('Starling ASR demo', () => {
  test(`record live transcription (${BACKEND})`, async ({ page }) => {
    test.setTimeout(220_000);

    const filename = await uploadAudioIfMissing(BASE_URL);

    // ----- Scene 1: arrival -----
    await page.goto(`${BASE_URL}/`);
    await expect(page.locator('#session-panel')).toBeVisible();
    const tStart = Date.now();
    await injectOverlay(page);
    await page.waitForTimeout(600);
    await caption(page, scenes, 1);

    // Switch to "Create New" tab in the STT pane (STT is the default top-tab).
    await page.click('.session-tab[data-tab="create"]');
    await expect(page.locator('#create-content')).toBeVisible();
    await page.waitForTimeout(400);
    await caption(page, scenes, 2);

    // ----- Scene 3: choose the media file + force German language -----
    // "Media Files" is the default source-tab, so we don't have to click it.
    await selectMediaFile(page, filename);

    // Force German language so transcription matches the audio content
    // (the dropdown often defaults to `de` already; explicit is safer).
    const langSelect = page.locator('#language-select');
    if (await langSelect.isVisible({ timeout: 2000 }).catch(() => false)) {
      await langSelect.selectOption({ value: 'de' }).catch(async () => {
        // Some servers expose language as a label; fall back to label match.
        await langSelect.selectOption({ label: 'German' });
      });
    }

    await page.waitForTimeout(500);
    await caption(page, scenes, 3);

    // ----- Scene 4: create + connect to the session -----
    await page.click('#create-session-btn');

    // Frontend auto-selects the new session and switches to the Sessions tab.
    // We still need to click Connect to open the WebSocket and start receiving
    // captions.
    const connectBtn = page.locator('#connect-btn');
    await expect(connectBtn).toBeEnabled({ timeout: 15_000 });
    await connectBtn.click();

    // Stamp the audio-start moment so the wrapper can mux broadcast_1_90s.wav
    // at the right offset. The video frame timeline is approximately page-creation
    // → page.goto → injectOverlay; tStart is captured right after page.goto
    // returned, which is ~150 ms after the first recorded frame in practice.
    const audioStartMs = Date.now() - tStart;
    console.log(`[BLOG_VIDEO_AUDIO_START_MS] ${audioStartMs}`);

    // Captions land in #live-subtitle .subtitle-text. Wait for the first
    // non-empty caption rather than relying on element visibility (the div is
    // technically rendered but empty / opacity-faded until content arrives).
    const subtitleText = page.locator('#live-subtitle .subtitle-text');
    await expect
      .poll(async () => ((await subtitleText.textContent()) ?? '').trim().length, {
        timeout: 30_000,
        intervals: [500, 1000, 2000],
      })
      .toBeGreaterThan(0);

    await caption(page, scenes, 4);

    // ----- Scene 5: backend identity -----
    // Let the captions roll for a while so the viewer can read the German text.
    await page.waitForTimeout(8_000);
    await caption(page, scenes, 5);

    // ----- Scenes 6-7: let the rest of the 90 s clip transcribe -----
    await page.waitForTimeout(15_000);
    await caption(page, scenes, 6);

    await page.waitForTimeout(20_000);
    await caption(page, scenes, 7);

    // Settle for a final beat so the video ends on the latest caption.
    await page.waitForTimeout(3_000);
  });
});
