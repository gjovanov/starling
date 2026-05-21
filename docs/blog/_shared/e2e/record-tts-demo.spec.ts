/**
 * TTS demo recording (Posts 3 & 4 of the Starling blog series).
 *
 * Runs against a live server with the TTS feature enabled:
 *   - vllm-server: `./init.sh --tts && ./start.sh` (port 8090)
 *   - burn-server: `./init.sh --tts && ./start.sh` (port 8091)
 *
 * Opens the TTS tab in the frontend, pastes the ~150-word excerpt of
 * broadcast_1.txt, clicks Generate, and waits for the streaming player to
 * reach the `ended` state. Caption overlay narrates ~5 scenes.
 *
 * Usage:
 *   BACKEND=vllm playwright test record-tts-demo.spec.ts
 *   BACKEND=burn playwright test record-tts-demo.spec.ts
 */
import { test, expect } from '@playwright/test';
import * as fs from 'node:fs';
import * as path from 'node:path';

import {
  injectOverlay,
  caption,
  backendBaseUrl,
  backendLabel,
  type Scene,
} from './overlay';

const EXCERPT_PATH = path.resolve(__dirname, '..', 'media', 'broadcast_1_excerpt.txt');
const EXCERPT = fs.readFileSync(EXCERPT_PATH, 'utf8').trim();

const BASE_URL = backendBaseUrl();
const BACKEND = backendLabel();

const SCENES_PATH = path.resolve(
  __dirname,
  'transcriptions',
  BACKEND === 'burn' ? 'tts-burn.json' : 'tts-vllm.json',
);
const scenes = JSON.parse(fs.readFileSync(SCENES_PATH, 'utf8')) as Scene[];

test.describe('Starling TTS demo', () => {
  test(`record streaming synthesis (${BACKEND})`, async ({ page }) => {
    test.setTimeout(220_000);

    await page.goto(`${BASE_URL}/`);
    await expect(page.locator('#session-panel')).toBeVisible();
    const tStart = Date.now();
    await injectOverlay(page);
    await page.waitForTimeout(500);

    // ----- Scene 1: open TTS tab -----
    await page.click('.top-tab[data-top="tts"]');
    await expect(page.locator('#tts-content')).toBeVisible();
    await caption(page, scenes, 1);

    // Wait until voices have loaded — Generate becomes enabled then.
    await expect(page.locator('#tts-generate-btn')).toBeEnabled({ timeout: 60_000 });

    // Pick the German male voice so the synthesis matches the (German) excerpt.
    const voiceSelect = page.locator('#tts-voice');
    await voiceSelect
      .selectOption({ value: 'de_male' })
      .catch(async () => {
        // Fall back to label-based match if value differs across builds.
        await voiceSelect.selectOption({ label: /German.*Male/i });
      });
    await caption(page, scenes, 2);

    // ----- Scene 3: paste the excerpt -----
    const textArea = page.locator('#tts-text');
    await textArea.click();
    // Type in chunks so the char counter visibly ticks; pressSequentially is
    // too slow for 150 words, so we fill in 3 passes.
    const third = Math.ceil(EXCERPT.length / 3);
    await textArea.pressSequentially(EXCERPT.slice(0, third), { delay: 4 });
    await textArea.pressSequentially(EXCERPT.slice(third, third * 2), { delay: 2 });
    await textArea.pressSequentially(EXCERPT.slice(third * 2), { delay: 2 });
    await caption(page, scenes, 3);

    // ----- Scene 4: click Generate -----
    await page.locator('#tts-generate-btn').click();

    // Stamp the audio-start moment so the wrapper can mux the
    // separately-synthesised WAV at the right offset.
    const audioStartMs = Date.now() - tStart;
    console.log(`[BLOG_VIDEO_AUDIO_START_MS] ${audioStartMs}`);

    await caption(page, scenes, 4);

    // The frontend exposes `window.__ttsPlayerState` for Playwright assertions.
    // Shape: { state: "idle"|"buffering"|"playing"|"paused"|"ended", ... }.
    const readState = () =>
      page.evaluate(() => {
        const s = (window as unknown as { __ttsPlayerState?: unknown }).__ttsPlayerState;
        if (typeof s === 'string') return s;
        if (s && typeof s === 'object' && 'state' in (s as Record<string, unknown>)) {
          return String((s as { state: unknown }).state);
        }
        return 'unknown';
      });

    // Wait for ANY state transition out of "idle" — playing/buffering means
    // synthesis is working, "ended" means the backend already finished or
    // returned an empty stream (burn-server's TTS sometimes does the latter
    // after a prewarm; the demo video still records the visible flow either
    // way, and the muxed audio comes from the prewarm WAV).
    await expect
      .poll(readState, { timeout: 30_000, intervals: [300, 500, 1000] })
      .not.toBe('idle');

    const stateAfterClick = await readState();
    if (stateAfterClick === 'ended') {
      // The in-browser synth returned an empty stream (typical on burn-server
      // after a prewarm: the CUDA index_select kernel hits a bounds assertion
      // on the second call within one process). The muxed audio comes from
      // the prewarm-captured WAV instead, so we hold the page here for the
      // duration of that audio (~45 s) to keep the recorded video long enough
      // for the muxed track to play out.
      console.log('[BLOG_VIDEO_INFO] state=ended early; holding for audio length');
      await caption(page, scenes, 5);
      await page.waitForTimeout(40_000);
    } else {
      await caption(page, scenes, 5);
      // Wait for natural "ended" up to ~120 s; treat timeout as fine.
      await Promise.race([
        page
          .waitForFunction(
            () => {
              const s = (window as unknown as { __ttsPlayerState?: unknown }).__ttsPlayerState;
              const v = s && typeof s === 'object' && 'state' in (s as Record<string, unknown>)
                ? (s as { state: unknown }).state
                : s;
              return v === 'ended';
            },
            undefined,
            { timeout: 120_000 },
          )
          .then(() => 'ended' as const)
          .catch(() => 'timeout' as const),
      ]);
    }
    console.log(`[BLOG_VIDEO_FINAL_STATE] ${await readState()}`);

    await caption(page, scenes, 6);
    await page.waitForTimeout(2_000);
  });
});
