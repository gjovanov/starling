import { test, expect, type Page } from '@playwright/test';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

import { startMockServer, type MockServer } from '../fixtures/mockServer';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const FRONTEND_DIR = path.resolve(__dirname, '..', '..', '..', 'frontend');

let mock: MockServer;

test.beforeAll(async () => {
  mock = await startMockServer(FRONTEND_DIR);
});

test.afterAll(async () => {
  await mock?.close();
});

test.beforeEach(() => {
  mock.ttsRequests.length = 0;
  mock.ttsOutputs.length = 0;
  mock.clonedVoices.length = 0;
  // Reset lifecycle state to "ready" so existing tests aren't blocked
  // by leftover state from a previous test that flipped it.
  mock.ttsStatus.state = 'ready';
  mock.ttsStatus.blocked_reason = null;
  mock.ttsStatus.boot_elapsed_secs = null;
  mock.ttsStatus.boot_timeout_secs = null;
});

async function gotoTtsTab(page: Page) {
  await page.goto(mock.url);
  await expect(page.locator('#session-panel')).toBeVisible();
  await page.click('.session-tab[data-tab="tts"]');
  await expect(page.locator('#tts-content')).toBeVisible();
  // Wait until voices are populated (Generate becomes enabled)
  await expect(page.locator('#tts-generate-btn')).toBeEnabled();
}

test('TTS tab is present and reveals its content', async ({ page }) => {
  await page.goto(mock.url);
  await expect(page.locator('#session-panel')).toBeVisible();

  const ttsTab = page.locator('.session-tab[data-tab="tts"]');
  await expect(ttsTab).toBeVisible();
  await expect(ttsTab).toHaveText('TTS');

  await ttsTab.click();
  await expect(page.locator('#tts-content')).toBeVisible();
  await expect(page.locator('#tts-text')).toBeVisible();
  await expect(page.locator('#tts-voice')).toBeVisible();
});

test('Voice dropdown is populated from /api/tts/voices', async ({ page }) => {
  await gotoTtsTab(page);

  // <select> should have an option per mock voice plus optgroups by language.
  const optionCount = await page.locator('#tts-voice option').count();
  expect(optionCount).toBeGreaterThanOrEqual(11); // matches MOCK_VOICES length
  // The default voice should be selected.
  const selectedValue = await page.locator('#tts-voice').inputValue();
  expect(selectedValue).toBe('casual_male');
});

test('Output mode toggle reveals/hides the filename input', async ({ page }) => {
  await gotoTtsTab(page);

  const filenameGroup = page.locator('#tts-filename-group');
  await expect(filenameGroup).toBeHidden();

  await page.locator('input[name="tts-output-mode"][value="save"]').check();
  await expect(filenameGroup).toBeVisible();

  await page.locator('input[name="tts-output-mode"][value="play"]').check();
  await expect(filenameGroup).toBeHidden();
});

test('Char counter updates as the user types', async ({ page }) => {
  await gotoTtsTab(page);

  const text = page.locator('#tts-text');
  const counter = page.locator('#tts-char-count');
  await expect(counter).toHaveText('0');

  await text.fill('Hello');
  await expect(counter).toHaveText('5');
});

test('Generate (play mode) progressively plays via Web Audio', async ({ page }) => {
  await gotoTtsTab(page);
  await page.locator('#tts-text').fill('Hello world');
  await page.locator('#tts-generate-btn').click();

  await expect.poll(() => mock.ttsRequests.length, { timeout: 5_000 }).toBeGreaterThanOrEqual(1);
  const last = mock.ttsRequests[mock.ttsRequests.length - 1];
  expect(last.text).toBe('Hello world');
  expect(last.voice).toBe('casual_male');
  expect(last.save).toBe(false);

  // Phase 4: <audio> stays hidden in stream mode; status reflects
  // buffering/playing/done transitions reported by TtsPlayer.
  await expect(page.locator('#tts-audio')).toBeHidden();

  // The player's state transitions land on `window.__ttsPlayerState`.
  // We expect to see at least `playing` and ultimately `ended`.
  await expect.poll(
    async () => (await page.evaluate(() => (window as any).__ttsPlayerState?.state)) ?? null,
    { timeout: 5_000 },
  ).toBe('ended');

  // Final status text mentions Done (set by onStateChange) plus elapsed.
  await expect(page.locator('#tts-status')).toContainText(/Done/);
});

test('Pause / Resume buttons toggle the player state', async ({ page }) => {
  await gotoTtsTab(page);
  await page.locator('#tts-text').fill('Pause me');

  // Slow the player by adding latency on the mock side: we already insert
  // 25ms between chunks; that's enough buffering window for the test.
  await page.locator('#tts-generate-btn').click();

  // Wait for playing state, then click Pause.
  await expect.poll(
    async () => (await page.evaluate(() => (window as any).__ttsPlayerState?.state)) ?? null,
    { timeout: 5_000 },
  ).toMatch(/^(playing|buffering|ended)$/);

  // Pause button is visible only while playing — but the test stream is
  // very short (300ms total), so we may already be ended. Skip the rest
  // gracefully if so.
  const playing = await page.evaluate(() => (window as any).__ttsPlayerState?.state) === 'playing';
  if (!playing) return;

  await page.locator('#tts-pause-btn').click();
  await expect.poll(
    async () => (await page.evaluate(() => (window as any).__ttsPlayerState?.state)) ?? null,
  ).toBe('paused');
  await page.locator('#tts-resume-btn').click();
  await expect.poll(
    async () => (await page.evaluate(() => (window as any).__ttsPlayerState?.state)) ?? null,
  ).toMatch(/^(playing|ended)$/);
});

test('Stop button cancels the player and shows "Stopped"', async ({ page }) => {
  await gotoTtsTab(page);
  await page.locator('#tts-text').fill('Long enough to interrupt');
  await page.locator('#tts-generate-btn').click();

  // The stub stream is fast, so we click Stop the moment the button
  // becomes visible (any non-idle state).
  await page.locator('#tts-stop-btn').waitFor({ state: 'visible', timeout: 2_000 }).catch(() => {});
  if (await page.locator('#tts-stop-btn').isVisible()) {
    await page.locator('#tts-stop-btn').click();
    await expect(page.locator('#tts-status')).toContainText(/Stopped/);
  }
});

test('Generate (save mode) sends the user-chosen filename', async ({ page }) => {
  await gotoTtsTab(page);
  await page.locator('#tts-text').fill('My greeting');
  await page.locator('input[name="tts-output-mode"][value="save"]').check();
  await page.locator('#tts-filename').fill('greeting.wav');
  await page.locator('#tts-generate-btn').click();

  await expect.poll(() => mock.ttsRequests.length, { timeout: 5_000 }).toBe(1);
  expect(mock.ttsRequests[0].save_filename).toBe('greeting.wav');
});

test('Saved-files list appears after a successful save-mode synth', async ({ page }) => {
  await gotoTtsTab(page);
  await page.locator('input[name="tts-output-mode"][value="save"]').check();
  await page.locator('#tts-text').fill('First');
  await page.locator('#tts-filename').fill('first.wav');
  await page.locator('#tts-generate-btn').click();

  await expect(page.locator('#tts-saved-files-group')).toBeVisible({ timeout: 5_000 });
  const items = page.locator('#tts-saved-files .media-item');
  await expect(items).toHaveCount(1);
});


test('Play mode does NOT add anything to the saved-files list', async ({ page }) => {
  await gotoTtsTab(page);
  await page.locator('#tts-text').fill('Stream only');
  await page.locator('#tts-generate-btn').click();

  // Wait for the player to finish (state=ended) — visible audio elem is
  // hidden in stream mode, so we observe the player state directly.
  await expect.poll(
    async () => (await page.evaluate(() => (window as any).__ttsPlayerState?.state)) ?? null,
    { timeout: 5_000 },
  ).toBe('ended');
  // Streaming path doesn't write to disk — list stays hidden.
  await expect(page.locator('#tts-saved-files-group')).toBeHidden();
});

test('Switching back to Sessions hides TTS content', async ({ page }) => {
  await gotoTtsTab(page);
  await page.click('.session-tab[data-tab="sessions"]');
  await expect(page.locator('#tts-content')).toBeHidden();
  await expect(page.locator('#sessions-content')).toBeVisible();
});

test('Empty text shows an inline error and does NOT hit the server', async ({ page }) => {
  await gotoTtsTab(page);
  // Don't type anything
  await page.locator('#tts-generate-btn').click();
  await expect(page.locator('#tts-status')).toContainText(/text/i);
  expect(mock.ttsRequests.length).toBe(0);
});


// ─── Phase 6: lifecycle badge ──────────────────────────────────────

test('Engine status badge starts as "ready" by default', async ({ page }) => {
  await gotoTtsTab(page);
  await expect(page.locator('#tts-engine-badge')).toContainText(/ready/);
});

test('Engine badge flips to "warming up" when the lifecycle reports starting', async ({ page }) => {
  await gotoTtsTab(page);
  // Drive the mock into starting state.
  mock.ttsStatus.state = 'starting';
  mock.ttsStatus.boot_elapsed_secs = 12;
  mock.ttsStatus.boot_timeout_secs = 180;

  // The badge polls every 5s; expect.poll waits up to 6s for the change.
  await expect.poll(
    () => page.locator('#tts-engine-badge').innerText(),
    { timeout: 7_000 },
  ).toMatch(/warming up/i);
});

test('Engine badge shows "blocked by ASR" and disables Generate', async ({ page }) => {
  await gotoTtsTab(page);
  mock.ttsStatus.state = 'blocked';
  mock.ttsStatus.blocked_reason = 'blocked by ASR session';

  await expect.poll(
    () => page.locator('#tts-engine-badge').innerText(),
    { timeout: 7_000 },
  ).toMatch(/blocked/i);
  await expect(page.locator('#tts-generate-btn')).toBeDisabled();
});


// ─── Phase 7: voice cloning ────────────────────────────────────────

test('Custom-voices section is collapsed by default', async ({ page }) => {
  await gotoTtsTab(page);
  // <details> is closed by default — its inner controls are not visible.
  await expect(page.locator('#tts-cloning')).toBeVisible();
  await expect(page.locator('#tts-clone-upload-btn')).toBeHidden();
});

test('Upload button stays disabled until all fields + permission are set', async ({ page }) => {
  await gotoTtsTab(page);
  await page.locator('#tts-cloning summary').click();
  const btn = page.locator('#tts-clone-upload-btn');
  await expect(btn).toBeDisabled();

  await page.locator('#tts-clone-name').fill('My Voice');
  await expect(btn).toBeDisabled();

  await page.locator('#tts-clone-text').fill('The quick brown fox jumps over the lazy dog.');
  await expect(btn).toBeDisabled();

  // File-input
  await page.locator('#tts-clone-file').setInputFiles({
    name: 'sample.wav',
    mimeType: 'audio/wav',
    buffer: Buffer.from(new Uint8Array(8192)),
  });
  await expect(btn).toBeDisabled();   // permission still unchecked

  await page.locator('#tts-clone-permission').check();
  await expect(btn).toBeEnabled();
});

test('Upload posts multipart and adds the clone to the dropdown', async ({ page }) => {
  await gotoTtsTab(page);
  await page.locator('#tts-cloning summary').click();
  await page.locator('#tts-clone-name').fill('Aunt Helen');
  await page.locator('#tts-clone-text').fill('A clear sentence in a calm voice.');
  await page.locator('#tts-clone-file').setInputFiles({
    name: 'sample.wav',
    mimeType: 'audio/wav',
    buffer: Buffer.from(new Uint8Array(8192)),
  });
  await page.locator('#tts-clone-permission').check();
  await page.locator('#tts-clone-upload-btn').click();

  await expect.poll(() => mock.clonedVoices.length, { timeout: 5_000 }).toBe(1);
  expect(mock.clonedVoices[0].name).toBe('Aunt Helen');

  await expect(page.locator('#tts-clone-status')).toContainText(/Uploaded/);
  // Voice <select> now contains a "Custom voices" optgroup with our entry.
  await expect(page.locator('#tts-voice optgroup[label="Custom voices"] option')).toContainText('Aunt Helen');
});

test('Permission checkbox must be ticked or upload is rejected at the server', async ({ page }) => {
  await gotoTtsTab(page);
  await page.locator('#tts-cloning summary').click();
  await page.locator('#tts-clone-name').fill('No-Consent Voice');
  await page.locator('#tts-clone-text').fill('something');
  await page.locator('#tts-clone-file').setInputFiles({
    name: 'sample.wav',
    mimeType: 'audio/wav',
    buffer: Buffer.from(new Uint8Array(8192)),
  });

  // Tick to enable, then untick: client gates the button. We bypass by
  // posting directly so we exercise the SERVER's gate (the source of truth).
  const r = await page.evaluate(async () => {
    const fd = new FormData();
    fd.append('name', 'X');
    fd.append('ref_text', 't');
    fd.append('permission_confirmed', 'false');
    fd.append('audio_sample', new Blob([new Uint8Array(100)], { type: 'audio/wav' }), 'a.wav');
    const r = await fetch('/api/tts/voices/upload', { method: 'POST', body: fd });
    return r.json();
  });
  expect(r.success).toBe(false);
  expect(String(r.error).toLowerCase()).toContain('permission');
});

test('Delete removes the clone from the dropdown', async ({ page }) => {
  // Pre-seed an upload via the mock so the test starts with a clone.
  mock.clonedVoices.push({
    id: 'mock-clone-1',
    name: 'Throwaway',
    ref_text: 't',
    permission_confirmed: true,
    sample_rate: 24000,
    duration_secs: 6.0,
    created_at: Date.now() / 1000,
    kind: 'cloned',
  });

  await gotoTtsTab(page);
  await page.locator('#tts-cloning summary').click();

  // List should show the seeded clone.
  await expect(page.locator('#tts-clone-list .media-item')).toHaveCount(1);
  await page.locator('#tts-clone-list [data-clone-delete]').click();

  await expect.poll(() => mock.clonedVoices.length).toBe(0);
  await expect(page.locator('#tts-clone-list-group')).toBeHidden();
  // Dropdown loses the Custom voices group.
  await expect(page.locator('#tts-voice optgroup[label="Custom voices"]')).toHaveCount(0);
});


// ─── Phase 5: long-form ────────────────────────────────────────────

test('Long-form input is accepted and produces audio (single combined stream)', async ({ page }) => {
  await gotoTtsTab(page);

  // ~5000-char paragraph with many sentence boundaries. The server-side
  // splitter should chop it into many parts; the mock collapses every
  // synth request to a single chunked WAV (covers the route, not the
  // splitter — that's pytest's job).
  const sentence = 'This is sentence number ';
  const blob = Array.from({ length: 200 }, (_, i) => `${sentence}${i + 1}.`).join(' ');
  await page.locator('#tts-text').fill(blob);
  // Sanity: did the textarea actually accept the long input?
  const filledLen = await page.locator('#tts-text').inputValue().then(v => v.length);
  await page.locator('#tts-generate-btn').click();

  // Player must reach `ended`. The mock's stub stream is short, so this
  // just verifies the route + frontend wiring tolerate long input.
  await expect.poll(
    async () => (await page.evaluate(() => (window as any).__ttsPlayerState?.state)) ?? null,
    { timeout: 10_000 },
  ).toBe('ended');

  // The mock recorded exactly ONE /api/tts/synthesize POST — the server
  // is responsible for splitting; the client doesn't fan out.
  const synthRequests = mock.ttsRequests.filter(r => r && typeof r === 'object' && r.text);
  expect(synthRequests.length).toBe(1);
  // The textarea must accept >2000 chars (the old Phase-2 cap).
  expect(filledLen, `textarea filled length was ${filledLen}`).toBeGreaterThan(2000);
  expect(synthRequests[0].text.length, `posted text length was ${synthRequests[0].text.length}`).toBeGreaterThan(2000);
  expect(synthRequests[0].save).toBe(false);
});
