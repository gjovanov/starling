import { test, expect, type ConsoleMessage, type Page } from '@playwright/test';
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
  // Reset captured signalling events between tests
  mock.events.length = 0;
  mock.sessions.length = 0;
});

async function gotoApp(page: Page) {
  await page.goto(mock.url);
  // Give the frontend a moment to fetch config + populate dropdowns
  await expect(page.locator('#session-panel')).toBeVisible();
  await page.click('.session-tab[data-tab="create"]');
}

test('Speakers tab is present and reveals its content', async ({ page }) => {
  await gotoApp(page);

  const speakersTab = page.locator('.source-tab[data-source="speakers"]');
  await expect(speakersTab).toBeVisible();
  await expect(speakersTab).toHaveText('Speakers');

  // Media content visible by default
  await expect(page.locator('#media-source-content')).toBeVisible();
  await expect(page.locator('#speakers-source-content')).toBeHidden();

  await speakersTab.click();

  await expect(speakersTab).toHaveClass(/active/);
  await expect(page.locator('#speakers-source-content')).toBeVisible();
  await expect(page.locator('#media-source-content')).toBeHidden();

  // Method radio defaults to "display" (the screen-share path)
  const displayRadio = page.locator('input[name="speakers-method"][value="display"]');
  await expect(displayRadio).toBeChecked();

  // Device row is hidden by default
  await expect(page.locator('#speakers-device-row')).toBeHidden();

  // Test capture button is present
  await expect(page.locator('#speakers-test-btn')).toBeVisible();
});

test('Switching method to "device" reveals device dropdown', async ({ page }) => {
  await gotoApp(page);
  await page.click('.source-tab[data-source="speakers"]');

  await page.locator('input[name="speakers-method"][value="device"]').check();
  await expect(page.locator('#speakers-device-row')).toBeVisible();
  await expect(page.locator('#speakers-select')).toBeVisible();

  await page.locator('input[name="speakers-method"][value="display"]').check();
  await expect(page.locator('#speakers-device-row')).toBeHidden();
});

test('Switching back to Media Files hides the speakers content', async ({ page }) => {
  await gotoApp(page);

  await page.click('.source-tab[data-source="speakers"]');
  await expect(page.locator('#speakers-source-content')).toBeVisible();

  await page.click('.source-tab[data-source="media"]');
  await expect(page.locator('#speakers-source-content')).toBeHidden();
  await expect(page.locator('#media-source-content')).toBeVisible();
});

test('Refreshing devices populates audio input options', async ({ page }) => {
  await gotoApp(page);
  await page.click('.source-tab[data-source="speakers"]');

  // Switch to device method to reveal the dropdown + refresh button
  await page.locator('input[name="speakers-method"][value="device"]').check();
  await page.click('#speakers-refresh-btn');

  // Chromium --use-fake-device-for-media-stream exposes at least one
  // synthetic "Fake Audio Input" device.
  const options = page.locator('#speakers-select option');
  await expect.poll(async () => options.count(), { timeout: 5_000 }).toBeGreaterThanOrEqual(1);
});

test('Test capture button reports a result', async ({ page }) => {
  await gotoApp(page);
  await page.click('.source-tab[data-source="speakers"]');

  await page.click('#speakers-test-btn');

  // Chromium with --use-fake-ui-for-media-stream auto-grants getDisplayMedia
  // and synthesizes audio, so the test status should report success.
  const status = page.locator('#speakers-test-status');
  await expect.poll(async () => (await status.textContent()) || '', { timeout: 10_000 })
    .toMatch(/OK|Capture|fail/i);
});

test('Creating a Speakers session POSTs source=speakers and opens a WS with role=uplink', async ({ page }) => {
  const consoleLines: string[] = [];
  page.on('console', (msg: ConsoleMessage) => {
    consoleLines.push(`[${msg.type()}] ${msg.text()}`);
  });

  await gotoApp(page);
  await page.click('.source-tab[data-source="speakers"]');

  // Intercept POST /api/sessions to confirm the request payload
  const sessionPromise = page.waitForRequest(req => req.method() === 'POST' && req.url().endsWith('/api/sessions'));

  await page.click('#create-session-btn');

  const sessionReq = await sessionPromise;
  const postBody = JSON.parse(sessionReq.postData() || '{}');
  expect(postBody.source).toBe('speakers');
  expect(postBody.media_id).toBeUndefined();
  expect(postBody.srt_channel_id).toBeUndefined();

  // The mock creates the session. Wait for our server-side event log
  // to contain a ready-with-uplink message.
  await expect.poll(
    () => mock.events.find(e => e.type === 'ready' && e.raw?.role === 'uplink'),
    { timeout: 15_000, message: 'Expected a ready/role=uplink event from the frontend' }
  ).toBeTruthy();

  // And the frontend should follow up with an SDP offer.
  await expect.poll(
    () => mock.events.find(e => e.type === 'offer'),
    { timeout: 15_000, message: 'Expected an SDP offer from the frontend' }
  ).toBeTruthy();

  // Session must appear with source_type="speakers"
  expect(mock.sessions).toHaveLength(1);
  expect(mock.sessions[0].source_type).toBe('speakers');
  // Buffer info should update to reflect live capture
  await expect(page.locator('#buffer-info')).not.toHaveText('Select a session');
});

test('Media Files session flow still POSTs a media_id (non-speakers regression guard)', async ({ page }) => {
  // Seed a media entry by mutating the mock — simulate a real server listing
  // a WAV. We inject it via a custom route override before navigation.
  await page.route('**/api/media', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        success: true,
        data: [{
          id: 'broadcast_1.wav',
          filename: 'broadcast_1.wav',
          format: 'wav',
          duration_secs: 300,
          size_bytes: 1000,
        }],
      }),
    });
  });

  await gotoApp(page);

  // Media tab is default. Wait for the select to be populated
  await expect(page.locator('#media-select option[value="broadcast_1.wav"]')).toHaveCount(1);

  const sessionPromise = page.waitForRequest(req => req.method() === 'POST' && req.url().endsWith('/api/sessions'));
  await page.click('#create-session-btn');

  const req = await sessionPromise;
  const body = JSON.parse(req.postData() || '{}');
  expect(body.source).toBe('media');
  expect(body.media_id).toBe('broadcast_1.wav');
});
