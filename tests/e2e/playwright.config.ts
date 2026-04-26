import { defineConfig, devices } from '@playwright/test';

/**
 * Shared Playwright config for both Starling frontends.
 *
 * Tests in `specs/speakers.spec.ts` parametrize over the two targets
 * (burn-server on :8091, vllm-server on :8090).  Each target reads its
 * base URL from the `STARLING_TARGETS` env var, defaulting to the two
 * ports documented in CLAUDE.md.
 *
 * The tests launch a mock signalling server that plays the server role
 * so that the frontend can exercise the full uplink flow without
 * depending on the real servers (which require model weights + vLLM).
 * See `fixtures/mockServer.ts`.
 */
export default defineConfig({
  testDir: './specs',
  timeout: 60_000,
  expect: { timeout: 10_000 },
  reporter: [['list'], ['html', { open: 'never', outputFolder: 'playwright-report' }]],
  fullyParallel: false,
  workers: 1,
  retries: 0,
  use: {
    trace: 'on-first-retry',
    video: 'retain-on-failure',
    // Route getUserMedia/getDisplayMedia to a synthetic fake source
    launchOptions: {
      args: [
        '--use-fake-device-for-media-stream',
        '--use-fake-ui-for-media-stream',
        '--autoplay-policy=no-user-gesture-required',
      ],
    },
    permissions: ['microphone'],
  },
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
  ],
});
