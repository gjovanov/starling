import { defineConfig, devices } from '@playwright/test';
import * as path from 'node:path';

// The two demo specs live one level up under `../e2e/`.  Tests pick the right
// server via the BACKEND env var (vllm → :8090, burn → :8091).
const E2E_DIR = path.resolve(__dirname, '..', 'e2e');

// Default base URL is set per-backend inside each spec.  We still set one here
// so playwright.test() can use page.goto('/') as a fallback during dev.
const BACKEND = process.env.BACKEND ?? 'vllm';
const DEFAULT_BASE_URL =
  BACKEND === 'burn' ? 'http://localhost:8091' : 'http://localhost:8090';

export default defineConfig({
  testDir: E2E_DIR,
  // Recording specs are long-running (≤ 120 s of captured video + setup).
  timeout: 240_000,
  expect: { timeout: 20_000 },
  reporter: [['list']],
  fullyParallel: false,
  workers: 1,
  retries: 0,
  outputDir: path.resolve(__dirname, '..', 'test-results'),
  use: {
    baseURL: DEFAULT_BASE_URL,
    // Record every test — this is the whole point of the project.
    video: {
      mode: 'on',
      size: { width: 1280, height: 720 },
    },
    viewport: { width: 1280, height: 720 },
    trace: 'off',
    launchOptions: {
      // The frontend's WebRTC paths still touch getUserMedia even when we're
      // pushing media files, so allow the fake sources just in case.
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
