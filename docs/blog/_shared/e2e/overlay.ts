/**
 * Caption overlay helpers for the Starling blog-series demo recordings.
 *
 * Ported from tickytack's `packages/e2e/video/record-intro.spec.ts` so the
 * visual style is consistent: dark frosted-glass pill, 22 px Inter, fixed to
 * the bottom-centre of the viewport, fades in/out.
 *
 * Usage from a spec:
 *
 *   import { injectOverlay, caption } from './overlay';
 *   import scenes from './transcriptions/asr-vllm.json' with { type: 'json' };
 *
 *   await injectOverlay(page);
 *   await caption(page, scenes, 1);   // shows scene #1 for its declared duration
 */
import type { Page } from '@playwright/test';

export type Scene = {
  scene: number;
  text: string;
  /** How long the caption stays on screen, in milliseconds. */
  duration: number;
};

export async function injectOverlay(page: Page): Promise<void> {
  await page.evaluate(() => {
    if (document.getElementById('tt-overlay')) return;
    const overlay = document.createElement('div');
    overlay.id = 'tt-overlay';
    Object.assign(overlay.style, {
      position: 'fixed',
      bottom: '40px',
      left: '50%',
      transform: 'translateX(-50%)',
      zIndex: '99999',
      background: 'rgba(30, 26, 23, 0.88)',
      color: '#FFF3E0',
      padding: '16px 32px',
      borderRadius: '12px',
      fontSize: '22px',
      fontFamily: "'Inter', 'Segoe UI', system-ui, sans-serif",
      fontWeight: '500',
      letterSpacing: '0.01em',
      maxWidth: '820px',
      textAlign: 'center',
      backdropFilter: 'blur(8px)',
      border: '1px solid rgba(233, 69, 96, 0.35)',
      boxShadow: '0 8px 32px rgba(0, 0, 0, 0.4)',
      opacity: '0',
      transition: 'opacity 0.4s ease',
      pointerEvents: 'none',
    });
    document.body.appendChild(overlay);
  });
}

export async function showCaption(page: Page, text: string): Promise<void> {
  await page.evaluate((t) => {
    const el = document.getElementById('tt-overlay');
    if (!el) return;
    el.textContent = t;
    el.style.opacity = '1';
  }, text);
}

export async function hideCaption(page: Page): Promise<void> {
  await page.evaluate(() => {
    const el = document.getElementById('tt-overlay');
    if (el) el.style.opacity = '0';
  });
}

export async function caption(
  page: Page,
  scenes: Scene[],
  sceneNumber: number,
): Promise<void> {
  const scene = scenes.find((s) => s.scene === sceneNumber);
  if (!scene) {
    throw new Error(`No scene with number ${sceneNumber}`);
  }
  await showCaption(page, scene.text);
  await page.waitForTimeout(scene.duration);
  await hideCaption(page);
  await page.waitForTimeout(400);
}

export function backendBaseUrl(): string {
  // BASE_URL env wins if set — useful when the user runs their server on a
  // non-default port (e.g., the production deployment on :80).
  if (process.env.BASE_URL) return process.env.BASE_URL;
  const backend = process.env.BACKEND ?? 'vllm';
  if (backend !== 'vllm' && backend !== 'burn') {
    throw new Error(`Unknown BACKEND=${backend}; expected "vllm" or "burn"`);
  }
  return backend === 'burn' ? 'http://localhost:8091' : 'http://localhost:8090';
}

export function backendLabel(): 'vllm' | 'burn' {
  const backend = process.env.BACKEND ?? 'vllm';
  if (backend !== 'vllm' && backend !== 'burn') {
    throw new Error(`Unknown BACKEND=${backend}; expected "vllm" or "burn"`);
  }
  return backend;
}
