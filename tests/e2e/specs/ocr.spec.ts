/**
 * OCR tab e2e — runs against the REAL ocr-server (Python) with the mock
 * engine: upload → WS-driven jobs table → viewer panes → engines panel.
 */
import { test, expect, type Page } from '@playwright/test';
import { spawnSync } from 'node:child_process';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

import { startOcrServer, type OcrServer } from '../fixtures/ocrServer';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const APP_DIR = path.resolve(__dirname, '..', '..', '..', 'apps', 'ocr-server');

let server: OcrServer;
let pdfPath: string;
let txtPath: string;
let delPdfPath: string;
let pagePdfPaths: string[];

test.beforeAll(async () => {
  server = await startOcrServer();

  // Synthesize the upload fixtures with the app venv's PyMuPDF:
  // page 1 has a text layer (fast path), page 2 is drawing-only (mock OCR).
  // delPdfPath has unique content (own sha) so the delete test is isolated
  // from the sha-dedupe of repeated pdfPath uploads.
  pdfPath = path.join(server.tmpDir, 'e2e-invoice.pdf');
  txtPath = path.join(server.tmpDir, 'not-a-doc.txt');
  delPdfPath = path.join(server.tmpDir, 'e2e-delete-me.pdf');
  const python = path.join(APP_DIR, '.venv', 'bin', 'python');
  const gen = spawnSync(python, ['-c', `
import pymupdf, sys
doc = pymupdf.open()
p = doc.new_page()
p.insert_text((72, 72), "INVOICE No. E2E-2026\\nSupplier: Playwright Ltd, Sofia\\nTotal: 42.00 EUR", fontsize=12)
p2 = doc.new_page()
p2.draw_rect(pymupdf.Rect(72, 72, 300, 200), color=(0, 0, 0), width=2)
doc.save(sys.argv[1])
open(sys.argv[2], "w").write("plain text, not a document")
d2 = pymupdf.open()
d2.new_page().insert_text((72, 72), "INVOICE No. DELETE-ME-9999\\nUnique content for deletion test", fontsize=12)
d2.save(sys.argv[3])
# 16 unique 1-page docs to exercise pagination (PAGE_SIZE = 15)
outdir = sys.argv[4]
for i in range(16):
    d = pymupdf.open()
    d.new_page().insert_text((72, 72), f"PAGINATION DOC {i:02d} — unique {i*7919}", fontsize=12)
    d.save(f"{outdir}/page-{i:02d}.pdf")
`, pdfPath, txtPath, delPdfPath, server.tmpDir]);
  if (gen.status !== 0) {
    throw new Error(`fixture generation failed: ${gen.stderr}`);
  }
  pagePdfPaths = Array.from({ length: 16 }, (_, i) =>
    path.join(server.tmpDir, `page-${String(i).padStart(2, '0')}.pdf`));
});

test.afterAll(async () => {
  await server?.close();
});

async function gotoOcrTab(page: Page) {
  await page.goto(server.url);
  await expect(page.locator('#session-panel')).toBeVisible();
  const ocrTab = page.locator('.top-tab[data-top="ocr"]');
  await expect(ocrTab).toBeVisible();
  await ocrTab.click();
  await expect(page.locator('#ocr-content')).toBeVisible();
  // setupOcrTab() populated the engine dropdown ⇒ tab is fully initialized.
  await expect(page.locator('#ocr-engine-select')).toHaveValue('mock');
}

test('OCR tab is present and reveals its content', async ({ page }) => {
  await gotoOcrTab(page);
  await expect(page.locator('#ocr-upload-zone')).toBeVisible();
  await expect(page.locator('#ocr-engine-select')).toHaveValue('mock');
  await expect(page.locator('#ocr-language-hint')).toHaveValue('bg,en');
  await expect(page.locator('#ocr-auto-extract')).toBeChecked();
});

test('WS live badge connects', async ({ page }) => {
  await gotoOcrTab(page);
  await expect(page.locator('#ocr-ws-badge')).toHaveText('live', { timeout: 10_000 });
});

test('upload → jobs row live-updates to done with valid extraction + durations', async ({ page }) => {
  await gotoOcrTab(page);
  await page.setInputFiles('#ocr-file-input', pdfPath);

  const row = page.locator('#ocr-jobs tr[data-run]').first();
  await expect(row).toBeVisible({ timeout: 10_000 });
  await expect(row).toContainText('e2e-invoice.pdf');
  await expect(row.locator('.ocr-badge.engine')).toHaveText('mock');

  // Status transitions arrive over WS — the frontend never polls jobs.
  await expect(row.locator('td:nth-child(4) .ocr-badge')).toHaveText('done', { timeout: 15_000 });
  await expect(row).toContainText('2/2');
  // Stage 2 chip: extraction completes and validates.
  await expect(row.locator('td:nth-child(5)')).toContainText('done', { timeout: 15_000 });
  await expect(row.locator('td:nth-child(5)')).toContainText('valid');
  // Both step durations render (OCR in col 4, extraction in col 5).
  await expect(row.locator('td:nth-child(4) .ocr-dur')).toBeVisible();
  await expect(row.locator('td:nth-child(5) .ocr-dur')).toBeVisible({ timeout: 15_000 });
});

test('delete removes the document row', async ({ page }) => {
  await gotoOcrTab(page);
  await page.setInputFiles('#ocr-file-input', delPdfPath);
  const delRow = page.locator('#ocr-jobs tr[data-run]', { hasText: 'e2e-delete-me.pdf' });
  await expect(delRow).toBeVisible({ timeout: 10_000 });

  page.once('dialog', (d) => d.accept());   // confirm() prompt
  await delRow.locator('button[data-role="delete"]').click();
  await expect(page.locator('#ocr-jobs tr[data-run]', { hasText: 'e2e-delete-me.pdf' }))
    .toHaveCount(0, { timeout: 10_000 });
});

test('jobs table stays consistent across a WS reconnect (no stuck rows)', async ({ page }) => {
  await gotoOcrTab(page);
  await page.setInputFiles('#ocr-file-input', pdfPath);
  const row = page.locator('#ocr-jobs tr[data-run]').first();
  await expect(row.locator('td:nth-child(4) .ocr-badge')).toHaveText('done', { timeout: 15_000 });

  // Kill + reboot the server (same DB) → WS drops and reconnects; the row must
  // reconcile to server truth (done), not stay stuck.
  await server.restart();
  await expect(page.locator('#ocr-ws-badge')).toHaveText('live', { timeout: 30_000 });
  await expect(page.locator('#ocr-jobs tr[data-run]').first().locator('td:nth-child(4) .ocr-badge'))
    .toHaveText('done', { timeout: 15_000 });
});

test('viewer tabs: image, markdown+source, extraction, and CSV download', async ({ page }) => {
  await gotoOcrTab(page);
  const row = page.locator('#ocr-jobs tr[data-run]').first();
  await expect(row.locator('td:nth-child(5)')).toContainText('valid', { timeout: 15_000 });
  await row.locator('button[data-role="view"]').click();

  await expect(page.locator('#ocr-viewer-section')).toBeVisible();
  await expect(page.locator('#ocr-viewer-title')).toContainText('e2e-invoice.pdf');

  // Viewer tab (default): page image loaded, source badge = text-layer.
  await expect(page.locator('.ocr-tab[data-otab="viewer"]')).toHaveClass(/active/);
  await expect(page.locator('#ocr-viewer-source')).toHaveText('text-layer');
  const imgLoaded = await page.locator('#ocr-viewer-image')
    .evaluate((el: HTMLImageElement) => el.complete && el.naturalWidth > 0);
  expect(imgLoaded).toBe(true);

  // Markdown tab: born-digital page → real invoice text.
  await page.click('.ocr-tab[data-otab="markdown"]');
  await expect(page.locator('#ocr-viewer-markdown')).toBeVisible();
  await expect(page.locator('#ocr-viewer-markdown')).toContainText('INVOICE No. E2E-2026');

  // Extraction tab: JSON + validation badge.
  await page.click('.ocr-tab[data-otab="extraction"]');
  await expect(page.locator('#ocr-viewer-validation')).toBeVisible();
  await expect(page.locator('#ocr-viewer-validation')).toHaveText('valid');
  await expect(page.locator('#ocr-viewer-json')).toContainText('"document_type"');

  // Download tab: preview table renders + CSV downloads via the endpoint.
  await page.click('.ocr-tab[data-otab="download"]');
  await expect(page.locator('#ocr-tab-pagenav')).toBeHidden();  // per-page nav hidden here
  await expect(page.locator('#ocr-download-preview')).toContainText('Invoice metadata');
  await expect(page.locator('#ocr-download-preview')).toContainText('Line items');
  await expect(page.locator('#ocr-download-csv')).toBeEnabled();
  const [download] = await Promise.all([
    page.waitForEvent('download'),
    page.click('#ocr-download-csv'),
  ]);
  expect(download.suggestedFilename()).toMatch(/\.csv$/);

  // Page 2 via the viewer tab: drawing-only → OCR'd by the mock engine.
  await page.click('.ocr-tab[data-otab="viewer"]');
  await expect(page.locator('#ocr-tab-pagenav')).toBeVisible();
  await page.locator('#ocr-page-next').click();
  await expect(page.locator('#ocr-viewer-source')).toHaveText('ocr');
  await page.click('.ocr-tab[data-otab="markdown"]');
  await expect(page.locator('#ocr-viewer-markdown')).toContainText('Mock OCR');

  await page.locator('#ocr-viewer-close').click();
  await expect(page.locator('#ocr-viewer-section')).toBeHidden();
});

test('unsupported file type is rejected with a message', async ({ page }) => {
  await gotoOcrTab(page);
  await page.setInputFiles('#ocr-file-input', txtPath);
  await expect(page.locator('#ocr-upload-status')).toContainText('Upload failed', { timeout: 10_000 });
  await expect(page.locator('#ocr-upload-status')).toContainText('unsupported');
});

test('compare view: bake-off re-run + runs grid + markdown diff', async ({ page }) => {
  await gotoOcrTab(page);
  // Self-sufficient: upload our own doc so this test doesn't depend on prior
  // state (the doc may already have runs — assertions below are relative).
  await page.setInputFiles('#ocr-file-input', pdfPath);
  const row = page.locator('#ocr-jobs tr[data-run]').first();
  await expect(row.locator('td:nth-child(4) .ocr-badge')).toHaveText('done', { timeout: 15_000 });

  // Click the document name → compare section opens with the runs grid.
  await row.locator('[data-role="compare"]').click();
  await expect(page.locator('#ocr-compare-section')).toBeVisible();
  await expect(page.locator('#ocr-compare-title')).toContainText('e2e-invoice.pdf');
  // header + ≥1 existing run (this doc may already have several runs).
  const gridRows = page.locator('#ocr-compare-grid tr');
  const before = await gridRows.count();
  expect(before).toBeGreaterThanOrEqual(2);

  // Re-run with a second engine: real tesseract when installed, mock otherwise.
  const engineOptions = await page.locator('#ocr-compare-rerun-engine option')
    .allTextContents();
  const second = engineOptions.includes('tesseract-bul') ? 'tesseract-bul' : 'mock';
  await page.selectOption('#ocr-compare-rerun-engine', second);
  await page.click('#ocr-compare-rerun');

  // The grid live-updates over WS as the new run completes → grows by one.
  await expect(gridRows).toHaveCount(before + 1, { timeout: 30_000 });
  await expect(page.locator('#ocr-compare-grid')).toContainText(second);

  // Diff panes carry both runs' markdown (retrying — they fill once the new
  // run's markdown is available). Line-level highlighting is exercised in the
  // lineDiff unit path; here we just assert both panes render content.
  await expect(page.locator('#ocr-compare-a-md')).not.toBeEmpty({ timeout: 15_000 });
  await expect(page.locator('#ocr-compare-b-md')).not.toBeEmpty({ timeout: 15_000 });

  await page.click('#ocr-compare-close');
  await expect(page.locator('#ocr-compare-section')).toBeHidden();
});

test('engines panel lists mock and load/unload round-trips via WS', async ({ page }) => {
  await gotoOcrTab(page);
  const row = page.locator('.ocr-engine-row[data-engine="mock"]');
  await expect(row).toBeVisible();

  await row.locator('button[data-role="load"]').click();
  await expect(row.locator('[data-role="status"]')).toHaveText('ready', { timeout: 10_000 });

  await row.locator('button[data-role="unload"]').click();
  await expect(row.locator('[data-role="status"]')).toHaveText('unloaded', { timeout: 10_000 });
});

test('WS reconnects after a server kill and keeps driving the jobs table', async ({ page }) => {
  await gotoOcrTab(page);
  await expect(page.locator('#ocr-ws-badge')).toHaveText('live', { timeout: 10_000 });

  // SIGKILL + fresh boot on the same port/DB — the badge must flip offline
  // and come back live via the exponential-backoff reconnect.
  await server.restart();
  await expect(page.locator('#ocr-ws-badge')).toHaveText('offline', { timeout: 15_000 });
  await expect(page.locator('#ocr-ws-badge')).toHaveText('live', { timeout: 30_000 });

  // The re-established socket still drives live updates end-to-end.
  const before = await page.locator('#ocr-jobs tr[data-run]').count();
  await page.setInputFiles('#ocr-file-input', pdfPath);
  await expect(page.locator('#ocr-jobs tr[data-run]')).toHaveCount(before + 1, { timeout: 15_000 });
  const newRow = page.locator('#ocr-jobs tr[data-run]').first();
  await expect(newRow.locator('td:nth-child(4) .ocr-badge')).toHaveText('done', { timeout: 15_000 });
});

test('jobs table paginates past 15 rows', async ({ page }) => {
  // Runs last: uploads 16 unique docs, so it must not precede tests that
  // assume e2e-invoice.pdf is the first row.
  await gotoOcrTab(page);
  await page.setInputFiles('#ocr-file-input', pagePdfPaths);   // one multipart upload

  // Pager appears once total rows exceed the page size (15).
  const pager = page.locator('.ocr-pager');
  await expect(pager).toBeVisible({ timeout: 20_000 });
  await expect(pager).toContainText('page 1 /');
  await expect(page.locator('#ocr-jobs tbody tr')).toHaveCount(15);   // full first page

  // Next → page 2 shows the remaining rows, Prev returns.
  await pager.locator('[data-role="next"]').click();
  await expect(page.locator('.ocr-pager')).toContainText('page 2 /');
  const page2 = await page.locator('#ocr-jobs tbody tr').count();
  expect(page2).toBeGreaterThan(0);
  expect(page2).toBeLessThanOrEqual(15);
  await page.locator('.ocr-pager [data-role="prev"]').click();
  await expect(page.locator('.ocr-pager')).toContainText('page 1 /');
  await expect(page.locator('#ocr-jobs tbody tr')).toHaveCount(15);
});
