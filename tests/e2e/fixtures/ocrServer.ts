/**
 * Launches the real ocr-server (Python/FastAPI) with the mock engine and
 * tmp storage roots — no GPU, no model weights. Requires the app venv
 * (apps/ocr-server/init.sh --no-gpu --no-model) to exist.
 */
import { spawn, type ChildProcess } from 'node:child_process';
import { mkdtempSync, rmSync, existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));

export interface OcrServer {
  url: string;
  tmpDir: string;
  /** SIGKILL the server and boot a fresh one on the same port + tmp dirs
   *  (same DB — exercises frontend WS reconnect and queue resume). */
  restart: () => Promise<void>;
  close: () => Promise<void>;
}

export async function startOcrServer(port = 8093): Promise<OcrServer> {
  const appDir = resolve(__dirname, '../../../apps/ocr-server');
  const python = join(appDir, '.venv/bin/python');
  if (!existsSync(python)) {
    throw new Error(
      `ocr-server venv missing at ${python} — run apps/ocr-server/init.sh --no-gpu --no-model`);
  }
  const tmpDir = mkdtempSync(join(tmpdir(), 'ocr-e2e-'));
  const env = {
    ...process.env,
    OCR_PORT: String(port),
    OCR_ENGINE: 'mock',
    OCR_EXTRACT_MODEL: 'mock-extract',
    OCR_MEDIA_DIR: join(tmpDir, 'media'),
    OCR_RESULTS_DIR: join(tmpDir, 'results'),
    OCR_DB_PATH: join(tmpDir, 'ocr.sqlite3'),
    OCR_MOCK_DELAY_SECS: '0.05',
  };
  const url = `http://127.0.0.1:${port}`;
  let proc: ChildProcess;
  const logs: string[] = [];

  async function boot(): Promise<void> {
    proc = spawn(
      python,
      ['-m', 'uvicorn', 'ocr_server.main:app', '--host', '127.0.0.1',
       '--port', String(port), '--log-level', 'warning'],
      { cwd: appDir, env, stdio: ['ignore', 'pipe', 'pipe'] },
    );
    proc.stdout?.on('data', (d) => logs.push(String(d)));
    proc.stderr?.on('data', (d) => logs.push(String(d)));
    const deadline = Date.now() + 30_000;
    while (Date.now() < deadline) {
      if (proc.exitCode !== null) {
        throw new Error(`ocr-server exited early (${proc.exitCode}):\n${logs.join('')}`);
      }
      try {
        const r = await fetch(`${url}/health`);
        if (r.ok) return;
      } catch {
        // not up yet
      }
      await new Promise((r) => setTimeout(r, 250));
    }
    proc.kill('SIGKILL');
    throw new Error(`ocr-server did not become healthy in 30s:\n${logs.join('')}`);
  }

  async function stop(sig: NodeJS.Signals): Promise<void> {
    if (proc.exitCode !== null) return;
    proc.kill(sig);
    await new Promise<void>((done) => {
      const t = setTimeout(() => { proc.kill('SIGKILL'); done(); }, 5000);
      proc.once('exit', () => { clearTimeout(t); done(); });
    });
  }

  await boot();
  return {
    url,
    tmpDir,
    restart: async () => {
      await stop('SIGKILL');
      // Give the OS a beat to release the port.
      await new Promise((r) => setTimeout(r, 500));
      await boot();
    },
    close: async () => {
      await stop('SIGTERM');
      rmSync(tmpDir, { recursive: true, force: true });
    },
  };
}
