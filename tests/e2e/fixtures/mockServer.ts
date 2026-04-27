/**
 * Minimal mock Starling server used by the Playwright E2E suite.
 *
 * Responsibilities:
 *   - Serve the shared `frontend/` directory statically.
 *   - Respond to `/api/*` endpoints with the minimum JSON shape the
 *     frontend expects (models, modes, media, sessions, ...).
 *   - Implement the WebSocket handshake for BOTH the regular "receiver"
 *     flow (`ready` → offer → answer) and the new "uplink" flow used by
 *     Speakers sessions (`ready {role:"uplink"}` → welcome → offer → answer).
 *
 * The mock is NOT a real WebRTC endpoint — it simply records what the
 * client sends and emits an SDP answer that the browser will treat as
 * valid until ICE fails. That is enough for the E2E test to verify:
 *     1. Device enumeration / source switching.
 *     2. Session creation via POST /api/sessions.
 *     3. Client sends `{type:"ready", role:"uplink"}` and then an SDP offer.
 */

import * as http from 'node:http';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { WebSocketServer } from 'ws';

type SessionInfo = {
  id: string;
  state: string;
  model_id: string;
  model_name: string;
  quant: string;
  media_id: string | null;
  media_filename: string;
  language: string;
  mode: string;
  duration_secs: number;
  progress_secs: number;
  created_at: number;
  client_count: number;
  source_type: string;
  noise_cancellation: string;
  diarization: boolean;
  sentence_completion: string;
};

export type SignalingEvent = {
  clientId: string;
  type: string;
  raw: any;
};

export type TtsOutputFile = {
  name: string;
  bytes: number;
  created_at: number;
};

export type TtsStatus = {
  state: 'idle' | 'starting' | 'ready' | 'stopping' | 'blocked';
  pid?: number | null;
  boot_started_at?: number | null;
  boot_elapsed_secs?: number | null;
  boot_timeout_secs?: number | null;
  blocked_reason?: string | null;
  inflight_synths?: number;
};

export type ClonedVoice = {
  id: string;
  name: string;
  ref_text: string;
  permission_confirmed: boolean;
  sample_rate: number;
  duration_secs: number;
  created_at: number;
  kind: 'cloned';
};

export type MockServer = {
  port: number;
  url: string;
  events: SignalingEvent[];
  sessions: SessionInfo[];
  ttsRequests: any[];
  ttsOutputs: TtsOutputFile[];
  /** Mutable: tests can set this to drive the lifecycle badge through
   * different states. Defaults to `{ state: 'ready' }`. */
  ttsStatus: TtsStatus;
  /** Voice clones uploaded during a test. Reset in beforeEach. */
  clonedVoices: ClonedVoice[];
  close: () => Promise<void>;
};

const MOCK_VOICES = [
  { id: 'casual_female',    display_name: 'Casual (Female)',    language: 'English',    language_code: 'en', gender: 'female' },
  { id: 'casual_male',      display_name: 'Casual (Male)',      language: 'English',    language_code: 'en', gender: 'male' },
  { id: 'cheerful_female',  display_name: 'Cheerful (Female)',  language: 'English',    language_code: 'en', gender: 'female' },
  { id: 'neutral_female',   display_name: 'Neutral (Female)',   language: 'English',    language_code: 'en', gender: 'female' },
  { id: 'neutral_male',     display_name: 'Neutral (Male)',     language: 'English',    language_code: 'en', gender: 'male' },
  { id: 'de_female',        display_name: 'German (Female)',    language: 'German',     language_code: 'de', gender: 'female' },
  { id: 'de_male',          display_name: 'German (Male)',      language: 'German',     language_code: 'de', gender: 'male' },
  { id: 'fr_male',          display_name: 'French (Male)',      language: 'French',     language_code: 'fr', gender: 'male' },
  { id: 'fr_female',        display_name: 'French (Female)',    language: 'French',     language_code: 'fr', gender: 'female' },
  { id: 'es_male',          display_name: 'Spanish (Male)',     language: 'Spanish',    language_code: 'es', gender: 'male' },
  { id: 'es_female',        display_name: 'Spanish (Female)',   language: 'Spanish',    language_code: 'es', gender: 'female' },
];

function stubWavBytes(): Buffer {
  // Minimal 44-byte RIFF/WAVE/PCM header + 16 zero samples (stub).
  const dataLen = 16;
  const buf = Buffer.alloc(44 + dataLen);
  buf.write('RIFF', 0);
  buf.writeUInt32LE(36 + dataLen, 4);
  buf.write('WAVE', 8);
  buf.write('fmt ', 12);
  buf.writeUInt32LE(16, 16);          // fmt chunk size
  buf.writeUInt16LE(1, 20);           // PCM
  buf.writeUInt16LE(1, 22);           // mono
  buf.writeUInt32LE(24000, 24);       // sample rate
  buf.writeUInt32LE(24000 * 2, 28);   // byte rate
  buf.writeUInt16LE(2, 32);           // block align
  buf.writeUInt16LE(16, 34);          // bits/sample
  buf.write('data', 36);
  buf.writeUInt32LE(dataLen, 40);
  return buf;
}

/** Streaming-WAV header (44 bytes) with 0xFFFFFFFF size placeholders.
 * Matches the real server's `voxtral_server/tts/wav.py::streaming_header`.
 */
function streamingWavHeader(): Buffer {
  const buf = Buffer.alloc(44);
  buf.write('RIFF', 0);
  buf.writeUInt32LE(0xFFFFFFFF, 4);
  buf.write('WAVE', 8);
  buf.write('fmt ', 12);
  buf.writeUInt32LE(16, 16);
  buf.writeUInt16LE(1, 20);
  buf.writeUInt16LE(1, 22);
  buf.writeUInt32LE(24000, 24);
  buf.writeUInt32LE(24000 * 2, 28);
  buf.writeUInt16LE(2, 32);
  buf.writeUInt16LE(16, 34);
  buf.write('data', 36);
  buf.writeUInt32LE(0xFFFFFFFF, 40);
  return buf;
}

/** Build a Buffer containing `samples` mono int16 zero-samples (silence).
 * Used by the streaming stub to emit several PCM bursts back-to-back. */
function silentPcm(samples: number): Buffer {
  return Buffer.alloc(samples * 2);
}

export async function startMockServer(
  frontendDir: string,
  opts: { port?: number } = {}
): Promise<MockServer> {
  const sessions: SessionInfo[] = [];
  const events: SignalingEvent[] = [];
  const ttsRequests: any[] = [];
  const ttsOutputs: TtsOutputFile[] = [];
  const ttsStatus: TtsStatus = { state: 'ready' };
  const clonedVoices: ClonedVoice[] = [];

  const server = http.createServer(async (req, res) => {
    try {
      const url = new URL(req.url ?? '/', 'http://localhost');
      const pathname = url.pathname;

      if (req.method === 'GET' && pathname === '/api/config') {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({
          wsUrl: `ws://localhost:${(server.address() as any)?.port}/ws`,
          iceServers: [{ urls: 'stun:stun.l.google.com:19302' }],
          iceTransportPolicy: 'all',
          audio: { sampleRate: 16000, channels: 1, bufferSize: 4096 },
          subtitles: { maxSegments: 1000, autoScroll: true, showTimestamps: true },
          speakerColors: ['#4A90D9'],
          reconnect: { enabled: false, delay: 2000, maxDelay: 30000, maxAttempts: 10 },
        }));
        return;
      }

      if (req.method === 'GET' && pathname === '/api/models') {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({
          success: true,
          data: [{
            id: 'voxtral-mini-4b',
            display_name: 'Voxtral Mini 4B (mock)',
            languages: ['en', 'de'],
            quant_options: ['q4'],
            is_loaded: true,
          }],
        }));
        return;
      }

      if (req.method === 'GET' && pathname === '/api/modes') {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({
          success: true,
          data: [
            { id: 'speedy', display_name: 'Speedy', description: 'low-latency' },
            { id: 'growing_segments', display_name: 'Growing Segments', description: '' },
            { id: 'pause_segmented', display_name: 'Pause-Segmented', description: '' },
          ],
        }));
        return;
      }

      if (req.method === 'GET' && pathname === '/api/media') {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ success: true, data: [] }));
        return;
      }

      if (req.method === 'GET' && pathname === '/api/noise-cancellation') {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ success: true, data: [{ id: 'none', name: 'None', description: '', available: true }] }));
        return;
      }

      if (req.method === 'GET' && pathname === '/api/diarization') {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ success: true, data: [{ id: 'none', name: 'None', description: '', available: true }] }));
        return;
      }

      if (req.method === 'GET' && pathname === '/api/srt-streams') {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ success: true, configured: false, streams: [] }));
        return;
      }

      if (req.method === 'GET' && pathname === '/api/sessions') {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ success: true, data: sessions }));
        return;
      }

      if (req.method === 'POST' && pathname === '/api/sessions') {
        const body = await readJson(req);
        const isSpeakers = body.source === 'speakers';
        const sessionId = `mock-${Date.now()}-${Math.random().toString(16).slice(2, 8)}`;
        const info: SessionInfo = {
          id: sessionId,
          state: 'created',
          model_id: body.model_id ?? '',
          model_name: 'Voxtral Mini 4B (mock)',
          quant: body.quant ?? 'q4',
          media_id: body.media_id ?? null,
          media_filename: isSpeakers ? 'Speakers (live capture)' : body.media_id ?? '',
          language: body.language ?? 'de',
          mode: body.mode ?? 'speedy',
          duration_secs: 0,
          progress_secs: 0,
          created_at: Date.now() / 1000,
          client_count: 0,
          source_type: isSpeakers ? 'speakers' : 'file',
          noise_cancellation: body.noise_cancellation ?? 'none',
          diarization: body.diarization ?? false,
          sentence_completion: body.sentence_completion ?? 'minimal',
        };
        sessions.push(info);
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ success: true, data: info }));
        return;
      }

      if (req.method === 'POST' && /\/api\/sessions\/[^/]+\/start$/.test(pathname)) {
        const sid = pathname.split('/')[3];
        const s = sessions.find(x => x.id === sid);
        if (s) s.state = 'running';
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ success: true, data: s ?? null }));
        return;
      }

      if (req.method === 'DELETE' && /\/api\/sessions\/[^/]+$/.test(pathname)) {
        const sid = pathname.split('/')[3];
        const idx = sessions.findIndex(x => x.id === sid);
        if (idx >= 0) sessions.splice(idx, 1);
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ success: true }));
        return;
      }

      // ─── TTS routes ───────────────────────────────────────────────
      if (req.method === 'GET' && pathname === '/api/tts/status') {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({
          success: true,
          data: { ...ttsStatus, autostart: true },
        }));
        return;
      }

      if (req.method === 'GET' && pathname === '/api/tts/voices') {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        const builtins = MOCK_VOICES.map(v => ({ ...v, kind: 'builtin' }));
        res.end(JSON.stringify({
          success: true,
          data: [...builtins, ...clonedVoices],
        }));
        return;
      }

      if (req.method === 'POST' && pathname === '/api/tts/voices/upload') {
        const fields = await readMultipartFields(req);
        const permission = (fields.permission_confirmed ?? '') === 'true';
        if (!permission) {
          res.writeHead(200, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({
            success: false,
            error: 'permission_confirmed must be true',
          }));
          return;
        }
        const id = `mock${Date.now()}${Math.random().toString(36).slice(2, 8)}`.toLowerCase();
        const voice: ClonedVoice = {
          id,
          name: fields.name ?? 'unnamed',
          ref_text: fields.ref_text ?? '',
          permission_confirmed: true,
          sample_rate: 24000,
          duration_secs: 6.0,
          created_at: Date.now() / 1000,
          kind: 'cloned',
        };
        clonedVoices.push(voice);
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ success: true, data: voice }));
        return;
      }

      if (req.method === 'DELETE' && /^\/api\/tts\/voices\/[^/]+$/.test(pathname)) {
        const id = decodeURIComponent(pathname.slice('/api/tts/voices/'.length));
        const idx = clonedVoices.findIndex(v => v.id === id);
        if (idx >= 0) clonedVoices.splice(idx, 1);
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({
          success: idx >= 0,
          data: { deleted: id },
          error: idx >= 0 ? null : 'voice ref not found',
        }));
        return;
      }

      if (req.method === 'GET' && pathname === '/api/tts/config') {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({
          success: true,
          data: {
            output_dir: '/tmp/tts_out_mock',
            max_chars: 20000,
            default_voice: 'casual_male',
            sample_rate: 24000,
            supported_formats: ['wav'],
            long_max_secs: 300,
          },
        }));
        return;
      }

      if (req.method === 'POST' && pathname === '/api/tts/synthesize') {
        const body = await readJson(req);
        ttsRequests.push(body);
        // Basic server-side validation mirroring the real route, just enough
        // for the negative-path E2E cases to assert error shapes.
        if (!body.text || !body.text.trim()) {
          if (body.save === false) {
            res.writeHead(400, { 'Content-Type': 'application/json' });
          } else {
            res.writeHead(200, { 'Content-Type': 'application/json' });
          }
          res.end(JSON.stringify({ success: false, error: 'text is empty' }));
          return;
        }
        if (body.voice && !MOCK_VOICES.some(v => v.id === body.voice)) {
          if (body.save === false) {
            res.writeHead(400, { 'Content-Type': 'application/json' });
          } else {
            res.writeHead(200, { 'Content-Type': 'application/json' });
          }
          res.end(JSON.stringify({ success: false, error: `unknown voice: '${body.voice}'` }));
          return;
        }

        // Streaming branch — chunked audio/wav with 0xFFFFFFFF placeholder
        // sizes, matching the real server's behaviour. We emit the header,
        // then 3 small PCM chunks with a 25ms gap between them so the
        // client-side player exercises buffer scheduling.
        if (body.save === false) {
          res.writeHead(200, {
            'Content-Type': 'audio/wav',
            'Cache-Control': 'no-store',
            'Transfer-Encoding': 'chunked',
          });
          res.write(streamingWavHeader());
          // 3 chunks × ~0.1 s of silence each (24 kHz mono) = 0.3 s total.
          // Emit them with small gaps so the player sees real chunked I/O.
          const chunk = silentPcm(2400);
          (async () => {
            for (let i = 0; i < 3; i++) {
              res.write(chunk);
              await new Promise(r => setTimeout(r, 25));
            }
            res.end();
          })().catch(() => res.end());
          return;
        }

        const filename = body.save_filename || `tts_${body.voice || 'casual_male'}_${Date.now()}.wav`;
        ttsOutputs.push({
          name: filename,
          bytes: 12_345,
          created_at: Date.now() / 1000,
        });
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({
          success: true,
          data: {
            filename,
            path: `/tmp/tts_out_mock/${filename}`,
            bytes: 12_345,
            voice: body.voice || 'casual_male',
            sample_rate: 24000,
            duration_secs: 0.5,
            elapsed_secs: 0.05,
          },
        }));
        return;
      }

      if (req.method === 'GET' && pathname === '/api/tts/output') {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ success: true, data: [...ttsOutputs] }));
        return;
      }

      if (req.method === 'GET' && /^\/api\/tts\/output\/[^/]+$/.test(pathname)) {
        // Serve a tiny valid WAV blob so <audio> can pretend to load it.
        const wav = stubWavBytes();
        res.writeHead(200, { 'Content-Type': 'audio/wav', 'Content-Length': String(wav.length) });
        res.end(wav);
        return;
      }

      if (req.method === 'DELETE' && /^\/api\/tts\/output\/[^/]+$/.test(pathname)) {
        const name = decodeURIComponent(pathname.slice('/api/tts/output/'.length));
        const idx = ttsOutputs.findIndex(o => o.name === name);
        if (idx >= 0) ttsOutputs.splice(idx, 1);
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ success: true, data: { deleted: name } }));
        return;
      }

      // Serve the shared frontend statically
      const rel = pathname === '/' ? '/index.html' : pathname;
      const filePath = path.normalize(path.join(frontendDir, rel));
      if (!filePath.startsWith(frontendDir)) {
        res.writeHead(403);
        res.end('Forbidden');
        return;
      }

      try {
        const content = await fs.promises.readFile(filePath);
        res.writeHead(200, { 'Content-Type': contentType(filePath) });
        res.end(content);
      } catch {
        res.writeHead(404);
        res.end('Not found');
      }
    } catch (e: any) {
      res.writeHead(500, { 'Content-Type': 'text/plain' });
      res.end(`mock server error: ${e.message ?? e}`);
    }
  });

  const wss = new WebSocketServer({ server, path: undefined });
  wss.on('connection', (ws, req) => {
    const pathname = new URL(req.url ?? '/', 'http://localhost').pathname;
    if (!pathname.startsWith('/ws/')) {
      ws.close();
      return;
    }
    const sid = pathname.slice('/ws/'.length);
    const clientId = Math.random().toString(36).slice(2, 10);
    let isUplink = false;

    ws.send(JSON.stringify({
      type: 'welcome',
      message: 'mock server',
      client_id: clientId,
      session: sessions.find(s => s.id === sid) ?? null,
    }));

    ws.on('message', (raw) => {
      let msg: any;
      try {
        msg = JSON.parse(raw.toString());
      } catch {
        return;
      }
      events.push({ clientId, type: msg.type ?? 'unknown', raw: msg });

      if (msg.type === 'ready' && msg.role === 'uplink') {
        isUplink = true;
        // A second welcome tells the uplink client we're ready for its offer
        ws.send(JSON.stringify({ type: 'welcome', message: 'uplink ready', client_id: clientId }));
        return;
      }

      if (msg.type === 'offer' && isUplink) {
        // Reply with a syntactically-valid but non-functional SDP answer.
        // The browser will parse it and attempt ICE; the ICE attempt fails
        // silently, which is fine for our assertions.
        const answerSdp = buildStubAnswer(msg.sdp ?? '');
        ws.send(JSON.stringify({ type: 'answer', sdp: answerSdp }));
        return;
      }

      if (msg.type === 'ice-candidate') {
        // Just acknowledge by pushing an empty ICE candidate back (end-of-candidates)
        ws.send(JSON.stringify({ type: 'ice-candidate', candidate: null }));
        return;
      }
    });
  });

  await new Promise<void>(res => server.listen(opts.port ?? 0, '127.0.0.1', () => res()));
  const addr = server.address() as any;
  const port = addr.port;

  return {
    port,
    url: `http://localhost:${port}`,
    events,
    sessions,
    ttsRequests,
    ttsOutputs,
    ttsStatus,
    clonedVoices,
    async close() {
      wss.close();
      await new Promise<void>(res => server.close(() => res()));
    },
  };
}

/** Tiny multipart parser: returns a flat object of name→last-string-value
 * for fields, ignoring binary parts. Sufficient for our voice-clone tests
 * where we don't need to re-derive the audio bytes. */
async function readMultipartFields(req: http.IncomingMessage): Promise<Record<string, string>> {
  const ct = req.headers['content-type'] ?? '';
  const boundary = /boundary=(.+)$/.exec(ct)?.[1];
  if (!boundary) return {};
  const body: Buffer = await new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    req.on('data', c => chunks.push(c));
    req.on('end', () => resolve(Buffer.concat(chunks)));
    req.on('error', reject);
  });
  const text = body.toString('latin1');     // preserve binary boundaries
  const parts = text.split(`--${boundary}`);
  const fields: Record<string, string> = {};
  for (const part of parts) {
    const m = /Content-Disposition: form-data; name="([^"]+)"(?:; filename="([^"]+)")?\r\n(?:Content-Type:[^\r]*\r\n)?\r\n([\s\S]*?)\r\n$/.exec(part);
    if (!m) continue;
    const [, name, filename, value] = m;
    if (filename) continue;          // skip the audio blob; we don't need it
    fields[name] = value;
  }
  return fields;
}

function contentType(p: string): string {
  const ext = path.extname(p).toLowerCase();
  switch (ext) {
    case '.html': return 'text/html; charset=utf-8';
    case '.js': return 'application/javascript; charset=utf-8';
    case '.css': return 'text/css; charset=utf-8';
    case '.json': return 'application/json; charset=utf-8';
    case '.wasm': return 'application/wasm';
    case '.svg': return 'image/svg+xml';
    default: return 'application/octet-stream';
  }
}

async function readJson(req: http.IncomingMessage): Promise<any> {
  const chunks: Buffer[] = [];
  for await (const c of req) chunks.push(c as Buffer);
  const text = Buffer.concat(chunks).toString('utf8') || '{}';
  return JSON.parse(text);
}

/**
 * Build a minimal, roughly-valid SDP answer derived from the offer.
 * We reuse the offer's mids and ice-ufrag to keep it parseable by the
 * browser; actual media won't flow because the answer has no reachable
 * ICE candidates, but SDP parsing succeeds.
 */
function buildStubAnswer(offer: string): string {
  const lines = offer.split(/\r?\n/);
  const mids: string[] = [];
  let version = 0;
  for (const line of lines) {
    if (line.startsWith('o=')) {
      const parts = line.split(' ');
      if (parts[2]) version = parseInt(parts[2], 10) || 0;
    }
    if (line.startsWith('a=mid:')) {
      mids.push(line.slice('a=mid:'.length).trim());
    }
  }
  if (mids.length === 0) mids.push('0');

  const header = [
    'v=0',
    `o=mock-server ${version + 1} 1 IN IP4 127.0.0.1`,
    's=-',
    't=0 0',
    `a=group:BUNDLE ${mids.join(' ')}`,
    'a=extmap-allow-mixed',
    'a=msid-semantic: WMS',
  ];

  const mediaSections = mids.map(mid =>
    [
      'm=audio 9 UDP/TLS/RTP/SAVPF 111',
      'c=IN IP4 0.0.0.0',
      `a=mid:${mid}`,
      'a=recvonly',
      'a=rtcp-mux',
      'a=ice-ufrag:mock',
      'a=ice-pwd:mockpwdmockpwdmockpwdmockpwd',
      'a=ice-options:trickle',
      'a=fingerprint:sha-256 00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00',
      'a=setup:active',
      'a=rtpmap:111 opus/48000/2',
      'a=fmtp:111 minptime=10;useinbandfec=1',
    ].join('\r\n')
  );

  return header.join('\r\n') + '\r\n' + mediaSections.join('\r\n') + '\r\n';
}
