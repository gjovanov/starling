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

export type MockServer = {
  port: number;
  url: string;
  events: SignalingEvent[];
  sessions: SessionInfo[];
  close: () => Promise<void>;
};

export async function startMockServer(
  frontendDir: string,
  opts: { port?: number } = {}
): Promise<MockServer> {
  const sessions: SessionInfo[] = [];
  const events: SignalingEvent[] = [];

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
    async close() {
      wss.close();
      await new Promise<void>(res => server.close(() => res()));
    },
  };
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
