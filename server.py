import asyncio
import base64
import json
import re
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from openai_client import transcribe_audio, text_to_speech, chat_response_stream

app = FastAPI()

SYSTEM_PROMPT = (
    "You are a helpful, friendly conversational AI assistant. "
    "Keep responses concise and natural-sounding for voice conversation. "
    "Avoid markdown, bullet points, numbered lists, or special characters — "
    "speak in plain, natural sentences only."
)

# Split at sentence-ending punctuation followed by whitespace or end-of-string
_SENT_RE = re.compile(r'[.!?]+(?:\s+|$)')


def _split_speakable(buf: str) -> tuple[str, str]:
    """Return (ready_to_speak, remainder) at the last sentence boundary."""
    matches = list(_SENT_RE.finditer(buf))
    if not matches:
        return "", buf
    m = matches[-1]
    return buf[: m.end()].strip(), buf[m.end() :]


# ── HTTP ──────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(HTML)


# ── WebSocket ─────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()

    history = [{"role": "system", "content": SYSTEM_PROMPT}]
    config = {"voice": "alloy", "language": "en"}
    active: asyncio.Task | None = None

    async def send(obj: dict):
        await ws.send_text(json.dumps(obj))

    async def cancel():
        nonlocal active
        if active and not active.done():
            active.cancel()
            try:
                await active
            except (asyncio.CancelledError, Exception):
                pass
        active = None

    try:
        while True:
            msg = await ws.receive()

            audio_bytes = msg.get("bytes")
            text_data   = msg.get("text")

            if audio_bytes:
                await cancel()
                active = asyncio.create_task(
                    _handle_turn(send, audio_bytes, history, config)
                )
            elif text_data:
                data = json.loads(text_data)
                kind = data.get("type", "")

                if kind == "interrupt":
                    await cancel()
                    await send({"type": "interrupted"})

                elif kind == "config":
                    config.update({k: v for k, v in data.items() if k != "type"})

                elif kind == "clear":
                    await cancel()
                    history[1:] = []
                    await send({"type": "cleared"})

    except WebSocketDisconnect:
        await cancel()


async def _handle_turn(send, audio_bytes: bytes, history: list, config: dict):
    voice = config.get("voice", "alloy")
    language = config.get("language", "en")

    try:
        await send({"type": "status", "msg": "Transcribing…"})
        transcript = await asyncio.to_thread(transcribe_audio, audio_bytes, language)

        if not transcript.strip():
            await send({"type": "empty"})
            return

        await send({"type": "transcript", "text": transcript})
        history.append({"role": "user", "content": transcript})

        await send({"type": "status", "msg": "Thinking…"})
        full_reply = ""
        buffer = ""

        async for token in chat_response_stream(history):
            full_reply += token
            buffer += token
            await send({"type": "token", "text": token})

            speakable, buffer = _split_speakable(buffer)
            if speakable:
                audio = await asyncio.to_thread(text_to_speech, speakable, voice)
                await send({"type": "audio_chunk", "data": base64.b64encode(audio).decode()})

        if buffer.strip():
            audio = await asyncio.to_thread(text_to_speech, buffer.strip(), voice)
            await send({"type": "audio_chunk", "data": base64.b64encode(audio).decode()})

        history.append({"role": "assistant", "content": full_reply})
        await send({"type": "done"})

    except asyncio.CancelledError:
        raise
    except Exception as exc:
        await send({"type": "error", "msg": str(exc)})


# ── HTML / CSS / JS ───────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Voice Assistant</title>
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  background: #0a0a14;
  color: #e2e8f0;
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 28px 16px 20px;
  gap: 20px;
}

h1 { font-size: 1.1rem; font-weight: 500; color: #94a3b8; letter-spacing: 0.08em; text-transform: uppercase; }

/* ── Orb ──────────────────────────────────────────── */
#orb-wrap {
  position: relative;
  width: 220px; height: 220px;
  display: flex; align-items: center; justify-content: center;
  flex-shrink: 0;
}

.ring {
  position: absolute;
  border-radius: 50%;
  border: 1.5px solid transparent;
  opacity: 0;
  pointer-events: none;
}

#orb {
  width: 110px; height: 110px;
  border-radius: 50%;
  background: radial-gradient(circle at 35% 32%, #3b2d6b, #1a0f3a);
  box-shadow: 0 0 30px rgba(99, 62, 180, 0.25);
  transition: background 0.5s, box-shadow 0.5s;
  position: relative;
  z-index: 1;
}

/* State: listening */
.state-listening .ring {
  border-color: rgba(99, 62, 180, 0.5);
  animation: ring-expand 2.4s ease-out infinite;
}
.state-listening .ring:nth-child(2) { animation-delay: 0.8s; }
.state-listening .ring:nth-child(3) { animation-delay: 1.6s; }
.state-listening #orb {
  background: radial-gradient(circle at 35% 32%, #6d4ae0, #3a1d8f);
  box-shadow: 0 0 50px rgba(99, 62, 180, 0.5);
  animation: orb-breathe 3s ease-in-out infinite;
}

/* State: recording */
.state-recording .ring {
  border-color: rgba(220, 38, 38, 0.5);
  animation: ring-expand 1.2s ease-out infinite;
}
.state-recording .ring:nth-child(2) { animation-delay: 0.4s; }
.state-recording .ring:nth-child(3) { animation-delay: 0.8s; }
.state-recording #orb {
  background: radial-gradient(circle at 35% 32%, #ef4444, #991b1b);
  box-shadow: 0 0 60px rgba(220, 38, 38, 0.6);
  animation: orb-pulse-fast 0.7s ease-in-out infinite;
}

/* State: processing */
.state-processing #orb {
  background: radial-gradient(circle at 35% 32%, #f59e0b, #92400e);
  box-shadow: 0 0 50px rgba(245, 158, 11, 0.4);
  animation: orb-spin 1.8s linear infinite;
}

/* State: playing */
.state-playing .ring {
  border-color: rgba(16, 185, 129, 0.4);
  animation: ring-expand 2s ease-out infinite;
}
.state-playing .ring:nth-child(2) { animation-delay: 0.67s; }
.state-playing .ring:nth-child(3) { animation-delay: 1.33s; }
.state-playing #orb {
  background: radial-gradient(circle at 35% 32%, #10b981, #065f46);
  box-shadow: 0 0 60px rgba(16, 185, 129, 0.5);
  animation: orb-breathe-play 1.4s ease-in-out infinite;
}

@keyframes ring-expand {
  0%   { width: 110px; height: 110px; opacity: 0.8; }
  100% { width: 220px; height: 220px; opacity: 0; }
}
@keyframes orb-breathe {
  0%, 100% { transform: scale(1);    }
  50%       { transform: scale(1.06); }
}
@keyframes orb-pulse-fast {
  0%, 100% { transform: scale(1);    }
  50%       { transform: scale(1.13); }
}
@keyframes orb-spin {
  from { filter: hue-rotate(0deg);   }
  to   { filter: hue-rotate(360deg); }
}
@keyframes orb-breathe-play {
  0%, 100% { transform: scale(1);    }
  50%       { transform: scale(1.09); }
}

/* ── Status ───────────────────────────────────────── */
#status {
  font-size: 0.88rem;
  color: #64748b;
  min-height: 1.2em;
  text-align: center;
  letter-spacing: 0.02em;
}

/* ── Chat ─────────────────────────────────────────── */
#chat {
  width: 100%; max-width: 640px;
  flex: 1;
  overflow-y: auto;
  display: flex; flex-direction: column; gap: 10px;
  padding: 0 4px;
  min-height: 100px;
  max-height: 38vh;
}

.bubble {
  padding: 10px 15px;
  border-radius: 18px;
  max-width: 82%;
  font-size: 0.93rem;
  line-height: 1.6;
  word-wrap: break-word;
}
.bubble.user {
  background: #2d1b69;
  align-self: flex-end;
  border-bottom-right-radius: 4px;
}
.bubble.assistant {
  background: #1e293b;
  align-self: flex-start;
  border-bottom-left-radius: 4px;
}
.bubble.live {
  background: #162032;
  align-self: flex-start;
  border-bottom-left-radius: 4px;
  border: 1px solid #334155;
  color: #94a3b8;
}
.bubble.live::after {
  content: '▋';
  animation: cursor-blink 0.8s step-end infinite;
  margin-left: 2px;
}
@keyframes cursor-blink {
  0%, 100% { opacity: 1; } 50% { opacity: 0; }
}

/* ── Controls ─────────────────────────────────────── */
#controls {
  width: 100%; max-width: 640px;
  display: flex; flex-direction: column;
  align-items: center; gap: 14px;
}

#btn {
  width: 130px; height: 48px;
  border-radius: 24px; border: none;
  font-size: 1rem; font-weight: 600; cursor: pointer;
  background: #6d4ae0; color: #fff;
  transition: background 0.2s, opacity 0.2s;
}
#btn:hover  { background: #7c5cef; }
#btn:active { opacity: 0.85; }
#btn.stopping { background: #475569; cursor: default; }

.row { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; justify-content: center; }

select, #clear-btn {
  background: #1e293b; color: #cbd5e1;
  border: 1px solid #334155; border-radius: 8px;
  padding: 6px 10px; font-size: 0.82rem; cursor: pointer;
}
select:focus, #clear-btn:focus { outline: none; border-color: #6d4ae0; }
#clear-btn:hover { border-color: #475569; color: #e2e8f0; }

#vol-wrap { width: 180px; height: 3px; background: #1e293b; border-radius: 2px; overflow: hidden; }
#vol-fill  { height: 100%; width: 0%; background: #6d4ae0; transition: width 0.05s; }
</style>
</head>
<body>

<h1>Voice Assistant</h1>

<div id="orb-wrap">
  <div class="ring"></div>
  <div class="ring"></div>
  <div class="ring"></div>
  <div id="orb"></div>
</div>

<div id="status">Press Start to begin</div>

<div id="chat"></div>

<div id="controls">
  <div id="vol-wrap"><div id="vol-fill"></div></div>
  <button id="btn" onclick="toggle()">Start</button>
  <div class="row">
    <select id="voice">
      <option value="alloy">Alloy</option>
      <option value="nova">Nova</option>
      <option value="shimmer">Shimmer</option>
      <option value="echo">Echo</option>
      <option value="onyx">Onyx</option>
      <option value="fable">Fable</option>
    </select>
    <select id="lang">
      <option value="en">English</option>
      <option value="hi">Hindi</option>
      <option value="es">Spanish</option>
      <option value="fr">French</option>
      <option value="de">German</option>
      <option value="ja">Japanese</option>
      <option value="zh">Chinese</option>
    </select>
    <button id="clear-btn" onclick="clearChat()">Clear</button>
  </div>
</div>

<script>
// ── Config ─────────────────────────────────────────────────────────────────────
const VAD_THRESHOLD  = 0.02;   // RMS for speech detection
const MIN_SPEECH_MS  = 300;    // ignore noise blips shorter than this
const SILENCE_GAP_MS = 1200;   // ms of silence → end turn
const VAD_INTERVAL_MS= 50;     // VAD check frequency

// ── State ──────────────────────────────────────────────────────────────────────
let phase = 'idle';   // idle | listening | recording | processing | playing
let running = false;

let ws = null;
let audioCtx = null;
let analyser = null;
let micStream = null;
let recorder = null;
let chunks   = [];

let speechStart  = null;
let silenceStart = null;
let vadTimer     = null;

// Audio playback queue state
let nextAudioAt   = 0;       // AudioContext.currentTime when next chunk can start
let activeSources = [];      // currently scheduled/playing AudioBufferSourceNodes
let pendingDecodes = 0;      // chunks received but not yet decoded+scheduled
let allChunksSent  = false;  // server sent "done"

// DOM refs
const orbWrap = document.getElementById('orb-wrap');
const statusEl= document.getElementById('status');
const volFill = document.getElementById('vol-fill');
const chatEl  = document.getElementById('chat');
const btn     = document.getElementById('btn');

// ── State helpers ──────────────────────────────────────────────────────────────
function setState(s) {
  phase = s;
  orbWrap.className = s !== 'idle' ? `state-${s}` : '';
  const labels = {
    idle:       'Press Start to begin',
    listening:  'Listening…',
    recording:  'Hearing you…',
    processing: 'Thinking…',
    playing:    'Speaking…',
  };
  setStatus(labels[s] || '');
}

function setStatus(msg) { statusEl.textContent = msg; }

// ── Main toggle ────────────────────────────────────────────────────────────────
async function toggle() {
  if (!running) await startSession();
  else          stopSession();
}

async function startSession() {
  try {
    micStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
  } catch {
    setStatus('Microphone permission denied'); return;
  }

  audioCtx = new AudioContext();
  analyser  = audioCtx.createAnalyser();
  analyser.fftSize = 512;
  audioCtx.createMediaStreamSource(micStream).connect(analyser);

  ws = new WebSocket(`ws://${location.host}/ws`);
  ws.binaryType = 'arraybuffer';

  ws.onmessage = e => handleServerMsg(JSON.parse(e.data));
  ws.onerror   = () => setStatus('Connection error');
  ws.onclose   = () => { if (running) setStatus('Disconnected'); };

  running = true;
  btn.textContent = 'Stop';

  sendConfig();
  setState('listening');
  vadTimer = setInterval(vadTick, VAD_INTERVAL_MS);
}

function stopSession() {
  running = false;
  clearInterval(vadTimer);
  stopAllAudio();
  if (recorder && recorder.state !== 'inactive') recorder.stop();
  micStream?.getTracks().forEach(t => t.stop());
  audioCtx?.close();
  ws?.close();
  audioCtx = analyser = micStream = recorder = ws = null;
  speechStart = silenceStart = null;
  volFill.style.width = '0%';
  removeLiveBubble();
  setState('idle');
  btn.textContent = 'Start';
}

// ── VAD tick ───────────────────────────────────────────────────────────────────
function vadTick() {
  // Mic is off during processing and playback — no interruption
  if (!running || phase === 'idle' || phase === 'processing' || phase === 'playing') return;

  const level = rms();
  const isSpeech = level > VAD_THRESHOLD;
  const now = Date.now();

  volFill.style.width = Math.min((level / VAD_THRESHOLD) * 60, 100) + '%';

  if (phase === 'listening') {
    if (isSpeech) {
      if (!speechStart) speechStart = now;
      if (now - speechStart >= MIN_SPEECH_MS) beginRecording();
    } else {
      speechStart = null;
    }
  }

  if (phase === 'recording') {
    if (isSpeech) {
      silenceStart = null;
    } else {
      if (!silenceStart) silenceStart = now;
      if (now - silenceStart >= SILENCE_GAP_MS) endTurn();
    }
  }
}

// ── Recording ─────────────────────────────────────────────────────────────────
function beginRecording() {
  setState('recording');
  silenceStart = null;
  speechStart  = null;
  chunks = [];
  recorder = new MediaRecorder(micStream, { mimeType: 'audio/webm' });
  recorder.ondataavailable = e => { if (e.data.size > 0) chunks.push(e.data); };
  recorder.start(100);
}

async function endTurn() {
  if (phase !== 'recording') return;
  setState('processing');
  silenceStart = speechStart = null;

  await new Promise(res => { recorder.onstop = res; recorder.stop(); });
  const blob = new Blob(chunks, { type: 'audio/webm' });
  chunks = [];

  // Reset audio queue state
  allChunksSent  = false;
  pendingDecodes  = 0;
  nextAudioAt    = 0;

  ws.send(await blob.arrayBuffer());
}

// ── Audio playback ─────────────────────────────────────────────────────────────
function ensureAudioCtx() {
  if (!audioCtx) {
    audioCtx = new AudioContext();
    analyser  = audioCtx.createAnalyser();
    analyser.fftSize = 512;
    if (micStream) audioCtx.createMediaStreamSource(micStream).connect(analyser);
  }
  if (audioCtx.state === 'suspended') audioCtx.resume();
}

async function scheduleChunk(b64mp3) {
  ensureAudioCtx();
  pendingDecodes++;

  const raw = atob(b64mp3);
  const ab  = new ArrayBuffer(raw.length);
  const u8  = new Uint8Array(ab);
  for (let i = 0; i < raw.length; i++) u8[i] = raw.charCodeAt(i);

  try {
    const decoded = await audioCtx.decodeAudioData(ab);
    if (phase !== 'playing' && phase !== 'processing') {
      // Interrupted before decode finished
      pendingDecodes--;
      return;
    }
    setState('playing');

    const src = audioCtx.createBufferSource();
    src.buffer = decoded;
    src.connect(audioCtx.destination);

    // Schedule seamlessly after the previous chunk
    const when = Math.max(audioCtx.currentTime + 0.05, nextAudioAt);
    src.start(when);
    nextAudioAt = when + decoded.duration;
    activeSources.push(src);

    src.onended = () => {
      activeSources = activeSources.filter(s => s !== src);
      pendingDecodes--;
      checkPlaybackDone();
    };
  } catch (err) {
    console.error('Audio decode error:', err);
    pendingDecodes--;
    checkPlaybackDone();
  }
}

function checkPlaybackDone() {
  // Accept both 'playing' and 'processing' — the latter covers the race where
  // "done" arrives before the first audio chunk finishes decoding.
  if (allChunksSent && pendingDecodes <= 0 && activeSources.length === 0
      && (phase === 'playing' || phase === 'processing')) {
    const live = document.getElementById('live-bubble');
    if (live) {
      live.id = '';
      live.className = 'bubble assistant';
    }
    if (running) setState('listening');
  }
}

function stopAllAudio() {
  for (const src of activeSources) {
    try { src.stop(0); src.disconnect(); } catch {}
  }
  activeSources = [];
  pendingDecodes  = 0;
  nextAudioAt    = 0;
  allChunksSent  = false;
}

// ── Server message handler ─────────────────────────────────────────────────────
function handleServerMsg(data) {
  switch (data.type) {
    case 'status':
      setStatus(data.msg);
      break;

    case 'transcript':
      addBubble('user', data.text);
      break;

    case 'token':
      appendLiveToken(data.text);
      break;

    case 'audio_chunk':
      scheduleChunk(data.data);
      break;

    case 'done':
      allChunksSent = true;
      checkPlaybackDone();
      break;

    case 'empty':
      setStatus('Didn\'t catch that — try again');
      if (running) setTimeout(() => setState('listening'), 1500);
      break;

    case 'interrupted':
      // Server acknowledged interrupt; client already started recording
      break;

    case 'cleared':
      chatEl.innerHTML = '';
      break;

    case 'error':
      setStatus('Error: ' + data.msg);
      if (running) setTimeout(() => setState('listening'), 2500);
      break;
  }
}

// ── Chat UI ────────────────────────────────────────────────────────────────────
function addBubble(role, text) {
  const d = document.createElement('div');
  d.className = `bubble ${role}`;
  d.textContent = text;
  chatEl.appendChild(d);
  chatEl.scrollTop = chatEl.scrollHeight;
}

function appendLiveToken(token) {
  let live = document.getElementById('live-bubble');
  if (!live) {
    live = document.createElement('div');
    live.id = 'live-bubble';
    live.className = 'bubble live';
    chatEl.appendChild(live);
  }
  live.textContent += token;
  chatEl.scrollTop = chatEl.scrollHeight;
}

function removeLiveBubble() {
  const live = document.getElementById('live-bubble');
  if (live) live.remove();
}

// ── Helpers ────────────────────────────────────────────────────────────────────
function rms() {
  if (!analyser) return 0;
  const buf = new Float32Array(analyser.fftSize);
  analyser.getFloatTimeDomainData(buf);
  let s = 0;
  for (const v of buf) s += v * v;
  return Math.sqrt(s / buf.length);
}

function sendConfig() {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({
      type:     'config',
      voice:    document.getElementById('voice').value,
      language: document.getElementById('lang').value,
    }));
  }
}

async function clearChat() {
  removeLiveBubble();
  chatEl.innerHTML = '';
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'clear' }));
  }
}

// Re-send config whenever the user changes a dropdown
document.getElementById('voice').addEventListener('change', sendConfig);
document.getElementById('lang').addEventListener('change', sendConfig);
</script>
</body>
</html>
"""
