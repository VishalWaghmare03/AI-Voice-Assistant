# Voice Assistant

A real-time voice assistant that streams LLM responses and text-to-speech audio back to you sentence by sentence — similar to ChatGPT Voice. Built with FastAPI, WebSockets, OpenAI Whisper, GPT-4o-mini, and the Web Audio API.

---

## Demo Flow

```
You speak  →  Whisper (STT)  →  GPT-4o-mini (streaming)  →  OpenAI TTS  →  You hear
               ~500ms              tokens as they arrive        per sentence
```

The assistant starts speaking after the **first sentence** is generated — not after the full response — which dramatically cuts perceived latency.

---

## Features

- **One-click session** — press Start once; the mic auto-reactivates after every response
- **Streaming pipeline** — LLM tokens arrive live, TTS runs sentence-by-sentence
- **Seamless audio queue** — `AudioContext` schedules MP3 chunks back-to-back with no gaps
- **Mic off during playback** — VAD is paused while the assistant is speaking
- **Animated orb UI** — visual state feedback (listening / recording / thinking / speaking)
- **6 voices** — Alloy, Nova, Shimmer, Echo, Onyx, Fable
- **7 languages** — English, Hindi, Spanish, French, German, Japanese, Chinese
- **Chat history** — transcript scrolls in real time as the LLM generates

---

## Architecture

```
Browser
  │
  │  WebSocket /ws
  │  ├── binary frames  →  audio blob (WebM/Opus from MediaRecorder)
  │  └── JSON frames    →  config / interrupt / clear
  │
FastAPI (server.py)
  │
  ├── asyncio.to_thread → Whisper API  (transcription)
  ├── AsyncOpenAI       → GPT-4o-mini  (streaming tokens)
  │     └── sentence buffer → TTS per sentence
  └── asyncio.to_thread → OpenAI TTS   (MP3 bytes)
  │
  │  WebSocket responses
  │  ├── {type: "transcript"}   — what you said
  │  ├── {type: "token"}        — live LLM output
  │  ├── {type: "audio_chunk"}  — base64 MP3 per sentence
  │  └── {type: "done"}         — playback complete, mic resumes
  │
Browser AudioContext
  └── decodeAudioData + scheduleBufferSource → seamless playback
```

---

## Prerequisites

- Python 3.11+
- An [OpenAI API key](https://platform.openai.com/api-keys)
- A modern browser (Chrome / Edge recommended; Firefox works)

---

## Setup

**1. Clone the repository**

```bash
git clone https://github.com/VishalWaghmare03/AI-Voice-Assistant.git
cd voice-assistant
```

**2. Create a virtual environment and install dependencies**

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**3. Add your OpenAI API key**

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-...
```

**4. Start the server**

```bash
uvicorn server:app --reload --host 0.0.0.0 --port 8000
```

**5. Open in your browser**

```
http://localhost:8000
```

---

## Usage

1. Click **Start** and allow microphone access when prompted
2. Speak naturally — the orb turns red while it is recording you
3. Pause for ~1.2 seconds — the app detects silence and sends your audio automatically
4. The orb turns yellow (thinking) then green (speaking)
5. After the response finishes, the orb turns purple and the mic is live again — no button needed
6. Click **Stop** only when you want to end the entire session

### Controls

| Control | Description |
|---|---|
| **Start / Stop** | Begin or end a session |
| **Voice** | Choose the TTS voice (Alloy, Nova, Shimmer, Echo, Onyx, Fable) |
| **Language** | Hint to Whisper for better accuracy on non-English speech |
| **Clear** | Wipe the chat history and reset conversation context |

---

## Project Structure

```
voice_app/
├── server.py          # FastAPI app — WebSocket endpoint + inline HTML/CSS/JS
├── openai_client.py   # OpenAI wrappers (Whisper, streaming GPT, TTS)
├── app.py             # Alternative Streamlit UI (simpler, no streaming)
├── audio_handler.py   # Standalone mic capture + WebRTC VAD (desktop use)
├── requirements.txt
└── .env               # Your OPENAI_API_KEY (not committed)
```

> `app.py` and `audio_handler.py` are an earlier Streamlit-based version. The primary app is `server.py`.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, FastAPI, asyncio |
| Real-time transport | WebSocket (Starlette) |
| Speech-to-text | OpenAI Whisper (`whisper-1`) |
| Language model | OpenAI GPT-4o-mini (streaming) |
| Text-to-speech | OpenAI TTS (`tts-1`) |
| Frontend audio capture | `getUserMedia` + `MediaRecorder` |
| Frontend VAD | Web Audio API `AnalyserNode` (RMS) |
| Frontend playback | `AudioContext` + `AudioBufferSourceNode` |
| UI | Vanilla HTML/CSS/JS (served inline) |

---

## Configuration

All VAD and timing constants are at the top of the `<script>` block in `server.py`:

| Constant | Default | Description |
|---|---|---|
| `VAD_THRESHOLD` | `0.02` | RMS amplitude that counts as speech |
| `MIN_SPEECH_MS` | `300` | Minimum speech duration before recording starts |
| `SILENCE_GAP_MS` | `1200` | Silence duration that ends a turn |
| `VAD_INTERVAL_MS` | `50` | How often VAD runs (ms) |

---

## .gitignore

Add this to avoid committing secrets and build artifacts:

```
.env
venv/
__pycache__/
*.pyc
```

---

## License

MIT
