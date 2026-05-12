import sounddevice as sd
import numpy as np
import webrtcvad
import queue
import threading
import time
from scipy.io import wavfile
import io
import wave


SAMPLE_RATE = 16000
CHANNELS = 1
FRAME_DURATION_MS = 30  # 10, 20, or 30 ms — WebRTC VAD requirement
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)
SILENCE_THRESHOLD_SECONDS = 1.2
VAD_AGGRESSIVENESS = 2  # 0–3, higher = more aggressive filtering


class AudioHandler:
    def __init__(self):
        self.vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self._stream = None
        self._buffer = []
        self._lock = threading.Lock()

    def _callback(self, indata, frames, time_info, status):
        if self.is_recording:
            self.audio_queue.put(indata.copy())

    def start_stream(self):
        self.is_recording = True
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype='int16',
            blocksize=FRAME_SIZE,
            callback=self._callback,
        )
        self._stream.start()

    def stop_stream(self):
        self.is_recording = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def capture_until_silence(self, status_callback=None):
        """
        Reads from the audio queue, uses VAD to detect speech,
        and returns a WAV bytes object once silence is detected after speech.
        """
        frames = []
        speech_frames = []
        silent_frames = 0
        speech_started = False
        frames_for_silence = int(SILENCE_THRESHOLD_SECONDS * 1000 / FRAME_DURATION_MS)

        if status_callback:
            status_callback("Listening... (speak now)")

        while True:
            try:
                chunk = self.audio_queue.get(timeout=0.1)
            except queue.Empty:
                if not self.is_recording:
                    break
                continue

            pcm = chunk.flatten().tobytes()

            # WebRTC VAD expects exactly FRAME_SIZE * 2 bytes (16-bit)
            if len(pcm) != FRAME_SIZE * 2:
                continue

            try:
                is_speech = self.vad.is_speech(pcm, SAMPLE_RATE)
            except Exception:
                is_speech = False

            if is_speech:
                speech_started = True
                silent_frames = 0
                speech_frames.append(chunk.flatten())
                if status_callback:
                    status_callback("Detecting speech...")
            else:
                if speech_started:
                    silent_frames += 1
                    speech_frames.append(chunk.flatten())
                    if silent_frames >= frames_for_silence:
                        if status_callback:
                            status_callback("Processing...")
                        break

        if not speech_frames:
            return None

        audio_data = np.concatenate(speech_frames).astype(np.int16)
        return _to_wav_bytes(audio_data, SAMPLE_RATE)


def _to_wav_bytes(audio_np: np.ndarray, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(audio_np.tobytes())
    buf.seek(0)
    return buf.read()


def play_audio_bytes(audio_bytes: bytes):
    """Play raw audio bytes (MP3 from OpenAI TTS) via sounddevice."""
    import tempfile
    import os
    import subprocess

    # Write to temp file and play using system player as fallback
    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name

    try:
        # Try ffplay (ffmpeg), aplay, or mpg123 — common on Linux
        for player in ['ffplay -nodisp -autoexit', 'mpg123', 'cvlc --play-and-exit']:
            cmd = player.split() + [tmp_path]
            result = subprocess.run(cmd, capture_output=True, timeout=30)
            if result.returncode == 0:
                break
    except Exception:
        pass
    finally:
        os.unlink(tmp_path)
