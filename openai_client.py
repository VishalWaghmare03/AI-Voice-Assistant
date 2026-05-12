import os
import io
from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

_client: OpenAI | None = None
_async_client: AsyncOpenAI | None = None


def _api_key() -> str:
    key = os.getenv("OPENAI_API_KEY", "").strip().strip('"').strip("'")
    if not key or key == "your_openai_api_key_here":
        raise ValueError("Set a valid OPENAI_API_KEY in your .env file.")
    return key


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=_api_key())
    return _client


def get_async_client() -> AsyncOpenAI:
    global _async_client
    if _async_client is None:
        _async_client = AsyncOpenAI(api_key=_api_key())
    return _async_client


def transcribe_audio(audio_bytes: bytes, language: str = "en") -> str:
    client = get_client()
    buf = io.BytesIO(audio_bytes)
    buf.name = "recording.webm"
    return client.audio.transcriptions.create(
        model="whisper-1",
        file=buf,
        language=language,
    ).text.strip()


def chat_response(messages: list[dict]) -> str:
    return get_client().chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.7,
    ).choices[0].message.content.strip()


async def chat_response_stream(messages: list[dict]):
    """Async generator yielding text tokens from GPT streaming."""
    stream = await get_async_client().chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.7,
        stream=True,
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def text_to_speech(text: str, voice: str = "alloy") -> bytes:
    return get_client().audio.speech.create(
        model="tts-1",
        voice=voice,
        input=text,
    ).content
