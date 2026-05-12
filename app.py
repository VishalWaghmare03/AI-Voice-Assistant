import streamlit as st
import base64
import hashlib
from openai_client import transcribe_audio, chat_response, text_to_speech

st.set_page_config(page_title="Voice Chat", page_icon="🎙️", layout="centered")

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": (
            "You are a helpful, conversational AI assistant. "
            "Keep responses concise and natural for voice conversation."
        )}
    ]
if "chat_display" not in st.session_state:
    st.session_state.chat_display = []  # list of (role, text)
if "last_audio" not in st.session_state:
    st.session_state.last_audio = None
if "processed_audio_id" not in st.session_state:
    st.session_state.processed_audio_id = None

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    tts_voice = st.selectbox("Voice", ["alloy", "echo", "fable", "onyx", "nova", "shimmer"])
    language = st.selectbox("Language", ["en", "hi", "es", "fr", "de", "ja", "zh"])
    if st.button("Clear chat"):
        st.session_state.messages = [st.session_state.messages[0]]
        st.session_state.chat_display = []
        st.session_state.last_audio = None
        st.session_state.processed_audio_id = None
        st.rerun()

# ── Title ─────────────────────────────────────────────────────────────────────
st.title("🎙️ Voice Chat")

# ── Chat history ──────────────────────────────────────────────────────────────
for role, text in st.session_state.chat_display:
    with st.chat_message(role):
        st.write(text)

# ── TTS playback ──────────────────────────────────────────────────────────────
if st.session_state.last_audio:
    b64 = base64.b64encode(st.session_state.last_audio).decode()
    st.markdown(
        f'<audio autoplay controls style="width:100%;margin-top:8px">'
        f'<source src="data:audio/mpeg;base64,{b64}" type="audio/mpeg">'
        f'</audio>',
        unsafe_allow_html=True,
    )

# ── Input row ─────────────────────────────────────────────────────────────────
audio_value = st.audio_input(" ", label_visibility="collapsed")
text_query  = st.chat_input("Type a message...")


# ── Pipeline ──────────────────────────────────────────────────────────────────
def run_pipeline(user_text: str):
    """Append user turn, get reply, generate TTS, rerun."""
    st.session_state.chat_display.append(("user", user_text))
    st.session_state.messages.append({"role": "user", "content": user_text})
    st.session_state.last_audio = None

    with st.spinner(""):
        reply = chat_response(st.session_state.messages)
        mp3   = text_to_speech(reply, voice=tts_voice)

    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.session_state.chat_display.append(("assistant", reply))
    st.session_state.last_audio = mp3
    st.rerun()


# ── Handle mic ────────────────────────────────────────────────────────────────
if audio_value is not None:
    audio_bytes = audio_value.read()
    audio_hash = hashlib.md5(audio_bytes).hexdigest()
    if audio_hash != st.session_state.processed_audio_id:
        st.session_state.processed_audio_id = audio_hash
        with st.spinner(""):
            try:
                user_text = transcribe_audio(audio_bytes, language=language)
            except Exception as e:
                st.error(str(e))
                st.stop()
        if user_text:
            run_pipeline(user_text)
        else:
            st.toast("Couldn't catch that — try again.")

# ── Handle text ───────────────────────────────────────────────────────────────
if text_query:
    try:
        run_pipeline(text_query)
    except Exception as e:
        st.error(str(e))
