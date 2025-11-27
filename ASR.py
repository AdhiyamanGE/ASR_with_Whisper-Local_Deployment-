import os
from datetime import datetime
import streamlit as st
import whisper
from streamlit_mic_recorder import mic_recorder

FFMPEG_BIN = r"C:\Users\adhiy\Downloads\ffmpeg-master-latest-win64-gpl\bin"
os.environ["PATH"] = FFMPEG_BIN + os.pathsep + os.environ.get("PATH", "")

st.title("Streamlit Mic + Whisper")

RECORD_DIR = os.path.join(os.path.dirname(__file__), "recordings")
os.makedirs(RECORD_DIR, exist_ok=True)


@st.cache_resource
def load_model():
    return whisper.load_model("base")


audio = mic_recorder(
    start_prompt="Start recording",
    stop_prompt="Stop recording",
    just_once=False,
    use_container_width=True,
    key="mic",
)

if audio is not None:
    st.audio(audio["bytes"], format="audio/wav")

if st.button("Submit") and audio is not None:
    with st.spinner("Transcribing..."):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"input_{ts}.wav"
        save_path = os.path.join(RECORD_DIR, filename)

        with open(save_path, "wb") as f:
            f.write(audio["bytes"])

        model = load_model()
        result = model.transcribe(
            save_path,
            task="translate",
            language=None,
        )
        text = result["text"]

    st.markdown("### Transcription")
    st.text_area("Edit transcription", text, height=150)
