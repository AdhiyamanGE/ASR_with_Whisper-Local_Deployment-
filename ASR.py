import os
import streamlit as st
import whisper

# Force ffmpeg path if needed
FFMPEG_BIN = r"C:\Users\adhiy\Downloads\ffmpeg-master-latest-win64-gpl\bin"
os.environ["PATH"] = FFMPEG_BIN + os.pathsep + os.environ.get("PATH", "")

st.title("Streamlit Voice Recorder + Whisper")

# Ensure recordings folder exists (relative to this script)
RECORD_DIR = os.path.join(os.path.dirname(__file__), "recordings")
os.makedirs(RECORD_DIR, exist_ok=True)

audio_file = st.audio_input("Record your message")


@st.cache_resource
def load_model():
    return whisper.load_model("base")


if audio_file is not None:
    st.audio(audio_file)

    # Choose a filename; you can use timestamp or counter
    filename = "input.wav"
    save_path = os.path.join(RECORD_DIR, filename)

    # Save UploadedFile bytes to your project folder
    with open(save_path, "wb") as f:
        f.write(audio_file.read())

    # Transcribe from this fixed path
    model = load_model()
    result = model.transcribe(
        save_path,
        task="translate",  # translate any language -> English
        language=None,  # None = auto-detect language
    )
    text = result["text"]  # this will be English

    st.markdown("### Transcription")
    st.write(text)
