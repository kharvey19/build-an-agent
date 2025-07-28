import os
import tempfile

import numpy as np
import sounddevice as sd
import streamlit as st
import whisper
from dotenv import load_dotenv
from openai import OpenAI
from scipy.io.wavfile import write
from utils.feedback import call_llm

# Load environment variables
load_dotenv()
API_KEY = os.getenv("NVIDIA_API_KEY")
MODEL_URL = "https://integrate.api.nvidia.com/v1"
MODEL_NAME = "meta/llama-3.3-70b-instruct"
client = OpenAI(base_url=MODEL_URL, api_key=API_KEY)

# Load Whisper model once
asr_model = whisper.load_model("base")

# Streamlit config
st.set_page_config(page_title="🎤 Speech Feedback App", layout="centered")
st.title("🎙️ Speech Feedback via Audio")

# Session state
if "recording" not in st.session_state:
    st.session_state.recording = False
if "audio_file" not in st.session_state:
    st.session_state.audio_file = None

# Global audio buffer
audio_buffer = []

# Survey
st.subheader("🧠 Speech Context")
goal = st.text_input("What is the goal of the speech?")
audience = st.text_input("Who’s your audience?")
fs = 16000  # 16 kHz sample rate


def audio_callback(indata, frames, time, status):
    if status:
        print("Recording error:", status)
    audio_buffer.append(indata.copy())


# Start/Stop buttons
col1, col2 = st.columns(2)
with col1:
    if st.button("▶️ Start Recording") and not st.session_state.recording:
        audio_buffer.clear()
        st.session_state.recording = True
        st.info("Recording started... press stop when you're done.")
        try:
            with sd.InputStream(callback=audio_callback, channels=1, samplerate=fs):
                sd.sleep(5000)  # record for 5 seconds
        except Exception as e:
            st.error(f"Audio stream error: {e}")
        st.session_state.recording = False

        # Save audio file
        if audio_buffer:
            audio_data = np.concatenate(audio_buffer, axis=0)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
                write(tmpfile.name, fs, audio_data)
                st.session_state.audio_file = tmpfile.name
                st.success("✅ Recording saved.")
                st.audio(tmpfile.name)

with col2:
    st.button("⏹️ Stop Recording", disabled=True)  # Placeholder for symmetry

# Process and transcribe
if st.session_state.audio_file:
    st.subheader("🧠 Feedback from Agent")
    try:
        st.info("Transcribing your speech...")
        result = asr_model.transcribe(st.session_state.audio_file)
        transcription = result["text"]
        st.success("✅ Transcription complete!")
        st.markdown(f"### 📝 Transcription:\n> {transcription}")

        memory = [
            {
                "role": "user",
                "content": f"""Speech context:
Goal: {goal}
Audience: {audience}

Transcript:
\"\"\"{transcription}\"\"\"

Please give feedback on:
- Filler words
- Projection
- Speed
- Understandability
""",
            }
        ]

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "detect_filler_words",
                    "description": "Detects filler words in a speech.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "speech": {
                                "type": "string",
                                "description": "The speech text or transcript.",
                            }
                        },
                        "required": ["speech"],
                    },
                },
            }
        ]

        llm_response = call_llm(client, MODEL_NAME, memory, tools)
        st.write(llm_response.get("content", "No feedback returned."))

    except Exception as e:
        st.error(f"Model call failed: {e}")
