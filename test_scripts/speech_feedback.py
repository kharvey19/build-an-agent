import os
import tempfile

import sounddevice as sd
import streamlit as st
import whisper
from dotenv import load_dotenv
from openai import OpenAI
from scipy.io.wavfile import write
from utils.feedback import call_llm

# import test_scripts.speech as speech  # Install this if needed: pip install -U openai-whisper

# Load environment
load_dotenv()
API_KEY = os.getenv("NVIDIA_API_KEY")
MODEL_URL = "https://integrate.api.nvidia.com/v1"
MODEL_NAME = "meta/llama-3.3-70b-instruct"
client = OpenAI(base_url=MODEL_URL, api_key=API_KEY)

# Load Whisper ASR model
asr_model = speech.load_model("base")

# Streamlit layout
st.set_page_config(page_title="🎤 Speech Feedback", layout="centered")
st.title("🗣️ Speech Feedback via Transcribed Audio")

# Survey context
st.subheader("🧠 Speech Context")
goal = st.text_input("What is the goal of the speech?")
audience = st.text_input("Who’s your audience?")

# Recording duration
fs = 16000
seconds = st.slider("Recording Duration (seconds)", 5, 30, 10)

if st.button("🎙️ Record and Analyze"):
    st.info("Recording will start in 2 seconds...")
    sd.sleep(2000)
    st.info("Recording... Speak now!")

    recording = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype="int16")
    sd.wait()

    # Save audio to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        write(tmpfile.name, fs, recording)
        audio_path = tmpfile.name
        st.success("✅ Recording complete.")
        st.audio(audio_path, format="audio/wav")

    # Transcribe audio with Whisper
    st.info("🧠 Transcribing audio...")
    result = asr_model.transcribe(audio_path)
    transcription = result["text"]
    st.success("✅ Transcription complete.")
    st.markdown("### 📝 Transcription:")
    st.markdown(f"> {transcription}")

    # Build prompt for LLM
    memory = [
        {
            "role": "user",
            "content": f"""Here is context for the speech: goal={goal}, audience={audience}.

Here is the transcription of the speech:
\"\"\"{transcription}\"\"\"

Give feedback on:
- Filler words
- Projection
- Speed
- Understandability
Please be specific and constructive.""",
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
                            "description": "The speech text to analyze.",
                        }
                    },
                    "required": ["speech"],
                },
            },
        }
    ]

    # Call the model
    st.subheader("📋 Feedback from Agent")
    try:
        llm_response = call_llm(client, MODEL_NAME, memory, tools)
        st.write(llm_response.get("content", "No feedback returned."))
    except Exception as e:
        st.error(f"⚠️ Model call failed: {e}")
