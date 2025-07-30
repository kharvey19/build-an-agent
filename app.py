import json
import os
import re
import tempfile
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import streamlit as st
import whisper
from dotenv import load_dotenv
from openai import OpenAI
from scipy.io.wavfile import write

from fix_speech import call_llm
from wrapper import transcribe_with_nvidia_asr

# Load environment variables
load_dotenv()
API_KEY = os.getenv("NVIDIA_API_KEY")
MODEL_URL = "https://integrate.api.nvidia.com/v1"
MODEL_NAME = "meta/llama-3.3-70b-instruct"
client = OpenAI(base_url=MODEL_URL, api_key=API_KEY)

# Create audio directory if it doesn't exist
AUDIO_DIR = Path("temp_audio")
AUDIO_DIR.mkdir(exist_ok=True)

# Load Whisper model once
# asr_model = whisper.load_model("base")

# Streamlit config
st.set_page_config(page_title="🎤 Speech Feedback App", layout="centered")
st.title("🎙️ Speech Feedback via Audio")

# Session state
if "recording" not in st.session_state:
    st.session_state.recording = False
if "audio_file" not in st.session_state:
    st.session_state.audio_file = None
if "stop_early" not in st.session_state:
    st.session_state.stop_early = False

# Global audio buffer
audio_buffer = []

# Survey
st.subheader("🧠 Speech Context")
goal = st.text_input("What is the goal of the speech?")
audience = st.text_input("Who's your audience?")
tone = st.text_input("What tone do you want (funny, serious, inspiring)?")
unsure_parts = st.text_input("Are there parts you're unsure about?")
feedback_on = st.text_input(
    "What do you want feedback on? (Grammar, Clarity, Tone, Projection, Emotional impact, Delivery, Understandability)"
)
recording_duration = st.number_input(
    "How long do you want to record for? (seconds)",
    min_value=1,
    max_value=60,
    value=5,
    step=1,
)

fs = 16000  # 16 kHz sample rate


def audio_callback(indata, frames, time, status):
    if status:
        print("Recording error:", status)
    audio_buffer.append(indata.copy())


def cleanup_old_audio_files():
    """Clean up audio files older than 1 hour"""
    try:
        current_time = time.time()
        for audio_file in AUDIO_DIR.glob("*.wav"):
            if current_time - audio_file.stat().st_mtime > 3600:  # 1 hour
                audio_file.unlink(missing_ok=True)
    except Exception as e:
        print(f"Cleanup error: {e}")


def normalize_audio(audio_data, target_level=0.8):
    """Normalize audio to target level to ensure proper volume"""
    if len(audio_data) == 0:
        return audio_data

    # Convert to float for processing
    if audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32)

    # Find the maximum absolute value
    max_val = np.max(np.abs(audio_data))

    # Avoid division by zero
    if max_val > 0:
        # Normalize to target level
        audio_data = audio_data * (target_level / max_val)

    # Convert to 16-bit integer for WAV file
    audio_data = (audio_data * 32767).astype(np.int16)

    return audio_data


# Clean up old files
cleanup_old_audio_files()

# Start/Stop buttons
if st.button("▶️ Start Recording") and not st.session_state.recording:
    audio_buffer.clear()
    st.session_state.recording = True
    st.session_state.stop_early = False
    st.info(f"🎤 Recording started... Speak clearly for {recording_duration} seconds!")

    # Use a progress bar to show recording progress
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        # Calculate iterations based on duration (10 iterations per second for 0.1s intervals)
        total_iterations = int(recording_duration * 10)

        # Record for specified duration with visual feedback
        with sd.InputStream(
            callback=audio_callback, channels=1, samplerate=fs, dtype=np.float32
        ):
            for i in range(total_iterations):
                if st.session_state.stop_early:
                    break
                sd.sleep(100)  # 0.1 second
                progress_bar.progress((i + 1) / total_iterations)
                remaining_time = recording_duration - (i + 1) * 0.1
                status_text.text(f"Recording... {remaining_time:.1f}s remaining")

    except Exception as e:
        st.error(f"Audio stream error: {e}")

    st.session_state.recording = False
    progress_bar.empty()
    status_text.empty()

    # Save audio file with better processing
    if audio_buffer:
        try:
            # Concatenate all audio chunks
            audio_data = np.concatenate(audio_buffer, axis=0)

            # Check if we actually recorded something
            if len(audio_data) == 0:
                st.error("No audio data recorded. Please check your microphone.")
            else:
                # Normalize the audio for better recognition
                audio_data = normalize_audio(audio_data)

                # Create filename with timestamp
                timestamp = int(time.time())
                audio_filename = f"recording_{timestamp}.wav"
                audio_path = AUDIO_DIR / audio_filename

                # Write the audio file
                write(str(audio_path), fs, audio_data)
                st.session_state.audio_file = str(audio_path)

                # Check file size to ensure it's not empty
                file_size = audio_path.stat().st_size
                duration = len(audio_data) / fs

                st.success(
                    f"✅ Recording saved! Duration: {duration:.1f}s, Size: {file_size/1024:.1f}KB"
                )
                st.audio(str(audio_path))

        except Exception as e:
            st.error(f"Error processing audio: {e}")
    else:
        st.error("No audio data captured. Please check your microphone permissions.")

# Process and transcribe
if st.session_state.audio_file and Path(st.session_state.audio_file).exists():
    st.subheader("🧠 Feedback from Agent")
    try:
        st.info("Transcribing your speech...")
        raw_output = transcribe_with_nvidia_asr(st.session_state.audio_file, API_KEY)

        # Extract transcript from the output
        transcript = ""
        if "Final transcript:" in raw_output:
            # Extract from "Final transcript:" line
            lines = raw_output.split("\n")
            for line in lines:
                if line.startswith("Final transcript:"):
                    transcript = line.replace("Final transcript:", "").strip()
                    break
        else:
            # Try to parse JSON if no "Final transcript:" line
            try:
                json_match = re.search(r"\{.*\}", raw_output, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    data = json.loads(json_str)
                    if "results" in data and len(data["results"]) > 0:
                        alternatives = data["results"][0].get("alternatives", [])
                        if alternatives and len(alternatives) > 0:
                            transcript = alternatives[0].get("transcript", "").strip()
            except:
                transcript = raw_output.strip()

        if not transcript:
            transcript = "No transcript found"

        st.success("✅ Transcription complete!")
        st.markdown(f"### 📝 Transcription:\n> {transcript}")

        memory = [
            {
                "role": "user",
                "content": f"""Speech context:
Goal: {goal}
Audience: {audience}
Tone: {tone}
Unsure parts: {unsure_parts}
Feedback on: {feedback_on}

Transcript:
\"\"\"{transcript}\"\"\"

Please give feedback on:
- Filler words
- Projection
- Speed
- Understandability
""",
            }
        ]

        # Remove tools to get direct feedback
        tools = []

        llm_response = call_llm(client, MODEL_NAME, memory, tools)

        feedback_content = llm_response.get("content", "No feedback returned.")
        if feedback_content and feedback_content != "No feedback returned.":
            st.markdown("### 🎯 AI Feedback:")
            st.markdown(feedback_content)
        else:
            st.error("❌ No feedback received. Please try again.")

    except Exception as e:
        st.error(f"Model call failed: {e}")
elif st.session_state.audio_file:
    st.warning("Audio file not found. Please record again.")
    st.session_state.audio_file = None
