import sounddevice as sd
from scipy.io.wavfile import write

# Settings
fs = 16000  # Sample rate (Hz)
seconds = 10  # Duration of recording

print("Recording will start in 2 seconds. Get ready!")
sd.sleep(2000)
print("Recording... Speak now!")
recording = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype="int16")
sd.wait()  # Wait until recording is finished
print("Recording finished.")

# Save as WAV file
write("audio.wav", fs, recording)
print("Saved as audio.wav")
