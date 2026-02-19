import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np

def record_audio(duration=20, fs=16000, filename="input.wav"):
    """
    Records audio from the microphone.
    
    Args:
        duration (int): Duration of recording in seconds.
        fs (int): Sample rate (16000 Hz is recommended for SpeechBrain).
        filename (str): Name of the file to save.
    """
    print(f"Recording for {duration} seconds...")
    
    # Record audio
    # channels=1 for mono (SER models usually expect mono)
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    
    # Wait until recording is finished
    sd.wait()
    
    # Save as WAV file
    write(filename, fs, recording)
    print(f"Recording saved to {filename}")

if __name__ == "__main__":
    # Test the recorder
    record_audio()
