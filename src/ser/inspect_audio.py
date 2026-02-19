import torchaudio
import soundfile
import sys

print(f"Python Version: {sys.version}")
print(f"Torchaudio Version: {torchaudio.__version__}")
try:
    print(f"Soundfile Version: {soundfile.__version__}")
except:
    print("Soundfile version check failed")

print("\nChecking backends...")
try:
    print(f"List audio backends: {torchaudio.list_audio_backends()}")
except Exception as e:
    print(f"list_audio_backends failed: {e}")

try:
    print(f"Get audio backend: {torchaudio.get_audio_backend()}")
except Exception as e:
    print(f"get_audio_backend failed: {e}")

print("\nChecking available attributes in torchaudio:")
attrs = [a for a in dir(torchaudio) if 'backend' in a or 'load' in a]
print(attrs)

# Try loading a dummy file
import numpy as np
import soundfile as sf
sf.write('test.wav', np.zeros(16000), 16000)

print("\nAttempting torchaudio.load('test.wav')...")
try:
    y, sr = torchaudio.load('test.wav')
    print("Success! loaded with default backend.")
except Exception as e:
    print(f"Failed with default: {e}")

print("\nAttempting torchaudio.load('test.wav', backend='soundfile')...")
try:
    y, sr = torchaudio.load('test.wav', backend='soundfile')
    print("Success! loaded with backend='soundfile'.")
except Exception as e:
    print(f"Failed with backend='soundfile': {e}")
