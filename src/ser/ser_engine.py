import torch
import torchaudio
import soundfile as sf
import numpy as np
import sys

# --- ROBUST MONKEYPATCH START ---
# The user's torchaudio (v2.9.1) seems to have broken bindings for 'torchcodec' 
# that crash even when backend='soundfile' is used.
# We will COMPLETELY BYPASS torchaudio.load and implement it using soundfile directly.

import huggingface_hub
import transformers

# Patch transformers.AutoModelWithLMHead (removed in v5)
# SpeechBrain 1.0.3 tries to import it.
try:
    from transformers import AutoModelWithLMHead
except ImportError:
    # Alias to AutoModelForCausalLM as per deprecation warnings in older versions
    if hasattr(transformers, "AutoModelForCausalLM"):
        transformers.AutoModelWithLMHead = transformers.AutoModelForCausalLM
    else:
        transformers.AutoModelWithLMHead = transformers.AutoModel

# Patch huggingface_hub.hf_hub_download to accept 'use_auth_token' by renaming it to 'token'
# This is needed because speechbrain sends 'use_auth_token' but newer huggingface_hub removed it.
_original_hf_hub_download = huggingface_hub.hf_hub_download

def _patched_hf_hub_download(*args, **kwargs):
    if 'use_auth_token' in kwargs:
        kwargs['token'] = kwargs.pop('use_auth_token')
    return _original_hf_hub_download(*args, **kwargs)

huggingface_hub.hf_hub_download = _patched_hf_hub_download


def _custom_load(filepath, **kwargs):
    """
    Replacements for torchaudio.load using soundfile.
    Returns: (Tensor[channels, time], int sample_rate)
    """
    try:
        # soundfile.read returns (data, samplerate)
        # data is (time, channels) or (time,) for mono
        data, samplerate = sf.read(filepath)
        
        # Convert to float32 (standard for torch audio)
        data = data.astype(np.float32)
        
        # specific to how soundfile handles mono (1D array)
        if data.ndim == 1:
            # Add channel dimension: (time,) -> (1, time)
            tensor = torch.from_numpy(data).unsqueeze(0)
        else:
            # Transpose: (time, channels) -> (channels, time)
            tensor = torch.from_numpy(data.transpose())
            
        return tensor, samplerate
    except Exception as e:
        print(f"CRITICAL: Implementation of custom load failed for {filepath}: {e}")
        raise e

# Overwrite the function in the torchaudio module
torchaudio.load = _custom_load

# Also patch list_audio_backends just in case
if not hasattr(torchaudio, "list_audio_backends"):
    def _list_audio_backends():
        return ["soundfile"]
    torchaudio.list_audio_backends = _list_audio_backends
# --- ROBUST MONKEYPATCH END ---

from speechbrain.inference.interfaces import foreign_class

class SEREngine:
    def __init__(self):
        print("Loading SpeechBrain SER Model (CPU Optimized)...", flush=True)
        # Suppress the specific warning about pretrained/inference redirection
        import warnings
        warnings.filterwarnings("ignore", message=".*speechbrain.pretrained.*")
        
        # Load the pre-trained model from HuggingFace Hub
        # run_opts={"device": "cpu"} allows us to force CPU usage
        self.classifier = foreign_class(
            source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
            pymodule_file="custom_interface.py",
            classname="CustomEncoderWav2vec2Classifier",
            run_opts={"device": "cpu"} 
        )
        print("Model loaded successfully!", flush=True)

    def predict_emotion(self, audio_file):
        """
        Predicts emotion from a wav file.
        """
        # Manual load to ensure we control the shape
        signal, fs = _custom_load(audio_file)
        
        # classify_batch expects (Batch, Time)
        out_prob, score, index, text_lab = self.classifier.classify_batch(signal)
        
        # Map labels to human readable
        label_map = {
            'hap': 'Happy',
            'ang': 'Angry',
            'neu': 'Neutral',
            'sad': 'Sad'
        }
        
        raw_label = text_lab[0]
        return label_map.get(raw_label, raw_label)

if __name__ == "__main__":
    import os
    if os.path.exists("input.wav"):
        engine = SEREngine()
        emotion = engine.predict_emotion("input.wav")
        print(f"Predicted Entity: {emotion}")
