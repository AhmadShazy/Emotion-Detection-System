import torch
import torchaudio
import soundfile as sf
import numpy as np
import sys
import os

# ── Compatibility patches ─────────────────────────────────────────────────────
# These fix two version mismatches between SpeechBrain 1.0.3 and newer libs.
# They are applied ONCE inside _apply_patches() which is called only from
# SEREngine.__init__() — NOT at module import time.
# This prevents side effects on any other module that imports torchaudio.

def _apply_patches():
    """
    Applies all compatibility patches needed for SpeechBrain + newer libs.
    Safe to call multiple times (idempotent).
    """

    # ── Patch 1: transformers.AutoModelWithLMHead removed in v5 ──────────────
    # SpeechBrain 1.0.3 tries to import it. We alias it to AutoModelForCausalLM.
    import transformers
    if not hasattr(transformers, "AutoModelWithLMHead"):
        if hasattr(transformers, "AutoModelForCausalLM"):
            transformers.AutoModelWithLMHead = transformers.AutoModelForCausalLM
        else:
            transformers.AutoModelWithLMHead = transformers.AutoModel

    # ── Patch 2: huggingface_hub.hf_hub_download dropped 'use_auth_token' ────
    # SpeechBrain sends use_auth_token= but newer huggingface_hub expects token=
    import huggingface_hub
    if not getattr(huggingface_hub, "_patched_by_ser_engine", False):
        _original = huggingface_hub.hf_hub_download

        def _patched(*args, **kwargs):
            if "use_auth_token" in kwargs:
                kwargs["token"] = kwargs.pop("use_auth_token")
            return _original(*args, **kwargs)

        huggingface_hub.hf_hub_download = _patched
        huggingface_hub._patched_by_ser_engine = True  # idempotency guard

    # ── Patch 3: replace torchaudio.load with soundfile-based implementation ──
    # torchaudio v2.9.1 has broken bindings for torchcodec that crash even when
    # backend='soundfile' is requested. We bypass it entirely.
    if not getattr(torchaudio, "_patched_by_ser_engine", False):
        torchaudio.load = _custom_load
        torchaudio._patched_by_ser_engine = True  # idempotency guard

        # Also add list_audio_backends if missing (some torchaudio builds lack it)
        if not hasattr(torchaudio, "list_audio_backends"):
            torchaudio.list_audio_backends = lambda: ["soundfile"]


# ── Custom audio loader (soundfile-based) ─────────────────────────────────────

def _custom_load(filepath, **kwargs):
    """
    Drop-in replacement for torchaudio.load using soundfile.
    Returns: (Tensor[channels, time], int sample_rate)
    """
    try:
        data, samplerate = sf.read(filepath)
        data = data.astype(np.float32)

        if data.ndim == 1:
            # Mono: (time,) → (1, time)
            tensor = torch.from_numpy(data).unsqueeze(0)
        else:
            # Multi-channel: (time, channels) → (channels, time)
            tensor = torch.from_numpy(data.transpose())

        return tensor, samplerate

    except Exception as e:
        print(f"CRITICAL: _custom_load failed for {filepath}: {e}")
        raise


# ── SER Engine ────────────────────────────────────────────────────────────────

class SEREngine:
    def __init__(self):
        # Apply patches here — not at module level — so importing this file
        # has zero side effects on other modules.
        _apply_patches()

        print("Loading SpeechBrain SER Model (CPU Optimized)...", flush=True)

        import warnings
        warnings.filterwarnings("ignore", message=".*speechbrain.pretrained.*")

        from speechbrain.inference.interfaces import foreign_class

        self.classifier = foreign_class(
            source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
            pymodule_file="custom_interface.py",
            classname="CustomEncoderWav2vec2Classifier",
            run_opts={"device": "cpu"},
        )
        print("SER Model loaded successfully!", flush=True)

    def predict_emotion(self, audio_file):
        """
        Predicts emotion from a WAV file.
        Returns one of: 'Happy', 'Angry', 'Neutral', 'Sad'
        """
        signal, _ = _custom_load(audio_file)

        # classify_batch expects shape (Batch, Time)
        out_prob, score, index, text_lab = self.classifier.classify_batch(signal)

        label_map = {
            "hap": "Happy",
            "ang": "Angry",
            "neu": "Neutral",
            "sad": "Sad",
        }
        return label_map.get(text_lab[0], text_lab[0])


# ── CLI test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if os.path.exists("input.wav"):
        engine  = SEREngine()
        emotion = engine.predict_emotion("input.wav")
        print(f"Predicted Emotion: {emotion}")
    else:
        print("No input.wav found. Place a WAV file named input.wav and re-run.")