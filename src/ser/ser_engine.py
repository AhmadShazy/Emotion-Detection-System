"""
src/ser/ser_engine.py
======================
Speech Emotion Recognition using SpeechBrain Wav2Vec2 IEMOCAP.
Model is owned by ModelRegistry — SEREngine is now a thin wrapper
that gets the classifier from the registry instead of loading it.
"""

import torch
import torchaudio
import soundfile as sf
import numpy as np
import sys
import os

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _apply_patches():
    """
    Compatibility patches for SpeechBrain 1.0.3 + newer libs.
    Order matters — torchaudio must be patched before speechbrain import.
    Still called here so ser_engine works correctly when used standalone
    via CLI (main.py). Registry calls its own copy before loading the model.
    """
    # ── Patch 1: torchaudio — MUST come first ─────────────────────────────────
    if not hasattr(torchaudio, "list_audio_backends"):
        torchaudio.list_audio_backends = lambda: ["soundfile"]

    if not hasattr(torchaudio, "get_audio_backend"):
        torchaudio.get_audio_backend = lambda: "soundfile"

    if not getattr(torchaudio, "_patched_by_ser_engine", False):
        torchaudio.load = _custom_load
        torchaudio._patched_by_ser_engine = True

    # ── Patch 2: transformers removed AutoModelWithLMHead in v5 ──────────────
    import transformers
    if not hasattr(transformers, "AutoModelWithLMHead"):
        transformers.AutoModelWithLMHead = getattr(
            transformers, "AutoModelForCausalLM", transformers.AutoModel
        )

    # ── Patch 3: huggingface_hub dropped use_auth_token param ─────────────────
    import huggingface_hub
    if not getattr(huggingface_hub, "_patched_by_ser_engine", False):
        _original = huggingface_hub.hf_hub_download

        def _patched(*args, **kwargs):
            if "use_auth_token" in kwargs:
                kwargs["token"] = kwargs.pop("use_auth_token")
            return _original(*args, **kwargs)

        huggingface_hub.hf_hub_download        = _patched
        huggingface_hub._patched_by_ser_engine = True

    # ── Cross-platform inspect patch for SpeechBrain LazyModule ──
    try:
        from speechbrain.utils.importutils import LazyModule
        import importlib
        import warnings
        from types import ModuleType

        def _patched_ensure_module(self, stacklevel: int) -> ModuleType:
            import sys
            import os
            import inspect
            importer_frame = None
            try:
                importer_frame = inspect.getframeinfo(sys._getframe(stacklevel + 1))
            except AttributeError:
                warnings.warn(
                    "Failed to inspect frame to check if we should ignore "
                    "importing a module lazily."
                )

            if importer_frame is not None and (
                importer_frame.filename.endswith("/inspect.py") or
                importer_frame.filename.endswith("\\inspect.py") or
                os.path.basename(importer_frame.filename) == "inspect.py"
            ):
                raise AttributeError()

            if self.lazy_module is None:
                try:
                    if self.package is None:
                        self.lazy_module = importlib.import_module(self.target)
                    else:
                        self.lazy_module = importlib.import_module(
                            f".{self.target}", self.package
                        )
                except Exception as e:
                    raise ImportError(f"Lazy import of {repr(self)} failed") from e

            return self.lazy_module

        LazyModule.ensure_module = _patched_ensure_module
    except Exception as pe:
        print(f"[SEREngine] Warning: LazyModule patch failed: {pe}")


def _custom_load(filepath, **kwargs):
    """
    Soundfile-based drop-in replacement for torchaudio.load.
    Returns: (Tensor[channels, time], int sample_rate)
    """
    try:
        data, samplerate = sf.read(filepath)
        data = data.astype(np.float32)
        if data.ndim == 1:
            tensor = torch.from_numpy(data).unsqueeze(0)
        else:
            tensor = torch.from_numpy(data.transpose())
        return tensor, samplerate
    except Exception as e:
        print(f"[SEREngine] CRITICAL: _custom_load failed for {filepath}: {e}")
        raise


class SEREngine:
    """
    Thin wrapper around the SpeechBrain classifier.
    Gets the model from ModelRegistry — does NOT load it.
    Loading is done once at startup by registry.load_all().
    """

    def __init__(self):
        # Prefer the registry's canonical patch set (already applied at startup).
        # Fall back to the local _apply_patches() only when used standalone via CLI
        # (i.e., when registry is not initialised yet).
        try:
            from src.core.model_registry import registry
            registry._apply_speechbrain_patches()
            self.classifier = registry.get("speechbrain")
        except Exception:
            # CLI / test context — apply patches locally and re-raise if still broken
            _apply_patches()
            from src.core.model_registry import registry
            self.classifier = registry.get("speechbrain")

        print("[SEREngine] Classifier obtained from registry.")

    def predict_emotion(self, audio_file: str) -> tuple:
        """
        Predicts emotion from a WAV file.

        Args:
            audio_file: Path to a 16kHz mono WAV file.

        Returns:
            tuple: (label: str, confidence: float)
                label      — one of "Happy", "Angry", "Neutral", "Sad"
                confidence — real softmax probability from SpeechBrain [0.0, 1.0]
        """
        signal, _ = _custom_load(audio_file)

        # classify_batch expects shape (Batch, Time)
        # Returns: out_prob (log-probs), score (top-class prob), index, text_lab
        out_prob, score, index, text_lab = self.classifier.classify_batch(signal)

        label_map = {
            "hap": "Happy",
            "ang": "Angry",
            "neu": "Neutral",
            "sad": "Sad",
        }
        label      = label_map.get(text_lab[0], text_lab[0])
        confidence = float(score[0])   # real softmax probability, not hardcoded

        print(f"[SEREngine] Prediction: {label} (conf={confidence:.3f})")
        return label, confidence


# ── CLI test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if os.path.exists("input.wav"):
        engine           = SEREngine()
        emotion, conf    = engine.predict_emotion("input.wav")
        print(f"Predicted Emotion: {emotion} (confidence: {conf:.3f})")
    else:
        print("No input.wav found. Place a WAV file named input.wav and re-run.")