"""
src/core/model_registry.py
===========================
Central singleton registry that owns all ML model instances.

Rules:
    - Models are loaded ONCE at startup via registry.load_all()
    - Every module that needs a model calls registry.get("model_name")
    - No module ever instantiates WhisperModel, SEREngine, or pipeline directly
    - Thread safe — _lock prevents double-loading
    - FastAPI lifespan guarantees load_all() completes before first request

Usage:
    # Startup (api.py lifespan):
    from src.core.model_registry import registry
    registry.load_all()

    # Anywhere in codebase:
    from src.core.model_registry import registry
    model = registry.get("whisper")
    model = registry.get("roberta")
    model = registry.get("speechbrain")
    model = registry.get("faster_whisper")
"""

import os
import sys
import threading
import time

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

# Cache paths — must match scripts/download_models.py exactly
WHISPER_CACHE        = os.path.join(PROJECT_ROOT, "external", "whisper")
FASTER_WHISPER_CACHE = os.path.join(PROJECT_ROOT, "external", "faster_whisper")
SPEECHBRAIN_CACHE    = os.path.join(PROJECT_ROOT, "external", "speechbrain")


class ModelRegistry:
    """
    Singleton that holds all ML model instances.

    Internal store:
        _models = {
            "roberta":        HuggingFace pipeline object,
            "whisper":        openai-whisper model object,
            "faster_whisper": faster_whisper WhisperModel object,
            "speechbrain":    SpeechBrain classifier object,
        }
    """

    def __init__(self):
        self._models:     dict = {}
        self._lock              = threading.Lock()
        self._loaded            = False
        self._load_times: dict  = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def load_all(self):
        """
        Loads every model in sequence.
        Idempotent — safe to call multiple times, only executes once.
        Must be called from api.py lifespan BEFORE yield.
        """
        with self._lock:
            if self._loaded:
                print("[Registry] Already loaded — skipping.")
                return

            print("\n[Registry] ══════════════════════════════════════")
            print("[Registry] Loading all models into memory...")
            print("[Registry] ══════════════════════════════════════")

            # Lightest first so startup feels progressive
            loaders = [
                ("roberta",        self._load_roberta),
                ("faster_whisper", self._load_faster_whisper),
                ("whisper",        self._load_whisper),
                ("speechbrain",    self._load_speechbrain),
            ]

            failed = []
            for name, fn in loaders:
                try:
                    start = time.time()
                    fn()
                    elapsed = time.time() - start
                    self._load_times[name] = elapsed
                    print(f"[Registry] ✅ {name:<18} loaded in {elapsed:.1f}s")
                except Exception as e:
                    print(f"[Registry] ❌ {name:<18} FAILED: {e}")
                    failed.append(name)

            self._loaded = True

            print("[Registry] ══════════════════════════════════════")
            if failed:
                print(f"[Registry] ⚠️  Failed: {failed}")
                print("[Registry] Affected endpoints will return 503.")
            else:
                print("[Registry] 🎉 All models ready.")
            print("[Registry] ══════════════════════════════════════\n")

    def get(self, name: str):
        """
        Returns model instance by name.
        Raises RuntimeError if model is unavailable.

        Valid names: "roberta", "whisper", "faster_whisper", "speechbrain"
        """
        if name not in self._models:
            raise RuntimeError(
                f"[Registry] Model '{name}' is not available. "
                f"Either load_all() was not called, it failed to load, "
                f"or download_models.py was never run. "
                f"Available: {list(self._models.keys())}"
            )
        return self._models[name]

    def is_available(self, name: str) -> bool:
        """Non-raising check — use this for graceful degradation."""
        return name in self._models

    def status(self) -> dict:
        """Returns health dict for the /health endpoint."""
        return {
            "loaded": self._loaded,
            "models": {
                name: {
                    "available":         name in self._models,
                    "load_time_seconds": round(self._load_times.get(name, 0), 2),
                }
                for name in ["roberta", "whisper", "faster_whisper", "speechbrain"]
            }
        }

    # ── Private loaders ───────────────────────────────────────────────────────

    def _load_roberta(self):
        from transformers import pipeline
        self._models["roberta"] = pipeline(
            "text-classification",
            model="SamLowe/roberta-base-go_emotions",
            top_k=None,
        )

    def _load_whisper(self):
        import whisper
        self._models["whisper"] = whisper.load_model(
            "base",
            download_root=WHISPER_CACHE,
        )

    def _load_faster_whisper(self):
        from faster_whisper import WhisperModel
        self._models["faster_whisper"] = WhisperModel(
            "tiny",
            device="cpu",
            compute_type="int8",
            download_root=FASTER_WHISPER_CACHE,
        )

    def _load_speechbrain(self):
        # Patches must run before speechbrain import
        self._apply_speechbrain_patches()
        from speechbrain.inference.interfaces import foreign_class
        self._models["speechbrain"] = foreign_class(
            source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
            pymodule_file="custom_interface.py",
            classname="CustomEncoderWav2vec2Classifier",
            savedir=SPEECHBRAIN_CACHE,
            run_opts={"device": "cpu"},
        )

    def _apply_speechbrain_patches(self):
        """
        Compatibility patches for SpeechBrain 1.0.3 + newer libs.
        Order matters — torchaudio must be fully patched before
        speechbrain is imported anywhere in the call stack.
        """
        # ── Patch 1: torchaudio — MUST come before speechbrain import ─────────
        import torchaudio
        import soundfile as sf
        import numpy as np
        import torch

        if not hasattr(torchaudio, "list_audio_backends"):
            torchaudio.list_audio_backends = lambda: ["soundfile"]

        if not hasattr(torchaudio, "get_audio_backend"):
            torchaudio.get_audio_backend = lambda: "soundfile"

        if not getattr(torchaudio, "_patched_by_ser_engine", False):
            def _custom_load(filepath, **kwargs):
                data, samplerate = sf.read(filepath)
                data = data.astype(np.float32)
                if data.ndim == 1:
                    tensor = torch.from_numpy(data).unsqueeze(0)
                else:
                    tensor = torch.from_numpy(data.transpose())
                return tensor, samplerate

            torchaudio.load = _custom_load
            torchaudio._patched_by_ser_engine = True

        # ── Patch 2: transformers removed AutoModelWithLMHead in v5 ──────────
        import transformers
        if not hasattr(transformers, "AutoModelWithLMHead"):
            transformers.AutoModelWithLMHead = getattr(
                transformers, "AutoModelForCausalLM", transformers.AutoModel
            )

        # ── Patch 3: huggingface_hub dropped use_auth_token param ─────────────
        import huggingface_hub
        if not getattr(huggingface_hub, "_patched_by_ser_engine", False):
            _original = huggingface_hub.hf_hub_download

            def _patched(*args, **kwargs):
                if "use_auth_token" in kwargs:
                    kwargs["token"] = kwargs.pop("use_auth_token")
                return _original(*args, **kwargs)

            huggingface_hub.hf_hub_download        = _patched
            huggingface_hub._patched_by_ser_engine = True


# ── Module-level singleton ────────────────────────────────────────────────────
# Import this everywhere: from src.core.model_registry import registry
registry = ModelRegistry()