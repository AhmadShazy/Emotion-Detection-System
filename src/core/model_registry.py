"""
src/core/model_registry.py
===========================
Central singleton registry that owns all ML model instances.

TEXT_ONLY_MODE=true  → loads only RoBERTa (~350MB RAM)
TEXT_ONLY_MODE=false → loads all 4 models (~1.4GB RAM)
"""

import os
import sys
import threading
import time

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

# Cache paths
WHISPER_CACHE        = os.path.join(PROJECT_ROOT, "external", "whisper")
FASTER_WHISPER_CACHE = os.path.join(PROJECT_ROOT, "external", "faster_whisper")
SPEECHBRAIN_CACHE    = os.path.join(PROJECT_ROOT, "external", "speechbrain")

# ── Read mode flag ─────────────────────────────────────────────────────────────
from src.core.config import TEXT_ONLY_MODE


class ModelRegistry:

    def __init__(self):
        self._models:     dict = {}
        self._lock              = threading.Lock()
        self._loaded            = False
        self._load_times: dict  = {}

    def load_all(self):
        with self._lock:
            if self._loaded:
                print("[Registry] Already loaded — skipping.")
                return

            print("\n[Registry] ══════════════════════════════════════")
            if TEXT_ONLY_MODE:
                print("[Registry] TEXT_ONLY_MODE=true — loading RoBERTa only")
            else:
                print("[Registry] Loading all models into memory...")
            print("[Registry] ══════════════════════════════════════")

            # ── Build loader list based on mode ───────────────────────────────
            if TEXT_ONLY_MODE:
                loaders = [
                    ("roberta", self._load_roberta),
                ]
            else:
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
            else:
                print("[Registry] 🎉 All models ready.")
            print("[Registry] ══════════════════════════════════════\n")

    def get(self, name: str):
        if name not in self._models:
            raise RuntimeError(
                f"[Registry] Model '{name}' is not available. "
                f"Available: {list(self._models.keys())}"
            )
        return self._models[name]

    def is_available(self, name: str) -> bool:
        return name in self._models

    def status(self) -> dict:
        return {
            "loaded":      self._loaded,
            "mode":        "text_only" if TEXT_ONLY_MODE else "full",
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
            "base", download_root=WHISPER_CACHE,
        )

    def _load_faster_whisper(self):
        from faster_whisper import WhisperModel
        self._models["faster_whisper"] = WhisperModel(
            "tiny", device="cpu", compute_type="int8",
            download_root=FASTER_WHISPER_CACHE,
        )

    def _load_speechbrain(self):
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

        import transformers
        if not hasattr(transformers, "AutoModelWithLMHead"):
            transformers.AutoModelWithLMHead = getattr(
                transformers, "AutoModelForCausalLM", transformers.AutoModel
            )

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
            print(f"[Registry] Warning: LazyModule patch failed: {pe}")


# ── Module-level singleton ────────────────────────────────────────────────────
registry = ModelRegistry()