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
        from src.core.speechbrain_compat import fetch_kwargs
        from speechbrain.inference.interfaces import foreign_class
        self._models["speechbrain"] = foreign_class(
            source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
            pymodule_file="custom_interface.py",
            classname="CustomEncoderWav2vec2Classifier",
            savedir=SPEECHBRAIN_CACHE,
            run_opts={"device": "cpu"},
            # Same fetch behaviour as the downloader, so what the image bakes in
            # is what the server expects to find.
            **fetch_kwargs(),
        )

    def _apply_speechbrain_patches(self):
        """
        Delegates to the single definition in src/core/speechbrain_compat.py.

        This body used to be a copy. So did SEREngine's, and a third partial
        copy in scripts/download_models.py that omitted the LazyModule patch —
        which is why the downloader failed on speechbrain 1.1.0 while the
        server loaded the same model without complaint.
        """
        from src.core.speechbrain_compat import apply_patches
        apply_patches()


# ── Module-level singleton ────────────────────────────────────────────────────
registry = ModelRegistry()