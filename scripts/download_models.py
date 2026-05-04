"""
scripts/download_models.py
===========================
One-time setup. Run ONCE before starting the API for the first time.
After this, the API never needs internet access.

Usage:
    python scripts/download_models.py

Total download: ~1.1 GB
"""

import os
import sys
import traceback

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

WHISPER_CACHE        = os.path.join(PROJECT_ROOT, "external", "whisper")
FASTER_WHISPER_CACHE = os.path.join(PROJECT_ROOT, "external", "faster_whisper")
SPEECHBRAIN_CACHE    = os.path.join(PROJECT_ROOT, "external", "speechbrain")


def _header(t): print(f"\n{'='*60}\n  {t}\n{'='*60}")
def _ok(m):     print(f"  ✅ {m}")
def _fail(m):   print(f"  ❌ {m}")
def _info(m):   print(f"  ℹ️  {m}")


# ── 1. RoBERTa ────────────────────────────────────────────────────────────────

def download_roberta():
    _header("1 / 4 — RoBERTa go_emotions (~500 MB)")
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    _info("Downloading tokenizer...")
    AutoTokenizer.from_pretrained("SamLowe/roberta-base-go_emotions")
    _info("Downloading model weights...")
    AutoModelForSequenceClassification.from_pretrained(
        "SamLowe/roberta-base-go_emotions"
    )
    _ok("RoBERTa cached.")


# ── 2. Whisper base ───────────────────────────────────────────────────────────

def download_whisper():
    _header("2 / 4 — Whisper base (~140 MB)")
    import whisper
    os.makedirs(WHISPER_CACHE, exist_ok=True)
    _info(f"Downloading to {WHISPER_CACHE} ...")
    model = whisper.load_model("base", download_root=WHISPER_CACHE)
    del model
    _ok("Whisper base cached.")


# ── 3. faster-whisper tiny ────────────────────────────────────────────────────

def download_faster_whisper():
    _header("3 / 4 — faster-whisper tiny (~75 MB)")
    from faster_whisper import WhisperModel
    os.makedirs(FASTER_WHISPER_CACHE, exist_ok=True)
    _info(f"Downloading to {FASTER_WHISPER_CACHE} ...")
    model = WhisperModel(
        "tiny",
        device="cpu",
        compute_type="int8",
        download_root=FASTER_WHISPER_CACHE,
    )
    del model
    _ok("faster-whisper tiny cached.")


# ── 4. SpeechBrain Wav2Vec2 IEMOCAP ──────────────────────────────────────────

def download_speechbrain():
    _header("4 / 4 — SpeechBrain Wav2Vec2 IEMOCAP (~360 MB)")

    # ── Patch torchaudio FIRST, before any other import ──────────────────────
    # SpeechBrain internally calls torchaudio.list_audio_backends() during
    # its own import chain. If we patch after importing speechbrain it is
    # already too late. The patch must be injected before speechbrain is
    # touched at all.
    import torchaudio

    if not hasattr(torchaudio, "list_audio_backends"):
        torchaudio.list_audio_backends = lambda: ["soundfile"]
        _info("Patched torchaudio.list_audio_backends")

    if not hasattr(torchaudio, "get_audio_backend"):
        torchaudio.get_audio_backend = lambda: "soundfile"
        _info("Patched torchaudio.get_audio_backend")

    import soundfile as sf
    import numpy as np
    import torch

    if not getattr(torchaudio, "_patched_by_ser_engine", False):
        def _custom_load(filepath, **kwargs):
            data, samplerate = sf.read(filepath)
            data = data.astype(np.float32)
            if data.ndim == 1:
                tensor = torch.from_numpy(data).unsqueeze(0)
            else:
                tensor = torch.from_numpy(data.transpose())
            return tensor, samplerate

        torchaudio.load                = _custom_load
        torchaudio._patched_by_ser_engine = True
        _info("Patched torchaudio.load with soundfile backend")

    # ── Patch transformers ────────────────────────────────────────────────────
    import transformers
    if not hasattr(transformers, "AutoModelWithLMHead"):
        transformers.AutoModelWithLMHead = getattr(
            transformers, "AutoModelForCausalLM", transformers.AutoModel
        )
        _info("Patched transformers.AutoModelWithLMHead")

    # ── Patch huggingface_hub ─────────────────────────────────────────────────
    import huggingface_hub
    if not getattr(huggingface_hub, "_patched_by_ser_engine", False):
        _orig = huggingface_hub.hf_hub_download

        def _patched_download(*args, **kwargs):
            if "use_auth_token" in kwargs:
                kwargs["token"] = kwargs.pop("use_auth_token")
            return _orig(*args, **kwargs)

        huggingface_hub.hf_hub_download    = _patched_download
        huggingface_hub._patched_by_ser_engine = True
        _info("Patched huggingface_hub.hf_hub_download")

    # ── Download SpeechBrain ──────────────────────────────────────────────────
    os.makedirs(SPEECHBRAIN_CACHE, exist_ok=True)
    _info(f"Downloading to {SPEECHBRAIN_CACHE} ...")

    try:
        from speechbrain.inference.interfaces import foreign_class

        clf = foreign_class(
            source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
            pymodule_file="custom_interface.py",
            classname="CustomEncoderWav2vec2Classifier",
            savedir=SPEECHBRAIN_CACHE,
            run_opts={"device": "cpu"},
        )
        del clf
        _ok(f"SpeechBrain cached to {SPEECHBRAIN_CACHE}")

    except Exception as e:
        traceback.print_exc()
        raise RuntimeError(f"SpeechBrain download failed: {e}") from e


# ── Verification ──────────────────────────────────────────────────────────────

def verify():
    _header("Verification")

    whisper_ok = (
        os.path.isdir(WHISPER_CACHE)
        and any(f.endswith(".pt") for f in os.listdir(WHISPER_CACHE))
    )
    fw_ok = (
        os.path.isdir(FASTER_WHISPER_CACHE)
        and len(os.listdir(FASTER_WHISPER_CACHE)) > 0
    )
    sb_ok = (
        os.path.isdir(SPEECHBRAIN_CACHE)
        and len(os.listdir(SPEECHBRAIN_CACHE)) > 0
    )

    hf_cache = os.path.expanduser("~/.cache/huggingface/hub")
    hf_dirs  = os.listdir(hf_cache) if os.path.isdir(hf_cache) else []
    roberta_ok = any(
        "roberta" in d.lower() or "go_emotion" in d.lower()
        for d in hf_dirs
    )

    checks = [
        (whisper_ok,  "Whisper base",   WHISPER_CACHE),
        (fw_ok,       "faster-whisper", FASTER_WHISPER_CACHE),
        (sb_ok,       "SpeechBrain",    SPEECHBRAIN_CACHE),
        (roberta_ok,  "RoBERTa",        "~/.cache/huggingface/hub"),
    ]

    all_ok = True
    for ok, name, path in checks:
        icon = "✅" if ok else "❌"
        print(f"  {icon}  {name:<20} → {path}")
        if not ok:
            all_ok = False

    if all_ok:
        print("\n  🎉 All models ready. Run: uvicorn api:app --reload")
    else:
        print("\n  ⚠️  Some models missing. Re-run this script.")

    return all_ok


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n🚀 Humanoid Assistant — One-Time Model Setup (~1.1 GB)")

    steps = [
        ("RoBERTa",        download_roberta),
        ("Whisper",        download_whisper),
        ("faster-whisper", download_faster_whisper),
        ("SpeechBrain",    download_speechbrain),
    ]

    failed = []
    for name, fn in steps:
        try:
            fn()
        except Exception as e:
            _fail(f"{name} failed: {e}")
            failed.append(name)

    verify()

    if failed:
        print(f"\n⚠️  Re-run after fixing: {failed}")
        sys.exit(1)

    sys.exit(0)