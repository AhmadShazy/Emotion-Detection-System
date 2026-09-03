"""
scripts/download_models.py
===========================
One-time setup. Run ONCE before starting the API for the first time.
After this, the API never needs internet access.

Usage:
    python scripts/download_models.py

Total download: ~1.7 GB (measured, across external/ and the HF cache)
"""

import os
import sys
import traceback

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# This script's progress output uses emoji, which raise UnicodeEncodeError
# on a default Windows cp1252 console. Widen the streams first.
from src.core.console import enable_utf8_console
enable_utf8_console()

WHISPER_CACHE        = os.path.join(PROJECT_ROOT, "external", "whisper")
FASTER_WHISPER_CACHE = os.path.join(PROJECT_ROOT, "external", "faster_whisper")
SPEECHBRAIN_CACHE    = os.path.join(PROJECT_ROOT, "external", "speechbrain")


def _header(t): print(f"\n{'='*60}\n  {t}\n{'='*60}")
def _ok(m):     print(f"  ✅ {m}")
def _fail(m):   print(f"  ❌ {m}")
def _info(m):   print(f"  ℹ️  {m}")


# ── 1. RoBERTa ────────────────────────────────────────────────────────────────

def download_roberta():
    _header("1 / 5 — RoBERTa go_emotions (~500 MB)")
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
    _header("2 / 5 — Whisper base (~140 MB)")
    import whisper
    os.makedirs(WHISPER_CACHE, exist_ok=True)
    _info(f"Downloading to {WHISPER_CACHE} ...")
    model = whisper.load_model("base", download_root=WHISPER_CACHE)
    del model
    _ok("Whisper base cached.")


# ── 3. faster-whisper tiny ────────────────────────────────────────────────────

def download_faster_whisper():
    _header("3 / 5 — faster-whisper tiny (~75 MB)")
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
    _header("4 / 5 — SpeechBrain Wav2Vec2 IEMOCAP (~360 MB)")

    # Compatibility patches come from the ONE shared definition. This step
    # used to carry its own partial copy that omitted the LazyModule patch, so
    # it failed on speechbrain 1.1.0 with
    #     ImportError: Lazy import of LazyModule(target=...k2_fsa) failed
    # while the running server loaded the same model fine. A Docker build runs
    # this script, so the broken copy was the one deployment depended on.
    from src.core.speechbrain_compat import apply_patches, fetch_kwargs
    apply_patches(log=_info)

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
            **fetch_kwargs(),
        )
        del clf
        _ok(f"SpeechBrain cached to {SPEECHBRAIN_CACHE}")

    except Exception as e:
        traceback.print_exc()
        raise RuntimeError(f"SpeechBrain download failed: {e}") from e


# ── 5. MediaPipe face landmarker ─────────────────────────────────────────────

MEDIAPIPE_CACHE = os.path.join(PROJECT_ROOT, "external", "mediapipe")
MEDIAPIPE_MODEL = os.path.join(MEDIAPIPE_CACHE, "face_landmarker.task")
MEDIAPIPE_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)


def download_mediapipe_face():
    _header("5 / 5 — MediaPipe face landmarker (~4 MB)")

    # Not a pip dependency — the mediapipe package ships the runtime, but the
    # model bundle is a separate download.
    import urllib.request

    os.makedirs(MEDIAPIPE_CACHE, exist_ok=True)

    if os.path.isfile(MEDIAPIPE_MODEL):
        _ok(f"Already present at {MEDIAPIPE_MODEL}")
        return

    _info(f"Downloading to {MEDIAPIPE_CACHE} ...")
    urllib.request.urlretrieve(MEDIAPIPE_URL, MEDIAPIPE_MODEL)
    _ok(f"MediaPipe face landmarker cached "
        f"({os.path.getsize(MEDIAPIPE_MODEL) / 1e6:.1f} MB)")


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

    # Honour HF_HOME. The Docker image points it at /app/external/hf so every
    # cache sits in one tree — checking the default path there would report
    # RoBERTa missing on an image that has it, and fail a correct build.
    hf_home  = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    hf_cache = os.path.join(hf_home, "hub")
    hf_dirs  = os.listdir(hf_cache) if os.path.isdir(hf_cache) else []
    roberta_ok = any(
        "roberta" in d.lower() or "go_emotion" in d.lower()
        for d in hf_dirs
    )

    # MediaPipe was missing from this list. It is the one model outside the
    # registry, and when its .task file is absent the resulting error is
    # swallowed into a value meaning "no face was visible" — so a missing file
    # is indistinguishable from a dark room and conflict_analysis can never
    # fire. Exactly the thing a verification step exists to catch.
    mp_ok = os.path.isfile(MEDIAPIPE_MODEL)

    checks = [
        (whisper_ok,  "Whisper base",   WHISPER_CACHE),
        (fw_ok,       "faster-whisper", FASTER_WHISPER_CACHE),
        (sb_ok,       "SpeechBrain",    SPEECHBRAIN_CACHE),
        (roberta_ok,  "RoBERTa",        hf_cache),
        (mp_ok,       "MediaPipe face", MEDIAPIPE_MODEL),
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
    print("\n🚀 Humanoid Assistant — One-Time Model Setup (~1.7 GB)")

    steps = [
        ("RoBERTa",        download_roberta),
        ("Whisper",        download_whisper),
        ("faster-whisper", download_faster_whisper),
        ("SpeechBrain",    download_speechbrain),
        ("MediaPipe face", download_mediapipe_face),
    ]

    failed = []
    for name, fn in steps:
        try:
            fn()
        except Exception as e:
            _fail(f"{name} failed: {e}")
            failed.append(name)

    verified = verify()

    # The exit code gates the Docker build, so it has to reflect what actually
    # landed on disk and not merely that no exception escaped. A download step
    # can return cleanly and still leave nothing behind — and an image missing a
    # model does not crash at runtime, it serves "neutral" behind a green health
    # check, which is the failure this whole script exists to prevent.
    if failed or not verified:
        if failed:
            print(f"\n⚠️  Re-run after fixing: {failed}")
        if not verified:
            print("\n⚠️  Some models are missing from disk after downloading.")
        sys.exit(1)

    sys.exit(0)