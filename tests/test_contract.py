"""
tests/test_contract.py
======================
Guards the JSON contract with the LLM team.

The payload shape is agreed with another module that is being built in
parallel. If it changes, their integration breaks — usually silently, and
usually noticed late. These tests turn that into a failing test instead.

Two layers:
  1. The frozen examples in contract/payloads/ must satisfy every invariant.
     Fast, no models needed.
  2. A live request through the real app must satisfy the SAME invariants, so
     the examples cannot drift away from what the API actually emits.

Run:  python -m pytest tests/ -v
Live layer only:  python -m pytest tests/ -v -m live
"""

import os
import sys
import json
import glob

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

PAYLOAD_DIR = os.path.join(PROJECT_ROOT, "contract", "payloads")


# ══════════════════════════════════════════════════════════════════════════════
# THE CONTRACT
# Changing anything in this block is a breaking change for the LLM team.
# ══════════════════════════════════════════════════════════════════════════════

TOP_LEVEL_KEYS = {
    "session_id", "user_input", "emotion_analysis",
    "tone_analysis", "conflict_analysis",
}

USER_INPUT_KEYS       = {"text", "timestamp"}
EMOTION_ANALYSIS_KEYS = {"dominant_emotion", "confidence", "emotion_probabilities"}
TONE_ANALYSIS_KEYS    = {"tone", "confidence"}
CONFLICT_KEYS         = {"detected", "type", "details"}

# The four mismatches the fusion engine can name, plus "none".
CONFLICT_TYPES = {
    "none",
    "masked_anger",            # happy face, angry voice
    "suppressed_frustration",  # neutral face, angry voice
    "masked_sadness",          # happy face, sad voice
    "internal_sadness",        # sad face, neutral voice
}

# The 15 emotion classes. Every payload carries all fifteen, always.
EMOTION_CLASSES = {
    "happy", "sad", "angry", "surprised", "neutral",
    "empathetic", "concerned", "fear", "disgust",
    "shame", "guilt", "anxiety", "frustration", "joy", "calm",
}

# Every tone the adapter can currently emit.
TONE_VALUES = {
    "frustrated", "hostile", "tense", "panicked", "nervous",
    "somber", "reflective", "amazed", "excited", "cheerful",
    "measured", "conversational", "neutral",
}

PROBABILITY_SUM_TOLERANCE = 0.02   # values are rounded to 2dp before summing


# ── Shared assertions ─────────────────────────────────────────────────────────

def assert_valid_payload(payload: dict, source: str):
    """Every invariant the LLM team is entitled to rely on."""

    assert isinstance(payload, dict), f"{source}: payload must be an object"

    # ── Exact top-level shape ────────────────────────────────────────────────
    assert set(payload) == TOP_LEVEL_KEYS, (
        f"{source}: top-level keys changed.\n"
        f"  expected: {sorted(TOP_LEVEL_KEYS)}\n"
        f"  got:      {sorted(payload)}\n"
        f"  This breaks the LLM integration."
    )

    # ── session_id ───────────────────────────────────────────────────────────
    assert isinstance(payload["session_id"], str) and payload["session_id"], \
        f"{source}: session_id must be a non-empty string"

    # ── user_input ───────────────────────────────────────────────────────────
    ui = payload["user_input"]
    assert set(ui) == USER_INPUT_KEYS, f"{source}: user_input keys changed: {sorted(ui)}"
    # May be empty (silence, or a filtered Whisper hallucination) but never null.
    assert isinstance(ui["text"], str), f"{source}: user_input.text must be a string, never null"
    assert isinstance(ui["timestamp"], str) and ui["timestamp"].endswith("Z"), \
        f"{source}: timestamp must be an ISO-8601 UTC string ending in Z"

    # ── emotion_analysis ─────────────────────────────────────────────────────
    ea = payload["emotion_analysis"]
    assert set(ea) == EMOTION_ANALYSIS_KEYS, \
        f"{source}: emotion_analysis keys changed: {sorted(ea)}"

    assert ea["dominant_emotion"] in EMOTION_CLASSES, (
        f"{source}: dominant_emotion '{ea['dominant_emotion']}' is outside the "
        f"agreed 15 classes"
    )

    conf = ea["confidence"]
    assert isinstance(conf, (int, float)) and 0.0 <= conf <= 1.0, \
        f"{source}: confidence must be a number in [0,1], got {conf!r}"

    probs = ea["emotion_probabilities"]
    assert set(probs) == EMOTION_CLASSES, (
        f"{source}: emotion_probabilities must contain exactly the 15 classes.\n"
        f"  missing: {sorted(EMOTION_CLASSES - set(probs))}\n"
        f"  extra:   {sorted(set(probs) - EMOTION_CLASSES)}"
    )
    for name, value in probs.items():
        assert isinstance(value, (int, float)) and 0.0 <= value <= 1.0, \
            f"{source}: probability '{name}' out of range: {value!r}"

    total = sum(probs.values())
    assert abs(total - 1.0) <= PROBABILITY_SUM_TOLERANCE, \
        f"{source}: probabilities sum to {total:.3f}, expected 1.0"

    # ── tone_analysis ────────────────────────────────────────────────────────
    ta = payload["tone_analysis"]
    assert set(ta) == TONE_ANALYSIS_KEYS, f"{source}: tone_analysis keys changed: {sorted(ta)}"
    assert ta["tone"] in TONE_VALUES, f"{source}: unknown tone '{ta['tone']}'"
    assert isinstance(ta["confidence"], (int, float)) and 0.0 <= ta["confidence"] <= 1.0, \
        f"{source}: tone confidence out of range"

    # ── conflict_analysis ────────────────────────────────────────────────────
    ca = payload["conflict_analysis"]
    assert set(ca) == CONFLICT_KEYS, \
        f"{source}: conflict_analysis keys changed: {sorted(ca)}"
    assert isinstance(ca["detected"], bool), \
        f"{source}: conflict detected must be a real boolean, not {type(ca['detected'])}"
    assert ca["type"] in CONFLICT_TYPES, \
        f"{source}: unknown conflict type '{ca['type']}'"
    assert isinstance(ca["details"], str), \
        f"{source}: conflict details must be a string, empty rather than null"

    # detected and type must agree — a mismatch here would mislead the LLM
    # in exactly the situation the field exists to clarify.
    if ca["detected"]:
        assert ca["type"] != "none", \
            f"{source}: conflict detected but type is 'none'"
        assert ca["details"], \
            f"{source}: conflict detected but no explanation given"
    else:
        assert ca["type"] == "none", \
            f"{source}: no conflict detected but type is '{ca['type']}'"


# ══════════════════════════════════════════════════════════════════════════════
# Layer 1 — the frozen examples (fast, no models)
# ══════════════════════════════════════════════════════════════════════════════

def _payload_files():
    files = sorted(glob.glob(os.path.join(PAYLOAD_DIR, "*.json")))
    return [f for f in files if not f.endswith("index.json")]


def test_payload_directory_is_populated():
    files = _payload_files()
    assert files, (
        "No contract payloads found. Run:\n"
        "    python scripts/generate_contract_payloads.py"
    )


@pytest.mark.parametrize("path", _payload_files(), ids=lambda p: os.path.basename(p)[:-5])
def test_frozen_payload_matches_contract(path):
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    assert_valid_payload(payload, os.path.basename(path))


def test_index_matches_files_on_disk():
    """The index the mock endpoint serves must not reference missing files."""
    index_path = os.path.join(PAYLOAD_DIR, "index.json")
    assert os.path.isfile(index_path), "index.json missing — re-run the generator"

    with open(index_path, encoding="utf-8") as f:
        index = json.load(f)

    listed = {s["name"] for s in index["scenarios"]}
    on_disk = {os.path.basename(p)[:-5] for p in _payload_files()}
    assert listed == on_disk, (
        f"index.json is out of sync with the payload files.\n"
        f"  listed but missing: {sorted(listed - on_disk)}\n"
        f"  on disk but unlisted: {sorted(on_disk - listed)}"
    )


def test_examples_cover_the_hard_cases():
    """
    The whole point of these examples is that the LLM team sees more than the
    easy path. If someone trims the set down to only simple cases, that defeats
    the purpose, so it is asserted.
    """
    names = {os.path.basename(p)[:-5] for p in _payload_files()}

    assert any(n.startswith("conflict_") for n in names), \
        "No conflict example — the LLM team must see the masked-emotion case"
    assert "no_speech_detected" in names, \
        "No silence example — the LLM team must handle an empty transcript"
    assert "low_confidence_disagreement" in names, \
        "No low-confidence example — the LLM team must handle weak evidence"


def test_conflict_examples_actually_report_their_conflict():
    """
    The conflict scenarios exist to prove the field works. If one of them comes
    back with detected=false, the detection has broken — which would be
    invisible otherwise, since the payload would still be structurally valid.
    """
    expected = {
        "conflict_masked_anger":           "masked_anger",
        "conflict_suppressed_frustration": "suppressed_frustration",
        "conflict_internal_sadness":       "internal_sadness",
    }
    for name, conflict_type in expected.items():
        path = os.path.join(PAYLOAD_DIR, f"{name}.json")
        assert os.path.isfile(path), f"missing example {name}.json"
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        ca = payload["conflict_analysis"]
        assert ca["detected"] is True, f"{name}: conflict was not detected"
        assert ca["type"] == conflict_type, \
            f"{name}: expected '{conflict_type}', got '{ca['type']}'"


def test_single_signal_payloads_report_no_conflict():
    """A conflict needs both a face and a voice. Text alone cannot produce one."""
    for name in ("text_only_neutral", "text_only_joy", "text_only_sadness"):
        with open(os.path.join(PAYLOAD_DIR, f"{name}.json"), encoding="utf-8") as f:
            payload = json.load(f)
        assert payload["conflict_analysis"]["detected"] is False, \
            f"{name}: text-only input cannot have a face/voice conflict"


def test_empty_transcript_is_representable():
    """Silence must produce an empty string, not a null or a missing key."""
    path = os.path.join(PAYLOAD_DIR, "no_speech_detected.json")
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    assert payload["user_input"]["text"] == "", \
        "The silence case must send an empty string so the LLM can detect it"


# ══════════════════════════════════════════════════════════════════════════════
# Layer 2 — the live API must satisfy the same contract
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.live
def test_live_text_endpoint_matches_contract():
    """
    Loads the real models, so it is slow. This is the test that catches the
    examples drifting away from reality.
    """
    from fastapi.testclient import TestClient
    from src.core.config import API_KEYS
    import api

    headers = {"X-API-Key": next(iter(API_KEYS))} if API_KEYS else {}

    with TestClient(api.app) as client:
        response = client.post(
            "/analyze/text",
            json={"text": "I am really frustrated with how this turned out."},
            headers=headers,
        )
        assert response.status_code == 200, response.text
        assert_valid_payload(response.json(), "live /analyze/text")


@pytest.mark.live
def test_live_session_continuity():
    """
    Replaying session_id must reuse the session and let confidence build.
    Guards the regression where the frontend dropped session_id, which silently
    disabled all temporal smoothing.
    """
    from fastapi.testclient import TestClient
    from src.core.config import API_KEYS
    import api

    headers = {"X-API-Key": next(iter(API_KEYS))} if API_KEYS else {}
    body = {"text": "I am absolutely furious about this"}

    with TestClient(api.app) as client:
        first = client.post("/analyze/text", json=body, headers=headers).json()
        session_id = first["session_id"]

        latest = first
        for _ in range(3):
            latest = client.post(
                "/analyze/text",
                json={**body, "session_id": session_id},
                headers=headers,
            ).json()

        assert latest["session_id"] == session_id, "session was not reused"
        assert latest["emotion_analysis"]["confidence"] > first["emotion_analysis"]["confidence"], (
            "Confidence did not rise across repeated turns — temporal smoothing "
            "is not accumulating."
        )


# ══════════════════════════════════════════════════════════════════════════════
# Cross-path consistency
# ══════════════════════════════════════════════════════════════════════════════

def test_voice_state_is_built_in_exactly_one_place():
    """
    The upload paths and the live call must agree on how SER confidence becomes
    the reliability the fusion engine weights by.

    They each had their own copy of the formula. Identical at the time, but that
    is precisely the shape of thing that drifts — tweak one and the same audio
    scores differently depending on which mode it arrived through.
    """
    import subprocess

    result = subprocess.run(
        ["git", "grep", "-n", "confidence + 0.15", "--", "src/"],
        capture_output=True, text=True, cwd=PROJECT_ROOT,
    )
    hits = [ln for ln in result.stdout.splitlines() if ln.strip()]

    assert len(hits) == 1, (
        "The SER reliability formula should exist in exactly one place "
        f"(unified_pipeline.build_voice_state). Found {len(hits)}:\n"
        + "\n".join(hits)
    )
    assert "unified_pipeline" in hits[0], f"unexpected location: {hits[0]}"


def test_speechbrain_patches_exist_in_exactly_one_place():
    """
    The most fragile code in the project is the SpeechBrain compatibility patch
    set: monkeypatches against a PRIVATE API, one of which exists only because
    transformers removed AutoModelWithLMHead in v5.

    It used to be written out three times — ModelRegistry, SEREngine, and a
    PARTIAL third copy in scripts/download_models.py. The partial one omitted
    the LazyModule patch, so on speechbrain 1.1.0 the downloader died with
    "Lazy import of LazyModule(target=...k2_fsa) failed" while the running
    server loaded the identical model fine. A Docker build runs the downloader,
    so the broken copy was the one deployment depended on — and the failure
    only surfaced when a build was first attempted.

    The reliability formula and the silence threshold each already have a guard
    like this. The thing most likely to break on a dependency bump had none.
    """
    import subprocess

    result = subprocess.run(
        # --untracked matters: without it a brand-new file is invisible to the
        # search, so adding a second copy in a file you have not committed yet
        # reads as "zero copies exist" rather than as the duplication it is.
        ["git", "grep", "-n", "--untracked", "def _patched_ensure_module", "--",
         "src/", "scripts/", "routers/", "api.py"],
        capture_output=True, text=True, cwd=PROJECT_ROOT,
    )
    assert result.returncode in (0, 1), (
        f"git grep failed to run: {result.stderr.strip()}"
    )
    hits = [ln for ln in result.stdout.splitlines() if ln.strip()]

    assert len(hits) == 1, (
        "The SpeechBrain LazyModule patch should exist in exactly one place "
        f"(src/core/speechbrain_compat.py). Found {len(hits)}:\n"
        + "\n".join(hits)
    )
    assert "speechbrain_compat" in hits[0].replace("\\", "/"), (
        f"unexpected location: {hits[0]}"
    )


def test_every_speechbrain_caller_fetches_the_same_way():
    """
    The downloader bakes the model into the image and the registry loads it at
    runtime. If they disagree about how files land in savedir, the image
    contains one shape and the server expects another.

    Both must pass fetch_kwargs(), which asks SpeechBrain to COPY rather than
    symlink into the Hugging Face cache. The symlink default left savedir
    holding five links that only resolve alongside that cache — a Windows
    privilege error locally, and a dangling-link trap across Docker layers.
    """
    import subprocess

    result = subprocess.run(
        ["git", "grep", "-n", "--untracked", "foreign_class(", "--",
         "src/", "scripts/"],
        capture_output=True, text=True, cwd=PROJECT_ROOT,
    )
    assert result.returncode in (0, 1), (
        f"git grep failed to run: {result.stderr.strip()}"
    )
    # Only the actual invocations, not the imports.
    call_sites = [
        ln for ln in result.stdout.splitlines()
        if ln.strip() and "import" not in ln
    ]
    assert call_sites, "expected to find foreign_class call sites"

    for site in call_sites:
        path = site.split(":")[0]
        source = open(os.path.join(PROJECT_ROOT, path), encoding="utf-8").read()
        assert "fetch_kwargs()" in source, (
            f"{path} calls foreign_class without fetch_kwargs(), so it may "
            f"fetch differently from the other callers"
        )


def test_silence_threshold_is_shared_between_paths():
    """
    Upload and live must agree on what counts as too quiet to be speech.

    /analyze/voice turns this into a hard 422, so a mismatch means the same
    quiet recording is rejected on one path and analysed on the other. Five of
    the eighteen recordings in data/recordings/ sit below the old hardcoded
    0.01, so this was not hypothetical.
    """
    import inspect
    from src.interactive_modes import _check_audio_has_speech
    from src.streaming.turn_detector import MIN_ABSOLUTE_THRESHOLD

    source = inspect.getsource(_check_audio_has_speech)
    assert "MIN_ABSOLUTE_THRESHOLD" in source, (
        "_check_audio_has_speech should use the live path's shared floor "
        "rather than its own hardcoded threshold"
    )
    assert MIN_ABSOLUTE_THRESHOLD < 0.01, (
        "the shared floor should be below the old 0.01, which rejected quiet "
        "but perfectly real speech"
    )


def test_config_is_the_only_module_reading_the_environment():
    """
    src/core/config.py documents itself as the single place environment
    variables are read. Capacity settings had leaked into a router and into the
    face analyzer, so tuning the system meant hunting through three files.
    """
    import subprocess

    result = subprocess.run(
        ["git", "grep", "-n", "-E", r"os\.environ|os\.getenv", "--",
         "src/", "routers/", "api.py"],
        capture_output=True, text=True, cwd=PROJECT_ROOT,
    )
    offenders = [
        ln for ln in result.stdout.splitlines()
        if ln.strip() and "src/core/config.py" not in ln.replace("\\", "/")
    ]

    assert not offenders, (
        "Only src/core/config.py should read the environment. Found:\n"
        + "\n".join(offenders)
    )
