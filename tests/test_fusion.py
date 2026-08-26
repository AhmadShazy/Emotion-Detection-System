"""
tests/test_fusion.py
====================
Direct tests for the fusion engine — the most intricate logic in the project
and, until this file existed, the only complex part with no test of its own.

Everything here runs against EmotionStateManager directly. It imports nothing
but `collections.deque`, so these tests need no models, no server and no audio,
and the whole file runs in milliseconds.

The conflict cases matter most. Detecting that a smile is covering anger is the
one thing this system does that a sentiment classifier cannot, and the only
coverage it used to have was a test that opened a checked-in JSON file and
asserted on its contents — which passes whether or not detect_conflict still
works.
"""

import os
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.streaming.emotion_state_manager import EmotionStateManager


# ── State builders, matching what the real workers emit ───────────────────────

def text_state(emotion, confidence, reliability=1.0):
    return {"emotion": emotion, "confidence": confidence,
            "reliability": reliability}


def voice_state(emotion, confidence, reliability=1.0):
    return {"source": "voice", "emotion": emotion, "confidence": confidence,
            "average_emotion": emotion, "peak_emotion": emotion,
            "reliability": reliability}


def face_state(emotion, confidence, reliability=1.0, instability=0.1):
    return {"source": "face", "emotion": emotion, "confidence": confidence,
            "reliability": reliability, "instability": instability}


def fuse_fresh(t=None, v=None, f=None):
    """One turn against a brand-new engine, so no history leaks in."""
    return EmotionStateManager().fuse(t, v, f)


# ── Conflict detection ────────────────────────────────────────────────────────
# The headline capability. Each row is (face, voice, expected type).

@pytest.mark.parametrize("face_emo,voice_emo,expected", [
    ("Happy",   "Angry",   "masked_anger"),
    ("Neutral", "Angry",   "suppressed_frustration"),
    ("Sad",     "Neutral", "internal_sadness"),
    ("Happy",   "Sad",     "masked_sadness"),
])
def test_each_conflict_type_is_detected(face_emo, voice_emo, expected):
    """
    Guards the elif chain in detect_conflict. Breaking any branch used to leave
    every test in the suite green.
    """
    result = fuse_fresh(
        t=text_state("neutral", 0.5),
        v=voice_state(voice_emo, 0.8),
        f=face_state(face_emo, 0.7),
    )
    conflict = result["conflict_analysis"]
    assert conflict["detected"] is True, (
        f"face={face_emo} voice={voice_emo} should report {expected}"
    )
    assert conflict["type"] == expected
    assert conflict["details"], "a detected conflict must explain itself"


def test_agreeing_modalities_report_no_conflict():
    result = fuse_fresh(
        t=text_state("sadness", 0.9),
        v=voice_state("Sad", 0.9),
        f=face_state("Sad", 0.9),
    )
    assert result["conflict_analysis"]["detected"] is False
    assert result["conflict_analysis"]["type"] == "none"
    assert result["conflict_analysis"]["details"] == ""


def test_a_conflict_needs_both_a_face_and_a_voice():
    """CONTRACT.md: 'Text-only requests can never report a conflict.'"""
    for v, f in (
        (None, None),
        (voice_state("Angry", 0.9), None),
        (None, face_state("Happy", 0.9)),
    ):
        result = fuse_fresh(t=text_state("neutral", 0.6), v=v, f=f)
        assert result["conflict_analysis"]["detected"] is False


# ── Confidence calibration ────────────────────────────────────────────────────

@pytest.mark.parametrize("emotion", [
    "neutral", "sadness", "joy", "anger", "fear",
])
def test_first_turn_never_exceeds_the_documented_ceiling(emotion):
    """
    CONTRACT.md states in bold that the first turn of any session cannot exceed
    0.70.

    This used to be false for exactly one class. The memory was pre-seeded with
    five neutrals, so a first-turn neutral carried a full historical score and
    reached 0.97 while everything else was capped at 0.70 — the system was most
    certain about its least actionable answer.
    """
    result = fuse_fresh(t=text_state(emotion, 0.99))
    assert result["confidence"] <= 0.70, (
        f"first-turn {emotion} scored {result['confidence']}, "
        f"breaking the documented 0.70 ceiling"
    )


def test_neutral_gets_no_head_start_over_other_emotions():
    """The same reading strength must score the same whatever the label."""
    neutral = fuse_fresh(t=text_state("neutral", 0.9))["confidence"]
    sad     = fuse_fresh(t=text_state("sadness", 0.9))["confidence"]
    assert neutral == pytest.approx(sad, abs=0.01), (
        f"neutral {neutral} vs sad {sad} — identical evidence must score alike"
    )


def test_adding_agreeing_modalities_never_lowers_confidence():
    """
    Calibration belongs on the weight, not the confidence.

    While it multiplied the confidence, it sat in the numerator of the fusion
    ratio while the denominator kept the uncalibrated weight — so voice (0.9)
    and face (0.8) dragged the result down just by being present. Measured at
    the time: 0.63 text-only, 0.59 with voice, 0.57 with face. More agreeing
    evidence made the system less sure.
    """
    t = text_state("sadness", 0.90)
    v = voice_state("Sad", 0.90)
    f = face_state("Sad", 0.90)

    text_only  = fuse_fresh(t=t)["confidence"]
    plus_voice = fuse_fresh(t=t, v=v)["confidence"]
    plus_face  = fuse_fresh(t=t, v=v, f=f)["confidence"]

    assert plus_voice >= text_only - 0.005, (
        f"adding an agreeing voice dropped confidence "
        f"{text_only} -> {plus_voice}"
    )
    assert plus_face >= plus_voice - 0.005, (
        f"adding an agreeing face dropped confidence "
        f"{plus_voice} -> {plus_face}"
    )


def test_confidence_rises_as_an_emotion_persists():
    """
    CONTRACT.md: 'Confidence rises as an emotion persists across turns.'

    Checked for neutral specifically, which used to start saturated at 0.93 and
    stay flat forever — it could not rise because it began at the top.
    """
    for emotion in ("neutral", "sadness"):
        manager = EmotionStateManager()
        run = [
            manager.fuse(text_state(emotion, 0.9), None, None)["confidence"]
            for _ in range(4)
        ]
        assert run == sorted(run), f"{emotion} did not rise monotonically: {run}"
        assert run[-1] > run[0], (
            f"{emotion} never rose across four identical turns: {run}"
        )


def test_genuine_disagreement_still_scores_low():
    """The calibration fix must not simply inflate everything."""
    result = fuse_fresh(
        t=text_state("sadness", 0.9),
        v=voice_state("Angry", 0.9),
        f=face_state("Happy", 0.9),
    )
    assert result["confidence"] < 0.50, (
        f"three-way disagreement scored {result['confidence']}"
    )


def test_no_signals_at_all_scores_zero():
    """Nothing in means no evidence, not a floor value."""
    result = fuse_fresh()
    assert result["confidence"] == 0.0
    assert result["dominant_emotion"] == "neutral"


# ── Structural invariants ─────────────────────────────────────────────────────

def test_probabilities_always_sum_to_one():
    cases = [
        (text_state("joy", 0.9), None, None),
        (text_state("neutral", 0.5), voice_state("Angry", 0.8), None),
        (text_state("neutral", 0.5), voice_state("Angry", 0.8),
         face_state("Happy", 0.7)),
        (None, None, None),
    ]
    for t, v, f in cases:
        probs = fuse_fresh(t, v, f)["emotion_probabilities"]
        assert sum(probs.values()) == pytest.approx(1.0, abs=0.02)
        assert all(0.0 <= p <= 1.0 for p in probs.values())


def test_confidence_is_always_in_range():
    """Fuzzed across the modality combinations the four paths can produce."""
    emotions = ["neutral", "joy", "sadness", "anger", "fear", "disgust"]
    for i, emo_a in enumerate(emotions):
        for emo_b in emotions:
            for conf in (0.0, 0.5, 1.0):
                result = fuse_fresh(
                    t=text_state(emo_a, conf),
                    v=voice_state(emo_b, conf),
                    f=face_state(emo_a, conf) if i % 2 else None,
                )
                assert 0.0 <= result["confidence"] <= 1.0
