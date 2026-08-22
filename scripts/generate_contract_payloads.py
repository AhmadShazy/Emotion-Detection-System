"""
scripts/generate_contract_payloads.py
=====================================
Regenerates the frozen example payloads in contract/payloads/.

These examples are produced by running the REAL fusion engine and LLM adapter,
never hand-written, so they cannot drift from what the API actually emits.
They serve two purposes:

  1. routers/mock.py serves them, so the LLM team can build against realistic
     payloads without running any ML models.
  2. tests/test_contract.py validates them, so a change to the payload shape
     fails a test instead of silently breaking the integration.

Run after any deliberate change to the payload:
    python scripts/generate_contract_payloads.py

Requires the models, because text states are produced by really running RoBERTa
rather than being invented.
"""

import os
import sys
import json
import datetime

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.core.console import enable_utf8_console
enable_utf8_console()

OUT_DIR = os.path.join(PROJECT_ROOT, "contract", "payloads")


# ── Helpers to build modality states in the exact shape the workers emit ──────

def voice_state(emotion: str, confidence: float, reliability: float) -> dict:
    """Matches what interactive_modes.process_voice_pipeline() produces."""
    return {
        "source":          "voice",
        "emotion":         emotion,
        "confidence":      confidence,
        "average_emotion": emotion,
        "peak_emotion":    emotion,
        "reliability":     reliability,
    }


def face_state(emotion: str, confidence: float,
               reliability: float, instability: float = 0.1) -> dict:
    """Matches what faceexpression.classifier.analyze_openface_csv() produces."""
    return {
        "source":      "face",
        "emotion":     emotion,
        "confidence":  confidence,
        "reliability": reliability,
        "instability": instability,
    }


# ── The scenarios ─────────────────────────────────────────────────────────────
# Chosen to span the range the LLM team must handle: single-signal, agreeing
# multi-signal, each of the four conflict types, disagreement with low
# confidence, and the silence case where there is no text at all.

SCENARIOS = [
    {
        "name": "text_only_neutral",
        "why":  "Plainest case. Text mode, nothing notable in the message.",
        "text": "Okay, that works for me.",
        "voice": None, "face": None,
    },
    {
        "name": "text_only_joy",
        "why":  "Strong positive from text alone. Note dominant_emotion refines to 'joy'.",
        "text": "This is absolutely wonderful news, I love it!",
        "voice": None, "face": None,
    },
    {
        "name": "text_only_sadness",
        "why":  "Negative from text alone. Guards the label-mapping regression where "
                "'my car broke down' used to come back as happy.",
        "text": "My car broke down again and I feel terrible about it.",
        "voice": None, "face": None,
    },
    {
        "name": "voice_and_text_agree",
        "why":  "Voice mode. Transcript and vocal tone point the same way, so confidence is higher.",
        "text": "I am really not happy with how this turned out.",
        "voice": voice_state("Angry", 0.78, 0.93), "face": None,
    },
    {
        "name": "conflict_masked_anger",
        "why":  "THE IMPORTANT ONE. Smiling face over an angry voice. The user says they are "
                "fine and they are not. Fusion re-weights toward the voice.",
        "text": "It is fine, really.",
        "voice": voice_state("Angry", 0.81, 0.95),
        "face":  face_state("Happy", 0.70, 1.0),
    },
    {
        "name": "conflict_suppressed_frustration",
        "why":  "Flat face, angry voice — someone holding a professional expression.",
        "text": "Sure, I can redo it.",
        "voice": voice_state("Angry", 0.74, 0.9),
        "face":  face_state("Neutral", 0.82, 1.0),
    },
    {
        "name": "conflict_internal_sadness",
        "why":  "Sad face, level voice. The one conflict type that trusts the FACE more.",
        "text": "I am okay. Just tired.",
        "voice": voice_state("Neutral", 0.66, 0.8),
        "face":  face_state("Sad", 0.71, 1.0),
    },
    {
        "name": "low_confidence_disagreement",
        "why":  "All three signals disagree and none is confident. The LLM should treat "
                "a payload like this as weak evidence, not a firm read.",
        "text": "I guess so.",
        "voice": voice_state("Happy", 0.41, 0.45),
        "face":  face_state("Sad", 0.38, 0.5, instability=0.6),
    },
    {
        "name": "no_speech_detected",
        "why":  "Silence, or a transcript filtered as a Whisper hallucination. "
                "user_input.text is an EMPTY STRING — the LLM must handle this.",
        "text": "",
        "voice": None, "face": None,
    },
]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    from src.core.model_registry import registry
    registry.load_all()

    from src.interactive_modes import process_text_emotion
    from src.streaming.unified_pipeline import process_and_print_unified_json

    index = []

    for spec in SCENARIOS:
        name = spec["name"]

        # Real RoBERTa pass, so the text state is never invented.
        text_state = process_text_emotion(spec["text"]) if spec["text"] else None

        payload = process_and_print_unified_json(
            text_state=text_state,
            voice_state=spec["voice"],
            face_state=spec["face"],
            raw_text=spec["text"],
            voice_emo_raw=(spec["voice"] or {}).get("emotion") or "neutral",
            face_emo_raw=(spec["face"] or {}).get("emotion") or "neutral",
            # Each scenario gets its own session so smoothing history never
            # leaks between examples.
            session_id=f"sess-mock-{name[:8]}",
        )

        path = os.path.join(OUT_DIR, f"{name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
            f.write("\n")

        index.append({
            "name":             name,
            "description":      spec["why"],
            "signals":          [s for s, present in
                                 (("text", bool(spec["text"])),
                                  ("voice", spec["voice"] is not None),
                                  ("face",  spec["face"] is not None)) if present],
            "dominant_emotion": payload["emotion_analysis"]["dominant_emotion"],
            "confidence":       payload["emotion_analysis"]["confidence"],
            "tone":             payload["tone_analysis"]["tone"],
        })

        print(f"  wrote {name}.json")

    index_doc = {
        "generated_by": "scripts/generate_contract_payloads.py",
        "generated_at": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "note": ("Produced by running the real fusion engine. Do not hand-edit — "
                 "re-run the generator instead."),
        "scenarios": index,
    }
    with open(os.path.join(OUT_DIR, "index.json"), "w", encoding="utf-8") as f:
        json.dump(index_doc, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"\n{len(index)} payloads written to {OUT_DIR}")


if __name__ == "__main__":
    main()
