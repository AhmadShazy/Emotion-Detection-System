"""
src/text_emotion/analysis.py
=============================
Text emotion analysis using RoBERTa (go_emotions).
Model is owned by ModelRegistry — never loaded here directly.
"""

from src.core.model_registry import registry


def load_emotion_model():
    """
    Kept for backward compatibility with any code that calls it.
    Actual loading is done by registry.load_all() at startup.
    This is now a no-op.
    """
    pass


def analyze_text_emotion(text: str, threshold: float = 0.1) -> list:
    """
    Analyzes emotion of given text using RoBERTa go_emotions model.

    Args:
        text:      Raw input string to classify.
        threshold: Minimum score to include a label. Default 0.1.

    Returns:
        List of dicts sorted by score descending:
        [{"label": "joy", "score": 0.91}, {"label": "optimism", "score": 0.45}, ...]
        Returns [] if text is empty or model unavailable.
    """
    if not text or not text.strip():
        return []

    # Graceful degradation — if registry failed to load roberta,
    # return empty instead of crashing the whole request
    if not registry.is_available("roberta"):
        print("[TextEmotion] ⚠️  RoBERTa unavailable — returning empty.")
        return []

    try:
        pipe    = registry.get("roberta")
        results = pipe(text)

        # HuggingFace pipeline returns [[{...}]] for single string input
        if isinstance(results, list) and len(results) > 0:
            predictions = results[0] if isinstance(results[0], list) else results

            filtered = [
                {"label": p["label"], "score": p["score"]}
                for p in predictions
                if p["score"] > threshold
            ]
            filtered.sort(key=lambda x: x["score"], reverse=True)
            return filtered

    except Exception as e:
        print(f"[TextEmotion] ❌ Error: {e}")

    return []


if __name__ == "__main__":
    # Quick test — only works after registry.load_all() has run
    from src.core.model_registry import registry
    registry.load_all()
    result = analyze_text_emotion("I am so happy this is working!")
    print(result)