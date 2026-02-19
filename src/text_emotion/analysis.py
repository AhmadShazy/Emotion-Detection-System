import os
from transformers import pipeline

# Global variable to cache the pipeline so we don't reload it every time
_emotion_pipeline = None

def load_emotion_model():
    global _emotion_pipeline
    if _emotion_pipeline is None:
        print("[INFO] Loading Text Emotion Model (SamLowe/roberta-base-go_emotions)...")
        try:
            # top_k=None returns all scores. We can also use a specific number.
            # But we want to filter by threshold mostly.
            _emotion_pipeline = pipeline("text-classification", 
                                       model="SamLowe/roberta-base-go_emotions", 
                                       top_k=None)
            print("✅ Text Emotion Model loaded.")
        except Exception as e:
            print(f"❌ Error loading Text Emotion Model: {e}")
            _emotion_pipeline = None
    return _emotion_pipeline

def analyze_text_emotion(text, threshold=0.1):
    """
    Analyzes the emotion of the given text using a multi-label model.
    Returns a list of dicts [{'label': 'joy', 'score': 0.9}, ...] 
    filtering for scores > threshold.
    """
    if not text or not text.strip():
        return []

    pipe = load_emotion_model()
    if pipe is None:
        return []

    try:
        # Pipeline with top_k=None returns a list of dicts (all labels)
        results = pipe(text)
        
        # Determine format and extract
        # results might be [[{'label': 'joy', 'score': 0.9}, ...]] if input is a list or single string depending on version
        if isinstance(results, list) and len(results) > 0:
            if isinstance(results[0], list): 
                # [[...]] format
                predictions = results[0]
            else:
                # [...] format (unlikely with text-classification usually, but just in case)
                predictions = results

            # Filter by threshold and sort by score descending
            filtered_emotions = [
                {'label': p['label'], 'score': p['score']} 
                for p in predictions 
                if p['score'] > threshold
            ]
            
            # Sort just in case the pipeline didn't (it usually does by score if top_k is set, but top_k=None might vary)
            filtered_emotions.sort(key=lambda x: x['score'], reverse=True)
            
            return filtered_emotions
            
    except Exception as e:
        print(f"❌ Error analyzing text emotion: {e}")
        return []

if __name__ == "__main__":
    # Test
    sample_text = "I am so happy that this is working!"
    print(f"Testing with: '{sample_text}'")
    result = analyze_text_emotion(sample_text)
    print(f"Result: {result}")
