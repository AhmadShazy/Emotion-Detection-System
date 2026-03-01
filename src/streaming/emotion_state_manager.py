class EmotionStateManager:
    def __init__(self):
        """
        Decision-Level Fusion Engine.
        Synchronously fuses multimodal emotion estimates using weighted scoring.
        """
        # Fusion parameters
        self.weights = {
            "voice": 0.4,
            "face": 0.3,
            "text": 0.3
        }
        
        self.current_stable_emotion = "Neutral"

    def _align_emotion(self, emotion):
        """Align disparate vocabulary to a core subset: Happy, Angry, Sad, Neutral"""
        if not emotion:
            return None
        
        e = str(emotion).lower()
        if e in ["hap", "happy", "joy", "excitement", "amusement", "optimism", "admiration", "approval", "caring", "desire", "gratitude", "love", "pride", "relief"]: return "Happy"
        if e in ["ang", "angry", "anger", "frustration", "annoyance", "disapproval", "disgust"]: return "Angry"
        if e in ["sad", "sadness", "grief", "disappointment", "remorse", "grief"]: return "Sad"
        if e in ["surprised", "surprise", "realization"]: return "Surprised"
        if e in ["fear", "nervousness", "confusion"]: return "Fear"
        
        return "Neutral"

    def fuse(self, text_state, voice_state, face_state):
        """
        Takes in the current states of text, voice, and face and returns the fused emotion.
        Each state is expected to be a dict e.g. {"emotion": "Happy", "confidence": 0.9}
        """
        scores = {}
        total_active_weight = 0.0
        
        inputs = {
            "text": text_state,
            "voice": voice_state,
            "face": face_state
        }
        
        for src, data in inputs.items():
            if data and data.get("emotion"):
                aligned_emo = self._align_emotion(data["emotion"])
                
                score_contrib = self.weights[src] * data.get("confidence", 0.0)
                total_active_weight += self.weights[src]
                
                scores[aligned_emo] = scores.get(aligned_emo, 0.0) + score_contrib

        if not scores:
            return self.current_stable_emotion
            
        if total_active_weight > 0:
            for em in scores:
                scores[em] /= total_active_weight

        best_emotion = max(scores.items(), key=lambda x: x[1])[0]
        
        if best_emotion != self.current_stable_emotion:
            self.current_stable_emotion = best_emotion
            print(f"\n[🧠 FUSION] Overall Emotion changed to: {self.current_stable_emotion.upper()}")
            
        return self.current_stable_emotion
