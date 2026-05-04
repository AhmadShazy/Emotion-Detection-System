class LLMAdapter:
    def __init__(self):
        # The 15 LLM-required emotional states
        self.all_emotions = [
            "happy", "sad", "angry", "surprised", "neutral",
            "empathetic", "concerned", "fear", "disgust",
            "shame", "guilt", "anxiety", "frustration",
            "joy", "calm",
        ]

    def map_emotion(self, base_emotion: str, text: str) -> str:
        """
        Refines the fused 7-class emotion into the full 15-class LLM space.
        base_emotion from fusion is the primary source of truth.
        Text content is secondary signal for disambiguation only.
        """
        base       = base_emotion.lower().strip()
        text_lower = text.lower()

        if base == "sad":
            if any(w in text_lower for w in
                   ["overwhelmed", "stress", "anxious", "worried"]):
                return "anxiety"
            if any(w in text_lower for w in
                   ["sorry", "regret", "my fault"]):
                return "guilt"
            if any(w in text_lower for w in
                   ["ashamed", "embarrassed", "humiliated"]):
                return "shame"
            return "sad"

        elif base == "angry":
            if any(w in text_lower for w in
                   ["annoy", "ugh", "frustrat", "irritat"]):
                return "frustration"
            if any(w in text_lower for w in
                   ["gross", "ew", "disgusting"]):
                return "disgust"
            return "angry"

        elif base == "disgust":
            return "disgust"

        elif base == "happy":
            if any(w in text_lower for w in
                   ["great", "amazing", "love", "wonderful", "fantastic"]):
                return "joy"
            return "happy"

        elif base == "neutral":
            if any(w in text_lower for w in
                   ["fine", "okay", "alright", "chill"]):
                return "calm"
            if any(w in text_lower for w in
                   ["hope", "wish", "care", "worried about"]):
                return "concerned"
            return "neutral"

        elif base == "fear":
            if any(w in text_lower for w in
                   ["worry", "nervous", "anxious", "panic"]):
                return "anxiety"
            return "fear"

        elif base == "surprised":
            return "surprised"

        if base in self.all_emotions:
            return base

        return "neutral"

    def build_probabilities(
        self,
        mapped_emotion: str,
        original_confidence: float,
        emotion_probs: dict = None,
    ) -> dict:
        """
        Creates the full 15-class probability dict for the LLM.

        Uses emotion_probs from fusion engine as base distribution
        instead of flat 0.01 for everything — prevents low-confidence
        emotions from appearing artificially high in the output.
        """
        probs = {}

        # Map 7-class fusion probs to 15-class space as base
        seven_to_fifteen = {
            "happy":     "happy",
            "sad":       "sad",
            "angry":     "angry",
            "surprised": "surprised",
            "fear":      "fear",
            "disgust":   "disgust",
            "neutral":   "neutral",
        }

        if emotion_probs:
            for seven_key, fifteen_key in seven_to_fifteen.items():
                if seven_key in emotion_probs:
                    probs[fifteen_key] = emotion_probs[seven_key]

        # Fill missing 15-class emotions with small baseline
        for emo in self.all_emotions:
            if emo not in probs:
                probs[emo] = 0.01

        # Set mapped emotion to actual confidence
        probs[mapped_emotion] = original_confidence

        # Normalize so sum = 1.0
        total      = sum(probs.values())
        normalized = {k: round(v / total, 2) for k, v in probs.items()}

        # Fix rounding error on dominant key
        max_key    = max(normalized, key=normalized.get)
        others_sum = sum(v for k, v in normalized.items() if k != max_key)
        normalized[max_key] = round(max(0.01, 1.0 - others_sum), 2)

        return normalized

    def analyze_tone(self, fusion_output: dict, raw_inputs: dict) -> tuple:
        """
        Determines HOW the person speaks (tone) vs WHAT they feel (emotion).
        Uses smoothed fusion output as anchor to correct noisy raw signals.
        """
        text      = raw_inputs.get("text", "").lower()
        voice_raw = raw_inputs.get("voice_emotion", "neutral")
        voice     = voice_raw.lower().strip() if voice_raw else "neutral"
        anchor    = fusion_output.get("dominant_emotion", "neutral").lower()

        if anchor in ["angry", "frustration", "disgust"]:
            if any(w in text for w in ["annoy", "ugh", "frustrat"]):
                return "frustrated", 0.85
            return "hostile" if voice == "angry" else "tense", 0.82

        elif anchor in ["anxiety", "fear"]:
            return "panicked" if voice in ["fear", "angry"] else "nervous", 0.80

        elif anchor in ["sad", "guilt", "shame"]:
            return "somber" if voice == "sad" else "reflective", 0.75

        elif anchor in ["happy", "joy", "surprised"]:
            if any(w in text for w in ["wow", "amazing", "incredible"]):
                return "amazed", 0.85
            return "excited" if voice in ["angry", "happy"] else "cheerful", 0.80

        elif anchor in ["calm", "neutral", "empathetic", "concerned"]:
            return "measured" if voice == "neutral" else "conversational", 0.75

        return "neutral", 0.70

    def process(
        self,
        fusion_output: dict,
        raw_inputs:    dict,
        context:       dict,
    ) -> dict:
        """
        Main pipeline — structures the final JSON payload.
        Passes emotion_probs to build_probabilities() so the
        probability distribution reflects actual fusion scores.
        """
        text = raw_inputs.get("text", "")

        # Step 1: Get fused emotion and confidence
        base_emotion  = fusion_output.get(
            "dominant_emotion", "neutral"
        ).lower().strip()
        confidence    = fusion_output.get("confidence", 0.0)

        # Step 2: Get raw probability distribution from fusion engine
        emotion_probs = fusion_output.get("emotion_probabilities", {})

        # Step 3: Expand to 15-class space
        mapped_emotion = self.map_emotion(base_emotion, text)

        # Step 4: Build probabilities using fusion distribution as base
        probs = self.build_probabilities(
            mapped_emotion, confidence, emotion_probs
        )

        # Step 5: Analyze tone
        tone, tone_conf = self.analyze_tone(fusion_output, raw_inputs)

        # Step 6: Build final payload
        session_id           = context.get("session_id", "sess-unknown")
        timestamp            = raw_inputs.get("timestamp", "")
        conversation_history = context.get("conversation_history", [])
        conversation_context = {
            "window_size": 6,
            "turns":       context.get("turns", []),
        }

        return {
            "session_id": session_id,
            "user_input": {
                "text":      text,
                "timestamp": timestamp,
            },
            "emotion_analysis": {
                "dominant_emotion":      mapped_emotion,
                "confidence":            confidence,
                "emotion_probabilities": probs,
            },
            "tone_analysis": {
                "tone":       tone,
                "confidence": tone_conf,
            },
            "context": {
                "conversation_history": conversation_history,
            },
            "conversation_context": conversation_context,
        }