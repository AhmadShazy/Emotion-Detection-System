class LLMAdapter:
    def __init__(self):
        # The 15 LLM-required emotional states
        self.all_emotions = [
            "happy", "sad", "angry", "surprised", "neutral",
            "empathetic", "concerned", "fear", "disgust",
            "shame", "guilt", "anxiety", "frustration",
            "joy", "calm"
        ]

    def map_emotion(self, base_emotion, text):
        """
        Interprets the base emotion into the expanded 15-class LLM required space.
        """
        base = base_emotion.lower().strip()
        text_lower = text.lower()

        if base == "sad":
            if "overwhelmed" in text_lower or "stress" in text_lower or "anxious" in text_lower:
                return "anxiety"
            if "sorry" in text_lower:
                return "guilt"
            return "sad"

        elif base == "angry":
            if "annoy" in text_lower or "ugh" in text_lower:
                return "frustration"
            if "gross" in text_lower or "ew" in text_lower:
                return "disgust"
            return "angry"

        elif base == "happy":
            if "great" in text_lower or "amazing" in text_lower or "love" in text_lower:
                return "joy"
            return "happy"

        elif base == "neutral":
            if "fine" in text_lower or "okay" in text_lower or "chill" in text_lower:
                return "calm"
            return "neutral"

        elif base == "fear":
            if "worry" in text_lower or "nervous" in text_lower:
                return "anxiety"
            return "fear"

        # If it happens to already be mapped or unrecognized, return it
        return base

    def build_probabilities(self, mapped_emotion, original_confidence):
        """
        Creates the full probability dictionary required by the LLM. 
        Assigns the original confidence to the mapped emotion and normalizes the rest.
        """
        probs = {mapped_emotion: original_confidence}
        
        # Add all missing emotions with a tiny baseline probability
        for emo in self.all_emotions:
            if emo not in probs:
                probs[emo] = 0.01
                
        # Normalize so they sum exactly to 1.0
        total = sum(probs.values())
        normalized_probs = {k: round(v / total, 2) for k, v in probs.items()}
        
        # Ensure the sum equals exactly 1.0 down to the rounding error
        max_key = max(normalized_probs, key=normalized_probs.get)
        others_sum = sum(v for k, v in normalized_probs.items() if k != max_key)
        normalized_probs[max_key] = round(max(0.01, 1.0 - others_sum), 2)
        
        return normalized_probs

    def analyze_tone(self, fusion_output, raw_inputs):
        """
        Separates 'how' they speak from 'what' they feel using the smoothed fusion engine 
        as an anchor to correct noisy raw acoustic models.
        """
        text = raw_inputs.get("text", "").lower()
        voice_raw = raw_inputs.get("voice_emotion", "neutral")
        voice = voice_raw.lower().strip() if voice_raw else "neutral"
        
        # The fusion engine is our main source of truth
        anchor = fusion_output.get("dominant_emotion", "neutral").lower()
        
        # Tone mappings
        if anchor in ["angry", "frustration", "disgust"]:
            if "annoy" in text or "ugh" in text:
                return "frustrated", 0.85
            return "hostile" if voice == "angry" else "tense", 0.82
            
        elif anchor in ["anxiety", "fear"]:
            return "panicked" if voice in ["fear", "angry"] else "nervous", 0.80
            
        elif anchor in ["sad", "guilt", "shame"]:
            return "somber" if voice == "sad" else "reflective", 0.75
            
        elif anchor in ["happy", "joy", "surprised"]:
            if "wow" in text or "amazing" in text:
                return "amazed", 0.85
            return "excited" if voice in ["angry", "happy"] else "cheerful", 0.80
            
        elif anchor in ["calm", "neutral", "empathetic"]:
            return "measured" if voice == "neutral" else "conversational", 0.75
            
        return "neutral", 0.70

    def process(self, fusion_output, raw_inputs, context):
        """
        The main pipeline to structure the final JSON output.
        """
        text = raw_inputs.get("text", "")
        
        # Step 1: Normalize fusion emotion
        base_emotion = fusion_output.get("dominant_emotion", "neutral").lower().strip()
        confidence = fusion_output.get("confidence", 0.0)
        
        # Step 2: Expand emotion space
        mapped_emotion = self.map_emotion(base_emotion, text)
        
        # Step 3: Build probabilities
        probs = self.build_probabilities(mapped_emotion, confidence)
        
        # Step 4: Analyze tone from voice and fusion context
        tone, tone_conf = self.analyze_tone(fusion_output, raw_inputs)
        
        # Step 5: Format final JSON exactly matching LLM schema requirements
        session_id = context.get("session_id", "sess-unknown")
        timestamp = raw_inputs.get("timestamp", "")
        conversation_history = context.get("conversation_history", [])
        
        # Build the exact contextual window required by the LLM
        conversation_context = {
            "window_size": 6,
            "turns": context.get("turns", [])
        }
        
        final_output = {
            "session_id": session_id,
            "user_input": {
                "text": text,
                "timestamp": timestamp
            },
            "emotion_analysis": {
                "dominant_emotion": mapped_emotion,
                "confidence": confidence,
                "emotion_probabilities": probs
            },
            "tone_analysis": {
                "tone": tone,
                "confidence": tone_conf
            },
            "context": {
                "conversation_history": conversation_history
            },
            "conversation_context": conversation_context
        }
        
        return final_output
