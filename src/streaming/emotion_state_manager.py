from collections import deque

class EmotionStateManager:
    def __init__(self, memory_size=5):
        """
        Decision-Level Fusion Engine.
        Synchronously fuses multimodal emotion estimates using adaptive weighting, temporal memory, and conflict resolution.
        """
        # Base fusion parameters
        self.base_weights = {
            "voice": 0.4,
            "face": 0.3,
            "text": 0.3
        }
        
        # Calibration Multipliers
        self.calibration = {
            "voice": 0.9,
            "face": 0.8,
            "text": 1.0
        }
        
        self.current_stable_emotion = "neutral"
        self.emotion_memory = deque(["neutral"] * memory_size, maxlen=memory_size)
        
    def _align_emotion(self, emotion):
        """Align disparate vocabulary to a core subset: happy, angry, sad, fear, surprised, neutral"""
        if not emotion:
            return None
        
        e = str(emotion).lower()
        if e in ["hap", "happy", "joy", "excitement", "amusement", "optimism", "admiration", "approval", "caring", "desire", "gratitude", "love", "pride", "relief"]: return "happy"
        if e in ["ang", "angry", "anger", "frustration", "annoyance", "disapproval", "disgust"]: return "angry"
        if e in ["sad", "sadness", "grief", "disappointment", "remorse", "grief"]: return "sad"
        if e in ["surprised", "surprise", "realization"]: return "surprised"
        if e in ["fear", "nervousness", "confusion"]: return "fear"
        
        return "neutral"

    def detect_conflict(self, text_state, voice_state, face_state):
        conflict_flag = False
        conflict_type = "none"
        conflict_details = ""
        
        v_emo = self._align_emotion(voice_state.get("emotion")) if voice_state else None
        f_emo = self._align_emotion(face_state.get("emotion")) if face_state else None
        
        # Example Conflict Rules
        if f_emo == "happy" and v_emo == "angry":
            conflict_flag = True
            conflict_type = "masked_anger"
            conflict_details = "Face appears happy but voice indicates anger."
        elif f_emo == "neutral" and v_emo == "angry":
            conflict_flag = True
            conflict_type = "suppressed_frustration"
            conflict_details = "Face is neutral but voice is angry."
        elif f_emo == "sad" and v_emo == "neutral":
            conflict_flag = True
            conflict_type = "internal_sadness"
            conflict_details = "Face shows sadness while voice tries to remain neutral."
        elif f_emo == "happy" and v_emo == "sad":
            conflict_flag = True
            conflict_type = "masked_sadness"
            conflict_details = "Face appears happy but voice indicates sadness."
            
        return conflict_flag, conflict_type, conflict_details

    def calculate_dynamics(self):
        emotions = list(self.emotion_memory)
        if not emotions:
            return "stable", 1.0, 0.0
            
        transitions = sum(1 for i in range(1, len(emotions)) if emotions[i] != emotions[i-1])
        volatility = transitions / max(1, len(emotions) - 1)
        stability = 1.0 - volatility
        
        trend = "stable"
        if len(emotions) >= 3:
            recent = emotions[-2:]
            past = emotions[:-2]
            if "angry" in recent and "angry" not in past:
                trend = "increasing_anger"
            elif "sad" in recent and "sad" not in past:
                trend = "increasing_sadness"
            elif "happy" in recent and "happy" not in past:
                trend = "increasing_happiness"
                
        return trend, stability, volatility

    def fuse(self, text_state, voice_state, face_state):
        """
        Fuses modalities adaptively and returns a rich JSON map.
        """
        scores = {}
        total_active_weight = 0.0
        
        # Extract default reliabilities if absent
        t_rel = text_state.get("reliability", 1.0) if text_state else 0.0
        v_rel = voice_state.get("reliability", 1.0) if voice_state else 0.0
        f_rel = face_state.get("reliability", 1.0) if face_state else 0.0
        
        # 1. Calibrate Confidences
        t_conf = text_state.get("confidence", 0.0) * self.calibration["text"] if text_state else 0.0
        v_conf = voice_state.get("confidence", 0.0) * self.calibration["voice"] if voice_state else 0.0
        f_conf = face_state.get("confidence", 0.0) * self.calibration["face"] if face_state else 0.0

        # 2. Conflict Resolver
        conflict_detected, conflict_type, conflict_details = self.detect_conflict(text_state, voice_state, face_state)
        
        # Adjust weights based on specific conflicts (e.g., trust Voice more in masked emotions)
        v_weight_mod = 1.0
        f_weight_mod = 1.0
        
        if conflict_detected:
            if conflict_type in ["masked_anger", "suppressed_frustration", "masked_sadness"]:
                v_weight_mod = 1.5
                f_weight_mod = 0.5
            elif conflict_type == "internal_sadness":
                 f_weight_mod = 1.5
             
        modality_contributions = {"voice": 0.0, "face": 0.0, "text": 0.0}
             
        # 3. Compute Adaptive Scores
        if text_state and text_state.get("emotion"):
            emo = self._align_emotion(text_state["emotion"])
            weight = self.base_weights["text"] * t_rel
            contrib = weight * t_conf
            scores[emo] = scores.get(emo, 0.0) + contrib
            total_active_weight += weight
            modality_contributions["text"] = contrib
            
        if voice_state and voice_state.get("emotion"):
            emo = self._align_emotion(voice_state["emotion"])
            weight = self.base_weights["voice"] * v_rel * v_weight_mod
            contrib = weight * v_conf
            scores[emo] = scores.get(emo, 0.0) + contrib
            total_active_weight += weight
            modality_contributions["voice"] = contrib
            
        if face_state and face_state.get("emotion"):
            emo = self._align_emotion(face_state["emotion"])
            weight = self.base_weights["face"] * f_rel * f_weight_mod
            contrib = weight * f_conf
            scores[emo] = scores.get(emo, 0.0) + contrib
            total_active_weight += weight
            modality_contributions["face"] = contrib

        raw_probabilities = {}
        target_emotion = "neutral"
        final_confidence = 0.0
        
        if total_active_weight > 0:
            # Normalize contributions to percentages
            total_contrib = sum(modality_contributions.values())
            if total_contrib > 0:
                for k in modality_contributions:
                    modality_contributions[k] = round(modality_contributions[k] / total_contrib, 2)
                    
            for em in scores:
                raw_probabilities[em] = scores[em] / total_active_weight
            
            target_emotion = max(raw_probabilities.items(), key=lambda x: x[1])[0]
            final_confidence = raw_probabilities[target_emotion]

        # 4. Temporal Smoothing
        alpha = 0.7
        historical_score = 0.0
        
        if self.emotion_memory:
            # Simple historical trend: frequency of the target emotion in memory
            history_count = sum(1 for e in self.emotion_memory if e == target_emotion)
            historical_score = history_count / len(self.emotion_memory)
            
        smoothed_confidence = (alpha * final_confidence) + ((1.0 - alpha) * historical_score)
        
        # 5. Update State and Memory
        self.current_stable_emotion = target_emotion
        self.emotion_memory.append(target_emotion)
        
        # 6. Emotion Dynamics
        trend, stability, volatility = self.calculate_dynamics()
        
        # Output probabilities padded to cover all core emotions
        probs_output = {e: 0.01 for e in ["happy", "angry", "sad", "surprised", "fear", "neutral"]}
        for k, v in raw_probabilities.items():
            probs_output[k] = v
        # Ensure sum approximates to 1
        max_k = target_emotion
        others_sum = sum(v for k, v in probs_output.items() if k != max_k)
        probs_output[max_k] = max(0.01, 1.0 - others_sum)

        result = {
            "dominant_emotion": self.current_stable_emotion,
            "confidence": round(float(smoothed_confidence), 2),
            "emotion_probabilities": {k: round(float(v), 2) for k, v in probs_output.items()},
            "emotion_dynamics": {
                "trend": str(trend),
                "stability": round(float(stability), 2),
                "volatility": round(float(volatility), 2)
            },
            "conflict_analysis": {
                "detected": bool(conflict_detected),
                "type": str(conflict_type),
                "details": str(conflict_details)
            },
            "modality_contributions": {k: float(v) for k, v in modality_contributions.items()},
            "reliability": {
                "voice": round(float(v_rel), 2),
                "face": round(float(f_rel), 2),
                "text": round(float(t_rel), 2)
            }
        }

        print(f"\n[FUSION] Overall Emotion: {self.current_stable_emotion.upper()} "
              f"(Conf: {smoothed_confidence:.2f}, Conflict: {conflict_type})")
              
        return result
