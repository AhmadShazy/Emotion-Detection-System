from collections import deque


class EmotionStateManager:
    def __init__(self, memory_size=5):
        """
        Decision-Level Fusion Engine.
        Fuses multimodal emotion estimates using adaptive weighting,
        temporal memory, and conflict resolution.
        """
        self.base_weights = {
            "voice": 0.4,
            "face":  0.3,
            "text":  0.3,
        }
        # How much a modality's self-reported confidence is worth. This is a
        # TRUST weight, so it belongs on the weight and not on the confidence
        # value — see fuse() for why that distinction decides whether agreement
        # raises confidence or lowers it.
        self.calibration = {
            "voice": 0.9,
            "face":  0.8,
            "text":  1.0,
        }
        self.current_stable_emotion = "neutral"
        self.memory_size            = memory_size

        # Starts EMPTY, not pre-filled with neutrals.
        #
        # Pre-seeding ["neutral"] * 5 gave the neutral class a historical score
        # of 1.0 on the very first turn while every other emotion got 0.0. A
        # first-turn neutral therefore scored 0.97 where a first-turn sadness
        # scored 0.69 — so the system was most confident about the least
        # actionable reading it can produce, and CONTRACT.md's bolded promise
        # that "the first turn of any session cannot exceed 0.70" was false for
        # exactly one emotion. Starting empty makes that promise true for all of
        # them.
        self.emotion_memory = deque(maxlen=memory_size)

    def _align_emotion(self, emotion):
        """
        Maps any emotion label to one of 7 core states:
        happy, angry, sad, fear, surprised, disgust, neutral.

        Covers all 28 RoBERTa go_emotions labels, SpeechBrain
        IEMOCAP 4-class labels, and OpenFace AU classifier labels.

        Previously only covered ~15 labels — causing RoBERTa outputs
        like 'remorse', 'disappointment', 'grief' to fall through
        to 'neutral' incorrectly. Now fully mapped.
        """
        if not emotion:
            return None

        e = str(emotion).lower().strip()

        # ── Happy / Positive ──────────────────────────────────────────────────
        if e in [
            "hap", "happy", "joy", "excitement", "amusement",
            "optimism", "admiration", "approval", "caring",
            "desire", "gratitude", "love", "pride", "relief",
            "curiosity", "surprise",
        ]:
            return "happy"

        # ── Angry ─────────────────────────────────────────────────────────────
        if e in [
            "ang", "angry", "anger", "annoyance", "disapproval",
            "frustration",
        ]:
            return "angry"

        # ── Disgust ───────────────────────────────────────────────────────────
        if e in ["disgust"]:
            return "disgust"

        # ── Sad ───────────────────────────────────────────────────────────────
        # Critical fix — remorse/disappointment/grief were previously
        # falling through to neutral causing wrong results like
        # "My car broke down" → happy
        if e in [
            "sad", "sadness", "grief", "disappointment",
            "remorse", "embarrassment", "realization",
        ]:
            return "sad"

        # ── Fear / Anxiety ────────────────────────────────────────────────────
        if e in [
            "fear", "nervousness", "confusion", "anxiety",
        ]:
            return "fear"

        # ── Surprised ─────────────────────────────────────────────────────────
        if e in ["surprised", "surprise"]:
            return "surprised"

        # ── Neutral ───────────────────────────────────────────────────────────
        if e in ["neutral", "calm"]:
            return "neutral"

        # Unrecognized — log so we can add it
        print(f"[EmotionStateManager] ⚠️  Unmapped label: '{e}' → neutral")
        return "neutral"

    def detect_conflict(self, text_state, voice_state, face_state):
        conflict_flag    = False
        conflict_type    = "none"
        conflict_details = ""

        v_emo = self._align_emotion(voice_state["emotion"]) if voice_state else None
        f_emo = self._align_emotion(face_state["emotion"])  if face_state  else None

        if f_emo == "happy" and v_emo == "angry":
            conflict_flag    = True
            conflict_type    = "masked_anger"
            conflict_details = "Face appears happy but voice indicates anger."
        elif f_emo == "neutral" and v_emo == "angry":
            conflict_flag    = True
            conflict_type    = "suppressed_frustration"
            conflict_details = "Face is neutral but voice is angry."
        elif f_emo == "sad" and v_emo == "neutral":
            conflict_flag    = True
            conflict_type    = "internal_sadness"
            conflict_details = "Face shows sadness while voice tries to remain neutral."
        elif f_emo == "happy" and v_emo == "sad":
            conflict_flag    = True
            conflict_type    = "masked_sadness"
            conflict_details = "Face appears happy but voice indicates sadness."

        return conflict_flag, conflict_type, conflict_details

    def calculate_dynamics(self):
        emotions = list(self.emotion_memory)
        if not emotions:
            return "stable", 1.0, 0.0

        transitions = sum(
            1 for i in range(1, len(emotions))
            if emotions[i] != emotions[i - 1]
        )
        volatility = transitions / max(1, len(emotions) - 1)
        stability  = 1.0 - volatility

        trend = "stable"
        if len(emotions) >= 3:
            recent = emotions[-2:]
            past   = emotions[:-2]
            if "angry" in recent   and "angry"   not in past:
                trend = "increasing_anger"
            elif "sad" in recent   and "sad"     not in past:
                trend = "increasing_sadness"
            elif "happy" in recent and "happy"   not in past:
                trend = "increasing_happiness"

        return trend, stability, volatility

    def fuse(self, text_state, voice_state, face_state):
        """
        Fuses modalities adaptively and returns a rich result map.
        """
        scores              = {}
        total_active_weight = 0.0

        t_rel  = text_state["reliability"]  if text_state  else 0.0
        v_rel  = voice_state["reliability"] if voice_state else 0.0
        f_rel  = face_state["reliability"]  if face_state  else 0.0

        # Confidences stay RAW here. Calibration is applied to the weights
        # below instead, so that it divides out of the final ratio.
        t_conf = text_state["confidence"]  if text_state  else 0.0
        v_conf = voice_state["confidence"] if voice_state else 0.0
        f_conf = face_state["confidence"]  if face_state  else 0.0

        conflict_detected, conflict_type, conflict_details = (
            self.detect_conflict(text_state, voice_state, face_state)
        )

        v_weight_mod = 1.0
        f_weight_mod = 1.0

        if conflict_detected:
            if conflict_type in [
                "masked_anger", "suppressed_frustration", "masked_sadness"
            ]:
                v_weight_mod = 1.5
                f_weight_mod = 0.5
            elif conflict_type == "internal_sadness":
                f_weight_mod = 1.5

        modality_contributions = {"voice": 0.0, "face": 0.0, "text": 0.0}

        # Calibration multiplies the WEIGHT, not the confidence.
        #
        # It used to multiply the confidence instead, which put it in the
        # numerator of the ratio below while the denominator kept the
        # uncalibrated weight. Every modality trusted at less than 1.0 — voice
        # at 0.9, face at 0.8 — therefore dragged the final number down just by
        # being present, even when it agreed perfectly. Measured on identical
        # readings at 0.90 confidence: text alone 0.63, text+voice 0.59,
        # text+voice+face 0.57. Adding agreeing evidence made the system less
        # sure, which is backwards, and it pushed the conflict cases — the ones
        # that need all three modalities to exist at all — under the 0.35 mark
        # CONTRACT.md tells the LLM to ignore.
        #
        # With calibration on the weight, the ratio is exactly
        #     (mean confidence of the modalities that agreed)
        #       x (share of total weight that agreed)
        # which is what the original design was reaching for.

        if text_state and text_state["emotion"]:
            emo    = self._align_emotion(text_state["emotion"])
            weight = (self.base_weights["text"] * t_rel
                      * self.calibration["text"])
            contrib = weight * t_conf
            scores[emo]                     = scores.get(emo, 0.0) + contrib
            total_active_weight            += weight
            modality_contributions["text"]  = contrib

        if voice_state and voice_state["emotion"]:
            emo    = self._align_emotion(voice_state["emotion"])
            weight = (self.base_weights["voice"] * v_rel * v_weight_mod
                      * self.calibration["voice"])
            contrib = weight * v_conf
            scores[emo]                      = scores.get(emo, 0.0) + contrib
            total_active_weight             += weight
            modality_contributions["voice"]  = contrib

        if face_state and face_state["emotion"]:
            emo    = self._align_emotion(face_state["emotion"])
            weight = (self.base_weights["face"] * f_rel * f_weight_mod
                      * self.calibration["face"])
            contrib = weight * f_conf
            scores[emo]                     = scores.get(emo, 0.0) + contrib
            total_active_weight            += weight
            modality_contributions["face"]  = contrib

        raw_probabilities = {}
        target_emotion    = "neutral"
        final_confidence  = 0.0

        if total_active_weight > 0:
            total_contrib = sum(modality_contributions.values())
            if total_contrib > 0:
                for k in modality_contributions:
                    modality_contributions[k] = round(
                        modality_contributions[k] / total_contrib, 2
                    )

            for em in scores:
                raw_probabilities[em] = scores[em] / total_active_weight

            target_emotion   = max(
                raw_probabilities.items(), key=lambda x: x[1]
            )[0]
            final_confidence = raw_probabilities[target_emotion]

        # ── Temporal smoothing ────────────────────────────────────────────────
        alpha            = 0.7
        historical_score = 0.0

        if self.emotion_memory:
            history_count = sum(
                1 for e in self.emotion_memory if e == target_emotion
            )
            # Divided by the FULL window, not by how much of it has filled up.
            # Dividing by len() would score one remembered turn as a unanimous
            # history and jump confidence from 0.63 to 0.93 between turns one
            # and two. Against the full window the same run climbs
            # 0.63 -> 0.69 -> 0.75 -> 0.81, which is the gradual rise
            # CONTRACT.md describes.
            historical_score = history_count / self.memory_size

        smoothed_confidence = (
            (alpha * final_confidence) + ((1.0 - alpha) * historical_score)
        )

        self.current_stable_emotion = target_emotion
        self.emotion_memory.append(target_emotion)

        trend, stability, volatility = self.calculate_dynamics()

        # ── Build full probability output — all 7 core emotions ───────────────
        probs_output = {
            e: 0.01
            for e in [
                "happy", "angry", "sad", "surprised",
                "fear", "disgust", "neutral",
            ]
        }
        for k, v in raw_probabilities.items():
            probs_output[k] = v

        others_sum = sum(
            v for k, v in probs_output.items() if k != target_emotion
        )
        probs_output[target_emotion] = max(0.01, 1.0 - others_sum)

        result = {
            "dominant_emotion": self.current_stable_emotion,
            "confidence":       round(float(smoothed_confidence), 2),
            "emotion_probabilities": {
                k: round(float(v), 2) for k, v in probs_output.items()
            },
            "emotion_dynamics": {
                "trend":      str(trend),
                "stability":  round(float(stability), 2),
                "volatility": round(float(volatility), 2),
            },
            "conflict_analysis": {
                "detected": bool(conflict_detected),
                "type":     str(conflict_type),
                "details":  str(conflict_details),
            },
            "modality_contributions": {
                k: float(v) for k, v in modality_contributions.items()
            },
            "reliability": {
                "voice": round(float(v_rel), 2),
                "face":  round(float(f_rel), 2),
                "text":  round(float(t_rel), 2),
            },
        }

        print(
            f"\n[FUSION] Overall Emotion: {self.current_stable_emotion.upper()} "
            f"(Conf: {smoothed_confidence:.2f}, Conflict: {conflict_type})"
        )

        return result