class AssistantResponseEngine:
    def __init__(self):
        """
        Synchronous engine that determines the assistant's behavioral persona based on emotion.
        """
        self.current_state = "Neutral"

    def react(self, emotion):
        """Adapts the tone of the assistant based on the fused emotion."""
        e = str(emotion).lower()
        
        if e == "angry":
            print("  ↳ [🤖 Assistant] Shifts to a CALM, DE-ESCALATING and ATTENTIVE tone.")
        elif e == "sad" or e == "fear":
            print("  ↳ [🤖 Assistant] Shifts to an EMPATHETIC, SOFT, and SUPPORTIVE tone.")
        elif e in ["happy", "surprised"]:
            print("  ↳ [🤖 Assistant] Shifts to an ENERGETIC, UPBEAT, and ENGAGED tone.")
        else:
            print("  ↳ [🤖 Assistant] Shifts to a REGULAR, HELPFUL, and NEUTRAL tone.")
