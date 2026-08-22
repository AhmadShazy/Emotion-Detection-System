---
title: Humanoid Assistant Input Module
emoji: 🎛️
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
short_description: Multimodal emotion analysis API — text mode + contract sandbox
---

# Humanoid Assistant — Input Module

The input stage of a three-part conversational assistant. It reads a person's
emotional state and hands a structured summary to the LLM stage, which decides
what to say, and then to an avatar stage that delivers it.

This deployment runs in **text-only mode**: real analysis of typed text, plus a
sandbox of example payloads so the LLM side can be built and tested without
running any models.

## Endpoints

All endpoints except `/health` and `/docs` require an `X-API-Key` header.

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness and current mode |
| `GET /docs` | Interactive API documentation |
| `POST /analyze/text` | Analyse text, return the emotion payload |
| `GET /mock/scenarios` | Every example payload, with descriptions |
| `GET /mock/payload/{name}` | One example payload |
| `POST /mock/emit/{name}` | Push an example to the configured LLM endpoint |

## Quick start

```bash
curl -H "X-API-Key: YOUR_KEY" \
     -H "Content-Type: application/json" \
     -d '{"text": "I am really not happy with how this turned out."}' \
     https://THIS-SPACE.hf.space/analyze/text
```

Pass the `session_id` from the response back on the next request to keep the
conversation's emotional history — confidence builds as an emotion persists.

## The payload

Five keys, identical across every mode:

```jsonc
{
  "session_id": "sess-a1b2c3d4",
  "user_input":        { "text": "...", "timestamp": "...Z" },
  "emotion_analysis":  { "dominant_emotion": "angry", "confidence": 0.29,
                         "emotion_probabilities": { /* all 15 classes */ } },
  "tone_analysis":     { "tone": "hostile", "confidence": 0.82 },
  "conflict_analysis": { "detected": true, "type": "masked_anger",
                         "details": "Face appears happy but voice indicates anger." }
}
```

Two things that surprise people:

- **`dominant_emotion` is often not the highest entry in
  `emotion_probabilities`.** Always use `dominant_emotion`. Taking the argmax
  returns the masked *surface* emotion instead of the real one.
- **The first turn of a session cannot exceed 0.70 confidence**, because the
  emotional history starts empty. Low early numbers are expected.

Full field reference: `contract/CONTRACT.md` in the repository.

## Not available in this deployment

Voice, multimodal video and live streaming. Those currently require a
microphone, a webcam and a Windows-only face-analysis binary on the host.
Reworking them to accept media from the browser is in progress.
