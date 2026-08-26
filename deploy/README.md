# Deploying the input module

The goal here is narrow and worth stating plainly: **get a URL your LLM
teammate can call.** That is all this deployment does.

It runs in **text-only mode** — `/analyze/text` plus the `/mock/*` contract
sandbox. Voice, multimodal and live-stream are *not* deployable yet: they
currently require the server to own a microphone and a webcam, and a
Windows-only OpenFace binary. Fixing that is Phases 3 and 4.

---

## What you get

| Endpoint | Needs models? | Purpose |
|---|---|---|
| `GET /health` | no | Liveness + which mode is running |
| `GET /mock/scenarios` | no | All 9 example payloads with descriptions |
| `GET /mock/payload/{name}` | no | One payload, real shape |
| `POST /mock/emit/{name}` | no | Push a payload to `LLM_ENDPOINT_URL` |
| `POST /analyze/text` | RoBERTa | Real analysis of real text |
| `GET /docs` | no | Interactive API documentation |

Your teammate can build her entire integration against `/mock/*` before you
ever get the full pipeline hosted.

---

## Option A — Hugging Face Spaces (recommended, free)

Best fit here: it is built for ML models, the free tier has enough RAM, and
model weights cache properly. Downside: free Spaces are publicly discoverable,
so the API key is doing real work — which is why we rotated it.

**1. Create the Space**

At <https://huggingface.co/new-space>: pick a name, choose **Docker** → **Blank**,
and set visibility. Public is fine — the API key protects the endpoints.

**2. Add the Space config**

Hugging Face reads its settings from YAML at the top of `README.md` *in the
Space repo*. Copy `deploy/huggingface-README.md` from this repo to `README.md`
in the Space.

**3. Set the secret**

In the Space: **Settings → Variables and secrets → New secret**

| Name | Value |
|---|---|
| `API_KEYS` | your key from `.env` (`cat .env` to read it) |

Add `LLM_ENDPOINT_URL` too once your teammate's endpoint exists.

> Use **Secret**, not **Variable**. Variables are visible in the UI.

**4. Push**

```bash
git remote add space https://huggingface.co/spaces/<user>/<space-name>
git push space rebuild/phase-0-foundation:main
```

First build takes 5–10 minutes, mostly downloading torch.

**5. Verify**

```bash
curl https://<user>-<space>.hf.space/health
curl -H "X-API-Key: YOUR_KEY" https://<user>-<space>.hf.space/mock/scenarios
```

`/health` should return `"mode": "text_only"`. If `/mock/scenarios` returns
403, the secret name is wrong or wasn't saved.

---

## Option B — Railway / Render / Fly.io

Any container host works — the `Dockerfile` is portable and reads `$PORT`.

1. Point the platform at this repo, branch `rebuild/phase-0-foundation`.
2. It will detect the `Dockerfile` automatically.
3. Set `API_KEYS` as an environment variable / secret.
4. Deploy.

⚠️ **Render's free tier is 512 MB RAM**, which is not enough for PyTorch plus
RoBERTa. You need a paid instance there. Railway and Fly are usage-based and
generally fine.

---

## Option C — Mock-only micro-deployment

If all you need *right now* is unblocking your teammate, note that
`routers/mock.py` loads **no models at all**. A deployment serving only
`/mock/*` needs no PyTorch and no RoBERTa — it fits any free tier trivially and
starts in seconds.

Not wired up as a separate build today, but worth remembering if hosting turns
out to be slow or expensive.

---

## Before you deploy

- [ ] `API_KEYS` set as a **secret** on the platform, never committed —
      leaving it unset disables authentication entirely
- [ ] `TEXT_ONLY_MODE` left at `true` (the image already defaults it)
- [ ] Teammate has the key, sent over something private — not the repo, not a
      public channel

## After you deploy

- [ ] `/health` returns `"mode": "text_only"`
- [ ] `/mock/scenarios` returns 9 scenarios **with** the key
- [ ] `/mock/scenarios` returns **403** without it — if it returns 200, the key
      was not picked up and the API is open
- [ ] Send your teammate the URL and `contract/CONTRACT.md`

---

## Notes

**Image size** is around 2–3 GB, dominated by PyTorch. That is normal for an ML
container and why the CPU-only torch index matters — the CUDA build would add
roughly another 2 GB for no benefit.

**Cold starts.** RoBERTa is baked into the image at build time, so a restart
does not re-download it. Expect roughly 10–20 seconds to first response.

**Trimming dependencies.** `requirements-text.txt` was derived by testing, not
guessing. Removing anything from it makes RoBERTa fail to load — and the failure
is *silent*: the server still starts, `/health` still returns 200, and every
analysis comes back `neutral`. Re-test if you change it.

**Old key in git history.** The previously-committed key still exists in earlier
commits. That is harmless now that it has been rotated, since the old value no
longer opens anything. Do not put the new one anywhere tracked.
