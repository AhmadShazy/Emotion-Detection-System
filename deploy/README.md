# Deploying the input module

The goal is narrow and worth stating plainly: **get a URL your LLM teammate can
call**, running the full system — text, voice, video and the live call.

The container image carries all five models, so a cold start never depends on
the network. CI builds it, pushes it, and verifies it by running it with no
network access at all.

---

## 1. Get the image

GitHub Actions already builds this image on every push to `main` or `dev`, and
verifies it by running it with no network access at all. **That verified image
is the one to deploy.**

### Deploying on Cloud Run: copy, do not rebuild

Cloud Run can only pull from Artifact Registry or GCR, not GHCR — so the image
has to be moved. Copy it:

```bash
read -rs GHCR_TOKEN && export GHCR_TOKEN     # a PAT with read:packages only
bash deploy/copy-image.sh latest
```

Run that in **Cloud Shell**. The script installs `crane`, creates the Artifact
Registry repository if needed, streams the image across without writing it to
local disk, and then confirms the source and destination **digests match** — so
what lands in Artifact Registry is provably the artefact CI tested, not merely
something that looks like it.

**Why not just run `cloudbuild.yaml`?** It would spend 30–60 build-minutes
reproducing something already proven, and the result would be a *different*
image: different layer hashes, and two Linux-only transitive packages (`triton`,
`sounddevice`) that `requirements.lock` cannot pin because it was frozen on
Windows. Copying deploys the thing that was tested. Rebuilding deploys something
that resembles it.

`cloudbuild.yaml` remains for the case where you need to build inside GCP —
after changing the Dockerfile, say, without going through GitHub.

**Do not build this locally unless you have a fast connection.** The image is
about 5 GB. Measured on the development machine, package downloads ran at
9–76 kB/s: roughly 8 hours to build, and closer to 18 to push, with no way to
resume a push from a cache. A CI runner does it in 11–12 minutes.

### The GHCR package is private

It inherits the repository's visibility. Keep it that way and authenticate with
a GitHub token scoped to `read:packages` and nothing else — which is what
`copy-image.sh` expects in `GHCR_TOKEN`. Making the package public also works
but is not necessary, since the copy happens once.

Use `read -rs` rather than `export GHCR_TOKEN=...` so the value never reaches
your shell history.

---

## 2. What the host must provide

These are requirements, not preferences. Each one has already bitten or would.

| Requirement | Value | Why |
|---|---|---|
| **Memory** | 4 GB minimum | Measured resident set is 1.54 GB with every model loaded. 2 GB tiers do not fit. |
| **Architecture** | `linux/amd64` | The image CI builds is x86-64. ARM hosts (Oracle Ampere, Graviton) need a separate ARM build and every pinned wheel re-verified for `aarch64`. |
| **HTTPS** | Required | Browsers expose the camera and microphone only in a secure context. Over plain `http://<ip>`, `navigator.mediaDevices` is undefined and **voice, video and the live call all stop working** — only text and file upload survive. A bare IP with no certificate is not a viable deployment. |
| **Instances** | Exactly one | Session state (`unified_pipeline`) and the live-call counter (`routers/stream.py`) are both per-process. A second instance treats a known `session_id` as brand new. |
| **Request timeout** | Raise from the default | Platforms apply their HTTP request timeout to WebSockets too. Cloud Run's default 300 s severs a live call mid-conversation at five minutes. |
| **Concurrency** | 4 or fewer | The app implements no admission control of its own beyond the live-call cap. |

---

## 3. Deploy

### Google Cloud Run

```bash
gcloud run deploy emotion-detection \
  --image=us-central1-docker.pkg.dev/YOUR_PROJECT/humanoid/humanoid-input:latest \
  --region=us-central1 \
  --memory=4Gi \
  --cpu=2 \
  --timeout=3600 \
  --max-instances=1 \
  --concurrency=4 \
  --set-secrets=API_KEYS=humanoid-api-keys:latest \
  --allow-unauthenticated
```

`--allow-unauthenticated` is correct here: the app serves its own frontend, and
the API key gates `/analyze/*` and `/ws/stream`. Cloud Run's own IAM would block
the browser before it could present a key. HTTPS and a valid certificate come
free, which settles the secure-context requirement.

### Anything else

Railway, Fly.io and similar work the same way: point the host at the image, give
it 4 GB, set `API_KEYS` as a secret, raise the request timeout, and make sure the
URL is HTTPS.

---

## 4. Configure

| Variable | Set it to | Notes |
|---|---|---|
| `API_KEYS` | Your key, as a **platform secret** | Leaving it unset **disables authentication entirely**. Startup prints a loud banner when that happens. |
| `PORT` | Whatever the platform injects | Defaults to 8000 |
| `LLM_ENDPOINT_URL` | Your teammate's receiver | Optional; fire-and-forget |

There is deliberately **no setting that skips authentication for local
requests**. A bypass keyed on where the caller connected from would mean the
auth path is the one path local testing never exercises — which is how
`/ws/stream` once shipped with no authentication at all.

---

## 5. Verify after deploying

- [ ] `/health` returns `"mode": "full"` and every model `"available": true`
- [ ] `/analyze/text` returns **403** without a key — if it returns 200, the key
      was not picked up and the API is open to anyone
- [ ] `/analyze/text` returns a payload **with** the key
- [ ] The URL is `https://`, and the live call can access the camera in a browser
- [ ] Send your teammate the URL and `contract/CONTRACT.md`

---

## Notes

**Why models are baked in.** This system degrades *silently*: the model registry
catches every loading exception and `/health` reports success regardless. An
instance missing a model does not crash — it starts cleanly, passes its health
check, and answers `neutral` forever. Downloading at boot makes that outcome
depend on network conditions, and Hugging Face has already returned HTTP 429
during a startup here. `scripts/verify_image.sh` is what stops a half-populated
image reaching a registry.

**Cold starts.** Models load from local disk, not the network: expect 20–90
seconds to first response. On a scale-to-zero platform that cost is paid on the
first request after idling, so consider a minimum instance if a demo must feel
instant.

**Image size** is around 5 GB, dominated by PyTorch and the models. The CPU-only
torch index matters: the CUDA build would add several GB for no benefit on a
host with no GPU.

**Old key in git history.** A previously-committed key still exists in earlier
commits. That is harmless now it has been rotated, since the old value no longer
opens anything. Do not put the new one anywhere tracked.
