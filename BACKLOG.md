# Backlog

Outstanding work, ordered by when it hurts. Produced by a six-dimension
architecture review of the whole codebase (Aug 2026), where every finding was
challenged by a reviewer whose job was to defend the existing code.

**The architecture itself is sound and does not need redesigning.** Not one
finding says a module boundary is in the wrong place, that the shared payload
builder should be dismantled, or that the concurrency model needs rethinking.
Everything below is additive: the edge of the system is unfinished, not its
shape.

Effort estimates are for someone who already knows the codebase.

---

## Done

- [x] **Confidence pointed away from conflict detection.** The emotional history
      was pre-seeded with five neutrals, so a first-turn neutral scored 0.97
      against a 0.70 documented ceiling and could never rise with persistence.
      Separately, per-modality calibration multiplied the confidence rather than
      the weight, so agreeing evidence *lowered* the score. `masked_anger` and
      friends landed under the 0.35 mark the contract tells the LLM to ignore.
      Fixed in `780660e`, contract bumped to v1.2, payloads regenerated.
- [x] **`voice_and_text_agree` was not an agreement case.** Its text came back
      from RoBERTa as sadness while its voice said anger, so the example named
      for agreement was a disagreement — and the example set had no genuine
      agreement case at all. Reworded in `780660e`.
- [x] **`/ws/stream` had no authentication.** Every middleware check sat inside
      `if scope["type"] == "http"`. Fixed in `17c0369`.
- [x] **The localhost bypass trusted the Host header**, contradicting the
      guarantee written in `src/core/config.py`. Now reads the socket peer.
      Fixed in `17c0369`.
- [x] **Two concurrent uploads could splice each other's transcripts** through
      the shared openai-whisper decoder, silently. Fixed in `17c0369`.

---

## Before deploying

### 1. Build an image that actually runs full mode
`Dockerfile`, `.dockerignore` · **half a day to a day** · *the big one*

The Dockerfile installs `requirements-text.txt`, has **no `apt-get` line at
all**, hardcodes `TEXT_ONLY_MODE=true`, and `.dockerignore` excludes
`external/` where every model lives. Its header still cites the mic/webcam/
OpenFace constraint that Phases 3–4 removed.

Flipping `TEXT_ONLY_MODE=false` on the current image does **not** crash — it
starts cleanly, reports healthy, and returns `neutral` for every voice and video
turn, because `interactive_modes.py` swallows the ImportError and
`routers/video.py` imports MediaPipe lazily. Silent degradation is worse than a
crashloop.

Needs:
- `RUN apt-get install -y --no-install-recommends ffmpeg` — `src/video/ingest.py`
  shells out to `ffprobe` and `ffmpeg`.
- The real `requirements.txt`, minus `sounddevice` (needs PortAudio; no request
  path uses it).
- **Every model baked in**, not just RoBERTa. `external/speechbrain` contains
  five zero-byte symlinks into the HuggingFace cache, so `external/` and the HF
  cache must be copied in the *same layer* or the links dangle. Do it with
  `RUN python scripts/download_models.py` and `HF_HOME` set inside the image.
- Verify with `docker run --network=none` that nothing downloads at boot.

Measured RAM floor: **1.54 GB idle** (RoBERTa 341 MB, Whisper base 431,
SpeechBrain 633, faster-whisper 61, MediaPipe 94). A 2 GB tier is not viable;
4 GB works only with models baked in; 8 GB is comfortable. Expect a 4–6 GB
image — the right trade against Cloud Run's 240 s startup probe.

### 2. Pin the dependencies
`requirements.txt`, `requirements-text.txt` · **1 hour** · *do this before #1*

`grep -c '=='` returns **0** for both files. Every load-bearing dependency
floats. The two interpreters on the dev machine already disagree across three
major-version boundaries — transformers 4.57 vs 5.8, speechbrain 1.0.3 vs 1.1.0,
torch 2.9 vs 2.11, starlette 0.27 vs 1.0.

The SpeechBrain compatibility patches in `src/ser/ser_engine.py` are documented
as "for SpeechBrain 1.0.3", monkeypatch a **private** API, and one exists
specifically because "transformers removed `AutoModelWithLMHead` in v5" — while
the `.venv` already runs transformers 5.8. That patch block is duplicated in
**three** places (`model_registry.py`, `ser_engine.py`,
`scripts/download_models.py`).

An unpinned resolver inside `docker build` produces different library code than
what was tested, and combined with `/health` always reporting ok (#3) that is
the silent-neutral-forever failure again. Baking models in does not help,
because the code that *loads* them changed. "84 tests pass" is currently true of
only one of the two environments.

Fix: `pip freeze` the working environment into `requirements.lock`, install that
in the Dockerfile, collapse the three patch copies into one module, add it to
the existing single-definition test in `tests/test_contract.py`.

### 3. Make `/health` tell the truth
`api.py`, `src/core/model_registry.py` · **30 min**

`load_all` catches every loader exception, appends to a local `failed` list that
goes out of scope, and sets `_loaded = True` unconditionally. `/health`
hardcodes `"status": "ok"`. A failed model fetch therefore serves `neutral`
forever at HTTP 200 behind a green health check, with one ❌ line in stdout as
the only trace.

Return 503 / `"degraded"` when a model required by the current mode is missing.
Add `active_calls` to the response while in there.

**Fold in:** MediaPipe is the one model outside the registry. When
`face_landmarker.task` is missing, the error is swallowed into a value meaning
"no face was visible" — so a missing model file is indistinguishable from a dark
room, and `conflict_analysis` can never fire. `README.md` still tells you to
download OpenFace and never mentions `scripts/download_models.py`.

### 4. Pin the PyTorch thread count
`src/core/model_registry.py`, `Dockerfile` · **15 min + re-measure**

No `set_num_threads`, no `OMP_NUM_THREADS`, no `MKL_NUM_THREADS` anywhere.
PyTorch sizes its pools from the CPUs *visible in the container*, and on a
shared cloud host a 1–2 vCPU allocation commonly still reports the host's full
core count. Three concurrent stages × 8–32 intra-op threads each, on a fraction
of a core, is the difference between "slower than the laptop" and "unusable" —
on a system whose whole value proposition is live-call latency.

Pin alongside `INFERENCE_WORKERS`, set both env vars in the Dockerfile, then
**re-measure**. Do not assume this explains the earlier "more workers = slower"
result; that link is unverified.

### 5. Harden `/analyze/voice`
`routers/voice.py` · **30 min**

The least defended route, and `/analyze/video` already does all three correctly.

- **No size cap** — bare `await file.read()`, where video streams in 1 MB chunks
  with a cap and a comment explaining why.
- **Leaks the original WAV** — `wav_path` is rebound to `ensure_16k_mono`'s
  return, so the `finally` deletes only the converted file. Every recording
  leaves a permanent copy of a user's voice on the server. **Do not** fix this by
  extending `sweep_stale_jobs` to `data/recordings` — that directory holds the
  test fixtures `tests/test_live_stream.py` globs.
- **Corrupt uploads report 500** — `VideoIngestError` falls through to the
  generic handler; video maps the identical exception to 400.

### 6. The live-call slot leak
`routers/stream.py:107` vs `162` · **5 min**

`_active_calls += 1` happens at line 107; the `try` whose `finally` holds the
only decrement does not open until 162. A peer reset in that window never
returns the slot. The counter is monotonic — no decay, no timeout, no
reconciliation — and on Render or Railway the process lives for days. The window
is only one client round-trip wide, but the counter never heals.

### 7. Decide and write down the deployment configuration
`deploy/README.md`, `Dockerfile`, platform flags · **1 hour, no code**

The runbook currently says voice, multimodal and live-stream "are not deployable
yet" and need a server mic, webcam and Windows-only OpenFace. It describes the
previous architecture.

- **One replica.** `_sessions` in `unified_pipeline.py` is a module-level dict.
  A second instance treats a known `session_id` as brand new. Set
  `--max-instances=1`. Do **not** build a Redis session store — wrong spend.
- **`--workers 1` explicit**, with a comment that `MAX_ACTIVE_CALLS` is
  per-process. Two workers silently double the cap and the 1.54 GB.
- **`--timeout=3600`.** Cloud Run applies its request timeout to WebSockets,
  default 300 s — a demo conversation is severed mid-turn at five minutes.
- **Lower `MAX_UPLOAD_BYTES`** from 200 MB. 30 s of 720p webm is under 15 MB.
- **Point the platform health check at `/health`** — Cloud Run ignores the
  Dockerfile `HEALTHCHECK`.
- **Leave `ALLOW_LOCALHOST=false`.** Behind a proxy the socket peer is the
  proxy, sometimes on loopback; turning it on would expose everything. If it
  ever must be on, run uvicorn with `--proxy-headers` and read
  `X-Forwarded-For`.

### 8. Decide the push-delivery story
`contract/CONTRACT.md`, `routers/text.py` · **15 min to document, 2 h to build**

`LLM_ENDPOINT_URL` appears in `config.py`, `routers/text.py` and
`routers/mock.py` — nowhere else. Voice returns, video returns, stream sends over
the socket. But `CONTRACT.md` says the module POSTs the payload "after each
turn", and "each turn" names the live call — the one path that definitely does
not push. `routers/mock.py` compounds it: "exactly as the live pipeline would."

The teammate will stand up a receiver, verify against `/mock/emit`, mark the
integration done, and get nothing from the live demo.

If implementing: call a small `deliver()` module from the three *transport*
sites, **not** from inside `process_and_print_unified_json` — the fixture
generator and the test suite both call that, and it would fire real POSTs at a
teammate's endpoint on every regeneration.

---

## Soon after

- **The live client has no uplink backpressure.** `frontend/app.js` — both
  senders fire whenever the socket is open, roughly 0.6–0.9 Mbit/s sustained.
  A grep for `bufferedAmount` returns nothing. On weak wifi the browser buffers
  without bound and the call drifts minutes behind before the tab dies.
  Server-side `BUSY` shedding cannot help — it sheds turns, and the detector
  only sees audio that already crossed the wire. **Two lines.**
- **Every live-socket error reaches the browser as a clean hangup.** The
  `finally` calls bare `websocket.close()`; Starlette's default is 1000, Normal
  Closure. Send a typed error and close 1011. **15 min.**
- **One log line in the whole system carries a session id.** Every error path
  prints a bare `str(exc)` with no correlation and no traceback. With two people
  talking at once the log is untagged interleaved lines. Pass the session id
  into the failure prints and use `traceback.format_exc()` in the four handlers
  that swallow a modality. **10 min.**
- **CORS is fully open** — `allow_origins=["*"]` with
  `allow_credentials=True`. Combined with an empty `API_KEYS`, any page a user
  visits could drive `/analyze/*` from their browser and read the results.
  **One line.**
- **The auth predicate is "does the last path segment contain a dot".** Nothing
  is exploitable today, but it silently un-authenticates any future route with a
  dotted path parameter. Serve the frontend from `/static` and check that prefix
  instead.
- **`test_live_stream.py` globs `data/recordings/` by relative path** while
  every other test computes `PROJECT_ROOT`, and `data/` is gitignored — so the
  only end-to-end live test silently skips on a fresh clone, and whenever pytest
  runs from another cwd. Synthesise the WAV with the existing `_tone()` helper
  and add `addopts = -ra` to `pytest.ini` so skip reasons print. **30 min.**
- **`/analyze/voice` has no tests at all**, including `ensure_16k_mono`, which
  guards the failure the code itself calls "a confident WRONG label". **30 min.**
- **Dead code claiming a safety property.** `call_session.py:148`
  `check_overflow()` tests against 480,000 samples but the detector resets the
  turn at 320,000, so `TURN_TOO_LONG` can never fire. **10 min.**
- **Stack traces leak into 500 responses.** `str(exc)` is interpolated into
  error bodies; a `subprocess.TimeoutExpired` puts the full container path and
  every ffmpeg flag in the response. The `code` field `schemas/emotion.py`
  declares is never set by any route. **45 min.**
- **Two enum values the system can never produce** — `dominant_emotion` can
  never be `"empathetic"`, `tone` can never be `"neutral"`. A demo-honesty
  issue: an examiner reading the contract may ask to see one. **20 min.**
- **The README documents endpoints that do not exist** —
  `/analyze/multimodal/start` and `/stop`, while omitting `/analyze/video` and
  every `/mock/*` route, and still crediting OpenFace. Five unused schema
  classes can go too.

---

## Deliberately accepted

Spending time here is a mistake at this stage.

- **The two parked STT/SER tradeoffs** (Whisper base on upload vs faster-whisper
  tiny on live; SER seeing the whole file vs the last 6 s). Each is a single call
  site behind the shared spine, so revisiting later is a local edit, not a
  refactor.
- **No shared session store.** One replica plus a written-down note is right
  here; a live call is inherently sticky anyway.
- **Unbounded `_sessions` growth.** ~2 KB per session with a working sweeper and
  a 30-minute TTL; threatening a 2 GB box needs ~300,000 live sessions.
- **No admission control on the HTTP routes.** The default executor is a bounded
  pool, so request nine queues rather than piling more model calls onto the
  cores — degradation is "slow", not "unbounded". Cloud Run's `--concurrency`
  is this control, for one flag.
- **No idle timeout on the WebSocket.** The AudioWorklet pushes PCM whether or
  not anyone speaks, so a tabbed-away user genuinely *is* occupying a call.
- **`print()` instead of `logging`.** The correlation and traceback fixes above
  are the part that matters; a project-wide migration buys filtering nobody will
  use before the demo.
- **Model accuracy and retraining.** Deferred by decision. Note that
  `text_only_neutral` maps "Okay, that works for me." to `happy` via RoBERTa's
  `approval` label — a label-mapping question, not a fusion one.
- **Missing `contract_version` and `turn_index`, `src/streaming` naming.** All
  real, all cosmetic with one consumer, one repo and one deadline.

---

## What must not be refactored

Verified as load-bearing and better than typical. Do not "clean these up".

- **The single payload builder.** All four paths funnel through one constructor,
  so they *cannot* structurally drift. This is why nearly every item above is an
  additive fix rather than a redesign.
- **The fusion engine's purity.** `emotion_state_manager.py` imports `deque` and
  nothing else; `llm_adapter.py` imports nothing. Fuzzed over 20,000 randomised
  modality combinations with zero contract violations.
- **"The server owns no capture device", honoured everywhere.** The one decision
  that makes concurrent users possible at all.
- **The concurrency shape.** Per-connection state touched only by that
  connection's coroutine, models shared read-only, the single-writer rule on the
  socket correctly enforced, nested pools creating *fresh* pools rather than
  resubmitting into the executor they run on.
- **Backpressure that refuses work instead of queueing it.** `BUSY` sheds a turn
  rather than building a queue — correct for a signal whose value decays.
- **The `/analyze/video` temp-file lifecycle.** `mkdtemp` before the `try`, one
  job dir, cleanup covering every status path and mid-upload disconnect, a
  startup sweeper for what SIGKILL leaves behind. **This is the template
  `/analyze/voice` should copy.**
- **Subprocess discipline in `ingest.py`.** Always `subprocess.run`, always a
  timeout, always `stdin=DEVNULL`.
- **Executable architecture rules** in `tests/test_contract.py`, which git-greps
  to fail the build if any module outside `config.py` reads the environment.
- **`routers/mock.py`.** Zero ML imports, both modes, real payloads. Unblocks the
  LLM teammate with no cold start.
- **`turn_detector.py`.** Pure numpy in, dict out, constructor-injected timings,
  a real invariance property across packet sizes.
