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
      guarantee written in `src/core/config.py`. Fixed in `17c0369`, then the
      bypass was **removed entirely** — a rule keyed on where the caller
      connected from meant the auth path was the one path local testing never
      exercised, and behind a cloud proxy the socket peer is the proxy,
      sometimes itself on loopback, so enabling it in a deployment would have
      opened everything rather than nothing. Local and deployed now take the
      same path. This also closes the `--proxy-headers` decision the review
      raised, since nothing reads the peer address any more.
- [x] **Two concurrent uploads could splice each other's transcripts** through
      the shared openai-whisper decoder, silently. Fixed in `17c0369`.
- [x] **The image now builds and runs full mode.** ffmpeg, the OpenCV and
      PortAudio system libraries, all five models baked in, CPU-only torch.
      Built by GitHub Actions and verified in CI with `--network=none`, so a
      half-populated image cannot reach a registry. Measured RAM floor:
      **1.54 GB idle**, which is why 4 GB is the deploy setting.
- [x] **TEXT_ONLY_MODE removed entirely.** The reduced mode existed for a free
      tier that could not host the full stack; once that plan was dropped it
      was a second code path nothing exercised — half the routes had a
      disabled twin, the registry had two loading strategies, and the
      frontend asked `/health` which one it was talking to. A test now guards
      its absence.
- [x] **Dependencies pinned.** `requirements.lock` holds 94 packages frozen
      from the environment the tests pass in, and the Dockerfile installs it
      rather than the unpinned list. The SpeechBrain compatibility patches
      turned out to be duplicated in **four** places, two of them incomplete —
      and the incomplete copies were the ones a container build depended on,
      so the downloader failed on speechbrain 1.1.0 while the running server
      loaded the same model fine. One definition now, with a test guarding it.
- [x] **`requirements-text.txt` deleted** along with TEXT_ONLY_MODE. It
      described a minimal install for a deployment that no longer exists.

---

## Before deploying

### 1. Make `/health` tell the truth
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

### 2. Pin the PyTorch thread count
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

### 3. Harden `/analyze/voice`
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

### 4. The live-call slot leak
`routers/stream.py:107` vs `162` · **5 min**

`_active_calls += 1` happens at line 107; the `try` whose `finally` holds the
only decrement does not open until 162. A peer reset in that window never
returns the slot. The counter is monotonic — no decay, no timeout, no
reconciliation — and on Render or Railway the process lives for days. The window
is only one client round-trip wide, but the counter never heals.

### 5. Decide and write down the deployment configuration
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
- **Set `API_KEYS` as a platform secret.** Leaving it unset is the only
  remaining way to run without authentication, and startup prints a loud banner
  when that happens.

### 6. Decide the push-delivery story
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
