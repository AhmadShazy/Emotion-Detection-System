#!/usr/bin/env bash
#
# scripts/verify_image.sh — proves a built image is actually deployable.
#
#   usage: bash scripts/verify_image.sh <image-tag>
#
# Runs on a CI runner against a built image, NOT inside it.
#
# Why this is a script and not inline CI steps
# --------------------------------------------
# Both the GitHub Actions workflow and cloudbuild.yaml need exactly this check.
# Written twice, the two copies drift, and the one that drifts is the one that
# stops catching things. This project has already been bitten by that: the
# SpeechBrain compatibility patches existed in four places, and the two partial
# copies were the ones a Docker build depended on.
#
# What it is guarding against
# ---------------------------
# This system fails SILENTLY. ModelRegistry.load_all() catches every loading
# exception and sets _loaded = True regardless, and /health hardcodes
# "status": "ok". An image missing a model therefore does not crash — it starts
# cleanly, passes its health check, and returns "neutral" for every voice, video
# and live-call request forever. Nothing downstream can tell that apart from a
# genuinely neutral user.
#
# So a green build is not evidence of anything. This is.

set -euo pipefail

IMAGE="${1:?usage: verify_image.sh <image-tag>}"
NAME="verify-$$"
PORT=8000

cleanup() { docker rm -f "$NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT

fail() { echo "FAIL: $*" >&2; exit 1; }

echo "=============================================================="
echo "Verifying $IMAGE"
echo "=============================================================="

# ── 1. Static checks, no server needed ───────────────────────────────────────
# Cheaper than booting, and they fail with a clearer message.

echo "-- binaries and model files present in the image"

# ingest.py shells out to both by bare name. Without them every voice and video
# request fails at decode time, and nothing at build time would have noticed.
docker run --rm --entrypoint sh "$IMAGE" -c 'command -v ffmpeg'  >/dev/null \
  || fail "ffmpeg is not on PATH in the image"
docker run --rm --entrypoint sh "$IMAGE" -c 'command -v ffprobe' >/dev/null \
  || fail "ffprobe is not on PATH in the image"
echo "   ffmpeg, ffprobe: ok"

# MediaPipe is the one model OUTSIDE the registry, so it never appears in the
# startup log and the log-based checks below cannot see it. When the file is
# missing the resulting error is swallowed into a value meaning "no face was
# visible" — indistinguishable from a dark room, and conflict_analysis can then
# never fire. It has to be checked as a file.
docker run --rm --entrypoint sh "$IMAGE" \
  -c 'test -s /app/external/mediapipe/face_landmarker.task' \
  || fail "MediaPipe face_landmarker.task is missing or empty in the image"
echo "   mediapipe face_landmarker.task: ok"

# A CUDA torch in a CPU image is several GB of libraries nothing can use.
docker run --rm --entrypoint python "$IMAGE" \
  -c "import torch,sys; sys.exit(0 if '+cpu' in torch.__version__ else 1)" \
  || fail "torch is not the CPU build"
echo "   torch is a CPU build: ok"

# ── 2. Boot with NO network at all ───────────────────────────────────────────
# The entire point of baking models into the image. If anything still reaches
# for Hugging Face at startup, a cold start on a host with restricted egress
# degrades silently instead of failing.

echo "-- starting container with --network=none"
docker run -d --name "$NAME" --network=none \
  -e API_KEYS=ci-verification-key -e "PORT=$PORT" "$IMAGE" >/dev/null

READY=""
for i in $(seq 1 90); do
  if ! docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
    echo "---- container exited early ----"
    docker logs "$NAME" 2>&1 | tail -60
    fail "container exited before becoming ready"
  fi
  if docker logs "$NAME" 2>&1 | grep -q "API ready"; then
    READY="yes"
    echo "   ready after ~$((i * 5))s"
    break
  fi
  sleep 5
done

# Explicit, because a for-loop that simply runs out is the classic way a
# verification step passes without ever verifying anything.
[ -n "$READY" ] || {
  echo "---- last log ----"; docker logs "$NAME" 2>&1 | tail -60
  fail "server never reported readiness within 450s"
}

echo "---- startup log ----"
docker logs "$NAME" 2>&1 | tail -40
echo "---------------------"

# ── 3. Every model really loaded ─────────────────────────────────────────────
LOGS="$(docker logs "$NAME" 2>&1)"

# Anchored on the SUCCESS line, "<name>  loaded in 1.2s".
#
# Grepping for the bare model name would be worthless: the failure line is
# "[Registry] ❌ whisper  FAILED: ..." and contains the name too, so a model
# that failed would satisfy the check. The leading whitespace class also stops
# "whisper" from matching inside "faster_whisper".
for m in roberta whisper faster_whisper speechbrain; do
  echo "$LOGS" | grep -qE "[[:space:]]${m}[[:space:]]+loaded in" \
    || fail "model '$m' did not report loading successfully"
  echo "   $m: loaded"
done

# Belt and braces: no model may have reported failure, and the registry's own
# summary must agree. That summary line is only printed when its failed list is
# empty, so it is a real gate rather than a restatement.
echo "$LOGS" | grep -q "FAILED:" \
  && fail "at least one model reported FAILED in the startup log"
echo "$LOGS" | grep -q "All models ready" \
  || fail "registry did not report 'All models ready'"

# The startup log used to carry a "Mode: FULL" line, checked here to catch an
# image accidentally built in the reduced text-only mode. That mode no longer
# exists, so there is nothing left to confuse it with — and the four per-model
# assertions above are a stronger guarantee than the banner ever was.

echo "=============================================================="
echo "PASS — no network was available, every model loaded, full mode"
echo "=============================================================="
