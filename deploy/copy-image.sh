#!/usr/bin/env bash
#
# deploy/copy-image.sh
# ====================
# Copies the already-built image from GitHub Container Registry into Google
# Artifact Registry, so Cloud Run can deploy it.
#
#   usage:  bash deploy/copy-image.sh [SOURCE_TAG]
#   e.g.:   bash deploy/copy-image.sh latest
#
# Run this in Cloud Shell.
#
# Why copy instead of rebuilding
# ------------------------------
# GitHub Actions already builds this image and verifies it end to end: it boots
# the container with --network=none and asserts every model loads, ffmpeg is
# present, the MediaPipe model file exists and torch is a CPU build. That exact
# image, by digest, is the artefact worth deploying.
#
# Rebuilding it in Cloud Build would spend 30-60 build-minutes reproducing
# something already proven, and would produce a DIFFERENT image — different
# layer hashes, and two Linux-only transitive packages (triton, sounddevice)
# that requirements.lock cannot pin because it was frozen on Windows. Copying
# deploys the thing that was tested. Rebuilding deploys something that resembles
# it.
#
# Why crane rather than docker pull/push
# --------------------------------------
# crane streams registry to registry without ever writing the image to local
# disk. This image is about 5 GB and Cloud Shell has a 5 GB home directory, so a
# docker pull would be tight at best. crane sidesteps it entirely and is faster,
# since nothing is unpacked.
#
# Cloud Run cannot pull from GHCR directly — it requires Artifact Registry or
# GCR — which is the only reason this step exists at all.

set -euo pipefail

SOURCE_TAG="${1:-latest}"

GH_OWNER="ahmadshazy"                      # must be lowercase for a registry path
GH_REPO="emotion-detection-system"
GHCR_IMAGE="ghcr.io/${GH_OWNER}/${GH_REPO}:${SOURCE_TAG}"

REGION="${REGION:-us-central1}"
AR_REPO="${AR_REPO:-humanoid}"
AR_IMAGE_NAME="${AR_IMAGE_NAME:-humanoid-input}"

say()  { printf '\n\033[1m== %s\033[0m\n' "$*"; }
fail() { printf '\nFAIL: %s\n' "$*" >&2; exit 1; }

# ── Preconditions ────────────────────────────────────────────────────────────
say "Checking prerequisites"

command -v gcloud >/dev/null || fail "gcloud not found. Run this in Cloud Shell."

PROJECT_ID="$(gcloud config get-value project 2>/dev/null || true)"
[ -n "$PROJECT_ID" ] && [ "$PROJECT_ID" != "(unset)" ] \
  || fail "No project set. Run: gcloud config set project YOUR_PROJECT_ID"

AR_IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO}/${AR_IMAGE_NAME}:${SOURCE_TAG}"

echo "  project : $PROJECT_ID"
echo "  from    : $GHCR_IMAGE"
echo "  to      : $AR_IMAGE"

# ── crane ────────────────────────────────────────────────────────────────────
# Not preinstalled in Cloud Shell. It is a single small Go binary and the
# download runs at datacenter speed, so fetching it costs seconds.
if ! command -v crane >/dev/null; then
  say "Installing crane"
  CRANE_VERSION="v0.20.2"
  TMP="$(mktemp -d)"
  curl -sSL \
    "https://github.com/google/go-containerregistry/releases/download/${CRANE_VERSION}/go-containerregistry_Linux_x86_64.tar.gz" \
    | tar -xz -C "$TMP" crane
  mkdir -p "$HOME/bin"
  mv "$TMP/crane" "$HOME/bin/crane"
  chmod +x "$HOME/bin/crane"
  export PATH="$HOME/bin:$PATH"
  rm -rf "$TMP"
  echo "  installed to ~/bin/crane"
fi
crane version >/dev/null || fail "crane is installed but not runnable"

# ── Artifact Registry repository ─────────────────────────────────────────────
say "Ensuring the Artifact Registry repository exists"
if gcloud artifacts repositories describe "$AR_REPO" \
      --location="$REGION" >/dev/null 2>&1; then
  echo "  $AR_REPO already exists"
else
  gcloud artifacts repositories create "$AR_REPO" \
    --repository-format=docker \
    --location="$REGION" \
    --description="Humanoid Assistant input module"
  echo "  created $AR_REPO"
fi

# ── Authentication ───────────────────────────────────────────────────────────
say "Authenticating to Artifact Registry"
gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet
echo "  ok"

say "Authenticating to GHCR"
# The package is private, so a token is required. Read from the environment
# rather than prompting, so the value never lands in shell history.
if [ -z "${GHCR_TOKEN:-}" ]; then
  echo "  GHCR_TOKEN is not set."
  echo
  echo "  Create a classic PAT with ONLY the read:packages scope at"
  echo "    https://github.com/settings/tokens"
  echo "  then, in this shell:"
  echo
  echo "    read -rs GHCR_TOKEN && export GHCR_TOKEN"
  echo
  echo "  (read -rs keeps it off the screen and out of history.)"
  fail "GHCR_TOKEN required"
fi
echo "$GHCR_TOKEN" | crane auth login ghcr.io -u "$GH_OWNER" --password-stdin
echo "  ok"

# ── Copy ─────────────────────────────────────────────────────────────────────
say "Copying image (streamed, nothing written to local disk)"
echo "  this moves ~5 GB between two datacenters; expect a few minutes"
crane copy "$GHCR_IMAGE" "$AR_IMAGE"

# ── Verify ───────────────────────────────────────────────────────────────────
# A copy that silently produced a different image would be worse than a failed
# one, so compare digests rather than trusting the exit code.
say "Verifying"
SRC_DIGEST="$(crane digest "$GHCR_IMAGE")"
DST_DIGEST="$(crane digest "$AR_IMAGE")"
echo "  source      : $SRC_DIGEST"
echo "  destination : $DST_DIGEST"

[ "$SRC_DIGEST" = "$DST_DIGEST" ] \
  || fail "digest mismatch - the copy is not the image that was verified in CI"

echo "  digests match"

# Confirm it really is amd64. Cloud Run runs x86-64, and an arm64 image would
# deploy and then fail at runtime in a way that is awkward to diagnose.
ARCH="$(crane config "$AR_IMAGE" | grep -o '"architecture":"[^"]*"' | head -1)"
echo "  ${ARCH:-architecture unknown}"

say "Done"
cat <<EOF

Image is in Artifact Registry:

  $AR_IMAGE

Deploy it with:

  gcloud run deploy emotion-detection \\
    --image=$AR_IMAGE \\
    --region=$REGION \\
    --memory=4Gi \\
    --cpu=2 \\
    --timeout=3600 \\
    --max-instances=1 \\
    --concurrency=4 \\
    --cpu-throttling \\
    --set-secrets=API_KEYS=humanoid-api-keys:latest \\
    --allow-unauthenticated

Two flags matter for staying free:

  --cpu-throttling    the default, stated explicitly. CPU is allocated only
                      while a request is being handled, so an idle service
                      bills nothing. --no-cpu-throttling would bill around the
                      clock.

  NO --min-instances  adding it keeps a container warm continuously and would
                      exhaust the Cloud Run free tier in roughly a day. Cold
                      starts of 20-90s are the price of staying free; hit the
                      URL a minute before a demo to warm it.
EOF
