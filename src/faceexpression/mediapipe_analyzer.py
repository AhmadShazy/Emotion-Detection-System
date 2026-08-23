"""
src/faceexpression/mediapipe_analyzer.py
========================================
Reads facial emotion from video frames using MediaPipe, replacing the OpenFace
subprocess.

Why the swap
------------
OpenFace is a Windows-only .exe that reads a camera device or a video file. Both
properties are fatal for a hosted Linux service. MediaPipe is a pip package that
consumes frames directly, so it works in a container and on browser-supplied
video.

What is preserved
-----------------
The same five labels (Happy, Surprised, Angry, Sad, Neutral) and the same
face_state contract the fusion engine already consumes, so nothing downstream
changes.

What is different, honestly
---------------------------
OpenFace emitted BINARY Action Unit flags from classifiers trained on
FACS-coded data. MediaPipe emits CONTINUOUS blendshape scores, which are rig
coefficients for animating a face rather than anatomical muscle codings. They
correspond closely enough for these rules, but the thresholds below are
sensitive to lighting, camera and individual face shape in a way OpenFace's
person-normalised AUs were not. Expect to tune THRESHOLD against real footage.
"""

import os
import threading
from contextlib import contextmanager

# Blendshape names → the Action Unit each one stands in for.
#   AU01 inner brow raise  → browInnerUp
#   AU02 outer brow raise  → browOuterUpLeft/Right
#   AU04 brow lowerer      → browDownLeft/Right
#   AU06 cheek raiser      → cheekSquintLeft/Right
#   AU07 lid tightener     → eyeSquintLeft/Right
#   AU12 lip corner puller → mouthSmileLeft/Right
#   AU15 corner depressor  → mouthFrownLeft/Right
#   AU26 jaw drop          → jawOpen
AU_BLENDSHAPES = {
    "AU01": ("browInnerUp",),
    "AU02": ("browOuterUpLeft", "browOuterUpRight"),
    "AU04": ("browDownLeft", "browDownRight"),
    "AU06": ("cheekSquintLeft", "cheekSquintRight"),
    "AU07": ("eyeSquintLeft", "eyeSquintRight"),
    "AU12": ("mouthSmileLeft", "mouthSmileRight"),
    "AU15": ("mouthFrownLeft", "mouthFrownRight"),
    "AU26": ("jawOpen",),
}

# Above this, a blendshape counts as "present", standing in for OpenFace's
# binary flag. Deliberately a single tunable rather than per-AU values until
# there is real footage to tune against.
THRESHOLD = 0.35

# How much REAL TIME the per-frame smoothing vote covers.
#
# Expressed in seconds rather than frames on purpose. The two callers sample at
# different rates — uploads at 5fps, live calls at ~3fps — so a fixed frame
# count would mean they smoothed over different durations, and the same footage
# would read differently depending on which path it arrived through. They
# previously used 10 and 5 frames, which happened to land near each other in
# time by coincidence rather than design.
SMOOTHING_SECONDS = 2.0


def _smoothing_window(fps: float) -> int:
    """Frames needed to cover SMOOTHING_SECONDS at the given sample rate."""
    return max(1, round(SMOOTHING_SECONDS * fps))

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "external", "mediapipe", "face_landmarker.task",
)

# ── Landmarker pool ───────────────────────────────────────────────────────────
# There used to be ONE landmarker behind a module-level lock, and analyze_frames
# held that lock for its entire frame loop. With one caller that is fine. With
# several it is not: one caller's batch locked out everyone else for its whole
# duration, so the lock delivered no parallelism at all and added contention on
# top of it.
#
# MediaPipe's detect() is C++ and does release the GIL, so separate instances DO
# run in parallel. But an instance per connection grows without bound and would
# oversubscribe the same cores that Whisper and SpeechBrain are already
# saturating — and those are the real bottleneck, not the face.
#
# So: a small fixed pool, leased through a queue. The queue doubles as admission
# control — when every instance is busy, the next caller waits rather than
# piling more work onto a saturated CPU.
#
# Building an instance is cheap (~90ms, 3.8MB bundle), so the pool is created
# lazily on first use rather than at import.

from src.core.config import FACE_POOL_WORKERS as POOL_SIZE

_pool: "queue.Queue" = None
_pool_lock = threading.Lock()


def _build_landmarker():
    if not os.path.isfile(MODEL_PATH):
        raise RuntimeError(
            f"MediaPipe face model missing at {MODEL_PATH}. "
            f"Run: python scripts/download_models.py"
        )

    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.IMAGE,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=False,
        num_faces=1,
    )
    return vision.FaceLandmarker.create_from_options(options)


def _get_pool():
    global _pool
    if _pool is not None:
        return _pool

    with _pool_lock:
        if _pool is not None:
            return _pool

        import queue as _queue
        pool = _queue.Queue()
        for _ in range(max(1, POOL_SIZE)):
            pool.put(_build_landmarker())
        _pool = pool
        print(f"[MediaPipe] Face landmarker pool ready ({POOL_SIZE} instances).")
        return _pool


@contextmanager
def lease_landmarker(timeout: float = 30.0):
    """
    Borrows a landmarker for the duration of the block.

    Always returned, including on exception — a leaked instance would shrink
    the pool permanently and eventually deadlock every caller.
    """
    pool = _get_pool()
    try:
        landmarker = pool.get(timeout=timeout)
    except Exception:
        raise RuntimeError(
            "No face landmarker became available within "
            f"{timeout}s — the server is overloaded."
        )
    try:
        yield landmarker
    finally:
        pool.put(landmarker)


def _get_landmarker():
    """
    Back-compat shim for callers that just want an instance without leasing.

    Prefer lease_landmarker(). This exists because tests and the Phase 3 batch
    path referenced it directly.
    """
    pool = _get_pool()
    landmarker = pool.get()
    pool.put(landmarker)
    return landmarker


def _scores(blendshapes) -> dict:
    return {c.category_name: c.score for c in blendshapes}


def _au_active(scores: dict, au: str) -> bool:
    """True when any blendshape standing in for this Action Unit is present."""
    return any(scores.get(name, 0.0) >= THRESHOLD for name in AU_BLENDSHAPES[au])


def classify_frame(scores: dict) -> str:
    """
    Applies the same rules, in the same order, as the OpenFace classifier.

    Order matters and the rules are NOT mutually exclusive — a face can satisfy
    several at once, and the first match wins. That behaviour is deliberately
    preserved so results stay comparable to the previous implementation.
    """
    # Happy — lip corner puller OR cheek raiser
    if _au_active(scores, "AU12") or _au_active(scores, "AU06"):
        return "Happy"

    # Surprised — jaw drop, or both brows raised together
    if _au_active(scores, "AU26") or (_au_active(scores, "AU01") and _au_active(scores, "AU02")):
        return "Surprised"

    # Angry — brow lowerer AND lid tightener, both required
    if _au_active(scores, "AU04") and _au_active(scores, "AU07"):
        return "Angry"

    # Sad — inner brow raise with lip corner depressor, or the depressor alone
    if _au_active(scores, "AU15"):
        return "Sad"

    return "Neutral"


def analyze_frames(frame_paths: list) -> tuple:
    """
    Reads every sampled frame and aggregates one face_state.

    Returns (timeline_str, face_state | None). face_state is None when no face
    was found in any frame — a legitimate outcome for a video shot in the dark
    or pointed at a ceiling, and the fusion engine already handles a missing
    face modality.
    """
    import mediapipe as mp
    import numpy as np
    from PIL import Image

    emotions = []
    frames_with_face = 0

    # One lease for the whole batch. This is a single upload being processed as
    # a unit, so holding an instance for its duration is right — and the pool
    # means other callers still have their own instances to work with.
    with lease_landmarker() as landmarker:
        for path in frame_paths:
            try:
                # Load fully and close immediately. PIL keeps a lazy file handle
                # otherwise, which blocks the cleanup rmtree on Windows.
                with Image.open(path) as img:
                    rgb = np.asarray(img.convert("RGB"))

                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect(image)

                if not result.face_blendshapes:
                    continue

                frames_with_face += 1
                emotions.append(classify_frame(_scores(result.face_blendshapes[0])))

            except Exception as exc:
                print(f"[MediaPipe] Frame {os.path.basename(path)} failed: {exc}")

    total = len(frame_paths)

    if not emotions:
        return ("No face detected in any frame.", None)

    from src.video.ingest import FRAME_SAMPLE_FPS
    smoothed = _smooth(emotions, window=_smoothing_window(FRAME_SAMPLE_FPS))

    from collections import Counter
    counts = Counter(smoothed)
    dominant, top_count = counts.most_common(1)[0]
    confidence = top_count / len(smoothed)

    # Reliability = how much of the clip actually contained a readable face.
    # The OpenFace path hardcoded this to 1.0 regardless of tracking quality,
    # which let a barely-usable video enter fusion at full weight. User-supplied
    # video varies far more than a fixed webcam, so this now reflects reality.
    reliability = frames_with_face / total if total else 0.0

    transitions = sum(1 for i in range(1, len(emotions)) if emotions[i] != emotions[i - 1])
    instability = transitions / len(emotions) if len(emotions) > 1 else 0.0

    face_state = {
        "source":      "face",
        "emotion":     dominant,
        "confidence":  confidence,
        "reliability": reliability,
        "instability": instability,
    }

    return (_timeline(smoothed), face_state)


def _smooth(emotions: list, window: int = 10) -> list:
    """Sliding-window majority vote, damping single-frame noise."""
    from collections import Counter
    out = []
    for i in range(len(emotions)):
        start = max(0, i - window)
        out.append(Counter(emotions[start:i + 1]).most_common(1)[0][0])
    return out


def _timeline(smoothed: list) -> str:
    """
    Collapses the per-frame labels into readable spans.

    Built in memory and returned; nothing is written to disk. The previous
    implementation wrote two fixed paths, so concurrent requests overwrote each
    other's output and could raise PermissionError on Windows.
    """
    if not smoothed:
        return ""

    from src.video.ingest import FRAME_SAMPLE_FPS
    seconds_per_frame = 1.0 / FRAME_SAMPLE_FPS

    lines = []
    current, start_idx = smoothed[0], 0

    for i in range(1, len(smoothed) + 1):
        if i == len(smoothed) or smoothed[i] != current:
            lines.append(
                f"{start_idx * seconds_per_frame:.2f}s - "
                f"{i * seconds_per_frame:.2f}s : {current}"
            )
            if i < len(smoothed):
                current, start_idx = smoothed[i], i

    return "\n".join(lines)


# ── Live path ─────────────────────────────────────────────────────────────────

def analyze_jpeg_frames(jpeg_frames: list) -> dict | None:
    """
    Aggregates a face_state from JPEG frames held in memory.

    The live-call counterpart to analyze_frames(). The batch path reads files
    written by ffmpeg; here the frames arrive over a WebSocket and are never
    touched by disk, so there is nothing to clean up and no path to collide on.

    Decodes with OpenCV rather than PIL — measured ~1.7x faster, and it hands
    back a numpy array directly instead of going through an Image object.

    Returns the same face_state shape the fusion engine already consumes, or
    None when no face was found in any frame (a real outcome: bad lighting, or
    the caller stepped out of frame).
    """
    if not jpeg_frames:
        return None

    import cv2
    import numpy as np
    import mediapipe as mp
    from collections import Counter

    emotions = []
    frames_with_face = 0

    with lease_landmarker() as landmarker:
        for raw in jpeg_frames:
            try:
                buf = np.frombuffer(raw, dtype=np.uint8)
                bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                if bgr is None:
                    continue

                # OpenCV decodes BGR; MediaPipe expects SRGB.
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                result = landmarker.detect(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                )

                if not result.face_blendshapes:
                    continue

                frames_with_face += 1
                emotions.append(classify_frame(_scores(result.face_blendshapes[0])))

            except Exception as exc:
                print(f"[MediaPipe] Live frame failed: {exc}")

    if not emotions:
        return None

    # The browser sends video at LIVE_VIDEO_FPS; smoothing covers the same real
    # duration here as it does on the upload path.
    from src.core.config import LIVE_VIDEO_FPS
    smoothed = _smooth(emotions, window=_smoothing_window(LIVE_VIDEO_FPS))
    counts = Counter(smoothed)
    dominant, top_count = counts.most_common(1)[0]

    transitions = sum(
        1 for i in range(1, len(emotions)) if emotions[i] != emotions[i - 1]
    )

    return {
        "source":      "face",
        "emotion":     dominant,
        "confidence":  top_count / len(smoothed),
        # How much of the turn actually contained a readable face. A caller who
        # looked away for most of it should carry less weight in fusion.
        "reliability": frames_with_face / len(jpeg_frames),
        "instability": transitions / len(emotions) if len(emotions) > 1 else 0.0,
    }
