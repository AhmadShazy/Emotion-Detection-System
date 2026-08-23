"""
src/streaming/turn_detector.py
==============================
Decides where one speech turn ends and the next begins, on a live audio stream.

Why this replaces the old detector
----------------------------------
streaming_stt.py used a FIXED absolute threshold: a 250ms chunk counted as
silence if its RMS fell below 0.01. That number is wrong on this project's own
data. Measuring the 18 recordings in data/recordings/, global RMS ranges from
0.0029 to 0.0419 — a 14x spread from the same machine and microphone — and
FIVE of the eighteen sit entirely below 0.01. On those files every chunk reads
as silence, `has_spoken` never latches, and no turn is ever cut. The user talks
and nothing happens.

Browser capture makes this worse, not better: getUserMedia applies
autoGainControl and noiseSuppression by default, and their behaviour differs
per browser and per device, so the absolute level of "speech" is not knowable
in advance.

So the threshold here is RELATIVE to the noise floor actually observed on this
connection, and it adapts while the call runs.

The state machine itself (accumulate, cut on trailing silence, discard if the
user never spoke) is kept from the original — it was sound, and counting in
SAMPLES rather than chunks means it does not care how the network packetises
the audio.
"""

import numpy as np

SAMPLE_RATE = 16000

# Decisions are made on 20ms frames. The old code measured RMS over a whole
# 250ms chunk, which smears the boundary: a short pause inside a word and a
# real end-of-sentence pause look the same at that resolution.
FRAME_SAMPLES = 320            # 20 ms @ 16 kHz

# ── Turn shape ────────────────────────────────────────────────────────────────
# How much trailing silence ends a turn.
#
# This is pure perceived latency: the caller has finished speaking and is
# waiting, but analysis cannot start until we are confident they actually
# stopped rather than drawing breath. The old value was 1.5s, which on top of
# ~3-4s of inference made the wait feel long.
#
# 0.9s still comfortably exceeds a normal inter-word pause (typically
# 0.15-0.4s) while returning ~0.6s to the caller on every single turn. Raise it
# if turns start cutting mid-sentence for slower speakers.
TRAILING_SILENCE_SECONDS = 0.9

# Ignore anything shorter than this; it is a cough, a click, or a door.
MIN_TURN_SECONDS = 0.7

# Hard cap on a single turn. Inference cost grows with turn length, and a caller
# who never pauses would otherwise grow the buffer without bound.
MAX_TURN_SECONDS = 20.0

# ── Adaptive threshold ────────────────────────────────────────────────────────
# Speech must exceed the noise floor by this much to open a turn...
SPEECH_ENTER_MULTIPLE = 3.5
# ...and fall below this multiple to count as silence again. The gap between
# the two is hysteresis: without it, audio hovering near the boundary flickers
# between speech and silence and chops one utterance into many turns.
SPEECH_EXIT_MULTIPLE = 2.0

# Absolute floor, so a perfectly silent (digital zero) input cannot make the
# adaptive threshold collapse to zero and treat its own dither as speech.
#
# Kept low (~-54 dBFS) on purpose. It only binds when the room is very quiet;
# the moment there is any real room tone the adaptive 3.5x term takes over and
# sits well above this. Setting it higher makes genuinely quiet speech
# undetectable — the recordings in data/recordings/ include one whose whole-file
# RMS is 0.0029, and its speech must still register.
MIN_ABSOLUTE_THRESHOLD = 0.002

# How fast the noise floor tracks the room. Slow enough to ignore speech,
# fast enough to follow a fan switching on.
NOISE_FLOOR_ALPHA = 0.05


class TurnDetector:
    """
    Feed it audio, it tells you when a turn is complete.

    One instance per connection — it holds per-caller state (that caller's
    noise floor, whether they are mid-utterance) and must never be shared.
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        trailing_silence_seconds: float = TRAILING_SILENCE_SECONDS,
        min_turn_seconds: float = MIN_TURN_SECONDS,
        max_turn_seconds: float = MAX_TURN_SECONDS,
    ):
        self.sample_rate = sample_rate
        self.trailing_silence_samples = int(trailing_silence_seconds * sample_rate)
        self.min_turn_samples = int(min_turn_seconds * sample_rate)
        self.max_turn_samples = int(max_turn_seconds * sample_rate)

        # Seeded at the absolute floor rather than 0: until we have heard the
        # room, assume it is quiet rather than assuming it is silent.
        self.noise_floor = MIN_ABSOLUTE_THRESHOLD

        self._reset_turn()

        # Leftover samples that did not fill a whole 20ms frame, carried into
        # the next call so frame alignment survives arbitrary packet sizes.
        self._pending = np.zeros(0, dtype=np.float32)

        self._calibrating = True
        self._calibration_frames = 0

    # ── State ────────────────────────────────────────────────────────────────

    def _reset_turn(self):
        self.in_speech = False
        self.turn_audio = []
        self.turn_samples = 0
        # Counted separately from turn_samples, which also includes the silence
        # between and after words. The minimum-length gate must test THIS —
        # otherwise a 0.2s cough followed by the 1.5s pause that closes the turn
        # measures 1.7s and sails past a 0.7s minimum.
        self.speech_samples = 0
        self.silence_run = 0

    @property
    def speech_threshold(self) -> float:
        return max(SPEECH_ENTER_MULTIPLE * self.noise_floor, MIN_ABSOLUTE_THRESHOLD)

    @property
    def silence_threshold(self) -> float:
        return max(SPEECH_EXIT_MULTIPLE * self.noise_floor, MIN_ABSOLUTE_THRESHOLD)

    # ── Main entry point ─────────────────────────────────────────────────────

    def push(self, samples: np.ndarray):
        """
        Adds audio and returns a completed turn, or None.

        Returns a dict when a turn closes:
            {"audio": np.float32 array, "duration": float, "reason": str}
        `reason` is "silence" for a normal end, or "max_length" when the caller
        talked past the cap.

        Accepts any number of samples — the caller does not have to align to
        frame boundaries.
        """
        if samples.dtype != np.float32:
            samples = samples.astype(np.float32)

        if self._pending.size:
            samples = np.concatenate((self._pending, samples))

        n_frames = len(samples) // FRAME_SAMPLES
        self._pending = samples[n_frames * FRAME_SAMPLES:].copy()

        completed = None

        for i in range(n_frames):
            frame = samples[i * FRAME_SAMPLES:(i + 1) * FRAME_SAMPLES]
            result = self._consume_frame(frame)
            if result is not None and completed is None:
                completed = result

        return completed

    def _consume_frame(self, frame: np.ndarray):
        rms = float(np.sqrt(np.mean(frame ** 2)))

        # ── Track the noise floor ────────────────────────────────────────────
        # Only while NOT in speech, so the speaker's own voice never inflates
        # the floor and desensitises the detector mid-sentence.
        if not self.in_speech:
            if self._calibrating:
                # Adopt the first second quickly instead of crawling up from
                # the seed value, so the very first utterance is not missed.
                self.noise_floor = (
                    rms if self._calibration_frames == 0
                    else 0.7 * self.noise_floor + 0.3 * rms
                )
                self._calibration_frames += 1
                if self._calibration_frames >= 50:      # 50 x 20ms = 1s
                    self._calibrating = False
            else:
                self.noise_floor = (
                    (1 - NOISE_FLOOR_ALPHA) * self.noise_floor
                    + NOISE_FLOOR_ALPHA * rms
                )
            self.noise_floor = max(self.noise_floor, 1e-6)

        # ── Speech / silence decision, with hysteresis ───────────────────────
        if self.in_speech:
            is_speech = rms >= self.silence_threshold
        else:
            is_speech = rms >= self.speech_threshold

        if not self.in_speech:
            if is_speech:
                self.in_speech = True
                self.turn_audio = [frame.copy()]
                self.turn_samples = len(frame)
                self.speech_samples = len(frame)
                self.silence_run = 0
            # Silence outside a turn is simply discarded.
            return None

        # ── Inside a turn ────────────────────────────────────────────────────
        self.turn_audio.append(frame.copy())
        self.turn_samples += len(frame)

        if is_speech:
            self.speech_samples += len(frame)
            self.silence_run = 0
        else:
            self.silence_run += len(frame)

        if self.turn_samples >= self.max_turn_samples:
            return self._close_turn("max_length")

        if self.silence_run >= self.trailing_silence_samples:
            return self._close_turn("silence")

        return None

    def _close_turn(self, reason: str):
        audio = (
            np.concatenate(self.turn_audio)
            if self.turn_audio else np.zeros(0, dtype=np.float32)
        )
        speech_samples = self.speech_samples
        trailing = self.silence_run
        self._reset_turn()

        # Gate on how much SPEECH there was, not how long the turn ran. A cough
        # plus the pause that ended it can easily exceed the minimum while
        # containing almost no voice.
        if speech_samples < self.min_turn_samples:
            return None

        # Trim the trailing pause before handing this to the models. It carries
        # no information, and both Whisper and SpeechBrain cost time roughly in
        # proportion to input length — sending 1.5s of silence on every turn is
        # pure latency. A little is kept so a trailing consonant is not clipped.
        if trailing > 0:
            keep = max(0, len(audio) - trailing + int(0.2 * self.sample_rate))
            if keep >= self.min_turn_samples:
                audio = audio[:keep]

        return {
            "audio":    audio,
            "duration": len(audio) / self.sample_rate,
            "reason":   reason,
        }

    def flush(self):
        """
        Closes whatever is in progress and returns it, or None.

        Call this when the stream ends — a caller who hangs up mid-sentence, or
        a finite recording that runs out while they are still talking. Without
        it that speech is silently discarded: the detector is still waiting for
        the trailing pause that will now never arrive.

        Applies the same minimum-speech rule as a normal close, so a hangup
        during a stray noise does not produce a junk turn.
        """
        if not self.in_speech or self.speech_samples < self.min_turn_samples:
            self._reset_turn()
            return None
        return self._close_turn("stream_ended")

    def abandon_turn(self):
        """
        Drops whatever is accumulated without emitting it.

        Used when the connection's buffer would overflow: an incomplete turn is
        worth less than a spliced one, because a transcript stitched across a
        gap reads as a real sentence while being wrong.
        """
        had = self.turn_samples > 0
        self._reset_turn()
        return had

    def stats(self) -> dict:
        return {
            "noise_floor":      round(self.noise_floor, 5),
            "speech_threshold": round(self.speech_threshold, 5),
            "in_speech":        self.in_speech,
            "turn_seconds":     round(self.turn_samples / self.sample_rate, 2),
            "speech_seconds":   round(self.speech_samples / self.sample_rate, 2),
            "calibrating":      self._calibrating,
        }
