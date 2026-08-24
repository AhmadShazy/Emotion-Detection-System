/**
 * pcm-worklet.js
 * ==============
 * Turns the microphone into 16 kHz signed-16-bit PCM frames for the live call.
 *
 * Why an AudioWorklet rather than ScriptProcessorNode (which the Voice tab
 * still uses): ScriptProcessorNode runs on the main thread, so a busy UI
 * causes dropouts in the captured audio, and it is deprecated. A worklet runs
 * on the audio rendering thread and is unaffected by page work.
 *
 * Why Int16 rather than Float32 on the wire: exactly half the bytes
 * (32 KB/s vs 64 KB/s) for zero real loss — the microphone ADC is 16-bit, so
 * float32 would be shipping precision the signal never had.
 *
 * The 16 kHz conversion is done by the browser, by constructing the
 * AudioContext at that rate. That is deliberate: nothing on the server
 * resamples, and SpeechBrain returns a confident WRONG label rather than an
 * error if handed the wrong sample rate.
 */

// 2048 samples = 128 ms at 16 kHz. Small enough that turn detection stays
// responsive, large enough that we are not sending hundreds of tiny frames
// per second.
const FRAME_SAMPLES = 2048;

class PCM16kProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this._buffer = new Int16Array(FRAME_SAMPLES);
        this._filled = 0;
    }

    process(inputs) {
        const channel = inputs[0] && inputs[0][0];

        // No input yet (or the track ended). Returning true keeps the node
        // alive so capture resumes if the source comes back.
        if (!channel) return true;

        for (let i = 0; i < channel.length; i++) {
            // Clamp before scaling: values slightly outside [-1, 1] are legal
            // in Web Audio and would wrap around on conversion.
            const s = Math.max(-1, Math.min(1, channel[i]));
            this._buffer[this._filled++] = s < 0 ? s * 32768 : s * 32767;

            if (this._filled === FRAME_SAMPLES) {
                // Transfer a copy — the underlying buffer is reused, so
                // posting it directly would let the main thread read samples
                // we are already overwriting.
                const out = new Int16Array(this._buffer);
                this.port.postMessage(out.buffer, [out.buffer]);
                this._filled = 0;
            }
        }

        return true;
    }
}

registerProcessor('pcm-16k', PCM16kProcessor);
