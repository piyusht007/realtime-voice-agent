// AudioWorklet: captures mic audio, resamples to 24 kHz mono and emits pcm16 frames (~40 ms).
const TARGET_RATE = 24000;
const FRAME_SAMPLES = 960; // 40 ms @ 24 kHz

class PcmCaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    // `sampleRate` is a global in the AudioWorkletGlobalScope.
    this.ratio = sampleRate / TARGET_RATE;
    this.pos = 0;
    this.out = new Int16Array(FRAME_SAMPLES);
    this.outIdx = 0;
  }

  push(sample) {
    const s = Math.max(-1, Math.min(1, sample));
    this.out[this.outIdx++] = s < 0 ? s * 0x8000 : s * 0x7fff;
    if (this.outIdx === FRAME_SAMPLES) {
      this.port.postMessage(this.out.buffer, [this.out.buffer]);
      this.out = new Int16Array(FRAME_SAMPLES);
      this.outIdx = 0;
    }
  }

  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    if (!channel) return true;
    if (this.ratio === 1) {
      for (let i = 0; i < channel.length; i++) this.push(channel[i]);
      return true;
    }
    // Linear-interpolation resampler; good enough for speech.
    while (this.pos < channel.length - 1) {
      const i = Math.floor(this.pos);
      const frac = this.pos - i;
      this.push(channel[i] * (1 - frac) + channel[i + 1] * frac);
      this.pos += this.ratio;
    }
    this.pos -= channel.length;
    if (this.pos < 0) this.pos = 0;
    return true;
  }
}

registerProcessor("pcm-capture", PcmCaptureProcessor);
