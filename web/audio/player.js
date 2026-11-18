// Gapless pcm16 playback queue that can be flushed instantly on barge-in.
export const SAMPLE_RATE = 24000;

export function int16ToFloat32(int16) {
  const out = new Float32Array(int16.length);
  for (let i = 0; i < int16.length; i++) out[i] = int16[i] / 0x8000;
  return out;
}

export function base64ToInt16(b64) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Int16Array(bytes.buffer, 0, Math.floor(bytes.length / 2));
}

export class PcmPlayer {
  constructor(ctx) {
    this.ctx = ctx;
    this.nextTime = 0;
    this.sources = new Set();
    this.items = new Map(); // itemId -> { startTime, scheduledMs }
    this.currentItem = null;
  }

  enqueue(itemId, int16) {
    if (!int16.length) return;
    const buffer = this.ctx.createBuffer(1, int16.length, SAMPLE_RATE);
    buffer.copyToChannel(int16ToFloat32(int16), 0);
    const src = this.ctx.createBufferSource();
    src.buffer = buffer;
    src.connect(this.ctx.destination);
    const startAt = Math.max(this.ctx.currentTime + 0.02, this.nextTime);
    src.start(startAt);
    this.nextTime = startAt + buffer.duration;
    this.sources.add(src);
    src.onended = () => this.sources.delete(src);

    let item = this.items.get(itemId);
    if (!item) {
      item = { startTime: startAt, scheduledMs: 0 };
      this.items.set(itemId, item);
    }
    item.scheduledMs += buffer.duration * 1000;
    this.currentItem = itemId;
  }

  /** Milliseconds of `itemId` that have actually come out of the speaker. */
  playedMs(itemId) {
    const item = this.items.get(itemId);
    if (!item) return 0;
    const ms = (this.ctx.currentTime - item.startTime) * 1000;
    return Math.max(0, Math.min(ms, item.scheduledMs));
  }

  isPlaying() {
    return this.nextTime > this.ctx.currentTime;
  }

  /** Stop everything immediately (barge-in). */
  flush() {
    for (const src of this.sources) {
      try { src.stop(); } catch { /* already stopped */ }
    }
    this.sources.clear();
    this.nextTime = this.ctx.currentTime;
    this.currentItem = null;
  }
}
