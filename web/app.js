// Browser client: mic -> pcm16 frames over WebSocket; server audio -> flushable playback queue.
import { PcmPlayer, base64ToInt16, SAMPLE_RATE } from "./audio/player.js";

const $ = (id) => document.getElementById(id);
const ui = {
  connect: $("connect"), hangup: $("hangup"), form: $("text-form"), input: $("text-input"),
  phase: $("phase"), provider: $("provider"), transcript: $("transcript"), events: $("events"),
  metrics: $("metrics"), handoffPanel: $("handoff-panel"), handoff: $("handoff"),
};

let ws = null;
let audioCtx = null; // 24 kHz playback context
let captureCtx = null; // native-rate capture context; the worklet resamples to 24 kHz
let player = null;
let micStream = null;
let progressTimer = null;
const bubbles = new Map();

function logEvent(text, cls = "") {
  const li = document.createElement("li");
  li.textContent = `${new Date().toLocaleTimeString()}  ${text}`;
  if (cls) li.className = cls;
  ui.events.prepend(li);
}

function upsertBubble(role, itemId, text, final) {
  const key = itemId || `${role}-${Date.now()}`;
  let el = bubbles.get(key);
  if (!el) {
    el = document.createElement("div");
    el.className = `msg ${role}`;
    bubbles.set(key, el);
    ui.transcript.append(el);
  }
  el.textContent = text;
  el.classList.toggle("partial", !final);
  ui.transcript.scrollTop = ui.transcript.scrollHeight;
}

function addMetricsRow(m) {
  const tr = document.createElement("tr");
  const tools = m.tool_calls.map((t) => `${t.name} ${Math.round(t.duration_ms)}ms${t.filler_played ? " +filler" : ""}`).join(", ");
  const fmt = (v) => (v == null ? "—" : `${Math.round(v)} ms`);
  for (const v of [m.turn_id, m.source, fmt(m.first_audio_latency_ms) + (m.first_audio_was_filler ? " (filler)" : ""), fmt(m.answer_latency_ms), tools || "—", m.barge_in ? "yes" : ""]) {
    const td = document.createElement("td");
    td.textContent = v;
    tr.append(td);
  }
  ui.metrics.prepend(tr);
}

function handleServer(msg) {
  switch (msg.type) {
    case "session.started":
      logEvent(`session ${msg.session_id} started`);
      break;
    case "state":
      ui.phase.textContent = msg.phase;
      break;
    case "audio":
      player.enqueue(msg.item_id, base64ToInt16(msg.delta));
      break;
    case "audio.flush":
      player.flush();
      logEvent(`barge-in: flushed playback (heard ${msg.played_ms} ms)`, "warn");
      break;
    case "transcript":
      upsertBubble(msg.role, msg.item_id, msg.text, msg.final);
      break;
    case "filler":
      logEvent(`tool slow → latency prompt: "${msg.text}"`, "warn");
      break;
    case "tool":
      logEvent(msg.status === "started" ? `tool ${msg.name} started` : `tool ${msg.name} finished in ${Math.round(msg.duration_ms)} ms`);
      break;
    case "metrics":
      addMetricsRow(msg);
      break;
    case "handoff":
      ui.handoffPanel.hidden = false;
      ui.handoff.textContent = JSON.stringify(msg.summary, null, 2);
      logEvent(`handoff queued: ${msg.summary.ticket_id} (${msg.summary.intent}, ${msg.summary.sentiment.label})`, "warn");
      break;
    case "session.ended":
      logEvent(`session ended: ${msg.reason}`);
      stopMic();
      break;
    case "error":
      logEvent(`error: ${msg.error?.message || JSON.stringify(msg.error)}`, "warn");
      break;
    default:
      break;
  }
}

async function startMic() {
  micStream = await navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
  });
  captureCtx = new AudioContext();
  await captureCtx.audioWorklet.addModule("/static/audio/capture-worklet.js");
  const source = captureCtx.createMediaStreamSource(micStream);
  const node = new AudioWorkletNode(captureCtx, "pcm-capture");
  node.port.onmessage = (e) => {
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(e.data);
  };
  source.connect(node);
}

function stopMic() {
  micStream?.getTracks().forEach((t) => t.stop());
  micStream = null;
  captureCtx?.close();
  captureCtx = null;
  clearInterval(progressTimer);
  ui.connect.disabled = false;
  ui.hangup.disabled = true;
  ui.input.disabled = true;
  ui.form.querySelector("button").disabled = true;
}

async function connect() {
  ui.connect.disabled = true;
  audioCtx = new AudioContext({ sampleRate: SAMPLE_RATE });
  player = new PcmPlayer(audioCtx);
  const health = await fetch("/healthz").then((r) => r.json()).catch(() => ({}));
  ui.provider.textContent = health.provider ? `provider: ${health.provider}` : "";

  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.binaryType = "arraybuffer";
  ws.onmessage = (e) => handleServer(JSON.parse(e.data));
  ws.onclose = () => { logEvent("socket closed"); stopMic(); };
  ws.onopen = async () => {
    ui.hangup.disabled = false;
    ui.input.disabled = false;
    ui.form.querySelector("button").disabled = false;
    // Tell the server how much of the current assistant item was actually heard.
    progressTimer = setInterval(() => {
      if (player.currentItem && player.isPlaying() && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: "playback.progress", item_id: player.currentItem, played_ms: player.playedMs(player.currentItem) }));
      }
    }, 100);
    try {
      await startMic();
      logEvent("microphone live — start talking");
    } catch (err) {
      logEvent(`mic unavailable (${err.message}); use the text box`, "warn");
    }
  };
}

ui.connect.onclick = () => connect().catch((e) => { logEvent(e.message, "warn"); ui.connect.disabled = false; });
ui.hangup.onclick = () => {
  ws?.send(JSON.stringify({ type: "hangup" }));
  player?.flush();
};
ui.form.onsubmit = (e) => {
  e.preventDefault();
  const text = ui.input.value.trim();
  if (!text || !ws) return;
  player.flush();
  ws.send(JSON.stringify({ type: "text", text }));
  upsertBubble("user", null, text, true);
  ui.input.value = "";
};
