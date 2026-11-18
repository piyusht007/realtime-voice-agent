# Architecture

This document explains how audio, events and control flow move through the system, and why
the orchestration lives in a server-side relay rather than in the browser.

## Components

```mermaid
flowchart LR
    subgraph Browser["Browser (vanilla JS, no build step)"]
        MIC["🎤 getUserMedia<br/>echoCancellation"] --> WL["AudioWorklet<br/>resample → pcm16 24 kHz<br/>40 ms frames"]
        PQ["PcmPlayer<br/>gapless queue<br/>flush() on barge-in"] --> SPK["🔊 speaker"]
        UI["UI: transcript, events,<br/>latency table, handoff"]
    end

    subgraph Relay["Relay server (Python · FastAPI)"]
        WS["/ws endpoint"] <--> ORCH["Orchestrator<br/>state machine"]
        ORCH --> TOOLS["ToolRegistry<br/>order_status · book_callback"]
        ORCH --> HQ["HandoffQueue<br/>(mock contact-center queue)"]
        ORCH --> MET["MetricsSink<br/>JSON turn logs"]
        ORCH <--> RC["RealtimeConnection"]
    end

    subgraph Model["Realtime LLM"]
        OAI["OpenAI Realtime"]
        AZ["Azure OpenAI Realtime"]
        MOCK["MockRealtimeConnection<br/>deterministic, in-process"]
    end

    WL -- "binary pcm16" --> WS
    WS -- "audio deltas (b64), flush,<br/>transcripts, metrics" --> PQ
    WS --> UI
    PQ -. "playback.progress (100 ms)" .-> WS
    RC <-- "JSON events over WSS" --> OAI
    RC <-- "JSON events over WSS" --> AZ
    RC <-- "same protocol, no network" --> MOCK
```

**Why a relay?** The API key never reaches the browser, tools run next to your data, and the
barge-in / filler / handoff logic is plain Python that can be unit-tested with fakes. The
browser stays a thin "audio terminal".

### Orchestrator states

```mermaid
stateDiagram-v2
    [*] --> listening: session.update sent
    listening --> user_speaking: speech_started
    user_speaking --> thinking: speech_stopped / text
    thinking --> assistant_speaking: first audio delta
    thinking --> tool_running: function_call
    tool_running --> assistant_speaking: filler audio (slow tool)
    tool_running --> thinking: result → function_call_output + response.create
    assistant_speaking --> user_speaking: speech_started (BARGE-IN)
    assistant_speaking --> listening: response.done
    thinking --> handing_off: escalate_to_human
    tool_running --> handing_off: escalate_to_human
    handing_off --> ended: farewell response.done / timeout
    ended --> [*]
```

### Key invariants

1. **At most one active response.** The Realtime API rejects `response.create` while a response
   is in flight, so every create goes through `_request_response`, which queues until
   `response.done`. The mock server enforces the same rule, so tests catch violations.
2. **The model's memory matches what the caller heard.** On barge-in the assistant item is
   truncated to `audio_end_ms = played_ms`, never to what was *generated*.
3. **Fillers are out-of-band.** They are created with `conversation: "none"` and `input: []`,
   so they never enter the conversation history and cannot confuse the next turn.
4. **Stale work is dropped.** Queued fillers are discarded when the tool finishes first or when
   the caller barges in; audio deltas for a truncated item are never forwarded.

## Sequence: normal turn

```mermaid
sequenceDiagram
    autonumber
    participant B as Browser
    participant R as Relay / Orchestrator
    participant M as Realtime API
    B->>R: pcm16 frames (binary)
    R->>M: input_audio_buffer.append
    M-->>R: speech_started
    M-->>R: speech_stopped
    Note over R: turn opens, t0 = now
    M-->>R: input_audio_transcription.completed
    R-->>B: transcript (user)
    M-->>R: response.created
    M-->>R: response.audio.delta (first)
    Note over R: first_audio_latency = now - t0
    R-->>B: audio (b64)
    M-->>R: audio_transcript.delta / done
    R-->>B: transcript (assistant)
    M-->>R: response.done
    Note over R: emit turn.metrics JSON
    R-->>B: metrics
```

## Sequence: barge-in

```mermaid
sequenceDiagram
    autonumber
    participant B as Browser
    participant R as Relay / Orchestrator
    participant M as Realtime API
    M-->>R: response.audio.delta × N (generated faster than realtime)
    R-->>B: audio × N (queued in PcmPlayer)
    B-->>R: playback.progress {item_id, played_ms}
    B->>R: caller starts talking (pcm16)
    R->>M: input_audio_buffer.append
    M-->>R: input_audio_buffer.speech_started
    R->>M: response.cancel
    R->>M: conversation.item.truncate {item_id, audio_end_ms = played_ms}
    R-->>B: audio.flush
    Note over B: PcmPlayer.flush() stops all scheduled buffers instantly
    M-->>R: response.done (cancelled)
    M-->>R: late audio.delta for item
    Note over R: dropped (item already truncated)
```

`played_ms` is the latest client report plus time elapsed since it, capped at the audio
actually sent. If the client never reported, the relay falls back to "time since first audio
byte". Server VAD is configured with `interrupt_response: false` so that this single code path
owns cancellation, truncation and flushing.

## Sequence: slow tool with latency prompt

```mermaid
sequenceDiagram
    autonumber
    participant B as Browser
    participant R as Relay / Orchestrator
    participant T as Tool (order_status)
    participant M as Realtime API
    M-->>R: function_call_arguments.done (order_status)
    R->>T: call(order_id) [asyncio task]
    M-->>R: response.done (function-call response)
    Note over R: wait(task, FILLER_THRESHOLD_MS = 700)
    R->>M: response.create {conversation:"none", purpose:"filler"}
    M-->>R: filler audio
    R-->>B: audio "One moment while I check that for you."
    T-->>R: result (1.2 s)
    R->>M: conversation.item.create function_call_output
    Note over R: answer response queued until filler response.done
    M-->>R: response.done (filler)
    R->>M: response.create {purpose:"answer"}
    M-->>R: answer audio
    R-->>B: audio "Order 1042 has shipped…"
```

If the tool returns before the threshold, no filler is sent. If it returns after the threshold
but before the filler could start (because another response was still active), the queued
filler is discarded.

## Sequence: handoff to a human

```mermaid
sequenceDiagram
    autonumber
    participant B as Browser
    participant R as Relay / Orchestrator
    participant Q as Handoff queue
    participant M as Realtime API
    M-->>R: function_call_arguments.done (escalate_to_human {reason})
    Note over R: stop forwarding mic audio
    R->>R: build summary: transcript + intent + sentiment + tools + latency
    R->>Q: publish(summary)
    R-->>B: handoff {summary}
    R->>M: function_call_output {status: transferring, ticket_id}
    R->>M: response.create {purpose:"farewell"} (queued if needed)
    M-->>R: farewell audio ("transferring you to a colleague…")
    R-->>B: audio
    M-->>R: response.done (farewell)
    R->>M: close()
    R-->>B: session.ended {reason: handoff}
```

A safety timer (`HANDOFF_FAREWELL_TIMEOUT_S`) ends the session even if the farewell never
completes, so a caller can never be stranded with a half-open AI session.

## Latency budget (where the milliseconds go)

| Stage | Typical | Lever in this repo |
|---|---|---|
| Mic frame buffering | 40 ms | `FRAME_SAMPLES` in the worklet |
| End-of-speech detection | 300–700 ms | `VAD_SILENCE_MS` (biggest single lever) |
| Model time-to-first-audio | 300–800 ms | model choice, short instructions |
| Tool execution | 0–∞ | async tools + filler after `FILLER_THRESHOLD_MS` |
| Playback scheduling | ~20 ms | `PcmPlayer` schedules 20 ms ahead |

`turn.metrics.first_audio_latency_ms` measures from `speech_stopped` (or text submit) to the
first audio byte the relay forwards, which is what the caller perceives as "dead air".
`answer_latency_ms` measures to the first byte of the *real* answer, so the effect of the
filler is visible as the gap between the two.
