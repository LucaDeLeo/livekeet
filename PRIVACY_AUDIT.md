# Privacy Audit

Livekeet aims to run 100% locally by default. The one exception is the optional Claude Haiku transcript-cleanup feature (`--cleanup`), which calls the Anthropic API via the `claude-runner` package. Cleanup is **opt-in** and **off by default**.

This document is both the audit prompt (so anyone can re-verify the claims) and the record of the most recent verification.

## How to verify

Paste the following prompt into Claude Code (or any AI with codebase access) from the repo root:

> Review the livekeet codebase and confirm which features make network calls and which run locally. Specifically verify:
>
> 1. **Speech-to-text** — `parakeet-mlx` inference runs entirely on-device. No audio is sent to any server.
> 2. **Audio capture** — microphone via `sounddevice`, system audio via the Swift `audiocapture` subprocess (ScreenCaptureKit). No audio is streamed or uploaded.
> 3. **Voice activity detection** — `webrtcvad` runs entirely on-device.
> 4. **Speaker diarization** — WeSpeaker (real-time) and pyannote (batch) both run entirely on-device. The pyannote pipeline uses a HuggingFace token only to authenticate the local model download; no audio is sent to HuggingFace.
> 5. **File output** — transcripts are written as local markdown; optional raw audio dumps (`--dump-audio`) are local WAVs. No cloud sync.
> 6. **Transcript cleanup (opt-in)** — when `--cleanup` is set (or `[cleanup] enabled = true` in config), each transcribed turn is sent to Claude Haiku via the `claude-runner` package, which reuses the user's local Claude Code authentication. Confirm this is the only code path that leaves the device with user content and that it is disabled by default.
> 7. **Model downloads** — STT models (Parakeet) and diarization models (WeSpeaker, pyannote) are fetched from Hugging Face on first use. No user data is sent.
> 8. **Update check** — `livekeet update` queries PyPI/GitHub for the latest version string. No user data is sent.
> 9. **No analytics or telemetry** — confirm there are no tracking SDKs (Firebase, Sentry, Mixpanel, Amplitude, PostHog, Datadog, Segment, Rollbar) anywhere in the codebase.
>
> For each item, check the relevant source files and identify every `urllib`/`requests`/`http`/`subprocess` call that hits the network. Output your findings as a checklist with pass/fail and a brief justification, citing file:line refs.

---

## Most recent audit

**Date:** 2026-04-19
**Auditor:** Claude Code (Opus 4.7)
**Commit:** `5e8c821`

### Core features (run locally)

| # | Feature | Files checked | Result |
|---|---------|--------------|--------|
| 1 | Speech-to-text | `livekeet.py` (`_transcribe_audio`, `Transcriber.__init__` → `parakeet_mlx.from_pretrained`) | Pass — MLX inference in-process. No network call in the transcription path. |
| 2 | Audio capture | `livekeet.py` (`AudioCaptureProcess`), `audiocapture/Sources/audiocapture/main.swift` | Pass — AVAudioEngine (mic) and ScreenCaptureKit (system). Swift subprocess writes raw PCM to stdout only. |
| 3 | Voice activity detection | `livekeet.py` (webrtcvad usage) | Pass — `webrtcvad` is a local C extension. |
| 4 | Speaker diarization | `diarization.py` (WeSpeaker), `diarization_pyannote.py` | Pass — local MLX/torch inference. HuggingFace token only used for model download authentication. |
| 5 | File output | `livekeet.py` (`_write_transcript`, `_rebuild_transcript`), `--dump-audio` path (`_dump_segment`) | Pass — local filesystem only. No cloud/iCloud sync. Atomic rewrite via `os.replace`. |
| 6 | No analytics or telemetry | Entire repo | Pass — no imports of tracking SDKs. |

### Network-connected features

| Feature | Default state | Endpoint | Data sent |
|---------|---------------|----------|-----------|
| **Transcript cleanup (Claude Haiku)** | **Opt-in** via `--cleanup` or `[cleanup] enabled = true` (default: `false`) | Anthropic API via `claude-runner` (reuses local Claude Code auth) | One transcribed turn per API call. Model: `claude-haiku-4-5`. See `livekeet_cleanup.py:_cleanup_haiku_async`. |
| Deterministic corrections | Always applied if `[cleanup.corrections]` is set | None (local regex) | N/A — runs locally, no network. |
| Model downloads | Automatic on first use of a new model | Hugging Face (mlx-community, pyannote, WeSpeaker) | No user content — only the model id being requested. |
| Audio capture binary download | Automatic on first run if the Swift binary isn't present | GitHub releases (`download_audiocapture` in `livekeet.py`) | No user content — only the version being fetched. |
| Update check | On startup (`check_for_update` in `livekeet.py`) | PyPI / GitHub version check | No user content — only the version being requested. |

### Verdict

**Default installation runs locally for transcription, diarization, and file output.** The only feature that transmits user content is the Claude Haiku cleanup path, which is disabled by default. Users who need a fully-offline workflow should leave `--cleanup` off and not set `[cleanup] enabled = true` in config.
