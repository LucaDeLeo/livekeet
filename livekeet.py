#!/usr/bin/env python3
"""
livekeet - Real-time audio transcription to markdown.
Optimized for Apple Silicon using NVIDIA Parakeet via MLX.
"""

import argparse
import os
import re
import signal
import stat
import struct
import subprocess
import sys
import tempfile
import time
import tomllib
import urllib.request
import warnings
import wave
from datetime import datetime, timedelta
from pathlib import Path
from queue import Empty, Queue
from threading import Lock, Thread

# Suppress known warnings before importing ML libraries
warnings.filterwarnings("ignore", message="torchcodec is not installed")
warnings.filterwarnings("ignore", message="std\\(\\): degrees of freedom")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered")

import numpy as np
import sounddevice as sd
import webrtcvad

# Audio settings
SAMPLE_RATE = 16000
VAD_FRAME_MS = 30
SILENCE_THRESHOLD = 0.6  # seconds of silence to end utterance
MIN_SPEECH_DURATION = 0.5  # minimum seconds of speech frames to transcribe
MIN_AUDIO_ENERGY = 0.005  # RMS energy floor to reject near-silent segments
STATUS_INTERVAL = 10  # seconds between status updates
NO_AUDIO_WARNING_SECONDS = 10

# Paths
DATA_DIR = Path.home() / ".local" / "share" / "livekeet"
CONFIG_DIR = Path.home() / ".config" / "livekeet"
CONFIG_FILE = CONFIG_DIR / "config.toml"

# Audio capture binary - check local (dev) first, then user data dir
_LOCAL_AUDIOCAPTURE = Path(__file__).parent / "audiocapture" / ".build" / "release" / "audiocapture"
_INSTALLED_AUDIOCAPTURE = DATA_DIR / "audiocapture"

# GitHub release URL for pre-built binary
GITHUB_RELEASE_URL = "https://github.com/LucaDeLeo/livekeet/releases/latest/download/audiocapture"

# Update checking
_GITHUB_REPO = "LucaDeLeo/livekeet"
_PACKAGE_NAME = "livekeet"
_UPDATE_CACHE = CONFIG_DIR / "update_check.json"


def _installed_version() -> str:
    from importlib.metadata import version
    return version(_PACKAGE_NAME)


def _latest_version() -> str | None:
    """Fetch latest version from GitHub (3s timeout)."""
    import json as _json
    url = f"https://raw.githubusercontent.com/{_GITHUB_REPO}/main/pyproject.toml"
    try:
        with urllib.request.urlopen(url, timeout=3) as resp:
            content = resp.read().decode()
        match = re.search(r'version\s*=\s*"([^"]+)"', content)
        return match.group(1) if match else None
    except Exception:
        return None


def _read_update_cache() -> dict:
    try:
        import json as _json
        return _json.loads(_UPDATE_CACHE.read_text()) if _UPDATE_CACHE.exists() else {}
    except Exception:
        return {}


def _write_update_cache(latest: str) -> None:
    try:
        import json as _json
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        _UPDATE_CACHE.write_text(_json.dumps({
            "latest_version": latest,
            "checked_at": time.time(),
        }))
    except Exception:
        pass


def check_for_update() -> None:
    """Print a notice to stderr if an update is available. Cached for 24h."""
    try:
        cache = _read_update_cache()
        if time.time() - cache.get("checked_at", 0) < 86400:
            latest = cache.get("latest_version")
        else:
            latest = _latest_version()
            if latest:
                _write_update_cache(latest)
        if latest and latest != _installed_version():
            print(
                f"Update available: {_installed_version()} → {latest}. "
                f"Run `livekeet update` to update.",
                file=sys.stderr,
            )
    except Exception:
        pass


def run_update() -> None:
    """Check for and install updates."""
    current = _installed_version()
    print(f"Current version: {current}")
    print("Checking for updates...")
    latest = _latest_version()
    if latest is None:
        print("Could not check for updates. Are you online?")
        sys.exit(1)
    if latest == current:
        print("Already up to date.")
        _write_update_cache(latest)
        return
    print(f"Updating: {current} → {latest}")
    result = subprocess.run(
        ["uv", "tool", "install", "--force",
         f"git+https://github.com/{_GITHUB_REPO}.git"],
    )
    if result.returncode == 0:
        print(f"\nUpdated to v{latest}.")
        _write_update_cache(latest)
    else:
        print(f"\nUpdate failed. Try manually:")
        print(f"  uv tool install --force git+https://github.com/{_GITHUB_REPO}.git")
        sys.exit(1)


def get_audiocapture_path() -> Path:
    """Get path to audiocapture binary, downloading if needed."""
    # Prefer local build (development)
    if _LOCAL_AUDIOCAPTURE.exists():
        return _LOCAL_AUDIOCAPTURE

    # Use installed binary
    if _INSTALLED_AUDIOCAPTURE.exists():
        return _INSTALLED_AUDIOCAPTURE

    # Need to download
    return _INSTALLED_AUDIOCAPTURE


def download_audiocapture() -> bool:
    """Download the audiocapture binary from GitHub releases."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading audio capture tool...", end=" ", flush=True)
    try:
        urllib.request.urlretrieve(GITHUB_RELEASE_URL, _INSTALLED_AUDIOCAPTURE)
        # Make executable
        _INSTALLED_AUDIOCAPTURE.chmod(_INSTALLED_AUDIOCAPTURE.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        print("done")
        return True
    except Exception as e:
        print(f"failed: {e}")
        return False

# Default configuration
DEFAULT_CONFIG = """\
# livekeet configuration

[output]
# Directory for transcripts (empty = current directory)
directory = ""
# Filename pattern: {date}, {time}, {datetime}, or any static name
# Examples: "{datetime}.md", "{date}-meeting.md", "transcript.md"
filename = "{datetime}.md"

[speaker]
# Your name in transcripts (when using system audio for calls)
name = "Me"

[defaults]
# Available models (downloaded automatically on first use):
#   mlx-community/parakeet-tdt-0.6b-v2 - English, highest accuracy (default)
#   mlx-community/parakeet-tdt-0.6b-v3  - Multilingual, 25 languages
model = "mlx-community/parakeet-tdt-0.6b-v2"
# diarize = false
# engine = "wespeaker"  # or "pyannote" for batch diarization

# [pyannote]
# token = "hf_..."  # HuggingFace token (or set HF_TOKEN env var)
"""


def load_config() -> dict:
    """Load configuration from file, creating default if needed."""
    config = {
        "output": {"directory": "", "filename": "{datetime}.md"},
        "speaker": {"name": "Me"},
        "defaults": {"model": "mlx-community/parakeet-tdt-0.6b-v2", "diarize": False, "engine": "wespeaker"},
    }

    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "rb") as f:
                user_config = tomllib.load(f)
            # Merge user config
            for section in config:
                if section in user_config:
                    config[section].update(user_config[section])
            # Preserve extra sections (e.g. [pyannote])
            for section in user_config:
                if section not in config:
                    config[section] = user_config[section]
        except Exception as e:
            print(f"Warning: Could not load config: {e}", file=sys.stderr)

    return config


def init_config() -> None:
    """Create default configuration file."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    if CONFIG_FILE.exists():
        print(f"Config already exists: {CONFIG_FILE}")
    else:
        with open(CONFIG_FILE, "w") as f:
            f.write(DEFAULT_CONFIG)
        print(f"Created: {CONFIG_FILE}")

    print("""
Settings:
  speaker.name     Your name in transcripts
  output.directory Where to save files (default: current dir)
  output.filename  Pattern: {date}, {time}, {datetime}
  defaults.model   Speech recognition model

Models (downloaded on first use):
  parakeet-tdt-0.6b-v2  English, highest accuracy (default)
  parakeet-tdt-0.6b-v3  Multilingual, 25 languages (--multilingual)""")


def resolve_output_path(config: dict, output_arg: str | None) -> Path:
    """Resolve the output file path from config and CLI args."""
    if output_arg:
        # CLI argument takes precedence
        path = Path(output_arg)
        if path.is_dir():
            # Use config filename pattern inside the given directory
            now = datetime.now()
            pattern = config["output"]["filename"]
            filename = pattern.format(
                date=now.strftime("%Y-%m-%d"),
                time=now.strftime("%H-%M-%S"),
                datetime=now.strftime("%Y-%m-%d-%H%M%S"),
            )
            return path / filename
        if not path.suffix:
            path = path.with_suffix(".md")
        return path

    # Use config pattern
    pattern = config["output"]["filename"]
    directory = config["output"]["directory"]

    # Expand pattern
    now = datetime.now()
    filename = pattern.format(
        date=now.strftime("%Y-%m-%d"),
        time=now.strftime("%H-%M-%S"),
        datetime=now.strftime("%Y-%m-%d-%H%M%S"),
    )

    if directory:
        base_dir = Path(directory).expanduser()
        base_dir.mkdir(parents=True, exist_ok=True)
        return base_dir / filename

    return Path(filename)


def ensure_unique_path(path: Path) -> tuple[Path, bool]:
    """Ensure the output path does not overwrite an existing file."""
    if not path.exists():
        return path, False

    suffix = path.suffix
    stem = path.stem
    match = re.match(r"^(.*?)-(\d+)$", stem)
    if match:
        base = match.group(1)
        counter = int(match.group(2)) + 1
    else:
        base = stem
        counter = 2

    while True:
        candidate = path.with_name(f"{base}-{counter}{suffix}")
        if not candidate.exists():
            return candidate, True
        counter += 1


def get_input_devices() -> list[tuple[int, dict]]:
    """Return (index, device) pairs for input-capable devices."""
    devices = sd.query_devices()
    return [(i, device) for i, device in enumerate(devices) if device["max_input_channels"] > 0]


def resolve_device(device_arg: str | int | None) -> tuple[int | str | None, str | None]:
    """Resolve a device arg to a valid input device."""
    if device_arg is None:
        return None, None

    devices = sd.query_devices()
    input_devices = get_input_devices()
    if not input_devices:
        print("Error: No audio input devices found", file=sys.stderr)
        sys.exit(1)

    if isinstance(device_arg, str):
        device_arg = device_arg.strip()
        if device_arg.isdigit():
            device_arg = int(device_arg)

    if isinstance(device_arg, int):
        if device_arg < 0 or device_arg >= len(devices) or devices[device_arg]["max_input_channels"] <= 0:
            print(f"Error: Invalid input device index: {device_arg}", file=sys.stderr)
            list_devices()
            sys.exit(1)
        return device_arg, devices[device_arg]["name"]

    name = str(device_arg).lower()
    exact_matches = [(i, d) for i, d in input_devices if d["name"].lower() == name]
    if exact_matches:
        idx, dev = exact_matches[0]
        return idx, dev["name"]

    partial_matches = [(i, d) for i, d in input_devices if name in d["name"].lower()]
    if len(partial_matches) == 1:
        idx, dev = partial_matches[0]
        return idx, dev["name"]
    if len(partial_matches) > 1:
        print(f"Error: Multiple devices match '{device_arg}':", file=sys.stderr)
        for idx, dev in partial_matches:
            print(f"  {idx}: {dev['name']}", file=sys.stderr)
        sys.exit(1)

    print(f"Error: No input device matches '{device_arg}'", file=sys.stderr)
    list_devices()
    sys.exit(1)


def warn_flag_interactions(args: argparse.Namespace) -> None:
    """Warn about ignored or overridden flags."""
    if args.multilingual and args.model:
        print("Warning: --multilingual overrides --model", file=sys.stderr)
    if args.mic_only and args.other_speaker:
        print("Warning: --with is ignored in --mic-only mode (system audio disabled)", file=sys.stderr)
    if getattr(args, "engine", None) == "pyannote" and not getattr(args, "diarize", False) and not args.other_speaker:
        print("Note: --engine pyannote implies --diarize", file=sys.stderr)


class AudioCaptureProcess:
    """Captures system audio + microphone via ScreenCaptureKit (Swift subprocess).

    Output is stereo: left channel = mic (you), right channel = system (other).
    """

    # Keywords that indicate an error message worth showing
    ERROR_KEYWORDS = ["error", "failed", "denied", "permission", "not found", "cannot", "unable", "warning", "restart"]

    def __init__(self, include_mic: bool = True):
        self.include_mic = include_mic
        self.process: subprocess.Popen | None = None
        self.running = False
        self._stderr_thread: Thread | None = None
        self._read_buffer = bytearray()

    def _stderr_reader(self):
        """Read stderr and only print error-related messages."""
        if not self.process or not self.process.stderr:
            return
        for line in self.process.stderr:
            line_str = line.decode("utf-8", errors="replace").strip()
            # Only show lines that look like errors
            line_lower = line_str.lower()
            if any(kw in line_lower for kw in self.ERROR_KEYWORDS):
                print(f"[audio] {line_str}", file=sys.stderr)

    def start(self) -> None:
        """Start the audio capture subprocess."""
        audiocapture_path = get_audiocapture_path()
        if not audiocapture_path.exists():
            raise FileNotFoundError(
                "Audio capture tool not found. "
                "Run 'make build' or check your internet connection."
            )

        cmd = [str(audiocapture_path)]
        if not self.include_mic:
            cmd.append("--no-mic")

        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,  # Capture stderr to filter errors
            bufsize=0,
        )
        self.running = True

        # Start stderr reader thread to surface errors only
        self._stderr_thread = Thread(target=self._stderr_reader, daemon=True)
        self._stderr_thread.start()

    def read_chunk(self, num_samples: int) -> tuple[np.ndarray, np.ndarray] | None:
        """Read stereo audio samples from subprocess.

        Accumulates partial reads across calls to handle short reads from the pipe.

        Returns:
            Tuple of (mic_audio, system_audio) as float32 arrays, or None on EOF.
            Each array is mono with num_samples samples.
        """
        if not self.process or not self.running:
            return None

        # Stereo: 2 channels * 2 bytes per sample
        num_bytes = num_samples * 2 * 2
        try:
            while len(self._read_buffer) < num_bytes:
                remaining = num_bytes - len(self._read_buffer)
                data = self.process.stdout.read(remaining)
                if not data:
                    return None  # EOF
                self._read_buffer.extend(data)

            pcm_bytes = bytes(self._read_buffer[:num_bytes])
            self._read_buffer = self._read_buffer[num_bytes:]

            # Interleaved stereo: [L0, R0, L1, R1, ...]
            audio_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
            audio_float = audio_int16.astype(np.float32) / 32768.0

            # Deinterleave: left = mic, right = system
            mic_audio = audio_float[0::2]  # Even indices (left channel)
            system_audio = audio_float[1::2]  # Odd indices (right channel)

            return mic_audio, system_audio
        except Exception:
            return None

    def stop(self) -> None:
        """Stop the audio capture subprocess."""
        self.running = False
        self._read_buffer.clear()
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None


class Transcriber:
    """VAD-based transcription with speaker labels for calls.

    When using system audio, detects who is speaking based on audio channel:
    - Left channel (mic) = you (speaker_name)
    - Right channel (system) = other person (other_name)
    """

    def __init__(
        self,
        model_name: str,
        output_file: Path,
        speaker_name: str = "Me",
        other_name: str = "Other",
        other_names: list[str] | None = None,
        device=None,
        system_audio: bool = True,
        status_enabled: bool = False,
        diarize: bool = False,
        engine: str = "wespeaker",
    ):
        self.output_file = output_file
        self.device = device
        self.speaker_name = speaker_name
        self.other_name = other_name
        self.other_names = other_names or []
        self.system_audio = system_audio
        self.status_enabled = status_enabled
        self.diarize = diarize
        self.engine = engine
        self.mic_queue: Queue = Queue()
        self.sys_queue: Queue = Queue()
        self.running = False
        self.audio_capture: AudioCaptureProcess | None = None
        self.state = "listening"
        self._started_at: float | None = None
        self._last_audio_time: float | None = None
        self._no_audio_warned = False
        self._last_system_audio_time: float | None = None
        self._system_audio_warned = False

        short_name = model_name.split("/")[-1] if "/" in model_name else model_name
        print(f"Loading {short_name}...", end=" ", flush=True)
        from parakeet_mlx import from_pretrained
        self.model = from_pretrained(model_name)
        print("ready")

        # Pyannote engine: in-memory PCM buffers for periodic diarization
        self._mic_pcm = bytearray()        # raw int16 PCM, append-only
        self._sys_pcm = bytearray()
        self._pyannote_turns: list | None = None  # latest diarization result
        self._diarizer_thread: Thread | None = None
        self._diarizer_pass = 0
        self._session_start_str: str = ""
        self._config: dict = {}
        self._recording_start: float | None = None
        self._segments: list[tuple[float, str, str, str]] = []  # (offset, text, channel, timestamp)
        if engine == "pyannote":
            from diarization_pyannote import check_pyannote_installed
            check_pyannote_installed()
            print("Pyannote engine selected (periodic live diarization)")

        # Speaker diarization — WeSpeaker (real-time, only for wespeaker engine)
        self.embedder = None
        self.mic_tracker = None
        self.sys_tracker = None
        if diarize and engine == "wespeaker":
            from diarization import SpeakerTracker, load_embedder
            print("Loading speaker embeddings...", end=" ", flush=True)
            self.embedder = load_embedder()
            self.mic_tracker = SpeakerTracker(
                primary_name=speaker_name,
                secondary_prefix="Local",
            )
            self.sys_tracker = SpeakerTracker(
                primary_name=other_name,
                secondary_prefix="Remote",
                secondary_names=other_names[1:] if other_names and len(other_names) > 1 else None,
            )
            print("ready")

        # VAD setup (separate instances — webrtcvad is stateful)
        # Aggressiveness 3 = most aggressive filtering of non-speech
        self.mic_vad = webrtcvad.Vad(3)
        self.sys_vad = webrtcvad.Vad(3)
        self.vad_frame_size = int(SAMPLE_RATE * VAD_FRAME_MS / 1000)

        # Locks for shared resources
        self._model_lock = Lock()
        self._write_lock = Lock()

    def audio_callback(self, indata, frames, time_info, status):
        """Called by sounddevice for each audio chunk (mic-only mode)."""
        if status:
            print(f"Audio status: {status}", file=sys.stderr)

        if indata.shape[1] > 1:
            mono = np.mean(indata, axis=1)
        else:
            mono = indata.flatten()
        mono = mono.astype(np.float32)
        self._last_audio_time = time.monotonic()

        # Save mono audio for pyannote periodic diarization
        if self.engine == "pyannote":
            self._mic_pcm.extend((mono * 32767).astype(np.int16).tobytes())

        self.mic_queue.put(mono)

    def system_audio_reader(self):
        """Background thread that reads stereo audio from the capture subprocess."""
        chunk_samples = 1024
        max_queue_size = 200
        while self.running and self.audio_capture:
            result = self.audio_capture.read_chunk(chunk_samples)
            if result is not None:
                mic_audio, system_audio = result
                self._last_audio_time = time.monotonic()

                # Save per-channel audio for pyannote periodic diarization
                if self.engine == "pyannote":
                    self._mic_pcm.extend((mic_audio * 32767).astype(np.int16).tobytes())
                    self._sys_pcm.extend((system_audio * 32767).astype(np.int16).tobytes())

                # Track system audio presence by RMS energy
                sys_rms = float(np.sqrt(np.mean(system_audio ** 2)))
                if sys_rms > 0.001:
                    self._last_system_audio_time = time.monotonic()
                    if self._system_audio_warned:
                        print("System audio recovered", file=sys.stderr)
                        self._system_audio_warned = False

                for q, chunk in ((self.mic_queue, mic_audio), (self.sys_queue, system_audio)):
                    while q.qsize() > max_queue_size:
                        try:
                            q.get_nowait()
                        except Empty:
                            break
                    q.put(chunk)

    def _status_worker(self):
        """Periodic status updates and no-audio warnings."""
        next_status = time.monotonic() + STATUS_INTERVAL
        while self.running:
            time.sleep(0.2)
            now = time.monotonic()

            if self.status_enabled and now >= next_status:
                print(f"Status: {self.state}...", file=sys.stderr)
                next_status = now + STATUS_INTERVAL

            if (
                not self._no_audio_warned
                and self._started_at is not None
                and self._last_audio_time is None
                and now - self._started_at >= NO_AUDIO_WARNING_SECONDS
            ):
                self._no_audio_warned = True
                if self.system_audio:
                    print(
                        "No audio detected yet. Check Screen Recording permission or try --mic-only.",
                        file=sys.stderr,
                    )
                else:
                    print(
                        "No audio detected yet. Check microphone permission or select a device with --device.",
                        file=sys.stderr,
                    )

            # Detect system audio going silent mid-session
            if (
                self.system_audio
                and not self._system_audio_warned
                and self._last_system_audio_time is not None
                and now - self._last_system_audio_time > NO_AUDIO_WARNING_SECONDS
            ):
                self._system_audio_warned = True
                print(
                    "System audio lost. Stream may have stalled — waiting for recovery.",
                    file=sys.stderr,
                )

    def _is_speech(self, audio_chunk: np.ndarray, vad: webrtcvad.Vad) -> bool:
        """Check if audio chunk contains speech using VAD."""
        audio_int16 = (audio_chunk * 32767).astype(np.int16)
        audio_bytes = struct.pack(f"{len(audio_int16)}h", *audio_int16)
        try:
            return vad.is_speech(audio_bytes, SAMPLE_RATE)
        except Exception:
            return False

    def _transcribe_audio(self, audio: np.ndarray):
        """Transcribe audio segment. Returns AlignedResult or None."""
        if len(audio) < SAMPLE_RATE * MIN_SPEECH_DURATION:
            return None

        # Reject near-silent segments (background noise that slipped past VAD)
        rms = np.sqrt(np.mean(audio ** 2))
        if rms < MIN_AUDIO_ENERGY:
            return None

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
            with wave.open(f, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(SAMPLE_RATE)
                audio_int16 = (audio * 32767).astype(np.int16)
                wav.writeframes(audio_int16.tobytes())

        try:
            with self._model_lock:
                result = self.model.transcribe(temp_path)
            if not result.text.strip():
                return None
            return result
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def _fallback_speaker(self, channel: str) -> str:
        """Return the default speaker label for a channel."""
        return self.speaker_name if channel == "mic" else self.other_name

    def _flush_speech_segment(self, speech_frames: list[np.ndarray], channel: str) -> None:
        """Transcribe and write one completed speech segment."""
        if not speech_frames:
            return

        # Require minimum number of speech frames to avoid noise-triggered hallucinations
        min_frames = int(MIN_SPEECH_DURATION * SAMPLE_RATE / self.vad_frame_size)
        if len(speech_frames) < min_frames:
            return

        speech_audio = np.concatenate(speech_frames)

        # Estimate when speech started in the recording timeline
        # (must be computed before transcription delay shifts monotonic clock)
        speech_offset = None
        if self.engine == "pyannote" and self._recording_start is not None:
            speech_duration = len(speech_audio) / SAMPLE_RATE
            speech_offset = time.monotonic() - self._recording_start - speech_duration

        if self.engine == "pyannote":
            # Store segment for periodic pyannote relabeling
            speaker = self._fallback_speaker(channel)
        elif self.diarize and self.embedder is not None:
            # Mel computation outside lock (CPU-only, ~5ms)
            mel = self.embedder.compute_mel(speech_audio)
            if mel is not None:
                with self._model_lock:
                    embedding = self.embedder.extract_embedding_from_mel(mel)
                if embedding is not None:
                    tracker = self.mic_tracker if channel == "mic" else self.sys_tracker
                    speaker = tracker.identify(embedding)
                else:
                    speaker = self._fallback_speaker(channel)
            else:
                speaker = self._fallback_speaker(channel)
        else:
            speaker = self._fallback_speaker(channel)

        self.state = "transcribing"
        result = self._transcribe_audio(speech_audio)
        self.state = "listening"
        if result is None:
            return

        sentences = result.sentences
        if not sentences:
            return

        if speech_offset is not None:
            # Pyannote path: one segment per sentence with precise audio-aligned offsets
            base_time = datetime.now() - timedelta(seconds=len(speech_audio) / SAMPLE_RATE)
            entries = []
            for sentence in sentences:
                sent_offset = speech_offset + sentence.start
                sent_time = base_time + timedelta(seconds=sentence.start)
                timestamp = sent_time.strftime("%H:%M:%S")
                entries.append((sent_offset, sentence.text.strip(), timestamp))
            with self._write_lock:
                for sent_offset, text, timestamp in entries:
                    self._segments.append((sent_offset, text, channel, timestamp))
                self._rebuild_transcript()
            for _, text, timestamp in entries:
                print(f"[{timestamp}] {speaker}: {text}")
            return

        # Non-pyannote path: one line per sentence with real sentence text
        for sentence in sentences:
            text = sentence.text.strip()
            if text:
                self._write_transcript(text, speaker=speaker)

    def channel_worker(self, queue: Queue, channel: str, vad: webrtcvad.Vad):
        """Background thread that runs VAD + transcription for one audio channel."""
        buffer = np.array([], dtype=np.float32)
        speech_frames: list[np.ndarray] = []
        silence_frames = 0
        in_speech = False
        silence_threshold_frames = int(SILENCE_THRESHOLD * SAMPLE_RATE / self.vad_frame_size)

        while self.running:
            try:
                chunk = queue.get(timeout=0.1)
                buffer = np.append(buffer, chunk)

                while len(buffer) >= self.vad_frame_size:
                    frame = buffer[:self.vad_frame_size]
                    buffer = buffer[self.vad_frame_size:]

                    if self._is_speech(frame, vad):
                        speech_frames.append(frame.copy())
                        silence_frames = 0
                        in_speech = True
                    elif in_speech:
                        silence_frames += 1
                        if silence_frames >= silence_threshold_frames:
                            self._flush_speech_segment(speech_frames, channel)
                            speech_frames = []
                            silence_frames = 0
                            in_speech = False

            except Empty:
                continue
            except Exception as e:
                print(f"Transcription error: {e}", file=sys.stderr)

        if speech_frames:
            self._flush_speech_segment(speech_frames, channel)

    def _write_transcript(self, text: str, speaker: str | None = None):
        """Append transcribed text to output file."""
        timestamp = datetime.now().strftime("%H:%M:%S")

        if speaker:
            line = f"[{timestamp}] **{speaker}**: {text}\n"
            console = f"[{timestamp}] {speaker}: {text}"
        else:
            line = f"[{timestamp}] {text}\n"
            console = f"[{timestamp}] {text}"

        with self._write_lock:
            with open(self.output_file, "a") as f:
                f.write(line)
                f.flush()
            print(console)

    def _resolve_speaker(self, offset: float, channel: str) -> str:
        """Resolve speaker for a segment using pyannote turns or channel fallback."""
        if self._pyannote_turns:
            # Only match turns from the same channel
            turns = [t for t in self._pyannote_turns if t.channel == channel]
            # Find turn containing this offset
            for turn in turns:
                if turn.start <= offset <= turn.end:
                    return turn.speaker
            # Closest turn within 2 seconds
            best = None
            best_dist = float("inf")
            for turn in turns:
                mid = (turn.start + turn.end) / 2
                dist = abs(offset - mid)
                if dist < best_dist:
                    best_dist = dist
                    best = turn.speaker
            if best and best_dist < 2.0:
                return best
        return self._fallback_speaker(channel)

    def _rebuild_transcript(self) -> None:
        """Rewrite the entire .md file from _segments + _pyannote_turns.

        Must be called under _write_lock.
        """
        lines = [f"# Transcription - {self._session_start_str}", ""]

        for offset, text, channel, timestamp in self._segments:
            speaker = self._resolve_speaker(offset, channel)
            lines.append(f"[{timestamp}] **{speaker}**: {text}")

        self.output_file.write_text("\n".join(lines) + "\n")

    @staticmethod
    def _pcm_to_temp_wav(pcm_data: bytes) -> Path:
        """Write raw int16 PCM bytes to a temporary mono WAV file."""
        path = Path(tempfile.mktemp(suffix=".wav"))
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(pcm_data)
        return path

    def _diarizer_worker(self) -> None:
        """Background thread: run pyannote periodically on all audio captured so far."""
        # Wait 30s before first run
        for _ in range(300):
            if not self.running:
                break
            time.sleep(0.1)

        diarizer = None  # lazy-loaded

        def run_pass(final: bool = False) -> None:
            nonlocal diarizer
            audio_seconds = len(self._mic_pcm) / (SAMPLE_RATE * 2)
            if audio_seconds < 30:
                return

            if diarizer is None:
                print("Loading pyannote pipeline...", flush=True)
                from diarization_pyannote import PyannoteDiarizer
                token = self._config.get("pyannote", {}).get("token")
                diarizer = PyannoteDiarizer(hf_token=token)

            if final:
                print("\nRunning final speaker analysis...", flush=True)

            # Snapshot PCM buffers (append-only, safe under GIL)
            mic_data = bytes(self._mic_pcm)
            sys_data = bytes(self._sys_pcm) if self.system_audio else b""

            mic_wav = self._pcm_to_temp_wav(mic_data)
            sys_wav = self._pcm_to_temp_wav(sys_data) if sys_data else None

            try:
                if sys_wav:
                    turns = diarizer.diarize_stereo(
                        mic_wav, sys_wav,
                        self.speaker_name, self.other_name, self.other_names,
                    )
                else:
                    turns = diarizer.diarize_mono(mic_wav, self.speaker_name)

                self._diarizer_pass += 1
                with self._write_lock:
                    self._pyannote_turns = turns
                    self._rebuild_transcript()

                print(
                    f"Speaker labels updated (pass {self._diarizer_pass}, "
                    f"{len(turns)} turns)",
                    flush=True,
                )
            finally:
                mic_wav.unlink(missing_ok=True)
                if sys_wav:
                    sys_wav.unlink(missing_ok=True)

        while self.running:
            try:
                run_pass()
            except Exception as e:
                print(f"Diarization error: {e}", file=sys.stderr)

            # Sleep ~2 min between runs
            for _ in range(1200):
                if not self.running:
                    break
                time.sleep(0.1)

        # Final run after stop
        try:
            run_pass(final=True)
        except Exception as e:
            print(f"Final diarization failed: {e}", file=sys.stderr)

    def start(self, config: dict | None = None):
        """Start transcription.

        Args:
            config: Configuration dict (needed for pyannote token lookup).
        """
        self._session_start_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(self.output_file, "w") as f:
            f.write(f"# Transcription - {self._session_start_str}\n\n")

        self.running = True
        self.state = "listening"
        self._started_at = time.monotonic()
        self._recording_start = time.monotonic()
        self._last_audio_time = None
        self._no_audio_warned = False

        # Start periodic diarizer for pyannote engine
        if self.engine == "pyannote":
            self._config = config or {}
            self._diarizer_thread = Thread(target=self._diarizer_worker, daemon=True)
            self._diarizer_thread.start()

        mic_worker = Thread(
            target=self.channel_worker,
            args=(self.mic_queue, "mic", self.mic_vad),
            daemon=True,
        )
        mic_worker.start()

        if self.system_audio:
            sys_worker = Thread(
                target=self.channel_worker,
                args=(self.sys_queue, "system", self.sys_vad),
                daemon=True,
            )
            sys_worker.start()

        status_thread = Thread(target=self._status_worker, daemon=True)
        status_thread.start()

        if self.system_audio:
            others = ", ".join(self.other_names) if self.other_names else self.other_name
            print(f"Recording → {self.output_file} ({self.speaker_name} / {others})")
        else:
            print(f"Recording → {self.output_file}")
        print("Press Ctrl+C to stop\n")

        try:
            if self.system_audio:
                self.audio_capture = AudioCaptureProcess(include_mic=True)
                self.audio_capture.start()

                reader_thread = Thread(target=self.system_audio_reader, daemon=True)
                reader_thread.start()

                while self.running:
                    time.sleep(0.1)
            else:
                with sd.InputStream(
                    callback=self.audio_callback,
                    channels=1,
                    samplerate=SAMPLE_RATE,
                    dtype=np.float32,
                    device=self.device,
                ):
                    while self.running:
                        sd.sleep(100)
        except KeyboardInterrupt:
            pass
        finally:
            # Ignore further Ctrl+C during cleanup
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            print("\nStopping...")
            self.stop()
            # Give channel workers time to flush remaining segments
            time.sleep(2)
            if self._diarizer_thread is not None:
                self._diarizer_thread.join(timeout=300)
            # Write footer
            with self._write_lock:
                with open(self.output_file, "a") as f:
                    f.write(f"\n---\n*Ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
            print(f"Saved: {self.output_file}")
            if self.engine != "pyannote" and self.system_audio:
                relabel_interactive(self.output_file)

    def stop(self):
        """Stop transcription and audio capture."""
        self.running = False
        if self.audio_capture:
            self.audio_capture.stop()
            self.audio_capture = None


def list_devices():
    """Print available audio input devices."""
    print("Available audio input devices:\n")
    default_name = sd.query_devices(kind="input")["name"]
    for i, device in get_input_devices():
        default = " (default)" if device["name"] == default_name else ""
        print(f"  {i}: {device['name']}{default}")
    print("\nUse --device <number or name> to select")


def relabel_interactive(filepath: str | Path) -> None:
    """Interactively rename speaker labels in a transcript file."""
    filepath = Path(filepath)
    if not filepath.exists():
        print(f"File not found: {filepath}", file=sys.stderr)
        return

    content = filepath.read_text()

    # Find all unique speaker names
    speakers = list(dict.fromkeys(re.findall(r'\*\*(.+?)\*\*:', content)))
    if len(speakers) <= 1:
        return

    print(f"\nRelabel speakers in {filepath.name}:")

    try:
        for speaker in speakers:
            # Gather all transcript lines for this speaker
            quotes = []
            for line in content.splitlines():
                match = re.search(rf'\*\*{re.escape(speaker)}\*\*:\s*(.+)', line)
                if match:
                    text = match.group(1).strip()
                    if text:
                        quotes.append(text[:80] + "..." if len(text) > 80 else text)

            shown = 0
            batch = 3

            while True:
                # Show speaker info and quotes
                end = min(shown + batch, len(quotes))
                if shown == 0:
                    print(f'\nSpeaker "{speaker}" ({len(quotes)} lines):')
                for q in quotes[shown:end]:
                    print(f'  > "{q}"')
                shown = end

                # Build prompt options
                options = "(n) Name  "
                if shown < len(quotes):
                    options += "(m) More  "
                options += "(s) Skip: "

                choice = input(f"\n  {options}").strip().lower()

                if choice == "n":
                    new_name = input("  New name: ").strip()
                    if new_name and new_name != speaker:
                        content = content.replace(f"**{speaker}**", f"**{new_name}**")
                        print(f'  Renamed "{speaker}" → "{new_name}"')
                    break
                elif choice == "m" and shown < len(quotes):
                    continue
                else:
                    break

        filepath.write_text(content)
    except KeyboardInterrupt:
        print("\n\nRelabeling cancelled.")


def _install_stderr_filter():
    """Filter macOS MallocStackLogging noise from MLX child processes.

    Redirects fd 2 through a pipe so a background thread can drop lines
    containing 'MallocStackLogging' before forwarding to real stderr.
    """
    real_fd = os.dup(2)
    r_fd, w_fd = os.pipe()
    os.dup2(w_fd, 2)
    os.close(w_fd)
    sys.stderr = open(2, "w", buffering=1, closefd=False)
    real_stderr = open(real_fd, "w", buffering=1, closefd=True)

    def _reader():
        try:
            with open(r_fd, buffering=1, closefd=True) as pipe:
                for line in pipe:
                    if "MallocStackLogging" not in line:
                        real_stderr.write(line)
                        real_stderr.flush()
        except (OSError, ValueError):
            pass

    Thread(target=_reader, daemon=True).start()


def main():
    _install_stderr_filter()

    # Handle subcommands before argparse
    if len(sys.argv) >= 2 and sys.argv[1] == "update":
        run_update()
        return
    if len(sys.argv) >= 2 and sys.argv[1] == "init":
        init_config()
        return
    if len(sys.argv) >= 2 and sys.argv[1] == "relabel":
        if len(sys.argv) < 3:
            print("Usage: livekeet relabel <file>")
            sys.exit(1)
        relabel_interactive(sys.argv[2])
        return

    parser = argparse.ArgumentParser(
        description="Real-time audio transcription to markdown",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  livekeet                  Start transcribing (uses config defaults)
  livekeet meeting.md       Output to meeting.md
  livekeet meetings/        Output into meetings/ directory
  livekeet --with "John"    Label other speaker as "John"
  livekeet --mic-only       Only capture microphone (no system audio)
  livekeet relabel notes.md  Rename speakers in existing file
  livekeet init              Create config file
  livekeet update            Update to latest version
        """,
    )
    parser.add_argument(
        "--version", action="version",
        version=f"livekeet {_installed_version()}",
    )
    parser.add_argument(
        "output",
        nargs="?",
        help="Output file or directory (default: from config, usually {datetime}.md)",
    )
    parser.add_argument(
        "--with", "-w",
        dest="other_speaker",
        metavar="NAME",
        help="Other speaker name(s), comma-separated (enables diarization when multiple)",
    )
    parser.add_argument(
        "--mic-only", "-m",
        action="store_true",
        help="Only capture microphone (no system audio)",
    )
    parser.add_argument(
        "--multilingual",
        action="store_true",
        help="Use multilingual model (parakeet-tdt-0.6b-v3)",
    )
    parser.add_argument(
        "--device", "-d",
        help="Audio input device (number or name)",
    )
    parser.add_argument(
        "--devices",
        action="store_true",
        help="List available audio devices",
    )
    parser.add_argument(
        "--model",
        choices=[
            "mlx-community/parakeet-tdt-0.6b-v2",
            "mlx-community/parakeet-tdt-0.6b-v3",
        ],
        help="Model to use (default: from config)",
    )
    parser.add_argument(
        "--init",
        action="store_true",
        help="Create default config file",
    )
    parser.add_argument(
        "--config",
        action="store_true",
        help="Show config file location",
    )
    parser.add_argument(
        "--diarize",
        action="store_true",
        help="Identify individual speakers per audio channel",
    )
    parser.add_argument(
        "--engine",
        choices=["wespeaker", "pyannote"],
        default=None,
        help="Diarization engine: wespeaker (real-time) or pyannote (batch, post-session)",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show periodic status updates while recording",
    )

    args = parser.parse_args()

    # Handle utility commands
    if args.init:
        init_config()
        return

    if args.config:
        print(f"Config file: {CONFIG_FILE}")
        if CONFIG_FILE.exists():
            print("(exists)")
        else:
            print("(not created yet - run 'livekeet --init')")
        return

    if args.devices:
        list_devices()
        return

    check_for_update()

    warn_flag_interactions(args)

    # Load config
    config = load_config()

    # Check system audio requirements
    system_audio = not args.mic_only
    if system_audio:
        audiocapture_path = get_audiocapture_path()
        if not audiocapture_path.exists():
            # Try to download
            if not download_audiocapture():
                print("Error: Could not get audio capture tool")
                print("Try: git clone the repo and run 'make build'")
                print("Or use --mic-only to capture microphone only")
                sys.exit(1)

    # Resolve output path
    output_path = resolve_output_path(config, args.output)
    output_path, suffixed = ensure_unique_path(output_path)
    if suffixed:
        print(f"Output exists; saving to {output_path}")

    # Resolve device (mic-only mode)
    device = None
    if args.mic_only and args.device is not None:
        device, device_name = resolve_device(args.device)
        if device_name:
            print(f"Using input device: {device_name}")

    # Parse comma-separated speaker names
    if args.other_speaker:
        other_names = [n.strip() for n in args.other_speaker.split(",") if n.strip()]
    else:
        other_names = []
    other_name = other_names[0] if other_names else "Other"

    # Get speaker names
    speaker_name = config["speaker"]["name"]

    # Resolve engine
    engine = args.engine or config["defaults"].get("engine", "wespeaker")

    # Diarize if flag, config, multiple --with names, or pyannote engine
    diarize = args.diarize or config["defaults"].get("diarize", False) or len(other_names) > 1
    if engine == "pyannote":
        diarize = True

    # Get model
    if args.multilingual:
        model = "mlx-community/parakeet-tdt-0.6b-v3"
    else:
        model = args.model or config["defaults"]["model"]

    # Start transcription
    transcriber = Transcriber(
        model_name=model,
        output_file=output_path,
        speaker_name=speaker_name,
        other_name=other_name,
        other_names=other_names,
        device=device,
        system_audio=system_audio,
        status_enabled=args.status,
        diarize=diarize,
        engine=engine,
    )
    transcriber.start(config=config)


if __name__ == "__main__":
    main()
