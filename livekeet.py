#!/usr/bin/env python3
"""
livekeet - Real-time audio transcription to markdown.
Optimized for Apple Silicon using NVIDIA Parakeet via MLX.
"""

import argparse
import re
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
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue
from threading import Thread

# Suppress known warnings before importing ML libraries
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
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
#   mlx-community/parakeet-tdt-0.6b-v2  - Fast, English only (default)
#   mlx-community/parakeet-tdt-0.6b-v3  - Fast, multilingual
#   mlx-community/parakeet-tdt-1.1b     - Slower, English, highest accuracy
model = "mlx-community/parakeet-tdt-0.6b-v2"
"""


def load_config() -> dict:
    """Load configuration from file, creating default if needed."""
    config = {
        "output": {"directory": "", "filename": "{datetime}.md"},
        "speaker": {"name": "Me"},
        "defaults": {"model": "mlx-community/parakeet-tdt-0.6b-v2"},
    }

    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "rb") as f:
                user_config = tomllib.load(f)
            # Merge user config
            for section in config:
                if section in user_config:
                    config[section].update(user_config[section])
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
  parakeet-tdt-0.6b-v2  Fast, English only (default)
  parakeet-tdt-0.6b-v3  Fast, multilingual (--multilingual)
  parakeet-tdt-1.1b     Slower, English, highest accuracy""")


def resolve_output_path(config: dict, output_arg: str | None) -> Path:
    """Resolve the output file path from config and CLI args."""
    if output_arg:
        # CLI argument takes precedence
        path = Path(output_arg)
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
        print("Warning: --with is ignored in --mic-only mode", file=sys.stderr)


class AudioCaptureProcess:
    """Captures system audio + microphone via ScreenCaptureKit (Swift subprocess).

    Output is stereo: left channel = mic (you), right channel = system (other).
    """

    # Keywords that indicate an error message worth showing
    ERROR_KEYWORDS = ["error", "failed", "denied", "permission", "not found", "cannot", "unable", "warning"]

    def __init__(self, include_mic: bool = True):
        self.include_mic = include_mic
        self.process: subprocess.Popen | None = None
        self.running = False
        self._stderr_thread: Thread | None = None

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

        Returns:
            Tuple of (mic_audio, system_audio) as float32 arrays, or None if no data.
            Each array is mono with num_samples samples.
        """
        if not self.process or not self.running:
            return None

        # Stereo: 2 channels * 2 bytes per sample
        num_bytes = num_samples * 2 * 2
        try:
            pcm_bytes = self.process.stdout.read(num_bytes)
            if not pcm_bytes or len(pcm_bytes) < num_bytes:
                return None

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
        device=None,
        system_audio: bool = True,
        status_enabled: bool = False,
    ):
        self.output_file = output_file
        self.device = device
        self.speaker_name = speaker_name
        self.other_name = other_name
        self.system_audio = system_audio
        self.status_enabled = status_enabled
        self.audio_queue: Queue = Queue()
        self.running = False
        self.audio_capture: AudioCaptureProcess | None = None
        self.state = "listening"
        self._started_at: float | None = None
        self._last_audio_time: float | None = None
        self._no_audio_warned = False

        short_name = model_name.split("/")[-1] if "/" in model_name else model_name
        print(f"Loading {short_name}...", end=" ", flush=True)
        from parakeet_mlx import from_pretrained
        self.model = from_pretrained(model_name)
        print("ready")

        # VAD setup
        self.vad = webrtcvad.Vad(2)
        self.vad_frame_size = int(SAMPLE_RATE * VAD_FRAME_MS / 1000)

    def audio_callback(self, indata, frames, time_info, status):
        """Called by sounddevice for each audio chunk (mic-only mode)."""
        if status:
            print(f"Audio status: {status}", file=sys.stderr)

        if indata.shape[1] > 1:
            mono = np.mean(indata, axis=1)
        else:
            mono = indata.flatten()
        self._last_audio_time = time.monotonic()
        # In mic-only mode: (mic_audio, None) - no system audio
        self.audio_queue.put((mono.astype(np.float32), None))

    def system_audio_reader(self):
        """Background thread that reads stereo audio from the capture subprocess."""
        chunk_samples = 1024
        max_queue_size = 200
        while self.running and self.audio_capture:
            result = self.audio_capture.read_chunk(chunk_samples)
            if result is not None:
                mic_audio, system_audio = result
                self._last_audio_time = time.monotonic()
                while self.audio_queue.qsize() > max_queue_size:
                    try:
                        self.audio_queue.get_nowait()
                    except:
                        break
                # Queue both channels separately for per-frame energy calculation
                self.audio_queue.put((mic_audio, system_audio))

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

    def _is_speech(self, audio_chunk: np.ndarray) -> bool:
        """Check if audio chunk contains speech using VAD."""
        audio_int16 = (audio_chunk * 32767).astype(np.int16)
        audio_bytes = struct.pack(f"{len(audio_int16)}h", *audio_int16)
        try:
            return self.vad.is_speech(audio_bytes, SAMPLE_RATE)
        except Exception:
            return False

    def _transcribe_audio(self, audio: np.ndarray) -> str:
        """Transcribe audio segment."""
        if len(audio) < SAMPLE_RATE * 0.3:
            return ""

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
            with wave.open(f, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(SAMPLE_RATE)
                audio_int16 = (audio * 32767).astype(np.int16)
                wav.writeframes(audio_int16.tobytes())

        try:
            result = self.model.transcribe(temp_path)
            return result.text.strip()
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def _determine_speaker(self, mic_energy: float, sys_energy: float) -> str | None:
        """Determine who was speaking based on channel energy.

        Returns speaker name, or None if uncertain (35-65% split).
        """
        total = mic_energy + sys_energy

        if total < 1e-6:
            return None  # Too quiet to determine

        mic_ratio = mic_energy / total

        # Use 65/35 threshold for clearer separation
        if mic_ratio > 0.65:
            return self.speaker_name
        elif mic_ratio < 0.35:
            return self.other_name
        else:
            # Ambiguous - could be crosstalk, echo, or both speaking
            return None

    def transcribe_worker(self):
        """Background thread that detects speech end and transcribes.

        Keeps mic and system audio in separate buffers to calculate energy
        per VAD frame accurately for speaker detection.
        """
        # Separate buffers for each channel
        mic_buffer = np.array([], dtype=np.float32)
        sys_buffer = np.array([], dtype=np.float32)

        # Speech accumulation buffers
        speech_buffer = np.array([], dtype=np.float32)  # Mixed audio for transcription
        silence_frames = 0
        in_speech = False
        silence_threshold_frames = int(SILENCE_THRESHOLD * SAMPLE_RATE / self.vad_frame_size)

        # Track energy per speech frame for speaker detection
        total_mic_energy = 0.0
        total_sys_energy = 0.0
        has_system_audio = False  # Track if we're in system audio mode

        while self.running:
            try:
                queue_item = self.audio_queue.get(timeout=0.1)
                mic_chunk, sys_chunk = queue_item

                # Append to channel buffers
                mic_buffer = np.append(mic_buffer, mic_chunk)
                if sys_chunk is not None:
                    sys_buffer = np.append(sys_buffer, sys_chunk)
                    has_system_audio = True

                # Process in VAD frame-sized chunks
                while len(mic_buffer) >= self.vad_frame_size:
                    mic_frame = mic_buffer[:self.vad_frame_size]
                    mic_buffer = mic_buffer[self.vad_frame_size:]

                    if has_system_audio and len(sys_buffer) >= self.vad_frame_size:
                        sys_frame = sys_buffer[:self.vad_frame_size]
                        sys_buffer = sys_buffer[self.vad_frame_size:]
                        # Average the channels (not sum) to keep in [-1, 1] range for VAD
                        mixed_frame = (mic_frame + sys_frame) * 0.5
                    else:
                        sys_frame = None
                        mixed_frame = mic_frame

                    is_speech = self._is_speech(mixed_frame)

                    if is_speech:
                        speech_buffer = np.append(speech_buffer, mixed_frame)
                        silence_frames = 0
                        in_speech = True
                        self.state = "listening"

                        # Calculate energy for THIS frame only (no double counting)
                        total_mic_energy += np.sum(mic_frame ** 2)
                        if sys_frame is not None:
                            total_sys_energy += np.sum(sys_frame ** 2)

                    elif in_speech:
                        speech_buffer = np.append(speech_buffer, mixed_frame)
                        silence_frames += 1

                        if silence_frames >= silence_threshold_frames:
                            self.state = "transcribing"
                            text = self._transcribe_audio(speech_buffer)
                            self.state = "listening"
                            if text:
                                speaker = self._determine_speaker(total_mic_energy, total_sys_energy)
                                self._write_transcript(text, speaker=speaker)

                            # Reset for next utterance
                            speech_buffer = np.array([], dtype=np.float32)
                            silence_frames = 0
                            in_speech = False
                            total_mic_energy = 0.0
                            total_sys_energy = 0.0

            except Empty:
                continue
            except Exception as e:
                print(f"Transcription error: {e}", file=sys.stderr)

    def _write_transcript(self, text: str, speaker: str | None = None):
        """Append transcribed text to output file."""
        timestamp = datetime.now().strftime("%H:%M:%S")

        if speaker:
            line = f"[{timestamp}] **{speaker}**: {text}\n"
            console = f"[{timestamp}] {speaker}: {text}"
        else:
            line = f"[{timestamp}] {text}\n"
            console = f"[{timestamp}] {text}"

        with open(self.output_file, "a") as f:
            f.write(line)
            f.flush()
        print(console)

    def start(self):
        """Start transcription."""
        with open(self.output_file, "w") as f:
            f.write(f"# Transcription - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        self.running = True
        self.state = "listening"
        self._started_at = time.monotonic()
        self._last_audio_time = None
        self._no_audio_warned = False

        worker_thread = Thread(target=self.transcribe_worker, daemon=True)
        worker_thread.start()
        status_thread = Thread(target=self._status_worker, daemon=True)
        status_thread.start()

        if self.system_audio:
            print(f"Recording → {self.output_file} ({self.speaker_name} / {self.other_name})")
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
            self.stop()

    def stop(self):
        """Stop transcription and cleanup."""
        self.running = False
        if self.audio_capture:
            self.audio_capture.stop()
            self.audio_capture = None
        with open(self.output_file, "a") as f:
            f.write(f"\n---\n*Ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
        print(f"\nSaved: {self.output_file}")


def list_devices():
    """Print available audio input devices."""
    print("Available audio input devices:\n")
    default_name = sd.query_devices(kind="input")["name"]
    for i, device in get_input_devices():
        default = " (default)" if device["name"] == default_name else ""
        print(f"  {i}: {device['name']}{default}")
    print("\nUse --device <number or name> to select")


def main():
    parser = argparse.ArgumentParser(
        description="Real-time audio transcription to markdown",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  livekeet                  Start transcribing (uses config defaults)
  livekeet meeting.md       Output to meeting.md
  livekeet --with "John"    Label other speaker as "John"
  livekeet --mic-only       Only capture microphone (no system audio)
  livekeet --init           Create config file
        """,
    )
    parser.add_argument(
        "output",
        nargs="?",
        help="Output file (default: from config, usually {datetime}.md)",
    )
    parser.add_argument(
        "--with", "-w",
        dest="other_speaker",
        metavar="NAME",
        help="Name of the other speaker (for calls)",
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
            "mlx-community/parakeet-tdt-1.1b",
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

    # Get speaker names
    speaker_name = config["speaker"]["name"]
    other_name = args.other_speaker or "Other"

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
        device=device,
        system_audio=system_audio,
        status_enabled=args.status,
    )
    transcriber.start()


if __name__ == "__main__":
    main()
