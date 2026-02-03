#!/usr/bin/env python3
"""
Real-time audio transcription to markdown file using NVIDIA Parakeet via MLX.
Optimized for Apple Silicon.
"""

import argparse
import os
import struct
import sys
import tempfile
import warnings
import wave
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue
from threading import Thread

# Suppress known warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
warnings.filterwarnings("ignore", message="torchcodec is not installed")
warnings.filterwarnings("ignore", message="std\\(\\): degrees of freedom")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered")

import mlx.core as mx
import numpy as np
import sounddevice as sd
import webrtcvad
from parakeet_mlx import from_pretrained

SAMPLE_RATE = 16000
CHUNK_DURATION = 0.5  # seconds per chunk (lower = faster, less accurate)
VAD_FRAME_MS = 30  # VAD frame size in ms (10, 20, or 30)
SILENCE_THRESHOLD = 0.6  # seconds of silence to end utterance


class StreamingTranscriber:
    def __init__(self, model_name: str, output_file: str, device=None):
        self.output_file = Path(output_file)
        self.audio_queue: Queue = Queue()
        self.running = False
        self.device = device

        print(f"Loading model: {model_name}")
        self.model = from_pretrained(model_name)
        print("Model loaded")

    def audio_callback(self, indata, frames, time, status):
        """Called by sounddevice for each audio chunk."""
        if status:
            print(f"Audio status: {status}", file=sys.stderr)

        # Mix channels: mic (ch0) + system audio (ch1-2 averaged)
        if indata.shape[1] >= 3:
            mic = indata[:, 0] * 2.0  # Boost mic (often quieter)
            system = np.mean(indata[:, 1:3], axis=1)
            mono = (mic + system) / 2.0
        elif indata.shape[1] > 1:
            mono = np.mean(indata, axis=1)
        else:
            mono = indata.flatten()
        self.audio_queue.put(mono.astype(np.float32))

    def transcribe_worker(self):
        """Background thread that processes audio using streaming API."""
        chunk_size = int(SAMPLE_RATE * CHUNK_DURATION)

        with self.model.transcribe_stream(context_size=(128, 128)) as transcriber:  # Smaller context = faster
            buffer = np.array([], dtype=np.float32)
            written_finalized_count = 0
            pending_text = ""  # Buffer for accumulating text

            while self.running:
                try:
                    chunk = self.audio_queue.get(timeout=0.5)
                    buffer = np.append(buffer, chunk)

                    if len(buffer) >= chunk_size:
                        # Convert numpy to MLX array (required for MLX 0.30+)
                        audio_mlx = mx.array(buffer[:chunk_size])
                        transcriber.add_audio(audio_mlx)
                        buffer = buffer[chunk_size:]

                        # Accumulate new finalized tokens
                        finalized = transcriber.finalized_tokens
                        if len(finalized) > written_finalized_count:
                            new_tokens = finalized[written_finalized_count:]
                            new_text = "".join(t.text for t in new_tokens)
                            pending_text += new_text
                            written_finalized_count = len(finalized)

                        # Write when we have a sentence or enough text
                        if pending_text:
                            # Check for sentence endings or sufficient length
                            sentences = []
                            remaining = pending_text

                            for end_char in [". ", "? ", "! ", ".\n", "?\n", "!\n"]:
                                while end_char in remaining:
                                    idx = remaining.index(end_char) + len(end_char)
                                    sentences.append(remaining[:idx].strip())
                                    remaining = remaining[idx:]

                            # Write complete sentences
                            for sentence in sentences:
                                if sentence:
                                    self._write_transcript(sentence)

                            # Keep remainder or write if it's getting long
                            pending_text = remaining
                            if len(pending_text) > 50:  # Write sooner (was 100)
                                self._write_transcript(pending_text.strip())
                                pending_text = ""

                except Empty:
                    # On timeout, flush pending text only if meaningful
                    clean_text = pending_text.strip().strip(".")
                    if clean_text:
                        self._write_transcript(clean_text)
                    pending_text = ""
                    continue
                except Exception as e:
                    print(f"Transcription error: {e}", file=sys.stderr)

    def _write_transcript(self, text: str):
        """Append transcribed text to output file."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{timestamp}] {text}\n"
        with open(self.output_file, "a") as f:
            f.write(line)
            f.flush()
        print(f"[{timestamp}] {text}")

    def start(self):
        """Start streaming transcription."""
        # Initialize output file with header
        with open(self.output_file, "w") as f:
            f.write(f"# Transcription - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        self.running = True

        # Start transcription thread
        worker_thread = Thread(target=self.transcribe_worker, daemon=True)
        worker_thread.start()

        # Start audio stream
        print(f"\nRecording... Output: {self.output_file}")
        print("Press Ctrl+C to stop\n")

        try:
            # Use 3 channels to capture both mic (ch1) and BlackHole (ch2-3) from Aggregate Device
            # Will be mixed down to mono in audio_callback
            num_channels = 3 if "Aggregate" in str(self.device or "") else 1
            with sd.InputStream(
                callback=self.audio_callback,
                channels=num_channels,
                samplerate=SAMPLE_RATE,
                dtype=np.float32,
                device=self.device,
            ):
                while self.running:
                    sd.sleep(100)
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            self.stop()

    def stop(self):
        """Stop transcription and cleanup."""
        self.running = False
        with open(self.output_file, "a") as f:
            f.write(f"\n---\n*Ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
        print(f"Transcript saved to: {self.output_file}")


class VADTranscriber:
    """VAD-based transcription - detects speech end and transcribes immediately (fast like batch)."""

    def __init__(self, model_name: str, output_file: str, device=None, diarize: bool = False):
        self.output_file = Path(output_file)
        self.device = device
        self.audio_queue: Queue = Queue()
        self.running = False
        self.diarize = diarize

        print(f"Loading model: {model_name}")
        self.model = from_pretrained(model_name)
        print("Model loaded")

        # VAD setup - aggressiveness 2 (0-3, higher = more aggressive filtering)
        self.vad = webrtcvad.Vad(2)
        self.vad_frame_size = int(SAMPLE_RATE * VAD_FRAME_MS / 1000)  # samples per VAD frame

    def audio_callback(self, indata, frames, time, status):
        """Called by sounddevice for each audio chunk."""
        if status:
            print(f"Audio status: {status}", file=sys.stderr)

        # Mix channels: mic (ch0) + system audio (ch1-2 averaged)
        if indata.shape[1] >= 3:
            mic = indata[:, 0] * 2.0
            system = np.mean(indata[:, 1:3], axis=1)
            mono = (mic + system) / 2.0

            # For diarization: track which channel has more energy
            if self.diarize:
                mic_energy = np.mean(np.abs(indata[:, 0]))
                system_energy = np.mean(np.abs(np.mean(indata[:, 1:3], axis=1)))
                # Store speaker info with audio: "You" if mic dominates, "Other" if system
                speaker = "You" if mic_energy > system_energy * 0.5 else "Other"
                self.audio_queue.put((mono.astype(np.float32), speaker))
                return
        elif indata.shape[1] > 1:
            mono = np.mean(indata, axis=1)
        else:
            mono = indata.flatten()

        if self.diarize:
            self.audio_queue.put((mono.astype(np.float32), None))
        else:
            self.audio_queue.put(mono.astype(np.float32))

    def _is_speech(self, audio_chunk: np.ndarray) -> bool:
        """Check if audio chunk contains speech using VAD."""
        # Convert float32 to int16 for webrtcvad
        audio_int16 = (audio_chunk * 32767).astype(np.int16)
        audio_bytes = struct.pack(f"{len(audio_int16)}h", *audio_int16)
        try:
            return self.vad.is_speech(audio_bytes, SAMPLE_RATE)
        except Exception:
            return False

    def _transcribe_audio(self, audio: np.ndarray) -> str:
        """Transcribe audio segment using batch transcription (fast)."""
        if len(audio) < SAMPLE_RATE * 0.3:  # Skip very short segments
            return ""

        # Save to temp file
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

    def transcribe_worker(self):
        """Background thread that detects speech end and transcribes."""
        buffer = np.array([], dtype=np.float32)
        speech_buffer = np.array([], dtype=np.float32)
        silence_frames = 0
        in_speech = False
        silence_threshold_frames = int(SILENCE_THRESHOLD * SAMPLE_RATE / self.vad_frame_size)

        # For diarization: track speaker votes during speech segment
        speaker_votes = {"You": 0, "Other": 0}

        while self.running:
            try:
                queue_item = self.audio_queue.get(timeout=0.1)

                # Handle diarization mode
                if self.diarize:
                    chunk, speaker = queue_item
                    if speaker:
                        speaker_votes[speaker] += 1
                else:
                    chunk = queue_item

                buffer = np.append(buffer, chunk)

                # Process in VAD frame-sized chunks
                while len(buffer) >= self.vad_frame_size:
                    frame = buffer[:self.vad_frame_size]
                    buffer = buffer[self.vad_frame_size:]

                    is_speech = self._is_speech(frame)

                    if is_speech:
                        speech_buffer = np.append(speech_buffer, frame)
                        silence_frames = 0
                        in_speech = True
                    elif in_speech:
                        # Still in speech mode but silence detected
                        speech_buffer = np.append(speech_buffer, frame)
                        silence_frames += 1

                        # End of utterance - silence exceeded threshold
                        if silence_frames >= silence_threshold_frames:
                            text = self._transcribe_audio(speech_buffer)
                            if text:
                                # Determine dominant speaker
                                if self.diarize and (speaker_votes["You"] > 0 or speaker_votes["Other"] > 0):
                                    dominant = "You" if speaker_votes["You"] > speaker_votes["Other"] else "Other"
                                    self._write_transcript(text, speaker=dominant)
                                else:
                                    self._write_transcript(text)
                            speech_buffer = np.array([], dtype=np.float32)
                            silence_frames = 0
                            in_speech = False
                            speaker_votes = {"You": 0, "Other": 0}

            except Empty:
                continue
            except Exception as e:
                print(f"Transcription error: {e}", file=sys.stderr)

    def _write_transcript(self, text: str, speaker: str | None = None):
        """Append transcribed text to output file."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        if speaker:
            line = f"[{timestamp}] **{speaker}**: {text}\n"
            print(f"[{timestamp}] {speaker}: {text}")
        else:
            line = f"[{timestamp}] {text}\n"
            print(f"[{timestamp}] {text}")
        with open(self.output_file, "a") as f:
            f.write(line)
            f.flush()

    def start(self):
        """Start VAD-based transcription."""
        # Initialize output file with header
        with open(self.output_file, "w") as f:
            f.write(f"# Transcription - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        self.running = True

        # Start transcription thread
        worker_thread = Thread(target=self.transcribe_worker, daemon=True)
        worker_thread.start()

        # Start audio stream
        print(f"\nRecording with VAD... Output: {self.output_file}")
        print("Speak naturally - transcription happens when you pause")
        print("Press Ctrl+C to stop\n")

        try:
            num_channels = 3 if "Aggregate" in str(self.device or "") else 1
            with sd.InputStream(
                callback=self.audio_callback,
                channels=num_channels,
                samplerate=SAMPLE_RATE,
                dtype=np.float32,
                device=self.device,
            ):
                while self.running:
                    sd.sleep(100)
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            self.stop()

    def stop(self):
        """Stop transcription and cleanup."""
        self.running = False
        with open(self.output_file, "a") as f:
            f.write(f"\n---\n*Ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
        print(f"Transcript saved to: {self.output_file}")


class DiarizedTranscriber(VADTranscriber):
    """VAD-based transcription with pyannote speaker diarization."""

    # Accumulate at least this much audio before diarizing (seconds)
    MIN_DIARIZE_DURATION = 5.0
    # Maximum audio to accumulate before forcing diarization (seconds)
    MAX_DIARIZE_DURATION = 30.0

    def __init__(self, model_name: str, output_file: str, device=None, hf_token: str | None = None):
        # Initialize parent without diarize flag (we handle it differently)
        super().__init__(model_name, output_file, device, diarize=False)

        # Get token from parameter or environment
        token = hf_token or os.environ.get("HF_TOKEN")
        if not token:
            raise ValueError(
                "HuggingFace token required for pyannote diarization.\n"
                "1. Accept terms at: https://huggingface.co/pyannote/speaker-diarization-3.1\n"
                "2. Get token from: https://huggingface.co/settings/tokens\n"
                "3. Use --hf-token <token> or set HF_TOKEN environment variable"
            )

        print("Loading pyannote diarization pipeline...")
        from pyannote.audio import Pipeline

        self.diarization = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=token,
        )
        print("Diarization pipeline loaded")

        # Buffer for accumulating audio across multiple utterances
        self.diarize_buffer = np.array([], dtype=np.float32)
        self.diarize_buffer_start_time = None

    def audio_callback(self, indata, frames, time, status):
        """Called by sounddevice for each audio chunk."""
        if status:
            print(f"Audio status: {status}", file=sys.stderr)

        # Mix channels to mono (same as VADTranscriber without diarize)
        if indata.shape[1] >= 3:
            mic = indata[:, 0] * 2.0
            system = np.mean(indata[:, 1:3], axis=1)
            mono = (mic + system) / 2.0
        elif indata.shape[1] > 1:
            mono = np.mean(indata, axis=1)
        else:
            mono = indata.flatten()
        self.audio_queue.put(mono.astype(np.float32))

    def _diarize_and_transcribe(self, audio: np.ndarray) -> list[tuple[str, str]]:
        """Run diarization then transcribe each speaker segment."""
        import torch

        if len(audio) < SAMPLE_RATE * 0.3:
            return []

        # Pass audio as waveform dict to bypass torchcodec/FFmpeg issues
        waveform = torch.from_numpy(audio).unsqueeze(0).float()  # Shape: (1, samples)
        audio_input = {"waveform": waveform, "sample_rate": SAMPLE_RATE}

        try:
            # Run diarization - pyannote 4.x returns DiarizeOutput dataclass
            diarization_output = self.diarization(audio_input)
            annotation = diarization_output.speaker_diarization

            results = []
            for turn, _, speaker in annotation.itertracks(yield_label=True):
                # Extract audio for this speaker turn
                start_sample = int(turn.start * SAMPLE_RATE)
                end_sample = int(turn.end * SAMPLE_RATE)
                segment_audio = audio[start_sample:end_sample]

                if len(segment_audio) < SAMPLE_RATE * 0.2:
                    continue

                # Save segment to temp file for transcription
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as seg_f:
                    seg_path = seg_f.name
                    with wave.open(seg_f, "wb") as wav:
                        wav.setnchannels(1)
                        wav.setsampwidth(2)
                        wav.setframerate(SAMPLE_RATE)
                        seg_int16 = (segment_audio * 32767).astype(np.int16)
                        wav.writeframes(seg_int16.tobytes())

                try:
                    result = self.model.transcribe(seg_path)
                    text = result.text.strip()
                    if text:
                        results.append((speaker, text))
                finally:
                    Path(seg_path).unlink(missing_ok=True)

            return results

        except Exception as e:
            print(f"Diarization error: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            return []

    def transcribe_worker(self):
        """Background thread that detects speech end and diarizes+transcribes."""
        import time

        buffer = np.array([], dtype=np.float32)
        speech_buffer = np.array([], dtype=np.float32)
        silence_frames = 0
        in_speech = False
        silence_threshold_frames = int(SILENCE_THRESHOLD * SAMPLE_RATE / self.vad_frame_size)

        # Track when we last added speech to diarize buffer
        last_speech_time = None
        # Flush after this much silence following speech (seconds)
        FLUSH_AFTER_SILENCE = 2.0

        while self.running:
            try:
                chunk = self.audio_queue.get(timeout=0.5)
                buffer = np.append(buffer, chunk)

                # Process in VAD frame-sized chunks
                while len(buffer) >= self.vad_frame_size:
                    frame = buffer[:self.vad_frame_size]
                    buffer = buffer[self.vad_frame_size:]

                    is_speech = self._is_speech(frame)

                    if is_speech:
                        speech_buffer = np.append(speech_buffer, frame)
                        silence_frames = 0
                        in_speech = True
                    elif in_speech:
                        speech_buffer = np.append(speech_buffer, frame)
                        silence_frames += 1

                        if silence_frames >= silence_threshold_frames:
                            # End of utterance - add to diarize buffer
                            self.diarize_buffer = np.append(self.diarize_buffer, speech_buffer)
                            last_speech_time = time.time()
                            buffer_duration = len(self.diarize_buffer) / SAMPLE_RATE
                            print(f"  [Buffer: {buffer_duration:.1f}s]", end="\r")

                            speech_buffer = np.array([], dtype=np.float32)
                            silence_frames = 0
                            in_speech = False

                            # Flush if we hit max duration
                            if buffer_duration >= self.MAX_DIARIZE_DURATION:
                                self._flush_diarize_buffer()
                                last_speech_time = None

                # Check if we should flush based on silence after speech
                if last_speech_time and len(self.diarize_buffer) > 0:
                    time_since_speech = time.time() - last_speech_time
                    buffer_duration = len(self.diarize_buffer) / SAMPLE_RATE

                    if time_since_speech >= FLUSH_AFTER_SILENCE and buffer_duration >= self.MIN_DIARIZE_DURATION:
                        self._flush_diarize_buffer()
                        last_speech_time = None

            except Empty:
                # Timeout - check if we should flush
                if last_speech_time and len(self.diarize_buffer) > 0:
                    buffer_duration = len(self.diarize_buffer) / SAMPLE_RATE
                    if buffer_duration >= self.MIN_DIARIZE_DURATION:
                        self._flush_diarize_buffer()
                        last_speech_time = None
                continue
            except Exception as e:
                print(f"Transcription error: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc()

    def _flush_diarize_buffer(self):
        """Process accumulated audio through diarization and transcription."""
        if len(self.diarize_buffer) < SAMPLE_RATE * 0.5:  # Skip if too short
            self.diarize_buffer = np.array([], dtype=np.float32)
            return

        duration = len(self.diarize_buffer) / SAMPLE_RATE
        print(f"\n  Processing {duration:.1f}s of audio...")

        results = self._diarize_and_transcribe(self.diarize_buffer)
        if not results:
            print("  No speech segments found")
        for speaker, text in results:
            self._write_transcript(text, speaker=speaker)
        self.diarize_buffer = np.array([], dtype=np.float32)

    def start(self):
        """Start diarized transcription."""
        with open(self.output_file, "w") as f:
            f.write(f"# Transcription - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        self.running = True

        worker_thread = Thread(target=self.transcribe_worker, daemon=True)
        worker_thread.start()

        print(f"\nRecording with speaker diarization... Output: {self.output_file}")
        print(f"Accumulating {self.MIN_DIARIZE_DURATION}-{self.MAX_DIARIZE_DURATION}s of audio before diarizing")
        print("Press Ctrl+C to stop\n")

        try:
            num_channels = 3 if "Aggregate" in str(self.device or "") else 1
            with sd.InputStream(
                callback=self.audio_callback,
                channels=num_channels,
                samplerate=SAMPLE_RATE,
                dtype=np.float32,
                device=self.device,
            ):
                while self.running:
                    sd.sleep(100)
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            self.stop()

    def stop(self):
        """Stop transcription and cleanup."""
        # Flush any remaining audio in the diarize buffer
        if len(self.diarize_buffer) > SAMPLE_RATE * 0.5:
            print("Processing remaining audio...")
            self._flush_diarize_buffer()

        self.running = False
        with open(self.output_file, "a") as f:
            f.write(f"\n---\n*Ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
        print(f"Transcript saved to: {self.output_file}")


class BatchTranscriber:
    """Fast batch transcription - records then transcribes all at once (like Spokenly)."""

    def __init__(self, model_name: str, output_file: str, device=None):
        self.output_file = Path(output_file)
        self.device = device
        self.audio_chunks = []

        print(f"Loading model: {model_name}")
        self.model = from_pretrained(model_name)
        print("Model loaded")

    def audio_callback(self, indata, frames, time, status):
        """Called by sounddevice for each audio chunk."""
        if status:
            print(f"Audio status: {status}", file=sys.stderr)

        # Mix channels: mic (ch0) + system audio (ch1-2 averaged)
        if indata.shape[1] >= 3:
            mic = indata[:, 0] * 2.0
            system = np.mean(indata[:, 1:3], axis=1)
            mono = (mic + system) / 2.0
        elif indata.shape[1] > 1:
            mono = np.mean(indata, axis=1)
        else:
            mono = indata.flatten()
        self.audio_chunks.append(mono.astype(np.float32))

    def start(self):
        """Record until Ctrl+C, then transcribe all at once."""
        print(f"\nRecording... Press Ctrl+C when done speaking\n")

        num_channels = 3 if "Aggregate" in str(self.device or "") else 1

        try:
            with sd.InputStream(
                callback=self.audio_callback,
                channels=num_channels,
                samplerate=SAMPLE_RATE,
                dtype=np.float32,
                device=self.device,
            ):
                while True:
                    sd.sleep(100)
        except KeyboardInterrupt:
            pass

        if not self.audio_chunks:
            print("No audio recorded")
            return

        # Combine all audio
        print("Transcribing...")
        audio = np.concatenate(self.audio_chunks)

        # Save to temp file
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
            text = result.text.strip()

            # Write to file
            with open(self.output_file, "w") as f:
                f.write(f"# Transcription - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(f"{text}\n")

            print(f"\n{text}\n")
            print(f"Saved to: {self.output_file}")
        finally:
            Path(temp_path).unlink(missing_ok=True)


def list_devices():
    """Print available audio input devices."""
    print("Available audio input devices:\n")
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if device["max_input_channels"] > 0:
            default = " (default)" if device["name"] == sd.query_devices(kind="input")["name"] else ""
            print(f"  {i}: {device['name']}{default}")
            print(f"      Channels: {device['max_input_channels']}, Sample Rate: {device['default_samplerate']}")
    print("\nUse --device <number or name> to select a device")


def main():
    parser = argparse.ArgumentParser(
        description="Real-time transcription to markdown using Parakeet MLX"
    )
    parser.add_argument(
        "--model",
        default="mlx-community/parakeet-tdt-0.6b-v3",
        choices=[
            "mlx-community/parakeet-tdt-0.6b-v2",
            "mlx-community/parakeet-tdt-0.6b-v3",
            "mlx-community/parakeet-tdt-1.1b",
        ],
        help="Model to use for transcription (default: v3 multilingual)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="transcript.md",
        help="Output markdown file path (default: transcript.md)",
    )
    parser.add_argument(
        "--device",
        "-d",
        default=None,
        help="Audio input device (number or name, e.g., 'BlackHole 2ch')",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio input devices and exit",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Batch mode: record until Ctrl+C, then transcribe all at once (faster, like Spokenly)",
    )
    parser.add_argument(
        "--vad",
        action="store_true",
        help="VAD mode: auto-detect speech end and transcribe immediately (fast like batch, continuous like stream)",
    )
    parser.add_argument(
        "--diarize",
        action="store_true",
        help="Enable speaker diarization (channel-based without --hf-token, pyannote with --hf-token)",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="HuggingFace token for pyannote speaker diarization (or set HF_TOKEN env var)",
    )

    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return

    # Parse device argument
    device = args.device
    if device is not None:
        try:
            device = int(device)
        except ValueError:
            # Keep as string (device name)
            pass

    # Check for HF token (CLI arg or environment)
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    if args.diarize and hf_token:
        # Full pyannote diarization with HF token
        transcriber = DiarizedTranscriber(
            model_name=args.model,
            output_file=args.output,
            device=device,
            hf_token=hf_token,
        )
    elif args.vad or args.diarize:
        # VAD mode, or channel-based diarization without HF token
        transcriber = VADTranscriber(
            model_name=args.model,
            output_file=args.output,
            device=device,
            diarize=args.diarize,
        )
    elif args.batch:
        transcriber = BatchTranscriber(
            model_name=args.model,
            output_file=args.output,
            device=device,
        )
    else:
        transcriber = StreamingTranscriber(
            model_name=args.model,
            output_file=args.output,
            device=device,
        )
    transcriber.start()


if __name__ == "__main__":
    main()
