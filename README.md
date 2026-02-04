# livekeet

Real-time audio transcription to markdown. Optimized for Apple Silicon.

Uses NVIDIA's Parakeet speech recognition model via MLX for fast, accurate, on-device transcription.

## Features

- **Real-time transcription** with voice activity detection (VAD)
- **Speaker detection** - automatically labels who's speaking in calls
- **System audio capture** for transcribing calls, meetings, videos
- **Microphone capture** for voice notes and dictation
- **Markdown output** with timestamps and speaker labels
- **Configurable** speaker names, output location, filename patterns

## Requirements

- macOS 13+ (Ventura or later)
- Apple Silicon (M1/M2/M3/M4)
- Python 3.12+

## Installation

```bash
uv tool install git+https://github.com/LucaDeLeo/livekeet.git
```

The audio capture binary is downloaded automatically on first run.

### From source (for development)

```bash
git clone https://github.com/LucaDeLeo/livekeet.git
cd livekeet
uv sync && make build
uv run livekeet
```

### Screen Recording Permission

For system audio capture, macOS requires Screen Recording permission. On first run, you'll be prompted to grant this in System Settings > Privacy & Security > Screen Recording.

## Usage

```bash
# Start transcribing (system audio + mic, outputs to timestamped file)
livekeet

# Output to specific file
livekeet meeting.md

# Name the other speaker (for 1:1 calls)
livekeet --with "John"

# Multilingual transcription
livekeet --multilingual

# Mic only (no system audio)
livekeet --mic-only

# List audio devices
livekeet --devices

# Use specific audio device
livekeet --device "BlackHole 2ch"

# Show periodic status updates
livekeet --status
```

**Flags**
- `--with`, `-w` Set the other speaker name (for calls)
- `--mic-only`, `-m` Record microphone only (no system audio)
- `--multilingual` Use the multilingual model (parakeet-tdt-0.6b-v3)
- `--model` Choose a model explicitly
- `--device`, `-d` Select input device by number or name (mic-only)
- `--devices` List available audio input devices
- `--init` Create the default config file
- `--config` Show the config file location
- `--status` Show periodic status updates while recording
Note: `--multilingual` overrides `--model` when both are set.

## Configuration

Create a config file to customize defaults:

```bash
livekeet --init
```

This creates `~/.config/livekeet/config.toml`:

```toml
[output]
# Directory for transcripts (empty = current directory)
directory = ""
# Filename pattern: {date}, {time}, {datetime}
filename = "{datetime}.md"

[speaker]
# Your name in transcripts
name = "Me"

[defaults]
# Available models (downloaded automatically on first use):
#   mlx-community/parakeet-tdt-0.6b-v2  - Fast, English only (default)
#   mlx-community/parakeet-tdt-0.6b-v3  - Fast, multilingual
#   mlx-community/parakeet-tdt-1.1b     - Slower, English, highest accuracy
model = "mlx-community/parakeet-tdt-0.6b-v2"
```

### Output Patterns

The filename pattern supports these variables:
- `{date}` - Current date (2024-01-15)
- `{time}` - Current time (14-30-25)
- `{datetime}` - Combined (2024-01-15-143025)

Examples:
- `"{datetime}.md"` → `2024-01-15-143025.md`
- `"{date}-meeting.md"` → `2024-01-15-meeting.md`
- `"transcript.md"` → `transcript.md` (auto-suffixes if exists)
If the resolved filename already exists, livekeet will save to `name-2.md`, `name-3.md`, and so on.

## How It Works

1. **Audio Capture**: Uses a Swift tool with ScreenCaptureKit to capture system audio and microphone as separate stereo channels
2. **Voice Activity Detection**: WebRTC VAD detects when speech starts/stops
3. **Speaker Detection**: Compares audio energy between mic (you) and system (other) channels to determine who's speaking
4. **Transcription**: When speech ends, audio is transcribed using Parakeet-MLX
5. **Output**: Text is written to markdown with timestamps and speaker labels

## Speaker Detection

When using system audio capture (the default), livekeet automatically detects who's speaking based on which audio channel is dominant:

```
┌─────────────────┐     ┌──────────────┐
│  Microphone     │────▶│ Left Channel │────▶ Your name
└─────────────────┘     └──────────────┘
┌─────────────────┐     ┌──────────────┐
│  System Audio   │────▶│ Right Channel│────▶ Other speaker
└─────────────────┘     └──────────────┘
```

### How It Works

1. Audio is captured as **stereo**: left = mic, right = system
2. **VAD** (Voice Activity Detection) triggers when anyone speaks
3. During speech, **energy is measured** on each channel per frame
4. When speech ends, the channel with **>65% of total energy** determines the speaker
5. If energy is ambiguous (35-65% split), text is transcribed without a label

### Example Output

```markdown
[14:32:15] **Me**: Hey, how's the project going?
[14:32:22] **John**: Pretty good, we just finished the API integration.
[14:32:28] **Me**: Great, let's review it tomorrow.
[14:32:35] Yeah, sounds good.  ← (no label when ambiguous)
```

### Configuration

- Set your name: edit `~/.config/livekeet/config.toml` (run `livekeet --init` first)
- Set other speaker: `livekeet --with "John"`

### Limitations

- Works best with **clear turn-taking** (one person speaks at a time)
- **Simultaneous speech** may not be labeled (ambiguous energy)
- **Echo/crosstalk** can affect accuracy (your voice through their speakers, etc.)

## Models

| Model | Size | Speed | Notes |
|-------|------|-------|-------|
| parakeet-tdt-0.6b-v2 | 600M | Fast | English only (default) |
| parakeet-tdt-0.6b-v3 | 600M | Fast | Multilingual (`--multilingual`) |
| parakeet-tdt-1.1b | 1.1B | Slower | English, highest accuracy |

Models are downloaded automatically on first use.

## Troubleshooting

### "Audio capture tool not found"

Build the Swift audio capture tool:
```bash
make build
```

### "Screen Recording permission required"

Go to System Settings > Privacy & Security > Screen Recording and enable your terminal app.

### No audio captured

If you see "No audio detected yet", check Screen Recording or microphone permissions.

1. Check audio devices: `livekeet --devices`
2. Try mic-only mode: `livekeet --mic-only`
3. Ensure the audio source is playing

## License

MIT
