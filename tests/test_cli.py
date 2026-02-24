from datetime import datetime as real_datetime
from types import SimpleNamespace

import livekeet


def test_ensure_unique_path_no_conflict(tmp_path):
    path = tmp_path / "meeting.md"
    resolved, suffixed = livekeet.ensure_unique_path(path)
    assert resolved == path
    assert suffixed is False


def test_ensure_unique_path_suffixing(tmp_path):
    path = tmp_path / "meeting.md"
    path.write_text("x")
    resolved, suffixed = livekeet.ensure_unique_path(path)
    assert resolved == tmp_path / "meeting-2.md"
    assert suffixed is True


def test_ensure_unique_path_increments_existing_suffix(tmp_path):
    path = tmp_path / "meeting-2.md"
    path.write_text("x")
    resolved, suffixed = livekeet.ensure_unique_path(path)
    assert resolved == tmp_path / "meeting-3.md"
    assert suffixed is True


def test_resolve_output_path_adds_suffix(tmp_path):
    config = {"output": {"directory": "", "filename": "{datetime}.md"}}
    path = livekeet.resolve_output_path(config, str(tmp_path / "notes"))
    assert path.suffix == ".md"


def test_resolve_output_path_pattern(monkeypatch, tmp_path):
    class FixedDatetime(real_datetime):
        @classmethod
        def now(cls, tz=None):
            return real_datetime(2024, 1, 2, 3, 4, 5)

    monkeypatch.setattr(livekeet, "datetime", FixedDatetime)
    config = {"output": {"directory": str(tmp_path), "filename": "{date}-meeting.md"}}
    path = livekeet.resolve_output_path(config, None)
    assert path.name == "2024-01-02-meeting.md"


def test_resolve_output_path_directory(monkeypatch, tmp_path):
    class FixedDatetime(real_datetime):
        @classmethod
        def now(cls, tz=None):
            return real_datetime(2024, 1, 2, 3, 4, 5)

    monkeypatch.setattr(livekeet, "datetime", FixedDatetime)
    config = {"output": {"directory": "", "filename": "{datetime}.md"}}
    path = livekeet.resolve_output_path(config, str(tmp_path))
    assert path.parent == tmp_path
    assert path.name == "2024-01-02-030405.md"


def test_resolve_output_path_directory_custom_pattern(monkeypatch, tmp_path):
    class FixedDatetime(real_datetime):
        @classmethod
        def now(cls, tz=None):
            return real_datetime(2024, 1, 2, 3, 4, 5)

    monkeypatch.setattr(livekeet, "datetime", FixedDatetime)
    config = {"output": {"directory": "", "filename": "{date}-meeting.md"}}
    path = livekeet.resolve_output_path(config, str(tmp_path))
    assert path.parent == tmp_path
    assert path.name == "2024-01-02-meeting.md"


def test_init_subcommand(monkeypatch):
    """Test that 'init' is recognized as a subcommand."""
    called = []
    monkeypatch.setattr(livekeet, "init_config", lambda: called.append(True))
    monkeypatch.setattr("sys.argv", ["livekeet", "init"])
    livekeet.main()
    assert called == [True]


def test_warn_multilingual_overrides_model(capsys):
    args = SimpleNamespace(multilingual=True, model="mlx-community/parakeet-tdt-0.6b-v2", mic_only=False, other_speaker=None)
    livekeet.warn_flag_interactions(args)
    captured = capsys.readouterr()
    assert "--multilingual overrides --model" in captured.err


def test_warn_with_ignored_in_mic_only(capsys):
    args = SimpleNamespace(multilingual=False, model=None, mic_only=True, other_speaker="John")
    livekeet.warn_flag_interactions(args)
    captured = capsys.readouterr()
    assert "--with is ignored in --mic-only mode" in captured.err


def test_comma_separated_with_names():
    """--with 'Tom,Bob' should parse into multiple names."""
    raw = "Tom,Bob"
    names = [n.strip() for n in raw.split(",") if n.strip()]
    assert names == ["Tom", "Bob"]


def test_comma_separated_with_spaces():
    """Spaces around commas should be stripped."""
    raw = " Alice , Bob , Charlie "
    names = [n.strip() for n in raw.split(",") if n.strip()]
    assert names == ["Alice", "Bob", "Charlie"]


def test_single_with_name_no_split():
    """A single name without commas returns one entry."""
    raw = "John"
    names = [n.strip() for n in raw.split(",") if n.strip()]
    assert names == ["John"]


def test_auto_diarize_multiple_names():
    """Diarization should auto-enable when multiple --with names given."""
    other_names = ["Tom", "Bob"]
    diarize_flag = False
    config_diarize = False
    diarize = diarize_flag or config_diarize or len(other_names) > 1
    assert diarize is True


def test_no_auto_diarize_single_name():
    """Diarization should not auto-enable for a single --with name."""
    other_names = ["Tom"]
    diarize_flag = False
    config_diarize = False
    diarize = diarize_flag or config_diarize or len(other_names) > 1
    assert diarize is False


def test_config_diarize_enables():
    """Config diarize=true should enable diarization."""
    other_names = []
    diarize_flag = False
    config_diarize = True
    diarize = diarize_flag or config_diarize or len(other_names) > 1
    assert diarize is True


def test_diarization_module_importable():
    """Ensure diarization.py is importable (catches packaging omissions)."""
    from diarization import SpeakerTracker, load_embedder
    assert callable(load_embedder)
    assert callable(SpeakerTracker)


def test_pcm_to_temp_wav(tmp_path):
    """_pcm_to_temp_wav produces a valid mono 16kHz WAV."""
    import wave
    import numpy as np

    # Generate 1 second of silence as int16 PCM
    samples = np.zeros(16000, dtype=np.int16)
    pcm_data = samples.tobytes()

    wav_path = livekeet.Transcriber._pcm_to_temp_wav(pcm_data)
    try:
        assert wav_path.exists()
        with wave.open(str(wav_path), "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == 16000
            assert wf.getnframes() == 16000
    finally:
        wav_path.unlink(missing_ok=True)


def _make_transcriber_stub(output_file):
    """Create a Transcriber-like object with just the fields _rebuild_transcript needs."""
    t = object.__new__(livekeet.Transcriber)
    t.output_file = output_file
    t.speaker_name = "Me"
    t.other_name = "Alice"
    t.other_names = ["Alice"]
    t._session_start_str = "2024-01-02 10:00:00"
    t._segments = []
    t._pyannote_turns = None
    return t


def test_rebuild_transcript_fallback(tmp_path):
    """Segments with no pyannote turns use channel-based fallback labels."""
    out = tmp_path / "test.md"
    t = _make_transcriber_stub(out)
    t._segments = [
        (5.0, "Hello there", "mic", "10:00:05"),
        (10.0, "Hi back", "system", "10:00:10"),
    ]

    t._rebuild_transcript()

    content = out.read_text()
    assert "**Me**:" in content
    assert "**Alice**:" in content
    assert "Hello there" in content
    assert "Hi back" in content
    assert content.startswith("# Transcription - 2024-01-02 10:00:00")


def test_rebuild_transcript_with_turns(tmp_path):
    """Segments + mock pyannote turns produce correct pyannote labels."""
    from types import SimpleNamespace

    out = tmp_path / "test.md"
    t = _make_transcriber_stub(out)
    t._segments = [
        (5.0, "Hello there", "mic", "10:00:05"),
        (10.0, "Hi back", "system", "10:00:10"),
    ]
    t._pyannote_turns = [
        SimpleNamespace(start=3.0, end=7.0, speaker="Luca", channel="mic"),
        SimpleNamespace(start=8.0, end=12.0, speaker="Bob", channel="system"),
    ]

    t._rebuild_transcript()

    content = out.read_text()
    assert "**Luca**:" in content
    assert "**Bob**:" in content
    # Fallback labels should not appear
    assert "**Me**:" not in content
    assert "**Alice**:" not in content


def test_resolve_speaker_filters_by_channel(tmp_path):
    """Turns from a different channel should not match — fall back to channel label."""
    from types import SimpleNamespace

    out = tmp_path / "test.md"
    t = _make_transcriber_stub(out)
    # System channel segment, but only mic channel turns exist
    t._pyannote_turns = [
        SimpleNamespace(start=3.0, end=7.0, speaker="Luca", channel="mic"),
    ]
    # Should fall back to channel label, not match the mic turn
    assert t._resolve_speaker(5.0, "system") == "Alice"
    # Same offset on mic channel should match
    assert t._resolve_speaker(5.0, "mic") == "Luca"
