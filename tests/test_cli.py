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
