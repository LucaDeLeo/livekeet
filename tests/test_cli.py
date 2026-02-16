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
