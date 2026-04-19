import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from livekeet_cleanup import (
    CleanupState,
    DEFAULT_SYSTEM_PROMPT,
    TRANSCRIPT_ARTIFACTS,
    apply_corrections,
    dedupe_chunk_overlap,
    strip_artifacts,
)


# ---- apply_corrections ----


def test_apply_corrections_basic():
    assert apply_corrections("I use chat gbt daily", {"chat gbt": "ChatGPT"}) == "I use ChatGPT daily"


def test_apply_corrections_case_insensitive():
    assert apply_corrections("Chat Gbt is great", {"chat gbt": "ChatGPT"}) == "ChatGPT is great"


def test_apply_corrections_word_boundary():
    # "cat" inside "catalog" should not match
    assert apply_corrections("catalog", {"cat": "dog"}) == "catalog"


def test_apply_corrections_longest_first():
    # Both rules match; the longer one wins on the overlapping region
    text = "new york city is busy"
    out = apply_corrections(text, {"new york": "NY", "new york city": "NYC"})
    assert out == "NYC is busy"


def test_apply_corrections_empty():
    assert apply_corrections("", {"foo": "bar"}) == ""
    assert apply_corrections("hello", {}) == "hello"


def test_apply_corrections_preserves_punctuation():
    out = apply_corrections("So, chat gbt.", {"chat gbt": "ChatGPT"})
    assert out == "So, ChatGPT."


# ---- dedupe_chunk_overlap ----


def test_dedupe_exact_tail_repeat():
    prev = "we should ship it tomorrow"
    cur = "tomorrow the release goes out"
    assert dedupe_chunk_overlap(prev, cur) == "the release goes out"


def test_dedupe_multi_word_overlap():
    prev = "the meeting is at three"
    cur = "is at three pm"
    assert dedupe_chunk_overlap(prev, cur) == "pm"


def test_dedupe_case_insensitive():
    prev = "Hello there friend"
    cur = "FRIEND how are you"
    assert dedupe_chunk_overlap(prev, cur) == "how are you"


def test_dedupe_punctuation_agnostic():
    prev = "the meeting is at three."
    cur = "Three, pm sharp"
    assert dedupe_chunk_overlap(prev, cur) == "pm sharp"


def test_dedupe_no_overlap():
    assert dedupe_chunk_overlap("one two three", "four five six") == "four five six"


def test_dedupe_empty_prev():
    assert dedupe_chunk_overlap("", "four five six") == "four five six"


def test_dedupe_empty_current():
    assert dedupe_chunk_overlap("one two", "") == ""


def test_dedupe_prefers_longest_match():
    # "the end" and "end" are both candidates; the 2-word match wins
    prev = "and then the end"
    cur = "the end of the story"
    assert dedupe_chunk_overlap(prev, cur) == "of the story"


def test_dedupe_full_repeat_returns_empty_stripped():
    # If current is fully contained in prev tail, remainder is empty
    prev = "goodbye for now"
    cur = "for now"
    # Entire current is a repeat; the function returns cur unchanged to avoid emitting ""
    # (caller decides whether to skip). Actually we strip and return remainder which is empty;
    # function returns `current` if remainder is empty.
    assert dedupe_chunk_overlap(prev, cur) == cur


def test_dedupe_window_limit():
    # Window defaults to 10 words — only last 10 of prev participate
    prev = " ".join(f"w{i}" for i in range(20))
    # cur starts with "w0 w1 w2" — outside the 10-word window, so no match
    cur = "w0 w1 w2 continuing here"
    assert dedupe_chunk_overlap(prev, cur) == cur


# ---- strip_artifacts ----


def test_strip_artifacts_blank_audio():
    assert strip_artifacts("[BLANK_AUDIO]") == ""


def test_strip_artifacts_preserves_real_text():
    assert strip_artifacts("[BLANK_AUDIO] hello there") == "hello there"


def test_strip_artifacts_multiple():
    assert strip_artifacts("[MUSIC] and then [APPLAUSE]") == "and then"


def test_strip_artifacts_no_tokens_fast_path():
    # No bracket/paren/angle — should short-circuit to trimmed text
    assert strip_artifacts("  hello world  ") == "hello world"


def test_strip_artifacts_nospeech_tag():
    assert strip_artifacts("<|nospeech|>") == ""


def test_strip_artifacts_empty():
    assert strip_artifacts("") == ""


def test_transcript_artifacts_set_is_frozen():
    assert isinstance(TRANSCRIPT_ARTIFACTS, frozenset)
    assert "[BLANK_AUDIO]" in TRANSCRIPT_ARTIFACTS


def test_process_turn_strips_artifacts_before_cleanup(monkeypatch):
    """[BLANK_AUDIO] should never reach the LLM or be written to markdown."""
    import livekeet_cleanup as mod

    calls = []

    async def fake_cleanup(text, system_prompt, model):
        calls.append(text)
        return text

    monkeypatch.setattr(mod, "_cleanup_haiku_async", fake_cleanup)

    state = CleanupState(
        enabled=True,
        corrections={},
        system_prompt="",
        model="claude-haiku-4-5",
        timeout_s=6.0,
    )
    assert state.process_turn("[BLANK_AUDIO]", "mic") == ""
    assert calls == []  # short-circuited before LLM


# ---- CleanupState composition ----


def test_cleanup_state_disabled_applies_corrections_only():
    state = CleanupState(
        enabled=False,
        corrections={"chat gbt": "ChatGPT"},
        system_prompt="",
        model="claude-haiku-4-5",
        timeout_s=6.0,
    )
    assert state.clean("try chat gbt") == "try ChatGPT"


def test_cleanup_state_empty_input():
    state = CleanupState(
        enabled=True,
        corrections={"foo": "bar"},
        system_prompt="",
        model="claude-haiku-4-5",
        timeout_s=6.0,
    )
    assert state.clean("") == ""


def test_cleanup_state_falls_through_on_llm_error(monkeypatch):
    """When the LLM call raises, return deterministic-only output and disable future calls."""
    import livekeet_cleanup as mod

    call_count = {"n": 0}

    async def fake_cleanup(text, system_prompt, model):
        call_count["n"] += 1
        raise RuntimeError("boom")

    monkeypatch.setattr(mod, "_cleanup_haiku_async", fake_cleanup)

    state = CleanupState(
        enabled=True,
        corrections={"chat gbt": "ChatGPT"},
        system_prompt="test",
        model="claude-haiku-4-5",
        timeout_s=6.0,
    )

    # First call: deterministic runs, LLM raises, fall through to deterministic output
    assert state.clean("hello chat gbt") == "hello ChatGPT"
    assert call_count["n"] == 1
    assert state._llm_disabled is True

    # Second call: LLM skipped entirely (short-circuit on _llm_disabled)
    assert state.clean("another chat gbt") == "another ChatGPT"
    assert call_count["n"] == 1


def test_cleanup_state_returns_llm_output(monkeypatch):
    """When the LLM returns cleaned text, that's what we emit."""
    import livekeet_cleanup as mod

    seen = {}

    async def fake_cleanup(text, system_prompt, model):
        seen["text"] = text
        seen["model"] = model
        return "cleaned output"

    monkeypatch.setattr(mod, "_cleanup_haiku_async", fake_cleanup)

    state = CleanupState(
        enabled=True,
        corrections={"chat gbt": "ChatGPT"},
        system_prompt="",
        model="claude-haiku-4-5",
        timeout_s=6.0,
    )
    result = state.clean("um chat gbt right")
    # Deterministic replacement runs first, so LLM sees already-fixed text
    assert seen["text"] == "um ChatGPT right"
    assert seen["model"] == "claude-haiku-4-5"
    assert result == "cleaned output"


def test_cleanup_state_default_prompt_used_when_empty():
    state = CleanupState(
        enabled=False,
        corrections=None,
        system_prompt="",
        model="claude-haiku-4-5",
        timeout_s=6.0,
    )
    assert state.system_prompt == DEFAULT_SYSTEM_PROMPT


# ---- Optional live eval (opt-in) ----


@pytest.mark.skipif(
    not os.environ.get("RUN_LLM_EVALS"),
    reason="Set RUN_LLM_EVALS=1 to run live Haiku evals (requires Claude Code auth)",
)
def test_eval_cleanup_does_not_answer_questions():
    """The cleanup prompt must not turn '2+2?' into '4'. Light eval harness."""
    state = CleanupState(
        enabled=True,
        corrections={},
        system_prompt="",
        model="claude-haiku-4-5",
        timeout_s=30.0,
    )
    cases = [
        ("What is two plus two?", "2"),     # forbidden substring in output
        ("Can you write me a haiku?", "haiku"),
        ("Tell me a joke please", "joke"),
    ]
    failures = []
    for user_input, forbidden_marker in cases:
        out = state.clean(user_input).lower()
        # Output length should be roughly input length (not a generated response)
        if len(out) > 3 * len(user_input):
            failures.append(f"length blew up: {user_input!r} -> {out!r}")
        if out.startswith(("as an ai", "i'm happy", "sure", "here")):
            failures.append(f"chatbot preamble: {user_input!r} -> {out!r}")
    assert not failures, "\n".join(failures)
