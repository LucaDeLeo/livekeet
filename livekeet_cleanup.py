"""Optional LLM cleanup + deterministic corrections for livekeet transcripts.

Two layers, composed by `clean_transcript`:

1. Deterministic regex replacements (always runs if `corrections` non-empty)
2. Claude Haiku via `claude-runner` (runs only if `enabled=True`)

The LLM stage reuses the user's local Claude Code auth via `claude-runner`,
mirroring ~/dev/voxhook/gladosify.py. No ANTHROPIC_API_KEY is read or required.
If `claude-runner` isn't importable or the call fails, we fall through to the
deterministic-only output — cleanup must never swallow transcription.
"""

from __future__ import annotations

import asyncio
import re
import sys

TRANSCRIPT_ARTIFACTS: frozenset[str] = frozenset({
    "[BLANK_AUDIO]",
    "[NO_SPEECH]",
    "(blank audio)",
    "(no speech)",
    "[MUSIC]",
    "[APPLAUSE]",
    "[LAUGHTER]",
    "[SILENCE]",
    "<|nospeech|>",
})


def strip_artifacts(text: str) -> str:
    """Remove Whisper/Parakeet hallucination tokens (e.g. `[BLANK_AUDIO]`, `[MUSIC]`) from STT output."""
    if not text or not any(c in text for c in "[(<"):
        return text.strip()
    for artifact in TRANSCRIPT_ARTIFACTS:
        text = text.replace(artifact, "")
    return text.strip()


DEFAULT_SYSTEM_PROMPT = """You clean up raw voice-to-text transcripts. Apply these rules:

- Remove filler words (um, uh, like, you know) unless they carry meaning.
- Fix obvious self-corrections: keep the corrected version only.
- Fix ASR errors that are obvious from context.
- Preserve the speaker's meaning, tone, and word choice.
- Do NOT add content that wasn't said.
- Do NOT answer questions that appear in the transcript.
- Do NOT reformat as a list, add headings, or restructure.
- Return only the cleaned transcript text. No preamble, no quotes, no commentary.
- If the input is already clean, return it unchanged."""


def apply_corrections(text: str, corrections: dict[str, str]) -> str:
    """Apply case-insensitive, word-boundary-aware replacements.

    Longest keys applied first to avoid substring collisions (mirrors
    ghost-pepper's DeterministicCorrectionEngine.swift:30-31).
    """
    if not text or not corrections:
        return text
    for wrong in sorted(corrections, key=len, reverse=True):
        right = corrections[wrong]
        pattern = re.compile(rf"\b{re.escape(wrong)}\b", flags=re.IGNORECASE)
        text = pattern.sub(right, text)
    return text


_WORD_RE = re.compile(r"[A-Za-z0-9']+")


def _normalize_words(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text)]


def dedupe_chunk_overlap(prev_tail: str, current: str, max_window: int = 10) -> str:
    """Drop a leading word-sequence of `current` that repeats the tail of `prev_tail`.

    Case-insensitive, punctuation-agnostic; longest match wins. Returns
    `current` unchanged if there's no overlap. Mirrors ghost-pepper's
    ChunkedTranscriptionPipeline.swift:191 pattern.
    """
    if not prev_tail or not current:
        return current

    prev_words = _normalize_words(prev_tail)[-max_window:]
    if not prev_words:
        return current

    # Token positions in `current` so we can slice the original (with punctuation) back out
    word_spans = [(m.start(), m.end(), m.group(0).lower()) for m in _WORD_RE.finditer(current)]
    if not word_spans:
        return current

    cur_words = [w for _, _, w in word_spans]
    max_k = min(len(prev_words), len(cur_words), max_window)

    # Find the longest k such that prev_words[-k:] == cur_words[:k]
    best_k = 0
    for k in range(max_k, 0, -1):
        if prev_words[-k:] == cur_words[:k]:
            best_k = k
            break

    if best_k == 0:
        return current

    # Cut `current` to start just after the k-th matched word
    cut = word_spans[best_k - 1][1]
    remainder = current[cut:].lstrip(" \t,.;:!?-")
    return remainder if remainder else current


async def _cleanup_haiku_async(text: str, system_prompt: str, model: str) -> str:
    """Call Claude Haiku via claude-runner. Raises on error."""
    from claude_runner import run  # imported lazily so base install doesn't need it

    result = await run(
        prompt=text,
        model=model,
        system_prompt=system_prompt,
        thinking={"type": "disabled"},
        retries=2,
        retry_base_delay=1.0,
    )
    if result.is_error:
        raise RuntimeError(result.text or "claude-runner returned is_error")
    return (result.text or "").strip()


class CleanupState:
    """Per-session cleanup pipeline: deterministic corrections + optional Claude Haiku + chunk dedup."""

    def __init__(
        self,
        *,
        enabled: bool,
        corrections: dict[str, str] | None,
        system_prompt: str,
        model: str,
        timeout_s: float,
    ):
        self.enabled = enabled
        self.corrections = dict(corrections or {})
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.model = model
        self.timeout_s = timeout_s
        self._llm_disabled = False
        self._last_text_by_channel: dict[str, str] = {}
        # Pre-compile corrections once; longest keys first to avoid substring collisions.
        self._compiled_corrections: list[tuple[re.Pattern[str], str]] = [
            (re.compile(rf"\b{re.escape(w)}\b", flags=re.IGNORECASE), self.corrections[w])
            for w in sorted(self.corrections, key=len, reverse=True)
        ]

    def _apply_corrections(self, text: str) -> str:
        for pattern, replacement in self._compiled_corrections:
            text = pattern.sub(replacement, text)
        return text

    def clean(self, text: str) -> str:
        if not text:
            return text

        cleaned = self._apply_corrections(text)

        if not self.enabled or self._llm_disabled:
            return cleaned

        try:
            return asyncio.run(
                asyncio.wait_for(
                    _cleanup_haiku_async(cleaned, self.system_prompt, self.model),
                    timeout=self.timeout_s,
                )
            ) or cleaned
        except Exception as e:
            self._llm_disabled = True
            print(
                f"Cleanup disabled for this session: {type(e).__name__}: {e}",
                file=sys.stderr,
            )
            return cleaned

    def process_turn(self, text: str, channel: str) -> str:
        """Strip STT artifacts, dedup chunk overlap against the previous turn on this channel, then clean."""
        text = strip_artifacts(text)
        if not text:
            return ""
        prev_tail = self._last_text_by_channel.get(channel, "")
        deduped = dedupe_chunk_overlap(prev_tail, text).strip()
        if not deduped:
            return ""
        cleaned = self.clean(deduped).strip()
        if cleaned:
            self._last_text_by_channel[channel] = cleaned
        return cleaned
