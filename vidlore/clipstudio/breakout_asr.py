"""Short-window ASR for real-audio Breakouts.

Whisper can return a plausible but truncated transcript for a 7-10 second
movie-dialogue clip.  A single pass then makes both admission and captions
believe the returned prefix is the whole utterance.  Overlapping short windows
make the tail independently observable and are shared by both callers.
"""
from __future__ import annotations

import math
import re
import subprocess
import tempfile
from pathlib import Path

from .ingest import probe
from vidlore.ffmpeg_tool import ffmpeg_exe


def _norm_token(text: str) -> str:
    return re.sub(r"[^a-z0-9']", "", (text or "").lower())


def _materialize_words(model, wav: Path) -> list[tuple[str, float, float, float]]:
    segs, _ = model.transcribe(str(wav), word_timestamps=True, vad_filter=False,
                               condition_on_previous_text=False)
    out = []
    for seg in list(segs):
        for word in (getattr(seg, "words", None) or []):
            text = str(getattr(word, "word", "") or "").strip()
            try:
                start, end = float(word.start), float(word.end)
            except (TypeError, ValueError):
                continue
            if (text and math.isfinite(start) and math.isfinite(end)
                    and end > start >= 0.0):
                out.append((text, start, end,
                            float(getattr(word, "probability", 1.0) or 1.0)))
    return out


def _dedupe_overlaps(words: list[tuple[str, float, float, float]]) -> list[tuple[str, float, float, float]]:
    """Merge duplicate observations in overlap zones without dropping real repeats."""
    words = sorted(words, key=lambda w: (w[1], w[2], -w[3]))
    out: list[tuple[str, float, float, float]] = []
    for w in words:
        norm = _norm_token(w[0])
        duplicate = None
        # Only the immediate tail can overlap.  Same lexical token + strongly
        # overlapping time is the same observation; repeated dialogue at a
        # later timestamp remains intact.
        for i in range(len(out) - 1, max(-1, len(out) - 6), -1):
            old = out[i]
            if w[1] - old[2] > 0.55:
                break
            overlap = min(w[2], old[2]) - max(w[1], old[1])
            if norm and norm == _norm_token(old[0]) and overlap > -0.12:
                duplicate = i
                break
        if duplicate is not None:
            if w[3] > out[duplicate][3]:
                out[duplicate] = w
            continue
        out.append(w)
    out.sort(key=lambda w: (w[1], w[2]))
    # Enforce a clean monotonic stream at overlap seams.  A tiny decoder-time
    # overlap is clamped; a wholly backwards duplicate was already removed.
    clean = []
    last_start = -1.0
    for text, start, end, prob in out:
        if start < last_start - 0.25:
            continue
        clean.append((text, start, max(start + 0.01, end), prob))
        last_start = max(last_start, start)
    return clean


def transcribe_breakout_words(wav_path, *, model=None, duration: float | None = None,
                              window: float = 3.8, overlap: float = 0.6) \
        -> list[tuple[str, float, float, float]]:
    """Transcribe a Breakout in overlapping windows with clip-relative times."""
    wav = Path(wav_path)
    if model is None:
        from faster_whisper import WhisperModel
        model = WhisperModel("base", device="cpu", compute_type="int8")
    if duration is None:
        duration = float(probe(wav).get("duration", 0.0) or 0.0)
    duration = float(duration or 0.0)
    if duration <= 0.05:
        return []
    window = max(1.5, float(window))
    overlap = min(max(0.2, float(overlap)), window * 0.45)
    step = window - overlap
    starts = [0.0]
    while starts[-1] + window < duration - 0.05:
        nxt = starts[-1] + step
        if duration - nxt < 0.8:
            break
        starts.append(nxt)
    # Always anchor a tail window to EOF.  This is the independent observation
    # that recovered the four seconds omitted by the whole-pass transcript.
    tail = max(0.0, duration - window)
    if tail > starts[-1] + 0.15:
        starts.append(tail)

    observed: list[tuple[str, float, float, float]] = []
    for offset in starts:
        span = min(window, duration - offset)
        with tempfile.TemporaryDirectory(prefix="breakout_asr_") as td:
            chunk = Path(td) / "chunk.wav"
            p = subprocess.run(
                [ffmpeg_exe(), "-y", "-loglevel", "error", "-ss", f"{offset:.3f}",
                 "-t", f"{span:.3f}", "-i", str(wav), "-ar", "16000", "-ac", "1", str(chunk)],
                capture_output=True, timeout=90)
            if p.returncode != 0 or not chunk.exists() or chunk.stat().st_size <= 44:
                continue
            try:
                for text, start, end, prob in _materialize_words(model, chunk):
                    observed.append((text, offset + start, min(duration, offset + end), prob))
            except Exception:
                continue
    return _dedupe_overlaps(observed)


def speech_seconds(words: list[tuple[str, float, float, float]]) -> float:
    """Union duration of ASR word spans (overlaps never double-counted)."""
    spans = sorted((float(w[1]), float(w[2])) for w in words if w[2] > w[1])
    total = 0.0
    a = b = None
    for start, end in spans:
        if a is None:
            a, b = start, end
        elif start <= b:
            b = max(b, end)
        else:
            total += b - a
            a, b = start, end
    if a is not None:
        total += b - a
    return round(total, 2)


def caption_coverage(spoken_words, captioned_words, *, tail_tolerance: float = 0.5) -> dict:
    """Measured Breakout-caption completeness; every field is audit-friendly."""
    spoken = list(spoken_words or [])
    captioned = list(captioned_words or [])
    spoken_n = len(spoken)
    captioned_n = len(captioned)
    spoken_last = max((float(w[2]) for w in spoken), default=0.0)
    captioned_last = max((float(w[2]) for w in captioned), default=0.0)
    coverage = captioned_n / float(spoken_n) if spoken_n else 0.0
    tail = max(0.0, spoken_last - captioned_last)
    return {
        "spoken_words": spoken_n,
        "captioned_words": captioned_n,
        "coverage": round(coverage, 3),
        "asr_last_word_s": round(spoken_last, 3),
        "caption_last_word_s": round(captioned_last, 3),
        "uncaptioned_tail_s": round(tail, 3),
        # "Most" dialogue is not complete dialogue.  A 90% floor allowed one
        # dropped middle word in ten to publish even when the tail was covered.
        # Every ASR-observed spoken word must have a caption; the tail check is
        # retained as an independent guard against a truncated ASR prefix.
        "passed": bool(spoken_n and captioned_n == spoken_n
                       and coverage >= 1.0 and tail <= tail_tolerance),
    }
