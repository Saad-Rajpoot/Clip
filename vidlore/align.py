"""Forced alignment: get the REAL spoken time of every script word.

Captions and on-screen graphic cards were timed by a proportional
*estimate* (word length / scene share). That can never truly match a
human voiceover — text drifts ahead of or behind the voice. Here we
transcribe the actual narration audio with faster-whisper (word-level
timestamps) and align those recognised words back onto the user's exact
script words, so every word has its true start/end second.

Pure best-effort: any failure (model missing, odd audio) returns None
and the caller falls back to the old proportional timing — a render
never dies because alignment had a bad day.
"""
from __future__ import annotations

import math
import re
from difflib import SequenceMatcher
from pathlib import Path

_MODEL = None


def _norm(tok: str) -> str:
    # MULTILINGUAL: keep Unicode letters/digits (é, ñ, 日本語, 한국어) so
    # alignment matches non-English script words too (old [^a-z0-9]
    # erased every non-ASCII word -> non-English never aligned).
    return re.sub(r"[^\w]", "", tok.lower(), flags=re.UNICODE)


def _model():
    global _MODEL
    if _MODEL is None:
        from faster_whisper import WhisperModel

        # MULTILINGUAL "base" (~140MB) instead of English-only "base.en":
        # auto-detects the spoken language so a Spanish / French / German
        # / Japanese / Korean voiceover word-aligns for exact sync too.
        _MODEL = WhisperModel("base", device="cpu", compute_type="int8")
    return _MODEL


def whisper_words(audio_path: str | Path) -> list[tuple[str, float, float]]:
    """[(word, start, end)] for the whole audio, or [] on failure."""
    try:
        segments, _ = _model().transcribe(
            str(audio_path),
            word_timestamps=True,
            beam_size=1,
            condition_on_previous_text=False,
            vad_filter=True,
        )
        out: list[tuple[str, float, float]] = []
        last = 0.0
        for seg in segments:
            for w in (seg.words or []):
                txt = (w.word or "").strip()
                if not txt:
                    continue
                st, en = float(w.start), float(w.end)
                # faster-whisper's feature extractor can overflow on some
                # audio ("invalid value encountered in matmul") and emit
                # NaN/inf or wildly out-of-order timestamps. Drop anything
                # non-finite or non-monotonic so a bad spectrogram can
                # never poison the scene boundaries downstream.
                if not (math.isfinite(st) and math.isfinite(en)):
                    continue
                if en < st or st < last - 1.0:
                    continue
                out.append((txt, st, en))
                last = max(last, en)
        return out
    except Exception:
        return []


def transcript_text(audio_path: str | Path) -> str:
    """Plain readable transcript of an audio file, or '' on failure.

    Used for the VOICEOVER-FIRST flow: when the user uploads their own
    narration we transcribe it and make THAT the script, so (a) the
    footage is matched to the words actually spoken and (b) word-level
    alignment is exact (the script == the audio). Joins Whisper's word
    tokens and tidies the spacing around punctuation so the sentence
    splitter downstream sees clean sentences.
    """
    words = whisper_words(audio_path)
    if not words:
        return ""
    raw = " ".join(w for w, _, _ in words if w)
    # Whisper emits punctuation as/with separate tokens after .strip();
    # pull stray spaces back before punctuation and collapse doubles.
    raw = re.sub(r"\s+([,.!?;:])", r"\1", raw)
    raw = re.sub(r"\s{2,}", " ", raw).strip()
    return raw


def align_script(
    audio_path: str | Path, script_words: list[str]
) -> list[tuple[float, float]] | None:
    """Map each entry of ``script_words`` to its real (start, end) second
    in ``audio_path``. Returns a list the SAME length as script_words, or
    None if alignment is not usable.

    Whisper's transcript ~= the script (it's being read aloud) but not
    identical (numbers spelled out, punctuation, the odd misheard word).
    So we sequence-align the two normalised token streams: matched words
    get their exact spoken time; unmatched script words are linearly
    interpolated between the surrounding matched anchors. Robust to small
    ASR errors without ever leaving a word un-timed.
    """
    hyp = whisper_words(audio_path)
    if not hyp or not script_words:
        return None

    s_norm = [_norm(w) for w in script_words]
    h_norm = [_norm(w) for w, _, _ in hyp]
    if not any(s_norm) or not any(h_norm):
        return None

    times: list[tuple[float, float] | None] = [None] * len(script_words)
    sm = SequenceMatcher(a=s_norm, b=h_norm, autojunk=False)
    for i1, j1, n in sm.get_matching_blocks():
        for k in range(n):
            _, st, en = hyp[j1 + k]
            times[i1 + k] = (st, en)

    n_anchor = sum(1 for t in times if t is not None)
    if not n_anchor:
        return None
    # Coverage guard: if Whisper only matched a small fraction of the
    # script (e.g. the mel spectrogram overflowed partway and the rest
    # was never transcribed), the unmatched tail gets extrapolated onto
    # ONE timestamp -> one scene balloons to tens of seconds while the
    # rest collapse. That is unusable; bail to proportional timing.
    if n_anchor < max(8, int(0.45 * len(script_words))):
        return None
    last_anchor_i = max(i for i, t in enumerate(times) if t is not None)
    if last_anchor_i < 0.5 * len(script_words):
        return None  # huge un-anchored tail -> boundaries would be junk

    # Fill gaps: before first anchor, after last, and between anchors —
    # spread evenly across the time span so nothing is left None.
    anchors = [(i, t) for i, t in enumerate(times) if t is not None]
    first_i, first_t = anchors[0]
    for i in range(first_i):
        times[i] = (max(0.0, first_t[0]), first_t[0])
    last_i, last_t = anchors[-1]
    for i in range(last_i + 1, len(times)):
        times[i] = (last_t[1], last_t[1])
    for (ia, ta), (ib, tb) in zip(anchors, anchors[1:]):
        if ib - ia <= 1:
            continue
        gap = ib - ia
        t0, t1 = ta[1], tb[0]
        step = (t1 - t0) / gap if t1 > t0 else 0.0
        for k in range(1, gap):
            s = t0 + step * (k - 1)
            e = t0 + step * k
            times[ia + k] = (s, max(s, e))

    out = [t if t is not None else (0.0, 0.0) for t in times]
    # Sanity: must be (roughly) monotonic and span a real duration.
    span = out[-1][1] - out[0][0]
    if span <= 0.5:
        return None
    return out
