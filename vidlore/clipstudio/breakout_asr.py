"""Short-window ASR for real-audio Breakouts.

Whisper can return a plausible but truncated transcript for a 7-10 second
movie-dialogue clip.  A single pass then makes both admission and captions
believe the returned prefix is the whole utterance.  Overlapping short windows
make the tail independently observable and are shared by both callers.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
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


# CONTENT-ADDRESSED TRANSCRIPT CACHE
#
# This function is the expensive half of breakout selection: every candidate window is decoded into
# 3-5 overlapping chunks and each chunk is put through Whisper. A breakout pass over ~25 quotes
# across ~41 sources therefore re-transcribes the same audio many times over — the same source
# window is reached again by a different quote, by the next build pass after self-heal, and by every
# resume — and the stage was measured taking hours, which is why breakouts were switched off for an
# emergency render. Nothing about that work is a judgement; it is the same bytes producing the same
# words.
#
# The cache is keyed on the AUDIO'S OWN BYTES, never on a path or an mtime: a wav rewritten in place
# with different content gets a different key, and a cached entry whose stored digest no longer
# matches the file is discarded rather than trusted. The key also carries the ASR model identity,
# the windowing parameters and a schema version, because any of those changes the words. Written
# atomically via a temp file and os.replace, so an interrupted run leaves either the old entry or
# the new one and never a half-written transcript.
#
# This is a pure memo of a deterministic computation. It cannot approve anything: every gate that
# consumed a fresh transcript consumes the cached one identically.
_ASR_CACHE_SCHEMA = 4
# hit/miss counters for the audit trail — reset by callers that want a per-stage figure
_ASR_CACHE_STATS = {"hit": 0, "miss": 0, "unidentified_model": 0, "incomplete": 0}

# The tag we stamp on a model WE construct, so its size is recoverable afterwards. faster-whisper
# does not keep it: see _model_identity.
_SPEC_ATTR = "_vidlore_asr_spec"


def _wav_digest(wav: Path) -> str:
    h = hashlib.sha256()
    with open(wav, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _decode_fingerprint() -> str:
    """The code that turns audio into words is itself an input to the transcript.

    A schema integer only invalidates the cache if a human remembers to raise it. These three
    functions decide every word returned — the decode flags, the chunk walk's overlap merge, and
    the token normalisation that merge compares on — so their source is hashed straight into the
    key. Editing any of them invalidates every entry automatically."""
    try:
        import inspect
        src = "".join(inspect.getsource(f)
                      for f in (_materialize_words, _dedupe_overlaps, _norm_token))
    except Exception:                                    # noqa: BLE001 — source unavailable
        return "unavailable"                             # a constant: falls back to the schema int
    return hashlib.sha256(src.encode("utf-8")).hexdigest()[:16]


_DECODE_FP = _decode_fingerprint()


def _model_identity(model) -> str | None:
    """A handle that changes whenever the model producing the words changes — or None.

    None means DO NOT CACHE. That is the whole point of this function. faster-whisper 1.2.1
    exposes neither `model_size_or_path` nor `model_path`, so the previous key fell all the way
    through to `type(model).__name__` — the string "WhisperModel", identical for `base` and
    `large-v3`. A key that cannot tell two models apart is not a cache; it is a way to serve one
    model's transcript as another's. Verified against the installed version: both attributes are
    absent, and the only real discriminators live on the CTranslate2 object underneath.

    So the size must be RESOLVABLE — from our own tag or from an attribute a future version
    restores — before anything is memoised. Everything else that can move the words (quantisation,
    device, mel bins, language head, library version) joins it. When the size cannot be resolved
    the caller simply transcribes: slower, never wrong."""
    spec = getattr(model, _SPEC_ATTR, "") \
        or getattr(model, "model_size_or_path", None) or getattr(model, "model_path", None)
    if not spec:
        return None
    ct2 = getattr(model, "model", None)
    bits = [f"spec={spec}"]
    for attr in ("compute_type", "device", "n_mels", "num_languages", "is_multilingual"):
        v = getattr(ct2, attr, None) if ct2 is not None else None
        if v is not None:
            bits.append(f"{attr}={v}")
    try:
        import faster_whisper as _fw
        bits.append(f"fw={getattr(_fw, '__version__', '?')}")
    except Exception:                                    # noqa: BLE001
        pass
    return "|".join(bits)


def _asr_cache_key(digest: str, model, window: float, overlap: float,
                   duration: float) -> dict | None:
    """Every input that can change the returned words — and nothing that cannot.

    `duration` is in the key because it is not merely descriptive: it decides where the chunk
    windows start, whether a tail window is anchored at all, and what the word end times are
    clamped to. Two calls over the same bytes with different declared durations legitimately
    produce different words."""
    ident = _model_identity(model)
    if ident is None:
        return None
    return {"schema": _ASR_CACHE_SCHEMA, "sha256": digest, "model": ident,
            "decode": _DECODE_FP,
            "window": round(float(window), 4), "overlap": round(float(overlap), 4),
            "duration": round(float(duration), 3)}


def _asr_cache_read(path: Path, key: dict):
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
    except Exception:                                    # noqa: BLE001 — unreadable == absent
        return None
    if not isinstance(blob, dict) or blob.get("key") != key:
        return None                                      # any input changed → recompute
    words = blob.get("words")
    if not isinstance(words, list):
        return None
    try:
        return [(str(w[0]), float(w[1]), float(w[2]), float(w[3])) for w in words]
    except Exception:                                    # noqa: BLE001 — corrupt payload
        return None


def _asr_cache_write(path: Path, key: dict, words: list) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(json.dumps({"key": key, "words": [list(w) for w in words]}),
                       encoding="utf-8")
        os.replace(tmp, path)                            # atomic: old entry or new, never partial
    except Exception:                                    # noqa: BLE001 — caching must never fail
        try:                                            # the transcription it is memoising
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def transcribe_breakout_words(wav_path, *, model=None, duration: float | None = None,
                              window: float = 3.8, overlap: float = 0.6,
                              cache: bool = True, with_status: bool = False):
    """Transcribe a Breakout in overlapping windows with clip-relative times.

    Memoised on the audio's content digest (see the note above). `cache=False` forces a fresh
    transcription, which is what the cache's own regression tests use to produce a control."""
    wav = Path(wav_path)
    if model is None:
        from faster_whisper import WhisperModel
        model = WhisperModel("base", device="cpu", compute_type="int8")
        # faster-whisper keeps no record of what it was asked to load, so remember it here —
        # without this the memo cannot tell this model from any other (see _model_identity).
        try:
            setattr(model, _SPEC_ATTR, "base")
        except Exception:                                # noqa: BLE001 — exotic model object
            pass
    if duration is None:
        duration = float(probe(wav).get("duration", 0.0) or 0.0)
    duration = float(duration or 0.0)
    if duration <= 0.05:
        return ([], {"complete": False, "chunks_planned": 0, "chunks_decoded": 0,
                     "words": 0}) if with_status else []
    window = max(1.5, float(window))
    overlap = min(max(0.2, float(overlap)), window * 0.45)
    _key = _cache_path = None
    if cache:
        try:
            _key = _asr_cache_key(_wav_digest(wav), model, window, overlap, duration)
            if _key is None:                             # unidentifiable model — transcribe, and
                _ASR_CACHE_STATS["unidentified_model"] += 1   # never risk another model's words
            else:
                _cache_path = wav.with_suffix(wav.suffix + ".asr.json")
                _hit = _asr_cache_read(_cache_path, _key)
                if _hit is not None:
                    _ASR_CACHE_STATS["hit"] += 1
                    # only complete observations are ever written, so a hit is complete
                    return ((_hit, {"complete": True, "chunks_planned": -1,
                                    "chunks_decoded": -1, "words": len(_hit)})
                            if with_status else _hit)
                _ASR_CACHE_STATS["miss"] += 1
        except Exception:                                # noqa: BLE001 — a cache fault must never
            _key = _cache_path = None                    # cost a transcript; fall through and work
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
    decoded = 0
    for offset in starts:
        span = min(window, duration - offset)
        with tempfile.TemporaryDirectory(prefix="breakout_asr_") as td:
            chunk = Path(td) / "chunk.wav"
            try:
                p = subprocess.run(
                    [ffmpeg_exe(), "-y", "-loglevel", "error", "-ss", f"{offset:.3f}",
                     "-t", f"{span:.3f}", "-i", str(wav), "-ar", "16000", "-ac", "1", str(chunk)],
                    capture_output=True, timeout=90)
            except subprocess.TimeoutExpired:
                continue                                 # a window we never heard — see below
            if p.returncode != 0 or not chunk.exists() or chunk.stat().st_size <= 44:
                continue
            try:
                for text, start, end, prob in _materialize_words(model, chunk):
                    observed.append((text, offset + start, min(duration, offset + end), prob))
            except Exception:                            # noqa: BLE001 — a decode that did not run
                continue
            decoded += 1
    _out = _dedupe_overlaps(observed)

    # A WINDOW WE FAILED TO LISTEN TO IS NOT A WINDOW THAT WAS SILENT.
    #
    # Each `continue` above drops a chunk on a transient fault — an ffmpeg hiccup, a decode error,
    # an extraction timeout. The words from that chunk are then simply absent, and absence reads
    # downstream as "nothing was said there". That inverts the caption-completeness gate, which
    # scores captioned/spoken: lose half the spoken words and coverage rises toward 1.0, so an
    # incomplete transcription looks MORE completely captioned than a real one. Persisting it made
    # the inflated verdict permanent for every later run over the same audio.
    #
    # So an incomplete observation is never written to the cache, and it is reported. A hit is
    # therefore complete by construction. Callers that gate on coverage must fail closed on
    # `complete=False` exactly as they do on no words at all.
    complete = (decoded == len(starts))
    if _key is not None and _cache_path is not None and complete:
        _asr_cache_write(_cache_path, _key, _out)
    if not complete:
        _ASR_CACHE_STATS["incomplete"] += 1
    if with_status:
        return _out, {"complete": complete, "chunks_planned": len(starts),
                      "chunks_decoded": decoded, "words": len(_out)}
    return _out


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
