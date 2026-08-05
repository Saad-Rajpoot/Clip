"""Stage 2 — index.

For each source: detect shots, transcribe (word-level), extract a keyframe per shot, embed it
with the engine's CLIP, and record face presence. Output per source:
  index/<sid>.shots.json   — Shot[]  (in/out, transcript, keyframe ref, face/embed refs)
  index/<sid>.embeds.npy   — float32 [n_shots, dim] CLIP keyframe embeddings
  index/<sid>/keyframes/   — shot_NNN.jpg

Reuses the engine's local ONNX CLIP (vidlore.visual_relevance) — no torch, no API. Heavy
imports (faster_whisper, scenedetect, numpy, PIL) are done lazily so importing this module is
cheap. Re-indexing is skipped when shots.json already exists (resume), unless force=True.
"""
from __future__ import annotations

import json
import hashlib
import inspect
import math
import os
import re
import subprocess
import uuid
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

from .models import Shot, SourceVideo, ClipProject, SOURCE_OK
from .config import ClipConfig, ffmpeg_exe
from .ingest import probe
from .segment import _STOP as _CONTENT_STOP

# Index schema. Bump to invalidate caches whose SHAPE changed (not merely their capabilities).
#   1 → shots + per-shot transcript only
#   2 → + <sid>.words.json (word-level ASR start/end/text) for quote-span location
INDEX_SCHEMA = 2
# Metadata-only evolution stays separate: INDEX_SCHEMA also versions embedding manifests, and an
# ASR provenance upgrade must not invalidate otherwise-current visual embeddings.
INDEX_ARTIFACT_BINDING_SCHEMA = 1
_ASR_PIPELINE_REVISION = 8
QUOTE_RETRIEVAL_SCHEMA = 2
_QUOTE_RETRIEVAL_REVISION = 2
_ASR_PROMPT_RESERVE_TOKENS = 16
_ASR_INITIAL_PROMPT_PREFIX = "Cast, character names, and quoted dialogue: "

_WHISPER = {}   # cache: exact Whisper constructor identity -> WhisperModel


# ---------------------------------------------------------------------------
# CLIP bridge
# ---------------------------------------------------------------------------

def clip_available() -> bool:
    try:
        from vidlore import visual_relevance as vr
        return bool(vr.available())
    except Exception:
        return False


def _vr():
    from vidlore import visual_relevance as vr
    return vr


# ---------------------------------------------------------------------------
# Shot detection
# ---------------------------------------------------------------------------

def detect_shots(path: Path, cfg: ClipConfig, total_dur: float = 0.0) -> list[tuple[float, float]]:
    """Content-aware shot boundaries → [(start,end)]. Falls back to fixed chunks on failure."""
    scenes: list[tuple[float, float]] = []
    try:
        from scenedetect import open_video, SceneManager, ContentDetector
        video = open_video(str(path))
        sm = SceneManager()
        sm.add_detector(ContentDetector(threshold=cfg.scene_threshold))
        sm.detect_scenes(video, show_progress=False)
        for s, e in sm.get_scene_list():
            scenes.append((float(s.get_seconds()), float(e.get_seconds())))
    except Exception:
        scenes = []
    if not scenes:
        # fallback: uniform chunks of ~2*target so matching still has candidates
        dur = total_dur or probe(path).get("duration", 0.0)
        step = max(2.0, cfg.target_clip_sec * 2)
        t = 0.0
        while t < dur:
            scenes.append((t, min(dur, t + step)))
            t += step
    # floor at min_clip_sec too: a shot shorter than the minimum cuttable clip would force
    # cut_selection to extend past the shot boundary into the NEXT, unrelated shot
    return _merge_short(scenes, max(cfg.min_shot_sec, cfg.min_clip_sec))


def _merge_short(scenes: list[tuple[float, float]], min_sec: float) -> list[tuple[float, float]]:
    """Merge sub-min_sec shots into the previous shot so every shot is usable."""
    if not scenes:
        return scenes
    out = [list(scenes[0])]
    for s, e in scenes[1:]:
        if (e - s) < min_sec:
            out[-1][1] = e                       # absorb into previous
        else:
            out.append([s, e])
    # ensure the very first shot meets min length too
    if len(out) > 1 and (out[0][1] - out[0][0]) < min_sec:
        out[1][0] = out[0][0]
        out.pop(0)
    return [(a, b) for a, b in out]


# ---------------------------------------------------------------------------
# Transcription (word-level)
# ---------------------------------------------------------------------------

def _whisper(cfg: ClipConfig):
    threads = int(getattr(cfg, "whisper_cpu_threads", 0) or 0)
    key = (str(cfg.whisper_model), str(cfg.whisper_compute), max(0, threads))
    if key not in _WHISPER:
        from faster_whisper import WhisperModel
        # cpu_threads: 0 = ctranslate2 default; a positive value pins the ASR pass to that many
        # cores (auto-scaled from the machine in config) so a powerful box transcribes much faster.
        _WHISPER[key] = WhisperModel(cfg.whisper_model, device="cpu",
                                     compute_type=cfg.whisper_compute,
                                     cpu_threads=max(0, threads))
    return _WHISPER[key]


def _asr_vocabulary_entry(value) -> str:
    """Return one comma-safe decoder-vocabulary entry.

    The bounded prompt uses commas as entry delimiters, while authored dialogue naturally contains
    commas.  Removing only that delimiter preserves the spoken words and prevents one quote from
    being split into several independently truncatable fragments.
    """
    text = str(value or "").translate(str.maketrans({"\u2018": "'", "\u2019": "'"}))
    return " ".join(text.replace(",", " ").split()).strip()


def _asr_vocabulary_key(value) -> str:
    """Punctuation-insensitive identity for deterministic vocabulary de-duplication."""
    text = _asr_vocabulary_entry(value).casefold()
    return " ".join(re.findall(r"[\w]+", text, flags=re.UNICODE))


def _asr_hotwords(roster) -> str:
    """Stable, comma-delimited vocabulary; order affects truncation and decoder output."""
    out: list[str] = []
    seen: set[str] = set()
    for raw in (roster or []):
        entry = _asr_vocabulary_entry(raw)
        key = _asr_vocabulary_key(entry)
        if not entry or not key or key in seen:
            continue
        seen.add(key)
        out.append(entry)
    return ", ".join(out)


def _project_authored_quotes(proj, analysis: dict) -> list[str]:
    """Collect current beat quotes plus analysis-authored anchor dialogue.

    Within the quote tier, compact lines come first so a fixed Whisper context carries the largest
    number of exact dialogue hints instead of being consumed by one long speech. Original beat
    order breaks ties, so the result stays stable when a reviewed guard moves a quote from the live
    beat into provenance metadata.
    """
    segments = list(getattr(proj, "segments", None) or [])
    rows: list[tuple[int, int, str]] = []

    def add(order, raw) -> None:
        try:
            stable_order = int(order)
        except (TypeError, ValueError):
            stable_order = len(segments) + len(rows)
        rows.append((stable_order, len(rows), str(raw or "")))

    for pos, seg in enumerate(segments):
        add(getattr(seg, "index", pos), getattr(seg, "quote", ""))
    anchor_order = max(
        [int(getattr(seg, "index", pos)) for pos, seg in enumerate(segments)] or [-1]) + 1
    for scene in (analysis.get("anchor_scenes") or []):
        if not isinstance(scene, dict):
            continue
        dialogue = scene.get("dialogue") or []
        if isinstance(dialogue, str):
            dialogue = [dialogue]
        for quote in dialogue:
            add(anchor_order, quote)
            anchor_order += 1

    # Deterministic contract normalization and reviewed footage-gap softening may clear a quote
    # from the active segment while retaining it as audited source-locator provenance.  Keep that
    # authored line in the decoder hints so a sanctioned softening does not invalidate its own ASR
    # generation immediately after mutation.  This is only a recognition hint; no removed quote is
    # restored as a publication promise by indexing it.
    grounding = analysis.get("beat_grounding_audit") or {}
    grounding_beats = grounding.get("beats") if isinstance(grounding, dict) else {}
    if isinstance(grounding_beats, dict):
        for beat_index, marker in grounding_beats.items():
            if isinstance(marker, dict):
                add(beat_index, marker.get("original_quote", ""))
    softening = (getattr(proj, "meta", {}) or {}).get(
        "selection_relevance_gap_softening") or {}
    for marker in (softening.get("beats") or []) if isinstance(softening, dict) else []:
        original = marker.get("original") if isinstance(marker, dict) else {}
        if isinstance(original, dict):
            add((marker.get("segment_index", marker.get("beat_index", marker.get("index")))
                 if isinstance(marker, dict) else None), original.get("quote", ""))

    # De-duplicate after restoring provenance rows to their original beat positions. Otherwise
    # clearing beat 0 and retaining its quote after beat 1 reverses equal-sized entries and forces
    # a second whole-pool ASR refresh despite an identical authored quote set.
    rows.sort(key=lambda row: (row[0], row[1]))
    unique: list[tuple[int, str]] = []
    seen: set[str] = set()
    for order, _encounter, raw in rows:
        quote = _asr_vocabulary_entry(raw)
        key = _asr_vocabulary_key(quote)
        if not quote or not key or key in seen:
            continue
        seen.add(key)
        unique.append((order, quote))
    unique.sort(key=lambda row: (len(row[1].split()), len(row[1]), row[0]))
    return [quote for _order, quote in unique]


def _project_asr_name_tiers(proj, fallback=None) -> tuple[list[str], list[str]]:
    """Return character and actor names separately so legacy prompting preserves priority."""
    analysis = (getattr(proj, "meta", {}) or {}).get("analysis") or {}
    characters = [str(row.get("name", "") or "")
                  for row in (analysis.get("characters") or []) if isinstance(row, dict)]
    actors = list(analysis.get("actors") or [])
    if not characters and not actors:
        characters = [fallback] if isinstance(fallback, str) else list(fallback or [])
    return characters, actors


def _project_asr_hotwords(proj, fallback=None) -> str:
    """Proper-name hotwords only; sentence-level dialogue must never enter decoder bias."""
    # Character names are the words the show actually speaks, so keep them first when an unusually
    # large cast must be bounded to Whisper's context. Actor names remain useful for interviews and
    # credits, but must not evict a scripted character name such as Cersei.
    characters, actors = _project_asr_name_tiers(proj, fallback=fallback)
    return _asr_hotwords([*characters, *actors])


def _project_authored_retrieval_prompt(proj) -> str:
    """Authored lines for an optional, separate retrieval-only decode.

    This value must never be passed to the decoder that produces the general ``.words.json``
    stream. Even context-only sentence prompting was reproduced to fabricate the prompted words
    repeatedly across unrelated audio. A future recall-oriented retrieval artifact may use this
    prompt only when it is separately named/provenanced and every hit receives an independent
    zero-prompt narrow-window confirmation before it can become evidence.
    """
    analysis = (getattr(proj, "meta", {}) or {}).get("analysis") or {}
    return _asr_hotwords(_project_authored_quotes(proj, analysis))


def _project_asr_initial_prompt(proj, fallback=None) -> str:
    """No sentence prompt is permitted in the persisted general word index."""
    del proj, fallback
    return ""


def _project_asr_legacy_initial_prompt(proj, fallback=None) -> str:
    """Proper-name-only fallback for Faster-Whisper versions without ``hotwords``."""
    return _project_asr_hotwords(proj, fallback=fallback)


def _project_quote_retrieval_legacy_prompt(proj, fallback=None) -> str:
    """Old-decoder single-channel equivalent of names + authored retrieval lines."""
    analysis = (getattr(proj, "meta", {}) or {}).get("analysis") or {}
    characters, actors = _project_asr_name_tiers(proj, fallback=fallback)
    return _asr_hotwords([
        *characters,
        *_project_authored_quotes(proj, analysis),
        *actors,
    ])


def _faster_whisper_version() -> str:
    try:
        from importlib.metadata import version
        return str(version("faster-whisper") or "unknown")
    except Exception:
        return "unknown"


def _asr_prompt_fingerprint(cfg: ClipConfig, hotwords: str,
                            initial_prompt: Optional[str] = None,
                            legacy_initial_prompt: Optional[str] = None) -> str:
    """Identity of every persisted-ASR input/options dependency relevant to word evidence."""
    payload = {
        "revision": _ASR_PIPELINE_REVISION,
        "model": str(getattr(cfg, "whisper_model", "") or ""),
        "compute": str(getattr(cfg, "whisper_compute", "") or ""),
        "faster_whisper": _faster_whisper_version(),
        "hotwords": str(hotwords or ""),
        "initial_prompt": str("" if initial_prompt is None else initial_prompt),
        "legacy_initial_prompt": str(
            hotwords if legacy_initial_prompt is None else legacy_initial_prompt),
        "dialogue_delivery": "excluded_from_general_word_index",
        "primary": {"word_timestamps": True, "vad_filter": True},
        "eof_rescue": {
            "window_sec": _ASR_EOF_WINDOW_SEC,
            "gap_sec": _ASR_EOF_RESCUE_GAP_SEC,
            "max_no_speech": _ASR_EOF_MAX_NO_SPEECH,
            "min_avg_logprob": _ASR_EOF_MIN_AVG_LOGPROB,
            "vad_filter": False,
            "condition_on_previous_text": False,
        },
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _quote_retrieval_fingerprint(proj, cfg: ClipConfig, *, fallback=None) -> str:
    """Identity of the separate authored-prompt candidate stream.

    Its bytes are never generic transcript evidence. A phrase located here is only a retrieval
    hint for the independent narrow zero-prompt confirmer.
    """
    payload = {
        "schema_version": QUOTE_RETRIEVAL_SCHEMA,
        "revision": _QUOTE_RETRIEVAL_REVISION,
        "model": str(getattr(cfg, "whisper_model", "") or ""),
        "compute": str(getattr(cfg, "whisper_compute", "") or ""),
        "faster_whisper": _faster_whisper_version(),
        "hotwords": _project_asr_hotwords(proj, fallback=fallback),
        "initial_prompt": _project_authored_retrieval_prompt(proj),
        "legacy_initial_prompt": _project_quote_retrieval_legacy_prompt(
            proj, fallback=fallback),
        "evidence_role": "retrieval_candidates_only_requires_unprompted_confirmation",
        "prompt_chunking": {
            "policy": "complete_authored_quote_coverage_v1",
            "entries_are_atomic": True,
            "each_entry_delivered_exactly_once": True,
            "streams_are_scanned_independently": True,
        },
        "primary": {"word_timestamps": True, "vad_filter": True},
        "eof_rescue": {
            "window_sec": _ASR_EOF_WINDOW_SEC,
            "gap_sec": _ASR_EOF_RESCUE_GAP_SEC,
            "max_no_speech": _ASR_EOF_MAX_NO_SPEECH,
            "min_avg_logprob": _ASR_EOF_MIN_AVG_LOGPROB,
            "vad_filter": False,
            "condition_on_previous_text": False,
        },
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _legacy_rev7_general_fingerprint(proj, cfg: ClipConfig, *, fallback=None) -> str:
    """Exact rev7 identity, used only to salvage already-computed prompted bytes as retrieval."""
    hotwords = _project_asr_hotwords(proj, fallback=fallback)
    prompt = _project_authored_retrieval_prompt(proj) or hotwords
    legacy = _project_quote_retrieval_legacy_prompt(proj, fallback=fallback)
    payload = {
        "revision": 7,
        "model": str(getattr(cfg, "whisper_model", "") or ""),
        "compute": str(getattr(cfg, "whisper_compute", "") or ""),
        "faster_whisper": _faster_whisper_version(),
        "hotwords": hotwords,
        "initial_prompt": prompt,
        "legacy_initial_prompt": legacy,
        "dialogue_delivery": "initial_prompt_only",
        "primary": {"word_timestamps": True, "vad_filter": True},
        "eof_rescue": {
            "window_sec": _ASR_EOF_WINDOW_SEC,
            "gap_sec": _ASR_EOF_RESCUE_GAP_SEC,
            "max_no_speech": _ASR_EOF_MAX_NO_SPEECH,
            "min_avg_logprob": _ASR_EOF_MIN_AVG_LOGPROB,
            "vad_filter": False,
            "condition_on_previous_text": False,
        },
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def asr_semantic_fingerprint(proj, cfg: ClipConfig) -> str:
    """Public checkpoint/cache identity for the ASR evidence a project requires."""
    return _asr_prompt_fingerprint(
        cfg, _project_asr_hotwords(proj), _project_asr_initial_prompt(proj),
        _project_asr_legacy_initial_prompt(proj))


def _token_ids(tokenizer, text: str) -> Optional[list[int]]:
    """Return tokenizer IDs across HuggingFace-tokenizers and simple test doubles."""
    try:
        try:
            encoded = tokenizer.encode(text, add_special_tokens=False)
        except TypeError:
            encoded = tokenizer.encode(text)
        ids = getattr(encoded, "ids", encoded)
        return [int(token) for token in ids]
    except Exception:
        return None


def _bound_asr_vocabulary(model, hotwords: str, *, duplicated: bool) -> str:
    """Keep the complete-name prefix that provably fits Whisper's decoder context.

    Faster-Whisper independently clips ``hotwords`` and ``initial_prompt`` to half of the context,
    then concatenates both. Near the boundary that can exceed the model's 448-token limit. Project
    vocabularies are comma-delimited so this function drops whole low-priority names, never half a
    name. The conservative character fallback is only for old/test models without a tokenizer.
    """
    raw = str(hotwords or "").strip()
    if not raw:
        return ""
    entries = [entry.strip() for entry in raw.split(",") if entry.strip()]
    if not entries:
        return ""
    tokenizer = getattr(model, "hf_tokenizer", None)
    max_length = max(32, int(getattr(model, "max_length", 448) or 448))
    half_limit = max(1, max_length // 2 - 1)

    def fits(value: str) -> bool:
        initial = f"{_ASR_INITIAL_PROMPT_PREFIX}{value}"
        if tokenizer is None:
            # Whisper uses byte-level BPE, so UTF-8 byte count is a conservative token upper bound
            # even for old wrappers/test doubles that do not expose ``hf_tokenizer``.
            initial_bound = len(initial.encode("utf-8"))
            hotword_bound = len(value.encode("utf-8")) if duplicated else 0
            return (initial_bound <= half_limit
                    and hotword_bound <= half_limit
                    and initial_bound + hotword_bound + _ASR_PROMPT_RESERVE_TOKENS
                    <= max_length)
        initial_ids = _token_ids(tokenizer, " " + initial.strip())
        hotword_ids = _token_ids(tokenizer, " " + value.strip()) if duplicated else []
        if initial_ids is None or hotword_ids is None:
            return False
        return (len(initial_ids) <= half_limit
                and len(hotword_ids) <= half_limit
                and len(initial_ids) + len(hotword_ids) + _ASR_PROMPT_RESERVE_TOKENS
                <= max_length)

    kept: list[str] = []
    for entry in entries:
        candidate = ", ".join([*kept, entry])
        if fits(candidate):
            kept.append(entry)
        else:
            # Inputs are priority ordered. Once the prefix is full, later entries cannot be added
            # without making prompt identity/order surprising on resume.
            break
    return ", ".join(kept)


def _model_supports_hotwords(model) -> bool:
    try:
        params = inspect.signature(model.transcribe).parameters.values()
        return any(
            p.name == "hotwords" or p.kind == inspect.Parameter.VAR_KEYWORD for p in params)
    except (TypeError, ValueError):
        return False


def _transcribe_with_vocabulary(model, path: Path, *, hotwords: str = "",
                                initial_prompt: str = "", legacy_initial_prompt: str = "",
                                **kwargs):
    """Call Faster-Whisper without breaking supported 1.0.x installs.

    ``hotwords`` arrived after 1.0.0 while requirements intentionally support 1.0+.  Every version
    accepts ``initial_prompt``; pass the stronger hotword hint only when the installed method (or a
    test double's **kwargs) advertises it.  Never retry a TypeError after decoding starts.
    """
    requested_prompt = str(initial_prompt or "")
    if requested_prompt or hotwords:
        supports_hotwords = _model_supports_hotwords(model)
        if supports_hotwords:
            prompt_vocabulary = requested_prompt
        else:
            # Indexing supplies character -> compact-dialogue -> actor priority explicitly. Keep a
            # compatible fallback for direct callers that only know the two modern channels.
            prompt_vocabulary = str(legacy_initial_prompt or "") or _asr_hotwords([
                *[entry.strip() for entry in str(hotwords or "").split(",") if entry.strip()],
                *[entry.strip() for entry in requested_prompt.split(",") if entry.strip()],
            ])
        if prompt_vocabulary:
            # Bound the complete context as though it were duplicated when hotwords are supported.
            bounded_prompt = _bound_asr_vocabulary(
                model, prompt_vocabulary, duplicated=bool(supports_hotwords and hotwords))
            if not bounded_prompt:
                # The requested vocabulary is part of persisted ASR identity. Decoding without it
                # and stamping the full fingerprint would be false provenance.
                raise RuntimeError("ASR vocabulary could not be bounded/tokenized safely")
            kwargs["initial_prompt"] = f"{_ASR_INITIAL_PROMPT_PREFIX}{bounded_prompt}"
        if supports_hotwords and hotwords:
            bounded_hotwords = _bound_asr_vocabulary(model, hotwords, duplicated=True)
            if not bounded_hotwords:
                raise RuntimeError("ASR hotwords could not be bounded/tokenized safely")
            kwargs["hotwords"] = bounded_hotwords
    return model.transcribe(str(path), **kwargs)


_ASR_EOF_WINDOW_SEC = 30.0
_ASR_EOF_RESCUE_GAP_SEC = 4.0
_ASR_EOF_MAX_NO_SPEECH = 0.60
_ASR_EOF_MIN_AVG_LOGPROB = -1.0


def _timed_words(segments, *, lo: float = 0.0, hi: float = float("inf")) \
        -> list[tuple[float, float, str]]:
    """Materialise only real, finite Whisper word intervals.

    A no-VAD pass can emit an otherwise plausible sentence whose words are all pinned to EOF with
    zero duration.  Those are decoder hallucinations, not timed audio evidence, and must never enter
    the quote index.
    """
    out: list[tuple[float, float, str]] = []
    for seg in segments:
        for w in (seg.words or []):
            try:
                start, end = float(w.start), float(w.end)
            except (TypeError, ValueError):
                continue
            text = str(w.word or "").strip()
            clipped_start, clipped_end = max(lo, start), min(hi, end)
            if (text and math.isfinite(start) and math.isfinite(end)
                    and end > start + 0.005 and start >= lo - 0.25 and end <= hi + 0.25
                    and clipped_end > clipped_start + 0.005):
                out.append((clipped_start, clipped_end, text))
    return out


def _merge_eof_words(primary: list[tuple[float, float, str]],
                     rescue: list[tuple[float, float, str]]) \
        -> list[tuple[float, float, str]]:
    """Add independently observed EOF words without duplicating the overlapping prefix."""
    out = list(primary)
    for start, end, word in rescue:
        norm = _norm_tok(word)
        mid = (start + end) / 2.0
        duplicate = any(
            norm and norm == _norm_tok(old_word)
            and abs(mid - ((old_start + old_end) / 2.0)) <= 0.8
            for old_start, old_end, old_word in out
        )
        if not duplicate:
            out.append((start, end, word))
    out.sort(key=lambda row: (row[0], row[1]))
    return out


def _rescue_eof_words_result(path: Path, model, primary: list[tuple[float, float, str]],
                             duration: float, *, hotwords: str = "",
                             initial_prompt: str = "", legacy_initial_prompt: str = "") \
        -> tuple[bool, list[tuple[float, float, str]]]:
    """Recover trailing dialogue that whole-source Silero VAD can omit.

    The independent pass is narrowly EOF-anchored and disables VAD/previous-text conditioning.  It
    is attempted only when the primary stream leaves a real tail gap, and accepted only when at
    least two newly timed words form a coherent utterance.  This avoids promoting a one-token outro
    hallucination into quote evidence.
    """
    duration = float(duration or 0.0)
    if duration <= 2.0:
        return True, primary
    last_end = max((float(row[1]) for row in primary), default=0.0)
    if duration - last_end < _ASR_EOF_RESCUE_GAP_SEC:
        return True, primary
    tail_start = max(0.0, duration - _ASR_EOF_WINDOW_SEC)
    try:
        segments, _info = _transcribe_with_vocabulary(
            model, path, word_timestamps=True, vad_filter=False,
            condition_on_previous_text=False,
            clip_timestamps=[tail_start, duration],
            hotwords=hotwords,
            initial_prompt=initial_prompt,
            legacy_initial_prompt=legacy_initial_prompt,
        )
        # no-VAD is required precisely because Silero missed this tail, but that also makes silent
        # outro hallucinations possible.  Faster-Whisper supplies independent decoder confidence
        # per segment; fail closed when it is missing/non-finite or says silence/low likelihood.
        credible = []
        confidence_unverifiable = False
        for seg in segments:
            try:
                no_speech = float(seg.no_speech_prob)
                avg_logprob = float(seg.avg_logprob)
            except (AttributeError, TypeError, ValueError):
                confidence_unverifiable = True
                continue
            if not (math.isfinite(no_speech) and math.isfinite(avg_logprob)):
                confidence_unverifiable = True
                continue
            if (no_speech <= _ASR_EOF_MAX_NO_SPEECH
                    and avg_logprob >= _ASR_EOF_MIN_AVG_LOGPROB):
                credible.append(seg)
        if confidence_unverifiable:
            return False, primary
        rescue = _timed_words(credible, lo=tail_start, hi=duration)
    except Exception:
        # The rescue was triggered because the primary decoder left a material unknown tail. A
        # technical failure here cannot certify that tail as silence or absence of a real quote.
        return False, primary

    # Require a coherent multi-word observation strictly after the persisted stream.  Existing
    # overlap words are still merged below, but cannot by themselves authorize the rescue.
    new = [row for row in rescue if row[0] >= last_end + 0.25]
    coherent = 1 if new else 0
    best = coherent
    for prev, cur in zip(new, new[1:]):
        coherent = coherent + 1 if cur[0] - prev[1] <= 4.0 else 1
        best = max(best, coherent)
    if best < 2:
        return True, primary
    return True, _merge_eof_words(primary, rescue)


def _rescue_eof_words(path: Path, model, primary: list[tuple[float, float, str]],
                      duration: float, *, hotwords: str = "",
                      initial_prompt: str = "", legacy_initial_prompt: str = "") \
        -> list[tuple[float, float, str]]:
    """Compatibility wrapper; persistence callers use the result form's success bit."""
    return _rescue_eof_words_result(
        path, model, primary, duration, hotwords=hotwords,
        initial_prompt=initial_prompt, legacy_initial_prompt=legacy_initial_prompt)[1]


def _transcribe_words_result(path: Path, cfg: ClipConfig, *, duration: float = 0.0,
                             hotwords: str = "", initial_prompt: str = "",
                             legacy_initial_prompt: str = "") \
        -> tuple[bool, list[tuple[float, float, str]]]:
    """Return ``(decode_succeeded, timed_words)`` without conflating silence and failure."""
    try:
        model = _whisper(cfg)
        segments, _info = _transcribe_with_vocabulary(
            model, path, word_timestamps=True, vad_filter=True, hotwords=hotwords,
            initial_prompt=initial_prompt, legacy_initial_prompt=legacy_initial_prompt)
        words = _timed_words(segments)
    except Exception:
        return False, []
    if not duration:
        try:
            duration = float(probe(path).get("duration", 0.0) or 0.0)
        except Exception:
            # Duration discovery only controls the optional rescue.  Never discard a successful
            # primary transcript, but do not certify EOF completeness when its boundary is unknown.
            return False, words
    return _rescue_eof_words_result(
        path, model, words, duration, hotwords=hotwords,
        initial_prompt=initial_prompt, legacy_initial_prompt=legacy_initial_prompt)


def transcribe_words(path: Path, cfg: ClipConfig, *, duration: float = 0.0,
                     hotwords: str = "", initial_prompt: str = "",
                     legacy_initial_prompt: str = "") \
        -> list[tuple[float, float, str]]:
    """Compatibility wrapper returning words; use result form when persistence depends on success."""
    return _transcribe_words_result(
        path, cfg, duration=duration, hotwords=hotwords,
        initial_prompt=initial_prompt, legacy_initial_prompt=legacy_initial_prompt)[1]


def transcribe_unprompted_window(
        path: Path | str, cfg: ClipConfig, start: float, end: float, *,
        sample_rate: int = 16000,
        max_no_speech: float = _ASR_EOF_MAX_NO_SPEECH,
        min_avg_logprob: float = _ASR_EOF_MIN_AVG_LOGPROB) -> dict:
    """Decode one physically extracted audio window with no authored decoder hints.

    Project ASR is deliberately prompted to improve recall. Its hits are retrieval candidates,
    not independent proof that an authored line exists. This narrow pass receives only waveform
    samples: no ``initial_prompt``, no ``hotwords``, no previous-text conditioning, and no whole-
    source feature extraction. The caller binds/caches the result to immutable source/quote inputs.

    ``status=ok`` means decoding completed and every returned segment carried finite confidence;
    it does not mean a particular quote matched. Any technical or confidence uncertainty is
    ``inconclusive`` so publication callers can fail closed.
    """
    try:
        lo, hi = float(start), float(end)
        rate = int(sample_rate)
        no_speech_floor = float(max_no_speech)
        logprob_floor = float(min_avg_logprob)
    except (TypeError, ValueError):
        return {"status": "inconclusive", "reason": "invalid_decode_parameters"}
    if (not math.isfinite(lo) or not math.isfinite(hi) or not hi > lo >= 0.0
            or rate < 8000 or not math.isfinite(no_speech_floor)
            or not math.isfinite(logprob_floor)):
        return {"status": "inconclusive", "reason": "invalid_decode_parameters"}

    try:
        from . import audio_align as _audio_window
        pcm, decode_error = _audio_window._pcm_from_media(
            path, lo, hi, sample_rate=rate)
    except Exception as exc:
        return {
            "status": "inconclusive",
            "reason": f"audio_extract_error:{type(exc).__name__}",
        }
    if decode_error:
        return {"status": "inconclusive", "reason": str(decode_error)}

    try:
        model = _whisper(cfg)
        # Do not route through `_transcribe_with_vocabulary`: even an empty authored prompt can
        # acquire proper-name hotwords there. This pass must be acoustically independent.
        segments, _info = model.transcribe(
            pcm, word_timestamps=True, vad_filter=False,
            condition_on_previous_text=False)
        materialized = list(segments)
    except Exception as exc:
        return {
            "status": "inconclusive",
            "reason": f"unprompted_decode_error:{type(exc).__name__}",
        }

    confidence_rows: list[dict] = []
    credible = []
    rejected_confidence = False
    if not materialized:
        return {"status": "inconclusive", "reason": "segment_confidence_absent"}
    for seg in materialized:
        try:
            no_speech = float(seg.no_speech_prob)
            avg_logprob = float(seg.avg_logprob)
        except (AttributeError, TypeError, ValueError):
            return {"status": "inconclusive", "reason": "segment_confidence_missing"}
        if not (math.isfinite(no_speech) and math.isfinite(avg_logprob)):
            return {"status": "inconclusive", "reason": "segment_confidence_nonfinite"}
        accepted = bool(no_speech <= no_speech_floor and avg_logprob >= logprob_floor)
        confidence_rows.append({
            "no_speech_prob": round(no_speech, 6),
            "avg_logprob": round(avg_logprob, 6),
            "accepted": accepted,
        })
        if accepted:
            credible.append(seg)
        else:
            rejected_confidence = True

    # A low-confidence segment is not evidence that the authored phrase was absent.  The phrase
    # may be inside exactly the segment the decoder marked uncertain, so caching that observation
    # as a conclusive negative would incorrectly turn a real quote into a paraphrase.  Keep the
    # independent pass fail-closed: any uncertain decoded segment makes this request inconclusive.
    if rejected_confidence:
        return {
            "status": "inconclusive",
            "reason": "segment_confidence_below_floor",
            "sample_rate_hz": rate,
            "decode_window": [round(lo, 3), round(hi, 3)],
            "segment_confidence": confidence_rows,
        }

    relative = _timed_words(credible, lo=0.0, hi=hi - lo)
    absolute = sorted(
        [[round(lo + a, 3), round(lo + b, 3), str(word)]
         for a, b, word in relative],
        key=lambda row: (row[0], row[1]))
    if not absolute:
        return {
            "status": "inconclusive",
            "reason": "timed_words_absent",
            "sample_rate_hz": rate,
            "decode_window": [round(lo, 3), round(hi, 3)],
            "segment_confidence": confidence_rows,
        }
    return {
        "status": "ok",
        "reason": "",
        "sample_rate_hz": rate,
        "decode_window": [round(lo, 3), round(hi, 3)],
        "timed_words": absolute,
        "segment_confidence": confidence_rows,
    }


def _assign_transcript(shots: list[Shot], words: list[tuple[float, float, str]]) -> None:
    """Attach to each shot the words whose midpoint falls inside it.

    NOTE: this bins by midpoint, so a single spoken SENTENCE routinely splits across two shots
    whenever the cut lands mid-line. That is correct for "what is said during this shot" but makes
    per-shot transcripts useless for locating a QUOTE — see `find_quote_span`, which matches the
    word stream instead. The (ws, we) timings are preserved separately by `save_words`."""
    if not words:
        return
    for sh in shots:
        toks = []
        for (ws, we, wt) in words:
            mid = (ws + we) / 2.0
            if sh.start <= mid < sh.end:
                toks.append(wt)
        sh.transcript = " ".join(toks).strip()


# --- word-level ASR: persistence + quote location -------------------------------------------
# The shipped render proved why this must survive indexing. `transcribe_words` already produced
# word timings, then _assign_transcript threw the timecodes away and kept only per-shot strings.
# Two measured consequences in tywin_lannister_dismis_f4c81b75:
#   * "Any man who must say I am the king is no true king" straddles a cut, so it lands as
#     "...Any man who must" + "say I am the king is no true king." — no single shot contains it,
#     so a per-shot substring search can never find the most iconic line in the video.
#   * The breakout in-point could only be a SHOT boundary (130.05s), 0.41s after that line ends
#     (129.64s) — and the window only extends forward, so the line was unreachable by construction.
def _norm_tok(t: str) -> str:
    return re.sub(r"[^a-z0-9']", "", (t or "").lower())


def save_words(proj: ClipProject, source_id: str, words: list[tuple[float, float, str]]) -> None:
    f = proj.index_dir / f"{source_id}.words.json"
    tmp = f.with_suffix(".json.tmp")
    tmp.write_text(json.dumps([[round(float(a), 3), round(float(b), 3), str(c)] for a, b, c in words]),
                   encoding="utf-8")
    tmp.replace(f)


def _load_words_result(proj: ClipProject, source_id: str) \
        -> tuple[bool, list[tuple[float, float, str]]]:
    """Distinguish a valid silent ``[]`` cache from missing, corrupt, or malformed JSON."""
    f = proj.index_dir / f"{source_id}.words.json"
    if not f.exists():
        return False, []
    try:
        payload = json.loads(f.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            return False, []
        words: list[tuple[float, float, str]] = []
        for row in payload:
            if not isinstance(row, (list, tuple)) or len(row) != 3:
                return False, []
            start, end, text = float(row[0]), float(row[1]), str(row[2])
            if (not math.isfinite(start) or not math.isfinite(end)
                    or start < 0.0 or end <= start or not text.strip()):
                return False, []
            words.append((start, end, text))
        return True, words
    except Exception:
        return False, []


def load_words(proj: ClipProject, source_id: str) -> list[tuple[float, float, str]]:
    return _load_words_result(proj, source_id)[1]


def _quote_retrieval_path(proj: ClipProject, source_id: str) -> Path:
    return Path(proj.index_dir) / f"{source_id}.quote_retrieval.json"


def _source_content_fingerprint(path) -> str:
    try:
        from .verify import _file_fingerprint
        return str(_file_fingerprint(path) or "")
    except Exception:
        return ""


def _artifact_content_sha256(path) -> str:
    """Full content identity for a small persisted index artifact."""
    try:
        candidate = Path(path)
        if not candidate.is_file():
            return ""
        digest = hashlib.sha256()
        with candidate.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        return digest.hexdigest()
    except Exception:
        return ""


def _index_artifact_bindings(proj: ClipProject, source: SourceVideo) -> dict:
    """Current byte identities for the ASR/index tuple certified by index.meta."""
    sid = str(getattr(source, "id", "") or "")
    return {
        "artifact_binding_schema": INDEX_ARTIFACT_BINDING_SCHEMA,
        "source_content_fingerprint": _source_content_fingerprint(
            str(getattr(source, "local_path", "") or "")),
        "words_content_sha256": _artifact_content_sha256(
            Path(proj.index_dir) / f"{sid}.words.json"),
        "shots_content_sha256": _artifact_content_sha256(proj.shots_path(sid)),
    }


def _index_artifact_provenance_result(
        proj: ClipProject, source: SourceVideo, meta: dict | None = None) \
        -> tuple[bool, str, dict]:
    """Validate that current source/words/shots bytes are the tuple certified by meta."""
    if not isinstance(meta, dict):
        try:
            meta = json.loads((Path(proj.index_dir)
                               / f"{source.id}.index.meta.json").read_text(
                                   encoding="utf-8"))
        except Exception:
            meta = {}
    if not isinstance(meta, dict):
        return False, "index_artifact_binding_metadata_invalid", {}
    try:
        binding_schema = int(meta.get("artifact_binding_schema", 0) or 0)
    except (TypeError, ValueError):
        binding_schema = 0
    if binding_schema != INDEX_ARTIFACT_BINDING_SCHEMA:
        return False, "index_artifact_binding_schema_missing_or_stale", {}
    current = _index_artifact_bindings(proj, source)
    fields = (
        "source_content_fingerprint", "words_content_sha256", "shots_content_sha256",
    )
    for field in fields:
        actual = str(current.get(field, "") or "")
        expected = str(meta.get(field, "") or "")
        if not expected:
            return False, f"{field}_missing", current
        if not actual:
            return False, f"{field}_unavailable", current
        if actual != expected:
            return False, f"{field}_mismatch", current
    return True, "", current


def _quote_retrieval_prompt_chunks(
        proj, cfg: ClipConfig, *, entries=None, fallback=None, model=None) -> tuple[list[str], str]:
    """Partition authored lines into prompts the installed decoder will actually receive.

    `_transcribe_with_vocabulary` intentionally drops whole suffix entries at the context limit.
    Retrieval absence is only meaningful when every authored line was delivered, so repeat that
    exact bounding policy until the full ordered quote list is covered.
    """
    authored = list(entries if entries is not None else _project_authored_quotes(
        proj, (getattr(proj, "meta", {}) or {}).get("analysis") or {}))
    if not authored:
        return [], ""
    try:
        model = model or _whisper(cfg)
        hotwords = _project_asr_hotwords(proj, fallback=fallback)
        duplicated = bool(_model_supports_hotwords(model) and hotwords)
        chunks: list[str] = []
        remaining = list(authored)
        while remaining:
            bounded = _bound_asr_vocabulary(
                model, _asr_hotwords(remaining), duplicated=duplicated)
            if not bounded:
                return [], "authored_quote_exceeds_decoder_context"
            delivered = [entry.strip() for entry in bounded.split(", ") if entry.strip()]
            if not delivered or delivered != remaining[:len(delivered)]:
                return [], "authored_quote_chunk_boundary_unverifiable"
            chunks.append(bounded)
            remaining = remaining[len(delivered):]
        return chunks, ""
    except Exception as exc:
        return [], f"authored_quote_chunking_error:{type(exc).__name__}"


def _normalized_retrieval_words(words) -> list[list] | None:
    rows = []
    try:
        for start, end, text in words:
            start, end, text = float(start), float(end), str(text)
            if (not math.isfinite(start) or not math.isfinite(end)
                    or start < 0.0 or end <= start or not text.strip()):
                return None
            rows.append([round(start, 3), round(end, 3), text])
    except Exception:
        return None
    return rows


def _save_quote_retrieval_streams(
        proj: ClipProject, source: SourceVideo, cfg: ClipConfig, streams: list[dict], *,
        fallback=None) -> bool:
    """Atomically persist independently-scanned prompt chunks and delivered quote coverage."""
    source_id = str(getattr(source, "id", "") or "")
    source_path = str(getattr(source, "local_path", "") or "")
    source_fingerprint = _source_content_fingerprint(source_path)
    if not source_id or source_fingerprint in ("", "missing", "unreadable"):
        return False
    normalized_streams = []
    for stream in streams or []:
        if not isinstance(stream, dict):
            return False
        covered = [_asr_vocabulary_entry(value)
                   for value in (stream.get("covered_authored_quotes") or [])]
        if not covered or any(not value for value in covered):
            return False
        prompt = str(stream.get("prompt", "") or "")
        if prompt != _asr_hotwords(covered):
            return False
        rows = _normalized_retrieval_words(stream.get("words") or [])
        if rows is None:
            return False
        normalized_streams.append({
            "prompt": prompt,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "covered_authored_quotes": covered,
            "words": rows,
        })
    payload = {
        "schema_version": QUOTE_RETRIEVAL_SCHEMA,
        "source_id": source_id,
        "source_content_fingerprint": source_fingerprint,
        "retrieval_fingerprint": _quote_retrieval_fingerprint(
            proj, cfg, fallback=fallback),
        "evidence_role": "retrieval_candidates_only_requires_unprompted_confirmation",
        "streams": normalized_streams,
    }
    path = _quote_retrieval_path(proj, source_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    # A portal retry and a manual repair can overlap briefly.  Keep each atomic writer's staging
    # name unique so one valid sidecar cannot delete or replace another writer's temporary bytes.
    tmp = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        tmp.replace(path)
    except OSError:
        return False
    finally:
        tmp.unlink(missing_ok=True)
    return True


def _save_quote_retrieval_words(
        proj: ClipProject, source: SourceVideo, cfg: ClipConfig,
        words: list[tuple[float, float, str]], *, fallback=None) -> bool:
    """Compatibility helper for tests/single-chunk callers; refuses silently truncated prompts."""
    expected = _project_authored_quotes(
        proj, (getattr(proj, "meta", {}) or {}).get("analysis") or {})
    class _ConservativeSingleChunkModel:
        # Whisper's decoder context is 448. With no tokenizer, `_bound_asr_vocabulary` uses UTF-8
        # bytes as a conservative token upper bound, so a one-chunk result cannot hide truncation.
        max_length = 448
        hf_tokenizer = None

        @staticmethod
        def transcribe(_path, **_kwargs):
            return iter(()), None

    chunks, _reason = _quote_retrieval_prompt_chunks(
        proj, cfg, entries=expected, fallback=fallback,
        model=_ConservativeSingleChunkModel())
    if len(chunks) != 1:
        return False
    return _save_quote_retrieval_streams(proj, source, cfg, [{
        "prompt": chunks[0],
        "covered_authored_quotes": expected,
        "words": words,
    }], fallback=fallback)


def _load_quote_retrieval_streams_result(
        proj: ClipProject, source: SourceVideo | str, cfg: ClipConfig, *, fallback=None,
        require_complete: bool = True) -> tuple[bool, list[dict], str, bool]:
    """Validate retrieval chunks, source bytes, generation, and per-quote delivered coverage."""
    if isinstance(source, str):
        source = proj.source(source)
    if source is None:
        return False, [], "source_missing", False
    source_id = str(getattr(source, "id", "") or "")
    try:
        raw = json.loads(_quote_retrieval_path(proj, source_id).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return False, [], "quote_retrieval_missing", False
    except Exception:
        return False, [], "quote_retrieval_unreadable", False
    if not isinstance(raw, dict) or raw.get("schema_version") != QUOTE_RETRIEVAL_SCHEMA:
        return False, [], "quote_retrieval_schema_invalid", False
    if str(raw.get("source_id", "") or "") != source_id:
        return False, [], "quote_retrieval_source_id_mismatch", False
    expected_retrieval = _quote_retrieval_fingerprint(proj, cfg, fallback=fallback)
    if str(raw.get("retrieval_fingerprint", "") or "") != expected_retrieval:
        return False, [], "quote_retrieval_fingerprint_mismatch", False
    source_fingerprint = _source_content_fingerprint(
        getattr(source, "local_path", "") or "")
    if (source_fingerprint in ("", "missing", "unreadable")
            or source_fingerprint != str(raw.get("source_content_fingerprint", "") or "")):
        return False, [], "quote_retrieval_source_content_mismatch", False
    if raw.get("evidence_role") != \
            "retrieval_candidates_only_requires_unprompted_confirmation":
        return False, [], "quote_retrieval_role_invalid", False
    payload = raw.get("streams")
    if not isinstance(payload, list):
        return False, [], "quote_retrieval_streams_invalid", False
    expected = _project_authored_quotes(
        proj, (getattr(proj, "meta", {}) or {}).get("analysis") or {})
    streams: list[dict] = []
    coverage: list[str] = []
    for stream in payload:
        if not isinstance(stream, dict):
            return False, [], "quote_retrieval_stream_invalid", False
        covered = [_asr_vocabulary_entry(value)
                   for value in (stream.get("covered_authored_quotes") or [])]
        prompt = str(stream.get("prompt", "") or "")
        if (not covered or prompt != _asr_hotwords(covered)
                or str(stream.get("prompt_sha256", "") or "")
                != hashlib.sha256(prompt.encode("utf-8")).hexdigest()):
            return False, [], "quote_retrieval_prompt_coverage_invalid", False
        rows = _normalized_retrieval_words(stream.get("words") or [])
        if rows is None:
            return False, [], "quote_retrieval_words_invalid", False
        coverage.extend(covered)
        streams.append({
            "prompt": prompt,
            "covered_authored_quotes": covered,
            "words": [(float(a), float(b), str(text)) for a, b, text in rows],
        })
    if coverage != expected[:len(coverage)] or len(coverage) > len(expected):
        return False, [], "quote_retrieval_coverage_order_invalid", False
    complete = coverage == expected
    if require_complete and not complete:
        return False, streams, "quote_retrieval_coverage_incomplete", False
    return True, streams, "", complete


def _load_quote_retrieval_words_result(
        proj: ClipProject, source: SourceVideo | str, cfg: ClipConfig, *, fallback=None) \
        -> tuple[bool, list[tuple[float, float, str]], str]:
    """Compatibility flattened view; proof paths must scan chunk streams independently."""
    valid, streams, reason, _complete = _load_quote_retrieval_streams_result(
        proj, source, cfg, fallback=fallback, require_complete=True)
    return valid, [word for stream in streams for word in stream["words"]], reason


def load_quote_retrieval_words(
        proj: ClipProject, source: SourceVideo | str, cfg: ClipConfig, *, fallback=None) \
        -> list[tuple[float, float, str]]:
    return _load_quote_retrieval_words_result(
        proj, source, cfg, fallback=fallback)[1]


# Sources whose ASR could not be re-transcribed under the CURRENT authored prompt. They keep their
# preserved words for every other purpose, but they cannot prove a quote under a prompt they were
# never decoded with, so quote retrieval and whole-pool absence must not count them. Populated by
# index_source; consulted by quote_retrieval_source_eligible.
_asr_upgrade_refused: set = set()

_ASR_REFUSED_META_KEY = "asr_upgrade_refused"


def _remember_asr_refusal(proj, sid: str) -> None:
    """Record the bar in process state AND on the project.

    The in-memory set is what `quote_retrieval_source_eligible` can reach — it is handed a source
    and its shots, never the project. But a render that resumes runs in a NEW process, where that
    set is empty, and the pool audit would then demand quote evidence from a source this run had
    already refused to take any from: the abort simply returns. So the bar is also written to the
    project, and `index_all` seeds the set from there before it indexes anything."""
    _asr_upgrade_refused.add(sid)
    try:
        meta = getattr(proj, "meta", None)
        if isinstance(meta, dict):
            cur = list(meta.get(_ASR_REFUSED_META_KEY) or [])
            if sid not in cur:
                cur.append(sid)
                meta[_ASR_REFUSED_META_KEY] = cur
    except Exception:                                    # noqa: BLE001 — bookkeeping, never fatal
        pass


def _seed_asr_refusals(proj) -> None:
    """Restore bars recorded by an earlier process before any audit consults them."""
    try:
        meta = getattr(proj, "meta", None)
        if isinstance(meta, dict):
            for sid in (meta.get(_ASR_REFUSED_META_KEY) or []):
                if isinstance(sid, str) and sid:
                    _asr_upgrade_refused.add(sid)
    except Exception:                                    # noqa: BLE001
        pass


def quote_retrieval_source_eligible(source: SourceVideo, shots: list[Shot]) -> bool:
    """Use the same trusted-show-audio boundary as whole-pool quote classification."""
    if str(getattr(source, "id", "") or "") in _asr_upgrade_refused:
        return False                       # not decoded under this prompt — cannot prove a quote
    title = str(getattr(source, "title", "") or "")
    try:
        from .build import _breakout_src_ok, _ESSAYISH_RX
        from .discover import _REACTION_TITLE
        if _ESSAYISH_RX.search(title) or _REACTION_TITLE.search(title):
            return False
        return bool(shots and _breakout_src_ok(source, shots))
    except Exception:
        # Eligibility uncertainty cannot certify whole-pool absence. The relevance contract records
        # it as incomplete; indexing likewise refuses to pretend no retrieval artifact is needed.
        return True


def asr_pool_cache_audit(proj, cfg: ClipConfig, sources=None) -> dict:
    """Cheap artifact check used before Resume trusts match and downstream checkpoints."""
    expected = asr_semantic_fingerprint(proj, cfg)
    invalid: list[dict] = []
    seen: set[str] = set()
    selected_shots: dict[str, set[int]] = {}
    for selection in (getattr(proj, "selections", None) or []):
        selected_sid = str(getattr(selection, "source_id", "") or "")
        if not selected_sid:
            continue
        try:
            selected_index = int(getattr(selection, "shot_index", -1))
        except (TypeError, ValueError):
            selected_index = -1
        if selected_index >= 0:
            selected_shots.setdefault(selected_sid, set()).add(selected_index)
    pool = (getattr(proj, "sources", None) or []) if sources is None else (sources or [])
    for source in pool:
        sid = str(getattr(source, "id", "") or "")
        if (not sid or sid in seen
                or str(getattr(source, "status", "") or "") != SOURCE_OK):
            continue
        seen.add(sid)
        try:
            shots = load_shots(proj, sid)
        except Exception:
            shots = []
        if not shots:
            invalid.append({"source_id": sid, "reason": "shots_cache_invalid_or_missing"})
            continue
        indexed_shots = {int(getattr(shot, "index", -1)) for shot in shots}
        if not selected_shots.get(sid, set()).issubset(indexed_shots):
            invalid.append({"source_id": sid, "reason": "selected_shot_missing_from_index"})
            continue
        valid_words, _words = _load_words_result(proj, sid)
        if not valid_words:
            invalid.append({"source_id": sid, "reason": "words_cache_invalid_or_missing"})
            continue
        meta_file = Path(proj.index_dir) / f"{sid}.index.meta.json"
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
        except Exception:
            meta = None
        if not isinstance(meta, dict):
            invalid.append({"source_id": sid, "reason": "index_metadata_invalid_or_missing"})
            continue
        if meta.get("asr_refresh_in_progress"):
            invalid.append({"source_id": sid, "reason": "asr_refresh_interrupted"})
            continue
        if meta.get("words") is not True:
            invalid.append({"source_id": sid, "reason": "word_evidence_not_certified"})
            continue
        try:
            schema_current = int(meta.get("schema", 0) or 0) >= INDEX_SCHEMA
        except (TypeError, ValueError):
            schema_current = False
        if not schema_current:
            invalid.append({"source_id": sid, "reason": "word_evidence_schema_stale"})
            continue
        if str(meta.get("asr_prompt_fingerprint", "") or "") != expected \
                and sid not in _asr_upgrade_refused:
            # A source barred from quote retrieval is OUT of the whole-pool quote scope, so its
            # transcript currency is not part of what that scope's soundness rests on. Demanding a
            # current prompt fingerprint from a source we have already refused to take quote
            # evidence FROM would only relocate the abort — and it did exactly that: the barred
            # sources reappeared here as `asr_prompt_fingerprint_mismatch` and stopped a 72-minute
            # run at the completeness check instead of at the source. It is excluded from the
            # requirement because it is excluded from the claim, never to make the claim easier.
            invalid.append({"source_id": sid, "reason": "asr_prompt_fingerprint_mismatch"})
            continue
        provenance_ok, provenance_reason, _provenance = \
            _index_artifact_provenance_result(proj, source, meta)
        if not provenance_ok:
            invalid.append({"source_id": sid, "reason": provenance_reason})
            continue
        if (_project_authored_retrieval_prompt(proj)
                and quote_retrieval_source_eligible(source, shots)):
            retrieval_ok, _retrieval_streams, retrieval_reason, retrieval_complete = \
                _load_quote_retrieval_streams_result(
                    proj, source, cfg, require_complete=True)
            if not retrieval_ok or not retrieval_complete:
                invalid.append({"source_id": sid, "reason": retrieval_reason})
    for sid in sorted(set(selected_shots) - seen):
        invalid.append({"source_id": sid, "reason": "selected_source_not_in_usable_pool"})
    return {
        "expected_asr_prompt_fingerprint": expected,
        "expected_quote_retrieval_fingerprint": (
            _quote_retrieval_fingerprint(proj, cfg)
            if _project_authored_retrieval_prompt(proj) else ""),
        "source_count": len(seen),
        "current_count": len(seen) - len(invalid),
        "invalid": invalid,
    }


def asr_pool_current(proj, cfg: ClipConfig, sources=None) -> bool:
    return not asr_pool_cache_audit(proj, cfg, sources).get("invalid")


def _tok_close(a: str, b: str, thresh: float = 0.8) -> bool:
    """Two tokens are 'the same word' if ASR merely garbled it. Never used to accept a match on its
    own — only ever as one term inside the PHRASE score in `find_quote_span`."""
    if a == b:
        return True
    if not a or not b or abs(len(a) - len(b)) > 3:
        return False
    return SequenceMatcher(None, a, b).ratio() >= thresh


def _matched_quote_positions(win, qt, matcher) -> set:
    """Which quote tokens a window actually accounted for — blocks plus positional fuzzy credit."""
    got = set()
    for bl in matcher.get_matching_blocks():
        for k in range(bl.size):
            got.add(bl.b + k)
    for idx, (a, b) in enumerate(zip(win, qt)):
        if a != b and _tok_close(a, b):
            got.add(idx)
    return got


# Only genuinely adjacent audio may be re-attached below.
#
# This number is deliberately not a tuned threshold, and that is measured rather than asserted.
# Across 50,033 real interword gaps from two jobs' ASR streams the median and p85 gap are both
# 0.000s and 85.5% are <= 0.0s — word timestamps are contiguous by construction, and a real pause
# is far outside this range. Sweeping the bound over 0.0 / 0.05 / 0.2 / 0.5 / 1.0 / 2.0s against 40
# real authored quotes x 150 real streams accepts exactly 51 matches at EVERY value: the constant
# is inert over its whole plausible range, so no outcome here rests on picking it.
_QUOTE_CONTIGUITY_S = 0.0


def _quote_window_pays_for_what_it_skips(st, i, L, qt, n, win, matcher, min_ratio: float) -> bool:
    """Would this window still qualify if it had to include the words it stopped short of?

    THE ARBITRAGE THIS CLOSES. The phrase score is ``2*(hits+fuzzy)/(L+n)``, so for a fixed number
    of matched tokens a SHORTER window scores HIGHER. Ending one word early is worth free ratio,
    and the search sweeps L across a slack range and keeps the best — which systematically rewards
    truncation. Measured case: authored quote "I want you to be my cupbearer." against audio that
    actually says "Good. I want you to be happy." (three independent decodes agree). Stopping after
    "be" gives 2*5/(5+7) = 0.833 and clears the 0.78 floor; including the next word "happy" gives
    2*5/(6+7) = 0.769 and does not. The quote's last two tokens are simply not in that audio.

    So when a window leaves quote tokens unaccounted for at either end it is extended by at most
    that many stream words — and only across genuinely adjacent audio, never across a pause — then
    re-scored with the identical formula. If the honest score no longer clears the caller's own
    floor, the candidate is refused.

    No new threshold is introduced: the bar is the caller's `min_ratio`, unchanged. This can only
    reject; it can never admit a window the current scorer refuses.
    """
    got = _matched_quote_positions(win, qt, matcher)
    if not got:
        return False
    lead_miss, trail_miss = min(got), n - 1 - max(got)
    if lead_miss <= 0 and trail_miss <= 0:
        return True                                   # the whole quote is accounted for
    lo, hi = i, i + L                                 # hi exclusive
    for _ in range(max(0, lead_miss)):
        if lo <= 0 or float(st[lo][1]) - float(st[lo - 1][2]) > _QUOTE_CONTIGUITY_S:
            break
        lo -= 1
    for _ in range(max(0, trail_miss)):
        if hi >= len(st) or float(st[hi][1]) - float(st[hi - 1][2]) > _QUOTE_CONTIGUITY_S:
            break
        hi += 1
    if lo == i and hi == i + L:
        return True                                   # nothing adjacent to re-attach
    win2 = [x[0] for x in st[lo:hi]]
    m2 = SequenceMatcher(None, win2, qt)
    hits2 = sum(bl.size for bl in m2.get_matching_blocks())
    fuzzy2 = sum(1 for a, b in zip(win2, qt) if a != b and _tok_close(a, b))
    return (2.0 * (hits2 + fuzzy2)) / float(len(win2) + n) >= min_ratio


def find_quote_span(words, quote: str, *, min_ratio: float = 0.72,
                    window_slack: int = 3, max_interword_gap: float = 4.0,
                    all_matches: bool = False):
    """Locate `quote` in a source's word stream. -> (start_s, end_s, ratio) or None.

    Aligns the quote against a sliding window of the word stream and scores the WHOLE PHRASE. A
    single garbled word cannot carry a match (that would let "sleep" alone anchor anywhere), and a
    single garbled word cannot break one either — which is what the measured data demands:

        source ASR : "Maester. Perhaps" | "a messence of nightshade to help him" | "sleep."
        script quote: "Perhaps some essence of nightshade to help him sleep."

    "messence"/"essence" and "a"/"some" differ, yet the phrase is unmistakably the same line. Shot
    boundaries are irrelevant here by construction — the stream is continuous.

    A word stream is continuous across *shots*, but it is not normally one utterance across an
    arbitrarily long silence.  Prefer a qualifying compact alignment over one that bridges such a
    gap: an isolated earlier ``I`` plus ``did warn you not to trust me`` sixteen seconds later must
    not beat the compact local line.  Keep a qualifying gapped alignment as fallback, though,
    because a real dramatic pause must remain typed as verbatim.  The phrase floor is unchanged."""
    qt = [t for t in (_norm_tok(w) for w in re.findall(r"[\w']+", quote or "")) if t]
    if len(qt) < 2 or not words:
        return [] if all_matches else None
    # A high score made solely from function words is not phrase proof.  Measured false positive:
    # quote "He was a monster" matched ASR "He was a fighter" at 0.857 because the slack window
    # was allowed to stop after the common prefix "he was a", omitting the only identifying word.
    # Require one substantive query token to survive exact/fuzzy alignment when the quote has one;
    # `_tok_close` retains the existing ASR-garble tolerance.
    substantive = [t for t in qt if t not in _CONTENT_STOP]
    st = [(_norm_tok(w[2]), w[0], w[1]) for w in words]
    st = [x for x in st if x[0]]
    if not st:
        return [] if all_matches else None
    n = len(qt)
    best = None
    gapped_best = None
    best_raw_ratio = None
    gapped_best_raw_ratio = None
    occurrences: list[tuple[float, float, float]] = []
    for i in range(len(st)):
        if not _tok_close(st[i][0], qt[0]) and not any(_tok_close(st[i][0], q) for q in qt[:2]):
            continue                      # cheap anchor: only start where the head plausibly begins
        local_best = None
        local_gapped_best = None
        local_best_raw_ratio = None
        local_gapped_raw_ratio = None
        for L in range(max(2, n - window_slack), min(len(st) - i, n + window_slack) + 1):
            span_words = st[i:i + L]
            # Prefer a compact local utterance over an apparent phrase assembled across a long
            # silence.  Do NOT discard the latter outright: a character can genuinely pause for
            # >4s mid-line, and whole-pool quote typing must still classify that authored quote as
            # verbatim.  A qualifying gapped alignment is retained as a fallback only when no
            # qualifying compact alignment exists.
            gapped = bool(max_interword_gap > 0 and any(
                float(b[1]) - float(a[2]) > max_interword_gap
                for a, b in zip(span_words, span_words[1:])))
            win = [x[0] for x in span_words]
            if substantive and not any(
                    _tok_close(actual, wanted)
                    for actual in win for wanted in substantive):
                continue
            m = SequenceMatcher(None, win, qt)
            hits = sum(bl.size for bl in m.get_matching_blocks())
            # credit near-miss tokens the block matcher rejected (ASR garble), positionally
            fuzzy = sum(1 for a, b in zip(win, qt) if a != b and _tok_close(a, b))
            ratio = (2.0 * (hits + fuzzy)) / float(L + n)
            # Two admissibility clauses, both reusing the caller's own `min_ratio` and neither
            # able to admit anything the scorer above refuses.
            #   * a window may not buy ratio by stopping short of the quote (see the helper)
            #   * a bridged alignment may not ALSO swallow speech the quote does not contain. A
            #     dramatic pause is a real utterance; a pause with someone else's words inside it
            #     is not the line. Without this, denying truncation merely displaces the match onto
            #     the gapped fallback.
            if ratio >= min_ratio:
                if not _quote_window_pays_for_what_it_skips(
                        st, i, L, qt, n, win, m, min_ratio):
                    continue
                if gapped and (L - (hits + fuzzy)) != 0:
                    continue
            target = gapped_best if gapped else best
            target_raw_ratio = gapped_best_raw_ratio if gapped else best_raw_ratio
            candidate_start = st[i][1]
            candidate_end = st[i + L - 1][2]
            # Whisper occasionally emits the same words twice: an early alignment whose first
            # word is stretched across several seconds, followed by the real compact utterance.
            # Both alignments have the same phrase ratio, and the old first-wins tie-break kept the
            # smeared span.  That made Window-QC shorten across several intervening shots and then
            # (correctly) reject the candidate for losing quote containment.  Phrase strength is
            # still the primary criterion; for an exact tie prefer the tighter temporal alignment.
            # This cannot turn a non-match into a match or lower the verbatim floor.
            tighter_tie = bool(
                target is not None
                and target_raw_ratio is not None
                and abs(float(ratio) - float(target_raw_ratio)) <= 1e-12
                and (float(candidate_end) - float(candidate_start))
                < (float(target[1]) - float(target[0])))
            if ratio >= min_ratio and (
                    target is None or ratio > float(target_raw_ratio) or tighter_tie):
                target = (candidate_start, candidate_end, round(min(1.0, ratio), 3))
                if gapped:
                    gapped_best = target
                    gapped_best_raw_ratio = ratio
                else:
                    best = target
                    best_raw_ratio = ratio
            local_target = local_gapped_best if gapped else local_best
            local_raw = local_gapped_raw_ratio if gapped else local_best_raw_ratio
            local_tighter = bool(
                local_target is not None and local_raw is not None
                and abs(float(ratio) - float(local_raw)) <= 1e-12
                and (float(candidate_end) - float(candidate_start))
                < (float(local_target[1]) - float(local_target[0])))
            if ratio >= min_ratio and (
                    local_target is None or ratio > float(local_raw) or local_tighter):
                local_target = (
                    candidate_start, candidate_end, round(min(1.0, ratio), 3))
                if gapped:
                    local_gapped_best = local_target
                    local_gapped_raw_ratio = ratio
                else:
                    local_best = local_target
                    local_best_raw_ratio = ratio
        local = local_best or local_gapped_best
        if local is not None:
            occurrences.append(local)
    if all_matches:
        # One best alignment per plausible starting word preserves repeated occurrences, including
        # the measured case where a tight prompted hallucination outranks a nearby real line. Exact
        # tuple de-dup removes only identical windows; overlapping/nested windows stay independent
        # retrieval candidates and each must face narrow unprompted confirmation.
        unique = list({(round(float(row[0]), 3), round(float(row[1]), 3),
                        round(float(row[2]), 3)) for row in occurrences})
        return sorted(unique, key=lambda row: (-row[2], row[1] - row[0], row[0]))
    return best or gapped_best


# ---------------------------------------------------------------------------
# Keyframes
# ---------------------------------------------------------------------------

def purge_source_index(proj, sid: str) -> None:
    """Delete every persisted index artifact for one source. REQUIRED whenever a source's
    MEDIA FILE is replaced after indexing (the 403 HD sweep swaps a 360p file for the HD
    copy): the index cache keys on shots-file existence + capability meta, NOT the media
    checksum, so stale 360p embeds/flags would silently survive the swap and change
    decisions. Missing files are fine (fresh source)."""
    import shutil
    try:
        for suf in (".shots.json", ".embeds.npy", ".embeds.manifest.json",
                    ".words.json", ".quote_retrieval.json", ".index.meta.json"):
            (proj.index_dir / f"{sid}{suf}").unlink(missing_ok=True)
        d = proj.index_dir / sid
        if d.is_dir():
            shutil.rmtree(d, ignore_errors=True)
    except Exception:                                    # noqa: BLE001 — purge is best-effort;
        pass                                             # a partial purge still fails the cache check


def extract_keyframe(path: Path, t: float, dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [ffmpeg_exe(), "-y", "-ss", f"{max(0.0, t):.3f}", "-i", str(path),
           "-frames:v", "1", "-q:v", "3", "-vf", "scale=512:-2", str(dest)]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=60)
        return dest.exists() and dest.stat().st_size > 0
    except Exception:
        return False


# ---------------------------------------------------------------------------
# MULTI-FRAME shot flags (start/mid/end samples) — the single keyframe misses
# INTERMITTENT defects: burned subs that appear mid-shot (observed: a Turkish
# "Gel, geçip odamda konuşalım." aired although the shot's keyframe was clean),
# fading channel bugs, and shots that are readable at the keyframe instant but
# near-black for most of their span. Computed once at index time, persisted on
# the Shot, and reused by every gate (match/build/stills/breakouts) without
# re-running detection.
# ---------------------------------------------------------------------------

# corner-bug mask geometry — MUST stay in sync with match._source_corner_logo (v3, calibrated):
# analysis at 640×360, 4 corner regions of 44×116 px, edge threshold 18, presence >2% of pixels.
_CORNER_REGIONS = {"tl": (slice(0, 44), slice(0, 116)), "tr": (slice(0, 44), slice(524, 640)),
                   "bl": (slice(316, 360), slice(0, 116)), "br": (slice(316, 360), slice(524, 640))}
_MASK_GRID = (11, 29)          # 44×116 region → 4×4 blocks


def _mask_to_hex(mask) -> str:
    import numpy as np
    bits = np.packbits(mask.astype("uint8").flatten())
    return bits.tobytes().hex()


def _mask_from_hex(h: str):
    import numpy as np
    try:
        bits = np.unpackbits(np.frombuffer(bytes.fromhex(h), dtype="uint8"))
        n = _MASK_GRID[0] * _MASK_GRID[1]
        if len(bits) < n:
            return None
        return bits[:n].reshape(_MASK_GRID).astype(bool)
    except Exception:
        return None


def _flags_from_frames(frames: list) -> dict:
    """Pure flag computation from up to 3 grayscale 640×360 float arrays (start/mid/end).
    Returns {subs_flag, text_conf, luma_avg, luma_hi, corner_masks}. Deterministic and
    frame-source-agnostic so it is unit-testable without video files."""
    import numpy as np
    out = {"subs_flag": 0, "text_conf": 0.0, "luma_avg": -1.0, "luma_hi": -1.0,
           "luma_min": -1.0, "luma_min_black_frac": -1.0, "corner_masks": {},
           "static_frac": -1.0, "pair_diff_max": -1.0, "pair_diff_mean": -1.0}
    frames = [f for f in frames if f is not None]
    if not frames:
        out["subs_flag"] = -1
        return out
    # STATIC-FRAME metrics — consecutive-pair mean|Δ|. A frozen image airing as footage
    # (thumbnail collage, AI-art still, promo composite) diffs at ~0 between samples; real
    # footage — even locked-off candlelit dialogue — carries codec grain and micro-motion
    # (measured live-action floor on 220 control shots: 0.97). Calibrated on job 5462677f95;
    # per-pair FREEZE threshold env-tunable.
    if len(frames) >= 2:
        _sth = float(os.environ.get("VIDLORE_CLIPSTUDIO_STATIC_PAIR_DIFF", "0.9") or 0.9)
        _pairs = [float(np.abs(frames[i + 1] - frames[i]).mean())
                  for i in range(len(frames) - 1)]
        out["static_frac"] = round(sum(1 for d in _pairs if d < _sth) / len(_pairs), 3)
        out["pair_diff_max"] = round(max(_pairs), 2)
        out["pair_diff_mean"] = round(sum(_pairs) / len(_pairs), 2)
    lumas, hi_vals, confs, dark_fracs = [], [], [], []
    corner_or = {k: None for k in _CORNER_REGIONS}
    subs_any = False
    for fr in frames:
        lumas.append(float(fr.mean()))
        hi_vals.append(float(np.percentile(fr, 99.8)))
        dark_fracs.append(float((fr < 16.0).mean()))    # black-pixel share of THIS sample
        # subtitle band on the 320×180 downsample — SAME calibrated geometry/thresholds as
        # match._shot_subtitle_band (Turkish/Arabic/English positives, 30+ clean negatives)
        small = fr[::2, ::2]
        gy, gx = np.gradient(small)
        E = np.hypot(gx, gy)
        band = E[137:175, 38:282] > 42.0
        bf = float(band.mean())
        mf = float((E[72:126, 38:282] > 42.0).mean())
        ys, xs = np.nonzero(band)
        colcov = len(np.unique(xs // 8)) / (244 // 8) if len(xs) else 0.0
        rowspread = ys.std() if len(ys) > 20 else 99.0
        confs.append(bf / max(mf, 0.008))
        if bf > 0.05 and bf > 2.0 * max(mf, 0.008) and colcov >= 0.28 and rowspread < 11.0:
            subs_any = True
        elif len(ys) >= 80 and mf <= 0.004 and rowspread < 8.0:
            # SHORT-LINE branch (calibrated on a leaked 2-word Turkish sub over a dark scene,
            # 0 FPs on 180 clean frames): a sparse but WIDE, row-tight stroke box in an
            # otherwise-empty band — too few edge pixels for the density path above.
            _h = ys.max() - ys.min() + 1
            _w = xs.max() - xs.min() + 1
            if _w >= 150 and _h <= 34:
                subs_any = True
        # corner edge masks at full 640×360 (same threshold as the source-level detector)
        gy2, gx2 = np.gradient(fr)
        E2 = np.hypot(gx2, gy2)
        for name, (rs, cs) in _CORNER_REGIONS.items():
            m = (E2[rs, cs] > 18.0)
            grid = m.reshape(_MASK_GRID[0], 4, _MASK_GRID[1], 4).mean(axis=(1, 3)) > 0.25
            corner_or[name] = grid if corner_or[name] is None else (corner_or[name] | grid)
    out["subs_flag"] = 1 if subs_any else 0
    out["text_conf"] = round(max(confs), 2) if confs else 0.0
    out["luma_avg"] = round(float(np.mean(lumas)), 1)
    out["luma_hi"] = round(float(max(hi_vals)), 1)
    # DARKEST SAMPLE. luma_avg is the mean across the shot's samples and luma_hi the brightest
    # highlight anywhere in it, so a shot that is black for half its span but carries a torch
    # reads as perfectly fine: the v2 render aired a beat whose shot measured avg 39.4 / hi 255
    # yet whose delivered frames sat at mean luma 2.5 with 100% of pixels below 16. Nothing in
    # the persisted stats could see an intra-shot dark SPAN. This is that missing number.
    out["luma_min"] = round(float(min(lumas)), 1)
    # ...and how much of the darkest sample is genuinely black, so a low-key-but-lit frame
    # (deep shadow round a face) is separable from an unusable one.
    out["luma_min_black_frac"] = round(float(dark_fracs[int(np.argmin(lumas))]), 3) \
        if dark_fracs else -1.0
    for name, g in corner_or.items():
        if g is not None and g.sum() >= 2:            # store only corners with real edge content
            out["corner_masks"][name] = _mask_to_hex(g)
    return out


# ── NON-SHOW GRAPHICS flag (per-shot, embed-space) ───────────────────────────────────────────
# Designed graphics that must NEVER air as footage: broadcast-news CGI, video-game-UI parody
# frames, cartoon/comic segments, posters, and painterly fan art. Observed leaks: a teal
# sports-news intro aired on a connector beat (the verifier rationalized it as "a transition"),
# and Jaime/Cersei fan art aired via an unverified secondary beat window from an illustrated
# book-essay source. Computed from the SAME persisted CLIP embedding the index already stores,
# so the gate costs dot products, not decodes.
#
# Rule (calibrated on a real 50-source render, 1957 shots — the parody/illustrated sources score
# 28.6-94.6% of shots vs <=6.1% for every real-footage source; 0 live-action false positives):
#   graphic_dom = max(sim to graphic anchors) - max(sim to real-photo anchors)
#   HARD  : graphic_dom > VIDLORE_CLIPSTUDIO_GRAPHIC_MAX  (default 0.036 — same bar as the
#           RC5 web-image gate this reuses)
#   BAND  : 0.010 < graphic_dom <= MAX additionally requires the photo-vs-art test to say "art"
#           (painterly fan art sits in this band; a normal film still does not)
#   GUARD : near-black frames (luma < 22) are never judged — their embeds are degenerate and
#           the unreadable gates own them.
_GFX_EXTRA_ANCHORS = (
    "a digital painting of fictional characters",
    "fan art illustration of a man and a woman",
    "a stylized digital painting portrait",
    "an anime or manga style illustration",
    "3D rendered broadcast television graphics with colorful icons",
)
_GFX_MATS: list = [None]        # [(G, P, PH, AR)] — lazily built text-anchor matrices


def _graphics_mats():
    """(graphic, realphoto, photo, art) anchor matrices, or None when CLIP is unavailable."""
    if _GFX_MATS[0] is not None:
        return _GFX_MATS[0]
    try:
        import numpy as np
        import vidlore.visual_relevance as _vrm
        if not _vrm.available():
            return None
        from .image_fallback import _PHOTO_PROMPTS, _ART_PROMPTS

        def _m(prompts):
            return np.stack([np.asarray(_vrm._txt_embed(t), dtype="float32") for t in prompts])
        _GFX_MATS[0] = (_m(tuple(_vrm._GRAPHIC_NEG) + _GFX_EXTRA_ANCHORS),
                        _m(_vrm._REALPHOTO_POS), _m(_PHOTO_PROMPTS), _m(_ART_PROMPTS))
        return _GFX_MATS[0]
    except Exception:
        return None


def graphics_flag_of(vec, luma_avg: float = -1.0) -> int:
    """Tiered designed-graphics verdict for one shot embedding:
      2 = HARD graphics (graphic_dom above the calibrated gate — always excluded)
      1 = BAND-art (weakly graphic AND the photo-vs-art test says art) — excluded only in a
          source that ALSO has hard evidence: a lone stylized live-action composition (observed:
          a high-angle drawbridge aerial) can land here, so band alone never gates a clean source
      0 = photographic · -1 = cannot judge."""
    if vec is None:
        return -1
    if 0.0 <= float(luma_avg) < 22.0:
        return 0                                       # near-black: the unreadable gates' business
    mats = _graphics_mats()
    if mats is None:
        return -1
    try:
        import os as _os
        G, P, PH, AR = mats
        d = float((G @ vec).max() - (P @ vec).max())
        try:
            gmax = float(_os.environ.get("VIDLORE_CLIPSTUDIO_GRAPHIC_MAX", "0.036") or 0.036)
        except (TypeError, ValueError):
            gmax = 0.036
        try:
            gsoft = float(_os.environ.get("VIDLORE_CLIPSTUDIO_GRAPHIC_SOFT", "0.010") or 0.010)
        except (TypeError, ValueError):
            gsoft = 0.010
        if d > gmax:
            return 2
        if d > gsoft:
            # photo-vs-art tiebreak on the SAME embedding (no extra image pass) — the identical
            # comparison image_fallback._photographic_ok runs, minus its file IO
            return 1 if float((PH @ vec).max()) < float((AR @ vec).max()) - 0.01 else 0
        return 0
    except Exception:
        return -1


def _sample_times(start: float, end: float) -> list:
    """Sample timestamps for one shot's flag pass. SHORT shots (<2s) get FIVE spread samples —
    they are exactly where a brief burned sub / bug flash slips between 3 samples (observed: a
    rare Turkish sub inside a 1.4s shot missed by 3 samples), and where the render pads the cut
    past the shot boundary, so their verdict must be the most reliable. LONG shots scale up to
    one sample per ~6s (cap 9): a 51s essay shot sampled at 3 points has ~17s blind stretches —
    observed: a commenter-avatar badge lived entirely between them and aired with empty ocr_text.
    Never returns a single mid-frame-only sample."""
    d = max(0.0, end - start)
    if d < 2.0:
        fr = (0.1, 0.3, 0.5, 0.7, 0.9)
    else:
        import math
        n = max(3, min(9, int(math.ceil(d / 6.0))))
        fr = tuple((i + 0.5) / n for i in range(n))
    return [start + f * d for f in fr]


def _band_ocr_hit(pil_frame) -> bool:
    """OCR the SUBTITLE BAND of one full-colour sample frame — the definitive catch for THIN
    Latin subs the edge-density heuristic under-detects at 320×180 (observed: a thin yellow
    'poison your three.' slipped both the density and short-line branches). ≥2 readable words
    of ≥3 letters in the band = burned text. Non-Latin scripts stay covered by the visual
    heuristic; OCR-unavailable environments simply skip this (fail-open)."""
    try:
        from . import ocr as _ocr
        if not _ocr.available():
            return False
        import re as _re
        import tempfile
        import os as _os
        w, h = pil_frame.size
        band = pil_frame.crop((int(w * 0.10), int(h * 0.74), int(w * 0.90), int(h * 0.99)))
        band = band.resize((band.width * 2, band.height * 2))
        fd, tmp = tempfile.mkstemp(suffix=".jpg")
        _os.close(fd)
        try:
            band.save(tmp, quality=88)
            txt = _ocr.read_text(tmp) or ""
        finally:
            try:
                _os.remove(tmp)
            except OSError:
                pass
        return len(_re.findall(r"[A-Za-zÀ-ÿĀ-žğışöüçÇĞİŞÖÜ]{3,}", txt)) >= 2
    except Exception:
        return False


def compute_shot_flags(path, shots: list, *, progress=None) -> int:
    """Decode 3-9 frames per shot (start..end) via PyAV and persist multi-frame flags on each
    Shot. Returns the number of shots flagged. Tolerant: a source that can't be decoded leaves
    the sentinel values (-1) in place and every gate falls back to keyframe heuristics.

    Fast path: ONE monotonic decode of the file assigns every (shot, sample) target — the
    per-sample seek path re-decoded each GOP 2-5x (sample density ~1/s vs 2-5s GOPs) and was
    the single largest index block (measured 29.5 min/render). Frame SELECTION is replicated
    exactly (first frame at/after the last keyframe ≤ t with time ≥ t-0.05; last-decoded-frame
    EOF fallback), so flags are bit-identical; any mid-walk decode error falls back to the
    original seek path for the whole source. VIDLORE_CLIPSTUDIO_FLAGS_FAST=0 restores the old
    path outright."""
    import os as _os
    if (_os.environ.get("VIDLORE_CLIPSTUDIO_FLAGS_FAST", "1").strip().lower()
            in ("0", "false", "no")):
        return _compute_shot_flags_seek(path, shots, progress=progress)
    try:
        return _compute_shot_flags_mono(path, shots, progress=progress)
    except Exception:
        # damaged/exotic media: recompute everything with the battle-tested seek walk
        # (assignments are overwritten wholesale, so a partial mono pass leaves no residue)
        return _compute_shot_flags_seek(path, shots, progress=progress)


def _flags_finish_shot(sh, frames, pil_mid) -> int:
    """Shared per-shot tail of both flag walks: flags math + gated band-OCR + assignment.
    Returns 1 when the shot got a real verdict. Byte-identical inputs in both walks."""
    flags = _flags_from_frames(frames)
    # band OCR on the mid sample — catches thin Latin subs the edge heuristic misses.
    # Only consulted when the visual pass reads clean AND the shot has dialogue (burned
    # subs track speech), keeping the OCR cost to the shots that can actually leak.
    if flags["subs_flag"] == 0 and pil_mid is not None \
            and (getattr(sh, "transcript", "") or "").strip() \
            and _band_ocr_hit(pil_mid):
        flags["subs_flag"] = 1
        flags["text_conf"] = max(flags["text_conf"], 9.9)
    sh.subs_flag = flags["subs_flag"]
    sh.text_conf = flags["text_conf"]
    sh.luma_avg = flags["luma_avg"]
    sh.luma_hi = flags["luma_hi"]
    sh.luma_min = flags.get("luma_min", -1.0)
    sh.luma_min_black_frac = flags.get("luma_min_black_frac", -1.0)
    sh.corner_masks = flags["corner_masks"]
    sh.static_frac = flags.get("static_frac", -1.0)
    sh.pair_diff_max = flags.get("pair_diff_max", -1.0)
    sh.pair_diff_mean = flags.get("pair_diff_mean", -1.0)
    return 1 if flags["subs_flag"] >= 0 else 0


def _compute_shot_flags_mono(path, shots: list, *, progress=None) -> int:
    """Single monotonic decode. Selection rule replicated from the seek walk EXACTLY:
    the seek path lands on the last keyframe ≤ t and takes the first frame with
    time ≥ t-0.05 from there — so a candidate frame seen BEFORE a later keyframe ≤ t
    must be discarded (the seek would never have decoded it). Decoder threading is safe
    here: h264/vp9/av1 decode is spec-exact, threads change scheduling, not pixels."""
    import numpy as np
    import av
    done = 0
    c = av.open(str(path))
    try:
        try:
            c.streams.video[0].thread_type = "AUTO"
        except Exception:
            pass
        # global ascending target list: (t, shot_idx, sample_idx)
        per_shot_ts = [_sample_times(sh.start, sh.end) for sh in shots]
        targets = []
        for k, ts in enumerate(per_shot_ts):
            for j, t in enumerate(ts):
                targets.append((float(t), k, j))
        targets.sort(key=lambda x: x[0])
        if not targets:
            return 0
        # per-shot result holders: {j: PIL-or-None}; converted at shot completion
        got_pil = [dict() for _ in shots]
        remaining = [len(ts) for ts in per_shot_ts]
        ti = 0
        cand = None            # first eligible candidate frame for targets[ti]
        last = None            # last decoded frame (EOF fallback)

        def _assign(fr):
            nonlocal ti, cand, done
            t, k, j = targets[ti]
            try:
                got_pil[k][j] = fr.to_image()
            except Exception:
                got_pil[k][j] = None            # per-sample tolerance, like the seek walk
            remaining[k] -= 1
            if remaining[k] == 0:
                done += _finish(k)
            ti += 1
            cand = None

        def _finish(k):
            sh = shots[k]
            ts = per_shot_ts[k]
            frames, pil_mid = [], None
            for j in range(len(ts)):
                pim = got_pil[k].get(j)
                if pim is None:
                    continue
                try:
                    if j == len(ts) // 2:
                        pil_mid = pim
                    im = pim.convert("L").resize((640, 360))
                    frames.append(np.asarray(im, dtype="float32"))
                except Exception:
                    continue
            got_pil[k].clear()
            n = _flags_finish_shot(sh, frames, pil_mid)
            if progress and (k % 50 == 0) and k:
                progress(f"index: multi-frame flags {k}/{len(shots)}")
            return n

        for fr in c.decode(video=0):
            last = fr
            ft = float(fr.time or 0.0)
            # a frame can satisfy several stacked targets (adjacent-shot boundaries)
            while ti < len(targets):
                t = targets[ti][0]
                if fr.key_frame and ft <= t:
                    cand = None                 # seek would restart from THIS keyframe
                if ft >= t - 0.05 and cand is None:
                    cand = fr
                if ft > t and cand is not None:
                    _assign(cand)
                    continue                    # re-check next target against this frame
                if cand is not None and ft >= t:
                    # candidate is this very frame at/after t — no later keyframe ≤ t can
                    # exist (times are monotonic), assign immediately
                    _assign(cand)
                    continue
                break
            if ti >= len(targets):
                break
        # EOF: unresolved targets take the pending candidate, else the last decoded frame
        while ti < len(targets):
            fr = cand if cand is not None else last
            if fr is None:
                # nothing decoded at all for this target: skip the sample (seek parity)
                t, k, j = targets[ti]
                got_pil[k][j] = None
                remaining[k] -= 1
                if remaining[k] == 0:
                    done += _finish(k)
                ti += 1
                cand = None
            else:
                _assign(fr)
    finally:
        try:
            c.close()
        except Exception:
            pass
    return done


def _compute_shot_flags_seek(path, shots: list, *, progress=None) -> int:
    """Original per-sample seek walk — kept verbatim as the fallback/reference path."""
    import numpy as np
    try:
        import av
    except Exception:
        return 0
    done = 0
    try:
        c = av.open(str(path))
    except Exception:
        return 0
    try:
        for k, sh in enumerate(shots):
            ts = _sample_times(sh.start, sh.end)
            frames = []
            pil_mid = None
            for j, t in enumerate(ts):
                try:
                    c.seek(int(t * 1e6))
                    got = None
                    for fr in c.decode(video=0):
                        got = fr
                        if float(fr.time or 0.0) >= t - 0.05:
                            break
                    if got is not None:
                        pim = got.to_image()
                        if j == len(ts) // 2:
                            pil_mid = pim
                        im = pim.convert("L").resize((640, 360))
                        frames.append(np.asarray(im, dtype="float32"))
                except Exception:
                    continue
            done += _flags_finish_shot(sh, frames, pil_mid)
            if progress and (k % 50 == 0) and k:
                progress(f"index: multi-frame flags {k}/{len(shots)}")
    finally:
        try:
            c.close()
        except Exception:
            pass
    return done


# ---------------------------------------------------------------------------
# Shot quality + perceptual hash + duplicate-scene detection
# ---------------------------------------------------------------------------

def _ahash(img_bgr) -> str:
    import cv2
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (8, 8), interpolation=cv2.INTER_AREA)
    mean = g.mean()
    val = 0
    for b in (g > mean).flatten():
        val = (val << 1) | int(b)
    return f"{val:016x}"


def _hamming(h1: str, h2: str) -> int:
    if not h1 or not h2:
        return 64
    try:
        return bin(int(h1, 16) ^ int(h2, 16)).count("1")
    except ValueError:
        return 64


def _quality(img_bgr, source_height: int = 0) -> float:
    """0..1 shot quality from sharpness (Laplacian var) + brightness + resolution."""
    import cv2
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(g, cv2.CV_64F).var()
    sharp = max(0.0, min(1.0, (lap - 25.0) / 155.0))
    m = float(g.mean()) / 255.0
    bright = 1.0 - min(1.0, abs(m - 0.5) / 0.5) * 0.6
    h = source_height or img_bgr.shape[0]
    res = 1.0 if h >= 720 else (0.8 if h >= 480 else (0.6 if h >= 360 else 0.4))
    return round(max(0.05, 0.5 * sharp + 0.2 * bright + 0.3 * res), 3)


def _dedup_shots(shots: list[Shot], hamming_thresh: int) -> None:
    """Mark near-identical keyframes (within a source) → sh.dup_of = representative index."""
    reps: list[Shot] = []
    for sh in shots:
        if not sh.phash:
            reps.append(sh)
            continue
        dup = next((k for k in reps if _hamming(sh.phash, k.phash) <= hamming_thresh), None)
        if dup is not None:
            sh.dup_of = dup.index
        else:
            reps.append(sh)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def index_source(proj: ClipProject, source: SourceVideo, cfg: ClipConfig,
                 *, references=None, faceid=None, roster=None,
                 force: bool = False, progress=None) -> list[Shot]:
    import numpy as np
    from PIL import Image

    def log(m):
        if progress:
            progress(m)

    shots_file = proj.shots_path(source.id)
    meta_file = proj.index_dir / f"{source.id}.index.meta.json"
    asr_hotwords = _project_asr_hotwords(proj, fallback=roster)
    asr_initial_prompt = _project_asr_initial_prompt(proj, fallback=roster)
    asr_legacy_initial_prompt = _project_asr_legacy_initial_prompt(
        proj, fallback=roster)
    asr_fingerprint = _asr_prompt_fingerprint(
        cfg, asr_hotwords, asr_initial_prompt, asr_legacy_initial_prompt)
    authored_retrieval_prompt = _project_authored_retrieval_prompt(proj)
    # capabilities this call wants — a cache built WITHOUT them (e.g. by the manual pipeline,
    # roster-less) must not satisfy an auto-mode call that needs Face-ID/OCR signals. "Wanted"
    # includes availability, else a missing OCR lib would force a futile re-index every run.
    want_caps = {"faceid": bool(faceid and references), "ocr": False}
    if cfg.detect_ocr:
        from . import ocr as _ocr_probe
        want_caps["ocr"] = _ocr_probe.available()
    # roster gives ocr_names (name-card corroboration) — a roster-less OCR cache must not
    # satisfy a roster-bearing auto run; only meaningful when OCR itself can run
    want_caps["roster"] = bool(roster) and want_caps["ocr"]
    # word-level ASR timings (INDEX_SCHEMA 2). A pre-schema cache has shots but no <sid>.words.json,
    # so quote location would silently fall back to per-shot substring search — the exact failure
    # that lost "…is no true king". Treat it as a missing capability so those sources re-index.
    want_caps["words"] = True
    if shots_file.exists() and not force:
        try:
            have_caps = json.loads(meta_file.read_text(encoding="utf-8"))
        except Exception:
            have_caps = {}
        if not isinstance(have_caps, dict):      # 'null'/list/string parses fine but isn't caps
            have_caps = {}
        try:
            cached_schema = int(have_caps.get("schema", 1) or 1)
        except (TypeError, ValueError):
            cached_schema = 1
        legacy_index_schema = cached_schema < INDEX_SCHEMA
        try:
            legacy_artifact_binding = int(
                have_caps.get("artifact_binding_schema", 0) or 0
            ) != INDEX_ARTIFACT_BINDING_SCHEMA
        except (TypeError, ValueError):
            legacy_artifact_binding = True
        if legacy_index_schema:
            have_caps["words"] = False           # schema bump invalidates the word cache
        words_cache_valid, old_words = _load_words_result(proj, source.id)
        if want_caps["words"] and not words_cache_valid:
            have_caps["words"] = False           # meta alone cannot certify missing/corrupt words
        try:
            cached_shots = load_shots(proj, source.id)
        except Exception:
            cached_shots = []
        provenance_ok, provenance_reason, _provenance = \
            _index_artifact_provenance_result(proj, source, have_caps)
        if not provenance_ok:
            # A schema-2 cache receives one clean ASR refresh, which republishes both dependent
            # JSON artifacts and binds them to the current source. Once artifact-binding schema 1
            # claims that tuple, source/shots drift is a visual-index fault and forces re-indexing.
            words_only_repair = bool(
                legacy_index_schema or legacy_artifact_binding
                or provenance_reason.startswith("words_content_sha256_"))
            if words_only_repair:
                have_caps["words"] = False
            else:
                want_caps["index_artifact_binding"] = True
                have_caps["index_artifact_binding"] = False
        retrieval_required = bool(
            authored_retrieval_prompt
            and quote_retrieval_source_eligible(source, cached_shots))
        retrieval_valid = True
        retrieval_preserved = False
        old_fingerprint = str(have_caps.get("asr_prompt_fingerprint", "") or "")
        if retrieval_required:
            retrieval_valid, _retrieval_streams, _retrieval_reason, retrieval_complete = \
                _load_quote_retrieval_streams_result(
                    proj, source, cfg, fallback=roster, require_complete=True)
            retrieval_valid = bool(retrieval_valid and retrieval_complete)
            # Rev7's general stream was authored-prompted. Salvage those already-paid bytes as a
            # retrieval-only sidecar before rev8 replaces `.words.json` with clean general ASR.
            if (not retrieval_valid and words_cache_valid
                    and have_caps.get("words") is True
                    and not have_caps.get("asr_refresh_in_progress")
                    and old_fingerprint == _legacy_rev7_general_fingerprint(
                        proj, cfg, fallback=roster)):
                try:
                    retrieval_model = _whisper(cfg)
                    retrieval_chunks, _chunk_reason = _quote_retrieval_prompt_chunks(
                        proj, cfg, fallback=roster, model=retrieval_model)
                    # Rev7's supported-decoder path used the same first bounded quote prompt plus
                    # proper-name hotwords. On legacy decoders the combined fallback prompt was
                    # different, so those bytes cannot be relabelled as a v2 chunk.
                    if retrieval_chunks and _model_supports_hotwords(retrieval_model):
                        first_prompt = retrieval_chunks[0]
                        first_covered = [entry.strip() for entry in first_prompt.split(", ")
                                         if entry.strip()]
                        retrieval_preserved = _save_quote_retrieval_streams(
                            proj, source, cfg, [{
                                "prompt": first_prompt,
                                "covered_authored_quotes": first_covered,
                                "words": old_words,
                            }], fallback=roster)
                        if retrieval_preserved:
                            retrieval_valid, _streams, _reason, _complete = \
                                _load_quote_retrieval_streams_result(
                                    proj, source, cfg, fallback=roster,
                                    require_complete=True)
                            log(f"index: {source.id} preserved rev7 prompted words as "
                                f"retrieval-only chunk coverage={len(first_covered)}/"
                                f"{len(_project_authored_quotes(proj, (proj.meta or {}).get('analysis') or {}))}")
                except Exception:
                    retrieval_preserved = False
            want_caps["quote_retrieval"] = True
            have_caps["quote_retrieval"] = retrieval_valid
        missing = [k for k, v in want_caps.items() if v and not have_caps.get(k)]
        if str(have_caps.get("asr_prompt_fingerprint", "") or "") != asr_fingerprint:
            missing.append("asr_prompt")
        missing = list(dict.fromkeys(missing))
        if missing:
            # A changed model/vocabulary/options dependency invalidates only ASR-derived files.
            # Refresh those transactionally and preserve every expensive visual artifact.
            if (set(missing).issubset({"words", "asr_prompt", "quote_retrieval"})
                    and source.status == SOURCE_OK and source.local_path):
                refreshed = cached_shots
                if "words" in missing or "asr_prompt" in missing:
                    refreshed = refresh_source_words(
                        proj, source, cfg, progress=progress, hotwords=asr_hotwords,
                        initial_prompt=asr_initial_prompt,
                        legacy_initial_prompt=asr_legacy_initial_prompt,
                        allow_empty=bool(
                            retrieval_preserved and old_fingerprint
                            == _legacy_rev7_general_fingerprint(
                                proj, cfg, fallback=roster)),
                        trusted_replaced_fingerprint=old_fingerprint)
                    if not refreshed:
                        state = ("corrupt/missing" if not words_cache_valid else
                                 ("non-empty" if old_words else "silent"))
                        # ONE SOURCE THAT CANNOT BE UPGRADED IS NOT A DEAD RENDER.
                        #
                        # Measured: a 101-beat validation ran 21 minutes and died here because a
                        # single 231-second lore-essay source ("Everything About Valyrian Steel")
                        # would not re-transcribe under the new authored prompt. Its whole existing
                        # transcript is FOUR words at t=223-230 — an end-credits cast read-out — and
                        # that valid cache was preserved, exactly as intended. Nothing was lost and
                        # nothing was corrupted; the render was simply killed.
                        #
                        # A source whose ASR could not be brought to the CURRENT prompt must not be
                        # trusted to prove a quote under that prompt, so it is barred from quote
                        # retrieval and from whole-pool absence claims. That is strictly less trust
                        # than before, not more — it cannot admit anything. It keeps its preserved
                        # words for every other purpose, and the run continues.
                        #
                        # A corrupt or missing cache is different in kind: there is no trustworthy
                        # prior state to fall back on, so that still raises.
                        if not (words_cache_valid and old_words):
                            raise RuntimeError(
                                f"ASR cache upgrade failed for {source.id}; {state} cache preserved")
                        if progress:
                            progress(f"index: {source.id} ASR could not be upgraded to the current "
                                     f"prompt ({state} cache preserved) — barred from quote "
                                     f"retrieval for this render, indexing continues")
                        _remember_asr_refusal(proj, source.id)
                        refreshed = cached_shots
                retrieval_needed_after_refresh = bool(
                    authored_retrieval_prompt
                    and quote_retrieval_source_eligible(source, refreshed))
                if retrieval_needed_after_refresh:
                    (retrieval_valid, _retrieval_streams, _retrieval_reason,
                     retrieval_complete) = _load_quote_retrieval_streams_result(
                        proj, source, cfg, fallback=roster, require_complete=True)
                    retrieval_valid = bool(retrieval_valid and retrieval_complete)
                if ("quote_retrieval" in missing or retrieval_needed_after_refresh) \
                        and not retrieval_valid:
                    retrieval_valid = refresh_source_quote_retrieval(
                        proj, source, cfg, progress=progress, fallback=roster)
                    if not retrieval_valid:
                        raise RuntimeError(
                            f"authored quote retrieval refresh failed for {source.id}; "
                            "whole-pool quote absence was not certified")
                log(f"index: {source.id} ASR caches upgraded "
                    f"({len(refreshed)} shots; visual index preserved)")
                return refreshed
            log(f"index: {source.id} re-indexing (cache lacks {'+'.join(missing)})")
        else:
            try:
                data = json.loads(shots_file.read_text(encoding="utf-8"))
                log(f"index: {source.id} cached ({len(data)} shots)")
                return [Shot.from_dict(d) for d in data]
            except Exception:
                log(f"index: ⚠ {source.id} cache unreadable — re-indexing")

    if source.status != SOURCE_OK or not source.local_path:
        log(f"index: {source.id} skipped (status={source.status})")
        return []

    path = Path(source.local_path)
    source_fingerprint_at_start = _source_content_fingerprint(path)
    if source_fingerprint_at_start in ("", "missing", "unreadable"):
        raise RuntimeError(f"source fingerprint unavailable before indexing {source.id}")
    proj.index_dir.mkdir(parents=True, exist_ok=True)
    kf_dir = proj.keyframes_dir(source.id)
    kf_dir.mkdir(parents=True, exist_ok=True)

    # 1) shots
    bounds = detect_shots(path, cfg, total_dur=source.duration)
    if not bounds:
        # zero shots = unreadable/zero-duration media; do NOT cache, so a later run retries.
        # Reaching here WITH an existing cache means a forced/capability re-index — drop the
        # stale files too, or match's load_shots silently serves the old index of media that
        # can no longer be read.
        for _stale in (shots_file, meta_file):
            try:
                _stale.unlink(missing_ok=True)
            except OSError:
                pass
        log(f"index: ⚠ {source.id} produced 0 shots (unreadable media?) — skipped, not cached")
        return []
    log(f"index: {source.id} {len(bounds)} shots detected")

    # 2) transcript
    asr_succeeded, words = _transcribe_words_result(
        path, cfg, duration=source.duration, hotwords=asr_hotwords,
        initial_prompt=asr_initial_prompt,
        legacy_initial_prompt=asr_legacy_initial_prompt)
    log(f"index: {source.id} transcript words={len(words)}")
    if not asr_succeeded:
        log(f"index: ⚠ {source.id} ASR failed; visual index will remain reusable but word "
            "evidence is not certified")
        prior_valid, prior_words = _load_words_result(proj, source.id)
        if prior_valid:
            # Retain earlier evidence bytes/meaning during a forced visual rebuild, but mark it
            # uncertified below. A later targeted refresh can repair it without rebuilding images.
            words = prior_words

    # 3) keyframes + CLIP embeds + faces + Face-ID + OCR + quality + phash
    import cv2
    use_clip = clip_available()
    vr = _vr() if use_clip else None
    do_faceid = bool(faceid and references)
    # OCR runs whenever enabled — the junk/watermark gates in match.py need ocr_text on EVERY
    # pipeline (the roster only adds name-card corroboration on top)
    do_ocr = bool(cfg.detect_ocr)
    _ocr = None
    if do_ocr:
        from . import ocr as _ocr
        do_ocr = _ocr.available()
    _fid = None
    if do_faceid:
        from . import faceid as _fid
    shots: list[Shot] = []
    embeds = []

    # SPEED: pre-extract every keyframe with the IDENTICAL ffmpeg argv in a small thread
    # pool (subprocesses release the GIL) — the serial loop paid one cold spawn per shot
    # (measured ~5.3 min/render). Same command → same jpg bytes; the pool result is
    # authoritative (single attempt, same 60s timeout), so the success/failure set keeps
    # single-attempt semantics. Never trusts a pre-existing file: only THIS run's recorded
    # result counts (a stale keyframe from an older index must not be resurrected).
    # VIDLORE_CLIPSTUDIO_KF_PREEXTRACT=0 restores the inline spawns.
    _prex: dict = {}
    if len(bounds) >= 4 and os.environ.get(
            "VIDLORE_CLIPSTUDIO_KF_PREEXTRACT", "1").strip().lower() not in ("0", "false", "no"):
        try:
            import concurrent.futures as _cf
            with _cf.ThreadPoolExecutor(max_workers=6) as _ex:
                _futs = {_ex.submit(extract_keyframe, path, (s + e) / 2.0,
                                    kf_dir / f"shot_{i:04d}.jpg"): i
                         for i, (s, e) in enumerate(bounds)}
                for _f in _cf.as_completed(_futs):
                    try:
                        _prex[_futs[_f]] = bool(_f.result())
                    except Exception:
                        _prex[_futs[_f]] = False
        except Exception:
            _prex = {}

    # SPEED: OCR runs in the persistent worker pool (child processes own their engines,
    # bit-identical output — see ocr.py) so it overlaps the main-thread CLIP/Face-ID work;
    # results are consumed strictly per shot index below, serial fallback on any failure.
    _ocr_futs: dict = {}
    if do_ocr and _prex:
        for _i, _ok in _prex.items():
            if _ok:
                _fut = _ocr.read_text_async(kf_dir / f"shot_{_i:04d}.jpg")
                if _fut is not None:
                    _ocr_futs[_i] = _fut

    for i, (s, e) in enumerate(bounds):
        sh = Shot(source_id=source.id, index=i, start=round(s, 3), end=round(e, 3))
        kf = kf_dir / f"shot_{i:04d}.jpg"
        if (_prex[i] if i in _prex else extract_keyframe(path, (s + e) / 2.0, kf)):
            sh.keyframe_path = str(kf)
            try:
                im = Image.open(kf)
                if use_clip:
                    v = vr._img_embed(im)
                    sh.embed_row = len(embeds)
                    embeds.append(np.asarray(v, dtype="float32"))
                    try:
                        _kl = float(np.asarray(im.convert("L"), dtype="float32").mean())
                        sh.graphics_flag = graphics_flag_of(
                            np.asarray(v, dtype="float32"), _kl)
                    except Exception:
                        sh.graphics_flag = -1
                if cfg.detect_faces and use_clip:
                    frac = float(vr._face_frac(im))
                    sh.scores["face_frac"] = round(frac, 4)
                    sh.faces = 1 if frac > 0.02 else 0
            except Exception:
                pass
            img_bgr = cv2.imread(str(kf))
            if img_bgr is not None:
                sh.phash = _ahash(img_bgr)
                sh.quality = _quality(img_bgr, source.height)
                if do_faceid:
                    try:
                        ids = _fid.identify_faces(kf, faceid, references)
                        sh.identities = ids[:4]
                        sh.face_ids = [d["name"] for d in ids if d.get("name")]
                        if ids:
                            sh.faces = max(sh.faces, len(ids))
                    except Exception:
                        pass
                if do_ocr:
                    try:
                        txt = None
                        if i in _ocr_futs:
                            try:
                                txt = _ocr_futs[i].result(timeout=120)
                            except Exception:
                                txt = None       # worker died → serial singleton below
                        if txt is None:
                            txt = _ocr.read_text(kf)
                        sh.ocr_text = txt[:200]
                        sh.ocr_names = _ocr.names_in_text(txt, roster)
                    except Exception:
                        pass
        shots.append(sh)
        if progress and (i % 25 == 0):
            log(f"index: {source.id} keyframed {i+1}/{len(bounds)}")

    _assign_transcript(shots, words)
    _dedup_shots(shots, cfg.dup_hamming)

    # 3b) MULTI-FRAME flags (start/mid/end sample per shot) — catches intermittent burned subs /
    # corner bugs / mid-shot darkness the single keyframe misses. Persisted on the Shot so every
    # downstream gate reuses them without re-decoding. env VIDLORE_CLIPSTUDIO_MULTIFRAME_FLAGS=0
    # disables (gates then fall back to keyframe heuristics).
    if os.environ.get("VIDLORE_CLIPSTUDIO_MULTIFRAME_FLAGS", "1").strip() \
            not in ("0", "false", "no"):
        try:
            _nf = compute_shot_flags(path, shots, progress=(log if progress else None))
            log(f"index: {source.id} multi-frame flags on {_nf}/{len(shots)} shots")
        except Exception as _e:
            log(f"index: {source.id} multi-frame flags skipped ({type(_e).__name__})")

    # 4) persist (atomic tmp+replace — an interrupted write must not brick later runs' resume).
    # embeds first, shots last: shots.json presence gates the cache, and match.py bounds-checks
    # embed_row, so a kill between the two degrades gracefully instead of pairing a valid cache
    # with a truncated matrix.
    if embeds:
        _etmp = proj.embeds_path(source.id).with_suffix(".tmp.npy")
        np.save(_etmp, np.vstack(embeds))
        _etmp.replace(proj.embeds_path(source.id))
        # EMBEDDING MANIFEST — certifies what each persisted row IS: index schema, the exact
        # embedding model, the dimension, and per-row (shot index, keyframe name, keyframe
        # content hash). A consumer may trust a stored vector ONLY when every one of these
        # matches its current world; anything else (model swap, re-extracted keyframe,
        # reordered rows, truncated matrix) must fall back to a live embed.
        try:
            write_embed_manifest(proj, source.id, shots, len(embeds),
                                 int(embeds[0].shape[-1]) if embeds else 0)
        except Exception as _me:
            log(f"index: {source.id} embed manifest skipped ({type(_me).__name__})")
    # words BEFORE shots: shots.json presence gates the cache, so writing words first means a kill
    # between the two re-indexes (safe) rather than serving shots with no word stream (silent
    # degradation back to per-shot quote search).
    save_words(proj, source.id, words)
    _tmp = shots_file.with_suffix(".json.tmp")
    _tmp.write_text(json.dumps([s.to_dict() for s in shots], indent=1), encoding="utf-8")
    _tmp.replace(shots_file)
    _mtmp = meta_file.with_suffix(".json.tmp")
    # "words" records that the word-level PASS RAN, never that it found anything — a genuinely
    # silent source yields [] and must still be cacheable, or it re-indexes on every run forever.
    index_meta = {"faceid": do_faceid, "ocr": do_ocr,
                  "roster": bool(roster), "words": bool(asr_succeeded),
                  "schema": INDEX_SCHEMA}
    if asr_succeeded:
        index_meta["asr_prompt_fingerprint"] = asr_fingerprint
    index_meta.update(_index_artifact_bindings(proj, source))
    if (not index_meta.get("source_content_fingerprint")
            or index_meta.get("source_content_fingerprint") != source_fingerprint_at_start
            or not index_meta.get("words_content_sha256")
            or not index_meta.get("shots_content_sha256")):
        index_meta["words"] = False
        index_meta["index_artifact_binding_error"] = "fingerprint_unavailable_or_source_changed"
    _mtmp.write_text(json.dumps(index_meta), encoding="utf-8")
    _mtmp.replace(meta_file)
    if index_meta.get("words") is not True:
        raise RuntimeError(
            f"index artifact provenance could not be certified for {source.id}")
    if (authored_retrieval_prompt
            and quote_retrieval_source_eligible(source, shots)):
        retrieval_valid, _retrieval_streams, _retrieval_reason, retrieval_complete = \
            _load_quote_retrieval_streams_result(
                proj, source, cfg, fallback=roster, require_complete=True)
        retrieval_valid = bool(retrieval_valid and retrieval_complete)
        if not retrieval_valid:
            retrieval_valid = refresh_source_quote_retrieval(
                proj, source, cfg, progress=progress, fallback=roster)
        if not retrieval_valid:
            log(f"index: ⚠ {source.id} quote retrieval incomplete; match checkpoint will block")
    log(f"index: {source.id} done — {len(shots)} shots, {len(embeds)} embeds, clip={use_clip}")
    return shots


def refresh_source_words(proj: ClipProject, source: SourceVideo, cfg: ClipConfig,
                         *, progress=None, hotwords: Optional[str] = None,
                         initial_prompt: Optional[str] = None,
                         legacy_initial_prompt: Optional[str] = None,
                         allow_empty: bool = False,
                         trusted_replaced_fingerprint: Optional[str] = None) -> list[Shot]:
    """Refresh one source's ASR words and shot transcripts without rebuilding visual indexes.

    This is the safe repair path for an otherwise-valid cached source whose transcript is known to
    be incomplete: keyframes, embeddings, OCR, face IDs, and shot boundaries remain byte-for-byte
    untouched.  Both dependent JSON files are atomically replaced only after ASR succeeds.
    """
    if source.status != SOURCE_OK or not source.local_path:
        return []
    source_fingerprint_at_start = _source_content_fingerprint(source.local_path)
    if source_fingerprint_at_start in ("", "missing", "unreadable"):
        return []
    shots_file = proj.shots_path(source.id)
    if not shots_file.exists():
        return []
    try:
        shots = load_shots(proj, source.id)
    except Exception:
        return []
    if hotwords is None:
        asr_hotwords = _project_asr_hotwords(proj)
    else:
        asr_hotwords = str(hotwords or "")
    if initial_prompt is None:
        asr_initial_prompt = _project_asr_initial_prompt(proj)
    else:
        asr_initial_prompt = str(initial_prompt or "")
    if legacy_initial_prompt is None:
        asr_legacy_initial_prompt = _project_asr_legacy_initial_prompt(proj)
    else:
        asr_legacy_initial_prompt = str(legacy_initial_prompt or "")
    old_valid, old_words = _load_words_result(proj, source.id)
    asr_succeeded, words = _transcribe_words_result(
        Path(source.local_path), cfg, duration=source.duration, hotwords=asr_hotwords,
        initial_prompt=asr_initial_prompt,
        legacy_initial_prompt=asr_legacy_initial_prompt)
    if not asr_succeeded:
        if progress:
            progress(f"index: ⚠ {source.id} word refresh failed technically; cache preserved")
        return []
    known_prompted_migration = False
    if allow_empty:
        try:
            old_meta = json.loads((Path(proj.index_dir)
                                   / f"{source.id}.index.meta.json").read_text(
                                       encoding="utf-8"))
            retrieval_ok, preserved_streams, _preserved_reason, _complete = \
                _load_quote_retrieval_streams_result(
                    proj, source, cfg, require_complete=False)
            preserved_words = (preserved_streams[0].get("words") or []) \
                if preserved_streams else []
            known_prompted_migration = bool(
                isinstance(old_meta, dict)
                and str(old_meta.get("asr_prompt_fingerprint", "") or "")
                == str(trusted_replaced_fingerprint or "")
                and len(str(trusted_replaced_fingerprint or "")) == 64
                and retrieval_ok and preserved_words == old_words)
        except Exception:
            known_prompted_migration = False
    if not words and (not old_valid or bool(old_words)) and not known_prompted_migration:
        # A successful no-word decode is trustworthy only when it confirms an already-valid silent
        # cache. It must never erase prior speech or turn malformed JSON into certified silence.
        if progress:
            state = "malformed/missing" if not old_valid else "non-empty"
            progress(f"index: ⚠ {source.id} silent refresh conflicts with {state} cache; "
                     "cache preserved")
        return []
    # Empty replacement is authorized only for the exact rev7→rev8 split above, after the prompted
    # bytes were preserved as retrieval-only candidates. Ordinary callers still cannot erase prior
    # speech by passing a permissive flag. Clear old per-shot text before assigning the clean stream.
    for shot in shots:
        shot.transcript = ""
    _assign_transcript(shots, words)
    words_file = Path(proj.index_dir) / f"{source.id}.words.json"
    meta_file = Path(proj.index_dir) / f"{source.id}.index.meta.json"
    words_tmp = words_file.with_name(words_file.name + ".refresh.tmp")
    shots_tmp = shots_file.with_name(shots_file.name + ".refresh.tmp")
    rollback_tmp = words_file.with_name(words_file.name + ".refresh.rollback.tmp")
    meta_tmp = meta_file.with_name(meta_file.name + ".refresh.tmp")
    meta_rollback_tmp = meta_file.with_name(meta_file.name + ".refresh.rollback.tmp")
    old_words_bytes = words_file.read_bytes() if words_file.exists() else None
    old_meta_bytes = meta_file.read_bytes() if meta_file.exists() else None
    try:
        meta = json.loads(old_meta_bytes.decode("utf-8")) if old_meta_bytes is not None else {}
    except Exception:
        meta = {}
    if not isinstance(meta, dict):
        meta = {}
    try:
        # Stage both complete payloads before replacing either member of the dependent pair.
        words_tmp.write_text(json.dumps([
            [round(float(a), 3), round(float(b), 3), str(c)] for a, b, c in words
        ]), encoding="utf-8")
        shots_tmp.write_text(
            json.dumps([shot.to_dict() for shot in shots], indent=1), encoding="utf-8")
        # Publish an invalid transaction marker BEFORE either dependent file can change. If the
        # process dies in the two-replace window, Resume sees this marker and must refresh again;
        # it can never trust new words paired with stale per-shot transcripts.
        in_progress_meta = dict(meta)
        in_progress_meta.update({
            "words": False,
            "schema": INDEX_SCHEMA,
            "asr_refresh_in_progress": True,
        })
        in_progress_meta.pop("asr_prompt_fingerprint", None)
        for field in (
                "artifact_binding_schema", "source_content_fingerprint",
                "words_content_sha256", "shots_content_sha256"):
            in_progress_meta.pop(field, None)
        meta_tmp.write_text(json.dumps(in_progress_meta), encoding="utf-8")
        meta_tmp.replace(meta_file)
        try:
            words_tmp.replace(words_file)
            shots_tmp.replace(shots_file)
        except Exception:
            # An ordinary replace failure is recoverable in-process. Restore both the old words and
            # old certification; a BaseException/process death deliberately leaves the invalid
            # marker behind so a later Resume repairs the ambiguous pair.
            if old_words_bytes is None:
                words_file.unlink(missing_ok=True)
            else:
                rollback_tmp.write_bytes(old_words_bytes)
                rollback_tmp.replace(words_file)
            if old_meta_bytes is None:
                meta_file.unlink(missing_ok=True)
            else:
                meta_rollback_tmp.write_bytes(old_meta_bytes)
                meta_rollback_tmp.replace(meta_file)
            raise
    finally:
        words_tmp.unlink(missing_ok=True)
        shots_tmp.unlink(missing_ok=True)
        rollback_tmp.unlink(missing_ok=True)
        meta_tmp.unlink(missing_ok=True)
        meta_rollback_tmp.unlink(missing_ok=True)
    # Same-process matching memoizes absent and present quote spans.  Invalidate only this source,
    # and only after the dependent cache pair committed successfully.
    try:
        from .match import _QSPAN_CACHE
        for key in [key for key in _QSPAN_CACHE if key and key[0] == source.id]:
            _QSPAN_CACHE.pop(key, None)
    except Exception:
        pass
    # Bind the refreshed stream to its exact model/options/vocabulary. A metadata-write failure is
    # safe: the next resume sees a mismatch and refreshes again rather than trusting unknown words.
    meta.update({
        "words": True,
        "asr_prompt_fingerprint": _asr_prompt_fingerprint(
            cfg, asr_hotwords, asr_initial_prompt, asr_legacy_initial_prompt),
        "schema": INDEX_SCHEMA,
    })
    meta.update(_index_artifact_bindings(proj, source))
    meta.pop("asr_refresh_in_progress", None)
    meta.pop("index_artifact_binding_error", None)
    if (not meta.get("source_content_fingerprint")
            or meta.get("source_content_fingerprint") != source_fingerprint_at_start
            or not meta.get("words_content_sha256")
            or not meta.get("shots_content_sha256")):
        if progress:
            progress(f"index: ⚠ {source.id} refreshed artifacts could not be provenance-bound")
        return []
    try:
        meta_tmp.write_text(json.dumps(meta), encoding="utf-8")
        meta_tmp.replace(meta_file)
    finally:
        meta_tmp.unlink(missing_ok=True)
    if progress:
        progress(f"index: {source.id} refreshed {len(words)} words across {len(shots)} shots")
    return shots


def refresh_source_quote_retrieval(
        proj: ClipProject, source: SourceVideo, cfg: ClipConfig, *, progress=None,
        fallback=None) -> bool:
    """Build only the separate authored-prompt retrieval stream; never touch general words."""
    if source.status != SOURCE_OK or not source.local_path:
        return False
    expected = _project_authored_quotes(
        proj, (getattr(proj, "meta", {}) or {}).get("analysis") or {})
    if not expected:
        return True
    hotwords = _project_asr_hotwords(proj, fallback=fallback)
    partial_ok, streams, _partial_reason, complete = \
        _load_quote_retrieval_streams_result(
            proj, source, cfg, fallback=fallback, require_complete=False)
    if not partial_ok:
        streams, complete = [], False
    if complete:
        return True
    covered_count = sum(len(stream.get("covered_authored_quotes") or [])
                        for stream in streams)
    remaining = expected[covered_count:]
    chunks, chunk_reason = _quote_retrieval_prompt_chunks(
        proj, cfg, entries=remaining, fallback=fallback)
    if not chunks:
        if progress:
            progress(f"index: ⚠ {source.id} authored quote retrieval prompt chunking failed "
                     f"({chunk_reason or 'no_chunks'})")
        return False
    new_streams = list(streams)
    decoded_words = 0
    for chunk_index, prompt in enumerate(chunks, start=len(streams)):
        covered = [entry.strip() for entry in prompt.split(", ") if entry.strip()]
        succeeded, words = _transcribe_words_result(
            Path(source.local_path), cfg, duration=source.duration, hotwords=hotwords,
            initial_prompt=prompt, legacy_initial_prompt=prompt)
        if not succeeded:
            if progress:
                progress(f"index: ⚠ {source.id} authored quote retrieval ASR failed "
                         f"on chunk {chunk_index + 1}/{len(streams) + len(chunks)}")
            return False
        decoded_words += len(words)
        new_streams.append({
            "prompt": prompt,
            "covered_authored_quotes": covered,
            "words": words,
        })
    if not _save_quote_retrieval_streams(
            proj, source, cfg, new_streams, fallback=fallback):
        if progress:
            progress(f"index: ⚠ {source.id} authored quote retrieval cache write failed")
        return False
    valid, _saved_streams, reason, complete = _load_quote_retrieval_streams_result(
        proj, source, cfg, fallback=fallback, require_complete=True)
    if not valid or not complete:
        if progress:
            progress(f"index: ⚠ {source.id} authored quote retrieval coverage invalid "
                     f"({reason})")
        return False
    if progress:
        progress(f"index: {source.id} authored retrieval chunks={len(new_streams)} "
                 f"new_candidate_words={decoded_words} coverage={len(expected)}/{len(expected)} "
                 "(not evidence until unprompted confirmation)")
    return True


def index_all(proj: ClipProject, cfg: ClipConfig, *, references=None, faceid=None, roster=None,
              force: bool = False, progress=None) -> dict[str, list[Shot]]:
    _seed_asr_refusals(proj)          # bars recorded by an earlier process outlive that process
    out: dict[str, list[Shot]] = {}
    for src in proj.sources:
        if src.status != SOURCE_OK:
            continue
        out[src.id] = index_source(proj, src, cfg, references=references, faceid=faceid,
                                   roster=roster, force=force, progress=progress)
    audit = asr_pool_cache_audit(proj, cfg)
    if audit["invalid"]:
        sample = [row["source_id"] for row in audit["invalid"][:8]]
        raise RuntimeError(
            f"ASR evidence incomplete for {len(audit['invalid'])}/{audit['source_count']} "
            f"usable source(s): {sample}; index/match checkpoints were not authorized")
    return out


def load_shots(proj: ClipProject, source_id: str) -> list[Shot]:
    from . import perf_metrics as _pm
    _pm.incr("index.load_shots")
    f = proj.shots_path(source_id)
    if not f.exists():
        return []
    return [Shot.from_dict(d) for d in json.loads(f.read_text(encoding="utf-8"))]


def load_embeds(proj: ClipProject, source_id: str):
    import numpy as np
    from . import perf_metrics as _pm
    _pm.incr("index.load_embeds")
    f = proj.embeds_path(source_id)
    return np.load(f) if f.exists() else None


def _manifest_path(proj: ClipProject, source_id: str):
    p = proj.embeds_path(source_id)
    return p.with_name(p.name.replace(".npy", "") + ".manifest.json")


def _kf_md5(path: str) -> str:
    import hashlib
    try:
        return hashlib.md5(Path(path).read_bytes()).hexdigest()
    except Exception:
        return ""


def write_embed_manifest(proj: ClipProject, source_id: str, shots, n_rows: int,
                         dim: int) -> None:
    """Atomic manifest write beside the embeds matrix (see index_source)."""
    from vidlore import visual_relevance as _vr_m
    rows = {}
    for sh in shots:
        r = getattr(sh, "embed_row", -1)
        r = -1 if r is None else int(r)
        if 0 <= r < n_rows and getattr(sh, "keyframe_path", ""):
            rows[str(r)] = {"shot": int(sh.index),
                            "kf": Path(sh.keyframe_path).name,
                            "kf_md5": _kf_md5(sh.keyframe_path)}
    mp = _manifest_path(proj, source_id)
    tmp = mp.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"schema": INDEX_SCHEMA, "model": _vr_m.model_identity(),
                               "dim": int(dim), "rows": int(n_rows), "row_map": rows}),
                   encoding="utf-8")
    tmp.replace(mp)


def load_embeds_verified(proj: ClipProject, source_id: str):
    """(matrix, row_map) — served ONLY when the manifest certifies the matrix for the
    CURRENT world: index schema, active embedding-model identity, dimension, and row count
    all match. Missing/corrupt/mismatched manifest (incl. every legacy pre-manifest index)
    -> (None, None): callers use the live embedding path, never an uncertified vector.
    Per-row keyframe identity is validated by the CONSUMER against row_map."""
    import numpy as np
    from . import perf_metrics as _pm
    from vidlore import visual_relevance as _vr_m
    mp = _manifest_path(proj, source_id)
    f = proj.embeds_path(source_id)
    if not (mp.exists() and f.exists()):
        _pm.incr("index.embeds.unverified_legacy")
        return None, None
    try:
        man = json.loads(mp.read_text(encoding="utf-8"))
        mat = np.load(f)
        ident = _vr_m.model_identity()
        if (int(man.get("schema", -1)) == INDEX_SCHEMA
                and ident and str(man.get("model", "")) == ident
                and int(man.get("dim", -1)) == int(mat.shape[1])
                and int(man.get("rows", -1)) == int(mat.shape[0])
                and isinstance(man.get("row_map"), dict)):
            _pm.incr("index.embeds.verified")
            return mat, man["row_map"]
    except Exception:
        pass
    _pm.incr("index.embeds.manifest_rejected")
    return None, None
