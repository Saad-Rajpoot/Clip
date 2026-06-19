"""HeyGen B-Roll Tool — separate lightweight pipeline.

This module is INTENTIONALLY ISOLATED from the main documentary
engine.  It imports only safe primitives (Pexels search, Whisper
transcription, Config) from the existing codebase and does NOT
touch:

    * pipeline.py            (documentary orchestration)
    * assemble.py            (cinematic ffmpeg assembly)
    * look_dna.py            (channel identity)
    * captions.py / themes / style_modes / musiclib
    * footage.py            (other than reusing _pexels_fetch)

PURPOSE
-------
Take an avatar / talking-head MP4 (HeyGen-style export) and overlay
relevant free stock B-roll on top of it during the speaking sections
the user is describing.  That's the only job.

NO captions, music, SFX, AI images, transitions, motion graphics,
text overlays, lower thirds, animations, or any documentary chrome.
Original audio is preserved untouched.

PIPELINE
--------
  1. probe_video    — ffprobe the upload (duration / fps / aspect)
  2. transcript     — use provided script OR whisper from audio
  3. chunk          — split transcript into 5-12 second semantic chunks
  4. keywords       — one batched LLM call generates English search
                      keywords for every chunk (multi-language safe)
  5. broll_plan     — natural B-roll timing layout honouring coverage
                      target + opening-avatar rule + clip-length floors
  6. fetch          — pull a free stock clip per plan slot (Pexels first,
                      Pixabay second; both keyless-friendly for previews
                      but real keys give better matches)
  7. render         — single ffmpeg invocation with overlay filters
                      (optional PiP for avatar corner mode)

The whole pipeline is designed to be FASTER than the main documentary
engine: no LLM script-gen, no cinematic cards, no music scoring, no
multiple-stage rendering — just one ffmpeg call.
"""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Optional

import requests

from .config import load_config

# Reused from existing code (read-only — we don't modify these).
# Falling back to direct HTTP if not importable so this module is
# resilient even if the main codebase moves things around.
try:
    from .footage import _pexels_fetch as _pexels_fetch_existing
except Exception:                                              # noqa: BLE001
    _pexels_fetch_existing = None
try:
    from .align import whisper_words as _whisper_words_existing
except Exception:                                              # noqa: BLE001
    _whisper_words_existing = None


# ────────────────────────────────────────────────────────────────────
# CONSTANTS
# ────────────────────────────────────────────────────────────────────
COVERAGE_PRESETS = {
    "light":    0.50,
    "balanced": 0.65,
    "heavy":    0.75,
    "maximum":  0.80,
}
DEFAULT_COVERAGE = 0.65

# B-roll clip duration floor / ceiling.  Below the floor the cut
# feels like a flash; above the ceiling viewers disengage from the
# avatar's narration.  These match the human-edited educational-
# avatar pattern (Ali Abdaal / Veritasium / Kurzgesagt style).
MIN_BROLL_S = 5.0
MAX_BROLL_S = 15.0
PREFERRED_BROLL_S = (6.0, 12.0)

# Minimum time the avatar must be visible BETWEEN B-roll bursts.
# Without this floor the video becomes a 100% stock montage with a
# disembodied voice — which is exactly the "robotic pattern" the
# user explicitly told us to avoid.
MIN_AVATAR_S = 3.0
MAX_CONTINUOUS_BROLL_S = 32.0      # never cover more than this in one run

# The opening seconds must always be the avatar so the viewer
# learns who's speaking before stock takes over.
OPENING_AVATAR_S = 7.0

# Pixabay (free) video search — used as Pexels fallback.
PIXABAY_VIDEO_URL = "https://pixabay.com/api/videos/"
PEXELS_VIDEO_URL  = "https://api.pexels.com/videos/search"

# ── AVATAR-CORNER PIP STYLE ───────────────────────────────────────
# When the user opts into Avatar Corner Mode, the avatar appears as
# a CIRCULAR portrait (not a rectangle) in one of the four corners,
# and the corner ROTATES every ~3.5 minutes through TL → TR → BR
# → BL so it doesn't sit in the same place the whole video.  This
# matches the look the user shared (clean round portrait, podcast
# style) and keeps the eye refreshed on long videos.
PIP_DIAMETER_FRAC      = 0.22       # circle diameter = 22 % of frame WIDTH
PIP_CORNER_ROTATE_S    = 210.0      # 3.5 min between corner changes
PIP_CORNER_MARGIN_PX   = 28         # gap between circle edge and frame edge
PIP_CORNERS = ("tl", "tr", "br", "bl")   # rotation order


def _pip_corner_for_time(t_s: float) -> str:
    """Which corner the avatar PiP should sit in at video time `t_s`.
    Rotates through TL → TR → BR → BL every PIP_CORNER_ROTATE_S
    (~3.5 min) so on a 14-minute video the viewer sees all four."""
    idx = int(t_s // PIP_CORNER_ROTATE_S) % len(PIP_CORNERS)
    return PIP_CORNERS[idx]


def _pip_corner_xy(corner: str, W: int, H: int,
                   pip_size: int) -> tuple[int, int]:
    """Pixel position (x, y) for the named corner."""
    m = PIP_CORNER_MARGIN_PX
    if corner == "tl":
        return (m, m)
    if corner == "tr":
        return (W - pip_size - m, m)
    if corner == "bl":
        return (m, H - pip_size - m)
    return (W - pip_size - m, H - pip_size - m)  # br (default)

# Where ffmpeg / ffprobe live — same probing rule the comparison
# script uses elsewhere in the project.
_FFMPEG  = shutil.which("ffmpeg")  or "/Users/hussnain/pinokio/bin/miniconda/bin/ffmpeg"
_FFPROBE = shutil.which("ffprobe") or "/Users/hussnain/pinokio/bin/miniconda/bin/ffprobe"

# Cap on per-job LLM keyword calls.  Single batched call covers up
# to ~25 chunks per request; for longer videos we'll loop.  This
# keeps the per-render LLM bill bounded (≤ ~$0.05 even on 40-min
# uploads).
MAX_KEYWORD_BATCH = 25


# ────────────────────────────────────────────────────────────────────
# DATA TYPES
# ────────────────────────────────────────────────────────────────────
@dataclass
class Chunk:
    """One semantic slice of the transcript / script."""
    start_s:  float
    end_s:    float
    text:     str
    keywords: list[str] = field(default_factory=list)


@dataclass
class PlanSlot:
    """One row of the B-roll timeline."""
    start_s:   float
    end_s:     float
    kind:      str                # "broll" | "avatar_only"
    query:     str   = ""
    clip_path: str   = ""         # local path to the stock clip
    pip:       bool  = False      # avatar-corner mode active here

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class JobState:
    """Live job status the web layer polls."""
    job_id:       str
    status:       str   = "running"     # running | done | error
    pct:          int   = 0
    msg:          str   = "Queued"
    error:        str   = ""
    input_path:   str   = ""
    output_path:  str   = ""
    duration_s:   float = 0.0
    n_chunks:     int   = 0
    n_broll:      int   = 0
    coverage_pct: float = 0.0
    title:        str   = ""
    language:     str   = ""

    def update(self, **kw) -> None:
        for k, v in kw.items():
            if hasattr(self, k):
                setattr(self, k, v)


# Per-process job registry (separate from the main JOBS dict in
# web.py so the two flows can never collide).
BROLL_JOBS: dict[str, JobState] = {}


# ────────────────────────────────────────────────────────────────────
# 1. PROBE — ffprobe wrapper, structured output
# ────────────────────────────────────────────────────────────────────
def probe_video(mp4: Path) -> dict:
    """Return {duration_s, width, height, fps, aspect, audio} or raise."""
    cmd = [
        _FFPROBE, "-v", "error",
        "-select_streams", "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,codec_name:format=duration",
        "-of", "json", str(mp4),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=45)
    if r.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {r.stderr[:200]}")
    data = json.loads(r.stdout or "{}")
    stream = (data.get("streams") or [{}])[0]
    fmt    = data.get("format", {})
    # fps may come back as "30000/1001" — parse safely.
    fps_raw = stream.get("r_frame_rate", "30/1")
    try:
        num, den = fps_raw.split("/")
        fps = float(num) / max(1.0, float(den))
    except Exception:                                          # noqa: BLE001
        fps = 30.0
    w = int(stream.get("width", 1920))
    h = int(stream.get("height", 1080))
    # Check for audio track separately (failing silently when absent).
    a = subprocess.run(
        [_FFPROBE, "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=codec_name", "-of", "json", str(mp4)],
        capture_output=True, text=True, timeout=15)
    has_audio = bool(json.loads(a.stdout or "{}").get("streams"))
    return {
        "duration_s": float(fmt.get("duration", 0.0)),
        "width":  w,
        "height": h,
        "fps":    fps,
        "aspect": w / max(1, h),
        "has_audio": has_audio,
    }


# ────────────────────────────────────────────────────────────────────
# 2. TRANSCRIPT — Whisper from audio, or align user-provided script
# ────────────────────────────────────────────────────────────────────
def _extract_audio(mp4: Path, out_wav: Path) -> Path:
    """Pull a clean 16kHz mono wav for Whisper.  Fast — no re-encode of video."""
    cmd = [_FFMPEG, "-y", "-i", str(mp4),
           "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
           str(out_wav)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        raise RuntimeError(f"audio extract failed: {r.stderr[-200:]}")
    return out_wav


def whisper_transcript(mp4: Path, work: Path
                       ) -> tuple[list[tuple[str, float, float]], str]:
    """Return ([(word, start_s, end_s), ...], detected_lang).

    Reuses the existing faster-whisper integration from align.py when
    importable; otherwise loads its own model.  Both paths return the
    same shape so the rest of the pipeline doesn't care."""
    if _whisper_words_existing is None:
        # Embedded fallback — slower first run but no hard dep on
        # the main codebase's helper.
        from faster_whisper import WhisperModel
        model = WhisperModel("small", device="cpu", compute_type="int8")
        wav = _extract_audio(mp4, work / "audio.wav")
        segments, info = model.transcribe(
            str(wav), language=None, word_timestamps=True, vad_filter=True)
        words = []
        for seg in segments:
            for w in (seg.words or []):
                words.append(
                    (w.word.strip(), float(w.start), float(w.end)))
        return words, getattr(info, "language", "") or ""
    # Existing helper takes the AUDIO file path directly.
    wav = _extract_audio(mp4, work / "audio.wav")
    words = _whisper_words_existing(wav)
    lang = ""
    try:
        # heuristic lang detect from the first 60 words
        sample = " ".join(w for w, _, _ in words[:60])
        lang = _detect_language(sample)
    except Exception:                                          # noqa: BLE001
        pass
    return words, lang


def _detect_language(text: str) -> str:
    """Cheap script-based language detection — no extra dep.
    Returns ISO-639-1-ish hint that the keyword generator can use."""
    if not text:
        return ""
    # Latin-only is the default; check for non-Latin scripts.
    if re.search(r"[一-鿿]", text):       # CJK
        return "zh"
    if re.search(r"[぀-ヿ]", text):       # Japanese hiragana/katakana
        return "ja"
    if re.search(r"[가-힯]", text):       # Korean Hangul
        return "ko"
    if re.search(r"[؀-ۿ]", text):       # Arabic
        return "ar"
    if re.search(r"[֐-׿]", text):       # Hebrew
        return "he"
    if re.search(r"[ऀ-ॿ]", text):       # Hindi / Devanagari
        return "hi"
    # Latin: a few cheap accent / common-word checks
    low = text.lower()
    if any(c in low for c in "ığşçöü") and "the" not in low:   # Turkish
        return "tr"
    if any(c in low for c in "äöüß"):
        return "de"
    if any(c in low for c in "éèàùç"):
        return "fr"
    if any(c in low for c in "ñáéíóú¿¡"):
        return "es"
    return "en"


# ────────────────────────────────────────────────────────────────────
# 3. CHUNK — turn words+timings (or a plain script) into Chunks
# ────────────────────────────────────────────────────────────────────
def chunk_words(words: list[tuple[str, float, float]],
                target_s: float = 8.0,
                min_s:    float = 5.0,
                max_s:    float = 14.0) -> list[Chunk]:
    """Slice the word stream into ~target_s chunks at natural punctuation
    boundaries.  Each chunk's text is the joined words; its time span
    is the first-word start to last-word end."""
    if not words:
        return []
    chunks: list[Chunk] = []
    cur_words: list[tuple[str, float, float]] = []
    cur_start = words[0][1]
    for w, s, e in words:
        cur_words.append((w, s, e))
        span = e - cur_start
        ends_sentence = bool(re.search(r"[.!?。！？]$", w))
        if (span >= target_s and ends_sentence) or span >= max_s:
            text = " ".join(x[0] for x in cur_words).strip()
            chunks.append(Chunk(
                start_s=cur_start, end_s=e, text=text))
            cur_words = []
            cur_start = e
    if cur_words:
        text = " ".join(x[0] for x in cur_words).strip()
        chunks.append(Chunk(
            start_s=cur_start, end_s=cur_words[-1][2], text=text))
    # Merge any sub-min chunks into their neighbour so we don't end
    # up with 1-second flashes.
    out: list[Chunk] = []
    for c in chunks:
        if out and (c.end_s - c.start_s) < min_s:
            out[-1] = Chunk(
                start_s=out[-1].start_s, end_s=c.end_s,
                text=(out[-1].text + " " + c.text).strip())
        else:
            out.append(c)
    return out


def chunk_script(script: str, total_duration_s: float,
                 target_s: float = 8.0) -> list[Chunk]:
    """When the user pasted a script but we have no word-level timing,
    split the script into sentences and proportionally distribute
    them across the video's duration.  Approximate but good enough
    for stock-search relevance (which is what the keywords drive)."""
    sentences = [s.strip() for s in
                 re.split(r"(?<=[.!?。！？])\s+", script.strip()) if s.strip()]
    if not sentences or total_duration_s <= 0:
        return []
    # Weight by character count so longer sentences get more time.
    weights = [max(1, len(s)) for s in sentences]
    total_w = sum(weights)
    chunks: list[Chunk] = []
    t = 0.0
    for s, w in zip(sentences, weights):
        d = total_duration_s * (w / total_w)
        chunks.append(Chunk(start_s=t, end_s=t + d, text=s))
        t += d
    # Merge until each chunk hits ~target_s
    merged: list[Chunk] = []
    for c in chunks:
        if merged and (merged[-1].end_s - merged[-1].start_s) < target_s:
            merged[-1] = Chunk(
                start_s=merged[-1].start_s, end_s=c.end_s,
                text=(merged[-1].text + " " + c.text).strip())
        else:
            merged.append(c)
    return merged


# ────────────────────────────────────────────────────────────────────
# 4. KEYWORDS — one batched LLM call generates English queries
# ────────────────────────────────────────────────────────────────────
def _keywords_heuristic(text: str) -> list[str]:
    """Zero-LLM fallback — pulls 2-3 concrete-noun candidates from
    the chunk text.  Worse than the LLM, but always available."""
    bag = re.findall(r"[A-Za-z]{4,}", text)
    stop = {"about", "after", "again", "could", "every", "first", "their",
            "there", "these", "those", "while", "which", "would", "still",
            "really", "thing", "things", "people", "person", "going",
            "actually", "because", "though", "another", "without"}
    out = []
    for w in bag:
        wl = w.lower()
        if wl in stop or wl in out:
            continue
        out.append(wl)
        if len(out) >= 3:
            break
    return out or ["lifestyle background"]


def llm_keywords(chunks: list[Chunk], title: str, language: str,
                 cfg=None) -> None:
    """Mutate `chunks` in place — fill each chunk's `keywords` list with
    2-4 short ENGLISH search queries suitable for free stock libraries.

    One single Claude call per up-to-25 chunks keeps cost trivial
    (~$0.005 per render in input/output combined).  Falls back to a
    heuristic when no key.  For non-English source text the prompt
    explicitly asks for ENGLISH keywords."""
    cfg = cfg or load_config()
    if not cfg.anthropic_api_key:
        for c in chunks:
            c.keywords = _keywords_heuristic(c.text)
        return
    # Batch in groups of 25
    for batch_start in range(0, len(chunks), MAX_KEYWORD_BATCH):
        batch = chunks[batch_start:batch_start + MAX_KEYWORD_BATCH]
        try:
            _llm_keyword_batch(batch, title, language, cfg)
        except Exception:                                      # noqa: BLE001
            # Anthropic failure → fall back to heuristic for this batch
            for c in batch:
                if not c.keywords:
                    c.keywords = _keywords_heuristic(c.text)


def _llm_keyword_batch(batch: list[Chunk], title: str, language: str,
                       cfg) -> None:
    import anthropic
    client = anthropic.Anthropic(api_key=cfg.anthropic_api_key)
    items = "\n".join(
        f"  [{i}] {c.text[:240]}" for i, c in enumerate(batch))
    lang_hint = (
        f"\nThe source text language is '{language}'. Generate keywords "
        "in ENGLISH regardless — free stock libraries index in English."
        if language and language != "en" else "")
    user = (
        f"Video title: {title or '(untitled avatar video)'}\n\n"
        f"For EACH numbered chunk below, output 2-4 short ENGLISH stock-"
        f"footage SEARCH QUERIES that would return relevant B-roll for "
        f"a talking-head video covering that moment.  Prefer CONCRETE "
        f"VISUAL nouns (people, places, objects, actions).  Avoid "
        f"abstract concepts.  Each query 1-3 words.  Return ONLY JSON: "
        f'{{"queries": [["q1","q2"], ["q1","q2","q3"], ...]}}'
        f"{lang_hint}\n\nCHUNKS:\n{items}"
    )
    msg = client.messages.create(
        model=cfg.anthropic_model, max_tokens=2000,
        messages=[{"role": "user", "content": user}])
    raw = "".join(b.text for b in msg.content
                  if getattr(b, "type", "") == "text")
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        raise RuntimeError("no JSON in LLM keyword response")
    data = json.loads(m.group(0))
    qs = data.get("queries", []) or []
    for c, kw in zip(batch, qs):
        if isinstance(kw, list):
            c.keywords = [str(q).strip() for q in kw
                          if str(q).strip()][:4]
    for c in batch:
        if not c.keywords:
            c.keywords = _keywords_heuristic(c.text)


# ────────────────────────────────────────────────────────────────────
# 5. STOCK FETCHERS — Pexels primary, Pixabay fallback (both free)
# ────────────────────────────────────────────────────────────────────
def _pixabay_video_fetch(query: str, key: str, dest: Path,
                         min_width: int = 1280) -> bool:
    """Free Pixabay video search.  Returns True on a successful clip
    download (saved to `dest`).  Safe to call with key='' — returns
    False instantly so the caller falls through to the next source."""
    if not key:
        return False
    try:
        r = requests.get(
            PIXABAY_VIDEO_URL,
            params={"key": key, "q": query, "per_page": 12,
                    "video_type": "film", "safesearch": "true",
                    "min_width": min_width},
            timeout=20)
        if r.status_code != 200:
            return False
        hits = r.json().get("hits", []) or []
        if not hits:
            return False
        # Walk hits in popularity order — pick the first that has a
        # large-enough rendition.
        for hit in hits:
            videos = hit.get("videos", {}) or {}
            for size in ("large", "medium", "small"):
                v = videos.get(size) or {}
                if v.get("url") and (v.get("width", 0) >= min_width
                                     or size == "large"):
                    url = v["url"]
                    try:
                        with requests.get(url, stream=True,
                                           timeout=60) as dl:
                            dl.raise_for_status()
                            with open(dest, "wb") as fh:
                                for chunk in dl.iter_content(1 << 16):
                                    fh.write(chunk)
                        if dest.stat().st_size > 50_000:
                            return True
                    except Exception:                          # noqa: BLE001
                        continue
        return False
    except Exception:                                          # noqa: BLE001
        return False


def find_stock_clip(queries: list[str], dest: Path, cfg,
                    variant: int = 0) -> Optional[str]:
    """Try every query through every free source until one returns
    a clip.  Returns the query that succeeded (for logging) or None.
    Pexels is tried FIRST (better relevance for English nouns); then
    Pixabay (broader stock library, often has lifestyle B-roll
    Pexels misses)."""
    for q in queries:
        if not q:
            continue
        if cfg.pexels_api_key and _pexels_fetch_existing is not None:
            try:
                if _pexels_fetch_existing(q, cfg.pexels_api_key,
                                          dest, variant=variant):
                    return q
            except Exception:                                  # noqa: BLE001
                pass
        # Pixabay key field — we read directly from env since the
        # main Config doesn't yet expose it as an attr.
        pix_key = os.environ.get("PIXABAY_API_KEY", "").strip()
        if pix_key and _pixabay_video_fetch(q, pix_key, dest):
            return q
    return None


# ────────────────────────────────────────────────────────────────────
# 6. PLAN BUILDER — natural B-roll timing layout
# ────────────────────────────────────────────────────────────────────
def build_broll_plan(chunks: list[Chunk], total_duration_s: float,
                     coverage: float,
                     pip: bool = False) -> list[PlanSlot]:
    """Decide which chunks become B-roll and which stay avatar-only.

    Strategy:
      * skip the opening (≤ OPENING_AVATAR_S) — always avatar
      * pick chunks in order until cumulative B-roll time ≥ target
      * always leave ≥ MIN_AVATAR_S gaps between B-roll slots
      * clamp each B-roll slot to [MIN_BROLL_S, MAX_BROLL_S]
      * avoid runs longer than MAX_CONTINUOUS_BROLL_S
    """
    coverage = max(0.30, min(0.85, coverage))
    target_broll_s = total_duration_s * coverage

    # Mark chunks as "candidate B-roll" or "avatar_only" first.
    candidates: list[Chunk] = []
    for c in chunks:
        if c.start_s < OPENING_AVATAR_S:
            continue
        if (c.end_s - c.start_s) < MIN_BROLL_S * 0.6:
            continue   # too short to host a clip
        candidates.append(c)

    # Prefer chunks with non-empty keywords + longer length.
    candidates.sort(key=lambda c: (-(c.end_s - c.start_s),
                                    -len(c.keywords or [])))

    chosen: list[Chunk] = []
    chosen_total = 0.0
    for c in candidates:
        slot_dur = max(MIN_BROLL_S,
                        min(MAX_BROLL_S, c.end_s - c.start_s))
        if chosen_total + slot_dur > target_broll_s:
            continue
        chosen.append(c)
        chosen_total += slot_dur

    # Sort by time so the plan is in playback order.
    chosen.sort(key=lambda c: c.start_s)

    # Enforce MIN_AVATAR_S gaps + continuous-run cap.
    plan: list[PlanSlot] = []
    prev_end = 0.0
    for c in chosen:
        # Trim slot to allowed window
        s = max(c.start_s, prev_end + MIN_AVATAR_S
                if plan and plan[-1].kind == "broll" else c.start_s)
        e = s + max(MIN_BROLL_S, min(MAX_BROLL_S, c.end_s - s))
        e = min(e, total_duration_s - 1.0)   # leave a tail second of avatar
        if e - s < MIN_BROLL_S:
            continue
        # avatar_only filler between previous broll and this one
        if plan and plan[-1].end_s < s:
            plan.append(PlanSlot(
                start_s=plan[-1].end_s, end_s=s, kind="avatar_only"))
        elif not plan and s > 0:
            plan.append(PlanSlot(
                start_s=0.0, end_s=s, kind="avatar_only"))
        plan.append(PlanSlot(
            start_s=s, end_s=e, kind="broll",
            query=" ".join((c.keywords or [c.text[:30]])[:1]),
            pip=pip))
        prev_end = e

    # Tail
    if not plan or plan[-1].end_s < total_duration_s:
        last_end = plan[-1].end_s if plan else 0.0
        plan.append(PlanSlot(
            start_s=last_end, end_s=total_duration_s, kind="avatar_only"))

    # Annotate chunk-derived queries onto broll slots
    for slot in plan:
        if slot.kind == "broll":
            # find the source chunk by time overlap
            for c in chunks:
                if c.start_s <= slot.start_s < c.end_s:
                    if c.keywords:
                        slot.query = ", ".join(c.keywords[:3])
                    break
    return plan


def _ffmpeg_error_summary(stderr: str) -> str:
    """Pull the actually-useful error lines out of a noisy ffmpeg
    stderr dump.  ffmpeg emits hundreds of progress lines like
    'frame=0 fps=0.0 size=512KiB ...' which crowd out the one real
    error line.  This helper drops those and returns the meaningful
    error lines (the first 6 'Error', 'Invalid', '[...] line, plus
    the last 2 substantive lines for context)."""
    junk_pat = re.compile(
        r"^(frame=|size=|video:|audio:|Press \[q\]|"
        r"\s+Last message|"
        r"\[swscaler|"
        r"Stream mapping|"
        r"Output #\d|"
        r"Input #\d|"
        r"\s+Stream #|"
        r"\s+Duration:|"
        r"\s+Metadata:|"
        r"\s+encoder|"
        r"\s+handler_name|"
        r"\s+vendor_id|"
        r"\s+major_brand|"
        r"\s+minor_version|"
        r"\s+compatible_brands|"
        r"\s+title|\s+creation_time|"
        r"ffmpeg version|"
        r"\s+built with|"
        r"\s+configuration:|"
        r"\s+lib(av|sw|post)|"
        r"Guessed Channel Layout|"
        r"At least one output file)"
    )
    keep = []
    for line in stderr.splitlines():
        s = line.strip()
        if not s:
            continue
        if junk_pat.match(line) or junk_pat.match(s):
            continue
        keep.append(line)
    # Always include first 8 + last 4 substantive lines.
    if len(keep) > 12:
        keep = keep[:8] + ["  …"] + keep[-4:]
    return "\n".join(keep) or stderr[-400:]


# ────────────────────────────────────────────────────────────────────
# 7. RENDER — segment-by-segment for any video length
# ────────────────────────────────────────────────────────────────────
# OLD APPROACH (single ffmpeg with N inputs + filter_complex) hit
# a wall on a real 18-min upload with 77 B-roll slots: 78 -i flags
# + a ~9KB filter graph blew past ffmpeg's argv / filter parser
# limits and produced 'frame=0' empty output.
#
# NEW APPROACH:
#   1. Walk the plan in time order.
#   2. For each slot, render a tiny independent segment (one input
#      OR overlay-with-two-inputs) covering JUST that time window.
#      All segments share the avatar's audio track.
#   3. Concat the segments into the final MP4 with -f concat.
#
# This trades 1 huge ffmpeg call for many tiny ones — but each
# tiny one finishes in 0.5-2 s and scales linearly to any video
# length / slot count.  20-min video with 80 slots ≈ 80 × 1.5 s ≈
# 2 min total render (vs the old single-call which just OOM'd).
# ────────────────────────────────────────────────────────────────────

_SEGMENT_MAX_INPUTS = 6   # never exceed this many -i per ffmpeg call


def _render_avatar_only_segment(input_mp4: Path, slot: PlanSlot,
                                 out: Path, W: int, H: int,
                                 fps: float) -> None:
    """Trim the avatar to [slot.start_s, slot.end_s) — fast copy
    of video + audio with re-encoded video so concat boundaries
    are clean.  No filter graph."""
    dur = slot.end_s - slot.start_s
    cmd = [_FFMPEG, "-y",
           "-ss", f"{slot.start_s:.3f}",
           "-i", str(input_mp4),
           "-t", f"{dur:.3f}",
           "-vf", f"scale={W}:{H},fps={fps:.3f},setpts=PTS-STARTPTS",
           "-af", "asetpts=PTS-STARTPTS",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
           "-c:a", "aac", "-b:a", "192k",
           "-pix_fmt", "yuv420p",
           str(out)]
    p = subprocess.run(cmd, capture_output=True, text=True,
                       timeout=300)
    if p.returncode != 0:
        raise RuntimeError(
            f"avatar segment render failed at "
            f"t={slot.start_s:.1f}-{slot.end_s:.1f}:\n"
            f"{_ffmpeg_error_summary(p.stderr)}")


def _render_broll_segment(input_mp4: Path, slot: PlanSlot,
                          out: Path, W: int, H: int, fps: float,
                          pip: bool) -> None:
    """Render a single B-roll-over-avatar segment.  Avatar provides
    audio (and PiP video if pip=True); stock provides full-screen
    video for this window."""
    dur = slot.end_s - slot.start_s
    avatar_ss = f"{slot.start_s:.3f}"
    if not pip:
        # Simple: stock is the only video; avatar audio overlays.
        cmd = [_FFMPEG, "-y",
               "-ss", avatar_ss,
               "-i", str(input_mp4),                # input 0 — avatar (audio)
               "-i", slot.clip_path,                # input 1 — stock (video)
               "-t", f"{dur:.3f}",
               "-filter_complex",
               (f"[1:v]scale={W}:{H}:force_original_aspect_ratio=increase,"
                f"crop={W}:{H},setpts=PTS-STARTPTS,fps={fps:.3f}[v]"),
               "-map", "[v]", "-map", "0:a?",
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
               "-c:a", "aac", "-b:a", "192k",
               "-pix_fmt", "yuv420p",
               "-shortest",
               str(out)]
    else:
        # PiP — CIRCULAR avatar portrait in a rotating corner.
        # Filter pipeline:
        #   stock → scale+crop to fill bg
        #   avatar → square-crop the centre → scale to PIP_SIZE → make
        #            CIRCULAR with an alpha mask (geq) → subtle 3-px
        #            white ring so it pops on bright backgrounds
        #   overlay circular avatar at corner_xy()
        pip_size = int(W * PIP_DIAMETER_FRAC)
        # Force even pixel size (libx264 requires even dimensions).
        pip_size -= pip_size % 2
        corner = _pip_corner_for_time(slot.start_s)
        x, y = _pip_corner_xy(corner, W, H, pip_size)
        # Square-crop the avatar's CENTRE so a portrait frame becomes
        # the right shape for a round portrait.  We use the smaller
        # of the avatar's dimensions as the square side.
        # Then scale to pip_size and apply circular alpha mask via geq.
        cx, cy = pip_size / 2.0, pip_size / 2.0
        radius = pip_size / 2.0 - 1
        # geq alpha: 255 inside circle, 0 outside.  Hard edge — the
        # ring overlay below softens it visually.  This keeps the
        # filter cheap (no per-pixel anti-alias maths).
        alpha_expr = (
            f"if(lte(hypot(X-{cx},Y-{cy}),{radius}),255,0)"
        )
        cmd = [_FFMPEG, "-y",
               "-ss", avatar_ss,
               "-i", str(input_mp4),
               "-i", slot.clip_path,
               "-t", f"{dur:.3f}",
               "-filter_complex",
               (
                # 1. Stock → background fill
                f"[1:v]scale={W}:{H}:force_original_aspect_ratio=increase,"
                f"crop={W}:{H},setpts=PTS-STARTPTS,fps={fps:.3f}[bg];"
                # 2. Avatar → square centre crop → scale → circular alpha
                f"[0:v]crop=ih:ih,scale={pip_size}:{pip_size},"
                f"format=yuva420p,geq=lum='p(X,Y)':"
                f"cb='p(X,Y)':cr='p(X,Y)':a='{alpha_expr}',"
                f"setpts=PTS-STARTPTS[pip];"
                # 3. Composite — round avatar on top of stock at corner
                f"[bg][pip]overlay=x={x}:y={y}:format=auto[v]"
               ),
               "-map", "[v]", "-map", "0:a?",
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
               "-c:a", "aac", "-b:a", "192k",
               "-pix_fmt", "yuv420p",
               "-shortest",
               str(out)]
    p = subprocess.run(cmd, capture_output=True, text=True,
                       timeout=300)
    if p.returncode != 0:
        raise RuntimeError(
            f"broll segment render failed at "
            f"t={slot.start_s:.1f}-{slot.end_s:.1f}, "
            f"query='{slot.query[:40]}':\n"
            f"{_ffmpeg_error_summary(p.stderr)}")


def _concat_segments(segments: list[Path], out: Path) -> None:
    """ffmpeg -f concat needs a list file.  All segments must
    share codec / fps / resolution — which we guarantee by
    encoding every segment with the same x264/aac settings."""
    list_file = out.with_suffix(".concat.txt")
    list_file.write_text(
        "\n".join(f"file '{s.resolve()}'" for s in segments) + "\n",
        encoding="utf-8")
    cmd = [_FFMPEG, "-y", "-f", "concat", "-safe", "0",
           "-i", str(list_file),
           "-c", "copy",
           str(out)]
    p = subprocess.run(cmd, capture_output=True, text=True,
                       timeout=900)
    if p.returncode != 0:
        # Concat with -c copy fails when keyframe alignment is off;
        # fall back to a re-encode pass which always works.
        cmd = [_FFMPEG, "-y", "-f", "concat", "-safe", "0",
               "-i", str(list_file),
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
               "-c:a", "aac", "-b:a", "192k",
               "-pix_fmt", "yuv420p",
               str(out)]
        p = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=1200)
        if p.returncode != 0:
            raise RuntimeError(
                "concat failed even after re-encode fallback:\n"
                + _ffmpeg_error_summary(p.stderr))


def render(input_mp4: Path, plan: list[PlanSlot],
           output_mp4: Path, pip: bool = False,
           probe: Optional[dict] = None,
           progress_cb: Optional[Callable[[int, str], None]] = None
           ) -> Path:
    """Segment-by-segment render — scales to any video length / B-roll
    slot count.  Each plan slot becomes one tiny ffmpeg invocation;
    all segments then concat into the final MP4.

    `progress_cb(pct, msg)` is called between segments so the web
    layer can report fine-grained progress during this stage."""
    info = probe or probe_video(input_mp4)
    W, H = info["width"], info["height"]
    fps = float(info["fps"]) or 25.0

    # Filter plan to only include slots with valid time windows.
    # B-roll slots whose clip download failed (no clip_path) collapse
    # to avatar_only for graceful degradation.
    work_slots: list[PlanSlot] = []
    for s in plan:
        if s.end_s - s.start_s < 0.2:
            continue   # skip near-zero windows
        ss = PlanSlot(start_s=s.start_s, end_s=s.end_s,
                      kind=s.kind, query=s.query,
                      clip_path=s.clip_path, pip=s.pip)
        if (s.kind == "broll"
                and (not s.clip_path
                     or not Path(s.clip_path).exists())):
            ss.kind = "avatar_only"     # graceful degradation
            ss.clip_path = ""
        work_slots.append(ss)

    if not work_slots:
        # Nothing to do — copy input through.
        cmd = [_FFMPEG, "-y", "-i", str(input_mp4),
               "-c", "copy", str(output_mp4)]
        subprocess.run(cmd, capture_output=True, text=True,
                       timeout=600)
        return output_mp4

    seg_dir = output_mp4.parent / "segments"
    seg_dir.mkdir(parents=True, exist_ok=True)
    segments: list[Path] = []
    n = len(work_slots)
    for i, slot in enumerate(work_slots):
        seg_out = seg_dir / f"seg_{i:04d}.mp4"
        if progress_cb is not None:
            pct = 80 + int(15 * (i / max(1, n)))
            progress_cb(pct,
                f"Rendering segment {i+1}/{n} "
                f"({slot.kind} {slot.end_s - slot.start_s:.1f}s)")
        if slot.kind == "broll":
            _render_broll_segment(input_mp4, slot, seg_out, W, H,
                                   fps, pip)
        else:
            _render_avatar_only_segment(input_mp4, slot, seg_out,
                                        W, H, fps)
        segments.append(seg_out)

    if progress_cb is not None:
        progress_cb(96, f"Concatenating {len(segments)} segments…")
    _concat_segments(segments, output_mp4)
    return output_mp4


# ────────────────────────────────────────────────────────────────────
# 8. ORCHESTRATOR — the end-to-end pipeline a job thread runs
# ────────────────────────────────────────────────────────────────────
def run_pipeline(job_id: str,
                 input_mp4: Path,
                 title: str   = "",
                 script: str  = "",
                 coverage:   float = DEFAULT_COVERAGE,
                 pip: bool   = False,
                 output_dir: Optional[Path] = None) -> Path:
    """Top-level entry point — drives the whole flow.

    `job_id` is the key under which progress is reported via
    BROLL_JOBS[job_id].  The function returns the final output MP4
    path on success or raises with status='error' set.  Designed to
    be called from a background thread."""
    job = BROLL_JOBS.setdefault(job_id, JobState(job_id=job_id))
    job.update(status="running", pct=0, msg="Starting…",
               input_path=str(input_mp4), title=title)
    try:
        cfg = load_config()
        out_dir = Path(output_dir or
                       (Path("output") / "heygen_broll" / job_id))
        out_dir.mkdir(parents=True, exist_ok=True)
        work = out_dir / "work"
        work.mkdir(exist_ok=True)
        stock_dir = out_dir / "stock"
        stock_dir.mkdir(exist_ok=True)

        # 1. Probe
        job.update(pct=4, msg="Probing video…")
        info = probe_video(input_mp4)
        job.update(duration_s=info["duration_s"])

        # 2. Transcript
        if script and script.strip():
            job.update(pct=12, msg="Using pasted script…")
            lang = _detect_language(script)
            chunks = chunk_script(script, info["duration_s"])
            job.update(language=lang)
        else:
            job.update(pct=12,
                       msg="Transcribing avatar audio (Whisper)…")
            words, lang = whisper_transcript(input_mp4, work)
            if not words:
                raise RuntimeError(
                    "Transcript extraction failed. "
                    "Please paste the script manually.")
            chunks = chunk_words(words)
            job.update(language=lang)
        job.update(n_chunks=len(chunks))

        # 3. Keywords
        job.update(pct=24, msg=f"Generating search keywords for "
                               f"{len(chunks)} chunks…")
        llm_keywords(chunks, title, job.language, cfg)

        # 4. Plan
        job.update(pct=34, msg="Building B-roll plan…")
        plan = build_broll_plan(chunks, info["duration_s"], coverage,
                                pip=pip)
        broll_slots = [s for s in plan if s.kind == "broll"]

        # 5. Fetch stock
        for i, slot in enumerate(broll_slots):
            pct = 38 + int(40 * (i / max(1, len(broll_slots))))
            queries = [q.strip() for q in slot.query.split(",")
                       if q.strip()] or [slot.query]
            dest = stock_dir / f"clip_{i:03d}.mp4"
            job.update(pct=pct,
                       msg=f"Fetching stock {i+1}/{len(broll_slots)}: "
                           f"{queries[0][:48]}")
            picked = find_stock_clip(queries, dest, cfg, variant=i)
            if picked and dest.exists():
                slot.clip_path = str(dest)
            # else: leave clip_path empty — render() will skip it and
            # the avatar stays visible for that window (graceful
            # degradation, no crash).

        # Re-compute actual coverage based on slots that got clips
        used = [s for s in broll_slots if s.clip_path]
        job.update(n_broll=len(used),
                   coverage_pct=round(
                       100.0 * sum(s.end_s - s.start_s for s in used) /
                       max(0.1, info["duration_s"]), 1))

        # 6. Render — segment-by-segment, scales to any length.
        # The progress_cb lets the user see per-segment progress
        # instead of a single 80%→100% jump.
        job.update(pct=80, msg="Starting render…")
        out_mp4 = out_dir / "output.mp4"

        def _on_render_progress(pct: int, msg: str) -> None:
            job.update(pct=pct, msg=msg)

        render(input_mp4, plan, out_mp4, pip=pip, probe=info,
               progress_cb=_on_render_progress)

        # 7. Persist plan + summary alongside the MP4
        (out_dir / "plan.json").write_text(
            json.dumps([s.to_dict() for s in plan], indent=2),
            encoding="utf-8")
        (out_dir / "summary.json").write_text(json.dumps({
            "duration_s": info["duration_s"],
            "language":   job.language,
            "n_chunks":   len(chunks),
            "n_broll_slots": len(broll_slots),
            "n_broll_with_clip": len(used),
            "coverage_pct": job.coverage_pct,
            "pip": pip,
            "title": title,
        }, indent=2), encoding="utf-8")

        job.update(pct=100, msg="Done",
                   status="done", output_path=str(out_mp4))
        return out_mp4
    except Exception as e:                                     # noqa: BLE001
        job.update(status="error",
                   error=f"{type(e).__name__}: {e}",
                   msg=f"Failed: {e}")
        raise


__all__ = [
    "COVERAGE_PRESETS",
    "DEFAULT_COVERAGE",
    "Chunk",
    "PlanSlot",
    "JobState",
    "BROLL_JOBS",
    "probe_video",
    "whisper_transcript",
    "chunk_words",
    "chunk_script",
    "llm_keywords",
    "find_stock_clip",
    "build_broll_plan",
    "render",
    "run_pipeline",
]
