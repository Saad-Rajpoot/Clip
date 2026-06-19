"""
research_engine.py — Documentary reference video analysis module for Vidlore's
autonomous improvement loop.

Analyzes YouTube documentary videos, extracts editing DNA into structured
patterns, and scores Vidlore renders against 15 quality dimensions.

Environment variables:
  RESEARCH_CACHE=1                    — force cache (skip LLM even if notes exist)
  RESEARCH_MAX_LLM_CALLS_PER_RUN=10  — hard cap on LLM calls per run (default 10)
  ANTHROPIC_API_KEY                   — required for LLM-backed analysis
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── Paths ─────────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_NOTES_DIR = _PROJECT_ROOT / "research" / "documentary_references" / "video_notes"
_DNA_LIBRARY = _PROJECT_ROOT / "research" / "documentary_references" / "extracted_patterns" / "dna_library.json"
_FFMPEG_BIN = Path(
    "/Users/hussnain/Desktop/vidrush-clone/.venv/lib/python3.9"
    "/site-packages/imageio_ffmpeg/binaries/ffmpeg-macos-aarch64-v7.1"
)
_FFPROBE_BIN = _FFMPEG_BIN.parent / _FFMPEG_BIN.name.replace("ffmpeg", "ffprobe")

# ── Budget globals ────────────────────────────────────────────────────────────
_MAX_LLM_CALLS: int = int(
    os.environ.get("RESEARCH_MAX_LLM_CALLS_PER_RUN", "10")
)
_llm_calls_this_run: int = 0

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [research_engine] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("research_engine")

# ── Banned templates (mirrors script_gen.py) ──────────────────────────────────
_BANNED_TEMPLATES: frozenset[str] = frozenset({
    "era_banner",
    "comparison",
    "stat_dashboard",
})


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cache_forced() -> bool:
    return os.environ.get("RESEARCH_CACHE", "").strip() in ("1", "true", "yes")


def _budget_ok() -> bool:
    return _llm_calls_this_run < _MAX_LLM_CALLS


def _charge_llm() -> None:
    global _llm_calls_this_run
    _llm_calls_this_run += 1


def _notes_path(video_id: str) -> Path:
    _NOTES_DIR.mkdir(parents=True, exist_ok=True)
    return _NOTES_DIR / f"{video_id}.json"


def _load_cache(video_id: str) -> dict | None:
    p = _notes_path(video_id)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:
            log.warning("Cache read failed for %s: %s", video_id, exc)
    return None


def _save_cache(video_id: str, notes: dict) -> None:
    p = _notes_path(video_id)
    p.write_text(json.dumps(notes, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Saved notes → %s", p)


def _run(cmd: list[str], timeout: int = 60) -> tuple[int, str, str]:
    """Run subprocess, return (returncode, stdout, stderr). Never raises."""
    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
        return r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "timeout"
    except FileNotFoundError as exc:
        return -1, "", str(exc)
    except Exception as exc:
        return -1, "", str(exc)


# ─────────────────────────────────────────────────────────────────────────────
# Transcript / metadata fetchers
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_metadata_ytdlp(video_id: str) -> dict:
    """Return flat JSON metadata dict via yt-dlp (no download)."""
    cmd = [
        "python3", "-m", "yt_dlp",
        "--skip-download",
        "--dump-json",
        "--no-playlist",
        f"https://www.youtube.com/watch?v={video_id}",
    ]
    rc, out, err = _run(cmd, timeout=30)
    if rc != 0 or not out.strip():
        log.debug("yt-dlp metadata failed: %s", err[:200])
        return {}
    try:
        return json.loads(out)
    except Exception:
        return {}


def _fetch_transcript_ytdlp(video_id: str) -> str:
    """Attempt to fetch auto/manual captions via yt-dlp. Returns plain text."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = [
            "yt-dlp",
            "--skip-download",
            "--write-auto-subs",
            "--write-subs",
            "--sub-langs", "en",
            "--sub-format", "vtt",
            "--no-playlist",
            "-o", f"{tmpdir}/%(id)s.%(ext)s",
            f"https://www.youtube.com/watch?v={video_id}",
        ]
        rc, out, err = _run(cmd, timeout=30)
        # Look for any .vtt file written
        vtt_files = list(Path(tmpdir).glob("*.vtt"))
        if not vtt_files:
            return ""
        try:
            raw = vtt_files[0].read_text(encoding="utf-8", errors="replace")
            return _vtt_to_text(raw)
        except Exception:
            return ""


def _fetch_transcript_api(video_id: str) -> str:
    """Fallback: youtube-transcript-api."""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi  # type: ignore
        entries = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
        return " ".join(e["text"].replace("\n", " ") for e in entries)
    except Exception as exc:
        log.debug("youtube_transcript_api failed: %s", exc)
        return ""


def _vtt_to_text(raw: str) -> str:
    """Strip VTT cue headers and tags, return plain text."""
    lines = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("WEBVTT") or "-->" in line or re.match(r"^\d+$", line):
            continue
        # strip <...> tags
        line = re.sub(r"<[^>]+>", "", line)
        if line:
            lines.append(line)
    # deduplicate consecutive identical lines (subtitle repetition)
    deduped: list[str] = []
    prev = None
    for ln in lines:
        if ln != prev:
            deduped.append(ln)
            prev = ln
    return " ".join(deduped)


def _fetch_transcript(video_id: str) -> str:
    """Try yt-dlp first, fall back to youtube-transcript-api."""
    t = _fetch_transcript_ytdlp(video_id)
    if t:
        return t
    return _fetch_transcript_api(video_id)


# ─────────────────────────────────────────────────────────────────────────────
# LLM analysis
# ─────────────────────────────────────────────────────────────────────────────

def _build_analysis_prompt(meta: dict, transcript: str, cfg: dict) -> str:
    title = meta.get("title", "Unknown")
    channel = meta.get("uploader", meta.get("channel", "Unknown"))
    duration = meta.get("duration", 0)
    description = (meta.get("description") or "")[:500]
    transcript_excerpt = transcript[:3000] if transcript else "(no transcript available)"

    return f"""You are a senior documentary editor analyzing a YouTube video to extract its editing DNA for a reference library.

VIDEO METADATA:
  Title:    {title}
  Channel:  {channel}
  Duration: {duration}s
  Description excerpt: {description}

TRANSCRIPT EXCERPT (first ~3000 chars):
{transcript_excerpt}

Analyze this video and return a SINGLE valid JSON object matching this exact schema. Be specific and concrete — no vague answers. For empty/unknown fields use sensible defaults.

{{
  "niche": "<single phrase: e.g. 'true crime', 'geopolitics', 'history', 'tech'>",
  "hook": {{
    "first_5s": "<describe exactly what happens in the first 5 seconds>",
    "first_15s": "<what happens in seconds 5-15>",
    "first_30s": "<what happens in seconds 15-30>",
    "technique": "<hook technique: question/cold-open/shocking-stat/reenactment/mystery/narration>"
  }},
  "pacing": {{
    "avg_shot_s": <estimated average shot length in seconds as a number>,
    "fast_section_s": [<list of timestamps (in seconds) where cutting accelerates>],
    "slow_section_s": [<list of timestamps where pacing slows to let moments breathe>],
    "tension_peaks": [<timestamps where audio/visual tension peaks>]
  }},
  "graphics": {{
    "types_used": [<list of graphic types: lower_third, map, document, stat, quote, etc.>],
    "frequency_per_min": <estimated graphic events per minute as a number>,
    "lower_thirds": <true/false>,
    "maps": <true/false>,
    "documents": <true/false>,
    "charts": <true/false>,
    "avoided": [<graphic/edit techniques this video deliberately avoids>]
  }},
  "sound": {{
    "whoosh_style": "<describe transition sound design: restrained/aggressive/absent/cinematic>",
    "silence_pockets": [<timestamps in seconds where silence is used for impact>],
    "music_psychology": "<describe music strategy: builds tension / emotional underscore / sparse>",
    "sfx_restrained": <true if SFX are used sparingly, false if heavy>
  }},
  "cinematography": {{
    "real_footage_pct": <estimated % of real/original footage>,
    "stock_pct": <estimated % of stock footage>,
    "archive_pct": <estimated % of archival footage>,
    "motion_style": "<slow-push-in / handheld / static / mixed>",
    "push_in_usage": "<never/subtle/moderate/frequent>"
  }},
  "text_design": {{
    "font_style": "<serif/sans-serif/mixed/monospace/handwritten>",
    "placement": "<lower-third/centered/corner/varied>",
    "animation": "<fade/slide/typewriter/cut/mixed>",
    "density": "<low|medium|high>"
  }},
  "anti_template": {{
    "custom_feel": <true if it feels handcrafted, false if template-y>,
    "repeated_cards": <true if same graphic layout repeats robotically>,
    "what_makes_it_real": [<list: specific techniques that give human/real feel>],
    "what_to_avoid": [<list: specific things this channel avoids that make it feel authentic>]
  }},
  "patterns_for_vidlore": [<list of 3-7 actionable pattern strings to adopt>],
  "improvements_suggested": [<list of 2-5 ways Vidlore could improve by studying this video>]
}}

Return ONLY the JSON object, no markdown fences, no commentary.
"""


def _call_llm(prompt: str, cfg: dict) -> str:
    """Call Anthropic Claude API. Returns raw text response."""
    global _llm_calls_this_run

    if not _budget_ok():
        raise RuntimeError(
            f"LLM budget exhausted: {_llm_calls_this_run}/{_MAX_LLM_CALLS} calls used this run"
        )

    api_key = cfg.get("anthropic_api_key") or os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise RuntimeError("No ANTHROPIC_API_KEY available for LLM analysis")

    model = cfg.get("anthropic_model", "claude-sonnet-4-6")

    try:
        import anthropic  # type: ignore
    except ImportError:
        raise RuntimeError("anthropic package not installed; run: pip install anthropic")

    client = anthropic.Anthropic(api_key=api_key)
    _charge_llm()

    msg = client.messages.create(
        model=model,
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text if msg.content else ""


def _parse_llm_json(raw: str) -> dict:
    """Extract JSON from LLM response, tolerating markdown fences."""
    raw = raw.strip()
    # Strip ```json ... ``` or ``` ... ```
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw.strip())
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Try to find the first {...} block
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
        log.warning("Could not parse LLM JSON response; returning empty dict")
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def analyze_youtube_video(video_id: str, cfg: dict) -> dict:
    """Full analysis of one YouTube video. Returns structured notes dict.

    Uses yt-dlp for transcript if available, else youtube-transcript-api.
    Never downloads video. Uses metadata + transcript + sampled info.
    Results are cached to disk; skips re-analysis if cache exists.
    """
    # ── Cache check ──────────────────────────────────────────────────────────
    if _cache_forced():
        cached = _load_cache(video_id)
        if cached:
            log.info("RESEARCH_CACHE=1: returning cached notes for %s", video_id)
            return cached

    cached = _load_cache(video_id)
    if cached:
        log.info("Cache hit for %s — skipping re-analysis", video_id)
        return cached

    log.info("Analyzing video %s …", video_id)

    # ── Fetch metadata ────────────────────────────────────────────────────────
    meta = _fetch_metadata_ytdlp(video_id)
    if not meta:
        log.warning("Could not fetch metadata for %s via yt-dlp", video_id)

    title = meta.get("title", "")
    channel = meta.get("uploader", meta.get("channel", ""))
    duration_s = int(meta.get("duration", 0) or 0)

    # ── Fetch transcript ──────────────────────────────────────────────────────
    transcript = _fetch_transcript(video_id)
    if transcript:
        log.info("Transcript fetched: %d chars", len(transcript))
    else:
        log.info("No transcript available for %s", video_id)

    # ── LLM analysis ─────────────────────────────────────────────────────────
    llm_data: dict = {}
    if _budget_ok():
        prompt = _build_analysis_prompt(meta, transcript, cfg)
        try:
            raw_response = _call_llm(prompt, cfg)
            llm_data = _parse_llm_json(raw_response)
        except RuntimeError as exc:
            log.warning("LLM call skipped: %s", exc)
        except Exception as exc:
            log.error("LLM analysis error: %s", exc)
    else:
        log.warning("LLM budget exhausted; skipping LLM analysis for %s", video_id)

    # ── Assemble notes with schema defaults ───────────────────────────────────
    notes: dict[str, Any] = {
        "video_id": video_id,
        "title": title,
        "channel": channel,
        "niche": llm_data.get("niche", ""),
        "duration_s": duration_s,
        "analyzed_at": datetime.now(timezone.utc).isoformat(),
        # IMP_028 — confidence/source so downstream NEVER over-trusts a
        # transcript-less analysis. When yt-dlp / youtube-transcript-api can't
        # fetch captions (common on premium channels), the LLM INFERS patterns
        # from title+description only ("host likely opens with…") — useful as
        # a hint but not an observation. Tag it so the DNA + any improvement
        # promotion can down-weight inferred entries.
        "transcript_available": bool(transcript),
        "transcript_chars": len(transcript or ""),
        "analysis_confidence": ("observed" if transcript
                                else "inferred-metadata-only"),
        "hook": {
            "first_5s": "",
            "first_15s": "",
            "first_30s": "",
            "technique": "",
            **llm_data.get("hook", {}),
        },
        "pacing": {
            "avg_shot_s": 0,
            "fast_section_s": [],
            "slow_section_s": [],
            "tension_peaks": [],
            **llm_data.get("pacing", {}),
        },
        "graphics": {
            "types_used": [],
            "frequency_per_min": 0,
            "lower_thirds": True,
            "maps": False,
            "documents": False,
            "charts": False,
            "avoided": [],
            **llm_data.get("graphics", {}),
        },
        "sound": {
            "whoosh_style": "",
            "silence_pockets": [],
            "music_psychology": "",
            "sfx_restrained": True,
            **llm_data.get("sound", {}),
        },
        "cinematography": {
            "real_footage_pct": 0,
            "stock_pct": 0,
            "archive_pct": 0,
            "motion_style": "",
            "push_in_usage": "",
            **llm_data.get("cinematography", {}),
        },
        "text_design": {
            "font_style": "",
            "placement": "",
            "animation": "",
            "density": "medium",
            **llm_data.get("text_design", {}),
        },
        "anti_template": {
            "custom_feel": True,
            "repeated_cards": False,
            "what_makes_it_real": [],
            "what_to_avoid": [],
            **llm_data.get("anti_template", {}),
        },
        "patterns_for_vidlore": llm_data.get("patterns_for_vidlore", []),
        "improvements_suggested": llm_data.get("improvements_suggested", []),
    }

    _save_cache(video_id, notes)
    return notes


def extract_editing_patterns(notes: dict) -> dict:
    """Convert raw video notes into a structured pattern library entry.

    Returns a dict suitable for merging into the DNA library.
    """
    video_id = notes.get("video_id", "unknown")
    niche = notes.get("niche", "unknown")
    duration_s = notes.get("duration_s", 0)

    hook = notes.get("hook", {})
    pacing = notes.get("pacing", {})
    graphics = notes.get("graphics", {})
    sound = notes.get("sound", {})
    cinematography = notes.get("cinematography", {})
    text_design = notes.get("text_design", {})
    anti_template = notes.get("anti_template", {})

    # Classify pacing tier
    avg_shot = pacing.get("avg_shot_s", 0)
    if avg_shot <= 0:
        pacing_tier = "unknown"
    elif avg_shot < 4:
        pacing_tier = "fast"
    elif avg_shot < 8:
        pacing_tier = "medium"
    else:
        pacing_tier = "slow"

    # Classify graphic density
    freq = graphics.get("frequency_per_min", 0)
    if freq <= 0:
        graphic_density = "none"
    elif freq < 2:
        graphic_density = "sparse"
    elif freq < 5:
        graphic_density = "moderate"
    else:
        graphic_density = "heavy"

    # Hook fingerprint
    hook_technique = hook.get("technique", "")
    hook_fingerprint = {
        "technique": hook_technique,
        "has_strong_hook": bool(hook.get("first_5s")),
        "first_5s": hook.get("first_5s", ""),
    }

    # Sound fingerprint
    sound_fingerprint = {
        "music_psychology": sound.get("music_psychology", ""),
        "sfx_restrained": sound.get("sfx_restrained", True),
        "uses_silence": bool(sound.get("silence_pockets")),
        "whoosh_style": sound.get("whoosh_style", ""),
    }

    # Cinematography fingerprint
    cine = {
        "motion_style": cinematography.get("motion_style", ""),
        "push_in_usage": cinematography.get("push_in_usage", ""),
        "footage_mix": {
            "real_pct": cinematography.get("real_footage_pct", 0),
            "stock_pct": cinematography.get("stock_pct", 0),
            "archive_pct": cinematography.get("archive_pct", 0),
        },
    }

    # Anti-template rules
    anti = {
        "custom_feel": anti_template.get("custom_feel", True),
        "avoids_repeated_cards": not anti_template.get("repeated_cards", False),
        "authenticity_techniques": anti_template.get("what_makes_it_real", []),
        "avoidance_rules": anti_template.get("what_to_avoid", []),
    }

    # Graphic types actually used
    graphic_types = graphics.get("types_used", [])

    pattern: dict[str, Any] = {
        "source_video_id": video_id,
        "source_title": notes.get("title", ""),
        "channel": notes.get("channel", ""),
        "niche": niche,
        "duration_s": duration_s,
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        # ── Core DNA ──────────────────────────────────────────────────────────
        "pacing_tier": pacing_tier,
        "avg_shot_s": avg_shot,
        "graphic_density": graphic_density,
        "graphic_freq_per_min": freq,
        "graphic_types": graphic_types,
        "text_density": text_design.get("density", "medium"),
        "text_animation": text_design.get("animation", ""),
        "font_style": text_design.get("font_style", ""),
        # ── Deep fingerprints ─────────────────────────────────────────────────
        "hook": hook_fingerprint,
        "sound": sound_fingerprint,
        "cinematography": cine,
        "anti_template": anti,
        # ── Actionable takeaways ──────────────────────────────────────────────
        "patterns_for_vidlore": notes.get("patterns_for_vidlore", []),
        "improvements_suggested": notes.get("improvements_suggested", []),
    }
    return pattern


def update_dna_library(patterns: dict, library_path: Path) -> None:
    """Merge new patterns into the documentary DNA library JSON.

    The library is a dict keyed by source_video_id. New entries are added;
    existing entries are updated (newer analysis wins).
    """
    library_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing library
    library: dict[str, Any] = {}
    if library_path.exists():
        try:
            library = json.loads(library_path.read_text(encoding="utf-8"))
        except Exception as exc:
            log.warning("Could not load DNA library at %s: %s", library_path, exc)
            library = {}

    video_id = patterns.get("source_video_id", "unknown")
    is_new = video_id not in library
    library[video_id] = patterns

    library_path.write_text(
        json.dumps(library, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    verb = "Added" if is_new else "Updated"
    log.info("%s pattern for %s in DNA library (%d total entries)", verb, video_id, len(library))


# ─────────────────────────────────────────────────────────────────────────────
# ffprobe helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ffprobe_bin() -> str:
    """Return path to ffprobe, preferring bundled binary."""
    # Try bundled ffprobe first
    if _FFPROBE_BIN.exists():
        return str(_FFPROBE_BIN)
    # Try ffprobe in same dir as ffmpeg
    ffprobe_sibling = _FFMPEG_BIN.parent / "ffprobe"
    if ffprobe_sibling.exists():
        return str(ffprobe_sibling)
    # Fall back to system ffprobe
    return "ffprobe"


def _ffmpeg_bin() -> str:
    if _FFMPEG_BIN.exists():
        return str(_FFMPEG_BIN)
    return "ffmpeg"


def _probe_duration(mp4_path: Path) -> float:
    """Return video duration in seconds via ffprobe."""
    cmd = [
        _ffprobe_bin(),
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        str(mp4_path),
    ]
    rc, out, err = _run(cmd, timeout=30)
    if rc != 0:
        log.debug("ffprobe duration failed: %s", err[:200])
        return 0.0
    try:
        data = json.loads(out)
        return float(data.get("format", {}).get("duration", 0))
    except Exception:
        return 0.0


def _probe_video_stream(mp4_path: Path) -> dict:
    """Return first video stream info dict."""
    cmd = [
        _ffprobe_bin(),
        "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        "-select_streams", "v:0",
        str(mp4_path),
    ]
    rc, out, _ = _run(cmd, timeout=30)
    if rc != 0:
        return {}
    try:
        data = json.loads(out)
        streams = data.get("streams", [])
        return streams[0] if streams else {}
    except Exception:
        return {}


def _detect_scene_changes(mp4_path: Path, threshold: float = 0.4) -> list[float]:
    """Detect scene-change timestamps using ffmpeg scene filter.

    Returns list of timestamps (seconds) where scene changes occur.
    Uses a temp output to avoid writing a full file.
    """
    cmd = [
        _ffmpeg_bin(),
        "-i", str(mp4_path),
        "-vf", f"select='gt(scene,{threshold})',showinfo",
        "-f", "null",
        "-",
    ]
    rc, out, err = _run(cmd, timeout=120)
    # showinfo writes to stderr
    combined = err + out
    timestamps: list[float] = []
    for m in re.finditer(r"pts_time:([\d.]+)", combined):
        try:
            timestamps.append(float(m.group(1)))
        except ValueError:
            pass
    return sorted(set(timestamps))


def _measure_audio_loudness(mp4_path: Path) -> dict:
    """Measure integrated loudness (LUFS) and dynamic range via ebur128.

    Returns dict with integrated_lufs, loudness_range, true_peak_dbfs.
    """
    cmd = [
        _ffmpeg_bin(),
        "-i", str(mp4_path),
        "-af", "ebur128=framelog=verbose",
        "-f", "null",
        "-",
    ]
    rc, out, err = _run(cmd, timeout=120)
    combined = err + out

    result = {
        "integrated_lufs": None,
        "loudness_range": None,
        "true_peak_dbfs": None,
    }

    m = re.search(r"I:\s+([-\d.]+)\s+LUFS", combined)
    if m:
        try:
            result["integrated_lufs"] = float(m.group(1))
        except ValueError:
            pass

    m = re.search(r"LRA:\s+([\d.]+)\s+LU", combined)
    if m:
        try:
            result["loudness_range"] = float(m.group(1))
        except ValueError:
            pass

    m = re.search(r"True peak:\s+([-\d.]+)\s+dBFS", combined)
    if m:
        try:
            result["true_peak_dbfs"] = float(m.group(1))
        except ValueError:
            pass

    return result


def _detect_black_frames(mp4_path: Path) -> int:
    """Count black frames (>0.5s of black) using blackdetect filter.

    Returns number of black frame segments detected.
    """
    cmd = [
        _ffmpeg_bin(),
        "-i", str(mp4_path),
        "-vf", "blackdetect=d=0.5:pix_th=0.10",
        "-f", "null",
        "-",
    ]
    rc, out, err = _run(cmd, timeout=120)
    combined = err + out
    count = len(re.findall(r"black_start:", combined))
    return count


# ─────────────────────────────────────────────────────────────────────────────
# Script analysis helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_script(script_json: Path) -> dict:
    """Load script JSON. Returns empty dict on failure."""
    if not script_json or not script_json.exists():
        return {}
    try:
        return json.loads(script_json.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("Could not load script JSON: %s", exc)
        return {}


def _analyze_script(script: dict) -> dict:
    """Extract quality-relevant metrics from script dict.

    Returns:
      total_scenes, graphic_kind_distribution, banned_template_count,
      graphic_density_ratio, has_title_card, has_lower_thirds,
      avg_narration_words, diversity_score
    """
    scenes = script.get("scenes", [])
    if not scenes and isinstance(script, list):
        scenes = script

    total = len(scenes)
    if total == 0:
        return {
            "total_scenes": 0,
            "graphic_kind_distribution": {},
            "banned_template_count": 0,
            "graphic_density_ratio": 0.0,
            "has_title_card": False,
            "has_lower_thirds": False,
            "avg_narration_words": 0.0,
            "diversity_score": 0.0,
        }

    kind_counts: dict[str, int] = {}
    banned_count = 0
    has_title_card = False
    has_lower_thirds = False
    narration_word_counts: list[int] = []

    for sc in scenes:
        if isinstance(sc, dict):
            gk = (sc.get("graphic_kind") or "").strip()
        else:
            gk = ""

        if gk:
            kind_counts[gk] = kind_counts.get(gk, 0) + 1
            if gk in _BANNED_TEMPLATES:
                banned_count += 1
            if gk == "title_card":
                has_title_card = True
            if gk == "lower_third":
                has_lower_thirds = True

        narration = ""
        if isinstance(sc, dict):
            narration = sc.get("narration", "") or sc.get("script", "") or ""
        if narration:
            narration_word_counts.append(len(narration.split()))

    scenes_with_graphics = sum(kind_counts.values())
    graphic_density_ratio = scenes_with_graphics / total if total > 0 else 0.0

    # Diversity score: how many unique graphic kinds / total graphic scenes
    unique_kinds = len(kind_counts)
    diversity_score = (
        unique_kinds / scenes_with_graphics if scenes_with_graphics > 0 else 0.0
    )

    avg_narration_words = (
        sum(narration_word_counts) / len(narration_word_counts)
        if narration_word_counts else 0.0
    )

    return {
        "total_scenes": total,
        "graphic_kind_distribution": kind_counts,
        "banned_template_count": banned_count,
        "graphic_density_ratio": round(graphic_density_ratio, 3),
        "has_title_card": has_title_card,
        "has_lower_thirds": has_lower_thirds,
        "avg_narration_words": round(avg_narration_words, 1),
        "diversity_score": round(diversity_score, 3),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Scoring helpers (0–10 per dimension)
# ─────────────────────────────────────────────────────────────────────────────

def _clamp(v: float, lo: float = 0.0, hi: float = 10.0) -> float:
    return max(lo, min(hi, v))


def _score_footage_relevance(script_analysis: dict, scene_changes: list[float], duration_s: float) -> float:
    """Score footage relevance: check scene diversity and coverage.

    Heuristic: more scene changes (up to a point) = more footage variety.
    Target ~10-20 cuts/min for documentary; too few or too many is bad.
    """
    if duration_s <= 0:
        return 5.0
    cuts_per_min = (len(scene_changes) / duration_s) * 60
    # Target range: 6–18 cuts/min for documentary
    if 6 <= cuts_per_min <= 18:
        return 8.0
    elif 3 <= cuts_per_min < 6 or 18 < cuts_per_min <= 25:
        return 6.0
    elif cuts_per_min > 25:
        return 4.0  # frantic
    else:
        return 3.0  # barely any cuts


def _score_scene_script_match(script_analysis: dict) -> float:
    """Score how well graphic kinds match the narration content.

    Proxy: diversity of graphic kinds used + absence of banned templates.
    """
    banned = script_analysis.get("banned_template_count", 0)
    diversity = script_analysis.get("diversity_score", 0.0)
    density = script_analysis.get("graphic_density_ratio", 0.0)

    base = 7.0
    # Good diversity (0.5–0.9) is ideal — not every scene same template
    if 0.4 <= diversity <= 0.85:
        base += 1.5
    elif diversity < 0.2:
        base -= 2.0
    # Density: 20–60% of scenes have graphics is ideal
    if 0.2 <= density <= 0.6:
        base += 1.0
    elif density > 0.8:
        base -= 1.0  # too graphic-heavy
    # Penalize banned templates hard
    base -= banned * 2.0
    return _clamp(base)


def _score_documentary_realism(script_analysis: dict, video_stream: dict) -> float:
    """Score how 'documentary real' it feels vs template-y."""
    banned = script_analysis.get("banned_template_count", 0)
    has_lower = script_analysis.get("has_lower_thirds", False)
    diversity = script_analysis.get("diversity_score", 0.0)
    density = script_analysis.get("graphic_density_ratio", 0.0)

    score = 7.0
    if banned == 0:
        score += 1.5
    else:
        score -= banned * 2.5
    if has_lower:
        score += 0.5
    if 0.3 <= diversity <= 0.8:
        score += 1.0
    if density > 0.75:
        score -= 1.0  # too many graphics = template feel
    return _clamp(score)


def _score_pacing(scene_changes: list[float], duration_s: float) -> float:
    """Score pacing based on scene change distribution.

    Good documentary pacing: variable rhythm, not uniform.
    """
    if duration_s <= 0 or len(scene_changes) < 2:
        return 4.0

    cuts_per_min = (len(scene_changes) / duration_s) * 60

    # Ideal range
    if 5 <= cuts_per_min <= 20:
        pacing_score = 8.0
    elif 3 <= cuts_per_min < 5 or 20 < cuts_per_min <= 30:
        pacing_score = 6.0
    else:
        pacing_score = 4.0

    # Check variance — good pacing has variance in shot lengths
    if len(scene_changes) >= 3:
        intervals = [
            scene_changes[i + 1] - scene_changes[i]
            for i in range(len(scene_changes) - 1)
        ]
        avg_interval = sum(intervals) / len(intervals)
        variance = sum((x - avg_interval) ** 2 for x in intervals) / len(intervals)
        cv = (variance ** 0.5) / avg_interval if avg_interval > 0 else 0
        if cv > 0.5:  # high variation = good dynamic pacing
            pacing_score = min(pacing_score + 1.0, 10.0)
        elif cv < 0.2:  # robotic uniform pacing
            pacing_score = max(pacing_score - 1.5, 0.0)

    return _clamp(pacing_score)


def _score_visual_variation(scene_changes: list[float], duration_s: float, script_analysis: dict) -> float:
    """Score visual variation via scene changes + graphic kind diversity."""
    n_scenes = len(scene_changes)
    diversity = script_analysis.get("diversity_score", 0.0)
    density = script_analysis.get("graphic_density_ratio", 0.0)

    if duration_s <= 0:
        return 5.0

    cuts_per_min = (n_scenes / duration_s) * 60
    base = _clamp(cuts_per_min / 2.0)  # 10 c/min → 5.0

    if 0.3 <= diversity <= 0.85:
        base += 2.0
    elif diversity < 0.15:
        base -= 2.0

    if 0.25 <= density <= 0.65:
        base += 1.0
    elif density > 0.8:
        base -= 0.5

    return _clamp(base)


def _score_motion_quality(video_stream: dict, duration_s: float) -> float:
    """Score render motion quality from video stream properties.

    Checks: framerate, resolution, bitrate.
    """
    score = 7.0

    # Check framerate
    fps_str = video_stream.get("r_frame_rate", "0/1")
    try:
        num, den = fps_str.split("/")
        fps = float(num) / float(den) if float(den) > 0 else 0
    except Exception:
        fps = 0.0

    if fps >= 29.97:
        score += 1.0
    elif fps >= 23.976:
        score += 0.5
    elif fps < 20:
        score -= 2.0

    # Check resolution
    width = int(video_stream.get("width", 0) or 0)
    height = int(video_stream.get("height", 0) or 0)
    if width >= 1920 and height >= 1080:
        score += 1.0
    elif width >= 1280 and height >= 720:
        score += 0.5
    elif width > 0 and width < 640:
        score -= 2.0

    # Check bitrate
    bit_rate = int(video_stream.get("bit_rate", 0) or 0)
    if bit_rate >= 4_000_000:
        score += 0.5
    elif 0 < bit_rate < 1_000_000:
        score -= 1.0

    return _clamp(score)


def _score_graphics_quality(script_analysis: dict) -> float:
    """Score graphics quality via kind distribution and absence of banned types."""
    banned = script_analysis.get("banned_template_count", 0)
    diversity = script_analysis.get("diversity_score", 0.0)
    has_title = script_analysis.get("has_title_card", False)
    has_lower = script_analysis.get("has_lower_thirds", False)
    dist = script_analysis.get("graphic_kind_distribution", {})

    score = 6.0
    if banned == 0:
        score += 2.0
    else:
        score -= banned * 2.5

    if 0.3 <= diversity <= 0.85:
        score += 1.0
    elif diversity < 0.1:
        score -= 1.0

    if has_title:
        score += 0.5
    if has_lower:
        score += 0.5

    return _clamp(score)


def _score_text_restraint(script_analysis: dict) -> float:
    """Score text restraint: too many graphics on every scene = bad."""
    density = script_analysis.get("graphic_density_ratio", 0.0)
    dist = script_analysis.get("graphic_kind_distribution", {})

    # Penalize over-reliance on any single heavy template
    max_single = max(dist.values()) if dist else 0
    total_scenes = script_analysis.get("total_scenes", 1) or 1
    single_dominance = max_single / total_scenes

    score = 8.0
    if density <= 0.5:
        score += 1.0
    elif 0.5 < density <= 0.7:
        pass
    else:
        score -= 2.0

    if single_dominance > 0.4:
        score -= 2.0  # one template dominates
    elif single_dominance <= 0.2:
        score += 0.5

    return _clamp(score)


def _score_audio_mix(loudness: dict) -> float:
    """Score audio mix quality based on integrated LUFS and dynamic range.

    Target: -16 to -12 LUFS integrated, LRA 8–12 LU, true peak < -1 dBFS.
    """
    lufs = loudness.get("integrated_lufs")
    lra = loudness.get("loudness_range")
    peak = loudness.get("true_peak_dbfs")

    if lufs is None:
        return 5.0  # no data

    score = 5.0

    # Integrated loudness: target -16 to -12 LUFS (YouTube normalizes to -14)
    if -16 <= lufs <= -11:
        score += 3.0
    elif -20 <= lufs < -16 or -11 < lufs <= -8:
        score += 1.5
    elif lufs > -6:
        score -= 1.0  # clipping risk
    elif lufs < -24:
        score -= 1.0  # too quiet

    # Dynamic range: 7–14 LU is ideal
    if lra is not None:
        if 7 <= lra <= 14:
            score += 1.5
        elif lra < 5:
            score -= 0.5  # over-compressed
        elif lra > 18:
            score -= 0.5  # too dynamic

    # True peak: should be below -1 dBFS
    if peak is not None:
        if peak <= -1.0:
            score += 0.5
        elif peak > 0.0:
            score -= 2.0  # clipping

    return _clamp(score)


def _score_sfx_restraint(script_analysis: dict) -> float:
    """Score SFX restraint based on sfx_cue graphic kind usage.

    Too many sfx_cue entries = heavy-handed; 0 or a few = restrained.
    """
    dist = script_analysis.get("graphic_kind_distribution", {})
    total_scenes = script_analysis.get("total_scenes", 1) or 1
    sfx_count = dist.get("sfx_cue", 0)
    sfx_ratio = sfx_count / total_scenes

    if sfx_ratio == 0:
        return 9.0
    elif sfx_ratio <= 0.1:
        return 8.0
    elif sfx_ratio <= 0.2:
        return 6.0
    elif sfx_ratio <= 0.35:
        return 4.0
    else:
        return 2.0


def _score_music_psychology(loudness: dict, duration_s: float) -> float:
    """Score music psychology proxy via LRA and overall audio balance.

    High LRA + reasonable LUFS suggests dynamic music bed (good).
    Flat loudness suggests boring static music or none.
    """
    lufs = loudness.get("integrated_lufs")
    lra = loudness.get("loudness_range")

    if lufs is None:
        return 5.0

    score = 6.0
    if lra is not None:
        if lra >= 8:  # dynamic music bed
            score += 2.0
        elif lra >= 5:
            score += 1.0
        else:
            score -= 1.0

    # Reasonable loudness range suggests mixed voice + music
    if lufs is not None and -18 <= lufs <= -10:
        score += 1.5
    elif lufs is not None and lufs > -8:
        score -= 0.5

    return _clamp(score)


def _score_no_template_feel(script_analysis: dict) -> float:
    """Score anti-template feel.

    Penalizes banned templates, single-kind dominance, and extreme graphic density.
    """
    banned = script_analysis.get("banned_template_count", 0)
    diversity = script_analysis.get("diversity_score", 0.0)
    density = script_analysis.get("graphic_density_ratio", 0.0)
    dist = script_analysis.get("graphic_kind_distribution", {})
    total = script_analysis.get("total_scenes", 1) or 1
    max_single = max(dist.values()) if dist else 0
    dominance = max_single / total

    score = 8.0

    if banned == 0:
        score += 1.0
    else:
        score -= banned * 3.0

    if dominance <= 0.25:
        score += 1.0
    elif dominance > 0.5:
        score -= 2.0

    if density > 0.75:
        score -= 1.5
    elif density <= 0.5:
        score += 0.5

    return _clamp(score)


def _score_render_reliability(black_frames: int, duration_s: float, video_stream: dict) -> float:
    """Score render reliability based on black frames and stream integrity."""
    score = 9.0

    if duration_s > 0 and black_frames > 3:
        score -= min(black_frames * 0.5, 4.0)

    width = int(video_stream.get("width", 0) or 0)
    height = int(video_stream.get("height", 0) or 0)
    codec = video_stream.get("codec_name", "")
    if width == 0 or height == 0:
        score -= 3.0
    if codec not in ("h264", "hevc", "vp9", "av1", ""):
        score -= 1.0

    return _clamp(score)


def _score_cost_efficiency(duration_s: float, script_analysis: dict) -> float:
    """Score cost efficiency: reasonable graphics-per-scene density.

    More graphics = higher render cost. Ideal is 20–50% graphic density
    with good coverage. Very low density wastes potential; very high = expensive.
    """
    density = script_analysis.get("graphic_density_ratio", 0.0)
    total = script_analysis.get("total_scenes", 0)

    if total == 0:
        return 5.0

    if 0.2 <= density <= 0.5:
        return 9.0
    elif 0.1 <= density < 0.2 or 0.5 < density <= 0.65:
        return 7.5
    elif density > 0.75:
        return 5.0  # expensive but maybe warranted
    else:
        return 6.0


# ─────────────────────────────────────────────────────────────────────────────
# Main scorer
# ─────────────────────────────────────────────────────────────────────────────

def score_vidlore_render(mp4_path: Path, script_json: Path, work_dir: Path) -> dict:
    """Score a Vidlore render across 15 quality dimensions.

    Uses ffprobe for: duration, shot detection (scene change filter),
    audio levels (ebur128), black frame detection.
    Uses script.json for: graphic_kind distribution, banned template check,
    density ratio.

    Returns dict with all 15 scores + metadata.
    """
    mp4_path = Path(mp4_path)
    work_dir = Path(work_dir)

    if not mp4_path.exists():
        raise FileNotFoundError(f"MP4 not found: {mp4_path}")

    log.info("Scoring render: %s", mp4_path.name)

    # ── ffprobe measurements ──────────────────────────────────────────────────
    duration_s = _probe_duration(mp4_path)
    video_stream = _probe_video_stream(mp4_path)

    log.info("Detecting scene changes (this may take a moment)…")
    scene_changes = _detect_scene_changes(mp4_path)

    log.info("Measuring audio loudness…")
    loudness = _measure_audio_loudness(mp4_path)

    log.info("Detecting black frames…")
    black_frames = _detect_black_frames(mp4_path)

    # ── Script analysis ───────────────────────────────────────────────────────
    script = _load_script(script_json)
    script_analysis = _analyze_script(script)

    log.info(
        "Render stats: duration=%.1fs, scene_changes=%d, black_frames=%d, "
        "integrated_lufs=%s, total_scenes=%d, banned_templates=%d",
        duration_s, len(scene_changes), black_frames,
        loudness.get("integrated_lufs"), script_analysis["total_scenes"],
        script_analysis["banned_template_count"],
    )

    # ── Score all 15 dimensions ───────────────────────────────────────────────
    scores: dict[str, float] = {}

    scores["footage_relevance"] = _score_footage_relevance(
        script_analysis, scene_changes, duration_s
    )
    scores["scene_script_match"] = _score_scene_script_match(script_analysis)
    scores["documentary_realism"] = _score_documentary_realism(
        script_analysis, video_stream
    )
    scores["pacing"] = _score_pacing(scene_changes, duration_s)
    scores["visual_variation"] = _score_visual_variation(
        scene_changes, duration_s, script_analysis
    )
    scores["motion_quality"] = _score_motion_quality(video_stream, duration_s)
    scores["graphics_quality"] = _score_graphics_quality(script_analysis)
    scores["text_restraint"] = _score_text_restraint(script_analysis)
    scores["audio_mix"] = _score_audio_mix(loudness)
    scores["sfx_restraint"] = _score_sfx_restraint(script_analysis)
    scores["music_psychology"] = _score_music_psychology(loudness, duration_s)
    scores["no_template_feel"] = _score_no_template_feel(script_analysis)
    scores["render_reliability"] = _score_render_reliability(
        black_frames, duration_s, video_stream
    )
    scores["cost_efficiency"] = _score_cost_efficiency(duration_s, script_analysis)

    # overall_quality: weighted average (realism, pacing, anti-template weighted higher)
    weights = {
        "footage_relevance": 1.0,
        "scene_script_match": 1.5,
        "documentary_realism": 2.0,
        "pacing": 1.5,
        "visual_variation": 1.0,
        "motion_quality": 1.0,
        "graphics_quality": 1.5,
        "text_restraint": 1.0,
        "audio_mix": 1.0,
        "sfx_restraint": 0.5,
        "music_psychology": 1.0,
        "no_template_feel": 2.0,
        "render_reliability": 1.5,
        "cost_efficiency": 0.5,
    }
    total_weight = sum(weights.values())
    weighted_sum = sum(scores[k] * w for k, w in weights.items())
    scores["overall_quality"] = round(weighted_sum / total_weight, 2)

    # Round all scores to 2 dp
    scores = {k: round(v, 2) for k, v in scores.items()}

    result: dict[str, Any] = {
        "mp4": str(mp4_path),
        "script_json": str(script_json) if script_json else None,
        "scored_at": datetime.now(timezone.utc).isoformat(),
        "duration_s": duration_s,
        "scene_changes_detected": len(scene_changes),
        "black_frames_detected": black_frames,
        "audio": loudness,
        "video_stream": {
            "codec": video_stream.get("codec_name", ""),
            "width": int(video_stream.get("width", 0) or 0),
            "height": int(video_stream.get("height", 0) or 0),
            "fps": video_stream.get("r_frame_rate", ""),
            "bit_rate": int(video_stream.get("bit_rate", 0) or 0),
        },
        "script_analysis": script_analysis,
        "scores": scores,
    }

    # Save score report alongside the mp4
    report_path = mp4_path.with_suffix(".score.json")
    try:
        report_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        log.info("Score report saved → %s", report_path)
    except Exception as exc:
        log.warning("Could not save score report: %s", exc)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Vidlore research engine — analyze YouTube docs and score renders"
    )
    p.add_argument("--analyze", metavar="VIDEO_ID", help="YouTube video ID to analyze")
    p.add_argument("--score", metavar="MP4_PATH", help="Path to mp4 to score")
    p.add_argument("--script", metavar="SCRIPT_JSON", help="Path to script.json for scoring")
    p.add_argument(
        "--work-dir", metavar="WORK_DIR",
        default=str(_PROJECT_ROOT / "research"),
        help="Working directory for score output (default: research/)",
    )
    p.add_argument(
        "--update-library", action="store_true",
        help="After analysis, extract patterns and update DNA library",
    )
    p.add_argument(
        "--model", default=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
        help="Anthropic model to use for analysis (default: claude-sonnet-4-6)",
    )
    args = p.parse_args()

    if not args.analyze and not args.score:
        p.print_help()
        sys.exit(0)

    # Load config from environment / .env
    try:
        sys.path.insert(0, str(_PROJECT_ROOT))
        from vidlore.config import load_config as _load_cfg  # type: ignore
        _cfg_obj = _load_cfg(_PROJECT_ROOT)
        cfg = {
            "anthropic_api_key": _cfg_obj.anthropic_api_key,
            "anthropic_model": args.model or _cfg_obj.anthropic_model,
        }
    except Exception:
        cfg = {
            "anthropic_api_key": os.environ.get("ANTHROPIC_API_KEY", ""),
            "anthropic_model": args.model or "claude-sonnet-4-6",
        }

    if args.analyze:
        log.info("=== Analyzing: %s ===", args.analyze)
        notes = analyze_youtube_video(args.analyze, cfg)
        print(json.dumps(notes, indent=2, ensure_ascii=False))

        if args.update_library:
            patterns = extract_editing_patterns(notes)
            update_dna_library(patterns, _DNA_LIBRARY)
            log.info("DNA library updated: %s", _DNA_LIBRARY)

    if args.score:
        mp4 = Path(args.score)
        script = Path(args.script) if args.script else Path("/dev/null")
        work = Path(args.work_dir)
        log.info("=== Scoring: %s ===", mp4.name)
        result = score_vidlore_render(mp4, script, work)
        scores = result["scores"]
        print("\n── Vidlore Render Scores ──")
        dims = [
            "footage_relevance", "scene_script_match", "documentary_realism",
            "pacing", "visual_variation", "motion_quality", "graphics_quality",
            "text_restraint", "audio_mix", "sfx_restraint", "music_psychology",
            "no_template_feel", "render_reliability", "cost_efficiency",
            "overall_quality",
        ]
        for dim in dims:
            val = scores.get(dim, "N/A")
            bar = "█" * int(round(float(val))) if isinstance(val, (int, float)) else ""
            label = "★" if dim == "overall_quality" else " "
            print(f"  {label} {dim:<25} {val:>5}  {bar}")
        print()
        print(json.dumps(result, indent=2, ensure_ascii=False))
