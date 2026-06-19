"""Vidlore Benchmark Engine — score a rendered MP4 across 15 quality
dimensions and track improvement trends across versions.

Usage (CLI):
    python tools/benchmark_engine.py score  <mp4> <script.json> [work_dir]
    python tools/benchmark_engine.py compare <slug>
    python tools/benchmark_engine.py sheet   <mp4> [output.png]

Usage (module):
    from tools.benchmark_engine import score_render, compare_versions
"""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

# ── Paths ────────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
_SCORES_FILE = _ROOT / "research" / "benchmark_scores.json"
_FFMPEG = (
    _ROOT / ".venv" / "lib" / "python3.9" / "site-packages"
    / "imageio_ffmpeg" / "binaries" / "ffmpeg-macos-aarch64-v7.1"
)

BANNED_TEMPLATES: set[str] = {"era_banner", "comparison", "stat_dashboard"}

# Weights for overall_quality (must sum to 1.0)
_WEIGHTS = {
    "footage_relevance":    0.15,
    "scene_script_match":   0.08,
    "documentary_realism":  0.15,
    "pacing":               0.10,
    "visual_variation":     0.07,
    "motion_quality":       0.05,
    "graphics_quality":     0.06,
    "text_restraint":       0.07,
    "audio_mix":            0.10,
    "sfx_restraint":        0.03,
    "music_psychology":     0.04,
    "no_template_feel":     0.05,
    "render_reliability":   0.05,
    "cost_efficiency":      0.00,  # informational only, excluded from weighted mean
}


# ── ffmpeg / ffprobe helpers ─────────────────────────────────────────────────

def _ffmpeg_exe() -> str:
    if _FFMPEG.exists():
        return str(_FFMPEG)
    import shutil
    sys_ff = shutil.which("ffmpeg")
    if sys_ff:
        return sys_ff
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        raise RuntimeError("Cannot locate ffmpeg binary")


def _ffprobe(mp4: Path) -> dict:
    """Return ffprobe JSON for the mp4."""
    ff = _ffmpeg_exe().replace("ffmpeg", "ffprobe")
    # If ffprobe doesn't exist alongside ffmpeg, use ffmpeg -i as fallback
    import shutil
    probe_bin = shutil.which("ffprobe") or ff
    if not Path(probe_bin).exists():
        probe_bin = shutil.which("ffprobe") or "ffprobe"
    try:
        r = subprocess.run(
            [probe_bin, "-v", "quiet", "-print_format", "json",
             "-show_streams", "-show_format", str(mp4)],
            capture_output=True, text=True, timeout=30,
        )
        if r.returncode == 0 and r.stdout.strip():
            return json.loads(r.stdout)
    except Exception:
        pass
    # FALLBACK: ffprobe is not bundled with imageio-ffmpeg, so on this
    # machine it is usually absent — parse `ffmpeg -i` stderr for the
    # container Duration instead. Without this the probe returned {} and
    # _video_duration -> 0.0, which silently TANKED the pacing score
    # (cuts/min became inf/0). Caught in the 2026-05-29 cumulative audit.
    return _ffmpeg_probe_fallback(mp4)


def _ffmpeg_probe_fallback(mp4: Path) -> dict:
    """Build a minimal probe dict ({'format': {'duration': s}}) by parsing
    `ffmpeg -i` stderr — used when the ffprobe binary is unavailable."""
    try:
        r = subprocess.run([_ffmpeg_exe(), "-hide_banner", "-i", str(mp4)],
                           capture_output=True, text=True, timeout=30)
        m = re.search(r"Duration:\s*(\d+):(\d+):([\d.]+)", r.stderr)
        if m:
            secs = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + \
                float(m.group(3))
            return {"format": {"duration": secs}, "streams": []}
    except Exception:                                      # noqa: BLE001
        pass
    return {}


def _ebur128(mp4: Path) -> dict:
    """Run EBU R128 loudness analysis; return integrated LUFS, true peak dB."""
    ff = _ffmpeg_exe()
    try:
        r = subprocess.run(
            [ff, "-hide_banner", "-i", str(mp4),
             "-af", "ebur128=peak=true", "-f", "null", "-"],
            capture_output=True, text=True, timeout=120,
        )
        out = r.stderr
        # ebur128 streams PER-FRAME readings (the first 'I:' is the gate
        # floor ~-70 LUFS at t=0 before audio ramps) and then a final
        # 'Summary:' block with the real INTEGRATED loudness + true peak.
        # We must take the LAST match (the summary), not the first — the
        # old re.search grabbed the -70 gate floor and tanked audio_mix.
        lufs = _parse_float_last(r"I:\s+([-\d.]+)\s+LUFS", out)
        # summary true-peak label is 'Peak:  -1.9 dBFS' (under 'True peak:');
        # per-frame is 'FTPK'. Take the last 'Peak:' (the summary value).
        peak = _parse_float_last(r"Peak:\s+([-\d.]+)\s+dBFS", out)
        return {"lufs": lufs, "true_peak_db": peak}
    except Exception:
        return {"lufs": None, "true_peak_db": None}


def _parse_float(pattern: str, text: str) -> Optional[float]:
    m = re.search(pattern, text)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return None


def _parse_float_last(pattern: str, text: str) -> Optional[float]:
    """Return the LAST regex-captured float (used for ffmpeg filters that
    stream per-frame readings then a final summary — we want the summary)."""
    matches = re.findall(pattern, text)
    if matches:
        try:
            return float(matches[-1])
        except (ValueError, TypeError):
            pass
    return None


def _video_duration(probe: dict) -> float:
    """Return video duration in seconds from probe data."""
    try:
        fmt = probe.get("format", {})
        dur = float(fmt.get("duration", 0))
        if dur > 0:
            return dur
    except Exception:
        pass
    for s in probe.get("streams", []):
        try:
            d = float(s.get("duration", 0))
            if d > 0:
                return d
        except Exception:
            pass
    return 0.0


# ── Script analysis helpers ──────────────────────────────────────────────────

def _load_script(script_json: Path) -> dict:
    if not script_json or not script_json.exists():
        return {}
    try:
        data = json.loads(script_json.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def _scenes(script: dict) -> list[dict]:
    return script.get("scenes", [])


# ── Individual scorers ───────────────────────────────────────────────────────

def _score_footage_relevance(scenes: list[dict]) -> float:
    """Match graphic_kind against narration keyword overlap."""
    if not scenes:
        return 5.0
    hits = 0
    for sc in scenes:
        kind = sc.get("graphic_kind", "")
        narration = sc.get("narration", "").lower()
        keywords = [k.lower() for k in sc.get("keywords", [])]
        visual = sc.get("visual", "").lower()
        # A scene with a graphic has text that should relate to narration
        if kind:
            # Check if graphic_text words appear in narration or keywords
            gt = sc.get("graphic_text", "").lower()
            gt_words = set(re.findall(r"\w+", gt)) - {"the", "a", "an", "of", "in"}
            nar_words = set(re.findall(r"\w+", narration))
            kw_words = set(keywords)
            if gt_words & (nar_words | kw_words):
                hits += 1
            elif gt_words:
                hits += 0.5
            else:
                hits += 0.7  # graphic exists but no text — neutral
        else:
            # Non-graphic scene: check keywords vs visual description
            kw_set = set(re.findall(r"\w+", " ".join(keywords)))
            vis_set = set(re.findall(r"\w+", visual))
            nar_set = set(re.findall(r"\w+", narration))
            overlap = kw_set & (vis_set | nar_set)
            score_frac = min(len(overlap) / max(len(kw_set), 1), 1.0)
            hits += 0.5 + score_frac * 0.5
    raw = hits / len(scenes)
    return round(min(raw * 10, 10.0), 2)


def _score_scene_script_match(scenes: list[dict], script: dict) -> float:
    """How well scenes cover expected narrative beats."""
    if not scenes:
        return 5.0
    expected_roles = {"hook", "problem", "evidence", "escalation",
                      "reveal", "climax", "resolution"}
    found_roles = {sc.get("role", "").lower() for sc in scenes}
    coverage = len(expected_roles & found_roles) / len(expected_roles)
    # Also check narration length variance — very short scenes indicate gaps
    nar_lengths = [len(sc.get("narration", "").split()) for sc in scenes]
    avg_len = sum(nar_lengths) / max(len(nar_lengths), 1)
    length_score = min(avg_len / 30.0, 1.0)  # target 30+ words/scene
    score = (coverage * 0.6 + length_score * 0.4) * 10
    return round(min(score, 10.0), 2)


def _score_documentary_realism(scenes: list[dict]) -> float:
    """Penalise if most scenes are templated; reward real-feel visual language."""
    if not scenes:
        return 5.0
    real_shot_types = {"establishing", "aerial", "detail", "archival",
                       "tracking", "portrait", "wide", "macro", "reaction"}
    graphic_count = sum(1 for sc in scenes if sc.get("graphic_kind"))
    real_count = sum(1 for sc in scenes
                     if sc.get("shot_type", "").lower() in real_shot_types)
    graphic_ratio = graphic_count / len(scenes)
    real_ratio = real_count / len(scenes)
    # Penalise if >60% scenes are graphics (template feel)
    graphic_penalty = max(0, graphic_ratio - 0.6) * 20
    score = (real_ratio * 7 + (1 - graphic_ratio) * 3) - graphic_penalty
    return round(min(max(score, 0), 10.0), 2)


def _load_render_meta(mp4: Path, work_dir: Optional[Path]) -> dict:
    """MNT_1 — load render_meta.json (the pipeline's authoritative beat/pacing
    numbers) from beside the MP4 or in work_dir. {} when absent (older renders
    fall back to ffmpeg scene-detection)."""
    for cand in [mp4.with_name("render_meta.json"),
                 (Path(work_dir) / "render_meta.json") if work_dir else None]:
        try:
            if cand and cand.is_file():
                return json.loads(cand.read_text(encoding="utf-8"))
        except Exception:                                      # noqa: BLE001
            pass
    return {}


def _count_cuts(mp4: Path, video_duration: float) -> int:
    """Count REAL visual cuts via ffmpeg scene-change detection. Vidlore
    cuts each narration scene into several beats, so the script-scene count
    massively under-counts on-screen shots — measuring the rendered video
    directly is the only honest pacing signal. Returns 0 if unavailable."""
    try:
        r = subprocess.run(
            [_ffmpeg_exe(), "-hide_banner", "-i", str(mp4),
             "-filter_complex", "select='gt(scene,0.35)',metadata=print",
             "-an", "-f", "null", "-"],
            capture_output=True, text=True, timeout=240)
        cuts = len(re.findall(r"lavfi\.scene_score", r.stderr))
        if not cuts:
            cuts = len(re.findall(r"pts_time:", r.stderr))
        return cuts
    except Exception:                                      # noqa: BLE001
        return 0


def _score_pacing(scenes: list[dict], video_duration: float,
                  cut_count: int = 0, beats: int = 0) -> float:
    """Avg shot duration variance. Documentary target: 3-6s avg.
    Shot-count priority (MNT_1): the render's OWN beat count (render_meta.json,
    ground truth) > ffmpeg scene-detection cut_count > script-scene count.
    The script-scene count is a poor proxy that scored a briskly-cut video as
    glacial; ffmpeg scene-detection under-counts camera-move beats."""
    if not scenes or video_duration <= 0:
        return 5.0
    if beats and beats >= len(scenes):
        n = beats                                  # ground-truth beat count
    elif cut_count >= len(scenes):
        n = cut_count + 1                          # cuts are transitions -> +1
    else:
        n = len(scenes)
    avg_shot = video_duration / n
    # Target band: 3-6s
    if 3.0 <= avg_shot <= 6.0:
        base = 10.0
    elif avg_shot < 3.0:
        # Too fast — each second below 3s costs 1.5 pts
        base = max(10.0 - (3.0 - avg_shot) * 1.5, 0.0)
    else:
        # Too slow — each second above 6s costs 1.0 pts
        base = max(10.0 - (avg_shot - 6.0) * 1.0, 2.0)
    # Intensity variance: check spread of intensities
    intensities = [sc.get("intensity", 0) for sc in scenes if sc.get("intensity")]
    if len(intensities) >= 3:
        mean_i = sum(intensities) / len(intensities)
        variance = sum((i - mean_i) ** 2 for i in intensities) / len(intensities)
        var_bonus = min(math.sqrt(variance), 2.0)  # up to +2 for good variation
        base = min(base + var_bonus * 0.3, 10.0)
    return round(base, 2)


def _score_visual_variation(scenes: list[dict]) -> float:
    """Diversity of shot types; penalise monotone repetition."""
    if not scenes:
        return 5.0
    shot_types = [sc.get("shot_type", "").lower() for sc in scenes if sc.get("shot_type")]
    if not shot_types:
        return 4.0
    unique_types = len(set(shot_types))
    total = len(shot_types)
    diversity_ratio = unique_types / min(total, 8)  # cap denominator at 8
    # Penalise consecutive same shots
    consecutive_penalty = 0
    for i in range(1, len(shot_types)):
        if shot_types[i] == shot_types[i - 1]:
            consecutive_penalty += 0.5
    score = diversity_ratio * 10 - consecutive_penalty
    return round(min(max(score, 0), 10.0), 2)


def _score_motion_quality(scenes: list[dict]) -> float:
    """Presence of motivated camera moves in visual descriptions."""
    if not scenes:
        return 5.0
    motion_words = {
        "push-in", "pull-back", "tracking", "pan", "tilt", "dolly",
        "crane", "handheld", "steadicam", "zoom", "orbit", "arc",
        "whip", "slow zoom", "push in", "pull back",
    }
    hits = 0
    for sc in scenes:
        visual = sc.get("visual", "").lower()
        if any(mw in visual for mw in motion_words):
            hits += 1
    ratio = hits / len(scenes)
    # Documentary ideal: 30-60% of scenes have motivated moves
    if 0.3 <= ratio <= 0.6:
        score = 10.0
    elif ratio < 0.3:
        score = max(ratio / 0.3 * 10, 3.0)
    else:
        # Too many moves = chaotic
        score = max(10.0 - (ratio - 0.6) * 15, 5.0)
    return round(score, 2)


def _score_graphics_quality(scenes: list[dict]) -> float:
    """Any curated (non-banned) template is a legitimate documentary device;
    only the banned full-screen-infographic types subtract. The old hardcoded
    5-name whitelist ({location,stat,number,label,document}) predated the
    100+ template registry, so real cards like currency_stat / typing_date /
    figure_locator scored ZERO and tanked this dim — a benchmark staleness
    bug caught in the 2026-05-29 cumulative audit. A small bonus rewards
    VARIETY (distinct kinds) so a wall of one repeated card scores lower."""
    graphic_scenes = [sc for sc in scenes if sc.get("graphic_kind")]
    if not graphic_scenes:
        return 7.0  # No graphics — neutral, not penalised
    kinds = [sc.get("graphic_kind", "").lower() for sc in graphic_scenes]
    banned_count = sum(1 for k in kinds if k in BANNED_TEMPLATES)
    good_count = len(graphic_scenes) - banned_count
    ratio_good = good_count / len(graphic_scenes)
    variety_bonus = min(len(set(kinds)) / len(graphic_scenes), 1.0) * 1.0
    score = ratio_good * 9 + variety_bonus - banned_count * 3
    return round(min(max(score, 0), 10.0), 2)


def _score_text_restraint(scenes: list[dict]) -> float:
    """Graphic density ratio; target max 35% of scenes."""
    if not scenes:
        return 8.0
    graphic_count = sum(1 for sc in scenes if sc.get("graphic_kind"))
    ratio = graphic_count / len(scenes)
    if ratio <= 0.35:
        score = 10.0
    else:
        # Each 10pp above 35% costs 2 pts
        score = max(10.0 - (ratio - 0.35) * 20, 0.0)
    return round(score, 2)


def _score_audio_mix(lufs: Optional[float], peak_db: Optional[float]) -> float:
    """Score EBU R128 result. Target: -16 LUFS, no clip (peak < 0 dBFS)."""
    if lufs is None:
        return 5.0  # unknown
    target_lufs = -16.0
    lufs_delta = abs(lufs - target_lufs)
    if lufs_delta <= 1.0:
        lufs_score = 10.0
    elif lufs_delta <= 3.0:
        lufs_score = 8.0
    elif lufs_delta <= 6.0:
        lufs_score = 6.0
    else:
        lufs_score = max(10.0 - lufs_delta, 2.0)
    # Penalise true peak clipping
    peak_penalty = 0.0
    if peak_db is not None and peak_db >= 0.0:
        peak_penalty = 3.0 + peak_db  # clipping is a serious problem
    score = lufs_score - peak_penalty
    return round(min(max(score, 0), 10.0), 2)


def _score_sfx_restraint(work_dir: Optional[Path]) -> float:
    """Infer SFX usage from render logs if available."""
    if not work_dir or not work_dir.exists():
        return 7.0
    # Look for any sfx-related log markers
    log_files = list(work_dir.glob("*.log")) + list(work_dir.glob("*.txt"))
    sfx_count = 0
    for lf in log_files:
        try:
            text = lf.read_text(encoding="utf-8", errors="ignore")
            sfx_count += len(re.findall(r"\bsfx\b|\bsound effect\b", text, re.I))
        except Exception:
            pass
    if sfx_count == 0:
        return 8.0  # sfx off — good (matches config default)
    elif sfx_count <= 3:
        return 6.0
    else:
        return max(8.0 - sfx_count * 0.5, 2.0)


def _score_music_psychology(scenes: list[dict]) -> float:
    """Infer music arc from scene intensity distribution."""
    intensities = [sc.get("intensity", 0) for sc in scenes if sc.get("intensity")]
    if len(intensities) < 3:
        return 6.0
    n = len(intensities)
    # Good documentary arc: low start, rise, peak in last third, resolution
    first_third = intensities[: n // 3]
    last_third = intensities[-(n // 3) :]
    mid_section = intensities[n // 3 : -(n // 3)] or intensities
    avg_start = sum(first_third) / len(first_third)
    avg_mid = sum(mid_section) / len(mid_section)
    avg_end = sum(last_third) / len(last_third)
    # Good arc: start low, mid higher, end resolves
    arc_ok = (avg_start < avg_mid) or (avg_mid > avg_end)
    # Peak check: highest intensity is in middle/late
    peak_idx = intensities.index(max(intensities))
    peak_in_good_zone = peak_idx > n // 3
    score = 5.0
    if arc_ok:
        score += 2.5
    if peak_in_good_zone:
        score += 2.0
    # Bonus for actual dynamic range
    dyn_range = max(intensities) - min(intensities)
    score += min(dyn_range * 0.25, 0.5)
    return round(min(score, 10.0), 2)


def _score_no_template_feel(scenes: list[dict]) -> float:
    """0 banned cards = 10/10; each banned card = -3."""
    banned_count = sum(
        1 for sc in scenes
        if sc.get("graphic_kind", "").lower() in BANNED_TEMPLATES
    )
    score = max(10.0 - banned_count * 3, 0.0)
    return round(score, 2)


def _score_render_reliability(mp4: Path, work_dir: Optional[Path]) -> float:
    """Completed without error = 10; crash indicators = 0."""
    if not mp4.exists() or mp4.stat().st_size < 1024:
        return 0.0
    # Check for crash log
    if work_dir and work_dir.exists():
        fail_log = work_dir / "last_ffmpeg_fail.txt"
        if fail_log.exists() and fail_log.stat().st_size > 0:
            return 2.0  # render happened but had ffmpeg errors
    return 10.0


def _score_cost_efficiency(work_dir: Optional[Path], render_time_s: float):
    """Based on work_dir file count / render time (informational). Returns
    None (UNMEASURED) when render time is unknown — e.g. a post-hoc audit — so
    it is EXCLUDED from the weighted overall rather than faking a 5.0 that
    silently drags the score (MNT_3: honest scoring)."""
    if render_time_s <= 0:
        return None
    if not work_dir or not work_dir.exists():
        return None
    try:
        file_count = sum(1 for _ in work_dir.rglob("*") if _.is_file())
        # Lower is better: fewer temp files per second of render = efficient
        ratio = file_count / render_time_s
        if ratio < 2:
            return 10.0
        elif ratio < 5:
            return 8.0
        elif ratio < 10:
            return 6.0
        else:
            return max(10.0 - math.log(ratio, 2), 2.0)
    except Exception:
        return 5.0


def _weighted_overall(scores: dict) -> float:
    # MNT_3 — an UNMEASURED dimension (value None, e.g. cost_efficiency in a
    # post-hoc audit with no render time) is EXCLUDED from the weighted overall
    # so it never silently drags the score; the remaining weights re-normalise.
    def _ok(k):
        return (k in scores and scores[k] is not None and _WEIGHTS[k] > 0)
    total_w = sum(_WEIGHTS[k] for k in _WEIGHTS if _ok(k))
    if total_w == 0:
        return 5.0
    weighted_sum = sum(scores[k] * _WEIGHTS[k] for k in _WEIGHTS if _ok(k))
    return round(weighted_sum / total_w, 3)


# ── Contact sheet ─────────────────────────────────────────────────────────────

def generate_contact_sheet(mp4: Path, output: Path, grid: str = "8x6") -> Path:
    """Extract a grid of frames (default 8 columns x 6 rows = 48 frames)
    and tile them into a PNG contact sheet."""
    cols_s, rows_s = grid.split("x")
    cols, rows = int(cols_s), int(rows_s)
    n_frames = cols * rows
    ff = _ffmpeg_exe()
    output = output if output.is_absolute() else mp4.parent / output
    output.parent.mkdir(parents=True, exist_ok=True)

    # Use ffmpeg's thumbnail filter + tile to build the contact sheet
    vf = (
        f"select='not(mod(n,trunc(nb_frames/{n_frames})))',"
        f"scale=320:180,"
        f"tile={cols}x{rows}"
    )
    try:
        r = subprocess.run(
            [ff, "-hide_banner", "-loglevel", "error", "-y",
             "-i", str(mp4),
             "-vf", vf,
             "-frames:v", "1",
             str(output)],
            capture_output=True, text=True, timeout=120,
        )
        if r.returncode != 0 or not output.exists():
            # Fallback: simpler fps-based selection
            vf_fallback = f"fps=1/10,scale=320:180,tile={cols}x{rows}"
            subprocess.run(
                [ff, "-hide_banner", "-loglevel", "error", "-y",
                 "-i", str(mp4),
                 "-vf", vf_fallback,
                 "-frames:v", "1",
                 str(output)],
                capture_output=True, text=True, timeout=120,
            )
    except Exception as e:
        print(f"[benchmark] contact sheet failed: {e}", file=sys.stderr)
    return output


# ── Persistence ───────────────────────────────────────────────────────────────

def load_scores() -> dict:
    """Load benchmark_scores.json -> dict keyed by slug."""
    if not _SCORES_FILE.exists():
        return {}
    try:
        return json.loads(_SCORES_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_score(score: dict) -> None:
    """Append/update score under its slug in benchmark_scores.json."""
    _SCORES_FILE.parent.mkdir(parents=True, exist_ok=True)
    all_scores = load_scores()
    slug = score.get("slug", "unknown")
    if slug not in all_scores:
        all_scores[slug] = []
    all_scores[slug].append(score)
    _SCORES_FILE.write_text(
        json.dumps(all_scores, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# ── Main scorer ──────────────────────────────────────────────────────────────

def score_render(
    mp4: Path,
    script_json: Path,
    work_dir: Optional[Path] = None,
    render_time_s: float = 0.0,
) -> dict:
    """Score a Vidlore render across 15 quality dimensions.

    Returns a dict with all dimension scores, metadata, and overall_quality.
    Also saves the result to benchmark_scores.json.
    """
    mp4 = Path(mp4)
    script_json = Path(script_json)

    # Derive slug from mp4 stem
    slug = mp4.stem

    print(f"[benchmark] Scoring: {mp4.name}")
    probe = _ffprobe(mp4)
    script = _load_script(script_json)
    scenes = _scenes(script)
    video_dur = _video_duration(probe)

    print(f"[benchmark]   scenes={len(scenes)}, duration={video_dur:.1f}s")

    # Audio analysis
    print("[benchmark]   Running EBU R128 loudness analysis...")
    audio = _ebur128(mp4)
    lufs = audio.get("lufs")
    peak = audio.get("true_peak_db")
    print(f"[benchmark]   LUFS={lufs}, TruePeak={peak}")

    scores = {}
    scores["footage_relevance"]    = _score_footage_relevance(scenes)
    scores["scene_script_match"]   = _score_scene_script_match(scenes, script)
    scores["documentary_realism"]  = _score_documentary_realism(scenes)
    meta = _load_render_meta(mp4, work_dir)
    beats = int(meta.get("beats", 0) or 0)
    if beats:
        cut_count = 0
        print(f"[benchmark]   render_meta beats={beats} "
              f"(~{meta.get('shot_len_s', {}).get('avg', 0)}s/shot, ground truth)")
    else:
        cut_count = _count_cuts(mp4, video_dur)
        print(f"[benchmark]   detected cuts={cut_count} "
              f"(~{video_dur/max(1,cut_count+1):.1f}s/shot; no render_meta)")
    scores["pacing"]               = _score_pacing(scenes, video_dur,
                                                   cut_count, beats)
    scores["visual_variation"]     = _score_visual_variation(scenes)
    scores["motion_quality"]       = _score_motion_quality(scenes)
    scores["graphics_quality"]     = _score_graphics_quality(scenes)
    scores["text_restraint"]       = _score_text_restraint(scenes)
    scores["audio_mix"]            = _score_audio_mix(lufs, peak)
    scores["sfx_restraint"]        = _score_sfx_restraint(work_dir)
    scores["music_psychology"]     = _score_music_psychology(scenes)
    scores["no_template_feel"]     = _score_no_template_feel(scenes)
    scores["render_reliability"]   = _score_render_reliability(mp4, work_dir)
    scores["cost_efficiency"]      = _score_cost_efficiency(work_dir, render_time_s)
    scores["overall_quality"]      = _weighted_overall(scores)

    result = {
        "slug":           slug,
        "mp4":            str(mp4),
        "script_json":    str(script_json),
        "scored_at":      time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "video_duration": video_dur,
        "scene_count":    len(scenes),
        "lufs":           lufs,
        "true_peak_db":   peak,
        "scores":         scores,
    }

    save_score(result)
    print(f"[benchmark]   overall_quality={scores['overall_quality']:.2f}")
    return result


# ── Version comparison ────────────────────────────────────────────────────────

def compare_versions(slug: str) -> dict:
    """Load all benchmark scores for a given slug and show trend."""
    all_scores = load_scores()
    versions = all_scores.get(slug, [])
    if not versions:
        return {"slug": slug, "versions": [], "trend": "no data"}

    print(f"\n[benchmark] Trend for '{slug}' ({len(versions)} renders):\n")
    dims = [
        "footage_relevance", "documentary_realism", "pacing",
        "audio_mix", "no_template_feel", "overall_quality",
    ]
    header = f"{'#':>3}  {'scored_at':>20}  " + "  ".join(f"{d[:10]:>10}" for d in dims)
    print(header)
    print("-" * len(header))
    for i, v in enumerate(versions, 1):
        sc = v.get("scores", {})
        row = f"{i:>3}  {v.get('scored_at', 'unknown'):>20}  "
        row += "  ".join(f"{sc.get(d, 0):>10.2f}" for d in dims)
        print(row)

    # Compute delta from first to last
    if len(versions) >= 2:
        first_sc = versions[0].get("scores", {})
        last_sc = versions[-1].get("scores", {})
        delta = {
            k: round(last_sc.get(k, 0) - first_sc.get(k, 0), 3)
            for k in last_sc
        }
        trend = "improving" if delta.get("overall_quality", 0) > 0 else "declining"
    else:
        delta = {}
        trend = "single render"

    return {
        "slug": slug,
        "versions": versions,
        "delta": delta,
        "trend": trend,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def _cli() -> int:
    p = argparse.ArgumentParser(
        description="Vidlore Benchmark Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd")

    sc_score = sub.add_parser("score", help="Score an MP4 render")
    sc_score.add_argument("mp4", type=Path)
    sc_score.add_argument("script_json", type=Path)
    sc_score.add_argument("work_dir", type=Path, nargs="?", default=None)

    sc_compare = sub.add_parser("compare", help="Show version trend for a slug")
    sc_compare.add_argument("slug")

    sc_sheet = sub.add_parser("sheet", help="Generate contact sheet PNG")
    sc_sheet.add_argument("mp4", type=Path)
    sc_sheet.add_argument("output", type=Path, nargs="?", default=None)
    sc_sheet.add_argument("--grid", default="8x6")

    args = p.parse_args()
    if args.cmd == "score":
        result = score_render(args.mp4, args.script_json, args.work_dir)
        print("\n" + json.dumps(result["scores"], indent=2))
        return 0
    elif args.cmd == "compare":
        r = compare_versions(args.slug)
        print(f"\nTrend: {r['trend']}")
        if r.get("delta"):
            print("\nDelta (first → last):")
            for k, v in sorted(r["delta"].items(), key=lambda x: -abs(x[1])):
                arrow = "+" if v > 0 else ""
                print(f"  {k:<25} {arrow}{v:.3f}")
        return 0
    elif args.cmd == "sheet":
        mp4 = args.mp4
        out = args.output or mp4.parent / (mp4.stem + "_contact_sheet.png")
        result = generate_contact_sheet(mp4, out, args.grid)
        print(f"Contact sheet saved: {result}")
        return 0
    else:
        p.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(_cli())
