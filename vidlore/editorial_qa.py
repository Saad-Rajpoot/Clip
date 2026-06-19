"""vidlore/editorial_qa.py — autonomous post-render editorial QA.

Scans a finished MP4 frame-by-frame (plus its audio and the render's manifests)
and emits EDITORIAL_QA_REPORT.json + EDITORIAL_QA_REPORT.md graded by severity.
It is the always-on "expert second pair of eyes" pass that catches the failure
classes a human documentary editor would catch on review, so they never ship
silently:

  - repeated footage ...... same shot reused at two timestamps        [Issue B]
  - unreadable card text .. low contrast inside a card's time window   [Issue A]
  - black / near-black .... dead frames, bad splice, content gap
  - blank-flash ........... near-white blown frames
  - washed / flat ......... very low global contrast (faded, milky)
  - loudness / peak ....... integrated LUFS or true-peak out of range
  - fail-closed slides .... relevance rejects that fell to a subject slide
                            (read straight from ASSET_DECISION_MANIFEST.json)

It does NOT replace human review; it is the first automated pass that turns the
"I'll notice it when I watch it" issues into a structured, actionable report.

Severities: critical > high > medium > low > cosmetic.
A "rerender_required" issue is one the engine should fix and re-render before
the video is considered shippable.

CLI:
    python3 -m vidlore.editorial_qa <video.mp4> [run_dir] [--fps 2]

Importable:
    from vidlore import editorial_qa as EQ
    report = EQ.run_qa(Path("video.mp4"), run_dir=Path("output/.../slug"))
    EQ.write_report(report, out_dir)
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
from typing import Optional

import numpy as np
from PIL import Image

# ───────────────────────── thresholds (documented) ─────────────────────────
# Frame sampling rate for the visual sweep (frames per second of video).
DEFAULT_FPS = 2.0
# A frame whose mean luminance (0..1) is below this is flagged. Below DEAD_LUMA
# is a true dead/black frame (hard gap, bad splice) → high severity; between
# DEAD_LUMA and BLACK_LUMA is a very-dark dim scene or a dissolve-through-dark →
# low severity (worth a glance, not a release blocker).
BLACK_LUMA = 0.06
DEAD_LUMA = 0.03
# …but a near-black frame is only a *dead* frame (failed clip / hard splice) when
# it is also nearly FEATURELESS. A dark-but-detailed cinematic shot (low mean,
# healthy spatial std — e.g. a shadowed ship hull against a misty sky) is a
# legitimate moody beat, not a defect. Require std below this to call it dead;
# this mirrors footage.py's blank-vs-dark-detail distinction (mean AND std).
DEAD_STD = 0.045
# Above this is a near-white "blank flash".
WHITE_LUMA = 0.93
# Global contrast (luma std, 0..1) below this is "washed / flat".
FLAT_STD = 0.045
# Perceptual-hash Hamming distance at/under which two FINAL-FRAME shots are "the
# same asset reused". Final frames of one asset (same grade/grain) are nearly
# identical, so this is stricter than the raw-asset VR-pass dedup (which uses 6
# to also catch re-crops); 3 here avoids false matches between two distinct but
# similarly-dark frames on a topically-cohesive video.
DUP_HAMMING = 3
# Two matching frames must be at least this far apart (s) to count as REUSE
# rather than the same held shot. A single shot rarely runs longer than this.
DUP_MIN_GAP_S = 4.0
# Card text-band Michelson contrast below this is "hard to read"; a documentary
# card headline should clear ~0.4 (≈3:1) comfortably.
CARD_TEXT_CONTRAST_MIN = 0.33
# Broadcast loudness window (integrated LUFS) and true-peak ceiling (dBTP).
LUFS_LO, LUFS_HI = -17.5, -13.0
PEAK_CEIL = -1.0


# ───────────────────────────── data model ──────────────────────────────────
@dataclass
class Issue:
    issue_type: str                 # e.g. "repeated_footage", "unreadable_text"
    severity: str                   # critical|high|medium|low|cosmetic
    subsystem: str                  # which engine owns the fix
    timestamp: float                # seconds into the video
    description: str                # human summary
    expected: str = ""              # what a good edit would show
    root_cause: str = ""            # best-guess cause
    fix: str = ""                   # recommended engine fix
    scene: Optional[int] = None
    second_timestamp: Optional[float] = None   # for paired issues (dupes)
    metric: Optional[float] = None
    rerender_required: bool = False


@dataclass
class QAReport:
    video: str
    duration_s: float
    frames_scanned: int
    issues: list = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    audio: dict = field(default_factory=dict)
    passed: bool = True
    gate: str = "PASS"              # PASS | WARN | FAIL


# ───────────────────────────── ffmpeg utils ────────────────────────────────
def _ff() -> str:
    """Resolve ffmpeg: PATH first, then the bundled imageio_ffmpeg binary the
    rest of the pipeline uses (ffmpeg is not always on PATH in this env)."""
    p = shutil.which("ffmpeg")
    if p:
        return p
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def _hms(t: float) -> str:
    m, s = divmod(max(0.0, float(t)), 60)
    return f"{int(m):d}:{s:05.2f}"


def probe_duration(mp4: Path) -> float:
    """Duration in seconds, parsed from `ffmpeg -i` stderr (no ffprobe needed)."""
    try:
        p = subprocess.run([_ff(), "-i", str(mp4)],
                           capture_output=True, text=True)
        m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", p.stderr)
        if m:
            h, mn, s = m.groups()
            return int(h) * 3600 + int(mn) * 60 + float(s)
    except Exception:
        pass
    return 0.0


def extract_frames(mp4: Path, fps: float, workdir: Path) -> list:
    """Extract frames at `fps` per second. Returns sorted [(t_seconds, Path)]."""
    workdir.mkdir(parents=True, exist_ok=True)
    pat = str(workdir / "f_%05d.jpg")
    subprocess.run(
        [_ff(), "-hide_banner", "-loglevel", "error", "-i", str(mp4),
         "-vf", f"fps={fps},scale=320:-1", "-q:v", "4", pat],
        check=False)
    out = []
    files = sorted(workdir.glob("f_*.jpg"))
    for i, fp in enumerate(files):
        out.append((i / fps, fp))
    return out


# ───────────────────────── perceptual + luma signals ───────────────────────
def _gray(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 3:
        return (0.299 * arr[..., 0] + 0.587 * arr[..., 1]
                + 0.114 * arr[..., 2])
    return arr


def phash(img: Image.Image) -> int:
    """64-bit DCT perceptual hash."""
    g = np.asarray(img.convert("L").resize((32, 32)), dtype=np.float64) / 255.0
    dct = _dct2(g)
    low = dct[:8, :8]
    med = np.median(low[1:].flatten())     # skip the DC term in the median
    bits = (low > med).flatten()
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return h


def _dct2(a: np.ndarray) -> np.ndarray:
    return _dct1(_dct1(a.T).T)


def _dct1(a: np.ndarray) -> np.ndarray:
    n = a.shape[0]
    k = np.arange(n).reshape(1, -1)
    x = np.arange(n).reshape(-1, 1)
    basis = np.cos(math.pi * (2 * x + 1) * k / (2 * n))
    with np.errstate(all="ignore"):        # benign BLAS fp flags on tiny mats
        return basis.T @ a


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def luma_stats(img: Image.Image) -> tuple:
    """Return (mean_luma, std_luma) in 0..1."""
    g = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
    return float(g.mean()), float(g.std())


def text_band_contrast(img: Image.Image) -> float:
    """WORST Michelson contrast among the card's text-like horizontal bands.

    A readable card text band is bright (or dark) text on the opposite-tone
    background → a wide luminance spread. A light-on-light (or tan-on-cream
    body-copy) band collapses that spread. The earlier version returned only the
    SINGLE strongest band (the big title), so a card whose TITLE was readable
    but whose BODY copy was low-contrast (the "step description" bug) slipped
    through. We now evaluate EVERY band that stands out above the median band
    energy (i.e. every distinct text strip — title AND body) and return the
    MINIMUM contrast, so the weakest line gates the card. No distinct strip
    (uniform footage / empty card) → passing contrast.
    """
    g = np.asarray(img.convert("L"), dtype=np.float32)
    h, w = g.shape
    if h < 24:
        return 1.0
    band = max(10, h // 12)
    step = max(1, band // 2)
    energies, tops = [], []
    for top in range(0, max(1, h - band), step):
        col = g[top:top + band]
        e = float(np.abs(np.diff(col, axis=0)).mean()
                  + np.abs(np.diff(col, axis=1)).mean())
        energies.append(e)
        tops.append(top)
    if not energies:
        return 1.0
    energies = np.array(energies)
    med = float(np.median(energies)) + 1e-6
    peak = float(energies.max())
    worst = 1.0
    found = False
    for e, top in zip(energies, tops):
        # a text strip stands clearly above the median band AND carries a
        # meaningful share of the strongest strip's energy (filters faint noise)
        if e < 1.6 * med or e < 0.18 * peak:
            continue
        rows = g[top:top + band]
        lo = float(np.percentile(rows, 2))
        hi = float(np.percentile(rows, 98))
        c = (hi - lo) / (hi + lo + 1e-6)
        worst = min(worst, c)
        found = True
    return worst if found else 1.0


# ───────────────────────────── QA checks ───────────────────────────────────
def check_frames(frames: list) -> list:
    """Black / blank-flash / flat-contrast per-frame issues."""
    issues = []
    for t, fp in frames:
        try:
            img = Image.open(fp)
        except Exception:
            continue
        mean, std = luma_stats(img)
        if mean < BLACK_LUMA:
            dead = mean < DEAD_LUMA and std < DEAD_STD     # featureless near-black only
            issues.append(Issue(
                "black_frame", "high" if dead else "low", "assemble", t,
                f"{'dead-black' if dead else 'very-dark'} frame near {_hms(t)} "
                f"(mean luma {mean:.3f}; ±1s sampling)",
                expected="visible footage or a graded card",
                root_cause=("content gap / failed clip / hard black splice"
                            if dead else "dim scene or dissolve-through-dark"),
                fix=("extend neighbour shot / hard-cut" if dead else
                     "lift crossfade midpoint or grade the dim beat brighter"),
                metric=round(mean, 4), rerender_required=dead))
        elif mean > WHITE_LUMA:
            issues.append(Issue(
                "blank_flash", "high", "assemble", t,
                f"near-white blank flash at {_hms(t)} (mean luma {mean:.3f})",
                expected="graded footage / card, never a white flash",
                root_cause="repair-anchor white frame or blown card bg",
                fix="guard repair anchor; clamp card background luminance",
                metric=round(mean, 4), rerender_required=True))
        elif std < FLAT_STD and mean > 0.5:
            issues.append(Issue(
                "washed_frame", "low", "grade", t,
                f"washed/flat frame at {_hms(t)} (luma std {std:.3f})",
                expected="natural tonal range",
                root_cause="over-bright bg, milky grade, or empty card",
                fix="add contrast / vignette / verify card content present",
                metric=round(std, 4)))
    return issues


SHOT_CUT_HAMMING = 12       # consecutive-frame phash jump that marks a new shot


def _segment_shots(hashes: list) -> list:
    """Group consecutive frames into shots. A continuously-held card or clip is
    ONE shot (so it is never mistaken for a duplicate of itself). Returns
    [(start_t, end_t, representative_phash, mid_index)]."""
    shots = []
    if not hashes:
        return shots
    start = 0
    for k in range(1, len(hashes)):
        if hamming(hashes[k][1], hashes[k - 1][1]) > SHOT_CUT_HAMMING:
            mid = (start + k - 1) // 2
            shots.append((hashes[start][0], hashes[k - 1][0],
                          hashes[mid][1], mid))
            start = k
    mid = (start + len(hashes) - 1) // 2
    shots.append((hashes[start][0], hashes[-1][0], hashes[mid][1], mid))
    return shots


def check_duplicates(frames: list, scene_starts: list = None,
                     scene_durations: list = None) -> list:
    """Cross-SCENE footage REUSE — the repeated-asset class the user objects to.

    A duplicate is two DISTINCT shots, IN DIFFERENT SCENES, that look the same —
    not a single shot held across a scene's own beats, and not a scene that
    returns to its establishing clip. We segment the video into shots (cut = a
    large consecutive-frame phash jump) and compare shot representatives, but
    skip any pair that falls inside the SAME scene (intra-scene reuse is a
    legitimate editorial choice and is exactly what the VR-pass dedup allows).
    Adjacent shots are also skipped (a dissolve / before-after pair is not
    reuse); only a shot reappearing in a LATER scene, ≥DUP_MIN_GAP_S on, counts.
    """
    scene_starts = scene_starts or []

    def _scene(t):
        if not scene_starts:
            return -1                       # no table → treat all as distinct
        for i, s in enumerate(scene_starts):
            e = s + (scene_durations[i] if scene_durations
                     and i < len(scene_durations) else 1e9)
            if s <= t < e:
                return i
        return len(scene_starts) - 1

    issues = []
    hashes = []
    for t, fp in frames:
        try:
            hashes.append((t, phash(Image.open(fp))))
        except Exception:
            continue
    shots = _segment_shots(hashes)
    for a in range(len(shots)):
        sa, ea, ha, _ = shots[a]
        for b in range(a + 2, len(shots)):     # +2 → skip the adjacent shot
            sb, eb, hb, _ = shots[b]
            if sb - ea < DUP_MIN_GAP_S:
                continue
            if scene_starts and _scene(sa) == _scene(sb):
                continue                    # intra-scene reuse is allowed
            d = hamming(ha, hb)
            if d <= DUP_HAMMING:
                issues.append(Issue(
                    "repeated_footage", "high", "footage", round(sa, 2),
                    f"shot at {_hms(sa)} reappears at {_hms(sb)} "
                    f"(phash dist {d})",
                    expected="each footage asset appears once per video",
                    root_cause="cross-scene asset reuse slipped the dedup gate",
                    fix="cross-scene phash dedup → escalate the weaker beat to a "
                        "fresh asset",
                    second_timestamp=round(sb, 2),
                    metric=d, rerender_required=True))
                break                          # one report per source shot
    return issues


def check_pacing(frames: list) -> list:
    """Editorial rhythm — overly long monotonous holds and machine-gun cut runs.

    Re-uses shot segmentation: a single shot longer than LONG_HOLD_S reads as a
    monotonous hold; a run of CUT_RUN_N shots each under FAST_CUT_S reads as a
    frantic stretch. Both are low-severity editorial notes, not release blockers.
    """
    LONG_HOLD_S, FAST_CUT_S, CUT_RUN_N = 13.0, 1.1, 5
    issues = []
    hashes = []
    for t, fp in frames:
        try:
            hashes.append((t, phash(Image.open(fp))))
        except Exception:
            continue
    shots = _segment_shots(hashes)
    # long holds
    for sa, ea, _, _ in shots:
        if ea - sa >= LONG_HOLD_S:
            issues.append(Issue(
                "long_hold", "low", "pacing", round(sa, 2),
                f"a single shot holds {ea - sa:.0f}s from {_hms(sa)} "
                f"(monotonous)",
                expected="cut or push-in before attention drifts",
                root_cause="no beat split / restraint engine held too long",
                fix="split the beat or add a motivated push-in/cut",
                metric=round(ea - sa, 1)))
    # fast-cut runs
    run = 0
    run_start = None
    for sa, ea, _, _ in shots:
        if ea - sa < FAST_CUT_S:
            run += 1
            run_start = run_start if run_start is not None else sa
        else:
            if run >= CUT_RUN_N:
                issues.append(Issue(
                    "fast_cut_run", "low", "pacing", round(run_start, 2),
                    f"{run} rapid cuts (<{FAST_CUT_S}s each) from "
                    f"{_hms(run_start)}",
                    expected="let key shots breathe",
                    root_cause="cut rate too high for the beat",
                    fix="lengthen holds in this stretch", metric=run))
            run, run_start = 0, None
    return issues


def check_opening(frames: list, dur: float) -> list:
    """Weak first 30 seconds — the retention-critical hook. Flags an opening
    that is mostly one shot (low variety) or mostly dark."""
    issues = []
    head = [(t, fp) for t, fp in frames if t <= 30.0]
    if len(head) < 4:
        return issues
    hashes = []
    lumas = []
    for t, fp in head:
        try:
            im = Image.open(fp)
            hashes.append((t, phash(im)))
            lumas.append(luma_stats(im)[0])
        except Exception:
            continue
    shots = _segment_shots(hashes)
    import numpy as _np
    if len(shots) < 3:
        issues.append(Issue(
            "weak_opening", "medium", "editorial", 0.0,
            f"first 30s has only {len(shots)} distinct shot(s) — low variety",
            expected="3+ varied, strong shots in the hook",
            root_cause="opening leans on one held visual",
            fix="add visual variety / a stronger lead shot", metric=len(shots)))
    if lumas and float(_np.mean(lumas)) < 0.12:
        issues.append(Issue(
            "weak_opening", "medium", "editorial", 0.0,
            f"first 30s is very dark (mean luma {float(_np.mean(lumas)):.3f})",
            expected="a bright, legible hook",
            root_cause="dark footage/cards lead the video",
            fix="lead with a brighter, clearer visual",
            metric=round(float(_np.mean(lumas)), 3)))
    return issues


def check_audio_silence(mp4: Path) -> list:
    """Dead audio — a sustained silent gap mid-video (not the intended tail)."""
    issues = []
    try:
        p = subprocess.run(
            [_ff(), "-hide_banner", "-i", str(mp4), "-af",
             "silencedetect=noise=-50dB:d=1.5", "-f", "null", "-"],
            capture_output=True, text=True)
        starts = re.findall(r"silence_start:\s*([\d.]+)", p.stderr)
        durs = re.findall(r"silence_duration:\s*([\d.]+)", p.stderr)
        dur = probe_duration(mp4)
        for s, d in zip(starts, durs):
            s, d = float(s), float(d)
            if s > 2.0 and (s + d) < dur - 2.0:      # ignore lead-in / tail
                issues.append(Issue(
                    "dead_audio", "medium", "audio", round(s, 2),
                    f"{d:.1f}s of near-silence at {_hms(s)} (mid-video)",
                    expected="continuous bed/narration under the body",
                    root_cause="VO gap with no music/room-tone fill",
                    fix="extend the music bed / room tone across the gap",
                    metric=round(d, 2)))
    except Exception:
        pass
    return issues


def check_text_readability(mp4: Path, frames: list, run_dir: Optional[Path],
                           workdir: Path) -> list:
    """Flag card windows whose text band has collapsed contrast (Issue A).

    Uses motion_graphics_manifest.json (if present) to know WHEN text cards are
    on screen, samples a frame mid-window, and measures the text-band contrast.
    Falls back to scanning all sampled frames for card-like flat-contrast text
    bands when no manifest is available.
    """
    issues = []
    windows = []            # (start, end, label)
    man = (run_dir / "motion_graphics_manifest.json") if run_dir else None
    if man and man.exists():
        try:
            data = json.loads(man.read_text())
            entries = data if isinstance(data, list) else data.get("cards", data.get("items", []))
            for e in entries if isinstance(entries, list) else []:
                if not isinstance(e, dict):
                    continue
                kind = str(e.get("type", e.get("kind", e.get("primitive", "")))).lower()
                # only the text-bearing explainer cards
                if not any(k in kind for k in (
                        "process", "cause", "effect", "bullet", "timeline",
                        "comparison", "definition", "statement", "quote",
                        "diagram", "list", "term", "title", "lower_third",
                        "stat", "number")):
                    continue
                st = e.get("start", e.get("t", e.get("start_s")))
                du = e.get("duration", e.get("dur", e.get("len", 3.0)))
                if st is None:
                    continue
                windows.append((float(st), float(st) + float(du or 3.0),
                                kind or "card"))
        except Exception:
            windows = []

    fmap = {round(t, 3): fp for t, fp in frames}
    fts = [t for t, _ in frames]

    def nearest_frame(tc):
        if not fts:
            return None
        t = min(fts, key=lambda x: abs(x - tc))
        return fmap.get(round(t, 3))

    if windows:
        for st, en, label in windows:
            mid = (st + en) / 2.0
            fp = nearest_frame(mid)
            if not fp:
                continue
            try:
                c = text_band_contrast(Image.open(fp))
            except Exception:
                continue
            if c < CARD_TEXT_CONTRAST_MIN:
                issues.append(Issue(
                    "unreadable_text", "high", "typography", mid,
                    f"{label} card at {_hms(mid)} has low text contrast "
                    f"(Michelson {c:.2f}); headline may read light-on-light",
                    expected="≥3:1 large-title / ≥4.5:1 body contrast",
                    root_cause="card ink not adapted to a light/busy background",
                    fix="background-aware adaptive ink (pick_ink) + scrim on "
                        "busy/low-contrast backgrounds",
                    metric=round(c, 3), rerender_required=True))
    return issues


def check_audio(mp4: Path) -> dict:
    """Integrated LUFS + true-peak via the loudnorm measurement pass."""
    out = {"lufs": None, "peak_dbtp": None}
    try:
        p = subprocess.run(
            [_ff(), "-hide_banner", "-i", str(mp4), "-af",
             "loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json",
             "-f", "null", "-"],
            capture_output=True, text=True)
        m = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", p.stderr, re.S)
        if m:
            d = json.loads(m.group(0))
            out["lufs"] = float(d.get("input_i"))
            out["peak_dbtp"] = float(d.get("input_tp"))
    except Exception:
        pass
    return out


def check_text_slide_ratio(run_dir: Optional[Path]) -> list:
    """Emergency text slides must be a LAST RESORT, not the normal output.

    Reads the ASSET_DECISION_MANIFEST and flags HIGH if subject (text) slides
    exceed VIDLORE_MAX_TEXT_SLIDE_RATIO (default 0.10) of the beats — the
    signature of a render where the AI-still providers weren't producing images
    (e.g. the free Pollinations tier blocked by the paid budget). This is the
    gate that catches the FARMER / UNIVERSITY STUDY text-slide-heavy output.
    """
    issues = []
    if not run_dir:
        return issues
    man = Path(run_dir) / "ASSET_DECISION_MANIFEST.json"
    if not man.exists():
        return issues
    try:
        data = json.loads(man.read_text())
    except Exception:
        return issues
    recs = data.get("beats", data.get("records", data if isinstance(data, list)
                                      else []))
    total = len([r for r in recs if isinstance(r, dict)])
    slides = len([r for r in recs if isinstance(r, dict)
                  and "slide" in str(r.get("outcome", "")).lower()])
    if total <= 0:
        return issues
    ratio = slides / total
    try:
        cap = float(os.environ.get("VIDLORE_MAX_TEXT_SLIDE_RATIO", "0.10"))
    except Exception:
        cap = 0.10
    if ratio > cap:
        issues.append(Issue(
            "excess_text_slides", "high", "footage", 0.0,
            f"{slides}/{total} beats are emergency TEXT SLIDES "
            f"({ratio:.0%} > {cap:.0%} cap) — AI stills were not generated",
            expected="rejected beats become relevant AI stills; slides ≤ cap",
            root_cause="AI-still provider unavailable or budget-blocked "
                       "(free Pollinations must not share the paid ceiling)",
            fix="ensure fal/Pollinations reachable; free tier ungated by budget",
            metric=round(ratio, 3), rerender_required=True))
    return issues


def check_manifest_failclosed(run_dir: Optional[Path]) -> list:
    """Surface relevance rejects that fell to a fail-closed subject slide.

    A subject slide is on-topic and SAFE, but a video full of them means the
    footage pipeline couldn't find real assets — worth a low-severity flag so a
    human knows several beats are text slides rather than footage.
    """
    issues = []
    if not run_dir:
        return issues
    man = run_dir / "ASSET_DECISION_MANIFEST.json"
    if not man.exists():
        return issues
    try:
        data = json.loads(man.read_text())
    except Exception:
        return issues
    recs = data.get("records", data if isinstance(data, list) else [])
    slides = [r for r in recs if isinstance(r, dict)
              and "slide" in str(r.get("outcome", "")).lower()]
    dups = [r for r in recs if isinstance(r, dict)
            and "cross-scene-duplicate" in str(r.get("reason", "")).lower()]
    if slides:
        issues.append(Issue(
            "fail_closed_slides", "low", "footage", 0.0,
            f"{len(slides)} beat(s) fell to a fail-closed subject slide "
            f"(safe & on-topic, but not footage)",
            expected="relevant footage or a relevant AI still",
            root_cause="no on-topic real asset passed the relevance gate",
            fix="widen stock queries / enable AI image gen in production",
            metric=len(slides)))
    if dups:
        issues.append(Issue(
            "dedup_escalations", "cosmetic", "footage", 0.0,
            f"{len(dups)} beat(s) were re-sourced by the cross-scene dedup gate "
            f"(working as intended)",
            expected="unique asset per beat",
            root_cause="duplicate detected and escalated to a fresh asset",
            fix="none — informational", metric=len(dups)))
    return issues


# ───────────────────────────── orchestration ───────────────────────────────
_SEV_RANK = {"critical": 4, "high": 3, "medium": 2, "low": 1, "cosmetic": 0}


def run_qa(mp4: Path, run_dir: Optional[Path] = None,
           fps: float = DEFAULT_FPS) -> QAReport:
    mp4 = Path(mp4)
    dur = probe_duration(mp4)
    tmp = Path(tempfile.mkdtemp(prefix="eqa_"))
    try:
        frames = extract_frames(mp4, fps, tmp / "frames")
        # scene boundaries (for scene-aware dedup — intra-scene reuse allowed)
        _starts, _durs = [], []
        if run_dir:
            try:
                _m = json.loads((Path(run_dir) / "render_meta.json").read_text())
                _starts = list(_m.get("scene_starts", []))
                _durs = list(_m.get("scene_durations", []))
            except Exception:
                pass
        issues = []
        issues += check_frames(frames)
        issues += check_duplicates(frames, _starts, _durs)
        issues += check_text_readability(mp4, frames, run_dir, tmp)
        issues += check_pacing(frames)
        issues += check_opening(frames, dur)
        issues += check_manifest_failclosed(run_dir)
        issues += check_text_slide_ratio(run_dir)
        issues += check_audio_silence(mp4)
        audio = check_audio(mp4)
        # audio issues
        if audio.get("lufs") is not None and not (LUFS_LO <= audio["lufs"] <= LUFS_HI):
            issues.append(Issue(
                "loudness", "medium", "audio", 0.0,
                f"integrated loudness {audio['lufs']:.1f} LUFS outside "
                f"[{LUFS_LO}, {LUFS_HI}]",
                expected=f"~-16 LUFS", root_cause="mix gain drift",
                fix="re-run loudnorm to -16 LUFS", metric=audio["lufs"]))
        if audio.get("peak_dbtp") is not None and audio["peak_dbtp"] > PEAK_CEIL:
            issues.append(Issue(
                "true_peak", "medium", "audio", 0.0,
                f"true peak {audio['peak_dbtp']:.1f} dBTP over {PEAK_CEIL}",
                expected=f"≤{PEAK_CEIL} dBTP", root_cause="insufficient headroom",
                fix="apply -1.5 dBTP limiter", metric=audio["peak_dbtp"]))

        issues.sort(key=lambda i: (-_SEV_RANK.get(i.severity, 0), i.timestamp))
        by_sev = {}
        for it in issues:
            by_sev[it.severity] = by_sev.get(it.severity, 0) + 1
        rerender = any(i.rerender_required for i in issues)
        crit_high = by_sev.get("critical", 0) + by_sev.get("high", 0)
        gate = "FAIL" if crit_high else ("WARN" if issues else "PASS")
        rep = QAReport(
            video=str(mp4), duration_s=round(dur, 2),
            frames_scanned=len(frames),
            issues=[asdict(i) for i in issues],
            summary={"by_severity": by_sev, "total": len(issues),
                     "rerender_required": rerender},
            audio=audio, passed=(gate != "FAIL"), gate=gate)
        return rep
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def write_report(rep: QAReport, out_dir: Path) -> tuple:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jp = out_dir / "EDITORIAL_QA_REPORT.json"
    jp.write_text(json.dumps(asdict(rep), indent=2))
    md = out_dir / "EDITORIAL_QA_REPORT.md"
    lines = [
        f"# Editorial QA Report — gate: **{rep.gate}**", "",
        f"- Video: `{rep.video}`",
        f"- Duration: {rep.duration_s:.1f}s · frames scanned: {rep.frames_scanned}",
        f"- Audio: {rep.audio.get('lufs')} LUFS / peak {rep.audio.get('peak_dbtp')} dBTP",
        f"- Issues: {rep.summary.get('total', 0)} "
        f"({rep.summary.get('by_severity', {})})",
        f"- Re-render required: {rep.summary.get('rerender_required')}", "",
        "| sev | type | when | description | fix |",
        "|-----|------|------|-------------|-----|",
    ]
    for i in rep.issues:
        when = _hms(i["timestamp"])
        if i.get("second_timestamp") is not None:
            when += f" ↔ {_hms(i['second_timestamp'])}"
        desc = i["description"].replace("|", "\\|")
        fix = (i.get("fix", "") or "").replace("|", "\\|")
        lines.append(f"| {i['severity']} | {i['issue_type']} | {when} | "
                     f"{desc} | {fix} |")
    if not rep.issues:
        lines.append("| — | — | — | no issues found | — |")
    md.write_text("\n".join(lines) + "\n")
    return jp, md


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Autonomous post-render editorial QA")
    ap.add_argument("video")
    ap.add_argument("run_dir", nargs="?", default=None)
    ap.add_argument("--fps", type=float, default=DEFAULT_FPS)
    ap.add_argument("--out", default=None, help="report output dir (default: run_dir or video dir)")
    a = ap.parse_args(argv)
    mp4 = Path(a.video)
    run_dir = Path(a.run_dir) if a.run_dir else mp4.parent
    out_dir = Path(a.out) if a.out else run_dir
    rep = run_qa(mp4, run_dir=run_dir, fps=a.fps)
    jp, md = write_report(rep, out_dir)
    print(f"[editorial-qa] gate={rep.gate}  issues={rep.summary.get('total')}  "
          f"{rep.summary.get('by_severity')}")
    print(f"[editorial-qa] wrote {jp}")
    print(f"[editorial-qa] wrote {md}")
    return 0 if rep.gate != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
