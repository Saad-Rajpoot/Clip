"""Forensic analysis tool — fingerprint any documentary video.

Pipeline goal: take a YouTube URL (or local MP4) of a top-tier
documentary and emit a JSON FINGERPRINT measuring the editorial
choices a human editor made:

  * shot-length distribution (median, p25, p75, std)
  * cut-rate over time (cuts per 10s window)
  * music presence map (% of timeline with audio bed above silence floor)
  * silence map (long pauses ≥ 0.8s — silence-as-design moments)
  * loudness curve (sampled every 2s)
  * dominant-colour palette over time (low-res posterised frames)
  * text-overlay density (edge density on sampled frames as a proxy
    for "how often does this video paint type on screen")
  * dominant motion direction (mean optical-flow scalar per scene)

Run repeatedly across multiple videos of the same style and the
fingerprints can be averaged into a STYLE PROFILE
(geopolitical / psychological / historical / investigative / business
/ survival).  Those style profiles eventually feed back into the
editor as topic-aware defaults.

CLI usage:
    python -m vidlore.forensics <youtube-url-or-path> [--out=PATH.json]
    python -m vidlore.forensics --compare a.json b.json

Long-form bias: input is expected to be ≥10 min.  We sample the
first ~10 min if longer (faster + the editorial fingerprint of the
first act usually matches the rest).
"""
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------- #
# ffmpeg locator (re-use vidlore's bundled binary so the tool works
# even in an env without a system ffmpeg)
def _ffmpeg_exe() -> str:
    try:
        from .ffmpeg_tool import ffmpeg_exe
        return ffmpeg_exe()
    except Exception:                                              # noqa: BLE001
        return shutil.which("ffmpeg") or "ffmpeg"


# ---------------------------------------------------------------- #
# YouTube download via yt_dlp (Python module — we don't shell out
# to the CLI so the tool runs in clean envs without yt-dlp on PATH).
def download_youtube(url: str, out_dir: Path,
                     max_minutes: int = 12,
                     cookies_from_browser: str | None = None) -> Path:
    """Download a YouTube video at ≤720p H.264 to `out_dir`.  Returns
    the local mp4 path.  We cap at ~720p because higher resolutions
    don't change the fingerprint and waste time / bandwidth.

    YouTube has been increasingly aggressive about blocking yt-dlp
    (HTTP 403 / PO-token requirement).  Pass `cookies_from_browser`
    = 'firefox' | 'chrome' | 'safari' etc. so yt-dlp can use your
    existing logged-in browser session — usually defeats 403."""
    try:
        import yt_dlp                                              # type: ignore
    except ImportError as e:                                       # noqa: BLE001
        raise SystemExit(
            "yt_dlp is required to download YouTube videos. "
            "Install with: pip install yt-dlp"
        ) from e

    out_dir.mkdir(parents=True, exist_ok=True)
    out_tpl = str(out_dir / "%(id)s.%(ext)s")
    opts = {
        "format": (
            "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/"
            "best[height<=720][ext=mp4]/best[ext=mp4]/best"
        ),
        "merge_output_format": "mp4",
        "outtmpl": out_tpl,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        # cap clip for analysis (faster) — yt-dlp's downloader_args
        # passes through to ffmpeg when --download-sections is used
        # via the postprocessor; simpler: download full then trim.
    }
    if cookies_from_browser:
        opts["cookiesfrombrowser"] = (cookies_from_browser,)
    with yt_dlp.YoutubeDL(opts) as ydl:                            # type: ignore
        info = ydl.extract_info(url, download=True)
    vid_id = info.get("id", "video")
    src = out_dir / f"{vid_id}.mp4"
    if not src.exists():
        # yt-dlp may pick a different ext if format constraints couldn't be met
        for p in out_dir.glob(f"{vid_id}.*"):
            if p.suffix in (".mp4", ".mkv", ".webm"):
                src = p
                break
    if not src.exists():
        raise SystemExit(f"yt-dlp finished but no file matches {vid_id}.*")
    # Trim to max_minutes for faster fingerprinting
    trimmed = out_dir / f"{vid_id}_trim.mp4"
    if trimmed.exists():
        trimmed.unlink()
    subprocess.run(
        [_ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error",
         "-i", str(src), "-t", str(max_minutes * 60),
         "-c", "copy", str(trimmed)],
        capture_output=True)
    if trimmed.exists() and trimmed.stat().st_size > 1000:
        return trimmed
    return src


# ---------------------------------------------------------------- #
# Probing helpers
def _probe_duration(path: Path) -> float:
    """Video duration in seconds via ffmpeg stderr Duration line."""
    r = subprocess.run([_ffmpeg_exe(), "-hide_banner", "-i", str(path)],
                       capture_output=True, text=True)
    m = re.search(r"Duration:\s*(\d+):(\d+):([\d.]+)", r.stderr or "")
    if not m:
        return 0.0
    h, mn, s = m.groups()
    return int(h) * 3600 + int(mn) * 60 + float(s)


# ---------------------------------------------------------------- #
# Cut / shot detection.  ffmpeg's `select='gt(scene,0.3)'` emits the
# timestamp of every frame whose scene-change score exceeds 0.3 —
# that's a reliable "hard cut detected" signal.  We dump those
# timestamps with showinfo and parse the pts_time field.
def detect_cuts(video_path: Path, threshold: float = 0.32) -> list[float]:
    """Return cut timestamps (seconds, sorted ascending) from a video."""
    r = subprocess.run(
        [_ffmpeg_exe(), "-hide_banner", "-i", str(video_path),
         "-filter:v", f"select='gt(scene,{threshold})',showinfo",
         "-f", "null", "-"],
        capture_output=True, text=True)
    cuts: list[float] = []
    for line in (r.stderr or "").splitlines():
        m = re.search(r"pts_time:([\d.]+)", line)
        if m:
            try:
                cuts.append(float(m.group(1)))
            except ValueError:
                continue
    return sorted(cuts)


# ---------------------------------------------------------------- #
# Silence detection — gives us the music-presence map.  If the
# entire video has ≤ 5% silence the editor lays a continuous
# music bed.  > 25% silence is a "music-light" cut (60 Minutes
# style).  Mid-range = cinematic ebb/flow.
def detect_silences(video_path: Path, *,
                    noise_db: float = -32.0,
                    min_dur: float = 0.5) -> list[tuple[float, float]]:
    """Return list of (start, end) silence intervals."""
    r = subprocess.run(
        [_ffmpeg_exe(), "-hide_banner", "-i", str(video_path),
         "-af", f"silencedetect=noise={noise_db}dB:duration={min_dur}",
         "-f", "null", "-"],
        capture_output=True, text=True)
    starts: list[float] = []
    intervals: list[tuple[float, float]] = []
    for line in (r.stderr or "").splitlines():
        m_s = re.search(r"silence_start:\s*([\d.]+)", line)
        m_e = re.search(r"silence_end:\s*([\d.]+)", line)
        if m_s:
            try: starts.append(float(m_s.group(1)))
            except ValueError: pass
        elif m_e and starts:
            try:
                intervals.append((starts.pop(0), float(m_e.group(1))))
            except (ValueError, IndexError):
                pass
    return intervals


# ---------------------------------------------------------------- #
# Sampled frame analyser — color palette + edge density (proxy for
# "type-on-screen" frequency).  We sample every `every` seconds and
# downscale to 160x90 for cheap stats.
def sample_frames(video_path: Path, *,
                  every: float = 3.0,
                  max_samples: int = 220) -> list[dict]:
    """Return a list of frame stats dicts (one per sample)."""
    duration = _probe_duration(video_path)
    if duration <= 0:
        return []
    samples: list[dict] = []
    n = min(max_samples, max(1, int(duration / every)))
    tmpdir = Path(tempfile.mkdtemp(prefix="forensic_"))
    try:
        for i in range(n):
            t = i * every + 0.5
            frame_path = tmpdir / f"f_{i:04d}.jpg"
            subprocess.run(
                [_ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error",
                 "-ss", f"{t:.2f}", "-i", str(video_path),
                 "-vf", "scale=160:90", "-frames:v", "1",
                 "-q:v", "5", str(frame_path)],
                capture_output=True, timeout=15)
            if not frame_path.exists() or frame_path.stat().st_size < 200:
                continue
            stat = _analyze_frame(frame_path, t)
            if stat:
                samples.append(stat)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    return samples


def _analyze_frame(path: Path, t: float) -> Optional[dict]:
    """Per-frame stats: dominant colours + edge-density + mean brightness."""
    try:
        from PIL import Image
        import numpy as np
    except ImportError:                                            # noqa: BLE001
        return None
    try:
        im = Image.open(path).convert("RGB")
    except Exception:                                              # noqa: BLE001
        return None
    a = np.asarray(im, dtype="float32") / 255.0
    L = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    brightness = float(L.mean())
    # Crude saturation proxy — distance between max and min channel
    sat = float((a.max(axis=2) - a.min(axis=2)).mean())
    # Edge density via simple gradient magnitude (no scipy needed)
    gx = np.diff(L, axis=1)
    gy = np.diff(L, axis=0)
    g = np.sqrt(gx[:-1, :] ** 2 + gy[:, :-1] ** 2)
    edge_density = float((g > 0.10).mean())
    # Posterised palette: quantise to 4 buckets per channel = 64 bins
    q = (a * 4).astype("uint8").clip(0, 3)
    bins = q[..., 0] * 16 + q[..., 1] * 4 + q[..., 2]
    top = np.bincount(bins.ravel(), minlength=64).argsort()[::-1][:3]
    palette = []
    for b in top:
        r = int(((b >> 4) & 3) * 64 + 32)
        g_ = int(((b >> 2) & 3) * 64 + 32)
        bl = int((b & 3) * 64 + 32)
        palette.append([r, g_, bl])
    return {
        "t": round(t, 2),
        "brightness": round(brightness, 4),
        "saturation": round(sat, 4),
        "edge_density": round(edge_density, 4),
        "palette": palette,
    }


# ---------------------------------------------------------------- #
# Loudness curve — sample LUFS-ish RMS every 2 seconds via astats.
def loudness_curve(video_path: Path, *,
                   every: float = 2.0) -> list[tuple[float, float]]:
    """Return list of (time, RMS_dB) tuples.  We segment-extract a
    small WAV per window and read peak-RMS via volumedetect — more
    reliable than ametadata's RMS_level extraction (which silently
    drops -inf / no-data segments)."""
    duration = _probe_duration(video_path)
    if duration <= 0:
        return []
    out: list[tuple[float, float]] = []
    n = max(1, int(duration / every))
    tmpdir = Path(tempfile.mkdtemp(prefix="forensic_loud_"))
    try:
        for i in range(n):
            t = i * every
            seg = tmpdir / f"s_{i:04d}.wav"
            subprocess.run(
                [_ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error",
                 "-ss", f"{t:.2f}", "-t", f"{every:.2f}",
                 "-i", str(video_path),
                 "-vn", "-ac", "1", "-ar", "22050",
                 str(seg)],
                capture_output=True, timeout=20)
            if not seg.exists() or seg.stat().st_size < 200:
                out.append((round(t, 2), -90.0))
                continue
            r = subprocess.run(
                [_ffmpeg_exe(), "-hide_banner", "-i", str(seg),
                 "-af", "volumedetect", "-f", "null", "-"],
                capture_output=True, text=True, timeout=10)
            m = re.search(r"mean_volume:\s*(-?[\d.]+)\s*dB", r.stderr or "")
            if m:
                try:
                    out.append((round(t, 2), round(float(m.group(1)), 2)))
                except ValueError:
                    out.append((round(t, 2), -90.0))
            else:
                out.append((round(t, 2), -90.0))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    return out


# ---------------------------------------------------------------- #
@dataclass
class Fingerprint:
    """The full editorial fingerprint of a video."""
    source: str
    duration_s: float
    fps: float = 0.0
    n_cuts: int = 0
    median_shot_s: float = 0.0
    p25_shot_s: float = 0.0
    p75_shot_s: float = 0.0
    longest_shot_s: float = 0.0
    cuts_per_min: float = 0.0
    music_presence_pct: float = 0.0          # % timeline with audio above silence floor
    n_long_silences: int = 0                 # silences ≥ 0.8s
    mean_loudness_db: float = 0.0
    loudness_std_db: float = 0.0
    mean_brightness: float = 0.0
    mean_saturation: float = 0.0
    mean_edge_density: float = 0.0
    text_density_proxy: float = 0.0          # high edge_density frames as %
    palette_sample: list = field(default_factory=list)
    cuts: list = field(default_factory=list)
    silences: list = field(default_factory=list)
    loudness: list = field(default_factory=list)
    frame_samples: list = field(default_factory=list)


def analyze(video_path: Path) -> Fingerprint:
    """Full forensic pipeline.  ~30-90s per 10-min video on a mac mini."""
    print(f"[forensic] probing {video_path.name}…", file=sys.stderr, flush=True)
    duration = _probe_duration(video_path)
    fp = Fingerprint(source=str(video_path), duration_s=round(duration, 2))

    print(f"[forensic] cuts…", file=sys.stderr, flush=True)
    cuts = detect_cuts(video_path)
    fp.cuts = cuts
    fp.n_cuts = len(cuts)
    if cuts and duration > 0:
        # shot lengths = gaps between cuts, plus the first shot (0→cut0)
        # and the last (last_cut→duration)
        boundaries = [0.0] + cuts + [duration]
        shots = [boundaries[i + 1] - boundaries[i]
                 for i in range(len(boundaries) - 1)
                 if boundaries[i + 1] - boundaries[i] > 0.05]
        if shots:
            fp.median_shot_s = round(statistics.median(shots), 2)
            fp.p25_shot_s = round(_percentile(shots, 25), 2)
            fp.p75_shot_s = round(_percentile(shots, 75), 2)
            fp.longest_shot_s = round(max(shots), 2)
        fp.cuts_per_min = round(len(cuts) / (duration / 60.0), 2)

    print(f"[forensic] silences…", file=sys.stderr, flush=True)
    silences = detect_silences(video_path)
    fp.silences = [[round(s, 2), round(e, 2)] for s, e in silences]
    total_silence = sum(e - s for s, e in silences)
    if duration > 0:
        fp.music_presence_pct = round(100.0 * (1 - total_silence / duration), 1)
    fp.n_long_silences = sum(1 for s, e in silences if (e - s) >= 0.8)

    print(f"[forensic] loudness…", file=sys.stderr, flush=True)
    loud = loudness_curve(video_path)
    fp.loudness = loud
    if loud:
        vals = [v for _, v in loud if v > -90]
        if vals:
            fp.mean_loudness_db = round(statistics.mean(vals), 2)
            if len(vals) > 1:
                fp.loudness_std_db = round(statistics.stdev(vals), 2)

    print(f"[forensic] frames…", file=sys.stderr, flush=True)
    samples = sample_frames(video_path)
    fp.frame_samples = samples
    if samples:
        fp.mean_brightness = round(
            statistics.mean(s["brightness"] for s in samples), 4)
        fp.mean_saturation = round(
            statistics.mean(s["saturation"] for s in samples), 4)
        fp.mean_edge_density = round(
            statistics.mean(s["edge_density"] for s in samples), 4)
        # high edge-density frames = motion graphics / type overlay frames
        fp.text_density_proxy = round(
            100.0 * sum(1 for s in samples
                        if s["edge_density"] > 0.18) / len(samples), 1)
        # palette sample: take first 5 frames' top colours
        pal: list = []
        for s in samples[:5]:
            for c in s.get("palette", []):
                if c not in pal:
                    pal.append(c)
        fp.palette_sample = pal[:8]
    return fp


def _percentile(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    if not xs:
        return 0.0
    k = (len(xs) - 1) * p / 100.0
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return xs[lo]
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


# ---------------------------------------------------------------- #
def build_style_profile(fp_paths: list[Path], name: str = "custom") -> dict:
    """Average 2+ fingerprints into a STYLE PROFILE.

    Used to derive an editorial "north star" from multiple videos of
    the same channel / family.  Numeric metrics get arithmetic mean,
    plus min/max bounds + std so the profile carries variance, not
    just an average.  Lists (cuts, silences, palette) are not
    averaged (they're per-video); we keep only the summary numbers.
    """
    fps = [json.loads(p.read_text()) for p in fp_paths]
    if len(fps) < 2:
        raise SystemExit("style profile needs ≥2 fingerprints to average")
    metrics = ("duration_s", "n_cuts", "median_shot_s", "p25_shot_s",
               "p75_shot_s", "longest_shot_s", "cuts_per_min",
               "music_presence_pct", "n_long_silences",
               "mean_loudness_db", "loudness_std_db",
               "mean_brightness", "mean_saturation",
               "mean_edge_density", "text_density_proxy")
    out: dict = {"profile": name, "n_videos": len(fps),
                 "sources": [fp.get("source", "?") for fp in fps]}
    for k in metrics:
        vals = [float(fp.get(k, 0.0) or 0.0) for fp in fps]
        if not vals:
            continue
        out[k] = {
            "mean": round(statistics.mean(vals), 3),
            "min": round(min(vals), 3),
            "max": round(max(vals), 3),
            "std": round(statistics.stdev(vals), 3) if len(vals) > 1
                   else 0.0,
        }
    # palette: keep all distinct top colours seen across the set
    seen_pal: list = []
    for fp in fps:
        for c in fp.get("palette_sample", []):
            if c not in seen_pal:
                seen_pal.append(c)
    out["palette_union"] = seen_pal[:16]
    return out


def compare(fp_a_path: Path, fp_b_path: Path) -> dict:
    """Quick numeric diff of two fingerprints — useful for "how far
    is our render from a Vox piece?"
    """
    a = json.loads(fp_a_path.read_text())
    b = json.loads(fp_b_path.read_text())
    diff: dict = {}
    for k in ("median_shot_s", "p25_shot_s", "p75_shot_s", "cuts_per_min",
              "music_presence_pct", "n_long_silences",
              "mean_loudness_db", "loudness_std_db",
              "mean_brightness", "mean_saturation",
              "mean_edge_density", "text_density_proxy"):
        if k in a and k in b:
            diff[k] = {"a": a[k], "b": b[k],
                       "delta": round(b[k] - a[k], 3)}
    return diff


# ---------------------------------------------------------------- #
def _is_url(s: str) -> bool:
    return s.startswith("http://") or s.startswith("https://")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="vidlore.forensics",
                                description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", nargs="*",
                   help="One or more YouTube URLs or local video paths. "
                        "When multiple, --out is treated as a DIRECTORY.")
    p.add_argument("--out", type=Path, default=None,
                   help="Where to write JSON fingerprint(s).  Single "
                        "source → file; multiple → directory.")
    p.add_argument("--max-min", type=int, default=12,
                   help="Cap analyzed length in minutes (default 12)")
    p.add_argument("--compare", nargs=2, metavar=("FP_A", "FP_B"),
                   help="Diff two fingerprint JSON files; skips analysis")
    p.add_argument("--style-profile", nargs="+", metavar="FP",
                   help="Average 2+ fingerprints into a STYLE PROFILE "
                        "JSON (use after fingerprinting multiple videos "
                        "from the same channel/family).")
    p.add_argument("--profile-name", default="custom",
                   help="Name for the style profile output "
                        "(default: 'custom')")
    p.add_argument("--keep-download", action="store_true",
                   help="Don't delete the downloaded mp4 after analysis")
    p.add_argument("--cookies-from-browser",
                   choices=["firefox", "chrome", "safari", "edge", "brave"],
                   help="Pass a browser name to yt-dlp so it reads your "
                        "current YouTube login cookies (defeats most 403s)")
    args = p.parse_args(argv)

    if args.compare:
        diff = compare(Path(args.compare[0]), Path(args.compare[1]))
        print(json.dumps(diff, indent=2, ensure_ascii=False))
        return 0

    if args.style_profile:
        out_path = args.out or Path.cwd() / f"style_{args.profile_name}.json"
        prof = build_style_profile([Path(p_) for p_ in args.style_profile],
                                    args.profile_name)
        out_path.write_text(json.dumps(prof, indent=2, ensure_ascii=False))
        print(json.dumps(prof, indent=2, ensure_ascii=False))
        print(f"[forensic] wrote {out_path}", file=sys.stderr)
        return 0

    if not args.source:
        p.print_help()
        return 1

    # ── BATCH MODE ──────────────────────────────────────────────
    if len(args.source) > 1:
        out_dir = args.out or (Path.cwd() / "fingerprints")
        out_dir.mkdir(parents=True, exist_ok=True)
        results = []
        for src_arg in args.source:
            print(f"\n[forensic] === {src_arg} ===", file=sys.stderr)
            single_argv = [src_arg, "--out",
                           str(out_dir / f"fp_{len(results)+1:02d}.json"),
                           "--max-min", str(args.max_min)]
            if args.keep_download:
                single_argv.append("--keep-download")
            if args.cookies_from_browser:
                single_argv += ["--cookies-from-browser",
                                args.cookies_from_browser]
            rc = main(single_argv)
            results.append(rc)
        print(f"\n[forensic] batch done: "
              f"{results.count(0)}/{len(results)} succeeded",
              file=sys.stderr)
        return 0 if all(r == 0 for r in results) else 1

    src_arg = args.source[0]
    cleanup_dir: Optional[Path] = None
    if _is_url(src_arg):
        dl_dir = Path(tempfile.mkdtemp(prefix="forensic_dl_"))
        if not args.keep_download:
            cleanup_dir = dl_dir
        video_path = download_youtube(src_arg, dl_dir,
                                       max_minutes=args.max_min)
    else:
        video_path = Path(src_arg).expanduser().resolve()
        if not video_path.exists():
            print(f"file not found: {video_path}", file=sys.stderr)
            return 2

    fp = analyze(video_path)
    out_path = args.out
    if out_path is None:
        if _is_url(src_arg):
            out_path = Path.cwd() / f"{video_path.stem}.fingerprint.json"
        else:
            out_path = video_path.with_suffix(".fingerprint.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(asdict(fp), indent=2, ensure_ascii=False))
    print(f"[forensic] wrote {out_path}", file=sys.stderr)

    # Human summary on stdout
    print(json.dumps({
        "source": fp.source,
        "duration_s": fp.duration_s,
        "n_cuts": fp.n_cuts,
        "cuts_per_min": fp.cuts_per_min,
        "median_shot_s": fp.median_shot_s,
        "p25_p75_shot_s": [fp.p25_shot_s, fp.p75_shot_s],
        "music_presence_pct": fp.music_presence_pct,
        "n_long_silences": fp.n_long_silences,
        "mean_loudness_db": fp.mean_loudness_db,
        "loudness_std_db": fp.loudness_std_db,
        "mean_brightness": fp.mean_brightness,
        "mean_saturation": fp.mean_saturation,
        "text_density_proxy_pct": fp.text_density_proxy,
        "palette_sample": fp.palette_sample,
    }, indent=2, ensure_ascii=False))

    if cleanup_dir and cleanup_dir.exists():
        shutil.rmtree(cleanup_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
