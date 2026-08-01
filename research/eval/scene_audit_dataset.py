"""Build a viewer's-eye audit set: what is ON SCREEN at time T vs what is BEING SAID at time T.

Deliberately reads only the FINISHED artifacts — the rendered mp4, its .srt and render_meta's
scene boundaries — never the pipeline's own beat bookkeeping. A render can believe it aired the
right footage and still show the wrong thing; an audit that starts from the same bookkeeping
inherits the same blind spot.

    python3 research/eval/scene_audit_dataset.py <job_dir> <out_dir> [--every N]

Writes <out_dir>/scene_NNN.jpg plus scenes.json: one row per scene with its span, the narration
spoken during it, and the aired source (joined on the clip order in aired_windows.json).
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path


FFMPEG = "ffmpeg"


def ffmpeg_exe() -> str:
    """ffmpeg is not on PATH in this environment — the pipeline ships its own."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from vidlore.ffmpeg_tool import ffmpeg_exe as _f       # noqa: PLC0415
    return _f()


def parse_srt(p: Path) -> list:
    def _t(s: str) -> float:
        h, m, rest = s.split(":")
        sec, ms = rest.split(",")
        return int(h) * 3600 + int(m) * 60 + int(sec) + int(ms) / 1000.0
    out = []
    for b in re.split(r"\n\s*\n", p.read_text().strip()):
        L = b.strip().splitlines()
        if len(L) >= 3 and "-->" in L[1]:
            a, z = L[1].split("-->")
            out.append((_t(a.strip()), _t(z.strip()), " ".join(L[2:]).strip()))
    return out


def narration_between(cues: list, a: float, z: float) -> str:
    """Every caption whose span overlaps [a, z) — what the viewer hears while this shot is up."""
    return " ".join(c[2] for c in cues if c[1] > a and c[0] < z).strip()


#  aired_windows records only the FOOTAGE clips (158). The timeline has 174 scenes because image
#  stills and breakouts also occupy one each, so scene index != clip index — joining on the index
#  mislabelled 151 of 158 rows, i.e. it would have blamed the wrong source for every finding.
#  Each footage scene is exactly `need - CROSSFADE` long, which is a clean discriminator.
CROSSFADE = 0.5
cursor = [0]


def take_clip(clips: list, cur: list, dur: float) -> dict:
    """The next unconsumed clip if this scene's duration matches it; {} for a still/breakout."""
    i = cur[0]
    if i < len(clips) and abs(float(clips[i].get("need") or 0) - (dur + CROSSFADE)) <= 0.12:
        cur[0] = i + 1
        return clips[i]
    return {}


def main() -> int:
    job = Path(sys.argv[1])
    out = Path(sys.argv[2])
    every = int(sys.argv[sys.argv.index("--every") + 1]) if "--every" in sys.argv else 1
    out.mkdir(parents=True, exist_ok=True)

    global FFMPEG
    FFMPEG = ffmpeg_exe()
    vids = sorted(job.glob("output/final*.mp4"))
    if not vids:
        print("no rendered mp4", file=sys.stderr)
        return 2
    vid = vids[0]
    meta = json.loads((job / "output" / "render_meta.json").read_text())
    starts = meta["scene_starts"]
    durs = meta["scene_durations"]
    cues = parse_srt(next(job.glob("output/final*.srt")))
    aw = json.loads((job / "output" / "aired_windows.json").read_text())
    clips = aw.get("clips") or []

    rows = []
    for i, (s, d) in enumerate(zip(starts, durs)):
        if i % every:
            continue
        mid = s + d / 2.0
        img = out / f"scene_{i:03d}.jpg"
        subprocess.run([FFMPEG, "-nostdin", "-v", "error", "-ss", f"{mid:.3f}", "-i", str(vid),
                        "-frames:v", "1", "-vf", "scale=960:-1", "-q:v", "3", "-y", str(img)],
                       check=False, capture_output=True)
        c = take_clip(clips, cursor, d)
        rows.append({
            "scene": i, "start": round(s, 2), "dur": round(d, 2), "mid": round(mid, 2),
            "image": img.name,
            "narration": narration_between(cues, s, s + d),
            "source_id": c.get("source_id", ""), "source_title": c.get("source_title", ""),
            "beat": c.get("beat"), "via": c.get("via", ""), "in": c.get("in"),
        })
    (out / "scenes.json").write_text(json.dumps(rows, indent=1))
    print(f"{len(rows)} scenes -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
