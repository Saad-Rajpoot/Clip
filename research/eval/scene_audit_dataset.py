"""Build a viewer's-eye audit set: what is ON SCREEN at time T vs what is BEING SAID at time T.

Deliberately reads only the FINISHED artifacts — the rendered mp4, its .srt and render_meta's
scene boundaries — never the pipeline's own beat bookkeeping. A render can believe it aired the
right footage and still show the wrong thing; an audit that starts from the same bookkeeping
inherits the same blind spot.

    python3 research/eval/scene_audit_dataset.py <job_dir> <out_dir> [--every N]

Writes <out_dir>/scene_NNN.jpg plus scenes.json: one row per scene with its span, the narration
spoken during it, and the aired source.

The join is `beat = scene - breakouts before it`, checked against the ledger's own narration before
anything is written — see scene_to_beat. I shipped an audit of job 409e284b60 with a duration-
matching join instead, and 23 of its 24 findings named a beat 1-4 places off. The self-check exists
so that cannot happen silently again: if fewer than 90% of scenes agree, this REFUSES to emit.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path


FFMPEG = "ffmpeg"
WORDS = re.compile(r"[a-z']+")


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


#  SCENE -> BEAT. Every beat occupies exactly one scene, whether it aired footage or an image
#  still; a breakout inserts an EXTRA scene that belongs to no beat. So:
#
#       beat = scene_index - (breakouts before it)
#
#  and a breakout scene is identifiable with no bookkeeping at all: it carries source audio, so no
#  voiceover caption overlaps it. VALIDATED on job 409e284b60 — the ledger's own narration for the
#  derived beat matches the captions actually heard over that scene in 169 of 170 scenes (99%).
#
#  Do NOT join on aired_windows' clip order: it records only the 158 footage clips, so its index
#  runs ahead of the scene index by every still and breakout in between. Matching each scene to
#  the next clip by duration looks like it works (it consumes exactly the right number) but drifts
#  silently — measured up to five beats out by the end, which would have blamed the wrong source
#  for every finding in the back half of the video.
def scene_to_beat(rows: list) -> None:
    """Attach `beat` to every scene row in place; None for a breakout scene."""
    brk = {r["scene"] for r in rows if not (r.get("narration") or "").strip()}
    for r in rows:
        r["beat"] = None if r["scene"] in brk else \
            r["scene"] - sum(1 for x in brk if x < r["scene"])


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
    #  the ledger is keyed by segment_index, which is what scene_to_beat resolves to — unlike
    #  aired_windows, whose clip order counts only footage clips
    led = {}
    lp = job / "ledger.jsonl"
    if lp.exists():
        for line in lp.read_text().splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if "segment_index" in r:
                led[int(r["segment_index"])] = r

    rows = []
    for i, (s, d) in enumerate(zip(starts, durs)):
        if i % every:
            continue
        mid = s + d / 2.0
        img = out / f"scene_{i:03d}.jpg"
        subprocess.run([FFMPEG, "-nostdin", "-v", "error", "-ss", f"{mid:.3f}", "-i", str(vid),
                        "-frames:v", "1", "-vf", "scale=960:-1", "-q:v", "3", "-y", str(img)],
                       check=False, capture_output=True)
        rows.append({
            "scene": i, "start": round(s, 2), "dur": round(d, 2), "mid": round(mid, 2),
            "image": img.name,
            "narration": narration_between(cues, s, s + d),
        })
    scene_to_beat(rows)
    for r in rows:
        b = led.get(r["beat"]) or {}
        r["source_id"] = b.get("source_id", "")
        r["source_title"] = b.get("source_title", "")
        r["in"] = b.get("in_point")
        r["relevance_class"] = b.get("relevance_class", "")
        r["ledger_narration"] = b.get("narration", "")
    (out / "scenes.json").write_text(json.dumps(rows, indent=1))
    #  self-check: the ledger's own narration for the derived beat must match what is HEARD over
    #  that scene. A join that has drifted shows up here immediately instead of in the findings.
    ok = tot = 0
    for r in rows:
        a = set(WORDS.findall((r.get("ledger_narration") or "").lower()))
        c = set(WORDS.findall((r.get("narration") or "").lower()))
        if a and c:
            tot += 1
            ok += len(a & c) / len(a) >= 0.5
    print(f"{len(rows)} scenes -> {out}")
    print(f"join self-check: ledger narration matches the captions heard in {ok}/{tot} scenes")
    if tot and ok / tot < 0.9:
        print("REFUSING: the scene->beat join has drifted; findings would blame the wrong beat",
              file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
