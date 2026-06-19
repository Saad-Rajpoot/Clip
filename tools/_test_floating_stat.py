"""IMP_023 floating-stat render test.
(1) unit: _graphic_card_filters on a footage-only floating_stat cue ->
    emits the lower-third pill (drawbox + count-up + landed value).
(2) visual: composite those exact filters over a real-photo 'footage' frame
    and extract mid-roll + landed frames for QA."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from vidlore import assemble as A          # noqa: E402
from vidlore.ffmpeg_tool import ffmpeg_exe  # noqa: E402

FFMPEG = ffmpeg_exe()
work = ROOT / "output" / "_imp023_stat_test"
work.mkdir(parents=True, exist_ok=True)
font = ROOT / "vidlore" / "assets" / "VidloreSans.ttf"

# floating_stat cue: (start,d,kind,raw,body,asset,rev_t,shot_type,map_fig)
cue = (0.0, 6.0, "floating_stat", "130,000", "", "", -1.0, "", "130,000")
out, stages, post, num_events = A._graphic_card_filters(
    [cue], str(font), (240, 196, 90), work)
print("comma_filters:", len(out), "| stages:", len(stages),
      "| post:", len(post), "| num_events:", num_events)
for f in out:
    print("  ·", f[:96])
assert out and any("drawbox" in f for f in out) and \
    sum("drawtext" in f for f in out) >= 2, "expected pill + count-up + landed"

# choose a real-photo 'footage' background
bg_src = ROOT / "research" / "imp010_real_photo.png"
base_png = work / "footage_base.png"
subprocess.run([FFMPEG, "-y", "-i", str(bg_src),
                "-vf", "scale=1920:1080:force_original_aspect_ratio=increase,"
                "crop=1920:1080", str(base_png)],
               capture_output=True, text=True)

# build the filtergraph: looping still + the emitted comma_filters
fc = "[0:v]" + ",".join(out) + "[outv]"
mp4 = work / "stat_composite.mp4"
cmd = [FFMPEG, "-y", "-loop", "1", "-i", str(base_png),
       "-filter_complex", fc, "-map", "[outv]", "-t", "6",
       "-r", "30", "-pix_fmt", "yuv420p", str(mp4)]
r = subprocess.run(cmd, capture_output=True, text=True)
if r.returncode != 0:
    print("FFMPEG FAILED\n", r.stderr[-2000:])
    sys.exit(1)
print("OK ->", mp4, mp4.stat().st_size, "bytes")
for t, tag in [(0.20, "before"), (3.10, "rolling"), (5.40, "landed")]:
    fr = work / f"frame_{tag}.png"
    subprocess.run([FFMPEG, "-y", "-ss", str(t), "-i", str(mp4),
                    "-frames:v", "1", str(fr)], capture_output=True, text=True)
    print(f"  frame@{t}s ({tag}) ->", fr.exists())
print("DONE")
