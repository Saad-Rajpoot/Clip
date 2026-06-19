"""IMP_021 assemble-branch proof: feed the EXACT filtergraph the
figure_locator branch in assemble._graphic_card_filters emits through
ffmpeg over a synthetic base clip and extract before/during/after frames.
Cheap (one 6s render, no TTS/footage/LLM)."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from vidlore.ffmpeg_tool import ffmpeg_exe  # noqa: E402

FFMPEG = ffmpeg_exe()
FPS = 30
asset = ROOT / "output" / "_imp021_dispatch_test" / "figloc_000.png"
assert asset.exists(), "run _test_figloc_dispatch.py first"
work = ROOT / "output" / "_imp021_assemble_test"
work.mkdir(parents=True, exist_ok=True)
mp4 = work / "figloc_composite.mp4"

# --- replicate the branch's arithmetic (start=0, d=6.0, emax=6.0) -------
start, d, emax = 0.0, 6.0, 6.0
fl_start = start + 0.40
b_ = min(start + d - 0.40, emax)
fout = b_ - 0.45
assert b_ - fl_start >= 1.4
win = f"between(t,{fl_start:.2f},{b_:.2f})"

# the exact two stages the branch appends, with {CUR}->base, {OUT}->outv
movie_stage = (
    f"movie='{asset}',format=rgba,loop=loop=-1:size=1,"
    f"setpts=N/{FPS}/TB,"
    f"fade=t=in:st={fl_start:.2f}:d=0.55:alpha=1,"
    f"fade=t=out:st={fout:.2f}:d=0.45:alpha=1[figl0]")
overlay_stage = f"[base][figl0]overlay=0:0:enable='{win}'[outv]"

# synth base: slow dark-blue drift so we can see the tether settle over it
base_src = (
    f"color=c=0x10141c:s=1920x1080:r={FPS}:d={d}[base]")
fc = f"{base_src};{movie_stage};{overlay_stage}"
print("FILTERGRAPH:\n ", fc.replace(";", ";\n  "))

cmd = [FFMPEG, "-y", "-f", "lavfi", "-i", base_src.split("[base]")[0],
       "-filter_complex", f"{movie_stage};[0:v][figl0]overlay=0:0:enable='{win}'[outv]",
       "-map", "[outv]", "-t", str(d), "-r", str(FPS),
       "-pix_fmt", "yuv420p", str(mp4)]
print("\nRUN:", " ".join(cmd[:6]), "... (filter_complex elided)")
r = subprocess.run(cmd, capture_output=True, text=True)
if r.returncode != 0:
    print("FFMPEG FAILED rc=", r.returncode)
    print(r.stderr[-2500:])
    sys.exit(1)
print("OK ->", mp4, mp4.stat().st_size, "bytes")

# extract 3 frames: before tether, mid (fully visible), after fade-out
for t, tag in [(0.20, "before"), (2.50, "during"), (5.85, "after")]:
    fr = work / f"frame_{tag}.png"
    subprocess.run([FFMPEG, "-y", "-ss", str(t), "-i", str(mp4),
                    "-frames:v", "1", str(fr)],
                   capture_output=True, text=True)
    print(f"  frame@{t}s ({tag}) ->", fr.exists())
print("DONE")
