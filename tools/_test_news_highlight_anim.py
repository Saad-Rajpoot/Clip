"""MNT_5 proof — news_article highlighter now ANIMATES (not pre-baked).
Renders the card (plain + _hl pages), runs the assemble xfade=wiperight filter
over a dark background, extracts before/mid/after frames + counts yellow pixels
in the highlight band to prove the marker draws in left->right."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from types import SimpleNamespace
from PIL import Image
from vidlore import footage as F
from vidlore.ffmpeg_tool import ffmpeg_exe

FF = ffmpeg_exe()
work = ROOT / "output" / "_mnt5_news_hl"
work.mkdir(parents=True, exist_ok=True)

sc = SimpleNamespace(index=3, narration="", graphic_text="", graphic_body="")
theme = {"accent": (214, 64, 54), "name": "investigation"}
dest = work / "na_003.png"
ok = F._render_news_article_card(
    sc, theme, dest,
    "FORBES MAGAZINE | 1982",
    "INTERNATIONAL BILLIONAIRES LIST | Pablo Escobar listed among the "
    "world's wealthiest individuals — a man who filed no taxes, "
    "incorporated no company, and attended no business school.")
hl = dest.with_name("na_003_hl.png")
print("render ok:", ok, "| plain:", dest.exists(), "| hl page:", hl.exists())
assert ok and dest.exists() and hl.exists(), "expected plain + _hl pages"


def _yellow_in_band(img_path):
    """Count strongly-yellow pixels in the excerpt band (rows 560-640) —
    where the highlighter sits in this Forbes card."""
    im = Image.open(img_path).convert("RGB")
    px = im.load()
    n = 0
    for y in range(555, 660, 2):
        for x in range(430, 1500, 2):
            r, g, b = px[x, y]
            if r > 180 and g > 165 and b < 140 and (r + g) > 2 * b + 120:
                n += 1
    return n


# the two PAGES must differ only by the marker: plain band ~0 yellow, hl band >0
print(f"plain-page yellow band px : {_yellow_in_band(dest)}")
print(f"hl-page    yellow band px : {_yellow_in_band(hl)}")

# replicate the assemble xfade=wiperight filter (start=0,d=6,rev_t=2.0)
na_start, b_, ws, mt = 0.35, 5.55, 2.0, 0.6
base = f"color=c=0x12141a:s=1920x1080:r=30:d=6[bg]"
fc = (
    f"movie='{dest.name}',format=yuva420p,loop=loop=-1:size=1,setpts=N/30/TB,fps=30[nap];"
    f"movie='{hl.name}',format=yuva420p,loop=loop=-1:size=1,setpts=N/30/TB,fps=30[nah];"
    f"[nap][nah]xfade=transition=wiperight:duration={mt}:offset={ws}[nax];"
    f"[nax]format=rgba,fade=t=in:st={na_start}:d=0.55:alpha=1[na];"
    f"[0:v][na]overlay=0:0:enable='between(t,{na_start},{b_})'[outv]")
mp4 = work / "news_hl_anim.mp4"
cmd = [FF, "-y", "-f", "lavfi", "-i", f"color=c=0x12141a:s=1920x1080:r=30:d=6",
       "-filter_complex", fc, "-map", "[outv]", "-t", "6", "-r", "30",
       "-pix_fmt", "yuv420p", str(mp4)]
r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(work))
if r.returncode != 0:
    print("FFMPEG FAILED\n", r.stderr[-1800:])
    sys.exit(1)
print("composite ok:", mp4.stat().st_size, "bytes")

frames = [(1.2, "before"), (2.35, "mid"), (4.0, "after")]
counts = {}
for t, tag in frames:
    fr = work / f"frame_{tag}.png"
    subprocess.run([FF, "-y", "-ss", str(t), "-i", str(mp4), "-frames:v", "1",
                    str(fr)], capture_output=True, text=True)
    counts[tag] = _yellow_in_band(fr)
    print(f"  t={t}s ({tag}): yellow band px = {counts[tag]}")

# contact sheet
ims = [Image.open(work / f"frame_{tag}.png").convert("RGB").resize((640, 360))
       for _, tag in frames]
sheet = Image.new("RGB", (640 + 32, (360 + 30) * 3 + 16), (20, 20, 24))
from PIL import ImageDraw
d = ImageDraw.Draw(sheet)
y = 8
for im, (t, tag) in zip(ims, frames):
    d.text((16, y), f"t={t}s  {tag}  (yellow px={counts[tag]})", fill=(255, 210, 90))
    y += 26
    sheet.paste(im, (16, y)); y += 360 + 8
cs = ROOT / "research" / "mnt5_news_highlight_anim.png"
sheet.save(cs)
print("contact sheet:", cs)

verdict = (counts["before"] < 40 and counts["mid"] > counts["before"] + 80
           and counts["after"] > counts["mid"] + 40)
print("\nVERDICT:", "PASS — highlight animates left->right" if verdict
      else "REVIEW", f"(before={counts['before']}, mid={counts['mid']}, after={counts['after']})")
