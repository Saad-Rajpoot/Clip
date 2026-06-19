"""MNT_6 — prove a stale PRE-MNT_5 baked news_article card is invalidated.
Plant an old baked na_000.png (yellow highlight, NO _hl page) in a work_dir,
run build_graphic_images, confirm it RE-RENDERS: fresh plain (0 baked yellow)
+ a new _hl page. No API/render."""
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from PIL import Image, ImageDraw
from vidlore import footage as F
from vidlore.script_gen import Scene

work = ROOT / "output" / "_mnt6_cache"
work.mkdir(parents=True, exist_ok=True)
stale = work / "na_000.png"
hl = work / "na_000_hl.png"
# clean slate
for p in (stale, hl):
    p.unlink(missing_ok=True)

# Plant a STALE pre-MNT_5 asset: a card with a BAKED yellow highlight band,
# and NO _hl page (exactly the old format).
img = Image.new("RGBA", (1920, 1080), (0, 0, 0, 0))
d = ImageDraw.Draw(img)
d.rectangle([700, 480, 1220, 600], fill=(248, 245, 235, 252))   # paper
d.rectangle([740, 560, 1040, 596], fill=(255, 230, 90, 255))    # BAKED highlight
img.save(stale)


def _yellow(p):
    im = Image.open(p).convert("RGB")
    px = im.load()
    n = 0
    for y in range(480, 600, 2):
        for x in range(700, 1220, 2):
            r, g, b = px[x, y]
            if r > 180 and g > 165 and b < 140 and (r + g) > 2 * b + 120:
                n += 1
    return n


print(f"planted stale na_000.png baked-yellow px = {_yellow(stale)} "
      f"(simulates old static highlight); _hl exists = {hl.exists()}")

sc = Scene(index=0, narration="",
           keywords=[], graphic_kind="news_article",
           graphic_text="FORBES MAGAZINE | 1982",
           graphic_body="INTERNATIONAL BILLIONAIRES LIST | Pablo Escobar "
                        "listed among the world's wealthiest individuals — a "
                        "man who filed no taxes and incorporated no company.")
cfg = SimpleNamespace(has_fal=False, fal_key="", fal_model="")
out = F.build_graphic_images([sc], cfg, work, theme={"accent": (214, 64, 54)})
print("build_graphic_images out:", out)
print(f"AFTER: plain na_000.png yellow-band px = {_yellow(stale)} "
      f"(should be ~0 — fresh plain, no baked highlight)")
print(f"AFTER: _hl page exists = {hl.exists()}"
      + (f"  yellow-band px = {_yellow(hl)}" if hl.exists() else ""))

ok = (hl.exists() and _yellow(stale) < 30 and _yellow(hl) > 40)
print("\nRESULT:", "PASS — stale baked asset invalidated + regenerated "
      "(plain has no highlight, _hl page created)" if ok else "FAIL")
