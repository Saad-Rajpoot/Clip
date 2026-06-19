"""Text-readability regression — the permanent guard for Issue A.

Two layers are tested:

  1. ENGINE-SIDE adaptive typography (vidlore/templates/_shared.py):
     pick_ink() must choose text ink that clears the WCAG contrast thresholds
     on ANY card background — the light-on-light headline bug
     ("HOW COPPER STOPS SLUGS" near-white on cream) can never reappear.
     Body text targets ~4.5:1, large titles ~3:1.

  2. DETECTOR-SIDE QA (vidlore/editorial_qa.py):
     text_band_contrast() must FLAG a rendered light-on-light card and PASS a
     readable one, with no false positives on empty cards or plain footage —
     so any regression is caught automatically on the post-render sweep.

Run:  python3 tools/test_text_readability.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                   # noqa: E402
from PIL import Image, ImageDraw, ImageFont          # noqa: E402

from vidlore.templates import _shared as S           # noqa: E402
from vidlore import editorial_qa as EQ               # noqa: E402

FONT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "vidlore", "assets", "VidloreSans-Bold.ttf")

# Card backgrounds the engine must adapt to (the real Look-DNA bg modes).
CREAM = (242, 230, 208)     # paper_scan — the bug surface
WHITE = (250, 250, 250)     # white_clinical
NAVY = (10, 18, 36)         # dark_panel
GOLD = (212, 175, 55)       # accent / number-card bg
SLATE = (40, 44, 52)        # matte_flat dark
PARCH = (224, 206, 178)     # parchment_sepia

_passed = 0
_failed = 0


def check(name, cond, detail=""):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  [PASS] {name}")
    else:
        _failed += 1
        print(f"  [FAIL] {name}  {detail}")


# ───────────── Layer 1: engine-side adaptive ink (pick_ink) ─────────────
print("Layer 1 — adaptive typography (pick_ink / contrast):")
for label, bg, big in [
    ("cream  → readable ink", CREAM, False),
    ("white  → readable ink", WHITE, False),
    ("navy   → readable ink", NAVY, False),
    ("gold   → readable ink", GOLD, False),
    ("slate  → readable ink", SLATE, False),
    ("parch. → readable ink", PARCH, False),
    ("cream  large-title", CREAM, True),
    ("navy   large-title", NAVY, True),
]:
    ink = S.pick_ink(bg)
    cr = S.contrast_ratio(ink, bg)
    thresh = 3.0 if big else 4.5
    check(f"{label} ({cr:.1f}:1 ≥ {thresh})", cr >= thresh,
          f"ink={ink} contrast={cr:.2f}")

# the exact historical bug: near-white headline on cream must NOT be chosen
ink_cream = S.pick_ink(CREAM)
check("cream picks DARK ink (not near-white)", sum(ink_cream) < 384,
      f"ink={ink_cream}")
check("navy picks LIGHT ink (not near-black)", sum(S.pick_ink(NAVY)) > 384,
      f"ink={S.pick_ink(NAVY)}")

# accent brackets must also stay legible on the card bg
for label, bg in [("accent on cream", CREAM), ("accent on navy", NAVY)]:
    acc = S.safe_accent_on(bg, (212, 175, 55))
    check(f"{label} ≥3:1", S.contrast_ratio(acc, bg) >= 3.0,
          f"acc={acc} cr={S.contrast_ratio(acc, bg):.2f}")

# WCAG primitives sanity
check("relative_luminance white≈1", abs(S.relative_luminance((255, 255, 255)) - 1.0) < 0.01)
check("relative_luminance black≈0", S.relative_luminance((0, 0, 0)) < 0.01)
check("contrast black/white == 21", abs(S.contrast_ratio((0, 0, 0), (255, 255, 255)) - 21.0) < 0.1)


# ───────────── Layer 2: detector-side QA (text_band_contrast) ───────────
print("\nLayer 2 — rendered-frame readability detector:")
font = ImageFont.truetype(FONT, 70)


def card(bg, fg, text="HOW COPPER STOPS SLUGS", draw_text=True):
    im = Image.new("RGB", (1280, 720), bg)
    if draw_text:
        ImageDraw.Draw(im).text((120, 300), text, fill=fg, font=font)
    return im


T = EQ.CARD_TEXT_CONTRAST_MIN
# the bug must be flagged
check("light-on-light card FLAGGED",
      EQ.text_band_contrast(card(CREAM, (245, 248, 255))) < T)
# the fix must pass
check("dark-on-cream card PASSES",
      EQ.text_band_contrast(card(CREAM, S.pick_ink(CREAM))) >= T)
check("light-on-navy card PASSES",
      EQ.text_band_contrast(card(NAVY, S.pick_ink(NAVY))) >= T)
# no false positives
check("empty card NOT flagged",
      EQ.text_band_contrast(card(CREAM, CREAM, draw_text=False)) >= T)
rng = np.random.RandomState(7)
foot = Image.fromarray((rng.rand(720, 1280, 3) * 255).astype("uint8"))
check("plain footage NOT flagged", EQ.text_band_contrast(foot) >= T)
# a second bug surface: gold-on-gold number card
check("gold-on-gold card FLAGGED",
      EQ.text_band_contrast(card(GOLD, (220, 185, 70), text="96%")) < T)


# ── Layer 3: BODY copy (the "step description" tan-on-cream bug) ──────────
print("\nLayer 3 — body-copy ink (pick_body_ink) + worst-band detector:")
fbody = ImageFont.truetype(FONT, 30)


def card2(bg, title_fill, body_fill):
    """Card with a readable TITLE but variable-contrast BODY copy — the
    process-diagram step-description surface."""
    im = Image.new("RGB", (1280, 720), bg)
    d = ImageDraw.Draw(im)
    d.text((120, 90), "HOW COPPER STOPS SLUGS", fill=title_fill, font=font)
    for i, t in enumerate(["Slug wet foot touches copper strip",
                           "Galvanic current fires instantly",
                           "Slug retreats permanently"]):
        d.text((120, 340 + i * 70), t, fill=body_fill, font=fbody)
    return im


# pick_body_ink must clear the body threshold on every bg AND stay muted < title
for label, bg in [("cream body ink", CREAM), ("navy body ink", NAVY),
                  ("parch body ink", PARCH), ("gold body ink", GOLD)]:
    bi = S.pick_body_ink(bg)
    cr = S.contrast_ratio(bi, bg)
    check(f"{label} ({cr:.1f}:1 ≥ 4.5)", cr >= 4.5, f"ink={bi} cr={cr:.2f}")

# the bug: readable title + light-tan body must be FLAGGED (worst band gates)
TAN = (176, 150, 120)
check("title-OK + tan body FLAGGED",
      EQ.text_band_contrast(card2(CREAM, S.pick_ink(CREAM), TAN)) < T)
# the fix: readable title + pick_body_ink body must PASS
check("title-OK + body-ink PASSES",
      EQ.text_band_contrast(card2(CREAM, S.pick_ink(CREAM),
                                  S.pick_body_ink(CREAM))) >= T)


print(f"\n{_passed}/{_passed + _failed} text-readability cases passed")
sys.exit(0 if _failed == 0 else 1)
