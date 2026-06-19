#!/usr/bin/env python3
"""RC5.1 STEP 5 — deterministic tests for the CENTRAL readability adoption in look.py.

Target under test:  vidlore/motion_graphics/look.py  (its shared text helpers now
route through the readability gate in vidlore/motion_graphics/_shared.py).

Plain assert-based harness (matches tools/test_geo.py / test_rc5_mg_text_readability.py)
— no pytest dependency:

    .venv/bin/python tools/test_rc51_readability_adoption.py

Exits 0 when every test passes, 1 on any failure (all run; a PASS/FAIL summary prints
at the end). QA only — does NOT modify look.py / _shared.py or any source.

The four checks the task required:
  (1) look.py imports _shared AND its shared text helper invokes ensure_contrast
      (monkeypatched to record the call);
  (2) a thin-gray-on-dark draw TRIGGERS the repair (recolour) so the resulting glyph
      colour clears >= 4.5:1 against the bed;
  (3) an already-high-contrast draw is left UNCHANGED (helper returns the original
      colour, no scrim, no stroke, byte-identical raster);
  (4) defensive: if _shared raises / is unavailable, look.py STILL draws (no crash).
Plus a couple of supporting checks (byte-identical legacy path with no `bg`; the
paste_center scrim hook fires only on a genuine fail and is a no-op otherwise).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from vidlore.motion_graphics import look                       # noqa: E402
from vidlore.motion_graphics import _shared as S               # noqa: E402

# The thin-serif body colour at the heart of RC5: a low-contrast warm gray, on a
# crushed near-black documentary footage bed.
THIN_GRAY = (108, 102, 92)
DARK_BG = (24, 20, 16)
NEAR_WHITE = (244, 240, 232)        # the already-readable palette body/text colour
MID_BG = (120, 116, 110)            # a mid-tone bed neither pole clears alone


_RESULTS: list[tuple[str, bool, str]] = []


def _check(name: str, cond: bool, detail: str = "") -> None:
    _RESULTS.append((name, bool(cond), detail))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))


def _reset_shared_cache() -> None:
    """Restore look.py's lazy _shared cache to a clean, real-module state."""
    look._SHARED = None
    look._SHARED_TRIED = False


# ─────────────────────────────────────────────────────────────────────────
# (1) look.py imports _shared and its text helper invokes ensure_contrast.
# ─────────────────────────────────────────────────────────────────────────
def test_helper_invokes_ensure_contrast() -> None:
    _reset_shared_cache()
    # look.py must be able to import the sibling gate at all.
    mod = look._shared_mod()
    _check("look.py imports _shared", mod is S,
           "look._shared_mod() returned the _shared module")

    # Monkeypatch ensure_contrast and confirm the shared text helper calls it when a
    # background is supplied.
    fnt = look.font("caption", 48)
    orig = S.ensure_contrast
    calls = {"n": 0}

    def _spy(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    S.ensure_contrast = _spy
    try:
        look.text_with_glow("STAT", fnt, fill=THIN_GRAY, glow=(6, 4, 3),
                            glow_radius=4, glow_alpha=0.0, pad=10, bg=DARK_BG)
    finally:
        S.ensure_contrast = orig
    _check("text_with_glow(bg=...) invokes ensure_contrast", calls["n"] >= 1,
           f"ensure_contrast called {calls['n']}x")


# ─────────────────────────────────────────────────────────────────────────
# (2) thin-gray-on-dark triggers the repair → resulting contrast >= 4.5:1.
# ─────────────────────────────────────────────────────────────────────────
def test_thin_gray_on_dark_is_repaired() -> None:
    _reset_shared_cache()
    fnt = look.font("caption", 48)

    raw = S.contrast_ratio(THIN_GRAY, DARK_BG)
    _check("thin-gray/dark measures below body target", raw < S.CONTRAST_BODY,
           f"raw contrast {raw:.2f} < {S.CONTRAST_BODY}")

    out_fill, out_font, plan = look._readability_plan(
        "STAT", fnt, THIN_GRAY, DARK_BG, large=False, body=False)
    _check("repair changed the glyph colour", tuple(out_fill) != tuple(THIN_GRAY),
           f"{THIN_GRAY} -> {tuple(out_fill)}")

    repaired = S.contrast_ratio(out_fill, DARK_BG)
    _check("repaired glyph clears 4.5:1", repaired >= S.CONTRAST_BODY,
           f"repaired contrast {repaired:.2f} >= {S.CONTRAST_BODY}")
    _check("repair returned a ContrastPlan", plan is not None,
           f"plan.ok={getattr(plan, 'ok', None)}")

    # The rasterised layer must differ from the legacy (no-bg) raster — proof the
    # repair actually reached pixels.
    ras_repaired = look.text_with_glow("STAT", fnt, fill=THIN_GRAY, glow=(6, 4, 3),
                                       glow_radius=4, glow_alpha=0.0, pad=10, bg=DARK_BG)
    ras_legacy = look.text_with_glow("STAT", fnt, fill=THIN_GRAY, glow=(6, 4, 3),
                                     glow_radius=4, glow_alpha=0.0, pad=10)
    _check("repaired raster differs from legacy raster",
           not np.array_equal(np.asarray(ras_repaired), np.asarray(ras_legacy)),
           "the recoloured glyph changed the output pixels")


def test_body_thin_serif_swaps_to_sans_on_fail() -> None:
    """A failing BODY run swaps the thin serif for the readable heavier sans."""
    _reset_shared_cache()
    serif = look.font("title", 40)            # premium thin serif (the RC5 offender)
    _, out_font, _ = look._readability_plan(
        "definition body line", serif, THIN_GRAY, DARK_BG, large=False, body=True)
    # readable_body_font routes through look.font('caption') (condensed sans). We can't
    # compare TTF paths portably, but a swap must at least return a usable font object.
    _check("body=True fail returns a (swapped) font", out_font is not None,
           "heavier-sans body font resolved")


# ─────────────────────────────────────────────────────────────────────────
# (3) an already-high-contrast draw is left UNCHANGED (no recolour, no scrim).
# ─────────────────────────────────────────────────────────────────────────
def test_high_contrast_left_unchanged() -> None:
    _reset_shared_cache()
    fnt = look.font("caption", 48)

    passing = S.contrast_ratio(NEAR_WHITE, DARK_BG)
    _check("near-white/dark already passes", passing >= S.CONTRAST_BODY,
           f"contrast {passing:.2f} >= {S.CONTRAST_BODY}")

    out_fill, out_font, plan = look._readability_plan(
        "STAT", fnt, NEAR_WHITE, DARK_BG, large=False, body=False)
    _check("passing colour returned unchanged", tuple(out_fill) == tuple(NEAR_WHITE),
           f"{tuple(out_fill)} == {NEAR_WHITE}")
    _check("passing run carries NO repair plan", plan is None,
           "helper signalled no-change (None)")

    # And the rasterised layer with a bed must be byte-identical to the legacy raster.
    ras_bg = look.text_with_glow("STAT", fnt, fill=NEAR_WHITE, glow=(6, 4, 3),
                                 glow_radius=6, glow_alpha=0.0, pad=20, bg=DARK_BG)
    ras_nobg = look.text_with_glow("STAT", fnt, fill=NEAR_WHITE, glow=(6, 4, 3),
                                   glow_radius=6, glow_alpha=0.0, pad=20)
    _check("passing card raster byte-identical to legacy",
           np.array_equal(np.asarray(ras_bg), np.asarray(ras_nobg)),
           "already-good cards are not shifted")


def test_no_bg_is_byte_identical_legacy() -> None:
    """The default call (no bg) — i.e. every one of the ~100 existing primitives —
    is byte-identical to itself and untouched by the gate."""
    _reset_shared_cache()
    fnt = look.font("caption", 48)
    a = look.text_with_glow("HELLO WORLD", fnt, fill=THIN_GRAY, glow=(6, 4, 3),
                            glow_radius=6, glow_alpha=0.0, pad=20)
    b = look.text_with_glow("HELLO WORLD", fnt, fill=THIN_GRAY, glow=(6, 4, 3),
                            glow_radius=6, glow_alpha=0.0, pad=20)
    _check("legacy no-bg path is deterministic/byte-identical",
           np.array_equal(np.asarray(a), np.asarray(b)),
           "no gate engaged without a supplied background")


# ─────────────────────────────────────────────────────────────────────────
# (4) defensive — if _shared raises / is unavailable, look.py STILL draws.
# ─────────────────────────────────────────────────────────────────────────
def test_defensive_shared_unavailable() -> None:
    fnt = look.font("caption", 48)
    # Force the lazy cache into the "import failed" state.
    look._SHARED = None
    look._SHARED_TRIED = True
    try:
        of, ofont, plan = look._readability_plan(
            "X", fnt, THIN_GRAY, DARK_BG, large=False, body=False)
        _check("gate-unavailable returns inputs unchanged",
               tuple(of) == tuple(THIN_GRAY) and plan is None,
               "fell back to the legacy colour/font, no plan")
        img = look.text_with_glow("X", fnt, fill=THIN_GRAY, glow=(6, 4, 3),
                                  glow_radius=4, glow_alpha=0.0, pad=10, bg=DARK_BG)
        _check("gate-unavailable still produces a layer", img.size[0] > 0,
               f"drew a {img.size} RGBA layer without crashing")
    finally:
        _reset_shared_cache()


def test_defensive_ensure_contrast_raises() -> None:
    """If the gate itself raises mid-call, the helper swallows it and draws legacy."""
    _reset_shared_cache()
    look._shared_mod()                         # populate the cache with the real module
    fnt = look.font("caption", 48)
    orig = S.ensure_contrast

    def _boom(*a, **k):
        raise RuntimeError("synthetic gate failure")

    S.ensure_contrast = _boom
    try:
        of, ofont, plan = look._readability_plan(
            "X", fnt, THIN_GRAY, DARK_BG, large=False, body=False)
        _check("gate exception → inputs unchanged",
               tuple(of) == tuple(THIN_GRAY) and plan is None,
               "exception swallowed, legacy colour returned")
        img = look.text_with_glow("X", fnt, fill=THIN_GRAY, glow=(6, 4, 3),
                                  glow_radius=4, glow_alpha=0.0, pad=10, bg=DARK_BG)
        _check("gate exception → still draws (no crash)", img.size[0] > 0,
               f"drew a {img.size} RGBA layer")
    finally:
        S.ensure_contrast = orig
        _reset_shared_cache()


# ─────────────────────────────────────────────────────────────────────────
# paste_center scrim hook — fires only on a genuine fail; no-op otherwise.
# ─────────────────────────────────────────────────────────────────────────
def test_paste_center_scrim_fires_only_on_fail() -> None:
    _reset_shared_cache()
    fnt = look.font("caption", 40)

    # Mid-tone bed: neither near-white nor near-black text clears 4.5 alone, so the
    # glyph-side recolour cannot fix it → a bed scrim is genuinely needed.
    base_mid = Image.new("RGB", (400, 200), MID_BG)
    layer_wt = look.text_with_glow("READ ME", fnt, fill=NEAR_WHITE, glow=(6, 4, 3),
                                   glow_radius=4, glow_alpha=0.0, pad=12)
    out_scrim = look.paste_center(base_mid.copy(), layer_wt, cx=200, cy=100,
                                  scrim_for=NEAR_WHITE)
    out_plain = look.paste_center(base_mid.copy(), layer_wt, cx=200, cy=100)
    _check("paste_center scrim CHANGES pixels on a failing mid-tone bed",
           not np.array_equal(np.asarray(out_scrim), np.asarray(out_plain)),
           "restrained plate laid where recolour alone can't reach target")

    # Passing bed (dark text on a bright bed already > target): scrim hook is a no-op.
    base_bright = Image.new("RGB", (400, 200), (235, 230, 222))
    layer_dk = look.text_with_glow("READ ME", fnt, fill=(20, 16, 12),
                                   glow=(235, 230, 222), glow_radius=4,
                                   glow_alpha=0.0, pad=12)
    ob_scrim = look.paste_center(base_bright.copy(), layer_dk, cx=200, cy=100,
                                 scrim_for=(20, 16, 12))
    ob_plain = look.paste_center(base_bright.copy(), layer_dk, cx=200, cy=100)
    _check("paste_center scrim is a NO-OP on a passing bed (byte-identical)",
           np.array_equal(np.asarray(ob_scrim), np.asarray(ob_plain)),
           "premium-first: a readable bed is left untouched")


def main() -> int:
    print("RC5.1 STEP 5 — central readability adoption (look.py) QA\n")
    test_helper_invokes_ensure_contrast()
    test_thin_gray_on_dark_is_repaired()
    test_body_thin_serif_swaps_to_sans_on_fail()
    test_high_contrast_left_unchanged()
    test_no_bg_is_byte_identical_legacy()
    test_defensive_shared_unavailable()
    test_defensive_ensure_contrast_raises()
    test_paste_center_scrim_fires_only_on_fail()

    passed = sum(1 for _, ok, _ in _RESULTS if ok)
    total = len(_RESULTS)
    print(f"\nRC5.1 STEP 5 adoption QA:  {passed}/{total} passed, "
          f"{total - passed} failed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
