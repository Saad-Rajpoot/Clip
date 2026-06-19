#!/usr/bin/env python3
"""RC5.1 — deterministic regression tests for the MG text-readability gate.

Target under test:  vidlore/motion_graphics/_shared.py

Plain assert-based harness (matches tools/test_geo.py) — no pytest dependency:

    .venv/bin/python tools/test_rc5_mg_text_readability.py

Exits 0 when every test passes, 1 on any failure (all run; a PASS/FAIL summary
prints at the end). QA only — does NOT modify _shared.py or any source.

Core asserts the task asked for:
  * the contrast calculator FLAGS a thin-gray-on-dark case as failing (< 4.5:1);
  * the repair ladder (colour shift + scrim) brings that case to >= 4.5:1;
  * a clean high-contrast case is left UNCHANGED (no recolour, no scrim).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from vidlore.motion_graphics import _shared as S   # noqa: E402

# The thin-serif body colour at the heart of RC5: a low-contrast warm gray.
THIN_GRAY = (108, 102, 92)
DARK_BG = (24, 20, 16)          # a crushed near-black documentary footage bed
WHITE_BG = (245, 242, 236)      # a clean bright bed


# ───────────────────────── helpers ─────────────────────────
def _flat(w, h, rgb):
    return Image.new("RGB", (w, h), tuple(int(v) for v in rgb))


def _noisy_dark(w, h, base=DARK_BG, amp=100, seed=7):
    """A genuinely BUSY dark bed: crushed mean + real high-frequency texture, the
    footage condition that destroys legibility even at an OK mean contrast."""
    rng = np.random.default_rng(seed)
    arr = np.empty((h, w, 3), np.float32)
    for i in range(3):
        arr[..., i] = base[i]
    arr += rng.normal(0, amp, (h, w, 3)).astype(np.float32)
    return Image.fromarray(np.clip(arr, 0, 255).astype("uint8"))


def _box(img):
    w, h = img.size
    return (int(w * 0.2), int(h * 0.4), int(w * 0.8), int(h * 0.6))


# ───────────────────────── tests ─────────────────────────
def test_wcag_math_anchors():
    """WCAG primitives match the published reference values."""
    # pure black vs pure white is the canonical 21:1
    assert abs(S.contrast_ratio((0, 0, 0), (255, 255, 255)) - 21.0) < 0.01
    # identical colours are 1:1
    assert abs(S.contrast_ratio((120, 120, 120), (120, 120, 120)) - 1.0) < 1e-6
    # relative luminance is ordered and bounded
    assert S.relative_luminance((0, 0, 0)) == 0.0
    assert abs(S.relative_luminance((255, 255, 255)) - 1.0) < 1e-9
    assert S.relative_luminance((255, 0, 0)) < S.relative_luminance((0, 255, 0))


def test_thin_gray_on_dark_is_flagged_failing():
    """THE core failing case: thin warm-gray body on a crushed dark bed must read
    as failing the 4.5:1 body target."""
    r = S.contrast_ratio(THIN_GRAY, DARK_BG)
    assert r < S.CONTRAST_BODY, f"expected fail (<4.5), got {r:.2f}"


def test_repair_brings_thin_gray_to_target():
    """The repair ladder (recolour + restrained scrim) must lift the same case to
    >= 4.5:1, measured against the ACTUAL plate-blended bed the plan produces."""
    img = _flat(640, 360, DARK_BG)
    box = _box(img)
    plan = S.ensure_contrast(None, box, THIN_GRAY, S.box_bg_sampler(img),
                             large=False, img=img)
    # something must have happened (recolour and/or scrim)
    assert plan.text_color != tuple(THIN_GRAY) or plan.scrim is not None, \
        "repair ladder did nothing on a failing case"
    # reconstruct the effective contrast the viewer actually sees
    bg = S.box_bg_stats(img, box)["mean_rgb"]
    if plan.scrim is not None and plan.scrim["alpha"] > 0:
        a = plan.scrim["alpha"]
        pr = plan.scrim["rgb"]
        bg = tuple(b * (1 - a) + p * a for b, p in zip(bg, pr))
    eff = S.contrast_ratio(plan.text_color, bg)
    assert eff >= S.CONTRAST_BODY, f"post-repair only {eff:.2f}, need >=4.5"
    assert plan.ok, "plan should report ok after a successful repair"


def test_clean_high_contrast_left_unchanged():
    """A clean high-contrast case (near-black text on a bright clean bed) must pass
    untouched: no recolour, no scrim, no stroke, no weight bump."""
    img = _flat(640, 360, WHITE_BG)
    box = _box(img)
    text = (14, 12, 10)
    # sanity: this really is high contrast to begin with
    assert S.contrast_ratio(text, WHITE_BG) >= S.CONTRAST_BODY
    plan = S.ensure_contrast(None, box, text, S.box_bg_sampler(img),
                             large=False, img=img)
    assert plan.ok
    assert plan.text_color == tuple(text), "colour was needlessly shifted"
    assert plan.scrim is None, "scrim added to an already-readable clean bed"
    assert plan.stroke_w == 0 and plan.weight_boost == 0, "needless heavy repair"


def test_busy_dark_bed_gets_a_plate():
    """A busy/noisy dark bed should trigger a readability plate (variance is the
    real legibility killer on footage), and the plate must stay a soft wash."""
    img = _noisy_dark(640, 360)
    box = _box(img)
    assert S.is_busy_background(img, box), "fixture should read as busy"
    plan = S.ensure_contrast(None, box, THIN_GRAY, S.box_bg_sampler(img),
                             large=False, img=img)
    assert plan.busy
    assert plan.scrim is not None and plan.scrim["alpha"] > 0
    assert plan.scrim["alpha"] <= S.SCRIM_MAX_ALPHA + 1e-9, "plate too heavy"


def test_large_text_uses_relaxed_target():
    """Large/title text uses the 3:1 target — a contrast that passes 'large' may
    still fail 'body', proving the two thresholds are wired."""
    # find a colour/bed pair whose ratio sits between 3.0 and 4.5
    bg = DARK_BG
    txt = (126, 121, 111)
    r = S.contrast_ratio(txt, bg)
    assert 3.0 <= r < 4.5, f"fixture ratio {r:.2f} not in the (3,4.5) band"
    img = _flat(640, 360, bg)
    box = _box(img)
    big = S.ensure_contrast(None, box, txt, S.box_bg_sampler(img), large=True, img=img)
    small = S.ensure_contrast(None, box, txt, S.box_bg_sampler(img), large=False, img=img)
    assert big.target == S.CONTRAST_LARGE and small.target == S.CONTRAST_BODY
    # large (3:1) passes UNTOUCHED at this ratio; body (4.5:1) must trigger SOME
    # repair (a colour nudge and/or a scrim — whichever the ladder reaches first).
    assert big.text_color == tuple(txt) and big.scrim is None, \
        "large text needlessly repaired at a 3:1-passing ratio"
    body_repaired = (small.text_color != tuple(txt)) or (small.scrim is not None)
    assert body_repaired, "body text should be repaired at this sub-4.5 ratio"
    # and the body repair actually reaches the body target
    assert small.ok and small.ratio >= S.CONTRAST_BODY - 1e-6


def test_body_font_is_sans_and_mobile_legible():
    """readable_body_font must (a) never drop below the mobile px floor and (b) be
    the CLEAN SANS family, not the thin Playfair serif used for headings."""
    h = 1080
    f = S.readable_body_font(h)
    assert f.size >= max(S.BODY_MIN_PX, int(h * S.BODY_MIN_FRAC)), "below mobile floor"
    body_name = f.getname()[0].lower()
    title_name = S.readable_title_font(h).getname()[0].lower()
    assert "playfair" not in body_name, f"body uses the thin serif: {body_name}"
    assert body_name != title_name, "body and heading resolve to the same family"
    # tiny canvas still respects the absolute floor
    assert S.readable_body_font(200).size >= S.BODY_MIN_PX


def test_wrap_caps_line_length():
    """wrap_to_width caps BOTH pixel width and characters-per-line, so a long run
    becomes a sane multi-line measure, never one 70-char line."""
    d = __import__("PIL.ImageDraw", fromlist=["ImageDraw"]).Draw(_flat(8, 8, (0, 0, 0)))
    fnt = S.readable_body_font(1080)
    text = ("the quick brown fox jumps over the lazy dog while reciting "
            "the entire history of cartography in one breath")
    lines = S.wrap_to_width(d, text, fnt, max_w=10_000, max_lines=8)  # width can't bite
    assert len(lines) >= 3, "char cap did not split a long run"
    assert all(len(ln) <= S.MAX_CHARS_PER_LINE for ln in lines[:-1]), \
        "a non-final line exceeded the char cap"
    # an unbreakable token never overflows a narrow box
    hard = S.wrap_to_width(d, "x" * 200, fnt, max_w=120, max_lines=4)
    assert len(hard) >= 2, "unbreakable token was not hard-split"


def test_apply_plan_only_marks_when_needed():
    """apply_plan is a no-op for a passing (scrim-less) plan and actually darkens a
    failing dark-bed plan (so the plate is real, not cosmetic)."""
    # passing plan -> identical pixels
    clean = _flat(320, 200, WHITE_BG)
    box = _box(clean)
    p_ok = S.ensure_contrast(None, box, (14, 12, 10), S.box_bg_sampler(clean), img=clean)
    out = S.apply_plan(clean.copy(), p_ok, box)
    assert np.array_equal(np.asarray(clean), np.asarray(out)), "no-op plan changed pixels"
    # failing MID-TONE plan -> the plated region gets darker (light text => dark
    # plate). A mid-grey bed is where neither pure white nor pure black text wins,
    # so the gate is forced to plate (start ratio ~3.5, under the 4.5 body target).
    mid = _flat(320, 200, (132, 126, 116))
    p_bad = S.ensure_contrast(None, box, (244, 240, 232), S.box_bg_sampler(mid), img=mid)
    assert p_bad.scrim is not None, "mid-tone failing case should force a plate"
    plated = S.apply_plan(mid.copy(), p_bad, box)
    cx, cy = 160, 100
    before = float(np.asarray(mid)[cy, cx].mean())
    after = float(np.asarray(plated)[cy, cx].mean())
    assert after < before - 3.0, f"dark plate did not darken bed ({before:.0f}->{after:.0f})"


def test_helpers_never_crash_on_degenerate_input():
    """The gate must degrade gracefully, never hard-crash (the task's no-crash
    requirement) on empty / off-canvas / tiny boxes and odd colours."""
    img = _flat(64, 64, DARK_BG)
    for box in [(-50, -50, -10, -10), (0, 0, 1, 1), (100, 100, 200, 200), (10, 10, 10, 10)]:
        S.box_bg_stats(img, box)
        plan = S.ensure_contrast(None, box, (130, 120, 110), S.box_bg_sampler(img), img=img)
        S.apply_plan(img.copy(), plan, box)
    # draw_* convenience entries run end-to-end without raising
    S.draw_body(img.copy(), (8, 8), "legible body copy over a dark bed")
    S.draw_headline(img.copy(), (32, 20), "A PREMIUM HEADLINE")
    S.draw_text(img.copy(), (8, 8), "run", S.readable_body_font(64), fill=THIN_GRAY)


# ───────────────────────── runner ─────────────────────────
def main() -> int:
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    passed = failed = 0
    failures = []
    for fn in tests:
        name = fn.__name__
        try:
            fn()
            passed += 1
            print(f"PASS  {name}")
        except AssertionError as ex:
            failed += 1
            failures.append((name, f"assert: {ex}"))
            print(f"FAIL  {name}  ->  {ex}")
        except Exception as ex:                                # noqa: BLE001
            failed += 1
            failures.append((name, f"{type(ex).__name__}: {ex}"))
            print(f"ERROR {name}  ->  {type(ex).__name__}: {ex}")
    total = passed + failed
    print("\n" + "=" * 60)
    print(f"RC5.1 MG text-readability QA:  {passed}/{total} passed, {failed} failed")
    if failures:
        print("-" * 60)
        for name, msg in failures:
            print(f"  FAIL {name}: {msg}")
    print("=" * 60)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
