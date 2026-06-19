"""Regression: supply_route_dashes must not divide-by-zero on a zero-length route.

TRIGGER (real render). The motion-graphics director derives
    source="Soviet Union", dests=["Iraq", "Moscow", "Baghdad"]
from narration like "...funnelling arms shipments from Moscow to Baghdad...".
The gazetteer (maps/geo.py) DELIBERATELY aliases both "Soviet Union" and "Moscow"
to Moscow's coordinate (37.62, 55.75), so the source and the "Moscow" destination
project to the SAME pixel. The traveling-dots block then computed
    ux, uy = (dp - src_pt) / L     with  L == math.hypot(0, 0) == 0
-> "float division by zero", which crashed render() and made the whole
supply_route_dashes card fall back to footage (manifest: "render error: float
division by zero").

FIX (maps/supply_route_dashes.py), two layers:
  * _drop_coincident() removes any destination that lands on the source pixel
    BEFORE drawing — the zero-length "Moscow" route is dropped (and no redundant
    pin + label is stacked on the source node).
  * the dots block guards `prog > 0.25 and L >= 1.0` (mirrors _dashed's
    `if L < 1: return`) as defence-in-depth, so a zero-length route can never
    reach the division even if de-dup is bypassed.

This test does NOT modify geo.py: the USSR/Soviet-Union -> Moscow coordinate
aliasing is intentional and is asserted here as the (intended) trigger condition.

Run:  .venv/bin/python tools/test_supply_route_zero_length.py
Exit 0 when every check passes, 1 otherwise (all checks run; summary at the end).
"""
import math
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vidlore.motion_graphics.maps import geo                         # noqa: E402
from vidlore.motion_graphics.maps import supply_route_dashes as srd  # noqa: E402

SOURCE = "Soviet Union"
DESTS = ["Iraq", "Moscow", "Baghdad"]
PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def _project(source, dests, w, h):
    """Replicate render()'s real-geo projection so the test sees the same pixels
    render() does (resolve -> region_bbox(pad_frac=0.5) -> make_projector)."""
    src = geo.resolve(source)
    gd = [(d, geo.resolve(d)) for d in dests]
    gd = [(d, ll) for d, ll in gd if ll]
    bbox = geo.region_bbox([src] + [ll for _, ll in gd], pad_frac=0.5)
    proj = geo.make_projector(bbox, w, h)
    return proj(*src), [proj(*ll) for _, ll in gd], [d for d, _ in gd]


def test_geo_alias_is_the_trigger():
    """Lock the deliberate gazetteer aliasing that makes source pixel == dest
    pixel. (geo.py must NOT be 'fixed' to make these differ — it is intentional.)"""
    print("\ngeo trigger (intended USSR -> Moscow aliasing):")
    s = geo.resolve(SOURCE)
    m = geo.resolve("Moscow")
    check("Soviet Union resolves", s is not None, str(s))
    check("Moscow resolves", m is not None, str(m))
    check("Soviet Union and Moscow share one coordinate (the alias)", s == m,
          f"{s} == {m}")
    src_pt, dest_pts, lbls = _project(SOURCE, DESTS, 960, 540)
    j = lbls.index("Moscow")
    L = math.hypot(dest_pts[j][0] - src_pt[0], dest_pts[j][1] - src_pt[1])
    check("'Moscow' destination projects ONTO the source pixel (L == 0)",
          L == 0.0, f"L={L}")


def test_drop_coincident_helper():
    """The de-dup primitive drops only the source-coincident destination."""
    print("\n_drop_coincident helper:")
    src_pt, dest_pts, lbls = _project(SOURCE, DESTS, 960, 540)
    kp, kl = srd._drop_coincident(src_pt, dest_pts, lbls)
    check("coincident 'Moscow' dropped", "Moscow" not in kl, str(kl))
    check("non-coincident dests kept (order preserved)", kl == ["Iraq", "Baghdad"],
          str(kl))
    check("kept points stay in sync with kept labels", len(kp) == len(kl) == 2)
    # edge cases
    none = srd._drop_coincident((0.0, 0.0), [(40.0, 0.0), (0.0, 90.0)], ["A", "B"])
    check("nothing dropped when none coincide", none[1] == ["A", "B"], str(none[1]))
    allc = srd._drop_coincident((5.0, 5.0), [(5.0, 5.0), (5.4, 5.0)], ["X", "Y"])
    check("all-coincident (incl. sub-pixel) -> empty lists", allc == ([], []),
          str(allc))
    mism = srd._drop_coincident((0.0, 0.0), [(0.0, 0.0)], ["A", "B"])
    check("length-mismatch returned unchanged (defensive)", mism[1] == ["A", "B"],
          str(mism))


def _render_exact(tmp):
    """Render the exact coincident case on a small/fast canvas. dur/fps are large
    enough that the staggered 'Moscow' route (j=1, t_start=0.85) advances past the
    dots threshold (prog > 0.25 -> t > ~0.94), i.e. it actually exercises the
    zero-length code path rather than skipping it."""
    out = Path(tmp) / "supply.mp4"
    res = srd.render(str(out), source=SOURCE, dests=DESTS,
                     w=480, h=270, dur=2.5, fps=8, palette_name="cold_steel")
    return out, res


def test_coincident_render_does_not_raise():
    """The exact real-render case must render end-to-end without raising."""
    print("\nrender exact coincident case (de-dup ON):")
    td = tempfile.mkdtemp()
    raised, out, res = None, None, {}
    try:
        out, res = _render_exact(td)
    except Exception as ex:                                       # noqa: BLE001
        raised = ex
    check("render() did not raise", raised is None,
          "" if raised is None else f"{type(raised).__name__}: {raised}")
    if raised is None:
        check("render reported ok", bool(res.get("ok")), res.get("err", ""))
        check("frames were produced", res.get("frames", 0) > 0, str(res.get("frames")))
        check("mp4 written", out.exists())
    shutil.rmtree(td, ignore_errors=True)


def test_dots_guard_protects_without_dedupe():
    """Defence-in-depth: with de-dup disabled, the dots block's `L >= 1.0` guard
    must STILL keep the zero-length 'Moscow' route from dividing by zero."""
    print("\nL>=1 dots-guard alone (de-dup monkeypatched OFF):")
    orig = srd._drop_coincident
    srd._drop_coincident = lambda src_pt, dest_pts, dests_lbl, **k: (dest_pts, dests_lbl)
    td = tempfile.mkdtemp()
    raised, res = None, {}
    try:
        _out, res = _render_exact(td)
    except Exception as ex:                                       # noqa: BLE001
        raised = ex
    finally:
        srd._drop_coincident = orig
        shutil.rmtree(td, ignore_errors=True)
    check("render() did not raise with de-dup OFF (guard holds)", raised is None,
          "" if raised is None else f"{type(raised).__name__}: {raised}")
    if raised is None:
        check("render still ok with the guard alone", bool(res.get("ok")),
              res.get("err", ""))


def main():
    assert geo.available(), "geo data must be available (world_countries.json missing?)"
    test_geo_alias_is_the_trigger()
    test_drop_coincident_helper()
    test_coincident_render_does_not_raise()
    test_dots_guard_protects_without_dedupe()
    print(f"\n{'=' * 56}\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILED:", ", ".join(FAIL))
        return 1
    print("ALL GREEN")
    return 0


if __name__ == "__main__":
    sys.exit(main())
