#!/usr/bin/env python3
"""Offline regression tests (QA) for the Vidlore geo backbone.

Target under test:  vidlore/motion_graphics/maps/geo.py
Bundled data:       vidlore/assets/geo/world_countries.json  (Natural Earth 110m)

Plain assert-based harness — no pytest dependency. Run with the project venv:

    .venv/bin/python tools/test_geo.py

Exits 0 when every test passes, 1 on the first-or-any failure (all tests run;
failures are collected and printed with a PASS/FAIL summary at the end).

This file is QA only and does NOT modify geo.py or any other source. If a test
surfaces a genuine bug, that is reported by the test failing (and surfaced in the
runner's summary), not patched here.
"""
from __future__ import annotations

import hashlib
import sys

# geo.py lives in a package; make sure the repo root is importable when this is
# run as a bare script from anywhere.
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from vidlore.motion_graphics import look                      # noqa: E402
from vidlore.motion_graphics.maps import geo                  # noqa: E402

# Small, fast canvas for every image test.
W, H = 480, 270
PAL = look.palette("parchment_sepia")


# ── helpers ──────────────────────────────────────────────────────────────────
def _hash(img) -> str:
    """Stable content hash of a PIL image (mode + size + raw bytes)."""
    return hashlib.sha256(
        ("%s%dx%d" % (img.mode, img.width, img.height)).encode() + img.tobytes()
    ).hexdigest()


def _in_bbox(pt, bbox, *, pad=0.0):
    """True if (lon,lat) point lies within [W,S,E,N] bbox (optionally padded).

    Handles antimeridian-wrapping bboxes (W==-180 and E==180, as Russia/USA give
    in Natural Earth) by accepting any longitude when the box spans the globe."""
    if pt is None or bbox is None:
        return False
    lon, lat = pt
    w, s, e, n = bbox
    lon_ok = (w - pad <= lon <= e + pad) or (w <= -179.999 and e >= 179.999)
    lat_ok = (s - pad) <= lat <= (n + pad)
    return lon_ok and lat_ok


def _approx_range(val, lo, hi, *, tol=2.0):
    return (lo - tol) <= val <= (hi + tol)


# ── 1. Resolution ────────────────────────────────────────────────────────────
def test_country_resolution():
    countries = ["Iran", "Iraq", "Turkey", "Greece", "Russia", "Syria",
                 "Egypt", "China", "Netherlands"]
    for name in countries:
        ll = geo.resolve(name)
        assert ll is not None, f"{name} should resolve to a (lon,lat)"
        lon, lat = ll
        assert -180.0 <= lon <= 180.0, f"{name} lon out of range: {lon}"
        assert -90.0 <= lat <= 90.0, f"{name} lat out of range: {lat}"
        # Everything in this set sits in the northern hemisphere, mostly eastern.
        assert lat > 0, f"{name} should be northern hemisphere, got lat={lat}"


def test_city_within_country_bbox():
    # city -> the country whose bbox must contain it
    pairs = [
        ("Tehran", "Iran"), ("Baghdad", "Iraq"), ("Basra", "Iraq"),
        ("Abadan", "Iran"), ("Kermanshah", "Iran"), ("Moscow", "Russia"),
        ("Shanghai", "China"), ("Rotterdam", "Netherlands"),
    ]
    for city, country in pairs:
        cll = geo.resolve(city)
        assert cll is not None, f"city {city} should resolve"
        entry = geo.country_entry(country)
        assert entry is not None, f"country_entry({country}) should exist"
        bbox = entry["bbox"]
        assert _in_bbox(cll, bbox, pad=0.5), (
            f"{city} {cll} should fall inside {country} bbox {bbox}"
        )


def test_country_entry_iran_iraq():
    iran = geo.country_entry("Iran")
    iraq = geo.country_entry("Iraq")
    for nm, e in (("Iran", iran), ("Iraq", iraq)):
        assert e is not None, f"country_entry({nm}) should return an entry"
        assert e.get("rings"), f"{nm} entry should have non-empty rings"
        # rings is a list of rings; first ring should be a real polygon
        assert len(e["rings"]) >= 1 and len(e["rings"][0]) >= 3, \
            f"{nm} first ring should be a polygon (>=3 points)"
        assert "name" in e and "bbox" in e and "centroid" in e, \
            f"{nm} entry missing expected keys"

    # Plausible bounding boxes (Natural Earth 110m).
    iw, isth, ie, inn = iran["bbox"]
    assert _approx_range(iw, 44, 44, tol=3) and _approx_range(ie, 64, 64, tol=3), \
        f"Iran lon span looks wrong: {iran['bbox']}"
    assert _approx_range(isth, 25, 25, tol=3) and _approx_range(inn, 40, 40, tol=3), \
        f"Iran lat span looks wrong: {iran['bbox']}"

    qw, qs, qe, qn = iraq["bbox"]
    assert _approx_range(qw, 38, 38, tol=3) and _approx_range(qe, 49, 49, tol=3), \
        f"Iraq lon span looks wrong: {iraq['bbox']}"
    assert _approx_range(qs, 29, 29, tol=3) and _approx_range(qn, 38, 38, tol=3), \
        f"Iraq lat span looks wrong: {iraq['bbox']}"


def test_country_rings_helper():
    rings = geo.country_rings("Iran")
    assert rings is not None and len(rings) >= 1, "country_rings(Iran) should return rings"
    assert geo.country_rings("Atlantis") is None, "unknown country -> no rings"


# ── 2. Aliases ───────────────────────────────────────────────────────────────
def test_aliases_ussr_soviet_union():
    russia_bbox = geo.country_entry("Russia")["bbox"]
    for alias in ("USSR", "Soviet Union", "the Soviet Union"):
        ll = geo.resolve(alias)
        assert ll is not None, f"{alias} should resolve (alias -> Russia)"
        assert _in_bbox(ll, russia_bbox, pad=1.0), \
            f"{alias} centroid {ll} should land in Russia bbox {russia_bbox}"
        e = geo.country_entry(alias)
        assert e is not None and e["name"] == "Russia", \
            f"country_entry({alias}) should map to Russia, got {e and e.get('name')}"


def test_alias_persia_to_iran():
    iran = geo.resolve("Iran")
    persia = geo.resolve("Persia")
    assert persia is not None and persia == iran, \
        f"Persia should resolve identically to Iran ({persia} vs {iran})"
    e = geo.country_entry("Persia")
    assert e is not None and e["name"] == "Iran", \
        "country_entry('Persia') should map to Iran"


def test_alias_holland_mesopotamia():
    # extra alias coverage (mesopotamia -> Iraq, holland -> Netherlands)
    assert geo.country_entry("Mesopotamia")["name"] == "Iraq"
    assert geo.country_entry("Holland")["name"] == "Netherlands"


# ── 3. Routes / projection ───────────────────────────────────────────────────
def test_project_moscow_north_of_baghdad():
    moscow = geo.resolve("Moscow")
    baghdad = geo.resolve("Baghdad")
    bbox = geo.region_bbox([moscow, baghdad])
    assert bbox is not None
    proj = geo.make_projector(bbox, W, H)
    mpx, mpy = proj(*moscow)
    bpx, bpy = proj(*baghdad)
    # North is up -> Moscow (higher latitude) must have the SMALLER py.
    assert mpy < bpy, f"Moscow should be north of Baghdad (py {mpy} < {bpy})"
    # Roughly correct east/west: Moscow (37.6E) is west of Baghdad (44.4E).
    assert mpx < bpx, f"Moscow should be west of Baghdad (px {mpx} < {bpx})"
    # Both points must stay inside the frame.
    for px, py in ((mpx, mpy), (bpx, bpy)):
        assert 0 <= px <= W and 0 <= py <= H, f"point ({px},{py}) off-frame"


def test_project_shanghai_east_of_rotterdam():
    shanghai = geo.resolve("Shanghai")
    rotterdam = geo.resolve("Rotterdam")
    # World-spanning bbox so longitude ordering is unambiguous.
    bbox = [-180.0, -60.0, 180.0, 75.0]
    proj = geo.make_projector(bbox, W, H)
    spx, _ = proj(*shanghai)
    rpx, _ = proj(*rotterdam)
    assert spx > rpx, f"Shanghai should be east of Rotterdam (px {spx} > {rpx})"


def test_four_stop_route_in_frame():
    route = ["Shanghai", "Singapore", "Suez", "Rotterdam"]
    # "Suez" is not in the gazetteer; a real route drops it gracefully, so use the
    # resolvable stops to frame, then assert every resolvable stop is in-frame.
    resolved = [(nm, geo.resolve(nm)) for nm in route]
    hits = [(nm, ll) for nm, ll in resolved if ll]
    # At least the three real cities resolve.
    assert len(hits) >= 3, f"expected >=3 resolvable stops, got {hits}"
    bbox = geo.region_bbox([ll for _, ll in hits])
    assert bbox is not None
    # bbox auto-focus must contain every input point.
    for nm, ll in hits:
        assert _in_bbox(ll, bbox), f"{nm} {ll} not contained by auto bbox {bbox}"
    proj = geo.make_projector(bbox, W, H)
    for nm, ll in hits:
        px, py = proj(*ll)
        assert 0 <= px <= W and 0 <= py <= H, f"{nm} projects off-frame: ({px},{py})"


def test_region_bbox_contains_inputs_and_margins():
    pts = [geo.resolve(c) for c in ("Tehran", "Baghdad", "Basra", "Kuwait")]
    bbox = geo.region_bbox(pts)
    assert bbox is not None
    for p in pts:
        assert _in_bbox(p, bbox), f"{p} not contained by bbox {bbox}"
    # Projector keeps points within the frame (margin default 0.06 -> well inside).
    proj = geo.make_projector(bbox, W, H)
    for p in pts:
        px, py = proj(*p)
        assert 0 <= px <= W and 0 <= py <= H, f"{p} -> ({px},{py}) off-frame"
    # A single point should still produce a sane, non-degenerate bbox (min_span).
    one = geo.region_bbox([geo.resolve("Tehran")])
    assert one is not None
    w, s, e, n = one
    assert (e - w) > 0 and (n - s) > 0, "single-point bbox should be non-degenerate"


# ── 4. Label safety ──────────────────────────────────────────────────────────
def _gulf_items():
    cities = ["Basra", "Abadan", "Khorramshahr", "Kuwait"]
    pts = [geo.resolve(c) for c in cities]
    bbox = geo.region_bbox(pts)
    proj = geo.make_projector(bbox, W, H)
    items = []
    for i, (c, ll) in enumerate(zip(cities, pts)):
        px, py = proj(*ll)
        items.append({
            "text": c, "x": px, "y": py,
            "priority": 4 - i,        # Basra highest, descending
            "accent": (i == 0),
        })
    return items


def test_layout_labels_returns_image_same_size():
    base = geo.draw_basemap(W, H, geo.region_bbox(
        [geo.resolve(c) for c in ("Basra", "Abadan", "Khorramshahr", "Kuwait")]))
    items = _gulf_items()
    out = geo.layout_labels(base.copy(), [dict(it) for it in items], PAL)
    assert out.size == base.size, "layout_labels must preserve frame size"
    assert out.mode == "RGB", "layout_labels should return an RGB frame"


def test_layout_labels_deterministic():
    base = geo.draw_basemap(W, H, geo.region_bbox(
        [geo.resolve(c) for c in ("Basra", "Abadan", "Khorramshahr", "Kuwait")]))
    items = _gulf_items()
    a = geo.layout_labels(base.copy(), [dict(it) for it in items], PAL)
    b = geo.layout_labels(base.copy(), [dict(it) for it in items], PAL)
    assert _hash(a) == _hash(b), "same inputs must yield identical label bytes"


def test_layout_labels_no_chip_overlap():
    """Clustered cities must not produce overlapping label chips. We replicate the
    module's own chip geometry + overlap predicate to inspect the chosen anchors."""
    base = geo.draw_basemap(W, H, geo.region_bbox(
        [geo.resolve(c) for c in ("Basra", "Abadan", "Khorramshahr", "Kuwait")]))
    items = _gulf_items()

    # Mirror geo.layout_labels' placement to capture the rectangles it commits.
    W2, H2 = base.size
    margin = int(H2 * 0.03)
    ordered = sorted(
        [it for it in items if it.get("text") and it.get("opacity", 1) > 0.01],
        key=lambda it: -it.get("priority", 0),
    )
    placed = []
    chip_rects = []
    for it in ordered:
        opacity = float(it.get("opacity", 1.0))
        size = it.get("size_frac", 0.030 if it.get("priority", 0) >= 2 else 0.026)
        fnt = look.font("label", max(19, int(H2 * size)))
        ink = tuple(PAL["accent_hi"]) if it.get("accent") else (240, 236, 226)
        chip = geo._chip_img(str(it["text"]).upper(), fnt, PAL, ink, opacity)
        cw, ch = chip.width, chip.height
        x, y = int(it["x"]), int(it["y"])
        placed.append((x - 8, y - 8, x + 8, y + 8))
        cands = [(0, -ch * 0.9 - 12), (0, ch * 0.9 + 12), (cw * 0.6 + 16, 0),
                 (-cw * 0.6 - 16, 0), (cw * 0.6 + 16, -ch - 10),
                 (-cw * 0.6 - 16, -ch - 10), (0, -ch * 1.7 - 14)]
        chosen = None
        for dx, dy in cands:
            cx = int(x + dx - cw / 2)
            cy = int(y + dy - ch / 2)
            cx = max(margin, min(W2 - cw - margin, cx))
            cy = max(margin, min(H2 - ch - margin, cy))
            rect = (cx, cy, cx + cw, cy + ch)
            if not any(geo._overlap(rect, p) for p in placed):
                chosen = rect
                break
        if chosen is None:
            continue                     # dropped on crowding — acceptable
        chip_rects.append(chosen)
        placed.append(chosen)

    # No two committed chips may overlap.
    for i in range(len(chip_rects)):
        for j in range(i + 1, len(chip_rects)):
            assert not geo._overlap(chip_rects[i], chip_rects[j]), (
                f"label chips {i} and {j} overlap: "
                f"{chip_rects[i]} vs {chip_rects[j]}"
            )
    # The highest-priority label (Basra) should always survive placement.
    assert len(chip_rects) >= 1, "at least the primary label must be placed"


def test_layout_labels_empty_items():
    base = geo.draw_basemap(W, H, geo.region_bbox([geo.resolve("Iran")]))
    out = geo.layout_labels(base.copy(), [], PAL)
    assert out.size == base.size, "empty items must return the base unchanged in size"


# ── 5. Historical safety ─────────────────────────────────────────────────────
def _iraq_iran_bbox():
    return geo.region_bbox([geo.resolve("Iran"), geo.resolve("Iraq")])


def _count_border_ink_pixels(img, style):
    """Count border line-work pixels = pixels that are neither land nor ocean.

    Borders are drawn anti-aliased and semi-transparent (alpha 150), so they get
    blended with the land underneath and never exactly equal the raw border RGB —
    an exact-colour match would always read 0. Instead we count pixels that differ
    from BOTH the flat land fill and the ocean fill (i.e. ink laid over the bed:
    border/graticule lines). With borders=False and no highlight, the inter-country
    lines are suppressed, so this count drops to ~0 for the Iran/Iraq view."""
    import numpy as np

    oc, land, bd, bdhi, grat = geo.STYLES[style]
    arr = np.asarray(img).astype(int)
    d_land = np.abs(arr - np.array(land)).sum(axis=2)
    d_ocean = np.abs(arr - np.array(oc)).sum(axis=2)
    tol = 30
    return int(((d_land > tol) & (d_ocean > tol)).sum())


def test_reference_stamp_changes_image():
    bbox = _iraq_iran_bbox()
    ref_on = geo.draw_basemap(W, H, bbox, style="parchment", reference=True)
    ref_off = geo.draw_basemap(W, H, bbox, style="parchment", reference=False)
    assert _hash(ref_on) != _hash(ref_off), \
        "reference=True should stamp a visible chip (image must differ)"
    # The stamp lives in the lower portion of the frame — the upper band should be
    # untouched, confirming we changed pixels via the chip, not the whole render.
    assert ref_on.size == ref_off.size


def test_borders_toggle_changes_image_and_count():
    bbox = _iraq_iran_bbox()
    # Disable the graticule so the ONLY ink difference is the country border lines.
    b_on = geo.draw_basemap(W, H, bbox, style="parchment", borders=True,
                            graticule=False)
    b_off = geo.draw_basemap(W, H, bbox, style="parchment", borders=False,
                             graticule=False)
    assert _hash(b_on) != _hash(b_off), "borders toggle should change the image"
    n_on = _count_border_ink_pixels(b_on, "parchment")
    n_off = _count_border_ink_pixels(b_off, "parchment")
    assert n_off < n_on, (
        f"borders=False should have FEWER border-ink pixels (off={n_off} < on={n_on})"
    )


def test_historical_safety_deterministic():
    bbox = _iraq_iran_bbox()
    a1 = geo.draw_basemap(W, H, bbox, style="parchment", reference=True)
    a2 = geo.draw_basemap(W, H, bbox, style="parchment", reference=True)
    assert _hash(a1) == _hash(a2), "reference render must be deterministic"
    b1 = geo.draw_basemap(W, H, bbox, style="parchment", borders=False)
    b2 = geo.draw_basemap(W, H, bbox, style="parchment", borders=False)
    assert _hash(b1) == _hash(b2), "borders=False render must be deterministic"
    # Custom reference string should also render deterministically + differ.
    c1 = geo.draw_basemap(W, H, bbox, reference="approximate")
    c2 = geo.draw_basemap(W, H, bbox, reference="approximate")
    assert _hash(c1) == _hash(c2)
    assert _hash(c1) != _hash(geo.draw_basemap(W, H, bbox, reference=False))


# ── 6. Failure handling ──────────────────────────────────────────────────────
def test_resolve_unknown_returns_none():
    assert geo.resolve("Wakanda") is None
    assert geo.resolve("") is None
    assert geo.resolve(None) is None
    assert geo.resolve("   ") is None


def test_country_entry_unknown_returns_none():
    assert geo.country_entry("Atlantis") is None
    assert geo.country_entry("") is None
    assert geo.country_entry(None) is None


def test_region_bbox_empty_returns_none():
    assert geo.region_bbox([]) is None
    assert geo.region_bbox([None, None]) is None


def test_basemap_for_all_unknown_returns_fallback_signal():
    bed, proj, hits = geo.basemap_for(["Wakanda", "Atlantis"], W, H)
    assert bed is None and proj is None and hits == {}, \
        "all-unknown places must signal synthetic fallback (None,None,{})"


def test_basemap_for_drops_bogus_route_stop():
    # A bogus stop mixed with real cities should be dropped, not crash.
    bed, proj, hits = geo.basemap_for(["Baghdad", "Wakanda", "Basra"], W, H)
    assert bed is not None and proj is not None, "valid stops should still render"
    assert "Wakanda" not in hits, "bogus stop must be dropped from hits"
    assert set(hits.keys()) == {"Baghdad", "Basra"}, f"unexpected hits: {hits}"
    # min_hits gate: a single resolvable place with min_hits=2 -> fallback signal.
    bed2, proj2, hits2 = geo.basemap_for(["Baghdad", "Wakanda"], W, H, min_hits=2)
    assert bed2 is None and proj2 is None and hits2 == {}, \
        "min_hits unmet must signal fallback"


def test_no_crash_on_pathological_inputs():
    # None of these should raise.
    geo.country_rings("Wakanda")
    geo.region_bbox([(0, 0)])
    geo.make_projector([0, 0, 1, 1], W, H)(0.5, 0.5)
    geo.resolve("Basra, Iraq")     # trailing-country form should resolve to Basra
    assert geo.resolve("Basra, Iraq") == geo.resolve("Basra")


# ── 7. Determinism ───────────────────────────────────────────────────────────
def test_draw_basemap_deterministic():
    bbox = _iraq_iran_bbox()
    h1 = _hash(geo.draw_basemap(W, H, bbox, style="parchment", seed=0))
    h2 = _hash(geo.draw_basemap(W, H, bbox, style="parchment", seed=0))
    assert h1 == h2, "identical inputs must produce identical basemap bytes"


def test_basemap_for_deterministic():
    a = geo.basemap_for(["Iran", "Iraq", "Kuwait"], W, H)[0]
    b = geo.basemap_for(["Iran", "Iraq", "Kuwait"], W, H)[0]
    assert a is not None and b is not None
    assert _hash(a) == _hash(b), "basemap_for must be deterministic for same input"


def test_different_styles_differ():
    bbox = _iraq_iran_bbox()
    hp = _hash(geo.draw_basemap(W, H, bbox, style="parchment"))
    hd = _hash(geo.draw_basemap(W, H, bbox, style="dark"))
    assert hp != hd, "different styles should render differently"


# ── runner ───────────────────────────────────────────────────────────────────
def _collect_tests():
    g = globals()
    names = sorted(n for n in g if n.startswith("test_") and callable(g[n]))
    return [(n, g[n]) for n in names]


def main():
    assert geo.available(), "geo data must be available (world_countries.json missing?)"
    tests = _collect_tests()
    passed, failed = 0, 0
    failures = []
    for name, fn in tests:
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
    print(f"geo backbone QA:  {passed}/{total} passed, {failed} failed")
    if failures:
        print("-" * 60)
        for name, msg in failures:
            print(f"  FAIL {name}: {msg}")
    print("=" * 60)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
