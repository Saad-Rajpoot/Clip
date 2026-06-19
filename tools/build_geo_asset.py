"""Build a COMPACT bundled geographic asset from Natural Earth 110m countries.

Input : /tmp/ne110.geojson  (public-domain Natural Earth 1:110m admin-0 countries)
Output: vidlore/assets/geo/world_countries.json

Natural Earth is PUBLIC DOMAIN (no attribution, no license restriction) — safe to
bundle. We Douglas-Peucker-simplify every polygon ring, drop tiny islands, round
coordinates to 2 dp (~1.1 km), and keep per-country {name, iso3, continent, bbox,
centroid, rings}. Result is a few hundred KB — small enough to ship in all 3 trees
and load instantly, deterministic, offline. This is the real-geography backbone
the map family projects onto (replacing synthetic blobs + normalized guesses).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

SRC = Path("/tmp/ne110.geojson")
OUT = Path("vidlore/assets/geo/world_countries.json")
TOL = 0.35          # Douglas-Peucker tolerance in degrees (~38 km) — recognizable
MIN_RING_AREA = 0.6  # drop rings whose bbox area (deg^2) is below this (tiny isles)
ROUND = 2           # coordinate decimals


def _perp(p, a, b):
    (px, py), (ax, ay), (bx, by) = p, a, b
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _dp(pts, tol):
    """Iterative Douglas-Peucker (avoids recursion limits)."""
    if len(pts) < 3:
        return pts
    keep = [False] * len(pts)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        i, j = stack.pop()
        dmax, idx = 0.0, -1
        for k in range(i + 1, j):
            d = _perp(pts[k], pts[i], pts[j])
            if d > dmax:
                dmax, idx = d, k
        if dmax > tol and idx != -1:
            keep[idx] = True
            stack.append((i, idx))
            stack.append((idx, j))
    return [p for p, k in zip(pts, keep) if k]


def _ring_bbox_area(ring):
    xs = [p[0] for p in ring]; ys = [p[1] for p in ring]
    return (max(xs) - min(xs)) * (max(ys) - min(ys))


def _polys(geom):
    """Yield each outer ring (lon,lat lists) from a Polygon/MultiPolygon."""
    t = geom["type"]; c = geom["coordinates"]
    if t == "Polygon":
        yield c[0]
    elif t == "MultiPolygon":
        for poly in c:
            yield poly[0]


def main():
    gj = json.loads(SRC.read_text())
    out = {}
    total_pts = 0
    for feat in gj["features"]:
        p = feat["properties"]
        iso = p.get("ISO_A3") or p.get("ADM0_A3") or p.get("NAME")
        name = p.get("NAME") or p.get("ADMIN") or iso
        if not iso or iso == "-99":
            iso = (p.get("NAME") or "").upper()[:3]
        rings = []
        for ring in _polys(feat["geometry"]):
            if _ring_bbox_area(ring) < MIN_RING_AREA:
                continue
            s = _dp([(round(x, ROUND), round(y, ROUND)) for x, y in ring], TOL)
            if len(s) >= 4:
                rings.append(s)
        if not rings:
            # keep the single largest ring even if "small" (micro-states)
            allr = list(_polys(feat["geometry"]))
            if allr:
                big = max(allr, key=_ring_bbox_area)
                s = _dp([(round(x, ROUND), round(y, ROUND)) for x, y in big], TOL)
                if len(s) >= 4:
                    rings = [s]
        if not rings:
            continue
        xs = [x for r in rings for x, _ in r]; ys = [y for r in rings for _, y in r]
        bbox = [min(xs), min(ys), max(xs), max(ys)]
        # area-weighted-ish centroid (use largest ring's vertex mean)
        big = max(rings, key=_ring_bbox_area)
        cx = sum(x for x, _ in big) / len(big)
        cy = sum(y for _, y in big) / len(big)
        total_pts += sum(len(r) for r in rings)
        out[iso] = {"name": name, "continent": p.get("CONTINENT", ""),
                    "bbox": [round(v, 2) for v in bbox],
                    "centroid": [round(cx, 2), round(cy, 2)], "rings": rings}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, separators=(",", ":")))
    kb = OUT.stat().st_size / 1024
    print(f"countries: {len(out)}  total ring points: {total_pts}  size: {kb:.0f} KB")
    # spot-check a few documentary-relevant countries
    for q in ("IRN", "IRQ", "USA", "RUS", "CHN", "GBR", "DEU", "FRA"):
        e = out.get(q)
        print(f"  {q}: {'OK '+e['name']+' bbox='+str(e['bbox']) if e else 'MISSING'}")


if __name__ == "__main__":
    main()
