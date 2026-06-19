"""Real-geography backbone for the map family.

Replaces synthetic landmass blobs + hand-guessed normalized positions with REAL
country boundaries (bundled public-domain Natural Earth 110m) + a gazetteer of
country centroids and major cities, an equirectangular projection with latitude
correction, automatic region-focus (bounding box of the places in play), and a
real coastline/border basemap renderer in several documentary styles.

Design: pure-local (json + PIL + numpy), deterministic, offline, Python-3.9 safe.
Every map primitive can:  resolve(name)->(lon,lat) · region_bbox(points) ·
make_projector(bbox,w,h) · draw_basemap(...).  When a place can't be resolved the
caller falls back to the prior synthetic bed + normalized positions, so nothing
regresses on unknown/abstract scenes.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .. import look

_ASSET = Path(__file__).resolve().parent.parent.parent / "assets" / "geo" / "world_countries.json"
_DATA = None

# ── country-name aliases → the NAME used in Natural Earth ────────────────────
_ALIASES = {
    "usa": "United States of America", "us": "United States of America",
    "united states": "United States of America", "america": "United States of America",
    "uk": "United Kingdom", "britain": "United Kingdom", "england": "United Kingdom",
    "great britain": "United Kingdom", "russia": "Russia", "ussr": "Russia",
    "soviet union": "Russia", "the soviet union": "Russia", "uae": "United Arab Emirates",
    "south korea": "South Korea", "north korea": "North Korea", "drc": "Dem. Rep. Congo",
    "congo": "Dem. Rep. Congo", "czech republic": "Czechia", "burma": "Myanmar",
    "persia": "Iran", "mesopotamia": "Iraq", "holland": "Netherlands",
    "ivory coast": "Côte d'Ivoire", "vietnam": "Vietnam", "syria": "Syria",
    "west germany": "Germany", "east germany": "Germany", "prussia": "Germany",
}

# ── major-city gazetteer (lon, lat) — documentary-relevant capitals + cities ──
_CITIES = {
    # Middle East (Iran-Iraq theatre + region)
    "baghdad": (44.36, 33.31), "tehran": (51.39, 35.69), "basra": (47.78, 30.50),
    "abadan": (48.30, 30.34), "khorramshahr": (48.18, 30.44), "kuwait": (47.98, 29.38),
    "kuwait city": (47.98, 29.38), "tabriz": (46.29, 38.07), "kermanshah": (47.06, 34.31),
    "dezful": (48.40, 32.38), "mosul": (43.13, 36.34), "kirkuk": (44.39, 35.47),
    "shiraz": (52.53, 29.59), "isfahan": (51.67, 32.65), "ahvaz": (48.69, 31.32),
    "damascus": (36.29, 33.51), "riyadh": (46.72, 24.69), "jerusalem": (35.21, 31.77),
    "tel aviv": (34.78, 32.08), "cairo": (31.24, 30.04), "ankara": (32.85, 39.93),
    "istanbul": (28.98, 41.01), "beirut": (35.50, 33.89), "amman": (35.93, 31.95),
    "dubai": (55.27, 25.20), "doha": (51.53, 25.29), "muscat": (58.41, 23.59),
    "suez": (32.55, 29.97), "port said": (32.30, 31.27), "aden": (45.04, 12.79),
    "alexandria": (29.92, 31.20),
    "sanaa": (44.21, 15.35), "kabul": (69.21, 34.56), "islamabad": (73.06, 33.69),
    # Europe
    "london": (-0.13, 51.51), "paris": (2.35, 48.86), "berlin": (13.40, 52.52),
    "moscow": (37.62, 55.75),
    # USSR/Soviet Union as a single POINT → the European core near Moscow (NOT
    # Russia's Siberian centroid); country_entry() still returns Russia for region
    # framing. Historical Soviet *borders* are handled via reference mode.
    "ussr": (37.62, 55.75), "soviet union": (37.62, 55.75),
    "the soviet union": (37.62, 55.75), "rome": (12.50, 41.90), "madrid": (-3.70, 40.42),
    "vienna": (16.37, 48.21), "warsaw": (21.01, 52.23), "kyiv": (30.52, 50.45),
    "kiev": (30.52, 50.45), "amsterdam": (4.90, 52.37), "rotterdam": (4.48, 51.92),
    "brussels": (4.35, 50.85), "geneva": (6.14, 46.20), "stockholm": (18.07, 59.33),
    "athens": (23.73, 37.98), "lisbon": (-9.14, 38.72), "oslo": (10.75, 59.91),
    "munich": (11.58, 48.14), "hamburg": (9.99, 53.55), "venice": (12.32, 45.44),
    "sarajevo": (18.41, 43.86), "belgrade": (20.46, 44.79),
    # Asia
    "beijing": (116.41, 39.90), "shanghai": (121.47, 31.23), "tokyo": (139.69, 35.69),
    "hong kong": (114.16, 22.32), "singapore": (103.82, 1.35), "delhi": (77.21, 28.61),
    "new delhi": (77.21, 28.61), "mumbai": (72.88, 19.08), "seoul": (126.98, 37.57),
    "bangkok": (100.50, 13.76), "jakarta": (106.85, -6.21), "hanoi": (105.83, 21.03),
    "manila": (120.98, 14.60), "karachi": (67.01, 24.86), "dhaka": (90.41, 23.81),
    "taipei": (121.56, 25.03), "pyongyang": (125.76, 39.04),
    # Africa
    "lagos": (3.38, 6.52), "nairobi": (36.82, -1.29), "addis ababa": (38.75, 9.03),
    "johannesburg": (28.05, -26.20), "cape town": (18.42, -33.92), "casablanca": (-7.59, 33.57),
    "algiers": (3.06, 36.75), "tunis": (10.18, 36.81), "tripoli": (13.19, 32.89),
    "khartoum": (32.53, 15.50), "kinshasa": (15.27, -4.44), "accra": (-0.19, 5.60),
    # Americas
    "new york": (-74.01, 40.71), "washington": (-77.04, 38.91), "washington dc": (-77.04, 38.91),
    "los angeles": (-118.24, 34.05), "chicago": (-87.63, 41.88), "san francisco": (-122.42, 37.77),
    "boston": (-71.06, 42.36), "houston": (-95.37, 29.76), "mexico city": (-99.13, 19.43),
    "toronto": (-79.38, 43.65), "ottawa": (-75.70, 45.42), "havana": (-82.38, 23.11),
    "bogota": (-74.07, 4.71), "lima": (-77.04, -12.05), "santiago": (-70.65, -33.45),
    "buenos aires": (-58.38, -34.60), "rio de janeiro": (-43.20, -22.91),
    "sao paulo": (-46.63, -23.55), "brasilia": (-47.93, -15.78), "caracas": (-66.92, 10.49),
    # Oceania
    "sydney": (151.21, -33.87), "melbourne": (144.96, -37.81), "canberra": (149.13, -35.28),
    "auckland": (174.76, -36.85), "wellington": (174.78, -41.29),
}


def _load():
    global _DATA
    if _DATA is None:
        try:
            _DATA = json.loads(_ASSET.read_text())
        except Exception:                                          # noqa: BLE001
            _DATA = {}
    return _DATA


def _country_by_name(name):
    data = _load()
    n = (name or "").strip()
    key = _ALIASES.get(n.lower(), n)
    kl = key.lower()
    for iso, e in data.items():
        if e["name"].lower() == kl:
            return e
    for iso, e in data.items():                # loose contains (e.g. "Korea")
        if kl and kl in e["name"].lower():
            return e
    return None


def resolve(name):
    """(lon, lat) for a city or country name, else None. City wins over country."""
    if not name:
        return None
    n = name.strip().lower()
    if n in _CITIES:
        return _CITIES[n]
    # strip a trailing ", Country" (e.g. "Basra, Iraq")
    if "," in n and n.split(",")[0].strip() in _CITIES:
        return _CITIES[n.split(",")[0].strip()]
    e = _country_by_name(name)
    if e:
        return tuple(e["centroid"])
    return None


def country_rings(name):
    """List of [(lon,lat),…] outer rings for a country, else None."""
    e = _country_by_name(name)
    return e["rings"] if e else None


def country_entry(name):
    return _country_by_name(name)


def region_bbox(points, *, pad_frac=0.35, min_span=6.0):
    """Bounding [W,S,E,N] for (lon,lat) points, padded; clamped to a sane min."""
    pts = [p for p in points if p]
    if not pts:
        return None
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    w, s, e, n = min(xs), min(ys), max(xs), max(ys)
    spanx, spany = max(e - w, 0.0), max(n - s, 0.0)
    if max(spanx, spany) < min_span:                # single point / tiny → widen
        cx, cy = (w + e) / 2, (s + n) / 2
        w, e = cx - min_span / 2, cx + min_span / 2
        s, n = cy - min_span / 2, cy + min_span / 2
        spanx, spany = e - w, n - s
    padx, pady = spanx * pad_frac, spany * pad_frac
    return [w - padx, s - pady, e + padx, n + pady]


def make_projector(bbox, w, h, *, margin=0.06):
    """Return fn(lon,lat)->(px,py) fitting bbox into (w,h), aspect-correct, with
    latitude cosine correction (less east-west stretch). North is up."""
    W, S, E, N = bbox
    mlon, mlat = (W + E) / 2.0, (S + N) / 2.0
    cosl = max(0.2, math.cos(math.radians(mlat)))
    projW = max(1e-6, (E - W) * cosl)
    projH = max(1e-6, (N - S))
    availW, availH = w * (1 - 2 * margin), h * (1 - 2 * margin)
    scale = min(availW / projW, availH / projH)
    cx, cy = w / 2.0, h / 2.0

    def proj(lon, lat):
        x = cx + (lon - mlon) * cosl * scale
        y = cy - (lat - mlat) * scale
        return (x, y)
    proj.scale = scale            # type: ignore[attr-defined]
    proj.bbox = bbox              # type: ignore[attr-defined]
    return proj


# ── basemap styles: (ocean, land, border, border_hi, graticule) ─────────────
STYLES = {
    "parchment":   ((214, 198, 166), (198, 178, 138), (150, 126, 88), (120, 96, 60), (168, 148, 112)),
    "dark":        ((10, 13, 18), (28, 33, 42), (70, 84, 100), (150, 180, 205), (40, 50, 62)),
    "satellite":   ((9, 16, 24), (26, 34, 24), (60, 78, 60), (150, 190, 150), (40, 56, 64)),
    "tactical":    ((18, 20, 22), (38, 40, 40), (96, 104, 96), (210, 180, 96), (52, 56, 56)),
    "archival":    ((205, 196, 176), (182, 170, 146), (132, 118, 92), (96, 80, 56), (160, 150, 128)),
    "clean":       ((22, 28, 36), (40, 50, 62), (90, 110, 130), (150, 200, 230), (50, 62, 76)),
    "conflict":    ((16, 14, 14), (40, 34, 32), (110, 80, 70), (224, 120, 96), (54, 46, 44)),
    "blueprint":   ((10, 22, 36), (16, 32, 52), (70, 120, 170), (150, 200, 240), (34, 58, 86)),
}
_PAL_STYLE = {"parchment_sepia": "parchment", "amber_gold": "archival",
              "cold_steel": "clean", "ember_red": "conflict"}


def style_for(palette_name, override=""):
    if override and override in STYLES:
        return override
    return _PAL_STYLE.get(palette_name, "parchment")


def draw_basemap(w, h, bbox, *, style="parchment", seed=0, proj=None,
                 graticule=True, highlight=None, borders=True, reference=False):
    """Render a REAL coastline/border basemap for `bbox` in `style`. `highlight`
    is an optional set/list of country names to fill with the border-hi accent.

    HISTORICAL SAFETY (see HISTORICAL_MAP_POLICY.md): Natural Earth gives MODERN
    boundaries. `borders=False` suppresses inter-country border lines so only
    coastlines + the highlighted region read (coastlines are historically stable;
    modern national borders are NOT) — for scenes whose borders materially
    changed. `reference=True` stamps a subtle 'REFERENCE MAP' chip so a modern
    map is never silently presented as a historically exact border.
    Returns an RGB PIL image; flat ocean fill if data is missing."""
    data = _load()
    oc, land, bd, bdhi, grat = STYLES.get(style, STYLES["parchment"])
    img = Image.new("RGB", (w, h), oc)
    d = ImageDraw.Draw(img, "RGBA")
    if proj is None:
        proj = make_projector(bbox, w, h)
    W, S, E, N = bbox
    hl = set(x.lower() for x in (highlight or []))
    # graticule (every ~5–10° depending on span)
    if graticule:
        span = max(E - W, N - S)
        step = 10 if span > 40 else (5 if span > 12 else 2)
        lon0 = math.floor(W / step) * step
        while lon0 <= E:
            p0, p1 = proj(lon0, S), proj(lon0, N)
            d.line([p0, p1], fill=(*grat, 70), width=1)
            lon0 += step
        lat0 = math.floor(S / step) * step
        while lat0 <= N:
            p0, p1 = proj(W, lat0), proj(E, lat0)
            d.line([p0, p1], fill=(*grat, 70), width=1)
            lat0 += step
    # land fills + borders for every country intersecting the view
    for iso, e in data.items():
        bw, bs, be, bn = e["bbox"]
        if be < W or bw > E or bn < S or bs > N:          # bbox cull
            continue
        is_hi = e["name"].lower() in hl
        for ring in e["rings"]:
            poly = [proj(lon, lat) for lon, lat in ring]
            if len(poly) < 3:
                continue
            fill = bdhi if is_hi else land
            fa = 235 if is_hi else 255
            d.polygon(poly, fill=(*fill, fa))
            # de-emphasised modern borders by default (read as reference geography,
            # not exact lines); suppressed entirely when borders=False except the
            # highlighted region's own outline.
            if is_hi:
                d.line(poly + [poly[0]], fill=(*bdhi, 255), width=2)
            elif borders:
                d.line(poly + [poly[0]], fill=(*bd, 150), width=1)
    if reference:
        # A CLEARLY READABLE honesty stamp (not hidden) so a modern map is never
        # mistaken for an exact historical boundary. `reference` may be:
        #   True              → "REFERENCE MAP · MODERN BORDERS FOR ORIENTATION"
        #   "approximate"     → "APPROXIMATE REGION"
        #   any string        → used verbatim (upper-cased)
        if reference is True:
            txt = "REFERENCE MAP · MODERN BORDERS FOR ORIENTATION"
        elif str(reference).strip().lower() in ("approx", "approximate", "approximate region"):
            txt = "APPROXIMATE REGION"
        else:
            txt = str(reference).strip().upper()
        rf = look.font("label", max(20, int(h * 0.028)))
        bx = ImageDraw.Draw(img).textbbox((0, 0), txt, font=rf)
        tw, th = bx[2] - bx[0], bx[3] - bx[1]
        padx, pady = int(th * 0.7), int(th * 0.5)
        chip = Image.new("RGBA", (tw + padx * 2 + int(th * 0.7), th + pady * 2), (0, 0, 0, 0))
        cdr = ImageDraw.Draw(chip)
        cdr.rounded_rectangle([0, 0, chip.width - 1, chip.height - 1],
            radius=int(th * 0.32), fill=(8, 9, 11, 224))
        cdr.rectangle([0, 0, int(th * 0.32), chip.height], fill=(228, 168, 84, 255))  # amber tab
        cdr.text((padx + int(th * 0.7) - bx[0], pady - bx[1]), txt, font=rf,
                 fill=(238, 232, 220, 255))
        img = img.convert("RGBA")
        img.alpha_composite(chip, (int(w * 0.5 - chip.width / 2),
                                   int(h * 0.93 - chip.height / 2)))
        img = img.convert("RGB")
    return img


def basemap_for(places, w, h, *, palette_name="parchment_sepia", style=None,
                seed=0, highlight=None, graticule=True, pad_frac=0.4, min_hits=1,
                borders=True, reference=False):
    """One-call real basemap for a set of place names. Resolves them, frames the
    region, renders a real coastline/border bed, and returns
    (bed_img, projector, {name:(lon,lat)}). Returns (None, None, {}) when fewer
    than `min_hits` places resolve — the caller then uses its synthetic fallback."""
    names = [p for p in (places or []) if p]
    res = {nm: resolve(nm) for nm in names}
    hits = {nm: ll for nm, ll in res.items() if ll}
    if len(hits) < max(1, min_hits):
        return None, None, {}
    bbox = region_bbox(list(hits.values()), pad_frac=pad_frac)
    st = style or style_for(palette_name)
    proj = make_projector(bbox, w, h)
    bed = draw_basemap(w, h, bbox, style=st, seed=seed, proj=proj,
                       graticule=graticule, highlight=highlight, borders=borders,
                       reference=reference)
    return bed, proj, hits


def place_label(base, text, cx, cy, pal, *, size_frac=0.031, opacity=1.0,
                dy=-26, accent=False):
    """Draw a MOBILE-READABLE place label: condensed all-caps on a soft dark
    pill (so it reads over busy map land), anchored above (dy<0) a node. Returns
    the composited RGB frame. Larger + scrimmed vs a bare glow — legible at small
    player sizes, never tiny dev-zoom-only text."""
    if not text or opacity <= 0.01:
        return base
    w, h = base.size
    fnt = look.font("label", max(20, int(h * size_frac)))
    ink = tuple(pal["accent_hi"]) if accent else (240, 236, 226)
    tmp = Image.new("RGBA", (8, 8)); box = ImageDraw.Draw(tmp).textbbox((0, 0), text.upper(), font=fnt)
    tw, th = box[2] - box[0], box[3] - box[1]
    pad_x, pad_y = int(th * 0.55), int(th * 0.34)
    chip = Image.new("RGBA", (tw + pad_x * 2, th + pad_y * 2), (0, 0, 0, 0))
    cd = ImageDraw.Draw(chip)
    cd.rounded_rectangle([0, 0, chip.width - 1, chip.height - 1],
                         radius=int(th * 0.34),
                         fill=(int(pal["bg_b"][0]), int(pal["bg_b"][1]),
                               int(pal["bg_b"][2]), int(190 * opacity)))
    cd.text((pad_x - box[0], pad_y - box[1]), text.upper(), font=fnt,
            fill=(*ink, int(255 * opacity)))
    x = int(cx - chip.width / 2)
    y = int(cy + dy - chip.height / 2)
    x = max(4, min(w - chip.width - 4, x))
    y = max(4, min(h - chip.height - 4, y))
    base = base.convert("RGBA")
    base.alpha_composite(chip, (x, y))
    return base.convert("RGB")


def _chip_img(text, fnt, pal, ink, opacity):
    tmp = Image.new("RGBA", (8, 8))
    box = ImageDraw.Draw(tmp).textbbox((0, 0), text, font=fnt)
    tw, th = box[2] - box[0], box[3] - box[1]
    pad_x, pad_y = int(th * 0.55), int(th * 0.34)
    chip = Image.new("RGBA", (tw + pad_x * 2, th + pad_y * 2), (0, 0, 0, 0))
    cd = ImageDraw.Draw(chip)
    cd.rounded_rectangle([0, 0, chip.width - 1, chip.height - 1], radius=int(th * 0.34),
                         fill=(int(pal["bg_b"][0]), int(pal["bg_b"][1]),
                               int(pal["bg_b"][2]), int(190 * opacity)))
    cd.text((pad_x - box[0], pad_y - box[1]), text, font=fnt, fill=(*ink, int(255 * opacity)))
    return chip


def _overlap(a, b):
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def layout_labels(base, items, pal, *, max_labels=None):
    """Place many node labels with COLLISION AVOIDANCE + crowd simplification.
    `items`: list of dicts {text, x, y, priority(↑=keep), accent(bool),
    opacity, size_frac}. High-priority labels place first + larger; each tries
    anchors (above/below/right/left, nudged) until it clears placed chips +
    stays in safe margins; when the map is crowded only the top `max_labels`
    survive (low-priority hidden). Endpoints (the node dots) stay uncovered
    because the preferred anchor is offset above the point. Returns RGB frame."""
    if not items:
        return base
    W, H = base.size
    margin = int(H * 0.03)
    items = sorted([it for it in items if it.get("text") and it.get("opacity", 1) > 0.01],
                   key=lambda it: -it.get("priority", 0))
    if max_labels is not None:
        items = items[:max_labels]
    base = base.convert("RGBA")
    placed = []                                   # occupied rects (incl. node dots)
    for it in items:
        opacity = float(it.get("opacity", 1.0))
        size = it.get("size_frac", 0.030 if it.get("priority", 0) >= 2 else 0.026)
        fnt = look.font("label", max(19, int(H * size)))
        ink = tuple(pal["accent_hi"]) if it.get("accent") else (240, 236, 226)
        chip = _chip_img(str(it["text"]).upper(), fnt, pal, ink, opacity)
        cw, ch = chip.width, chip.height
        x, y = int(it["x"]), int(it["y"])
        placed.append((x - 8, y - 8, x + 8, y + 8))     # reserve the node dot
        # candidate anchor offsets (dx, dy of chip CENTER from node) — above first
        cands = [(0, -ch * 0.9 - 12), (0, ch * 0.9 + 12), (cw * 0.6 + 16, 0),
                 (-cw * 0.6 - 16, 0), (cw * 0.6 + 16, -ch - 10),
                 (-cw * 0.6 - 16, -ch - 10), (0, -ch * 1.7 - 14)]
        chosen = None
        for dx, dy in cands:
            cx = int(x + dx - cw / 2); cy = int(y + dy - ch / 2)
            cx = max(margin, min(W - cw - margin, cx))
            cy = max(margin, min(H - ch - margin, cy))
            rect = (cx, cy, cx + cw, cy + ch)
            if not any(_overlap(rect, p) for p in placed):
                chosen = (cx, cy, rect); break
        if chosen is None:                          # crowded → drop this label
            continue
        cx, cy, rect = chosen
        base.alpha_composite(chip, (cx, cy))
        placed.append(rect)
    return base.convert("RGB")


def available():
    return bool(_load())
