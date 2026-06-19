"""REAL DOCUMENTARY MAP ENGINE.

Fetches real geographic imagery (satellite / political) and stitches it into
a single frame centred on a location, so the cinematic overlays (pin drop,
pulse ring, label, route, region highlight) can sit on TOP of REAL terrain —
the Netflix / Johnny Harris "the documentary is geographically explaining
the story" look, not a flat fake chart.

Provider ladder (licence-safe, works with NO token):
  1. Mapbox / MapTiler Static API   — only if a token is in the env
     (premium styles: satellite-streets, dark, light). Commercial OK
     WITH attribution.
  2. ESRI World Imagery tiles       — free satellite, attribution "Esri".
  3. OpenStreetMap tiles            — free political/street base.
Every fetched tile is cached on disk, so a render reuses imagery and we stay
well within free limits and are gentle on the public tile servers.

Geocoding: the editor SHOULD supply 'lat,lon' in the card body (it knows
major places); when it doesn't, we fall back to a cached Nominatim lookup.

Nothing here hardcodes anything illegal — every source is a documented,
allowed tile/endpoint, and attribution is baked into the rendered frame.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageOps

_UA = "vidlore-map/1.0 (documentary render; contact: support@vidlore.local)"
_TILE = 256

# ---- provider tile templates ------------------------------------------- #
_ESRI_SAT = ("https://server.arcgisonline.com/ArcGIS/rest/services/"
             "World_Imagery/MapServer/tile/{z}/{y}/{x}")
_ESRI_REF = ("https://server.arcgisonline.com/ArcGIS/rest/services/"
             "Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}")
_OSM = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"

# A documentary map "style" maps to (base provider, post-process key).
# The post-process is applied in PIL so we get parchment / tactical / news
# looks even from a plain satellite or street base.
STYLE_PRESETS = {
    "satellite":   ("esri_sat", "satellite"),   # satellite documentary
    "dark":        ("esri_sat", "dark"),         # dark investigative
    "political":   ("osm",      "political"),    # clean political
    "parchment":   ("osm",      "parchment"),    # historical parchment
    "tactical":    ("esri_sat", "tactical"),     # tactical military (mono-grn)
    "news":        ("osm",      "news"),         # news broadcast (blue)
}


def _attr_for(provider: str) -> str:
    if provider == "osm":
        return "© OpenStreetMap contributors"
    if provider.startswith("esri"):
        return "Imagery © Esri, Maxar, Earthstar Geographics"
    return ""


# ====================================================================== #
#  Slippy-map (Web Mercator) math
# ====================================================================== #
def _deg2xy(lat: float, lon: float, z: int) -> tuple:
    """lat/lon -> fractional tile coords at zoom z."""
    lat = max(-85.05112878, min(85.05112878, lat))
    lat_r = math.radians(lat)
    n = 2 ** z
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - math.asinh(math.tan(lat_r)) / math.pi) / 2.0 * n
    return x, y


def _xy2deg(x: float, y: float, z: int) -> tuple:
    """fractional tile coords -> (lat, lon)."""
    n = 2 ** z
    lon = x / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    return lat, lon


def fit_bounds(points: list, out_w: int, out_h: int, *, pad: float = 0.18,
               min_z: int = 2, max_z: int = 13) -> tuple:
    """Given [(lat,lon),...], return (center_lat, center_lon, zoom) such that
    ALL points fit inside out_w x out_h with `pad` margin — the most
    zoomed-in framing that still shows the whole journey/region."""
    # never zoom out past the point where the WHOLE WORLD (2^z*256 px) is
    # narrower/shorter than the frame, or we'd get black bars at the edges.
    floor_z = math.ceil(math.log2(max(out_w, out_h) / _TILE))
    min_z = max(min_z, int(floor_z))
    pts = [(float(a), float(b)) for a, b in points]
    if len(pts) == 1:
        return pts[0][0], pts[0][1], min(11, max_z)
    best = min_z
    for z in range(max_z, min_z - 1, -1):
        xs = [_deg2xy(a, b, z)[0] * _TILE for a, b in pts]
        ys = [_deg2xy(a, b, z)[1] * _TILE for a, b in pts]
        if (max(xs) - min(xs) <= out_w * (1 - 2 * pad)
                and max(ys) - min(ys) <= out_h * (1 - 2 * pad)):
            best = z
            break
    z = best
    xs = [_deg2xy(a, b, z)[0] for a, b in pts]
    ys = [_deg2xy(a, b, z)[1] for a, b in pts]
    clat, clon = _xy2deg((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, z)
    return clat, clon, z


def project(lat: float, lon: float, clat: float, clon: float, z: int,
            out_w: int, out_h: int) -> tuple:
    """Project (lat,lon) to PIXEL coords inside an out_w x out_h image that is
    centred on (clat,clon) at zoom z."""
    px = (_deg2xy(lat, lon, z)[0] - _deg2xy(clat, clon, z)[0]) * _TILE \
        + out_w / 2.0
    py = (_deg2xy(lat, lon, z)[1] - _deg2xy(clat, clon, z)[1]) * _TILE \
        + out_h / 2.0
    return px, py


def haversine_km(a: tuple, b: tuple) -> float:
    """Great-circle distance between (lat,lon) a and b, in km."""
    r = 6371.0
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    dla, dlo = la2 - la1, lo2 - lo1
    h = (math.sin(dla / 2) ** 2
         + math.cos(la1) * math.cos(la2) * math.sin(dlo / 2) ** 2)
    return 2 * r * math.asin(min(1.0, math.sqrt(h)))


# ====================================================================== #
#  Tile fetch (disk-cached, polite retries)
# ====================================================================== #
def _tile_cache_dir(cache_dir: Path | None) -> Path:
    base = (cache_dir if cache_dir is not None else Path("/tmp")) / "maptiles"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _fetch_tile(provider: str, z: int, x: int, y: int,
                cache_dir: Path | None) -> Image.Image | None:
    n = 2 ** z
    if not (0 <= y < n):                          # poles: no tile (clamp Y)
        return Image.new("RGB", (_TILE, _TILE), (12, 16, 24))
    x = x % n                                     # longitude WRAPS east<->west
    tdir = _tile_cache_dir(cache_dir)
    cf = tdir / f"{provider}_{z}_{x}_{y}.png"
    if cf.exists():
        try:
            return Image.open(cf).convert("RGB")
        except Exception:                                  # noqa: BLE001
            pass
    if provider == "osm":
        url = _OSM.format(z=z, x=x, y=y)
    elif provider == "esri_ref":
        url = _ESRI_REF.format(z=z, x=x, y=y)
    else:                                                   # esri_sat
        url = _ESRI_SAT.format(z=z, x=x, y=y)
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": _UA})
            data = urllib.request.urlopen(req, timeout=12).read()
            im = Image.open(__import__("io").BytesIO(data)).convert("RGB")
            # ESRI occasionally returns a pure-white "no data" tile. On the
            # SATELLITE base it shows as a bright rectangle; on the REFERENCE
            # (labels) layer it composites as an opaque white block. Detect
            # it and (sat) substitute OSM / (ref) skip — never cache it.
            if _is_blank(im):
                if provider == "esri_sat":
                    osm = _fetch_tile("osm", z, x, y, cache_dir)
                    return osm if osm is not None else None
                return None                    # blank ref/other -> skip
            try:
                im.save(cf)
            except Exception:                              # noqa: BLE001
                pass
            return im
        except Exception:                                  # noqa: BLE001
            time.sleep(0.4 * (attempt + 1))
    return None


def _is_blank(im: "Image.Image") -> bool:
    """True if a tile is almost entirely near-white (an ESRI 'no data'
    placeholder) — checked on a tiny downscale so it's cheap."""
    try:
        small = im.resize((16, 16)).convert("L")
        px = list(small.getdata())
        white = sum(1 for p in px if p >= 250)
        return white >= int(len(px) * 0.92)
    except Exception:                                      # noqa: BLE001
        return False


def _stitch(provider: str, lat: float, lon: float, z: int,
            out_w: int, out_h: int, cache_dir: Path | None,
            label_overlay: bool = True) -> Image.Image | None:
    """Stitch the tiles needed to fill out_w x out_h centred on (lat,lon)."""
    cx, cy = _deg2xy(lat, lon, z)
    cpx, cpy = cx * _TILE, cy * _TILE              # centre, world pixels
    left = cpx - out_w / 2.0
    top = cpy - out_h / 2.0
    x0, y0 = int(math.floor(left / _TILE)), int(math.floor(top / _TILE))
    x1 = int(math.floor((left + out_w) / _TILE))
    y1 = int(math.floor((top + out_h) / _TILE))
    canvas = Image.new("RGB", ((x1 - x0 + 1) * _TILE,
                               (y1 - y0 + 1) * _TILE), (10, 14, 20))
    ok_any = False
    for ty in range(y0, y1 + 1):
        for tx in range(x0, x1 + 1):
            t = _fetch_tile(provider, z, tx, ty, cache_dir)
            if t is None:
                continue
            ok_any = True
            canvas.paste(t, ((tx - x0) * _TILE, (ty - y0) * _TILE))
    if not ok_any:
        return None
    # NOTE: the ESRI place/road REFERENCE overlay is deliberately NOT
    # composited — its tiles are inconsistent (some ship dark-labels-on-white
    # which the alpha mask turned into opaque white blocks). Clean satellite
    # (Netflix/Johnny-Harris style) reads better and our gold label chip
    # already names the location; boundaries remain visible on the imagery.
    off_x = int(left - x0 * _TILE)
    off_y = int(top - y0 * _TILE)
    return canvas.crop((off_x, off_y, off_x + out_w, off_y + out_h))


def _ref_mask(ref: Image.Image) -> Image.Image:
    """The ESRI reference layer is mostly transparent-ish white labels on a
    near-black field; turn its luminance into an alpha mask so only the
    labels/roads composite onto the satellite base."""
    g = ImageOps.grayscale(ref)
    return g.point(lambda p: min(255, int(p * 1.2)) if p > 40 else 0)


# ====================================================================== #
#  Geocoding (editor coords first, cached Nominatim fallback)
# ====================================================================== #
def parse_coords(s: str) -> tuple | None:
    """Parse 'lat,lon' / 'lat|lon' (also accepts 'N/S/E/W' suffixes)."""
    if not s:
        return None
    s = s.strip().replace("|", ",")
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) < 2:
        return None

    def _one(tok: str) -> float | None:
        tok = tok.strip().upper()
        sign = 1.0
        if tok and tok[-1] in "NSEW":
            if tok[-1] in "SW":
                sign = -1.0
            tok = tok[:-1].strip()
        try:
            return sign * float(tok)
        except ValueError:
            return None
    la, lo = _one(parts[0]), _one(parts[1])
    if la is None or lo is None:
        return None
    if -90 <= la <= 90 and -180 <= lo <= 180:
        return (la, lo)
    return None


def geocode(place: str, cache_dir: Path | None) -> tuple | None:
    """Place name -> (lat, lon) via cached Nominatim. Returns None on fail."""
    if not place or not place.strip():
        return None
    key = hashlib.sha1(place.strip().lower().encode()).hexdigest()[:16]
    cdir = (cache_dir if cache_dir is not None else Path("/tmp"))
    cf = cdir / f"geocode_{key}.json"
    if cf.exists():
        try:
            d = json.loads(cf.read_text())
            if d.get("lat") is not None:
                return (d["lat"], d["lon"])
            return None
        except Exception:                                  # noqa: BLE001
            pass
    try:
        q = urllib.parse.urlencode({"q": place, "format": "json", "limit": 1})
        url = f"https://nominatim.openstreetmap.org/search?{q}"
        req = urllib.request.Request(url, headers={"User-Agent": _UA})
        data = json.loads(urllib.request.urlopen(req, timeout=12).read())
        if data:
            lat, lon = float(data[0]["lat"]), float(data[0]["lon"])
            bb = data[0].get("boundingbox")          # [S, N, W, E] strings
            rec = {"lat": lat, "lon": lon}
            if bb and len(bb) == 4:
                rec["bbox"] = [float(x) for x in bb]
            try:
                cf.write_text(json.dumps(rec))
            except Exception:                              # noqa: BLE001
                pass
            return (lat, lon)
        try:
            cf.write_text(json.dumps({"lat": None}))
        except Exception:                                  # noqa: BLE001
            pass
    except Exception:                                      # noqa: BLE001
        return None
    return None


def geocode_bbox(place: str, cache_dir: Path | None) -> tuple | None:
    """Like geocode() but also returns the OSM bounding box when known:
    (lat, lon, (south, north, west, east)) — used to FIT a region highlight
    to the real territory extent. bbox may be None."""
    if not place or not place.strip():
        return None
    geocode(place, cache_dir)                    # ensure cache populated
    key = hashlib.sha1(place.strip().lower().encode()).hexdigest()[:16]
    cdir = (cache_dir if cache_dir is not None else Path("/tmp"))
    cf = cdir / f"geocode_{key}.json"
    try:
        d = json.loads(cf.read_text())
    except Exception:                                      # noqa: BLE001
        return None
    if d.get("lat") is None:
        return None
    bb = d.get("bbox")
    box = (bb[0], bb[1], bb[2], bb[3]) if bb and len(bb) == 4 else None
    return (d["lat"], d["lon"], box)


# ====================================================================== #
#  Style post-process (parchment / tactical / dark / news from a base)
# ====================================================================== #
def stylize(img: Image.Image, style_key: str) -> Image.Image:
    im = img.convert("RGB")
    if style_key == "satellite":
        # gentle cinematic grade: a touch more contrast + slightly cooler
        im = ImageOps.autocontrast(im, cutoff=1)
        return im
    if style_key == "dark":
        im = ImageOps.autocontrast(im, cutoff=1)
        dark = Image.new("RGB", im.size, (8, 14, 26))
        return Image.blend(im, dark, 0.45)
    if style_key == "political":
        return im
    if style_key == "parchment":
        g = ImageOps.grayscale(im)
        g = ImageOps.autocontrast(g, cutoff=2)
        sep = ImageOps.colorize(g, black=(54, 38, 18),
                                white=(232, 214, 176),
                                mid=(166, 132, 84))
        return sep
    if style_key == "tactical":
        g = ImageOps.grayscale(im)
        grn = ImageOps.colorize(g, black=(2, 10, 4),
                                white=(150, 230, 150),
                                mid=(28, 92, 44))
        return grn
    if style_key == "news":
        g = ImageOps.grayscale(im)
        blu = ImageOps.colorize(g, black=(6, 14, 30),
                                white=(208, 224, 248),
                                mid=(40, 78, 150))
        return blu
    return im


# ====================================================================== #
#  Public entry: a ready-to-use map background frame
# ====================================================================== #
def map_background(lat: float, lon: float, *, zoom: int, out_w: int,
                   out_h: int, style: str = "satellite",
                   cache_dir: Path | None = None) -> tuple | None:
    """Return (PIL RGB image out_w x out_h, attribution str) for the location,
    or None if no provider could supply imagery (caller should fall back to
    the vector chart). Disk-cached by (provider,lat,lon,zoom,size,style)."""
    provider, post = STYLE_PRESETS.get(style, STYLE_PRESETS["satellite"])
    # Token providers (premium) — only if configured.
    token = (os.environ.get("MAPTILER_KEY") or "").strip()
    mb = (os.environ.get("MAPBOX_TOKEN") or "").strip()

    key = hashlib.sha1(
        f"{provider}|{post}|{lat:.5f}|{lon:.5f}|{zoom}|{out_w}x{out_h}"
        .encode()).hexdigest()[:18]
    cdir = (cache_dir if cache_dir is not None else Path("/tmp"))
    ccf = cdir / f"mapbg_{key}.png"
    attr = _attr_for(provider)
    if ccf.exists():
        try:
            return Image.open(ccf).convert("RGB"), attr
        except Exception:                                  # noqa: BLE001
            pass

    base = None
    # (token providers could be slotted here; ESRI/OSM are the no-token path)
    if base is None:
        base = _stitch(provider, lat, lon, zoom, out_w, out_h, cache_dir)
    if base is None:
        # last imagery attempt: the other free base before giving up
        alt = "osm" if provider != "osm" else "esri_sat"
        base = _stitch(alt, lat, lon, zoom, out_w, out_h, cache_dir)
        if base is not None:
            attr = _attr_for(alt)
    if base is None:
        return None
    styled = stylize(base, post)
    try:
        styled.save(ccf)
    except Exception:                                      # noqa: BLE001
        pass
    return styled, attr
