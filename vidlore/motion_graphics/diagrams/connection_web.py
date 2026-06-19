"""Primitive: connection_web.

A relationship beat — named nodes ring the frame and thin gold lines draw between
the ones that are linked, exposing a web of who-knew-who: a conspiracy, a network
of shell companies, a cast of connected players.

Distinct from `org_hierarchy_tree` (a strict root→children TREE) and
`money_flow_empire` (a centre with radiating money branches): this is a
NON-HIERARCHICAL graph — any node can link to any other, and the pattern of links
IS the story.

Forensic principle (NOT asset copy): MagnatesMedia reveals hidden connections by
drawing the lines one by one; the premium is the graded field + restrained gold
nodes + thin glowing links + a hub highlight, not a hairball. Pure-local (PIL +
numpy → ffmpeg). No paid API. Deterministic.

    render("x.mp4", nodes=["The Senator", "The Banker", "The Lobbyist",
           "The Editor", "The General"], links=["The Senator-The Banker",
           "The Banker-The Lobbyist", "The Lobbyist-The Editor",
           "The Senator-The General"], title="THE WEB OF INFLUENCE")
"""
from __future__ import annotations

import math
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .. import look

SPEC = {
    "id": "connection_web", "family": "diagrams",
    "roles": ["network", "connections", "relationships", "conspiracy", "web"],
    "niches_ok": ["crime", "geopolitics", "business", "biography", "history", "tech"],
    "intensity_range": [3, 5], "duration_range": [4.5, 7.0],
    "easing": "easeOutCubic", "audio_cue": "soft_link_pulse",
    "repeat_cooldown_s": 55, "per_video_cap": 2, "cost": "low",
    "layout_variants": ["ring_graph"],
    "review_override": ["nodes", "links", "title", "palette"],
    "fallback": "org_hierarchy_tree if the relationships are strictly hierarchical",
}


def _coerce_nodes(nodes):
    out = []
    for nd in (nodes or []):
        if isinstance(nd, dict):
            out.append(str(nd.get("name") or nd.get("label") or "").strip())
        elif isinstance(nd, (list, tuple)) and nd:
            out.append(str(nd[0]).strip())
        elif nd:
            out.append(str(nd).strip())
    return [n for n in out if n][:7]


def _coerce_links(links, names):
    idx = {n.lower(): i for i, n in enumerate(names)}
    out = []
    for lk in (links or []):
        a = b = None
        if isinstance(lk, str):
            parts = lk.replace("—", "-").replace("–", "-").split("-")
            if len(parts) >= 2:
                a, b = parts[0].strip(), parts[1].strip()
        elif isinstance(lk, (list, tuple)) and len(lk) >= 2:
            a, b = lk[0], lk[1]
        if a is None:
            continue
        ai = a if isinstance(a, int) else idx.get(str(a).strip().lower())
        bi = b if isinstance(b, int) else idx.get(str(b).strip().lower())
        if (ai is not None and bi is not None and ai != bi
                and 0 <= ai < len(names) and 0 <= bi < len(names)):
            pair = tuple(sorted((ai, bi)))
            if pair not in out:
                out.append(pair)
    return out


def render(out_path, *, nodes=None, links=None, title: str = "", dur: float = 6.0,
           fps: int = 30, w: int = 1920, h: int = 1080,
           palette_name: str = "ember_red", layout: str = "", seed: int = 0,
           crf: int = 18) -> dict:
    pal = look.palette(palette_name)
    t0 = time.time()
    n = max(2, int(round(dur * fps)))
    names = _coerce_nodes(nodes)
    if len(names) < 2:
        names = (names + ["NODE A", "NODE B"])[:2]
    m = len(names)
    pairs = _coerce_links(links, names)
    if not pairs:                                              # default: ring chain + a hub spoke
        pairs = [tuple(sorted((k, (k + 1) % m))) for k in range(m)]
        pairs += [tuple(sorted((0, k))) for k in range(2, m)]
        pairs = list(dict.fromkeys(pairs))

    cx, cy = w / 2, int(h * 0.55)         # nudged down so the top label clears title
    Rr = int(h * 0.30)
    ang = [-90 + k * 360 / m for k in range(m)]
    pos = [(cx + math.cos(math.radians(a)) * Rr,
            cy + math.sin(math.radians(a)) * Rr) for a in ang]

    title_font = look.font("label", int(h * 0.030))
    name_font = look.font("label", int(h * 0.026))
    nr = int(h * 0.018)

    td = Path(tempfile.mkdtemp())
    for i in range(n):
        t = i / fps
        pr = i / max(1, n - 1)
        frame = look.graded_background(w, h, pal, seed=seed, drift=pr)
        d = ImageDraw.Draw(frame, "RGBA")
        if title:
            ta = look.ease_out_cubic(min(1.0, t / 0.5))
            ti = look.text_with_glow(title.upper(), title_font, fill=pal["text"],
                                     glow=pal["bg_b"], glow_radius=4,
                                     glow_alpha=0.0, pad=16)
            frame = look.paste_center(frame, ti, cx=w * 0.5, cy=h * 0.11,
                                      opacity=ta)
            d = ImageDraw.Draw(frame, "RGBA")
        # links draw in (staggered), each line grows from a → b
        for li, (a, b) in enumerate(pairs):
            le = look.ease_out_cubic(max(0.0, (t - (0.5 + li * 0.12)) / 0.5))
            if le <= 0.01:
                continue
            ax, ay = pos[a]; bx, by = pos[b]
            ex, ey = ax + (bx - ax) * le, ay + (by - ay) * le
            d.line([(ax, ay), (ex, ey)], fill=(*pal["accent"], 150), width=2)
        # nodes pop in (staggered, before links) + labels
        for k in range(m):
            ne = look.ease_out_cubic(max(0.0, (t - k * 0.12) / 0.4))
            if ne <= 0.01:
                continue
            px, py = pos[k]
            is_hub = (k == 0)
            rr = int(nr * (1.25 if is_hub else 1.0) * ne)
            col = pal["accent_hi"] if is_hub else pal["accent"]
            d.ellipse([px - rr - 5, py - rr - 5, px + rr + 5, py + rr + 5],
                      fill=(*col, int(60 * ne)))
            d.ellipse([px - rr, py - rr, px + rr, py + rr], fill=(*col, 255))
            d.ellipse([px - rr, py - rr, px + rr, py + rr],
                      outline=(*pal["bg_b"], 200), width=2)
            # label pushed radially outward from centre
            la = math.radians(ang[k])
            lx = px + math.cos(la) * int(h * 0.055)
            ly = py + math.sin(la) * int(h * 0.055)
            ni = look.text_with_glow(names[k].upper(), name_font,
                                     fill=(pal["text"] if is_hub else pal["muted"]),
                                     glow=pal["bg_b"], glow_radius=3,
                                     glow_alpha=0.0, pad=8)
            frame = look.paste_center(frame, ni, cx=lx, cy=ly, opacity=ne)
            d = ImageDraw.Draw(frame, "RGBA")

        frame = look.vignette(frame, strength=0.6)
        frame = look.film_grain(frame, seed=seed, amount=4.5, t=t)
        fa = look.fade_alpha(t, dur, fps)
        if fa < 1.0:
            frame = look.fade_frame(frame, fa, pal)
        frame.save(td / f"f{i:05d}.png")

    try:
        from vidlore.ffmpeg_tool import ffmpeg_exe
        ff = ffmpeg_exe()
    except Exception:                                          # noqa: BLE001
        ff = "ffmpeg"
    cmd = [ff, "-y", "-hide_banner", "-loglevel", "error", "-framerate", str(fps),
           "-i", f"{td}/f%05d.png", "-c:v", "libx264", "-crf", str(crf),
           "-pix_fmt", "yuv420p", "-colorspace", "bt709", "-color_primaries",
           "bt709", "-color_trc", "bt709", "-movflags", "+faststart", str(out_path)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    ok = r.returncode == 0 and Path(out_path).exists()
    look.cleanup_frames(td)
    return {"ok": ok, "path": str(out_path), "frames": n, "dur_s": round(dur, 2),
            "render_s": round(time.time() - t0, 2), "w": w, "h": h,
            "err": (r.stderr[-200:] if not ok else "")}
