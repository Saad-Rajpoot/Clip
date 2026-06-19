"""Primitive: evidence_connection_board.

The "it's all connected" beat — a restrained, premium take on the red-string
board. On a charcoal stage, 3–5 pinned evidence cards are placed around a central
subject, and connecting threads (with a gentle string sag) draw in one at a time
to reveal the relationships. Deliberately controlled — not the chaotic conspiracy
wall — so it reads as an investigator's case map: spy / true-crime / corruption /
history.

Distinct from `org_hierarchy_tree` (a clean top-down structure) and
`relationship_tree` (genealogy): this is a non-hierarchical web of EVIDENCE with
a stressed central node and sequentially drawn links.

Forensic principle (NOT an asset copy): premium docs connect pinned evidence
cards to a central subject with threads that resolve one by one. Rebuilt from that
PRINCIPLE with Vidlore's own look system. Pure-local, deterministic, no paid API.

    render("x.mp4", title="THE NETWORK", center="The Banker",
           nodes=["Shell company","Offshore account","The courier","Senator X"])
"""
from __future__ import annotations

import math
import subprocess
import tempfile
import textwrap
import time
from pathlib import Path

from PIL import Image, ImageDraw

from .. import look

SPEC = {
    "id": "evidence_connection_board", "family": "investigation",
    "roles": ["connection_board", "evidence_board", "network", "links",
              "its_all_connected", "web", "conspiracy_map"],
    "niches_ok": ["crime", "spy", "history", "geopolitics", "business"],
    "intensity_range": [3, 4], "duration_range": [5.5, 8.5],
    "easing": "easeOutCubic", "audio_cue": "pin_soft",
    "repeat_cooldown_s": 60, "per_video_cap": 2, "cost": "low",
    "layout_variants": ["hub"],
    "review_override": ["title", "center", "nodes", "links", "palette"],
    "fallback": "org_hierarchy_tree when the relationship is strictly hierarchical",
    "required_inputs": ["nodes"],
}


def _card(text, cw, ch, pal, stressed=False):
    card = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
    d = ImageDraw.Draw(card)
    fillc = tuple(pal["bg_a"]) if not stressed else tuple(
        int(a * 0.5 + b * 0.5) for a, b in zip(pal["bg_a"], pal["accent"]))
    d.rounded_rectangle([0, 0, cw - 1, ch - 1], radius=8, fill=(*fillc, 246))
    edge = tuple(pal["accent_hi"]) if stressed else tuple(pal["muted"])
    d.rounded_rectangle([0, 0, cw - 1, ch - 1], radius=8, outline=(*edge, 255),
                        width=3 if stressed else 2)
    f = look.font("label", int(ch * (0.26 if stressed else 0.23)))
    avg = f.getlength("n") or 10
    wrapn = max(8, int((cw - 18) / avg))
    lines = textwrap.wrap(text, wrapn)[:2]
    fg = tuple(pal["text"])
    total_h = len(lines) * int(f.size * 1.18)
    y = (ch - total_h) // 2
    for ln in lines:
        li = look.text_with_glow(ln, f, fill=fg, glow=pal["bg_b"], glow_radius=3,
                                 glow_alpha=0.0, pad=3)
        if li.width > cw - 12:
            li = li.resize((cw - 12, int(li.height * (cw - 12) / li.width)), Image.LANCZOS)
        card.alpha_composite(li, ((cw - li.width) // 2, y))
        y += int(f.size * 1.18)
    # pin head
    d.ellipse([cw // 2 - 6, -2, cw // 2 + 6, 10],
              fill=(*(tuple(pal["accent_hi"]) if stressed else tuple(pal["muted"])), 255))
    return card


def _norm_nodes(nodes):
    out = []
    for nn in (nodes or []):
        if isinstance(nn, str):
            out.append({"label": nn.strip(), "photo": None})
        elif isinstance(nn, dict):
            out.append({"label": str(nn.get("label") or nn.get("name") or "").strip(),
                        "photo": nn.get("photo")})
    return [x for x in out if x["label"]][:5]


def render(out_path, *, title: str = "", center: str = "", nodes=None, links=None,
           dur: float = 7.0, fps: int = 30, w: int = 1920, h: int = 1080,
           palette_name: str = "cold_steel", layout: str = "", seed: int = 0,
           crf: int = 18) -> dict:
    pal = look.palette(palette_name)
    t0 = time.time()
    n = max(2, int(round(dur * fps)))
    items = _norm_nodes(nodes)
    accent = tuple(pal["accent_hi"]); ink = tuple(pal["text"]); muted = tuple(pal["muted"])
    thread = (200, 96, 82)                                          # restrained string red
    has_center = bool(str(center or "").strip())

    bed = look.graded_background(w, h, pal, seed=seed) if hasattr(look, "graded_background") \
        else Image.new("RGB", (w, h), tuple(pal["bg_b"]))

    # layout: optional center node + ring of evidence cards
    cx, cy = int(w * 0.5), int(h * 0.54)
    ring_rx, ring_ry = int(w * 0.30), int(h * 0.27)
    k = len(items)
    positions = []
    for j in range(k):
        ang = -math.pi / 2 + (2 * math.pi * j / max(1, k))
        positions.append((int(cx + math.cos(ang) * ring_rx),
                          int(cy + math.sin(ang) * ring_ry)))

    # links: explicit, else hub-and-spoke from the centre (or a ring chain)
    edges = []
    if links:
        for e in links:
            if isinstance(e, (list, tuple)) and len(e) >= 2:
                edges.append((int(e[0]), int(e[1])))
    elif has_center:
        edges = [(-1, j) for j in range(k)]                        # -1 == centre node
    else:
        edges = [(j, (j + 1) % k) for j in range(k)] if k > 2 else [(0, 1)] if k == 2 else []

    cw, ch = int(w * 0.155), int(h * 0.105)
    ccw, cch = int(w * 0.17), int(h * 0.12)
    cards = [_card(it["label"], cw, ch, pal) for it in items]
    center_card = _card(center, ccw, cch, pal, stressed=True) if has_center else None

    def _node_xy(idx):
        return (cx, cy) if idx == -1 else positions[idx]

    title_font = look.font("title", int(h * 0.036))

    td = Path(tempfile.mkdtemp())
    for i in range(n):
        t = i / fps
        frame = bed.copy()
        d = ImageDraw.Draw(frame, "RGBA")

        ta = look.ease_out_cubic(min(1.0, t / 0.5))
        if title and ta > 0.01:
            ti = look.text_with_glow(title.upper(), title_font, fill=ink, glow=pal["bg_b"],
                                     glow_radius=5, glow_alpha=0.0, pad=8)
            d.rectangle([int(w * 0.06), int(h * 0.085), int(w * 0.06) + int(ti.width * 0.5),
                         int(h * 0.09)], fill=(*accent, int(200 * ta)))
            frame = look.paste_center(frame, ti, cx=int(w * 0.06) + ti.width // 2,
                                      cy=int(h * 0.125), opacity=ta)
            d = ImageDraw.Draw(frame, "RGBA")

        # threads draw one at a time after nodes are up (start 1.3s, 0.45s each)
        for ei, (a, b) in enumerate(edges):
            start = 1.3 + ei * 0.45
            ea = look.ease_out_cubic(min(1.0, max(0.0, (t - start) / 0.4)))
            if ea <= 0.01:
                continue
            p0 = _node_xy(a); p2 = _node_xy(b)
            sag = int(h * 0.05) + int(math.hypot(p2[0] - p0[0], p2[1] - p0[1]) * 0.04)
            p1 = ((p0[0] + p2[0]) / 2, (p0[1] + p2[1]) / 2 + sag)
            pts = []
            steps = 36
            mm = max(1, int(steps * ea))
            for s in range(mm + 1):
                tt = s / steps
                x = (1 - tt) ** 2 * p0[0] + 2 * (1 - tt) * tt * p1[0] + tt * tt * p2[0]
                y = (1 - tt) ** 2 * p0[1] + 2 * (1 - tt) * tt * p1[1] + tt * tt * p2[1]
                pts.append((x, y))
            if len(pts) >= 2:
                d.line(pts, fill=(*thread, int(200 * ea)), width=2, joint="curve")

        # evidence cards pop in (staggered 0.5s start)
        for j, (px, py) in enumerate(positions):
            ca = look.ease_out_cubic(min(1.0, max(0.0, (t - (0.5 + j * 0.18)) / 0.4)))
            if ca <= 0.01:
                continue
            frame = look.paste_center(frame, cards[j], cx=px, cy=py, opacity=ca)
        d = ImageDraw.Draw(frame, "RGBA")

        # centre subject card on top (pops first)
        if center_card is not None:
            cca = look.ease_out_cubic(min(1.0, max(0.0, (t - 0.3) / 0.4)))
            if cca > 0.01:
                frame = look.paste_center(frame, center_card, cx=cx, cy=cy, opacity=cca)
                d = ImageDraw.Draw(frame, "RGBA")

        frame = look.vignette(frame, strength=0.6)
        frame = look.film_grain(frame, seed=seed, amount=5.0, t=t)
        fa = look.fade_alpha(t, dur, fps)
        if fa < 1.0:
            frame = look.fade_frame(frame, fa, pal)
        frame.save(td / f"f{i:05d}.png")

    try:
        from vidlore.ffmpeg_tool import ffmpeg_exe
        ff = ffmpeg_exe()
    except Exception:                                              # noqa: BLE001
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
