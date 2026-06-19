"""Primitive: packet_path_trace  (Section C · systems).

Grounded in a REAL frame audit — Wendover "Cities at Sea / How Aircraft Carriers
Work" 04:08–04:13 (`reference_videos/frame_sequences/wendover/02_map_route_singapore/`,
frames 19–24): a FLOW is drawn as a PATH — a glowing line travels hop→hop across a
faint topology (a satellite map there), with a BRIGHT NODE landing at each waypoint
and a single white scrim callout that BRIGHTENS as the path arrives ("Louisville" →
"New York" → "Anchorage"). The motion is one travelling head with a follow-feel, not
a swarm — exactly the "flow = a path with a travelling token" lesson.

Crossed with the ColdFusion anti-dashboard doctrine
(`audit_reports/coldfusion_FRAME_AUDIT.md`): tech is a TREATMENT, not a literal
SaaS network diagram — ColdFusion never draws packet/neural/flow graphs; it keeps
graphics still, lets ONE highlight travel, holds ≤3 emphasized elements, uses a
two-pole dark palette, and dissolves in/out of footage. So this is NOT a dense
network graph and NOT a persistent HUD: a single packet/signal traces ONE path
through a faint NETWORK of nodes, ≤3 hops emphasized, one accent, dissolve via
`fade_frame`.

Original Vidlore design — the PRINCIPLE only (path-as-flow + travelling token +
arrival-brightened label + faint topology), rebuilt on the charcoal `look` system
with our cold_steel palette, typography, easing and grain. No asset, layout, colour
or label copied from the reference. Pure-local, deterministic, footage-first, no
paid API.

    render("x.mp4", hops=[{"label":"Your device"},{"label":"ISP router"},
                          {"label":"Exchange"},{"label":"Server"}])
"""
from __future__ import annotations

import math
import subprocess
import tempfile
import time
from pathlib import Path

from PIL import Image, ImageDraw

from .. import look

SPEC = {
    "id": "packet_path_trace", "family": "systems",
    "roles": ["packet", "signal", "network_path", "trace", "route", "request_path",
              "data_path", "hops", "flow", "traversal"],
    "niches_ok": ["tech", "business"],
    "intensity_range": [2, 4], "duration_range": [5.0, 7.0],
    "easing": "easeInOutCubic", "audio_cue": "ping_soft",
    "repeat_cooldown_s": 60, "per_video_cap": 1, "cost": "low",
    "layout_variants": ["pathtrace"],
    "review_override": ["hops", "palette"],
    "fallback": "system_planview_flow for an ordered stage/zone path",
    "required_inputs": ["hops"],
    "grounded_in": "wendover/02_map_route_singapore (04:08-04:13) + coldfusion treatment",
}


def _norm(hops):
    out = []
    for hpv in (hops or []):
        if isinstance(hpv, str) and hpv.strip():
            out.append({"label": hpv.strip(), "note": ""})
        elif isinstance(hpv, dict):
            lbl = str(hpv.get("label") or hpv.get("name") or "").strip()
            if lbl:
                out.append({"label": lbl,
                            "note": str(hpv.get("note") or hpv.get("metric") or "").strip()})
    return out[:5]


def _hop_points(items, w, h, seed):
    """Lay the hops out as a gently meandering left→right poly-path across the
    stage (the packet travels through the topology), with a deterministic vertical
    wander so it reads as a NETWORK route, not a straight pipe. Returns the list of
    (x, y) waypoints for the emphasized hops."""
    import random
    rng = random.Random(seed * 131 + len(items))
    k = len(items)
    x0, x1 = w * 0.13, w * 0.87
    cy = h * 0.52
    amp = h * 0.16
    pts = []
    for i in range(k):
        fx = i / max(1, k - 1)
        x = x0 + (x1 - x0) * fx
        # smooth alternating wander, damped at the ends, plus a small jitter
        wob = math.sin(fx * math.pi * (1.6 + 0.4 * (k - 2))) * (0.55 + 0.45 * math.sin(fx * math.pi))
        jit = (rng.random() - 0.5) * 0.28
        y = cy + amp * (wob + jit)
        y = max(h * 0.30, min(h * 0.74, y))
        pts.append((x, y))
    return pts


def _smooth_path(pts, samples_per_seg: int = 22):
    """Catmull-Rom spline through the hop waypoints → a gently curved network
    route (reads organic, not a zig-zag line chart). Passes EXACTLY through every
    waypoint; returns (curve_pts, anchor_idx) where anchor_idx[i] is the index in
    curve_pts that coincides with hop i (so the token lands precisely on nodes)."""
    if len(pts) < 3:
        return list(pts), list(range(len(pts)))
    P = [pts[0]] + list(pts) + [pts[-1]]            # clamp endpoints
    curve = []
    anchors = [0]
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        for s in range(samples_per_seg):
            tt = s / samples_per_seg
            t2, t3 = tt * tt, tt * tt * tt
            x = 0.5 * ((2 * p1[0]) + (-p0[0] + p2[0]) * tt +
                       (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2 +
                       (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3)
            y = 0.5 * ((2 * p1[1]) + (-p0[1] + p2[1]) * tt +
                       (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2 +
                       (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3)
            curve.append((x, y))
        anchors.append(len(curve))                  # next hop sits here
    curve.append(pts[-1])
    return curve, anchors


def _scatter_nodes(items_pts, w, h, seed):
    """A FAINT scatter of topology nodes (the network the packet lives in) — small
    charcoal dots, a few of them lightly connected, deliberately low-contrast so
    they recede behind the lit path. NOT a dense graph. Returns
    (nodes:list[(x,y,r)], links:list[(i,j)])."""
    import random
    rng = random.Random(seed * 977 + 7)
    nodes = []
    # keep faint topology nodes clear of the hop waypoints so the path stays clean
    avoid = [(px, py) for (px, py) in items_pts]
    tries = 0
    target = 26
    while len(nodes) < target and tries < target * 12:
        tries += 1
        x = rng.uniform(w * 0.06, w * 0.94)
        y = rng.uniform(h * 0.18, h * 0.86)
        if any((x - ax) ** 2 + (y - ay) ** 2 < (w * 0.045) ** 2 for ax, ay in avoid):
            continue
        if any((x - nx) ** 2 + (y - ny) ** 2 < (w * 0.035) ** 2 for nx, ny, _ in nodes):
            continue
        nodes.append((x, y, rng.uniform(1.6, 3.2)))
    # a sparse set of faint links between near neighbours (network texture)
    links = []
    for i in range(len(nodes)):
        best = None
        for j in range(len(nodes)):
            if i == j:
                continue
            dd = (nodes[i][0] - nodes[j][0]) ** 2 + (nodes[i][1] - nodes[j][1]) ** 2
            if dd < (w * 0.14) ** 2 and (best is None or dd < best[1]):
                if rng.random() < 0.6:
                    best = (j, dd)
        if best is not None and i < best[0]:
            links.append((i, best[0]))
    return nodes, links[:14]


def _draw_topology(bed, nodes, links, muted):
    """Paint the faint static network topology ONCE onto the bed (so it is part of
    the charcoal stage, never animated/busy)."""
    d = ImageDraw.Draw(bed, "RGBA")
    for i, j in links:
        d.line([(nodes[i][0], nodes[i][1]), (nodes[j][0], nodes[j][1])],
               fill=(*muted, 26), width=1)
    for (x, y, r) in nodes:
        d.ellipse([x - r, y - r, x + r, y + r], fill=(*muted, 60))
        d.ellipse([x - r - 2, y - r - 2, x + r + 2, y + r + 2], outline=(*muted, 24), width=1)
    return bed


def _poly_len(pts):
    seg = []
    tot = 0.0
    for i in range(1, len(pts)):
        dx = pts[i][0] - pts[i - 1][0]
        dy = pts[i][1] - pts[i - 1][1]
        d = math.hypot(dx, dy)
        seg.append(d)
        tot += d
    return seg, max(1e-6, tot)


def _point_at(pts, seg, tot, frac):
    """Position (x,y) and the index of the most-recently-passed waypoint at path
    fraction `frac` in [0,1] along the poly-line."""
    frac = max(0.0, min(1.0, frac))
    target = frac * tot
    acc = 0.0
    for i in range(len(seg)):
        if acc + seg[i] >= target or i == len(seg) - 1:
            r = (target - acc) / max(1e-6, seg[i])
            r = max(0.0, min(1.0, r))
            x = pts[i][0] + (pts[i + 1][0] - pts[i][0]) * r
            y = pts[i][1] + (pts[i + 1][1] - pts[i][1]) * r
            return x, y, i + (1 if r >= 0.999 else 0)
        acc += seg[i]
    return pts[-1][0], pts[-1][1], len(pts) - 1


def _draw_path(d, pts, seg, reach_frac, accent, accent_hi, glow):
    """Draw the lit path from the start UP TO the current reach (a glowing trail
    with a soft halo) and return the head (x,y). Drawn as short segments so it
    grows smoothly; faint un-reached path is hinted ahead as a guide."""
    _, tot = seg, None
    seglen, total = _poly_len(pts)
    # faint guide for the not-yet-traced remainder (very low contrast)
    for i in range(1, len(pts)):
        d.line([pts[i - 1], pts[i]], fill=(*accent, 28), width=2)
    # lit trail up to reach
    steps = 220
    last = pts[0]
    hx, hy = pts[0]
    for s in range(1, steps + 1):
        f = (s / steps) * reach_frac
        x, y, _ = _point_at(pts, seglen, total, f)
        # glow underlay (wide, soft) then bright core
        d.line([last, (x, y)], fill=(*glow, 60), width=8)
        d.line([last, (x, y)], fill=(*accent, 150), width=4)
        d.line([last, (x, y)], fill=(*accent_hi, 210), width=2)
        last = (x, y)
        hx, hy = x, y
    return hx, hy


def render(out_path, *, hops=None, dur: float = 6.0, fps: int = 30,
           w: int = 1920, h: int = 1080, palette_name: str = "cold_steel",
           layout: str = "", seed: int = 0, crf: int = 18) -> dict:
    pal = look.palette(palette_name)
    t0 = time.time()
    n = max(2, int(round(dur * fps)))
    items = _norm(hops)
    if len(items) < 2:
        items = [{"label": "Your device", "note": ""}, {"label": "Network", "note": ""},
                 {"label": "Server", "note": ""}]
    k = len(items)
    accent = tuple(pal["accent"]); accent_hi = tuple(pal["accent_hi"])
    glow = tuple(pal["glow"]); ink = tuple(pal["text"]); muted = tuple(pal["muted"])

    # charcoal cinematic stage (lifted off black) + a faint, static topology of
    # network nodes the packet lives in (painted once → never busy/animated).
    floor = getattr(look, "CARD_STAGE_FLOOR", 52.0)
    bed = look.graded_background(w, h, pal, seed=seed, floor=floor) \
        if hasattr(look, "graded_background") else Image.new("RGB", (w, h), tuple(pal["bg_b"]))
    pts = _hop_points(items, w, h, seed)
    # smooth the hop waypoints into a gently curved route (organic network, not a
    # zig-zag chart). The token & path travel `curve`; nodes/labels live on the
    # original `pts` (== curve[anchor]) so they land exactly on each hop.
    curve, anchors = _smooth_path(pts)
    cseg, ctot = _poly_len(curve)
    # cumulative length up to each anchor → the path-fraction the token reaches
    # each hop at (so node landings + arrival pings + label brightening line up).
    cum = [0.0]
    for s in cseg:
        cum.append(cum[-1] + s)
    arr_fr = [cum[min(a, len(cum) - 1)] / ctot for a in anchors]
    nodes, links = _scatter_nodes(pts, w, h, seed)
    bed = _draw_topology(bed, nodes, links, muted)

    title_font = look.font("title", int(h * 0.034))
    hop_font = look.font("label", int(h * 0.028))
    note_font = look.font("label", int(h * 0.020))

    # the packet traverses the whole path between t_start and t_end (leaving a
    # short settle head/tail for the dissolve). One travelling head, not a swarm.
    t_start, t_end = 0.9, max(1.4, dur - 1.05)

    td = Path(tempfile.mkdtemp())
    for i in range(n):
        t = i / fps
        frame = bed.copy()
        d = ImageDraw.Draw(frame, "RGBA")

        # progress of the travelling token along the path (eased for follow-feel)
        sp = (t - t_start) / max(0.1, (t_end - t_start))
        reach = look.ease_in_out_cubic(max(0.0, min(1.0, sp)))

        # ACTIVE hop = the waypoint the packet is at / heading INTO — so the bright
        # callout FOLLOWS the token (Wendover: the label brightens as the path
        # arrives, the previous one dims) rather than the last-arrived lingering.
        active = 0
        for hi in range(k):
            if reach >= arr_fr[hi] - 1e-3:
                active = min(k - 1, hi + 1)
        if reach >= 1.0 - 1e-3:
            active = k - 1

        # the lit path drawn up to the token + its glowing head
        if reach > 0.001:
            hx, hy = _draw_path(d, curve, (cseg, ctot), reach, accent, accent_hi, glow)
            # travelling packet token: a bright comet head that DOMINATES every
            # static node — wide soft halo, motion trail behind, white-hot core.
            if 0.0 < reach < 0.999:
                # multi-tap comet trail (fading taps behind the head = follow-feel)
                for ti, (tb, tw, ta) in enumerate(((0.045, 5, 150), (0.085, 3, 90),
                                                   (0.13, 2, 50))):
                    bx, by, _ = _point_at(curve, cseg, ctot, max(0.0, reach - tb))
                    d.line([(bx, by), (hx, hy)], fill=(*accent_hi, ta), width=tw)
            d.ellipse([hx - 22, hy - 22, hx + 22, hy + 22], fill=(*glow, 70))
            d.ellipse([hx - 13, hy - 13, hx + 13, hy + 13], fill=(*glow, 130))
            d.ellipse([hx - 8, hy - 8, hx + 8, hy + 8], fill=(*accent_hi, 245))
            d.ellipse([hx - 3.5, hy - 3.5, hx + 3.5, hy + 3.5], fill=(255, 255, 255, 255))
        # each waypoint NODE lands as the token arrives + a soft arrival "ping"
        # ring pulses outward (the arrival beat). Landed nodes sit a touch smaller
        # & cooler than the live comet head so the TOKEN always leads the eye.
        for hi, (px, py) in enumerate(pts):
            af = arr_fr[hi]
            # arrival progress (0 just before, 1 fully landed) over a small window
            ap = look.ease_out_cubic(max(0.0, min(1.0, (reach - (af - 0.10)) / 0.10)))
            if hi == 0:
                ap = 1.0  # source node present from the start
            if ap <= 0.01:
                continue
            # ping ring (expands + fades over a generous window right at arrival)
            ping = max(0.0, min(1.0, (reach - af) / 0.18))
            if 0.0 < ping < 1.0 and hi > 0:
                rr = 9 + ping * 46
                d.ellipse([px - rr, py - rr, px + rr, py + rr],
                          outline=(*accent_hi, int(170 * (1 - ping))), width=2)
                rr2 = 9 + ping * 30
                d.ellipse([px - rr2, py - rr2, px + rr2, py + rr2],
                          outline=(*glow, int(90 * (1 - ping))), width=1)
            # the settled node marker (smaller than the head; glow halo + cool core)
            base_r = 5
            d.ellipse([px - base_r - 3, py - base_r - 3, px + base_r + 3, py + base_r + 3],
                      fill=(*glow, int(60 * ap)))
            d.ellipse([px - base_r, py - base_r, px + base_r, py + base_r],
                      fill=(*accent_hi, int(225 * ap)))
            d.ellipse([px - 1.5, py - 1.5, px + 1.5, py + 1.5],
                      fill=(255, 255, 255, int(220 * ap)))

        # ── labels: a readable scrim callout per emphasized hop that BRIGHTENS as
        # the token arrives (the Wendover arrival-label move). Already-passed hops
        # dim to muted; the active hop is inked + accented. ≤ the hop count.
        for hi, it in enumerate(items):
            px, py = pts[hi]
            af = arr_fr[hi]
            ap = look.ease_out_cubic(max(0.0, min(1.0, (reach - (af - 0.10)) / 0.12)))
            if hi == 0:
                ap = look.ease_out_cubic(max(0.0, min(1.0, (t - 0.5) / 0.4)))
            if ap <= 0.02:
                continue
            is_active = (hi == active)
            txt_col = ink if is_active else muted
            lbl = look.text_with_glow(it["label"], hop_font, fill=txt_col,
                                      glow=pal["bg_b"], glow_radius=3, glow_alpha=0.0, pad=6)
            note = it["note"]
            ni = None
            if note and is_active:
                ni = look.text_with_glow(note, note_font, fill=muted, glow=pal["bg_b"],
                                         glow_radius=2, glow_alpha=0.0, pad=4)
            # callout box geometry — placed above or below the node to stay on-screen
            bw = max(lbl.width, (ni.width if ni else 0)) + 26
            bh = lbl.height + (ni.height + 4 if ni else 0) + 14
            above = py > h * 0.5
            boxy = (py - 30 - bh) if above else (py + 30)
            boxx = int(min(max(px - bw / 2, w * 0.035), w * 0.965 - bw))
            boxy = int(min(max(boxy, h * 0.04), h * 0.94 - bh))
            # scrim (dark rounded plate) + accent edge for the active hop
            scr_a = int(150 * ap) if not is_active else int(185 * ap)
            d.rounded_rectangle([boxx, boxy, boxx + bw, boxy + bh], radius=10,
                                fill=(8, 11, 16, scr_a),
                                outline=((*accent_hi, int(200 * ap)) if is_active
                                         else (*muted, int(90 * ap))),
                                width=2 if is_active else 1)
            # leader line from node to the callout
            lead_y = (boxy + bh) if above else boxy
            d.line([(px, py), (px, lead_y)], fill=(*accent, int(150 * ap)), width=2)
            # composite the text
            frame = look.paste_center(frame, lbl, cx=boxx + bw / 2,
                                      cy=boxy + 7 + lbl.height / 2, opacity=ap)
            if ni:
                frame = look.paste_center(frame, ni, cx=boxx + bw / 2,
                                          cy=boxy + bh - 7 - ni.height / 2, opacity=ap)
            d = ImageDraw.Draw(frame, "RGBA")

        frame = look.vignette(frame, strength=0.62)
        frame = look.film_grain(frame, seed=seed, amount=4.5, t=t)
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
