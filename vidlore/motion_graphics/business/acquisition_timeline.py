"""Primitive: acquisition_timeline  (Section C · business).

Grounded in a REAL frame audit — MagnatesMedia "J.P. Morgan: The Man Who Owned
America" 10:39-10:46 (the M&A / Northern-Securities consolidation beat), with the
network-reveal grammar cross-checked against the same film's profiteering beat
04:02-04:15.

  · reference_videos/frame_sequences/magnatesmedia/04_ma_network_companies/
    (10:39-10:46): a PERSISTENT spine of connector lines holds a hub; the absorbed
    companies do NOT appear all-at-once — each one folds onto the spine ONE AT A
    TIME (GE, then AT&T, …), revealed progressively over time.
  · reference_videos/frame_sequences/magnatesmedia/01_moneyflow_profiteering/
    (04:02-04:15): ONE persistent horizontal connector RAIL across a dark gritty
    field; a generic ICON+paper-label node sits on the rail (NOT a logo); a value
    slides along the rail and a count-up tally ("600%", "$33 BILLION") ticks up.

PRINCIPLE (the only thing borrowed): a parent absorbs targets over time on ONE
persistent connector spine; targets fold into the parent ONE AT A TIME (progressive
reveal, never all-at-once); a running count-up tally; GENERIC chips, never real
company logos.

This is an ORIGINAL Vidlore composition — the PRINCIPLE only, rebuilt on the
charcoal `look` system (our palette, condensed/serif type, easing, vignette,
grain). No MagnatesMedia asset/layout/colour/icon reproduced: no orange paper
cards, no teal radial web, no brand logos, no money-shower. A parent hub is fixed
left on a single horizontal rail; each target chip slides in from the right and its
connector branch DRAWS to the hub one at a time; an amber count-up tally updates as
each lands; restraint keeps ≤5 chips visible (earliest fades when a 6th arrives).
Pure-local, deterministic, footage-first, no paid API.

    render("x.mp4", parent="Northern Securities",
           targets=[{"label": "Great Northern", "year": "1901"},
                    {"label": "Northern Pacific", "year": "1901"},
                    {"label": "CB&Q", "year": "1901"}])
"""
from __future__ import annotations

import subprocess
import tempfile
import time
from pathlib import Path

from PIL import Image, ImageDraw

from .. import look

SPEC = {
    "id": "acquisition_timeline", "family": "business",
    "roles": ["acquisition", "acquisitions", "mergers", "m&a", "rollup",
              "roll_up", "consolidation", "takeover", "absorbed", "buyout",
              "holding_company", "empire_building", "subsidiaries", "absorb"],
    "niches_ok": ["business", "history"],
    "intensity_range": [2, 4], "duration_range": [5.5, 8.0],
    "easing": "easeInOutCubic", "audio_cue": "acquire_soft",
    "repeat_cooldown_s": 60, "per_video_cap": 1, "cost": "low",
    "layout_variants": ["spine_left"], "fullscreen": True,
    "review_override": ["parent", "targets", "palette"],
    "fallback": "bullet_list for a plain ordered list (no acquirer hub)",
    "required_inputs": ["parent", "targets"],
    "grounded_in": "magnatesmedia/04_ma_network_companies (10:39-10:46) + "
                   "01_moneyflow_profiteering (04:02-04:15)",
}

# the principle's hard line: progressive reveal, never an all-at-once explosion;
# restraint keeps the spine legible (the reference never crowds the frame).
_MAX_VISIBLE = 5


def _norm_targets(targets):
    out = []
    for tg in (targets or []):
        if isinstance(tg, str) and tg.strip():
            out.append({"label": tg.strip(), "year": ""})
        elif isinstance(tg, dict):
            lbl = str(tg.get("label") or tg.get("name") or "").strip()
            if lbl:
                yr = str(tg.get("year") or tg.get("date") or tg.get("note") or "").strip()
                out.append({"label": lbl, "year": yr})
    return out[:7]


def render(out_path, *, parent: str = "", targets=None, dur: float = 6.5,
           fps: int = 30, w: int = 1920, h: int = 1080,
           palette_name: str = "amber_gold", layout: str = "", seed: int = 0,
           crf: int = 18, tally_label: str = "acquired",
           tally_dollars: str = "") -> dict:
    """Build the acquisition-timeline clip.

    parent       : the acquirer / holding company (str).
    targets      : [{label, year?}, …] absorbed one at a time, in order.
    tally_label  : word after the count-up ("acquired" → "3 / 5 acquired").
    tally_dollars: optional $ figure that count-ups instead of the N/total
                   (e.g. "$400M") — the moneyflow tally variant.
    """
    pal = look.palette(palette_name)
    t0 = time.time()
    n = max(2, int(round(dur * fps)))
    parent = (parent or "Parent Co.").strip()
    items = _norm_targets(targets)
    if not items:
        items = [{"label": "Target A", "year": ""},
                 {"label": "Target B", "year": ""},
                 {"label": "Target C", "year": ""}]
    k = len(items)

    accent = tuple(pal["accent"]); accent_hi = tuple(pal["accent_hi"])
    ink = tuple(pal["text"]); muted = tuple(pal["muted"]); glow = tuple(pal["glow"])

    # ── persistent charcoal stage (lifted off-black so a dissolve never reads
    #    as a black frame — this composition is sparse like a stat card) ──
    bed = look.graded_background(w, h, pal, seed=seed, floor=look.CARD_STAGE_FLOOR)
    gd = ImageDraw.Draw(bed, "RGBA")
    step = int(h * 0.05)
    for gx in range(0, w, step):                                   # faint graticule
        gd.line([(gx, 0), (gx, h)], fill=(*muted, 10), width=1)
    for gy in range(0, h, step):
        gd.line([(0, gy), (w, gy)], fill=(*muted, 10), width=1)

    # ── geometry: ONE horizontal rail; parent hub fixed at left ──
    rail_y = int(h * 0.56)
    rail_x0, rail_x1 = int(w * 0.07), int(w * 0.95)
    hub_cx = int(w * 0.20)                                          # parent hub
    hub_w, hub_h = int(w * 0.20), int(h * 0.13)
    hub_x0, hub_x1 = hub_cx - hub_w // 2, hub_cx + hub_w // 2
    hub_y0, hub_y1 = rail_y - hub_h // 2, rail_y + hub_h // 2
    # target landing slots: evenly spaced along the rail to the right of the hub
    slot_x0 = int(w * 0.42); slot_x1 = int(w * 0.88)
    chip_w, chip_h = int(w * 0.155), int(h * 0.105)
    slots = k if k > 1 else 1
    slot_cx = [int(slot_x0 + (slot_x1 - slot_x0) * (i / max(1, slots - 1)))
               for i in range(k)] if k > 1 else [int((slot_x0 + slot_x1) / 2)]

    title_font = look.font("title", int(h * 0.040))
    hub_font = look.font("label", int(h * 0.030))
    chip_font = look.font("label", int(h * 0.024))
    year_font = look.font("label", int(h * 0.018))
    tally_num_font = look.font("numeral", int(h * 0.090))
    tally_lab_font = look.font("label", int(h * 0.026))

    # ── timeline: targets fold in ONE AT A TIME, evenly across the active window
    intro = 0.55                       # rail + hub establish
    outro = 0.55                       # settle before the dissolve-out
    win0, win1 = intro + 0.45, max(intro + 1.0, dur - outro)
    cadence = (win1 - win0) / max(1, k)          # seconds between arrivals
    reveal_dur = min(0.62, cadence * 0.92)       # one chip's slide+draw time
    arrive_t = [win0 + i * cadence for i in range(k)]

    title = f"{parent.upper()}  ·  ACQUISITIONS"

    td = Path(tempfile.mkdtemp())
    for fi in range(n):
        t = fi / fps
        frame = bed.copy()
        d = ImageDraw.Draw(frame, "RGBA")

        # how many have landed (for the tally); plus the in-flight one
        landed = sum(1 for at in arrive_t if t >= at + reveal_dur)

        # title (top-left, with an accent underscore that wipes in)
        ta = look.ease_out_cubic(min(1.0, t / 0.45))
        if ta > 0.01:
            ti = look.text_with_glow(title, title_font, fill=ink, glow=pal["bg_b"],
                                     glow_radius=4, glow_alpha=0.0, pad=8)
            d.rectangle([rail_x0, int(h * 0.155),
                         rail_x0 + int(ti.width * 0.46 * ta), int(h * 0.160)],
                        fill=(*accent, int(210 * ta)))
            frame = look.paste_center(frame, ti, cx=rail_x0 + ti.width // 2,
                                      cy=int(h * 0.205), opacity=ta)
            d = ImageDraw.Draw(frame, "RGBA")

        # ── persistent rail draws out from the hub (0.10-0.55s) ──
        ra = look.ease_out_cubic(min(1.0, max(0.0, (t - 0.10) / 0.45)))
        if ra > 0.01:
            rx = int(hub_cx + (rail_x1 - hub_cx) * ra)
            # base rail (muted) + an amber core for life
            d.line([(rail_x0, rail_y), (rx, rail_y)], fill=(*muted, 150), width=2)
            d.line([(hub_cx, rail_y), (rx, rail_y)], fill=(*accent, int(180 * ra)), width=3)

        # ── parent hub (fixed; establishes 0.0-0.5s) ──
        ha = look.ease_out_cubic(min(1.0, t / 0.5))
        if ha > 0.01:
            ho = int(255 * ha)
            d.rounded_rectangle([hub_x0, hub_y0, hub_x1, hub_y1], radius=int(hub_h * 0.22),
                                fill=(*pal["bg_b"], int(210 * ha)),
                                outline=(*accent_hi, ho), width=3)
            # a generic "hub" glyph: three converging ticks (NOT a logo)
            gx0 = hub_x0 + int(hub_w * 0.12)
            for dy in (-int(hub_h * 0.22), 0, int(hub_h * 0.22)):
                d.line([(gx0, rail_y + dy), (gx0 + int(hub_w * 0.10), rail_y)],
                       fill=(*accent, int(200 * ha)), width=3)
            hl = look.text_with_glow(parent, hub_font, fill=ink, glow=pal["bg_b"],
                                     glow_radius=3, glow_alpha=0.0, pad=6)
            maxw = hub_w - int(hub_w * 0.30)
            if hl.width > maxw:
                hl = hl.resize((maxw, max(1, int(hl.height * maxw / hl.width))), Image.LANCZOS)
            frame = look.paste_center(frame, hl, cx=hub_cx + int(hub_w * 0.10),
                                      cy=rail_y, opacity=ha)
            d = ImageDraw.Draw(frame, "RGBA")

        # ── targets fold in ONE AT A TIME (progressive reveal) ──
        # restraint: only the most recent _MAX_VISIBLE stay solid; the earliest
        # fade out (but their connector stub persists) when a 6th+ arrives.
        for i, it in enumerate(items):
            at = arrive_t[i]
            if t < at:
                continue                                    # not yet revealed
            p = look.ease_out_cubic(min(1.0, (t - at) / max(0.05, reveal_dur)))
            # fade-out for over-cap chips (keep the spine legible)
            vis = 1.0
            older_after = sum(1 for j in range(i + 1, k) if t >= arrive_t[j])
            if older_after >= _MAX_VISIBLE:
                fo = older_after - _MAX_VISIBLE + 1
                vis = max(0.18, 1.0 - 0.55 * fo)

            cx_to = slot_cx[i]
            cx_from = slot_cx[i] + int(w * 0.10)            # slides in from the right
            cx = int(cx_from + (cx_to - cx_from) * p)
            cy = rail_y - int(h * 0.215)                    # chips ride above the rail
            cy_from = cy - int(h * 0.05)
            cyy = int(cy_from + (cy - cy_from) * p)

            # connector branch DRAWS from the rail node up to the chip (one at a time)
            node_x = slot_cx[i]
            d.ellipse([node_x - 6, rail_y - 6, node_x + 6, rail_y + 6],
                      fill=(*accent_hi, int(235 * p * vis)))
            draw_p = look.ease_out_cubic(min(1.0, p * 1.15))
            conn_y1 = int(rail_y + (cyy + chip_h // 2 - rail_y) * draw_p)
            d.line([(node_x, rail_y), (node_x, conn_y1)],
                   fill=(*accent, int(210 * vis)), width=3)

            # the chip itself (generic rounded card + label + optional year scrim)
            co = max(0.0, min(1.0, p * vis))
            cx0, cx1 = cx - chip_w // 2, cx + chip_w // 2
            cy0, cy1 = cyy - chip_h // 2, cyy + chip_h // 2
            d.rounded_rectangle([cx0, cy0, cx1, cy1], radius=int(chip_h * 0.20),
                                fill=(*pal["bg_b"], int(205 * co)),
                                outline=(*muted, int(220 * co)), width=2)
            # a small "absorbed" amber pip on the chip's left edge
            d.ellipse([cx0 + int(chip_w * 0.06) - 5, cyy - 5,
                       cx0 + int(chip_w * 0.06) + 5, cyy + 5],
                      fill=(*accent, int(235 * co)))
            lab = look.text_with_glow(it["label"], chip_font, fill=ink, glow=pal["bg_b"],
                                      glow_radius=3, glow_alpha=0.0, pad=5)
            maxw = chip_w - int(chip_w * 0.22)
            if lab.width > maxw:
                lab = lab.resize((maxw, max(1, int(lab.height * maxw / lab.width))), Image.LANCZOS)
            ly = cyy - (int(chip_h * 0.14) if it["year"] else 0)
            frame = look.paste_center(frame, lab, cx=cx + int(chip_w * 0.06), cy=ly, opacity=co)
            d = ImageDraw.Draw(frame, "RGBA")
            if it["year"]:
                yr = look.text_with_glow(it["year"], year_font, fill=accent_hi,
                                         glow=pal["bg_b"], glow_radius=2, glow_alpha=0.0, pad=4)
                frame = look.paste_center(frame, yr, cx=cx + int(chip_w * 0.06),
                                          cy=cyy + int(chip_h * 0.22), opacity=co * 0.95)
                d = ImageDraw.Draw(frame, "RGBA")

        # ── running count-up tally (amber numeral with bloom), bottom-right ──
        tcx, tcy = int(w * 0.80), int(h * 0.80)
        tly = look.ease_out_cubic(min(1.0, max(0.0, (t - intro) / 0.4)))
        if tly > 0.01:
            if tally_dollars:
                # $ figure counts toward the final as chips land (+ in-flight ease)
                frac = 0.0
                for i in range(k):
                    if t >= arrive_t[i] + reveal_dur:
                        frac += 1.0
                    elif t >= arrive_t[i]:
                        frac += look.ease_out_cubic((t - arrive_t[i]) / max(0.05, reveal_dur))
                frac = frac / max(1, k)
                num_txt = _scale_dollars(tally_dollars, frac)
                lab_txt = tally_label.upper() if tally_label else ""
            else:
                num_txt = f"{landed} / {k}"
                lab_txt = tally_label.upper()
            ni = look.gold_fill(num_txt, tally_num_font, pal, glow_radius=14, pad=18)
            frame = look.paste_center(frame, ni, cx=tcx, cy=tcy, opacity=tly)
            d = ImageDraw.Draw(frame, "RGBA")
            if lab_txt:
                li = look.text_with_glow(lab_txt, tally_lab_font, fill=muted, glow=pal["bg_b"],
                                         glow_radius=2, glow_alpha=0.0, pad=4)
                frame = look.paste_center(frame, li, cx=tcx, cy=tcy + int(h * 0.075), opacity=tly * 0.9)
                d = ImageDraw.Draw(frame, "RGBA")
            # a hairline tying the tally to the rail's right end
            look.hairline(d, tcx, tcy - int(h * 0.075), int(w * 0.08), pal, alpha=tly)

        frame = look.vignette(frame, strength=0.62)
        frame = look.film_grain(frame, seed=seed, amount=4.5, t=t)
        fa = look.fade_alpha(t, dur, fps)
        if fa < 1.0:
            frame = look.fade_frame(frame, fa, pal)
        frame.save(td / f"f{fi:05d}.png")

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
            "parent": parent, "targets": k,
            "err": (r.stderr[-200:] if not ok else "")}


def _scale_dollars(final: str, frac: float) -> str:
    """Count a '$400M' / '$1.2B' style figure up to `final` by `frac` (0..1).
    Keeps the prefix/suffix; only the number scales. Falls back to the literal."""
    s = final.strip()
    pre = "$" if s.startswith("$") else ""
    body = s[len(pre):]
    suf = ""
    while body and body[-1].isalpha():
        suf = body[-1] + suf
        body = body[:-1]
    try:
        val = float(body)
    except ValueError:
        return final
    cur = val * max(0.0, min(1.0, frac))
    txt = f"{cur:.1f}".rstrip("0").rstrip(".") if "." in body or cur < 10 else f"{int(round(cur))}"
    return f"{pre}{txt}{suf}"
