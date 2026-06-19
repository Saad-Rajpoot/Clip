"""Primitive: cause_effect_chain.

A causation beat — 2-4 labelled CAUSE cards laid left→right, linked by bold gold
chevron ARROWS that draw in as each card lands, revealed like falling dominoes
("this led to this led to this"). The FINAL card is the OUTCOME / consequence:
crowned in gold (accent_hi), softly glowing and slightly larger, so it reads as
the climactic payoff and lands last with emphasis.

Distinct from `bullet_list` (numbered circular nodes explaining HOW a
mechanism works — the recipe) and `chronology_timeline` (dated events on a spine
— WHEN): this is WHY → and so → and so → THEREFORE. Rectangular cards, double
gold chevrons, a crowned outcome — not a numbered method flow.

Forensic principle (NOT asset copy): MagnatesMedia reveals a causal chain one
link at a time so the viewer feels the inevitability; the premium is the grade +
condensed caps on framed cards + clean drawn chevrons + a single emphasised
outcome, never a clip-art flowchart. Pure-local (PIL + numpy → ffmpeg). No API.

    render("x.mp4", steps=["Treaty humiliation", "Economic collapse",
           "Radical politics", "The march to war"],
           title="HOW ONE TREATY LIT THE FUSE")
"""
from __future__ import annotations

import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .. import look

SPEC = {
    "id": "cause_effect_chain", "family": "diagrams",
    "roles": ["cause", "effect", "causation", "consequence", "chain"],
    "niches_ok": ["history", "geopolitics", "business", "crime", "biography", "tech"],
    "intensity_range": [3, 5], "duration_range": [4.5, 7.0],
    "easing": "easeOutCubic", "audio_cue": "soft_domino_tick",
    "repeat_cooldown_s": 55, "per_video_cap": 2, "cost": "low",
    "layout_variants": ["domino_row"],
    "review_override": ["steps", "title", "palette"],
    "fallback": "bullet_list if the steps are a neutral sequence, not a causal chain",
}


def _coerce(steps):
    out = []
    for s in (steps or []):
        if isinstance(s, dict):
            out.append(str(s.get("label") or s.get("text") or "").strip())
        elif isinstance(s, (list, tuple)) and s:
            out.append(str(s[0]).strip())
        elif s:
            out.append(str(s).strip())
    return [s for s in out if s][:4]


def _wrap(d, text, font, max_w):
    """Wrap to <=2 short lines; if it still overflows, hard-truncate the 2nd."""
    words, lines, cur = text.split(), [], ""
    for wd in words:
        t = (cur + " " + wd).strip()
        if d.textlength(t, font=font) <= max_w or not cur:
            cur = t
        else:
            lines.append(cur); cur = wd
    if cur:
        lines.append(cur)
    return lines[:2]


def render(out_path, *, steps=None, title: str = "", dur: float = 6.0,
           fps: int = 30, w: int = 1920, h: int = 1080,
           palette_name: str = "amber_gold", layout: str = "", seed: int = 0,
           crf: int = 18) -> dict:
    pal = look.palette(palette_name)
    t0 = time.time()
    n = max(2, int(round(dur * fps)))
    data = _coerce(steps)
    if not data:
        data = ["Cause"]
    m = len(data)

    accent = tuple(pal["accent"]); accent_hi = tuple(pal["accent_hi"])
    text_c = tuple(pal["text"]); bg_b = tuple(pal["bg_b"])
    muted = tuple(pal["muted"])

    # ── card geometry: horizontal row, sized down as the chain grows ────────
    # Longest label biases the font down so 4 long causes stay readable.
    longest = max((len(s) for s in data), default=1)
    fscale = 1.0
    if m >= 4 or longest > 16:
        fscale = 0.86
    if m >= 4 and longest > 18:
        fscale = 0.78
    lab_font = look.font("label", int(h * 0.0335 * fscale))
    title_font = look.font("label", int(h * 0.030))

    span = w * (0.70 if m > 1 else 0.0)         # row width centre-to-centre
    left = w * 0.5 - span / 2
    xs = [left + (span * (k / (m - 1)) if m > 1 else 0) for k in range(m)]
    cy = int(h * 0.50)                          # card vertical centre

    # card body sizing (the last/outcome card is ~1.12x → reserve gaps for it)
    gap = (span / (m - 1)) if m > 1 else (w * 0.5)
    cw = int(min(w * 0.205, gap * 0.74)) if m > 1 else int(w * 0.30)
    ch = int(h * 0.205)
    rr = int(h * 0.022)                          # corner radius
    OUT = 1.12                                   # outcome card scale

    # connector chevron geometry (bold double chevron between adjacent cards)
    chev_h = int(ch * 0.20)

    def _card_layer(label, scale, is_outcome, reveal):
        """Render one card (RGBA, padded) at native size; caller scales/pastes.
        reveal in [0,1] drives the gold border 'ignition' on the outcome."""
        bw = int(cw * scale); bh = int(ch * scale)
        pad = int(h * 0.05)
        L = Image.new("RGBA", (bw + pad * 2, bh + pad * 2), (0, 0, 0, 0))
        dd = ImageDraw.Draw(L, "RGBA")
        x0, y0, x1, y1 = pad, pad, pad + bw, pad + bh
        if is_outcome:
            # outcome: warm gold-tinted plate, bright border, soft inner crown
            fill = (int(bg_b[0] * 0.45 + accent[0] * 0.32),
                    int(bg_b[1] * 0.45 + accent[1] * 0.28),
                    int(bg_b[2] * 0.45 + accent[2] * 0.18), 244)
            bord = accent_hi
            bordw = max(3, int(h * 0.0042))
        else:
            fill = (*bg_b, 232)
            bord = accent
            bordw = max(2, int(h * 0.0030))
        dd.rounded_rectangle([x0, y0, x1, y1], radius=int(rr * scale),
                             fill=fill, outline=bord, width=bordw)
        # top accent rule inside the card (a refined header tick)
        tickw = int(bw * 0.34)
        tcx = x0 + bw // 2
        dd.line([(tcx - tickw // 2, y0 + int(bh * 0.16)),
                 (tcx + tickw // 2, y0 + int(bh * 0.16))],
                fill=(*(accent_hi if is_outcome else accent), 255),
                width=max(2, int(h * 0.0030)))
        # label, wrapped + centred
        lines = _wrap(dd, label.upper(), lab_font, bw * 0.84)
        lh = int(h * 0.040 * scale)
        ty = y0 + bh // 2 - (len(lines) - 1) * lh // 2 + int(bh * 0.05)
        fillc = accent_hi if is_outcome else text_c
        for ln in lines:
            lw = dd.textlength(ln, font=lab_font)
            dd.text((tcx - lw / 2, ty - lab_font.size * 0.5), ln,
                    font=lab_font, fill=(*fillc, 255))
            ty += lh
        return L

    td = Path(tempfile.mkdtemp())
    for i in range(n):
        t = i / fps
        pr = i / max(1, n - 1)
        frame = look.graded_background(w, h, pal, seed=seed, drift=pr)
        d = ImageDraw.Draw(frame, "RGBA")

        # ── title on a hairline at the top ──────────────────────────────────
        if title:
            ta = look.ease_out_cubic(min(1.0, t / 0.5))
            ti = look.text_with_glow(title.upper(), title_font, fill=muted,
                                     glow=bg_b, glow_radius=4, glow_alpha=0.0,
                                     pad=16)
            frame = look.paste_center(frame, ti, cx=w * 0.5, cy=h * 0.165,
                                      opacity=ta)
            d = ImageDraw.Draw(frame, "RGBA")
            hw = look.ease_out_cubic(min(1.0, (t - 0.15) / 0.6))
            if hw > 0.01:
                look.hairline(d, int(w * 0.5), int(h * 0.215),
                              int(w * 0.20 * hw), pal)

        # ── chevron connectors first (so cards overlap them) ────────────────
        # Each chevron pair draws in right after its LEFT card has seated, then
        # the head sharpens — the "domino tips into the next" tick.
        for k in range(m - 1):
            aa = look.ease_out_cubic(max(0.0, (t - (0.55 + k * 0.62)) / 0.34))
            if aa <= 0.01:
                continue
            x0 = int(xs[k] + cw * 0.5 + cw * 0.04)
            x1 = int(xs[k + 1] - cw * (0.5 * (OUT if k + 1 == m - 1 else 1.0))
                     - cw * 0.04)
            if x1 <= x0:
                continue
            xe = int(x0 + (x1 - x0) * aa)
            col = (*accent, int(235 * aa))
            lw = max(3, int(h * 0.0050))
            # twin chevrons pointing right, spaced along the gap (drawn up to xe)
            seg = x1 - x0
            for cidx in (0.40, 0.78):
                cx0 = int(x0 + seg * cidx)
                if cx0 + chev_h > xe and aa < 1.0:
                    continue
                d.line([(cx0 - chev_h, cy - chev_h), (cx0, cy)], fill=col, width=lw)
                d.line([(cx0 - chev_h, cy + chev_h), (cx0, cy)], fill=col, width=lw)
            # leading connector spine
            d.line([(x0, cy), (xe, cy)], fill=(*accent, int(150 * aa)),
                   width=max(2, lw - 2))
            if aa > 0.9 and k + 1 == m - 1:
                # brighten the final chevron as it hands off to the outcome
                cb = int(x1)
                d.line([(cb - chev_h, cy - chev_h), (cb, cy)],
                       fill=(*accent_hi, 255), width=lw + 1)
                d.line([(cb - chev_h, cy + chev_h), (cb, cy)],
                       fill=(*accent_hi, 255), width=lw + 1)

        # ── cards, staggered left→right like dominoes ───────────────────────
        for k, lab in enumerate(data):
            is_out = (k == m - 1) and (m > 1)
            ca = look.ease_out_cubic(max(0.0, (t - (0.35 + k * 0.62)) / 0.5))
            if ca <= 0.01:
                continue
            rise = int(h * 0.045 * (1 - ca))     # slide up into place
            scale = (1.0 + (OUT - 1.0) * ca) if is_out else 1.0
            # outcome gets a soft gold bloom behind it as it seats
            if is_out and ca > 0.05:
                gl = Image.new("RGBA", (int(cw * 1.5), int(ch * 1.5)), (0, 0, 0, 0))
                gd = ImageDraw.Draw(gl)
                gp = int(cw * 0.16)
                gd.rounded_rectangle([gp, gp, gl.width - gp, gl.height - gp],
                                     radius=int(rr * 1.4),
                                     fill=(*pal["glow"], int(60 * ca)))
                gl = gl.filter(ImageFilter.GaussianBlur(int(h * 0.018)))
                frame = look.paste_center(frame, gl, cx=xs[k], cy=cy + rise,
                                          opacity=ca)
                d = ImageDraw.Draw(frame, "RGBA")
            cardL = _card_layer(lab, scale, is_out, ca)
            frame = look.paste_center(frame, cardL, cx=xs[k], cy=cy + rise,
                                      opacity=ca)
            d = ImageDraw.Draw(frame, "RGBA")
            # crown tick above the outcome card once it has fully landed
            if is_out and ca > 0.85:
                cr = look.ease_out_cubic((ca - 0.85) / 0.15)
                top_y = cy + rise - int(ch * OUT * 0.5) - int(h * 0.022)
                hw2 = int(cw * 0.20 * cr)
                d.line([(int(xs[k]) - hw2, top_y), (int(xs[k]) + hw2, top_y)],
                       fill=(*accent_hi, int(255 * cr)), width=max(2, int(h * 0.0040)))
                # two small flanking diamonds = the 'crown'
                for dx in (-hw2, hw2):
                    px = int(xs[k]) + dx
                    s = int(h * 0.007 * cr)
                    if s > 0:
                        d.polygon([(px, top_y - s), (px + s, top_y),
                                   (px, top_y + s), (px - s, top_y)],
                                  fill=(*accent_hi, int(255 * cr)))

        frame = look.vignette(frame, strength=0.58)
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
