"""Primitive: flowchart_decision.

A judgement beat — a single yes/no DECISION FORK. One QUESTION node sits at
top-centre (a gold-outlined diamond holding the question wrapped to <=2 condensed
caps lines); from its base TWO connector lines DIVERGE down-left and down-right to
two OUTCOME cards, each branch carrying a small chip near its midpoint labelled
YES / NO. If a side is `chosen`, that whole path (line + chip + outcome) ignites
to accent_hi gold and the other dims — the verdict.

Distinct from `bullet_list` (a linear numbered left→right row — HOW) and
`cause_effect_chain` (a linear domino row — WHY → THEREFORE): this one BRANCHES.
It is a Y-shaped fork, not a row — "if this, then that; if not, then this".

Forensic principle (NOT asset copy): MagnatesMedia stages a decision so the
viewer feels the pivot — the question lands, the branches draw, the verdict
brightens. The premium is the grade + serif/condensed type + clean drawn
diverging connectors + restraint, never a clip-art flowchart. Pure-local.

    render("x.mp4", question="Did the alibi hold up?",
           yes="Released without charge", no="Held for questioning",
           chosen="no", title="THE DECISION")
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
    "id": "flowchart_decision", "family": "diagrams",
    "roles": ["decision", "branch", "flowchart", "choice", "fork"],
    "niches_ok": ["business", "tech", "crime", "geopolitics", "history", "biography"],
    "intensity_range": [2, 4], "duration_range": [4.5, 7.0],
    "easing": "easeOutCubic", "audio_cue": "soft_branch_tick",
    "repeat_cooldown_s": 55, "per_video_cap": 2, "cost": "low",
    "layout_variants": ["yes_no_fork"],
    "review_override": ["question", "yes", "no", "chosen", "palette"],
    "fallback": "bullet_list if the logic is linear rather than a branch",
}


def _txt(v) -> str:
    """Defensive scalar→str: tolerate None / list / tuple / dict."""
    if v is None:
        return ""
    if isinstance(v, (list, tuple)):
        v = v[0] if v else ""
    if isinstance(v, dict):
        v = v.get("label") or v.get("text") or v.get("value") or ""
    return str(v).strip()


def _wrap(d, text, font, max_w, max_lines=2):
    """Wrap to <=max_lines lines; hard-truncate the last with an ellipsis."""
    words, lines, cur = text.split(), [], ""
    for wd in words:
        t = (cur + " " + wd).strip()
        if d.textlength(t, font=font) <= max_w or not cur:
            cur = t
        else:
            lines.append(cur)
            cur = wd
    if cur:
        lines.append(cur)
    if len(lines) > max_lines:
        keep = lines[:max_lines]
        last = keep[-1]
        while last and d.textlength(last + "…", font=font) > max_w:
            last = last[:-1].rstrip()
        keep[-1] = (last + "…") if last else "…"
        lines = keep
    return lines


def _norm_chosen(v) -> str:
    s = _txt(v).lower()
    if s in ("yes", "y", "true", "left", "1"):
        return "yes"
    if s in ("no", "n", "false", "right", "0"):
        return "no"
    return ""


def render(out_path, *, question: str = "", yes: str = "", no: str = "",
           yes_label: str = "YES", no_label: str = "NO", chosen: str = "",
           title: str = "", dur: float = 6.0, fps: int = 30, w: int = 1920,
           h: int = 1080, palette_name: str = "amber_gold", layout: str = "",
           seed: int = 0, crf: int = 18) -> dict:
    pal = look.palette(palette_name)
    t0 = time.time()
    n = max(2, int(round(dur * fps)))

    q_text = _txt(question) or "Decision"
    yes_out = _txt(yes) or "Yes"
    no_out = _txt(no) or "No"
    yl = (_txt(yes_label) or "YES").upper()
    nl = (_txt(no_label) or "NO").upper()
    pick = _norm_chosen(chosen)

    accent = tuple(pal["accent"]); accent_hi = tuple(pal["accent_hi"])
    text_c = tuple(pal["text"]); bg_b = tuple(pal["bg_b"])
    muted = tuple(pal["muted"]); glow_c = tuple(pal["glow"])

    title_font = look.font("label", int(h * 0.030))
    # question font shrinks for long prompts so it stays inside the diamond
    qf_scale = 1.0
    if len(q_text) > 22:
        qf_scale = 0.86
    if len(q_text) > 34:
        qf_scale = 0.74
    q_font = look.font("label", int(h * 0.036 * qf_scale))
    chip_font = look.font("label", int(h * 0.026))
    # outcome label font shrinks as the text grows
    longest = max(len(yes_out), len(no_out))
    of_scale = 1.0
    if longest > 16:
        of_scale = 0.88
    if longest > 22:
        of_scale = 0.78
    out_font = look.font("label", int(h * 0.0325 * of_scale))

    # ── geometry ────────────────────────────────────────────────────────────
    qcx = int(w * 0.5)
    qcy = int(h * 0.34)                       # question node centre
    q_rx = int(w * 0.150)                     # diamond half-width
    q_ry = int(h * 0.130)                     # diamond half-height

    ocy = int(h * 0.78)                       # outcome cards vertical centre
    lx = int(w * 0.255)                       # left  outcome centre x
    rx = int(w * 0.745)                       # right outcome centre x
    cw = int(w * 0.300)                       # outcome card width
    chh = int(h * 0.150)                      # outcome card height
    rr = int(h * 0.024)                       # card corner radius

    # branch endpoints (top-centre of each outcome card)
    fork_x, fork_y = qcx, qcy + q_ry         # lines leave the diamond's bottom tip
    lex, ley = lx, ocy - chh // 2 - int(h * 0.006)
    rex, rey = rx, ocy - chh // 2 - int(h * 0.006)
    # chip positions: roughly the midpoint of each diverging line
    lmx, lmy = int((fork_x + lex) / 2), int((fork_y + ley) / 2)
    rmx, rmy = int((fork_x + rex) / 2), int((fork_y + rey) / 2)

    def _diamond_pts(cx, cy, rxx, ryy, s=1.0):
        rxx = int(rxx * s); ryy = int(ryy * s)
        return [(cx, cy - ryy), (cx + rxx, cy), (cx, cy + ryy), (cx - rxx, cy)]

    def _branch_color(side, base_a):
        """Highlighted side → accent_hi; dimmed side → muted-ish; neutral → accent."""
        if pick == side:
            return (*accent_hi, int(255 * base_a))
        if pick and pick != side:
            return (*muted, int(150 * base_a))
        return (*accent, int(235 * base_a))

    def _chip_layer(label, side, reveal):
        """Small pill chip with the branch label (RGBA, padded)."""
        hi = (pick == side)
        dim = (pick and pick != side)
        pad = int(h * 0.05)
        tmp = Image.new("RGBA", (8, 8))
        td_ = ImageDraw.Draw(tmp)
        tw = int(td_.textlength(label, font=chip_font))
        thh = int(chip_font.size * 1.0)
        cw_ = tw + int(h * 0.044)
        ch_ = thh + int(h * 0.026)
        L = Image.new("RGBA", (cw_ + pad * 2, ch_ + pad * 2), (0, 0, 0, 0))
        dd = ImageDraw.Draw(L, "RGBA")
        x0, y0, x1, y1 = pad, pad, pad + cw_, pad + ch_
        if hi:
            face = (int(bg_b[0] * 0.40 + accent[0] * 0.42),
                    int(bg_b[1] * 0.40 + accent[1] * 0.36),
                    int(bg_b[2] * 0.40 + accent[2] * 0.24), 248)
            bord = accent_hi; txtc = accent_hi
            bw_ = max(2, int(h * 0.0034))
        elif dim:
            face = (*bg_b, 220)
            bord = muted; txtc = muted
            bw_ = max(2, int(h * 0.0024))
        else:
            face = (*bg_b, 236)
            bord = accent; txtc = accent_hi
            bw_ = max(2, int(h * 0.0028))
        dd.rounded_rectangle([x0, y0, x1, y1], radius=ch_ // 2,
                             fill=face, outline=bord, width=bw_)
        lw = dd.textlength(label, font=chip_font)
        dd.text(((L.width - lw) / 2, (L.height - chip_font.size) / 2 - h * 0.004),
                label, font=chip_font, fill=(*txtc, 255))
        return L

    def _outcome_layer(label, side, reveal):
        """One outcome card (RGBA, padded). Chosen → gold plate + bright border."""
        hi = (pick == side)
        dim = (pick and pick != side)
        pad = int(h * 0.05)
        L = Image.new("RGBA", (cw + pad * 2, chh + pad * 2), (0, 0, 0, 0))
        dd = ImageDraw.Draw(L, "RGBA")
        x0, y0, x1, y1 = pad, pad, pad + cw, pad + chh
        if hi:
            face = (int(bg_b[0] * 0.46 + accent[0] * 0.32),
                    int(bg_b[1] * 0.46 + accent[1] * 0.28),
                    int(bg_b[2] * 0.46 + accent[2] * 0.18), 246)
            bord = accent_hi
            bw_ = max(3, int(h * 0.0044))
            fillc = accent_hi
        elif dim:
            face = (*bg_b, 214)
            bord = muted
            bw_ = max(2, int(h * 0.0026))
            fillc = muted
        else:
            face = (*bg_b, 232)
            bord = accent
            bw_ = max(2, int(h * 0.0032))
            fillc = text_c
        dd.rounded_rectangle([x0, y0, x1, y1], radius=rr, fill=face,
                             outline=bord, width=bw_)
        # refined header tick inside the card
        tickw = int(cw * 0.30)
        tcx = x0 + cw // 2
        tick_c = accent_hi if hi else (muted if dim else accent)
        dd.line([(tcx - tickw // 2, y0 + int(chh * 0.18)),
                 (tcx + tickw // 2, y0 + int(chh * 0.18))],
                fill=(*tick_c, 255), width=max(2, int(h * 0.0030)))
        # wrapped centred label
        lines = _wrap(dd, label.upper(), out_font, cw * 0.84, max_lines=2)
        lh = int(out_font.size * 1.22)
        ty = y0 + int(chh * 0.52) - (len(lines) - 1) * lh // 2
        for ln in lines:
            lw = dd.textlength(ln, font=out_font)
            dd.text((tcx - lw / 2, ty - out_font.size * 0.5), ln,
                    font=out_font, fill=(*fillc, 255))
            ty += lh
        return L

    # reveal schedule (seconds)
    T_Q = 0.45        # question node seats by ~0.45+0.55
    T_LINE = 1.15     # branch lines start drawing
    T_LINE_D = 0.55   # line draw duration
    T_CHIP = 1.85     # chips pop
    T_CARD = 2.20     # outcome cards rise

    arrow = int(h * 0.018)            # arrowhead size

    td = Path(tempfile.mkdtemp())
    for i in range(n):
        t = i / fps
        pr = i / max(1, n - 1)
        frame = look.graded_background(w, h, pal, seed=seed, drift=pr)
        d = ImageDraw.Draw(frame, "RGBA")

        # ── title on a hairline at the very top ─────────────────────────────
        if title:
            ta = look.ease_out_cubic(min(1.0, t / 0.5))
            ti = look.text_with_glow(_txt(title).upper(), title_font, fill=muted,
                                     glow=bg_b, glow_radius=4, glow_alpha=0.0,
                                     pad=16)
            frame = look.paste_center(frame, ti, cx=w * 0.5, cy=h * 0.085,
                                      opacity=ta)
            d = ImageDraw.Draw(frame, "RGBA")
            hw = look.ease_out_cubic(min(1.0, (t - 0.15) / 0.6))
            if hw > 0.01:
                look.hairline(d, int(w * 0.5), int(h * 0.130),
                              int(w * 0.22 * hw), pal)

        # ── branch connectors (drawn before nodes so nodes overlap them) ────
        for side, (ex, ey, mx, my) in (("yes", (lex, ley, lmx, lmy)),
                                        ("no", (rex, rey, rmx, rmy))):
            la = look.ease_out_cubic(max(0.0, (t - T_LINE) / T_LINE_D))
            if la <= 0.01:
                continue
            cx_e = int(fork_x + (ex - fork_x) * la)
            cy_e = int(fork_y + (ey - fork_y) * la)
            col = _branch_color(side, 1.0)
            lw = max(3, int(h * 0.0048))
            if pick == side:
                lw = max(4, int(h * 0.0058))
            d.line([(fork_x, fork_y), (cx_e, cy_e)], fill=col, width=lw)
            # arrowhead at the landing corner once the line has nearly arrived
            if la > 0.92:
                ah = look.ease_out_cubic((la - 0.92) / 0.08)
                dxn = (ex - fork_x); dyn = (ey - fork_y)
                dl = max(1.0, (dxn * dxn + dyn * dyn) ** 0.5)
                ux, uy = dxn / dl, dyn / dl
                px, py = -uy, ux
                s = int(arrow * ah)
                tipx, tipy = ex, ey
                d.polygon([(tipx, tipy),
                           (int(tipx - ux * s * 1.7 + px * s), int(tipy - uy * s * 1.7 + py * s)),
                           (int(tipx - ux * s * 1.7 - px * s), int(tipy - uy * s * 1.7 - py * s))],
                          fill=(col[0], col[1], col[2], 255))

        # small fork node (a dot) at the base of the diamond where lines split
        forka = look.ease_out_cubic(max(0.0, (t - (T_LINE - 0.1)) / 0.3))
        if forka > 0.01:
            fr = int(h * 0.009 * forka)
            fc = accent_hi if pick else accent
            d.ellipse([fork_x - fr, fork_y - fr, fork_x + fr, fork_y + fr],
                      fill=(*fc, int(255 * forka)))

        # ── question node: a gold-outlined diamond, fades + scales in first ──
        qa = look.ease_out_cubic(max(0.0, (t - T_Q + 0.45) / 0.55))
        if qa > 0.01:
            qs = 0.86 + 0.14 * qa                # subtle scale-in
            # soft bloom behind the diamond as it seats
            if qa > 0.05:
                gw, gh = int(q_rx * 2.6), int(q_ry * 2.6)
                gl = Image.new("RGBA", (gw, gh), (0, 0, 0, 0))
                gd = ImageDraw.Draw(gl)
                gd.polygon(_diamond_pts(gw // 2, gh // 2, q_rx, q_ry, qs * 0.96),
                           fill=(*glow_c, int(46 * qa)))
                gl = gl.filter(ImageFilter.GaussianBlur(int(h * 0.020)))
                frame = look.paste_center(frame, gl, cx=qcx, cy=qcy, opacity=qa)
                d = ImageDraw.Draw(frame, "RGBA")
            # the diamond itself (filled dark plate + double gold outline)
            pts = _diamond_pts(qcx, qcy, q_rx, q_ry, qs)
            d.polygon(pts, fill=(*bg_b, int(238 * qa)),
                      outline=(*accent_hi, int(255 * qa)),
                      width=max(3, int(h * 0.0042)))
            # inner hairline diamond for a refined double-border
            ipts = _diamond_pts(qcx, qcy, q_rx, q_ry, qs * 0.90)
            d.line(ipts + [ipts[0]], fill=(*accent, int(170 * qa)),
                   width=max(1, int(h * 0.0016)))
            # question text, wrapped to <=2 lines, centred in the diamond
            lines = _wrap(d, q_text.upper(), q_font, q_rx * 1.30, max_lines=2)
            lh = int(q_font.size * 1.24)
            qty = qcy - (len(lines) - 1) * lh // 2
            for ln in lines:
                qi = look.text_with_glow(ln, q_font, fill=text_c, glow=bg_b,
                                         glow_radius=4, glow_alpha=0.0, pad=10)
                frame = look.paste_center(frame, qi, cx=qcx, cy=qty, opacity=qa)
                qty += lh
            d = ImageDraw.Draw(frame, "RGBA")

        # ── branch chips at each midpoint, pop after the lines land ─────────
        for side, (mx, my), label in (("yes", (lmx, lmy), yl),
                                       ("no", (rmx, rmy), nl)):
            ka = look.ease_out_cubic(max(0.0, (t - T_CHIP) / 0.45))
            if ka <= 0.01:
                continue
            pop = 0.80 + 0.20 * ka               # tiny pop-in
            chip = _chip_layer(label, side, ka)
            frame = look.paste_center(frame, chip, cx=mx, cy=my,
                                      scale=pop, opacity=ka)
            d = ImageDraw.Draw(frame, "RGBA")

        # ── outcome cards rise + fade in last ───────────────────────────────
        for side, ccx, label in (("yes", lx, yes_out), ("no", rx, no_out)):
            ca = look.ease_out_cubic(max(0.0, (t - T_CARD) / 0.55))
            if ca <= 0.01:
                continue
            rise = int(h * 0.05 * (1 - ca))
            # chosen card gets a soft gold bloom behind it
            if pick == side and ca > 0.05:
                gl = Image.new("RGBA", (int(cw * 1.4), int(chh * 1.6)),
                               (0, 0, 0, 0))
                gd = ImageDraw.Draw(gl)
                gp = int(cw * 0.14)
                gd.rounded_rectangle([gp, gp, gl.width - gp, gl.height - gp],
                                     radius=int(rr * 1.4),
                                     fill=(*glow_c, int(64 * ca)))
                gl = gl.filter(ImageFilter.GaussianBlur(int(h * 0.018)))
                frame = look.paste_center(frame, gl, cx=ccx, cy=ocy + rise,
                                          opacity=ca)
                d = ImageDraw.Draw(frame, "RGBA")
            cardL = _outcome_layer(label, side, ca)
            frame = look.paste_center(frame, cardL, cx=ccx, cy=ocy + rise,
                                      opacity=ca)
            d = ImageDraw.Draw(frame, "RGBA")

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
