"""Primitive: labeled_cross_section.

A science / engineering EXPLAINER cutaway. A stylised SECTIONED vessel (a capsule
column with a bright accent CUT-EDGE and 2-3 interior compartments) sits on the
graded charcoal stage; an animated ACTIVE-PATH line is drawn down its axis (the
"beam / flow / signal"); thin LEADER LINES run from interior points to small label
chips parked in negative space; a title settles below. Everything ENTERS and EXITS
eased. Rides look.CARD_STAGE_FLOOR so a cut/dissolve never reads as dead black.

Grounded in the labeled-cross-section grammar of real explainers (Branch Education
electron-microscope cutaway: a sectioned column with the electron beam down the
axis + a "Focal Point" leader + a "Magnetic Lenses" title — research
.../science_explainer_family/frame_audit/branch_microscope/). This is an ORIGINAL
Vidlore 2D composition — a clean stylised vessel drawn with PIL on the `look`
system — NOT the reference's 3D models, copper-coil art, palette, fonts, or
branding.

Pure-local (PIL + numpy frames -> ffmpeg encode). No paid API. Deterministic.

    render("xs.mp4", title="How an electron microscope focuses",
           parts=[{"label": "Magnetic lens", "pos": [0.5, 0.30]},
                  {"label": "Focal point", "pos": [0.5, 0.62]}],
           flow="Electron beam", palette_name="cold_steel")

Returns a dict: {ok, path, frames, dur_s, render_s, w, h, err}.
"""
from __future__ import annotations

import subprocess
import tempfile
import time
import unicodedata
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .. import look

SPEC = {
    "id": "labeled_cross_section", "family": "science",
    "roles": ["explain", "mechanism", "how", "internal", "reveal", "anatomy"],
    "niches_ok": ["science", "tech", "engineering", "history"],
    "intensity_range": [2, 4], "duration_range": [4.5, 7.5],
    "easing": "easeInOutCubic", "audio_cue": "soft_reveal_sweep",
    "repeat_cooldown_s": 60, "per_video_cap": 2, "cost": "low",
    "layout_variants": ["capsule_column"],
    "required_inputs": ["parts"],
    "review_override": ["title", "parts", "flow", "palette"],
    "fallback": "annotated_detail_callout for a single labelled detail on a still",
}


# ───────────────────────── helpers ─────────────────────────
def _clean(s) -> str:
    try:
        s = "" if s is None else str(s)
    except Exception:                                          # noqa: BLE001
        return ""
    s = unicodedata.normalize("NFC", s)
    out = []
    for ch in s:
        if unicodedata.category(ch)[0] == "C" and ch not in ("\n", "\t", " "):
            continue
        out.append(" " if ch in ("\n", "\t") else ch)
    return " ".join("".join(out).split()).strip()


def _norm_parts(parts) -> list:
    """Coerce `parts` into [{'label': str, 'pos': (x, y)}] with x,y in 0..1
    (relative to the vessel box). Strings auto-distribute down the axis. Never
    raises; clamps to <=4 parts."""
    out = []
    if isinstance(parts, (str, dict)):
        parts = [parts]
    try:
        seq = list(parts or [])
    except Exception:                                          # noqa: BLE001
        seq = []
    n = max(1, min(4, len(seq)))
    for i, p in enumerate(seq[:4]):
        label, pos = "", None
        if isinstance(p, dict):
            label = _clean(p.get("label") or p.get("name") or p.get("text"))
            pos = p.get("pos") or p.get("point")
        else:
            label = _clean(p)
        # auto-place down the axis, alternating side bias for readability
        ay = 0.20 + (0.60 * (i / max(1, n - 1))) if n > 1 else 0.42
        ax = 0.5 + (0.10 if i % 2 else -0.10)
        if isinstance(pos, (list, tuple)) and len(pos) >= 2:
            try:
                ax = min(0.96, max(0.04, float(pos[0])))
                ay = min(0.96, max(0.04, float(pos[1])))
            except Exception:                                  # noqa: BLE001
                pass
        if label:
            out.append({"label": label[:42], "pos": (ax, ay)})
    return out


def _fit_font(role: str, text: str, max_w: int, start_px: int, min_px: int):
    tmp = ImageDraw.Draw(Image.new("L", (8, 8)))
    px = max(min_px, int(start_px))
    while px > min_px:
        f = look.font(role, px)
        try:
            b = tmp.textbbox((0, 0), text, font=f)
            if (b[2] - b[0]) <= max_w:
                return f
        except Exception:                                      # noqa: BLE001
            return f
        px -= max(2, int(px * 0.06))
    return look.font(role, min_px)


def _chip(text: str, pal: dict, h: int) -> Image.Image:
    """A small label chip: text on a soft rounded scrim pill (RGBA)."""
    f = _fit_font("label", text, int(h * 0.34), int(h * 0.030),
                  max(int(h * 0.020), 11))
    tmp = ImageDraw.Draw(Image.new("L", (8, 8)))
    b = tmp.textbbox((0, 0), text, font=f)
    tw, th = b[2] - b[0], b[3] - b[1]
    padx, pady = int(h * 0.016), int(h * 0.012)
    cw, ch = tw + padx * 2, th + pady * 2
    chip = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
    d = ImageDraw.Draw(chip)
    fl = int(look.CARD_STAGE_FLOOR)
    d.rounded_rectangle([0, 0, cw - 1, ch - 1], radius=int(ch * 0.42),
                        fill=(fl + 2, fl + 3, fl + 6, 210),
                        outline=(*pal["accent"], 150), width=2)
    d.text((padx - b[0], pady - b[1]), text, font=f, fill=pal["text"])
    return chip


# ───────────────────────── render ─────────────────────────
def render(out_path, *, parts, title: str = "", flow: str = "",
           bg_image=None, dur: float = 6.0, fps: int = 30, w: int = 1920,
           h: int = 1080, palette_name: str = "cold_steel", seed: int = 0,
           crf: int = 18) -> dict:
    t0 = time.time()
    out_path = str(out_path)
    try:
        pal = look.palette(palette_name)
        title = _clean(title)
        flow = _clean(flow)
        plist = _norm_parts(parts)
        if not plist:
            return {"ok": False, "path": out_path, "frames": 0, "dur_s": 0.0,
                    "render_s": round(time.time() - t0, 2), "w": w, "h": h,
                    "err": "no parts"}

        dur = max(2.0, float(dur))
        fps = max(1, int(fps))
        n = max(2, int(round(dur * fps)))
        fl = int(look.CARD_STAGE_FLOOR)
        accent = tuple(int(c) for c in pal["accent"])
        accent_hi = tuple(int(c) for c in pal["accent_hi"])

        # vessel geometry (a tall capsule column, left-of-centre to leave room
        # for the label chips on the right)
        vcx, vcy = w * 0.40, h * 0.46
        vw, vh = int(w * 0.20), int(h * 0.60)
        vx0, vy0 = int(vcx - vw / 2), int(vcy - vh / 2)
        vx1, vy1 = int(vcx + vw / 2), int(vcy + vh / 2)
        vrad = int(vw * 0.42)

        # pre-bake the static vessel tile (body + cut-edge + interior chambers)
        # so each frame only composites + animates overlays.
        vessel = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        vd = ImageDraw.Draw(vessel)
        # soft glow halo behind the vessel
        halo = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        ImageDraw.Draw(halo).rounded_rectangle(
            [vx0 - 18, vy0 - 18, vx1 + 18, vy1 + 18], radius=vrad + 18,
            fill=(*pal["glow"], 46))
        vessel.alpha_composite(halo.filter(ImageFilter.GaussianBlur(int(h * 0.03))))
        # body
        vd.rounded_rectangle([vx0, vy0, vx1, vy1], radius=vrad,
                             fill=(fl + 13, fl + 15, fl + 19, 255))
        # interior compartments (3 stacked rounded rects, darker)
        mgx = int(vw * 0.16)
        comp_n = 3
        gap = int(vh * 0.04)
        ch_h = int((vh - 2 * mgx - (comp_n - 1) * gap) / comp_n)
        for c in range(comp_n):
            cy0 = vy0 + mgx + c * (ch_h + gap)
            vd.rounded_rectangle([vx0 + mgx, cy0, vx1 - mgx, cy0 + ch_h],
                                 radius=int(ch_h * 0.30),
                                 fill=(fl + 5, fl + 6, fl + 9, 255),
                                 outline=(*accent, 70), width=2)
        # bright CUT-EDGE stroke (the signature of a cross-section)
        edge = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        ImageDraw.Draw(edge).rounded_rectangle(
            [vx0, vy0, vx1, vy1], radius=vrad, outline=(*accent_hi, 255),
            width=max(3, int(h * 0.0035)))
        vessel.alpha_composite(edge.filter(ImageFilter.GaussianBlur(1)))

        # label lock-ups (chips) + interior anchor points (px)
        chips = []
        for i, p in enumerate(plist):
            ax, ay = p["pos"]
            px = int(vx0 + ax * vw)
            py = int(vy0 + ay * vh)
            chip = _chip(p["label"], pal, h)
            # park chip on the right, vertically spread to match its anchor
            chx = int(vx1 + w * 0.05 + chip.width / 2)
            chy = int(vy0 + (0.16 + 0.66 * (i / max(1, len(plist) - 1)
                                            if len(plist) > 1 else 0.5)) * vh)
            chips.append({"px": px, "py": py, "chip": chip, "chx": chx, "chy": chy})

        # title lock-up
        title_img = None
        if title:
            tf = _fit_font("title", title, int(w * 0.84), int(h * 0.050),
                           max(int(h * 0.032), 14))
            title_img = look.text_with_glow(title, tf, fill=pal["text"],
                                            glow=pal["glow"],
                                            glow_radius=max(6, int(h * 0.012)),
                                            glow_alpha=0.50, pad=int(h * 0.04))
        flow_img = None
        if flow:
            ff = _fit_font("label", flow.upper(), int(w * 0.26),
                           int(h * 0.026), max(int(h * 0.018), 10))
            flow_img = look.text_with_glow(flow.upper(), ff, fill=accent_hi,
                                           glow=pal["glow"], glow_radius=5,
                                           glow_alpha=0.0, pad=int(h * 0.03))

        # optional graded footage/still bed behind the stage
        bed = None
        if bg_image:
            try:
                if Path(str(bg_image)).exists():
                    src = Image.open(str(bg_image)).convert("RGB")
                    bed = look.grade_media(
                        look.face_safe_crop(src, w, h) if hasattr(
                            look, "face_safe_crop") else src.resize((w, h)),
                        pal, strength=0.7)
            except Exception:                                  # noqa: BLE001
                bed = None

        td = Path(tempfile.mkdtemp())
        for i in range(n):
            t = i / fps
            pr = i / max(1, n - 1)
            frame = look.graded_background(w, h, pal, seed=seed, drift=pr,
                                           floor=look.CARD_STAGE_FLOOR)
            if bed is not None:
                frame = Image.blend(frame, bed.convert("RGB"), 0.22)
            fa = look.fade_alpha(t, dur, fps, fin=0.42, fout=0.45)

            # vessel: eased settle (scale up a touch + fade)
            ent = look.ease_out_cubic(min(1.0, t / 0.7))
            vsc = 0.965 + 0.035 * ent
            frame = look.paste_center(frame, vessel, cx=vcx, cy=vcy,
                                      scale=vsc, opacity=fa)

            frame = frame.convert("RGB")
            dr = ImageDraw.Draw(frame)

            # ACTIVE-PATH line drawn down the axis with a travelling head
            if flow_img is not None or True:
                pp = look.ease_out_cubic(max(0.0, min(1.0, (t - 0.55) / 0.95)))
                if pp > 0.01:
                    ytop = int(vy0 + vh * 0.06)
                    ylen = int(vh * 0.88 * pp)
                    lw = max(2, int(h * 0.006))
                    ax_x = int(vcx)
                    dr.line([(ax_x, ytop), (ax_x, ytop + ylen)],
                            fill=accent_hi, width=lw)
                    # travelling glow knot at the head
                    kd = max(4, int(h * 0.010))
                    ky = ytop + ylen
                    dr.ellipse([ax_x - kd, ky - kd, ax_x + kd, ky + kd],
                               fill=accent_hi)

            # leader lines + chips (staggered)
            for li, c in enumerate(chips):
                lead = 0.85 + li * 0.20
                la = look.ease_out_cubic(max(0.0, (t - lead) / 0.6))
                la = min(1.0, la) * fa
                if la <= 0.02:
                    continue
                # leader: from interior anchor to the chip's left edge
                tx = int(c["chx"] - c["chip"].width / 2 - int(w * 0.008))
                ty = c["chy"]
                hx = int(c["px"] + (tx - c["px"]) * la)
                hy = int(c["py"] + (ty - c["py"]) * la)
                dr.line([(c["px"], c["py"]), (hx, hy)], fill=accent,
                        width=max(1, int(h * 0.0022)))
                nd = max(3, int(h * 0.005))
                dr.ellipse([c["px"] - nd, c["py"] - nd, c["px"] + nd,
                            c["py"] + nd], fill=accent_hi)
                if la > 0.6:
                    frame = look.paste_center(frame, c["chip"], cx=c["chx"],
                                              cy=c["chy"], opacity=(la - 0.6) / 0.4)
                    dr = ImageDraw.Draw(frame)

            # flow label (small, top-left of vessel)
            if flow_img is not None:
                ka = look.ease_out_cubic(max(0.0, (t - 0.5) / 0.6)) * fa
                if ka > 0.01:
                    frame = look.paste_center(frame, flow_img,
                                              cx=vcx, cy=vy0 - h * 0.045,
                                              opacity=ka)

            # title (lower third, settles after the reveal)
            if title_img is not None:
                ta = look.ease_out_cubic(max(0.0, (t - 0.7) / 0.7)) * fa
                if ta > 0.01:
                    ry = (1.0 - min(1.0, ta)) * h * 0.02
                    frame = look.paste_center(frame, title_img, cx=w * 0.5,
                                              cy=h * 0.88 + ry, opacity=ta)

            frame = look.vignette(frame, strength=0.60)
            frame = look.film_grain(frame, seed=seed, amount=5.0, t=t)
            frame.save(td / f"f{i:05d}.png")

        return _encode(td, out_path, fps, crf, n, dur, w, h, t0)
    except Exception as e:                                     # noqa: BLE001
        return {"ok": False, "path": out_path, "frames": 0, "dur_s": 0.0,
                "render_s": round(time.time() - t0, 2), "w": w, "h": h,
                "err": f"render: {type(e).__name__}: {e}"}


def _encode(td, out_path, fps, crf, n, dur, w, h, t0) -> dict:
    try:
        from vidlore.ffmpeg_tool import ffmpeg_exe
        ff = ffmpeg_exe()
    except Exception:                                          # noqa: BLE001
        ff = "ffmpeg"
    cmd = [ff, "-y", "-hide_banner", "-loglevel", "error", "-framerate",
           str(fps), "-i", f"{td}/f%05d.png", "-c:v", "libx264", "-crf",
           str(crf), "-pix_fmt", "yuv420p", "-colorspace", "bt709",
           "-color_primaries", "bt709", "-color_trc", "bt709", "-movflags",
           "+faststart", str(out_path)]
    err = ""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True)
        ok = (r.returncode == 0 and Path(out_path).exists())
        if not ok:
            err = (r.stderr or "")[-200:]
    except Exception as e:                                     # noqa: BLE001
        ok, err = False, f"encode: {e}"
    look.cleanup_frames(td)
    return {"ok": ok, "path": str(out_path), "frames": n, "dur_s": round(dur, 2),
            "render_s": round(time.time() - t0, 2), "w": w, "h": h, "err": err}
