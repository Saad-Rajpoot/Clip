"""Primitive: suspect_profile_card.

A police-dossier "person of interest" beat — on a charcoal stage, a booking-style
identity tile (front portrait or initials plate) sits behind faint height-chart
hairlines and registration corner marks, while a CLASSIFICATION stamp (PERSON OF
INTEREST / WANTED / SUSPECT / CLEARED) lands with an ink-impact and dot-leader
data rows (alias, status, last seen, known for) reveal in sequence. The grammar
of "here is who we are looking at, and what the file says" — spy / true-crime /
history dossiers.

Distinct from `witness_testimony_card` (a person's WORDS) and `pull_quote_portrait`
(a held quotation): this CLASSIFIES and tabulates a person from a case file.

Forensic principle (NOT an asset copy): premium investigation docs profile a
subject with a booking-frame photo, a stamped classification, and tabular file
fields on a desaturated stage. Rebuilt from that PRINCIPLE with Vidlore's own
look system. Pure-local, deterministic, no paid API.

    render("x.mp4", name="Kim Philby", alias="Stanley · Sonny",
           status="DOUBLE AGENT", fields=[["Recruited","1934"],["Role","SIS"]])
"""
from __future__ import annotations

import subprocess
import tempfile
import time
from pathlib import Path

from PIL import Image, ImageDraw

from .. import look

SPEC = {
    "id": "suspect_profile_card", "family": "investigation",
    "roles": ["suspect", "profile", "dossier", "person_of_interest", "wanted",
              "identity", "subject"],
    "niches_ok": ["crime", "spy", "history", "geopolitics", "biography"],
    "intensity_range": [2, 4], "duration_range": [4.5, 7.0],
    "easing": "easeOutCubic", "audio_cue": "stamp_soft",
    "repeat_cooldown_s": 45, "per_video_cap": 3, "cost": "low",
    "layout_variants": ["photo_left", "photo_right"],
    "review_override": ["name", "alias", "status", "fields", "photo", "case_no",
                        "palette"],
    "fallback": "witness_testimony_card / mini_bio_card if no classification context",
    "required_inputs": ["name"],
}

_STATUS_ACCENT = {                                  # which palette tone fits a status
    "wanted": "ember", "fugitive": "ember", "armed": "ember", "double agent": "ember",
    "suspect": "ember", "person of interest": "accent", "cleared": "cool",
    "deceased": "muted", "defector": "accent", "informant": "accent",
}


def _ident_tile(photo, fw, fh, pal, name):
    if photo and Path(str(photo)).exists():
        try:
            im = Image.open(str(photo)).convert("RGB")
            inner = look.face_safe_crop(im, fw, fh) if hasattr(look, "face_safe_crop") \
                else im.resize((fw, fh), Image.LANCZOS)
            return look.grade_media(inner, pal, strength=0.55).convert("RGB")
        except Exception:                                          # noqa: BLE001
            pass
    inner = Image.new("RGB", (fw, fh), tuple(pal["bg_a"]))
    d = ImageDraw.Draw(inner)
    for y in range(fh):
        t = y / fh
        d.line([(0, y), (fw, y)],
               fill=tuple(int(c * (0.65 + 0.4 * t)) for c in pal["bg_a"]))
    ini = "".join(w[0] for w in (name or "?").split()[:2]).upper() or "?"
    f = look.font("numeral", int(fh * 0.32))
    ti = look.text_with_glow(ini, f, fill=pal["muted"], glow=pal["bg_b"],
                             glow_radius=4, glow_alpha=0.0, pad=6)
    inner.paste(ti.convert("RGB"), ((fw - ti.width) // 2, (fh - ti.height) // 2), ti)
    return inner


def _stamp_chip(text, pal, tone):
    f = look.font("title", 58)
    col = {"ember": (206, 84, 70), "accent": tuple(pal["accent_hi"]),
           "cool": (120, 168, 200), "muted": tuple(pal["muted"])}.get(tone,
                                                                       tuple(pal["accent_hi"]))
    pad_x, pad_y = 34, 18
    tw = int(f.getlength(text))
    bb = f.getbbox("Hg"); th = bb[3] - bb[1]
    cw, ch = tw + pad_x * 2, th + pad_y * 2
    chip = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
    cd = ImageDraw.Draw(chip)
    cd.rectangle([2, 2, cw - 3, ch - 3], outline=(*col, 255), width=4)
    cd.rectangle([10, 10, cw - 11, ch - 11], outline=(*col, 110), width=1)
    cd.text((pad_x, pad_y - bb[1]), text, font=f, fill=(*col, 255))
    return chip


def render(out_path, *, name: str = "", alias: str = "", status: str = "PERSON OF INTEREST",
           fields=None, photo=None, case_no: str = "", dur: float = 6.0, fps: int = 30,
           w: int = 1920, h: int = 1080, palette_name: str = "cold_steel",
           layout: str = "", seed: int = 0, crf: int = 18) -> dict:
    pal = look.palette(palette_name)
    t0 = time.time()
    n = max(2, int(round(dur * fps)))
    name = str(name or "").strip()
    alias = str(alias or "").strip()
    status = str(status or "").strip().upper()
    right = layout == "photo_right"
    ink = tuple(pal["text"]); muted = tuple(pal["muted"]); accent = tuple(pal["accent_hi"])
    tone = _STATUS_ACCENT.get(status.lower(), "accent")

    rows = []
    if fields:
        for it in fields:
            if isinstance(it, (list, tuple)) and len(it) >= 2:
                rows.append((str(it[0]).upper(), str(it[1])))
            elif isinstance(it, dict):
                rows.append((str(it.get("label", "")).upper(), str(it.get("value", ""))))
    if alias and not any(r[0] == "ALIAS" for r in rows):
        rows.insert(0, ("ALIAS", alias))
    if not any(r[0] == "STATUS" for r in rows):
        rows.insert(0, ("STATUS", status.title()))
    rows = rows[:5]

    bed = look.graded_background(w, h, pal, seed=seed) if hasattr(look, "graded_background") \
        else Image.new("RGB", (w, h), tuple(pal["bg_b"]))

    fw, fh = int(w * 0.27), int(h * 0.56)
    border = int(min(fw, fh) * 0.03)
    tile = _ident_tile(photo, fw - 2 * border, fh - 2 * border, pal, name)
    photo_cx = int(w * (0.73 if right else 0.27)); photo_cy = int(h * 0.52)
    col_x = int(w * (0.07 if right else 0.46)); col_w = int(w * 0.45)

    name_font = look.font("title", int(h * 0.052))
    head_font = look.font("label", int(h * 0.021))
    lab_font = look.font("label", int(h * 0.0225))
    val_font = look.font("numeral", int(h * 0.030))
    stamp = _stamp_chip(status or "PERSON OF INTEREST", pal, tone)
    # pre-rotate the stamp once
    stamp_rot = stamp.rotate(7, expand=True, resample=Image.BICUBIC)

    td = Path(tempfile.mkdtemp())
    for i in range(n):
        t = i / fps
        frame = bed.copy()

        # header tab
        ha = look.ease_out_cubic(min(1.0, t / 0.4))
        if ha > 0.01:
            tab = "DOSSIER" + (f" · {case_no}" if case_no else "")
            ci = look.text_with_glow(tab, head_font, fill=muted, glow=pal["bg_b"],
                                     glow_radius=3, glow_alpha=0.0, pad=6)
            frame = look.paste_center(frame, ci, cx=col_x + ci.width // 2,
                                      cy=int(h * 0.135), opacity=ha)

        # identity tile with booking height-chart hairlines + registration corners
        pa = look.ease_out_cubic(min(1.0, max(0.0, (t - 0.1) / 0.55)))
        if pa > 0.01:
            dx = int((1 - pa) * w * 0.035) * (1 if right else -1)
            x0 = photo_cx - fw // 2 + dx; y0 = photo_cy - fh // 2
            fr = Image.new("RGBA", (fw, fh), (0, 0, 0, 0))
            fd = ImageDraw.Draw(fr)
            fd.rectangle([0, 0, fw - 1, fh - 1], fill=(*pal["bg_a"], int(220 * pa)))
            for hy in range(1, 6):                       # height-chart hairlines
                yy = int(fh * hy / 6.0)
                fd.line([(border, yy), (fw - border, yy)],
                        fill=(*muted, int(40 * pa)), width=1)
            fr.paste(tile, (border, border))
            cl = max(14, int(fw * 0.08))                 # registration corners
            for (cx0, cy0, sx, sy) in ((border, border, 1, 1), (fw - border, border, -1, 1),
                                       (border, fh - border, 1, -1),
                                       (fw - border, fh - border, -1, -1)):
                fd.line([(cx0, cy0), (cx0 + sx * cl, cy0)], fill=(*accent, int(210 * pa)), width=3)
                fd.line([(cx0, cy0), (cx0, cy0 + sy * cl)], fill=(*accent, int(210 * pa)), width=3)
            base = frame.convert("RGBA"); base.alpha_composite(fr, (x0, y0))
            frame = base.convert("RGB")

        # name
        na = look.ease_out_cubic(min(1.0, max(0.0, (t - 0.45) / 0.5)))
        y_cur = int(h * 0.215)
        if name and na > 0.01:
            ni = look.text_with_glow(name, name_font, fill=ink, glow=pal["bg_b"],
                                     glow_radius=5, glow_alpha=0.0, pad=8)
            frame = look.paste_center(frame, ni, cx=col_x + ni.width // 2,
                                      cy=y_cur + ni.height // 2, opacity=na)
            y_cur += int(ni.height * 1.25)

        # data rows with dot-leaders
        for ri, (lab, val) in enumerate(rows):
            ra = look.ease_out_cubic(min(1.0, max(0.0, (t - (0.75 + ri * 0.28)) / 0.4)))
            if ra <= 0.01:
                break
            li = look.text_with_glow(lab, lab_font, fill=muted, glow=pal["bg_b"],
                                     glow_radius=2, glow_alpha=0.0, pad=4)
            vi = look.text_with_glow(val, val_font, fill=ink, glow=pal["bg_b"],
                                     glow_radius=3, glow_alpha=0.0, pad=4)
            rowy = y_cur + ri * int(h * 0.066)
            frame = look.paste_center(frame, li, cx=col_x + li.width // 2,
                                      cy=rowy + li.height // 2, opacity=ra)
            vx = col_x + col_w - vi.width
            frame = look.paste_center(frame, vi, cx=vx + vi.width // 2,
                                      cy=rowy + vi.height // 2, opacity=ra)
            d = ImageDraw.Draw(frame, "RGBA")           # dot leader between
            lx0 = col_x + li.width + 14; lx1 = vx - 14
            yy = rowy + li.height // 2
            x = lx0
            while x < lx1:
                d.ellipse([x, yy - 1, x + 2, yy + 1], fill=(*muted, int(150 * ra)))
                x += 12

        # classification stamp — ink impact (scale-snap + settle)
        sa = min(1.0, max(0.0, (t - 1.05) / 0.32))
        if sa > 0.01:
            sc = 1.0 + (1.0 - look.ease_out_cubic(sa)) * 0.7   # 1.7 -> 1.0
            sw = max(1, int(stamp_rot.width * sc)); sh = max(1, int(stamp_rot.height * sc))
            simg = stamp_rot.resize((sw, sh), Image.BICUBIC)
            op = min(0.92, look.ease_out_cubic(sa) * 0.92)
            if op < 0.92:
                a = simg.split()[3].point(lambda v: int(v * op / 0.92))
                simg.putalpha(a)
            scx = col_x + int(col_w * 0.62); scy = int(h * 0.80)
            base = frame.convert("RGBA")
            base.alpha_composite(simg, (scx - sw // 2, scy - sh // 2))
            frame = base.convert("RGB")

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
