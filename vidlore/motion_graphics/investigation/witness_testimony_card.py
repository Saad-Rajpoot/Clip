"""Primitive: witness_testimony_card.

A humanize-the-evidence beat — on a charcoal archival stage, a file-bordered
IDENTITY photo (or an initials placeholder in a police-file frame) sits left
while the right column reveals a NAME (serif), a ROLE/affiliation (condensed,
muted), and a QUOTE that unfolds LINE-BY-LINE in serif, synced to the beat, with
an opening quotation glyph. A faint case-number + "TESTIMONY" file tab and grain
sell the archival feel. The grammar of "here is the person, in their own words" —
witnesses, defectors, suspects, sources in spy / true-crime / history.

Distinct from `pull_quote_portrait` (a single held quotation, business/biography
tone) and `framed_evidence_spotlight` (an exhibit, not a person): this is a
FILE-style human testimony with a staged line reveal.

Forensic principle (NOT an asset copy): premium investigation docs frame a
witness with a file-border photo + line-revealed quote on a desaturated stage.
Rebuilt from that PRINCIPLE with Vidlore's own look system. Pure-local,
deterministic, no paid API.

    render("x.mp4", name="Oleg Gordievsky", role="KGB Colonel · Defector",
           quote="They never suspected the man at the very top.")
"""
from __future__ import annotations

import math
import subprocess
import tempfile
import textwrap
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .. import look

SPEC = {
    "id": "witness_testimony_card", "family": "investigation",
    "roles": ["testimony", "witness", "quote", "statement", "source", "interview"],
    "niches_ok": ["crime", "spy", "history", "geopolitics", "biography"],
    "intensity_range": [2, 4], "duration_range": [4.5, 7.5],
    "easing": "easeOutCubic", "audio_cue": "paper_soft",
    "repeat_cooldown_s": 45, "per_video_cap": 3, "cost": "low",
    "layout_variants": ["photo_left", "photo_right"],
    "review_override": ["name", "role", "quote", "photo", "case_no", "palette"],
    "fallback": "pull_quote_portrait if no name/role context is available",
    "required_inputs": ["quote"],
}


def _file_photo(photo, fw, fh, pal, name):
    """A file-border identity tile: graded portrait if supplied, else an initials
    silhouette plate. Returns an RGB image of size (fw, fh)."""
    inner = None
    if photo and Path(str(photo)).exists():
        try:
            im = Image.open(str(photo)).convert("RGB")
            if hasattr(look, "face_safe_crop"):
                inner = look.face_safe_crop(im, fw, fh)
            else:
                inner = im.resize((fw, fh), Image.LANCZOS)
            inner = look.grade_media(inner, pal, strength=0.5)
        except Exception:                                          # noqa: BLE001
            inner = None
    if inner is None:
        inner = Image.new("RGB", (fw, fh), tuple(pal.get("bg_a", (30, 32, 38))))
        d = ImageDraw.Draw(inner)
        for y in range(fh):                                        # subtle gradient
            t = y / fh
            d.line([(0, y), (fw, y)], fill=tuple(int(c * (0.7 + 0.3 * t))
                                                 for c in pal.get("bg_a", (30, 32, 38))))
        ini = "".join(w[0] for w in (name or "?").split()[:2]).upper() or "?"
        f = look.font("numeral", int(fh * 0.34))
        ti = look.text_with_glow(ini, f, fill=pal["muted"], glow=pal["bg_b"],
                                 glow_radius=4, glow_alpha=0.0, pad=6)
        inner.alpha_composite(ti.convert("RGBA"), ((fw - ti.width) // 2,
                                                   (fh - ti.height) // 2)) \
            if inner.mode == "RGBA" else inner.paste(
            ti.convert("RGB"), ((fw - ti.width) // 2, (fh - ti.height) // 2),
            ti)
    return inner.convert("RGB")


def render(out_path, *, name: str = "", role: str = "", quote: str = "",
           photo=None, case_no: str = "", dur: float = 6.0, fps: int = 30,
           w: int = 1920, h: int = 1080, palette_name: str = "cold_steel",
           layout: str = "", seed: int = 0, crf: int = 18) -> dict:
    pal = look.palette(palette_name)
    t0 = time.time()
    n = max(2, int(round(dur * fps)))
    name = str(name or "").strip()
    role = str(role or "").strip()
    quote = str(quote or "").strip().strip('"“”')
    right = layout == "photo_right"
    accent_hi = tuple(pal["accent_hi"]); ink = tuple(pal["text"])
    muted = tuple(pal["muted"])

    # archival charcoal stage
    if hasattr(look, "graded_background"):
        bed = look.graded_background(w, h, pal, seed=seed)
    else:
        bed = Image.new("RGB", (w, h), tuple(pal["bg_b"]))

    # file photo tile
    fw, fh = int(w * 0.26), int(h * 0.52)
    border = int(min(fw, fh) * 0.035)
    tile = _file_photo(photo, fw - 2 * border, fh - 2 * border, pal, name)
    photo_cx = int(w * (0.74 if right else 0.26))
    photo_cy = int(h * 0.5)
    # text column anchor
    col_x = int(w * (0.06 if right else 0.45))
    col_w = int(w * 0.46)

    name_font = look.font("title", int(h * 0.050))
    role_font = look.font("label", int(h * 0.028))
    quote_font = look.font("numeral", int(h * 0.040))
    mark_font = look.font("numeral", int(h * 0.13))
    case_font = look.font("label", int(h * 0.020))

    # wrap quote to the column
    avg_char = quote_font.getlength("n") or (h * 0.02)
    wrap_n = max(14, int(col_w / max(1.0, avg_char)))
    qlines = textwrap.wrap('"' + quote + '"', wrap_n) if quote else []
    line_h = int(h * 0.058)

    td = Path(tempfile.mkdtemp())
    for i in range(n):
        t = i / fps
        frame = bed.copy()
        d = ImageDraw.Draw(frame, "RGBA")

        # file tab + case number (top), fades in first
        ta = look.ease_out_cubic(min(1.0, t / 0.4))
        if ta > 0.01:
            tab = "TESTIMONY" + (f" · {case_no}" if case_no else "")
            ci = look.text_with_glow(tab, case_font, fill=muted, glow=pal["bg_b"],
                                     glow_radius=3, glow_alpha=0.0, pad=6)
            d.rectangle([col_x, int(h * 0.16), col_x + int(col_w * 0.5), int(h * 0.165)],
                        fill=(*accent_hi, int(180 * ta)))
            frame = look.paste_center(frame, ci,
                                      cx=col_x + ci.width // 2 + 4,
                                      cy=int(h * 0.13), opacity=ta)
            d = ImageDraw.Draw(frame, "RGBA")

        # photo tile slides + fades in (0.15-0.8s) with a file border
        pa = look.ease_out_cubic(min(1.0, max(0.0, (t - 0.15) / 0.6)))
        if pa > 0.01:
            dx = int((1 - pa) * w * 0.04) * (1 if right else -1)
            x0 = photo_cx - fw // 2 + dx; y0 = photo_cy - fh // 2
            fr = Image.new("RGBA", (fw, fh), (0, 0, 0, 0))
            ImageDraw.Draw(fr).rectangle([0, 0, fw - 1, fh - 1],
                                         fill=(236, 232, 224, int(235 * pa)))
            fr.paste(tile, (border, border))
            ImageDraw.Draw(fr).rectangle([border - 1, border - 1, fw - border, fh - border],
                                         outline=(*pal["bg_b"], int(200 * pa)), width=1)
            base = frame.convert("RGBA"); base.alpha_composite(fr, (x0, y0))
            frame = base.convert("RGB"); d = ImageDraw.Draw(frame, "RGBA")

        # name + role (0.5-1.0s)
        na = look.ease_out_cubic(min(1.0, max(0.0, (t - 0.5) / 0.5)))
        y_cursor = int(h * 0.24)
        if name and na > 0.01:
            ni = look.text_with_glow(name, name_font, fill=ink, glow=pal["bg_b"],
                                     glow_radius=5, glow_alpha=0.0, pad=8)
            frame = look.paste_center(frame, ni, cx=col_x + ni.width // 2,
                                      cy=y_cursor + ni.height // 2, opacity=na)
            y_cursor += int(ni.height * 0.9)
            d = ImageDraw.Draw(frame, "RGBA")
        if role and na > 0.01:
            ri = look.text_with_glow(role.upper(), role_font, fill=accent_hi,
                                     glow=pal["bg_b"], glow_radius=4, glow_alpha=0.0, pad=6)
            frame = look.paste_center(frame, ri, cx=col_x + ri.width // 2,
                                      cy=y_cursor + ri.height, opacity=na)
            y_cursor += int(ri.height * 1.6)
            d = ImageDraw.Draw(frame, "RGBA")

        # big opening quotation mark
        if na > 0.2:
            qm = look.text_with_glow("“", mark_font, fill=accent_hi,
                                     glow=pal["bg_b"], glow_radius=6, glow_alpha=0.0, pad=4)
            frame = look.paste_center(frame, qm, cx=col_x + int(h * 0.03),
                                      cy=y_cursor + int(h * 0.02),
                                      opacity=min(0.5, na * 0.5))
            d = ImageDraw.Draw(frame, "RGBA")

        # quote reveals LINE BY LINE (start 0.9s, ~0.45s/line)
        qy = y_cursor + int(h * 0.04)
        for li, line in enumerate(qlines):
            la = look.ease_out_cubic(min(1.0, max(0.0, (t - (0.9 + li * 0.45)) / 0.4)))
            if la <= 0.01:
                break
            qi = look.text_with_glow(line, quote_font, fill=ink, glow=pal["bg_b"],
                                     glow_radius=4, glow_alpha=0.0, pad=6)
            frame = look.paste_center(frame, qi, cx=col_x + qi.width // 2,
                                      cy=qy + line_h // 2, opacity=la)
            qy += line_h
            d = ImageDraw.Draw(frame, "RGBA")

        frame = look.vignette(frame, strength=0.62)
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
