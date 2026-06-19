"""Primitive: classified_stamp_reveal.

A declassification beat — an aged file page sits on the charcoal stage with a
header and typewritten body lines, a TOP SECRET stamp lands with an ink-impact,
and then the redaction bar over the key line WIPES away to reveal the suppressed
fact, finished by a DECLASSIFIED stamp. The grammar of "the file said this — here
is what they hid" in spy / true-crime / political history.

Distinct from `redacted_document` (a static censored page) and `classified dossier`
cards: the motion IS the point here — stamp impact, then an animated un-redaction
reveal of a single key line.

Forensic principle (NOT an asset copy): premium docs stamp a file, then lift the
redaction to reveal the buried line. Rebuilt from that PRINCIPLE with Vidlore's
own look system — original typography, layout and colour. Pure-local,
deterministic, no paid API.

    render("x.mp4", header="MEMORANDUM · EYES ONLY · 1974",
           body=["SUBJECT: Operation codename", "STATUS: Active"],
           reveal="The operation was authorized at the highest level.")
"""
from __future__ import annotations

import subprocess
import tempfile
import textwrap
import time
from pathlib import Path

from PIL import Image, ImageDraw

from .. import look

SPEC = {
    "id": "classified_stamp_reveal", "family": "investigation",
    "roles": ["classified", "declassified", "redacted_reveal", "top_secret",
              "dossier_reveal", "leak", "suppressed"],
    "niches_ok": ["spy", "crime", "history", "geopolitics", "war"],
    "intensity_range": [3, 5], "duration_range": [5.0, 7.5],
    "easing": "easeOutCubic", "audio_cue": "stamp_hard",
    "repeat_cooldown_s": 55, "per_video_cap": 2, "cost": "low",
    "layout_variants": ["file_page"],
    "review_override": ["header", "body", "reveal", "stamp_text", "palette"],
    "fallback": "redacted_document for a static censored page (no reveal)",
    "required_inputs": ["reveal"],
}

_PAPER = (226, 220, 205)
_PAPER_SH = (206, 198, 181)
_INK = (44, 41, 36)
_INK_SOFT = (96, 90, 80)
_BAR = (24, 22, 20)
_RED = (176, 44, 40)


def _stamp(text, col, size=66):
    f = look.font("title", size)
    tw = int(f.getlength(text)); bb = f.getbbox("HgY"); th = bb[3] - bb[1]
    pad_x, pad_y = 30, 16
    cw, ch = tw + pad_x * 2, th + pad_y * 2
    chip = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
    d = ImageDraw.Draw(chip)
    d.rectangle([3, 3, cw - 4, ch - 4], outline=(*col, 255), width=5)
    d.rectangle([12, 12, cw - 13, ch - 13], outline=(*col, 120), width=1)
    d.text((pad_x, pad_y - bb[1]), text, font=f, fill=(*col, 255))
    return chip


def render(out_path, *, header: str = "", body=None, reveal: str = "",
           reveal_label: str = "", stamp_text: str = "TOP SECRET",
           dur: float = 6.0, fps: int = 30, w: int = 1920, h: int = 1080,
           palette_name: str = "cold_steel", layout: str = "", seed: int = 0,
           crf: int = 18) -> dict:
    pal = look.palette(palette_name)
    t0 = time.time()
    n = max(2, int(round(dur * fps)))
    header = str(header or "CLASSIFIED FILE").strip()
    reveal = str(reveal or "").strip()
    stamp_text = str(stamp_text or "TOP SECRET").strip().upper()
    body = [str(b).strip() for b in (body or []) if str(b).strip()][:4]

    bed = look.graded_background(w, h, pal, seed=seed) if hasattr(look, "graded_background") \
        else Image.new("RGB", (w, h), tuple(pal["bg_b"]))

    # ---- build the file page once (paper + header + body + the reveal text) ----
    pw, ph = int(w * 0.60), int(h * 0.74)
    page = Image.new("RGB", (pw, ph), _PAPER)
    pd = ImageDraw.Draw(page)
    pd.rectangle([0, 0, pw - 1, ph - 1], outline=_PAPER_SH, width=2)
    margin = int(pw * 0.08)
    # header
    hf = look.font("label", int(ph * 0.040))
    pd.text((margin, int(ph * 0.06)), header.upper(), font=hf, fill=_INK_SOFT)
    pd.line([(margin, int(ph * 0.13)), (pw - margin, int(ph * 0.13))], fill=_INK_SOFT, width=2)
    # body lines (typewritten, muted)
    bf = look.font("numeral", int(ph * 0.040))
    yb = int(ph * 0.20)
    for ln in body:
        for sub in textwrap.wrap(ln, 46)[:2]:
            pd.text((margin, yb), sub, font=bf, fill=_INK)
            yb += int(ph * 0.064)
        yb += int(ph * 0.012)
    # a couple of decorative (permanent) redactions on filler lines
    for fy in (yb + int(ph * 0.02), yb + int(ph * 0.085)):
        pd.text((margin, fy), "REF:", font=bf, fill=_INK_SOFT)
        bx0 = margin + int(bf.getlength("REF:  "))
        pd.rectangle([bx0, fy - 2, bx0 + int(pw * (0.34 if fy == yb + int(ph * 0.02) else 0.5)),
                      fy + int(bf.size)], fill=_BAR)
    # the KEY line — reveal text drawn into the page (covered by an animated bar)
    kf = look.font("numeral", int(ph * 0.050))
    key_y = int(ph * 0.70)
    rlabel = (reveal_label or "FINDING").upper()
    pd.text((margin, key_y - int(ph * 0.052)), rlabel, font=look.font("label", int(ph * 0.030)),
            fill=_RED)
    rl_lines = textwrap.wrap(reveal, 40)[:2]
    ky = key_y
    key_spans = []
    for sub in rl_lines:
        pd.text((margin, ky), sub, font=kf, fill=_INK)
        key_spans.append((ky, int(kf.getlength(sub))))
        ky += int(ph * 0.066)

    px0, py0 = (w - pw) // 2, (h - ph) // 2
    stamp_top = _stamp(stamp_text, _RED, size=int(h * 0.066))
    stamp_top = stamp_top.rotate(11, expand=True, resample=Image.BICUBIC)
    stamp_dc = _stamp("DECLASSIFIED", (58, 132, 96), size=int(h * 0.05))
    stamp_dc = stamp_dc.rotate(-8, expand=True, resample=Image.BICUBIC)

    td = Path(tempfile.mkdtemp())
    for i in range(n):
        t = i / fps
        frame = bed.copy()

        # page slides up + fades in (0-0.6s)
        pa = look.ease_out_cubic(min(1.0, t / 0.6))
        if pa <= 0.01:
            frame = look.film_grain(look.vignette(frame, strength=0.6), seed=seed, amount=5.0, t=t)
            frame.save(td / f"f{i:05d}.png"); continue
        dy = int((1 - pa) * h * 0.04)
        base = frame.convert("RGBA")
        pg = page.convert("RGBA")
        if pa < 1.0:
            a = pg.split()[3].point(lambda v: int(v * pa))
            pg.putalpha(a)
        base.alpha_composite(pg, (px0, py0 + dy))
        frame = base.convert("RGB")
        d = ImageDraw.Draw(frame, "RGBA")

        # key-line redaction bar — covers, then wipes left->right to reveal (1.7-2.7s)
        rev = look.ease_out_cubic(min(1.0, max(0.0, (t - 1.7) / 1.0)))
        for (ky_rel, span_w) in key_spans:
            ay = py0 + dy + ky_rel - 4
            bx0 = px0 + margin - 6
            bx1 = px0 + margin + span_w + 10
            cover_x = int(bx0 + (bx1 - bx0) * rev)
            if cover_x < bx1:
                d.rectangle([cover_x, ay, bx1, ay + int(kf.size) + 6], fill=(*_BAR, 255))
                if 0.02 < rev < 0.98:                              # bright declassify scan edge
                    d.line([(cover_x, ay), (cover_x, ay + int(kf.size) + 6)],
                           fill=(*tuple(pal["accent_hi"]), 220), width=3)

        # TOP SECRET stamp — ink impact at ~0.85s
        sa = min(1.0, max(0.0, (t - 0.85) / 0.3))
        if sa > 0.01:
            sc = 1.0 + (1.0 - look.ease_out_cubic(sa)) * 0.8
            sw = max(1, int(stamp_top.width * sc)); sh = max(1, int(stamp_top.height * sc))
            simg = stamp_top.resize((sw, sh), Image.BICUBIC)
            op = min(0.9, look.ease_out_cubic(sa) * 0.9)
            if op < 0.9:
                aa = simg.split()[3].point(lambda v: int(v * op / 0.9))
                simg.putalpha(aa)
            scx = px0 + int(pw * 0.62); scy = py0 + dy + int(ph * 0.30)
            b2 = frame.convert("RGBA"); b2.alpha_composite(simg, (scx - sw // 2, scy - sh // 2))
            frame = b2.convert("RGB"); d = ImageDraw.Draw(frame, "RGBA")

        # DECLASSIFIED stamp — impact once the reveal completes
        if rev > 0.85:
            da = min(1.0, (rev - 0.85) / 0.15)
            sc = 1.0 + (1.0 - da) * 0.6
            sw = max(1, int(stamp_dc.width * sc)); sh = max(1, int(stamp_dc.height * sc))
            simg = stamp_dc.resize((sw, sh), Image.BICUBIC)
            op = min(0.92, da * 0.92)
            if op < 0.92:
                aa = simg.split()[3].point(lambda v: int(v * op / 0.92))
                simg.putalpha(aa)
            scx = px0 + int(pw * 0.40); scy = py0 + dy + int(ph * 0.90)
            b2 = frame.convert("RGBA"); b2.alpha_composite(simg, (scx - sw // 2, scy - sh // 2))
            frame = b2.convert("RGB")

        frame = look.vignette(frame, strength=0.62)
        frame = look.film_grain(frame, seed=seed, amount=5.5, t=t)
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
