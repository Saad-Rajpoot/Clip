"""Primitive: redacted_document.

A secrecy beat — a classified page where black redaction bars sweep across most
of the typed lines while ONE line is left legible, and a red TOP SECRET stamp
lands askew. What they tried to hide, and the one thing they couldn't.

Distinct from `headline_document_reveal` (a newspaper headline on newsprint) and
`framed_evidence_spotlight` (an artifact under a spotlight): this is the
GOVERNMENT/CORPORATE FILE — concealment made visible, the redaction itself the
story.

Forensic principle (NOT asset copy): the bars and the stamp are the drama; the
premium is the aged file paper + typed mono lines + clean black bars sweeping in
+ a worn red stamp + the single damning line that survives. Pure-local. No paid
API. Deterministic.

    render("x.mp4", reveal="...authorized the operation personally.",
           title="MEMORANDUM — EYES ONLY", stamp="TOP SECRET")
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
    "id": "redacted_document", "family": "documents",
    "roles": ["classified", "redacted", "secret", "evidence", "reveal"],
    "niches_ok": ["crime", "geopolitics", "history", "biography", "business", "tech"],
    "intensity_range": [3, 5], "duration_range": [4.0, 6.5],
    "easing": "easeOutCubic", "audio_cue": "soft_redact_thunk",
    "repeat_cooldown_s": 55, "per_video_cap": 2, "cost": "low",
    "layout_variants": ["classified_file"],
    "review_override": ["reveal", "title", "stamp", "palette"],
    "fallback": "headline_document_reveal if redaction would read as noise",
}


def _wrap(d, text, font, max_w):
    words, lines, cur = text.split(), [], ""
    for wd in words:
        t = (cur + " " + wd).strip()
        if d.textlength(t, font=font) <= max_w or not cur:
            cur = t
        else:
            lines.append(cur); cur = wd
    if cur:
        lines.append(cur)
    return lines


def _paper(pw, ph, seed):
    rng = np.random.default_rng(int(seed or 7) & 0xFFFFFFFF)
    arr = np.empty((ph, pw, 3), np.float32)
    arr[:] = np.array([224, 216, 200], np.float32)
    arr += rng.normal(0, 6.0, (ph, pw, 1))
    yy, xx = np.mgrid[0:ph, 0:pw].astype(np.float32)
    r = np.sqrt(((xx - pw / 2) / (pw * 0.62)) ** 2 + ((yy - ph / 2) / (ph * 0.62)) ** 2)
    arr -= np.clip(r - 0.55, 0, 1)[..., None] * 38
    return Image.fromarray(np.clip(arr, 0, 255).astype("uint8"), "RGB")


def render(out_path, *, reveal: str = "", title: str = "", stamp: str = "TOP SECRET",
           dur: float = 5.5, fps: int = 30, w: int = 1920, h: int = 1080,
           palette_name: str = "ember_red", layout: str = "", seed: int = 0,
           crf: int = 18) -> dict:
    pal = look.palette(palette_name)
    t0 = time.time()
    n = max(2, int(round(dur * fps)))
    reveal = (reveal or "").strip() or "[ redacted ]"
    title = (title or "CLASSIFIED MEMORANDUM").strip()

    pw, ph = int(w * 0.62), int(h * 0.78)
    px0, py0 = (w - pw) // 2, (h - ph) // 2
    paper = _paper(pw, ph, seed)
    pd = ImageDraw.Draw(paper)
    ink, faint = (44, 38, 30), (150, 132, 108)
    hdr_font = look.font("label", int(ph * 0.045))
    line_font = look.font("caption", int(ph * 0.040))
    # classification banner top + bottom
    for by in (int(ph * 0.04), int(ph * 0.92)):
        pd.rectangle([int(pw * 0.06), by, int(pw * 0.94), by + int(ph * 0.045)],
                     fill=(120, 24, 16))
    bn = look.font("label", int(ph * 0.030))
    for by in (int(ph * 0.045), int(ph * 0.925)):
        tw = pd.textlength(stamp.upper(), font=bn)
        pd.text(((pw - tw) / 2, by), stamp.upper(), font=bn, fill=(238, 228, 214))
    # title / header
    pd.text((int(pw * 0.1), int(ph * 0.13)), title.upper(), font=hdr_font, fill=ink)
    pd.line([(int(pw * 0.1), int(ph * 0.20)), (int(pw * 0.9), int(ph * 0.20))],
            fill=faint, width=2)
    # body lines: typed faux text, the reveal line kept real + legible
    reveal_lines = _wrap(pd, reveal, line_font, pw * 0.78)[:2]
    rows = []                                                  # (y, kind, payload)
    yy = ph * 0.27
    reveal_at = 4
    li = 0
    while yy < ph * 0.86 and li < 9:
        if li == reveal_at:
            for rl in reveal_lines:
                rows.append((yy, "reveal", rl)); yy += ph * 0.058
        else:
            seg = []                                           # redacted line: bar segments
            x = pw * 0.1
            rng2 = np.random.default_rng((seed or 7) * 31 + li)
            while x < pw * 0.86:
                wseg = pw * float(rng2.uniform(0.10, 0.26))
                seg.append((x, min(x + wseg, pw * 0.88)))
                x += wseg + pw * 0.03
            rows.append((yy, "redact", seg)); yy += ph * 0.058
        li += 1

    pcx, pcy = w / 2, h / 2

    td = Path(tempfile.mkdtemp())
    for i in range(n):
        t = i / fps
        pr = i / max(1, n - 1)
        frame = look.graded_background(w, h, pal, seed=seed, drift=pr)
        page = paper.copy()
        d = ImageDraw.Draw(page)
        # reveal lines type/fade in first; redaction bars sweep in after
        for ri, (ry, kind, payload) in enumerate(rows):
            if kind == "reveal":
                ra = look.ease_out_cubic(max(0.0, (t - 0.3) / 0.5))
                if ra > 0.01:
                    col = tuple(int(c + (44 - c) * 1) for c in (44, 38, 30))
                    d.text((int(pw * 0.1), int(ry)), payload, font=line_font,
                           fill=(44, 38, 30))
            else:
                for (sx, ex) in payload:
                    ba = look.ease_out_cubic(max(0.0, (t - 0.6 - ri * 0.06) / 0.4))
                    if ba <= 0.01:
                        continue
                    ex2 = sx + (ex - sx) * ba
                    d.rounded_rectangle([int(sx), int(ry - ph * 0.004),
                                         int(ex2), int(ry + ph * 0.040)],
                                        radius=3, fill=(18, 16, 14))
        # composite page onto frame with a soft shadow
        sh = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        ImageDraw.Draw(sh).rectangle([px0 + 12, py0 + 16, px0 + pw + 12, py0 + ph + 16],
                                     fill=(0, 0, 0, 150))
        frame = Image.alpha_composite(frame.convert("RGBA"),
                                      sh.filter(ImageFilter.GaussianBlur(18))).convert("RGB")
        frame.paste(page, (px0, py0))
        d = ImageDraw.Draw(frame, "RGBA")
        # red TOP SECRET stamp lands askew over the page
        sa = look.ease_out_cubic(max(0.0, (t - (0.5 + len(rows) * 0.06)) / 0.4))
        if sa > 0.01:
            stf = look.font("title", int(h * 0.06))
            stamp_im = Image.new("RGBA", (int(w * 0.5), int(h * 0.14)), (0, 0, 0, 0))
            sd = ImageDraw.Draw(stamp_im)
            tw = sd.textlength(stamp.upper(), font=stf)
            sd.rounded_rectangle([10, 10, tw + 50, int(h * 0.10)], radius=8,
                                 outline=(180, 30, 22), width=6)
            sd.text((30, int(h * 0.018)), stamp.upper(), font=stf, fill=(180, 30, 22))
            stamp_im = stamp_im.rotate(-11, expand=True, resample=Image.BICUBIC)
            sc = 1.18 - 0.18 * sa
            # sit the stamp LOW over the redacted bars so it never obscures the
            # one legible reveal line near centre.
            frame = look.paste_center(frame, stamp_im, cx=pcx + w * 0.04,
                                      cy=pcy + h * 0.20, scale=sc,
                                      opacity=min(1.0, sa) * 0.92)

        frame = look.vignette(frame, strength=0.62)
        frame = look.film_grain(frame, seed=seed, amount=5.0, t=t)
        fade = look.fade_alpha(t, dur, fps)
        if fade < 1.0:
            frame = look.fade_frame(frame, fade, pal)
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
