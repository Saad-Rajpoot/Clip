"""Primitive: era_stamp_overlay  (Section · hybrid, biography family).

A compact ERA MARKER that rides OVER living footage — a time/place stamp without
a full card. A big bold YEAR settles into a safe lower-third corner (its last
digit optionally ODOMETER-ROLLS into place, e.g. 2013→2014); beneath it an
optional small place/era line and an optional UNIT COUNTER ("N <pip> CARS/DAY")
count up beside a tiny original pictograph pip. The real footage stays clearly
visible — only a soft, low scrim sits behind the two text rows so they read; it
is NEVER a full card and NEVER covers the centre/faces.

This is a HYBRID OVERLAY. It accepts an optional `bg_image` (a footage frame);
if absent it renders over a neutral graded bed (charcoal, lifted off black) to
SIMULATE the live shot. If the supplied frame is near-black its shadows are
lifted to CARD_STAGE_FLOOR so a dissolve landing on the clip never reads as a
black frame. The in/out envelope fades the FOREGROUND ONLY (text + scrim) toward
the charcoal stage — the footage bed is always present — so a cut/dissolve into
the stamp never dips to black.

GROUNDING (forensic GRAMMAR only, never pixels): a modern exposé-biography
"era-year stamp + iconographic production-rate counter" beat
(`research/motion_graphics_expansion/biography_family/frame_sequences/musk/
seq11_era_rate_counter/`, P11 in MUSK_FRAME_AUDIT.md, BIOGRAPHY_FAMILY_PLAN
item 7): a huge year with an odometer-rolling last digit (2013→2014) sits over
continuing factory footage, with a count-up rate ("521→…→1,000 /week") and a
small unit pictograph beneath it. We reproduce ONLY the GRAMMAR — era stamp +
odometer year + count-up unit rate + pictograph, kept as a restrained footage
overlay with a light scrim. We DO NOT copy its centred white-Helvetica layout,
its chromatic-aberration, its tunnel/factory footage, or — explicitly — its red
CAR icon: Vidlore parks the stamp in a SAFE CORNER (never centre), wears the
charcoal `look` palette / serif / grain, and draws an ORIGINAL abstract amber
"pip" pictograph (a rounded marker), not any vehicle/competitor glyph.

Pure-local (PIL + numpy frames → ffmpeg encode). No paid API. Deterministic.
Never crashes on a missing/bad bg, a bad/unicode year, or missing unit fields.

    render("x.mp4", year=2014, place="FREMONT, CALIFORNIA",
           unit_label="CARS/WEEK", unit_count=1000, bg_image="factory.jpg")
    render("x.mp4", year=1913, palette_name="parchment_sepia")

Returns a dict: {ok, path, frames, dur_s, render_s, w, h, ...}.
"""
from __future__ import annotations

import re
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter

from .. import look

SPEC = {
    "id": "era_stamp_overlay", "family": "hybrid",
    "roles": ["context", "establish", "era", "chronology", "setup"],
    "niches_ok": ["biography", "history", "geopolitics", "crime"],
    "niches_unsuitable": ["science", "comedy", "lifestyle", "reaction"],
    "intensity_range": [1, 3], "duration_range": [2.5, 4.0],
    "easing": "easeOutCubic", "audio_cue": "tick_soft",
    "repeat_cooldown_s": 45, "per_video_cap": 3, "cost": "low",
    "overlay": True,                      # hybrid: drawn OVER footage/bg or sim bed
    "review_override": ["year", "place", "unit_label", "unit_count", "palette"],
    "required_inputs": ["year"],
    "fallback": False,
    "grounded_in": "musk/seq11_era_rate_counter (~5:12-5:21) — era-year stamp + "
                   "iconographic production-rate counter (GRAMMAR only)",
}


# ───────────────────────── input coercion (never crash) ─────────────────────
def _year_str(year) -> str:
    """Coerce `year` to a clean display string. Accepts an int/float, a string
    like '2014', 'c. 1913', '1840s', '1913 AD', or junk/unicode — always returns
    a short printable token (falls back to the raw stripped text, then '—')."""
    if year is None:
        return "—"
    if isinstance(year, float):
        if year != year or year in (float("inf"), float("-inf")):   # NaN / inf
            return "—"
        year = int(round(year))
    if isinstance(year, int):
        return str(year)
    s = str(year).strip()
    if not s:
        return "—"
    # keep it compact for a corner stamp (years/eras are short)
    return s[:14]


def _roll_digits(disp: str):
    """Decide which trailing run of ASCII digits can odometer-ROLL into place.

    Returns (prefix, roll_digits, suffix) where `roll_digits` is the trailing
    numeric run (rolled from the previous value, e.g. ...3 → ...4). If the token
    has no trailing digit (e.g. an era word) `roll_digits` is "" and the whole
    thing is treated as a static prefix. Only the LAST digit actually animates —
    the rest are anchors — matching the seq11 single-digit odometer feel."""
    m = re.search(r"(\d+)(\D*)$", disp)
    if not m:
        return disp, "", ""
    digits = m.group(1)
    suffix = m.group(2)
    prefix = disp[: m.start(1)]
    return prefix, digits, suffix


def _num_str(v) -> str | None:
    """Coerce a unit count to a grouped integer string ('1,000'); None/garbage →
    None (the counter row is then suppressed). Tolerates '1,000', '1.2k', 50000."""
    if v is None:
        return None
    if isinstance(v, bool):                       # guard: bool is an int subclass
        return None
    if isinstance(v, (int, float)):
        try:
            if isinstance(v, float) and (v != v):
                return None
            return f"{int(round(float(v))):,}"
        except (ValueError, OverflowError):
            return None
    s = str(v).strip().lower().replace(",", "")
    mult = 1.0
    for suf, m in (("billion", 1e9), ("million", 1e6),
                   ("b", 1e9), ("m", 1e6), ("k", 1e3)):
        if s.endswith(suf):
            s = s[: -len(suf)].strip()
            mult = m
            break
    try:
        return f"{int(round(float(s) * mult)):,}"
    except (TypeError, ValueError):
        return None


def _clean_line(s, limit: int) -> str:
    """A safe one-line label: stripped, collapsed whitespace, length-capped."""
    s = re.sub(r"\s+", " ", str(s or "").strip())
    return s[:limit]


# ───────────────────────── footage bed ─────────────────────────
def _bed(bg_image, w, h, pal, seed):
    """The live shot the stamp rides over: a graded footage frame if one is
    supplied AND loads, otherwise a neutral graded charcoal bed (lifted off
    black) to SIMULATE footage. A real frame is graded toward the palette,
    cover-fit to the canvas, gently darkened so the corner stamp stays legible
    while the footage CLEARLY stays the hero — and luma-floored to
    CARD_STAGE_FLOOR so a near-black frame never reads as a dead-black dissolve."""
    if bg_image is not None:
        try:
            ok = isinstance(bg_image, Image.Image) or Path(str(bg_image)).exists()
            if ok:
                src = bg_image if isinstance(bg_image, Image.Image) \
                    else Image.open(str(bg_image))
                im = look.grade_media(src.convert("RGB"), pal, strength=0.45)
                iw, ih = im.size                       # cover-fit to the canvas
                s = max(w / iw, h / ih)
                im = im.resize((max(w, int(iw * s)), max(h, int(ih * s))),
                               Image.LANCZOS)
                x = (im.width - w) // 2
                y = (im.height - h) // 2
                im = im.crop((x, y, x + w, y + h))
                # gentle darken (keep footage clearly visible — light, not heavy)
                arr = np.asarray(im).astype(np.float32) * 0.88
                im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")
                # luma floor: a near-black footage still must never read as a
                # dead-black frame under a dissolve (CARD_STAGE_FLOOR contract).
                fl = int(look.CARD_STAGE_FLOOR * 0.85)
                return ImageChops.lighter(
                    im, Image.new("RGB", im.size, (fl, fl, fl)))
        except Exception:                                          # noqa: BLE001
            pass
    # simulated footage bed: charcoal graded background, lifted off black so the
    # frame never reads as black on a dissolve in/out.
    return look.graded_background(w, h, pal, seed=seed, drift=0.22,
                                  floor=look.CARD_STAGE_FLOOR)


# ───────────────────────── original unit pictograph ────────────────────────
def _unit_pip(d: ImageDraw.ImageDraw, cx: int, cy: int, r: int, accent,
              accent_hi, alpha: float):
    """An ORIGINAL abstract 'unit' pictograph — a small rounded amber marker
    (a filled pip with a soft ring + a tiny inner notch) standing in for ONE
    counted unit. Deliberately NOT a vehicle/competitor glyph: it is a neutral
    quantity token that suits cars, barrels, troops, victims, launches, etc."""
    a = look.clamp01(alpha)
    if a <= 0.01:
        return
    # outer soft ring
    d.ellipse([cx - r, cy - r, cx + r, cy + r],
              outline=(*accent, int(150 * a)), width=2)
    # solid pip
    ir = int(r * 0.56)
    d.ellipse([cx - ir, cy - ir, cx + ir, cy + ir],
              fill=(*accent_hi, int(225 * a)))
    # tiny inner notch (top) — a small original detail, not an icon
    nk = max(2, int(r * 0.22))
    d.line([(cx, cy - ir + 1), (cx, cy - ir + 1 + nk)],
           fill=(*accent, int(200 * a)), width=2)


# ───────────────────────── scrim (NOT a full card) ─────────────────────────
def _row_scrim(w, h, x0, y0, x1, y1, alpha: float, radius: int):
    """A soft, low rounded dark scrim behind a text ROW so it reads over busy
    footage WITHOUT a hard opaque box (light scrim only — never a full card).
    RGBA, frame-sized, gaussian-softened."""
    band = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    a = look.clamp01(alpha)
    if a <= 0.01:
        return band
    d = ImageDraw.Draw(band)
    d.rounded_rectangle([int(x0), int(y0), int(x1), int(y1)],
                        radius=int(radius), fill=(8, 9, 12, int(118 * a)))
    return band.filter(ImageFilter.GaussianBlur(7))


def render(out_path, *, year, place: str = "", unit_label: str = "",
           unit_count=None, bg_image=None, dur: float = 3.2, fps: int = 30,
           w: int = 1920, h: int = 1080, palette_name: str = "amber_gold",
           seed: int = 0, crf: int = 18, **_ignored) -> dict:
    """Render an era-stamp overlay. `year` is coerced to a display string (the
    trailing digit odometer-rolls in). `place` is an optional small era/city
    line; `unit_label` + `unit_count` add an optional count-up unit row with an
    original pip pictograph. All overlay text lives in a SAFE lower-left corner /
    lower-third — the centre and any faces stay clear."""
    pal = look.palette(palette_name)
    t0 = time.time()
    dur = float(min(4.0, max(2.5, dur)))
    n = max(2, int(round(dur * fps)))

    # ---- coerce inputs (never crash) ---------------------------------------
    disp = _year_str(year)
    pre, roll, suf = _roll_digits(disp)
    place_txt = _clean_line(place, 34).upper()
    unit_lab = _clean_line(unit_label, 18).upper()
    unit_num = _num_str(unit_count)
    has_unit = bool(unit_num and unit_lab)

    accent = tuple(pal["accent"]); accent_hi = tuple(pal["accent_hi"])
    ink = tuple(pal["text"]); muted = tuple(pal["muted"])

    bed = _bed(bg_image, w, h, pal, seed)

    # ---- fonts (year is the hero; place/unit are small & restrained) -------
    year_font = look.font("numeral", int(h * 0.150))   # big bold serif year
    place_font = look.font("label", int(h * 0.026))
    unit_num_font = look.font("numeral", int(h * 0.052))
    unit_lab_font = look.font("label", int(h * 0.026))

    # ---- safe-corner anchor (lower-LEFT lower-third; never centre/faces) ----
    # the year baseline band sits higher when a place/unit row hangs below it, so
    # the whole stamp keeps a comfortable title-safe bottom margin.
    margin_x = int(w * 0.065)
    has_place = bool(_clean_line(place, 34))
    extra_rows = (1 if has_place else 0) + (1 if has_unit else 0)
    base_y = int(h * (0.755 if extra_rows else 0.815))  # year baseline band

    # measure the year so the scrim hugs it (not a full card)
    _tmp = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    yb = _tmp.textbbox((0, 0), disp, font=year_font)
    year_w, year_h = yb[2] - yb[0], yb[3] - yb[1]
    # full text-block height (year + hairline + optional place + optional unit)
    # so the soft scrim can wash the WHOLE block (still a scrim, never a card).
    block_h = year_h + int(h * 0.030)                   # year + hairline gap
    if has_place:
        block_h += int(h * 0.046)
    if has_unit:
        block_h += int(h * 0.052)

    # animation phases (fractions of dur):
    #   year rises + fades in early; its last digit odometer-rolls into place;
    #   place line fades up; unit row counts up; clean dissolve out via fade_alpha.
    roll_t0, roll_t1 = 0.18, 0.72                       # odometer window (s-frac)

    td = Path(tempfile.mkdtemp())
    for i in range(n):
        t = i / fps
        tf = t / dur                                    # normalised 0..1
        frame = bed.copy()

        # ---- entrance envelope for the whole stamp (rise + fade) -----------
        ent = look.ease_out_cubic(min(1.0, t / 0.55))
        rise = int((1.0 - ent) * h * 0.045)             # small upward settle
        stamp_a = ent                                   # foreground alpha

        # geometry for this frame's year top-left
        yx = margin_x
        yy = base_y - year_h + rise

        # ---- soft scrim washing the whole text block (light, NOT a card) ---
        # one low, gaussian-soft rounded wash behind the year + place + unit
        # rows so muted text reads over bright footage without an opaque plate.
        if stamp_a > 0.01:
            pad_x, pad_y = int(h * 0.024), int(h * 0.018)
            # block width = widest of the year and the estimated lower rows
            row_w = year_w
            if has_place:
                row_w = max(row_w, _tmp.textbbox(
                    (0, 0), place_txt, font=place_font)[2])
            if has_unit:
                est = (_tmp.textbbox((0, 0), (unit_num or "") + "  ",
                                     font=unit_num_font)[2]
                       + int(h * 0.06)
                       + _tmp.textbbox((0, 0), unit_lab, font=unit_lab_font)[2])
                row_w = max(row_w, est)
            sc = _row_scrim(w, h, yx - pad_x, yy - pad_y,
                            yx + row_w + pad_x, yy + block_h + pad_y,
                            stamp_a * 0.85, radius=int(h * 0.022))
            frame = Image.alpha_composite(frame.convert("RGBA"), sc).convert("RGB")

        # ---- the YEAR: static prefix digits + odometer-rolling last digit --
        if stamp_a > 0.01:
            # static part (everything but the rolling trailing run's LAST digit)
            roll_p = look.ease_out_cubic(
                (tf - roll_t0) / max(1e-3, roll_t1 - roll_t0))
            if roll:
                static_head = pre + roll[:-1]           # anchors
                last_d = roll[-1]
            else:
                static_head, last_d = pre, ""
            # draw the static head (+ suffix later) as one glowing serif block
            head_txt = static_head
            # build the full settled string for width anchoring of the suffix
            cur_x = yx
            if head_txt:
                head_layer = look.text_with_glow(
                    head_txt, year_font, fill=ink, glow=pal["glow"],
                    glow_radius=int(h * 0.016), glow_alpha=0.30, pad=int(h * 0.05))
                frame = look.paste_center(
                    frame, head_layer,
                    cx=cur_x + (_tmp.textbbox((0, 0), head_txt,
                                              font=year_font)[2]) / 2.0,
                    cy=yy + year_h / 2.0, opacity=stamp_a)
                cur_x += _tmp.textbbox((0, 0), head_txt, font=year_font)[2]

            # the odometer last digit: roll from the PREVIOUS digit up to the
            # target as roll_p goes 0→1 (a vertical wheel — prev slides up & out,
            # target slides up & in), then settles. Single-digit, on-theme.
            if last_d:
                dw = (_tmp.textbbox((0, 0), last_d, font=year_font)[2]
                      - _tmp.textbbox((0, 0), last_d, font=year_font)[0])
                cell_cx = cur_x + dw / 2.0 + int(h * 0.004)
                cell_cy = yy + year_h / 2.0
                if roll_p >= 0.999:
                    dl = look.text_with_glow(
                        last_d, year_font, fill=ink, glow=pal["glow"],
                        glow_radius=int(h * 0.016), glow_alpha=0.30,
                        pad=int(h * 0.05))
                    frame = look.paste_center(frame, dl, cx=cell_cx, cy=cell_cy,
                                              opacity=stamp_a)
                else:
                    prev_d = str((int(last_d) - 1) % 10)
                    travel = year_h * 0.92               # wheel travel distance
                    # easing for the wheel
                    wp = look.ease_in_out_cubic(roll_p)
                    # outgoing previous digit slides UP and out (fades)
                    out_layer = look.text_with_glow(
                        prev_d, year_font, fill=ink, glow=pal["glow"],
                        glow_radius=int(h * 0.014), glow_alpha=0.20,
                        pad=int(h * 0.05))
                    frame = look.paste_center(
                        frame, out_layer, cx=cell_cx,
                        cy=cell_cy - wp * travel,
                        opacity=stamp_a * (1.0 - wp))
                    # incoming target digit rises into place from below
                    in_layer = look.text_with_glow(
                        last_d, year_font, fill=ink, glow=pal["glow"],
                        glow_radius=int(h * 0.016), glow_alpha=0.30,
                        pad=int(h * 0.05))
                    frame = look.paste_center(
                        frame, in_layer, cx=cell_cx,
                        cy=cell_cy + (1.0 - wp) * travel,
                        opacity=stamp_a * wp)
                cur_x = cell_cx + dw / 2.0

            # trailing suffix (e.g. 's' in '1840s', 'AD') after the digits
            if suf:
                sl = look.text_with_glow(
                    suf, place_font, fill=muted, glow=pal["bg_b"],
                    glow_radius=2, glow_alpha=0.0, pad=int(h * 0.02))
                frame = look.paste_center(
                    frame, sl,
                    cx=cur_x + (_tmp.textbbox((0, 0), suf,
                                              font=place_font)[2]) / 2.0 + 4,
                    cy=yy + int(h * 0.010), opacity=stamp_a * 0.9)

        # ---- thin accent hairline under the year (anchors the stamp) -------
        if stamp_a > 0.01:
            hp = look.ease_out_cubic(min(1.0, (tf - 0.10) / 0.35))
            if hp > 0.01:
                d = ImageDraw.Draw(frame, "RGBA")
                hl_y = yy + year_h + int(h * 0.018)
                hl_w = int(year_w * 0.82 * hp)
                d.line([(yx + 2, hl_y), (yx + 2 + hl_w, hl_y)],
                       fill=(*accent, int(210 * stamp_a)), width=3)
                d.line([(yx + 2, hl_y + 2), (yx + 2 + hl_w, hl_y + 2)],
                       fill=(*accent_hi, int(70 * stamp_a)), width=1)

        # ---- optional small PLACE / era line (under the hairline) ----------
        place_y = yy + year_h + int(h * 0.052)
        if place_txt:
            pa = look.ease_out_cubic(min(1.0, (tf - 0.22) / 0.34)) * stamp_a
            if pa > 0.02:
                pl = look.text_with_glow(
                    place_txt, place_font, fill=muted, glow=pal["bg_b"],
                    glow_radius=2, glow_alpha=0.0, pad=int(h * 0.02))
                # left-anchored: paste so its left edge sits at margin_x
                plw = pl.width
                frame = look.paste_center(frame, pl, cx=yx + plw / 2.0,
                                          cy=place_y, opacity=pa)

        # ---- optional UNIT COUNTER row: N <pip> LABEL (count-up) -----------
        if has_unit:
            ua = look.ease_out_cubic(min(1.0, (tf - 0.30) / 0.40)) * stamp_a
            if ua > 0.02:
                unit_y = (place_y + int(h * 0.045)) if place_txt \
                    else (yy + year_h + int(h * 0.058))
                # count-up: ease the displayed number from 0 → target
                cfrac = look.ease_out_cubic(min(1.0, (tf - 0.30) / 0.46))
                try:
                    target = int(unit_num.replace(",", ""))
                except ValueError:
                    target = 0
                cur_v = int(round(target * cfrac))
                num_s = f"{cur_v:,}"
                # number (small serif, palette ink) — left-anchored
                nl = look.text_with_glow(
                    num_s, unit_num_font, fill=ink, glow=pal["glow"],
                    glow_radius=int(h * 0.010), glow_alpha=0.22, pad=int(h * 0.03))
                # measure settled width for stable layout (no horizontal jitter)
                settled_w = (_tmp.textbbox((0, 0), unit_num.replace(",", "0"),
                                           font=unit_num_font)[2])
                nlw = nl.width
                nl_cx = yx + nlw / 2.0
                frame = look.paste_center(frame, nl, cx=nl_cx, cy=unit_y,
                                          opacity=ua)
                # original pip pictograph after the number
                pip_r = int(h * 0.018)
                pip_cx = yx + settled_w + int(h * 0.028)
                d = ImageDraw.Draw(frame, "RGBA")
                _unit_pip(d, int(pip_cx), int(unit_y), pip_r,
                          accent, accent_hi, ua)
                # unit label after the pip
                ll = look.text_with_glow(
                    unit_lab, unit_lab_font, fill=muted, glow=pal["bg_b"],
                    glow_radius=2, glow_alpha=0.0, pad=int(h * 0.02))
                lab_cx = pip_cx + pip_r + int(h * 0.012) + ll.width / 2.0
                frame = look.paste_center(frame, ll, cx=lab_cx, cy=unit_y,
                                          opacity=ua * 0.96)

        # ---- finish: vignette + grain; FOREGROUND-ONLY in/out fade ---------
        frame = look.vignette(frame, strength=0.45)
        frame = look.film_grain(frame, seed=seed, amount=4.0, t=t)
        fa = look.fade_alpha(t, dur, fps)
        if fa < 1.0:
            frame = look.fade_frame(frame, fa, pal)     # toward charcoal, not black
        frame.save(td / f"f{i:05d}.png")

    return _encode(td, out_path, fps, crf, n, dur, w, h, t0, disp, has_unit)


def _encode(td, out_path, fps, crf, n, dur, w, h, t0, disp, has_unit) -> dict:
    try:
        from vidlore.ffmpeg_tool import ffmpeg_exe
        ff = ffmpeg_exe()
    except Exception:                                              # noqa: BLE001
        ff = "ffmpeg"
    out_path = str(out_path)
    cmd = [ff, "-y", "-hide_banner", "-loglevel", "error", "-framerate", str(fps),
           "-i", f"{td}/f%05d.png", "-c:v", "libx264", "-crf", str(crf),
           "-pix_fmt", "yuv420p", "-colorspace", "bt709", "-color_primaries",
           "bt709", "-color_trc", "bt709", "-movflags", "+faststart", out_path]
    r = subprocess.run(cmd, capture_output=True, text=True)
    ok = (r.returncode == 0 and Path(out_path).exists())
    look.cleanup_frames(td)
    return {"ok": ok, "path": out_path, "frames": n, "dur_s": round(dur, 2),
            "render_s": round(time.time() - t0, 2), "w": w, "h": h,
            "year": disp, "unit": bool(has_unit),
            "err": (r.stderr[-200:] if not ok else "")}
