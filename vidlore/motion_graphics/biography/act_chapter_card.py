"""Primitive: act_chapter_card.

A structural ACT DIVIDER that punctuates a long biography/history piece — a small
tracked KICKER (an act label, an era, a year) over a large serif THESIS TITLE that
settles in from a gentle per-letter scatter, with one accent rule drawn beneath
it. It rides over LIVE footage (a graded ``bg_image`` still that keeps pushing in
underneath), NOT a black slate, so the film never stops moving. Short cinematic
punctuation: it scatters-in, holds for a single breath, then a soft mask eases it
back out — the footage carries straight on into the next act.

This is the editorial cousin of a film act-break / chapter heading. Distinct from
its siblings (original Vidlore composition — no asset, font, layout or wording
copied):
  * ``kinetic_keyword`` reveals ONE concept word with a bottom-up mask wipe; it is
    an emphasis stab, not a two-tier structural divider over footage.
  * ``portrait_legend_reveal`` introduces a PERSON (face + name); this introduces
    an ACT (a thesis line + label), with no portrait.
  * ``gold_number_callout`` is a single hero numeral. This carries prose, not data.

If ``bg_image`` is near-black it is lifted to ``look.CARD_STAGE_FLOOR`` (mirroring
gold_number_callout's media floor) so a dissolve/cut landing on the divider never
reads as a dead-black frame; the footage detail still shows through. With no
``bg_image`` it falls back to a graded charcoal stage.

Pure-local (PIL + numpy frames -> ffmpeg encode). No paid API. Deterministic.

    render("x.mp4", title="THE RISE", kicker="CHAPTER TWO",
           bg_image="refinery.jpg", palette_name="amber_gold")

Returns a dict: {ok, path, frames, dur_s, render_s, w, h, ...}.

Grounding (forensic PRINCIPLE, not pixels): R seq02 — a small tracked act-eyebrow
over a large serif title that resolves from a light letter-scatter and settles to
a clean centred baseline, all over a near-full-screen tableau that keeps pushing
in, then dissolves straight into the act (a genuine breath/reset). We reproduce
that GRAMMAR — kicker+title hierarchy, settle-from-scatter, landing over live
imagery, short cinematic punctuation — and deliberately NOT the channel's
"[ADJECTIVE] MR. [SURNAME]" naming formula, its burning-refinery art, or its
exact serif. The caller supplies whatever kicker/title it wants.
"""
from __future__ import annotations

import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter

from .. import look

SPEC = {
    "id": "act_chapter_card", "family": "typography",
    "roles": ["hook", "context", "turn", "escalation", "act", "chapter"],
    "niches_ok": ["biography", "history", "business", "crime", "spy",
                  "geopolitics"],
    "intensity_range": [2, 4], "duration_range": [3.0, 5.0],
    "easing": "easeOutCubic", "audio_cue": "soft_whoosh",
    "repeat_cooldown_s": 60, "per_video_cap": 3, "cost": "low",
    "required_inputs": ["title"],
    "review_override": ["title", "kicker", "bg_image", "palette"],
    "fallback": False,
}


# ───────────────────────── text helpers ─────────────────────────
def _txt_w(draw: "ImageDraw.ImageDraw", s: str, fnt) -> int:
    box = draw.textbbox((0, 0), s, font=fnt)
    return box[2] - box[0]


def _wrap_to_lines(draw, text: str, fnt, max_w: int, max_lines: int) -> list:
    """Greedy word-wrap into at most `max_lines` lines that each fit `max_w`.
    Falls back to a hard character split for a single unbreakable long token so a
    URL-ish / no-space title never overflows. Never raises."""
    words = [w for w in str(text).split() if w]
    if not words:
        return [""]
    lines, cur = [], ""
    for w in words:
        trial = w if not cur else cur + " " + w
        if _txt_w(draw, trial, fnt) <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
        if len(lines) >= max_lines:               # out of line budget
            break
    if cur and len(lines) < max_lines:
        lines.append(cur)
    # any remaining words get folded onto the last line (will be shrunk to fit)
    used = sum(len(l.split()) for l in lines)
    if used < len(words) and lines:
        lines[-1] = " ".join([lines[-1]] + words[used:])
    # hard-split a single token still wider than the box (no spaces to wrap on)
    if len(lines) == 1 and _txt_w(draw, lines[0], fnt) > max_w and " " not in lines[0]:
        s = lines[0]
        lo, mid = 0, len(s) // 2
        for cut in range(len(s) - 1, 0, -1):
            if _txt_w(draw, s[:cut], fnt) <= max_w:
                mid = cut
                break
        lines = [s[:mid], s[mid:]][:max_lines]
    return lines[:max_lines]


def _fit_title(text: str, w: int, h: int, max_lines: int = 2):
    """Auto-fit: shrink the serif title until it wraps to <= max_lines within the
    safe width. Returns (font, lines, line_h). Robust to very long / unicode /
    single-token titles."""
    safe_w = int(w * 0.82)
    size = int(h * 0.150)                 # generous start; hero line, big
    min_size = max(12, int(h * 0.060))    # never below the h*0.06 readability floor
    scratch = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    fnt = look.font("title", size)
    lines = _wrap_to_lines(scratch, text, fnt, safe_w, max_lines)
    while size > min_size:
        fnt = look.font("title", size)
        lines = _wrap_to_lines(scratch, text, fnt, safe_w, max_lines)
        widest = max((_txt_w(scratch, l, fnt) for l in lines), default=0)
        if widest <= safe_w and len(lines) <= max_lines:
            break
        size = int(size * 0.92)
    fnt = look.font("title", max(size, min_size))
    box = scratch.textbbox((0, 0), "AÉQy", font=fnt)
    line_h = int((box[3] - box[1]) * 1.18)
    return fnt, lines, line_h


def _tracked_layer(text: str, fnt, fill, glow, *, tracking: float = 0.30,
                   glow_radius: int = 6, glow_alpha: float = 0.0,
                   pad: int = 28) -> Image.Image:
    """Render letter-spaced (tracked) caps as one RGBA layer — the small kicker.
    `tracking` is extra spacing as a fraction of the font size. A soft glow keeps
    the tiny eyebrow legible over busy footage."""
    s = str(text)
    scratch = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    asc = fnt.size
    track_px = int(asc * tracking)
    # measure
    widths = [scratch.textbbox((0, 0), ch, font=fnt) for ch in s]
    adv = [(b[2] - b[0]) for b in widths]
    total = sum(adv) + track_px * max(0, len(s) - 1)
    bb = scratch.textbbox((0, 0), s if s else "X", font=fnt)
    th = bb[3] - bb[1]
    W = max(1, total) + pad * 2
    H = th + pad * 2
    oy = pad - bb[1]
    glow_img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    txt = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    gd, tdr = ImageDraw.Draw(glow_img), ImageDraw.Draw(txt)
    x = pad
    for ch, a in zip(s, adv):
        b = scratch.textbbox((0, 0), ch, font=fnt)
        ox = x - b[0]
        if glow_alpha > 0:
            gd.text((ox, oy), ch, font=fnt,
                    fill=(*glow, int(235 * look.clamp01(glow_alpha))))
        tdr.text((ox, oy), ch, font=fnt, fill=(*fill, 255))
        x += a + track_px
    if glow_alpha > 0:
        glow_img = glow_img.filter(ImageFilter.GaussianBlur(glow_radius))
    return Image.alpha_composite(glow_img, txt)


def _line_glyphs(line: str, fnt):
    """Per-glyph advance metadata for one title line so each letter can be placed
    individually (needed for the settle-from-scatter motion). Returns
    (glyphs[(ch, x_offset)], total_w)."""
    scratch = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    glyphs, x = [], 0
    for ch in line:
        b = scratch.textbbox((0, 0), ch, font=fnt)
        adv = b[2] - b[0]
        glyphs.append((ch, x, b[0]))
        # space has zero ink-width bbox; give it a real advance
        if adv <= 0:
            sb = scratch.textbbox((0, 0), "n", font=fnt)
            adv = int((sb[2] - sb[0]) * 0.55)
        x += adv + max(1, int(fnt.size * 0.02))
    return glyphs, x


# ───────────────────────── render ─────────────────────────
def render(out_path, *, title: str, kicker: str = "", bg_image=None,
           dur: float = 4.0, fps: int = 30, w: int = 1920, h: int = 1080,
           palette_name: str = "amber_gold", seed: int = 0,
           crf: int = 18) -> dict:
    pal = look.palette(palette_name)
    t0 = time.time()
    # clamp duration into the SPEC window so it stays cinematic punctuation
    dur = float(max(2.6, min(5.5, dur if dur and dur > 0 else 4.0)))
    n = max(2, int(round(dur * fps)))
    title = ("" if title is None else str(title)).strip() or " "
    kicker = ("" if kicker is None else str(kicker)).strip()

    # ---- live footage bed (graded, floored) OR charcoal stage --------------
    media = None
    if bg_image and Path(str(bg_image)).exists():
        try:
            src = Image.open(str(bg_image)).convert("RGB")
            media = look.grade_media(src, pal, strength=0.58)
            media = media.resize((w, h), Image.LANCZOS)
        except Exception:                          # noqa: BLE001 — never crash
            media = None

    # ---- type: auto-fit title to <=2 lines + tracked kicker ----------------
    title_font, lines, line_h = _fit_title(title, w, h, max_lines=2)
    # pre-compute per-glyph layout for each title line (for scatter-settle)
    line_meta = []
    for ln in lines:
        glyphs, total_w = _line_glyphs(ln, title_font)
        line_meta.append((ln, glyphs, total_w))
    kick_font = look.font("label", max(12, int(h * 0.030)))
    kick_layer = None
    if kicker:
        kick_layer = _tracked_layer(
            kicker.upper(), kick_font, fill=pal["accent_hi"], glow=pal["glow"],
            tracking=0.42, glow_radius=max(3, int(h * 0.008)), glow_alpha=0.35,
            pad=int(h * 0.03))

    # block geometry (centred). Title block sits a touch above centre; kicker +
    # rule above it, so the lockup reads top-down: kicker · rule · title.
    n_lines = len(lines)
    title_block_h = n_lines * line_h
    cy = h * 0.50
    title_top = cy - title_block_h / 2 + line_h * 0.12
    kicker_y = title_top - h * 0.055
    rule_y = title_top - h * 0.018          # accent rule just under the title
    rule_y = title_top + title_block_h + h * 0.030

    # seeded per-glyph scatter vectors (gentle: small horizontal spread + slight
    # vertical lift + a hair of rotation-feel via x jitter). Each glyph settles on
    # a staggered schedule so the line "resolves" left-to-right-ish, not all at
    # once — the seq02 "settle from a light scatter" read.
    rng = np.random.default_rng(int(seed) & 0x7FFFFFFF)
    glyph_anim = []
    for li, (ln, glyphs, total_w) in enumerate(line_meta):
        gx0 = w / 2 - total_w / 2
        per = []
        ng = max(1, len(glyphs))
        for gi, (ch, gx, lb) in enumerate(glyphs):
            dx = float(rng.uniform(-1, 1)) * (w * 0.060)     # gentle spread
            dy = float(rng.uniform(-1, 1)) * (h * 0.045)
            # stagger start across the line + between lines
            order = (gi / ng) * 0.55 + li * 0.10
            per.append((ch, gx0 + gx - lb, dx, dy, order))
        glyph_anim.append((li, per))

    # phase timing (fractions of dur). SHORT: scatter-settle, one breath, ease-out.
    settle_end = 0.42           # title fully settled by here
    hold_end = 0.66             # brief settled beat
    # exit (mask wipe) runs hold_end..1.0

    accent = tuple(pal["accent"])
    accent_hi = tuple(pal["accent_hi"])
    floor_rgb = (int(look.CARD_STAGE_FLOOR),) * 3

    td = Path(tempfile.mkdtemp())
    for i in range(n):
        t = i / fps
        pr = i / max(1, n - 1)

        # --- background: live footage pushing in (or charcoal stage) ---------
        if media is not None:
            zb = 1.0 + 0.055 * look.ease_out_cubic(pr)     # slow continuous push
            bw, bh = int(w * zb), int(h * zb)
            frame = media.resize((bw, bh), Image.LANCZOS).crop(
                ((bw - w) // 2, (bh - h) // 2,
                 (bw - w) // 2 + w, (bh - h) // 2 + h))
            # media luma-floor: a near-black still must never read dead-black under
            # a dissolve/cut. Lift shadows to the charcoal stage floor (footage
            # detail/highlights survive; the black bed becomes deep charcoal).
            frame = ImageChops.lighter(frame, Image.new("RGB", frame.size, floor_rgb))
        else:
            frame = look.graded_background(w, h, pal, seed=seed, drift=pr,
                                           floor=look.CARD_STAGE_FLOOR)

        # --- readability scrim behind the text block (NOT a black slate) -----
        # a soft, centre-weighted vertical darkening so the title stays legible
        # over busy footage without hiding the footage. Fades with the foreground.
        fa = look.fade_alpha(t, dur, fps, fin=0.22, fout=0.34)
        scrim_a = 0.40 * fa
        if scrim_a > 0.01:
            yy = np.linspace(0, 1, h, dtype=np.float32)
            band = np.exp(-((yy - 0.50) ** 2) / (2 * 0.20 ** 2))   # gaussian band
            mul = (1.0 - scrim_a * band)[:, None, None]
            frame = Image.fromarray(
                np.clip(np.asarray(frame).astype(np.float32) * mul, 0, 255)
                .astype(np.uint8), "RGB")

        # --- exit mask: a soft vertical wipe that eases the LOCKUP out, footage
        # continuing underneath. progress 0 (fully shown) .. 1 (gone). Computed
        # once per frame; applied as an alpha multiplier on every text layer. ---
        ex = 0.0
        if pr > hold_end:
            ex = look.ease_in_out_cubic((pr - hold_end) / max(1e-3, 1.0 - hold_end))
        # The soft upward wipe itself is applied per-row inside `_apply_alpha`
        # below (a feathered gradient edge driven by `ex`), so the footage keeps
        # running underneath as the lockup eases out.

        # global foreground alpha: clip envelope (fa) + entry fade-in only. The
        # EXIT is the masked vertical wipe (per-row gradient in `_apply_alpha`,
        # driven by `ex`) — NOT a whole-block fade — so the lockup is carried off
        # by a soft mask while the footage runs straight on underneath. `fa`'s
        # own fout is kept short so it doesn't pre-empt the wipe.
        fg_in = look.ease_out_cubic(min(1.0, t / 0.34))
        base_a = fa * fg_in
        # the small kicker + thin rule ride the exit as a quick scalar fade (a
        # directional wipe on a 1-line eyebrow / hairline would read oddly); the
        # big title gets the masked vertical wipe. Both leave together.
        small_exit = (1.0 - ex)

        d = ImageDraw.Draw(frame, "RGBA")

        # --- kicker (tracked caps), rises + fades in just before the title ---
        if kick_layer is not None:
            ka = look.ease_out_cubic(max(0.0, (t - 0.04) / 0.40)) * base_a * small_exit
            if ka > 0.02:
                ky = int(kicker_y - (1 - look.ease_out_cubic(min(1.0, t / 0.45))) * h * 0.018)
                frame = look.paste_center(frame, kick_layer, cx=w / 2, cy=ky,
                                          opacity=ka)
                d = ImageDraw.Draw(frame, "RGBA")

        # --- title: per-glyph settle-from-scatter ---------------------------
        # draw onto an offscreen RGBA so glow + per-glyph opacity composite cleanly
        tlayer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        tdraw = ImageDraw.Draw(tlayer)
        for li, per in glyph_anim:
            base_y = title_top + li * line_h
            for (ch, fx, dx, dy, order) in per:
                # settle progress for this glyph (staggered window inside settle_end)
                span = max(1e-3, settle_end - order * (settle_end * 0.55))
                lp = look.ease_out_cubic(look.clamp01((pr - order * settle_end * 0.55) / span))
                gx = fx + dx * (1.0 - lp)
                gy = base_y + dy * (1.0 - lp)
                ga = lp                              # fade each glyph in as it lands
                if ch.strip() == "":
                    continue
                col = (*pal["text"], int(255 * look.clamp01(ga)))
                tdraw.text((gx, gy), ch, font=title_font, fill=col)

        # soft gold bloom behind the (settled) title for premium depth
        if base_a > 0.02:
            glow = tlayer.filter(ImageFilter.GaussianBlur(max(3, int(h * 0.012))))
            ga_arr = np.asarray(glow).astype(np.float32)
            # recolour the blurred copy toward palette glow, keep its alpha shape
            gcol = np.array(pal["glow"], np.float32)
            rgb = ga_arr[..., :3]
            ga_arr[..., :3] = gcol[None, None]
            ga_arr[..., 3] = np.clip(ga_arr[..., 3] * 0.5, 0, 255)
            glow = Image.fromarray(ga_arr.astype(np.uint8), "RGBA")
            # apply exit wipe + base alpha to BOTH glow and crisp text
            def _apply_alpha(layer):
                a = np.asarray(layer).astype(np.float32)
                # vertical wipe gradient (per-row) for the eased exit
                if ex > 0.0:
                    rows = np.arange(h, dtype=np.float32)
                    edge = (title_top + title_block_h + h * 0.06) - ex * (title_block_h + h * 0.30)
                    feather = h * 0.11
                    wcol = np.clip((rows - edge) / feather, 0.0, 1.0)  # 0 above edge→hidden
                    a[..., 3] *= wcol[:, None]
                a[..., 3] *= look.clamp01(base_a)
                return Image.fromarray(a.astype(np.uint8), "RGBA")
            frame = Image.alpha_composite(frame.convert("RGBA"), _apply_alpha(glow))
            frame = Image.alpha_composite(frame, _apply_alpha(tlayer)).convert("RGB")
            d = ImageDraw.Draw(frame, "RGBA")

        # --- one accent rule, drawn in under the title (then wiped on exit) --
        rule_full = w * 0.16
        rp = look.ease_out_cubic(look.clamp01((pr - settle_end * 0.55) / 0.30))
        rule_a = rp * base_a * small_exit
        if rule_a > 0.02:
            half = rule_full * rp * 0.5
            ry = int(rule_y)
            cxp = w / 2
            # glow underlay
            gl = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            ImageDraw.Draw(gl).line(
                [(cxp - half, ry), (cxp + half, ry)],
                fill=(*pal["glow"], int(150 * rule_a)), width=max(3, int(h * 0.006)))
            gl = gl.filter(ImageFilter.GaussianBlur(max(2, int(h * 0.006))))
            frame = Image.alpha_composite(frame.convert("RGBA"), gl).convert("RGB")
            d = ImageDraw.Draw(frame, "RGBA")
            d.line([(cxp - half, ry), (cxp + half, ry)],
                   fill=(*accent_hi, int(235 * rule_a)), width=max(2, int(h * 0.003)))
            # two small end-ticks for a refined, engraved feel
            for sgn in (-1, 1):
                ex_x = cxp + sgn * half
                d.line([(ex_x, ry - int(h * 0.008)), (ex_x, ry + int(h * 0.008))],
                       fill=(*accent, int(200 * rule_a)), width=max(1, int(h * 0.0025)))

        # --- finish: vignette + grain on the composite ----------------------
        frame = look.vignette(frame, strength=0.58)
        frame = look.film_grain(frame, seed=seed, amount=4.5, t=t)
        frame.save(td / f"f{i:05d}.png")

    return _encode(td, out_path, fps, crf, n, dur, w, h, t0)


def _encode(td, out_path, fps, crf, n, dur, w, h, t0) -> dict:
    try:
        from vidlore.ffmpeg_tool import ffmpeg_exe
        ff = ffmpeg_exe()
    except Exception:                                           # noqa: BLE001
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
            "err": (r.stderr[-200:] if not ok else "")}
