"""Primitive: verdict_duality_card.

A biography FINALE JUDGMENT. One editorial thesis line (`verdict`) types on
clause-by-clause in large serif, optionally framed above by a two-pole DUALITY
(`pole_a` vs `pole_b`, e.g. "INNOVATOR" ↔ "MANIPULATOR", "ADMIRED" ↔ "CRITICIZED",
"VISIONARY" ↔ "CONTROVERSIAL"). The two poles are weighted EQUALLY — same scale,
same colour, mirrored about a thin gold fulcrum — so they read as a FAIR tension,
never an accusation or a clickbait split-screen. The verdict accumulates one
clause at a time, framed top and bottom by a thin gold rule, on a graded charcoal
stage that rides look.CARD_STAGE_FLOOR (a cut/dissolve into it never reads black).
Reserved for the close: judge the subject, do not recap.

Grounded in the Rockefeller verdict beat (research .../seq11_verdict_duality/,
audit pattern P12): the moral summation staged as a designed device — a thesis
that builds clause-by-clause for gravity, framed by a good-side↔bad-side duality,
bracketed by horizontal rules, one warm palette throughout. This is an ORIGINAL
Vidlore composition — a clean charcoal stage, a balanced gold fulcrum between two
equal poles, and thin gold framing rules — NOT the reference's handwritten yellow
sticky-note, scribe-at-desk tableau, GOOD/BAD red thought-bubbles, mirrored-
silhouette art, burning-face plate, its font, or its sentence. The poles are a
fair balance beam, deliberately the opposite of a sensational split-screen.

Pure-local (PIL + numpy frames -> ffmpeg encode). No paid API. Deterministic.

    render("verdict.mp4",
           verdict="He built the modern world and broke the rules to do it.",
           pole_a="INNOVATOR", pole_b="MANIPULATOR",
           palette_name="cold_steel")

Returns a dict: {ok, path, frames, dur_s, render_s, w, h, err}.
"""
from __future__ import annotations

import re
import subprocess
import tempfile
import time
import unicodedata
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .. import look

SPEC = {
    "id": "verdict_duality_card", "family": "statements",
    "roles": ["thesis", "verdict", "payoff", "resolution", "legacy"],
    # The biography close that JUDGES rather than recaps — distinct from
    # statement_card (a single static thesis, no clause-by-clause build and no
    # duality framing). Reserved for the finale of a character piece.
    "niches_ok": ["biography", "history", "crime", "business"],
    "intensity_range": [3, 5], "duration_range": [4.5, 6.5],
    "easing": "easeOutCubic", "audio_cue": "low_resolve",
    "repeat_cooldown_s": 999, "per_video_cap": 1, "cost": "low",
    "required_inputs": ["verdict"],
    "review_override": ["verdict", "pole_a", "pole_b", "palette"],
    "fallback": False,
}

_VERDICT_MAX = 90          # hard clamp on the thesis length (chars)
_POLE_MAX = 22             # a pole is a single word/short label


# ───────────────────────── text helpers ─────────────────────────
def _clean(s) -> str:
    """Coerce to a safe single-line string: drop control glyphs, collapse
    whitespace, normalise unicode (so a unicode/emoji verdict never crashes the
    typesetter). Never raises."""
    try:
        s = "" if s is None else str(s)
    except Exception:                                          # noqa: BLE001
        return ""
    s = unicodedata.normalize("NFC", s)
    out = []
    for ch in s:
        cat = unicodedata.category(ch)
        if cat[0] == "C" and ch not in ("\n", "\t", " "):      # control glyphs
            continue
        out.append(" " if ch in ("\n", "\t") else ch)
    return " ".join("".join(out).split()).strip()


def _clamp(s: str, n: int) -> str:
    """Clamp to <= n chars on a word boundary, with an ellipsis if cut."""
    if len(s) <= n:
        return s
    cut = s[:n].rstrip()
    sp = cut.rfind(" ")
    if sp >= int(n * 0.6):                                     # keep whole words
        cut = cut[:sp].rstrip()
    return (cut.rstrip(",.;:—- ") + "…")


def _clauses(text: str) -> list[str]:
    """Split a one-line thesis into the natural clauses that should type on one
    at a time. Break at commas / semicolons / colons / em- or en-dashes (the
    rhetorical 'breath' points), keeping the delimiter glued to the clause it
    closes so the punctuation appears WITH its clause, not orphaned. Falls back
    to ~5-6 word chunks for a long clause with no internal punctuation so a
    delimiter-free sentence still builds progressively rather than popping whole.
    Always returns at least one non-empty clause when `text` is non-empty."""
    text = text.strip()
    if not text:
        return []
    # split but keep the delimiter token attached to the preceding clause
    raw = re.split(r"(\s*[,;:]\s+|\s+[—–-]\s+)", text)
    parts, buf = [], ""
    for tok in raw:
        if tok is None:
            continue
        if re.fullmatch(r"\s*[,;:]\s+|\s+[—–-]\s+", tok):
            buf += tok.strip()                                 # keep punctuation
            parts.append(buf.strip())
            buf = ""
        else:
            buf += tok
    if buf.strip():
        parts.append(buf.strip())
    parts = [p for p in (p.strip() for p in parts) if p]
    if not parts:
        parts = [text]

    # a single very long clause (no internal breaks) -> chunk by words so it
    # still types on in stages instead of one big block
    if len(parts) == 1 and len(parts[0].split()) > 9:
        words = parts[0].split()
        step = 5 if len(words) <= 14 else 6
        parts = [" ".join(words[i:i + step]) for i in range(0, len(words), step)]
    return parts


def _fit_font(role: str, text: str, max_w: int, start_px: int,
              min_px: int):
    """Largest serif/condensed font (role) whose `text` fits in `max_w`.
    Shrinks from start_px down to min_px. Always returns a usable font."""
    tmp = Image.new("L", (8, 8))
    d = ImageDraw.Draw(tmp)
    px = max(min_px, int(start_px))
    while px > min_px:
        f = look.font(role, px)
        try:
            box = d.textbbox((0, 0), text, font=f)
            if (box[2] - box[0]) <= max_w:
                return f
        except Exception:                                      # noqa: BLE001
            return f
        px -= max(2, int(px * 0.06))
    return look.font(role, min_px)


def _wrap(d, text, font, max_w: int) -> list[str]:
    """Greedy word-wrap `text` to lines no wider than `max_w`."""
    words, lines, cur = text.split(), [], ""
    for wd in words:
        t = (cur + " " + wd).strip()
        try:
            fits = d.textlength(t, font=font) <= max_w
        except Exception:                                      # noqa: BLE001
            fits = len(t) < 40
        if fits or not cur:
            cur = t
        else:
            lines.append(cur)
            cur = wd
    if cur:
        lines.append(cur)
    return lines


def _layout_clauses(clauses: list[str], fnt, max_w: int,
                    d) -> tuple[list[str], list[int]]:
    """Wrap each clause to fit `max_w`, returning the flat list of display lines
    plus, for each clause, the index of its LAST line (so the type-on can reveal
    a clause one whole clause at a time, line-wraps included)."""
    lines: list[str] = []
    last_line_of_clause: list[int] = []
    for cl in clauses:
        wl = _wrap(d, cl, fnt, max_w) or [cl]
        lines.extend(wl)
        last_line_of_clause.append(len(lines) - 1)
    return lines, last_line_of_clause


# ───────────────────────── render ─────────────────────────
def render(out_path, *, verdict: str, pole_a: str = "", pole_b: str = "",
           dur: float = 5.5, fps: int = 30, w: int = 1920, h: int = 1080,
           palette_name: str = "cold_steel", seed: int = 0,
           crf: int = 18) -> dict:
    t0 = time.time()
    out_path = str(out_path)
    try:
        pal = look.palette(palette_name)
        verdict = _clamp(_clean(verdict), _VERDICT_MAX)
        pole_a = _clamp(_clean(pole_a), _POLE_MAX).upper()
        pole_b = _clamp(_clean(pole_b), _POLE_MAX).upper()
        if not verdict:
            return {"ok": False, "path": out_path, "frames": 0, "dur_s": 0.0,
                    "render_s": round(time.time() - t0, 2), "w": w, "h": h,
                    "err": "empty verdict"}
        # a duality needs BOTH poles; one pole alone is not a fair tension, so
        # fall back to a verdict-only card (never a lone, accusatory label).
        has_poles = bool(pole_a and pole_b)

        dur = max(2.0, float(dur))
        fps = max(1, int(fps))
        n = max(2, int(round(dur * fps)))

        probe = ImageDraw.Draw(Image.new("L", (8, 8)))

        # ── poles (balanced duality) ─────────────────────────────────────────
        pole_font = None
        pole_glyph_h = 0
        if has_poles:
            # one shared size for BOTH poles → neither dominates. Fit the LONGER
            # label so both stay inside their half of the stage.
            longer = pole_a if len(pole_a) >= len(pole_b) else pole_b
            half_w = int(w * 0.40)                              # each pole's lane
            pole_font = _fit_font("title", longer, half_w, int(h * 0.066),
                                  max(int(h * 0.034), 14))
            try:
                pb = probe.textbbox((0, 0), longer, font=pole_font)
                pole_glyph_h = pb[3] - pb[1]
            except Exception:                                  # noqa: BLE001
                pole_glyph_h = int(h * 0.06)

        # ── verdict thesis (clause-by-clause) ────────────────────────────────
        # size the serif to the verdict length (shorter → bigger), then fit.
        vlen = len(verdict)
        base = 0.082 if vlen < 46 else (0.066 if vlen < 70 else 0.056)
        vmax_w = int(w * 0.80)
        v_font = _fit_font("title", verdict, vmax_w, int(h * base),
                           max(int(h * 0.044), 14))
        clauses = _clauses(verdict)
        v_lines, last_line_of_clause = _layout_clauses(clauses, v_font,
                                                       vmax_w, probe)
        v_lines = v_lines[:5]                                   # readability cap
        n_clauses = max(1, len(last_line_of_clause))
        try:
            vb = probe.textbbox((0, 0), max(v_lines, key=len), font=v_font)
            v_glyph_h = vb[3] - vb[1]
        except Exception:                                      # noqa: BLE001
            v_glyph_h = int(h * base)
        v_lh = int(v_glyph_h * 1.34)
        v_block_h = v_lh * len(v_lines)

        # ── vertical layout anchors ─────────────────────────────────────────
        # poles ride the upper third; the verdict block is centred in the lower
        # middle, framed by two thin rules. Everything fits the safe area.
        if has_poles:
            poles_cy = h * 0.30
            v_top = h * 0.52
        else:
            poles_cy = 0.0
            v_top = h * 0.5 - v_block_h / 2.0
        rule_pad = int(h * 0.055)                              # rule gap above/below
        rule_top_y = int(v_top - rule_pad)
        rule_bot_y = int(v_top + v_block_h + rule_pad * 0.55)
        rule_half = int(w * 0.30)

        td = Path(tempfile.mkdtemp())
        for i in range(n):
            t = i / fps
            pr = i / max(1, n - 1)

            # graded charcoal stage (slow drift) on the dark stage floor
            frame = look.graded_background(w, h, pal, seed=seed, drift=pr,
                                           floor=look.CARD_STAGE_FLOOR)

            # foreground in/out envelope — NEVER multiplies the whole frame to
            # black (the lit stage always remains; only overlays fade).
            fa = look.fade_alpha(t, dur, fps, fin=0.40, fout=0.42)

            d = ImageDraw.Draw(frame, "RGBA")

            # ── duality: two equal poles + a fair gold fulcrum ──────────────
            if has_poles:
                # both poles fade in TOGETHER (symmetric) so neither is named
                # first — a fair tension, not an accusation.
                pa_in = look.ease_out_cubic(max(0.0, (t - 0.30) / 0.6))
                pole_alpha = pa_in * fa
                lane = w * 0.255                                # offset from centre
                pcy = poles_cy
                if pole_alpha > 0.01:
                    rise = (1.0 - pa_in) * h * 0.018
                    for lbl, sign in ((pole_a, -1.0), (pole_b, +1.0)):
                        timg = look.text_with_glow(
                            lbl, pole_font, fill=pal["text"], glow=pal["glow"],
                            glow_radius=max(6, int(h * 0.012)), glow_alpha=0.35,
                            pad=int(h * 0.04))
                        frame = look.paste_center(
                            frame, timg, cx=w * 0.5 + sign * lane,
                            cy=pcy + rise, opacity=pole_alpha)
                    d = ImageDraw.Draw(frame, "RGBA")

                # the fulcrum: a thin vertical gold rule between the poles with a
                # small diamond node at centre — a balance-beam pivot, the
                # deliberate OPPOSITE of a hard split-screen seam. Draws after
                # the poles settle, growing symmetrically from the centre.
                fw = look.ease_out_cubic(max(0.0, (t - 0.62) / 0.6))
                fw = min(fw, 1.0) * fa
                if fw > 0.01:
                    acc = tuple(int(c) for c in pal["accent"])
                    acc_hi = tuple(int(c) for c in pal["accent_hi"])
                    half_v = int(pole_glyph_h * 0.62 * fw)
                    fx = int(w * 0.5)
                    d.line([(fx, int(pcy) - half_v), (fx, int(pcy) + half_v)],
                           fill=(*acc, 235), width=max(2, int(h * 0.003)))
                    nd = int(max(4, int(h * 0.009)) * fw)
                    if nd >= 2:                                 # gold pivot diamond
                        d.polygon([(fx, int(pcy) - nd), (fx + nd, int(pcy)),
                                   (fx, int(pcy) + nd), (fx - nd, int(pcy))],
                                  fill=(*acc_hi, 255))

            # ── framing rules (top + bottom of the verdict block) ───────────
            # thin gold rules bracket the thesis (a restrained reading of the
            # reference's horizontal bracket bars — gold hairlines, not bars).
            rw = look.ease_out_cubic(max(0.0, (t - 0.18) / 0.55)) * fa
            if rw > 0.01:
                acc = tuple(int(c) for c in pal["accent"])
                hh = int(rule_half * rw)
                ru = ImageDraw.Draw(frame, "RGBA")
                lw = max(2, int(h * 0.0028))
                ru.line([(w // 2 - hh, rule_top_y), (w // 2 + hh, rule_top_y)],
                        fill=(*acc, 230), width=lw)
                ru.line([(w // 2 - hh, rule_bot_y), (w // 2 + hh, rule_bot_y)],
                        fill=(*acc, 200), width=lw)

            # ── verdict: clauses type on one at a time ──────────────────────
            # each clause owns a window of the clip; within its window its
            # display lines fade up together (rise + fade), then hold.
            lead = 0.85                                         # first clause start
            span = 0.92                                         # seconds per clause
            for ci, last_li in enumerate(last_line_of_clause):
                start_li = 0 if ci == 0 else last_line_of_clause[ci - 1] + 1
                c_start = lead + ci * span
                ca_in = look.ease_out_cubic(max(0.0, (t - c_start) / 0.62))
                ca = ca_in * fa
                if ca <= 0.01:
                    continue
                rise = (1.0 - ca_in) * h * 0.022
                for li in range(start_li, min(last_li + 1, len(v_lines))):
                    timg = look.text_with_glow(
                        v_lines[li], v_font, fill=pal["text"], glow=pal["bg_b"],
                        glow_radius=5, glow_alpha=0.0, pad=int(h * 0.03))
                    ly = v_top + li * v_lh + v_lh * 0.5 + rise
                    frame = look.paste_center(frame, timg, cx=w * 0.5, cy=ly,
                                              opacity=ca)

            # finish: vignette + grain on the full composite
            frame = look.vignette(frame, strength=0.6)
            frame = look.film_grain(frame, seed=seed, amount=4.5, t=t)
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
    look.cleanup_frames(td)                  # never leak PNG frames to /tmp
    return {"ok": ok, "path": str(out_path), "frames": n,
            "dur_s": round(dur, 2), "render_s": round(time.time() - t0, 2),
            "w": w, "h": h, "err": err}
