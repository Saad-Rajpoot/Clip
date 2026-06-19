"""Primitive: relationship_roster.

Introduce "the cast." A full-screen, charcoal-stage roster that opens on ONE
portrait card and then RESOLVES that single card into a clean, evenly-spaced row
of N (2-5) people — each a face-safe portrait tile (or a graded monogram plate
when no photo exists), a bold serif NAME, and a muted ROLE caption chip. The
"focus on one, then reveal the set" choreography is the grammar of the "and there
were others" beat: founders, suspects/players, rivals, the supporting cast.

Forensic principle (NOT an asset copy): premium documentaries stage a named-cast
build as 1->N framed portrait cards with a consistent name+role lockup and a
staggered, eased entrance. Rebuilt from that PRINCIPLE with Vidlore's own dark
look system (deep-charcoal CARD_STAGE_FLOOR stage, warm gold accent, serif
typography) — never a competitor's portraits, wordmark, fonts, chrome or layout.
The reference rosters sit on a bright white infographic plate; this is its dark,
cinematic inverse.

Distinct from `org_hierarchy_tree` / `connection_web` (abstract leader-line
NODES) — this is a flat row of *portrait cards* that LABELS a named cast.

Pure-local (PIL + numpy frames -> ffmpeg encode). No paid API. Deterministic.

    render("x.mp4", people=[
        {"name": "Martin Eberhard", "role": "Co-Founder · 2003"},
        {"name": "Marc Tarpenning", "role": "Co-Founder"},
        {"name": "Ze'ev Drori",     "role": "CEO · 2007"},
    ])
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
    "id": "relationship_roster", "family": "biography",
    "roles": ["context", "relationship", "cast", "reveal"],
    "niches_ok": ["biography", "business", "crime", "history"],
    "intensity_range": [2, 4], "duration_range": [4.0, 6.5],
    "easing": "easeOutCubic", "audio_cue": "soft_per_tile_tick",
    "repeat_cooldown_s": 90, "per_video_cap": 1, "cost": "low",
    "review_override": ["people", "kicker", "palette"],
    # introduce-the-cast: not a single-subject hold, not an abstract node graph
    "fallback": False,
    "required_inputs": ["people"],
}

_MAX_TILES = 5


# ───────────────────────── input normalisation ─────────────────────────
def _normalise_people(people) -> list[dict]:
    """Accept a list of dicts ({name, role, note?, portrait?/photo?/path?/image?})
    OR plain strings OR (name, role) pairs. Skip empty entries, clamp to 5, and
    never crash on malformed input."""
    out: list[dict] = []
    if not people:
        return out
    if isinstance(people, (str, bytes)):                # a single bare name
        people = [people]
    try:
        seq = list(people)
    except TypeError:
        return out
    for it in seq:
        name, role, photo = "", "", None
        if isinstance(it, dict):
            name = str(it.get("name") or it.get("label") or it.get("title") or "").strip()
            role = str(it.get("role") or it.get("caption") or it.get("sub")
                       or it.get("note") or "").strip()
            photo = (it.get("portrait") or it.get("portrait_path") or it.get("photo")
                     or it.get("path") or it.get("image"))
        elif isinstance(it, (list, tuple)):
            if len(it) >= 1:
                name = str(it[0] or "").strip()
            if len(it) >= 2:
                role = str(it[1] or "").strip()
            if len(it) >= 3 and it[2]:
                photo = it[2]
        elif it is not None:
            name = str(it).strip()
        if not name:
            continue
        if photo is not None:
            photo = str(photo)
            if not (photo and Path(photo).exists()):
                photo = None
        out.append({"name": name, "role": role, "photo": photo})
        if len(out) >= _MAX_TILES:
            break
    return out


def _initials(name: str) -> str:
    parts = [w for w in str(name or "").replace(".", " ").split() if w]
    if not parts:
        return "?"
    if len(parts) == 1:
        return parts[0][:2].upper()
    return (parts[0][0] + parts[-1][0]).upper()


# ───────────────────────── tile rendering ─────────────────────────
def _portrait_inner(photo, iw: int, ih: int, pal: dict, name: str, seed: int) -> Image.Image:
    """The iw×ih image INSIDE a tile's border: a face-safe graded portrait when a
    photo exists, else a graded monogram plate (warm vertical gradient + crushed
    initials) so a personless cast still reads as a designed card, never a hole."""
    if photo:
        try:
            im = Image.open(photo).convert("RGB")
            crop = (look.face_safe_crop(im, iw, ih) if hasattr(look, "face_safe_crop")
                    else im.resize((iw, ih), Image.LANCZOS))
            return look.grade_media(crop, pal, strength=0.5).convert("RGB")
        except Exception:                                          # noqa: BLE001
            pass
    # ── monogram plate ──────────────────────────────────────────
    a = np.array(look._lift_color(pal["bg_a"], look.CARD_STAGE_FLOOR * 0.9), np.float32)
    b = np.array(look._lift_color(pal["bg_b"], look.CARD_STAGE_FLOOR * 0.55), np.float32)
    yy = np.linspace(0.0, 1.0, ih, dtype=np.float32)[:, None, None]
    grad = (a[None, None] * (1.0 - yy) + b[None, None] * yy)
    rng = np.random.default_rng(seed + len(name))
    grad = np.clip(grad + rng.normal(0, 3.0, (ih, iw, 1)), 0, 255)
    inner = Image.fromarray(grad.astype(np.uint8), "RGB")
    f = look.font("numeral", int(ih * 0.34))
    ini = look.text_with_glow(_initials(name), f, fill=pal["accent_hi"],
                              glow=pal["glow"], glow_radius=int(ih * 0.05),
                              glow_alpha=0.42, pad=int(ih * 0.10))
    inner = inner.convert("RGBA")
    inner.alpha_composite(ini, ((iw - ini.width) // 2, int(ih * 0.50 - ini.height / 2)))
    return inner.convert("RGB")


def _build_tile(person: dict, tw: int, th: int, pal: dict, seed: int) -> Image.Image:
    """A finished RGBA portrait tile: drop shadow + gold inset border + face-safe
    portrait/monogram. Sized so the bordered photo is tw×th; the returned canvas
    has padding for the shadow. The NAME + ROLE are drawn separately (so they can
    reveal a beat after the tile lands)."""
    bd = max(3, int(min(tw, th) * 0.022))             # gold border thickness
    iw, ih = tw - 2 * bd, th - 2 * bd
    inner = _portrait_inner(person.get("photo"), iw, ih, pal, person["name"], seed)

    pad = max(16, int(tw * 0.14))                     # room for the soft shadow
    canvas = Image.new("RGBA", (tw + pad * 2, th + pad * 2), (0, 0, 0, 0))
    # drop shadow (offset down/right, blurred)
    sh = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    so = max(6, int(tw * 0.05))
    ImageDraw.Draw(sh).rectangle(
        [pad + so, pad + so + so // 2, pad + so + tw, pad + so + so // 2 + th],
        fill=(0, 0, 0, 175))
    sh = sh.filter(ImageFilter.GaussianBlur(max(8, int(tw * 0.06))))
    canvas.alpha_composite(sh)
    # gold frame plate behind the photo
    frame = Image.new("RGBA", (tw, th),
                      (*[int(c * 0.82) for c in pal["accent"]], 255))
    frame.paste(inner, (bd, bd))
    # thin bright inner keyline for a refined edge
    fd = ImageDraw.Draw(frame)
    fd.rectangle([bd - 1, bd - 1, tw - bd, th - bd],
                 outline=(*pal["accent_hi"], 150), width=1)
    canvas.alpha_composite(frame, (pad, pad))
    return canvas


def _caption(name: str, role: str, name_font, role_font, pal: dict, max_w: int):
    """RGBA name + role caption lockup (centered), height-trimmed. Returns the
    composited layer + the y where the name baseline block ends (for stacking)."""
    ink = tuple(pal["text"]); muted = tuple(pal["muted"])
    nm = look.text_with_glow(name, name_font, fill=ink, glow=pal["bg_b"],
                             glow_radius=5, glow_alpha=0.0, pad=10)
    if nm.width > max_w:                              # keep names inside the column
        s = max_w / nm.width
        nm = nm.resize((max_w, max(1, int(nm.height * s))), Image.LANCZOS)
    parts = [nm]
    gap = int(name_font.size * 0.28)
    chip = None
    if role:
        rl = look.text_with_glow(role.upper(), role_font, fill=muted, glow=pal["bg_b"],
                                 glow_radius=2, glow_alpha=0.0, pad=8)
        if rl.width > max_w:
            s = max_w / rl.width
            rl = rl.resize((max_w, max(1, int(rl.height * s))), Image.LANCZOS)
        # muted role CHIP: faint rounded plate behind the role text
        cpx, cpy = int(role_font.size * 0.55), int(role_font.size * 0.30)
        cw, ch = rl.width + cpx * 2, rl.height + cpy * 2
        chip = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
        cd = ImageDraw.Draw(chip)
        rad = ch // 2
        cd.rounded_rectangle([0, 0, cw - 1, ch - 1], radius=rad,
                             fill=(*[int(c * 0.55) for c in pal["bg_a"]], 150),
                             outline=(*muted, 90), width=1)
        chip.alpha_composite(rl, (cpx, cpy))
        parts.append(chip)
    total_h = sum(p.height for p in parts) + (gap if len(parts) > 1 else 0)
    total_w = max(p.width for p in parts)
    lay = Image.new("RGBA", (total_w, total_h), (0, 0, 0, 0))
    y = 0
    lay.alpha_composite(nm, ((total_w - nm.width) // 2, y))
    y += nm.height + gap
    if chip is not None:
        lay.alpha_composite(chip, ((total_w - chip.width) // 2, y))
    return lay


# ───────────────────────── render ─────────────────────────
def render(out_path, *, people, dur: float = 5.5, fps: int = 30,
           w: int = 1920, h: int = 1080, palette_name: str = "amber_gold",
           kicker: str = "", seed: int = 0, crf: int = 18) -> dict:
    pal = look.palette(palette_name)
    t0 = time.time()
    cast = _normalise_people(people)
    if not cast:                                       # never crash on empty input
        return {"ok": False, "err": "relationship_roster: no valid people",
                "path": str(out_path)}
    k = len(cast)
    n = max(2, int(round(dur * fps)))

    # ── layout: an evenly-spaced row of k tiles ─────────────────
    # tile size shrinks as the cast grows so 5 tiles still breathe.
    tw_frac = {1: 0.30, 2: 0.26, 3: 0.225, 4: 0.195, 5: 0.165}[k]
    tw = int(w * tw_frac)
    th = int(tw * 1.28)                                # ~portrait 4:5
    row_cy = int(h * 0.45)
    margin = w * (0.085 if k >= 4 else 0.12)
    if k == 1:
        slot_x = [w * 0.5]
    else:
        span = w - 2 * margin
        slot_x = [margin + span * (j / (k - 1)) for j in range(k)]
    col_w = int((w - 2 * margin) / k * 0.96) if k > 1 else int(w * 0.5)

    name_font = look.font("title", int(h * (0.040 if k >= 4 else 0.046)))
    role_font = look.font("label", int(h * (0.0215 if k >= 4 else 0.024)))
    kick_font = look.font("label", int(h * 0.026))

    # pre-build tiles + captions once (static content)
    tiles = [_build_tile(p, tw, th, pal, seed + idx) for idx, p in enumerate(cast)]
    caps = [_caption(p["name"], p.get("role", ""), name_font, role_font, pal, col_w)
            for p in cast]
    cap_cy = int(row_cy + th * 0.5 + h * 0.058)

    bed = (look.graded_background(w, h, pal, seed=seed, floor=look.CARD_STAGE_FLOOR)
           if hasattr(look, "graded_background")
           else Image.new("RGB", (w, h), tuple(pal["bg_b"])))

    # ── choreography timing ─────────────────────────────────────
    # Phase A: hero (tile 0) holds enlarged near the focus point.
    # Phase B: hero eases to its slot while the rest stagger-resolve into the row.
    focus_x = w * (0.30 if k > 1 else 0.5)             # where the hero opens
    hero_in = 0.55                                     # hero fades up by here
    resolve_at = max(0.95, dur * 0.20)                 # row build begins
    per_stagger = min(0.26, (dur * 0.42) / max(1, k))  # gap between tile lands
    tile_win = 0.55                                    # each tile's ease window
    cap_lag = 0.28                                     # caption trails its tile

    # accent rule (under the row) reveal
    rule_at = resolve_at + 0.15

    td = Path(tempfile.mkdtemp())
    for i in range(n):
        t = i / fps
        drift = i / max(1, n - 1)
        frame = (look.graded_background(w, h, pal, seed=seed, drift=drift,
                                        floor=look.CARD_STAGE_FLOOR)
                 if hasattr(look, "graded_background") else bed.copy())

        # kicker label (small gold caps) above the row
        ka = look.ease_out_cubic(min(1.0, t / 0.5))
        if kicker and ka > 0.01:
            ki = look.text_with_glow(kicker.upper(), kick_font, fill=pal["accent_hi"],
                                     glow=pal["glow"], glow_radius=int(h * 0.012),
                                     glow_alpha=0.35, pad=12)
            frame = look.paste_center(frame, ki, cx=w / 2,
                                      cy=int(h * 0.135), opacity=ka * 0.95)

        # thin accent rule sweeping under the row (the single accent gesture)
        ra = look.ease_out_cubic(min(1.0, max(0.0, (t - rule_at) / 0.6)))
        if ra > 0.01 and k > 1:
            d = ImageDraw.Draw(frame, "RGBA")
            ry = int(row_cy + th * 0.5 + h * 0.028)
            half = (w * 0.5 - margin * 0.7) * ra
            d.line([(w / 2 - half, ry), (w / 2 + half, ry)],
                   fill=(*pal["accent"], int(190 * ra)), width=2)
            for sx in (-1, 1):                          # gold end-pips
                d.ellipse([w / 2 + sx * half - 3, ry - 3, w / 2 + sx * half + 3, ry + 3],
                          fill=(*pal["accent_hi"], int(210 * ra)))

        for j in range(k):
            # entrance schedule: tile 0 is the hero (enters first, at the focus
            # point, oversized) then migrates to its slot; tiles 1..k staggered.
            if j == 0:
                ent = look.ease_out_cubic(min(1.0, t / hero_in))
                mig = look.ease_in_out_cubic(min(1.0, max(0.0, (t - resolve_at) / 0.85)))
                # hero is bigger during phase A, settles to row scale in phase B
                hero_scale = 1.34 - 0.34 * mig if k > 1 else 1.0
                cx = focus_x + (slot_x[0] - focus_x) * mig
                cy = row_cy
                scale = hero_scale * (0.96 + 0.04 * ent)
                op = ent
                cap_a = look.ease_out_cubic(min(1.0, max(0.0, (t - (resolve_at + cap_lag)) / tile_win)))
            else:
                delay = resolve_at + j * per_stagger
                ent = look.ease_out_cubic(min(1.0, max(0.0, (t - delay) / tile_win)))
                if ent <= 0.001:
                    continue
                # rise + settle in from slightly below, scaling up
                cx = slot_x[j]
                cy = row_cy + (1.0 - ent) * h * 0.045
                scale = 0.90 + 0.10 * ent
                op = ent
                cap_a = look.ease_out_cubic(min(1.0, max(0.0, (t - (delay + cap_lag)) / tile_win)))

            frame = look.paste_center(frame, tiles[j], cx=cx, cy=cy,
                                      scale=scale, opacity=op)
            # caption tracks the tile's current x (so the hero's caption follows it)
            if cap_a > 0.01:
                frame = look.paste_center(frame, caps[j], cx=cx,
                                          cy=cap_cy + (1.0 - cap_a) * h * 0.018,
                                          opacity=cap_a)

        frame = look.vignette(frame, strength=0.6)
        frame = look.film_grain(frame, seed=seed, amount=5.0, t=t)
        fa = look.fade_alpha(t, dur, fps)
        if fa < 1.0:
            frame = (look.fade_frame(frame, fa, pal) if hasattr(look, "fade_frame")
                     else frame)
        frame.save(td / f"f{i:05d}.png")

    return _encode(td, out_path, fps, crf, n, dur, w, h, t0)


def _encode(td, out_path, fps, crf, n, dur, w, h, t0) -> dict:
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
    if hasattr(look, "cleanup_frames"):
        look.cleanup_frames(td)
    else:
        import shutil as _sh
        _sh.rmtree(td, ignore_errors=True)
    return {"ok": ok, "path": str(out_path), "frames": n, "dur_s": round(dur, 2),
            "render_s": round(time.time() - t0, 2), "w": w, "h": h,
            "err": (r.stderr[-200:] if not ok else "")}
