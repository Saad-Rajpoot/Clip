"""Auto background score (Vidlore always lays a music bed; without one the
video feels empty). The bed is *synthesized* with ffmpeg — two detuned
sine drones + brown-noise air, slow tremolo, lowpassed so it sits under
the narration. Being generated, it is inherently royalty-free and keyless.

A user-supplied track (brief.music / --music) always overrides this.
"""
from __future__ import annotations

from pathlib import Path

from .cache import scene_key
from .ffmpeg_tool import run, seeded_noise as _seeded

# Per-theme drone: (root Hz, fifth-ish Hz, tremolo Hz, lowpass cutoff Hz).
# Lower/darker for crime, warmer for history, brighter for modern.
# tremolo Hz must be >= 0.1 (ffmpeg filter minimum); these stay slow/ambient.
_THEME_TONE = {
    "crime":      (55.0, 82.5, 0.10, 850),
    "history":    (65.4, 98.0, 0.12, 1000),
    "modern":     (73.4, 110.0, 0.16, 1200),
    "minimalist": (55.0, 55.0, 0.10, 700),
    "standard":   (65.4, 98.0, 0.12, 1000),
}


# Each ~75s the score moves to a new "movement": the chord is
# transposed by one of these (cohesive, mostly minor-ish steps) and the
# tremolo/feel shifts slightly. Same timbre & theme palette, so it
# evolves naturally with the script instead of one endless loop.
_MOVES = (
    (1.000, 1.00), (0.917, 0.85), (1.122, 1.15),
    (0.841, 0.92), (1.059, 1.08), (0.794, 0.80),
)
_SECTION = 75.0
_XF = 3.0          # crossfade between movements (s)


def _arc_section_intensity(arc, n, section=_SECTION):
    """Map a per-scene emotional arc onto each of the ``n`` music
    movements (PHASE 3.2; sharpened during full-doc QA).

    ``arc`` = list of ``(duration_seconds, energy)`` per scene. Returns
    ``n`` intensities in 0..1 for movements of length ``section`` seconds
    each. The score follows the emotional curve, not a blind timer.

    QA fix: each movement's intensity is a BLEND of the window's mean
    energy and its PEAK (max) energy, not the mean alone. A real arc
    spikes at the climax for just a scene or two; a pure mean washed
    that spike out (a 2.5-min doc came back dead-flat because each long
    movement averaged the whole rise-and-fall to the same middling
    value). Weighting the peak lets the movement that actually covers
    the climax swell, while calm stretches still sit low."""
    if not arc:
        return [0.5] * n
    spans, es, t = [], [], 0.0
    for d, e in arc:
        d = max(0.0, float(d))
        spans.append((t, t + d, float(e)))
        es.append(float(e))
        t += d
    total = t
    lo, hi = min(es), max(es)
    rng = (hi - lo) if hi > lo else 1.0
    stride = section
    out = []
    for k in range(n):
        ws = k * stride
        we = min(ws + section, total) if total > 0 else ws + section
        num = den = 0.0
        peak = lo
        for (s, e, en) in spans:
            ov = min(we, e) - max(ws, s)
            if ov > 0:
                num += en * ov
                den += ov
                if en > peak:
                    peak = en
        mean_e = (num / den) if den > 0 else es[min(k, len(es) - 1)]
        score = 0.45 * mean_e + 0.55 * peak       # peak-biased
        out.append(max(0.0, min(1.0, (score - lo) / rng)))
    return out


def _intensity_to_params(intensity, k):
    """An emotional intensity (0..1) + movement index -> the movement's
    musical character: ``(transpose_ratio, tremolo_factor, level)``.

    Calm beats render lower/darker and softer; peaks lift brighter and
    swell louder. A tiny neighbour-distinct shimmer keeps two
    equal-intensity movements from crossfading into the exact same
    chord (so the bed always feels like it's moving).

    The loudness swing is driven by ``level`` (the post-limiter output
    gain in ``_render_section``), kept DECOUPLED from pitch: the
    transposition is a gentle harmonic drift (~0.96..1.04) for emotional
    colour, while ``level`` (calm ~0.45 → full 1.0) lands a controlled,
    predictable ~9-10 LUFS perceived swing across the arc. (Driving the
    swing through pitch instead is fragile — the bed's echo/resonance
    makes K-weighted loudness jump sharply for tiny transpose changes.)"""
    i = max(0.0, min(1.0, float(intensity)))
    tr = (0.96 + 0.08 * i) * (1.0 + (0.012 if k % 2 else -0.012))
    tf = 0.80 + 0.45 * i
    lvl = 0.38 + 0.62 * i
    return round(tr, 4), tf, round(lvl, 3)


_REF_LUFS = -23.0  # every movement is loudness-normalized to this
#                    (perceived, K-weighted) before the arc gain is
#                    applied (see build_music_bed)


def _render_section(f1, f2, trem, cutoff, seclen, out):
    """One self-contained movement of the score, rendered at a NEUTRAL
    level (no arc gain here).

    PHASE 3.2 note: the per-section emotional swing is applied later in
    ``build_music_bed`` as a loudness-normalized gain, NOT inside this
    render. That is deliberate — the bed's echo/resonance makes the
    section's perceived (K-weighted) loudness lurch by >15 LUFS for a
    <3% transposition change, so baking a gain in here would fight an
    unpredictable baseline. Rendering neutral + normalizing on the
    section's MEASURED LUFS decouples loudness from pitch, so the arc
    gain produces a clean, predictable swing."""
    p1, p2, p3 = round(f1 * 4, 2), round(f2 * 4, 2), round(f1 * 6, 2)
    mids = max(int(cutoff), 2200)
    fo = max(0.0, seclen - 1.6)
    fc = (
        "[0:a]volume=0.34[a0];[1:a]volume=0.24[a1];"
        "[2:a]lowpass=f=420,volume=0.13[a2];"
        "[3:a]vibrato=f=0.18:d=0.6,volume=0.78[a3];"
        "[4:a]vibrato=f=0.15:d=0.5,volume=0.56[a4];"
        "[5:a]vibrato=f=0.12:d=0.5,volume=0.34[a5];"
        "[a0][a1][a2][a3][a4][a5]amix=inputs=6:normalize=0,"
        f"tremolo=f={max(0.1, trem):.3f}:d=0.35,"
        f"highpass=f=38,lowpass=f={mids},aecho=0.8:0.75:680:0.30,"
        "afade=t=in:st=0:d=1.6,"
        f"afade=t=out:st={fo:.2f}:d=1.6,"
        "aformat=sample_fmts=s16:sample_rates=44100:"
        "channel_layouts=stereo,"
        "volume=2.0,alimiter=limit=0.18:level=0[m]"
    )
    run([
        "-f", "lavfi", "-i", f"sine=frequency={f1}:sample_rate=44100",
        "-f", "lavfi", "-i", f"sine=frequency={f2}:sample_rate=44100",
        "-f", "lavfi",
        "-i", _seeded("anoisesrc=color=brown:sample_rate=44100:amplitude=0.06"),
        "-f", "lavfi", "-i", f"sine=frequency={p1}:sample_rate=44100",
        "-f", "lavfi", "-i", f"sine=frequency={p2}:sample_rate=44100",
        "-f", "lavfi", "-i", f"sine=frequency={p3}:sample_rate=44100",
        "-filter_complex", fc,
        "-map", "[m]", "-t", f"{seclen:.2f}", str(out),
    ])


def _measure_lufs(path) -> float:
    """Integrated (K-weighted, perceived) loudness in LUFS of a rendered
    section, via ebur128. RMS is NOT a safe reference here: the bed's
    echo/resonance can shift a section's PERCEIVED loudness by >20 LUFS
    for the same RMS depending on where its spectral energy lands under
    the K-weighting curve. Normalizing on measured LUFS is what makes
    the arc swing perceptually clean and predictable."""
    import re
    import subprocess

    from .ffmpeg_tool import ffmpeg_exe

    proc = subprocess.run(
        [ffmpeg_exe(), "-hide_banner", "-nostats", "-i", str(path),
         "-af", "ebur128", "-f", "null", "-"],
        capture_output=True, text=True)
    vals = re.findall(r"I:\s*(-?\d+(?:\.\d+)?)\s*LUFS", proc.stderr or "")
    if not vals:
        return _REF_LUFS
    val = float(vals[-1])                       # Summary integrated value
    if val < -70.0:                             # below gate / silence
        return _REF_LUFS
    return val


def _norm_gain(measured_lufs: float, arc_gain: float) -> float:
    """Linear volume that normalizes a section from its measured LUFS to
    the fixed perceptual reference, then scales by the emotional
    ``arc_gain`` (so the swing is clean and pitch-independent)."""
    norm = 10.0 ** ((_REF_LUFS - float(measured_lufs)) / 20.0)
    norm = max(0.2, min(6.0, norm))            # sanity clamp
    return round(norm * max(0.2, min(1.0, float(arc_gain))), 4)


def _scale_swing(lvl: float, intensity: float) -> float:
    """Widen/narrow the arc gain around its mid so a Style Mode can make
    the score swell BIGGER (epic, intensity>1) or flatter (intensity<1)."""
    if abs(intensity - 1.0) < 1e-3:
        return lvl
    mid = 0.69
    return max(0.2, min(1.0, mid + (lvl - mid) * float(intensity)))


def build_music_bed(
    theme_name: str,
    duration: float,
    dest: Path,
    *,
    cache_dir: Path | None = None,
    arc: list | None = None,
    intensity: float = 1.0,
) -> Path:
    """Copyright-free synthesized score that EVOLVES with the story.

    A new movement (transposed chord + shifted feel) every ~75s,
    crossfaded smoothly, instead of one endless looping drone. When an
    ``arc`` (list of ``(scene_duration, energy)``) is supplied, each
    movement's transposition, loudness and tremolo are driven by the
    emotional intensity of the scenes it spans (PHASE 3.2) — the score
    rises into climaxes/reveals and settles into calm/resolution
    instead of stepping through a blind 75 s timer. Cached by
    (theme, duration, arc-shape)."""
    import math

    dur = max(2.0, round(float(duration), 2))
    f1, f2, trem, cutoff = _THEME_TONE.get(
        theme_name, _THEME_TONE["standard"])

    arc_fp = (tuple((round(float(d), 1), round(float(e), 2))
                    for d, e in arc) if arc else None)

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        key = scene_key("music5", theme_name, int(round(dur)), arc_fp,
                        round(float(intensity), 2))
        cached = cache_dir / f"{key}.wav"
        if cached.exists():
            import shutil

            shutil.copyfile(cached, dest)
            return dest

    # Movement length. Without an arc, keep the classic ~75 s movements.
    # WITH an arc, make the movement length ADAPTIVE so the score tracks
    # the emotional curve at the right resolution: a ~75 s movement is
    # too coarse for a 2-3 min doc (the whole hook->climax->resolution
    # arc fits in ~2 movements, which average to the same level and the
    # bed comes out flat — caught in full-doc QA). Aim for ~5 movements
    # (clamped 2..8), ~26-75 s each, so each movement sits over a
    # distinct emotional phase.
    if arc and dur > (_SECTION + _XF):
        target_n = max(2, min(8, int(round(dur / 38.0))))
        n = target_n
        section = max(24.0, (dur - _XF) / n)
    else:
        section = _SECTION
        n = max(1, math.ceil(dur / (section - _XF)))
    intens = _arc_section_intensity(arc, n, section)
    if n == 1:
        if arc:
            tr, tf, lvl = _intensity_to_params(intens[0], 0)
            lvl = _scale_swing(lvl, intensity)
        else:
            tr, tf, lvl = 1.0, 1.0, 1.0
        raw = dest.parent / "_mbsec_raw.wav"
        _render_section(round(f1 * tr, 2), round(f2 * tr, 2),
                        trem * tf, cutoff, dur, raw)
        g = _norm_gain(_measure_lufs(raw), lvl)
        run(["-i", str(raw), "-af",
             f"volume={g:.4f},alimiter=limit=0.5:level=0",
             str(dest)])
        raw.unlink(missing_ok=True)
    else:
        secs, gains = [], []
        for k in range(n):
            if arc:
                tr, tf, lvl = _intensity_to_params(intens[k], k)
                lvl = _scale_swing(lvl, intensity)
            else:
                tr, tf = _MOVES[k % len(_MOVES)]
                lvl = 1.0
            sp = dest.parent / f"_mbsec_{k}.wav"
            _render_section(round(f1 * tr, 2), round(f2 * tr, 2),
                            trem * tf, cutoff, section + _XF, sp)
            # loudness-normalize each movement, then apply its arc gain so
            # the swing is clean & predictable (immune to pitch resonance).
            gains.append(_norm_gain(_measure_lufs(sp), lvl))
            secs.append(sp)
        # pre-scale each movement, then chain-crossfade into one bed
        ins, parts = [], []
        for j, sp in enumerate(secs):
            ins += ["-i", str(sp)]
        for j in range(n):
            parts.append(f"[{j}:a]volume={gains[j]:.4f}[g{j}]")
        prev = "[g0]"
        for j in range(1, n):
            o = "[mix]" if j == n - 1 else f"[x{j}]"
            parts.append(
                f"{prev}[g{j}]acrossfade=d={_XF}:c1=tri:c2=tri{o}")
            prev = o
        fc = ";".join(parts) + (
            f";[mix]afade=t=in:st=0:d=2,"
            f"afade=t=out:st={max(0.0, n * section - 3.0):.2f}:d=3,"
            "alimiter=limit=0.7:level=0"
            "[out]"
        )
        run(ins + ["-filter_complex", fc, "-map", "[out]", str(dest)])
        for sp in secs:
            sp.unlink(missing_ok=True)

    if cache_dir is not None:
        import shutil

        shutil.copyfile(dest, cached)
    return dest


def _sfx_seed(t: float) -> float:
    """Deterministic 0..1 jitter from an event time, so repeated hits in
    the same video vary (length, tone) instead of sounding stamped."""
    return ((int(round(t * 1000)) * 2654435761) % 1000) / 1000.0


def build_sfx_bed(
    events: list,
    duration: float,
    dest: Path,
    *,
    cache_dir: Path | None = None,
) -> Path:
    """Motivated, layered sound design — placed ONLY where it earns it,
    never a stamp on every cut (PHASE 3.3). Three synthesized,
    royalty-free elements, each processed PER EVENT so no two are
    identical:

      * 'boom'   — a soft low impact when a high-energy scene LANDS; its
                   depth (sub cutoff), tail length and level all scale
                   with the moment's intensity, so a climax hits harder
                   than an ordinary high beat.
      * 'riser'  — a quiet tension build that ramps INTO the impact; its
                   length varies per hit (~1.0-1.7s) so it never feels
                   canned.
      * 'whoosh' — a soft filtered noise sweep used ONLY on a motivated
                   emotional DISSOLVE (a story exhale / time or place
                   shift), never on ordinary cuts. Low and short so it
                   reads as a transition, not an effect.

    ``events`` = list of ``(time_seconds, kind)`` or
    ``(time_seconds, kind, intensity[0..1])``. 'riser' is placed to END
    at its event time (the impact); 'whoosh' leads slightly into it.
    Everything else stays silent so the music bed and narration breathe.
    """
    dur = max(1.0, round(float(duration), 2))

    def _norm(ev):
        if len(ev) >= 3:
            t, k, q = ev[0], ev[1], ev[2]
        else:
            t, k, q = ev[0], ev[1], 0.7
        return round(float(t), 2), k, max(0.0, min(1.0, float(q)))

    evs = [_norm(e) for e in (events or [])]
    booms = sorted({(t, q) for t, k, q in evs if k == "boom" and 0.0 <= t < dur})
    risers = sorted({(t, q) for t, k, q in evs if k == "riser" and 0.0 <= t < dur})
    whooshes = sorted({(t, q) for t, k, q in evs
                       if k == "whoosh" and 0.0 <= t < dur})

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        key = scene_key("sfx3", int(round(dur)),
                        tuple(booms), tuple(risers), tuple(whooshes))
        cached = cache_dir / f"{key}.wav"
        if cached.exists():
            import shutil

            shutil.copyfile(cached, dest)
            return dest

    if not booms and not risers and not whooshes:
        run(["-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
             "-t", f"{dur:.2f}", str(dest)])
    else:
        nb, nr, nw = len(booms), len(risers), len(whooshes)
        # inputs: 0=boom sine · 1=riser noise · 2=whoosh noise · 3=base
        boom_src = "sine=frequency=52:sample_rate=44100:duration=2.2"
        riser_src = _seeded(
            "anoisesrc=color=brown:sample_rate=44100:amplitude=0.6:d=2.0")
        whoosh_src = _seeded(
            "anoisesrc=color=brown:sample_rate=44100:amplitude=0.7:d=1.0")

        parts = [f"[3:a]atrim=0:{dur:.2f},asetpts=PTS-STARTPTS[base]"]
        # split each raw source into one copy per event (asplit=N)
        if nb:
            parts.append("[0:a]asetpts=PTS-STARTPTS"
                         + (f",asplit={nb}" if nb > 1 else "")
                         + "".join(f"[bsrc{i}]" for i in range(nb)))
        if nr:
            parts.append("[1:a]asetpts=PTS-STARTPTS"
                         + (f",asplit={nr}" if nr > 1 else "")
                         + "".join(f"[rsrc{i}]" for i in range(nr)))
        if nw:
            parts.append("[2:a]asetpts=PTS-STARTPTS"
                         + (f",asplit={nw}" if nw > 1 else "")
                         + "".join(f"[wsrc{i}]" for i in range(nw)))

        mix = "[base]"
        for i, (t, q) in enumerate(booms):
            # deeper/longer/louder with intensity
            lp = int(round(200 - 78 * q))          # 200..122 Hz sub
            tail = 0.50 + 0.55 * q                  # 0.50..1.05s decay
            vol = 0.60 + 0.46 * q                   # 0.60..1.06
            ln = 0.18 + tail
            ms = int(t * 1000)
            parts.append(
                f"[bsrc{i}]atrim=0:{ln:.2f},asetpts=PTS-STARTPTS,"
                f"lowpass=f={lp},afade=t=in:d=0.02,"
                f"afade=t=out:st=0.12:d={tail:.2f},volume={vol:.3f},"
                f"adelay={ms}|{ms}[bd{i}]")
            mix += f"[bd{i}]"
        for i, (t, q) in enumerate(risers):
            seed = _sfx_seed(t)
            ln = 1.0 + 0.7 * seed                   # 1.0..1.7s, varied
            vol = 0.40 + 0.18 * q                   # a touch louder if hot
            start = max(0.0, t - ln)
            ms = int(start * 1000)
            fo = max(0.05, ln - 0.12)
            parts.append(
                f"[rsrc{i}]atrim=0:{ln:.2f},asetpts=PTS-STARTPTS,"
                "highpass=f=240,lowpass=f=3800,"
                f"volume='min(1\\,t/{ln:.2f})':eval=frame,"
                f"afade=t=out:st={fo:.2f}:d=0.12,volume={vol:.3f},"
                f"adelay={ms}|{ms}[rd{i}]")
            mix += f"[rd{i}]"
        for i, (t, q) in enumerate(whooshes):
            seed = _sfx_seed(t + 0.5)
            ln = 0.42 + 0.20 * seed                 # 0.42..0.62s
            # band varies a little per hit so repeats don't sound stamped
            hp = int(round(380 + 260 * seed))       # 380..640
            lpf = int(round(3000 + 1400 * seed))    # 3000..4400
            # audible cinematic reveal whoosh (was 0.20-0.30, too quiet to
            # FEEL on a text reveal) — still restrained, not trailer-loud.
            vol = 0.34 + 0.22 * q                   # 0.34..0.56
            start = max(0.0, t - 0.22)              # lead into the cut
            ms = int(start * 1000)
            fo = max(0.05, ln - 0.14)
            parts.append(
                f"[wsrc{i}]atrim=0:{ln:.2f},asetpts=PTS-STARTPTS,"
                f"highpass=f={hp},lowpass=f={lpf},"
                f"afade=t=in:d=0.10,afade=t=out:st={fo:.2f}:d=0.14,"
                f"volume={vol:.3f},adelay={ms}|{ms}[wd{i}]")
            mix += f"[wd{i}]"

        k = mix.count("[")
        parts.append(
            f"{mix}amix=inputs={k}:duration=first:normalize=0,"
            "aformat=sample_fmts=s16:sample_rates=44100:"
            "channel_layouts=stereo[out]")
        run([
            "-f", "lavfi", "-i", boom_src,
            "-f", "lavfi", "-i", riser_src,
            "-f", "lavfi", "-i", whoosh_src,
            "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
            "-filter_complex", ";".join(parts),
            "-map", "[out]", "-t", f"{dur:.2f}", str(dest),
        ])

    if cache_dir is not None:
        import shutil

        shutil.copyfile(dest, cached)
    return dest


def build_typewriter_sfx(
    events: list,
    duration: float,
    dest: Path,
    *,
    cache_dir: Path | None = None,
) -> Path:
    """Synced TYPEWRITER click track. ``events`` = list of
    ``(char_times[list], enter_time)``. A short mechanical key click is
    placed at EVERY character-appearance time (so the audio syncs exactly
    to the on-screen letters, never a disconnected loop) + a softer ENTER
    accent when the line finishes. Restrained level — tactile under the
    voice, never a hacker-movie clatter."""
    dur = max(1.0, round(float(duration), 2))
    # CONTEXT-AWARE click voice — each typing event carries a style so a
    # historical doc clatters like a warm mechanical TYPEWRITER, a military
    # terminal ticks like a crisp DIGITAL keyboard, and classified intel
    # uses a restrained SECURE electronic tick. (hp, lp, length, volume)
    _STYLE = {
        "mech":    (420, 3000, 0.075, 0.50),   # warm, older machine
        "digital": (1150, 6200, 0.055, 0.42),  # crisp, sharp UI
        "secure":  (1500, 4400, 0.050, 0.34),  # subtle, professional
    }
    clicks: list = []   # (time, style)
    for ev in (events or []):
        sty = ev[2] if len(ev) > 2 and ev[2] in _STYLE else "digital"
        for t in (ev[0] or []):
            if 0.0 <= float(t) < dur:
                clicks.append((round(float(t), 2), sty))
    clicks = sorted(set(clicks))
    enters = sorted({round(float(ev[1]), 2) for ev in (events or [])
                     if len(ev) > 1 and 0.0 <= float(ev[1]) < dur})
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        key = scene_key("twsfx2", int(round(dur)), tuple(clicks),
                        tuple(enters))
        cached = cache_dir / f"{key}.wav"
        if cached.exists():
            import shutil
            shutil.copyfile(cached, dest)
            return dest
    if not clicks:
        run(["-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
             "-t", f"{dur:.2f}", str(dest)])
        return dest
    nC, nE = len(clicks), len(enters)
    click_src = _seeded("anoisesrc=color=white:sample_rate=44100:amplitude=0.55:d=0.12")
    enter_src = "sine=frequency=190:sample_rate=44100:duration=0.20"
    parts = [f"[2:a]atrim=0:{dur:.2f},asetpts=PTS-STARTPTS[base]"]
    parts.append("[0:a]asetpts=PTS-STARTPTS"
                 + (f",asplit={nC}" if nC > 1 else "")
                 + "".join(f"[c{i}]" for i in range(nC)))
    if nE:
        parts.append("[1:a]asetpts=PTS-STARTPTS"
                     + (f",asplit={nE}" if nE > 1 else "")
                     + "".join(f"[e{i}]" for i in range(nE)))
    mix = "[base]"
    for i, (t, sty) in enumerate(clicks):
        ms = int(t * 1000)
        hp, lp, ln, vol = _STYLE[sty]
        fo_ = max(0.012, ln - 0.025)
        parts.append(
            f"[c{i}]atrim=0:{ln:.3f},asetpts=PTS-STARTPTS,"
            f"highpass=f={hp},lowpass=f={lp},afade=t=in:d=0.004,"
            f"afade=t=out:st={fo_:.3f}:d=0.02,volume={vol:.2f},"
            f"adelay={ms}|{ms}[cd{i}]")
        mix += f"[cd{i}]"
    for i, t in enumerate(enters):
        ms = int(t * 1000)
        parts.append(
            f"[e{i}]atrim=0:0.20,asetpts=PTS-STARTPTS,"
            "lowpass=f=2600,afade=t=in:d=0.01,"
            f"afade=t=out:st=0.06:d=0.13,volume=0.34,"
            f"adelay={ms}|{ms}[ed{i}]")
        mix += f"[ed{i}]"
    k = mix.count("[")
    parts.append(
        f"{mix}amix=inputs={k}:duration=first:normalize=0,"
        "aformat=sample_fmts=s16:sample_rates=44100:"
        "channel_layouts=stereo[out]")
    run([
        "-f", "lavfi", "-i", click_src,
        "-f", "lavfi", "-i", enter_src,
        "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
        "-filter_complex", ";".join(parts),
        "-map", "[out]", "-t", f"{dur:.2f}", str(dest),
    ])
    if cache_dir is not None:
        import shutil
        shutil.copyfile(dest, cached)
    return dest


def build_number_sfx(
    events: list,
    duration: float,
    dest: Path,
    *,
    cache_dir: Path | None = None,
) -> Path:
    """Cinematic COUNT-UP sound. ``events`` = list of (roll_start, land).
    While a number rapidly counts up, soft rapid 'counter' clicks tick
    along; the instant it LANDS a crisp mid/high 'stat stinger' hits
    (audible on any speaker — never the old inaudible sub-bass thud).
    Sparse (only figure cards) so it punctuates without ever becoming
    the annoying repeating SFX the user disliked."""
    dur = max(1.0, round(float(duration), 2))
    ev = [(float(a), float(b)) for a, b in (events or [])
          if 0.0 <= float(b) < dur]
    impacts = sorted({round(b, 2) for _a, b in ev})
    clicks: list[float] = []
    for a, b in ev:
        if b - a > 0.07:                       # there was a real roll
            tt = a
            while tt < b - 0.03:
                clicks.append(round(tt, 3))
                tt += 0.062
    clicks = sorted({c for c in clicks if 0.0 <= c < dur})

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        key = scene_key("numsfx2", int(round(dur)),
                        tuple(impacts), tuple(clicks))
        cached = cache_dir / f"{key}.wav"
        if cached.exists():
            import shutil

            shutil.copyfile(cached, dest)
            return dest

    if not impacts:
        run(["-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
             "-t", f"{dur:.2f}", str(dest)])
    else:
        # 0 body 220 · 1 blip 1200 · 2 transient 2600 (LANDING stinger)
        # 3 counter click ~2300 (rolling) · 4 silent base (length)
        ni, nc = len(impacts), len(clicks)

        def _split(src, chain, n, lbl):
            return (f"[{src}:a]{chain}"
                    + (f",asplit={n}" if n > 1 else "")
                    + "".join(f"[{lbl}{i}]" for i in range(n)))

        parts = [f"[4:a]atrim=0:{dur:.2f},asetpts=PTS-STARTPTS[base]"]
        parts.append(_split(
            0, "atrim=0:0.26,asetpts=PTS-STARTPTS,afade=t=in:d=0.006,"
            "afade=t=out:st=0.06:d=0.20,lowpass=f=900,volume=1.0",
            ni, "a"))
        parts.append(_split(
            1, "atrim=0:0.20,asetpts=PTS-STARTPTS,afade=t=in:d=0.004,"
            "afade=t=out:st=0.05:d=0.15,volume=1.05", ni, "b"))
        parts.append(_split(
            2, "atrim=0:0.09,asetpts=PTS-STARTPTS,afade=t=in:d=0.002,"
            "afade=t=out:st=0.018:d=0.07,volume=0.95", ni, "c"))
        if nc:
            parts.append(_split(
                3, "atrim=0:0.035,asetpts=PTS-STARTPTS,"
                "afade=t=in:d=0.002,afade=t=out:st=0.012:d=0.022,"
                "volume=0.42", nc, "k"))
        mix = "[base]"
        for i, t in enumerate(impacts):
            ms = int(t * 1000)
            parts.append(f"[a{i}]adelay={ms}|{ms}[ad{i}]")
            parts.append(f"[b{i}]adelay={ms}|{ms}[bd{i}]")
            parts.append(f"[c{i}]adelay={ms}|{ms}[cd{i}]")
            mix += f"[ad{i}][bd{i}][cd{i}]"
        for i, t in enumerate(clicks):
            ms = int(t * 1000)
            parts.append(f"[k{i}]adelay={ms}|{ms}[kd{i}]")
            mix += f"[kd{i}]"
        k = mix.count("[")
        parts.append(
            f"{mix}amix=inputs={k}:duration=first:normalize=0,"
            "aformat=sample_fmts=s16:sample_rates=44100:"
            "channel_layouts=stereo[out]"
        )
        run([
            "-f", "lavfi", "-i",
            "sine=frequency=220:sample_rate=44100:duration=0.5",
            "-f", "lavfi", "-i",
            "sine=frequency=1200:sample_rate=44100:duration=0.5",
            "-f", "lavfi", "-i",
            "sine=frequency=2600:sample_rate=44100:duration=0.5",
            "-f", "lavfi", "-i",
            "sine=frequency=2300:sample_rate=44100:duration=0.5",
            "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
            "-filter_complex", ";".join(parts),
            "-map", "[out]", "-t", f"{dur:.2f}", str(dest),
        ])

    if cache_dir is not None:
        import shutil

        shutil.copyfile(dest, cached)
    return dest


def build_archival_bed(
    windows: list,
    duration: float,
    dest: Path,
    *,
    cache_dir: Path | None = None,
) -> Path:
    """Faint analog ROOM TONE only over 'archival' scenes — soft tape
    hiss + a low projector/transport hum, edges gently faded so it never
    pops. Very low by design: it sits UNDER everything and just makes
    recovered footage feel real (NOT a loud retro effect)."""
    dur = max(1.0, round(float(duration), 2))
    wins = [(max(0.0, float(s)), min(dur, float(e)))
            for s, e in (windows or []) if float(e) - float(s) > 0.4]
    wins = [(s, e) for s, e in wins if e > s]

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        key = scene_key("archbed", int(round(dur)),
                        tuple((round(s, 1), round(e, 1)) for s, e in wins))
        cached = cache_dir / f"{key}.wav"
        if cached.exists():
            import shutil

            shutil.copyfile(cached, dest)
            return dest

    if not wins:
        run(["-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
             "-t", f"{dur:.2f}", str(dest)])
    else:
        # 0 = hiss (band-limited noise) · 1 = low hum · 2 = silent base
        parts = [f"[2:a]atrim=0:{dur:.2f},asetpts=PTS-STARTPTS[base]"]
        mix = "[base]"
        for i, (s, e) in enumerate(wins):
            ln = max(0.5, e - s)
            ms = int(s * 1000)
            fo = max(0.05, ln - 0.45)
            parts.append(
                f"[0:a]atrim=0:{ln:.2f},asetpts=PTS-STARTPTS,"
                "highpass=f=1000,lowpass=f=6500,"
                f"afade=t=in:d=0.45,afade=t=out:st={fo:.2f}:d=0.45,"
                f"volume=0.060,adelay={ms}|{ms}[h{i}]")
            parts.append(
                f"[1:a]atrim=0:{ln:.2f},asetpts=PTS-STARTPTS,"
                f"lowpass=f=160,afade=t=in:d=0.5,"
                f"afade=t=out:st={fo:.2f}:d=0.5,"
                f"volume=0.045,adelay={ms}|{ms}[u{i}]")
            mix += f"[h{i}][u{i}]"
        k = mix.count("[")
        parts.append(
            f"{mix}amix=inputs={k}:duration=first:normalize=0,"
            "aformat=sample_fmts=s16:sample_rates=44100:"
            "channel_layouts=stereo[out]")
        run([
            "-f", "lavfi", "-i",
            _seeded("anoisesrc=color=pink:sample_rate=44100:amplitude=0.5"),
            "-f", "lavfi", "-i",
            "sine=frequency=58:sample_rate=44100",
            "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
            "-filter_complex", ";".join(parts),
            "-map", "[out]", "-t", f"{dur:.2f}", str(dest),
        ])

    if cache_dir is not None:
        import shutil

        shutil.copyfile(dest, cached)
    return dest


# --------------------------------------------------------------------------- #
# ATMOSPHERE ENGINE (Human-Editor #8) — subconscious environmental immersion
# --------------------------------------------------------------------------- #
# Per-environment ambience textures, ALL synthesized (royalty-free) and kept
# very low so they sit UNDER the narration — the viewer never "hears" them,
# they just feel like they're inside a world. Built from noise + tones +
# filters; noise never loops audibly. A scene with no clear environment gets
# NO atmosphere (silence is respected). Value = (lavfi inputs, core filter
# producing [core], base volume).
_ATMOS: dict[str, tuple[tuple[str, ...], str, float]] = {
    # desert / sand — dry wind air, gusting slowly, no rumble
    "desert": (("anoisesrc=color=pink:sample_rate=44100:amplitude=0.5",),
               "[0:a]highpass=f=190,lowpass=f=1500,"
               "tremolo=f=0.13:d=0.7[core]", 0.060),
    # cave / underground — deep low rumble + cavern echo (space)
    "cave": (("sine=frequency=46:sample_rate=44100",
              "anoisesrc=color=brown:sample_rate=44100:amplitude=0.4"),
             "[0:a]volume=0.5[s];[1:a]lowpass=f=190,volume=0.5[n];"
             "[s][n]amix=inputs=2:normalize=0,"
             "aecho=0.9:0.85:550:0.4[core]", 0.060),
    # water / ocean — soft broadband wash, slow swell like distant surf
    "water": (("anoisesrc=color=pink:sample_rate=44100:amplitude=0.5",),
              "[0:a]highpass=f=300,lowpass=f=2600,"
              "tremolo=f=0.2:d=0.6[core]", 0.052),
    # city / urban — distant low traffic rumble
    "urban": (("anoisesrc=color=brown:sample_rate=44100:amplitude=0.5",),
              "[0:a]lowpass=f=520,tremolo=f=0.1:d=0.3[core]", 0.050),
    # room / lab / investigation — faint mains hum + soft hiss
    "room": (("sine=frequency=60:sample_rate=44100",
              "anoisesrc=color=pink:sample_rate=44100:amplitude=0.4"),
             "[0:a]volume=0.4[h];[1:a]highpass=f=1800,lowpass=f=6500,"
             "volume=0.5[s];[h][s]amix=inputs=2:normalize=0[core]", 0.042),
    # tension / military / conspiracy — low detuned tonal pressure
    "tension": (("sine=frequency=68:sample_rate=44100",
                 "sine=frequency=102:sample_rate=44100"),
                "[0:a]volume=0.5[a];[1:a]volume=0.4[b];"
                "[a][b]amix=inputs=2:normalize=0,lowpass=f=420,"
                "tremolo=f=0.08:d=0.5[core]", 0.055),
    # space / cosmic — airy high shimmer, very faint
    "space": (("anoisesrc=color=pink:sample_rate=44100:amplitude=0.5",),
              "[0:a]highpass=f=1500,vibrato=f=0.12:d=0.6[core]", 0.040),
    # nature / forest — soft mid air texture
    "nature": (("anoisesrc=color=pink:sample_rate=44100:amplitude=0.45",),
               "[0:a]highpass=f=350,lowpass=f=3200,"
               "tremolo=f=0.12:d=0.4[core]", 0.045),
}


def build_atmosphere_bed(
    windows: list,
    duration: float,
    dest: Path,
    *,
    cache_dir: Path | None = None,
) -> Path:
    """Subtle, evolving environmental ambience (Human-Editor #8).

    ``windows`` = list of ``(start, end, kind, intensity)`` — each a span
    that should carry a faint world texture (kind in ``_ATMOS``;
    intensity 0..1 lifts the level a touch on tense beats). Everything
    between windows stays silent so the doc still breathes. Each window
    is rendered as its own faded, delayed leg and mixed under a silent
    base — sits far below narration (almost subconscious)."""
    dur = max(1.0, round(float(duration), 2))
    wins = []
    for w in (windows or []):
        s, e = max(0.0, float(w[0])), min(dur, float(w[1]))
        kind = w[2] if len(w) > 2 else ""
        q = max(0.0, min(1.0, float(w[3]))) if len(w) > 3 else 0.5
        # R1 cinematic weight (0 = legacy/off): a per-window sustained low-end
        # level threaded from assemble.py (flag-gated there). Carried in the
        # cache key so weighted/unweighted beds never collide.
        wt = max(0.0, min(2.0, float(w[4]))) if len(w) > 4 else 0.0
        if e - s >= 1.0 and kind in _ATMOS:
            wins.append((round(s, 2), round(e, 2), kind, round(q, 2), round(wt, 3)))

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        key = scene_key("atmos1", int(round(dur)), tuple(wins))
        cached = cache_dir / f"{key}.wav"
        if cached.exists():
            import shutil

            shutil.copyfile(cached, dest)
            return dest

    if not wins:
        run(["-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
             "-t", f"{dur:.2f}", str(dest)])
        if cache_dir is not None:
            import shutil
            shutil.copyfile(dest, cached)
        return dest

    legs: list[Path] = []
    for i, w in enumerate(wins):
        s, e, kind, q = w[0], w[1], w[2], w[3]
        wt = w[4] if len(w) > 4 else 0.0
        ln = round(e - s, 2)
        srcs, core, base = _ATMOS[kind]
        vol = base * (0.7 + 0.5 * q)               # subtle intensity lift
        fo = max(0.3, ln - 1.0)
        inputs: list[str] = []
        for src in srcs:
            inputs += ["-f", "lavfi", "-i", _seeded(src)]
        if wt > 0.0:
            # R1 — sustained cinematic LOW-END weight UNDER the texture: a
            # detuned sub sine pair (42/63 Hz) lowpassed to stay SUB (no mud,
            # no overlap with the 300-3400 Hz voice band), gently undulating,
            # faded with the window (no clicks). Gain scales with the per-scene
            # weight + is hard-capped for limiter safety. The texture keeps its
            # own tiny `vol`; the sub does NOT inherit it.
            ns = len(srcs)
            inputs += ["-f", "lavfi", "-i", "sine=frequency=42:sample_rate=44100",
                       "-f", "lavfi", "-i", "sine=frequency=63:sample_rate=44100"]
            wgain = min(0.15, 0.10 * float(wt))
            fc = (
                core + ";"
                f"[core]volume={vol:.4f}[texv];"
                f"[{ns}:a]volume=0.6[wa];[{ns + 1}:a]volume=0.28[wb];"
                "[wa][wb]amix=inputs=2:normalize=0,lowpass=f=90,"
                f"tremolo=f=0.13:d=0.4,volume={wgain:.4f}[wsub];"
                "[texv][wsub]amix=inputs=2:normalize=0[body];[body]"
                f"afade=t=in:d=0.9,afade=t=out:st={fo:.2f}:d=1.0,"
                "aformat=sample_fmts=s16:sample_rates=44100:"
                "channel_layouts=stereo[out]"
            )
        else:
            fc = (
                core + ";[core]"
                f"afade=t=in:d=0.9,afade=t=out:st={fo:.2f}:d=1.0,"
                f"volume={vol:.4f},"
                "aformat=sample_fmts=s16:sample_rates=44100:"
                "channel_layouts=stereo[out]"
            )
        leg = dest.parent / f"_atmos_{i}.wav"
        run(inputs + ["-filter_complex", fc, "-map", "[out]",
                      "-t", f"{ln:.2f}", str(leg)])
        legs.append(leg)

    parts = [f"[{len(legs)}:a]atrim=0:{dur:.2f},asetpts=PTS-STARTPTS[base]"]
    mix = "[base]"
    for i, w in enumerate(wins):
        ms = int(w[0] * 1000)
        parts.append(f"[{i}:a]adelay={ms}|{ms}[d{i}]")
        mix += f"[d{i}]"
    k = mix.count("[")
    parts.append(
        f"{mix}amix=inputs={k}:duration=first:normalize=0,"
        "aformat=sample_fmts=s16:sample_rates=44100:"
        "channel_layouts=stereo[out]")
    ins: list[str] = []
    for leg in legs:
        ins += ["-i", str(leg)]
    ins += ["-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo"]
    run(ins + ["-filter_complex", ";".join(parts),
               "-map", "[out]", "-t", f"{dur:.2f}", str(dest)])
    for leg in legs:
        leg.unlink(missing_ok=True)

    if cache_dir is not None:
        import shutil

        shutil.copyfile(dest, cached)
    return dest

