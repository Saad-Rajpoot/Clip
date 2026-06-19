"""Reusable cinematic motion-graphics primitives — the shared MOTION
LANGUAGE every template inherits.

WHY THIS EXISTS
    Before this module each card hand-rolled its own fade durations and
    the odd slide, all of them LINEAR. That reads as "overlay animation",
    not "a motion-graphics editor worked on this". This module gives the
    whole engine ONE cinematic vocabulary: a small set of eases, timing
    curves, intensity profiles and a stagger scheduler, so motion is
    consistent, restrained and physically believable everywhere.

HOW IT FITS THE ARCHITECTURE
    The documentary renderer is FFmpeg-filtergraph based, so the core of
    this module EMITS FFmpeg expression strings (for `overlay` x/y and
    `zoompan`/`scale`, evaluated per-frame with `eval=frame`). It is the
    proven GSAP / Remotion easing math — `power*.out`, `back.out`
    overshoot, `expo.out`, in-out cubic — recreated as FFmpeg `expr`
    rather than reinvented. Python easing callables are also provided for
    PIL per-frame card rendering (kinetic type, count-up numbers, etc.).

DESIGN REFERENCES (adapted, not copied)
    • GSAP eases — power1..4.out, back.out(overshoot), expo.out.
    • Remotion — interpolate() value-mapping + frame-delay stagger.
    • Classic animation principles — slow-in/slow-out, overshoot+settle,
      anticipation, follow-through — applied with RESTRAINT (documentary,
      not TikTok): subtle distances, short overshoot, motivated timing.

STYLE CONTRACT
    Motion must guide attention, support emotion, clarify and immerse —
    never feel flashy. So: no bounce spam, modest overshoot only on the
    'dramatic' profile, gentle distances, smooth deceleration as the
    default (reveals decelerate into place, like a camera settling).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

FPS = 30

# --------------------------------------------------------------------------- #
# 1) EASING — the timing curves.
#
# Two parallel implementations of the SAME curves so the look is identical
# whether motion is done in FFmpeg (position/scale of overlays) or in PIL
# (per-frame redraw for kinetic type / counters):
#   • EASE_FF[name](p)  -> an FFmpeg expr string, given a progress expr `p`
#   • EASE_PY[name](x)  -> a Python float, given progress x in [0,1]
#
# Names mirror GSAP so they're familiar:
#   out_cubic  ~ power2/3.out  (the UI sweet spot — default)
#   out_quint  ~ power4.out    (dramatic deceleration)
#   out_expo   ~ expo.out      (very fast in, long glide to rest)
#   out_back   ~ back.out      (overshoot + settle — use sparingly)
#   in_out_cubic               (smooth symmetric — for moves, not reveals)
# --------------------------------------------------------------------------- #

# back-ease overshoot constant (GSAP default 1.70158). Kept modest so the
# overshoot is a gentle "settle", never a cartoon bounce.
_BACK_C1 = 1.70158
_BACK_C3 = _BACK_C1 + 1.0


def _wrap(p: str) -> str:
    return f"({p})"


EASE_FF = {
    "linear": lambda p: _wrap(p),
    "out_cubic": lambda p: f"(1-pow(1-{_wrap(p)},3))",
    "out_quint": lambda p: f"(1-pow(1-{_wrap(p)},5))",
    "out_expo": lambda p: f"(1-pow(2,-10*{_wrap(p)}))",
    "out_back": (
        lambda p: f"(1+{_BACK_C3:.5f}*pow({_wrap(p)}-1,3)"
                  f"+{_BACK_C1:.5f}*pow({_wrap(p)}-1,2))"
    ),
    "in_out_cubic": (
        lambda p: f"(if(lt({_wrap(p)},0.5),4*pow({_wrap(p)},3),"
                  f"1-pow(-2*{_wrap(p)}+2,3)/2))"
    ),
}


def _py_out_back(x: float) -> float:
    return 1.0 + _BACK_C3 * (x - 1) ** 3 + _BACK_C1 * (x - 1) ** 2


EASE_PY = {
    "linear": lambda x: x,
    "out_cubic": lambda x: 1 - (1 - x) ** 3,
    "out_quint": lambda x: 1 - (1 - x) ** 5,
    "out_expo": lambda x: 1.0 if x >= 1 else 1 - 2 ** (-10 * x),
    "out_back": _py_out_back,
    "in_out_cubic": (
        lambda x: 4 * x ** 3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2
    ),
}


# --------------------------------------------------------------------------- #
# 2) MOTION PROFILES — intensity presets.
#
# A profile is the restraint dial. 'subtle' for calm/expository beats,
# 'standard' for most, 'dramatic' for a climax/reveal (and the ONLY profile
# that uses overshoot). Style Modes can pick a profile so an Epic doc moves
# slower & grander while True Crime moves tighter.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MotionProfile:
    name: str
    slide_dur: float      # seconds for a slide-in to settle
    slide_dist: int       # px the element travels in
    fade_in: float        # seconds
    fade_out: float       # seconds
    stagger: float        # seconds between staggered siblings
    ease: str             # default ease for slides/reveals
    zoom_rate: float      # camera push/drift strength multiplier


PROFILES = {
    "subtle": MotionProfile("subtle", 0.60, 64, 0.55, 0.45, 0.16,
                            "out_cubic", 0.7),
    "standard": MotionProfile("standard", 0.52, 104, 0.45, 0.45, 0.22,
                              "out_cubic", 1.0),
    "dramatic": MotionProfile("dramatic", 0.66, 150, 0.55, 0.50, 0.30,
                              "out_back", 1.25),
}


def profile(name: str = "standard") -> MotionProfile:
    return PROFILES.get((name or "standard").lower(), PROFILES["standard"])


# --------------------------------------------------------------------------- #
# 3) INTERPOLATE — the core value-mapper (Remotion's interpolate()).
#
# Returns an FFmpeg expr that eases a scalar from `a` to `b` across the
# window [st, st+dur] and HOLDS `b` afterwards (and `a` before). Everything
# else (slide, scale, count-up) is built on this.
# --------------------------------------------------------------------------- #
def progress_expr(st: float, dur: float) -> str:
    """Clamped 0..1 progress over [st, st+dur] as an FFmpeg time expr."""
    dur = max(1e-3, dur)
    return f"clip((t-{st:.4f})/{dur:.4f},0,1)"


def interp_ff(a: float, b: float, st: float, dur: float,
              ease: str = "out_cubic") -> str:
    """FFmpeg expr: value eased a->b over [st,st+dur], holding b after."""
    p = progress_expr(st, dur)
    e = EASE_FF.get(ease, EASE_FF["out_cubic"])(p)
    return f"({a:.3f}+({(b - a):.3f})*{e})"


def interp_py(a: float, b: float, x: float, ease: str = "out_cubic") -> float:
    """Python: value eased a->b at progress x in [0,1]."""
    x = max(0.0, min(1.0, x))
    e = EASE_PY.get(ease, EASE_PY["out_cubic"])(x)
    return a + (b - a) * e


# --------------------------------------------------------------------------- #
# 4) SLIDE ENGINE — directional slide + overshoot + settle.
#
# Returns the FFmpeg `overlay` coordinate expression for an element that
# slides in to `base` from `direction`, decelerating into place. Use with
# overlay x=...:y=...:eval=frame. The companion axis stays constant.
# 'dramatic' profile's out_back gives the gentle overshoot+settle.
# --------------------------------------------------------------------------- #
_DIR_SIGN = {"left": -1, "right": +1, "up": -1, "down": +1}


def slide_in(base: int, direction: str, st: float,
             prof: MotionProfile, *, dist: int | None = None,
             ease: str | None = None) -> str:
    """Eased slide-in coordinate expr ending at `base`.

    direction: where the element comes FROM ('left' enters from left, etc.)
    """
    d = int(dist if dist is not None else prof.slide_dist)
    sign = _DIR_SIGN.get(direction, -1)
    start = base + sign * d                      # off-position it enters from
    return interp_ff(start, base, st, prof.slide_dur,
                     ease or prof.ease)


def slide_out(base: int, direction: str, st: float,
              prof: MotionProfile, *, dist: int | None = None) -> str:
    """Eased slide-OUT (exit toward `direction`), starting at `base`."""
    d = int(dist if dist is not None else prof.slide_dist)
    sign = _DIR_SIGN.get(direction, +1)
    end = base + sign * d
    # exits accelerate away -> in-cubic feel via reversed out on a short dur
    return interp_ff(base, end, st, prof.fade_out, "in_out_cubic")


# --------------------------------------------------------------------------- #
# 5) STAGGER — cascade siblings (Remotion frame-delay stagger).
# --------------------------------------------------------------------------- #
def stagger(n: int, base_st: float, prof: MotionProfile,
            *, step: float | None = None) -> list[float]:
    """Per-item start times so a list/group reveals as a cascade, not a
    wall. Restrained: the step is small so the build feels like one move."""
    s = float(step if step is not None else prof.stagger)
    return [base_st + i * s for i in range(max(0, n))]


# --------------------------------------------------------------------------- #
# 6) CINEMATIC FADE — eased alpha timing helper.
#
# FFmpeg's `fade` alpha ramp is linear; for layered reveals we want soft,
# slightly delayed opacity. This returns the (st, dur) pair for fade=in and
# the start for fade=out given a window + profile, so all cards share the
# same breathing in/out timing. (Position easing carries the "physical"
# feel; alpha stays a clean soft dissolve.)
# --------------------------------------------------------------------------- #
def fade_window(start: float, end: float, prof: MotionProfile,
                *, lead: float = 0.0) -> tuple[float, float, float, float]:
    """Returns (fin_st, fin_d, fout_st, fout_d) clamped to [start,end]."""
    fin_st = start + lead
    fin_d = min(prof.fade_in, max(0.05, (end - fin_st) * 0.5))
    fout_d = min(prof.fade_out, max(0.05, (end - fin_st) * 0.5))
    fout_st = max(fin_st + fin_d, end - fout_d)
    return fin_st, fin_d, fout_st, fout_d


def zoompan_expr(direction: str, st: float, dur: float,
                 prof: MotionProfile, *, amount: float = 0.06) -> str:
    """Eased scale expr for a slow camera push/pull on a still (Ken Burns,
    Remotion-style interpolated scale). Returns a multiplier expr for use
    with scale=iw*EXPR. amount scaled by the profile's zoom_rate."""
    amt = amount * prof.zoom_rate
    if direction == "out":
        return interp_ff(1.0 + amt, 1.0, st, dur, "out_cubic")
    return interp_ff(1.0, 1.0 + amt, st, dur, "out_cubic")
