# RC5.1 — GLOBAL OVERLAY-RESTRAINT regression.
#
# Proves the bounded overlay-restraint policy in vidlore.assemble:
#   1) every strength knob CLAMPS to its safe maximum (a knob pushed to 1.0 —
#      or beyond / garbage — can never exceed the per-layer ceiling, so footage
#      can't be made muddier than the cap), and
#   2) the automatic CLARITY GATE reduces / de-stacks the heaviest layers
#      (vignette / darken / texture / grain) when the projected stack is
#      excessive, while always preserving a luma-floor colour grade.
#
# Pure parameter-level — no ffmpeg, no pixels. Sets the env knobs to their
# MAX then rebuilds the policy object so the clamp is exercised directly.
#
# Run:  PYTHONPATH=. .venv/bin/python tests/test_rc51_overlay_restraint.py
import os

import vidlore.assemble as A

_passed = 0


def check(name, cond):
    global _passed
    assert cond, "FAIL: " + name
    print("  ok ", name)
    _passed += 1


# ---------------------------------------------------------------------------- #
# 1) STRENGTH CLAMPS — knobs pushed to / past 1.0 cap at the SAFE MAXIMUMS.
# ---------------------------------------------------------------------------- #
# Push every knob WAY past 1.0 (and master past 1.0) — the clamp must hold.
for k in ("VIDLORE_OVERLAY_STRENGTH", "VIDLORE_GRAIN_STRENGTH",
          "VIDLORE_VIGNETTE_STRENGTH", "VIDLORE_TEXTURE_STRENGTH",
          "VIDLORE_DARKEN_STRENGTH"):
    os.environ[k] = "9.9"                         # absurd over-push
maxed = A._OverlayRestraint()

# knobs themselves clamp to [0,1]
check("master knob clamps to 1.0", maxed.master == 1.0)
check("grain knob clamps to 1.0", maxed.grain == 1.0)
check("vignette knob clamps to 1.0", maxed.vignette == 1.0)
check("texture knob clamps to 1.0", maxed.texture == 1.0)
check("darken knob clamps to 1.0", maxed.darken == 1.0)

# at the cap, the produced film-grain equals the SAFE MAX (never above).
check("footage grain caps at GRAIN_MAX",
      maxed.grain_amount() == maxed.GRAIN_MAX)
check("footage grain never exceeds GRAIN_MAX",
      maxed.grain_amount() <= maxed.GRAIN_MAX)
check("archival grain caps at GRAIN_ARCH_MAX",
      maxed.arch_grain_amount() == maxed.GRAIN_ARCH_MAX)
check("final grain caps at GRAIN_FINAL_MAX",
      maxed.final_grain_amount() == maxed.GRAIN_FINAL_MAX)

# vignette divisor at the cap == the HARDEST allowed angle (smallest divisor).
# Even maxed, it must NOT go below the hard cap (== never crush harder than cap).
check("footage vignette caps at hardest divisor (>= VIGN_HARD_ANGLE)",
      maxed.vignette_angle() >= maxed.VIGN_HARD_ANGLE - 1e-9)
check("footage vignette equals hard cap when maxed",
      abs(maxed.vignette_angle() - maxed.VIGN_HARD_ANGLE) < 1e-6)
check("archival vignette caps at hardest divisor (>= VIGN_ARCH_HARD)",
      maxed.arch_vignette_angle() >= maxed.VIGN_ARCH_HARD - 1e-9)

# texture multiplier caps at TEXTURE_MAX.
check("texture scale caps at TEXTURE_MAX",
      abs(maxed.texture_scale() - maxed.TEXTURE_MAX) < 1e-6)

# darken: even with darken pushed to max, a crushing proposed gamma is lifted
# to the luma FLOOR — footage can never be darkened below it.
check("darken_gamma never below floor (crushing input)",
      maxed.darken_gamma(0.80) >= maxed.DARKEN_FLOOR_GAMMA - 1e-9)
check("darken_gamma exactly floor for a sub-floor proposal",
      abs(maxed.darken_gamma(0.80) - maxed.DARKEN_FLOOR_GAMMA) < 1e-6)

# ---------------------------------------------------------------------------- #
# 2) The SAFE MAXIMUMS are genuinely restrained vs the legacy hardcoded values
#    (so 'pushed' really is <= what the engine used to bake by default).
# ---------------------------------------------------------------------------- #
check("GRAIN_MAX (9) <= legacy archival grain (16)", maxed.GRAIN_MAX <= 16)
check("GRAIN_ARCH_MAX (12) < legacy 16", maxed.GRAIN_ARCH_MAX < 16)
check("archival vignette hard cap softer than legacy PI/3.8",
      maxed.VIGN_ARCH_HARD > 3.8)        # larger divisor = softer = less crush
check("darken floor is a real luma floor (>1.0, no shadow crush)",
      maxed.DARKEN_FLOOR_GAMMA > 1.0)

# ---------------------------------------------------------------------------- #
# 3) DEFAULT knobs (no env) sit at-or-below the caps and are conservative.
# ---------------------------------------------------------------------------- #
for k in ("VIDLORE_OVERLAY_STRENGTH", "VIDLORE_GRAIN_STRENGTH",
          "VIDLORE_VIGNETTE_STRENGTH", "VIDLORE_TEXTURE_STRENGTH",
          "VIDLORE_DARKEN_STRENGTH"):
    os.environ.pop(k, None)
dft = A._OverlayRestraint()
check("default footage grain < cap (lighter than max)",
      dft.grain_amount() < dft.GRAIN_MAX)
check("default footage vignette softer than hard cap",
      dft.vignette_angle() > dft.VIGN_HARD_ANGLE)
check("default archival grain well below legacy 16",
      dft.arch_grain_amount() < 14)
check("default texture scale < 1.0", dft.texture_scale() < 1.0)
# garbage env value falls back to the conservative default (never blows up).
os.environ["VIDLORE_GRAIN_STRENGTH"] = "not-a-number"
check("garbage knob falls back to default",
      A._OverlayRestraint().grain == dft.grain)
os.environ.pop("VIDLORE_GRAIN_STRENGTH", None)

# ---------------------------------------------------------------------------- #
# 4) CLARITY GATE — reduces the stack when inputs are EXCESSIVE.
# ---------------------------------------------------------------------------- #
# An over-treated footage scene: heavy grain + hard vignette + many textures +
# a crushing gamma. The gate must shed layers (heaviest first) AND lift gamma.
excessive = dft.clarity_gate(
    grain=14, vignette_div=3.8, texture_layers=4, darken_gamma=0.80,
    scene_kind="footage")
check("gate fires reductions on excessive input",
      len(excessive["reductions"]) > 0)
check("gate lifts crushing gamma to >= floor",
      excessive["darken_gamma"] >= dft.DARKEN_FLOOR_GAMMA - 1e-9)
check("gate softens the hard vignette (divisor raised)",
      excessive["vignette_div"] > 3.8)
check("gate reduces overall heaviness (<= what it started with)",
      excessive["grain"] <= 14 and excessive["texture_layers"] <= 4)
check("gate records a darken floor / vignette reduction",
      any("darken" in r or "vignette" in r for r in excessive["reductions"]))

# A clean / mild footage scene: the gate should NOT over-correct — grade kept,
# no crushing, minimal (ideally zero) reductions.
mild = dft.clarity_gate(
    grain=5, vignette_div=6.8, texture_layers=1, darken_gamma=1.10,
    scene_kind="footage")
check("mild scene keeps a lifted (non-crushed) gamma",
      mild["darken_gamma"] >= dft.DARKEN_FLOOR_GAMMA - 1e-9)
check("mild scene is not heavily de-staged",
      len(mild["reductions"]) == 0)

# ARCHIVAL has a TIGHTER budget (1 heavy layer) — same input sheds MORE.
arch = dft.clarity_gate(
    grain=12, vignette_div=4.0, texture_layers=2, darken_gamma=0.95,
    scene_kind="archival")
check("archival gate fires (tighter budget)", len(arch["reductions"]) > 0)
check("archival gate never crushes below floor",
      arch["darken_gamma"] >= dft.DARKEN_FLOOR_GAMMA - 1e-9)

# CARD backgrounds collapse to the lightest treatment — NO cinematic stack.
card = dft.clarity_gate(
    grain=9, vignette_div=5.0, texture_layers=3, darken_gamma=1.05,
    scene_kind="card")
check("card drops all texture layers", card["texture_layers"] == 0)
check("card uses a very soft (>=6.8) vignette", card["vignette_div"] >= 6.8)
check("card keeps only minimal grain", card["grain"] <= 3)
check("card marks itself clean", "card→clean" in card["reductions"])

# ---------------------------------------------------------------------------- #
# 5) The live baked strings actually flow through the policy (regression vs the
#    legacy hardcoded muddy values).
# ---------------------------------------------------------------------------- #
import importlib

importlib.reload(A)   # reload with the default (no-env) knobs in effect
check("_VINTAGE no longer darkens (no gamma<1.0)",
      "gamma=0.9" not in A._VINTAGE)
check("_VINTAGE grain reduced from 16",
      "noise=alls=16" not in A._VINTAGE)
check("_CINEMA_FINISH grain reduced from 7",
      "noise=alls=7" not in A._CINEMA_FINISH)
check("_OVERLAY_BASE vignette softened from PI/5.0",
      "vignette=angle=PI/5.0" not in A._OVERLAY_BASE)

print(f"\nAll {_passed} RC5.1 overlay-restraint checks passed.")
