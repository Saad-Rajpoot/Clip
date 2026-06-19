"""Vidlore Motion-Graphics Engine.

Premium, cinematic, non-template motion graphics for faceless documentaries.
Extracted (principles only) from a full local forensic of a MagnatesMedia
documentary — see research/forensic_refs/magnatesmedia_motion_graphics/.

Architecture:
  look.py            — the shared "cinematic composite" look layer (grade,
                       vignette, film grain, gold bloom, framed-subject
                       compositor, slow parallax, palettes, fonts, easing).
  numbers/           — stat / currency / count-up callouts
  portraits/         — framed graded portraits, name-over-map, parallax
  typography/        — kinetic keyword, titles, labels
  documents/         — headline / evidence / highlight-sweep
  maps/ charts/ timelines/ evidence/ transitions/ audio_sync/
  director.py        — editorial chooser (which primitive, when, how strong)
  qa/                — reusable motion-QA (frames, checks, verdict)

Every primitive is a modular spec + a pure-local renderer (ffmpeg + PIL +
numpy). No paid API. Each registers a MotionGraphicManifest-compatible spec so
the existing assemble overlay-baking + Review-Editor override path can drive it.
"""

__all__ = ["look"]
__version__ = "0.1.0"
