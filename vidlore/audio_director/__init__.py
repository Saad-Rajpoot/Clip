"""Vidlore audio director package.

Permanent, reusable engine-level audio intelligence that sits ON TOP of the
mature `musiclib` / `sfx` / `assemble` core (which is preserved unchanged):

  • music_director       — chapter-level music cue planning + niche intro profiles
  • sfx_director         — motivated, restrained SFX cue planning per primitive
  • audio_usage_history  — cross-video anti-repetition (category + track + SFX family)

Every module is defensively wired by its callers (try/except -> legacy behaviour),
so a failure here can never break a render. Editorial config is data-driven
(JSON next to these modules / in audio_library) so it can be tuned without code
changes and refined by the audio-engine design workflow.
"""

__all__ = ["audio_usage_history", "music_director", "sfx_director"]
