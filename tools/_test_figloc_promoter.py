"""Deterministic test of _promote_figure_locator (no LLM, no render).
POSITIVE: protagonist + control-cue + place -> fires once.
NEGATIVE: no person<->place spine -> must not fire."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from vidlore.script_gen import Scene, _apply_graphic_caps  # noqa: E402


def kinds(scenes):
    return [(s.index, s.graphic_kind or "-", s.graphic_text, s.graphic_body)
            for s in scenes if (s.graphic_kind or "")]


# ---- POSITIVE: a Milosevic-style bio with a clear who-controls-where beat
pos = [
    Scene(0, "In the heart of the Balkans, a storm was gathering.",
          role="hook", intensity=3),
    Scene(1, "Meet Slobodan Milosevic, the lawyer who would seize a nation.",
          role="intro", intensity=3, graphic_kind="name_reveal",
          graphic_text="SLOBODAN MILOSEVIC"),
    Scene(2, "From Belgrade, he ruled Serbia with an iron grip for a decade.",
          role="rise", intensity=4),
    Scene(3, "His speeches stirred a fervor that swept the region.",
          role="escalation", intensity=4),
    Scene(4, "By 2000, the streets turned against him and his regime fell.",
          role="resolution", intensity=5),
]
_apply_graphic_caps(pos, None)
pos_hit = any((s.graphic_kind or "") == "figure_locator" for s in pos)
print("POSITIVE kinds:", kinds(pos))
print("POSITIVE figure_locator fired:", pos_hit)

# ---- NEGATIVE: a generic science doc, no person ruling a place
neg = [
    Scene(0, "The ocean covers most of our planet.", role="hook", intensity=2),
    Scene(1, "Beneath the waves, pressure builds to crushing levels.",
          role="problem", intensity=3),
    Scene(2, "Strange creatures thrive where no light reaches.",
          role="evidence", intensity=3),
    Scene(3, "Scientists still map these depths today.", role="reveal",
          intensity=3),
    Scene(4, "The deep sea remains our last frontier.", role="resolution",
          intensity=2),
]
_apply_graphic_caps(neg, None)
neg_hit = any((s.graphic_kind or "") == "figure_locator" for s in neg)
print("NEGATIVE kinds:", kinds(neg))
print("NEGATIVE figure_locator fired:", neg_hit)

print("\nRESULT:", "PASS" if (pos_hit and not neg_hit) else "FAIL")
