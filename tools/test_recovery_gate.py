#!/usr/bin/env python3
"""Behavioral tests for the bounded-recovery gate (R4-5) — which EXACT beats trigger a targeted
rediscovery→download→index→rematch→reverify round, and which do NOT. Drives the real decision
`_beat_is_unresolved` with real ScriptSegment + selection stand-ins (no source-greps)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vidlore.clipstudio.orchestrate import _beat_is_unresolved     # noqa: E402
from vidlore.clipstudio import policy as _policy                    # noqa: E402
from vidlore.clipstudio.models import ScriptSegment                 # noqa: E402

PASS = FAIL = 0


def _say(ok, msg):
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  ok   {msg}")
    else:
        FAIL += 1
        print(f"  FAIL {msg}")


class Sel:
    def __init__(self, source_id="", verdict=None, status="ok", image_path="", image_source=""):
        self.source_id = source_id
        self.verifier = ({"status": status, "verdict": verdict} if verdict is not None else {})
        self.image_path = image_path
        self.image_meta = ({"source": image_source} if image_source else {})


def exact_seg(i=1):
    # an exact_scene beat: a precise moment the render must actually show
    s = ScriptSegment(index=i, text="Tywin dismisses the small council",
                      scene_query="tywin lannister dismisses the small council chamber",
                      required_entity="Tywin Lannister", required_kind="character",
                      visual_policy="exact_scene")
    return s


def generic_seg(i=2):
    s = ScriptSegment(index=i, text="Power is a curious thing",
                      expected_visual="abstract castle imagery", visual_policy="generic_filler")
    return s


def main():
    ex = exact_seg()
    gen = generic_seg()

    # a verifier-KEPT exact beat with real footage → resolved (no recovery)
    _say(not _beat_is_unresolved(Sel("src_A", verdict="keep"), ex, _policy),
         "verifier-kept exact beat is RESOLVED (no recovery)")

    # a verifier-REJECTED exact beat (replace) → UNRESOLVED (recover)
    _say(_beat_is_unresolved(Sel("src_A", verdict="replace"), ex, _policy),
         "verifier-rejected exact beat is UNRESOLVED → recovery")

    # an exact beat with NO source at all → UNRESOLVED
    _say(_beat_is_unresolved(Sel(""), ex, _policy),
         "exact beat with no source is UNRESOLVED → recovery")

    # a missing selection for an exact beat → UNRESOLVED
    _say(_beat_is_unresolved(None, ex, _policy),
         "exact beat with no selection at all is UNRESOLVED → recovery")

    # an exact beat covered by a REAL source-frame still → resolved (still IS coverage)
    _say(not _beat_is_unresolved(Sel("", verdict="replace", image_path="/x.jpg",
                                     image_source="source-frame"), ex, _policy),
         "exact beat with a real source-frame still is RESOLVED (still is coverage)")
    _say(not _beat_is_unresolved(Sel("", image_path="/x.jpg", image_source="web-exact-scene"),
                                 ex, _policy),
         "exact beat with a validated web-exact-scene still is RESOLVED")

    # A verifier-REJECTED non-exact beat IS unresolved. This test used to assert the opposite —
    # the original R4-5 contract exempted every non-exact policy — but that exemption was removed
    # deliberately: a rejected character/filler clip whose editorial hold later fails the R4-3/R4-4
    # validity checks release-blocks the finished render exactly like an exact one (observed: 7
    # character_specific beats FATALed a 4½-hour render while recovery reported ZERO unresolved).
    # Recovery and the build-stage gate must see the SAME set. Only the REJECTION makes it
    # unresolved, though — a non-exact beat that simply has no source still has its fallback.
    _say(_beat_is_unresolved(Sel("", verdict="replace"), gen, _policy),
         "verifier-rejected generic beat IS unresolved (gate would block it → recovery)")
    _say(not _beat_is_unresolved(Sel(""), gen, _policy),
         "generic beat with no source is NOT unresolved (no rejection = fallback treatment)")

    # a verifier ERROR (transient) is NOT a rejection → not unresolved on that basis alone
    _say(not _beat_is_unresolved(Sel("src_A", status="error", verdict="replace"), ex, _policy),
         "a transient verifier ERROR is not a REJECTION (has source → not unresolved)")

    print(f"\n{PASS} passed · {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
