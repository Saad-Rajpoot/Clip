#!/usr/bin/env python3
"""Replay job 6a26707939's REAL breakout candidates through the admission gate, live.

Unit tests script the judge's replies, so they prove the WIRING and can never prove the JUDGMENT.
This runs the real provider against the real (beat, aired-audio) pairs that render produced, and
fails if any of the three decisions it got wrong before is still wrong.

Every string below is transcribed from that job — build.log, project.json and the per-source word
streams under index/ — not invented:

  beat 18   THE NEGATION. The beat promised "I have seen the future in the flames." and
            find_quote_span matched it at phrase ratio 0.8 against audio that says the OPPOSITE.
            The old gate skipped the relevance question entirely on that "quote anchor" and
            admitted it 3-judges-to-0 against; only a downstream coverage floor kept it off screen.
  beat 113  THE GARBLE. Real ASR from game_of_thrones_s06e04_a18f768e. All three single-stage
            samples answered "Melisandre declares Jon Snow is the prince that was promised" —
            which is a paraphrase of the BEAT they were shown in the same prompt, not of the audio.
  beat 53   THE CONTROL. A genuinely correct breakout (3/3 belongs=true, confidence 1.00). It must
            still air: a gate that only ever says no is not a fixed gate.

Usage:  python3 tools/breakout_admit_real_cases.py
Costs a handful of small text calls. Exit code 0 only if all cases decide correctly.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vidlore.clipstudio import build as B  # noqa: E402
from vidlore.clipstudio import llm as L  # noqa: E402

SHOW = "Game of Thrones"

CASES = [
    dict(
        name="beat 18 — promised line matched its own NEGATION",
        must_admit=False,
        beat_text="The counter-argument is not a supply route or a battle plan. It is her.",
        beat_subject="character: Melisandre",
        promised_quote="I have seen the future in the flames.",
        quote_authored=True,          # find_quote_span DID locate it — that is the whole trap
        aired_text="I don't know your grace. I can't see the future in the flames.",
        was="ADMITTED (quote-anchored, 3/3) with all three judges saying belongs=false",
    ),
    dict(
        name="beat 113 — unintelligible ASR answered from the beat",
        must_admit=False,
        beat_text="Azor Ahai the prince that was promised is the central identification of her "
                  "religion",
        beat_subject="character: Melisandre",
        promised_quote="The prince that was promised will bring the dawn.",
        quote_authored=False,
        aired_text="My lady? Sidervos. We'll use to hear it, cut some black. I will do us.",
        was="ADMITTED — 3/3 'Melisandre declares Jon Snow is the prince that was promised'",
    ),
    dict(
        name="beat 53 — a genuinely correct breakout (control)",
        must_admit=True,
        beat_text="Stannis says he will risk everything",
        beat_subject="",
        promised_quote="",
        quote_authored=False,
        aired_text="When I'm out of food, we kind of want to supply line some clears. This is the "
                   "right time, and I will risk everything. Because if I don't, we've lost. "
                   "We march to victory.",
        was="ADMITTED correctly (3/3, conf 1.00) — then lost downstream to window-QC legibility",
    ),
]


def main() -> int:
    if not L.has_llm(None):
        print("no LLM configured — cannot run the live probe", file=sys.stderr)
        return 2
    os.environ.setdefault("VIDLORE_CLIPSTUDIO_BREAKOUT_ADMIT_SAMPLES", "3")

    bad = []
    for c in CASES:
        ok, why, verdicts = B._breakout_window_admissible(
            c["aired_text"], SHOW,
            beat_text=c["beat_text"], beat_subject=c["beat_subject"],
            promised_quote=c["promised_quote"], quote_authored=c["quote_authored"])
        want = c["must_admit"]
        good = (ok is want)
        print(f"\n{'ok  ' if good else 'FAIL'} {c['name']}")
        print(f"       before: {c['was']}")
        print(f"       now:    {'ADMIT' if ok else 'REJECT'} — {why}")
        for v in verdicts:
            print(f"         · intelligible={v.get('intelligible')} "
                  f"speaker={v.get('speaker')!r} scene={str(v.get('scene'))[:52]!r} "
                  f"belongs={v.get('belongs')} conf={v.get('confidence')}")
        if not good:
            bad.append(c["name"])

    print("\n" + "=" * 78)
    if bad:
        print(f"FAILED {len(bad)}/{len(CASES)}:")
        for n in bad:
            print(f"  - {n}")
        return 1
    print(f"SAFETY: all {len(CASES)} real cases decide correctly "
          f"(both known-bad windows refused, the correct one still airs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
