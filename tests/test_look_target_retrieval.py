"""P1 — deictic visual sync: broadened target extraction, the CLIP target probe, and the
retrieval wiring (match bonus → deep-bench ordering → still query → recovery acquisition).

    python3 tests/test_look_target_retrieval.py
"""
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vidlore.clipstudio import policy as P                  # noqa: E402
from vidlore.clipstudio.match import _target_pool_scores    # noqa: E402

FAILS = []
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def check(name, cond):
    print(f"  {'ok  ' if cond else 'FAIL'} {name}")
    if not cond:
        FAILS.append(name)


def seg(text, ent="", kind=""):
    s = types.SimpleNamespace()
    s.text, s.required_entity, s.required_kind = text, ent, kind
    return s


def test_extraction_breadth():
    cases = [
        ("Keep your eye on the dagger in that room", "", "", "the dagger"),
        ("Watch the chalice itself. Joffrey drinks.", "", "", "the chalice itself"),
        ("Watch Bran", "", "", "Bran"),
        ("Watch Olenna Tyrell in this scene", "", "", "Olenna Tyrell"),
        ("And notice what nobody asks him.", "Grand Maester Pycelle", "character",
         "Grand Maester Pycelle"),
        ("watch her while she reads out the charges", "Sansa Stark", "character",
         "Sansa Stark"),
        ("We can watch Olenna's hand take it off.", "", "", "Olenna's hand"),
        ("watch the trial the way Bran watched it", "", "", "the trial"),
    ]
    for text, ent, kind, want in cases:
        got = P.deictic_target(seg(text, ent, kind))
        check(f"extract {text[:40]!r} -> {want!r}", got == want)


def test_extraction_still__rejects_abstracts():
    for text in ("notice the division of labour", "watch his strategy unfold",
                 "watch the way it changes", "that is the tragedy"):
        got = P.deictic_target(seg(text))
        check(f"abstract stays empty: {text[:38]!r}", got == "")


def test_pronoun_rule_needs_character_entity():
    check("pronoun without an entity yields nothing",
          P.deictic_target(seg("notice what nobody asks him")) == "")
    check("pronoun with a non-character entity yields nothing",
          P.deictic_target(seg("notice what nobody asks him", "the chalice", "object")) == "")


def test_target_pool_scores_adaptive():
    import numpy as np

    class _VR:
        def _txt_embed(self, prompt):
            v = np.zeros(8, dtype="float32")
            v[0 if "chalice" in prompt else 1] = 1.0
            return v

    def ps(sid, idx, vec):
        return types.SimpleNamespace(sid=sid, shot=types.SimpleNamespace(index=idx),
                                     embed=np.asarray(vec, dtype="float32"))

    pool = [ps("a", i, [0.9, 0.1] + [0] * 6) for i in range(3)]          # chalice-y
    pool += [ps("b", i, [0.1, 0.9] + [0] * 6) for i in range(9)]         # generic
    sc = _target_pool_scores("the chalice", pool, _VR())
    check("probe returns scores for the pool", sc is not None and len(sc) == 12)
    check("target-bearing shots rank at 1.0",
          sc is not None and all(sc[("a", i)] == 1.0 for i in range(3)))
    check("generic shots rank at 0.0",
          sc is not None and all(sc[("b", i)] == 0.0 for i in range(9)))
    check("tiny pools are undecidable (None)",
          _target_pool_scores("the chalice", pool[:4], _VR()) is None)
    check("cache never crosses targets",
          _target_pool_scores("", pool, _VR()) is None)


def test_wiring():
    m = open(os.path.join(ROOT, "vidlore", "clipstudio", "match.py")).read()
    check("match computes the probe per beat",
          "_tgt01 = _target_pool_scores(_tgt_phrase, pool, vr)" in m)
    check("match records the target_vis signal", '"target_vis"' in m)
    check("bonus is ranking-only (rides on bonus, env weight)",
          "VIDLORE_CLIPSTUDIO_LOOK_MATCH_W" in m and
          "VIDLORE_CLIPSTUDIO_LOOK_MATCH_BONUS" in m)
    v = open(os.path.join(ROOT, "vidlore", "clipstudio", "verify.py")).read()
    check("primary + prefetch fingerprints carry must_see",
          v.count("must_see=_must_see(seg))") >= 2)
    check("keep verdicts with a missing target are now flagged",
          "FLAG-ON-ANY-VERDICT" in v)
    check("deep bench orders by target_vis on a look miss",
          'get("target_vis", 0.0)), reverse=True)' in v)
    i = open(os.path.join(ROOT, "vidlore", "clipstudio", "image_fallback.py")).read()
    check("still query leads with the deictic target", "deictic_target(seg)" in i)
    o = open(os.path.join(ROOT, "vidlore", "clipstudio", "orchestrate.py")).read()
    check("recovery adds bounded look-miss acquisition",
          "VIDLORE_CLIPSTUDIO_LOOK_RECOVERY" in o and "_look_aug" in o)
    check("recovery augments the search query with the target",
          "close-up" in o.split("_look_aug[s.index]")[1][:40])


if __name__ == "__main__":
    for fn in sorted(k for k in dict(globals()) if k.startswith("test_")):
        print(f"[{fn}]")
        globals()[fn]()
    print(f"\n{'FAILED: ' + ', '.join(FAILS) if FAILS else 'all passed'}")
    sys.exit(1 if FAILS else 0)
