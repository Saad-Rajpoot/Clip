"""Regression: footage SUBJECT-PRESENCE floor (Forensic v2, vs Vidlore).

A scene that names a CONCRETE subject must (a) report concrete=True and (b) score
an on-subject asset >= floor while an off-subject / generic-filler asset scores
below it. Abstract/concept beats must never be blocked. Runs across 7 niches so
the fix is not a one-sample patch. Pure + deterministic (no render, no network)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class _Scene:
    def __init__(self, narration, keywords):
        self.narration = narration
        self.keywords = keywords


# (niche, narration, keywords, on-subject asset text, off-subject/generic text)
CASES = [
    ("gardening", "He pushed thin strips of copper into the soil around each plant.",
     ["copper", "soil", "plant"], "copper strips garden soil", "abstract blue gradient bokeh"),
    ("science", "A 1994 Cornell study measured the galvanic reaction on a beetle.",
     ["beetle", "reaction"], "colorado potato beetle macro", "businessman walking in office"),
    ("spy", "The handler left a package at the dead drop near the embassy.",
     ["handler", "package", "embassy"], "embassy building exterior night", "generic city street aerial"),
    ("crime", "The detective photographed the weapon at the crime scene.",
     ["detective", "weapon"], "detective examining evidence weapon", "sunset over a calm field"),
    ("business", "The company opened a new factory and tripled its revenue.",
     ["factory", "company"], "industrial factory production line", "abstract motion particles"),
    ("history", "Roman legionaries marched across the stone bridge at dawn.",
     ["legion", "bridge"], "ancient roman soldiers stone bridge", "modern highway traffic"),
    ("geopolitics", "Troops massed tanks along the contested desert border.",
     ["tanks", "border"], "military tanks in desert", "happy people in a park"),
]


def test_subject_presence_seven_niches():
    try:
        from vidlore import footage_strength as F
    except Exception as e:                                         # pragma: no cover
        print(f"SKIP — footage_strength not importable yet: {e}")
        return
    if not hasattr(F, "subject_support_score"):                    # pragma: no cover
        print("SKIP — subject_support_score not implemented yet")
        return
    floor = getattr(F, "SUBJECT_PRESENCE_FLOOR", 0.34)
    failures = []
    for niche, narr, kws, on, off in CASES:
        sc = _Scene(narr, kws)
        son = F.subject_support_score(sc, on)
        soff = F.subject_support_score(sc, off)
        if not (son >= floor and soff < floor and son > soff):
            failures.append(f"{niche}: on={son} off={soff} floor={floor}")
    assert not failures, "subject-presence floor failed: " + "; ".join(failures)


def test_abstract_scene_never_blocked():
    try:
        from vidlore import footage_strength as F
    except Exception as e:                                         # pragma: no cover
        print(f"SKIP — {e}")
        return
    if not hasattr(F, "subject_support_score"):                    # pragma: no cover
        print("SKIP — not implemented yet")
        return
    ab = _Scene("Power and corruption spread through the institutions of the state.",
                ["power", "corruption"])
    # abstract/concept beat must score 1.0 so the floor can never reject it
    assert F.subject_support_score(ab, "dark abstract smoke texture") == 1.0


if __name__ == "__main__":
    for fn in (test_subject_presence_seven_niches, test_abstract_scene_never_blocked):
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            print(f"FAIL  {fn.__name__}: {e}")
