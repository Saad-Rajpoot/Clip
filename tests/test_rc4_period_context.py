"""RC4 regression: post-1945 CONFLICT topics must resolve to their TRUE decade.

Root cause this guards: `period_guard._ERA_WORDS` had no post-1945 conflict bucket
(the latest was `midcentury`/1960; ww2 ends 1945), so a 1980s war documentary
(Iran–Iraq, Gulf, Falklands…) whose narration said "modern"/"today" fell through to
the `modern` catch-all (year 2010). detect_era then reported year=2010, and the
downstream query_bias/reject_terms pushed 2010s stock onto an 1980s topic.

These tests assert detect_era routes such a topic to ~1980 (NOT 2010) while a
genuinely modern smartphone/startup video still lands "modern", and that the
pre-1945 buckets (Napoleon → period_sensitive, WWII → ww2) are unchanged. Pure +
deterministic: no network, no render. Runs under pytest OR as a plain script."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vidlore import period_guard as pg


# (1) Iran–Iraq War (1980-1988) → real decade (1980s), and NOT "modern".
def test_iran_iraq_resolves_to_1980s_not_modern():
    era = pg.detect_era(
        "The Iran-Iraq War was one of the longest conventional conflicts of the "
        "20th century, a brutal 1980s war between Iran and Iraq, still studied today "
        "for its use of modern weapons and trench tactics.")
    assert era["year"] is not None, era
    assert 1980 <= era["year"] <= 1989, f"expected 1980s, got {era['year']} ({era})"
    assert era["year"] != 2010, f"regressed to the modern catch-all: {era}"
    assert era["label"] != "modern", f"must not be the modern bucket: {era}"

    # Even a minimal mention (no date words) must not fall through to modern/2010.
    bare = pg.detect_era("A documentary about the Iran-Iraq war.")
    assert bare["year"] == 1980 and bare["label"] != "modern", bare

    # The real point of the fix: query bias now targets the 1980s, not present day.
    qb = pg.query_bias(era)
    assert any("1980" in t for t in qb), f"query_bias should target 1980s: {qb}"
    assert "archival" in qb, qb
    rt = pg.reject_terms(era)
    assert "smartphone" in rt and "drone" in rt, rt


# (2) Napoleon (1812) → pre-1945, period_sensitive True (unchanged behaviour).
def test_napoleon_is_period_sensitive():
    era = pg.detect_era(
        "Napoleon's 1812 invasion of Russia; the Grande Armee marched on Moscow.")
    assert era["label"] == "napoleonic", era
    assert era["year"] == 1812, era
    assert era["period_sensitive"] is True, era
    assert era["year"] < pg.MODERN_ERA_YEAR, era


# (3) WWII text → the ww2 bucket (unchanged behaviour).
def test_ww2_bucket():
    era = pg.detect_era(
        "World War II and the 1944 D-Day Normandy landings against the Nazi Wehrmacht; "
        "the Blitz and Pearl Harbor reshaped the war.")
    assert era["label"] == "ww2", era
    assert era["year"] == 1942, era
    assert era["period_sensitive"] is True, era


# (4) A clearly modern smartphone/startup video → still "modern" (no over-trigger).
def test_modern_tech_stays_modern():
    era = pg.detect_era(
        "A modern startup builds a smartphone app, using social media, the internet "
        "and digital tools that define the contemporary 21st century, today, in the 2020s.")
    assert era["label"] == "modern", era
    assert era["year"] == 2010, era
    assert era["period_sensitive"] is False, era
    # Contemporary topics must never be period-biased (would over-filter the search).
    assert pg.query_bias(era) == [], pg.query_bias(era)
    assert pg.reject_terms(era) == [], pg.reject_terms(era)


# (5) Neutral landscape → does not crash and is not wrongly flagged.
def test_neutral_landscape_not_flagged():
    era = pg.detect_era("A calm mountain lake at sunrise with mist over the forest.")
    # No period signal → unknown / no year / not period-sensitive; must not raise.
    assert era["label"] == "unknown", era
    assert era["year"] is None, era
    assert era["period_sensitive"] is False, era
    # Downstream helpers must be empty (no spurious bias) and must not crash.
    assert pg.query_bias(era) == []
    assert pg.reject_terms(era) == []
    assert pg.period_risk("modern city skyline with cars and traffic", era)["risk"] == 0
    # And empty input is safe too.
    assert pg.detect_era("")["label"] == "unknown"


# (bonus) Sibling new-bucket coverage so the fix is not a one-phrase patch.
def test_other_post1945_conflicts_resolve():
    for text, lo, hi in [
        ("Operation Desert Storm and the Gulf War.", 1980, 1989),
        ("The Falklands War between Britain and Argentina.", 1980, 1989),
        ("The Soviet-Afghan war in the mountains.", 1980, 1989),
        ("The Bosnian war during the Balkans war of the 1990s.", 1990, 1999),
    ]:
        era = pg.detect_era(text)
        assert era["label"] != "modern", (text, era)
        assert lo <= (era["year"] or 0) <= hi, (text, era)


if __name__ == "__main__":
    tests = [
        test_iran_iraq_resolves_to_1980s_not_modern,
        test_napoleon_is_period_sensitive,
        test_ww2_bucket,
        test_modern_tech_stays_modern,
        test_neutral_landscape_not_flagged,
        test_other_post1945_conflicts_resolve,
    ]
    failures = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL  {fn.__name__}: {e}")
    print(f"\nRC4 PERIOD CONTEXT: {len(tests) - failures}/{len(tests)} PASS")
    sys.exit(1 if failures else 0)
