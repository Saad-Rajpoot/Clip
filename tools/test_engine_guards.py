#!/usr/bin/env python3
"""Reusable regression tests for the four permanent engine guards + QA layer:
portrait_intel · period_guard · niche_palette · card_style_guard · asset_qa.

Proves the RULES generalise to many future topics — not only the 3 re-rendered
samples. Pure-logic, offline, no network/render. Run:

  python3 tools/test_engine_guards.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vidlore import (asset_qa, card_style_guard, niche_palette,  # noqa: E402
                     period_guard, portrait_intel)

_P = _F = 0


def ok(cond, msg):
    global _P, _F
    if cond:
        _P += 1
    else:
        _F += 1
        print(f"  FAIL: {msg}")


# ── 1. portrait_intel ────────────────────────────────────────────────────────
def test_portrait_intel():
    print("portrait_intel")
    pre = ["Napoleon Bonaparte", "Julius Caesar", "Genghis Khan",
           "Alexander the Great", "George Washington"]
    photo = ["Abraham Lincoln", "Winston Churchill", "John D. Rockefeller"]
    for n in pre:
        cls, why = portrait_intel.era_class(n)
        ok(cls == "pre_photographic", f"{n} should be pre_photographic, got {cls} ({why})")
        ok(portrait_intel.prefers_artwork(n), f"{n} should prefer artwork")
        q = portrait_intel.portrait_queries(n)
        ok("painting" in q[0].lower(), f"{n} first query should be a painting: {q[0]}")
        ok(not any("photograph" in x.lower() for x in q),
           f"{n} queries must not ask for a photograph: {q}")
    for n in photo:
        cls, why = portrait_intel.era_class(n)
        ok(cls == "photographic", f"{n} should be photographic, got {cls} ({why})")
        ok(not portrait_intel.prefers_artwork(n), f"{n} should NOT prefer artwork")
        q = portrait_intel.portrait_queries(n)
        ok(any("photograph" in x.lower() for x in q), f"{n} should allow photo: {q}")
    # strict name match
    ok(portrait_intel.name_match("Napoleon Bonaparte", "Portrait of Napoleon Bonaparte") >= 0.5,
       "name match should accept matching title")
    ok(portrait_intel.name_match("Napoleon Bonaparte", "A Random Modern Soldier") == 0.0,
       "name match should reject unrelated title")
    # acceptable_source: pre-photo + non-artwork + untrusted → reject; artwork → ok
    bad, why = portrait_intel.acceptable_source(
        "Napoleon Bonaparte", title="napoleon costume photo", prov={"domain": "stockphotos.com"})
    ok(not bad, f"pre-photo non-artwork untrusted should reject (got {why})")
    good, why = portrait_intel.acceptable_source(
        "Napoleon Bonaparte", title="Napoleon Bonaparte portrait painting by David",
        prov={"domain": "commons.wikimedia.org"})
    ok(good, f"pre-photo artwork on commons should be accepted (got {why})")
    # modern stock / reenactor rejection (any era)
    bad2, _ = portrait_intel.acceptable_source(
        "Al Capone", title="Al Capone reenactor actor at film premiere")
    ok(not bad2, "reenactor/actor imagery should be rejected")


# ── 2. period_guard ──────────────────────────────────────────────────────────
def test_period_guard():
    print("period_guard")
    cases = {
        "Napoleon's invasion of Russia in 1812": ("napoleonic", True),
        "the Roman Empire and its legions": ("ancient", True),
        "medieval knights and castles in the Middle Ages": ("medieval", True),
        "World War II and the year 1942": ("ww2", True),
        "the Industrial Revolution of the 1800s": ("industrial", True),
        "a modern smartphone startup in 2021": ("modern", False),
    }
    for text, (label, sensitive) in cases.items():
        era = period_guard.detect_era(text)
        ok(era["label"] == label, f"'{text[:30]}' → era {era['label']} (want {label})")
        ok(era["period_sensitive"] == sensitive,
           f"'{text[:30]}' period_sensitive {era['period_sensitive']} (want {sensitive})")
    # modern markers in a period scene → high risk
    era1812 = period_guard.detect_era("Napoleon 1812 retreat")
    r = period_guard.period_risk("modern city skyline with cars and highway", era=era1812)
    ok(r["risk"] >= 30, f"modern city in 1812 should be high risk, got {r['risk']}")
    ok("car" in r["markers"] or "cars" in r["markers"], f"should flag cars: {r['markers']}")
    # genuine landscape in a period scene → low risk (no over-filtering)
    r2 = period_guard.period_risk("snowy mountain landscape at dawn", era=era1812)
    ok(r2["risk"] == 0, f"landscape should not be penalised, got {r2['risk']}")
    # modern scene → never period-risky
    eram = period_guard.detect_era("a startup in 2021")
    r3 = period_guard.period_risk("city skyline with cars", era=eram)
    ok(r3["risk"] == 0, "modern scene candidates are never period-risky")
    # reject terms only for period scenes
    ok("skyline" in period_guard.reject_terms(era1812), "1812 should reject skyline")
    ok(period_guard.reject_terms(eram) == [], "modern scene → no reject terms")
    # WWII allows period vehicles but not glass towers
    eraww2 = period_guard.detect_era("World War II 1942 battle")
    rww = period_guard.period_risk("soldiers with a truck and a tank", era=eraww2)
    ok(rww["risk"] == 0, f"WWII trucks/tanks are period-appropriate, got {rww['risk']}")
    rww2 = period_guard.period_risk("modern glass tower downtown", era=eraww2)
    ok(rww2["risk"] > 0, "WWII scene should still reject modern glass towers")
    # REGRESSION: an ambiguous nationality ("Egyptian-born") in a MODERN doc must
    # NOT trigger the 'ancient' era (this false-positived a 1960s spy doc).
    espy = period_guard.detect_era(
        "an Egyptian-born Israeli intelligence agent in Damascus in 1965")
    ok(espy["label"] == "modern", f"Egyptian-born 1965 spy → modern, got {espy['label']}")
    ok(not espy["period_sensitive"], "modern spy doc must not be period-sensitive")
    erome = period_guard.detect_era("a Roman Catholic priest in 1985")
    ok(not erome["period_sensitive"], "'Roman Catholic' in 1985 is not ancient")
    # ── SEARCH-SIDE markers (2026-06-03): real Pexels slugs that LEAKED modern
    # stock onto the Napoleon-1812 sweep must score risk; genuine period footage
    # must stay at 0. These run over a clip's slug at fetch time (footage.
    # _period_blocked). See project_period_search_guard.
    for _mslug in ("people wearing face mask while protesting",
                   "traffic in moscow at night", "covid protest rally",
                   "vibrant military parade in urban setting",
                   "back view of blm protestors marching on city streets"):
        ok(period_guard.period_risk(_mslug, era=era1812)["risk"] >= 18,
           f"modern slug must be flagged for 1812: {_mslug!r}")
    for _gslug in ("soldiers marching on the field",
                   "french revolutionary soldiers marching together",
                   "men walking on a snow covered field",
                   "aerial view of winter landscape",
                   "silhouettes of group of historic german soldiers"):
        ok(period_guard.period_risk(_gslug, era=era1812)["risk"] == 0,
           f"genuine period footage must stay risk 0: {_gslug!r}")
    # a COVID face-mask clip is still anachronistic for a 1942 (WWII) scene …
    ok(period_guard.period_risk("face mask", era=eraww2)["risk"] >= 18,
       "COVID face-mask clip is anachronistic for a 1942 scene")
    # … but the new markers must NEVER fire on a modern documentary (no over-filter)
    ok(period_guard.period_risk(
        "urban traffic and a protest crowd",
        era=period_guard.detect_era("a startup in 2021"))["risk"] == 0,
       "modern doc → search-side markers never fire")
    # EXPLICIT POST-1945 year / decade in a slug = anachronism for a pre-1945
    # scene (catches mid-century "vintage" home-movie stock the era-bias surfaces)
    for _yslug in ("vintage-suburban-street-from-the-1950s-31683904",
                   "vintage-smokehouse-tower-from-1960s-film-31938870",
                   "city-view-1972", "a-scene-from-2020"):
        ok(period_guard.period_risk(_yslug, era=era1812)["risk"] >= 18,
           f"post-1945 year/decade must flag for 1812: {_yslug!r}")
    # pre-1945 dates + period phrasing + the 8-digit Pexels clip-ID must NOT flag
    for _ok in ("1920s-vintage-street-fight-footage-31613384",
                "woman-in-19th-century-costume-6718799",
                "vintage-family-walk-on-snowy-winter-day-31808738"):
        ok(period_guard.period_risk(_ok, era=era1812)["risk"] == 0,
           f"pre-1945 / period slug must stay risk 0: {_ok!r}")


# ── 3. niche_palette ─────────────────────────────────────────────────────────
def test_niche_palette():
    print("niche_palette")
    # crime trends ember_red and NEVER amber_gold; still varies
    crime = [niche_palette.select_palette("crime", s)[0] for s in range(200)]
    ember = crime.count("ember_red")
    ok(ember > 100, f"crime should lean ember_red (got {ember}/200)")
    ok("amber_gold" not in crime, "crime must never get warm business amber_gold")
    ok(len(set(crime)) >= 2, "crime should still vary across topics (controlled variation)")
    # business trends amber_gold
    biz = [niche_palette.select_palette("business", s)[0] for s in range(200)]
    ok(biz.count("amber_gold") > 100, f"business should lean amber_gold ({biz.count('amber_gold')}/200)")
    # determinism
    ok(niche_palette.select_palette("crime", 42)[0] == niche_palette.select_palette("crime", 42)[0],
       "selection must be deterministic for a fixed seed")
    # anti-repeat — pick a seed whose NATURAL choice is ember_red, then prove the
    # anti-repeat nudges away from it and logs the reason.
    s_ember = next(s for s in range(500)
                   if niche_palette.select_palette("crime", s)[0] == "ember_red")
    pal, reason = niche_palette.select_palette("crime", s_ember, recent=["ember_red"])
    ok(pal != "ember_red", f"anti-repeat should avoid last palette, got {pal}")
    ok("anti-repeat" in reason, f"anti-repeat reason should be logged, got '{reason}'")
    # alias normalization
    ok(niche_palette.normalize_niche("true_crime") == "crime", "true_crime→crime")
    ok(niche_palette.normalize_niche("spy_intel") == "spy", "spy_intel→spy")
    ok(niche_palette.normalize_niche("Mossad") == "spy", "Mossad→spy")
    # multiple distinct crime topics shouldn't all be identical. Use a DETERMINISTIC spread of
    # seeds (0..19) — the old `hash(t)` on topic strings was randomized per-process by
    # PYTHONHASHSEED, so occasionally all five hashed to ember_red-picking seeds and this flaked.
    topics = [niche_palette.select_palette("crime", s)[0] for s in range(20)]
    ok(len(set(topics)) >= 2, f"distinct crime seeds should differ somewhat: {set(topics)}")


# ── 4. card_style_guard ──────────────────────────────────────────────────────
def test_card_style_guard():
    print("card_style_guard")
    ok(card_style_guard.is_dark_niche("spy"), "spy is dark")
    ok(card_style_guard.is_dark_niche("true_crime"), "true_crime is dark")
    ok(not card_style_guard.is_dark_niche("business"), "business is not dark")
    # dark niche + light statement → forced dark
    r = card_style_guard.resolve_card_variant("spy", "statement")
    ok(not r["allow_light"], "spy statement should not be light")
    ok(r["kind"] == "text_on_black", f"spy statement should route to text_on_black, got {r['kind']}")
    # light niche keeps light
    r2 = card_style_guard.resolve_card_variant("business", "statement")
    ok(r2["allow_light"], "business statement may stay light")
    # authorised rare contrast beat
    r3 = card_style_guard.resolve_card_variant("spy", "statement",
                                               recipe_allows_contrast=True, contrast_used=0)
    ok(r3["allow_light"] and r3["contrast_beat"], "authorised contrast beat allowed once")
    r4 = card_style_guard.resolve_card_variant("spy", "statement",
                                               recipe_allows_contrast=True, contrast_used=1)
    ok(not r4["allow_light"], "second contrast beat should be denied (rarity)")
    # non-light kinds untouched
    r5 = card_style_guard.resolve_card_variant("spy", "text_on_black")
    ok(r5["kind"] == "text_on_black", "non-light kind unchanged")
    # palette/card compatibility
    bad, _ = card_style_guard.palette_card_compatible("ember_red", light_card=True)
    ok(not bad, "light card on ember_red is incompatible")


# ── 5. asset_qa ──────────────────────────────────────────────────────────────
def test_asset_qa():
    print("asset_qa")
    w = asset_qa.check_portrait("Napoleon Bonaparte", prov={}, ai_generated=True)
    ok(any(x["check"] == "prephoto_ai_face" for x in w), "AI face for pre-photo person flagged")
    w2 = asset_qa.check_portrait(
        "Winston Churchill", title="Winston Churchill photograph",
        prov={"domain": "commons.wikimedia.org", "source": "Wikimedia"}, name_match_score=1.0)
    ok(not any(x["check"].startswith("prephoto") for x in w2),
       "photographic person with good photo should not be prephoto-flagged")
    w3 = asset_qa.check_footage_period(
        scene_text="Napoleon 1812 campaign",
        candidate_title="aerial drone shot of modern city skyline with cars")
    ok(any(x["check"] == "modern_footage_in_period" for x in w3), "modern footage in period flagged")
    ok(any(x["check"] == "contemporary_city_in_period" for x in w3), "contemporary city flagged")
    w4 = asset_qa.check_palette_niche("crime", "amber_gold")
    ok(any(x["check"] == "palette_niche_mismatch" for x in w4), "crime+amber flagged")
    ok(not asset_qa.check_palette_niche("crime", "ember_red"), "crime+ember not flagged")
    w5 = asset_qa.check_card_brightness("spy", "statement", allow_light=True)
    ok(any(x["check"] == "bright_card_in_dark_niche" for x in w5), "bright card in dark niche flagged")
    ok(not asset_qa.check_card_brightness("spy", "statement", allow_light=False),
       "dark card in dark niche not flagged")
    # summary
    s = asset_qa.summarize(w3 + w4 + w5)
    ok(s["total"] >= 4, "summary counts warnings")


def main():
    for t in (test_portrait_intel, test_period_guard, test_niche_palette,
              test_card_style_guard, test_asset_qa):
        t()
    print(f"\n{_P} passed · {_F} failed")
    sys.exit(1 if _F else 0)


if __name__ == "__main__":
    main()
