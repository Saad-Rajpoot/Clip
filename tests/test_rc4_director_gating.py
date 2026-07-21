"""RC4 — director gating regression (off-theme business primitive on war docs).

Root cause (confirmed on an Iran-Iraq War documentary): a `wealth_arc_counter`
(a business/finance MG) fired on a geopolitics/history/war beat showing an
ungrounded number, and a `relationship_roster` mixed a person (Saddam) with
places (Khuzestan, Gulf, Iran). This locks in the fixes:

  * registry.eligible() must NOT offer wealth_arc_counter for geopolitics /
    history / war, but MUST still offer it for business (its real home).
  * the director must only derive a wealth `value` from a number ACTUALLY present
    in the narration (no fabricated figure from money keywords alone).
  * relationship_roster must be PEOPLE-ONLY — any entity that resolves as a place
    via the geo backbone is dropped, and if <2 real people remain the roster is
    not emitted (and no place string ever appears as a roster "person").
  * REGISTRY is 70: this RC4 work removes nothing itself (it is a gate, not a
    deletion), but RC5.1 later cut `process_flow_steps` (cheap four-box) from
    production, dropping the count 71 → 70.

Pure + deterministic: no render, no network. Uses the real director.plan(...) /
director._best_for_scene(...) on tiny synthetic scene dicts.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vidlore.motion_graphics import director as D          # noqa: E402
from vidlore.motion_graphics import registry               # noqa: E402

# places the Iran-Iraq beat mixed into "people"; the geo backbone resolves the
# countries (Iran). "Saddam" is the only genuine person in that bug report.
_PLACES = {"khuzestan", "gulf", "iran", "iraq", "france", "germany", "japan"}


def _roster_people(scene, niche):
    """Return the list of person-name strings the director would feed into a
    relationship_roster for this scene/niche, or None if no roster is built."""
    _pid, _s, ins = D._best_for_scene(scene, niche)
    ppl = ins.get("people")
    if not ppl:
        return None
    return [p["name"] if isinstance(p, dict) else p for p in ppl]


# ── 1) eligible() does NOT offer wealth_arc_counter on war/geopolitics/history ──
def test_wealth_arc_counter_blocked_on_offtheme_niches():
    for nm in ("geopolitics", "history", "war"):
        ids = {e["id"] for e in registry.eligible(niche=nm, have_inputs={"value"})}
        assert "wealth_arc_counter" not in ids, (
            f"wealth_arc_counter must be ineligible for niche={nm!r}")
        # the sibling business/finance arcs are denylisted on war/geopolitics too
        if nm in ("geopolitics", "war"):
            assert "money_flow_empire" not in ids, (
                f"money_flow_empire must be ineligible for niche={nm!r}")


# ── 2) eligible() DOES allow wealth_arc_counter for business ──────────────────
def test_wealth_arc_counter_allowed_for_business():
    ids = {e["id"] for e in registry.eligible(niche="business", have_inputs={"value"})}
    assert "wealth_arc_counter" in ids, (
        "wealth_arc_counter must remain eligible for business")
    # biography too (its other legitimate home)
    bio = {e["id"] for e in registry.eligible(niche="biography", have_inputs={"value"})}
    assert "wealth_arc_counter" in bio


# ── 3) a people-only entity list yields a relationship_roster ─────────────────
def test_people_only_yields_relationship_roster():
    scene = {"index": 0, "role": "context", "intensity": 3,
             "narration": ("Andrew Carnegie was the rival and sometime partner of "
                           "Henry Frick, his closest ally and friend.")}
    people = _roster_people(scene, "business")
    assert people is not None, "people-only beat should derive a roster"
    assert len(people) >= 2
    assert any("Carnegie" in p for p in people)
    assert any("Frick" in p for p in people)
    # no place leaked in
    assert not any(p.lower() in _PLACES for p in people)


# ── 4) a mixed person+place list does NOT yield a place-as-person roster ──────
def test_mixed_person_place_list_skips_or_strips_places():
    # (a) The actual bug niche: on geopolitics the biography derivation is gated
    # OFF entirely, so the roster is never built for an Iran-Iraq beat.
    geo_scene = {"index": 0, "role": "context", "intensity": 3,
                 "narration": ("Saddam, the rival and enemy, turned his family "
                               "against Iran and the Gulf and Khuzestan.")}
    assert _roster_people(geo_scene, "geopolitics") is None, (
        "geopolitics must not derive a relationship_roster at all")

    # (b) Even where derivation DOES run (business), every entity that resolves as
    # a place is dropped, and with <2 real people the roster is not emitted —
    # never a place string standing in as a "person". One person + three
    # resolvable countries collapses to 1 person => no roster.
    biz_scene = {"index": 0, "role": "context", "intensity": 3,
                 "narration": ("Saddam was the rival and enemy of France and "
                               "Germany and Japan, his family at war.")}
    biz_people = _roster_people(biz_scene, "business")
    assert biz_people is None, (
        "1 person + resolvable places must collapse below 2 => no roster")

    # (c) Hard invariant for the literal Iran-Iraq mix in ANY niche where a roster
    # could form: a geo-resolvable place must never appear as a roster person.
    mix_scene = {"index": 0, "role": "context", "intensity": 3,
                 "narration": ("Saddam was the rival and enemy of Iran and Iraq, "
                               "his family and allies divided.")}
    for nm in ("business", "biography", "crime", "geopolitics"):
        ppl = _roster_people(mix_scene, nm)
        if ppl:
            assert not any(p.lower() in _PLACES for p in ppl), (
                f"a place leaked into roster people for niche={nm!r}: {ppl}")


# ── 5) money keywords with NO number do not produce a wealth value ────────────
def test_no_number_means_no_wealth_value():
    # business niche so the wealth derivation gate IS active; the prose has money
    # WORDS ("wealth", "fortune", "richest") but no digit — value must stay unset.
    scene = {"index": 0, "role": "reveal", "intensity": 4,
             "narration": ("His wealth and fortune were beyond measure, the "
                           "richest of his age.")}
    _pid, _s, ins = D._best_for_scene(scene, "business")
    assert not ins.get("value"), (
        "no number in narration => the director must not fabricate a wealth value")

    # and with a real number present, the value IS grounded (sanity counter-check)
    grounded = {"index": 0, "role": "reveal", "intensity": 4,
                "narration": "His net worth reached 40 billion dollars at its peak."}
    _p2, _s2, ins2 = D._best_for_scene(grounded, "business")
    assert ins2.get("value"), "a real figure in narration should ground a value"
    assert any(c.isdigit() for c in str(ins2["value"]))


# ── 6) REGISTRY length (RC5.1: 70 after process_flow_steps was removed) ───────
def test_registry_length_unchanged():
    # RC4 itself removes nothing; RC5.1 cut `process_flow_steps` from production
    # (cheap four-box template). Pinning an exact count made every legitimate template
    # addition a false failure — validate the registry's INVARIANTS instead:
    assert len(registry.REGISTRY) >= 60, "registry unexpectedly gutted"
    assert "process_flow_steps" not in registry.REGISTRY, \
        "the RC5.1 removal must stay removed"
    for key, ent in registry.REGISTRY.items():
        assert isinstance(ent, dict), key
        assert ent.get("id") == key, f"{key}: id must equal its registry key"
        for req in ("family", "roles", "intensity_range", "duration_range",
                    "per_video_cap", "cost"):
            assert req in ent, f"{key}: missing required field {req!r}"
        lo, hi = ent["duration_range"]
        assert 0 < lo <= hi, f"{key}: nonsensical duration_range"
        assert int(ent["per_video_cap"]) >= 0, f"{key}: negative per_video_cap"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nALL {len(fns)} TESTS PASSED — REGISTRY={len(registry.REGISTRY)}")
