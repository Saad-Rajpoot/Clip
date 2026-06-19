#!/usr/bin/env python3
"""Regression tests for the permanent audio-director engine layer:
music_director · sfx_director · audio_usage_history.

Proves the editorial RULES generalise (niche intro shaping, per-primitive SFX
restraint, cross-video anti-repetition) — pure logic, offline, no render. Run:

  python3 tools/test_audio_director.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vidlore.audio_director import (audio_usage_history as H,  # noqa: E402
                                    music_director as MD, sfx_director as SD)

_P = _F = 0


def ok(cond, msg):
    global _P, _F
    if cond:
        _P += 1
    else:
        _F += 1
        print(f"  FAIL: {msg}")


def test_niche_normalize():
    print("music_director.normalize_niche")
    for raw, want in [("spy", "spy"), ("Mossad", "spy"), ("true_crime", "crime"),
                      ("business", "business"), ("geopolitics", "geopolitics"),
                      ("mystery", "mystery")]:
        ok(MD.normalize_niche(raw) == want, f"{raw} -> {want}, got {MD.normalize_niche(raw)}")
    # unknown niche must still resolve to a valid canonical niche (never crash)
    ok(MD.normalize_niche("zzz") in ("spy", "crime", "business", "history",
       "geopolitics", "mystery"), "unknown niche resolves to a canonical one")


def test_intro_intelligence():
    print("music_director intro intelligence")
    # business opens the most confident (highest start), mystery the quietest
    biz = MD.intro_relative_envelope("business")["start_mult"]
    mys = MD.intro_relative_envelope("mystery")["start_mult"]
    ok(biz > mys, f"business intro should open hotter than mystery ({biz} vs {mys})")
    # every niche's intro is bounded so it can never bury the voice
    for n in ("spy", "crime", "business", "history", "geopolitics", "mystery"):
        e = MD.intro_relative_envelope(n)
        # v14 hold-then-recede (validated): intros open stronger and HOLD across the
        # hook, clamped by music_director._INTRO_START_MAX (2.8) / _INTRO_RECEDE_MAX
        # (30 s). Rails widened to those documented-safe ceilings (+ small margin).
        # Safe: it's a RELATIVE lift on the proven absolute mix + sidechain duck;
        # loudnorm -16 LUFS + limiter 0.85 prevent masking / clipping.
        ok(e["start_mult"] <= 2.85, f"{n} intro start within safety rail")
        ok(4.0 <= e["recede_s"] <= 30.5, f"{n} recede within rail")
    # the volume expr is a real recede for shaped niches, empty (no-op) for default
    ok("if(lt(t," in MD.intro_volume_expr("spy"), "spy gets a recede expr")
    ok(MD.intro_volume_expr("default") == "", "unknown niche = no-op intro (byte-identical)")
    # regression guard (2026-06-04): the default / unrecognised niche must stay a
    # no-op (start_mult ~1.0) — never an unvalidated intro swell.
    ok(abs(MD.intro_relative_envelope("default")["start_mult"] - 1.0) < 0.05,
       "default niche intro is a no-op (start_mult ~1.0)")


def test_mix_character():
    print("music_director per-niche mix character")
    # bounded multipliers — never beyond the safety rails
    for n in ("spy", "crime", "business", "history", "geopolitics", "mystery"):
        m = MD.niche_mix(n)
        ok(0.7 <= m["music_bed_mult"] <= 1.3, f"{n} bed mult bounded")
        ok(0.6 <= m["atmos_mult"] <= 1.2, f"{n} atmos mult bounded")
        ok(0.8 <= m["reveal_duck_scale"] <= 1.4, f"{n} reveal duck bounded")
    # crime should run a quieter/deeper bed than history (genre character)
    ok(MD.bed_mult("crime") < MD.bed_mult("history"),
       "crime bed quieter than history (genre character)")
    # default niche returns the proven base values (no behavioural change)
    d = MD.niche_mix("zzz_unknown_force_default")
    # unknown still maps to a canonical niche, so just assert it's a valid dict
    ok("music_bed_mult" in d, "unknown niche still returns a usable mix profile")


def test_sfx_restraint():
    print("sfx_director restraint")
    # silence-default text cards stay silent
    ok(SD.should_silence(gk="statement", kind="reveal"), "statement card defaults silent")
    ok(SD.should_silence(gk="quote_highlight", kind="word_pop"), "quote card defaults silent")
    ok(not SD.should_silence(gk="redacted", kind="stamp"), "redacted foley is NOT silent")
    # a rare deliberate accent overrides silence-default
    ok(not SD.should_silence(gk="statement", kind="reveal", is_accent=True),
       "accent beat overrides silence-default")
    # intensity is capped by the per-primitive max
    ok(SD.cap_intensity(0.95, gk="countdown_clock", kind="impact") <= 0.75 + 1e-6,
       "countdown capped at its max")
    ok(SD.cap_intensity(0.9, gk="statement", kind="reveal") < 0.5,
       "statement reveal capped low")
    # niche restraint: mystery caps lower + cools longer than business
    ok(SD.cap_intensity(0.9, gk="map_reveal", kind="map_pin", niche="mystery")
       <= SD.cap_intensity(0.9, gk="map_reveal", kind="map_pin", niche="business") + 1e-6,
       "mystery caps <= business")
    ok(SD.cooldown_s(gk="map_reveal", kind="map_pin", niche="mystery")
       >= SD.cooldown_s(gk="map_reveal", kind="map_pin", niche="business"),
       "mystery cools longer than business")
    # family routing: whoosh family detected; foley never mis-tagged as whoosh
    ok(SD.family_of("reveal") == "whoosh", "reveal is whoosh family")
    ok(SD.family_of("stamp", "redacted") == "foley_doc", "stamp is foley_doc")
    ok(SD.family_of("money_tick") == "foley_money", "money_tick is foley_money")


def test_cross_video_antirep():
    print("audio_usage_history cross-video anti-rep")
    hist = H._empty()
    H.record_video(hist, video_id="v1", niche="spy",
                   categories=["suspense", "dark_investigation"],
                   tracks=["suspense/a"], sfx_families=["whoosh"],
                   lead_category="suspense")
    # a recently-used category sinks to the back of the chain (never dropped)
    chain = ["suspense", "mystery", "dark_investigation", "ambient"]
    biased = H.filter_categories(hist, chain, gap=2)
    ok(set(biased) == set(chain), "no category dropped")
    ok(biased[0] != "suspense", "recently-used category sinks to back")
    ok(H.category_recently_used(hist, "suspense", 2), "suspense flagged recent")
    ok(not H.category_recently_used(hist, "ambient", 2), "unused category not flagged")
    # deterministic per-video seed: stable for same id, different across ids
    ok(H.video_seed("v1", "spy") == H.video_seed("v1", "spy"), "seed stable")
    ok(H.video_seed("v1", "spy") != H.video_seed("v2", "spy"), "seed varies per video")
    # re-recording the same video_id does not inflate history
    H.record_video(hist, video_id="v1", niche="spy", categories=["mystery"])
    ok(len(hist["videos"]) == 1, "re-render of same video updates in place")


def test_cue_sheets():
    print("cue-sheet builders")
    seg = [({"category": "suspense", "role": "hook", "start": 0, "end": 28,
             "events": [{"tier": 3}], "swell": True}, {"name": "cold-open"}, 28.0),
           ({"category": "aftermath", "role": "resolution", "start": 28, "end": 90,
             "events": []}, {"name": "settle"}, 62.0)]
    ms = MD.build_cue_sheet(seg, 90.0, niche="spy", video_id="vX")
    ok(ms["chapters"] == 2, "music cue sheet has 2 chapters")
    ok(ms["cues"][0]["is_intro"] and ms["cues"][-1]["is_outro"], "intro/outro flagged")
    ok(ms["lead_category"] == "suspense", "lead category captured")
    ok(ms["cues"][0]["reveal_tiers"]["tier3"] == 1, "reveal tier counted")
    evs = [{"time": 5, "kind": "reveal", "intensity": 0.5, "gk": "map_reveal"},
           {"time": 40, "kind": "stamp", "intensity": 0.6, "gk": "redacted"}]
    ss = SD.build_cue_sheet(evs, 90.0, niche="spy", video_id="vX")
    ok(ss["total_events"] == 2, "sfx cue sheet counts events")
    ok("whoosh" in ss["families_used"] and "foley_doc" in ss["families_used"],
       "sfx families classified")


def main():
    for t in (test_niche_normalize, test_intro_intelligence, test_mix_character,
              test_sfx_restraint, test_cross_video_antirep, test_cue_sheets):
        t()
    print(f"\n{_P} passed · {_F} failed")
    sys.exit(1 if _F else 0)


if __name__ == "__main__":
    main()
