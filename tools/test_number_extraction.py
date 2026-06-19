"""Regression — director stat-number extraction (the 96%-shown-as-64% class).

Verifies the director derives the CORRECT numeric payload from narration, that
the scene's `emphasis` disambiguates a multi-percentage line, and that the
motion-graphics cache key is sensitive to the numeric value (so a 96% card can
never reuse a cached 64% render). Run:  python3 tools/test_number_extraction.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vidlore.motion_graphics import director as D
from vidlore.motion_graphics import render_dispatch as RD


def _derive(scene):
    """Return the (value, suffix) the director derives for a scene, via the
    real _best_for_scene path (assets win, else narration/emphasis derived)."""
    _pid, _s, ins = D._best_for_scene(scene, "science")
    return ins.get("value"), ins.get("suffix"), ins.get("share")


def case(name, scene, want_value, *, allow_share=False):
    val, suf, share = _derive(scene)
    got = val if val is not None else (share if allow_share else None)
    ok = (got == want_value)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: got value={val} share={share} "
          f"suffix={suf!r}  (want {want_value})")
    return ok


def main():
    results = []
    # 1) single 96% — the headline regression
    results.append(case("96% single", {
        "narration": "A controlled study confirmed a 96% reduction in pests.",
        "emphasis": "96%", "keywords": ["study"]}, 96.0, allow_share=True))
    # 2) single 64% — must extract 64, not anything else
    results.append(case("64% single", {
        "narration": "Only a 64% success rate was observed in the trial.",
        "emphasis": "64%", "keywords": ["trial"]}, 64.0, allow_share=True))
    # 3) TWO percentages in one line — emphasis must disambiguate to 96
    results.append(case("two-pct emphasis=96", {
        "narration": "Not 40, not 60 — the researchers measured 96% in the field.",
        "emphasis": "96%", "keywords": ["research"]}, 96.0, allow_share=True))
    # 4) two percentages, emphasis picks the SECOND (60->96 but land on 60)
    results.append(case("two-pct emphasis=60", {
        "narration": "It rose from 60% to 96% over the decade.",
        "emphasis": "60%", "keywords": ["data"]}, 60.0, allow_share=True))
    # 5) percentage PLUS a year — must pick the %, never the year 1994
    results.append(case("pct + year", {
        "narration": "In 1994 a university study found a 96% reduction.",
        "emphasis": "96%", "keywords": ["study"]}, 96.0, allow_share=True))

    # 6) cache key is value-sensitive: 96 vs 64 -> different keys (no stale reuse)
    k96 = RD.cache_key("gold_number_callout", {"value": 96.0, "suffix": "%"}, 4.0)
    k64 = RD.cache_key("gold_number_callout", {"value": 64.0, "suffix": "%"}, 4.0)
    ok_cache = (k96 != k64)
    print(f"  [{'PASS' if ok_cache else 'FAIL'}] cache key value-sensitive: "
          f"96->{k96} 64->{k64}")
    results.append(ok_cache)

    # 7) director-injected number card actually fires + carries the value
    pid, score, ins = D._best_for_scene({
        "narration": "The Cornell study confirmed a 96% reduction in damage.",
        "emphasis": "96%", "keywords": ["cornell", "study"],
        "graphic_kind": ""}, "science")
    fired = pid in ("gold_number_callout", "proportion_ring",
                    "statistic_bar_reveal", "pictograph_scale")
    carries = (ins.get("value") == 96.0 or ins.get("share") == 96.0)
    print(f"  [{'PASS' if (fired and carries) else 'FAIL'}] injected number card: "
          f"primitive={pid} value={ins.get('value')} share={ins.get('share')}")
    results.append(fired and carries)

    n_pass = sum(1 for r in results if r)
    print(f"\n{n_pass}/{len(results)} number-extraction regression cases passed")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
