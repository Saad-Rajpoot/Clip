#!/usr/bin/env python3
"""No-API regression suite for Vidlore measurement / scoring helpers (MNT_4).

Pure deterministic checks — no renders, no LLM, no network — so the
benchmark-honesty + duration work (IMP_023/025/026/029, MNT_1/3) can't silently
regress. Run: python3 tools/test_maintenance.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_passed, _failed = 0, 0


def check(name, got, want):
    global _passed, _failed
    ok = got == want
    if ok:
        _passed += 1
    else:
        _failed += 1
        print(f"  FAIL {name}: got {got!r}, want {want!r}")
    return ok


def approx(name, got, lo, hi):
    global _passed, _failed
    ok = (got is not None) and (lo <= got <= hi)
    _passed += int(ok)
    _failed += int(not ok)
    if not ok:
        print(f"  FAIL {name}: {got!r} not in [{lo}, {hi}]")


# ── IMP_025 — spelled-out magnitude detection (precision) ────────────────
from vidlore.assemble import _spelled_to_number as s2n, _best_stat_figure as bsf
check("s2n million", s2n("four hundred and twenty million dollars"), 420000000)
check("s2n thousand", s2n("more than four thousand people"), 4000)
check("s2n billion", s2n("two billion dollars"), 2000000000)
check("s2n year-reject", s2n("on December second, nineteen ninety-three"), None)
check("s2n ordinal-reject", s2n("his forty-fourth birthday"), None)
check("s2n count-reject", s2n("three judges resigned"), None)
check("s2n vague-reject", s2n("hundreds of people fled"), None)
check("bsf picks-largest", bsf("130,000 troops and four thousand tanks"), "130,000")
check("bsf spelled", bsf("four hundred and twenty million"), "420,000,000")
check("bsf none", bsf("no numbers here"), "")
# MNT_10 — descending magnitudes must ADD a new group, not re-multiply the
# accumulator (old bug: "two million five hundred thousand" -> 2,000,500,000)
check("s2n million+thousand", s2n("two million five hundred thousand"), 2_500_000)
check("s2n million+thousand b", s2n("one million two hundred thousand"), 1_200_000)
check("s2n billion+million", s2n("two billion five hundred million"), 2_500_000_000)
check("s2n hundred-thousand", s2n("three hundred thousand"), 300_000)
check("s2n multi-pick-largest",
      s2n("four thousand troops and two million dollars"), 2_000_000)
check("bsf descending-mag", bsf("two million five hundred thousand"), "2,500,000")

# ── benchmark scoring honesty (MNT_1/MNT_3) ──────────────────────────────
import importlib
import tools.benchmark_engine as B
importlib.reload(B)
_sc = [{"intensity": 3} for _ in range(38)]
# pacing: real beats (123 over 391.5s = 3.18s) -> in 3-6 band -> ~10
approx("pacing beats-groundtruth", B._score_pacing(_sc, 391.5, 0, 123), 9.0, 10.0)
# pacing fallback: few script scenes -> still uses cut_count when given
approx("pacing cuts-fallback", B._score_pacing(_sc, 391.5, 110, 0), 9.0, 10.0)
# cost_efficiency: None when render time unknown
check("cost None@unknown", B._score_cost_efficiency(ROOT, 0.0), None)
# weighted_overall excludes None dims (no crash, re-normalises)
_w = B._weighted_overall({"audio_mix": 10.0, "pacing": 10.0,
                          "cost_efficiency": None})
approx("overall skips None", _w, 9.5, 10.0)
# graphics_quality: non-banned templates count as good (currency_stat/typing_date)
approx("graphics non-banned good",
       B._score_graphics_quality([{"graphic_kind": "currency_stat"},
                                  {"graphic_kind": "typing_date"},
                                  {"graphic_kind": "document"}]), 9.0, 10.0)
# render_meta loader returns {} when absent
check("meta absent -> {}", B._load_render_meta(ROOT / "nope.mp4", ROOT / "nope"), {})

# ── IMP_026/029 — duration bucket projections land in requested range ────
from vidlore.brief import DURATION_BUCKETS as DB, WORDS_PER_SECOND as WPS
for key, spec in DB.items():
    lo, hi = (float(x) for x in key.split("-"))
    mins = spec["words"] / WPS / 60.0
    approx(f"bucket {key} centres in range", mins, lo - 0.6, hi + 0.6)
    # scenes decoupled from points (more scenes than acts)
    check(f"bucket {key} scenes>=points",
          spec["scenes"][0] >= spec["points"][0], True)

# ── MNT_7 — render-log -> structured-metrics parser ──────────────────────
from tools.parse_render_log import parse_log
_LOG = (
    "  [script] 38 scenes, ~1206 words\n"
    "  [2/5] narration 391.5s audio (0/38 scenes reused) · TTS wall 6.2s\n"
    "  [NIM] active niche: true_crime (cold case) — 6 topic-word hits\n"
    "  [3/5] ai usage: 20/20 (26.0% of 77 beats) · priority_target:20\n"
    "  graphic art: 30 card images baked\n"
    "  transitions: 9 motivated (dissolve×6, whip×3)\n"
    "  [5/5] scored music: 11 cues (118-track library)\n"
    "  [5/5] 123 beats from 38 scenes · hard cuts · shot len 1.8-12.4s\n"
    "  [5/5] black-frame repair: 1 span(s) detected — freeze-holding\n"
    "  [5/5] black-frame repair: clean (0 black spans)\n"
    "  assembly wall 497.9s\n"
    "  [web-footage] decisions: 2 selected\n"
    "  WARNING: pexels clip low-res, fell back to AI\n")
_m = parse_log(_LOG)
check("parse scenes", _m["scenes"], 38)
check("parse words", _m["script_words"], 1206)
check("parse beats (final line)", _m["beats"], 123)
check("parse narration", _m["narration_audio_s"], 391.5)
check("parse card images", _m["graphic_card_images"], 30)
check("parse transitions", _m["transitions_motivated"], 9)
check("parse music cues", _m["music_cues"], 11)
check("parse black detected", _m["black_spans_detected"], 1)
check("parse black after-repair", _m["black_spans_after_repair"], 0)
check("parse assembly wall", _m["assembly_wall_s"], 497.9)
check("parse shot-len min", _m["shot_len_min_s"], 1.8)
check("parse web-footage src", _m["sources"]["web_footage_selected"], 2)
check("parse warn captured", _m["warning_error_count"] >= 1, True)
# empty log must not crash and yields Nones / zero counts
_e = parse_log("")
check("parse empty scenes None", _e["scenes"], None)
check("parse empty black 0", _e["black_spans_detected"], 0)

# ── MNT_8 — QA verdict gate (aligned to benchmark constants) ──────────────
from tools.parse_render_log import qa_verdict
_clean = {"scenes": 38, "black_spans_after_repair": 0,
          "audio": {"lufs": -16.0, "true_peak_db": -1.9},
          "warnings_errors": [], "warning_error_count": 0,
          "card_types": {"a": 5}, "narration_audio_s": 391.5}
check("qa clean PASS", qa_verdict(_clean)["verdict"], "PASS")
# residual black frames -> FAIL
_bf = dict(_clean, black_spans_after_repair=2)
check("qa residual-black FAIL", qa_verdict(_bf)["verdict"], "FAIL")
# clipping (peak >= 0) -> FAIL
_clip = dict(_clean, audio={"lufs": -16.0, "true_peak_db": 0.3})
check("qa clip FAIL", qa_verdict(_clip)["verdict"], "FAIL")
# loudness slightly off -> WARN, way off -> FAIL
check("qa lufs-warn", qa_verdict(dict(_clean,
      audio={"lufs": -12.0, "true_peak_db": -1.9}))["verdict"], "WARN")
check("qa lufs-fail", qa_verdict(dict(_clean,
      audio={"lufs": -8.0, "true_peak_db": -1.9}))["verdict"], "FAIL")
# duration off-target (the IMP_026 bug class) -> FAIL at 60% off
check("qa duration FAIL", qa_verdict(
      dict(_clean, narration_audio_s=143.2), target_minutes=6)["verdict"], "FAIL")
check("qa duration on-target PASS", qa_verdict(
      _clean, target_minutes=7)["verdict"], "PASS")
# hard error line -> FAIL
check("qa traceback FAIL", qa_verdict(dict(_clean,
      warnings_errors=["Traceback (most recent call last):"],
      warning_error_count=1))["verdict"], "FAIL")
# missing scene count (truncated/aborted) -> FAIL
check("qa no-scenes FAIL", qa_verdict(dict(_clean, scenes=None))["verdict"], "FAIL")
# graphics clutter -> WARN; but a healthy ~39% density must NOT false-alarm
check("qa gfx-clutter WARN", qa_verdict(dict(_clean,
      scenes=10, card_types={"a": 9}))["verdict"], "WARN")
check("qa gfx-healthy PASS", qa_verdict(dict(_clean,
      scenes=38, card_types={"a": 15}))["verdict"], "PASS")

# ── IMP_026 duration gate — pure (no-API) contract checks ────────────────
from types import SimpleNamespace
from vidlore.script_gen import _count_words, _validate_and_expand
_scn = [SimpleNamespace(narration="one two three", index=0),
        SimpleNamespace(narration="four five", index=1),
        SimpleNamespace(narration="", index=2)]
check("count_words sums narration", _count_words(_scn), 5)
check("count_words handles None",
      _count_words([SimpleNamespace(narration=None)]), 0)
# without an LLM the gate must short-circuit and never crash / mutate
_brief = SimpleNamespace(target_words=300, target_minutes=(2.0, 3.0),
                         target_points=(2, 3))
_nollm = SimpleNamespace(has_llm=False)
check("gate no-LLM returns same list",
      _validate_and_expand(list(_scn), _brief, _nollm), _scn)
# CONTRACT (documents IMP_030 gap): a too-LONG script is returned UNCHANGED —
# compress-if-long is not yet implemented (needs an LLM condense pass, API).
# When IMP_030 lands this expectation must change.
_long = [SimpleNamespace(narration=" ".join(["w"] * 200), index=i)
         for i in range(6)]                     # 1200w vs 300 target = 4x
check("gate long-script unchanged (IMP_030 pending)",
      len(_validate_and_expand(list(_long), _brief, _nollm)), 6)

# ── MNT_9 — VO-synced reveal time for news_article (was fixed-offset) ─────
from vidlore.assemble import _reveal_time, _VO_REVEAL_KINDS
_W = [SimpleNamespace(word="Pablo", start=1.0),
      SimpleNamespace(word="Escobar", start=1.4),
      SimpleNamespace(word="billionaire.", start=2.3)]
check("news_article in VO-reveal set", "news_article" in _VO_REVEAL_KINDS, True)
check("document still in VO-reveal set", "document" in _VO_REVEAL_KINDS, True)
check("reveal_time matches word", _reveal_time("Escobar", _W), 1.4)
check("reveal_time punct-insensitive", _reveal_time("billionaire", _W), 2.3)
check("reveal_time no-match -> -1", _reveal_time("Medellin", _W), -1.0)
check("reveal_time empty-emph -> -1", _reveal_time("", _W), -1.0)
check("reveal_time empty-words -> -1", _reveal_time("Escobar", None), -1.0)

print(f"\n{'PASS' if _failed == 0 else 'FAIL'} — {_passed} passed, {_failed} failed")
sys.exit(1 if _failed else 0)
