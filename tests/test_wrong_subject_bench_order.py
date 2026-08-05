"""When the verifier says the wrong person is on screen, look at the right person first.

Measured, job ee93371e41 beat 134. The beat needs Shae (required_entity 'Shae'); the verifier
returned correct_subject_visible=False and wrong_subject_visible=True on a pick taken from an
Oberyn Martell compilation. Its bench was NOT missing the footage — 24 of the beat's 61 candidates
come from Shae/Tywin-titled sources, including "Betrayal of Shae with Tyrion Lannister | S04E06".
The bench was simply ordered by a ranking that knows nothing about the verdict just returned, so
the first three candidates the strict bar saw were Oberyn-vs-the-Mountain, Tywin-talks-to-Jaime and
Tywin-and-Oberyn, and the beat shipped the compilation.

This is the same shape as the look-miss rule that already sits beside it: when the narration points
at something the verifier could not see, bench candidates whose CLIP probe DID see it are surfaced
first. Ordering only, in both cases — `_try_promote` applies the identical strict bar to whatever
it is handed, so this changes which candidate is examined first and can never admit one that would
otherwise be refused.

Affinity uses two independent signals so a mislabelled title alone cannot carry a candidate: the
source title naming the subject, and the candidate's own match-time face/identity signals. Beats
name CHARACTERS while Face-ID and many titles name ACTORS, so the project roster is applied first
or "Shae" would never match a title reading "Sibel Kekilli".
"""
from __future__ import annotations

import inspect

import pytest

from vidlore.clipstudio import verify as V


class Seg:
    def __init__(self, entity="Shae", kind="character"):
        self.required_entity = entity
        self.required_kind = kind


class Cand:
    def __init__(self, source_id="", signals=None):
        self.source_id = source_id
        self.signals = signals or {}


class Proj:
    """Minimal stand-in: _project_source_title reads proj.source(id).title."""
    def __init__(self, titles):
        self._t = titles

    def source(self, sid):
        t = self._t.get(sid)
        return type("S", (), {"title": t})() if t is not None else None


# ------------------------------------------------------------------ subject terms
def test_the_character_name_becomes_a_search_term():
    assert "shae" in V._subject_terms(Seg("Shae"), {})


def test_the_actor_name_is_included_via_the_roster():
    """Face-ID and uploaders name actors; the beat names a character."""
    terms = V._subject_terms(Seg("Shae"), {"shae": "Sibel Kekilli"})
    assert "shae" in terms and "sibel" in terms and "kekilli" in terms


def test_a_multi_name_requirement_contributes_every_name():
    terms = V._subject_terms(Seg("Jaime Lannister, Varys"), {})
    assert {"jaime", "lannister", "varys"} <= terms


def test_short_fragments_are_ignored():
    """One- and two-letter fragments match almost every title."""
    assert not any(len(t) < 3 for t in V._subject_terms(Seg("Ed Ng"), {}))


def test_a_beat_with_no_named_subject_yields_no_terms():
    assert V._subject_terms(Seg("", "object"), {}) == set()


def test_an_object_requirement_is_not_treated_as_a_person():
    assert V._subject_terms(Seg("flayed man banner", "object"), {}) == set()


def test_a_montage_kind_still_resolves_its_named_people():
    """beat 134's own kind is 'montage' — outside the analyzer's documented enum — and it still
    names a person who has to be found."""
    assert "shae" in V._subject_terms(Seg("Shae", "montage"), {})


# ------------------------------------------------------------------ affinity
def test_a_title_naming_the_subject_outranks_one_that_does_not():
    proj = Proj({"a": "Betrayal of Shae with Tyrion Lannister S04E06",
                 "b": "Oberyn Martell vs. The Mountain"})
    hi = V._subject_affinity(Cand("a"), {"shae"}, proj)
    lo = V._subject_affinity(Cand("b"), {"shae"}, proj)
    assert hi > lo and lo == 0.0


def test_face_identity_signals_count_independently_of_the_title():
    """So a clip whose uploader never wrote the name can still surface on face evidence."""
    proj = Proj({"a": "Game of Thrones S04E06"})
    assert V._subject_affinity(Cand("a", {"identity": "Sibel Kekilli"}),
                               {"sibel", "kekilli"}, proj) > 0


def test_no_terms_means_no_opinion():
    proj = Proj({"a": "anything at all"})
    assert V._subject_affinity(Cand("a"), set(), proj) == 0.0


def test_affinity_never_returns_a_negative_score():
    proj = Proj({"a": "unrelated"})
    assert V._subject_affinity(Cand("a"), {"shae"}, proj) >= 0.0


# ------------------------------------------------------------------ wiring
def test_the_bench_is_reordered_only_when_the_subject_was_reported_absent():
    src = inspect.getsource(V.verify_and_repair) if hasattr(V, "verify_and_repair") \
        else inspect.getsource(V)
    assert 'v.get("correct_subject_visible") is False' in src, \
        "the reorder must key on the verdict that was actually returned"


def test_it_is_ordering_only_and_never_admits_a_candidate():
    """The strict bar stays exactly where it was: the sort feeds `_try_promote`, which decides."""
    src = inspect.getsource(V)
    i = src.index("if _bench and _wrong_subject:")
    j = src.index("_subject_affinity(c, _want, proj), reverse=True)", i)
    reorder = src[i:j]
    assert "_bench.sort(" in reorder
    assert "_try_promote" not in reorder, \
        "the reorder block must only sort — it must not admit, drop or judge anything"
    assert "_try_promote(downgrade=False, pool=_bench)" in src[j:j + 400], \
        "the reordered bench must then go through the SAME strict promotion as before"


def test_the_look_miss_ordering_is_left_intact():
    """The two rules are siblings; adding one must not disturb the other."""
    src = inspect.getsource(V)
    assert 'v.get("target_visible") is False' in src
    assert '"target_vis"' in src


# ------------------------------------------------------------------ what the reorder could not fix
#
# MEASURED on job ee93371e41 beat 134, on real extracted frames — not on titles:
#   * the shipped pick (got_oberyn_martell_the_582c9993 @173.94) is a warm daylight Tyrion close-up:
#     no Shae, no bed, no cell
#   * every one of the 60 bench candidates was viewed; NONE shows Shae or Tywin's bedchamber, so no
#     ordering of that bench can satisfy the beat
#   * the shot that literally shows "Shae in his father's bed" (tyrion_kills_tywin_lan_47364efd,
#     0.000-9.343, 720p, clean) sits at CLIP rank 563 of 4942 for this beat's query because it is a
#     near-black night interior
#
# The bench is match's one-candidate-per-source ranking over the top CLIP-ranked sources, so a shot
# retrieval cannot see is structurally absent from it. Sorting it better cannot help.
def test_a_wrong_subject_failure_does_not_stop_at_the_bench():
    """It falls through to pools built by scene affinity rather than by CLIP rank."""
    src = inspect.getsource(V)
    i = src.index("THE BENCH CANNOT HOLD WHAT RETRIEVAL COULD NOT SEE")
    tail = src[i:i + 3600]
    assert "_strict_scene_neighborhood_candidates(" in tail
    assert "_venue_candidates(" in tail


def test_the_fallthrough_uses_the_identical_strict_bar():
    """downgrade=False — this changes what is EXAMINED, never what is admitted."""
    src = inspect.getsource(V)
    i = src.index("THE BENCH CANNOT HOLD WHAT RETRIEVAL COULD NOT SEE")
    tail = src[i:i + 3600]
    assert "_try_promote(downgrade=False, pool=_pool," in tail
    assert "downgrade=True" not in tail, "a rescue must not buy itself a downgrade"


def test_a_pool_that_cannot_be_built_fails_closed():
    src = inspect.getsource(V)
    i = src.index("THE BENCH CANNOT HOLD WHAT RETRIEVAL COULD NOT SEE")
    tail = src[i:i + 3600]
    assert "except Exception" in tail and "continue" in tail


def test_the_fallthrough_only_runs_on_a_wrong_subject_verdict():
    src = inspect.getsource(V)
    i = src.index("THE BENCH CANNOT HOLD WHAT RETRIEVAL COULD NOT SEE")
    assert "if _wrong_subject:" in src[i:i + 3600]


# ------------------------------------------------------------------ a title is not a frame
#
# _subject_affinity documents two independent signals "so a mislabelled title alone cannot carry a
# candidate". That holds only where the second signal CAN fire. On beat 134 the project roster has
# 9 characters and Shae is not among them, so the identity arm is 0.0 on every candidate, the score
# collapses to the title arm, and the three candidates it scored 1.0 are Tywin on the Iron Throne.
def test_a_title_only_match_cannot_outrank_an_identity_corroborated_one():
    class C:
        def __init__(self, sid, sig=None):
            self.source_id = sid
            self.signals = sig or {}

    class P:
        def source(self, sid):
            return type("S", (), {"title": "Shae and Tyrion — every scene"})() \
                if sid == "titled" else type("S", (), {"title": "some compilation"})()

    proj = P()
    titled = C("titled")                                   # title says Shae, nothing saw her
    seen = C("other", {"identity": "shae"})                # the pipeline saw her
    assert V._subject_affinity(seen, {"shae"}, proj) > V._subject_affinity(titled, {"shae"}, proj)


def test_a_title_only_match_still_counts_for_something():
    """Weak evidence is not no evidence — it must still order above a candidate with neither."""
    class C:
        def __init__(self, sid):
            self.source_id = sid
            self.signals = {}

    class P:
        def source(self, sid):
            return type("S", (), {"title": "Shae scenes" if sid == "titled" else "unrelated"})()

    proj = P()
    assert V._subject_affinity(C("titled"), {"shae"}, proj) > \
        V._subject_affinity(C("nope"), {"shae"}, proj) == 0.0


def test_the_title_only_weight_is_named_and_below_one():
    assert 0.0 < V._TITLE_ONLY_AFFINITY < 1.0
