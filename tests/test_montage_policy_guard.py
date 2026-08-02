"""A beat that names four people in four scenes is not an exact scene.

Job 6a26707939 died at the release gate with four unresolved beats. One of them, beat 114, read
"and she makes that identification three separate times about three separate people" and carried
required_kind="montage", required_entity="Melisandre, Stannis, Jon Snow, Daenerys Targaryen",
visual_policy=exact_scene. No single window in any source can satisfy that, so every candidate
failed the verifier, the still/web fallback ladder found nothing, and the render refused to publish.

Two things went wrong and both are general:

  * "montage" is not one of the values the analyzer prompt asks for
    (actor|character|object|scene|event|location) — analyze.py stores whatever the model replies,
    with no validation, so an unknown kind becomes a demand nothing downstream can read.
  * A requirement naming three or more subjects cannot be a single exact scene, whatever the kind
    string says.

The demotion is to CHARACTER rather than ABSTRACT on purpose. Showing ONE of the named people is an
honest visual for that sentence; dropping to an abstract effect would throw away a specificity the
beat genuinely has.
"""
from __future__ import annotations

import pytest

from vidlore.clipstudio import policy as P


class Seg:
    def __init__(self, **kw):
        self.index = kw.pop("index", 0)
        self.text = kw.pop("text", "")
        self.quote = kw.pop("quote", "")
        self.required_kind = kw.pop("required_kind", "")
        self.required_entity = kw.pop("required_entity", "")
        self.visual_policy = kw.pop("visual_policy", "")
        self.is_specific_claim = kw.pop("is_specific_claim", False)
        for k, v in kw.items():
            setattr(self, k, v)


# ------------------------------------------------------------------ the beat that blocked release
def test_the_montage_beat_is_no_longer_an_exact_scene_demand():
    seg = Seg(index=114,
              text="and she makes that identification three separate times about three separate "
                   "people.",
              required_kind="montage",
              required_entity="Melisandre, Stannis, Jon Snow, Daenerys Targaryen",
              visual_policy="exact_scene")
    assert P.policy_of(seg) == P.CHARACTER


def test_it_is_demoted_to_character_not_thrown_away():
    """CHARACTER keeps the beat pointed at the right people; ABSTRACT would license anything."""
    seg = Seg(required_kind="montage", required_entity="A, B, C, D", visual_policy="exact_scene",
              text="three separate people")
    assert P.policy_of(seg) not in (P.ABSTRACT, P.FILLER)


# ------------------------------------------------------------------ what must NOT change
@pytest.mark.parametrize("entity,text", [
    ("Melisandre, Jon Snow", "Then Stannis dies, and the name moves to Jon Snow."),
    ("Melisandre and Stannis", "There is a fair objection that none of this was ever a mistake."),
    ("Melisandre, Daenerys Targaryen", "the same prophecy is restated in front of Daenerys."),
])
def test_a_two_hander_stays_an_exact_scene(entity, text):
    """Measured on the same job: 3 two-entity beats, every one a real scene with both people in it.
    Two names must never be read as a montage."""
    seg = Seg(required_kind="character", required_entity=entity, visual_policy="exact_scene",
              text=text)
    assert P.policy_of(seg) == P.EXACT


def test_a_single_named_subject_is_untouched():
    seg = Seg(required_kind="object", required_entity="flayed man banner",
              visual_policy="exact_scene", text="The flayed man banners brought down to the ground.")
    assert P.policy_of(seg) == P.EXACT


def test_the_guard_does_not_promote_anything():
    """Shorten-only: this rule may lower a policy, never raise one."""
    seg = Seg(required_kind="montage", required_entity="A, B, C", visual_policy="generic_filler",
              text="filler")
    assert P.policy_of(seg) == P.FILLER


# ------------------------------------------------------------------ the unvalidated-enum half
@pytest.mark.parametrize("kind", ["montage", "sequence", "collage", "supercut"])
def test_a_kind_outside_the_documented_enum_cannot_demand_an_exact_scene(kind):
    """analyze.py stores required_kind unvalidated, so the prompt's enum is a promise, not a rule.
    An unreadable kind must fail SAFE (a looser policy), never into an unsatisfiable demand."""
    seg = Seg(required_kind=kind, required_entity="Melisandre", visual_policy="exact_scene",
              text="whatever")
    assert P.policy_of(seg) == P.CHARACTER


@pytest.mark.parametrize("kind", ["actor", "character", "object", "scene", "event", "location", ""])
def test_every_documented_kind_still_reaches_exact_scene(kind):
    seg = Seg(required_kind=kind, required_entity="Melisandre", visual_policy="exact_scene",
              text="Melisandre burns Shireen outside Winterfell.")
    assert P.policy_of(seg) == P.EXACT


def test_deixis_still_outranks_the_montage_guard():
    """A pointing beat is exact by objective evidence — that outranks every label heuristic."""
    seg = Seg(required_kind="montage", required_entity="A, B, C",
              visual_policy="exact_scene", text="watch what happens at that table")
    assert P.policy_of(seg) == P.EXACT
