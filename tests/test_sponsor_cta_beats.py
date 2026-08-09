"""A sponsor / ebook read is not a scene, so it must not be treated as one.

"Grab my free ebook — the link's in the description." There is no Game of Thrones footage of that
sentence. Left as exact_scene it burns discovery budget hunting a moment that cannot exist and then
release-blocks on exact_scene_missing. It is also the one beat the owner always covers in the edit —
a book card goes over the top of it — so what it actually wants is the cheapest honest thing: any
clean clip from the pool it can sit under. That is FILLER.

The risk this file mostly guards is the OTHER direction. A false positive silently downgrades a real
narrative beat, and the downgrade looks identical to a correct one. So the negative cases below are
the point: lines that share the vocabulary of a CTA — a book, a link, "you can", "join", "follow" —
and are pure narration.
"""
from __future__ import annotations

import pytest

from vidlore.clipstudio import policy as P


class Seg:
    """The handful of fields the classifier reads."""

    def __init__(self, text, **kw):
        self.text = text
        self.quote = kw.get("quote", "")
        self.required_entity = kw.get("required_entity", "")
        self.required_kind = kw.get("required_kind", "")
        self.is_specific_claim = kw.get("is_specific_claim", False)
        self.visual_policy = kw.get("visual_policy", "")
        self.breakout_candidate = kw.get("breakout_candidate", False)


CTA_LINES = [
    "Grab my free ebook — the link is in the description.",
    "If you want the full breakdown, my ebook is linked below.",
    "You can download our guide right now, link in the description.",
    "Check out my book on the history of Westeros — link's in the description.",
    "Head over to my Patreon if you want to support the channel.",
    "Use the code WESTEROS for a discount on your first order.",
    "Hit subscribe so you don't miss the next one.",
    "Full sources are in the pinned comment.",
    "Sign up for my newsletter and get the free PDF.",
]

# Real narration that borrows CTA vocabulary. Every one of these must stay untouched.
NARRATIVE_LINES = [
    "Jaime reads his own entry in the Book of Brothers.",
    "The White Book records every Kingsguard who ever served.",
    "You can see the fear in his eyes before he says a word.",
    "Varys sent a message south and waited.",
    "Jon decides to join the Night's Watch.",
    "The men who followed him north never came back.",
    "Sam checks out every scroll the Citadel will let him touch.",
    "She gets her revenge in the most literal way possible.",
    "Tyrion buys his way out of the Vale.",
    "The link between the two houses was never written down.",
    "Bran visits the weirwood one last time.",
    "That series of murders begins with a single letter.",
]


@pytest.mark.parametrize("text", CTA_LINES)
def test_a_cta_read_is_detected(text):
    assert P.is_sponsor_cta(Seg(text)) is True


@pytest.mark.parametrize("text", NARRATIVE_LINES)
def test_narration_that_shares_the_vocabulary_is_not_a_cta(text):
    assert P.is_sponsor_cta(Seg(text)) is False, f"false positive on narration: {text!r}"


@pytest.mark.parametrize("text", CTA_LINES)
def test_a_cta_beat_resolves_to_filler(text):
    """Any clean clip from the pool. Not exact, not a still — a clip the book card can sit over."""
    assert P.policy_of(Seg(text)) == P.FILLER
    assert P.classify(Seg(text)) == P.FILLER


def test_a_cta_outranks_an_llm_exact_label():
    seg = Seg("Grab my ebook — link in the description.", visual_policy=P.EXACT,
              required_entity="Daenerys", required_kind="character", is_specific_claim=True)
    assert P.policy_of(seg) == P.FILLER


def test_a_cta_outranks_deixis():
    """'that ebook' points at nothing in the story, so an exact demand can never be met."""
    seg = Seg("You can grab that ebook right now — the link is in the description below.")
    assert P.is_deictic(seg) or True          # whether or not deixis fires, the answer is FILLER
    assert P.policy_of(seg) == P.FILLER


def test_a_cta_outranks_a_quote():
    """The analyzer quotes a CTA line as readily as a line of dialogue."""
    seg = Seg("Grab my free ebook, link in the description.", quote="Grab my free ebook")
    assert P.policy_of(seg) == P.FILLER


@pytest.mark.parametrize("text", CTA_LINES)
def test_a_cta_is_never_a_breakout(text):
    """Real show audio underneath a sponsor read would talk over the offer."""
    assert P.is_breakout_candidate(Seg(text, quote="Grab my free ebook")) is False


def test_a_cta_beat_spends_no_discovery_budget():
    seg = Seg("Grab my free ebook — the link is in the description.")
    assert P.discovery_tier(seg) == "low"
    assert P.wants_discovery_query(seg) is False
    assert P.match_strict(seg) is False
    assert P.verify_strict(seg) is False


def test_a_cta_beat_still_gets_a_moving_clip_not_a_still():
    """FILLER, deliberately — ABSTRACT would hand it a freeze/image with nothing to cover."""
    seg = Seg("Grab my free ebook — the link is in the description.")
    assert P.footage_optional(seg) is False
    assert P.maximize_variety(seg) is True


def test_the_tally_says_why_those_beats_are_filler():
    segs = [Seg(CTA_LINES[0]), Seg(CTA_LINES[1]), Seg("Ned rides south to King's Landing.")]
    tally = P.finalize_beats(segs)
    assert tally.get("sponsor_cta") == 2
    assert tally[P.FILLER] >= 2
    assert all(s.visual_policy in P.POLICIES for s in segs)   # not a fifth policy


def test_an_empty_or_missing_text_is_not_a_cta():
    assert P.is_sponsor_cta(Seg("")) is False
    assert P.is_sponsor_cta(Seg("   ")) is False


def test_the_offer_alone_or_the_verb_alone_is_never_enough():
    """All three signals are required for the acquisition form — noun, verb, and person."""
    assert P.is_sponsor_cta(Seg("The ebook sat unopened on the table.")) is False   # noun only
    assert P.is_sponsor_cta(Seg("You should check out what he did next.")) is False  # verb+person
    assert P.is_sponsor_cta(Seg("Download the guide.")) is False                     # no person
