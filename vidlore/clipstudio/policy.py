"""Unified per-beat VISUAL POLICY — the single source of truth for how each script beat is treated
across EVERY stage (discovery, matching, verification, breakouts, image/effect fallback, QC).

The owner's rule: relevance is contextual, not blanket. A line that names a precise scene/quote/
action demands the EXACT footage; a generic line only needs relevant filler; an abstract/emotional
line wants an image/effect, not a literal clip. This module classifies each beat into ONE policy
and exposes per-stage helpers so the whole pipeline reasons from the same decision.

Four mutually-exclusive treatments (`ScriptSegment.visual_policy`):
  exact_scene        — a specific scene / quote / character action / plot event. STRICT matching
                       (dialogue/ASR + Face-ID + CLIP + scene-query + verifier). ONLY real matching
                       footage or an EXACT source-frame screenshot may air — never a web/AI image or
                       a loose filler. If none is found → flag exact_scene_missing → manual review
                       (NEVER silently covered by weak filler). Aggressive discovery, high budget.
  character_specific — about a named character/object/place in general (not a precise moment). Needs
                       the right subject on screen (Face-ID matters) but any clean shot. Medium budget.
  generic_filler     — generic/explanatory narration. Any thematically-relevant clip is fine; source
                       VARIETY is actively maximised. Low budget / reuse the existing pool.
  abstract_effect    — abstract/emotional/meta line with no literal visual. Prefer image/effect/
                       freeze/B-roll over downloading footage.

`breakout_candidate` is an ORTHOGONAL flag (composes with any policy): a beat that quotes an iconic
line/action is a candidate for a real-audio breakout.
"""
from __future__ import annotations

import re

EXACT = "exact_scene"
CHARACTER = "character_specific"
FILLER = "generic_filler"
ABSTRACT = "abstract_effect"
POLICIES = (EXACT, CHARACTER, FILLER, ABSTRACT)

# abstract / emotional / meta commentary with NO concrete on-screen subject → image/effect, not a clip
_ABSTRACT_RX = re.compile(
    r"\b(imagine|think about|ask yourself|in the end|the truth is|the point is|the lesson|"
    r"what (this|that|it) (means|tells|teaches|says|reveals)|sit with (that|this)|let (that|this) sink|"
    r"this is what|that'?s the (point|truth|tragedy|irony)|it'?s (really )?about|"
    r"represent|symboli[sz]|a metaphor|stands for|"
    r"everything chang(es|ed)|nothing (would|was) (ever )?(be )?the same|never the same|"
    r"the beginning of the end|and that'?s (why|how|exactly)|here'?s the (thing|truth|point)|"
    r"matters? (because|is)|the real (reason|question|tragedy))\b", re.I)


# DEIXIS — a pointing word bound to a scene noun ("at THAT TABLE", "in THOSE NINETY SECONDS",
# "THERE IT IS"). The referent is the anchor scene, so the line is as specific as a named one even
# though it names nothing: the essayist is pointing at the moment on screen. Measured failure this
# guards: "The point is what everyone else at that table heard" shipped as generic_filler over Ned
# Stark's execution, and "Somewhere in those ninety seconds" as abstract_effect over an inn brawl —
# both because deixis reads as vague to a keyword classifier. Worse, _ABSTRACT_RX below matches
# "the point is", so deictic lines were actively pushed AWAY from exact.
#
# Deliberately TIGHT: a pointing determiner (or a fixed phrase) plus a noun from a closed SCENE
# vocabulary, within 2 intervening words ("those NINETY seconds"). Bare anaphora ("that's the
# tragedy", "this is why") must NOT match, or genuinely generic beats lose their filler.
_DEICTIC_RX = re.compile(
    r"\b(?:that|this|those|these)\s+(?:\w+\s+){0,2}"
    r"(?:tables?|rooms?|chambers?|scenes?|moments?|meetings?|councils?|seconds?|minutes?|"
    r"exchanges?|conversations?|sentences?|lines?|pauses?|silences?|dinners?|suppers?|feasts?)\b"
    r"|\bthere it is\b|\bright there\b|\bwatch it again\b", re.I)


# RHETORICAL CONNECTORS — transition/question lines whose only job is moving the argument along:
# "And that raises the next obvious question." / "Where does it even come from?" / "the answer
# stops being obvious fast". They have no scene, no era, no character — there is nothing exact TO
# show. Measured failure this guards: the analyzer LLM labelled 11 such lines `exact_scene` in one
# render; each demanded footage that cannot exist, failed the verifier, ground through the entire
# still/web fallback ladder (~7 hours of vision calls) and still release-blocked the video. The
# competitor covered the same lines with 2s of related filler and moved on.
_RHETORIC_RX = re.compile(
    r"\b(?:raises? the (?:next |obvious |same )*question|the answer (?:stops|is|isn'?t|starts)|"
    r"where does (?:it|that|this|the \w+) (?:even )?come from|"
    r"which brings us (?:back )?to|(?:but|and|so) (?:that|this|it) (?:changes|leaves|means)|"
    r"(?:let'?s|we need to) (?:start|go back|look|talk about)|"
    r"(?:that|this|it)'?s not a (?:gap|coincidence|accident|mystery)|"
    r"(?:here|now) (?:is|'?s) (?:where|why|what|the)|start with)\b", re.I)


def _is_rhetorical_connector(seg) -> bool:
    """A connector has NO concrete hook: no quote, no required entity, and either a connector
    phrase or a short pure question. Any concrete hook keeps the beat out of this bucket."""
    if (getattr(seg, "quote", "") or "").strip():
        return False
    if (getattr(seg, "required_entity", "") or "").strip():
        return False
    t = (getattr(seg, "text", "") or "").strip()
    if _RHETORIC_RX.search(t):
        return True
    # a short pure question with no entity is asking the AUDIENCE, not describing a scene
    return t.endswith("?") and len(t.split()) <= 10


def is_deictic(seg) -> bool:
    """True when the beat POINTS at the anchor scene instead of naming it. Such a beat inherits the
    exact_scene treatment: 'that table' can only ever be satisfied by THAT table."""
    return bool(_DEICTIC_RX.search(getattr(seg, "text", "") or ""))


# INSTRUCTED LOOKING — the narrator tells the viewer to look at a NAMED thing: "keep your eye on
# the dagger", "watch his face", "count the chairs", "notice the division of labour". Different from
# _DEICTIC_RX above, which catches a pointing determiner bound to a scene noun ("that table") and
# only decides POLICY. Here the sentence names its own target, and that target is checkable: either
# the dagger is on screen or it is not.
#
# These are the beats a viewer notices breaking. Measured on the v4 render: of 30 such beats the
# named thing was clearly visible in 8, partly in 10, and absent in 12 — and the two the essay turns
# on ("keep your eye on the dagger", "watch the trial the way Bran watched it") were both misses.
_LOOK_RX = re.compile(
    # 'listen to' removed: an AUDIO instruction ("listen to the register") must never become a
    # VISUAL must-see — the verifier's target_visible question and the target-led still query
    # are nonsensical for a sound, and the A/B showed such beats steering footage.
    r"\b(?:watch|look at|keep (?:your )?eyes? on|notice|count|study)\s+"
    r"((?:the|his|her|their|its|that|this|those|these|\w+'s)\b.*)", re.I)

# the same verbs aimed at a BARE PROPER NOUN — "watch Bran", "watch Olenna, not the cup". The
# determiner-led pattern above cannot see these (audited misses: 'watch Bran's face' matched but
# 'watch Bran' yielded no target at all). Case-sensitive capture: a capitalized token after the
# verb is a name; 'watch the' is the other pattern's business.
_LOOK_PROPER_RX = re.compile(
    r"\b(?:[Ww]atch|[Ll]ook at|[Kk]eep (?:your )?eyes? on|[Nn]otice|[Ss]tudy)\s+"
    r"([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})?(?:'s\s+\w+)?)")

# the verbs aimed at a PRONOUN — "notice what nobody asks him", "watch her while she reads".
# The pronoun's referent is the beat's required entity; the entity IS the visual target.
_LOOK_PRONOUN_RX = re.compile(
    r"\b(?:watch|look at|keep (?:your )?eyes? on|notice|study)\b[^.!?]{0,40}"
    r"\b(?:him|her|them)\b", re.I)

# a target must be something you can SEE — drop the abstract nouns the same phrasing produces
# ("notice the division of labour", "watch the way it changes")
_TARGET_STOP = frozenset("""way ways point points fact facts idea ideas reason reasons question
questions answer answers thing things order timing logic argument division labour labor
difference differences pattern patterns detail details sense
strategy tactics approach method plan intent motive meaning implication""".split())

# a target phrase ENDS here — the trailing clause is about the target, not part of it:
# "the dagger IN that room", "the trial THE WAY Bran watched it", "his strategy IN the next…"
_TARGET_CUT = frozenset("""in on at to for of with from by as and or but because while when
before after during until since so that which who whom whose again
the a an his her their its this these those""".split())


def deictic_target(seg) -> str:
    """The concrete thing this beat tells the viewer to look at, or "".

    Returns the noun phrase ("the dagger", "his face", "the chairs") so the verifier can require it
    on screen instead of settling for a clip that merely has the right people in it. The phrase is
    cut at the first preposition or conjunction: greedy capture turned "watch the trial the way Bran
    watched it" into "the trial the way", whose head noun then read as abstract and lost the beat."""
    txt = getattr(seg, "text", "") or ""
    got = _det_target(txt)
    if got:
        return got
    # bare proper noun: "watch Bran", "watch Olenna Tyrell", "notice Sansa's necklace"
    mp = _LOOK_PROPER_RX.search(txt)
    if mp:
        cand = mp.group(1).strip(".,;:!?\"'")
        head = cand.split()[-1].lower().rstrip("'s")
        if (head not in _TARGET_STOP and len(head) >= 3
                and cand.split()[0] not in _CAP_PRONOUNS):
            return cand
    # pronoun referent: "notice what nobody asks him" — the required entity is the target
    if _LOOK_PRONOUN_RX.search(txt):
        ent = (getattr(seg, "required_entity", "") or "").strip()
        if ent and (getattr(seg, "required_kind", "") or "") == "character":
            return ent
    return ""


# a mid-sentence capitalized pronoun must never read as a proper name ("They watch Him ascend")
_CAP_PRONOUNS = frozenset("""Him Her Them He She They It His Hers Their Its You Your Yours
Me My Mine Us Our Ours Who Whom""".split())


def _det_target(txt: str) -> str:
    """The original determiner-led extraction ('watch the chalice'), hardened: the capture stops
    at a sentence boundary (greedy split crossed 'itself. Joffrey' into the target), and ANY
    abstract token — not only the head — rejects the phrase ('his strategy unfold' dodged the
    head test because the trailing verb became the head)."""
    m = _LOOK_RX.search(txt)
    if not m:
        return ""
    words, out = m.group(1).split(), []
    for i, w in enumerate(words):
        bare = w.lower().strip(".,;:!?\"")
        ends_sentence = w.rstrip("\"'").endswith((".", "!", "?"))
        if bare.endswith("'s") and i == 0:
            out.append(w.strip(".,;:!?\""))
            if ends_sentence:
                break
            continue
        bare = bare.strip("'")
        if i and (bare in _TARGET_CUT or not bare):
            break
        out.append(w.strip(".,;:!?\"'"))
        if ends_sentence or len(out) >= 4:
            break
    if len(out) < 2:
        return ""
    head = out[-1].lower().rstrip("'s")
    if head in _TARGET_STOP or len(head) < 3:
        return ""
    if any(t.lower().rstrip("'s") in _TARGET_STOP for t in out):
        return ""
    return " ".join(out)


def has_deictic_target(seg) -> bool:
    return bool(deictic_target(seg))


def classify(seg) -> str:
    """Heuristic policy from the signals the analyzer already produces. Deterministic — always works
    even with no LLM. The LLM's explicit `visual_policy` (when present + valid) takes precedence."""
    # deixis outranks every other signal, including the abstract heuristic below
    if is_deictic(seg):
        return EXACT
    kind = (getattr(seg, "required_kind", "") or "").lower()
    ent = (getattr(seg, "required_entity", "") or "").strip()
    specific = bool(getattr(seg, "is_specific_claim", False))
    quote = (getattr(seg, "quote", "") or "").strip()
    text = (getattr(seg, "text", "") or "")
    # 1) a quoted iconic line, a specific visual claim, or a scene/event → the PRECISE moment
    if quote or specific or kind in ("scene", "event"):
        return EXACT
    # 2) a named person/object/place in general (no precise moment) → medium, right-subject
    if ent and kind in ("character", "actor", "object", "location"):
        return CHARACTER
    # 3) abstract/emotional commentary with no concrete subject → image/effect/freeze
    if not ent and _ABSTRACT_RX.search(text):
        return ABSTRACT
    # 4) everything else → generic filler
    return FILLER


def policy_of(seg) -> str:
    """The beat's resolved policy: an explicit (LLM-set) visual_policy if valid, else the heuristic.

    Deixis overrides the LLM label. The LLM sees one beat's words, not what they point AT, so it
    labels 'at that table' generic — and a generic label is a licence to air anything. The pointing
    word is objective evidence about the referent that outranks the model's guess."""
    if is_deictic(seg):
        return EXACT
    p = (getattr(seg, "visual_policy", "") or "").strip().lower()
    # DEMOTION GUARD — the mirror of the deixis promotion. A SPECIFIC label (exact/character) is a
    # demand for particular footage, so it must be earned by at least one concrete hook: a quote, a
    # required entity, or scene-describing text. A rhetorical connector has none; keeping the LLM's
    # exact_scene label on "And that raises the next obvious question." creates a beat that CANNOT
    # be satisfied — measured: 11 such beats release-blocked a render after ~7h of doomed fallback
    # work. Genuinely specific beats are untouched (any quote or entity keeps the label).
    if p in (EXACT, CHARACTER) and _is_rhetorical_connector(seg):
        return ABSTRACT
    return p if p in POLICIES else classify(seg)


def finalize_beats(segs) -> dict:
    """Persist a resolved visual_policy + breakout flag on EVERY beat after analysis, so all
    downstream stages + the ledger/review read one stable decision. Returns a class→count tally."""
    tally: dict = {}
    for s in segs or []:
        try:
            p = policy_of(s)
            s.visual_policy = p
            s.breakout_candidate = is_breakout_candidate(s)
            tally[p] = tally.get(p, 0) + 1
        except Exception:
            pass
    return tally


def normalize(value: str) -> str:
    """Coerce an LLM-supplied policy string to a valid one, or '' if unrecognised (→ heuristic)."""
    v = (value or "").strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "exact": EXACT, "scene": EXACT, "specific_scene": EXACT, "exact_scene": EXACT,
        "character": CHARACTER, "character_specific": CHARACTER, "person": CHARACTER, "subject": CHARACTER,
        "generic": FILLER, "filler": FILLER, "generic_filler": FILLER, "broll": FILLER, "b_roll": FILLER,
        "abstract": ABSTRACT, "abstract_effect": ABSTRACT, "emotional": ABSTRACT, "effect": ABSTRACT,
    }
    return aliases.get(v, v if v in POLICIES else "")


# --- simple predicates -----------------------------------------------------
def is_exact(seg) -> bool:      return policy_of(seg) == EXACT
def is_character(seg) -> bool:  return policy_of(seg) == CHARACTER
def is_filler(seg) -> bool:     return policy_of(seg) == FILLER
def is_abstract(seg) -> bool:   return policy_of(seg) == ABSTRACT


def is_breakout_candidate(seg) -> bool:
    return bool(getattr(seg, "breakout_candidate", False)) \
        or bool((getattr(seg, "quote", "") or "").strip())


# --- discovery budget (req. 8) --------------------------------------------
# high   = aggressive per-beat discovery + larger download budget (exact scenes must be found)
# medium = some targeted discovery (right character)
# low    = minimal / reuse the existing pool (don't spend on generic filler)
# none   = don't download footage; image/effect/freeze instead
def discovery_tier(seg) -> str:
    return {EXACT: "high", CHARACTER: "medium", FILLER: "low", ABSTRACT: "none"}[policy_of(seg)]


def wants_discovery_query(seg) -> bool:
    """Only exact/character beats earn their own targeted scene-query search; filler reuses the pool
    and abstract wants no footage — so we don't waste discovery budget on them."""
    return policy_of(seg) in (EXACT, CHARACTER)


# --- matching (reqs. 1,2,3,5) ---------------------------------------------
def match_strict(seg) -> bool:
    """exact_scene → strict: a weak/loose match must LOSE (and then flag missing) rather than air."""
    return policy_of(seg) == EXACT


def maximize_variety(seg) -> bool:
    """filler/character beats should spread across sources — never repeat a dominant one."""
    return policy_of(seg) in (FILLER, CHARACTER)


def footage_optional(seg) -> bool:
    """abstract beats don't need literal footage — the image/effect path may take them."""
    return policy_of(seg) == ABSTRACT


# --- verification (reqs. 1,4) ---------------------------------------------
def verify_strict(seg) -> bool:
    """Hold exact_scene beats to the precise scene; be lenient elsewhere (relevant filler is fine)."""
    return policy_of(seg) == EXACT


# --- image / effect fallback (final image policy) -------------------------
def prefers_image(seg) -> bool:
    """abstract_effect: prefer an image / effect / freeze / B-roll over literal footage."""
    return policy_of(seg) == ABSTRACT


def allows_web_image(seg) -> bool:
    """A real, validated LIVE-ACTION web still is a LAST resort. PREFER exact_scene only. A
    character_specific beat may use one ONLY when it names a SPECIFIC moment (a strong
    scene_query / expected_visual) — never a bare "show this character" portrait. Generic filler +
    abstract beats NEVER use a web image (source frames only)."""
    p = policy_of(seg)
    if p == EXACT:
        return True
    if p == CHARACTER:
        sq = (getattr(seg, "scene_query", "") or getattr(seg, "expected_visual", "") or "").strip()
        return len(sq) >= 12          # a specific moment, not a generic portrait
    return False


def allows_ai_image(seg) -> bool:
    """AI-generated images are GLOBALLY BANNED — no AI art, 3D renders, cartoon/fan-art/game/toy/
    illustration, or AI-generator domains, for ANY beat."""
    return False
