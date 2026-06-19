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


def classify(seg) -> str:
    """Heuristic policy from the signals the analyzer already produces. Deterministic — always works
    even with no LLM. The LLM's explicit `visual_policy` (when present + valid) takes precedence."""
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
    """The beat's resolved policy: an explicit (LLM-set) visual_policy if valid, else the heuristic."""
    p = (getattr(seg, "visual_policy", "") or "").strip().lower()
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
