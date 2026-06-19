"""FOOTAGE-STRENGTH DECISION LOGIC — Phase B, priority #1.

The brain that decides, per scene: do we stay FOOTAGE-FIRST, or shift into
MOTION-GRAPHICS storytelling? A documentary cannot lean only on stock
footage — for abstract, unfilmable, historically-missing or
content-desert topics, generic B-roll reads as "they had no footage".
A real cinematic storyteller instead VISUALISES the concept (animated
timeline, map, process, infographic, symbolic motion).

This module is the editorial gate. It is intentionally CONSERVATIVE
(the user's rule: do NOT overuse motion graphics — smart substitution
only). A scene is routed to motion-graphics ONLY when footage genuinely
can't communicate it well:

  • slide-fallback   — no real footage was found at all (weakest tier)
  • abstract concept — the line is about an idea/system/force a camera
                       cannot literally show, and only a generic
                       still/slide was available
  • repetition       — the same clip is being reused (content desert)

Scenes that already carry a strong editor-chosen graphic are left alone
(already handled), and scenes with strong, relevant footage stay
footage-first. The OUTPUT is a per-scene plan the pipeline uses to assign
an appropriate explainer template (routed by what the scene is ABOUT).

Decisions are deterministic (no LLM call) so they're cheap, reproducible
and cache-stable.
"""
from __future__ import annotations

import re
from pathlib import Path

from .script_gen import _ABSTRACT, _VISUAL_LEXICON

# --------------------------------------------------------------------------- #
# Concept routing lexicons — what KIND of explainer best fits the scene.
# --------------------------------------------------------------------------- #
_PROCESS_WORDS = set(
    """caused cause causes led leads leading because therefore thus then after
    before triggered triggers trigger resulted result results sequence step
    steps process processes how mechanism mechanisms chain reaction flow
    spread spreads spreading collapse collapsed collapsing began begins start
    started escalated escalating unfolded happened occurs occurred""".split()
)
_MONEY_WORDS = set(
    """money dollar dollars billion billions million millions trillion cost
    costs price prices economy economic inflation deflation debt market markets
    revenue profit profits loss losses budget gdp wealth fortune fund funds
    investment investments tax taxes payment payments wage wages""".split()
)
_SYMBOLIC_WORDS = set(
    """corruption surveillance power control propaganda manipulation conspiracy
    intelligence network networks influence ideology secrecy secret secrets
    espionage spying censorship authority regime dominance oppression freedom
    threat threats fear paranoia distrust betrayal cover-up coverup hidden""".split()
)
_PLACE_WORDS = set(
    """country countries nation nations border borders territory territories
    region regions empire kingdom city cities capital route routes invasion
    expansion advance retreat front frontline coast sea ocean continent
    continents map maps location locations world global europe asia africa
    america pacific atlantic""".split()
)
_YEAR_RE = re.compile(r"\b(?:1[0-9]{3}|20[0-9]{2})\b|\bcentur|\bdecade|\bera\b|\bbc\b|\bad\b")
_NUM_RE = re.compile(r"\b\d[\d,\.]*\s*(?:%|percent|tons?|miles?|km|years?|"
                     r"people|times|x)\b|\b(?:billion|million|trillion)\b", re.I)

# explainer templates that ALREADY exist in the catalogue and read as
# premium motion-graphics storytelling (NOT powerpoint). These are the
# routing targets. All are rendered with the new motion engine.
_EXPLAINER_KINDS = {
    "timeline", "map_reveal", "map_route", "cause_effect", "process_diagram",
    "network_graph", "relationship_tree", "conspiracy_board", "stat_dashboard",
    "chart", "number", "quote_highlight", "comparison", "bullet_list",
}

# kinds that, if the editor already chose one, mean the scene is ALREADY a
# deliberate graphic moment — never override these.
_ALREADY_GRAPHIC = _EXPLAINER_KINDS | {
    "title_card", "evidence", "evidence_tag", "document", "redacted",
    "classified", "portrait", "era_banner", "lower_third", "location",
    "stat", "newspaper", "news_article",
}


# --------------------------------------------------------------------------- #
# Footage tier — how strong is the resolved asset, by source.
# --------------------------------------------------------------------------- #
def footage_tier(item) -> tuple[float, str]:
    """(strength 0..1, label) from the resolved footage source. Real stock
    video is strongest; a themed slide means NO footage was found."""
    name = Path(str(getattr(item, "path", ""))).name.lower()
    if getattr(item, "is_video", False):
        return 0.85, "stock-video"
    if name.startswith("fal_"):
        return 0.70, "ai-video"
    if name.startswith(("aimg_", "ov_")):
        return 0.50, "ai-still"
    return 0.18, "slide-fallback"


def is_abstract_scene(scene) -> bool:
    """True when the scene is about something a camera can't literally show
    (an idea / system / force), so a generic clip wouldn't communicate it.
    Uses the same lexicons the keyword extractor uses, so it agrees with
    how footage was searched in the first place."""
    kws = [str(k).lower() for k in (getattr(scene, "keywords", None) or [])]
    blob = (getattr(scene, "narration", "") or "").lower()
    concrete = sum(1 for k in kws if k in _VISUAL_LEXICON)
    if concrete >= 1:
        return False                       # has a real depictable subject
    abstract = sum(1 for k in kws if k in _ABSTRACT)
    symbolic = any(w in blob for w in _SYMBOLIC_WORDS)
    return symbolic or abstract >= 1 or not kws


# --------------------------------------------------------------------------- #
# Forensic v2 (vs Vidlore) — SUBJECT-PRESENCE floor.  Vidlore wins most visual
# beats by showing the LITERAL named subject; Vidlore's existing gates reject
# blank/generic junk but happily PASS a clean, on-grade clip of the WRONG
# subject (b03 "two cheap metals" showing zero metal; b09 the beetle never
# appears; b11 copper shown as diagrams).  These two pure, deterministic
# helpers let the footage tier ladder REJECT an on-grade but off-subject asset
# and escalate (re-query subject -> archival/web image -> AI image -> generic
# atmosphere ONLY as last resort).  Gated OFF for abstract/concept scenes so
# concept beats are untouched.  No LLM, cache-stable.
# --------------------------------------------------------------------------- #
_GENERIC_FILLER = set(
    """background abstract texture bokeh light lights gradient pattern motion
    blur smoke particles aerial nature landscape scenery sky clouds cloud
    sunset sunrise dawn dusk grass meadow forest woods water generic stock
    footage video clip cinematic atmosphere mood ambient ambience b-roll
    person people man woman walking city street office desk technology""".split()
)
_SW = set(
    """the a an of to in on at by for with and or but from into over under is
    are was were be been being this that these those it its his her their them
    they he she you your we our as not no about above can will would what which
    who whom whose when where why how then than so very just also only more
    most some any each every both one two three your has have had they're""".split()
)


def _content_words(text: str) -> list[str]:
    """Lowercased content tokens (>=3 chars) minus stopwords + generic filler.
    Order-preserving + de-duped.  Cheap stand-in for noun extraction (no NLP)."""
    out, seen = [], set()
    for w in re.findall(r"[a-z][a-z\-]{2,}", (text or "").lower()):
        if w in _SW or w in _GENERIC_FILLER or w in seen:
            continue
        seen.add(w)
        out.append(w)
    return out


def subject_presence(scene) -> dict:
    """What concrete thing must be ON SCREEN for this scene?  Returns
    {nouns:set, verbs:set, process:bool, concrete:bool, expected_subject:str}.
    Deterministic; reuses the same visual lexicon footage was searched with."""
    blob = (getattr(scene, "narration", "") or "")
    low = blob.lower()
    kws = [str(k).lower() for k in (getattr(scene, "keywords", None) or [])]
    depictable = [k for k in kws if k in _VISUAL_LEXICON]
    nouns = list(dict.fromkeys(depictable + _content_words(blob)[:8]))
    verbs = {w for w in re.findall(r"[a-z]{3,}", low)
             if w.endswith(("ing", "ed")) and w not in _SW}
    process = any(w in low for w in _PROCESS_WORDS)
    concrete = not is_abstract_scene(scene)
    expected = depictable[0] if depictable else (nouns[0] if nouns else "")
    return {"nouns": set(nouns), "verbs": verbs, "process": process,
            "concrete": bool(concrete), "expected_subject": expected}


def subject_support_score(scene, asset_text: str) -> float:
    """0..1 — does this asset (its slug / caption / query text) visibly depict
    the scene's expected subject?  Generic-filler-only text scores 0.  Used by
    the footage tier ladder: a CONCRETE scene whose best asset scores below the
    SUBJECT_PRESENCE_FLOOR is rejected and escalated.  Never raises."""
    try:
        sp = subject_presence(scene)
        if not sp["concrete"]:
            return 1.0                         # abstract/concept beat — never block
        want = sp["nouns"]
        if not want:
            return 1.0                         # nothing concrete demanded
        atoks = set(re.findall(r"[a-z][a-z\-]{2,}", (asset_text or "").lower())) - _SW
        have = atoks - _GENERIC_FILLER
        if not have:
            return 0.0                         # empty or generic-filler-only asset

        def _stem_hit(word, pool):
            stem = word[:max(4, len(word) - 2)]
            return any(stem in p or p in word for p in pool)

        matches = sum(1 for w in want if w in have or _stem_hit(w, have))
        if matches == 0:
            return 0.0                         # asset depicts NONE of the subject
        return round(min(1.0, 0.45 + 0.25 * matches), 3)   # 1 match→0.70, 2→0.95
    except Exception:                          # never break a render
        return 1.0


# default subject-presence floor (env-tunable at the call site in footage.py).
SUBJECT_PRESENCE_FLOOR = 0.34


def suggest_explainer(scene) -> str:
    """Route a scene to the explainer template that best VISUALISES what it
    is about. Order matters — most specific signal wins."""
    blob = (getattr(scene, "narration", "") or "").lower()
    kws = " ".join(str(k).lower() for k in
                   (getattr(scene, "keywords", None) or []))
    text = f"{blob} {kws}"

    # 1) time / dates -> animated timeline
    if _YEAR_RE.search(text):
        return "timeline"
    # 2) geography / movement -> dynamic map
    if any(w in text for w in _PLACE_WORDS):
        return "map_reveal"
    # 3) numbers / economics -> infographic stat
    if _NUM_RE.search(text) or any(w in text for w in _MONEY_WORDS):
        return "stat_dashboard"
    # 4) cause / process -> clean editorial document card (the two-box
    #    cause_effect template was retired — template-y / repetitive)
    if any(w in text for w in _PROCESS_WORDS):
        return "document"
    # 5) abstract power/secrecy systems -> symbolic network / board
    if any(w in text for w in _SYMBOLIC_WORDS):
        return "network_graph"
    # 6) fallback -> kinetic typography of the key line (always cinematic,
    #    never powerpoint): let the sentence itself carry the screen.
    return "quote_highlight"


# --------------------------------------------------------------------------- #
# The plan — per-scene decision the pipeline acts on.
# --------------------------------------------------------------------------- #
_YEAR_TOKEN = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")
_NUM_TOKEN = re.compile(
    r"(\$?\d[\d,\.]*\s?(?:%|percent|tons?|billion|million|trillion|"
    r"miles?|km|years?|x)?)", re.I)


def _short_caps(text: str, n: int) -> str:
    t = re.sub(r"[^\w \-/$%.]", "", (text or "")).strip().upper()
    if len(t) <= n:
        return t
    return (t[:n].rsplit(" ", 1)[0] or t[:n]).rstrip()   # word boundary


def synthesize_content(scene, suggested: str) -> tuple[str, str, str]:
    """Build (kind, graphic_text, graphic_body) for a motion-graphics
    substitution from the scene's own narration. Where the structured
    data isn't cleanly extractable, fall back to KINETIC TYPOGRAPHY of
    the scene's key line (always cinematic, never powerpoint). Keeps
    quality high: a structured explainer only fires when the content
    genuinely supports it."""
    nar = (getattr(scene, "narration", "") or "").strip()
    emph = (getattr(scene, "emphasis", "") or "").strip()
    first = re.split(r"(?<=[.!?])\s+", nar)[0] if nar else ""

    def _clip_words(text: str, limit: int) -> str:
        """Trim to <=limit chars on a WORD boundary (never mid-word) so
        the on-screen line is always complete, not '...the questi'."""
        text = (text or "").strip().rstrip(".,;:")
        if len(text) <= limit:
            return text
        cut = text[:limit].rsplit(" ", 1)[0]
        return cut.rstrip(",;:") or text[:limit]

    # KINETIC TYPOGRAPHY fallback — a SHORT, complete key line as an
    # on-screen statement. quote_highlight wraps ~3 lines comfortably, so
    # cap ~96 chars on a word boundary. Prefer the first clause; if that's
    # still long, clip cleanly. (The richer per-scene structured content
    # comes from the LLM-editor explainer pass — this stays readable.)
    key = _clip_words(first or nar, 96) or (emph or "…")
    kinetic = ("quote_highlight", key, "")

    if suggested == "timeline":
        years = _YEAR_TOKEN.findall(nar)
        if len(years) >= 2:
            evs = []
            for y in years[:4]:
                tail = nar.split(y, 1)[1] if y in nar else ""
                label = _short_caps(tail, 22) or "EVENT"
                evs.append(f"{y}|{label}")
            return ("timeline", "KEY EVENTS", ";".join(evs))
        return kinetic

    if suggested in ("stat_dashboard", "number"):
        m = _NUM_TOKEN.search(nar)
        if m and any(c.isdigit() for c in m.group(1)):
            fig = m.group(1).strip()
            return ("number", fig, _short_caps(first, 32))
        return kinetic

    # map / cause_effect / network_graph need richer structured data than
    # we can reliably synthesise here -> kinetic typography for now (the
    # LLM-editor explainer pass will supply structured content next).
    return kinetic


def plan_motion_graphics(scenes: list, items: list,
                         *, weak_threshold: float = 0.45) -> list[dict]:
    """Decide, per scene, whether to substitute weak footage with a
    motion-graphics explainer. Returns one dict per scene:
        {index, strength, tier, abstract, decision, suggested, reason}
    decision ∈ {'footage', 'motion-graphics', 'keep-graphic'}.

    CONSERVATIVE: only flips to motion-graphics when footage is genuinely
    weak (slide fallback / repeated) OR the concept is abstract AND the
    footage is non-video. Scenes already carrying a deliberate graphic are
    left untouched ('keep-graphic')."""
    by_index = {getattr(it, "index", i): it for i, it in enumerate(items)}

    # repetition: same resolved asset stem reused across scenes
    stems: dict[str, int] = {}
    for it in items:
        stem = re.sub(r"_b\d+|_v\d+", "",
                      Path(str(getattr(it, "path", ""))).stem)
        stems[stem] = stems.get(stem, 0) + 1

    out: list[dict] = []
    for i, sc in enumerate(scenes):
        gk = (getattr(sc, "graphic_kind", "") or "").strip()
        it = by_index.get(getattr(sc, "index", i))
        strength, tier = footage_tier(it) if it is not None else (0.18,
                                                                  "none")
        abstract = is_abstract_scene(sc)

        # repetition penalty
        if it is not None:
            stem = re.sub(r"_b\d+|_v\d+", "",
                          Path(str(getattr(it, "path", ""))).stem)
            if stems.get(stem, 0) > 1:
                strength *= 0.55
        # abstract concept with only a still/slide -> footage can't carry it
        if abstract and strength < 0.7:
            strength *= 0.6

        if gk in _ALREADY_GRAPHIC:
            decision, suggested, reason = "keep-graphic", gk, \
                "scene already a deliberate graphic"
        elif strength < weak_threshold:
            decision = "motion-graphics"
            suggested = suggest_explainer(sc)
            reason = (f"{tier} + {'abstract' if abstract else 'weak'} "
                      f"(strength {strength:.2f})")
        else:
            decision, suggested, reason = "footage", None, \
                f"{tier} footage strong ({strength:.2f})"

        out.append(dict(index=getattr(sc, "index", i),
                        strength=round(strength, 2), tier=tier,
                        abstract=abstract, decision=decision,
                        suggested=suggested, reason=reason))
    return out
