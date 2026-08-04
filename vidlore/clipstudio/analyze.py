"""Phase 1 — script intelligence.

Turns a narration script (+ topic, + optional movie name) into structured film intelligence:
the movie title/year, the actor + character roster, key scenes/events/locations, visual
keywords, emotional moments — and tags each narration beat with the SPECIFIC entity it demands
(which actor / character / object / scene). This drives source discovery (what to search for)
and Face-ID (whose reference faces to build).

Uses the engine's Claude (key already in .env). Degrades to a heuristic roster if no LLM.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict

from .models import ScriptSegment
from .config import ClipConfig
from .segment import segment_script
from . import policy as _policy


@dataclass
class ScriptAnalysis:
    topic: str = ""
    movie_title: str = ""
    year: str = ""
    synopsis: str = ""               # the whole-story through-line (global context for matching)
    tone: str = ""                   # emotional register (tragic / triumphant / ominous …)
    # VIDEO TYPE drives the whole footage strategy:
    #   "single_scene" = a deep-dive on ONE moment/scene (the video re-watches & dissects it) → the
    #     editor must find the RAW footage of that ONE scene and play it THROUGH, tracking the
    #     narration (like a real essayist), not scatter unrelated clips.
    #   "multi_scene"  = a broad arc spanning many scenes (a character/season retrospective).
    video_type: str = ""             # "single_scene" | "multi_scene"
    # The 1–3 CORE scenes the whole video is ABOUT, each with the best RAW-footage search string.
    # For a single-scene video this is the one canonical scene the editor anchors everything on.
    anchor_scenes: list[dict] = field(default_factory=list)   # [{"name":..., "query":...}]
    # "S02E09"-style code when the LLM names the episode — used as a high-precision search
    # variant (the raw-scene uploads are very often titled with it).
    # It is a HINT, not a fact: it comes from one LLM field with no second opinion and has been
    # measured wrong (S04E01 for a scene that is S03E10). Until `episode_hint_verified` is True it
    # may be used to SEARCH but never to purge sources, confer anchor status, or constrain a beat's
    # era — see era.verified_episode_hint.
    episode_hint: str = ""
    episode_hint_verified: bool = False
    episode_hint_reason: str = ""
    actors: list[str] = field(default_factory=list)
    characters: list[dict] = field(default_factory=list)   # {"name":..., "actor":...}
    locations: list[str] = field(default_factory=list)
    key_scenes: list[str] = field(default_factory=list)
    events: list[str] = field(default_factory=list)
    visual_keywords: list[str] = field(default_factory=list)
    emotional_moments: list[str] = field(default_factory=list)
    # Persisted audit of the conservative beat-local exact-storyboard guard.  The per-segment
    # marker is process-local because ScriptSegment deliberately has no free-form metadata field;
    # this analysis-level copy makes every downgrade and its reason inspectable after resume.
    beat_grounding_audit: dict = field(default_factory=dict)
    source: str = "heuristic"        # "claude" | "heuristic"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ScriptAnalysis":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def reference_identities(self) -> list[dict]:
        """Unique ACTORS to build Face-ID references for (characters resolve to their actor, so
        every reference is one real face keyed by the actor's name)."""
        out, seen = [], set()
        for a in self.actors:
            if a and a.lower() not in seen:
                seen.add(a.lower())
                out.append({"name": a, "kind": "actor", "actor": a})
        for c in self.characters:
            ac = (c.get("actor") or "").strip()
            if ac and ac.lower() not in seen:
                seen.add(ac.lower())
                out.append({"name": ac, "kind": "actor", "actor": ac})
        return out

    def char_to_actor(self) -> dict:
        """Lowercased character-name / actor-name → actor full name (for resolving required entity)."""
        m = {}
        for c in self.characters:
            nm = (c.get("name") or "").strip()
            ac = (c.get("actor") or "").strip()
            if nm and ac:
                m[nm.lower()] = ac
        for a in self.actors:
            m.setdefault(a.lower(), a)
        return m


_SYS = (
    "You are a film researcher and footage editor. Given a faceless-video narration script about "
    "a movie (or its cast), extract precise, factual film intelligence and, for each numbered "
    "narration beat, decide the SINGLE most specific thing that must appear on screen. Be concrete "
    "and visual. Never invent actors/films you are unsure of — leave fields empty instead. "
    "Reply with ONLY a JSON object."
)


def _llm_analyze(script_text: str, topic: str, movie_hint: str, beats: list[ScriptSegment],
                 eng_cfg, progress=None):
    """Two-stage so it scales to ANY script length: (1) one compact high-level call, then
    (2) per-beat enrichment in small BATCHES. A single combined call for a long script blows the
    output token budget and returns truncated/invalid JSON (the 107-beat Cersei failure)."""
    from . import llm as _llm

    def _log(m):
        if progress:
            progress(m)

    # --- stage 1: high-level film intelligence (small, reliable output) ---
    hi_user = (
        f"TOPIC: {topic or '(none)'}\nKNOWN MOVIE: {movie_hint or '(infer from script)'}\n\n"
        f"NARRATION SCRIPT:\n{script_text[:4200]}\n\n"
        "First UNDERSTAND the whole video — its argument, arc and tone — then extract the film "
        "intelligence. Return ONLY this JSON object:\n"
        '{"movie_title":"","year":"",'
        '"synopsis":"2-3 sentences: what this video is ABOUT — the through-line/argument the '
        "narration makes start to finish, so every footage choice stays coherent with the whole "
        'story (not just the current line)",'
        '"tone":"the emotional register in 1-4 words (e.g. tragic, triumphant, ominous, reflective)",'
        '"video_type":"single_scene if the ENTIRE video dissects/re-watches ONE specific scene or '
        "moment (the narration walks through that single scene beat-by-beat); multi_scene if it "
        'spans many different scenes across the story",'
        '"anchor_scenes":[{"name":"the CANONICAL name of this scene as fans and clip-uploaders '
        "call it — the recognized scene + its setting (e.g. 'the Small Council meeting', 'the "
        "Battle of the Bastards', 'the Red Wedding'), NOT a thematic description you invented\","
        '"episode":"the exact episode if a TV show, as SxxExx AND its title if known (e.g. '
        "\\\"S03E03 Walk of Punishment\\\"); '' for a film or if unsure\","
        '"query":"the BEST YouTube search string for the RAW in-show footage of THIS exact scene '
        "— movie + the CANONICAL scene/location name + key characters + episode code, the way a "
        "clip uploader actually titles it (e.g. 'Game of Thrones Small Council scene Walk of "
        "Punishment S03E03 Littlefinger Tywin'). Use the canonical scene name, never a vague "
        "theme like 'chair test'\","
        '"dialogue":["2-5 VERBATIM lines actually SPOKEN in this scene, exactly as said on screen '
        "(e.g. 'You're just like me, only smaller'). Verbatim only — never paraphrase. NEVER song "
        "lyrics or sung lines (those appear in every cover/lyric upload and poison the search); "
        'spoken dialogue between characters only; [] if unsure"]}],  // 1 for single_scene, '
        "up to 3 for multi_scene — the scenes EVERYTHING should be cut from",
        '"actors":["full names"],'
        '"characters":[{"name":"character","actor":"who plays them"}],  // EVERY named '
        "character in the script PLUS the title's OTHER major recurring characters who commonly "
        "appear in clips of this story (e.g. siblings, rivals, the people in the same scenes) — "
        "each with the actor who plays them. List 6-12 when the title is a big ensemble. This "
        "roster is used to RECOGNIZE faces so footage of the WRONG character is never shown when "
        "a specific person is named,"
        '"locations":["places in the film"],'
        '"key_scenes":["SPECIFIC memorable scenes to find footage of"],'
        '"events":["plot events referenced"],'
        '"visual_keywords":["concrete search terms"],'
        '"emotional_moments":["beats with strong emotion"]}'
    )
    def _parse_obj(txt):
        """Extract a JSON object from an LLM reply — tolerant of a reasoning model (pro) that may
        wrap it in a ```json fence or add preamble/closing prose."""
        if not txt or not txt.strip():
            return None
        t = txt.strip()
        fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", t, re.S)
        if fence:
            t = fence.group(1)
        mm = re.search(r"\{.*\}", t, re.S)
        if not mm:
            return None
        try:
            return json.loads(mm.group(0))
        except Exception:
            return None

    hi_txt = _llm.complete(system=_SYS, messages=[{"role": "user", "content": hi_user}],
                           max_tokens=2000, eng_cfg=eng_cfg)
    analysis = _parse_obj(hi_txt)
    if analysis is None:
        # RELIABILITY: a reasoning model (deepseek-v4-pro) can occasionally return empty/garbled
        # JSON (reasoning overran the budget, or stray prose). Rather than collapse to the heuristic
        # (much worse), retry the high-level once on the FAST DeepSeek model so analysis stays on an
        # LLM. No-op for non-DeepSeek providers (fast_deepseek_model used only as an explicit retry).
        fb = getattr(_llm, "fast_deepseek_model", lambda: "")()
        if fb and _llm._provider() in ("deepseek", "ds"):
            hi_txt = _llm.complete(system=_SYS, messages=[{"role": "user", "content": hi_user}],
                                   max_tokens=2000, eng_cfg=eng_cfg, model=fb)
            analysis = _parse_obj(hi_txt)
    if analysis is None:
        return None

    # --- stage 2: per-beat visual targeting, in batches (robust for long scripts) ---
    # GLOBAL CONTEXT header — the whole topic/synopsis/tone + the full narration go into EVERY batch so
    # the model resolves each line WITHIN the arc (e.g. "she makes a decision" → the King's Landing
    # burning), not in isolation. This is what makes the footage choices coherent with the whole story.
    mv = analysis.get("movie_title", "") or movie_hint
    _ctx = (
        f"TOPIC: {topic or '(none)'}\nMOVIE/SHOW: {mv or '(infer)'}\n"
        f"WHAT THIS VIDEO IS ABOUT: {analysis.get('synopsis', '') or '(infer from the script)'}\n"
        f"TONE: {analysis.get('tone', '') or '(infer)'}\n\n"
        f"FULL NARRATION (context — understand how every beat fits this whole arc):\n"
        f"{script_text[:4500]}\n\n"
    )
    beat_out = []
    BATCH = 18
    for start in range(0, len(beats), BATCH):
        chunk = beats[start:start + BATCH]
        numbered = [{"i": b.index, "line": b.text} for b in chunk]
        bu = (
            _ctx +
            "You know this film/show scene-by-scene. For EACH beat, first classify only the visual "
            "promise actually made by its NARRATION. Do not invent an exact action, location, pose, "
            "camera angle, or scene from whole-story context for a generic, character-general, or "
            "abstract line. If a beat says a character learns/discovers something but does not name "
            "how, keep that information event exact without inventing a child, whisper, letter, or "
            "other delivery mechanism. Only after a beat genuinely names a precise moment should "
            "you identify that exact scene. Reply ONLY a JSON array, one "
            "object per beat, same order:\n"
            '[{"i":int,"expected_visual":"concrete shot description of the exact moment",'
            '"scene_query":"a precise search string to find THIS exact scene clip — movie + character + '
            "the specific action/location (e.g. 'Game of Thrones Daenerys walks into fire unburnt "
            "Drogo funeral pyre'), or '' if the line is generic\",\"quote\":\"the iconic line of "
            "DIALOGUE actually spoken in the SAME CONTINUOUS <=8-SECOND WINDOW as expected_visual, "
            "verbatim if known (e.g. 'I am the dragon's daughter'), or '' if none. A line from the "
            "same scene but before/after that window is NOT a match. NEVER put the essayist's "
            "narration/paraphrase here, and "
            "NEVER borrow an iconic line from another scene or copy one across adjacent beats just "
            "because the same character appears; if you are not confident it is spoken in THIS "
            "scene, use ''\",\"shot_intent\":\"the KIND of shot that best serves this beat: "
            "establishing|action|emotional_closeup|reaction|symbolic|montage\","
            "\"required_entity\":\"the ONE specific "
            "actor/character/object/scene this line demands, or ''\","
            '"required_kind":"actor|character|object|scene|event|location|",'
            '"visual_policy":"ONE of exact_scene|character_specific|generic_filler|abstract_effect — '
            "exact_scene=a precise scene/quote/character-action/plot-event (needs the EXACT clip); "
            "character_specific=a named person/thing in general (needs the right subject, any clean shot); "
            "generic_filler=generic/explanatory line (any relevant clip); abstract_effect=abstract/"
            'emotional/meta with no literal visual (use an image/effect). For character_specific, '
            'describe only the named subject generally and set specific=false; for generic_filler or '
            'abstract_effect leave scene_query empty and set specific=false",'
            '"emotion":"one word or '
            "''\",\"specific\":true}]\n\nBEATS:\n" + json.dumps(numbered, ensure_ascii=False)
        )
        parsed = None
        for _try in range(2):                      # one retry — a truncated/invalid/EMPTY reply
            bt = _llm.complete(system=_SYS, messages=[{"role": "user", "content": bu}],
                               max_tokens=min(8000, 600 + len(chunk) * 180), eng_cfg=eng_cfg,
                               model=_llm.beat_model())   # fast model for high-volume per-beat work
            mm = re.search(r"\[.*\]", bt, re.S)
            if mm:
                try:
                    parsed = json.loads(mm.group(0))
                except Exception:
                    parsed = None
                if isinstance(parsed, list) and parsed:
                    break
                parsed = None                      # '[]' is no enrichment — spend the retry
        if parsed:
            beat_out.extend(parsed)
        else:
            _log(f"analyze: ⚠ beats {start}-{min(start + BATCH, len(beats)) - 1} enrichment "
                 f"failed (invalid LLM JSON twice) — they fall back to heuristic visuals")
        _log(f"analyze: per-beat {min(start + BATCH, len(beats))}/{len(beats)}")
    return {"analysis": analysis, "beats": beat_out}


_GROUNDING_STOP = frozenset(
    "a an the of to in on at for and or but so then with from into as it its is are was were be "
    "been being this that these those his her their he she they him them who which what when where "
    "movie show scene scenes clip clips shot shots footage game thrones got hd official season "
    "episode part full closeup wide extreme camera".split()
)
_GROUNDING_WORD_RX = re.compile(r"[a-z][a-z'-]*", re.I)

# An exact directive may refer indirectly to a moment named immediately before it.  Keep this
# deliberately narrower than general demonstratives: ``that is a rarer mind`` is commentary, while
# ``inside that scene`` and ``moments later`` really do point at an authored moment.
_BEAT_LOCAL_SCENE_POINTER_RX = re.compile(
    r"\b(?:that|this|those|these)\s+(?:\w+\s+){0,2}"
    r"(?:scene|moment|meeting|exchange|conversation|attack|betrayal|death|trial|feast|fight|event)\b"
    r"|\b(?:at that point|right there|there it is|moments? later|seconds? later)\b"
    r"|^\s*(?:then|next)\b", re.I)

# These are not claims about one observable instant even when they contain a verb.  The narrow
# forms below are measured analyzer failures; a blanket ``never|always`` rule would incorrectly
# demote lines such as "Ned never sees the dagger before Jaime attacks".
_GENERAL_RELATION_RX = re.compile(
    r"\bnever\s+had\s+to\b|\b(?:was|is|were|are)\s+(?:not\s+)?(?:a\s+)?"
    r"rarer\s+(?:mind|willingness)\b",
    re.I,
)
_NOMINAL_FRAGMENT_RX = re.compile(r"^\s*to\b", re.I)
_VAGUE_SUBJECT_RX = re.compile(
    r"\b(?:those|these)\s+people\b|\bsomeone\s+(?:is\s+)?(?:checking|verifying)\b",
    re.I,
)
# Information-acquisition narration names an exact event, but it does not name the mechanism by
# which the character learns it.  The beat model has repeatedly turned ``Varys learns she is in the
# city`` into a literal child-whisper/eye-widening shot that does not exist.  Keep the real event
# exact, while removing only that unsupported staging.  These patterns are intentionally narrow:
# physical actions (putting a dagger on a table, holding a poison cup, an actual attack) never enter
# this branch.
_INFORMATION_EVENT_RX = re.compile(
    r"\b(?:learns?|learned|discovers?|discovered|finds?\s+out|found\s+out|"
    r"realizes?|realized|is\s+told|was\s+told|hears?|heard)\b",
    re.I,
)
_INFORMATION_MECHANISM_RX = re.compile(
    r"\b(?:whispers?|whispered|whispering|ravens?|letters?|messages?|child|children|"
    r"spies|spy|reads?|reading|informs?|informed)\b",
    re.I,
)
# A quote copied onto a neighbouring, independently narrated physical beat creates an impossible
# conjunction even when both pieces occur somewhere in the same long scene.  We only remove the
# copy when a nearby beat carries the identical line with substantially stronger beat-local support.
_DISTINCT_PHYSICAL_EVENT_RX = re.compile(
    r"\b(?:holds?|holding|held|carries|carrying|carried|wears?|wearing|wore|drinks?|"
    r"drinking|drank|pushes|pushing|pushed|stabs?|stabbing|stabbed|shoots?|shooting|"
    r"burns?|burning|burned|places?|placing|placed|puts?|putting|falls?|falling|fell|"
    r"collapses?|collapsing|collapsed)\b",
    re.I,
)
_SCENE_ACTION_ROOTS = frozenset({
    "answer", "arrive", "attack", "betray", "bow", "burn", "carry", "chase", "check",
    "choke", "collapse", "confess", "confront", "discover", "die", "draw", "drink",
    "enter", "escape", "fall", "fight", "find", "give", "hand", "hold", "kill", "kneel",
    "learn", "leave", "look", "open", "order", "persuade", "poison", "push", "read",
    "realize", "reveal", "ride", "run", "say", "shoot", "stab", "stand", "take", "tell",
    "touch", "turn", "wait", "walk", "watch", "whisper", "write",
})
_NAMED_SUBJECT_KINDS = frozenset({
    "actor", "character", "object", "prop", "weapon", "animal", "creature", "location",
    "place", "building", "vehicle",
})
_NON_SUBJECT_GROUNDING_TERMS = _SCENE_ACTION_ROOTS | frozenset({
    "belief", "believe", "check", "clever", "decision", "die", "general", "information",
    "mind", "motion", "people", "plan", "rare", "rarer", "reason", "verification",
    "willing", "willingness",
})


def _grounding_stem(word: str) -> str:
    """Small deterministic stemmer for overlap evidence; it is not a semantic classifier."""
    w = word.lower().strip("'-")
    if len(w) <= 3:
        return w
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("ing") and len(w) > 5:
        root = w[:-3]
        if len(root) > 3 and root[-1:] == root[-2:-1]:
            root = root[:-1]
        return root
    if w.endswith("ed") and len(w) > 4:
        if w == "died":
            return "die"
        root = w[:-2]
        if len(root) > 3 and root[-1:] == root[-2:-1]:
            root = root[:-1]
        return root
    if w.endswith("es") and len(w) > 4:
        return w[:-2]
    if w.endswith("s") and len(w) > 4:
        return w[:-1]
    return w


def _grounding_terms(value: str) -> set[str]:
    return {
        stem for raw in _GROUNDING_WORD_RX.findall(value or "")
        if raw.lower() not in _GROUNDING_STOP and len(stem := _grounding_stem(raw)) >= 3
    }


def _narrates_authored_quote(narration: str, quote: str) -> bool:
    """True only when the beat itself says the analyzer-authored dialogue.

    This is intentionally independent of whole-pool quote typing.  A real line from some other
    scene cannot make a generic narration beat exact, but dialogue actually present in the beat is
    a beat-local promise and must not be demoted here.
    """
    q_raw = " ".join(_GROUNDING_WORD_RX.findall(quote or "")).lower()
    n_raw = " ".join(_GROUNDING_WORD_RX.findall(narration or "")).lower()
    if q_raw and q_raw in n_raw:
        return True
    q_terms = _grounding_terms(quote)
    n_terms = _grounding_terms(narration)
    return bool(q_terms and len(q_terms & n_terms) / len(q_terms) >= 0.8)


def _has_scene_action(text: str) -> bool:
    terms = _grounding_terms(text)
    if terms & _SCENE_ACTION_ROOTS:
        return True
    # Covers productive verbs outside the small root vocabulary while excluding a bare copula.
    return bool(re.search(r"\b[a-z]{4,}(?:ing|ed)\b", text or "", re.I))


def _exact_direction_grounding(beat: ScriptSegment, directive: dict) -> dict:
    """Classify whether an exact storyboard is supported by this beat's own narration.

    The analyzer sees the entire essay and sometimes assigns a vivid scene from neighbouring/global
    context to a generic line.  This guard only rejects *obvious* cases: an explicitly vague subject
    with zero storyboard overlap, a subject-only nominal fragment, or a measured general relationship
    construction.  Ambiguous, indirect, dialogue-bearing, and genuinely action-bearing lines remain
    exact; downstream media verification is still responsible for proving their footage.
    """
    narration = str(getattr(beat, "text", "") or "")
    quote = str(directive.get("quote", "") or "")
    narration_terms = _grounding_terms(narration)
    entity_terms = _grounding_terms(str(directive.get("required_entity", "") or ""))
    storyboard_terms = _grounding_terms(" ".join(str(directive.get(k, "") or "") for k in (
        "expected_visual", "scene_query", "required_entity")))
    shared = narration_terms & storyboard_terms
    entity_shared = narration_terms & entity_terms

    if _narrates_authored_quote(narration, quote):
        return {"grounded": True, "reason": "authored_dialogue_in_narration",
                "shared_terms": sorted(shared)[:8], "entity_grounded": bool(entity_shared)}
    if _BEAT_LOCAL_SCENE_POINTER_RX.search(narration):
        return {"grounded": True, "reason": "indirect_scene_reference_in_narration",
                "shared_terms": sorted(shared)[:8], "entity_grounded": bool(entity_shared)}

    expected_visual = str(directive.get("expected_visual", "") or "")
    # Preserve the exact information event, but do not let whole-story context fabricate the
    # physical delivery mechanism.  ``scene_query`` is retained: it is often the clean, grounded
    # event query ("Varys learns Catelyn is in the city") while only the prose storyboard and its
    # borrowed line contain the invented child/whisper staging.
    if (_INFORMATION_EVENT_RX.search(narration)
            and entity_shared and len(shared) >= 2
            and not _INFORMATION_MECHANISM_RX.search(narration)
            and _INFORMATION_MECHANISM_RX.search(expected_visual)):
        sanitize_fields = ["expected_visual"]
        if quote and not _narrates_authored_quote(narration, quote):
            sanitize_fields.append("quote")
        return {
            "grounded": True,
            "reason": "unsupported_information_staging_removed",
            "shared_terms": sorted(shared)[:8],
            "entity_grounded": bool(entity_shared),
            "sanitize_fields": sanitize_fields,
        }

    has_action = _has_scene_action(narration)
    required_kind = str(directive.get("required_kind", "") or "").strip().lower()
    subject_terms = shared - _NON_SUBJECT_GROUNDING_TERMS
    # A canonical entity can be an alias rather than a lexical copy ("the master-at-arms" →
    # "Aron Santagar").  One shared subject phrase is enough for subject-level footage when the
    # declared kind is a concrete named thing; it is not enough to prove an exact action.
    subject_named = bool(entity_shared or len(subject_terms) >= 2
                         or (subject_terms and required_kind in _NAMED_SUBJECT_KINDS))
    if _GENERAL_RELATION_RX.search(narration):
        return {"grounded": False, "reason": "general_relationship_not_exact_moment",
                "shared_terms": sorted(shared)[:8], "entity_grounded": bool(entity_shared),
                "subject_named": subject_named}
    if _NOMINAL_FRAGMENT_RX.search(narration) and not has_action:
        return {"grounded": False, "reason": "named_subject_without_exact_action",
                "shared_terms": sorted(shared)[:8], "entity_grounded": bool(entity_shared),
                "subject_named": subject_named}
    if not shared and _VAGUE_SUBJECT_RX.search(narration):
        return {"grounded": False, "reason": "vague_subject_has_no_storyboard_overlap",
                "shared_terms": [], "entity_grounded": False, "subject_named": False}

    # Once the narration itself names the subject and an action/event, uncertainty belongs to the
    # footage verifier, not this lexical guard.  Keeping this branch permissive protects indirect
    # references and verbs/synonyms outside the tiny deterministic vocabulary.
    if has_action:
        return {"grounded": True, "reason": "named_subject_and_scene_action",
                "shared_terms": sorted(shared)[:8], "entity_grounded": bool(entity_shared)}
    if len(shared) >= 3:
        return {"grounded": True, "reason": "specific_event_terms_in_narration",
                "shared_terms": sorted(shared)[:8], "entity_grounded": bool(entity_shared)}
    # No deterministic signal proved a mismatch.  Fail closed *against demotion*: the exact media
    # verifier remains the authoritative gate, and this analyzer guard must not mass re-label an
    # edit merely because a grounded paraphrase has low lexical overlap.
    return {"grounded": True, "reason": "conservative_no_mismatch_proven",
            "shared_terms": sorted(shared)[:8], "entity_grounded": bool(entity_shared)}


def _apply_beat_direction(beat: ScriptSegment, directive: dict) -> None:
    """Apply one analyzer reply while keeping policy and specificity internally coherent."""
    resolved = _policy.normalize(directive.get("visual_policy", ""))
    grounding = (_exact_direction_grounding(beat, directive)
                 if resolved == _policy.EXACT else None)
    if directive.get("expected_visual"):
        beat.expected_visual = str(directive["expected_visual"])[:200]
    beat.scene_query = str(directive.get("scene_query", ""))[:120]
    beat.quote = str(directive.get("quote", ""))[:160]
    beat.shot_intent = str(directive.get("shot_intent", ""))[:24]
    beat.required_entity = str(directive.get("required_entity", ""))[:80]
    beat.required_kind = str(directive.get("required_kind", ""))[:20]
    beat.emotion = str(directive.get("emotion", ""))[:24]
    if grounding is not None:
        sanitize_fields = list(grounding.get("sanitize_fields") or [])
        marker = {
            "branch": ("grounded_exact_sanitized" if grounding["grounded"] and sanitize_fields
                       else "grounded_exact" if grounding["grounded"]
                       else "ungrounded_exact_downgrade"),
            "reason": grounding["reason"],
            "from_policy": _policy.EXACT,
            "shared_terms": grounding.get("shared_terms", []),
            "entity_grounded": bool(grounding.get("entity_grounded")),
        }
        if sanitize_fields:
            marker["sanitized_fields"] = sanitize_fields
            if "expected_visual" in sanitize_fields:
                beat.expected_visual = str(getattr(beat, "text", "") or "")[:200]
            if "quote" in sanitize_fields:
                beat.quote = ""
        if not grounding["grounded"]:
            # A narration-named subject still merits subject-correct footage, but no exact action.
            # With no named subject, ordinary semantic filler is the honest promise.  Clear every
            # exact-only field before quote typing: a real quote from another scene must not revive
            # a storyboard this beat never authored.
            subject_named = bool(grounding.get("subject_named")
                                 or grounding.get("entity_grounded"))
            resolved = _policy.CHARACTER if subject_named else _policy.FILLER
            marker["to_policy"] = resolved
            beat.expected_visual = str(getattr(beat, "text", "") or "")[:200]
            beat.scene_query = ""
            beat.quote = ""
            beat.is_specific_claim = False
            if resolved == _policy.FILLER:
                beat.required_entity = ""
                beat.required_kind = ""
        setattr(beat, "_analyzer_grounding_guard", marker)
    # Policy and specificity are one contract. A character-general or abstract label cannot
    # simultaneously carry an analyzer-invented `specific=true` that steers title affinity and
    # downstream prompts back toward a fabricated exact storyboard.
    if grounding is not None and not grounding["grounded"]:
        beat.is_specific_claim = False
    elif resolved in (_policy.CHARACTER, _policy.ABSTRACT):
        beat.is_specific_claim = False
    elif directive.get("specific"):
        beat.is_specific_claim = True
    if resolved == _policy.ABSTRACT:
        # Abstract means there is no literal searchable subject. Keep expected_visual as an
        # effects/art-direction hint, but remove contradictory retrieval fields.
        beat.scene_query = ""
        beat.required_entity = ""
        beat.required_kind = ""
    if resolved:
        beat.visual_policy = resolved


def _normalized_grounding_quote(value: str) -> str:
    """Stable identity for detecting an analyzer line copied across neighbouring beats."""
    return " ".join(_GROUNDING_WORD_RX.findall(value or "")).lower()


def _quote_narration_support(narration: str, quote: str) -> float:
    """Beat-local support only; this does not attempt to decide whether dialogue is real."""
    if _narrates_authored_quote(narration, quote):
        return 1.0
    q_terms = _grounding_terms(quote)
    if not q_terms:
        return 0.0
    return len(q_terms & _grounding_terms(narration)) / len(q_terms)


def _sanitize_adjacent_quote_borrowing(beats) -> int:
    """Remove only a demonstrably copied quote from a distinct, grounded visual event.

    The beat analyzer works in global-context batches and can paste the strongest line from a long
    scene onto several adjacent beats.  When one beat narrates a concrete physical instant and a
    nearby beat carries the identical quote with strong local wording support, requiring both in one
    <=8-second window is an analyzer-authored conjunction—not a narration promise.  The event remains
    EXACT and all visual fields remain intact; only the copied quote requirement is removed.
    """
    rows = list(beats or [])
    changed = 0
    for beat in rows:
        quote = str(getattr(beat, "quote", "") or "")
        qkey = _normalized_grounding_quote(quote)
        marker = getattr(beat, "_analyzer_grounding_guard", None)
        narration = str(getattr(beat, "text", "") or "")
        expected = str(getattr(beat, "expected_visual", "") or "")
        if (not qkey or not isinstance(marker, dict)
                or not str(marker.get("branch", "")).startswith("grounded_exact")
                or not _DISTINCT_PHYSICAL_EVENT_RX.search(narration)
                or len(_grounding_terms(narration) & _grounding_terms(expected)) < 3):
            continue
        local_support = _quote_narration_support(narration, quote)
        stronger = []
        for other in rows:
            if other is beat or abs(int(getattr(other, "index", -9999))
                                    - int(getattr(beat, "index", 9999))) > 3:
                continue
            if _normalized_grounding_quote(str(getattr(other, "quote", "") or "")) != qkey:
                continue
            support = _quote_narration_support(str(getattr(other, "text", "") or ""), quote)
            if support >= 0.8 and support >= local_support + 0.35:
                stronger.append((support, int(getattr(other, "index", -1))))
        if not stronger:
            continue
        strongest_support, strongest_index = max(stronger)
        beat.quote = ""
        marker["branch"] = "grounded_exact_sanitized"
        fields = list(marker.get("sanitized_fields") or [])
        if "quote" not in fields:
            fields.append("quote")
        marker["sanitized_fields"] = fields
        reasons = list(marker.get("sanitization_reasons") or [])
        reasons.append("adjacent_quote_borrowed_into_distinct_visual_event")
        marker["sanitization_reasons"] = reasons
        marker["quote_support_beat"] = strongest_index
        marker["quote_support"] = round(strongest_support, 4)
        changed += 1
    return changed


def _record_beat_grounding_audit(analysis: ScriptAnalysis, beats) -> dict:
    """Persist process-local grounding markers in the project's serialized analysis metadata."""
    records = {}
    for beat in beats or []:
        marker = getattr(beat, "_analyzer_grounding_guard", None)
        if isinstance(marker, dict):
            records[str(int(getattr(beat, "index", -1)))] = dict(marker)
    grounded = sum(str(m.get("branch", "")).startswith("grounded_exact")
                   for m in records.values())
    downgraded = sum(m.get("branch") == "ungrounded_exact_downgrade"
                     for m in records.values())
    sanitized = sum(m.get("branch") == "grounded_exact_sanitized"
                    for m in records.values())
    counts = {
        "exact_directives": len(records),
        "grounded_exact": grounded,
        "sanitized_exact": sanitized,
        "information_staging_sanitized": sum(
            m.get("reason") == "unsupported_information_staging_removed"
            for m in records.values()),
        "adjacent_quote_copy_sanitized": sum(
            "adjacent_quote_borrowed_into_distinct_visual_event"
            in (m.get("sanitization_reasons") or []) for m in records.values()),
        "downgraded": downgraded,
        "to_character_specific": sum(
            m.get("to_policy") == _policy.CHARACTER for m in records.values()),
        "to_generic_filler": sum(
            m.get("to_policy") == _policy.FILLER for m in records.values()),
    }
    analysis.beat_grounding_audit = {
        "schema": 2,
        "counts": counts,
        "beats": records,
    }
    return counts


_ANCHOR_STOP = set(          # (was `_STOPQ if False else set(...)` — the dead branch never ran,
                             # but it left the package's only undefined-name warning, which is the
                             # exact signal that catches bugs like the recovery NameError)
    "the a an of to in on at for and or but with from into as it is are was were be scene clip "
    "game thrones got movie show season episode hd official part full his her their he she they "
    "video this that what why how when where who which".split())


def _derive_anchors(beats, movie_title: str):
    """Heuristic anchor/video-type derivation from the beats' scene_queries — a safety net so the
    footage strategy never collapses when the LLM omits anchor_scenes (it returns them inconsistently
    under load). The DOMINANT cluster of scene-query tokens across the script IS the scene the video
    keeps returning to; if it covers most beats, the video is a single-scene deep-dive."""
    import re as _re
    from collections import Counter
    mvtoks = {w for w in _re.findall(r"\w+", (movie_title or "").lower()) if len(w) > 2}
    per_beat = []
    for b in beats:
        sq = (getattr(b, "scene_query", "") or "")
        if not sq:                                  # heuristic path has no scene_query — use the line
            sq = " ".join(list(getattr(b, "keywords", []) or []) +
                          list(getattr(b, "entities", []) or [])) or (getattr(b, "text", "") or "")
        toks = {w for w in _re.findall(r"\w+", sq.lower())
                if len(w) > 2 and w not in _ANCHOR_STOP and w not in mvtoks}
        per_beat.append(toks)
    freq = Counter(w for ts in per_beat for w in ts)
    if not freq:
        return "multi_scene", []
    nbeats = max(1, sum(1 for ts in per_beat if ts))
    # tokens shared by a meaningful fraction of beats = the recurring (anchor) scene
    top = [w for w, c in freq.most_common(8) if c >= max(2, nbeats * 0.25)]
    if not top:
        top = [w for w, _ in freq.most_common(5)]
    # how concentrated is the script? fraction of beats that touch the top cluster
    covered = sum(1 for ts in per_beat if ts and (set(top) & ts)) / nbeats
    vtype = "single_scene" if covered >= 0.6 else "multi_scene"
    query = (f"{movie_title} " + " ".join(top[:6])).strip()
    name = " ".join(top[:6])
    return vtype, [{"name": name, "query": query[:120]}]


def analyze_script(script_text: str, *, topic: str = "", movie_hint: str = "",
                   eng_cfg=None, cfg: ClipConfig, progress=None):
    """Return (ScriptAnalysis, list[ScriptSegment] enriched with required entity per beat)."""
    beats = segment_script(script_text, cfg)

    def log(m):
        if progress:
            progress(m)

    from . import llm as _llm
    if not (eng_cfg and _llm.has_llm(eng_cfg)):
        ents, seen = [], set()
        for b in beats:
            for e in b.entities:
                if e.lower() not in seen:
                    seen.add(e.lower())
                    ents.append(e)
        analysis = ScriptAnalysis(topic=topic, movie_title=movie_hint, actors=ents[:12],
                                  visual_keywords=[k for b in beats for k in b.keywords][:20],
                                  source="heuristic")
        vt, anch = _derive_anchors(beats, movie_hint)
        analysis.video_type, analysis.anchor_scenes = vt, anch
        log("analyze: heuristic (no LLM key)")
        return analysis, beats

    try:
        data = _llm_analyze(script_text, topic, movie_hint, beats, eng_cfg, progress)
    except Exception as e:                              # noqa: BLE001
        log(f"analyze: LLM failed ({str(e)[:80]}) — heuristic fallback")
        data = None
    if not data:
        return analyze_script(script_text, topic=topic, movie_hint=movie_hint,
                              eng_cfg=None, cfg=cfg, progress=progress)

    a = data.get("analysis", {})
    analysis = ScriptAnalysis(
        topic=topic,
        movie_title=a.get("movie_title", "") or movie_hint,
        year=str(a.get("year", "")),
        synopsis=str(a.get("synopsis", ""))[:600],
        tone=str(a.get("tone", ""))[:40],
        video_type=("single_scene" if "single" in str(a.get("video_type", "")).lower()
                    else "multi_scene"),
        anchor_scenes=[{"name": str(s.get("name", ""))[:120], "query": str(s.get("query", ""))[:140],
                        "episode": str(s.get("episode", ""))[:40],
                        "dialogue": [str(d)[:120] for d in (s.get("dialogue") or [])
                                     if isinstance(d, str) and len(d.split()) >= 3][:5]}
                       for s in a.get("anchor_scenes", [])
                       if isinstance(s, dict) and (s.get("query") or s.get("name"))][:3],
        actors=[str(x) for x in a.get("actors", []) if x][:14],
        characters=[c for c in a.get("characters", []) if isinstance(c, dict) and c.get("name")][:14],
        locations=[str(x) for x in a.get("locations", [])][:12],
        key_scenes=[str(x) for x in a.get("key_scenes", [])][:18],
        events=[str(x) for x in a.get("events", [])][:14],
        visual_keywords=[str(x) for x in a.get("visual_keywords", [])][:24],
        emotional_moments=[str(x) for x in a.get("emotional_moments", [])][:14],
        source=_llm.active_provider(eng_cfg).split()[0],
    )
    # the LLM often returns "Show (Season 2, Episode 9)" — a terrible SEARCH PREFIX (every query
    # inherits the parenthetical). Split it: clean title for queries + S02E09 episode hint.
    _em = re.search(r"\(?\s*Season\s*(\d{1,2})\s*,?\s*Episode\s*(\d{1,2})\s*\)?", analysis.movie_title, re.I)
    if _em:
        analysis.episode_hint = f"S{int(_em.group(1)):02d}E{int(_em.group(2)):02d}"
        analysis.movie_title = re.sub(
            r"\s*\(?\s*Season\s*\d{1,2}\s*,?\s*Episode\s*\d{1,2}\s*\)?", "",
            analysis.movie_title, flags=re.I).strip(" -–—:") or analysis.movie_title
    # else pull the SxxExx from the anchor scene's episode field (the LLM now names it there)
    if not analysis.episode_hint:
        for sc in analysis.anchor_scenes:
            m2 = re.search(r"S(\d{1,2})\s*E(\d{1,2})|Season\s*(\d{1,2}).{0,12}Episode\s*(\d{1,2})",
                           sc.get("episode", ""), re.I)
            if m2:
                g = [x for x in m2.groups() if x]
                analysis.episode_hint = f"S{int(g[0]):02d}E{int(g[1]):02d}"
                break
    # CORROBORATE. The hint arrives from a single LLM field with no second opinion, and it is wrong
    # often enough to matter: for the small-council scene in S03E10 "Mhysa" the model returned
    # "S04E01 Two Swords" while, in the SAME dict, quoting the S03E10 dialogue verbatim — and the
    # script said "This is the last episode of season three" out loud. Both signals were in hand and
    # neither was read, so one wrong string purged 354 shots and made wrong-episode clips anchors.
    # An unverified hint stays a fine search keyword; it just may not gate anything downstream.
    from . import era as _era_mod
    _h, _ok, _why = _era_mod.verified_episode_hint(analysis, script_text=script_text or "")
    analysis.episode_hint_verified = _ok
    analysis.episode_hint_reason = _why
    if _h:
        log(f"analyze: episode hint {_h!r} — {'CORROBORATED' if _ok else 'UNVERIFIED'} ({_why})"
            + ("" if _ok else "; it will NOT purge sources, confer anchor status, or "
                              "constrain beats"))
    def _beat_i(o):
        try:
            return int(o.get("i", -1))
        except (TypeError, ValueError):     # a non-numeric index must not crash the whole analyze
            return -1
    by_i = {_beat_i(o): o for o in data.get("beats", []) if isinstance(o, dict)}
    for b in beats:
        o = by_i.get(b.index)
        if not o:
            continue
        _apply_beat_direction(b, o)
    _sanitize_adjacent_quote_borrowing(beats)
    _grounding_counts = _record_beat_grounding_audit(analysis, beats)
    if _grounding_counts["exact_directives"]:
        log("analyze: beat-local exact grounding — "
            f"retained={_grounding_counts['grounded_exact']} "
            f"(sanitized={_grounding_counts['sanitized_exact']}), "
            f"downgraded={_grounding_counts['downgraded']} "
            f"(character={_grounding_counts['to_character_specific']}, "
            f"filler={_grounding_counts['to_generic_filler']})")
    # ROBUSTNESS: the LLM returns anchor_scenes inconsistently under load. If it omitted them, derive
    # the recurring (anchor) scene + video_type from the per-beat scene_queries so the footage strategy
    # never silently degrades to "scatter clips".
    if not analysis.anchor_scenes:
        vt, anch = _derive_anchors(beats, analysis.movie_title)
        analysis.anchor_scenes = anch
        if not analysis.video_type or analysis.video_type == "multi_scene":
            analysis.video_type = vt
        if anch:
            log(f"analyze: derived anchor → {anch[0].get('query','')[:70]!r} (type={analysis.video_type})")
    log(f"analyze: {analysis.source} · movie={analysis.movie_title!r} actors={len(analysis.actors)} "
        f"scenes={len(analysis.key_scenes)} · tone={analysis.tone!r} · type={analysis.video_type}")
    if analysis.synopsis:
        log(f"analyze: context → {analysis.synopsis[:140]}")
    if analysis.anchor_scenes:
        log(f"analyze: anchor scene(s) → " +
            " | ".join(s.get("query", "") or s.get("name", "") for s in analysis.anchor_scenes)[:200])
    return analysis, beats
