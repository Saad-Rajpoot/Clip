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
            "You know this film/show scene-by-scene. Using the WHOLE-STORY context above, for EACH of "
            "the beats below identify the EXACT moment it describes (resolve vague lines from the arc) "
            "so the viewer sees THAT specific scene, not generic footage. Reply ONLY a JSON array, one "
            "object per beat, same order:\n"
            '[{"i":int,"expected_visual":"concrete shot description of the exact moment",'
            '"scene_query":"a precise search string to find THIS exact scene clip — movie + character + '
            "the specific action/location (e.g. 'Game of Thrones Daenerys walks into fire unburnt "
            "Drogo funeral pyre'), or '' if the line is generic\",\"quote\":\"the iconic line of "
            "DIALOGUE actually spoken in this exact moment, verbatim if known (e.g. 'I am the dragon's "
            "daughter'), or '' if none\",\"shot_intent\":\"the KIND of shot that best serves this beat: "
            "establishing|action|emotional_closeup|reaction|symbolic|montage\","
            "\"required_entity\":\"the ONE specific "
            "actor/character/object/scene this line demands, or ''\","
            '"required_kind":"actor|character|object|scene|event|location|",'
            '"visual_policy":"ONE of exact_scene|character_specific|generic_filler|abstract_effect — '
            "exact_scene=a precise scene/quote/character-action/plot-event (needs the EXACT clip); "
            "character_specific=a named person/thing in general (needs the right subject, any clean shot); "
            "generic_filler=generic/explanatory line (any relevant clip); abstract_effect=abstract/"
            'emotional/meta with no literal visual (use an image/effect)",'
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
        if o.get("expected_visual"):
            b.expected_visual = str(o["expected_visual"])[:200]
        b.scene_query = str(o.get("scene_query", ""))[:120]
        b.quote = str(o.get("quote", ""))[:160]
        b.shot_intent = str(o.get("shot_intent", ""))[:24]
        b.required_entity = str(o.get("required_entity", ""))[:80]
        b.required_kind = str(o.get("required_kind", ""))[:20]
        b.emotion = str(o.get("emotion", ""))[:24]
        if o.get("specific"):
            b.is_specific_claim = True
        _vp = _policy.normalize(o.get("visual_policy", ""))   # LLM's explicit class (validated)
        if _vp:
            b.visual_policy = _vp
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
