"""Phase 2 — automatic source discovery.

From the script intelligence (movie + actors + scenes + keywords), auto-search multiple
providers for 15–20 highly relevant source videos — WITHOUT the user supplying any URLs.
Ranks by script relevance, rejects reaction/review/low-res/duplicate/too-short/too-long, caps
per-channel for diversity, and records every candidate (url, title, duration, resolution,
relevance, provider, reject reason). Metadata only — no download happens here.

Providers: yt-dlp `ytsearch` (no API key) + archive.org (public-domain pool). Resolution is
resolved by a bounded second pass on the top candidates so low-res can be rejected.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict

from .config import ClipConfig
from .analyze import ScriptAnalysis
from . import policy as _policy

# Titles that signal NOT clean in-show footage (talking-head / interview / annotated / parody / rip).
_REJECT_TITLE = re.compile(
    # NOTE: this whole pattern is wrapped \b(...)\b — the TRAILING \b means every alternative whose
    # match ends in a letter must be PLURAL-SAFE (write reactions?, not reaction), else the 's'/suffix
    # right after the stem kills the match. A 'MORE Reactors Reactions' upload slipped through exactly
    # because 'reaction'/'reacts?' missed the plurals. Keep new junk markers plural-safe.
    r"\b(reactions?|reactors?|reacts?|reacting|first time (watching|seeing)|watch ?along|watch party|"
    # audiobook uploads are a narrator over a static image — zero scene footage, and their ASR is
    # book narration that could poison quote mining (observed: 'Tyrion kills Tywin || Audiobook'
    # downloaded + indexed at 360p). Plural-safe per the trailing-\b note above.
    r"audio ?books?|"
    r"reviews?|reviewers?|review(ed|ing)?|commentar(y|ies)|podcasts?|tier list|ranking every|honest trailer|"
    r"everything wrong|cinemasins|recaps?|recap(ped|ping)?|explained|breakdowns?|easter eggs|"
    r"spoiler talk|let'?s play|live ?stream|full movie|whole movie|"
    # talking-head / press / cast content — NOT scene footage:
    r"interviews?|interviewed|interviewing|talks? about|talk show|jimmy kimmel|jimmy fallon|colbert|"
    r"graham norton|conan|ellen|red carpet|press (junket|conference|tour)|reveal(s|ed|ing)?|"
    r"featurettes?|on playing|filmisnow|film is now|movie extras|vs fan|plot holes?|q ?& ?a|"
    r"in real life|irl|behind the scenes|bloopers?|gag reel|outtakes?|"
    r"cast (reacts|interview|on|talk|remembers)|goodbye|auditions?|reunions?|then and now|"
    r"when i (died|was killed)|gag(s| reel|ging)?|mesmeriz|kissing scene|"
    # parody / BTS / fan-made / fake-future-season / press — NONE are clean in-show scene footage:
    r"the musical|musical (teaser|parody|number)|parod(y|ies)|comedy sketch|sketch comedy|skits?|"
    # 'Abridged' series are fan parodies with redubbed audio — one satisfied a TRIAL anchor slot
    # ('Tyrion's Trial 2: Medieval Boogaloo | ... Abridged') because its title shares scene tokens.
    # Foreign-language re-uploads carry burned subs + dubbed/subbed ASR that poisons quote mining
    # (plural-safe per the trailing-\b note above):
    r"abridged|legendado|legendada|subtitulado|sub ita|vostfr|deutsch|dublado|"
    r"set tour|tour the [a-z ]{0,20}set|studio tour|set visit|on the set of|set with the cast|"
    r"concept (trailer|art|teaser)|announcement trailer|teaser concept|fan[- ]?made|fan trailer|"
    r"fan film|fan edit|what if|re-?imagined|ai[- ]?(generated|trailer|concept)|deepfakes?|"
    # making-of / breakdown featurettes (show the film crew + equipment, NOT the scene) and
    # joke/meme re-edits (fart/crack/YTP edits, bad lip reading) — both aired junk on the
    # Tyrion render (an 'Anatomy of a Scene' BTS clip showed lights+reflectors mid-conclusion):
    r"anatomy of (a|the) scene|making[- ]of|the making of|fart edition|\bfart\b|"
    r"crack ?(vid|video|edition)|\bytp\b|bad lip reading|"
    r"record straight|sets? the record|addresses?[a-z' ]{0,18}(rumor|salary)|salary (rumor|talk)|"
    r"season (9|10|11|12|nine|ten|eleven|twelve)|red nose day|fundrais|"
    # ESSAY / COMMENTARY / VIDEO-ESSAY uploads — these narrate OVER the footage (the very thing we
    # are MAKING); they are not clean raw scene clips. The exact-scene clip we want is the raw upload,
    # not someone else's analysis of it.
    r"the real reason|reason why|here'?s why|the truth (about|behind)|what (really )?happened|"
    r"video essays?|deep[- ]?dive|\banaly(s|z)|\btheor(y|ies)\b|\bdecoded\b|the meaning of|"
    r"story behind|the story of|character arcs?|\bruins?\b [a-z' ]{0,30}(character|arc|story)|"
    # 'why <verb> ...' analysis titles — extended verbs catch 'Why Omitting ... Ruins ...'
    r"why [a-z]+ (is|was|are|were|did|didn'?t|never|really|actually|"
    r"ruins?|ruined|matters?|works?|fail(s|ed)?|destroys?|omitting|cutting|removing|changing)|"
    r"ending explained|things you (missed|didn'?t)|details you missed|hidden details|"
    # fan MUSIC-edits / AMVs (scene footage recolored + cut to a SONG) + recap 'clue(s)' essays —
    # observed leaking into the Tyrion render: 'Everything is Sadder with The Leftovers Music', and
    # 'Red Wedding - The clues' (whose red 'the murder of Rob Starks, Catelyn' overlay AIRED).
    r"the clues?\b|(sadder|better|cooler|epic|emotional) with (the )?[a-z' ]{0,20}music|"
    r"with the leftovers|set to [a-z' ]{0,18}music|\bamv\b|music video|"
    # PODCAST / VLOG / on-camera MAKEUP-BTS / transformation / table-read / retrospective + theory-
    # analysis uploads: modern people on camera or narration OVER footage, NOT clean scene clips.
    # Added after a Night-King render leaked a two-host PODCAST ('Bran's Elaborate Plan...'), a
    # White-Walker MAKEUP/prosthetics BTS clip ('I Got Transformed Into the Night King'), an actor
    # retrospective ('A Decade of Game of Thrones | John Bradley on...'), a table-read ('... Script
    # Reading'), and a theory upload ('Is The Night King A Targaryen? (SPIRAL SYMBOL Meaning)') —
    # all of whose titles dodged the gate. (Inside the \b(...)\b wrapper -> keep alts plural-safe.)
    r"\bvlog(s|ger|gers)?\b|\bmakeup\b|prosthetics?|"
    r"transformed into|i got transformed|turn(ed|ing) into (the|a)|becom(e|es|ing) the (night king|character)|"
    r"script reading|table read(ing|s)?|read[- ]?throughs?|a decade of|"
    # cast-interview / press-junket recap formats: '<Actor> Relives the Red Wedding', 'Breaks Down
    # the ... Scene', 'Looks Back at ...' — an actor ON CAMERA discussing the scene, not the scene
    # itself (observed: a Richard Madden interview matched a Tyrion beat and release-blocked it)
    r"\breliv(es?|ing)\b|breaks? down (the|his|her|every)|looks? back (at|on)|"
    r"remembers? (the|filming|shooting)|on filming|cast (interview|remembers|discusses)|"
    r"elaborate plans?|plan to (beat|defeat|stop|kill|save|destroy)|spiral symbols?|\bsymbolism\b|"
    r"hot/?badass|hot (scene|moment)s? timing|funniest|best moments compilation)\b", re.I)

# REACTION / FACECAM videos. These embed the REAL scene (its audio + a small inset of the show)
# behind a large webcam of modern people reacting — so they PASS dialogue-verification (they
# literally play the scene's lines) yet their FOOTAGE is a person in headphones on a couch, not the
# show. They must stay rejected even when anchor-verified. Kept separate from _REJECT_TITLE so the
# dialogue-verify re-admit step (which rescues clean essays) can specifically exclude them.
_REACTION_TITLE = re.compile(
    r"\b(reaction|reactions|reactor|reactors|reacting|reacts?|re-?act|"
    r"first time (watching|seeing|watch)|first watch(es)?|blind (watch|reaction)|"
    r"watch(es|ing)? (got|game of thrones|this)|live ?reaction|"
    r"couples? (reacts?|reaction|watch(es|ing)?)|we react|i react|husband|wife react)\b", re.I)

# NON-SHOW / WRONG-MEDIUM footage — looks battle-relevant by title but is NOT the live-action
# show: strategy-game battles (Total War etc.), lore/animated-map recreations, and epic-music
# AMVs. These leak in on a "war" topic and read instantly as irrelevant stock (game HUD numbers,
# CGI armies, montage). KEPT SEPARATE from _REJECT_TITLE so it can be tuned independently.
_NONSHOW_TITLE = re.compile(
    r"\b(total war|three kingdoms|mount ?& ?blade|bannerlord|crusader kings|ck2|ck3|"
    r"gameplay|game ?play|playthrough|mod(ded| showcase| review)?|"
    # RENDERED / CUT-SCENE GAME footage — the Telltale 'Game of Thrones' game (and other cinematic
    # games) look cartoonish/rendered, NOT the live-action show, but their cut-scene compilations are
    # titled 'Telltale' / 'all cutscenes' / 'game movie' with no 'gameplay' word, so they slipped the
    # gate and flooded a Theon/Ironborn render (the game is set in that exact era). Block them by name.
    r"telltale|cut[- ]?scenes?|game movie|all cutscenes|walkthrough|longplay|video ?game|"
    r"machinima|minecraft|roblox|fortnite|simulation|simulator|"
    r"two steps from hell|audiomachine|epic music|soundtrack|\bost\b|\bamv\b|gmv|"
    r"music video|lyric video|tribute|fan ?edit|"
    # live-concert / orchestral / single-SONG music uploads (composer performances, e.g. the
    # 'The Long Night Song' + 'Ramin Djawadi - The Night King Live Concert' that leaked) — not
    # scene footage. \bsong\b is lookahead-guarded so 'A Song of Ice and Fire' never false-trips.
    r"\bconcerts?\b|\bsong\b(?! of ice)|djawadi|orchestra(l)?|symphon(y|ic)|"
    r"animated|animation|cartoon|stick figure|lego|"
    # toy / clay / miniature recreations + AI-generated art — NOT live-action show footage
    r"playmobil|stop ?motion|claymation|plasticine|diorama|figurine|action figure|"
    r"miniature|recreated|recreation|reconstruct|reimagined in|"
    r"\bai\b ?(art|render|generated|animation|recreation)|midjourney|dall[- ]?e|"
    r"sora|runway|kling|stable diffusion|"
    r"all battles|every battle|battle simulator|cinematic battle|"
    r"\(\d{3,4}\s?ac\)|\d{3,4}\s?ac\b|house \w+ vs\.? house \w+|"
    r"\bvs\.?\b.{0,30}\bbattle\b|\bbattle\b.{0,30}\bvs\.?\b)", re.I)
_TYPE_BONUS = re.compile(r"\b(scene|clip|fight|battle|death|kills?|moment|confronts?|speech)\b", re.I)


def is_unwanted_source_title(title: str) -> bool:
    """True when a source's TITLE marks it as NON-scene footage: reaction / facecam / review /
    commentary / interview / essay / parody / non-show (game/AMV/animated). Such uploads are gated
    out of clip selection at discovery, and their indexed keyframes (reactor facecams, channel logos,
    'GAME OF THRONES' title cards, burned-in reactor-name overlays like 'AmberReacts') must ALSO be
    excluded from the image-recovery STILL pool — otherwise a beat with no exact footage can pull a
    reactor frame as a 'source-frame' still. Mirrors the discovery-time _reject() title gates."""
    t = title or ""
    return bool(_REJECT_TITLE.search(t) or _REACTION_TITLE.search(t) or _NONSHOW_TITLE.search(t))


@dataclass
class SourceCandidate:
    url: str
    id: str = ""
    title: str = ""
    provider: str = "youtube"          # youtube | archive
    duration: float = 0.0
    height: int = 0
    channel: str = ""
    view_count: int = 0
    relevance: float = 0.0
    query: str = ""
    permission_hint: str = ""          # e.g. "public_domain" for archive.org listings
    reject_reason: str = ""            # "" = kept
    download_status: str = "pending"
    # True = this candidate's own subtitles/captions contain the anchor scene's spoken lines —
    # content-level proof it IS the exact scene (title matching can't give that)
    anchor_verified: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SourceCandidate":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


_STOPQ = set("the a an of to in on at for and or but with from into as it its this that "
             "his her their he she they him them you your we our us is are was were be been "
             "like who which what when where why how then than too very just only also not no "
             "one two new got had has have do does did will would can could should may must".split())


def _scene_quotes(segments, analysis=None) -> list[str]:
    """Distinctive spoken lines to lock onto — the anchor scenes' VERBATIM dialogue first (the
    LLM is asked for exact lines there; per-beat quotes are often script paraphrases)."""
    out, seen = [], set()

    def add(qt):
        qt = (qt or "").strip().strip('"“”')
        if len(qt) >= 12 and len(qt.split()) >= 3 and qt.lower() not in seen:
            seen.add(qt.lower())
            out.append(qt)

    for sc in (getattr(analysis, "anchor_scenes", None) or []) if analysis is not None else []:
        for d in (sc.get("dialogue") or []):
            add(d)
    for s in (segments or []):
        add(getattr(s, "quote", "") or "")
    return out


def anchor_queries(a: ScriptAnalysis, segments=None) -> list[str]:
    """MULTI-VARIANT searches for the anchor scene(s). One long verbatim query routinely misses
    the raw-scene uploads (they're titled 'Bronn and The Hound | S02E09', not the LLM's prose) —
    so also search character pairs, the scene NAME, the iconic QUOTES, and the episode code."""
    mv = a.movie_title.strip()
    ep = (getattr(a, "episode_hint", "") or "").strip()
    chars = [c.get("name", "").strip() for c in (a.characters or []) if c.get("name")]
    out: list[str] = []

    def add(s):
        s = " ".join((s or "").split())
        if len(s) > 6:
            out.append(s[:100])

    for sc in (getattr(a, "anchor_scenes", None) or []):
        sq = (sc.get("query") or sc.get("name") or "").strip()
        nm = (sc.get("name") or "").strip()
        if sq:
            add(sq if mv.lower() in sq.lower() else f"{mv} {sq}")
        if nm and nm.lower() != sq.lower():
            add(nm if mv.lower() in nm.lower() else f"{mv} {nm}")
        # character-pair variants from the anchor text (matches how raw clips are titled)
        toks = set(re.findall(r"\w+", (sq + " " + nm).lower()))
        sc_chars = [c for c in chars if any(w in toks for w in re.findall(r"\w+", c.lower()))][:2]
        if len(sc_chars) == 2:
            add(f"{mv} {sc_chars[0]} and {sc_chars[1]} scene")
            if ep:
                add(f"{mv} {ep} {sc_chars[0]} {sc_chars[1]}")
        elif sc_chars:
            add(f"{mv} {sc_chars[0]} scene {' '.join(list(toks)[:2])}")
        if ep:
            add(f"{mv} {ep} scene")
    # iconic dialogue = the sharpest key to the exact upload
    for qt in _scene_quotes(segments, a)[:3]:
        add(f"{mv} {qt}")
    # DATA-DRIVEN canonical scene phrase: the LLM's anchor query is sometimes a vague theme
    # ("chair test"), but the PER-BEAT scene_queries often name the real scene/setting
    # ("Small Council", "throne room"). Mine the most frequent multi-word location/scene phrase
    # across the beats and search it deeply — this finds the raw scene even when the anchor was weak.
    for ph in _dominant_scene_phrases(segments, a)[:2]:
        add(f"{mv} {ph} scene")
        if ep:
            add(f"{mv} {ep} {ph}")
    # ANGLE / VERSION variants for the PRIMARY character pair — different uploaders title the SAME
    # scene differently ("full scene", "HD", "4K", "no music"), so these surface DISTINCT uploads
    # (and thus distinct camera angles / crops / edits) of the exact same scene. More visually
    # different REAL footage for the matcher = less repetition WITHOUT leaving the scene. Single-
    # scene deep-dives benefit most; env-gated so it can be tuned off.
    import os as _os_av
    if _os_av.environ.get("VIDLORE_CLIPSTUDIO_ANGLE_VARIANTS", "1").strip() not in ("0", "false", "no"):
        pair = [c for c in chars][:2]
        base = (f"{mv} {pair[0]} and {pair[1]}" if len(pair) == 2
                else (f"{mv} {pair[0]}" if pair else (f"{mv} {ep}".strip() if ep else mv)))
        for variant in ("full scene", "scene HD", "scene 4k", "extended scene",
                        "full scene no music", "complete scene"):
            add(f"{base} {variant}")
    # dedupe preserving order
    seen, ded = set(), []
    for x in out:
        if x.lower() not in seen:
            seen.add(x.lower())
            ded.append(x)
    return ded[:18]


# recognized "scene/setting" words that name a canonical GoT-style moment — a multi-word phrase
# built around one of these, recurring across beats, is the real scene to search for
_SCENE_NOUNS = re.compile(
    r"\b(council|throne|tavern|wedding|battle|trial|duel|dungeon|crypt|sept|hall|chamber|"
    r"meeting|feast|court|arena|pit|cell|brothel|camp|gate|wall|bridge|courtyard|tent|"
    r"room|table|chair|chairs)\b", re.I)


def _dominant_scene_phrases(segments, analysis) -> list[str]:
    """The most frequent 1-2 word scene/setting phrases across the beats' scene_queries — a
    data-driven canonical-scene signal independent of the LLM's anchor query."""
    from collections import Counter
    mvt = {w for w in re.findall(r"\w+", (getattr(analysis, "movie_title", "") or "").lower())}
    bigrams: Counter = Counter()
    unigrams: Counter = Counter()
    for s in (segments or []):
        ws = [w for w in re.findall(r"[a-z]+", (getattr(s, "scene_query", "") or "").lower())
              if w not in mvt and w not in _STOPQ and len(w) > 2]
        for i, w in enumerate(ws):
            if _SCENE_NOUNS.search(w):
                unigrams[w] += 1
                if i > 0:
                    bigrams[f"{ws[i-1]} {w}"] += 1
    out = []
    for ph, n in bigrams.most_common(4):
        if n >= 2:
            out.append(ph)
    for w, n in unigrams.most_common(4):
        if n >= 2 and not any(w in p for p in out):
            out.append(w)
    return out


def _era_hint(a: ScriptAnalysis) -> str:
    """'season N' from the analysis episode_hint (SxxEyy) — used to bias character B-roll to the
    RIGHT era so a single-scene deep-dive doesn't pull a later-season look of the character."""
    import re as _re
    m = _re.search(r"s0*(\d+)\s*e", (getattr(a, "episode_hint", "") or ""), _re.I)
    return f"season {int(m.group(1))}" if m else ""


def build_queries(a: ScriptAnalysis, segments=None) -> list[str]:
    """Targeted search queries. The big relevance lever is PER-BEAT queries: search the SPECIFIC
    scene each narration line needs (e.g. 'Game of Thrones Tyrion kills Tywin crossbow'), plus the
    analyzer's key scenes — not just generic 'movie scene' (which returns trailers + interviews)."""
    q: list[str] = []
    mv = a.movie_title.strip()
    yr = a.year.strip()
    # 0) ANCHOR SCENES come FIRST — all variants (verbatim, character-pair, scene name, quotes,
    #    episode code). For a single-scene deep-dive the raw footage of THIS scene IS the video.
    q.extend(anchor_queries(a, segments))
    if mv:
        q.append(f"{mv} {yr} scene".strip())
        if getattr(a, "video_type", "") != "single_scene":
            q.append(f"{mv} official trailer")
    # 1) the analyzer's key scenes ARE the specific moments — query every one
    for sc in a.key_scenes[:14]:
        if mv and sc:
            q.append(f"{mv} {sc}"[:90])
    # 2) PER-BEAT: the LLM's PRECISE scene query is the strongest signal — it names the exact moment,
    #    so search it verbatim (this is what actually downloads the on-script scene). Fall back to the
    #    entity + action keywords when the line is generic. Collected SEPARATELY with each beat's
    #    timeline position so the query budget can be STRATIFIED across early/middle/late beats — the
    #    old flat segment-order truncation starved late exact beats (a 238-beat script never searched
    #    its coin-handoff / blind-Arya scenes because the first ~40 beats filled the 60-query cap).
    _beat_q: list = []                                 # (seg_index, query)
    _seg_list = list(segments or [])
    _maxidx = max((int(getattr(s, "index", i)) for i, s in enumerate(_seg_list)), default=1) or 1
    for i, s in enumerate(_seg_list):
        # POLICY budget (req. 8): only EXACT/CHARACTER beats earn a dedicated scene-query search —
        # generic_filler reuses the existing pool and abstract_effect wants an image, so spending
        # discovery/download budget on them is waste.
        if not _policy.wants_discovery_query(s):
            continue
        _si = int(getattr(s, "index", i))
        sq = (getattr(s, "scene_query", "") or "").strip()
        if sq and len(sq) > 6:
            # the LLM usually already prefixes the movie; only add it if missing
            _beat_q.append((_si, (sq if mv.lower() in sq.lower() else f"{mv} {sq}").strip()[:100]))
        ent = (getattr(s, "required_entity", "") or "").strip()
        kws = [k for k in (getattr(s, "keywords", []) or [])
               if k not in _STOPQ and k.lower() != ent.lower()][:3]
        if mv and ent and len(ent) > 2 and ent.lower() not in ("game of thrones", "the cast"):
            _beat_q.append((_si, f"{mv} {ent} {' '.join(kws)}".strip()[:80]))
        elif mv and len(kws) >= 2:
            _beat_q.append((_si, f"{mv} {' '.join(kws)}".strip()[:80]))
    # 3) a few character/actor scene queries for face coverage — ERA-ANCHORED so a single-scene
    #    deep-dive gets ON-ERA footage of each character (e.g. long-hair S1 Cersei, not short-hair
    #    S6 Cersei spliced in from a multi-season compilation). The season comes from episode_hint.
    _era = _era_hint(a)
    for ch in a.characters[:6]:
        nm = (ch.get("name") or "").strip()
        if nm and mv:
            q.append(f"{mv} {nm} scene")
            if _era:
                q.append(f"{mv} {nm} {_era} scene")
    # 3b) B-ROLL / CONTEXT VARIETY — distinct-but-relevant footage so the editor isn't forced to
    #     RE-AIR the same handful of scene shots (the single-scene repetition problem). Use only
    #     LOCATION establishing shots (safe). NOTE: do NOT query analysis `events` here — a GoT
    #     event like "Fall of the Mad King"/"Death of Lyanna" returns HOUSE OF THE DRAGON (the
    #     prequel) footage, contaminating the video with the wrong show. Era/character queries
    #     above already add on-era variety without that cross-show risk.
    for loc in (a.locations or [])[:3]:
        if mv and loc and loc.lower() not in ("westeros",):     # too generic alone
            q.append(f"{mv} {loc}".strip()[:80])

    # --- combine: structural queries (anchors/key-scenes/characters/locations) FIRST, then the
    #     per-beat exact queries STRATIFIED across the timeline so late exact beats are never starved.
    _mv_toks = {w for w in re.findall(r"[a-z0-9']+", mv.lower())}

    def _eqkey(s: str) -> str:
        """Equivalent-query key: drop the movie title + stopwords, sort the rest — so 'Tywin Arya
        dinner Harrenhal' and 'Arya dinner at Harrenhal Tywin' collapse to ONE search (dedupe
        equivalent scene queries, req.). Single DIGITS are kept (a season/episode number is a real
        distinguisher — 'Tywin scene season 2' must NOT collapse into 'Tywin scene season 6')."""
        toks = [w for w in re.findall(r"[a-z0-9']+", s.lower())
                if w not in _mv_toks and w not in _STOPQ and (len(w) > 1 or w.isdigit())]
        return " ".join(sorted(set(toks))) or s.lower().strip()

    seen_key, out = set(), []

    def _emit(x: str):
        x = (x or "").strip()
        if not x:
            return
        k = _eqkey(x)
        if k in seen_key:
            return
        seen_key.add(k)
        out.append(x)

    for x in q:                                          # structural queries keep priority/order
        _emit(x)
    # stratify the per-beat queries into early/middle/late thirds and round-robin interleave, so a
    # forced truncation (very long script) is SPREAD across the whole timeline instead of dropping
    # every late beat. Dedupe equivalent queries within the beat pool first.
    _bseen, _b = set(), []
    for _si, x in _beat_q:
        k = _eqkey(x)
        if k not in _bseen:
            _bseen.add(k)
            _b.append((_si, x))
    thirds = ([], [], [])
    for _si, x in _b:
        thirds[min(2, int(3 * _si / max(1, _maxidx + 1)))].append(x)
    _il = 0
    while any(thirds):
        for t in thirds:
            if t:
                _emit(t.pop(0))
        _il += 1
    from .config import _i as _cfg_i
    _qcap = _cfg_i("VIDLORE_CLIPSTUDIO_QUERY_CAP", 60)    # tolerant: bad value → default
    _hard = _cfg_i("VIDLORE_CLIPSTUDIO_QUERY_CAP_MAX", 200)
    # scale the cap with the beat count so a long script can search ALL its exact scenes (more
    # queries = better candidate discovery only; downloads stay separately budgeted). Bounded by
    # _hard so a pathological script can't explode the search API call count.
    _qcap = max(_qcap, min(_hard, len(out)))
    return out[:_qcap]


# Provider call statuses — callers can now tell these APART (they were all a silent []):
#   ok         >=1 result
#   empty      the call succeeded and there genuinely were no results (never retried)
#   throttled  HTTP 429 / rate-limit language (retried with a longer backoff)
#   timeout    socket/read timeout (retried)
#   transport  any other network/extract failure (retried)
STATUS_OK, STATUS_EMPTY = "ok", "empty"
STATUS_THROTTLED, STATUS_TIMEOUT, STATUS_TRANSPORT = "throttled", "timeout", "transport"
_RETRYABLE = (STATUS_THROTTLED, STATUS_TIMEOUT, STATUS_TRANSPORT)


def _classify_net_error(exc) -> str:
    """Map a provider exception to a status. Message-based for yt-dlp (it wraps HTTP errors
    into DownloadError strings); type-based where real types exist."""
    import socket
    try:
        import requests as _rq
        if isinstance(exc, _rq.Timeout):
            return STATUS_TIMEOUT
        if isinstance(exc, _rq.RequestException):
            _resp = getattr(exc, "response", None)
            if _resp is not None and getattr(_resp, "status_code", 0) == 429:
                return STATUS_THROTTLED
            return STATUS_TRANSPORT
    except Exception:
        pass
    if isinstance(exc, (socket.timeout, TimeoutError)):
        return STATUS_TIMEOUT
    msg = str(exc).lower()
    if "429" in msg or "too many requests" in msg or "rate limit" in msg or "rate-limit" in msg:
        return STATUS_THROTTLED
    if "timed out" in msg or "timeout" in msg:
        return STATUS_TIMEOUT
    return STATUS_TRANSPORT


def _ytsearch_ex(query: str, n: int) -> tuple[list[SourceCandidate], str]:
    """YouTube search -> (candidates, status). Never raises."""
    import yt_dlp
    opts = {"quiet": True, "no_warnings": True, "skip_download": True,
            "extract_flat": True, "noplaylist": True, "socket_timeout": 20}
    out = []
    try:
        with yt_dlp.YoutubeDL(opts) as y:
            info = y.extract_info(f"ytsearch{n}:{query}", download=False)
        for e in (info.get("entries") or []):
            if not e:
                continue
            vid = e.get("id", "")
            out.append(SourceCandidate(
                url=e.get("url") or (f"https://www.youtube.com/watch?v={vid}" if vid else ""),
                id=vid, title=e.get("title", "") or "", provider="youtube",
                duration=float(e.get("duration") or 0), channel=e.get("channel") or e.get("uploader") or "",
                view_count=int(e.get("view_count") or 0), query=query))
    except Exception as exc:                              # noqa: BLE001 — classified, never raised
        return [c for c in out if c.url], _classify_net_error(exc)
    out = [c for c in out if c.url]
    return out, (STATUS_OK if out else STATUS_EMPTY)


def _ytsearch(query: str, n: int) -> list[SourceCandidate]:
    return _ytsearch_ex(query, n)[0]


def _archive_search_ex(query: str, n: int) -> tuple[list[SourceCandidate], str]:
    """archive.org search -> (candidates, status). Never raises."""
    import requests
    out = []
    try:
        r = requests.get("https://archive.org/advancedsearch.php",
                         params={"q": f"({query}) AND mediatype:(movies)",
                                 "fl[]": ["identifier", "title"], "rows": n, "output": "json"},
                         timeout=20, headers={"User-Agent": "ClipStudio/0.1 (testing)"})
        if getattr(r, "status_code", 200) == 429:
            return out, STATUS_THROTTLED
        for d in r.json().get("response", {}).get("docs", []):
            ident = d.get("identifier", "")
            if not ident:
                continue
            out.append(SourceCandidate(
                url=f"https://archive.org/details/{ident}", id=ident,
                title=str(d.get("title", "") or ident), provider="archive",
                permission_hint="public_domain", query=query))
    except Exception as exc:                              # noqa: BLE001 — classified, never raised
        return out, _classify_net_error(exc)
    return out, (STATUS_OK if out else STATUS_EMPTY)


def _archive_search(query: str, n: int) -> list[SourceCandidate]:
    return _archive_search_ex(query, n)[0]


def _anchor_token_sets(a: ScriptAnalysis) -> list[set]:
    """Scene-specific token set per anchor (movie/stop words stripped) — used to recognise candidates
    that ARE the core scene the video is about, so they dominate the footage."""
    mtoks = {w for w in re.findall(r"\w+", a.movie_title.lower()) if len(w) > 2}
    out = []
    for sc in (getattr(a, "anchor_scenes", None) or []):
        toks = {w for w in re.findall(r"\w+", (sc.get("query", "") + " " + sc.get("name", "")).lower())
                if len(w) > 2 and w not in _STOPQ and w not in mtoks}
        if toks:
            out.append(toks)
    return out


def _relevance(c: SourceCandidate, a: ScriptAnalysis, anchor_sets: list[set] | None = None) -> float:
    t = c.title.lower()
    score = 0.0
    single = getattr(a, "video_type", "") == "single_scene"
    mt = [w for w in re.findall(r"\w+", a.movie_title.lower()) if len(w) > 2]
    abbr = "".join(w[0] for w in a.movie_title.split() if w) if len(a.movie_title.split()) >= 2 else ""  # "game of thrones" -> "got"
    # 1) MOVIE present — relaxed + DELIBERATELY not dominant. A fan-cut scene clip titled "GOT - Bronn
    #    sings Rains of Castamere" (no literal "game of thrones") used to score ~0 here and got buried
    #    under generic "Game of Thrones Battle ..." uploads. Accept the abbreviation, and weight it low
    #    so the SPECIFIC-scene signal (below) decides.
    if mt and all(w in t for w in mt):
        score += 0.22
    elif (abbr and len(abbr) >= 2 and re.search(rf"\b{re.escape(abbr)}\b", t)) or any(w in t for w in mt):
        score += 0.12
    # 2) QUERY MATCH — the dominant signal. How many of the SPECIFIC tokens of the query that surfaced
    #    this candidate appear in its title. A title echoing "bronn rains castamere tavern" IS the exact
    #    scene we searched for; a same-movie upload that merely shares the franchise name is not. This
    #    is what makes the on-script raw scene outrank a generic battle/trailer upload.
    mvset = set(mt) | ({abbr} if abbr else set())
    qw = [w for w in re.findall(r"\w+", (c.query or "").lower())
          if len(w) > 2 and w not in mvset and w not in _STOPQ]
    if qw:
        hits = sum(1 for w in set(qw) if w in t)
        score += min(0.55, 0.12 * hits)
    # 3) character / actor surnames present (capped)
    asc = 0.0
    for actor in a.actors:
        sn = actor.split()[-1].lower() if actor.split() else ""
        if len(sn) > 2 and sn in t:
            asc += 0.10
    score += min(0.20, asc)
    # 4) scene keywords
    kws = set()
    for s in (a.key_scenes + a.visual_keywords):
        for w in re.findall(r"\w+", s.lower()):
            if len(w) > 3:
                kws.add(w)
    score += min(0.20, 0.03 * sum(1 for w in kws if w in t))
    if _TYPE_BONUS.search(t):
        score += 0.10
    # 5) a TRAILER/teaser/promo is a montage of many scenes — almost never the exact moment a
    #    deep-dive needs, and its fast cuts wreck single-scene continuity. Demote it.
    if re.search(r"\btrailer\b|\bteaser\b|\bpromo\b", t):
        score -= 0.12
    # 5c) a CHARACTER COMPILATION ("all scenes / best scenes / best of / every moment / supercut")
    #     is a montage spanning the whole show — for a SINGLE-SCENE deep-dive it scatters footage
    #     across dozens of unrelated moments instead of holding the one scene. Demote hard so the
    #     raw single-scene upload outranks it. (Harmless for multi_scene, which wants breadth.)
    if single and re.search(r"\ball scenes?\b|\bbest scenes?\b|\bbest of\b|\bevery (scene|moment|"
                            r"time)\b|\bsupercut\b|\bcompilation\b|\ball (his|her|their) \w+\b|"
                            r"\bmoments\b|\bfunniest\b|\bgreatest\b", t):
        score -= 0.30
    # 6c) EPISODE-CODE match — a raw-scene upload titled with the exact SxxExx is almost always
    #     the scene itself, not a compilation. Strong boost.
    ep = (getattr(a, "episode_hint", "") or "").strip().lower()
    if ep and ep in t.replace(" ", ""):
        score += 0.25
    # 5b) MUSIC-ONLY uploads (lyric videos, covers, OST rips) title-match scene queries that
    #     contain a song name ("Rains of Castamere") but contain ZERO scene footage — they were
    #     crowding out the real scene. Hard demotion.
    if _MUSIC_ONLY_RX.search(t):
        score -= 0.45
    if a.year and a.year in t:
        score += 0.04
    if c.provider == "archive":
        score += 0.04                          # mild preference: cleaner rights posture
    # 6) ANCHOR BOOST — this candidate's title echoes the CORE scene the whole video is about. This is
    #    the decisive signal that elevates the on-script raw scene above an equally-well-titled but
    #    OFF-topic same-movie upload (a Goldroad battle clip matches its own query just as well, but it
    #    is not what THIS video is about). For single-scene videos this makes the one scene dominate.
    if anchor_sets:
        best = max((sum(1 for w in s if w in t) for s in anchor_sets), default=0)
        if best >= 2:
            score += min(0.45, 0.15 * best)
    return min(1.0, round(max(0.0, score), 4))


def _reject(c: SourceCandidate, cfg: ClipConfig) -> str:
    import os
    if _REJECT_TITLE.search(c.title):
        return "reaction/review/annotated"
    # REACTION / facecam gate (separate regex with the PLURAL forms 'reactions/reactors'). _REJECT_TITLE
    # is wrapped \b(...)\b, so its singular alternatives ('reaction','reacts?') MISS the plurals — a
    # 'MORE Reactors Reactions ...' upload slipped past _REJECT_TITLE, got downloaded+indexed, and its
    # reactor-facecam keyframes leaked into the image-recovery still pool. _REACTION_TITLE lists the
    # plural/derivative forms explicitly, so it must be checked here too (root gate, before download).
    if _REACTION_TITLE.search(c.title):
        return "reaction/facecam"
    if os.environ.get("VIDLORE_CLIPSTUDIO_NONSHOW_GATE", "1").strip() not in ("0", "false", "no") \
            and _NONSHOW_TITLE.search(c.title):
        return "non-show (game/AMV/animated/wrong-medium)"
    if c.duration and c.duration < cfg.discover_min_sec:
        return f"too short ({c.duration:.0f}s)"
    if c.duration and c.duration > cfg.discover_max_sec:
        return f"too long ({c.duration:.0f}s)"
    if c.height and c.height < cfg.discover_min_height:
        return f"low-res ({c.height}p)"
    return ""


def _norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()


_MUSIC_ONLY_RX = re.compile(
    r"\blyrics?\b|\bcover\b|\bacoustic\b|\bslowed\b|\breverb\b|\binstrumental\b|"
    r"\bpiano\b|\bkaraoke\b|\bsoundtrack\b|\bost\b|\btheme song\b|\bmusic video\b|"
    r"\b(10|1) hours?\b|\bremix\b|\bepic version\b|\bextended version\b|\bbest version\b|"
    r"\bfull song\b", re.I)

# WRONG-SHOW / WRONG-INSTALLMENT guard. A franchise prequel/spin-off depicts the SAME world with a
# DIFFERENT cast and era, so its footage looks plausibly on-topic (Targaryens, the Iron Throne) but
# is the wrong production — e.g. a Game of Thrones (S1, Robert/Cersei) video must NOT pull House of
# the Dragon clips (Aegon II's green-wildfire coronation, white-haired Targaryens over dragon eggs).
# Keyed on the TARGET show: for each franchise, sibling installments whose titles disqualify a clip.
# Only the franchise we're editing is consulted, so a HotD video can still legitimately use HotD.
_WRONGSHOW_SIBLINGS = {
    "game of thrones": re.compile(
        r"\bhouse of the dragon\b|\bhotd\b|\bdragon'?s? prequel\b|"
        r"\bdunk (and|&) egg\b|\bknight of the seven kingdoms\b|\bhedge knight\b|"
        # HotD-exclusive characters/dragons/topics — a sibling clip titled by CHARACTER (not by the
        # show name) otherwise slips this gate (observed: "Aemond Destroys King Aegon…" aired as the
        # cold-open of a Daenerys video). Only names that cannot appear in a genuine GoT scene title
        # (Aegon/Viserys/Rhaenys are shared backstory, so left out to avoid false positives).
        r"\baemond\b|\brhaenyra\b|\balicent\b|\bdaemon targaryen\b|\botto hightower\b|"
        r"\bcriston cole\b|\bdance of the dragons\b|\btargaryen civil war\b|"
        r"\bvhagar\b|\bcaraxes\b|\blucerys\b|\bjacaerys\b|\bgreen council\b", re.I),
    "house of the dragon": re.compile(
        r"\bgame of thrones\b|\bgot s\d|\bgot season\b", re.I),
}


def _wrong_installment(target_title: str, candidate_title: str) -> bool:
    """True if candidate_title declares a DIFFERENT installment of target_title's franchise.

    Catches sibling-show contamination that survives every other gate because the clip is
    'right world, wrong production'. Self-references are excluded: a HotD-target never rejects
    HotD, and the GoT pattern is only consulted when GoT is the target."""
    tl = (target_title or "").lower()
    cl = (candidate_title or "").lower()
    for show, rx in _WRONGSHOW_SIBLINGS.items():
        if show not in tl or not rx.search(cl):
            continue
        # Don't reject a clip that ALSO names the target show — it's a self/comparison reference
        # ("GoT vs House of the Dragon"), and the target's own footage may well be in it.
        if show in cl:
            continue
        return True
    return False


# ---------------------------------------------------------------------------
# CONTENT-LEVEL anchor verification: does a candidate's own subtitle track contain the anchor
# scene's spoken lines? A title can lie ("Rains of Castamere" fits three different scenes);
# the dialogue can't. Costs one metadata call + one tiny caption download per candidate.
# ---------------------------------------------------------------------------

# Per-URL result caches (process-lifetime). A render calls _fetch_subs_text / the HD probe for
# overlapping URL sets on every anchor-verify AND every bounded-recovery round — the measured
# recovery rounds re-paid ~330-510s each mostly re-fetching the same URLs. A video's caption
# track and real resolution ladder are effectively immutable for the lifetime of one render
# process, so a POSITIVE result is safe to reuse. Failures ('' / h<=0) are NEVER cached — a
# transient fetch failure must stay retryable on the next round, exactly as before.
_SUBS_TEXT_CACHE: dict = {}
_PROBE_H_CACHE: dict = {}


def _fetch_subs_text(url: str, timeout: int = 45) -> str:
    """Best-effort English subtitle/auto-caption text for a video ('' on any failure).
    HD runtime FIRST: plain clients no longer receive caption tracks (PO-token gated), so the
    in-process path below almost always finds none."""
    from . import perf_metrics as _pm_s
    if url in _SUBS_TEXT_CACHE:
        _pm_s.incr("discover.subs.cache_hit")
        return _SUBS_TEXT_CACHE[url]
    _pm_s.incr("discover.subs.fetch")
    _got = _fetch_subs_text_uncached(url, timeout)
    if _got:
        _SUBS_TEXT_CACHE[url] = _got                     # positive results only — '' stays retryable
    return _got


def _fetch_subs_text_uncached(url: str, timeout: int = 45) -> str:
    import json as _json
    import urllib.request
    try:
        from . import hd_download as _hd
        if _hd.available() and _hd.is_youtube(url) and _hd.ensure_pot_server():
            import glob
            import shutil
            import subprocess
            import tempfile
            d = tempfile.mkdtemp(prefix="cs_subs_")
            try:
                cmd = [_hd.HD_PY, "-m", "yt_dlp",
                       "--cookies-from-browser", _hd.COOKIES_BROWSER,
                       "--extractor-args",
                       f"youtubepot-bgutilhttp:base_url=http://127.0.0.1:{_hd.POT_PORT}",
                       "--skip-download", "--write-auto-subs", "--write-subs",
                       "--sub-langs", "en.*", "--sub-format", "json3",
                       "--no-playlist", "--no-warnings", "--quiet",
                       "-o", f"{d}/s.%(ext)s", url]
                p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                                   env=_hd._env_with_runtimes())
                files = sorted(glob.glob(f"{d}/s*.json3"))
                if p.returncode == 0 and files:
                    data = _json.loads(open(files[0], encoding="utf-8").read())
                    return " ".join(seg.get("utf8", "") for ev in data.get("events", [])
                                    for seg in (ev.get("segs") or []))
            finally:
                shutil.rmtree(d, ignore_errors=True)
    except Exception:
        pass
    import yt_dlp
    try:
        with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "skip_download": True,
                               "noplaylist": True, "socket_timeout": timeout}) as y:
            info = y.extract_info(url, download=False)
    except Exception:
        return ""
    tracks = {}
    for pool in (info.get("subtitles") or {}, info.get("automatic_captions") or {}):
        for lang, entries in pool.items():
            if lang.startswith("en") and entries and lang not in tracks:
                tracks[lang] = entries
    for entries in tracks.values():
        ent = next((e for e in entries if e.get("ext") == "json3"), None) or \
              next((e for e in entries if e.get("ext") in ("srv1", "vtt")), None)
        if not (ent and ent.get("url")):
            continue
        try:
            with urllib.request.urlopen(ent["url"], timeout=timeout) as r:
                raw = r.read(2 << 20).decode("utf-8", "ignore")
        except Exception:
            continue
        if ent.get("ext") == "json3":
            try:
                data = _json.loads(raw)
                return " ".join(seg.get("utf8", "") for ev in data.get("events", [])
                                for seg in (ev.get("segs") or []))
            except Exception:
                continue
        # srv1/vtt: strip tags/timing lines
        txt = re.sub(r"<[^>]+>", " ", raw)
        txt = "\n".join(l for l in txt.splitlines()
                        if l.strip() and "-->" not in l and not l.strip().isdigit()
                        and not l.startswith(("WEBVTT", "Kind:", "Language:")))
        return txt
    return ""


def _norm_ws(s: str) -> list[str]:
    return [w for w in re.findall(r"[a-z']+", (s or "").lower()) if w]


def _subs_quote_hits(subs: str, quotes: list[str]) -> int:
    """How many DISTINCT quotes are spoken in this caption text."""
    if not subs or not quotes:
        return 0
    tw = _norm_ws(subs)
    tstr = " " + " ".join(tw) + " "
    hits = 0
    for q in quotes:
        qw = _norm_ws(q)[:6]                       # leading phrase is enough, ASR drifts later
        hit = len(qw) >= 3 and (" " + " ".join(qw) + " ") in tstr
        if not hit and len(qw) >= 4:               # tolerate one-word ASR drift: 3-word sub-runs
            hit = any((" " + " ".join(qw[i:i + 3]) + " ") in tstr for i in range(len(qw) - 2))
        if hit:
            hits += 1
    return hits


def _subs_contain_quote(subs: str, quotes: list[str]) -> bool:
    """VERIFIED = at least one of the scene's DIALOGUE lines is spoken in the captions. Dialogue
    is scene-specific (a different scene of the same show won't say "you're just like me, only
    smaller"), so one solid hit is decisive. The lyric-poisoning guard is NOT a hit-count — it's
    the music-title exclusion in verify_anchor_candidates (a cover/lyric upload never reaches
    here), because the LLM occasionally lists only ONE correct line among paraphrased ones, and a
    2-hit rule then rejected the real scene."""
    return _subs_quote_hits(subs, quotes) >= 1


def verify_anchor_candidates(kept: list, analysis, segments, *, limit: int = 14,
                             anchor_qset=None, extra=None, progress=None) -> int:
    """Mark candidates whose captions SPEAK the anchor scene's quotes (anchor_verified) and boost
    them; candidates that merely echo song/scene words in the title get no boost. `extra` =
    title-REJECTED essay/breakdown candidates — if one verifies, it carries the exact scene and
    earns its way back in (its talking-head/text frames are filtered later by the OCR/graphic
    gates, and ClipStudio strips source audio anyway). Returns verified count."""
    quotes = _scene_quotes(segments, analysis)
    if not quotes:
        return 0
    anchor_sets = _anchor_token_sets(analysis)
    aq = {q.lower() for q in (anchor_qset or set())}

    def _tok_hit(c):
        t = c.title.lower()
        return any(sum(1 for w in s if w in t) >= 2 for s in anchor_sets) if anchor_sets else False

    pool = []
    for c in list(kept) + list(extra or []):
        if c.provider != "youtube":
            continue
        if _MUSIC_ONLY_RX.search(c.title):
            continue                       # a lyric/cover video can't carry the scene
        if _tok_hit(c) or c.relevance >= 0.55 or (c.query or "").lower() in aq:
            pool.append(c)
    # spend the budget on the LIKELIEST first: surfaced by an anchor query, then relevance
    pool.sort(key=lambda c: (((c.query or "").lower() in aq), _tok_hit(c), c.relevance),
              reverse=True)
    n = 0
    for c in pool[:limit]:
        subs = _fetch_subs_text(c.url)
        if _subs_contain_quote(subs, quotes):
            c.anchor_verified = True
            c.relevance = min(1.0, round(c.relevance + 0.5, 4))
            n += 1
            if progress:
                progress(f"discover: ✓ anchor VERIFIED by dialogue → {c.title[:56]!r}")
    return n


def resolve_quality(cands: list[SourceCandidate], cfg: ClipConfig, progress=None) -> None:
    """Second pass: resolve each top candidate's TRUE max resolution + exact duration. The legacy
    yt-dlp probe (any in-engine client) only sees YouTube's SABR-collapsed ~360p ladder, so a 1080p
    upload was indistinguishable from a 360p one — the HD-preference ranking was blind. We probe the
    real ladder via the SABR-capable setup (hd_download.probe_max_height) and fall back to the legacy
    extract only when that's unavailable. Env VIDLORE_CLIPSTUDIO_HD_PROBE=0 disables (faster)."""
    import os
    from . import hd_download as _hd
    use_hd = (os.environ.get("VIDLORE_CLIPSTUDIO_HD_PROBE", "1").strip() not in ("0", "false", "no", "")
              and _hd.available())
    import yt_dlp
    opts = {"quiet": True, "no_warnings": True, "skip_download": True,
            "noplaylist": True, "socket_timeout": 25,
            "extractor_args": {"youtube": {"player_client": ["web_safari", "web"]}}}
    todo = [c for c in cands if c.provider == "youtube"][:cfg.discover_resolve_limit]
    with yt_dlp.YoutubeDL(opts) as y:
        for c in todo:
            _pk = (c.url, int(getattr(cfg, "max_height", 0) or 0))   # cap changes the answer
            if use_hd and _pk in _PROBE_H_CACHE:
                h = _PROBE_H_CACHE[_pk]
            elif use_hd:
                h = _hd.probe_max_height(c.url, max_height=cfg.max_height)
                if h > 0:
                    _PROBE_H_CACHE[_pk] = h              # positive probes only — failures retry
            else:
                h = 0
            if h <= 0:                                    # HD probe off/failed → legacy estimate
                try:
                    info = y.extract_info(c.url, download=False)
                    heights = [f.get("height") or 0 for f in (info.get("formats") or [])]
                    h = int(max(heights or [info.get("height") or 0]) or 0)
                    c.duration = c.duration or float(info.get("duration") or 0)
                    c.view_count = c.view_count or int(info.get("view_count") or 0)
                except Exception:
                    pass
            if h:
                c.height = h
            if progress:
                progress(f"discover: resolved {c.height}p · {c.title[:40]!r}")


def discover_sources(analysis: ScriptAnalysis, cfg: ClipConfig, *, segments=None,
                     progress=None) -> list[SourceCandidate]:
    """Return a ranked, de-duplicated, diverse candidate list (~discover_target items).
    `segments` enables PER-BEAT targeted queries (the exact scene each line needs)."""
    def log(m):
        if progress:
            progress(m)

    queries = build_queries(analysis, segments)
    # the anchor-scene queries are the most important searches — go DEEPER on them (more results) so
    # the exact raw scene clip can't be missed under the noise of essays/compilations.
    anchor_qset = {q.lower() for q in anchor_queries(analysis, segments)}
    log(f"discover: {len(queries)} queries ({len(anchor_qset)} anchor) · type={getattr(analysis,'video_type','')}")
    from . import perf_metrics as _pm

    def _depths(q: str) -> tuple[int, int]:
        yt_n = cfg.discover_per_query * 2 if q.lower() in anchor_qset else cfg.discover_per_query
        return yt_n, max(2, cfg.discover_per_query // 2)

    def _one_query_ex(q: str) -> dict:
        """One query's per-provider results + status, in the serial loop's exact internal
        order (yt then archive). Statuses let the caller distinguish real results, a
        legitimate empty answer, throttling, a timeout, and a transport failure — the
        old code collapsed all five into a silent []."""
        yt_n, ar_n = _depths(q)
        with _pm.timed("discover.query"):
            _pm.incr("discover.query")
            yt_res, yt_st = _ytsearch_ex(q, yt_n)
            ar_res, ar_st = _archive_search_ex(q, ar_n)
        for _st in (yt_st, ar_st):
            if _st in _RETRYABLE:
                _pm.incr(f"discover.provider.{_st}")
        return {"yt": (yt_res, yt_st), "ar": (ar_res, ar_st)}

    def _retry_failed_providers(q: str, bucket: dict) -> dict:
        """SERIAL bounded-backoff retry of ONLY the failed/throttled providers of one query.
        A provider's successful partial results are preserved untouched; a legitimately
        EMPTY answer is never retried (it is an answer). Throttling backs off longer."""
        import time as _t
        yt_n, ar_n = _depths(q)
        for prov, fn, n in (("yt", _ytsearch_ex, yt_n), ("ar", _archive_search_ex, ar_n)):
            res, st = bucket[prov]
            attempt = 0
            while st in _RETRYABLE and attempt < 2:
                attempt += 1
                _pm.incr("discover.retry.serial")
                _t.sleep(min(8.0, (4.0 if st == STATUS_THROTTLED else 1.5) * attempt))
                res, st = fn(q, n)
            bucket[prov] = (res, st)
        return bucket

    def _bucket_list(bucket: dict) -> list:
        return list(bucket["yt"][0]) + list(bucket["ar"][0])

    # BOUNDED PARALLEL FAN-OUT, ORDER-PRESERVING AGGREGATION. The queries are independent
    # network waits (measured: 200 queries × ~3s strictly serial = ~13 min of the render).
    # Each query's results are collected in their OWN bucket and the buckets are concatenated
    # in the ORIGINAL query order, so the raw sequence — and therefore the first-occurrence
    # id/title dedupe below, every stable sort after it, the per-channel caps, and the chosen
    # sources — is identical to the serial loop over the same result sets. Failed/throttled
    # provider buckets are retried SERIALLY in query order (bounded backoff) after the pool
    # drains, keeping each query's successful partial results.
    # VIDLORE_CLIPSTUDIO_DISCOVER_WORKERS=1 restores the pure serial loop.
    import os as _os_dw
    try:
        _dw = int(_os_dw.environ.get("VIDLORE_CLIPSTUDIO_DISCOVER_WORKERS", "4") or 4)
    except (TypeError, ValueError):
        _dw = 4
    raw: list[SourceCandidate] = []
    _EMPTY_BUCKET = {"yt": ([], STATUS_TRANSPORT), "ar": ([], STATUS_TRANSPORT)}
    if _dw > 1 and len(queries) > 1:
        import concurrent.futures as _cf
        _buckets: dict[int, dict] = {}
        with _cf.ThreadPoolExecutor(max_workers=_dw) as _ex:
            _futs = {_ex.submit(_one_query_ex, q): i for i, q in enumerate(queries)}
            for _fu in _cf.as_completed(_futs):
                _qi = _futs[_fu]
                try:
                    _buckets[_qi] = _fu.result()
                except Exception:                        # noqa: BLE001 — retried serially below
                    _pm.incr("discover.query.error")
                    _buckets[_qi] = {k: (list(v[0]), v[1]) for k, v in _EMPTY_BUCKET.items()}
        for _qi, q in enumerate(queries):                # serial in-order retry of failures
            if any(_buckets[_qi][p][1] in _RETRYABLE for p in ("yt", "ar")):
                _buckets[_qi] = _retry_failed_providers(q, _buckets[_qi])
        for _qi, q in enumerate(queries):
            raw += _bucket_list(_buckets[_qi])
            log(f"discover: '{q[:40]}' → {len(raw)} cumulative")
    else:
        for q in queries:
            _b = _one_query_ex(q)
            if any(_b[p][1] in _RETRYABLE for p in ("yt", "ar")):
                _b = _retry_failed_providers(q, _b)
            raw += _bucket_list(_b)
            log(f"discover: '{q[:40]}' → {len(raw)} cumulative")

    # dedupe by id and by normalized title
    seen_id, seen_title, uniq = set(), set(), []
    for c in raw:
        key_id = (c.provider, c.id or c.url)
        key_t = _norm_title(c.title)
        if key_id in seen_id or (key_t and key_t in seen_title):
            continue
        seen_id.add(key_id)
        if key_t:
            seen_title.add(key_t)
        uniq.append(c)

    # score + first-pass reject (title/duration)
    import os as _os_ws
    anchor_sets = _anchor_token_sets(analysis)
    _wrongshow_on = _os_ws.environ.get("VIDLORE_CLIPSTUDIO_WRONGSHOW_GATE", "1").strip() \
        not in ("0", "false", "no")
    for c in uniq:
        c.relevance = _relevance(c, analysis, anchor_sets)
        c.reject_reason = _reject(c, cfg)
        if not c.reject_reason and _wrongshow_on \
                and _wrong_installment(analysis.movie_title, c.title):
            c.reject_reason = "wrong-show (franchise sibling/prequel)"
    kept = sorted([c for c in uniq if not c.reject_reason], key=lambda c: c.relevance, reverse=True)

    # CONTENT-LEVEL anchor verification — the title can echo the right words on the WRONG scene
    # (a "Rains of Castamere" Red Wedding clip is not the Bronn tavern scene); the spoken lines
    # can't lie. Verified candidates get a decisive boost; check is skippable via env.
    # Famous scenes often exist ONLY inside essay/breakdown videos — those are title-rejected,
    # but a dialogue-verified one carries the exact scene, so it earns its way back in.
    import os as _os4
    if (getattr(analysis, "anchor_scenes", None) and segments is not None
            and _os4.environ.get("VIDLORE_CLIPSTUDIO_SUB_VERIFY", "1").strip()
            not in ("0", "false", "no")):
        essay_pool = [c for c in uniq
                      if c.reject_reason == "reaction/review/annotated"
                      and not _MUSIC_ONLY_RX.search(c.title)
                      # a REACTION/facecam video plays the real scene audio (so it WOULD verify)
                      # but its footage is people on a couch — never re-admit it.
                      and not _REACTION_TITLE.search(c.title)]
        nv = verify_anchor_candidates(kept, analysis, segments, anchor_qset=anchor_qset,
                                      extra=essay_pool, progress=progress)
        log(f"discover: dialogue-verified {nv} anchor candidate(s)")
        if nv:
            for c in essay_pool:
                if c.anchor_verified:
                    c.reject_reason = ""
                    kept.append(c)
            kept.sort(key=lambda c: c.relevance, reverse=True)

    # resolve resolution on the top slice, then re-reject on low-res
    if cfg.discover_resolve_quality and kept:
        resolve_quality(kept, cfg, progress)
        for c in kept:
            if not c.reject_reason and not c.anchor_verified:
                # dialogue-verified candidates are exempt — they carry the exact scene, which
                # often exists ONLY inside an (otherwise title-rejected) essay/breakdown upload
                c.reject_reason = _reject(c, cfg)
        kept = [c for c in kept if not c.reject_reason]

        def _hd_bonus(h):
            # GRADUATED so 1080p outranks 720p outranks 480p (the old binary bonus gave 720p and
            # 1080p the SAME score, so a video ended up a mix — the user wants the crispest
            # available). Tiers anchor on cfg.discover_prefer_height (default 720 → 1080/720/480).
            ph = cfg.discover_prefer_height or 720
            if h >= max(1080, ph):          # raising the knob raises the TOP tier, never
                return 0.22                  # weakens the lower ones
            if h >= min(720, ph):
                return 0.12
            if h >= max(300, min(480, ph - 240)):
                return 0.05
            return 0.0
        # rank by relevance + a graduated HD bonus, then by raw height — prefer the crispest source
        kept.sort(key=lambda c: (round(c.relevance + _hd_bonus(c.height), 4), c.height), reverse=True)

    # per-channel diversity cap + target count
    final, per_ch = [], {}
    for c in kept:
        ch = c.channel or c.provider
        if per_ch.get(ch, 0) >= cfg.discover_max_per_channel:
            continue
        per_ch[ch] = per_ch.get(ch, 0) + 1
        final.append(c)
        if len(final) >= cfg.discover_target:
            break

    # ENTITY COVERAGE: a multi-character script can fill up with only the most-clipped pairing
    # (e.g. Hound+Arya) and starve other characters (Hound+Sansa). Guarantee each major character
    # with available footage gets at least one source, up to a small overflow.
    if segments is not None:
        names = []
        for ch_ in analysis.characters[:8]:
            nm = (ch_.get("name") or "").strip().split()
            if nm and len(nm[0]) > 2:
                names.append(nm[0])
        for s in segments:
            e = (getattr(s, "required_entity", "") or "").strip().split()
            if e and len(e[0]) > 2:
                names.append(e[0])
        # token (word/prefix) matching, not raw substring — "jon" must not count "jonquil"
        # as coverage, while "dragon" still matches "dragons"
        def _twords(title):
            return set(re.findall(r"\w+", (title or "").lower()))

        def _hit(tok, words):
            # prefix only for short suffixes (plural/possessive): "dragon"→"dragons" yes,
            # "jon"→"jonquil" no
            return any(w == tok or (w.startswith(tok) and len(w) - len(tok) <= 2)
                       for w in words)

        covered = set()
        for c in final:
            covered |= _twords(c.title)
        seen_names, added = set(), 0
        for nm in names:
            key = nm.lower()
            if key in seen_names or _hit(key, covered) or key in ("the", "game", "thrones"):
                continue
            seen_names.add(key)
            extra = next((c for c in kept if c not in final and _hit(key, _twords(c.title))), None)
            if extra is not None and added < cfg.discover_coverage_extra:
                final.append(extra)
                covered |= _twords(extra.title)
                added += 1
                log(f"discover: +coverage '{nm}' → {extra.title[:42]!r}")

    # ANCHOR COVERAGE: the scene(s) the whole video is ABOUT must be downloaded, even if the global
    # ranking + per-channel cap would have dropped them. For each anchor query, force-include the best
    # kept candidates whose title actually echoes that scene (raw footage, not the franchise name).
    # single_scene wants SEVERAL uploads of the one scene (coverage / quality / alternate angles).
    if anchor_qset:
        want = 4 if getattr(analysis, "video_type", "") == "single_scene" else 2
        present = {(c.provider, c.id or c.url) for c in final}
        for sc in (getattr(analysis, "anchor_scenes", None) or []):
            toks = [w for w in re.findall(r"\w+", (sc.get("query", "") + " " + sc.get("name", "")).lower())
                    if len(w) > 2 and w not in _STOPQ]
            mtoks = [w for w in re.findall(r"\w+", analysis.movie_title.lower()) if len(w) > 2]
            stoks = [w for w in toks if w not in mtoks]            # the scene-specific tokens
            if not stoks:
                continue
            # candidates that share ≥2 scene-specific tokens with this anchor, best relevance first
            # (token match on title words — prefix only for short plural/possessive suffixes).
            # Dialogue-VERIFIED candidates qualify regardless of title and outrank everything.
            matches = sorted(
                (c for c in kept
                 if c.anchor_verified
                 or sum(1 for w in set(stoks)
                        if any(t == w or (t.startswith(w) and len(t) - len(w) <= 2)
                               for t in re.findall(r"\w+", c.title.lower()))) >= 2),
                key=lambda c: (c.anchor_verified, c.relevance), reverse=True)
            added = 0
            for c in matches:
                kk = (c.provider, c.id or c.url)
                if kk in present:
                    added += 1
                    continue
                if added >= want:
                    break
                final.append(c)
                present.add(kk)
                added += 1
                log(f"discover: +anchor '{(sc.get('name') or sc.get('query'))[:34]}' → {c.title[:42]!r}")

    # KEY-SCENE COVERAGE: the script often CITES scenes beyond the anchor(s) — a later confession,
    # a trial, an origin flashback. Those live in analysis.key_scenes, and until now they had NO
    # download guarantee: global ranking concentrates on the dominant scene, so the cited-scene
    # candidates get queried, found… and dropped at the cut. Measured on the Olenna render:
    # 'Olenna confesses to Jaime' was queried 3× with candidates found, yet 0/45 downloaded
    # sources contained the confession, and the same starvation left Pycelle's trial testimony
    # uncovered — the two scenes the script explicitly retells. One source per cited scene, best
    # candidate first, same token matching as anchors, small bounded overflow.
    if anchor_qset:
        import os as _os_ks
        _ks_extra = int(_os_ks.environ.get("VIDLORE_CLIPSTUDIO_KEYSCENE_COVERAGE", "4") or 4)
        present = {(c.provider, c.id or c.url) for c in final}
        _mtoks = [w for w in re.findall(r"\w+", analysis.movie_title.lower()) if len(w) > 2]
        _anchor_blob = " ".join((sc.get("query", "") + " " + sc.get("name", "")).lower()
                                for sc in (getattr(analysis, "anchor_scenes", None) or []))
        added_ks = 0
        for ks in (getattr(analysis, "key_scenes", None) or []):
            if added_ks >= _ks_extra:
                break
            kst = [w for w in re.findall(r"\w+", str(ks).lower())
                   if len(w) > 2 and w not in _STOPQ and w not in _mtoks]
            if not kst:
                continue
            # already covered by an anchor scene? (the anchors got their own pass above)
            if sum(1 for w in set(kst) if w in _anchor_blob) >= max(2, len(set(kst)) - 1):
                continue
            covered = any(
                sum(1 for w in set(kst)
                    if any(t == w or (t.startswith(w) and len(t) - len(w) <= 2)
                           for t in re.findall(r"\w+", c.title.lower()))) >= 2
                for c in final)
            if covered:
                continue
            best = next((c for c in sorted(kept, key=lambda c: (c.anchor_verified, c.relevance),
                                           reverse=True)
                         if (c.provider, c.id or c.url) not in present
                         and sum(1 for w in set(kst)
                                 if any(t == w or (t.startswith(w) and len(t) - len(w) <= 2)
                                        for t in re.findall(r"\w+", c.title.lower()))) >= 2), None)
            if best is not None:
                final.append(best)
                present.add((best.provider, best.id or best.url))
                added_ks += 1
                log(f"discover: +key-scene '{str(ks)[:36]}' → {best.title[:42]!r}")

    log(f"discover: {len(uniq)} unique → {len(final)} selected "
        f"(rejected {sum(1 for c in uniq if c.reject_reason)})")
    return final
