"""Era / episode reasoning — one place, so every stage answers "which episode?" the same way.

Written after a render whose whole footage pool was decided by one unvalidated LLM string. The
analyzer labelled the anchor scene "S04E01 Two Swords"; it is S03E10 "Mhysa". Nothing checked, and
the label then behaved as ground truth in three separate places:

  * discovery injected it into queries ('Game of Thrones S04E01 small council', and the
    self-evidently broken 'Game of Thrones S04E01 red wedding' -- the Red Wedding is S03E09);
  * an episode code in a TITLE alone conferred ANCHOR status, skipping the ASR dialogue check, so
    'S04E01 - Arya Stark and the Hound' became an anchor of a small-council video;
  * the era filter PURGED every source whose title declared a different season -- 354 shots, every
    run -- including 'Game Of Thrones S03E10 Red Wedding Aftermath scene', the correct episode.

Sources with NO season in their title survived. The filter therefore punished honest labelling and
rewarded vague titles; the one clip holding both iconic lines survived only because its title omits
the season.

Two rules follow, and this module exists to make them structural rather than remembered:

  1. An UNVERIFIED hint may narrow nothing. It may not purge a source, confer anchor status, or
     constrain a beat. It is a hypothesis until corroborated (`verified_episode_hint`).
  2. Era is BEAT-LOCAL. A beat that names its own event is talking about that event, even inside a
     single-scene essay: the Red Wedding is S03E09 no matter what the anchor scene is. The global
     hint fills a gap; it never overrides local evidence.
"""
from __future__ import annotations

import re

# spelled-out numerals: the narration says "This is the last episode of season three." — a digits-
# only regex could not see the one disconfirming signal the pipeline already had in hand.
_WORD_NUM = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}
_NUM_ALT = "|".join(_WORD_NUM)

_SEASON_ONLY_RX = re.compile(
    r"\bseason\s+(?:(\d{1,2})|(" + _NUM_ALT + r"))\b", re.I)
_SXXEYY_RX = re.compile(r"\bs\s?0?(\d{1,2})\s?[ex]\s?0?(\d{1,2})\b", re.I)
_NXNN_RX = re.compile(r"\b(\d{1,2})\s?x\s?(\d{1,2})\b", re.I)
# bare "S3" / "GoT S4" — a real era leak on clip titles. Must not swallow S01E07 (handled above).
_BARE_S_RX = re.compile(r"\bs0*(\d{1,2})\b(?![\dex])", re.I)


def parse_episode(text: str):
    """-> (season, episode) or None. Only a FULL code counts as an episode."""
    t = text or ""
    m = _SXXEYY_RX.search(t) or _NXNN_RX.search(t)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


def parse_season(text: str):
    """-> season int or None. Understands S03E10, 3x01, 'season 3', 'season three', bare 'S3'."""
    t = text or ""
    ep = parse_episode(t)
    if ep:
        return ep[0]
    m = _SEASON_ONLY_RX.search(t)
    if m:
        return int(m.group(1)) if m.group(1) else _WORD_NUM[m.group(2).lower()]
    m = _BARE_S_RX.search(t)
    return int(m.group(1)) if m else None


def as_era(text: str) -> str:
    """Canonical era string for a beat constraint: 'season N', or '' when nothing is named."""
    n = parse_season(text)
    return f"season {n}" if n else ""


def episodes_conflict(a: str, b: str) -> bool:
    """True only when both name a FULL episode and they differ. S04E01 vs 'season 4' agree;
    S04E01 vs S04E10 conflict; anything unknown never conflicts."""
    ea, eb = parse_episode(a or ""), parse_episode(b or "")
    if ea and eb:
        return ea != eb
    sa, sb = parse_season(a or ""), parse_season(b or "")
    return bool(sa and sb and sa != sb)


def verified_episode_hint(analysis, script_text: str = "") -> tuple[str, bool, str]:
    """Corroborate the analyzer's episode_hint against evidence the system ALREADY holds.

    -> (hint, verified, reason). An unverified hint is still returned (it is a fine search
    keyword among others) but callers must not let it gate anything.

    Corroboration sources, all local — no extra LLM call, nothing to go wrong at 3am:
      * the narration script's own era claim ("the last episode of season three");
      * the anchor scene's verbatim dialogue vs the beats that quote it.
    The failing render had BOTH in hand: the same LLM dict that said "S04E01 Two Swords" carried
    the verbatim S03E10 lines, and the script said season three out loud."""
    hint = (getattr(analysis, "episode_hint", "") or "").strip()
    if not hint:
        return "", False, "no hint"
    hs = parse_season(hint)
    if hs is None:
        return hint, False, "unparseable hint"

    script_season = parse_season(script_text or "")
    if script_season is not None:
        if script_season == hs:
            return hint, True, f"script says season {script_season}"
        return hint, False, (f"CONTRADICTED — hint says season {hs}, the script says "
                             f"season {script_season}")

    # no independent evidence either way: a hint nobody can check must not gain hard power
    return hint, False, "uncorroborated (no era claim in the script)"


_TITLE_SEASON_RANGE_RX = re.compile(r"\bseasons?\s*0?(\d{1,2})\s*[-–—]\s*0?(\d{1,2})\b", re.I)


def title_seasons(title: str) -> set:
    """EVERY season a source title declares, range-aware. 'Seasons 1-8' → {1..8}; 'S04E02 + 7x03
    compilation' → {4, 7}. Empty set = the title declares nothing (never a conflict). parse_season
    returns only the FIRST hit, which mis-read span compilations as single-season sources."""
    t = title or ""
    out: set = set()
    for m in _TITLE_SEASON_RANGE_RX.finditer(t):
        a, b = int(m.group(1)), int(m.group(2))
        if 0 < a <= b <= 20:
            out |= set(range(a, b + 1))
    for m in _SXXEYY_RX.finditer(t):
        out.add(int(m.group(1)))
    for m in _NXNN_RX.finditer(t):
        out.add(int(m.group(1)))
    for m in _SEASON_ONLY_RX.finditer(t):
        out.add(int(m.group(1)) if m.group(1) else _WORD_NUM[m.group(2).lower()])
    return out


def title_era_conflicts(beat_era_str: str, title: str) -> bool:
    """Deterministic beat-vs-source-title era test: True ONLY when the beat declares a season AND
    the title declares seasons AND the beat's season is not among them. Range compilations that
    INCLUDE the beat's season never conflict; undeclared either side never conflicts."""
    s = parse_season(beat_era_str or "")
    if not s:
        return False
    declared = title_seasons(title)
    return bool(declared) and s not in declared


def anchor_token_eras(analysis) -> list:
    """[(scene-token set, era string)] for every anchor scene that DECLARES an episode. Tokens are
    the anchor's scene-specific words (movie-title + stop tokens removed) — the same token currency
    as discover's anchor/key-scene coverage and verify's venue mapping."""
    def _get(obj, key):
        if isinstance(obj, dict):
            return obj.get(key)
        return getattr(obj, key, None)
    out = []
    mv = {w for w in re.findall(r"[a-z']+", str(_get(analysis, "movie_title") or "").lower())
          if len(w) > 2}
    try:
        from .discover import _STOPQ as _STOP
    except Exception:
        _STOP = set()
    for sc in (_get(analysis, "anchor_scenes") or []):
        if not isinstance(sc, dict):
            continue
        era = as_era(sc.get("episode") or "") or as_era(sc.get("query") or "")
        if not era:
            continue
        toks = {w for w in re.findall(
                    r"[a-z']+", ((sc.get("name", "") or "") + " " + (sc.get("query", "") or "")).lower())
                if len(w) > 2 and w not in mv and w not in _STOP}
        if toks:
            out.append((toks, era))
    return out


def beat_era(seg, global_era: str, *, single_scene: bool, global_verified: bool,
             event_eras: dict | None = None, anchor_eras: list | None = None) -> str:
    """The era constraint for ONE beat.

    Order matters, and it is the inverse of what the pipeline used to do (which returned the global
    hint immediately for single-scene videos and never read the beat at all):

      1. the beat names an era itself            -> that era wins, always
      2. the beat names a known non-anchor event -> that event's era
      3. core-scene / deictic beat + VERIFIED global hint -> the global era
      4. otherwise                               -> unconstrained

    (4) is deliberate. An unconstrained beat is not an unchecked beat — it still has to win on
    dialogue / Face-ID / CLIP. Guessing an era is strictly worse than declining to: a wrong guess
    deletes the right footage, which is precisely how 'S03E10 Red Wedding Aftermath' scored zero
    shots on a video about S03E10."""
    for attr in ("scene_query", "expected_visual", "text"):
        e = as_era(getattr(seg, attr, "") or "")
        if e:
            return e
    if event_eras:
        blob = " ".join((getattr(seg, a, "") or "") for a in ("text", "expected_visual",
                                                              "scene_query")).lower()
        for name, era in event_eras.items():
            if name and era and name.lower() in blob:
                return era
    # ANCHOR-SCENE INHERITANCE: the beat's scene_query names the SCENE (not the episode) — 'the
    # Purple Wedding feast crowd' declares no era itself, so its beats ran unconstrained and a
    # S7/S8 Winterfell great-hall shot aired under S4 wedding narration (the vision verifier
    # cannot read a season off a torch-lit hall). When the scene_query shares >=2 scene-specific
    # tokens with an anchor scene that DECLARES an episode, the beat inherits that anchor's era.
    # Same token currency as coverage/venue mapping; >=2 avoids one-word collisions ('poison').
    # This never guesses: no declared anchor era, or no 2-token mapping → next rung.
    if anchor_eras:
        sq = {w for w in re.findall(r"[a-z']+", (getattr(seg, "scene_query", "") or "").lower())
              if len(w) > 2}
        if sq:
            best_era, best_hits = "", 0
            for toks, era in anchor_eras:
                hits = sum(1 for w in toks
                           if any(t == w or (t.startswith(w) and len(t) - len(w) <= 2)
                                  for t in sq))
                if hits > best_hits:
                    best_hits, best_era = hits, era
            if best_hits >= 2:
                return best_era
    if single_scene and global_verified and global_era:
        # verbatim, NOT normalised to 'season N': the hint may carry a full episode code and
        # callers compare at episode granularity (S04E01 vs S04E10 must still conflict).
        return global_era
    return ""


def event_eras_from(analysis) -> dict:
    """{event name -> 'season N'} for events the analyzer named an episode for. Only used to route
    a beat AWAY from the anchor era, never to invent one: an event with no stated episode yields no
    entry and its beats stay unconstrained."""
    out: dict = {}
    for sc in (getattr(analysis, "anchor_scenes", None) or []):
        if not isinstance(sc, dict):
            continue
        name, era = (sc.get("name") or "").strip(), as_era(sc.get("episode") or "")
        if name and era:
            out[name] = era
    return out
