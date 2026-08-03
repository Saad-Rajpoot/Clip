"""Stage 3 — segment the narration script into clip-sized beats.

Each segment carries the text, a short `expected_visual` (the CLIP query), keywords, named
entities, an `is_specific_claim` flag (lines asserting a precise visual fact → forced review),
and an estimated screen time. A heuristic segmenter runs offline and deterministically; an
optional Claude pass enriches `expected_visual`/entities/keywords when an API key is present
(reusing the engine's anthropic config).
"""
from __future__ import annotations

import re
from .models import ScriptSegment
from .config import ClipConfig

_WORDS_PER_SEC = 2.6                      # ~156 wpm narration
_STOP = set("""a an the of to in on at for and or but so then this that these those is are was
were be been being with from into as it its it's they them their he she his her you your we our
us i me my a about over under after before while when where which who whom whose what why how
than too very just only also not no yes if up out down off again more most some any all each
every both either neither one two three new like got had has have do does did will would can
could should may might must let's there here""".split())

_CLAIM_RX = re.compile(
    r"\b(the scene where|the moment (?:when|that)|this is the|the exact|in this (?:clip|shot|scene)|"
    r"here (?:we see|is|'s)|the part where|you can see|pictured|shown here|the very|caught on)\b",
    re.I,
)
_YEAR_RX = re.compile(r"\b(1[89]\d{2}|20\d{2})\b")
_ENTITY_RX = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
# Split only on commas + coordinating conjunctions — NOT subordinating ones (who/which/because/
# while), which would fragment a sentence into ungrammatical narration for the TTS.  The delimiter
# is CAPTURED because it is authored narration, not disposable parsing syntax.  The old split
# consumed commas and words such as "and", permanently changing both TTS and captions (for example
# "in secret, hidden her face, and told" became "in secret hidden her face told").
_CLAUSE_SPLIT = re.compile(r"(,\s+|\s+(?:and|but|so|then)\s+)", re.I)


def _word_count(s: str) -> int:
    return len([w for w in re.findall(r"\w+", s)])


def _split_long(sentence: str, lo: int, hi: int) -> list[str]:
    """Break a long sentence into ~lo..hi-word chunks at clause boundaries, else word windows."""
    if _word_count(sentence) <= hi:
        return [sentence.strip()]
    # Convert the captured-token stream into independently chunkable clauses while preserving the
    # exact punctuation/words.  A comma stays on the clause before it; a coordinating conjunction
    # stays at the start of the clause after it.  Joining the resulting pieces with one space is
    # therefore text-preserving after the module's documented whitespace normalization.
    split = _CLAUSE_SPLIT.split(sentence)
    parts: list[str] = []
    current = split[0].strip() if split else ""
    for i in range(1, len(split), 2):
        sep = split[i]
        following = split[i + 1].strip() if i + 1 < len(split) else ""
        if sep.lstrip().startswith(","):
            if current:
                parts.append(current.rstrip() + ",")
            current = following
        else:
            if current:
                parts.append(current)
            conjunction = sep.strip()
            current = f"{conjunction} {following}".strip()
    if current:
        parts.append(current)
    chunks, cur = [], ""
    for p in parts:
        cand = (cur + " " + p).strip() if cur else p
        if _word_count(cand) > hi and cur:
            chunks.append(cur)
            cur = p
        else:
            cur = cand
    if cur:
        chunks.append(cur)
    # any still-too-long chunk → hard word window
    out = []
    for c in chunks:
        ws = c.split()
        if len(ws) > hi:
            for i in range(0, len(ws), hi):
                out.append(" ".join(ws[i:i + hi]))
        else:
            out.append(c)
    return out


def _keywords(text: str) -> list[str]:
    toks = re.findall(r"[A-Za-z][A-Za-z'-]+", text)
    kws = []
    for t in toks:
        tl = t.lower()
        if tl in _STOP or len(tl) < 3:
            continue
        kws.append(tl)
    # dedupe preserve order
    seen, out = set(), []
    for k in kws:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out[:8]


def _entities(text: str) -> list[str]:
    ents = []
    for m in _ENTITY_RX.finditer(text):
        e = m.group(1).strip()
        # drop sentence-initial single capitalized words that are just the first token
        if " " in e or text.find(e) > 0:
            ents.append(e)
    seen, out = set(), []
    for e in ents:
        if e not in seen and e.lower() not in _STOP:
            seen.add(e)
            out.append(e)
    return out[:5]


def segment_script(script_text: str, cfg: ClipConfig) -> list[ScriptSegment]:
    """Heuristic segmentation into clip-sized beats."""
    # Keep natural units WHOLE up to max_scene_sec so scene length tracks the writing's own rhythm
    # (short line → quick cut, long dramatic sentence → long hold). Only very long sentences split.
    hi = max(8, int(cfg.max_scene_sec * _WORDS_PER_SEC))
    lo = max(3, cfg.min_scene_words)

    raw = re.sub(r"\s+", " ", script_text.strip())
    sentences = [s.strip() for s in _SENT_SPLIT.split(raw) if s.strip()]
    pieces: list[str] = []
    for sent in sentences:
        pieces.extend(_split_long(sent, lo, hi))

    # merge sub-minimum fragments into a neighbor so no scene is jarringly short
    merged: list[str] = []
    for p in pieces:
        if merged and _word_count(p) < lo:
            merged[-1] = (merged[-1] + " " + p).strip()
        else:
            merged.append(p)

    segs: list[ScriptSegment] = []
    for i, text in enumerate(merged):
        wc = _word_count(text)
        seg = ScriptSegment(
            index=i,
            text=text,
            expected_visual=text,                      # default CLIP query = the line itself
            keywords=_keywords(text),
            entities=_entities(text),
            is_specific_claim=bool(_CLAIM_RX.search(text) or _YEAR_RX.search(text)),
            est_duration=round(min(cfg.max_scene_sec, max(cfg.min_clip_sec, wc / _WORDS_PER_SEC)), 2),
        )
        segs.append(seg)
    return segs


# ---------------------------------------------------------------------------
# Optional Claude enrichment (visual director hint per segment)
# ---------------------------------------------------------------------------

_ENRICH_SYS = (
    "You are a film-clip editor. For each narration line, return what should be ON SCREEN as a "
    "short concrete visual phrase (people, objects, setting — not abstract), the named entities "
    "(people/films/characters) it references, and whether it asserts a precise visual fact that a "
    "viewer could verify in the footage (is_specific_claim). Reply ONLY with JSON."
)


def enrich_with_llm(segments: list[ScriptSegment], engine_cfg, *, batch: int = 25,
                    progress=None) -> None:
    """Fill expected_visual/entities/keywords/is_specific_claim via the LLM brain (Gemini→Claude),
    in place. No-op if no brain available. Falls back silently to heuristic values on any error."""
    import json as _json
    from . import llm as _llm
    if not _llm.has_llm(engine_cfg):
        return
    for start in range(0, len(segments), batch):
        chunk = segments[start:start + batch]
        lines = [{"i": s.index, "line": s.text} for s in chunk]
        prompt = (
            "Return a JSON array; one object per input with keys "
            '{"i":int,"visual":str,"entities":[str],"keywords":[str],"specific":bool}.\n'
            + _json.dumps(lines)
        )
        try:
            txt = _llm.complete(system=_ENRICH_SYS, max_tokens=2000,
                                messages=[{"role": "user", "content": prompt}], eng_cfg=engine_cfg)
            m = re.search(r"\[.*\]", txt, re.S)
            data = _json.loads(m.group(0)) if m else []

            def _bi(o):
                try:
                    return int(o.get("i", -1))
                except (TypeError, ValueError):    # '"i": null'/"three" must not drop the batch
                    return -1
            by_i = {_bi(o): o for o in data if isinstance(o, dict)}
            for s in chunk:
                o = by_i.get(s.index)
                if not o:
                    continue
                if o.get("visual"):
                    s.expected_visual = str(o["visual"])[:160]
                if o.get("entities"):
                    s.entities = [str(x) for x in o["entities"]][:6]
                if o.get("keywords"):
                    s.keywords = [str(x).lower() for x in o["keywords"]][:8]
                if "specific" in o:
                    s.is_specific_claim = bool(o["specific"]) or s.is_specific_claim
            if progress:
                progress(f"segment: enriched {start+len(chunk)}/{len(segments)}")
        except Exception:
            continue                                   # keep heuristic values for this batch
