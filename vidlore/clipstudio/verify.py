"""Phase 6 — mandatory AI visual verification.

A second pass that shows each selected clip's representative frame to Claude (vision) alongside
the narration line + the entity that line demands + the automatic Face-ID result, and asks:
does this clip actually match? is the correct actor/character visible? is it specific enough?
is the quality acceptable? On a "replace" verdict the clip is swapped for the next-best
alternate and re-verified — so weak/wrong/blurry picks are repaired automatically.

Uses the engine's Claude key. If no key, this pass is skipped (the pipeline still produces a
video; the QC report notes verification was unavailable). The verifier never claims certainty —
its verdict is recorded per clip for the QC report and to drive replacement.
"""
from __future__ import annotations

import base64
import copy
import json
import re
from collections import Counter
from pathlib import Path

from .models import (ClipProject, ScriptSegment, ClipSelection, FLAG_EXACT_MISSING,
                     FLAG_VERIFIER_UNVERIFIED)
from .config import ClipConfig
from . import index as _index
from . import cut as _cut
from . import policy as _policy
from . import era as _era

REUSE_CAP_OVERFLOW_EXACT_CONTRACT = "reuse_cap_overflow_exact_contract"

_VSYS = (
    "You are a strict film-footage QC editor. You judge whether ONE clip's representative frame "
    "correctly illustrates a narration line. Be skeptical: if the specific person/character/object "
    "the line is about is not clearly visible, or the frame is blurry, a title card, a watermark, or "
    "only loosely related, you must fail it. Reply with ONLY a JSON object."
)


_ABSENCE_PLACE_RX = (
    r"room|chamber|hall|castle|tent|garden|ship|boat|bay|harbour|"
    r"harbor|city|street|road|court|tower|island|"
    r"forest|camp|gate|wall|keep|palace|bedroom|brothel|cell|dungeon"
)


def _absence_elsewhere_context(narration: str, required_entity: str, *,
                               expected_visual: str = "", scene_query: str = "") -> bool:
    """Whether a locative absence at place A is storyboarded at a distinct place B.

    This deliberately requires both named-subject grammar and two concrete, different location
    nouns.  Merely supplying a generic character close-up cannot turn ``X is absent`` into a pass.
    """
    text = " ".join(str(narration or "").replace("\u2019", "'").split())
    entity = str(required_entity or "").strip()
    storyboard = " ".join(x for x in (
        str(expected_visual or ""), str(scene_query or "")) if x)
    if not text or not entity or not storyboard:
        return False
    aliases = [tok for tok in re.findall(r"[A-Za-z][A-Za-z'-]{2,}", entity)
               if tok.lower() not in {"ser", "lord", "lady", "king", "queen", "prince"}]
    if not aliases:
        return False
    excluded_places: set[str] = set()
    absence_rx = re.compile(
        rf"\s*\b(?:is|was|are|were)\s+not(?:\s+even)?\s+"
        rf"(?:in|at|inside)\s+(?:the\s+)?(?P<place>{_ABSENCE_PLACE_RX})\b|"
        rf"\s*\b(?:isn't|wasn't|aren't|weren't)\s+"
        rf"(?:in|at|inside)\s+(?:the\s+)?(?P<place_short>{_ABSENCE_PLACE_RX})\b",
        re.I)
    for alias in aliases:
        for named in re.finditer(rf"\b{re.escape(alias)}\b", text, re.I):
            # The absence predicate must belong grammatically to THIS subject. Searching anywhere
            # later in the sentence misread ``Baelish accused Varys, who was not in the room`` as
            # an assertion about Baelish. An anchored match still handles either full-name token:
            # ``Petyr`` is skipped, while the adjacent ``Baelish is not...`` token succeeds.
            match = absence_rx.match(text, named.end(), min(len(text), named.end() + 96))
            if match is not None:
                excluded_places.add(str(match.group("place") or match.group("place_short") or "")
                                    .lower())
    if not excluded_places:
        return False
    storyboard_places = {
        match.group(0).lower()
        for match in re.finditer(rf"\b(?:{_ABSENCE_PLACE_RX})\b", storyboard, re.I)
    }
    equivalent_place = {
        "harbour": "harbor",
        "boat": "ship",
        "keep": "castle", "palace": "castle",
        "chamber": "room",
        "road": "street",
    }
    excluded_places = {equivalent_place.get(place, place) for place in excluded_places}
    storyboard_places = {equivalent_place.get(place, place) for place in storyboard_places}
    # Mentioning the excluded room anywhere in the storyboard is ambiguous, even when another
    # place also appears (``the council room aboard a ship``). Require a wholly distinct concrete
    # location; uncertainty retains the ordinary contradiction rule.
    return bool(storyboard_places) and storyboard_places.isdisjoint(excluded_places)


def _absence_elsewhere_instruction(narration: str, required_entity: str, *,
                                   expected_visual: str = "", scene_query: str = "") -> str:
    """Clarify an authored *absence at place A* visualized by the subject at place B.

    The verifier's ordinary contradiction example deliberately rejects ``X is absent`` over a
    frame showing X.  That is right unless the storyboard explicitly asks to show X somewhere
    else: e.g. ``Baelish is not even in the room`` over Baelish aboard the escape ship.  In that
    case his presence at the different location is the visual proof, not a contradiction.

    Keep this narrow and deterministic.  It activates only when a token from the required entity
    is named in the narration, followed by an explicit locative absence, and a concrete storyboard
    was authored for the alternate location. A generic negation (``X is not the killer``) does not
    match and retains the normal strict rule.
    """
    if not _absence_elsewhere_context(
            narration, required_entity, expected_visual=expected_visual,
            scene_query=scene_query):
        return ""
    return (
        "ABSENCE-ELSEWHERE CONTEXT — the narration says the required subject is absent from a "
        "particular place, while the storyboard intentionally shows that subject somewhere else. "
        "Seeing the subject at the storyboard's different location SUPPORTS the narration; do not "
        "mark a contradiction merely because the subject is visible. Still mark 'replace' if the "
        "frame shows the subject in the excluded place, or if it is a different scene/location "
        "than the storyboard.\n"
    )


def _img_block(path: Path) -> dict:
    data = base64.standard_b64encode(Path(path).read_bytes()).decode()
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}}


def verify_frame(keyframe_path, narration: str, required_entity: str, required_kind: str,
                 faceid_names: list[str], eng_cfg, model: str = "", is_specific: bool = True,
                 *, expected_visual: str = "", scene_query: str = "", era_hint: str = "",
                 multiframe: bool = False, venue_fallback: bool = False,
                 must_see: str = "", exact_cast_warning: str = "") -> dict | None:
    """One vision verdict for a frame (Gemini brain → Claude fallback). None on error.

    `is_specific` carries the beat's is_specific_claim: a SPECIFIC line ("Tyrion shoots Tywin with a
    crossbow") demands the EXACT scene; a GENERIC line ("and everything changed") only needs a
    thematically-relevant filler — so the verifier is told to grade leniently there.

    For a specific/venue question, `expected_visual`/`scene_query` give the verifier the beat's
    STORYBOARD so a right-character / wrong-moment frame fails. They are deliberately excluded from
    generic and character-general questions, where an aspirational storyboard is not narration.
    `era_hint` remains active at every policy level to reject clearly wrong-era footage.

    `keyframe_path` may be a single frame or a pre-built start→mid→end contact sheet (set
    `multiframe=True`) so an ACTION beat is judged on whether the action actually occurs, not on one
    ambiguous instant."""
    if not keyframe_path or not Path(keyframe_path).exists():
        return None
    from . import llm as _llm
    _rule = (
        "This line refers to a SPECIFIC scene/moment — the footage must show THAT exact scene/"
        "subject. Be STRICT: the correct character ALONE is not enough — if the frame shows the right "
        "person but a DIFFERENT scene, moment, action, or era than the one described, mark 'replace'.\n"
        if is_specific else
        "This is a GENERIC or CHARACTER-GENERAL narration line (no exact-scene claim) — a "
        "thematically RELEVANT clip with the required subject is acceptable. Judge only the "
        "narration's actual claim and required subject; do not demand an invented pose, room, action, "
        "or camera angle. Set matches_narration=true and specific_enough=true when the footage is a "
        "clean, relevant illustration at this policy level. Mark 'replace' ONLY if it is off-topic, "
        "jarring, contradicts the line, or shows the WRONG character/era — NOT merely because it "
        "isn't a specific/exact scene.\n")
    # INSTRUCTED LOOKING — the narration tells the viewer to look at a NAMED thing ("keep your eye
    # on the dagger", "watch Bran's face"). These are the beats a viewer notices breaking, and on
    # the v4 render the named thing was absent on 12 of them. When the line points at something,
    # "the right people are on screen" is not an answer — the thing itself has to be there.
    _look = ""
    if must_see:
        _look = (f"THE NARRATION TELLS THE VIEWER TO LOOK AT: {must_see}\n"
                 f"Report target_visible=true ONLY if {must_see} is actually visible and "
                 f"identifiable in the frame. If it is not, set target_visible=false — a clip with "
                 f"the right characters or the right scene but WITHOUT {must_see} does not satisfy "
                 f"this line. Judge the object/face itself, not the surrounding context.\n")
    _story = ""
    # Storyboards/search queries are strict-scene evidence only. The analyzer may provide an
    # aspirational shot for a character-general line ("Baelish smirking in his brothel"); injecting
    # that into a lenient question made correct Baelish footage fail `specific_enough`. Venue stills
    # explicitly ask a scene-context question and therefore retain the storyboard.
    if expected_visual and (is_specific or venue_fallback):
        _story += f"The exact moment should LOOK LIKE: {expected_visual}\n"
    if scene_query and (is_specific or venue_fallback):
        _story += f"Target scene: {scene_query}\n"
    _absence_elsewhere = (
        _absence_elsewhere_instruction(
            narration, required_entity, expected_visual=expected_visual,
            scene_query=scene_query)
        if is_specific or venue_fallback else ""
    )
    _contradiction_instruction = (
        "For contradicts_narration: this is the narrow ABSENCE-ELSEWHERE exception described "
        "above. Set false when the named subject is visibly at the storyboard's distinct location, "
        "because that supports their absence from the excluded place. Set true if the subject is "
        "in the excluded place, at a different location than the storyboard, or the frame otherwise "
        "directly negates the line. The exact storyboard scene/location is still required.\n"
        if _absence_elsewhere else
        "For contradicts_narration: set true when what is visibly shown directly negates the line "
        "(for example, the line says a named person is absent but that person is visibly present, "
        "or it names one person's death while the clip shows another person's death). This is "
        "stronger than merely being an inexact or contextual shot.\n")
    if era_hint:
        _story += (f"Era/season context: {era_hint} — footage from a clearly different era/season "
                   f"than the moment described is WRONG even if the character matches.\n"
                   f"Set era_ok=false when the frame is clearly from another season/era than "
                   f"{era_hint} — a visibly younger or older cast, or a location that belongs to a "
                   f"different point in the story. Set era_ok=true when it is consistent or you "
                   f"cannot tell.\n")
    _mf = ("The image is a START -> MIDDLE -> END contact sheet (three moments of the clip, left to "
           "right). Judge whether the described ACTION actually happens across them — a single frame "
           "cannot prove an action, so require visible progression consistent with the line.\n"
           if multiframe else "")
    _cast_warning = ""
    if exact_cast_warning and is_specific:
        _cast_warning = (
            "SOURCE-TITLE CAST WARNING — the upload title conflicts with co-character(s) named "
            "by the exact storyboard: " + str(exact_cast_warning).strip() + "\n"
            "A title describes the whole upload and is NOT proof that this selected shot is wrong. "
            "Resolve the warning from the ACTUAL PIXELS only. Set source_title_conflict_resolved="
            "true only when this frame/contact sheet itself gives affirmative visual evidence for "
            "the storyboard's exact scene and expected co-character(s), not merely the required "
            "main character. If the expected scene/cast cannot be established from the pixels, set "
            "it false and mark replace.\n")
    # MICRO-OBJECT beats: 'the poison stone' / 'a vial' / 'a coin' cannot be resolved in a frame,
    # so demanding the object itself made the verifier reject the beat's OWN scene at every layer
    # (measured: frames of the exact gem-plucking moment — right characters, right table, necklace
    # in frame — rejected as "no poison is visible", and the beat release-blocked with the correct
    # footage downloaded). Ask the answerable question instead: is this the described MOMENT — the
    # characters who handle the object, the setting, the staging. Wrong scene/characters/era still
    # fail; this narrows the question, it does not lower the bar.
    _obj = (
        "NOTE — the required subject is an OBJECT/PROP that may be too small to resolve in a frame. "
        "Do NOT demand the object itself be identifiable. Verify the moment's CONTEXT instead: the "
        "frame must show the described scene — the right characters, setting and staging for THIS "
        "moment (see the storyboard above). The storyboard's FRAMING is aspirational: never mark "
        "'replace' because the frame is a wide/medium shot instead of the described close-up — "
        "judge WHO, WHERE and WHEN, not the camera distance. Set correct_subject_visible=true when "
        "the character(s) who hold/handle the object are clearly present in the right scene, even "
        "if the object is not resolvable. A wrong scene, wrong characters, or wrong era is still "
        "'replace'.\n"
        if "object" in (required_kind or "").lower() else "")
    # VENUE-FALLBACK stills: this question is only asked AFTER every strict layer (match pick,
    # alternates, contextual downgrade, venue promotion, rediscovery) refused — the exact footage
    # is established as unobtainable, and the still layer's whole design is "a right-scene frame
    # beats a dead render, subject to no contradiction". The measured failure: for a beat citing a
    # micro-action inside a known scene (a witness examining evidence at the trial), every
    # right-VENUE frame was rejected because the micro-action wasn't visible in it — which it never
    # can be. Scene/era/character contradictions still fail; the still is installed honestly
    # labeled contextual_fallback.
    _venue = (
        "FALLBACK CONTEXT — the exact footage of this moment is unavailable; this frame is a "
        "candidate HOLDING IMAGE. Mark 'keep' when the frame shows the right SCENE/VENUE for the "
        "moment: the correct location and era, featuring the moment's characters or setting (see "
        "the storyboard). Do NOT demand the described action/subject itself be visible. Mark "
        "'replace' only for a DIFFERENT scene/location, a different era/season, unrelated or wrong "
        "characters, or footage that would contradict the narration.\n"
        if venue_fallback else "")
    # NON-SHOW HARD RULE — applies to EVERY rung (strict, generic, object, venue-fallback). The
    # verifier once rationalized a sports-news CGI intro as "suitable for a general transition"
    # on an abstract-effect beat, and painterly fan art as "shows the correct character": a
    # designed image can be thematically perfect and still must never air as footage.
    _nonshow = (
        "HARD RULE — the frame must be REAL footage from the actual show itself, with the show's "
        "real actors. Mark 'replace' no matter how thematically fitting it is if it is any of:\n"
        "• a designed / non-show image — drawing, painting, fan art, comic/anime frame, poster, "
        "video-game graphics or UI, news/broadcast motion graphics, a logo/title card;\n"
        "• an AI-GENERATED or deepfake image OR VIDEO — tells: waxy/over-smooth or plastic skin, "
        "faces that morph or don't match the show's real actors, garbled hands/armor/props, an "
        "unnaturally 'cinematic' sepia AI grade, impossible costumes/weapons;\n"
        "• footage from a DIFFERENT PRODUCTION — a fan film, a fan-made recreation, another "
        "movie/series, or a re-enactment — i.e. the location, costumes, or actors are clearly NOT "
        "the real show's (amateur or mismatched production values, unfamiliar faces in the role);\n"
        "• BEHIND-THE-SCENES / production footage — the film crew, a camera, dolly, boom, lighting "
        "rig, monitors, marks or equipment cases in frame; crew in modern clothing (jackets, "
        "baseball caps, trainers, hi-vis) on or beside the set; a rehearsal, stunt practice, a "
        "blooper, or an actor out of character. This is REAL footage OF the production, which is "
        "why it slips the tests above — but the essay is about the STORY, so it must never air.\n"
        "This is about AUTHENTICITY, not resolution: genuine show footage that is merely low-res, "
        "dark, blurry, or heavily colour-graded is FINE. Real props, maps and documents FILMED "
        "WITHIN a live-action scene are fine. When the frame looks AI-generated OR like a different "
        "production, set correct_subject_visible=false and verdict='replace'.\n")
    txt = (
        f'Narration line: "{narration}"\n'
        f"This clip should show: {required_entity or '(a general scene fitting the line)'} "
        f"(kind: {required_kind or 'any'}).\n"
        + _story + _absence_elsewhere + _look + _mf + _cast_warning + _rule + _nonshow + _obj + _venue +
        f"Automatic Face-ID on this frame detected: {', '.join(faceid_names) if faceid_names else 'none'}.\n\n"
        "For wrong_subject_visible: set true ONLY if a DIFFERENT specific character (clearly NOT the "
        "one this line is about) is the main subject of the frame; set false for a wide / crowd / "
        "reaction / establishing shot where the required person may be present off-centre or unclear.\n"
        + _contradiction_instruction +
        "Answer ONLY this JSON:\n"
        '{"matches_narration": true/false, "correct_subject_visible": true/false, '
        '"wrong_subject_visible": true/false, "contradicts_narration": true/false, '
        + ('"target_visible": true/false, ' if must_see else "")
        + ('"source_title_conflict_resolved": true/false, '
           if exact_cast_warning and is_specific else "")
        # era_ok is asked whenever the beat declares an era. It gates the CONTEXTUAL FALLBACK, which
        # used to re-admit a clip on "the subject is visible" alone and shipped season-1 child Bran
        # under season-8 lines.
        + ('"era_ok": true/false, ' if era_hint else "")
        + '"specific_enough": true/false, "quality_ok": true/false, '
        '"confidence": 0.0-1.0, "verdict": "keep" or "replace", "reason": "one short sentence"}'
    )
    import time
    from . import perf_metrics as _pm
    content = [_img_block(Path(keyframe_path)), {"type": "text", "text": txt}]
    _pm.incr("verify.vision_call")
    if venue_fallback:
        _pm.incr("verify.vision_call.venue")
    for attempt in range(1, 5):                       # retry transient overload / rate limits
        try:
            with _pm.timed("verify.vision_call"):
                out, _meta = _llm.complete_ex(system=_VSYS, max_tokens=400,
                                              messages=[{"role": "user", "content": content}],
                                              eng_cfg=eng_cfg, model=model)
            m = re.search(r"\{.*\}", out, re.S)
            if not m:
                return None
            v = json.loads(m.group(0))
            if isinstance(v, dict):
                # provenance: the provider that ACTUALLY served this judgment (vision_config's
                # canonical format). Cache writers key the verdict by this, never by a prediction.
                v["vision_served_by"] = str(_meta.get("served") or "")
            return v
        except Exception:                             # transient overload / rate limit → back off
            if attempt == 4:
                return None
            time.sleep(min(1.5 * (2 ** attempt), 16))
    return None


_SEASON_RX = re.compile(
    r"\bS0?(\d{1,2})\s?E0?\d{1,2}\b|\bseason\s+(\d{1,2})\b|\b(\d{1,2})\s?x\s?\d{2}\b", re.I)

# Bump whenever the verifier PROMPT or its JSON contract changes: a verdict is only reusable if it
# was produced by the same question. Part of the fingerprint below.
# BUMPED with the behind-the-scenes clause. The verdict cache keys on this string, so a prompt
# change that did NOT bump it would serve answers to a different question — the whole point of the
# fingerprint. The cost is one cold verify pass on the next render (~$1); the alternative is a
# silently stale cache, which is worse.
PROMPT_VERSION = "v9-2026-08"           # strict/exact prompt remains byte-identical to v9
LENIENT_PROMPT_VERSION = "v10-2026-08"  # policy-typed generic/character/venue question
# Bump when the contact-sheet SAMPLING changes (frame count/positions/layout). The sheet is the
# image the verifier judges, so a different sampling is a different question even for the same shot.
SHEET_VERSION = "sheet-v3-selected-window-15-50-85-max4x1"
# A persisted positive verdict is not publication evidence until it is bound to the selection that
# will air.  The verdict-cache fingerprint identifies the vision QUESTION; this schema adds the
# selected source/shot/trim tuple so a later project.json mutation cannot carry a stale `keep` onto
# different footage.  Bump this whenever the persisted binding shape changes.
SELECTION_EVIDENCE_SCHEMA = 1
# Consecutive transient failures after which the vision backend is declared DOWN. Measured: over an
# 11-hour run the verifier degraded 176 replaced -> 180 -> 55 -> 0, and at exactly 0 the release
# gate passed and published. Nothing noticed, because "0 rejections" and "nothing checked" were the
# same number. The breaker exists so the pipeline can tell those two apart.
VERIFIER_BREAKER_TRIP = 8


def effective_deictic_target(seg) -> str:
    """Current look target under the verifier's supported LOOK_GATE kill switch."""
    import os as _os_look
    if _os_look.environ.get("VIDLORE_CLIPSTUDIO_LOOK_GATE", "1").strip().lower() \
            in ("0", "false", "no"):
        return ""
    try:
        return _policy.deictic_target(seg)
    except Exception:                                   # noqa: BLE001 — disabled/unknown is empty
        return ""


class NonRetryableBuildError(RuntimeError):
    """A CONTENT verdict: the render is wrong and re-running it unchanged cannot help.

    Release-blocks and relevance failures are of this kind. They were being raised as bare
    RuntimeErrors, so an outer driver happily restarted the whole pipeline — 8 times in the render
    that prompted this. Each attempt re-ran the verifier, and the last attempt "passed" only
    because the vision API had finally died: 0 verdicts, 0 rejections, 0 unresolved, publish.
    Scene 25 was never fixed. It just stopped being checked.

    Retry transient plumbing. Never retry a judgment.

    `kind` is the machine-readable identity of the gate that raised — routing (e.g. the in-build
    heal-and-rebuild catch) MUST dispatch on it, never on message substrings: the catch that
    matched "NO valid fallback" against a message that actually says "no valid editorial hold"
    was dead code for its whole life."""

    def __init__(self, msg: str = "", *, kind: str = ""):
        super().__init__(msg)
        self.kind = kind


class VisionBackendError(RuntimeError):
    """The vision backend was UNAVAILABLE (billing/quota exhausted, bad key, or persistent outage),
    so footage could not be verified. This is INFRASTRUCTURE, not content: unlike a
    NonRetryableBuildError it IS retryable — once the backend is restored (e.g. API credits topped
    up), a resume re-runs verification and the render completes. Carries `.kind` in
    {'billing','auth','down'} so the caller can show an actionable message and mark the job retryable.

    Distinguished so a render never (a) grinds hours of doomed image-fallback against a dead API,
    (b) checkpoints the errored verify stage as 'done' (which made Resume skip verify and re-hit the
    same wall), or (c) reports 'footage gap' when the real problem is an unpaid API bill."""

    def __init__(self, message: str, kind: str = "down"):
        super().__init__(message)
        self.kind = kind


def is_content_stop(exc) -> bool:
    """Whether a failure is the kind of CONTENT verdict a review draft may legitimately deliver.

    Every driver that offers "…→ auto-build a REVIEW DRAFT instead" has to answer this question, and
    each of them used to answer it alone, with `isinstance(exc, NonRetryableBuildError)`. That misses
    the semantic-recovery pagination guard, which is every bit a content verdict — beats fail the
    publication contract and the repair walk is out of pages — but is raised as a PipelineError by
    machinery that lives in orchestrate, not verify. Job 0321078108 died on it after 6h20m with the
    portal's auto-review sitting right there, ineligible.

    So the test is by identity, not by class alone, and it lives in ONE place. Infrastructure is
    never a content stop: a dead vision backend needs the backend back, not a draft.
    """
    if isinstance(exc, VisionBackendError):
        return False
    if isinstance(exc, NonRetryableBuildError):
        return True
    # A typed content stop raised by non-verify machinery. Deliberately only this kind: integrity
    # kinds such as scene_lineage mean the artifact may not be ours and stay fatal in every mode.
    return str(getattr(exc, "kind", "") or "") == "selection_relevance"


def _file_fingerprint(path) -> str:
    """Content id for a file: a FULL sha256, memoized on disk against (size, mtime).

    Head-only + size was too weak — a re-encode or a trim can preserve both while changing every
    frame the verifier judged, and container metadata lives at the head, so two different cuts of
    one upload can share a head block. Sampling head/middle/tail is better but still blind to a
    change between the sampled windows, and "probably caught it" is not an identity.

    So: hash the whole file, once. The cost is bounded, not repeated — the digest is cached beside
    the media keyed by (size, mtime), so a 200MB source costs ~1s on first sight and nothing
    thereafter. Anything that rewrites the bytes moves mtime and re-hashes. This is the strong
    option and it is affordable precisely because it is memoized."""
    import hashlib
    # is_file(), not exists(): Path("") is PosixPath('.') — the CWD — which exists, so an empty
    # local_path sailed past an exists() guard and then blew up in with_suffix, taking the whole
    # verify pass with it. A directory is not a source either.
    if not path:
        return "missing"
    p = Path(path)
    if not p.is_file():
        return "missing"
    try:
        st = p.stat()
        # Nanoseconds matter: project/source rewrites often happen in one burst, and the old
        # whole-second stamp could hand back a pre-mutation content hash for a same-size rewrite.
        stamp = f"{st.st_size}:{int(st.st_mtime_ns)}"
        side = p.with_suffix(p.suffix + ".fp.json")
        try:
            prev = json.loads(side.read_text(encoding="utf-8"))
            if prev.get("stamp") == stamp and prev.get("fp"):
                return str(prev["fp"])
        except Exception:
            pass
        h = hashlib.sha256()
        with p.open("rb") as fh:
            for blk in iter(lambda: fh.read(1 << 22), b""):
                h.update(blk)
        fp = h.hexdigest()[:20]
        try:
            side.write_text(json.dumps({"stamp": stamp, "fp": fp}), encoding="utf-8")
        except OSError:
            pass                                        # a read-only cache dir must not fail a build
        return fp
    except OSError:
        return "unreadable"


def _norm_faces(names) -> str:
    """Face-ID names, order-independent and case-folded. They are IN the prompt ('Automatic Face-ID
    on this frame detected: …'), so they change the answer and must change the key — but a reordered
    list is the same evidence and must not."""
    toks = sorted({(n or "").strip().lower() for n in (names or []) if (n or "").strip()})
    return ",".join(toks)


def verdict_fingerprint(*, src_hash: str, source_id: str, shot_start: float, shot_end: float,
                        beat_text: str, required_entity: str, required_kind: str = "",
                        expected_visual: str = "", scene_query: str = "", era: str = "",
                        visual_policy: str = "", is_specific: bool = True,
                        faceid_names=(), multiframe: bool = False, image_id: str = "",
                        model: str = "", venue_fallback: bool = False,
                        must_see: str = "", exact_cast_warning: str = "") -> str:
    """Identity of a verdict: EVERY input that can change the answer.

    A verdict is reusable only when the QUESTION is byte-identical. The first cut of this keyed on
    beat text + shot + era + policy + model, which left real holes — each of these is interpolated
    into the prompt or decides which prompt is sent, so omitting any of them silently reuses the
    answer to a DIFFERENT question:

      required_kind    -> "(kind: character)" in the prompt
      expected_visual  -> "The exact moment should LOOK LIKE: …"
      scene_query      -> "Target scene: …"
      is_specific      -> selects the STRICT rule vs the lenient one. Same frame, opposite verdict.
      faceid_names     -> "Automatic Face-ID on this frame detected: …"
      multiframe       -> a start/mid/end contact sheet asks a different question than one frame
      image_id         -> the actual pixels judged (keyframe/sheet), which shot bounds do not pin:
                          a re-index can rewrite a keyframe while start/end stay put
      model            -> the REAL vision provider+model (see llm.vision_config), not the configured
                          text brain: with the deepseek default, vision is really Gemini, so keying
                          on eng_cfg.anthropic_model made Gemini and Claude verdicts collide.
      venue_fallback   -> selects the still layer's HOLDING-IMAGE question (the _venue prompt
                          block): the same frame under the venue question can legitimately get the
                          opposite verdict, so the two must never share a key. Appended to the hash
                          ONLY when True so every pre-existing (venue-less) cache key stays valid."""
    import hashlib
    # Mirror verify_frame exactly: non-specific moving-footage questions do not contain the
    # analyzer's storyboard/query. Besides avoiding needless cache misses, this makes the evidence
    # fingerprint describe the question that was actually asked. Venue fallback retains both.
    if not is_specific and not venue_fallback:
        expected_visual = ""
        scene_query = ""
    absence_elsewhere = bool(_absence_elsewhere_instruction(
        beat_text, required_entity, expected_visual=expected_visual, scene_query=scene_query))
    # Exact prompts did not change, so preserve their warm v9 cache. Every non-specific question
    # uses the rewritten generic rule (including venue fallback) and must cold-miss under v10.
    prompt_version = PROMPT_VERSION if is_specific else LENIENT_PROMPT_VERSION
    h = hashlib.sha256()
    parts = [src_hash, source_id, f"{float(shot_start):.3f}", f"{float(shot_end):.3f}",
             (beat_text or "").strip(), (required_entity or "").strip().lower(),
             (required_kind or "").strip().lower(), (expected_visual or "").strip(),
             (scene_query or "").strip(), (era or "").strip().lower(),
             (visual_policy or "").strip().lower(), "1" if is_specific else "0",
             _norm_faces(faceid_names), "mf" if multiframe else "sf",
             (image_id or ""), (model or "").strip(),
             prompt_version, SHEET_VERSION]
    if venue_fallback:
        parts.append("venue")
    # This is a conditional prompt clause, so only affected absence-at-A / subject-at-B questions
    # cold-miss. Every unrelated exact verdict keeps its warm v9 cache.
    if absence_elsewhere:
        parts.append("absence-elsewhere-v2")
    # must_see changes the QUESTION ("is the dagger visible?"), so a verdict cached without it
    # answers something else entirely and must not be reused. Appended only when set, so every
    # pre-existing key stays valid.
    if must_see:
        parts.append("look:" + must_see.strip().lower())
    # A source-title mismatch does not prove a shot wrong. It changes the strict question by asking
    # vision to resolve that warning from pixels, so only warned exact calls cold-miss their cache.
    if exact_cast_warning and is_specific:
        parts.append("exact-cast-warning:" + " ".join(exact_cast_warning.lower().split()))
    for part in parts:
        h.update(str(part).encode("utf-8", "replace"))
        h.update(b"\x1f")
    return h.hexdigest()[:32]


def _project_beat_era(proj: ClipProject, seg: ScriptSegment) -> str:
    """Re-derive the exact beat-local era string used by ``verify_and_repair``.

    This is deliberately module-level so the pre-render relevance contract can ask the same
    question without trusting a stale era string persisted beside the verdict.
    """
    analysis = (getattr(proj, "meta", {}) or {}).get("analysis", {}) or {}
    single = str(analysis.get("video_type", "") or "") == "single_scene"
    global_era = str(analysis.get("episode_hint", "") or "")
    global_ok = bool(analysis.get("episode_hint_verified", False))
    shim = type("A", (), {
        "anchor_scenes": analysis.get("anchor_scenes"),
        "movie_title": analysis.get("movie_title", ""),
        "characters": analysis.get("characters"),
        "actors": analysis.get("actors"),
    })()
    return _beat_era(
        seg, global_era, single, global_verified=global_ok,
        event_eras=_era.event_eras_from(shim), anchor_eras=_era.anchor_token_eras(shim))


def _selection_evidence_image_id(
        proj: ClipProject, shot, multiframe: bool, *, window_start: float = 0.0,
        window_end: float = 0.0) -> str:
    """Content identity of the actual single frame or deterministic contact sheet judged."""
    if shot is None:
        return ""
    if multiframe:
        src = proj.source(getattr(shot, "source_id", "") or "")
        src_hash = _file_fingerprint(getattr(src, "local_path", "") or "") if src else "missing"
        return f"sheet:{src_hash}:{float(window_start):.3f}-{float(window_end):.3f}"
    keyframe = str(getattr(shot, "keyframe_path", "") or "")
    return f"kf:{_file_fingerprint(keyframe)}" if keyframe else "kf:none"


def selection_verifier_evidence_record(
        proj: ClipProject, sel: ClipSelection, seg: ScriptSegment, *, shot=None,
        model: str, is_specific: bool, multiframe: bool, faceid_names=(),
        era: str, must_see: str) -> dict:
    """Return the immutable identity of the moving-footage judgment that will be persisted.

    ``verdict_fingerprint`` already covers source content, shot bounds, every prompt field, model,
    face evidence and frame/contact-sheet identity.  The wrapper below additionally covers the
    selected shot number and exact in/out window.  Those values are not part of the vision prompt,
    but they are part of what the renderer airs and therefore must invalidate evidence when edited.
    """
    if shot is None:
        try:
            shot = _shot_lookup(proj)(getattr(sel, "source_id", ""),
                                      getattr(sel, "shot_index", -1))
        except Exception:
            shot = None
    sid = str(getattr(sel, "source_id", "") or "")
    if (shot is None or not sid or str(getattr(shot, "source_id", "") or "") != sid
            or int(getattr(shot, "index", -1)) != int(getattr(sel, "shot_index", -1))):
        return {}
    src = proj.source(sid)
    source_fp = _file_fingerprint(getattr(src, "local_path", "") or "") if src else "missing"
    selection_in = float(getattr(sel, "in_point", 0.0) or 0.0)
    selection_out = float(getattr(sel, "out_point", 0.0) or 0.0)
    image_id = _selection_evidence_image_id(
        proj, shot, bool(multiframe), window_start=selection_in, window_end=selection_out)
    if (source_fp in ("", "missing", "unreadable")
            or image_id in ("", "kf:none", "kf:missing", "kf:unreadable")
            or not (selection_out > selection_in >= 0.0)):
        return {}
    model_id = str(model or "").strip()
    if not model_id:
        return {}
    faces = list(faceid_names or [])
    exact_cast_warning = _project_exact_cast_warning(proj, seg, sid) if is_specific else ""
    question_fp = verdict_fingerprint(
        src_hash=source_fp, source_id=sid,
        shot_start=(selection_in if multiframe else getattr(shot, "start", 0.0)),
        shot_end=(selection_out if multiframe else getattr(shot, "end", 0.0)),
        beat_text=getattr(seg, "text", ""),
        required_entity=getattr(seg, "required_entity", ""),
        required_kind=getattr(seg, "required_kind", ""),
        expected_visual=getattr(seg, "expected_visual", "") or "",
        scene_query=getattr(seg, "scene_query", "") or "", era=str(era or ""),
        visual_policy=_policy.policy_of(seg), is_specific=bool(is_specific),
        faceid_names=faces, multiframe=bool(multiframe), image_id=image_id,
        model=model_id, must_see=str(must_see or ""),
        exact_cast_warning=exact_cast_warning)
    import hashlib as _hashlib_ev
    parts = [
        f"selection-evidence-v{SELECTION_EVIDENCE_SCHEMA}", question_fp, sid,
        str(int(getattr(sel, "shot_index", -1))),
        f"{selection_in:.6f}", f"{selection_out:.6f}",
    ]
    digest = _hashlib_ev.sha256("\x1f".join(parts).encode("utf-8", "replace")).hexdigest()
    return {
        "schema_version": SELECTION_EVIDENCE_SCHEMA,
        "fingerprint": digest,
        "question_fingerprint": question_fp,
        "source_content_fingerprint": source_fp,
        "source_id": sid,
        "shot_index": int(getattr(sel, "shot_index", -1)),
        "selection_in": round(selection_in, 6),
        "selection_out": round(selection_out, 6),
        "shot_start": round(float(getattr(shot, "start", 0.0) or 0.0), 6),
        "shot_end": round(float(getattr(shot, "end", 0.0) or 0.0), 6),
        "image_id": image_id,
        "multiframe": bool(multiframe),
        "model": model_id,
        "is_specific": bool(is_specific),
        "faceid_names": sorted({str(x).strip().lower() for x in faces if str(x).strip()}),
        "era": str(era or ""),
        "must_see": str(must_see or ""),
        "exact_cast_warning": exact_cast_warning,
        "prompt_version": (PROMPT_VERSION if is_specific else LENIENT_PROMPT_VERSION),
        "sheet_version": SHEET_VERSION,
    }


def bind_selection_verifier_evidence(
        proj: ClipProject, sel: ClipSelection, seg: ScriptSegment, verdict: dict, *, shot=None,
        model: str, is_specific: bool, multiframe: bool, faceid_names=(),
        era: str, must_see: str) -> dict:
    """Bind ``verdict`` in place to the current selection, dropping any stale prior binding."""
    verdict.pop("selection_evidence", None)
    record = selection_verifier_evidence_record(
        proj, sel, seg, shot=shot, model=model, is_specific=is_specific,
        multiframe=multiframe, faceid_names=faceid_names, era=era, must_see=must_see)
    if record:
        verdict["selection_evidence"] = record
    return verdict


_VERDICT_TRANSITION_FIELDS = (
    "downgraded", "relaxed", "relevance_class", "contract_rejected",
)


def _clear_verifier_transition_state(verdict: dict) -> dict:
    """Drop selection-lifecycle labels from a newly answered verifier question.

    Vision-cache rows are question answers.  Downgrade/relaxation labels and a prior contract
    rejection describe what a caller subsequently did with an answer, so carrying them into a
    fresh strict scoped pass creates an impossible mixed record (strict bound evidence labelled as
    an earlier contextual verdict).  Callers may add current-run transition labels again after the
    strict answer has been evaluated; they must never arrive through the answer/cache boundary.
    """
    if isinstance(verdict, dict):
        for field in _VERDICT_TRANSITION_FIELDS:
            verdict.pop(field, None)
    return verdict


def selection_verifier_evidence_reason(
        proj: ClipProject, sel: ClipSelection, seg: ScriptSegment, verdict: dict) -> str:
    """Return a stable blocker code when persisted moving-footage proof is stale or absent."""
    record = (verdict or {}).get("selection_evidence")
    if not isinstance(record, dict):
        return "verifier_evidence_absent"
    if int(record.get("schema_version", 0) or 0) != SELECTION_EVIDENCE_SCHEMA:
        return "verifier_evidence_schema_mismatch"
    if (_policy.policy_of(seg) in (_policy.EXACT, _policy.CHARACTER)
            and record.get("multiframe") is not True):
        return "verifier_evidence_window_not_sampled"
    model = str(record.get("model", "") or "")
    served = str((verdict or {}).get("vision_served_by", "") or "")
    if not model or (served and served != "none" and served != model):
        return "verifier_evidence_model_mismatch"
    try:
        shot = _shot_lookup(proj)(getattr(sel, "source_id", ""),
                                  getattr(sel, "shot_index", -1))
        faces = (getattr(shot, "face_ids", None) or
                 ([getattr(sel, "identity", "")] if getattr(sel, "identity", "") else []))
        expected = selection_verifier_evidence_record(
            proj, sel, seg, shot=shot, model=model,
            # Preserve whether this was the strict or the deliberate lenient question: the
            # publication contract separately rejects a lenient proof for a current EXACT beat and
            # recognizes completed exact→contextual downgrades as content (not infrastructure).
            # The current deictic target below, however, changes the semantic question itself and
            # must never be borrowed from the old record.
            is_specific=bool(record.get("is_specific", False)),
            multiframe=bool(record.get("multiframe", False)), faceid_names=faces,
            era=_project_beat_era(proj, seg), must_see=effective_deictic_target(seg))
    except Exception:
        expected = {}
    if not expected:
        return "verifier_evidence_unrecomputable"
    if (str(record.get("fingerprint", "") or "") != expected["fingerprint"]
            or str(record.get("question_fingerprint", "") or "")
            != expected["question_fingerprint"]):
        return "verifier_evidence_mismatch"
    return ""


def _hit_provider_ok(entry, expected_model: str) -> bool:
    """A cached verdict may serve a lookup ONLY when the provider that actually produced it
    matches the model identity in the key it was found under. Verdicts now record
    `vision_served_by` (the ACTUAL server); a Claude-fallback answer is stored under a
    Claude-keyed fingerprint, so a Gemini-keyed lookup can never return it — this check is
    the belt-and-suspenders that also drops any mislabeled legacy-style entry. Entries
    without provenance (pre-upgrade caches) are accepted as-is: they were stored under the
    predicted key when prediction and server agreed."""
    sb = str((entry or {}).get("vision_served_by") or "")
    return (not sb) or sb == (expected_model or "")


def _verdict_schema_ok(v, *, required_entity: str = "", must_see: str = "",
                       complete_keep: bool = True, exact_cast_warning: str = "") -> bool:
    """Whether a verdict is complete enough to store in, or serve from, the vision cache.

    A ``replace`` is conclusive negative evidence once its status/verdict/confidence envelope is
    valid.  A ``keep`` is stronger: every boolean consumed by the publication contract must be
    explicitly typed.  This distinction prevents malformed positive replies (observed in production
    as ``matches_naration``) from becoming permanent cache hits that scoped re-verification can never
    repair.  Subject and instructed-look facts are conditional because those questions are required
    only when the corresponding prompt inputs were present.
    """
    if not isinstance(v, dict):
        return False
    if str(v.get("status", "ok")) not in ("ok", ""):
        return False
    if v.get("verdict") not in ("keep", "replace"):
        return False
    if not isinstance(v.get("confidence", 0.0), (int, float)):
        return False
    if v.get("verdict") == "replace" or not complete_keep:
        return True
    for field in ("matches_narration", "specific_enough", "quality_ok",
                  "wrong_subject_visible", "contradicts_narration"):
        if not isinstance(v.get(field), bool):
            return False
    if str(required_entity or "").strip() \
            and not isinstance(v.get("correct_subject_visible"), bool):
        return False
    if str(must_see or "").strip() and not isinstance(v.get("target_visible"), bool):
        return False
    if str(exact_cast_warning or "").strip() \
            and not isinstance(v.get("source_title_conflict_resolved"), bool):
        return False
    return True


def _load_verdict_cache(proj) -> dict:
    f = Path(proj.root) / "verdict_cache.json"
    if not f.exists():
        return {}
    try:
        d = json.loads(f.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def _save_verdict_cache(proj, cache: dict) -> None:
    f = Path(proj.root) / "verdict_cache.json"
    try:
        tmp = f.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(cache), encoding="utf-8")
        tmp.replace(f)
    except OSError:
        pass


def _beat_era(seg, global_era: str, single_scene: bool, *, global_verified: bool = False,
              event_eras: dict | None = None, anchor_eras: list | None = None) -> str:
    """The era/season constraint for ONE beat — see `era.beat_era` for the ordering and why.

    This used to return the global hint IMMEDIATELY for single-scene videos, never reading the
    beat. That made one unvalidated LLM string ("S04E01" for a scene that is S03E10) the era of all
    229 beats at once, including the ones about the Red Wedding (S03E09). Era is beat-local now,
    and an unverified global hint constrains nothing."""
    return _era.beat_era(seg, global_era, single_scene=single_scene,
                         global_verified=global_verified, event_eras=event_eras,
                         anchor_eras=anchor_eras)


_CONTACT_SHEET_MAX_ASPECT = 4


def _action_contact_sheet(src_path: str, shot_start: float, shot_end: float, dest: Path):
    """Build a 15% -> 50% -> 85% sheet from the exact selected source window.

    The caller used to pass the entire detected shot.  A target elsewhere in a long shot could then
    earn ``keep`` even when the much shorter trim that actually aired omitted it.  All strict
    publication evidence now calls this with the selected in/out window, and the persisted evidence
    fingerprint binds that same window.  Returns ``dest`` or ``None``.
    """
    import subprocess
    from .config import ffmpeg_exe
    if not src_path or not Path(src_path).exists():
        return None
    a, b = float(shot_start), float(shot_end)
    if b - a < 0.5:
        return None
    span = b - a
    ff = ffmpeg_exe()
    try:
        from PIL import Image
    except Exception:
        return None
    frames = []
    for i, t in enumerate((a + span * 0.15, a + span * 0.50, a + span * 0.85)):
        fp = dest.with_name(f"{dest.stem}_{i}.jpg")
        subprocess.run([ff, "-y", "-loglevel", "error", "-ss", f"{max(0.0, t):.2f}", "-i", str(src_path),
                        "-frames:v", "1", "-vf", "scale=426:-1", str(fp)], capture_output=True, timeout=20)
        if fp.exists():
            frames.append(fp)
    if len(frames) < 3:
        for fp in frames:
            fp.unlink(missing_ok=True)
        return None
    try:
        ims = [Image.open(f).convert("RGB") for f in frames]
        h = min(im.height for im in ims)
        ims = [im.resize((int(im.width * h / im.height), h)) for im in ims]
        sheet = Image.new("RGB", (sum(im.width for im in ims), h))
        x = 0
        for im in ims:
            sheet.paste(im, (x, 0)); x += im.width
        # Three 16:9 frames make a 5.33:1 strip. Gemini can reject particular strips at prompt
        # admission with ``BlockedReason.OTHER`` and zero candidates even though every source
        # frame is valid (measured beat 65). Preserve all START/MIDDLE/END pixels and their
        # left-to-right order; add only a neutral letterbox so the provider accepts the same
        # multiframe evidence. Falling back to one frame would weaken the exact-action gate.
        min_h = (sheet.width + _CONTACT_SHEET_MAX_ASPECT - 1) // \
            _CONTACT_SHEET_MAX_ASPECT
        if sheet.height < min_h:
            padded = Image.new("RGB", (sheet.width, min_h), (16, 16, 16))
            padded.paste(sheet, (0, (min_h - sheet.height) // 2))
            sheet = padded
        sheet.save(dest, quality=88)
    except Exception:
        dest = None
    finally:
        for fp in frames:
            fp.unlink(missing_ok=True)
    return dest if (dest and Path(dest).exists()) else None


def _shot_lookup(proj: ClipProject):
    cache: dict[str, dict] = {}

    def get(source_id, shot_index):
        if not source_id:
            return None
        if source_id not in cache:
            cache[source_id] = {s.index: s for s in _index.load_shots(proj, source_id)}
        return cache[source_id].get(shot_index)

    def all_shots(source_id):
        if not source_id:
            return []
        if source_id not in cache:
            cache[source_id] = {s.index: s for s in _index.load_shots(proj, source_id)}
        return list(cache[source_id].values())

    get.all_shots = all_shots
    return get


_DIRECT_ABSENCE_RX = re.compile(
    r"\b(?:(?:is|was|are|were)\s+not|isn['’]?t|wasn['’]?t|aren['’]?t|weren['’]?t)\s+"
    r"(?:\w+\s+){0,3}(?:(?:in|inside|at)\s+(?:the\s+)?"
    r"(?:room|chamber|hall|meeting|council|scene|castle|tent|garden|ship|battle|wedding|trial|feast)\b"
    r"|(?:present|there)\b)", re.I)

_DEATH_CLAIM_RX = re.compile(
    r"\b(?:dies?|died|death|dead|killed|murdered|assassinated|executed|poisoned|slain)\b", re.I)
_TITLE_DEATH_NOUN = r"(?:death|killing|murder|execution|assassination|poisoning)"
_TITLE_DEATH_VERB = r"(?:dies|died|killed|murdered|assassinated|executed|poisoned|slain)"


def _direct_negative_contradiction(seg, vd) -> str:
    """Return a reason for the narrow, deterministic absence contradiction, else ``""``.

    This deliberately does not interpret general negation ("Jon is not in control"). It only fires
    for a named character/actor, an explicit room/place absence assertion, and positive visual
    evidence that the required person is on screen. The measured case is "Baelish is not even in
    the room" over a Baelish close-up.
    """
    if not isinstance(vd, dict) or vd.get("correct_subject_visible") is not True:
        return ""
    if (getattr(seg, "required_kind", "") or "").strip().lower() not in ("character", "actor"):
        return ""
    text = (getattr(seg, "text", "") or "").lower()
    absence = _DIRECT_ABSENCE_RX.search(text)
    if absence is None:
        return ""
    # A different concrete location is sometimes the authored visual proof of the absence (the
    # measured line says Baelish is not in the room while intentionally showing him on the escape
    # ship).  The vision reply must still explicitly deny contradiction; this only prevents the
    # deterministic backstop from undoing that valid judgment based on subject presence alone.
    if _absence_elsewhere_context(
            getattr(seg, "text", "") or "", getattr(seg, "required_entity", "") or "",
            expected_visual=getattr(seg, "expected_visual", "") or "",
            scene_query=getattr(seg, "scene_query", "") or ""):
        return ""
    ent = (getattr(seg, "required_entity", "") or "").lower()
    name_tokens = [t for t in re.findall(r"[a-z0-9]+", ent)
                   if len(t) >= 4 and t not in {"king", "queen", "lord", "lady", "body", "death"}]
    for token in name_tokens:
        named = re.search(rf"\b{re.escape(token)}\b", text)
        if named is not None and named.start() <= absence.start() \
                and absence.start() - named.end() <= 80:
            return (f"narration explicitly says {getattr(seg, 'required_entity', token)!r} is "
                    "absent from the place, but the verifier confirms that subject on screen")
    return ""


def _source_title_named_death_conflict(seg, source_title: str, char2actor=None) -> str:
    """Return a reason when a source title explicitly names a *different* roster member's death.

    Titles are negative evidence only: this never proves that a clip is correct. It requires a
    death claim in the beat, a character roster, and possessive/verb grammar that binds a different
    roster name to death in the title. Thus "Joffrey reacts to Tywin's death" is not misread as
    Joffrey's death, while "King Joffrey's Death" conflicts with a Jon Arryn death beat.
    """
    if not source_title or not char2actor:
        return ""
    # The NARRATION itself must make the death claim. A storyboard can mention a death merely as
    # scene context for an object/action beat; treating that as the line's subject falsely blocked
    # a necklace beat whose selected source was legitimately titled for Joffrey's death scene.
    if not _DEATH_CLAIM_RX.search(getattr(seg, "text", "") or ""):
        return ""
    required_entity = (getattr(seg, "required_entity", "") or "").lower()
    if not re.search(r"\b(?:death|body|corpse|remains)\b", required_entity):
        return ""                 # cannot safely identify which mentioned person is the decedent
    title = source_title.lower()
    target_tokens = {t for t in re.findall(r"[a-z0-9]+", (
        required_entity))
                     if len(t) >= 4 and t not in {"body", "death", "scene"}}
    for character in (char2actor or {}):
        ctoks = [t for t in re.findall(r"[a-z0-9]+", str(character).lower()) if len(t) >= 3]
        if not ctoks or target_tokens.intersection(ctoks):
            continue                                      # title names the beat's own person
        aliases = [r"\s+".join(re.escape(t) for t in ctoks)]
        # A distinctive given/single name is safe; do not use a surname alone ("Stark"/"Lannister"
        # would conflate relatives). Short first names such as Jon require the full name.
        if len(ctoks[0]) >= 4:
            aliases.append(re.escape(ctoks[0]))
        for alias in dict.fromkeys(aliases):
            owns_death = re.search(
                rf"\b(?:{alias})(?:['’]s)?\s+{_TITLE_DEATH_NOUN}\b", title)
            death_of = re.search(
                rf"\b{_TITLE_DEATH_NOUN}\s+of\s+(?:king\s+|queen\s+|lord\s+|lady\s+)?"
                rf"(?:{alias})\b", title)
            dies = re.search(
                rf"\b(?:{alias})\s+(?:is\s+|was\s+)?{_TITLE_DEATH_VERB}\b", title)
            if owns_death or death_of or dies:
                return (f"source title explicitly identifies {character!r} as the person who dies, "
                        f"while the beat "
                        f"requires {getattr(seg, 'required_entity', '')!r}")
    return ""


class _CharacterRoster(dict):
    """Canonical character→actor mapping with corpus-proven identity aliases."""

    def __init__(self, *args, identity_aliases=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.identity_aliases = dict(identity_aliases or {})


def _corpus_character_aliases(proj, canonical_names) -> dict[str, tuple[str, ...]]:
    """Infer only explicit alias statements repeated across independent source titles.

    A versus title is not alias evidence (``Oberyn vs The Mountain`` names two people).  Accepted
    forms must state an identity relation such as ``became/known as`` or use the common title-card
    apposition ``Canonical Name || The Epithet``.  Two distinct source titles must agree before an
    alias affects a cast warning.  This turns corpus evidence like ``Oberyn Martell became The Red
    Viper`` + ``Oberyn Martell || The Red Viper`` into a safe bridge for ``The Viper vs The
    Mountain`` without hard-coding one show or trusting an analyzer guess.
    """
    titles = [str(getattr(src, "title", "") or "")
              for src in (getattr(proj, "sources", None) or [])]
    evidence: dict[str, dict[str, set[str]]] = {}
    for canonical in canonical_names:
        words = re.findall(r"[a-z0-9]+", str(canonical or "").lower())
        if len(words) < 2:
            continue
        name_rx = r"\s+".join(re.escape(word) for word in words)
        per_alias: dict[str, set[str]] = {}
        for title in titles:
            normalized = " ".join(re.findall(r"[a-z0-9|]+", title.lower()))
            candidates = []
            for pattern in (
                    rf"\b{name_rx}\s+(?:became|(?:is\s+|was\s+)?known\s+as|a\s*k\s*a)\s+"
                    r"(?:the\s+)?([a-z][a-z0-9'-]*(?:\s+[a-z][a-z0-9'-]*)?)",
                    rf"\b{name_rx}\s*\|\|\s*(?:the\s+)?"
                    r"([a-z][a-z0-9'-]*(?:\s+[a-z][a-z0-9'-]*)?)",
            ):
                match = re.search(pattern, normalized, re.I)
                if match:
                    candidates.append(" ".join(re.findall(
                        r"[a-z0-9]+", match.group(1).lower())))
            for alias in candidates:
                if alias and alias != " ".join(words):
                    per_alias.setdefault(alias, set()).add(title)
        evidence[" ".join(words)] = per_alias

    out: dict[str, tuple[str, ...]] = {}
    canonical_tokens = {token for name in canonical_names
                        for token in re.findall(r"[a-z0-9]+", str(name).lower())}
    for canonical, aliases in evidence.items():
        admitted = []
        for alias, supporting_titles in sorted(aliases.items()):
            if len(supporting_titles) < 2:
                continue
            admitted.append(alias)
            parts = alias.split()
            # A distinctive epithet noun remains recognizable when an uploader omits its modifier
            # (``Red Viper`` → ``Viper``).  Require five characters and no canonical-name collision.
            if len(parts) > 1 and len(parts[-1]) >= 5 and parts[-1] not in canonical_tokens:
                admitted.append(parts[-1])
        if admitted:
            out[canonical] = tuple(dict.fromkeys(admitted))
    return out


def _project_char2actor(proj) -> dict[str, str]:
    """Return one canonical project roster for verification, relevance and build.

    Character names remain useful even when the analyzer omitted an actor.  Empty/``None`` actor
    values are deliberately not turned into aliases (the three publication paths previously built
    subtly different maps, including an accidental ``"none"`` actor in the still path).
    """
    analysis = (getattr(proj, "meta", {}) or {}).get("analysis", {}) or {}
    roster: dict[str, str] = {}
    for row in (analysis.get("characters") or []):
        if not isinstance(row, dict):
            continue
        name = str(row.get("name") or "").strip().lower()
        if name:
            roster[name] = str(row.get("actor") or "").strip()
    return _CharacterRoster(
        roster, identity_aliases=_corpus_character_aliases(proj, roster.keys()))


def _named_roster_characters(text: str, char2actor=None) -> set[str]:
    """Return canonical roster characters explicitly named by text or a safe actor alias."""
    normalized = " ".join(re.findall(r"[a-z0-9]+", str(text or "").lower()))
    found: set[str] = set()
    for raw_character, raw_actor in (char2actor or {}).items():
        character = " ".join(re.findall(r"[a-z0-9]+", str(raw_character).lower()))
        actor = " ".join(re.findall(r"[a-z0-9]+", str(raw_actor).lower()))
        if not character:
            continue
        char_parts = character.split()
        actor_parts = actor.split()
        aliases = [character]
        aliases.extend(
            str(alias or "").strip().lower()
            for alias in (getattr(char2actor, "identity_aliases", {}) or {}).get(
                character, ()) if str(alias or "").strip())
        # Distinctive given names are safe; surnames alone conflate Stark/Lannister relatives.
        if char_parts and len(char_parts[0]) >= 4:
            aliases.append(char_parts[0])
        if actor:
            aliases.append(actor)
            if actor_parts and len(actor_parts[0]) >= 4:
                aliases.append(actor_parts[0])
        if any(re.search(rf"\b{re.escape(alias)}\b", normalized) for alias in aliases):
            found.add(character)
    return found


def _exact_cast_expected_characters(seg, char2actor=None) -> set[str]:
    required = _named_roster_characters(
        getattr(seg, "required_entity", "") or "", char2actor)
    storyboard = " ".join((
        getattr(seg, "expected_visual", "") or "",
        getattr(seg, "scene_query", "") or "",
    ))
    return _named_roster_characters(storyboard, char2actor) - required


def _source_title_exact_cast_conflict(seg, source_title: str, char2actor=None) -> str:
    """Return a strict-scene *warning* when a title names different co-character(s).

    A source title describes a whole upload, not necessarily the selected shot, so this signal is
    never a general narration contradiction.  Strict verification uses it to demand affirmative
    pixel-level resolution; the contextual/abstract softening ladder remains available.  It
    requires an EXACT beat, a secondary roster character explicitly named in the storyboard, and
    a title that names different cast while naming none of the expected secondary cast.
    """
    if not source_title or not char2actor or not _policy.verify_strict(seg):
        return ""

    required = _named_roster_characters(
        getattr(seg, "required_entity", "") or "", char2actor)
    expected_secondary = _exact_cast_expected_characters(seg, char2actor)
    if not expected_secondary:
        return ""
    titled = _named_roster_characters(source_title, char2actor)
    # The job-local roster is intentionally compact and may omit a character named by a source
    # title.  A different given name attached to an expected character's exact surname is still
    # deterministic negative evidence (expected Catelyn Stark, title says Arya Stark).  This does
    # not guess unrelated names and is suppressed whenever the expected character is also named.
    title_text = str(source_title or "")
    non_names = {
        "the", "this", "that", "lady", "lord", "ser", "sir", "king", "queen",
        "prince", "princess", "house", "family", "clan", "team", "young", "old",
    }
    for expected in expected_secondary:
        parts = expected.split()
        if len(parts) < 2 or expected in titled:
            continue
        expected_given, expected_surname = parts[0], parts[-1]
        # The compact job roster may omit a character named by a title (e.g. Arya/Bran Stark).
        # Accept only a title-cased or all-caps person-shaped token and explicitly exclude common
        # honorific/family grammar.  The old normalized regex invented people named "Lady Stark",
        # "House Stark" and "The Stark".
        for match in re.finditer(
                rf"\b([A-Z][a-z]{{2,}}|[A-Z]{{3,}})\s+"
                rf"{re.escape(parts[-1].title())}\b", title_text):
            other_given = match.group(1).lower()
            if other_given != expected_given and other_given not in non_names:
                titled.add(f"{other_given} {expected_surname}")
    wrong = titled - required - expected_secondary
    if wrong and not (titled & expected_secondary):
        return (f"exact storyboard names {sorted(expected_secondary)!r}, but source title names "
                f"different cast {sorted(wrong)!r} and none of the expected co-characters")
    return ""


def _cast_warning_resolution_reason(vd, seg, char2actor=None) -> str:
    """Require a focused cast-warning KEEP to identify expected cast in its pixel evidence.

    A model once set ``source_title_conflict_resolved=true`` on a Joffrey-only dagger shot while
    its own reason never mentioned Catelyn.  The title still is not treated as frame-level proof;
    this merely requires the claimed pixel resolution to say whom it actually saw.  Bound Face-ID
    names may provide the same affirmative evidence.
    """
    if not isinstance(vd, dict) or vd.get("source_title_conflict_resolved") is not True:
        return "source-title cast warning was not affirmatively resolved from selected pixels"
    expected = _exact_cast_expected_characters(seg, char2actor)
    if not expected:
        return "source-title cast warning has no auditable expected co-character"
    faceid_text = " ".join(str(name) for name in (
        ((vd.get("selection_evidence") or {}).get("faceid_names") or [])))
    faceid_characters = _named_roster_characters(faceid_text, char2actor)
    if expected & faceid_characters:
        return ""
    evidence = " ".join((
        str(vd.get("reason", "") or ""),
        str(vd.get("source_title_conflict_evidence", "") or ""),
    ))
    # A name appearing in a negative explanation is not pixel proof.  Judge each contrast clause
    # independently so ``not present in frame one, but visible in frame two`` can still resolve.
    clauses = [" ".join(re.findall(r"[a-z0-9]+", clause.lower()))
               for clause in re.split(r"[.;]|\bbut\b|\bhowever\b|\balthough\b", evidence,
                                      flags=re.I)]
    affirmative = (
        "show", "shows", "shown", "showing", "depict", "depicts", "depicted",
        "feature", "features", "featured", "include", "includes", "included",
        "identify", "identifies", "identified", "confirm", "confirms", "confirmed",
        "recognize", "recognizes", "recognized", "recognise", "recognises", "recognised",
        "visible", "present", "seen", "appears", "appearing", "facing", "beside",
        "holding", "standing", "sitting", "speaking", "walking", "entering", "in frame",
        "on screen", "selected pixels", "face id",
    )
    roster_surname_counts = Counter(
        parts[-1] for character in (char2actor or {})
        if len(parts := str(character).lower().split()) >= 2)
    for character in expected:
        actor = str((char2actor or {}).get(character, "") or "").lower()
        aliases = [character]
        char_parts = character.split()
        actor_parts = actor.split()
        if char_parts and len(char_parts[0]) >= 4:
            aliases.append(char_parts[0])
        if actor:
            aliases.append(actor)
            if actor_parts and len(actor_parts[0]) >= 4:
                aliases.append(actor_parts[0])
        # Honorific+surnames are useful only when the roster makes them unambiguous.  A generic
        # ``Lady Stark`` must not prove Catelyn when Arya/Sansa/another Stark is also in scope.
        if len(char_parts) >= 2 and roster_surname_counts[char_parts[-1]] == 1:
            aliases.append(f"lady {char_parts[-1]}")
        for clause in clauses:
            # If the full canonical/actor name is present, do not retry its shorter given-name
            # alias after a negative full-name clause.  Otherwise ``Catelyn Stark is not present``
            # could be rejected for the full name and then accidentally accepted as ``Catelyn``.
            if re.search(rf"\b{re.escape(character)}\b", clause):
                clause_aliases = [character]
            elif actor and re.search(rf"\b{re.escape(actor)}\b", clause):
                clause_aliases = [actor]
            else:
                clause_aliases = aliases
            for alias in clause_aliases:
                match = re.search(rf"\b{re.escape(alias)}\b", clause)
                if not match:
                    continue
                words = clause.split()
                alias_words = alias.split()
                try:
                    start = next(i for i in range(len(words))
                                 if words[i:i + len(alias_words)] == alias_words)
                except StopIteration:
                    continue
                window = " ".join(words[max(0, start - 6):start + len(alias_words) + 7])
                alias_rx = rf"\b{re.escape(alias)}\b"
                negative_before = re.search(
                    rf"\b(?:no|without)\s+(?:the\s+)?{alias_rx}"
                    rf"|\b(?:not|never|cannot|can\s+not|can\s+t|unable\s+to)\s+"
                    rf"(?:(?:show|shows|showing|depict|depicts|feature|features|include|includes|"
                    rf"identify|identifies|confirm|confirms|see|sees)\s+)?(?:the\s+)?{alias_rx}"
                    rf"|\b(?:unclear|uncertain)\s+(?:whether|if)\s+(?:the\s+)?{alias_rx}",
                    clause)
                negative_after = re.search(
                    rf"{alias_rx}\s+(?:(?:is|was|does|did|appears|seems|may|might|can|could)\s+)?"
                    rf"(?:clearly\s+)?(?:not|absent|missing|offscreen|off\s+screen|"
                    rf"outside\s+the\s+frame)\b", clause)
                if negative_before or negative_after:
                    continue
                if any(re.search(rf"\b{re.escape(term)}\b", window) for term in affirmative):
                    return ""
    return ("source-title cast warning resolution did not identify any expected co-character "
            f"with affirmative pixel evidence ({sorted(expected)!r})")


def _project_source_title(proj, source_id: str) -> str:
    getter = getattr(proj, "source", None) if proj is not None else None
    source = getter(str(source_id or "")) if callable(getter) and source_id else None
    return ((getattr(source, "title", "") or "") + " " + str(source_id or ""))


def _subject_terms(seg, char2actor: dict) -> set:
    """Every name that would identify this beat's required subject on screen.

    The beat names a CHARACTER; Face-ID and many upload titles name the ACTOR, so the roster
    mapping has to be applied or "Shae" never matches a title that says "Sibel Kekilli". Returns
    lowercased word-level terms, ignoring one- and two-letter fragments that match everything."""
    ent = (getattr(seg, "required_entity", "") or "").strip()
    if not ent or (getattr(seg, "required_kind", "") or "").lower() not in (
            "character", "actor", "montage", ""):
        return set()
    out: set = set()
    for name in re.split(r",|\band\b|/|\+|;", ent):
        name = name.strip().lower()
        if not name:
            continue
        for form in (name, str((char2actor or {}).get(name, "") or "").lower()):
            for w in re.findall(r"[a-z']{3,}", form):
                out.add(w)
    return out


# A title-only subject match orders BELOW an identity-corroborated one. Not zero — an uploader's
# title is real, if weak, evidence — but it must never outrank a candidate the pipeline itself saw
# the subject in. See _subject_affinity.
_TITLE_ONLY_AFFINITY = 0.5


def _subject_affinity(cand, terms: set, proj) -> float:
    """How strongly this candidate is associated with the wanted subject. ORDERING ONLY.

    Nothing here admits a candidate: `_try_promote` applies the same strict verifier bar to
    whatever it is handed. This only decides which candidate that bar looks at first, which is the
    difference between finding the right person on the bench and never reaching them.

    Two independent signals, so a mislabelled title alone cannot carry a candidate:
      * the source TITLE naming the subject (what a human uploader wrote about the clip)
      * the match-time face/identity signals on the candidate itself

    That claim only holds when the second signal CAN fire. Measured on job ee93371e41 beat 134
    (required_entity 'Shae'): the project roster holds 9 characters and Shae is not one of them, so
    the identity arm is 0.0 on every candidate and the score collapses to the title arm alone —
    scores were 1.0, 1.0, 1.0 and then a cliff to 0.0, and the three "1.0" candidates are, on real
    frames, Tywin on the Iron Throne with no Shae in them. A title is not a frame, and a compilation
    titled for a character contains long stretches without them.

    So a title-only match is capped BELOW an identity-corroborated one. Ordering is unchanged
    wherever both signals exist; where only the title speaks, it can still bring a candidate
    forward — it simply cannot outrank a candidate the pipeline actually saw the subject in.
    """
    if not terms:
        return 0.0
    score = 0.0
    title = _project_source_title(proj, getattr(cand, "source_id", "") or "").lower()
    title_hit = 0.0
    if title:
        hits = sum(1 for t in terms if t in title)
        title_hit = min(1.0, hits / max(1, min(2, len(terms))))
    sig = getattr(cand, "signals", None) or {}
    ident = " ".join(str(x) for x in (
        sig.get("identity", ""), sig.get("face_ids", ""), sig.get("identities", ""))).lower()
    ident_hit = bool(ident and any(t in ident for t in terms))
    # an uploader's title is weaker evidence than the pipeline's own look at the pixels
    score += title_hit * (1.0 if ident_hit else _TITLE_ONLY_AFFINITY)
    if ident_hit:
        score += 1.0
    score += 0.25 * float(sig.get("faceid", 0.0) or 0.0)
    return score


def strict_window_verdict(av, seg, alt, proj, cfg, char2actor, *, downgrade: bool,
                          exact: bool, character: bool, must_see) -> dict:
    """The accept/reject decision for ONE candidate window — the whole of it, in one place.

    This logic used to live inside `_try_promote_inner`, unreachable from anywhere else. That is
    why nothing could audit the pool the way the gate judges it: any external check had to
    re-implement the bar, and a re-implementation that drifts is worse than no check at all.
    `_try_promote_inner` now calls this; there is no second copy.

    `av` is the vision verdict, or None for a transport error. A transport error is NOT a
    judgement: it returns `status="incomplete"` and `accept=False`, and a caller counting
    rejections must not count it as one. Same for any exception raised while judging.

    Returns the decision plus everything needed to explain it: the verdict fields, the rejection
    reason, the contradiction, the rung, and the window's identity.
    """
    ident = {
        "source_id": str(getattr(alt, "source_id", "") or ""),
        "shot_index": int(getattr(alt, "shot_index", -1) or -1),
        "in_point": float(getattr(alt, "in_point", 0.0) or 0.0),
        "out_point": float(getattr(alt, "out_point", 0.0) or 0.0),
    }
    if av is None:
        return {"accept": False, "status": "incomplete", "reason": "verifier_transport_error",
                "window": ident, "verdict": None}
    try:
        src = proj.source(ident["source_id"])
        title = ((getattr(src, "title", "") or "") + " " + ident["source_id"])
        conflict = _contradiction_reason(seg, av, title, char2actor)
        if conflict:
            av["contradicts_narration"] = True
            av["contradiction_reason"] = conflict

        if downgrade:
            accept = _exact_contextual_ok(av, seg, title, char2actor)
            reject = "" if accept else "contextual_bar_not_met"
        else:
            reject = ""
            if exact:
                reject = _strict_keep_rejection_reason(
                    av, seg, title, char2actor, must_see=must_see)
                if not reject:
                    reaction = _exact_reaction_context_evidence(proj, alt, seg, cfg=cfg)
                    if reaction.get("required"):
                        av["exact_reaction_context"] = reaction
                        if not reaction.get("passed"):
                            reject = str(reaction.get("reason")
                                         or "exact_reaction_context_unproven")
            elif character:
                reject = _character_keep_rejection_reason(
                    av, seg, title, char2actor, must_see=must_see)
            accept = (av.get("verdict") == "keep" and not conflict and not reject
                      and (not exact or not must_see or av.get("target_visible") is True))
            if not accept and not reject:
                reject = ("contradicts_narration" if conflict else
                          ("target_not_visible"
                           if (exact and must_see and av.get("target_visible") is not True)
                           else f"verdict_{av.get('verdict')}"))
    except Exception as exc:                             # noqa: BLE001 — a fault is not a rejection
        return {"accept": False, "status": "incomplete",
                "reason": f"strict_judgement_error:{type(exc).__name__}",
                "window": ident, "verdict": av.get("verdict") if isinstance(av, dict) else None}

    return {
        "accept": bool(accept),
        "status": "judged",
        "reason": "" if accept else str(reject or "rejected"),
        "window": ident,
        "verdict": av.get("verdict"),
        "matches_narration": av.get("matches_narration"),
        "correct_subject_visible": av.get("correct_subject_visible"),
        "wrong_subject_visible": av.get("wrong_subject_visible"),
        "specific_enough": av.get("specific_enough"),
        "target_visible": av.get("target_visible"),
        "contradiction_reason": av.get("contradiction_reason", ""),
        "model": ((av.get("selection_evidence") or {}).get("model", "")
                  if isinstance(av.get("selection_evidence"), dict) else ""),
        "prompt_version": ((av.get("selection_evidence") or {}).get("prompt_version", "")
                           if isinstance(av.get("selection_evidence"), dict) else ""),
        "evidence": av.get("selection_evidence") or {},
        "rung": "contextual" if downgrade else ("exact" if exact else
                                                ("character" if character else "generic")),
    }


def _project_exact_cast_warning(proj, seg, source_id: str) -> str:
    """Compute the conditional strict-prompt warning from the canonical project roster."""
    return _source_title_exact_cast_conflict(
        seg, _project_source_title(proj, source_id), _project_char2actor(proj))


def _contradiction_reason(seg, vd, source_title: str = "", char2actor=None) -> str:
    """Combine explicit vision evidence with narrowly deterministic contradiction checks."""
    if isinstance(vd, dict) and vd.get("contradicts_narration") is True:
        return "vision verifier explicitly marked the footage as contradicting the narration"
    return (_direct_negative_contradiction(seg, vd)
            or _source_title_named_death_conflict(seg, source_title, char2actor))


def _exact_positive_evidence_ok(vd, seg=None, source_title: str = "", char2actor=None) -> bool:
    """Positive evidence required before a rejected exact beat may enter a contextual rung."""
    if not isinstance(vd, dict):
        return False
    if vd.get("matches_narration") is not True or vd.get("specific_enough") is not True:
        return False
    if vd.get("wrong_subject_visible") is True or vd.get("quality_ok") is False:
        return False
    if seg is not None and _contradiction_reason(seg, vd, source_title, char2actor):
        return False
    return True


def _strict_keep_rejection_reason(vd, seg=None, source_title: str = "", char2actor=None,
                                  *, must_see: str = "") -> str:
    """Return why a purported strict ``keep`` cannot satisfy the publication contract.

    The verifier asks for all of these facts in one response.  Consuming only its top-level
    ``verdict`` let internally contradictory answers through repair (the measured case said
    ``verdict=keep`` while also saying ``correct_subject_visible=false``), so the expensive repair
    ladder stopped on footage the final relevance gate was guaranteed to reject.  Keep this
    predicate aligned with ``relevance_contract.evaluate_selection_relevance``: every positive
    promise must be explicit, while an absent ``era_ok`` remains unknown-but-not-disproven exactly
    as the publication gate defines it.
    """
    if not isinstance(vd, dict) or vd.get("verdict") != "keep":
        return "verifier did not return keep"
    for field in ("matches_narration", "specific_enough", "quality_ok"):
        if vd.get(field) is not True:
            return f"{field} is not positively true"
    if vd.get("wrong_subject_visible") is not False:
        return "wrong_subject_visible is not explicitly false"
    if str(getattr(seg, "required_entity", "") or "").strip() \
            and vd.get("correct_subject_visible") is not True:
        return "required subject is not positively visible"
    contradiction = _contradiction_reason(seg, vd, source_title, char2actor) if seg is not None \
        else ("vision verifier explicitly marked a contradiction"
              if vd.get("contradicts_narration") is True else "")
    if contradiction:
        return contradiction
    cast_warning = (_source_title_exact_cast_conflict(
        seg, source_title, char2actor) if seg is not None else "")
    if cast_warning:
        cast_resolution = _cast_warning_resolution_reason(vd, seg, char2actor)
        if cast_resolution:
            return f"{cast_resolution} ({cast_warning})"
    if vd.get("era_ok") is False:
        return "era_ok is false"
    if str(must_see or "").strip() and vd.get("target_visible") is not True:
        return f"instructed look target {must_see!r} is not positively visible"
    return ""


def _character_keep_rejection_reason(vd, seg=None, source_title: str = "", char2actor=None,
                                     *, must_see: str = "") -> str:
    """Reject a CHARACTER keep that disproves its own required-subject promise.

    Character prompts intentionally permit thematic right-subject filler, so this does not add the
    exact-scene cast/moment bar.  Their own schema still promises narration relevance, sufficient
    character coverage, quality, and a positively visible named subject.  The publication gate
    already requires those same fields; rejecting an internally contradictory keep here lets
    scoped recovery try positive Face-ID/vision alternates instead of stopping the whole render
    after verification has declared success.
    """
    if not isinstance(vd, dict) or vd.get("verdict") != "keep":
        return "verifier did not return keep"
    for field in ("matches_narration", "specific_enough", "quality_ok"):
        if vd.get(field) is not True:
            return f"{field} is not positively true"
    if vd.get("wrong_subject_visible") is not False:
        return "wrong_subject_visible is not explicitly false"
    if str(getattr(seg, "required_entity", "") or "").strip() \
            and vd.get("correct_subject_visible") is not True:
        return "required subject is not positively visible"
    contradiction = _contradiction_reason(seg, vd, source_title, char2actor) if seg is not None \
        else ("vision verifier explicitly marked a contradiction"
              if vd.get("contradicts_narration") is True else "")
    if contradiction:
        return contradiction
    if vd.get("era_ok") is False:
        return "era_ok is false"
    if str(must_see or "").strip() and vd.get("target_visible") is not True:
        return f"instructed look target {must_see!r} is not positively visible"
    return ""


def _exact_contextual_ok(vd, seg=None, source_title: str = "", char2actor=None) -> bool:
    """Exact→contextual acceptance: positive exact evidence plus the existing subject/era bar."""
    return (_exact_positive_evidence_ok(vd, seg, source_title, char2actor)
            and _contextual_subject_ok(vd))


def _contextual_subject_ok(vd) -> bool:
    """Is a verifier-rejected clip a legitimate NON-CONTRADICTORY contextual fallback? The single
    reliable signal is the REQUIRED SUBJECT being confirmed on screen (correct_subject_visible is
    True) — right character/scene, merely not the exact moment. matches_narration is NOT usable on
    its own: the AI verifier returns it False for nearly all META / COMMENTARY narration ("he isn't
    king anymore") even when the right subject is plainly visible, and the analyzer over-marks
    is_specific_claim on every beat, so neither can gate this. A clip whose subject is WRONG
    (correct_subject_visible is False) is contradictory and never accepted. (A clip that literally
    matches the narration with the subject not-disproven is also accepted.)"""
    # ERA still disqualifies. The verifier is already told that footage from a clearly different
    # season is wrong even when the character matches — but that only drove `verdict`, and this
    # function then re-admitted the clip because the subject was on screen. That is precisely how
    # season-1 child Bran ships under a season-8 Dragonpit line: 16-18 beats tagged wrong_era on the
    # frame eval, one of the two largest fallback failure modes.
    #
    # Read as "not disproven": a verdict cached before this field existed carries no era_ok and
    # still passes, so the gate tightens on fresh verdicts without invalidating the whole cache
    # (~$1 of vision calls per render).
    import os as _os_era
    if vd.get("era_ok") is False and _os_era.environ.get(
            "VIDLORE_CLIPSTUDIO_ERA_FALLBACK_GATE", "1").strip() not in ("0", "false", "no"):
        return False
    if vd.get("contradicts_narration") is True:
        return False
    return (vd.get("correct_subject_visible") is True
            or (bool(vd.get("matches_narration"))
                and vd.get("correct_subject_visible") is not False))


_EXACT_REACTION_INACTION_RX = re.compile(
    r"\b(?:does\s+not|doesn['’]?t|did\s+not|didn['’]?t|without|"
    r"refus(?:e|es|ed|ing)|silent(?:ly)?|no\s+argument|not\s+argu(?:e|ing))\b",
    re.I,
)
_EXACT_REACTION_CONTEXT_PRE_SEC = 30.0
_EXACT_REACTION_CONTEXT_POST_SEC = 5.0
_EXACT_REACTION_CONTEXT_MAX_SPAN_SEC = 30.0
_EXACT_REACTION_CONTEXT_MAX_EDIT_GAP = 4
_EXACT_REACTION_CONTEXT_MAX_TIME_GAP_SEC = 18.0
_EXACT_REACTION_CONTEXT_SCHEMA = 1


def _exact_reaction_context_required(seg) -> bool:
    """Whether pixels alone cannot prove this exact reaction/inaction claim.

    A silent face of the correct actor occurs in many unrelated scenes.  Tighten only the measured
    failure shape: an EXACT beat explicitly labelled as a reaction whose narration/storyboard says
    that the subject *does not* act, refuses nothing, or remains silent.  Ordinary reaction shots,
    visible actions and character filler retain their existing contracts.
    """
    if seg is None or not _policy.verify_strict(seg):
        return False
    if str(getattr(seg, "shot_intent", "") or "").strip().lower() != "reaction":
        return False
    text = " ".join((
        str(getattr(seg, "text", "") or ""),
        str(getattr(seg, "expected_visual", "") or ""),
    ))
    return bool(_EXACT_REACTION_INACTION_RX.search(text))


def _exact_reaction_context_evidence(proj, selection, seg, *, cfg=None) -> dict:
    """Re-derive a source/window-bound scene locator for an exact silent reaction.

    Vision must still return a strict KEEP, but that answer is not sufficient for an invisible
    non-action (the measured model hallucinated a trial decision over a marriage conversation).
    Require at least two non-roster storyboard terms in compact timed ASR immediately around the
    selected reaction.  The result is recomputed from the current indexed source by both verifier
    repair and the final publication audit; a persisted model assertion never proves this lock.
    """
    required = _exact_reaction_context_required(seg)
    detail = {
        "schema_version": _EXACT_REACTION_CONTEXT_SCHEMA,
        "required": required,
        "passed": not required,
        "reason": "not_required" if not required else "exact_reaction_context_unproven",
    }
    if not required:
        return detail
    sid = str(getattr(selection, "source_id", "") or "")
    try:
        w0 = float(getattr(selection, "in_point", 0.0) or 0.0)
        w1 = float(getattr(selection, "out_point", 0.0) or 0.0)
    except (TypeError, ValueError):
        return detail
    detail.update({
        "source_id": sid,
        "selection_window": [round(w0, 3), round(w1, 3)],
        "pre_roll_sec": _EXACT_REACTION_CONTEXT_PRE_SEC,
        "post_roll_sec": _EXACT_REACTION_CONTEXT_POST_SEC,
    })
    if not sid or not (w1 > w0 >= 0.0):
        return detail
    try:
        source = proj.source(sid)
    except Exception:
        source = None
    if source is None or str(getattr(source, "status", "") or "") != "ok":
        return detail
    source_fp = _file_fingerprint(str(getattr(source, "local_path", "") or ""))
    if source_fp in ("", "missing", "unreadable"):
        return detail
    detail["source_content_fingerprint"] = source_fp

    # The complete pipeline validates every ASR sidecar before match.  Re-check the selected
    # source here as well so a standalone publication audit cannot certify edited/stale words.
    try:
        if cfg is None:
            from .config import load_clip_config
            use_cfg = load_clip_config()
        else:
            use_cfg = cfg
        meta_path = Path(proj.index_dir) / f"{sid}.index.meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        expected_asr = _index.asr_semantic_fingerprint(proj, use_cfg)
        provenance_ok, provenance_reason, provenance = \
            _index._index_artifact_provenance_result(proj, source, meta)
        meta_current = bool(
            isinstance(meta, dict)
            and not meta.get("asr_refresh_in_progress")
            and meta.get("words") is True
            and int(meta.get("schema", 0) or 0) >= int(_index.INDEX_SCHEMA)
            and str(meta.get("asr_prompt_fingerprint", "") or "") == expected_asr
            and provenance_ok)
    except Exception:
        meta_current = False
        expected_asr = ""
        provenance_reason = "index_artifact_binding_validation_error"
        provenance = {}
    detail["asr_prompt_fingerprint_expected"] = expected_asr
    detail["index_artifact_provenance_reason"] = str(provenance_reason or "")
    if provenance:
        detail["index_artifact_fingerprints_current"] = provenance
    if not meta_current:
        detail["reason"] = "exact_reaction_context_asr_provenance_invalid"
        return detail

    try:
        from .discover import _STOPQ as stop_words
    except Exception:
        stop_words = set()

    def _terms(value: str) -> set[str]:
        out = set()
        for raw_word in re.findall(r"[a-z0-9']+", str(value or "").lower()):
            word = raw_word[:-2] if raw_word.endswith("'s") else raw_word
            if len(word) >= 4 and word not in stop_words:
                out.add(word)
        return out

    analysis = (getattr(proj, "meta", None) or {}).get("analysis", {}) or {}
    excluded = _terms(str(analysis.get("movie_title", "") or ""))
    excluded |= _terms(str(getattr(seg, "required_entity", "") or ""))
    excluded |= _terms(" ".join(str(x or "") for x in (
        getattr(seg, "entities", None) or [])))
    roster = _project_char2actor(proj)
    for character, actor in (roster or {}).items():
        excluded |= _terms(character)
        excluded |= _terms(actor)
    for aliases in (getattr(roster, "identity_aliases", {}) or {}).values():
        excluded |= _terms(" ".join(str(alias or "") for alias in aliases))
    # These describe the invisible decision, not the scene which independently locates it.
    excluded |= {
        "argue", "argued", "arguing", "argument", "doesn", "grant", "grants", "granted",
        "refuse", "refused", "refuses", "refusing", "silent", "silently", "without",
    }
    generic = {
        "game", "thrones", "scene", "episode", "season", "clip", "video", "exact",
        "character", "watching", "reaction", "shows", "showing", "visible", "moment",
    }
    target_terms = (
        _terms(str(getattr(seg, "scene_query", "") or ""))
        | _terms(str(getattr(seg, "expected_visual", "") or ""))
    ) - excluded - generic
    detail["target_terms"] = sorted(target_terms)
    import hashlib as _hashlib_context
    detail["target_hash"] = _hashlib_context.sha256(
        "\x1f".join(sorted(target_terms)).encode("utf-8", "replace")).hexdigest()
    if len(target_terms) < 2:
        return detail

    def _token_matches(target: str, actual: str) -> bool:
        target = re.sub(r"[^a-z0-9]", "", target.lower())
        actual = re.sub(r"[^a-z0-9]", "", actual.lower())
        if not target or not actual:
            return False
        if target == actual:
            return True
        return bool(
            min(len(target), len(actual)) >= 5
            and abs(len(target) - len(actual)) <= 3
            and (target.startswith(actual) or actual.startswith(target)))

    rows = []
    context0, context1 = w0 - _EXACT_REACTION_CONTEXT_PRE_SEC, \
        w1 + _EXACT_REACTION_CONTEXT_POST_SEC
    try:
        shots = sorted(_index.load_shots(proj, sid), key=lambda shot: (
            float(getattr(shot, "start", 0.0) or 0.0), int(getattr(shot, "index", -1))))
    except Exception:
        shots = []
    try:
        selected_shot_index = int(getattr(selection, "shot_index", -1))
    except (TypeError, ValueError):
        selected_shot_index = -1
    selected_shot = next((shot for shot in shots
                          if int(getattr(shot, "index", -1)) == selected_shot_index), None)
    detail["selected_shot_index"] = selected_shot_index
    if selected_shot is None:
        return detail
    try:
        selected_shot_start = float(selected_shot.start)
        selected_shot_end = float(selected_shot.end)
    except (TypeError, ValueError, AttributeError):
        return detail
    selection_overlaps_shot = max(w0, selected_shot_start) < min(w1, selected_shot_end)
    detail["selection_overlaps_indexed_shot"] = selection_overlaps_shot
    if not selection_overlaps_shot:
        return detail
    for shot in shots:
        try:
            s0, s1 = float(shot.start), float(shot.end)
        except (TypeError, ValueError, AttributeError):
            continue
        if s1 < context0 or s0 > context1:
            continue
        actual_tokens = re.findall(
            r"[a-z0-9']+", str(getattr(shot, "transcript", "") or "").lower())
        matched = {
            target for target in target_terms
            if any(_token_matches(target, token) for token in actual_tokens)
        }
        if matched:
            rows.append((s0, s1, int(getattr(shot, "index", -1)), matched))
    best = None
    for start in range(len(rows)):
        matched = set()
        for end in range(start, len(rows)):
            span0 = rows[start][0]
            span1 = rows[end][1]
            if span1 - span0 > _EXACT_REACTION_CONTEXT_MAX_SPAN_SEC:
                break
            matched |= rows[end][3]
            if len(matched) >= 2:
                last_match_index = rows[end][2]
                edit_gap = selected_shot_index - last_match_index
                time_gap = max(0.0, w0 - span1)
                if (edit_gap < 0 or edit_gap > _EXACT_REACTION_CONTEXT_MAX_EDIT_GAP
                        or time_gap > _EXACT_REACTION_CONTEXT_MAX_TIME_GAP_SEC):
                    continue
                rank = (len(matched), -(span1 - span0), -span0)
                if best is None or rank > best[0]:
                    best = (rank, span0, span1, rows[start][2], last_match_index,
                            set(matched), edit_gap, time_gap)
    if best is None:
        detail["matched_terms"] = []
        return detail
    _rank, span0, span1, first_shot, last_shot, matched, edit_gap, time_gap = best
    detail.update({
        "passed": True,
        "reason": "timed_asr_scene_locator",
        "matched_terms": sorted(matched),
        "context_span": [round(span0, 3), round(span1, 3)],
        "context_shots": [first_shot, last_shot],
        "context_to_selection_edit_gap": int(edit_gap),
        "context_to_selection_time_gap_sec": round(float(time_gap), 3),
    })
    return detail


def _season_num(text: str):
    """Season number declared anywhere in a string (S03E10 / 'season 3' / 'season three' / 3x10)."""
    return _era.parse_season(text or "")


_EPISODE_RX = re.compile(r"\bS0?\d{1,2}\s?E0?(\d{1,2})\b|\b\d{1,2}\s?x\s?0?(\d{1,2})\b", re.I)


def _episode_num(text: str):
    """Episode number declared anywhere in a string (S03E10 / 3x10), else None."""
    m = _EPISODE_RX.search(text or "")
    if m:
        n = m.group(1) or m.group(2)
        return int(n) if n else None
    return None


def _era_conflict(era_a: str, era_b: str) -> bool:
    """Do two era strings CONTRADICT each other? Era strings arrive in mixed formats —
    _beat_era returns the project's raw episode hint ('S04E01') for single-scene videos while
    _title_season normalizes to 'season 4' — so a naive string != is NOT an era test: it
    rejected every same-season still candidate as 'wrong era (beat S04E01 vs source season 4)'
    and release-blocked a finished render. Compare CANONICALLY: a conflict needs both sides to
    declare a season and the seasons to differ, or (same/undeclared season) both to declare an
    episode and the episodes to differ. An era only one side declares can't contradict."""
    sa, sb = _season_num(era_a), _season_num(era_b)
    if sa is not None and sb is not None and sa != sb:
        return True
    ea, eb = _episode_num(era_a), _episode_num(era_b)
    return ea is not None and eb is not None and ea != eb


def _beat_mention_tokens(seg) -> set:
    """Every person/thing this beat MENTIONS (required_entity + its entities list). A shot showing
    any of these characters is CO-MENTIONED — narratively relevant, not contradictory (e.g. a Tywin
    shot on 'Joffrey calls Tywin a coward')."""
    names = [getattr(seg, "required_entity", "") or ""] + list(getattr(seg, "entities", []) or [])
    toks = set()
    for nm in names:
        toks |= {w for w in re.findall(r"[a-z0-9]+", (nm or "").lower()) if len(w) > 2}
    return toks


def _confirmed_wrong_character(seg, faceid_names, extra_ok_tokens=frozenset(),
                               char2actor=None) -> bool:
    """True IFF Face-ID POSITIVELY identifies a specific person who is NEITHER the beat's required/
    co-mentioned entity NOR in extra_ok_tokens (the scene roster for a single-scene deep-dive, where
    any main-cast member is contextually valid). This is the ONLY hard block for a character
    fallback — an EMPTY / unconfirmed Face-ID is NOT a confirmed wrong character (the required person
    may be present off-face), so it does not block.

    Face-ID reports ACTOR names while beats name CHARACTERS, so the roster must map between them:
    without it a PERFECT Joffrey frame (face 'jack gleeson') reads as a confirmed WRONG character
    for a beat about 'Joffrey Baratheon'. That never bit before only because Face-ID resolved no
    leads at all in the failing render — fixing the reference builder would have activated it."""
    ok = _beat_mention_tokens(seg) | set(extra_ok_tokens)
    for ch, ac in (char2actor or {}).items():
        cht = {w for w in re.findall(r"[a-z0-9]+", (ch or "").lower()) if len(w) > 2}
        act = {w for w in re.findall(r"[a-z0-9]+", (ac or "").lower()) if len(w) > 2}
        if cht and act and (cht <= ok or act <= ok):
            ok |= cht | act                            # same person under either naming
    for nm in (faceid_names or []):
        nt = {w for w in re.findall(r"[a-z0-9]+", (nm or "").lower()) if len(w) > 2}
        if nt and ok and not (nt & ok):
            return True                                # a DIFFERENT identified person → contradictory
    return False


def _entity_face_confirmed(seg, faceid_names, char2actor=None) -> bool:
    """True IFF Face-ID POSITIVELY places the beat's required entity in the shot.

    The counterpart to _confirmed_wrong_character, and the one that was missing. Beats were kept on
    the ABSENCE of a wrong face; nothing ever required the PRESENCE of the right one. Face-ID
    identifies actors while beats name characters, so match either way round."""
    ent = (getattr(seg, "required_entity", "") or "").strip().lower()
    if not ent or not faceid_names:
        return False
    from .orchestrate import entity_name_variants
    face_toks = {w for w in re.findall(r"[a-z0-9]+", " ".join(faceid_names).lower()) if len(w) > 2}
    if not face_toks:
        return False
    for v in entity_name_variants(ent, char2actor):
        if v and all(t in face_toks for t in v):
            return True
    return False


def _generic_filler_ok(vd, seg, src_title, faceid_names, beat_era, ok_tokens=frozenset(),
                       char2actor=None) -> tuple:
    """May an exact beat air its clip as honestly-labelled GENERIC FILLER? -> (ok, why).

    `vd` must be a FRESH LENIENT verdict on the footage that would actually air — not the strict
    verdict that just rejected it, and never a recycled one. The old code had no fresh pass at all:
    it relabelled the rejecting verdict as "keep", so the verifier's own judgment was overwritten
    and shipped.

    Every condition is POSITIVE. "No wrong face was confirmed" is not evidence of anything — it was
    vacuously true for every frame in the failing render, because Face-ID could not resolve the
    leads. Requires all of:
      1. a judgment exists (an outage is not a pass);
      2. the LENIENT pass still says keep — asked the easy question, it must at least answer yes;
      3. it affirms matches_narration — on-topic, asserted, not merely un-refuted;
      4. quality is not rejected — a blurry/unreadable frame is not editorially relevant;
      5. no different identified person, and no wrong main subject seen (contradictory);
      6. no era conflict with the source's declared season (same-show/era)."""
    if vd is None:
        return False, "no lenient judgment (verifier unavailable) — an outage is not a pass"
    if vd.get("verdict") != "keep":
        return False, "the lenient pass ALSO rejected this footage"
    if vd.get("matches_narration") is not True:
        return False, "lenient pass did not affirm the footage is on-topic"
    if vd.get("quality_ok") is False:
        return False, "lenient pass rejected the quality"
    contradiction = _contradiction_reason(seg, vd, src_title, char2actor)
    if contradiction:
        return False, f"direct contradiction: {contradiction}"
    if vd.get("wrong_subject_visible") is True:
        return False, "a different character is the main subject (contradictory)"
    if _confirmed_wrong_character(seg, faceid_names, ok_tokens, char2actor):
        return False, "Face-ID confirms a DIFFERENT person in this shot (contradictory)"
    _bn, _sn = _season_num(beat_era), _season_num(src_title)
    if _bn is not None and _sn is not None and _bn != _sn:
        return False, f"wrong era (beat season {_bn} vs source season {_sn})"
    return True, (f"lenient keep + matches_narration, no wrong subject, era ok "
                  f"(conf {vd.get('confidence', 0.0)})")


def _present_unconfirmed_ok(vd, seg, src_title, faceid_names, beat_era, ok_tokens=frozenset(),
                            *, char2actor=None) -> bool:
    """May a CHARACTER beat whose exact footage the verifier rejected still air as CONTEXTUAL?

    Only on POSITIVE evidence that the required person is there. The old rule accepted on the
    absence of a wrong one, which is not the same claim — and in the render that exposed this it was
    vacuously true everywhere, because Face-ID had NO REFERENCE for Jack Gleeson (Joffrey, the
    co-lead), Conleth Hill (Varys) or Julian Glover (Pycelle). With the leads unidentifiable, "no
    confirmed wrong character" is satisfied by every frame in existence, so 121 exact beats were
    downgraded to contextual and kept, "honestly labeled", over whatever happened to be there.

    An empty Face-ID is UNKNOWN, never innocent. Requires ALL of:
      (1) vision positively affirms matches_narration AND specific_enough;
      (2) vision did not see a different main subject or direct contradiction;
      (3) no DIFFERENT identified person in the shot;
      (4) Face-ID POSITIVELY confirms the required entity — the evidence that was never demanded;
      (5) a POSITIVE same-era signal (an unconstrained era proves nothing)."""
    if not _exact_positive_evidence_ok(vd, seg, src_title, char2actor):
        return False                                   # explicit mismatch/insufficiency cannot downgrade
    if vd.get("wrong_subject_visible") is True:
        return False                                   # vision saw a different main subject
    if _confirmed_wrong_character(seg, faceid_names, ok_tokens, char2actor):
        return False                                   # a DIFFERENT identified person → contradictory
    if not _entity_face_confirmed(seg, faceid_names, char2actor):
        return False                                   # UNKNOWN ≠ present. Positive evidence only.
    _bn = _season_num(beat_era)
    if _bn is None:
        return False                                   # unconstrained era → no positive signal → block
    _sn = _season_num(src_title)
    if _sn is not None and _sn != _bn:
        return False                                   # source declares a DIFFERENT season → wrong era
    return True                                        # right person, right era, no wrong character


def _scene_affinity_order(alts, seg, proj, orig_source_id: str):
    """Stable reorder of a beat's alternates so SCENE-AFFINE sources are tried first when repairing
    an exact beat. The vision verifier judges one frame against one narration line — it cannot see
    what episode a frame comes from, so visually-plausible wrong-scene shots pass ('the king at a
    table with wine' verifies against a pie-moment beat even from a different season's dinner).
    Measured in a full render: a beat whose scene_query named the cited scene was repaired with a
    shot from a source sharing ZERO scene tokens while four sources titled with the cited scene sat
    in the pool — 3 of the 5 wrong-footage beats in that render's audit shared this signature.

    Tiers (relevance order preserved within each — the sort is stable):
      0  source is dialogue-verified for the anchor scene (anchor_verified), or its TITLE shares
         >=2 scene-specific tokens with the beat's scene_query (same token rule as discover's
         anchor/key-scene coverage: word/prefix match, movie-title + stop tokens excluded)
      1  the source of the ORIGINAL rejected pick — match chose this source for the scene; the
         verifier rejected one FRAME of it, which is no evidence against its other shots
      2  everything else
    Ordering only — every candidate still faces the same verifier, window-QC, and reuse gates."""
    import re as _re_aff
    try:
        from .discover import _STOPQ as _AFF_STOP
    except Exception:
        _AFF_STOP = set()
    _mv = {w for w in _re_aff.findall(
        r"[a-z']+", (((getattr(proj, "meta", None) or {}).get("analysis", {}) or {})
                     .get("movie_title", "") or "").lower()) if len(w) > 2}

    def _aff_terms(value) -> set[str]:
        return {
            w for w in _re_aff.findall(r"[a-z']+", str(value or "").lower())
            if len(w) > 2 and w not in _mv and w not in _AFF_STOP
        }

    # Character names identify a cast member, not a scene.  Counting Tywin+Tyrion as two scene
    # anchors promoted every conversation between them into the trial tier (the measured marriage
    # false-positive).  Remove the complete project roster and beat entities; two *event/location*
    # terms such as trial+combat are still required for tier zero.
    roster_terms = _aff_terms(getattr(seg, "required_entity", "") or "")
    roster_terms |= _aff_terms(" ".join(
        str(value or "") for value in (getattr(seg, "entities", None) or [])))
    roster = _project_char2actor(proj)
    for character, actor in (roster or {}).items():
        roster_terms |= _aff_terms(character)
        roster_terms |= _aff_terms(actor)
    for aliases in (getattr(roster, "identity_aliases", {}) or {}).values():
        roster_terms |= _aff_terms(" ".join(str(alias or "") for alias in aliases))
    toks = _aff_terms(getattr(seg, "scene_query", "") or "") - roster_terms

    def _tier(a):
        try:
            src = proj.source(a.source_id)
        except Exception:
            src = None
        if src is not None and (getattr(src, "extra", None) or {}).get("anchor_verified"):
            return 0
        tw = set(_re_aff.findall(r"[a-z']+", ((getattr(src, "title", "") if src else "") or "").lower()))
        if toks and sum(1 for w in toks
                        if any(t == w or (t.startswith(w) and len(t) - len(w) <= 2)
                               for t in tw)) >= 2:
            return 0
        if a.source_id == orig_source_id:
            return 1
        return 2
    return sorted(alts, key=_tier)


def _venue_candidates(sel, seg, proj, get_shot, beat_era: str, cap: int = 8):
    """Bounded candidate pool for the scene-VENUE contextual rung (see the call site for the full
    rationale). Finds the anchor scene the beat's scene_query points at (>=1 shared scene-specific
    token), then returns ClipCandidates from sources matching THAT anchor — anchor_verified, or a
    >=2 anchor-token title match — skipping shots already tried as alternates, era-conflicting
    sources, and sub-2s shots. Ordered: anchor_verified sources first, then title-match strength,
    then shot length (legibility proxy). Empty list = no venue evidence → the beat still blocks."""
    import re as _re_v
    from .models import ClipCandidate
    try:
        from .discover import _STOPQ as _VSTOP
    except Exception:
        _VSTOP = set()
    ana = (getattr(proj, "meta", None) or {}).get("analysis", {}) or {}
    _mv = {w for w in _re_v.findall(r"[a-z']+", (ana.get("movie_title", "") or "").lower())
           if len(w) > 2}
    sqt = {w for w in _re_v.findall(r"[a-z']+", (getattr(seg, "scene_query", "") or "").lower())
           if len(w) > 2 and w not in _mv and w not in _VSTOP}
    if not sqt:
        return []
    best_st, best_ov = set(), 0
    for sc in (ana.get("anchor_scenes") or []):
        st = {w for w in _re_v.findall(
                  r"[a-z']+", ((sc.get("name", "") or "") + " " + (sc.get("query", "") or "")).lower())
              if len(w) > 2 and w not in _mv and w not in _VSTOP}
        ov = len(sqt & st)
        if ov > best_ov:
            best_ov, best_st = ov, st
    if best_ov < 1:
        return []
    tried = {(a.source_id, a.shot_index) for a in (sel.alternates or [])}
    tried.add((sel.source_id, sel.shot_index))
    hold = max(0.0, float(getattr(sel, "out_point", 0.0)) - float(getattr(sel, "in_point", 0.0)))
    want_dur = max(4.0, min(8.0, hold or 4.0))
    scored = []
    for src in proj.sources:
        if getattr(src, "status", "") != "ok":
            continue
        title = (getattr(src, "title", "") or "")
        if _era_conflict(beat_era, title):
            continue                                   # a declared wrong-season source never airs
        tw = set(_re_v.findall(r"[a-z']+", title.lower()))
        hits = sum(1 for w in best_st
                   if any(t == w or (t.startswith(w) and len(t) - len(w) <= 2) for t in tw))
        averi = bool((getattr(src, "extra", None) or {}).get("anchor_verified"))
        if not averi and hits < 2:
            continue
        all_shots = getattr(get_shot, "all_shots", lambda _s: [])(src.id)
        for sh in all_shots or []:
            k = (src.id, getattr(sh, "index", -1))
            if k in tried:
                continue
            d = float(getattr(sh, "end", 0.0)) - float(getattr(sh, "start", 0.0))
            if d < 2.0:
                continue
            scored.append(((0 if averi else 1, -hits, -d), src.id, sh))
    scored.sort(key=lambda x: x[0])
    out = []
    for _k, sid, sh in scored[:cap]:
        s0 = float(sh.start)
        out.append(ClipCandidate(segment_index=sel.segment_index, source_id=sid,
                                 shot_index=int(sh.index), score=0.30, in_point=s0,
                                 out_point=min(float(sh.end), s0 + want_dur),
                                 signals={"venue_fallback": True}))
    return out


def _strict_scene_neighborhood_candidates(sel, seg, proj, get_shot, cfg, *,
                                          exclude=None, beat_era: str = "",
                                          cap: int = 12, radius: int = 6,
                                          source_cap: int = 4,
                                          allow_indexed_pool_sources: bool = False,
                                          whole_source_probe: bool = False):
    """Return a small strict-repair pool around already-ranked, scene-affine seeds.

    Match deliberately keeps only a few shots per source in ``alternates``/``deep_alternates``.
    That is good for global variety, but it creates a verifier blind spot: the right *source* can
    be on the bench while the exact action is one detected shot beside the retained seed.  Do not
    re-rank the global matcher to repair that local miss.  Instead, only after the chosen footage
    has failed strict verification, expand at most ``radius`` shot indexes around the selection's
    existing seeds, and only inside the most scene-affine eligible sources.

    Source affinity is derived from the beat's scene query + expected visual + required entity and
    the source title/acquisition query.  A required-entity-only match is intentionally insufficient:
    every compilation about a lead character would otherwise become a neighborhood source.  One
    disjoint five-shot region may also be reserved when the already-retained source's *timed shot
    transcripts* contain at least two rare scene-specific anchors.  This is a local verifier repair,
    not a new global retrieval channel.  Within the top affine sources, persisted CLIP similarity
    orders unseen shots when available; distance from a retained seed is the deterministic fallback.
    The caller still applies the unchanged strict vision, exact-window QC, reuse ledger and
    materialization transaction.

    ``allow_indexed_pool_sources`` is reserved for the scoped publication-recovery pass, after a
    complete global rematch has already failed.  In that lane only, at most two native-HD sources
    whose title/acquisition query match at least two non-generic storyboard terms may seed this
    same bounded pool.  This repairs the measured failure where the exact source was indexed but
    sat below match's source-diversity/deep-bench cut.  It does not change normal matching, enlarge
    ``cap``, or approve pixels: every candidate still faces the unchanged strict vision, Window-QC,
    reuse, materialization, and final publication contracts.

    ``whole_source_probe`` is the final, still-bounded action locator for that same scoped lane.
    It inspects at most five hard-eligible, CLIP-ranked shots from the strongest native-HD source
    bound to the beat's anchor episode, and only outside every retained local/deep/timed region.
    The caller invokes it only after the ordinary neighborhood failed and sends its candidates
    through the exact same strict vision, Window-QC, reuse and materialization transaction.
    """
    from .models import ClipCandidate

    try:
        cap = max(0, min(5 if whole_source_probe else 24, int(cap)))
        radius = max(1, min(12, int(radius)))
        source_cap = max(1, min(8, int(source_cap)))
    except (TypeError, ValueError):
        return []
    if cap <= 0:
        return []
    if whole_source_probe and not allow_indexed_pool_sources:
        return []

    try:
        from .discover import _STOPQ as _NSTOP
    except Exception:
        _NSTOP = set()
    ana = (getattr(proj, "meta", None) or {}).get("analysis", {}) or {}
    movie_tokens = {
        w for w in re.findall(r"[a-z']+", (ana.get("movie_title", "") or "").lower())
        if len(w) > 2
    }
    # These words describe the *container*, not the scene.  Removing them keeps a source from
    # qualifying merely because every upload is titled "Game of Thrones scene/episode".
    generic = {"game", "thrones", "scene", "scenes", "episode", "season", "clip", "clips",
               "full", "hd", "official", "video", "moment", "moments", "camera", "shot",
               "shots", "closeup", "wide", "medium", "aerial", "overhead", "establishing",
               "tracking", "pulling", "framing", "zoom", "pan", "because", "could",
               "couldn't", "cannot", "test", "tested"}

    def _tokens(text):
        return {
            w for w in re.findall(r"[a-z']+", (text or "").lower())
            if len(w) > 2 and w not in _NSTOP and w not in movie_tokens and w not in generic
        }

    q_tokens = _tokens(getattr(seg, "scene_query", "") or "")
    visual_tokens = _tokens(getattr(seg, "expected_visual", "") or "")
    entity_tokens = _tokens(getattr(seg, "required_entity", "") or "")
    # No scene/visual description means there is no principled source-affinity decision.  Preserve
    # the old verifier path instead of widening a generic character beat.
    if not (q_tokens or visual_tokens):
        return []

    def _matching_terms(wanted, present):
        return {w for w in wanted if any(
            t == w or (t.startswith(w) and len(t) - len(w) <= 2)
            or (w.startswith(t) and len(w) - len(t) <= 2)
            for t in present)}

    def _hits(wanted, present):
        return len(_matching_terms(wanted, present))

    def _episode_code(text):
        """Return a normalized (season, episode) pair from an explicit upload label."""
        value = str(text or "").lower()
        match = re.search(
            r"\bs(?:eason)?\s*0*(\d{1,2})\s*[:._ -]*"
            r"e(?:pisode)?\s*#?\s*0*(\d{1,2})\b",
            value)
        if not match:
            match = re.search(r"\b0*(\d{1,2})\s*x\s*0*(\d{1,2})\b", value)
        return ((int(match.group(1)), int(match.group(2))) if match else None)

    # An anchor's explicit episode is authoritative *source-location* evidence.  Literal title
    # overlap alone cannot distinguish the episode that depicts an event from a later recap or
    # confession about it (measured: the S04E02 necklace action lost all five reserve slots to an
    # Olenna compilation discussing the poison). Bind only when at least two storyboard terms
    # identify one configured anchor; one broad word such as "wedding" is intentionally insufficient.
    anchor_episode = None
    anchor_rank = None
    storyboard_tokens = q_tokens | visual_tokens
    for anchor_order, anchor in enumerate(ana.get("anchor_scenes") or []):
        if not isinstance(anchor, dict):
            continue
        episode = _episode_code(" ".join((
            str(anchor.get("episode", "") or ""),
            str(anchor.get("query", "") or ""),
        )))
        if episode is None:
            continue
        anchor_tokens = _tokens(" ".join((
            str(anchor.get("name", "") or ""),
            str(anchor.get("query", "") or ""),
        )))
        overlap = _hits(storyboard_tokens, anchor_tokens)
        if overlap < 2:
            continue
        rank = (overlap, -anchor_order)
        if anchor_rank is None or rank > anchor_rank:
            anchor_rank, anchor_episode = rank, episode

    # Transcript prose is useful for finding a *passage* inside a retained long source, but only
    # when it says something genuinely diagnostic about the storyboard.  These are common visual
    # directions, not scene identifiers.  Letting any pair of them open a distant region ("man
    # talks", "woman in room") would turn the strict local rung into another global search.
    timed_text_generic = {
        "appears", "close", "closeup", "dark", "face", "faces", "girl", "girls",
        "give", "gives", "giving", "gave", "hand", "hands", "holding", "holds",
        "inside", "light", "look", "looking", "looks", "man", "men", "outside",
        "people", "person", "room", "said", "says", "show", "showing", "shown",
        "sit", "sits", "someone", "something", "speak", "speaks", "stand", "stands",
        "take", "takes", "taking", "talk", "talking", "talks", "tell", "tells",
        "visible", "walk", "walking", "walks", "watch", "wear", "wearing", "wears",
        "woman", "women",
    }

    # Possessives and short inflections are one semantic anchor, not independent proof.  Without
    # this collapse, one ASR word ("betrayed") matched both storyboard targets ``betray`` and
    # ``betrayed`` and falsely satisfied the two-anchor contract.
    raw_timed_fields = tuple(
        {token[:-2] if token.endswith("'s") else token for token in field}
        for field in (q_tokens, visual_tokens, entity_tokens))
    raw_timed_targets = set().union(*raw_timed_fields)
    timed_aliases = {}
    ordered_targets = sorted(raw_timed_targets, key=lambda token: (len(token), token))
    for token in ordered_targets:
        alias = next((shorter for shorter in ordered_targets
                      if (shorter != token and len(shorter) >= 5
                          and token.startswith(shorter)
                          and len(token) - len(shorter) <= 3)), token)
        timed_aliases[token] = alias
    timed_fields = tuple({timed_aliases[token] for token in field}
                         for field in raw_timed_fields)
    timed_entity_tokens = timed_fields[2]
    timed_targets = set().union(*timed_fields) - timed_text_generic
    timed_field_count = {
        token: sum(1 for field in timed_fields if token in field)
        for token in timed_targets
    }

    def _timed_token_match(target: str, actual_tokens: list[str]) -> bool:
        """ASR-tolerant token match, including split proper names such as don+toes -> Dontos.

        Fuzz never decides a region by itself: `_timed_text_region` requires two distinct target
        tokens in one compact rolling passage, including at least one source-rare token.
        """
        target = re.sub(r"[^a-z0-9]", "", str(target or "").lower())
        if len(target) < 4:
            return False
        for pos, raw in enumerate(actual_tokens):
            actual = re.sub(r"[^a-z0-9]", "", str(raw or "").lower())
            if not actual:
                continue
            if actual == target:
                return True
            # Conservative morphology (poison/poisoned), then Whisper-style single-token garble.
            if (min(len(actual), len(target)) >= 5
                    and abs(len(actual) - len(target)) <= 3
                    and (actual.startswith(target) or target.startswith(actual))):
                return True
            try:
                if (actual[0] == target[0] and min(len(actual), len(target)) >= 5
                        and _index._tok_close(
                            actual, target, thresh=0.84)):
                    return True
            except Exception:
                pass
            # Proper names are often split into two phonetic words by ASR.  Do not concatenate
            # arbitrary long phrases: two/three short adjacent tokens is the measured failure.
            for width in (2, 3):
                if pos + width > len(actual_tokens):
                    continue
                compound = "".join(re.sub(r"[^a-z0-9]", "", str(piece).lower())
                                   for piece in actual_tokens[pos:pos + width])
                if len(target) < 5 or abs(len(compound) - len(target)) > 3:
                    continue
                try:
                    if (compound and compound[0] == target[0]
                            and _index._tok_close(compound, target, thresh=0.84)):
                        return True
                except Exception:
                    continue
        return False

    def _timed_text_region(shots, seeds):
        """Return one strong disjoint transcript passage, or ``None``.

        Per-shot transcripts retain timing via each shot's start/end.  A five-shot rolling window
        tolerates dialogue crossing edits while remaining local.  Source rarity prevents a common
        character name repeated throughout a compilation from carrying the decision.
        """
        if len(timed_targets) < 2 or not shots:
            return None
        rows = []
        for shot in shots:
            try:
                shot_index = int(getattr(shot, "index", -1))
            except (TypeError, ValueError):
                continue
            raw_tokens = re.findall(r"[a-z0-9']+", str(
                getattr(shot, "transcript", "") or "").lower())
            matched = {
                target for target in timed_targets
                if _timed_token_match(target, raw_tokens)
            }
            rows.append((shot_index, shot, matched))
        if not rows:
            return None
        # A token appearing throughout a source is contextual chatter, not a locator.  Five per
        # cent (with a two-shot floor for short clips) keeps genuine repeated proper names usable
        # while giving one-off object/name pairs the intended discriminative weight.
        rare_limit = max(2, (len(rows) + 19) // 20)
        doc_freq = {
            target: sum(1 for _index_v, _shot, matched in rows if target in matched)
            for target in timed_targets
        }
        rare_targets = {target for target, count in doc_freq.items()
                        if 0 < count <= rare_limit}
        if not rare_targets:
            return None

        best = None
        roll_radius = 2                  # at most five adjacent timed shots establish a passage
        reaction_tail = _exact_reaction_context_required(seg)
        # A silent reaction can resolve several edits after the line which locates its scene.  The
        # measured Tywin reaction is +4 edits / 17s after "trial by combat".  Widen only that
        # fail-closed reaction/inaction shape; the ordinary timed reserve stays byte-for-byte +/-2.
        reserve_radius = 4 if reaction_tail else 2
        for center in range(len(rows)):
            lo = max(0, center - roll_radius)
            hi = min(len(rows), center + roll_radius + 1)
            local = rows[lo:hi]
            matches = set().union(*(row[2] for row in local))
            rare_matches = matches & rare_targets
            if len(matches) < 2 or not rare_matches:
                continue
            non_entity_matches = matches - timed_entity_tokens
            required_kind = str(getattr(seg, "required_kind", "") or "").lower()
            # On event/character beats, a name plus one generic mood word is not a scene locator
            # (measured false anchors: Catelyn+"honor" for Ned's betrayal, Stark+"trust" in an
            # unrelated council). Require two scene terms beyond the named entity. Object beats may
            # use the stricter object+one-participant shape (Dontos + necklace).
            min_non_entity = 1 if "object" in required_kind else 2
            if len(non_entity_matches) < min_non_entity:
                continue
            # Map every matching token back to a timed shot.  Centre the reserve between the first
            # and last evidence-bearing edits rather than on an arbitrary rolling-window edge.
            evidence_positions = [
                pos for pos in range(lo, hi) if rows[pos][2] & matches
            ]
            if not evidence_positions:
                continue
            first_pos, last_pos = min(evidence_positions), max(evidence_positions)
            anchor_pos = (first_pos + last_pos) // 2
            anchor_index = rows[anchor_pos][0]
            # A transcript reserve exists only for a passage the ordinary +/-radius scan cannot
            # already reach.  Requiring the whole +/-2 reserve to be disjoint keeps the old local
            # candidate set byte-for-byte intact for nearby evidence.
            try:
                if min(abs(anchor_index - int(seed)) for seed in seeds) \
                        <= radius + reserve_radius:
                    continue
            except (TypeError, ValueError):
                continue
            try:
                span_start = float(rows[first_pos][1].start)
                span_end = float(rows[last_pos][1].end)
            except (TypeError, ValueError, AttributeError):
                continue
            if span_end - span_start > 45.0:
                continue                  # not one local utterance/passage
            field_coverage = sum(1 for field in timed_fields if matches & field)
            weighted = sum(
                1.0 + 0.35 * timed_field_count.get(token, 0)
                + (1.0 / max(1, doc_freq.get(token, 1)))
                for token in matches)
            # Leading tier 2 means "two-or-more explicit semantic transcript anchors".  It beats a
            # CLIP/object-only deep reserve (tier 0) and a weak per-shot transcript hint (tier 1),
            # but not an independently located quote/moment lock (tier 3).
            rarest_df = min(doc_freq[token] for token in rare_matches)
            # The rarest local identifier leads: Dontos spoken once is more diagnostic than three
            # broad words repeated across an essay's later poison recap. Count/coverage break ties.
            explicit_strength = (2, round(1.0 / rarest_df, 4), field_coverage,
                                 len(matches), round(weighted, 4))
            rank_key = (explicit_strength, -(last_pos - first_pos), -anchor_index)
            record = {
                "anchor": anchor_index,
                "reserve_radius": reserve_radius,
                "reaction_tail": reaction_tail,
                "matches": tuple(sorted(matches)),
                "rare_matches": tuple(sorted(rare_matches)),
                "score": round(weighted, 4),
                "strength": explicit_strength,
                "first_match_index": rows[first_pos][0],
                "last_match_index": rows[last_pos][0],
                "span_end": round(span_end, 3),
            }
            if best is None or rank_key > best[0]:
                best = (rank_key, record)
        return best[1] if best else None

    # The retained candidates are evidence about *where* match believed the scene could be.  Keep
    # their first-seen order as the final deterministic tiebreak, while collecting every seed shot
    # from a source so a sibling can be near any of them.
    head_seed_rows = [(getattr(sel, "source_id", ""), getattr(sel, "shot_index", -1))]
    head_seed_rows += [(getattr(c, "source_id", ""), getattr(c, "shot_index", -1))
                       for c in (getattr(sel, "alternates", None) or [])]
    deep_seed_rows = [(getattr(c, "source_id", ""), getattr(c, "shot_index", -1), c)
                      for c in (getattr(sel, "deep_alternates", None) or [])]
    source_seeds: dict[str, list[int]] = {}
    source_order: dict[str, int] = {}
    head_source_ids = {str(sid or "") for sid, _shot_index in head_seed_rows if sid}
    # A long exact-scene upload can contain two useful regions.  Match deliberately retains only a
    # few shots per source, so the normal bench may point at a contextual face while a later deep
    # candidate points at the actual action (measured: necklace 22 -> 1, dagger 29 -> 20, gold-cloak
    # betrayal 17 -> 61).  The old blanket same-source suppression below made those exact regions
    # unreachable.  Preserve at most ONE evidence-backed disjoint region per head source; candidate
    # generation later gives that region its own tiny, hard-bounded nearest-shot reserve instead of
    # letting it widen or crowd the ordinary neighborhood.
    # value = (existing deterministic rank evidence, anchor shot, explicit-text strength)
    supported_deep_regions: dict[str, tuple[tuple, int, tuple]] = {}

    def _deep_region_evidence(candidate, order: int):
        signals = getattr(candidate, "signals", None) or {}

        def _num(key):
            try:
                return float(signals.get(key, 0.0) or 0.0)
            except (TypeError, ValueError):
                return 0.0

        dialogue_evidence = _num("dialogue")
        transcript_evidence = _num("transcript")
        moment_evidence = _num("moment_lock")
        text_evidence = max(dialogue_evidence, transcript_evidence, moment_evidence)
        clip_evidence = _num("clip")
        try:
            retained_score = float(getattr(candidate, "score", 0.0) or 0.0)
        except (TypeError, ValueError):
            retained_score = 0.0
        # A bare late bench sibling is not permission to scan another part of a long compilation.
        # It must carry candidate-level retrieval evidence and a solid retained score.  The source
        # still has to pass the unchanged scene-affinity test below, and strict vision remains the
        # acceptance authority.
        supported = retained_score >= 0.65 and (
            text_evidence > 0.0 or clip_evidence >= 0.65
            or _num("object") > 0.0 or _num("faceid") > 0.0)
        if not supported:
            return None
        # Timed/text evidence disambiguates two distant regions in the same upload (the measured
        # throne-room source retained an early Ned shot and the later betrayal; only the latter had
        # transcript overlap).  CLIP and the retained score are deterministic fallbacks.
        evidence = (1 if text_evidence > 0.0 else 0, text_evidence,
                    clip_evidence, retained_score, -int(order))
        if moment_evidence > 0.0 or dialogue_evidence >= 0.78:
            explicit_strength = (3, round(max(moment_evidence, dialogue_evidence), 4))
        elif text_evidence > 0.0:
            explicit_strength = (1, round(text_evidence, 4))
        else:
            explicit_strength = (0, 0.0)
        return evidence, explicit_strength

    def _add_seed(order, sid, shot_index, *, deep=False):
        sid = str(sid or "")
        try:
            shot_index = int(shot_index)
        except (TypeError, ValueError):
            return
        if not sid or shot_index < 0:
            return
        # A normal alternate is a stronger location hint than a deep-bench sibling.  Once a source
        # has a primary/alternate seed, distant deep seeds from the same long upload must not widen
        # its neighborhood (the measured trial source had head seed 35 but deep seeds 19/23; using
        # all three crowded the actual +6 kneeling shot out with unrelated earlier material).
        if deep and sid in source_seeds:
            return
        source_order.setdefault(sid, order)
        if shot_index not in source_seeds.setdefault(sid, []):
            source_seeds[sid].append(shot_index)

    for order, (sid, shot_index) in enumerate(head_seed_rows):
        _add_seed(order, sid, shot_index)
    _deep_base = len(head_seed_rows)
    for offset, (sid, shot_index, candidate) in enumerate(deep_seed_rows):
        # For a deep-only source, its first retained shot is the bounded anchor. Later sibling
        # entries from the same source are ranking evidence, not permission to scan many regions.
        sid = str(sid or "")
        if sid in source_seeds:
            try:
                shot_index = int(shot_index)
            except (TypeError, ValueError):
                continue
            if shot_index < 0 or min(abs(shot_index - seed)
                                     for seed in source_seeds[sid]) <= radius:
                continue
            # A second deep-only sibling is ranking evidence, not permission to open another
            # distant region. Only a source already retained by the primary/normal bench may use
            # the same-source blind-spot repair.
            if sid not in head_source_ids:
                continue
            evidence_row = _deep_region_evidence(candidate, _deep_base + offset)
            previous = supported_deep_regions.get(sid)
            if evidence_row is not None:
                evidence, explicit_strength = evidence_row
                if previous is None or evidence > previous[0]:
                    supported_deep_regions[sid] = (
                        evidence, shot_index, explicit_strength)
            continue
        _add_seed(_deep_base + offset, sid, shot_index, deep=True)

    try:
        from .match import banned_source_ids
        banned = banned_source_ids(proj, include_auto=True)
    except Exception:
        banned = set()

    # SCOPED INDEXED-POOL SOURCE LANE.  A global rematch can retain the right file only below the
    # four-source strict-neighborhood cut (beat 51), or omit it from the deep bench entirely despite
    # an exact acquisition query (beat 3).  Do not re-rank the project or widen normal verification.
    # Once publication recovery explicitly opts in, admit only the two strongest metadata-bound,
    # natively publishable sources as *seeds* inside the existing source/candidate caps.  A required
    # entity alone is insufficient: one additional scene/visual term must independently match.
    strict_pool_sources: dict[str, dict] = {}
    if allow_indexed_pool_sources:
        try:
            from .quality_contract import native_video_ok as _native_pool_ok
            from .quality_contract import probe_native_video_info as _probe_native_pool

            ranked_pool_sources = []
            seen_pool_sources = set()
            storyboard_tokens = q_tokens | visual_tokens
            for source_position, src in enumerate(getattr(proj, "sources", None) or []):
                sid = str(getattr(src, "id", "") or "")
                if (not sid or sid in seen_pool_sources or sid in banned
                        or str(getattr(src, "status", "") or "") != "ok"):
                    continue
                seen_pool_sources.add(sid)
                title = str(getattr(src, "title", "") or "")
                extra = getattr(src, "extra", None) or {}
                source_text = " ".join((
                    title, sid.replace("_", " "), str(extra.get("query", "") or "")))
                if beat_era and _era_conflict(beat_era, source_text):
                    continue
                try:
                    if not _native_pool_ok(_probe_native_pool(
                            str(getattr(src, "local_path", "") or ""))):
                        continue
                except Exception:
                    continue                         # unknown bytes never gain a recovery seed
                source_tokens = _tokens(source_text)
                semantic_matches = _matching_terms(storyboard_tokens, source_tokens)
                non_entity_matches = semantic_matches - entity_tokens
                if len(semantic_matches) < 2 or not non_entity_matches:
                    continue
                title_matches = _matching_terms(storyboard_tokens, _tokens(title))
                query_matches = _matching_terms(
                    storyboard_tokens, _tokens(str(extra.get("query", "") or "")))
                try:
                    discovery_relevance = float(extra.get("relevance", 0.0) or 0.0)
                except (TypeError, ValueError):
                    discovery_relevance = 0.0
                # The persisted acquisition query records why this file entered the project and
                # is less vulnerable than a catchy title to incidental prose (measured: "All
                # because I couldn't..." matched a beat containing "could not be tested").
                rank = (
                    len(semantic_matches), len(non_entity_matches), len(query_matches),
                    len(title_matches), round(discovery_relevance, 4), -source_position)
                ranked_pool_sources.append((rank, sid))

            ranked_pool_sources.sort(key=lambda row: row[0], reverse=True)
            # Resolve every prospective seed before mutating the retained-source map.  If source
            # inspection unexpectedly fails halfway through, the opt-in lane must leave the old
            # pool byte-for-byte intact rather than silently retaining a partial first mutation.
            pending_pool_sources = {}
            pending_new_seeds = []
            for pool_rank, (_rank, sid) in enumerate(ranked_pool_sources[:2]):
                if sid in source_seeds:
                    pending_pool_sources[sid] = {
                        "match_count": int(_rank[0]),
                        "branch": "reprioritized_retained",
                    }
                    continue
                try:
                    pool_shots = sorted(
                        getattr(get_shot, "all_shots", lambda _s: [])(sid) or [],
                        key=lambda shot: int(getattr(shot, "index", -1)))
                    seed_index = next(
                        int(getattr(shot, "index", -1)) for shot in pool_shots
                        if int(getattr(shot, "index", -1)) >= 0)
                except (StopIteration, TypeError, ValueError):
                    continue
                pending_pool_sources[sid] = {
                    "match_count": int(_rank[0]),
                    "branch": "admitted_new_source",
                }
                pending_new_seeds.append((
                    _deep_base + len(deep_seed_rows) + pool_rank, sid, seed_index))
            for seed_order, sid, seed_index in pending_new_seeds:
                _add_seed(seed_order, sid, seed_index, deep=True)
            strict_pool_sources = pending_pool_sources
        except Exception:
            # This is retrieval opportunity, never approval.  Any metadata/probe failure preserves
            # the old retained-source pool and lets the unchanged strict publication gate block.
            strict_pool_sources = {}

    affine_sources = []
    anchor_episode_sources = set()
    for sid, seeds in source_seeds.items():
        try:
            src = proj.source(sid)
        except Exception:
            src = None
        if src is None or str(getattr(src, "status", "") or "") != "ok" or sid in banned:
            continue
        title = str(getattr(src, "title", "") or "")
        extra = getattr(src, "extra", None) or {}
        source_text = " ".join((title, sid.replace("_", " "), str(extra.get("query", "") or "")))
        source_tokens = _tokens(source_text)
        qh = _hits(q_tokens, source_tokens)
        vh = _hits(visual_tokens, source_tokens)
        eh = _hits(entity_tokens, source_tokens)
        anchored = bool(extra.get("anchor_verified"))
        # Entity agreement improves ordering but cannot qualify a broad character compilation by
        # itself.  At least one scene/visual token (or explicit anchor proof) is mandatory.
        if not anchored and qh + vh <= 0:
            continue
        if beat_era and _era_conflict(beat_era, title + " " + sid):
            continue
        source_episode = _episode_code(source_text)
        anchor_episode_match = bool(
            anchor_episode is not None and source_episode == anchor_episode)
        if anchor_episode_match:
            anchor_episode_sources.add(sid)
        # Exact episode identity is stronger than noisy title prose, while explicit
        # ``anchor_verified`` remains the strongest source contract. This changes only the bounded
        # strict-repair ordering; it neither widens the retained source set nor bypasses vision/QC.
        affinity = ((100 if anchored else 0) + (50 if anchor_episode_match else 0)
                    + 5 * qh + 3 * vh + 2 * eh)
        # Source affinity is primary, with a small penalty for how far down the already-ranked seed
        # list this source appeared. This keeps strong literal sources ahead of generic ones while
        # preserving the useful matcher fact that an early alternate is more plausible than a late
        # deep-bench compilation. It puts the measured attack/trial/execution sources in the top 2.
        priority = affinity - source_order[sid]
        # Normal alternates are the matcher's stronger source evidence. Deep-only sources remain
        # available when the head is sparse, but a late character compilation cannot displace the
        # exact-scene source already present in the normal alternate list merely on noisy CLIP/title
        # overlap.
        seed_tier = (-1 if sid in strict_pool_sources else
                     (0 if sid in head_source_ids else 1))
        affine_sources.append((seed_tier, -priority, source_order[sid], sid, seeds, affinity))
    affine_sources.sort(key=lambda row: (row[0], row[1], row[2], row[3]))

    # Scan transcript text only in sources the matcher already retained and the affinity gate just
    # accepted.  This may inspect a deep-bench source that sits below the four-source vision cap;
    # explicit local text can replace (never add to) the last source slot.  No image/API call is
    # spent by this scan and the global matcher remains untouched.
    all_affine_sources = list(affine_sources)
    shots_cache: dict[str, list] = {}
    timed_regions: dict[str, dict] = {}
    for _row in all_affine_sources:
        _sid, _seeds = _row[3], _row[4]
        try:
            _shots = sorted(getattr(get_shot, "all_shots", lambda _s: [])(_sid) or [],
                            key=lambda sh: int(getattr(sh, "index", -1)))
        except Exception:
            _shots = []
        shots_cache[_sid] = _shots
        _timed = _timed_text_region(_shots, _seeds)
        if _timed:
            # Keep this evidence even when it overlaps the old deep anchor.  It reuses the SAME
            # five-call reserve, but carries stronger ordering and a post-dialogue action trim; the
            # measured throne-room miss had deep#61 and timed#62 pointing at the same passage.
            timed_regions[_sid] = _timed

    affine_sources = all_affine_sources[:source_cap]
    protected_timed_sid = ""
    if timed_regions:
        _timed_rows = [row for row in all_affine_sources if row[3] in timed_regions]
        _best_timed_row = max(
            _timed_rows,
            key=lambda row: (timed_regions[row[3]]["strength"], int(row[5]),
                             -int(row[2]), row[3]))
        _best_timed_sid = _best_timed_row[3]
        _deep_strength = max(
            (supported_deep_regions[row[3]][2]
             for row in affine_sources if row[3] in supported_deep_regions),
            default=(0, 0.0))
        # A deep source below the normal source cap enters only when two-or-more explicit timed
        # anchors are stronger than the best old deep evidence.  Replace the last slot so the
        # unchanged source_cap remains a hard bound.
        if timed_regions[_best_timed_sid]["strength"] > _deep_strength:
            protected_timed_sid = _best_timed_sid
            if all(row[3] != _best_timed_sid for row in affine_sources):
                if len(affine_sources) >= source_cap:
                    affine_sources = affine_sources[:-1]
                affine_sources.append(_best_timed_row)
    if not affine_sources:
        return []

    excluded = set(exclude or ())
    # The selected shot already received a strict primary verdict even when the caller did not pass
    # an exclusion set.  Never pay to ask the same pixels the same question again.
    try:
        selected_shot_index = int(getattr(sel, "shot_index", -1))
    except (TypeError, ValueError):
        selected_shot_index = -1
    excluded.add((str(getattr(sel, "source_id", "") or ""), selected_shot_index))

    # The shallow normal promotion may stop at max_replacements=3 even though match already ranked
    # an exact candidate fifth.  Keep a score-ordered list of at most one reserve opportunity here;
    # hard-shot eligibility is checked below before its source can consume one of source_cap's slots.
    affine_by_sid = {row[3]: row for row in all_affine_sources}
    unseen_retained_rows = []
    for alt_order, alt in enumerate(getattr(sel, "alternates", None) or []):
        sid = str(getattr(alt, "source_id", "") or "")
        try:
            shot_index = int(getattr(alt, "shot_index", -1))
            retained_score = float(getattr(alt, "score", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        if (sid, shot_index) in excluded or not sid or shot_index < 0:
            continue
        source_row = affine_by_sid.get(sid)
        if source_row is None:
            # Candidate-level semantic affinity is allowed only for this ONE already-ranked frame,
            # never for a +/- neighborhood.  This covers an exact frame inside a poorly titled
            # compilation (measured beat98) without declaring the whole source scene-affine.
            signals = getattr(alt, "signals", None) or {}
            try:
                clip_affinity = float(signals.get("clip", 0.0) or 0.0)
            except (TypeError, ValueError):
                clip_affinity = 0.0
            try:
                src = proj.source(sid)
            except Exception:
                src = None
            title = str(getattr(src, "title", "") or "") if src is not None else ""
            if (retained_score < 0.70 or clip_affinity < 0.90 or src is None
                    or str(getattr(src, "status", "") or "") != "ok" or sid in banned
                    or (beat_era and _era_conflict(beat_era, title + " " + sid))):
                continue
        unseen_retained_rows.append(
            ((-retained_score, alt_order, sid, shot_index), alt, source_row))
    unseen_retained_rows.sort(key=lambda row: row[0])

    from . import image_fallback as _IF_n
    from . import match as _M_n
    import os as _os_n
    try:
        _black_floor = float(_os_n.environ.get(
            "VIDLORE_CLIPSTUDIO_BLACK_FLOOR", "0.10") or 0.10)
    except (TypeError, ValueError):
        _black_floor = 0.10
    embed_cache: dict = {}
    raw_embed_cache: dict = {}
    graphics_tier_cache: dict = {}
    rel_memo: dict = {}

    def _embeds_of(sid):
        if sid not in embed_cache:
            try:
                embed_cache[sid] = _index.load_embeds_verified(proj, sid)
            except Exception:
                embed_cache[sid] = (None, None)
        return embed_cache[sid]

    def _graphics_tiers(sid, shots):
        """Mirror match's per-source graphics context for these sibling shots.

        A tier-1 frame is excluded only when the source also carries at least three hard graphics
        frames.  Looking at one sibling in isolation would miss that source-level arm of
        ``_load_pool`` and let a shot which match deliberately removed reach strict promotion.
        """
        if sid in graphics_tier_cache:
            return graphics_tier_cache[sid]
        try:
            mat = raw_embed_cache.setdefault(sid, _index.load_embeds(proj, sid))
        except Exception:
            mat = None
            raw_embed_cache[sid] = None
        tiers = {}
        for pos, candidate_shot in enumerate(shots):
            vec = None
            try:
                row = getattr(candidate_shot, "embed_row", -1)
                row = -1 if row is None else int(row)
                if mat is not None and 0 <= row < len(mat):
                    vec = mat[row]
            except Exception:
                vec = None
            try:
                tiers[int(getattr(candidate_shot, "index", pos))] = \
                    _M_n._shot_graphics_tier(candidate_shot, vec)
            except Exception:
                tiers[int(getattr(candidate_shot, "index", pos))] = -1
        graphics_tier_cache[sid] = (tiers, sum(1 for value in tiers.values() if value >= 2))
        return graphics_tier_cache[sid]

    from .match import (_shot_unreadable, _shot_featureless, _ocr_is_junk,
                        _ocr_text_heavy, _shot_overlay_badge,
                        _shot_static_collage, _shot_numeral_overlay,
                        _shot_subtitle_band)

    def _hard_shot_eligible(sid, shot, shots):
        """Mirror match's deterministic hard-shot admission before any reserve is allocated."""
        try:
            shot_index = int(getattr(shot, "index", -1))
            kf = str(getattr(shot, "keyframe_path", "") or "")
            if not kf or not Path(kf).is_file():
                return False
            tiers, hard_graphics = _graphics_tiers(sid, shots)
            tier = int(tiers.get(shot_index, -1))
            quality_raw = getattr(shot, "quality", None)
            quality = 1.0 if quality_raw is None else float(quality_raw)
            return not (
                _shot_unreadable(shot) or _shot_featureless(shot)
                or _ocr_is_junk(shot) or _ocr_text_heavy(shot)
                or _shot_overlay_badge(shot) or _shot_static_collage(shot)
                or _shot_numeral_overlay(shot) or _shot_subtitle_band(shot)
                or tier >= 2 or (tier == 1 and hard_graphics >= 3)
                or int((getattr(shot, "scores", None) or {}).get("bonus_tail", 0) or 0)
                or (_black_floor > 0 and quality < _black_floor))
        except Exception:
            return False

    # Reserve at most ONE normal alternate that the shallow promotion did not reach.  Its source
    # replaces the weakest unprotected source slot when necessary; explicit timed-text evidence is
    # never displaced.  Strict vision, Window-QC, reuse and materialization remain downstream.
    retained_alt_choice = None
    retained_direct_only_sid = ""
    for _rank, alt, source_row in unseen_retained_rows:
        sid = str(getattr(alt, "source_id", "") or "")
        try:
            shot_index = int(getattr(alt, "shot_index", -1))
        except (TypeError, ValueError):
            continue
        shots = shots_cache.get(sid)
        if shots is None:
            try:
                shots = sorted(getattr(get_shot, "all_shots", lambda _s: [])(sid) or [],
                               key=lambda sh: int(getattr(sh, "index", -1)))
            except Exception:
                shots = []
            shots_cache[sid] = shots
        shot = next((candidate_shot for candidate_shot in shots
                     if int(getattr(candidate_shot, "index", -1)) == shot_index), None)
        if shot is None or not _hard_shot_eligible(sid, shot, shots):
            continue
        if all(row[3] != sid for row in affine_sources):
            direct_only = source_row is None
            if source_row is None:
                # Strong persisted candidate-level CLIP admits only this exact matcher-ranked shot,
                # not its surrounding source.  A synthetic one-seed row lets the shared hard/vision
                # path process it while `retained_direct_only_sid` blocks every sibling.
                source_row = (0, 0, int(source_order.get(sid, 10_000)), sid,
                              [shot_index], 0)
            if len(affine_sources) < source_cap:
                affine_sources.append(source_row)
            else:
                replace_index = next(
                    (pos for pos in range(len(affine_sources) - 1, -1, -1)
                     if affine_sources[pos][3] != protected_timed_sid), None)
                if replace_index is None:
                    continue
                affine_sources[replace_index] = source_row
            if direct_only:
                retained_direct_only_sid = sid
        retained_alt_choice = alt
        break
    retained_alt_key = ((str(getattr(retained_alt_choice, "source_id", "") or ""),
                         int(getattr(retained_alt_choice, "shot_index", -1)))
                        if retained_alt_choice is not None else None)

    query = " ".join(x for x in (
        getattr(seg, "scene_query", "") or "",
        getattr(seg, "expected_visual", "") or "",
        getattr(seg, "required_entity", "") or "",
    ) if x)
    moment_cache: dict = {}
    ranked = []
    deep_region_ranked: dict[int, list] = {}
    timed_region_ranked: dict[int, list] = {}
    retained_normal_ranked = []

    # SCOPED WHOLE-SOURCE ACTION PROBE.  A silent visual climax can sit many edits away from every
    # retained matcher seed even though the correct full-scene upload is already indexed.  Timed
    # transcript reserves cannot locate such an action, and widening the ordinary +/-radius would
    # spend calls throughout long compilations.  The publication-recovery lane may therefore make
    # one final five-shot probe inside the strongest native-HD source whose explicit episode agrees
    # with the beat's anchor.  Candidate regions must be disjoint from every normal, supported-deep,
    # and timed-text seed; this is new evidence, not a second look at the local rung.  Local CLIP
    # ranks only the five questions.  It never approves a frame: the caller still applies the same
    # strict vision, Window-QC, reuse and materialization transaction.
    if whole_source_probe:
        if anchor_episode is None:
            return []
        try:
            from .quality_contract import native_video_ok as _native_probe_ok
            from .quality_contract import probe_native_video_info as _probe_native_source
        except Exception:
            return []

        probe_source_rows = []
        for source_row in all_affine_sources:
            sid = source_row[3]
            if sid not in anchor_episode_sources:
                continue
            try:
                source_obj = proj.source(sid)
                if source_obj is None or not _native_probe_ok(_probe_native_source(
                        str(getattr(source_obj, "local_path", "") or ""))):
                    continue
            except Exception:
                continue
            probe_source_rows.append(source_row)
        if not probe_source_rows:
            return []

        # The selected source is direct matcher evidence and is preferable when its literal anchor
        # affinity is competitive.  This avoids letting a high-CLIP trailer/official excerpt whose
        # title names the episode displace the already-selected full-scene upload (the measured
        # excerpt cut to an end card immediately before the fatal action).  One scene-token weight
        # is the same competitiveness margin used by the ordinary deep-region chooser above.
        strongest_probe_row = max(
            probe_source_rows, key=lambda row: (int(row[5]), -int(row[2]), row[3]))
        selected_sid = str(getattr(sel, "source_id", "") or "")
        selected_probe_row = next(
            (row for row in probe_source_rows if row[3] == selected_sid), None)
        probe_source_row = (
            selected_probe_row
            if (selected_probe_row is not None
                and int(selected_probe_row[5]) >= int(strongest_probe_row[5]) - 5)
            else strongest_probe_row)

        _seed_tier, _neg_aff, _seed_order, sid, seeds, affinity = probe_source_row
        source_shots = shots_cache.get(sid, [])
        if not source_shots:
            return []
        disjoint_anchors = [int(seed) for seed in seeds]
        supported_region = supported_deep_regions.get(sid)
        if supported_region is not None:
            try:
                disjoint_anchors.append(int(supported_region[1]))
            except (TypeError, ValueError):
                pass
        timed_region = timed_regions.get(sid)
        if timed_region is not None:
            try:
                disjoint_anchors.append(int(timed_region["anchor"]))
            except (KeyError, TypeError, ValueError):
                pass
        if not disjoint_anchors:
            return []

        probe_rows = []
        for sh in source_shots:
            try:
                shot_index = int(getattr(sh, "index", -1))
            except (TypeError, ValueError):
                continue
            key = (sid, shot_index)
            if (shot_index < 0 or key in excluded
                    or min(abs(shot_index - anchor) for anchor in disjoint_anchors) <= radius
                    or not _hard_shot_eligible(sid, sh, source_shots)):
                continue
            kf = str(getattr(sh, "keyframe_path", "") or "")
            try:
                rel = _IF_n._shot_relevance(
                    sh, Path(kf), query, embeds_of=_embeds_of, rel_memo=rel_memo)
                rel = float(rel)
            except Exception:
                continue
            if rel < 0.0:
                continue
            if sid not in moment_cache:
                try:
                    moment_cache[sid] = _M_n.locate_beat_moment(proj, sid, seg)
                except Exception:
                    moment_cache[sid] = None
            moment = moment_cache.get(sid)
            try:
                in_point, out_point = _M_n._trim_window(sh, seg, cfg, moment)
            except Exception:
                in_point, out_point = float(sh.start), float(sh.end)
            try:
                dialogue = float(_M_n._dialogue_match(
                    seg, getattr(sh, "transcript", "") or "",
                    quote_branch=_M_n._effective_matcher_quote_branch(seg, proj=proj)))
            except Exception:
                dialogue = 0.0
            try:
                moment_lock = float(_M_n._moment_proximity(sh, moment)) if moment else 0.0
            except Exception:
                moment_lock = 0.0
            dialogue = max(dialogue, moment_lock)
            signals = {
                "strict_scene_neighborhood": True,
                "strict_whole_source_probe": True,
                "scene_affinity": int(affinity),
                "visual_relevance": round(rel, 4),
                "dialogue": round(dialogue, 3),
                "quality": round(float(getattr(sh, "quality", 0.0) or 0.0), 3),
            }
            if sid in strict_pool_sources:
                signals["strict_indexed_pool_source"] = True
                branch = str(strict_pool_sources[sid]["branch"])
                signals[f"strict_indexed_pool_{branch}"] = 1.0
                signals["source_metadata_match_count"] = int(
                    strict_pool_sources[sid]["match_count"])
            if moment and moment_lock > 0.0:
                signals["moment_lock"] = round(moment_lock, 3)
                try:
                    signals["moment_ratio"] = round(float(moment[2]), 3)
                except Exception:
                    pass
            candidate = ClipCandidate(
                segment_index=int(getattr(sel, "segment_index", -1)), source_id=sid,
                shot_index=shot_index, score=round(max(0.0, min(1.0, rel)), 4),
                in_point=round(float(in_point), 3), out_point=round(float(out_point), 3),
                signals=signals)
            probe_rows.append(((-rel, shot_index), candidate))
        probe_rows.sort(key=lambda row: row[0])
        return [candidate for _rank, candidate in probe_rows[:cap]]

    for source_rank, (_seed_tier, _neg_aff, _seed_order, sid, seeds, affinity) \
            in enumerate(affine_sources):
        shots = shots_cache.get(sid, [])
        if not shots:
            continue
        _deep_anchor = (supported_deep_regions.get(sid) or (None, None, None))[1]
        _timed_info = timed_regions.get(sid)
        _timed_anchor = int(_timed_info["anchor"]) if _timed_info else None
        _timed_radius = int(_timed_info["reserve_radius"]) if _timed_info else 0
        for sh in shots:
            try:
                shot_index = int(getattr(sh, "index", -1))
                distance = min(abs(shot_index - seed) for seed in seeds)
            except (TypeError, ValueError):
                continue
            if retained_direct_only_sid == sid and (sid, shot_index) != retained_alt_key:
                continue
            deep_distance = (abs(shot_index - int(_deep_anchor))
                             if _deep_anchor is not None else radius + 1)
            timed_distance = (abs(shot_index - _timed_anchor)
                              if _timed_anchor is not None else radius + 1)
            # A supported deep region owns only pixels which the ordinary head neighborhood could
            # not already reach.  This prevents double counting when its anchor is just beyond the
            # radius and keeps the old head-region search byte-for-byte unchanged.
            in_deep_region = deep_distance <= radius and distance > radius
            _ordinary_timed = bool(_timed_info and timed_distance <= 2)
            _reaction_tail = False
            if _timed_info and _timed_info.get("reaction_tail"):
                try:
                    _reaction_tail = bool(
                        shot_index > int(_timed_info["last_match_index"])
                        and shot_index - int(_timed_info["anchor"]) <= _timed_radius
                        and float(getattr(sh, "start", 0.0) or 0.0)
                        - float(_timed_info["span_end"])
                        <= _EXACT_REACTION_CONTEXT_MAX_SPAN_SEC)
                except (KeyError, TypeError, ValueError):
                    _reaction_tail = False
            in_timed_region = bool(
                _timed_info and (_ordinary_timed or _reaction_tail) and distance > radius)
            key = (sid, shot_index)
            if (key in excluded
                    or (distance > radius and not in_deep_region and not in_timed_region)):
                continue
            kf = str(getattr(sh, "keyframe_path", "") or "")
            if not kf or not Path(kf).is_file():
                continue
            # Neighbors were not necessarily retained by match, so re-apply the complete
            # deterministic hard-shot predicate from `_load_pool` + `_score_pool` before they can
            # consume a vision call.  Strict vision is not permission to resurrect pixels which
            # match already ruled unairable (featureless cards, sub bands, hard graphics, etc.).
            if not _hard_shot_eligible(sid, sh, shots):
                continue
            try:
                rel = _IF_n._shot_relevance(
                    sh, Path(kf), query, embeds_of=_embeds_of, rel_memo=rel_memo)
            except Exception:
                rel = -1.0
            if sid not in moment_cache:
                try:
                    moment_cache[sid] = _M_n.locate_beat_moment(proj, sid, seg)
                except Exception:
                    moment_cache[sid] = None
            _moment = moment_cache.get(sid)
            try:
                in_point, out_point = _M_n._trim_window(
                    sh, seg, cfg, _moment)
            except Exception:
                in_point, out_point = float(sh.start), float(sh.end)
            # Promotion replaces the selection's signals wholesale.  Reconstruct the same dialogue
            # / moment evidence match would have attached; otherwise a genuinely quote-containing
            # neighborhood rescue passes strict vision and then deterministically fails the final
            # verbatim contract merely because this new rung erased its proof.
            try:
                dialogue = float(_M_n._dialogue_match(
                    seg, getattr(sh, "transcript", "") or "",
                    quote_branch=_M_n._effective_matcher_quote_branch(seg, proj=proj)))
            except Exception:
                dialogue = 0.0
            try:
                moment_lock = float(_M_n._moment_proximity(sh, _moment)) if _moment else 0.0
            except Exception:
                moment_lock = 0.0
            dialogue = max(dialogue, moment_lock)
            score = max(0.0, min(1.0, float(rel))) if rel is not None and rel >= 0 else 0.0
            signals = {"strict_scene_neighborhood": True,
                       "scene_affinity": int(affinity),
                       "neighbor_distance": int(distance),
                       "visual_relevance": round(float(rel), 4) if rel is not None else -1.0,
                       "dialogue": round(dialogue, 3),
                       "quality": round(float(getattr(sh, "quality", 0.0) or 0.0), 3)}
            if sid in strict_pool_sources:
                signals["strict_indexed_pool_source"] = True
                branch = str(strict_pool_sources[sid]["branch"])
                signals[f"strict_indexed_pool_{branch}"] = 1.0
                signals["source_metadata_match_count"] = int(
                    strict_pool_sources[sid]["match_count"])
            if _moment and moment_lock > 0.0:
                signals["moment_lock"] = round(moment_lock, 3)
                try:
                    signals["moment_ratio"] = round(float(_moment[2]), 3)
                except Exception:
                    pass
            def _candidate(_in, _out, _signals):
                return ClipCandidate(
                    segment_index=int(getattr(sel, "segment_index", -1)), source_id=sid,
                    shot_index=shot_index, score=round(score, 4),
                    in_point=round(float(_in), 3), out_point=round(float(_out), 3),
                    signals=_signals)
            is_retained_reserve = retained_alt_key == (sid, shot_index)
            if is_retained_reserve:
                retained_signals = dict(
                    getattr(retained_alt_choice, "signals", None) or {})
                retained_signals.update(signals)
                retained_signals["strict_scene_retained_alternate"] = True
                try:
                    retained_in = float(getattr(retained_alt_choice, "in_point"))
                    retained_out = float(getattr(retained_alt_choice, "out_point"))
                    if retained_out <= retained_in:
                        raise ValueError("empty retained window")
                except (TypeError, ValueError):
                    retained_in, retained_out = float(in_point), float(out_point)
                retained_normal_ranked.append(
                    ((-float(getattr(retained_alt_choice, "score", 0.0) or 0.0),
                      source_rank, shot_index),
                     _candidate(retained_in, retained_out, retained_signals)))
            # Real persisted/live CLIP scores lead. If CLIP is unavailable, source affinity then
            # nearest-shot distance provides a fully deterministic fallback.
            order_key = ((0, -float(rel), distance, shot_index)
                         if rel is not None and rel >= 0 else
                         (1, 0.0, distance, shot_index))
            if in_deep_region:
                # Distance leads only inside the five-call reserve.  A low-CLIP transition/action
                # sibling two shots from a strong deep seed must still be seen by strict vision;
                # this is exactly how the gold-cloak swords frame survived noisy CLIP ordering.
                deep_signals = dict(signals)
                deep_signals["neighbor_distance"] = int(deep_distance)
                deep_signals["strict_scene_deep_region"] = True
                deep_region_ranked.setdefault(source_rank, []).append(
                    ((deep_distance, order_key, shot_index),
                     _candidate(in_point, out_point, deep_signals)))
            if in_timed_region:
                timed_signals = dict(signals)
                timed_signals.update({
                    "neighbor_distance": int(timed_distance),
                    "strict_scene_timed_text_region": True,
                    # ClipCandidate.signals is a numeric ledger contract.  The matched token
                    # names are only temporary ranking diagnostics; persisting their lists makes
                    # the final QC ledger crash when it rounds every signal.  Counts retain the
                    # strength evidence without allowing structured data into that contract.
                    "timed_text_match_count": len(_timed_info["matches"]),
                    "timed_text_rare_match_count": len(_timed_info["rare_matches"]),
                    "timed_text_score": float(_timed_info["score"]),
                })
                timed_in, timed_out = float(in_point), float(out_point)
                # Dialogue often names an action immediately before the edit that shows it.  For a
                # long, silent action shot after the first matched line, the ordinary midpoint trim
                # can omit the resolving tail (measured: gold-cloak swords, then Ned enters frame).
                # Bias only this explicitly timed action candidate to its tail; all other windows
                # retain `_trim_window` unchanged and downstream exact-window QC still adjudicates.
                try:
                    action_intent = str(getattr(seg, "shot_intent", "") or "").lower() == "action"
                    shot_start, shot_end = float(sh.start), float(sh.end)
                    window_len = max(0.0, timed_out - timed_in)
                    post_dialogue = (
                        shot_index > int(_timed_info["first_match_index"])
                        and not str(getattr(sh, "transcript", "") or "").strip())
                    if (action_intent and post_dialogue and window_len > 0.0
                            and shot_end - shot_start > window_len + 0.5):
                        timed_out = shot_end
                        timed_in = max(shot_start, timed_out - window_len)
                        timed_signals["timed_text_tail_window"] = True
                except (TypeError, ValueError, AttributeError):
                    pass
                if _timed_info.get("reaction_tail"):
                    # Under the unchanged five-call reserve, see the four post-dialogue edits
                    # before the line itself; those pixels are where a silent reaction can exist.
                    timed_rank = (
                        0 if shot_index > int(_timed_info["last_match_index"]) else 1,
                        timed_distance, order_key, shot_index)
                else:
                    timed_rank = (timed_distance, order_key, shot_index)
                timed_region_ranked.setdefault(source_rank, []).append(
                    (timed_rank, _candidate(timed_in, timed_out, timed_signals)))
            if not in_deep_region and not in_timed_region and not is_retained_reserve:
                ranked.append((source_rank, order_key,
                               _candidate(in_point, out_point, signals)))

    # Source identity is stronger than cross-source CLIP noise for this rung: the whole reason it
    # exists is that match already found the right file but retained the neighboring second. Rank by
    # persisted CLIP *within* each top affine source, and reserve six slots apiece for the top two
    # sources (default cap 12). A global CLIP sort let high-scoring wrong-source faces crowd the
    # measured +6 kneeling shot out of the pool before strict vision could judge it.
    by_source: dict[int, list] = {}
    for source_rank, order_key, cand in ranked:
        by_source.setdefault(source_rank, []).append((order_key, cand))
    for rows in by_source.values():
        rows.sort(key=lambda row: row[0])
    chosen = []
    chosen_keys = set()
    # First resolve the old evidence-backed deep reserve exactly as before: prefer a competitive
    # selected source, otherwise the highest-affinity supported source.
    deep_source_rank = None
    if deep_region_ranked:
        selected_sid = str(getattr(sel, "source_id", "") or "")
        selected_deep_rank = next((rank for rank, row in enumerate(affine_sources)
                                   if row[3] == selected_sid and rank in deep_region_ranked), None)
        strongest_deep_rank = min(
            deep_region_ranked,
            key=lambda rank: (-int(affine_sources[rank][5]), rank))
        # "Selected source" is useful evidence only when its literal scene affinity is competitive.
        # A contextual wrong-source pick may itself have a distant bench region (measured beat 94);
        # preferring it over a source whose title/query is 26 affinity points stronger would merely
        # repeat the original mismatch. One scene-token weight (5) is the existing affinity unit.
        selected_is_affine = (
            selected_deep_rank is not None
            and int(affine_sources[selected_deep_rank][5])
            >= int(affine_sources[strongest_deep_rank][5]) - 5)
        deep_source_rank = selected_deep_rank if selected_is_affine else strongest_deep_rank

    # There may likewise be only ONE transcript reserve.  It displaces the old deep reserve only
    # when its explicit two-token timed evidence is strictly stronger; equality preserves the old
    # path.  Thus a real quote/moment lock still wins, while CLIP-only and one-token hints do not
    # hide the measured Dontos/necklace passage.
    timed_source_rank = None
    if timed_region_ranked:
        timed_source_rank = max(
            timed_region_ranked,
            key=lambda rank: (timed_regions[affine_sources[rank][3]]["strength"],
                              int(affine_sources[rank][5]), -rank))
    deep_strength = ((supported_deep_regions.get(
        affine_sources[deep_source_rank][3]) or (None, None, (0, 0.0)))[2]
        if deep_source_rank is not None else (0, 0.0))
    timed_strength = (timed_regions[affine_sources[timed_source_rank][3]]["strength"]
                      if timed_source_rank is not None else (0, 0.0))
    reserve_rows = []
    deep_sid = (affine_sources[deep_source_rank][3]
                if deep_source_rank is not None else "")
    timed_sid = (affine_sources[timed_source_rank][3]
                 if timed_source_rank is not None else "")
    # Two timed words in a recap locate discussion of the event, not necessarily the pixels which
    # depict it. When the analyzer bound this beat to a concrete anchor episode and match retained
    # a supported distant region from that episode, a non-episode transcript reserve must not evict
    # it. The reserve is still exactly five calls and every frame still faces strict vision/QC.
    episode_deep_protected = bool(
        deep_sid in anchor_episode_sources and timed_sid not in anchor_episode_sources)
    if (timed_source_rank is not None and timed_strength > deep_strength
            and not episode_deep_protected):
        reserve_rows = sorted(timed_region_ranked[timed_source_rank], key=lambda row: row[0])
    elif deep_source_rank is not None:
        reserve_rows = sorted(deep_region_ranked[deep_source_rank], key=lambda row: row[0])
    # Five nearest shots cover the evidence anchor and +/-2 siblings while leaving seven of the
    # default twelve calls for the old head search. This reallocates the existing cap; it does not
    # enlarge the source/candidate caps or alter global ranking.
    for _key, cand in reserve_rows[:min(5, cap)]:
        ck = (cand.source_id, cand.shot_index)
        if ck not in chosen_keys:
            chosen.append(cand)
            chosen_keys.add(ck)
    # One matcher-ranked normal alternate can otherwise remain invisible solely because the
    # shallow strict pass stopped at three.  Spend one slot from this same cap, never an extra call.
    for _key, cand in sorted(retained_normal_ranked, key=lambda row: row[0])[:1]:
        if len(chosen) >= cap:
            break
        ck = (cand.source_id, cand.shot_index)
        if ck not in chosen_keys:
            chosen.append(cand)
            chosen_keys.add(ck)
    top_source_ranks = sorted(by_source)[:2]
    remaining = max(0, cap - len(chosen))
    if chosen and len(top_source_ranks) >= 2:
        # The local reserve consumes at most five calls from the unchanged twelve-call cap. Split
        # the remainder across the same top two ordinary sources.
        quotas = {
            rank: remaining // 2 + (1 if pos < remaining % 2 else 0)
            for pos, rank in enumerate(top_source_ranks)
        }
    else:
        old_quota = max(1, min(6, (cap + 1) // 2))
        quotas = {rank: old_quota for rank in top_source_ranks}
    for source_rank in top_source_ranks:
        for _key, cand in by_source[source_rank][:quotas[source_rank]]:
            if len(chosen) >= cap:
                break
            chosen.append(cand)
            chosen_keys.add((cand.source_id, cand.shot_index))
    # Sparse top sources must not waste the bound. Fill remaining slots source-first, preserving
    # each source's persisted-CLIP/distance order and excluding the reserved candidates.
    if len(chosen) < cap:
        for source_rank in sorted(by_source):
            for _key, cand in by_source[source_rank]:
                ck = (cand.source_id, cand.shot_index)
                if ck in chosen_keys:
                    continue
                chosen.append(cand)
                chosen_keys.add(ck)
                if len(chosen) >= cap:
                    break
            if len(chosen) >= cap:
                break
    return chosen


def verify_and_repair(proj: ClipProject, segments: list[ScriptSegment], cfg: ClipConfig,
                      eng_cfg, *, max_replacements: int = 3, only_indices=None, progress=None,
                      materialize_promotions: bool = True,
                      persist_project: bool = True,
                      strict_pool_recovery: bool = False) -> dict:
    """Verify every selection; replace failures with the best passing alternate; re-cut swaps by default.
    Returns a summary. No-op (records 'unavailable') if there's no LLM key.
    `only_indices` (a set of segment indices) restricts verification to just those beats — used by
    the bounded recovery pass to re-verify ONLY the beats it re-matched, instead of re-running the
    whole (very expensive) verifier over every beat. Beats outside the set keep their prior verdict.
    `materialize_promotions=False` is for transactional exploratory recovery: accepted alternates
    update selection metadata so the strict contract can judge them, but their deterministic clip
    filenames are not overwritten until the caller commits the accepted scoped selections.
    `persist_project=False` keeps those exploratory metadata mutations in memory as well. Verdict
    cache writes remain durable because they do not alter the selection/clip lineage transaction.
    `strict_pool_recovery=True` is for the publication-recovery transaction only: it permits the
    bounded strict-neighborhood rung to inspect strongly metadata-bound native-HD sources already
    indexed but omitted by the global variety/deep-bench cut. It does not alter any acceptance gate."""
    _subset = set(only_indices) if only_indices is not None else None

    def log(m):
        if progress:
            progress(m)

    from . import llm as _llm
    if not _llm.has_llm(eng_cfg):
        if _subset is None:                    # a full pass with no LLM stamps every beat unavailable
            for sel in proj.selections:
                sel.verifier = {"status": "unavailable", "reason": "no LLM key"}
        log("verify: skipped (no LLM key)")
        return {"verified": 0, "replaced": 0, "failed": 0, "available": False}

    get_shot = _shot_lookup(proj)
    by_idx = {s.index: s for s in segments}
    model = eng_cfg.anthropic_model
    verified = replaced = failed = 0
    # REUSE LEDGER (Stage 5) — verify_and_repair mutates selections AFTER match's greedy loop, which
    # is where the per-shot reuse cap lives; without its own ledger it promoted ONE high-scoring
    # alternate into many beats (observed: a single Jaqen closeup into 9 beats vs a cap of 2), which
    # then re-aired that look across the timeline. Seed a counter from the CURRENT selections and skip
    # an over-reused alternate on promotion (falling to the next relevance-ranked one). The sole
    # bounded exception below is an exact, quote-locked contract after every ordinary under-cap
    # candidate fails; that path selects the least-used strict pass and records the overflow.
    from collections import Counter as _Counter
    _reuse = _Counter()
    for _s in proj.selections:
        if getattr(_s, "source_id", ""):
            _reuse[(_s.source_id, _s.shot_index)] += 1
    _reuse_cap = int(getattr(cfg, "max_reuse_per_shot", 2) or 2)
    import os as _os_ms
    # ERA POLICY (Gap 2): a project-level episode hint may be used GLOBALLY only for a genuinely
    # single-scene video. A multi_scene essay spans many eras, so a global season hint is unsafe —
    # each beat's era must come from its OWN local evidence (scene_query/expected_visual/narration),
    # and a beat with no reliable local era is left UNCONSTRAINED (empty) rather than guessed.
    _vtype = str((proj.meta.get("analysis", {}) or {}).get("video_type", "") or "")
    _single = (_vtype == "single_scene")
    _global_era = str((proj.meta.get("analysis", {}) or {}).get("episode_hint", "") or "")
    # an episode hint only constrains a beat once corroborated — see era.verified_episode_hint
    _global_ok = bool((proj.meta.get("analysis", {}) or {}).get("episode_hint_verified", False))
    _ana_shim = type("A", (), {
        "anchor_scenes": (proj.meta.get("analysis", {}) or {}).get("anchor_scenes"),
        "movie_title": (proj.meta.get("analysis", {}) or {}).get("movie_title", ""),
        "characters": (proj.meta.get("analysis", {}) or {}).get("characters"),
        "actors": (proj.meta.get("analysis", {}) or {}).get("actors")})()
    _event_eras = _era.event_eras_from(_ana_shim)
    _anchor_eras = _era.anchor_token_eras(_ana_shim)

    def _era_of(_s):
        return _beat_era(_s, _global_era, _single, global_verified=_global_ok,
                         event_eras=_event_eras, anchor_eras=_anchor_eras)

    def _src_title_of(_sel):
        return _project_source_title(proj, _sel.source_id)

    # character -> actor. Face-ID reports ACTOR names while beats name CHARACTERS, so confirming
    # "Joffrey is on screen" from a face labelled 'jack gleeson' needs the roster mapping.
    _char2actor = _project_char2actor(proj)

    def _cast_warning_of(_seg, source_id: str, strict_flag: bool) -> str:
        if not strict_flag:
            return ""
        return _source_title_exact_cast_conflict(
            _seg, _project_source_title(proj, source_id), _char2actor)

    _vcache = _load_verdict_cache(proj)
    _vcache_n0 = len(_vcache)
    _errored = _reused = _consec_err = _skipped_breaker = _fp_mismatch = 0
    _materialization_errors = 0
    _breaker_open = False
    # The REAL vision provider+model, not the configured text brain. With the deepseek default a
    # vision call is actually served by Gemini (DeepSeek cannot see images), so keying on
    # eng_cfg.anthropic_model named a model that never ran and let Gemini/Claude verdicts collide.
    try:
        from . import llm as _llm_id
        _vmodel = _llm_id.vision_config(eng_cfg)
    except Exception:
        _vmodel = str(getattr(eng_cfg, "anthropic_model", "") or "")
    _src_hash_cache: dict = {}

    def _src_hash_of(src):
        sid = getattr(src, "id", "") or ""
        if sid not in _src_hash_cache:
            _src_hash_cache[sid] = _file_fingerprint(getattr(src, "local_path", "") or "")
        return _src_hash_cache[sid]
    # SCENE ROSTER (single-scene deep-dive only): every main-cast character/actor of the video. In a
    # single-scene deep-dive ANY roster member is contextually valid for any beat (they are all in
    # the one scene), so a roster face is never a "wrong character". For a multi-scene essay the
    # roster spans eras/scenes and is NOT auto-allowed — only the beat's own co-mentioned entities.
    _roster_toks: set = set()
    if _single:
        _an = proj.meta.get("analysis", {}) or {}
        for _c in (_an.get("characters") or []):
            _nm = _c.get("name", "") if isinstance(_c, dict) else str(_c)
            _roster_toks |= {w for w in re.findall(r"[a-z0-9]+", (_nm or "").lower()) if len(w) > 2}
        for _a in (_an.get("actors") or []):
            _roster_toks |= {w for w in re.findall(r"[a-z0-9]+", (str(_a) or "").lower()) if len(w) > 2}
    # roster tokens allowed as "co-mentioned" context in a single-scene deep-dive — LOOP-INVARIANT,
    # so compute ONCE here. (Was assigned only inside the character-beat branch, so an object/scene
    # beat that skipped that branch hit an UnboundLocalError in the generic-filler rung below —
    # crashing verify on beat 11 of a single-scene render and trapping it in a resume→re-crash loop.)
    _ok_toks = _roster_toks if _single else frozenset()
    _mf_on = _os_ms.environ.get("VIDLORE_CLIPSTUDIO_VERIFY_ACTION_SHEET", "1").strip() \
        not in ("0", "false", "no")

    def _selected_window(window, ashot) -> tuple[float, float]:
        try:
            a, b = float(window[0]), float(window[1])
        except Exception:
            a = float(getattr(ashot, "start", 0.0) or 0.0)
            b = float(getattr(ashot, "end", 0.0) or 0.0)
        return a, b

    def _will_sheet(ashot, _seg, window) -> bool:
        """Predict, WITHOUT building it, whether this call uses a contact sheet. Must mirror
        _verify_ctx's own condition — the prediction is part of the cache key, and _verify_ctx
        reports what actually happened so a wrong prediction can never be stored."""
        if (not _mf_on or ashot is None
                or _policy.policy_of(_seg) not in (_policy.EXACT, _policy.CHARACTER)):
            return False
        a, b = _selected_window(window, ashot)
        if not (b - a >= 0.5 and a >= 0.0):
            return False
        _sid = getattr(ashot, "source_id", "") or ""
        _src = proj.source(_sid) if _sid else None
        return bool(getattr(_src, "local_path", "") if _src else "")

    def _image_id(kf_path, ashot, want_sheet: bool, window=None) -> str:
        """Identity of the PIXELS the verifier will judge.

        Shot bounds do not pin this: a re-index can rewrite a keyframe while start/end stay put, and
        the stale verdict would be reused against a different image. For a sheet the id is derived
        (source content + span + SHEET_VERSION) rather than measured, so a cache HIT costs no ffmpeg
        work — the sheet is a pure function of those inputs."""
        if want_sheet and ashot is not None:
            _src = proj.source(getattr(ashot, "source_id", "") or "")
            a, b = _selected_window(window, ashot)
            return f"sheet:{_src_hash_of(_src)}:{a:.3f}-{b:.3f}"
        return f"kf:{_file_fingerprint(kf_path)}" if kf_path else "kf:none"

    def _verify_ctx(kf_path, ashot, _seg, _exact, faceids, window=None):
        """Verify one candidate against pixels sampled from the exact selected window.

        Exact/concrete character beats use a 15/50/85 contact sheet of that trim, never the wider
        detected shot.  If sheet generation fails the call remains a single-frame judgment, but its
        persisted binding says ``multiframe=false`` and the publication contract blocks it.
        -> (verdict|None, actually_used_a_sheet)"""
        sheet, is_mf = kf_path, False
        if _will_sheet(ashot, _seg, window):
            try:
                _sid = getattr(ashot, "source_id", "") or ""
                _src = proj.source(_sid) if _sid else None
                _sp = getattr(_src, "local_path", "") if _src else ""
                if _sp:
                    # UNIQUE dest per call: concurrent warms of the SAME beat's alternates can
                    # share a shot index — a shared name would judge one alternate's pixels
                    # against another's question. The name is not a fingerprint input.
                    import uuid as _uuid_vs
                    _dest = proj.clips_dir / (
                        f"_vsheet_{_seg.index}_"
                        f"{(getattr(ashot, 'source_id', '') or 'x')[:10]}_"
                        f"{getattr(ashot, 'index', 0)}_{_uuid_vs.uuid4().hex[:6]}.jpg")
                    _wa, _wb = _selected_window(window, ashot)
                    _got = _action_contact_sheet(_sp, _wa, _wb, _dest)
                    if _got:
                        sheet, is_mf = str(_got), True
            except Exception:
                sheet, is_mf = kf_path, False              # any sheet failure → single-frame path
        try:
            _cast_warning = _cast_warning_of(
                _seg, getattr(ashot, "source_id", "") or "", bool(_exact))
            return verify_frame(sheet, _seg.text, _seg.required_entity, _seg.required_kind, faceids,
                                eng_cfg, model, is_specific=_exact,
                                expected_visual=getattr(_seg, "expected_visual", "") or "",
                                scene_query=getattr(_seg, "scene_query", "") or "",
                                era_hint=_era_of(_seg), multiframe=is_mf,
                                must_see=_must_see(_seg),
                                exact_cast_warning=_cast_warning), is_mf
        finally:
            if is_mf:
                try:
                    Path(sheet).unlink(missing_ok=True)
                except Exception:
                    pass

    _vcache_dirty = 0

    _look_scope = {"on": True}          # False while judging ALTERNATES (see _try_promote)

    def _must_see(_seg) -> str:
        """The thing this beat tells the viewer to LOOK at, or "".

        Env: VIDLORE_CLIPSTUDIO_LOOK_GATE=0 disables. When set, the verifier is asked whether that
        specific thing is on screen and a frame without it cannot satisfy the beat — "the right
        people are present" is not an answer to "keep your eye on the dagger"."""
        if not _look_scope["on"]:
            return ""
        return effective_deictic_target(_seg)

    def _served_model_of(v) -> str:
        """The provider that ACTUALLY served this fresh verdict (canonical id), falling back
        to the prediction for stubbed/legacy verdicts that carry no provenance."""
        _sb = str((v or {}).get("vision_served_by") or "")
        return _sb if _sb and _sb != "none" else _vmodel

    def _bind_evidence(vv, _sel, _seg, _shot, strict_flag: bool, faces, used_sheet: bool,
                       actual_must_see: str) -> None:
        """Persist proof of exactly which selection/window/pixels the verdict judged."""
        bind_selection_verifier_evidence(
            proj, _sel, _seg, vv, shot=_shot, model=_served_model_of(vv),
            is_specific=strict_flag, multiframe=used_sheet, faceid_names=list(faces or []),
            era=_era_of(_seg), must_see=actual_must_see)

    def _rung_fingerprint(
            ashot, _seg, strict_flag: bool, faceids, window, model_id: str = ""):
        """Fingerprint of a FALLBACK-RUNG question — the exact same derivation as the primary
        path (keep the two in sync), parameterized on the candidate shot and the rung's
        strictness. `faceids` must be the SAME list the prompt will carry (the caller may
        substitute sel.identity when the shot has no face detections — the fingerprint hashes
        what is actually asked, never a proxy). venue_fallback stays False here because
        _verify_ctx never asks the venue question (the still layer does)."""
        if ashot is None:
            return "", False
        _src_a = proj.source(getattr(ashot, "source_id", "") or "")
        _ws = _will_sheet(ashot, _seg, window)
        _kf_a = getattr(ashot, "keyframe_path", "") or ""
        _wa, _wb = _selected_window(window, ashot)
        return verdict_fingerprint(
            src_hash=_src_hash_of(_src_a), source_id=getattr(ashot, "source_id", "") or "",
            shot_start=(_wa if _ws else getattr(ashot, "start", 0.0)),
            shot_end=(_wb if _ws else getattr(ashot, "end", 0.0)),
            beat_text=getattr(_seg, "text", ""),
            required_entity=getattr(_seg, "required_entity", ""),
            required_kind=getattr(_seg, "required_kind", ""),
            expected_visual=getattr(_seg, "expected_visual", "") or "",
            scene_query=getattr(_seg, "scene_query", "") or "",
            era=_era_of(_seg), visual_policy=_policy.policy_of(_seg),
            is_specific=strict_flag, faceid_names=list(faceids or []),
            multiframe=_ws, image_id=_image_id(_kf_a, ashot, _ws, window),
            model=(model_id or _vmodel), must_see=_must_see(_seg),
            exact_cast_warning=_cast_warning_of(
                _seg, getattr(ashot, "source_id", "") or "", strict_flag)), _ws

    def _cached_verify_ctx(
            kf_path, ashot, _seg, strict_flag: bool, faceids, window, rung: str):
        """One cache layer for every fallback-rung verdict (strict promotion, contextual
        downgrade, venue-contextual promotion, lenient generic-filler re-ask). A rung verdict
        is reusable ONLY when its complete fingerprint — content hash, shot bounds, judged
        image identity, every prompt field, strictness, model, prompt/sheet version — is
        byte-identical (the same doctrine, and the same key derivation, as the primary path).
        Only successful schema-valid verdicts are stored; a transport error / breaker miss /
        malformed reply is never cached, so retry and circuit-breaker behavior is untouched.
        Returns (verdict|None, used_sheet) exactly like _verify_ctx."""
        nonlocal _vcache_dirty
        from . import perf_metrics as _pm_r
        _fp_r, _ws_r = _rung_fingerprint(ashot, _seg, strict_flag, faceids, window)
        _required_r = getattr(_seg, "required_entity", "") or ""
        _must_see_r = _must_see(_seg)
        _cast_warning_r = _cast_warning_of(
            _seg, getattr(ashot, "source_id", "") or "", strict_flag)
        _hit = _vcache.get(_fp_r) if _fp_r else None
        if _hit is not None and _verdict_schema_ok(
                _hit, required_entity=_required_r, must_see=_must_see_r,
                exact_cast_warning=_cast_warning_r) \
                and _hit_provider_ok(_hit, _vmodel):
            _pm_r.incr(f"verify.rung.{rung}.cache_hit")
            # Selection-lifecycle labels must not cross the immutable question-answer cache
            # boundary.  Copy first because callers intentionally mutate the returned verdict.
            return _clear_verifier_transition_state(dict(_hit)), _ws_r
        _pm_r.incr(f"verify.rung.{rung}.call")
        v_r, used_r = _verify_ctx(kf_path, ashot, _seg, strict_flag, faceids, window)
        if v_r is not None:
            _clear_verifier_transition_state(v_r)
        if _fp_r and v_r is not None and used_r == _ws_r \
                and _verdict_schema_ok(
                    {**v_r, "status": "ok"}, required_entity=_required_r,
                    must_see=_must_see_r, exact_cast_warning=_cast_warning_r):
            # store under the ACTUAL server's key — a Claude fallback answer must never sit
            # under a Gemini-predicted fingerprint (and vice versa)
            _served_r = _served_model_of(v_r)
            _fp_store = _fp_r if _served_r == _vmodel else \
                _rung_fingerprint(
                    ashot, _seg, strict_flag, faceids, window, model_id=_served_r)[0]
            if _fp_store:
                _vcache[_fp_store] = {k: val for k, val in v_r.items() if k != "reused"}
                _vcache_dirty += 1
        elif _fp_r and v_r is not None and used_r != _ws_r:
            _pm_r.incr("verify.rung.sheet_mismatch")     # not cached (answer ≠ keyed question)
        return v_r, used_r

    # CONCURRENT VERDICT PREFETCH — the wall-clock fix for the verify stage. The decision loop
    # below is inherently serial (reuse ledger, breaker, repair promotions share state), but the
    # EXPENSIVE part — one vision call per beat — has no cross-beat dependency at all. Measured:
    # ~148 serial calls ≈ 25-90 min of the render. This pass computes each pending selection's
    # fingerprint (the SAME derivation as the loop below — keep them in sync) and warms the verdict
    # cache with a small worker pool; the serial loop then hits cache. Failure semantics are
    # UNCHANGED: a failed prefetch simply isn't cached, so the serial loop re-asks with its own
    # retry/backoff and the circuit breaker still governs; a burst of prefetch failures aborts the
    # prefetch early and leaves everything to the serial path. Sheet-prediction rule preserved
    # (store only when used_sheet == want_sheet).
    #
    # OPT-IN (default 1 = off): the breaker/outage suites encode EXACT call-count contracts for
    # the serial path ("stop after 8 consecutive errors"); a default-on pool would change those
    # counts under failure. Concurrency here is a turbo switch, like MAX_CPU — the portal/runners
    # enable it explicitly with VIDLORE_CLIPSTUDIO_VERIFY_WORKERS=4.
    import os as _os_pf
    try:
        _pf_workers = int(_os_pf.environ.get("VIDLORE_CLIPSTUDIO_VERIFY_WORKERS", "1") or 1)
    except (TypeError, ValueError):
        _pf_workers = 1
    if _pf_workers > 1 and not _breaker_open:
        _pending = []
        _prim_items = []      # EVERY eligible selection (cached or not) — feeds phase-2 rung warms
        for sel in proj.selections:
            if _subset is not None and sel.segment_index not in _subset:
                continue
            if not sel.source_id:
                continue
            seg = by_idx.get(sel.segment_index)
            if seg is None:
                continue
            shot = get_shot(sel.source_id, sel.shot_index)
            if shot is None:
                continue
            kf = shot.keyframe_path if shot else ""
            faceid_names = (shot.face_ids if shot else []) or ([sel.identity] if sel.identity else [])
            _exact = _policy.verify_strict(seg)
            _src_obj = proj.source(sel.source_id)
            _window = (sel.in_point, sel.out_point)
            _want_sheet = _will_sheet(shot, seg, _window)
            _wa, _wb = _selected_window(_window, shot)
            _fp = verdict_fingerprint(
                src_hash=_src_hash_of(_src_obj), source_id=sel.source_id or "",
                shot_start=(_wa if _want_sheet else getattr(shot, "start", 0.0)),
                shot_end=(_wb if _want_sheet else getattr(shot, "end", 0.0)),
                beat_text=getattr(seg, "text", ""),
                required_entity=getattr(seg, "required_entity", ""),
                required_kind=getattr(seg, "required_kind", ""),
                expected_visual=getattr(seg, "expected_visual", "") or "",
                scene_query=getattr(seg, "scene_query", "") or "",
                era=_era_of(seg), visual_policy=_policy.policy_of(seg), is_specific=_exact,
                faceid_names=faceid_names, multiframe=_want_sheet,
                image_id=_image_id(kf, shot, _want_sheet, _window), model=_vmodel,
                must_see=_must_see(seg),
                exact_cast_warning=_cast_warning_of(seg, sel.source_id, _exact))
            _c0 = _vcache.get(_fp)
            if _fp:
                _prim_items.append((_fp, sel, seg, shot, kf, faceid_names, _exact, _window))
            if _fp and (_c0 is None or not _verdict_schema_ok(
                            _c0, required_entity=getattr(seg, "required_entity", "") or "",
                            must_see=_must_see(seg),
                            exact_cast_warning=_cast_warning_of(
                                seg, sel.source_id, _exact))
                        or not _hit_provider_ok(_c0, _vmodel)):
                _pending.append(
                    (_fp, seg, shot, kf, faceid_names, _exact, _want_sheet, _window))
        if _pending:
            log(f"verify: prefetching {len(_pending)} fresh verdict(s) "
                f"({_pf_workers} workers; serial decisions unchanged)")
            import concurrent.futures as _cf
            _pf_fail = _pf_ok = 0
            with _cf.ThreadPoolExecutor(max_workers=_pf_workers) as _ex:
                _futs = {_ex.submit(_verify_ctx, kf9, shot9, seg9, ex9, fids9, win9):
                         (fp9, ws9, seg9, shot9, ex9)
                         for (fp9, seg9, shot9, kf9, fids9, ex9, ws9, win9) in _pending}
                for _fu in _cf.as_completed(_futs):
                    _fp9, _ws9, _seg_schema9, _shot_schema9, _exact_schema9 = _futs[_fu]
                    try:
                        _v9, _us9 = _fu.result()
                    except Exception:                     # noqa: BLE001
                        _v9, _us9 = None, _ws9
                    if _v9 is not None and _us9 == _ws9 \
                            and _verdict_schema_ok(
                                {**_v9, "status": "ok"},
                                required_entity=getattr(
                                    _seg_schema9, "required_entity", "") or "",
                                must_see=_must_see(_seg_schema9),
                                exact_cast_warning=_cast_warning_of(
                                    _seg_schema9,
                                    getattr(_shot_schema9, "source_id", "") or "",
                                    _exact_schema9)):
                        _served9 = _served_model_of(_v9)
                        _fp_store9 = _fp9
                        if _served9 != _vmodel:
                            # a fallback provider answered — key by the ACTUAL server
                            (_, _seg9, _shot9, _kf9, _fids9, _ex9b, _ws9b, _win9) = \
                                next(p for p in _pending if p[0] == _fp9)
                            _src9 = proj.source(getattr(_shot9, "source_id", "") or "")
                            _wa9, _wb9 = _selected_window(_win9, _shot9)
                            _fp_store9 = verdict_fingerprint(
                                src_hash=_src_hash_of(_src9),
                                source_id=getattr(_shot9, "source_id", "") or "",
                                shot_start=(_wa9 if _ws9b else getattr(_shot9, "start", 0.0)),
                                shot_end=(_wb9 if _ws9b else getattr(_shot9, "end", 0.0)),
                                beat_text=getattr(_seg9, "text", ""),
                                required_entity=getattr(_seg9, "required_entity", ""),
                                required_kind=getattr(_seg9, "required_kind", ""),
                                expected_visual=getattr(_seg9, "expected_visual", "") or "",
                                scene_query=getattr(_seg9, "scene_query", "") or "",
                                era=_era_of(_seg9), visual_policy=_policy.policy_of(_seg9),
                                is_specific=_ex9b, faceid_names=_fids9,
                                multiframe=_ws9b,
                                image_id=_image_id(_kf9, _shot9, _ws9b, _win9),
                                model=_served9, must_see=_must_see(_seg9),
                                exact_cast_warning=_cast_warning_of(
                                    _seg9, getattr(_shot9, "source_id", "") or "", _ex9b))
                        _vcache[_fp_store9] = {k: val for k, val in _v9.items()
                                               if k != "reused"}
                        _pf_ok += 1
                        _pf_fail = 0
                    else:
                        _pf_fail += 1
                        if _pf_fail >= VERIFIER_BREAKER_TRIP:
                            for _f2 in _futs:
                                _f2.cancel()
                            log("verify: prefetch aborted — repeated transport failures; the "
                                "serial loop (and its circuit breaker) takes over")
                            break
            log(f"verify: prefetch warmed {_pf_ok}/{len(_pending)} verdict(s)")
            _save_verdict_cache(proj, _vcache)

        # PHASE-2 RUNG PREFETCH (the unimplemented half of prior-audit OPT-3) — the repair
        # chain's ~450-600 vision calls stayed SERIAL even with phase-1 on (measured ~17-20
        # min: 146/202 beats entered repair). For every EXACT beat whose warmed PRIMARY verdict
        # will enter repair (an explicit ``replace`` OR a self-contradictory ``keep`` rejected by
        # the strict publication facts), warm the questions the serial loop will provably ask:
        #   (a) the strict-promotion rung over the first max_replacements alternates in the
        #       exact serial order (_scene_affinity_order when enabled), including the SAME
        #       named-look-target question the serial strict promotion asks; a slot-consuming
        #       shotless alternate is skipped exactly like the serial walk;
        #   (b) the lenient generic-filler re-ask of the ORIGINAL shot (look scope ON).
        # All warms flow through _cached_verify_ctx — identical fingerprints, store-on-
        # success-only, breaker semantics untouched. The serial loop runs UNCHANGED and hits
        # cache. Superset waste is bounded (the loop stops at its first accept; a warm it
        # never reads is just an unused cache entry). The two sub-passes run one after the
        # other so the shared _look_scope flag is uniform within each pool.
        # VIDLORE_CLIPSTUDIO_VERIFY_PREFETCH_RUNGS=0 disables phase-2 only.
        if _prim_items and _os_pf.environ.get(
                "VIDLORE_CLIPSTUDIO_VERIFY_PREFETCH_RUNGS", "1").strip().lower() \
                not in ("0", "false", "no"):
            _aff_on2 = _os_pf.environ.get("VIDLORE_CLIPSTUDIO_SCENE_AFFINITY", "1").strip() \
                not in ("0", "false", "no", "")
            # STAGED, not eager. The serial loop stops at its first ACCEPTED alternate, so warming
            # all max_replacements up front bought verdicts nobody reads (measured: 13 of 15
            # lenient warms unread on one render). Depth is therefore paid one WAVE at a time —
            # alternate #1 for every beat concurrently, #2 only for the beats whose #1 came back
            # 'replace', and so on — which keeps essentially all of the wall-clock win (the
            # non-accepting majority still ends up fully warmed) while not buying the tail.
            # Warming FEWER questions can never change a decision: anything unwarmed is simply
            # asked fresh by the serial loop, exactly as it behaves today.
            _beat_alts, _len_jobs = [], []
            for (_fpP, selP, segP, shotP, kfP, fidsP, _exP, _winP) in _prim_items:
                _v0 = _vcache.get(_fpP)
                if not (_v0 is not None and _verdict_schema_ok(
                            _v0, required_entity=getattr(segP, "required_entity", "") or "",
                            must_see=_must_see(segP),
                            exact_cast_warning=_cast_warning_of(
                                segP, selP.source_id, _exP))
                        and _hit_provider_ok(_v0, _vmodel)):
                    continue                             # primary not warm → serial pays as today
                _primary_strict_reject = (
                    _strict_keep_rejection_reason(
                        _v0, segP, _src_title_of(selP), _char2actor,
                        must_see=_must_see(segP))
                    if _exP and str(_v0.get("verdict")) == "keep" else "")
                if (not _exP or (str(_v0.get("verdict")) != "replace"
                                 and not _primary_strict_reject)):
                    continue
                _altsP = selP.alternates
                if _aff_on2:
                    try:
                        _altsP = _scene_affinity_order(selP.alternates, segP, proj,
                                                       selP.source_id)
                    except Exception:                    # noqa: BLE001
                        _altsP = selP.alternates
                _t2, _chain = 0, []
                for _altP in _altsP:
                    if _t2 >= max_replacements:
                        break
                    _t2 += 1                             # slot consumed even when shotless
                    _ashP = get_shot(_altP.source_id, _altP.shot_index)
                    if _ashP is None:
                        continue
                    _chain.append((_ashP, segP, (_altP.in_point, _altP.out_point)))
                _bi_here = len(_beat_alts)
                if _chain:
                    _beat_alts.append(_chain)
                # The lenient re-ask fires serially ONLY at the generic-filler rung, which is
                # reached only when nothing swapped. A primary verdict that satisfies
                # _exact_contextual_ok routes the beat to a keep-contextual downgrade (both
                # branches set swapped=True), so that question is provably never asked — and it
                # is the single largest unread-warm bucket. Env gates mirrored so the warm never
                # asks a question the serial loop cannot reach.
                if (_exact_contextual_ok(_v0, segP, _src_title_of(selP), _char2actor)
                        or _os_pf.environ.get("VIDLORE_CLIPSTUDIO_EXACT_CONTEXTUAL_DOWNGRADE",
                                              "1").strip() in ("0", "false", "no")
                        or _os_pf.environ.get("VIDLORE_CLIPSTUDIO_GENERIC_FILLER_DOWNGRADE",
                                              "1").strip() in ("0", "false", "no")):
                    continue
                _len_jobs.append(
                    (_bi_here if _chain else -1, (kfP, shotP, segP, fidsP, _winP)))
            _alt_waves = []
            for _d in range(max_replacements):           # wave d = every beat's alternate #d
                _alt_waves.append([(c[_d], i) for i, c in enumerate(_beat_alts) if _d < len(c)])
            if _beat_alts or _len_jobs:
                log(f"verify: rung prefetch — {len(_beat_alts)} beat(s) staged (wave 1 = "
                    f"{len(_alt_waves[0]) if _alt_waves else 0} promotion) + {len(_len_jobs)} "
                    f"lenient question(s) ({_pf_workers} workers; the serial repair loop and "
                    f"every gate stay unchanged)")
                import concurrent.futures as _cf2
                _rp_fail = 0

                def _warm(fn, jobs):
                    nonlocal _rp_fail
                    with _cf2.ThreadPoolExecutor(max_workers=_pf_workers) as _ex2:
                        _fs2 = [_ex2.submit(fn, j) for j in jobs]
                        for _f2 in _fs2:
                            try:
                                _ok2 = _f2.result()
                            except Exception:            # noqa: BLE001
                                _ok2 = False
                            if _ok2:
                                _rp_fail = 0
                            else:
                                _rp_fail += 1
                                if _rp_fail >= VERIFIER_BREAKER_TRIP:
                                    for _f3 in _fs2:
                                        _f3.cancel()
                                    log("verify: rung prefetch aborted — repeated transport "
                                        "failures; the serial loop takes over")
                                    return False
                    return True

                _alt_done = {}                           # beat idx -> True once a warm said 'keep'

                def _warm_alt(j):
                    (_a, _s, _win_a), _bi = j
                    _v_w, _ = _cached_verify_ctx(_a.keyframe_path, _a, _s, True,
                                                 _a.face_ids or [], _win_a,
                                                 rung="strict_promote")
                    _look_w = _must_see(_s)
                    _src_w = proj.source(getattr(_a, "source_id", "") or "")
                    _title_w = ((getattr(_src_w, "title", "") or "") + " "
                                + (getattr(_a, "source_id", "") or ""))
                    if (_v_w is not None and not _strict_keep_rejection_reason(
                            _v_w, _s, _title_w, _char2actor, must_see=_look_w)):
                        # the serial loop may still refuse this alternate on the reuse cap or
                        # window-QC and walk on — then the next alternate is simply unwarmed and
                        # asked fresh, which is today's behaviour for anything not prefetched
                        _alt_done[_bi] = True
                    return _v_w is not None

                def _warm_len(j):
                    _kf_w, _sh_w, _s_w, _fi_w, _win_w = j
                    _v_w, _ = _cached_verify_ctx(_kf_w, _sh_w, _s_w, False, _fi_w,
                                                 _win_w,
                                                 rung="lenient_filler")
                    return _v_w is not None

                _go_on = True
                # Strict alternates must answer the named-target question too.  Keeping the scope
                # on here makes these warms byte-identical to the serial path; otherwise every
                # deictic candidate misses cache and is paid for twice.
                _look_scope["on"] = True
                for _wi, _wave in enumerate(_alt_waves):
                    _todo = [j for j in _wave if not _alt_done.get(j[1])]
                    if not _todo:
                        break                        # every staged beat already has a valid keep
                    if _wi:
                        from . import perf_metrics as _pm_pf
                        _pm_pf.incr(f"verify.rung.wave{_wi + 1}", len(_todo))
                    _go_on = _warm(_warm_alt, _todo)
                    if not _go_on:
                        break
                if _go_on:
                    # The lenient rung is reached ONLY when nothing swapped, so a beat whose
                    # strict warm already came back 'keep' will (barring a reuse-cap/window-QC
                    # refusal, which simply leaves the question unwarmed and asked fresh) never
                    # get there. Drop those — measured 6 of 6 unread on the parity fixture.
                    _len_todo = [j for _bi, j in _len_jobs if not _alt_done.get(_bi)]
                    _warm(_warm_len, _len_todo)          # original shots: look scope ON
                _save_verdict_cache(proj, _vcache)

    for sel in proj.selections:
        if _subset is not None and sel.segment_index not in _subset:
            continue                           # recovery pass: verify only the re-matched beats
        if not sel.source_id:
            continue
        seg = by_idx.get(sel.segment_index)
        if seg is None:
            continue
        shot = get_shot(sel.source_id, sel.shot_index)
        kf = shot.keyframe_path if shot else ""
        faceid_names = (shot.face_ids if shot else []) or ([sel.identity] if sel.identity else [])
        _resolved_policy = _policy.policy_of(seg)
        _exact = _policy.verify_strict(seg)               # exact_scene → strict; else lenient (filler ok)
        _character = _resolved_policy == _policy.CHARACTER
        # REUSE a verdict only when the QUESTION is byte-identical (see verdict_fingerprint). This
        # is what lets a restart keep explicitly-proven judgments instead of re-rolling them against
        # a dying API — the failure mode that published this render.
        _fp, _want_sheet = "", False
        if shot is not None:
            _src_obj = proj.source(sel.source_id)
            _window = (sel.in_point, sel.out_point)
            _want_sheet = _will_sheet(shot, seg, _window)
            _wa, _wb = _selected_window(_window, shot)
            _fp = verdict_fingerprint(
                src_hash=_src_hash_of(_src_obj), source_id=sel.source_id or "",
                shot_start=(_wa if _want_sheet else getattr(shot, "start", 0.0)),
                shot_end=(_wb if _want_sheet else getattr(shot, "end", 0.0)),
                beat_text=getattr(seg, "text", ""),
                required_entity=getattr(seg, "required_entity", ""),
                required_kind=getattr(seg, "required_kind", ""),
                expected_visual=getattr(seg, "expected_visual", "") or "",
                scene_query=getattr(seg, "scene_query", "") or "",
                era=_era_of(seg), visual_policy=_policy.policy_of(seg), is_specific=_exact,
                faceid_names=faceid_names, multiframe=_want_sheet,
                image_id=_image_id(kf, shot, _want_sheet, _window), model=_vmodel,
                must_see=_must_see(seg),
                exact_cast_warning=_cast_warning_of(seg, sel.source_id, _exact))
        # only a SUCCESSFUL, schema-valid verdict is reusable — never an error stub or a malformed
        # reply whose missing "verdict" key would read as falsy and quietly pass
        _cached = _vcache.get(_fp) if _fp else None
        _used_sheet = _want_sheet
        if _cached is not None and _verdict_schema_ok(
                _cached, required_entity=getattr(seg, "required_entity", "") or "",
                must_see=_must_see(seg),
                exact_cast_warning=_cast_warning_of(seg, sel.source_id, _exact)) \
                and _hit_provider_ok(_cached, _vmodel):
            from . import perf_metrics as _pm_v
            _pm_v.incr("verify.primary.cache_hit")
            v = dict(_cached)
            v["reused"] = True
            _reused += 1
        else:
            from . import perf_metrics as _pm_m
            _pm_m.incr("verify.primary.cache_miss")
            if _cached is not None:
                _vcache.pop(_fp, None)                 # poisoned entry — drop it
            if _breaker_open:
                # BREAKER OPEN — do not call. See the note where it trips: past the threshold the
                # backend is down, and every further call is latency spent to learn that again.
                v, _used_sheet = None, _want_sheet
            else:
                v, _used_sheet = _verify_ctx(
                    kf, shot, seg, _exact, faceid_names, (sel.in_point, sel.out_point))
        _schema_failure = ""
        _current_cast_warning = _cast_warning_of(seg, sel.source_id, _exact)
        if (v is not None and _current_cast_warning and v.get("verdict") == "keep"
                and not isinstance(v.get("source_title_conflict_resolved"), bool)):
            # This is a malformed answer to the focused question, not evidence that the footage is
            # wrong.  Route it through the same technical retry/breaker lane as native-still schema
            # failures; otherwise a missing JSON field could buy acquisition or specificity loss.
            _schema_failure = "focused source-title conflict field is missing or malformed"
            v = None
        if v is None:
            # FAIL CLOSED. "No judgment" is not a synonym for "acceptable". The old code set
            # status=error and `continue`d, so a beat nobody could check looked exactly like a beat
            # that passed — and a TOTAL outage produced zero rejections, which the release gate read
            # as "nothing wrong" and shipped.
            sel.verifier = {"status": "breaker_open" if _breaker_open else "error"}
            if _schema_failure:
                sel.verifier["reason"] = _schema_failure
            _errored += 1
            if _breaker_open:
                _skipped_breaker += 1
            else:
                _consec_err += 1
            if FLAG_VERIFIER_UNVERIFIED not in sel.flag_reasons:
                sel.flag_reasons.append(FLAG_VERIFIER_UNVERIFIED)
            sel.flagged = True
            if _exact:
                # an exact_scene beat we could not check is UNRESOLVED, never a pass
                failed += 1
                log(f"verify: seg{sel.segment_index} UNVERIFIED "
                    f"({'breaker open' if _breaker_open else 'verifier error'}) — exact_scene "
                    f"beat is unresolved, not accepted")
            if _consec_err >= VERIFIER_BREAKER_TRIP and not _breaker_open:
                _breaker_open = True
                log(f"verify: ⛔ CIRCUIT BREAKER OPEN — {_consec_err} consecutive verifier errors; "
                    f"the vision backend is down. NO further verifier requests will be made; every "
                    f"remaining beat is marked unverified and exact beats will release-block.")
            continue
        # A scoped retry may hit a cache written by an older lifecycle which accidentally stored
        # selection-transition metadata beside the vision answer.  Establish a clean strict answer
        # before applying this run's contract/downgrade decisions and binding it to current pixels.
        _clear_verifier_transition_state(v)
        _consec_err = 0
        verified += 1                    # counts SUCCESSES — never attempts (see the breaker note)
        # STORE only a schema-valid verdict, and only when the sheet prediction that went INTO the
        # key actually held. A sheet build can fail and silently fall back to one frame — storing
        # that single-frame answer under a multiframe key would hand back the wrong judgment later.
        if _fp and not v.get("reused") and _verdict_schema_ok(
                {**v, "status": "ok"},
                required_entity=getattr(seg, "required_entity", "") or "",
                must_see=_must_see(seg),
                exact_cast_warning=_cast_warning_of(seg, sel.source_id, _exact)):
            if _used_sheet == _want_sheet:
                # key by the ACTUAL server (a fallback provider's answer must never sit under
                # the predicted provider's fingerprint)
                _served_p = _served_model_of(v)
                _fp_store_p = _fp
                if _served_p != _vmodel and shot is not None:
                    _src_obj = proj.source(sel.source_id)
                    _window = (sel.in_point, sel.out_point)
                    _wa, _wb = _selected_window(_window, shot)
                    _fp_store_p = verdict_fingerprint(
                        src_hash=_src_hash_of(_src_obj), source_id=sel.source_id or "",
                        shot_start=(_wa if _want_sheet else getattr(shot, "start", 0.0)),
                        shot_end=(_wb if _want_sheet else getattr(shot, "end", 0.0)),
                        beat_text=getattr(seg, "text", ""),
                        required_entity=getattr(seg, "required_entity", ""),
                        required_kind=getattr(seg, "required_kind", ""),
                        expected_visual=getattr(seg, "expected_visual", "") or "",
                        scene_query=getattr(seg, "scene_query", "") or "",
                        era=_era_of(seg), visual_policy=_policy.policy_of(seg),
                        is_specific=_exact, faceid_names=faceid_names,
                        multiframe=_want_sheet,
                        image_id=_image_id(kf, shot, _want_sheet, _window),
                        model=_served_p, must_see=_must_see(seg),
                        exact_cast_warning=_cast_warning_of(seg, sel.source_id, _exact))
                _vcache[_fp_store_p] = {k: val for k, val in v.items() if k != "reused"}
                _vcache_dirty += 1
            else:
                _fp_mismatch += 1
        v["status"] = "ok"
        v["visual_policy"] = _policy.policy_of(seg)
        # Deterministic contradiction evidence is negative-only and therefore safe to combine with
        # vision: it never proves a source correct. An exact verdict that explicitly says mismatch/
        # insufficient, or whose footage directly contradicts the line, cannot enter the repair
        # ladder as a keep. Non-exact leniency below remains unchanged.
        _primary_conflict = _contradiction_reason(seg, v, _src_title_of(sel), _char2actor)
        if _primary_conflict:
            v["contradicts_narration"] = True
            v["contradiction_reason"] = _primary_conflict
        _primary_contract_rejection = ""
        if v.get("verdict") == "keep":
            if _exact:
                _primary_contract_rejection = _strict_keep_rejection_reason(
                    v, seg, _src_title_of(sel), _char2actor, must_see=_must_see(seg))
                if not _primary_contract_rejection:
                    _reaction_context = _exact_reaction_context_evidence(
                        proj, sel, seg, cfg=cfg)
                    if _reaction_context.get("required"):
                        v["exact_reaction_context"] = _reaction_context
                        if not _reaction_context.get("passed"):
                            _primary_contract_rejection = str(
                                _reaction_context.get("reason")
                                or "exact_reaction_context_unproven")
            elif _character:
                _primary_contract_rejection = _character_keep_rejection_reason(
                    v, seg, _src_title_of(sel), _char2actor, must_see=_must_see(seg))
        if _primary_contract_rejection:
            v["verdict"] = "replace"
            v["contract_rejected"] = _primary_contract_rejection
        # NON-EXACT LENIENCY (user rule: exact clip only for a SPECIFIC scene; a relevant FILLER is
        # fine for generic/character/abstract beats). Don't replace an on-topic, right-subject clip on
        # a non-exact beat just because it isn't the exact scene — only off-topic / wrong-character.
        if not _exact and v.get("verdict") == "replace" and _contextual_subject_ok(v):
            # A provider may put ``replace`` at the top while all character-filler facts are
            # positive.  Preserve that established leniency, but never flip an answer whose own
            # fields would be blocked by the publication contract.
            _relaxed = dict(v)
            _relaxed["verdict"] = "keep"
            _character_rejection = (
                _character_keep_rejection_reason(
                    _relaxed, seg, _src_title_of(sel), _char2actor,
                    must_see=_must_see(seg)) if _character else "")
            if not _character_rejection:
                v["verdict"] = "keep"
                v["relaxed"] = "non-exact beat: relevant right-subject filler accepted"
            else:
                v["contract_rejected"] = _character_rejection
        _bind_evidence(v, sel, seg, shot, _exact, faceid_names, _used_sheet, _must_see(seg))
        sel.verifier = v

        # FLAG-ON-ANY-VERDICT: a keep (or leniency-flipped) verdict with the named look-target
        # absent previously left the beat unflagged — the gate lived only inside the replace
        # branch, so "watch the chalice" over a kept chalice-less clip got no still coverage.
        # The FOOTAGE decision is untouched (measured: swapping usable picks regressed −4.00);
        # the flag only routes the beat to the still pass, which shows a frame OF the moment.
        if _must_see(seg) and v.get("target_visible") is False:
            try:
                if "look_target_missing" not in (sel.flag_reasons or []):
                    sel.flag_reasons = list(sel.flag_reasons or []) + ["look_target_missing"]
            except Exception:                                # noqa: BLE001
                pass

        if v.get("verdict") == "replace":
            swapped = False
            failed_wins: list = []      # alternates the verifier explicitly REJECTED on the way
            # Primary pixels already received the strict verdict above.  Promotion adds every
            # strict candidate it actually considers so the bounded neighborhood rung never asks
            # the same source/shot twice.
            strict_tried = {(str(getattr(sel, "source_id", "") or ""),
                             int(getattr(sel, "shot_index", -1)))}
            # Promotion changes many coupled fields (source/window/signals/verifier/beat windows) and
            # overwrites a deterministic clip filename.  Keep one pre-promotion state for the whole
            # ladder so a cut failure restores the actual rejected selection, including alternate
            # Window-QC mutations made while searching.
            _promotion_base_state = copy.deepcopy(vars(sel))
            _promotion_materialization_error = {"detail": ""}
            # The semantic quote-window rung marks hypotheses whose selected bytes are bound to a
            # whole-pool verbatim location.  The ordinary neighborhood verifier used to replace
            # such a hypothesis with a visually convincing adjacent shot that did not contain the
            # quote (measured beat 55: quote window 125.254--130.853 -> shot39 120.587--123.557).
            # The final contract caught the loss and rolled back, but recovery could never install
            # the clean copy.  Preserve that constraint across every promotion: another prebuilt
            # quote candidate already carries its own proof; a same-target PCM candidate may be
            # rebound only when its final Window-QC trim still contains the transferred phrase.
            _quote_locked_signals = dict(getattr(sel, "signals", None) or {})
            _quote_locked = bool(_quote_locked_signals.get("quote_pool_exact"))
            _quote_transfer = _quote_locked_signals.get("quote_audio_transfer_evidence")

            def _preserve_quote_lock(alt) -> bool:
                if not _quote_locked:
                    return True
                alt_signals = dict(getattr(alt, "signals", None) or {})
                # Rebuilt below only when confirmation is bound to this exact candidate/window.
                alt_signals.pop("quote_unprompted_confirmation", None)
                # ``quote_pool_exact`` is a matcher claim, not independent proof.  Prompted ASR may
                # retrieve the location, but it cannot certify a phrase suggested by its own prompt.
                # Re-locate inside THIS candidate's post-Window-QC window, then require the separate
                # no-prompt narrow decoder to confirm the hit and bind its confirmed timestamps.
                try:
                    from . import index as _index_quote_lock
                    from .relevance_contract import (
                        QUOTE_RETRIEVAL_MAX_OCCURRENCES_PER_STREAM as _quote_bound_lock,
                        QUOTE_WINDOW_TOLERANCE_SEC as _quote_tol,
                        _confirm_prompted_quote_span_unprompted as _confirm_quote_lock,
                        _quote_confirmation_summary as _quote_confirmation_summary_lock,
                        _quote_requires_exact_contiguous_match as _quote_exact_required_lock,
                        _prompted_quote_candidate_spans as _quote_candidate_spans_lock,
                    )
                    window0, window1 = float(alt.in_point), float(alt.out_point)
                    quote_text = str(getattr(seg, "quote", "") or "")
                    source_id = str(getattr(alt, "source_id", "") or "")
                    source = proj.source(source_id)
                    exact_required = _quote_exact_required_lock(
                        quote_text, index_module=_index_quote_lock)
                    general_words = _index_quote_lock.load_words(proj, source_id)
                    retrieval_ok, retrieval_streams, _retrieval_reason, retrieval_complete = \
                        _index_quote_lock._load_quote_retrieval_streams_result(
                            proj, source, cfg, require_complete=True)
                    streams = [general_words]
                    if retrieval_ok and retrieval_complete:
                        streams.extend(stream["words"] for stream in retrieval_streams)
                    candidates = []
                    for all_words in streams:
                        # Local-window retrieval prevents a duplicate line elsewhere in the source
                        # from legitimizing this alternate. Tolerance mirrors final containment.
                        local_words = []
                        for row in all_words:
                            try:
                                word0, word1 = float(row[0]), float(row[1])
                            except (IndexError, TypeError, ValueError):
                                continue
                            if word1 >= window0 - float(_quote_tol) \
                                    and word0 <= window1 + float(_quote_tol):
                                local_words.append(row)
                        candidates.extend(_quote_candidate_spans_lock(
                            local_words, quote_text,
                            exact_contiguous_required=exact_required,
                            index_module=_index_quote_lock))
                    unique = []
                    seen = set()
                    for candidate in candidates:
                        key = tuple(round(float(value), 3) for value in candidate)
                        if key not in seen:
                            seen.add(key)
                            unique.append(candidate)
                    direct = None
                    confirmed = None
                    confirmation = {"status": "inconclusive",
                                    "reason": "selected_quote_candidate_absent"}
                    direct_contained = False
                    if len(unique) <= _quote_bound_lock:
                        for candidate in unique:
                            checked = _confirm_quote_lock(
                                proj, source, quote_text, candidate, cfg,
                                exact_contiguous_required=exact_required)
                            checked_span = (checked.get("confirmed_span")
                                            if checked.get("status") == "confirmed" else None)
                            contained = bool(
                                checked_span and window1 > window0 >= 0.0
                                and float(checked_span[1]) > float(checked_span[0]) >= 0.0
                                and float(checked_span[0]) >= window0 - float(_quote_tol)
                                and float(checked_span[1]) <= window1 + float(_quote_tol))
                            if contained:
                                direct, confirmed = candidate, checked_span
                                confirmation, direct_contained = checked, True
                                break
                except Exception as exc:                       # decode/cache uncertainty rejects alt
                    direct = None
                    confirmed = None
                    confirmation = {"status": "inconclusive",
                                    "reason": ("selected_quote_confirmation_failed:"
                                               f"{type(exc).__name__}")}
                    direct_contained = False
                if direct_contained and confirmed is not None:
                    try:
                        dialogue_signal = max(
                            float(alt_signals.get("dialogue", 0.0) or 0.0),
                            float(confirmed[2]))
                    except (IndexError, TypeError, ValueError):
                        return False
                    alt_signals["quote_pool_exact"] = True
                    alt_signals["dialogue"] = dialogue_signal
                    alt_signals["quote_unprompted_confirmation"] = \
                        _quote_confirmation_summary_lock(confirmation)
                    alt.signals = alt_signals
                    return True
                # A same-target PCM transfer remains a separate valid proof path.  Prefer evidence
                # already bound to the alternate; fall back to the primary lock's transfer only
                # when it explicitly targets this source and survives the existing rebind checks.
                alt_transfer = alt_signals.get("quote_audio_transfer_evidence")
                if not isinstance(alt_transfer, dict):
                    alt_transfer = _quote_transfer
                if not isinstance(alt_transfer, dict):
                    return False
                if str(alt_transfer.get("target_source_id", "") or "") != \
                        str(getattr(alt, "source_id", "") or ""):
                    return False
                try:
                    target_q0, target_q1 = (
                        float(alt_transfer["target_quote_span"][0]),
                        float(alt_transfer["target_quote_span"][1]))
                    window0, window1 = float(alt.in_point), float(alt.out_point)
                    from .relevance_contract import QUOTE_WINDOW_TOLERANCE_SEC as _quote_tol
                    contained = (
                        window1 > window0 >= 0.0 and target_q1 > target_q0 >= 0.0
                        and target_q0 >= window0 - float(_quote_tol)
                        and target_q1 <= window1 + float(_quote_tol))
                except (IndexError, KeyError, TypeError, ValueError):
                    contained = False
                if not contained:
                    return False
                try:
                    from . import audio_align as _audio_quote_lock
                    rebound = _audio_quote_lock.rebind_transfer_evidence_window(
                        alt_transfer, [window0, window1])
                except Exception:
                    rebound = {}
                if not rebound:
                    return False
                alt_signals[_audio_quote_lock.AUDIO_QUOTE_TRANSFER_SIGNAL] = rebound
                alt_signals["quote_audio_transfer"] = True
                alt_signals["quote_pool_exact"] = True
                try:
                    alt_signals["dialogue"] = max(
                        float(alt_signals.get("dialogue", 0.0) or 0.0),
                        float(rebound.get("reference_asr_ratio", 0.0) or 0.0))
                except (TypeError, ValueError):
                    return False
                alt.signals = alt_signals
                return True

            def _ordered_promotion_alts(pool=None):
                """Return the exact stable order consumed by one promotion rung."""
                import os as _os_aff
                _alts = list(pool if pool is not None else (sel.alternates or []))
                if pool is None and _exact \
                        and _os_aff.environ.get(
                            "VIDLORE_CLIPSTUDIO_SCENE_AFFINITY", "1").strip() \
                        not in ("0", "false", "no", ""):
                    try:
                        _alts = _scene_affinity_order(_alts, seg, proj, sel.source_id)
                    except Exception:
                        pass
                return _alts

            def _try_promote(downgrade: bool, pool=None, label: str = "",
                             attempt_cap: int | None = None) -> bool:
                """Scan the beat's relevance-ranked alternates and promote the first acceptable one.
                downgrade=False → the ORIGINAL strict promotion (verify at the beat's own strictness,
                accept only an explicit verdict==keep). downgrade=True → the EXACT→CONTEXTUAL rung:
                verify LENIENTLY and accept a right-subject / on-topic clip that simply isn't the
                exact moment (wrong-show/era/character still fail and are skipped). `pool` overrides
                the candidate list (the scene-VENUE rung passes its bounded venue candidates); `label`
                overrides the downgrade tag for honest audit labeling. Returns True on a swap. All
                the production safeguards (reuse-ledger cap, Window-QC, beat_windows rewrite,
                re-cut) are shared by all modes."""
                nonlocal swapped, replaced
                _prior_look_scope = _look_scope["on"]
                # A strict replacement must satisfy every promise the primary was asked, including
                # "look at the dagger".  Contextual downgrade deliberately remains the softer rung
                # and keeps that question off; a missed primary target is routed to exact search /
                # still fallback below rather than silently declared satisfied.
                _look_scope["on"] = not downgrade
                try:
                    # A real-quote recovery may have only one canonical shot while the authored
                    # script deliberately returns to that exact moment more often than the soft
                    # variety cap.  Under-cap candidates retain absolute priority.  Only after
                    # that ordinary bounded walk fails do we consider a previously capped-out
                    # candidate, least-used first.  The second walk re-runs the unchanged strict
                    # vision, Window-QC and quote-containment checks; this is not an approval and
                    # does not change the global cap or any publication threshold.
                    if not downgrade and _exact and _quote_locked:
                        ordered = _ordered_promotion_alts(pool)
                        under_cap = [alt for alt in ordered if _reuse[
                            (str(getattr(alt, "source_id", "") or ""),
                             int(getattr(alt, "shot_index", -1)))] < _reuse_cap]
                        if _try_promote_inner(
                                False, under_cap, label, attempt_cap,
                                allow_reuse_overflow=False):
                            return True
                        if _promotion_materialization_error["detail"]:
                            return False
                        original_rank = {id(alt): rank for rank, alt in enumerate(ordered)}
                        over_cap = [alt for alt in ordered if _reuse[
                            (str(getattr(alt, "source_id", "") or ""),
                             int(getattr(alt, "shot_index", -1)))] >= _reuse_cap]
                        over_cap.sort(key=lambda alt: (
                            _reuse[(str(getattr(alt, "source_id", "") or ""),
                                    int(getattr(alt, "shot_index", -1)))],
                            original_rank.get(id(alt), len(ordered))))
                        if over_cap:
                            return _try_promote_inner(
                                False, over_cap, label, attempt_cap,
                                allow_reuse_overflow=True)
                        return False
                    return _try_promote_inner(
                        downgrade, pool, label, attempt_cap,
                        allow_reuse_overflow=False)
                finally:
                    _look_scope["on"] = _prior_look_scope

            def _try_promote_inner(downgrade: bool, pool=None, label: str = "",
                                   attempt_cap: int | None = None, *,
                                   allow_reuse_overflow: bool = False) -> bool:
                nonlocal swapped, replaced, _errored, _materialization_errors
                tried = 0
                _attempt_cap = max_replacements if attempt_cap is None else max(0, int(attempt_cap))
                # SCENE-AFFINITY ordering for exact beats — try same-scene sources first (see
                # _scene_affinity_order). Ordering only; every gate below still applies.
                _alts = _ordered_promotion_alts(pool)
                for alt in _alts:
                    if tried >= _attempt_cap:
                        break
                    tried += 1
                    if not downgrade:
                        strict_tried.add((str(getattr(alt, "source_id", "") or ""),
                                          int(getattr(alt, "shot_index", -1))))
                    ashot = get_shot(alt.source_id, alt.shot_index)
                    if ashot is None:
                        continue
                    anames = ashot.face_ids or []
                    # Finalize/validate the candidate trim BEFORE vision.  Window-QC may shorten
                    # ``alt.in_point/out_point``; judging first and then binding the verdict to the
                    # shortened tuple would falsely claim the verifier saw pixels it never saw.
                    import os as _os_w
                    if _os_w.environ.get("VIDLORE_CLIPSTUDIO_WINDOW_QC", "1").strip() \
                            not in ("0", "false", "no"):
                        from .match import validate_candidate_window, _wqc_log_line
                        _wshots = getattr(get_shot, "all_shots", lambda _s: [])(alt.source_id)
                        _wact, _wwhy, _wmeta = validate_candidate_window(
                            alt, ashot, _wshots, cfg, seg)
                        if _wact == "rejected":
                            log(f"window-qc: rejected verify-promotion seg{sel.segment_index} "
                                f"alt={alt.source_id[:28]} "
                                f"{_wqc_log_line(_wact, _wmeta, _wwhy)}")
                            failed_wins.append((alt.source_id, float(alt.in_point)))
                            continue
                        if _wact == "shortened":
                            log(f"window-qc: shortened verify-promotion seg{sel.segment_index} "
                                f"{_wqc_log_line(_wact, _wmeta, _wwhy)}")
                    if not _preserve_quote_lock(alt):
                        # A visually plausible neighbour without the phrase is not a valid quote
                        # recovery candidate.  Reject it before spending vision; the unchanged
                        # final contract remains the authoritative backstop.
                        failed_wins.append((alt.source_id, float(alt.in_point)))
                        continue
                    if _breaker_open:
                        break                           # backend is down — promotion cannot verify
                    _alt_strict = False if downgrade else _exact
                    _alt_must_see = _must_see(seg)
                    av, _av_used_sheet = _cached_verify_ctx(
                        ashot.keyframe_path, ashot, seg, _alt_strict, anames,
                        (alt.in_point, alt.out_point),
                        rung=("strict_scene_neighborhood"
                              if label == "strict_scene_neighborhood" and not downgrade else
                              ("venue" if pool is not None else
                              ("contextual" if downgrade else "strict_promote"))))
                    if av is None:
                        continue                        # transport error, NOT a judgment
                    # ONE judge, called from here and from anywhere that must judge identically.
                    _decision = strict_window_verdict(
                        av, seg, alt, proj, cfg, _char2actor, downgrade=downgrade,
                        exact=_exact, character=_character, must_see=_alt_must_see)
                    _accept = _decision["accept"]
                    if not _accept:
                        # an explicit non-keep judgment (av None = transport error, handled above)
                        failed_wins.append((alt.source_id, float(alt.in_point)))
                        continue
                    # REUSE LEDGER — do not promote a look that already airs on >= cap beats (that is
                    # how one clip got re-aired 9×). Skip to the next relevance-ranked alternate; if
                    # none survive, the beat stays flagged and image-fallback gives it a DISTINCT still.
                    _reuse_before = _reuse[(alt.source_id, alt.shot_index)]
                    _overflow_contract = bool(
                        allow_reuse_overflow and not downgrade and _exact and _quote_locked
                        and _reuse_before >= _reuse_cap)
                    if (_reuse[(alt.source_id, alt.shot_index)] >= _reuse_cap
                            and not _overflow_contract):
                        failed_wins.append((alt.source_id, float(alt.in_point)))
                        continue
                    if allow_reuse_overflow and not _overflow_contract:
                        # The deferred pool is constructed only from capped-out exact-quote
                        # candidates.  If shared state ever makes that premise stale, fail closed
                        # instead of turning this path into an ordinary reorder/bypass.
                        continue
                    # Snapshot the deterministic clip bytes BEFORE any selection mutation. A failed
                    # ffmpeg invocation may leave a non-empty partial at this same filename, so the
                    # return value alone is not enough to make rollback possible.
                    _clip = proj.clips_dir / f"seg_{int(sel.segment_index):03d}.mp4"
                    _clip_existed = _clip.is_file()
                    _clip_backup = None
                    try:
                        if materialize_promotions and _clip_existed:
                            _clip_backup = _clip.read_bytes()

                        # Promote the alternate into the selection, but do not update reuse/replaced
                        # accounting or emit a success log until its clip is proven materialized.
                        old_sid, old_shot, old_in, old_out = (
                            sel.source_id, sel.shot_index, sel.in_point, sel.out_point)
                        _old_key = (sel.source_id, sel.shot_index)
                        sel.source_id = alt.source_id
                        sel.shot_index = alt.shot_index
                        sel.in_point = alt.in_point
                        sel.out_point = alt.out_point
                        sel.signals = (dict(alt.signals or {})
                                       if _overflow_contract else alt.signals)
                        if _overflow_contract:
                            sel.signals[REUSE_CAP_OVERFLOW_EXACT_CONTRACT] = 1.0
                        sel.confidence = alt.score
                        sel.source_url = (proj.source(alt.source_id).url
                                          if proj.source(alt.source_id) else "")
                        sel.identity = (anames[0] if anames else "")
                        # build_video plays the scene's beats from beat_windows (rejected pick is
                        # FIRST there) — drop it AND every alternate the verifier explicitly failed
                        # on the way here, then lead with the promoted window.
                        new_win = [alt.source_id, round(alt.in_point, 3), round(alt.out_point, 3)]
                        kept = [w for w in (sel.beat_windows or [])
                                if not (w and w[0] == old_sid
                                        and abs(float(w[1]) - float(old_in)) < 0.05)
                                and not (w and w[0] == new_win[0]
                                        and abs(float(w[1]) - new_win[1]) < 0.05)
                                and not any(w and w[0] == fs
                                            and abs(float(w[1]) - fi) < 0.05
                                            for fs, fi in failed_wins)]
                        sel.beat_windows = [new_win] + kept
                        av["status"] = "ok"
                        av["replaced_from"] = {
                            "source_id": str(old_sid or ""),
                            "shot": int(old_shot),
                            "in_point": round(float(old_in), 3),
                            "out_point": round(float(old_out), 3),
                        }
                        if downgrade:
                            av["verdict"] = "keep"
                            av["downgraded"] = label or "exact→contextual"
                            av["relevance_class"] = "contextual_fallback"
                        if _overflow_contract:
                            # Do not write this selection-specific fact into the reusable vision
                            # cache. `_cached_verify_ctx` returns a copy, so the marker belongs only
                            # to the promoted selection/verifier audit record.
                            av[REUSE_CAP_OVERFLOW_EXACT_CONTRACT] = True
                            av["reuse_count_before"] = int(_reuse_before)
                            av["reuse_cap"] = int(_reuse_cap)
                        _bind_evidence(av, sel, seg, ashot, _alt_strict, anames,
                                       _av_used_sheet, _alt_must_see)
                        sel.verifier = av
                        if materialize_promotions:
                            made = _cut.cut_selection(proj, sel, cfg)
                            made_path = Path(made) if made is not None else None
                            if (made_path is None or not made_path.is_file()
                                    or made_path.stat().st_size <= 0):
                                raise RuntimeError("promotion cut returned no complete clip")
                            sel.clip_path = str(made_path)
                    except Exception as exc:             # noqa: BLE001 — disk transaction rollback
                        vars(sel).clear()
                        vars(sel).update(copy.deepcopy(_promotion_base_state))
                        _restore_error = ""
                        if materialize_promotions:
                            try:
                                if _clip_existed:
                                    _tmp_restore = _clip.with_suffix(
                                        _clip.suffix + ".verify_rollback.tmp")
                                    try:
                                        _tmp_restore.write_bytes(_clip_backup or b"")
                                        _tmp_restore.replace(_clip)
                                    finally:
                                        _tmp_restore.unlink(missing_ok=True)
                                else:
                                    _clip.unlink(missing_ok=True)
                            except Exception as restore_exc:  # noqa: BLE001 — remain inconclusive
                                _restore_error = (
                                    f"; clip rollback failed: {type(restore_exc).__name__}: "
                                    f"{restore_exc}")
                        _materialization_errors += 1
                        _errored += 1
                        _promotion_materialization_error["detail"] = (
                            f"{type(exc).__name__}: {exc}{_restore_error}")
                        if persist_project:
                            try:
                                proj.save()
                            except Exception as save_exc:  # noqa: BLE001 — summary stays technical
                                _promotion_materialization_error["detail"] += (
                                    f"; metadata rollback save failed: "
                                    f"{type(save_exc).__name__}: {save_exc}")
                        log(f"verify: seg{sel.segment_index} promotion MATERIALIZATION ERROR — "
                            f"rolled back selection/clip; "
                            f"{_promotion_materialization_error['detail']}")
                        return False
                    _reuse[(alt.source_id, alt.shot_index)] += 1   # this look now airs one more time
                    if _reuse[_old_key] > 0:
                        _reuse[_old_key] -= 1                       # the replaced pick no longer airs here
                    replaced += 1
                    swapped = True
                    if _overflow_contract:
                        log(f"verify: seg{sel.segment_index} "
                            f"{REUSE_CAP_OVERFLOW_EXACT_CONTRACT} → "
                            f"{alt.source_id}#{alt.shot_index} "
                            f"(reuse {_reuse_before}/{_reuse_cap}; all ordinary under-cap "
                            "strict candidates failed)")
                    else:
                        log(f"verify: seg{sel.segment_index} "
                            f"{'exact→contextual' if downgrade else 'replaced'} → "
                            f"{alt.source_id}#{alt.shot_index}")
                    return True
                return False

            # The ordinary pass deliberately stops after ``max_replacements`` candidates to keep
            # full-project verification bounded.  A scoped publication-recovery pass is different:
            # it is already operating on a small page of proven blockers, before paying for new
            # discovery, and match's bounded ``alternates`` may contain a strict-passing frame just
            # below that latency cap.  Measured on portal job ee93371e41 beat 125, the sixth retained
            # alternate was an unused native-HD, strict-verified Tywin trial reaction while the
            # three-candidate pass settled on contextual footage and release-blocked.  Exhaust the
            # existing bounded head only in this scoped lane (hard cap 12).  Every candidate still
            # faces the unchanged strict vision, Window-QC, quote lock, reuse and materialization
            # gates; normal verification and all acceptance thresholds are untouched.
            _head_attempt_cap = max_replacements
            if strict_pool_recovery:
                _head_attempt_cap = max(
                    max_replacements,
                    min(12, len(getattr(sel, "alternates", None) or [])))
            _try_promote(downgrade=False, attempt_cap=_head_attempt_cap)
            if _promotion_materialization_error["detail"]:
                # A cut/rollback failure is infrastructure uncertainty, not evidence that the
                # accepted alternate was semantically bad. Do not run contextual/filler rungs or
                # write a semantic rejection; the summary's errored/materialization counts make
                # every scoped/full caller retry this beat.
                continue

            # STRICT SCENE-NEIGHBORHOOD EXPANSION. Match may retain the correct source but only a
            # neighboring shot from it; global variety/de-dup ranking should not be changed to fix
            # that local verifier miss. Search a bounded ±6 neighborhood around the selection,
            # alternates and deep-bench seeds from scene-affine sources, ordered by persisted CLIP
            # when available. Crucially this reuses `_try_promote`: strict vision, exact selected-
            # window sheets, Window-QC, reuse cap and clip transaction are byte-for-byte the same.
            # The rung owns a cap independent of `max_replacements=3`; inheriting that shallow cap
            # would stop before measured exact moments at ranks 5-6.
            _neighborhood_on = _os_ms.environ.get(
                "VIDLORE_CLIPSTUDIO_STRICT_NEIGHBORHOOD", "1").strip() \
                not in ("0", "false", "no", "")
            _character_pool_subject_miss = bool(
                strict_pool_recovery
                and _policy.policy_of(seg) == _policy.CHARACTER
                and str(getattr(seg, "required_entity", "") or "").strip()
                and (v.get("correct_subject_visible") is False
                     or v.get("wrong_subject_visible") is True))
            if not swapped and (_exact or _character_pool_subject_miss) and _neighborhood_on:
                try:
                    _n_cap = max(0, min(24, int(_os_ms.environ.get(
                        "VIDLORE_CLIPSTUDIO_STRICT_NEIGHBORHOOD_CANDS", "12") or 12)))
                    _n_radius = max(1, min(12, int(_os_ms.environ.get(
                        "VIDLORE_CLIPSTUDIO_STRICT_NEIGHBORHOOD_RADIUS", "6") or 6)))
                    _n_sources = max(1, min(8, int(_os_ms.environ.get(
                        "VIDLORE_CLIPSTUDIO_STRICT_NEIGHBORHOOD_SOURCES", "4") or 4)))
                except (TypeError, ValueError):
                    _n_cap, _n_radius, _n_sources = 12, 6, 4
                try:
                    _npool = _strict_scene_neighborhood_candidates(
                        sel, seg, proj, get_shot, cfg, exclude=strict_tried,
                        beat_era=_era_of(seg), cap=_n_cap, radius=_n_radius,
                        source_cap=_n_sources,
                        allow_indexed_pool_sources=bool(strict_pool_recovery))
                except Exception as _n_exc:              # fail closed; old ladder remains intact
                    _npool = []
                    log(f"verify: seg{sel.segment_index} strict scene-neighborhood unavailable "
                        f"({type(_n_exc).__name__})")
                if _npool:
                    _neighborhood_kind = ("subject" if _character_pool_subject_miss
                                          and not _exact else "scene")
                    log(f"verify: seg{sel.segment_index} strict {_neighborhood_kind}-neighborhood "
                        "— trying "
                        f"{len(_npool)} unseen candidate(s) within ±{_n_radius} shots")
                    if _try_promote(downgrade=False, pool=_npool,
                                    label="strict_scene_neighborhood",
                                    attempt_cap=_n_cap):
                        if _exact:
                            log(f"verify: seg{sel.segment_index} rescued by strict "
                                "scene-neighborhood — exact scene found, no contextual downgrade")
                        else:
                            log(f"verify: seg{sel.segment_index} rescued by strict "
                                "subject-neighborhood — required subject confirmed")
                if _promotion_materialization_error["detail"]:
                    continue

            # A concrete silent action may be much farther than +/-6 edits from every retained
            # seed even when the selected upload is the correct full scene.  Only the scoped
            # publication-recovery lane may spend this final five-call probe, and only after the
            # local rung above failed.  Candidate generation is confined to one native-HD source
            # bound to the beat's anchor episode; `_try_promote` keeps every acceptance gate shared.
            _whole_action_probe = bool(
                not swapped and _exact and strict_pool_recovery and _neighborhood_on
                and str(getattr(seg, "shot_intent", "") or "").strip().lower() == "action")
            if _whole_action_probe:
                try:
                    _action_pool = _strict_scene_neighborhood_candidates(
                        sel, seg, proj, get_shot, cfg, exclude=strict_tried,
                        beat_era=_era_of(seg), cap=5, radius=_n_radius, source_cap=1,
                        allow_indexed_pool_sources=True, whole_source_probe=True)
                except Exception as _action_exc:         # fail closed; publication gate still blocks
                    _action_pool = []
                    log(f"verify: seg{sel.segment_index} strict whole-source action probe "
                        f"unavailable ({type(_action_exc).__name__})")
                if _action_pool:
                    log(f"verify: seg{sel.segment_index} strict whole-source action probe — "
                        f"trying {len(_action_pool)} unseen disjoint candidate(s)")
                    if _try_promote(downgrade=False, pool=_action_pool,
                                    label="strict_scene_neighborhood", attempt_cap=5):
                        log(f"verify: seg{sel.segment_index} rescued by strict whole-source "
                            "action probe — exact action found")
                if _promotion_materialization_error["detail"]:
                    continue

            # EXACT→CONTEXTUAL DOWNGRADE (relevance hierarchy: exact → contextual_fallback → filler).
            # The strict verifier rejected every candidate for not being the EXACT moment — but a clip
            # whose REQUIRED SUBJECT is confirmed on screen is a legitimate contextual fallback (a
            # right-character/scene moving clip beats a frozen still and never black-blocks). Prefer
            # keeping the ORIGINAL pick (already cut — no re-cut) when its subject is confirmed; else
            # promote the first alternate whose subject is confirmed. A clip whose subject is WRONG
            # (correct_subject_visible False) is CONTRADICTORY — it is never downgraded and falls
            # through to the honest still / release-block below. env-gated (default ON).
            _downgrade_on = _os_ms.environ.get(
                "VIDLORE_CLIPSTUDIO_EXACT_CONTEXTUAL_DOWNGRADE", "1").strip() \
                not in ("0", "false", "no")
            # DEEP BENCH — only for beats whose ORIGINAL pick is genuinely bad.
            #
            # The strict pass above sees just 6 alternates out of a ~4000-shot pool, so when none is
            # the exact moment the code below keeps the original and relabels it contextual_fallback:
            # 113 of 268 beats on job 69d80e9dd4_v4, averaging 4.34 on the frame eval against 5.92
            # for beats that got a real replacement. Match now keeps a deeper ranked bench for those.
            #
            # But it must NOT be tried on a pick that is already fine. Measured over the 11 beats the
            # bench first rescued: the group went 4.18 -> 5.36, and the two deictic beats the video
            # turns on went 1 -> 6 ("keep your eye on the dagger") and 2 -> 9 ("watch the trial the
            # way Bran watched it") — but two beats already scoring 9 fell to 4 and 5, because the
            # verifier only knows "not the exact scene", not "already good", and swapped a strong
            # clip for a strict-passing weaker one. `_contextual_subject_ok` separates the two cases
            # exactly: the rescued beats failed it, the regressed ones passed it. So the bench is
            # reached only down the branch where the original has ALREADY been judged unusable.
            # ~14 extra vision calls per affected beat (~$0.007 each at the measured rate).
            # Kill switch: VIDLORE_CLIPSTUDIO_DEEP_BENCH=0.
            def _wrong_subject_rescue() -> bool:
                """Reach past the bench when the required subject is not on screen.

                THE BENCH CANNOT HOLD WHAT RETRIEVAL COULD NOT SEE. Measured on job ee93371e41
                beat 134, on real extracted frames: the shipped pick is a daylight Tyrion close-up
                with no Shae, and every one of the 60 bench candidates was viewed — none shows Shae
                or Tywin's bedchamber, so no ordering of that bench can satisfy the beat. The shot
                that literally shows "Shae in his father's bed" sits at CLIP rank 563 of 4942 for
                this beat's query because it is a near-black night interior. The bench is match's
                one-candidate-per-source ranking over the top CLIP-ranked sources, so a shot
                retrieval cannot see is structurally absent from it.

                Deliberately NOT placed inside `_deep_bench`: that runs only under `_exact`, and
                this beat is `character_specific` (required_kind 'montage'), so a rescue living
                there could never run for it — my first attempt did exactly that and was dead code
                for the case it was written for. A wrong-subject verdict is policy-blind, and a
                character beat is if anything MORE likely to name a person the pick lacks.

                Venue first: the neighbourhood pool searches around the wrong pick's own scene
                seeds. Measured on this beat, all 8 venue candidates come from the source that
                actually holds the narrated image; none of the 12 neighbourhood ones do.

                `_try_promote` runs at downgrade=False, the identical strict bar — this changes
                what is EXAMINED, never what is admitted, and a beat with genuinely no right
                footage still blocks.
                """
                _want = _subject_terms(seg, _char2actor)
                _bench_ids = {id(c) for c in (getattr(sel, "deep_alternates", None) or [])}
                _rungs = [("venue", lambda: _venue_candidates(
                    sel, seg, proj, get_shot, _era_of(seg)))]
                # The neighbourhood pool opens the indexed pool and belongs to the scoped recovery
                # stage — ordinary verification deliberately does not reach it, and
                # test_normal_character_verification_does_not_open_indexed_pool_neighborhood pins
                # that. Costs this rescue nothing: on the motivating beat all 8 venue candidates
                # come from the source holding the narrated image and none of the neighbourhood
                # ones do.
                if strict_pool_recovery:
                    _rungs.append(
                        ("scene-neighbourhood", lambda: _strict_scene_neighborhood_candidates(
                            sel, seg, proj, get_shot, cfg, beat_era=_era_of(seg),
                            allow_indexed_pool_sources=True)))
                for _label, _mk in _rungs:
                    try:
                        _pool = [c for c in (_mk() or []) if id(c) not in _bench_ids]
                    except Exception as _exc:            # noqa: BLE001 — fail closed
                        log(f"verify: seg{sel.segment_index} wrong-subject {_label} pool "
                            f"unavailable ({type(_exc).__name__})")
                        continue
                    if not _pool:
                        continue
                    if _want:
                        _pool.sort(key=lambda c: _subject_affinity(c, _want, proj), reverse=True)
                    log(f"verify: seg{sel.segment_index} required subject not on screen and the "
                        f"bench had no answer — examining {len(_pool)} {_label} candidate(s) at "
                        f"the same strict bar")
                    if _try_promote(downgrade=False, pool=_pool,
                                    label=f"wrong_subject_{_label}"):
                        log(f"verify: seg{sel.segment_index} rescued from the {_label} pool — "
                            f"required subject found, no downgrade")
                        return True
                return False

            def _deep_bench() -> bool:
                if _os_ms.environ.get("VIDLORE_CLIPSTUDIO_DEEP_BENCH", "1").strip() \
                        in ("0", "false", "no"):
                    return False
                _bench = list(getattr(sel, "deep_alternates", None) or [])
                # LOOK-MISS ordering: when the narration's named target is absent from the pick,
                # surface bench candidates whose match-time CLIP probe saw the target (the
                # target_vis signal) first — the strict bar still decides, this is order only.
                if _bench and _must_see(seg) and v.get("target_visible") is False:
                    _bench.sort(key=lambda c: float(
                        (getattr(c, "signals", None) or {}).get("target_vis", 0.0)), reverse=True)
                # WRONG-SUBJECT ordering: the same idea for the other way a pick can be wrong about
                # WHO is on screen. When the verifier reports the beat's required subject absent and
                # a different one present, the bench is very often holding the right person — it is
                # simply further down a ranking that knows nothing about the verdict that was just
                # returned. Measured, job ee93371e41 beat 134 (required_entity 'Shae', verdict
                # correct_subject_visible=False / wrong_subject_visible=True): 24 of its 61
                # candidates come from Shae/Tywin-titled sources, including the S04E06 Shae scene,
                # and the beat still shipped an Oberyn compilation. Nothing was missing from the
                # bench; nothing put the right people at the front of it.
                # Ordering ONLY, exactly like the look-miss rule above: `_try_promote` still applies
                # the same strict bar to whatever it reaches, so this can change which candidate is
                # examined first and can never admit one that would otherwise be refused.
                _wrong_subject = v.get("correct_subject_visible") is False
                if _bench and _wrong_subject:
                    _want = _subject_terms(seg, _char2actor)
                    if _want:
                        _bench.sort(key=lambda c: _subject_affinity(c, _want, proj), reverse=True)
                if _bench and _try_promote(downgrade=False, pool=_bench):
                    log(f"verify: seg{sel.segment_index} rescued from the deep bench "
                        f"({len(_bench)} extra candidate(s)) — exact scene found, no downgrade")
                    return True

                return False

            # A beat that POINTS at something cannot be satisfied by "the right people are here".
            # "Keep your eye on the dagger" over a clip with no dagger is the failure the owner
            # raises most, and the contextual downgrade below is exactly what lets it through: the
            # required subject IS on screen, so the beat is kept and relabelled. When the narration
            # names a target and the verifier could not see it, refuse that shortcut and let the
            # deep bench / alternates look for footage that actually shows it.
            _look_for = _must_see(seg)
            _look_missed = bool(_look_for) and v.get("target_visible") is False
            if _look_missed:
                log(f"verify: seg{sel.segment_index} narration points at {_look_for!r} and it is "
                    f"NOT on screen — no contextual downgrade, searching for footage that shows it")
                # the still pass reads this: when nothing in the pool shows the named thing, a frame
                # OF THAT MOMENT beats moving footage of the wrong one
                try:
                    if "look_target_missing" not in (sel.flag_reasons or []):
                        sel.flag_reasons = list(sel.flag_reasons or []) + ["look_target_missing"]
                except Exception:                                # noqa: BLE001
                    pass

            if not swapped and _exact and _downgrade_on:
                def _keep_contextual(why: str) -> None:
                    v["verdict"] = "keep"
                    v["downgraded"] = "exact→contextual"
                    v["relevance_class"] = "contextual_fallback"
                    sel.verifier = v
                    log(f"verify: seg{sel.segment_index} exact→contextual downgrade ({why})")

                _orig_ok = _exact_contextual_ok(v, seg, _src_title_of(sel), _char2actor)
                if _look_missed and _orig_ok:
                    # The named thing is not on screen — but the pick is otherwise usable, so it is
                    # NOT thrown away. Measured twice: routing these beats to the deep bench moved 4
                    # of them from 7.25 to 3.25 (8/9/9 -> 4/2/3), because the bench's "strict-
                    # passing" replacement was worse than what it discarded and the eval disagreed
                    # with the verifier's strict judgment. That is the same rule the bench already
                    # follows — never replace a pick that clears the contextual bar — and the look
                    # gate must not carve an exception out of it. The owner's own rule says the same:
                    # "agar koi exact scene dastyab na ho, to wo uski zid na kare".
                    # The beat still carries look_target_missing, so the still pass covers the moment
                    # with a frame OF it, which is where the real gain is.
                    _keep_contextual(f"{_look_for!r} is not on screen and no better clip exists — "
                                     f"kept; still pass will cover the moment")
                    replaced += 1
                    swapped = True
                elif _look_missed:
                    # original is unusable AND the target is missing → worth searching
                    if not _deep_bench():
                        _try_promote(downgrade=True)
                elif _orig_ok:
                    _keep_contextual("required subject on screen — kept, honestly labeled "
                                     "contextual_fallback")
                    replaced += 1
                    swapped = True
                elif not _deep_bench():              # original unusable → reach for the deep bench
                    _try_promote(downgrade=True)     # then scan alternates for a right-subject clip

            # SCENE-VENUE CONTEXTUAL EXPANSION — the rung between "no alternate passes" and the
            # honest gap. The alternates come from match's visual ranking, so when a beat cites a
            # MICRO-moment whose footage simply isn't in the pool (measured: 'a maester examines
            # the necklace at trial' — no downloaded source contains the testimony; the word
            # 'strangler' appears only in an essay upload's ASR), every alternate is a wrong-scene
            # candidate and all rungs above correctly refuse. But the SCENE the moment belongs to
            # (the trial itself) IS in the pool — its uploads just never entered this beat's
            # alternates because they share one query token and rank low visually. What a human
            # editor airs there is the venue: the verified scene the narration's moment happens
            # inside. So: find the ANCHOR scene this beat's scene_query points at (>=1 shared
            # scene token), build a bounded candidate pool from sources matching THAT anchor
            # (anchor_verified or >=2 anchor-token title match, era non-conflicting), and run the
            # SAME contextual promotion over it — lenient vision verdict, _contextual_subject_ok
            # acceptance, reuse cap, window-QC. No gate is weakened: a shot that doesn't
            # positively show the right subject/scene is still refused, and the beat still
            # release-blocks. Env VIDLORE_CLIPSTUDIO_VENUE_FALLBACK=0 disables.
            _venue_on = _os_ms.environ.get("VIDLORE_CLIPSTUDIO_VENUE_FALLBACK", "1").strip() \
                not in ("0", "false", "no")
            if not swapped and _exact and _downgrade_on and _venue_on:
                try:
                    _vpool = _venue_candidates(sel, seg, proj, get_shot, _era_of(seg))
                except Exception:
                    _vpool = []
                if _vpool:
                    log(f"verify: seg{sel.segment_index} venue fallback — trying "
                        f"{len(_vpool)} scene-venue candidate(s) from anchor-affine sources")
                    _try_promote(downgrade=True, pool=_vpool, label="exact→venue_contextual")

            # CHARACTER beat, subject PRESENT-BUT-UNCONFIRMED. A character beat whose exact-moment
            # footage was rejected with correct_subject_visible=False is normally left unresolved (a
            # wrong-character read is contradictory). But when the shot is the RIGHT scene/era with no
            # DIFFERENT character identified, the required person is almost certainly present off-face
            # (a wide / reaction shot) — a legitimate contextual fallback. _present_unconfirmed_ok
            # fails CLOSED unless there is a POSITIVE same-era confirmation and no wrong Face-ID, so a
            # confirmed wrong character or a wrong/unconstrained-era source still blocks.
            if not swapped and _exact and _downgrade_on \
                    and (getattr(seg, "required_kind", "") or "").lower() in ("character", "actor"):
                _src_r = proj.source(sel.source_id)
                _src_title = ((getattr(_src_r, "title", "") or "") + " " + (sel.source_id or ""))
                if _present_unconfirmed_ok(v, seg, _src_title, faceid_names,
                                           _era_of(seg), _ok_toks, char2actor=_char2actor):
                    # right scene/era, subject present-but-unconfirmed → CONTEXTUAL
                    v["verdict"] = "keep"
                    v["downgraded"] = "exact→contextual(present-unconfirmed)"
                    v["relevance_class"] = "contextual_fallback"
                    sel.verifier = v
                    replaced += 1
                    swapped = True
                    log(f"verify: seg{sel.segment_index} exact→contextual (character present-"
                        f"unconfirmed; right scene/era, no wrong character — contextual_fallback)")
            # EXACT→GENERIC-FILLER — the last rung before an honest gap, and the one that used to
            # give everything away. It had two holes, both of which aired footage on NO new evidence:
            #
            #   * a CHARACTER beat was kept whenever `not _confirmed_wrong_character(...)` — i.e. on
            #     the ABSENCE of an accusation. With Face-ID unable to resolve Joffrey/Varys/Pycelle
            #     that was true of every frame in existence.
            #   * a NON-CHARACTER beat was kept unconditionally, by relabelling the SAME verdict `v`
            #     that had just said "replace". The verifier's rejection was overwritten with
            #     "keep" and shipped as "honestly labeled" filler.
            #
            # Now: exact and contextual must be genuinely exhausted (both promotion passes ran and
            # swapped nothing), AND a FRESH LENIENT verdict on the footage that would actually air
            # must POSITIVELY prove it is on-topic, same-show/era, non-contradictory and worth
            # airing. No proof → exact_scene_missing → still/hold/manual review → release-block.
            # This does not touch genuinely generic narration: a beat whose own policy is
            # generic_filler never reaches here (it is not `_exact`).
            _filler_on = _os_ms.environ.get(
                "VIDLORE_CLIPSTUDIO_GENERIC_FILLER_DOWNGRADE", "1").strip() \
                not in ("0", "false", "no")
            if not swapped and _exact and _downgrade_on and _filler_on:
                _fresh = None
                _fresh_used_sheet = False
                if not _breaker_open:
                    _fresh, _fresh_used_sheet = _cached_verify_ctx(
                        kf, shot, seg, False, faceid_names,
                        (sel.in_point, sel.out_point),
                        rung="lenient_filler")                               # LENIENT re-ask
                _ok_f, _why_f = _generic_filler_ok(
                    _fresh, seg, _src_title_of(sel), faceid_names, _era_of(seg),
                    _ok_toks, _char2actor)
                if _ok_f:
                    v = dict(_fresh)
                    v["status"] = "ok"
                    v["verdict"] = "keep"
                    v["downgraded"] = "exact→generic_filler"
                    v["relevance_class"] = "generic_filler"
                    v["filler_evidence"] = _why_f
                    _bind_evidence(v, sel, seg, shot, False, faceid_names,
                                   _fresh_used_sheet, _must_see(seg))
                    sel.verifier = v
                    replaced += 1
                    swapped = True
                    log(f"verify: seg{sel.segment_index} exact→generic_filler — a fresh lenient "
                        f"pass PROVES relevance ({_why_f}); honestly labeled")
                else:
                    log(f"verify: seg{sel.segment_index} generic-filler fallback REFUSED — "
                        f"{_why_f}; the beat stays unresolved rather than airing unproven footage")

            # Last chance before the beat is written off. Policy-independent by design: see
            # _wrong_subject_rescue.
            if not swapped and v.get("correct_subject_visible") is False:
                try:
                    if _wrong_subject_rescue():
                        replaced += 1
                        swapped = True
                except Exception as _wsr_exc:            # noqa: BLE001 — never break the ladder
                    log(f"verify: seg{sel.segment_index} wrong-subject rescue unavailable "
                        f"({type(_wsr_exc).__name__})")

            if not swapped:
                failed += 1
                if "verifier_failed" not in sel.flag_reasons:
                    sel.flag_reasons.append("verifier_failed")
                # EXACT-SCENE MISSING (req. 9): an exact_scene beat with no passing real footage AND no
                # relevant contextual clip must be marked for MANUAL REVIEW — the image-fallback will
                # NOT silently cover it with a web/AI image or loose filler (only a real source-frame of
                # the exact scene may), and build release-blocks rather than air contradictory footage.
                if _exact and FLAG_EXACT_MISSING not in sel.flag_reasons:
                    sel.flag_reasons.append(FLAG_EXACT_MISSING)
                    log(f"verify: seg{sel.segment_index} EXACT-SCENE MISSING → manual review "
                        f"(no exact footage AND no relevant contextual clip — only contradictory)")
                sel.flagged = True
                log(f"verify: seg{sel.segment_index} FAILED, no passing alternate")
        if _vcache_dirty:
            # INCREMENTAL atomic save — a kill mid-verify must not lose the rung verdicts already
            # paid for (they are exactly what makes a retry/resume cheap). Same writer, same file.
            _save_verdict_cache(proj, _vcache)
            _vcache_dirty = 0
        if progress and sel.segment_index % 10 == 0:
            log(f"verify: {verified} checked, {replaced} replaced, {failed} unresolved")

    if persist_project:
        proj.save()
    if len(_vcache) != _vcache_n0:
        _save_verdict_cache(proj, _vcache)
    _attempted = verified + _errored
    log(f"verify: done — {verified} verified ({_reused} reused), {_errored} ERRORED"
        + (f" ({_skipped_breaker} skipped, breaker open)" if _skipped_breaker else "")
        + f", {replaced} replaced, {failed} unresolved")
    if _fp_mismatch:
        log(f"verify: {_fp_mismatch} verdict(s) not cached — the contact-sheet build fell back to a "
            f"single frame, so the answer does not match the key")
    # LIVENESS. Reported as its own fact, never folded into 'unresolved': a run that checked
    # nothing must not read like a run that found nothing wrong. This is a SECOND line of defence —
    # the primary one is per-beat (an unverifiable exact beat is already unresolved above), because
    # a global ratio alone would happily pass a render whose 20% of failures were all exact beats.
    if _errored:
        log(f"verify: ⚠ {_errored}/{_attempted} beats could not be verified "
            f"(backend errors){' — CIRCUIT BREAKER OPEN' if _breaker_open else ''}")
    return {"verified": verified, "replaced": replaced, "failed": failed, "available": True,
            "errored": _errored, "reused": _reused, "attempted": _attempted,
            "materialization_errors": _materialization_errors,
            "verifier_down": bool(_breaker_open), "breaker_skipped": _skipped_breaker,
            "vision_config": _vmodel,
            "verified_frac": (verified / _attempted) if _attempted else 1.0}
