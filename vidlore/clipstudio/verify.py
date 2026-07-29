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
import json
import re
from pathlib import Path

from .models import (ClipProject, ScriptSegment, ClipSelection, FLAG_EXACT_MISSING,
                     FLAG_VERIFIER_UNVERIFIED)
from .config import ClipConfig
from . import index as _index
from . import cut as _cut
from . import policy as _policy
from . import era as _era

_VSYS = (
    "You are a strict film-footage QC editor. You judge whether ONE clip's representative frame "
    "correctly illustrates a narration line. Be skeptical: if the specific person/character/object "
    "the line is about is not clearly visible, or the frame is blurry, a title card, a watermark, or "
    "only loosely related, you must fail it. Reply with ONLY a JSON object."
)


def _img_block(path: Path) -> dict:
    data = base64.standard_b64encode(Path(path).read_bytes()).decode()
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}}


def verify_frame(keyframe_path, narration: str, required_entity: str, required_kind: str,
                 faceid_names: list[str], eng_cfg, model: str = "", is_specific: bool = True,
                 *, expected_visual: str = "", scene_query: str = "", era_hint: str = "",
                 multiframe: bool = False, venue_fallback: bool = False,
                 must_see: str = "") -> dict | None:
    """One vision verdict for a frame (Gemini brain → Claude fallback). None on error.

    `is_specific` carries the beat's is_specific_claim: a SPECIFIC line ("Tyrion shoots Tywin with a
    crossbow") demands the EXACT scene; a GENERIC line ("and everything changed") only needs a
    thematically-relevant filler — so the verifier is told to grade leniently there.

    `expected_visual`/`scene_query`/`era_hint` give the verifier the beat's STORYBOARD — what the
    exact moment should look like, which scene it is, and the era/season. Without them the verifier
    only knew the required character, so it rationalized wrong-scene keeps ("Arya is visible looking
    up at Jon Snow (the most powerful man)" for a Daenerys beat; "holding a coin-like object" for the
    Jaqen coin handoff). With the storyboard it can fail a right-character / wrong-moment frame.

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
        "This is a GENERIC narration line (no specific scene claim) — a thematically RELEVANT filler "
        "clip is acceptable. Mark 'replace' ONLY if the footage is off-topic, jarring, or shows the "
        "WRONG character/era — NOT merely because it isn't a specific/exact scene.\n")
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
    if expected_visual:
        _story += f"The exact moment should LOOK LIKE: {expected_visual}\n"
    if scene_query:
        _story += f"Target scene: {scene_query}\n"
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
        "the real show's (amateur or mismatched production values, unfamiliar faces in the role).\n"
        "This is about AUTHENTICITY, not resolution: genuine show footage that is merely low-res, "
        "dark, blurry, or heavily colour-graded is FINE. Real props, maps and documents FILMED "
        "WITHIN a live-action scene are fine. When the frame looks AI-generated OR like a different "
        "production, set correct_subject_visible=false and verdict='replace'.\n")
    txt = (
        f'Narration line: "{narration}"\n'
        f"This clip should show: {required_entity or '(a general scene fitting the line)'} "
        f"(kind: {required_kind or 'any'}).\n"
        + _story + _look + _mf + _rule + _nonshow + _obj + _venue +
        f"Automatic Face-ID on this frame detected: {', '.join(faceid_names) if faceid_names else 'none'}.\n\n"
        "For wrong_subject_visible: set true ONLY if a DIFFERENT specific character (clearly NOT the "
        "one this line is about) is the main subject of the frame; set false for a wide / crowd / "
        "reaction / establishing shot where the required person may be present off-centre or unclear.\n"
        "Answer ONLY this JSON:\n"
        '{"matches_narration": true/false, "correct_subject_visible": true/false, '
        '"wrong_subject_visible": true/false, '
        + ('"target_visible": true/false, ' if must_see else "")
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
PROMPT_VERSION = "v7-2026-07"          # v7: + non-show/illustration hard rule; v6: venue-fallback
# Bump when the contact-sheet SAMPLING changes (frame count/positions/layout). The sheet is the
# image the verifier judges, so a different sampling is a different question even for the same shot.
SHEET_VERSION = "sheet-v1-startmidend"
# Consecutive transient failures after which the vision backend is declared DOWN. Measured: over an
# 11-hour run the verifier degraded 176 replaced -> 180 -> 55 -> 0, and at exactly 0 the release
# gate passed and published. Nothing noticed, because "0 rejections" and "nothing checked" were the
# same number. The breaker exists so the pipeline can tell those two apart.
VERIFIER_BREAKER_TRIP = 8


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
        stamp = f"{st.st_size}:{int(st.st_mtime)}"
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
                        must_see: str = "") -> str:
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
    h = hashlib.sha256()
    parts = [src_hash, source_id, f"{float(shot_start):.3f}", f"{float(shot_end):.3f}",
             (beat_text or "").strip(), (required_entity or "").strip().lower(),
             (required_kind or "").strip().lower(), (expected_visual or "").strip(),
             (scene_query or "").strip(), (era or "").strip().lower(),
             (visual_policy or "").strip().lower(), "1" if is_specific else "0",
             _norm_faces(faceid_names), "mf" if multiframe else "sf",
             (image_id or ""), (model or "").strip(),
             PROMPT_VERSION, SHEET_VERSION]
    if venue_fallback:
        parts.append("venue")
    # must_see changes the QUESTION ("is the dagger visible?"), so a verdict cached without it
    # answers something else entirely and must not be reused. Appended only when set, so every
    # pre-existing key stays valid.
    if must_see:
        parts.append("look:" + must_see.strip().lower())
    for part in parts:
        h.update(str(part).encode("utf-8", "replace"))
        h.update(b"\x1f")
    return h.hexdigest()[:32]


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


def _verdict_schema_ok(v) -> bool:
    """A cached verdict is reusable only if it is a SUCCESSFUL, well-formed one.

    Guards two ways of poisoning the cache with something that was never a judgment: storing an
    error/unavailable stub, and storing a malformed reply whose missing `verdict` key would later
    read as falsy ("not a replace") and quietly pass."""
    if not isinstance(v, dict):
        return False
    if str(v.get("status", "ok")) not in ("ok", ""):
        return False
    if v.get("verdict") not in ("keep", "replace"):
        return False
    return isinstance(v.get("confidence", 0.0), (int, float))


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


def _action_contact_sheet(src_path: str, shot_start: float, shot_end: float, dest: Path):
    """Build a START -> MIDDLE -> END horizontal contact sheet from a shot's source span, so an
    ACTION beat is judged on whether the action actually happens (one keyframe can't prove motion —
    'he catches her by the throat' verified fine on a single ambiguous frame). Returns dest or None."""
    import subprocess
    from .config import ffmpeg_exe
    if not src_path or not Path(src_path).exists():
        return None
    a, b = float(shot_start), float(shot_end)
    if b - a < 0.5:
        return None
    mid = (a + b) / 2.0
    ff = ffmpeg_exe()
    try:
        from PIL import Image
    except Exception:
        return None
    frames = []
    for i, t in enumerate((a + 0.12, mid, max(a + 0.2, b - 0.12))):
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
    return (vd.get("correct_subject_visible") is True
            or (bool(vd.get("matches_narration"))
                and vd.get("correct_subject_visible") is not False))


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
      (1) vision did not see a different main subject (wrong_subject_visible is a hard rejection);
      (2) no DIFFERENT identified person in the shot;
      (3) Face-ID POSITIVELY confirms the required entity — the evidence that was never demanded;
      (4) a POSITIVE same-era signal (an unconstrained era proves nothing)."""
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
    toks = {w for w in _re_aff.findall(r"[a-z']+", (getattr(seg, "scene_query", "") or "").lower())
            if len(w) > 2 and w not in _mv and w not in _AFF_STOP}

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


def verify_and_repair(proj: ClipProject, segments: list[ScriptSegment], cfg: ClipConfig,
                      eng_cfg, *, max_replacements: int = 3, only_indices=None, progress=None) -> dict:
    """Verify every selection; replace failures with the best passing alternate; re-cut swaps.
    Returns a summary. No-op (records 'unavailable') if there's no LLM key.
    `only_indices` (a set of segment indices) restricts verification to just those beats — used by
    the bounded recovery pass to re-verify ONLY the beats it re-matched, instead of re-running the
    whole (very expensive) verifier over every beat. Beats outside the set keep their prior verdict."""
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
    # an over-reused alternate on promotion (falling to the next relevance-ranked one; if all are
    # exhausted, allow the least-used so repair success is preserved).
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
        "movie_title": (proj.meta.get("analysis", {}) or {}).get("movie_title", "")})()
    _event_eras = _era.event_eras_from(_ana_shim)
    _anchor_eras = _era.anchor_token_eras(_ana_shim)

    def _era_of(_s):
        return _beat_era(_s, _global_era, _single, global_verified=_global_ok,
                         event_eras=_event_eras, anchor_eras=_anchor_eras)

    def _src_title_of(_sel):
        _s = proj.source(_sel.source_id)
        return ((getattr(_s, "title", "") or "") + " " + (_sel.source_id or ""))

    # character -> actor. Face-ID reports ACTOR names while beats name CHARACTERS, so confirming
    # "Joffrey is on screen" from a face labelled 'jack gleeson' needs the roster mapping.
    _char2actor: dict = {}
    for _c in ((proj.meta.get("analysis", {}) or {}).get("characters") or []):
        if isinstance(_c, dict) and _c.get("name") and _c.get("actor"):
            _char2actor[str(_c["name"]).strip().lower()] = str(_c["actor"]).strip()

    _vcache = _load_verdict_cache(proj)
    _vcache_n0 = len(_vcache)
    _errored = _reused = _consec_err = _skipped_breaker = _fp_mismatch = 0
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

    def _will_sheet(ashot, _exact) -> bool:
        """Predict, WITHOUT building it, whether this call uses a contact sheet. Must mirror
        _verify_ctx's own condition — the prediction is part of the cache key, and _verify_ctx
        reports what actually happened so a wrong prediction can never be stored."""
        if not (_mf_on and _exact and ashot is not None):
            return False
        _sid = getattr(ashot, "source_id", "") or ""
        _src = proj.source(_sid) if _sid else None
        return bool(getattr(_src, "local_path", "") if _src else "")

    def _image_id(kf_path, ashot, want_sheet: bool) -> str:
        """Identity of the PIXELS the verifier will judge.

        Shot bounds do not pin this: a re-index can rewrite a keyframe while start/end stay put, and
        the stale verdict would be reused against a different image. For a sheet the id is derived
        (source content + span + SHEET_VERSION) rather than measured, so a cache HIT costs no ffmpeg
        work — the sheet is a pure function of those inputs."""
        if want_sheet and ashot is not None:
            _src = proj.source(getattr(ashot, "source_id", "") or "")
            return (f"sheet:{_src_hash_of(_src)}:{float(getattr(ashot, 'start', 0.0)):.3f}"
                    f"-{float(getattr(ashot, 'end', 0.0)):.3f}")
        return f"kf:{_file_fingerprint(kf_path)}" if kf_path else "kf:none"

    def _verify_ctx(kf_path, ashot, _seg, _exact, faceids):
        """verify one candidate with the beat's storyboard context + (for specific action beats) a
        start/mid/end contact sheet built from the shot's source span.
        -> (verdict|None, actually_used_a_sheet)"""
        sheet, is_mf = kf_path, False
        if _mf_on and _exact and ashot is not None:
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
                    _got = _action_contact_sheet(_sp, getattr(ashot, "start", 0.0),
                                                 getattr(ashot, "end", 0.0), _dest)
                    if _got:
                        sheet, is_mf = str(_got), True
            except Exception:
                sheet, is_mf = kf_path, False              # any sheet failure → single-frame path
        try:
            return verify_frame(sheet, _seg.text, _seg.required_entity, _seg.required_kind, faceids,
                                eng_cfg, model, is_specific=_exact,
                                expected_visual=getattr(_seg, "expected_visual", "") or "",
                                scene_query=getattr(_seg, "scene_query", "") or "",
                                era_hint=_era_of(_seg), multiframe=is_mf,
                                must_see=_must_see(_seg)), is_mf
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
        import os as _os_ls
        if not _look_scope["on"]:
            return ""
        if _os_ls.environ.get("VIDLORE_CLIPSTUDIO_LOOK_GATE", "1").strip() in ("0", "false", "no"):
            return ""
        try:
            return _policy.deictic_target(_seg)
        except Exception:                                    # noqa: BLE001
            return ""

    def _served_model_of(v) -> str:
        """The provider that ACTUALLY served this fresh verdict (canonical id), falling back
        to the prediction for stubbed/legacy verdicts that carry no provenance."""
        _sb = str((v or {}).get("vision_served_by") or "")
        return _sb if _sb and _sb != "none" else _vmodel

    def _rung_fingerprint(ashot, _seg, strict_flag: bool, faceids, model_id: str = ""):
        """Fingerprint of a FALLBACK-RUNG question — the exact same derivation as the primary
        path (keep the two in sync), parameterized on the candidate shot and the rung's
        strictness. `faceids` must be the SAME list the prompt will carry (the caller may
        substitute sel.identity when the shot has no face detections — the fingerprint hashes
        what is actually asked, never a proxy). venue_fallback stays False here because
        _verify_ctx never asks the venue question (the still layer does)."""
        if ashot is None:
            return "", False
        _src_a = proj.source(getattr(ashot, "source_id", "") or "")
        _ws = _will_sheet(ashot, strict_flag)
        _kf_a = getattr(ashot, "keyframe_path", "") or ""
        return verdict_fingerprint(
            src_hash=_src_hash_of(_src_a), source_id=getattr(ashot, "source_id", "") or "",
            shot_start=getattr(ashot, "start", 0.0), shot_end=getattr(ashot, "end", 0.0),
            beat_text=getattr(_seg, "text", ""),
            required_entity=getattr(_seg, "required_entity", ""),
            required_kind=getattr(_seg, "required_kind", ""),
            expected_visual=getattr(_seg, "expected_visual", "") or "",
            scene_query=getattr(_seg, "scene_query", "") or "",
            era=_era_of(_seg), visual_policy=_policy.policy_of(_seg),
            is_specific=strict_flag, faceid_names=list(faceids or []),
            multiframe=_ws, image_id=_image_id(_kf_a, ashot, _ws),
            model=(model_id or _vmodel), must_see=_must_see(_seg)), _ws

    def _cached_verify_ctx(kf_path, ashot, _seg, strict_flag: bool, faceids, rung: str):
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
        _fp_r, _ws_r = _rung_fingerprint(ashot, _seg, strict_flag, faceids)
        _hit = _vcache.get(_fp_r) if _fp_r else None
        if _hit is not None and _verdict_schema_ok(_hit) and _hit_provider_ok(_hit, _vmodel):
            _pm_r.incr(f"verify.rung.{rung}.cache_hit")
            return dict(_hit), _ws_r                     # copy: callers mutate their verdict
        _pm_r.incr(f"verify.rung.{rung}.call")
        v_r, used_r = _verify_ctx(kf_path, ashot, _seg, strict_flag, faceids)
        if _fp_r and v_r is not None and used_r == _ws_r \
                and _verdict_schema_ok({**v_r, "status": "ok"}):
            # store under the ACTUAL server's key — a Claude fallback answer must never sit
            # under a Gemini-predicted fingerprint (and vice versa)
            _served_r = _served_model_of(v_r)
            _fp_store = _fp_r if _served_r == _vmodel else \
                _rung_fingerprint(ashot, _seg, strict_flag, faceids, model_id=_served_r)[0]
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
            _want_sheet = _will_sheet(shot, _exact)
            _fp = verdict_fingerprint(
                src_hash=_src_hash_of(_src_obj), source_id=sel.source_id or "",
                shot_start=getattr(shot, "start", 0.0), shot_end=getattr(shot, "end", 0.0),
                beat_text=getattr(seg, "text", ""),
                required_entity=getattr(seg, "required_entity", ""),
                required_kind=getattr(seg, "required_kind", ""),
                expected_visual=getattr(seg, "expected_visual", "") or "",
                scene_query=getattr(seg, "scene_query", "") or "",
                era=_era_of(seg), visual_policy=_policy.policy_of(seg), is_specific=_exact,
                faceid_names=faceid_names, multiframe=_want_sheet,
                image_id=_image_id(kf, shot, _want_sheet), model=_vmodel,
                must_see=_must_see(seg))
            _c0 = _vcache.get(_fp)
            if _fp:
                _prim_items.append((_fp, sel, seg, shot, kf, faceid_names, _exact))
            if _fp and (_c0 is None or not _verdict_schema_ok(_c0)
                        or not _hit_provider_ok(_c0, _vmodel)):
                _pending.append((_fp, seg, shot, kf, faceid_names, _exact, _want_sheet))
        if _pending:
            log(f"verify: prefetching {len(_pending)} fresh verdict(s) "
                f"({_pf_workers} workers; serial decisions unchanged)")
            import concurrent.futures as _cf
            _pf_fail = _pf_ok = 0
            with _cf.ThreadPoolExecutor(max_workers=_pf_workers) as _ex:
                _futs = {_ex.submit(_verify_ctx, kf9, shot9, seg9, ex9, fids9):
                         (fp9, ws9)
                         for (fp9, seg9, shot9, kf9, fids9, ex9, ws9) in _pending}
                for _fu in _cf.as_completed(_futs):
                    _fp9, _ws9 = _futs[_fu]
                    try:
                        _v9, _us9 = _fu.result()
                    except Exception:                     # noqa: BLE001
                        _v9, _us9 = None, _ws9
                    if _v9 is not None and _us9 == _ws9 \
                            and _verdict_schema_ok({**_v9, "status": "ok"}):
                        _served9 = _served_model_of(_v9)
                        _fp_store9 = _fp9
                        if _served9 != _vmodel:
                            # a fallback provider answered — key by the ACTUAL server
                            (_, _seg9, _shot9, _kf9, _fids9, _ex9b, _ws9b) = \
                                next(p for p in _pending if p[0] == _fp9)
                            _src9 = proj.source(getattr(_shot9, "source_id", "") or "")
                            _fp_store9 = verdict_fingerprint(
                                src_hash=_src_hash_of(_src9),
                                source_id=getattr(_shot9, "source_id", "") or "",
                                shot_start=getattr(_shot9, "start", 0.0),
                                shot_end=getattr(_shot9, "end", 0.0),
                                beat_text=getattr(_seg9, "text", ""),
                                required_entity=getattr(_seg9, "required_entity", ""),
                                required_kind=getattr(_seg9, "required_kind", ""),
                                expected_visual=getattr(_seg9, "expected_visual", "") or "",
                                scene_query=getattr(_seg9, "scene_query", "") or "",
                                era=_era_of(_seg9), visual_policy=_policy.policy_of(_seg9),
                                is_specific=_ex9b, faceid_names=_fids9,
                                multiframe=_ws9b, image_id=_image_id(_kf9, _shot9, _ws9b),
                                model=_served9)
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
        # min: 146/202 beats entered repair). For every beat whose warmed PRIMARY verdict is
        # 'replace' and that is EXACT, warm the questions the serial loop will provably ask:
        #   (a) the strict-promotion rung over the first max_replacements alternates in the
        #       exact serial order (_scene_affinity_order when enabled), judged WITHOUT the
        #       look question (_look_scope off — same as _try_promote); a slot-consuming
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
            _alt_jobs, _len_jobs = [], []
            for (_fpP, selP, segP, shotP, kfP, fidsP, _exP) in _prim_items:
                _v0 = _vcache.get(_fpP)
                if not (_v0 is not None and _verdict_schema_ok(_v0)
                        and _hit_provider_ok(_v0, _vmodel)):
                    continue                             # primary not warm → serial pays as today
                if str(_v0.get("verdict")) != "replace" or not _exP:
                    continue
                _altsP = selP.alternates
                if _aff_on2:
                    try:
                        _altsP = _scene_affinity_order(selP.alternates, segP, proj,
                                                       selP.source_id)
                    except Exception:                    # noqa: BLE001
                        _altsP = selP.alternates
                _t2 = 0
                for _altP in _altsP:
                    if _t2 >= max_replacements:
                        break
                    _t2 += 1                             # slot consumed even when shotless
                    _ashP = get_shot(_altP.source_id, _altP.shot_index)
                    if _ashP is None:
                        continue
                    _alt_jobs.append((_ashP, segP))
                # lenient re-ask fires serially only when the downgrade + filler rungs are
                # enabled — mirror those envs so the warm never asks a question the serial
                # loop provably won't
                if (_os_pf.environ.get("VIDLORE_CLIPSTUDIO_EXACT_CONTEXTUAL_DOWNGRADE",
                                       "1").strip() not in ("0", "false", "no")
                        and _os_pf.environ.get("VIDLORE_CLIPSTUDIO_GENERIC_FILLER_DOWNGRADE",
                                               "1").strip() not in ("0", "false", "no")):
                    _len_jobs.append((kfP, shotP, segP, fidsP))
            if _alt_jobs or _len_jobs:
                log(f"verify: rung prefetch — warming {len(_alt_jobs)} promotion + "
                    f"{len(_len_jobs)} lenient question(s) ({_pf_workers} workers; the serial "
                    f"repair loop and every gate stay unchanged)")
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

                def _warm_alt(j):
                    _a, _s = j
                    _v_w, _ = _cached_verify_ctx(_a.keyframe_path, _a, _s, True,
                                                 _a.face_ids or [], rung="strict_promote")
                    return _v_w is not None

                def _warm_len(j):
                    _kf_w, _sh_w, _s_w, _fi_w = j
                    _v_w, _ = _cached_verify_ctx(_kf_w, _sh_w, _s_w, False, _fi_w,
                                                 rung="lenient_filler")
                    return _v_w is not None

                _look_scope["on"] = False                # alternates: no look question
                try:
                    _go_on = _warm(_warm_alt, _alt_jobs)
                finally:
                    _look_scope["on"] = True
                if _go_on:
                    _warm(_warm_len, _len_jobs)          # original shots: look scope ON
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
        _exact = _policy.verify_strict(seg)               # exact_scene → strict; else lenient (filler ok)
        # REUSE a verdict only when the QUESTION is byte-identical (see verdict_fingerprint). This
        # is what lets a restart keep explicitly-proven judgments instead of re-rolling them against
        # a dying API — the failure mode that published this render.
        _fp, _want_sheet = "", False
        if shot is not None:
            _src_obj = proj.source(sel.source_id)
            _want_sheet = _will_sheet(shot, _exact)
            _fp = verdict_fingerprint(
                src_hash=_src_hash_of(_src_obj), source_id=sel.source_id or "",
                shot_start=getattr(shot, "start", 0.0), shot_end=getattr(shot, "end", 0.0),
                beat_text=getattr(seg, "text", ""),
                required_entity=getattr(seg, "required_entity", ""),
                required_kind=getattr(seg, "required_kind", ""),
                expected_visual=getattr(seg, "expected_visual", "") or "",
                scene_query=getattr(seg, "scene_query", "") or "",
                era=_era_of(seg), visual_policy=_policy.policy_of(seg), is_specific=_exact,
                faceid_names=faceid_names, multiframe=_want_sheet,
                image_id=_image_id(kf, shot, _want_sheet), model=_vmodel,
                must_see=_must_see(seg))
        # only a SUCCESSFUL, schema-valid verdict is reusable — never an error stub or a malformed
        # reply whose missing "verdict" key would read as falsy and quietly pass
        _cached = _vcache.get(_fp) if _fp else None
        _used_sheet = _want_sheet
        if _cached is not None and _verdict_schema_ok(_cached) \
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
                v, _used_sheet = _verify_ctx(kf, shot, seg, _exact, faceid_names)
        if v is None:
            # FAIL CLOSED. "No judgment" is not a synonym for "acceptable". The old code set
            # status=error and `continue`d, so a beat nobody could check looked exactly like a beat
            # that passed — and a TOTAL outage produced zero rejections, which the release gate read
            # as "nothing wrong" and shipped.
            sel.verifier = {"status": "breaker_open" if _breaker_open else "error"}
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
        _consec_err = 0
        verified += 1                    # counts SUCCESSES — never attempts (see the breaker note)
        # STORE only a schema-valid verdict, and only when the sheet prediction that went INTO the
        # key actually held. A sheet build can fail and silently fall back to one frame — storing
        # that single-frame answer under a multiframe key would hand back the wrong judgment later.
        if _fp and not v.get("reused") and _verdict_schema_ok({**v, "status": "ok"}):
            if _used_sheet == _want_sheet:
                # key by the ACTUAL server (a fallback provider's answer must never sit under
                # the predicted provider's fingerprint)
                _served_p = _served_model_of(v)
                _fp_store_p = _fp
                if _served_p != _vmodel and shot is not None:
                    _src_obj = proj.source(sel.source_id)
                    _fp_store_p = verdict_fingerprint(
                        src_hash=_src_hash_of(_src_obj), source_id=sel.source_id or "",
                        shot_start=getattr(shot, "start", 0.0),
                        shot_end=getattr(shot, "end", 0.0),
                        beat_text=getattr(seg, "text", ""),
                        required_entity=getattr(seg, "required_entity", ""),
                        required_kind=getattr(seg, "required_kind", ""),
                        expected_visual=getattr(seg, "expected_visual", "") or "",
                        scene_query=getattr(seg, "scene_query", "") or "",
                        era=_era_of(seg), visual_policy=_policy.policy_of(seg),
                        is_specific=_exact, faceid_names=faceid_names,
                        multiframe=_want_sheet, image_id=_image_id(kf, shot, _want_sheet),
                        model=_served_p)
                _vcache[_fp_store_p] = {k: val for k, val in v.items() if k != "reused"}
                _vcache_dirty += 1
            else:
                _fp_mismatch += 1
        v["status"] = "ok"
        v["visual_policy"] = _policy.policy_of(seg)
        # NON-EXACT LENIENCY (user rule: exact clip only for a SPECIFIC scene; a relevant FILLER is
        # fine for generic/character/abstract beats). Don't replace an on-topic, right-subject clip on
        # a non-exact beat just because it isn't the exact scene — only off-topic / wrong-character.
        if not _exact and v.get("verdict") == "replace" and _contextual_subject_ok(v):
            v["verdict"] = "keep"
            v["relaxed"] = "non-exact beat: relevant right-subject filler accepted"
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

            def _try_promote(downgrade: bool, pool=None, label: str = "") -> bool:
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
                _look_scope["on"] = False        # alternates are judged WITHOUT the look question
                try:
                    return _try_promote_inner(downgrade, pool, label)
                finally:
                    _look_scope["on"] = True

            def _try_promote_inner(downgrade: bool, pool=None, label: str = "") -> bool:
                nonlocal swapped, replaced
                tried = 0
                # SCENE-AFFINITY ordering for exact beats — try same-scene sources first (see
                # _scene_affinity_order). Ordering only; every gate below still applies.
                import os as _os_aff
                _alts = pool if pool is not None else sel.alternates
                if pool is None and _exact \
                        and _os_aff.environ.get("VIDLORE_CLIPSTUDIO_SCENE_AFFINITY", "1").strip() \
                        not in ("0", "false", "no", ""):
                    try:
                        _alts = _scene_affinity_order(sel.alternates, seg, proj, sel.source_id)
                    except Exception:
                        _alts = sel.alternates
                for alt in _alts:
                    if tried >= max_replacements:
                        break
                    tried += 1
                    ashot = get_shot(alt.source_id, alt.shot_index)
                    if ashot is None:
                        continue
                    anames = ashot.face_ids or []
                    if _breaker_open:
                        break                           # backend is down — promotion cannot verify
                    av, _ = _cached_verify_ctx(ashot.keyframe_path, ashot, seg,
                                               (False if downgrade else _exact), anames,
                                               rung=("venue" if pool is not None else
                                                     ("contextual" if downgrade else "strict_promote")))
                    if av is None:
                        continue                        # transport error, NOT a judgment
                    if downgrade:
                        _accept = _contextual_subject_ok(av)
                    else:
                        _accept = av.get("verdict") == "keep"
                    if not _accept:
                        # an explicit non-keep judgment (av None = transport error, handled above)
                        failed_wins.append((alt.source_id, float(alt.in_point)))
                        continue
                    # REUSE LEDGER — do not promote a look that already airs on >= cap beats (that is
                    # how one clip got re-aired 9×). Skip to the next relevance-ranked alternate; if
                    # none survive, the beat stays flagged and image-fallback gives it a DISTINCT still.
                    if _reuse[(alt.source_id, alt.shot_index)] >= _reuse_cap:
                        failed_wins.append((alt.source_id, float(alt.in_point)))
                        continue
                    # CUT-WINDOW FLAG VALIDATION on the promotion — the repair must not swap a
                    # rejected clip for one whose PADDED render window airs an adjacent shot's
                    # burned subs / logo / murk. Same PRODUCTION validator as match selections:
                    # moment-locked beats (exact/quote/character) may only shorten around the
                    # alternate's own selected moment — never slide to a different moment —
                    # else this alternate is skipped for the next relevance-ranked one.
                    import os as _os_w
                    if _os_w.environ.get("VIDLORE_CLIPSTUDIO_WINDOW_QC", "1").strip() \
                            not in ("0", "false", "no"):
                        from .match import validate_candidate_window, _wqc_log_line
                        # stub-tolerant: tests monkeypatch _shot_lookup with a bare function —
                        # no shot list then means nothing to validate (fail-open)
                        _wshots = getattr(get_shot, "all_shots", lambda _s: [])(alt.source_id)
                        _wact, _wwhy, _wmeta = validate_candidate_window(
                            alt, ashot, _wshots, cfg, seg)
                        if _wact == "rejected":
                            log(f"window-qc: rejected verify-promotion seg{sel.segment_index} "
                                f"alt={alt.source_id[:28]} {_wqc_log_line(_wact, _wmeta, _wwhy)}")
                            failed_wins.append((alt.source_id, float(alt.in_point)))
                            continue
                        if _wact == "shortened":
                            log(f"window-qc: shortened verify-promotion seg{sel.segment_index} "
                                f"{_wqc_log_line(_wact, _wmeta, _wwhy)}")
                    # promote the alternate into the selection
                    old_sid, old_in = sel.source_id, sel.in_point
                    _old_key = (sel.source_id, sel.shot_index)
                    sel.source_id = alt.source_id
                    sel.shot_index = alt.shot_index
                    sel.in_point = alt.in_point
                    sel.out_point = alt.out_point
                    sel.signals = alt.signals
                    sel.confidence = alt.score
                    sel.source_url = (proj.source(alt.source_id).url if proj.source(alt.source_id) else "")
                    sel.identity = (anames[0] if anames else "")
                    # build_video plays the scene's beats from beat_windows (rejected pick is
                    # FIRST there) — drop it AND every alternate the verifier explicitly failed
                    # on the way here, then lead with the promoted window; otherwise rejected
                    # footage still airs on the scene's later beats.
                    new_win = [alt.source_id, round(alt.in_point, 3), round(alt.out_point, 3)]
                    kept = [w for w in (sel.beat_windows or [])
                            if not (w and w[0] == old_sid and abs(float(w[1]) - float(old_in)) < 0.05)
                            and not (w and w[0] == new_win[0] and abs(float(w[1]) - new_win[1]) < 0.05)
                            and not any(w and w[0] == fs and abs(float(w[1]) - fi) < 0.05
                                        for fs, fi in failed_wins)]
                    sel.beat_windows = [new_win] + kept
                    av["status"] = "ok"
                    av["replaced_from"] = {"shot": shot.index if shot else -1}
                    if downgrade:
                        av["verdict"] = "keep"
                        av["downgraded"] = label or "exact→contextual"
                        av["relevance_class"] = "contextual_fallback"
                    sel.verifier = av
                    _cut.cut_selection(proj, sel, cfg)     # re-cut the new in/out
                    _reuse[(alt.source_id, alt.shot_index)] += 1   # this look now airs one more time
                    if _reuse[_old_key] > 0:
                        _reuse[_old_key] -= 1                       # the replaced pick no longer airs here
                    replaced += 1
                    swapped = True
                    log(f"verify: seg{sel.segment_index} "
                        f"{'exact→contextual' if downgrade else 'replaced'} → "
                        f"{alt.source_id}#{alt.shot_index}")
                    return True
                return False

            _try_promote(downgrade=False)     # ORIGINAL strict/normal promotion (unchanged behavior)

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

                _orig_ok = _contextual_subject_ok(v)
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
                if not _breaker_open:
                    _fresh, _ = _cached_verify_ctx(kf, shot, seg, False, faceid_names,
                                                   rung="lenient_filler")         # LENIENT re-ask
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
                    sel.verifier = v
                    replaced += 1
                    swapped = True
                    log(f"verify: seg{sel.segment_index} exact→generic_filler — a fresh lenient "
                        f"pass PROVES relevance ({_why_f}); honestly labeled")
                else:
                    log(f"verify: seg{sel.segment_index} generic-filler fallback REFUSED — "
                        f"{_why_f}; the beat stays unresolved rather than airing unproven footage")

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
            "verifier_down": bool(_breaker_open), "breaker_skipped": _skipped_breaker,
            "vision_config": _vmodel,
            "verified_frac": (verified / _attempted) if _attempted else 1.0}
