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
                 multiframe: bool = False) -> dict | None:
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
    _story = ""
    if expected_visual:
        _story += f"The exact moment should LOOK LIKE: {expected_visual}\n"
    if scene_query:
        _story += f"Target scene: {scene_query}\n"
    if era_hint:
        _story += (f"Era/season context: {era_hint} — footage from a clearly different era/season "
                   f"than the moment described is WRONG even if the character matches.\n")
    _mf = ("The image is a START -> MIDDLE -> END contact sheet (three moments of the clip, left to "
           "right). Judge whether the described ACTION actually happens across them — a single frame "
           "cannot prove an action, so require visible progression consistent with the line.\n"
           if multiframe else "")
    txt = (
        f'Narration line: "{narration}"\n'
        f"This clip should show: {required_entity or '(a general scene fitting the line)'} "
        f"(kind: {required_kind or 'any'}).\n"
        + _story + _mf + _rule +
        f"Automatic Face-ID on this frame detected: {', '.join(faceid_names) if faceid_names else 'none'}.\n\n"
        "For wrong_subject_visible: set true ONLY if a DIFFERENT specific character (clearly NOT the "
        "one this line is about) is the main subject of the frame; set false for a wide / crowd / "
        "reaction / establishing shot where the required person may be present off-centre or unclear.\n"
        "Answer ONLY this JSON:\n"
        '{"matches_narration": true/false, "correct_subject_visible": true/false, '
        '"wrong_subject_visible": true/false, '
        '"specific_enough": true/false, "quality_ok": true/false, '
        '"confidence": 0.0-1.0, "verdict": "keep" or "replace", "reason": "one short sentence"}'
    )
    import time
    content = [_img_block(Path(keyframe_path)), {"type": "text", "text": txt}]
    for attempt in range(1, 5):                       # retry transient overload / rate limits
        try:
            out = _llm.complete(system=_VSYS, max_tokens=400,
                                messages=[{"role": "user", "content": content}],
                                eng_cfg=eng_cfg, model=model)
            m = re.search(r"\{.*\}", out, re.S)
            return json.loads(m.group(0)) if m else None
        except Exception:                             # transient overload / rate limit → back off
            if attempt == 4:
                return None
            time.sleep(min(1.5 * (2 ** attempt), 16))
    return None


_SEASON_RX = re.compile(
    r"\bS0?(\d{1,2})\s?E0?\d{1,2}\b|\bseason\s+(\d{1,2})\b|\b(\d{1,2})\s?x\s?\d{2}\b", re.I)

# Bump whenever the verifier PROMPT or its JSON contract changes: a verdict is only reusable if it
# was produced by the same question. Part of the fingerprint below.
PROMPT_VERSION = "v3-2026-07"
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

    Retry transient plumbing. Never retry a judgment."""


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
                        model: str = "") -> str:
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
                          on eng_cfg.anthropic_model made Gemini and Claude verdicts collide."""
    import hashlib
    h = hashlib.sha256()
    for part in (src_hash, source_id, f"{float(shot_start):.3f}", f"{float(shot_end):.3f}",
                 (beat_text or "").strip(), (required_entity or "").strip().lower(),
                 (required_kind or "").strip().lower(), (expected_visual or "").strip(),
                 (scene_query or "").strip(), (era or "").strip().lower(),
                 (visual_policy or "").strip().lower(), "1" if is_specific else "0",
                 _norm_faces(faceid_names), "mf" if multiframe else "sf",
                 (image_id or ""), (model or "").strip(),
                 PROMPT_VERSION, SHEET_VERSION):
        h.update(str(part).encode("utf-8", "replace"))
        h.update(b"\x1f")
    return h.hexdigest()[:32]


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
              event_eras: dict | None = None) -> str:
    """The era/season constraint for ONE beat — see `era.beat_era` for the ordering and why.

    This used to return the global hint IMMEDIATELY for single-scene videos, never reading the
    beat. That made one unvalidated LLM string ("S04E01" for a scene that is S03E10) the era of all
    229 beats at once, including the ones about the Red Wedding (S03E09). Era is beat-local now,
    and an unverified global hint constrains nothing."""
    return _era.beat_era(seg, global_era, single_scene=single_scene,
                         global_verified=global_verified, event_eras=event_eras)


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
    _event_eras = _era.event_eras_from(
        type("A", (), {"anchor_scenes": (proj.meta.get("analysis", {}) or {}).get("anchor_scenes")})())

    def _era_of(_s):
        return _beat_era(_s, _global_era, _single, global_verified=_global_ok,
                         event_eras=_event_eras)

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
                    _dest = proj.clips_dir / f"_vsheet_{_seg.index}_{getattr(ashot, 'index', 0)}.jpg"
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
                                era_hint=_era_of(_seg), multiframe=is_mf), is_mf
        finally:
            if is_mf:
                try:
                    Path(sheet).unlink(missing_ok=True)
                except Exception:
                    pass

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
                image_id=_image_id(kf, shot, _want_sheet), model=_vmodel)
        # only a SUCCESSFUL, schema-valid verdict is reusable — never an error stub or a malformed
        # reply whose missing "verdict" key would read as falsy and quietly pass
        _cached = _vcache.get(_fp) if _fp else None
        _used_sheet = _want_sheet
        if _cached is not None and _verdict_schema_ok(_cached):
            v = dict(_cached)
            v["reused"] = True
            _reused += 1
        else:
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
                _vcache[_fp] = {k: val for k, val in v.items() if k != "reused"}
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
                    av, _ = _verify_ctx(ashot.keyframe_path, ashot, seg,
                                        (False if downgrade else _exact), anames)
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
            if not swapped and _exact and _downgrade_on:
                if _contextual_subject_ok(v):
                    v["verdict"] = "keep"
                    v["downgraded"] = "exact→contextual"
                    v["relevance_class"] = "contextual_fallback"
                    sel.verifier = v
                    replaced += 1
                    swapped = True
                    log(f"verify: seg{sel.segment_index} exact→contextual downgrade "
                        f"(required subject on screen — kept, honestly labeled contextual_fallback)")
                else:
                    _try_promote(downgrade=True)     # scan alternates for a right-subject clip

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
                _ok_toks = _roster_toks if _single else frozenset()
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
                    _fresh, _ = _verify_ctx(kf, shot, seg, False, faceid_names)   # LENIENT re-ask
                _ok_f, _why_f = _generic_filler_ok(
                    _fresh, seg, _src_title_of(sel), faceid_names, _era_of(seg),
                    _ok_toks if _single else frozenset(), _char2actor)
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
