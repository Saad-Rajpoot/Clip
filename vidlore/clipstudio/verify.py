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


def _source_fingerprint(path) -> str:
    """Cheap, stable content id: size + a hash of the head. Full hashing of multi-GB sources every
    run costs more than the verdicts it protects."""
    import hashlib
    p = Path(path or "")
    if not p.exists():
        return "missing"
    h = hashlib.sha256()
    h.update(str(p.stat().st_size).encode())
    try:
        with p.open("rb") as fh:
            h.update(fh.read(1 << 20))
    except OSError:
        return "unreadable"
    return h.hexdigest()[:16]


def verdict_fingerprint(*, src_hash: str, source_id: str, shot_start: float, shot_end: float,
                        beat_text: str, required_entity: str, era: str, visual_policy: str,
                        model: str) -> str:
    """Identity of a verdict: every input that could change the answer.

    A verdict may only be reused when the QUESTION is identical. Keying on the beat alone would
    silently reuse a verdict across a re-matched shot, a changed era constraint, a policy
    promotion (deixis!), or a different verifier model — each of which changes the answer."""
    import hashlib
    h = hashlib.sha256()
    for part in (src_hash, source_id, f"{float(shot_start):.3f}", f"{float(shot_end):.3f}",
                 (beat_text or "").strip(), (required_entity or "").strip().lower(),
                 (era or "").strip().lower(), (visual_policy or "").strip().lower(),
                 (model or "").strip(), PROMPT_VERSION):
        h.update(str(part).encode("utf-8", "replace"))
        h.update(b"\x1f")
    return h.hexdigest()[:32]


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

    # character -> actor. Face-ID reports ACTOR names while beats name CHARACTERS, so confirming
    # "Joffrey is on screen" from a face labelled 'jack gleeson' needs the roster mapping.
    _char2actor: dict = {}
    for _c in ((proj.meta.get("analysis", {}) or {}).get("characters") or []):
        if isinstance(_c, dict) and _c.get("name") and _c.get("actor"):
            _char2actor[str(_c["name"]).strip().lower()] = str(_c["actor"]).strip()

    _vcache = _load_verdict_cache(proj)
    _vcache_n0 = len(_vcache)
    _errored = _reused = _consec_err = 0
    _breaker_open = False
    _vmodel = str(getattr(eng_cfg, "anthropic_model", "") or "")
    _src_hash_cache: dict = {}

    def _src_hash_of(src):
        sid = getattr(src, "id", "") or ""
        if sid not in _src_hash_cache:
            _src_hash_cache[sid] = _source_fingerprint(getattr(src, "local_path", "") or "")
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

    def _verify_ctx(kf_path, ashot, _seg, _exact, faceids):
        """verify one candidate with the beat's storyboard context + (for specific action beats) a
        start/mid/end contact sheet built from the shot's source span."""
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
                                era_hint=_era_of(_seg), multiframe=is_mf)
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
        _fp = ""
        if shot is not None:
            _src_obj = proj.source(sel.source_id)
            _fp = verdict_fingerprint(
                src_hash=_src_hash_of(_src_obj), source_id=sel.source_id or "",
                shot_start=getattr(shot, "start", 0.0), shot_end=getattr(shot, "end", 0.0),
                beat_text=getattr(seg, "text", ""), required_entity=getattr(seg, "required_entity", ""),
                era=_era_of(seg), visual_policy=_policy.policy_of(seg), model=_vmodel)
        _cached = _vcache.get(_fp) if _fp else None
        if _cached:
            v = dict(_cached)
            v["reused"] = True
            _reused += 1
        else:
            v = _verify_ctx(kf, shot, seg, _exact, faceid_names)
        if v is None:
            # FAIL CLOSED. "No judgment" is not a synonym for "acceptable". The old code set
            # status=error and `continue`d, so a beat nobody could check looked exactly like a beat
            # that passed — and a TOTAL outage produced zero rejections, which the release gate read
            # as "nothing wrong" and shipped.
            sel.verifier = {"status": "error"}
            _errored += 1
            _consec_err += 1
            if FLAG_VERIFIER_UNVERIFIED not in sel.flag_reasons:
                sel.flag_reasons.append(FLAG_VERIFIER_UNVERIFIED)
            sel.flagged = True
            if _exact:
                # an exact_scene beat we could not check is UNRESOLVED, never a pass
                failed += 1
                log(f"verify: seg{sel.segment_index} UNVERIFIED (verifier error) — exact_scene "
                    f"beat is unresolved, not accepted")
            if _consec_err >= VERIFIER_BREAKER_TRIP and not _breaker_open:
                _breaker_open = True
                log(f"verify: ⛔ CIRCUIT BREAKER — {_consec_err} consecutive verifier errors; the "
                    f"vision backend is down. Remaining beats will NOT be silently accepted.")
            continue
        _consec_err = 0
        verified += 1                    # counts SUCCESSES — never attempts (see the breaker note)
        if _fp and not v.get("reused"):
            _vcache[_fp] = {k: val for k, val in v.items() if k != "reused"}
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

            def _try_promote(downgrade: bool) -> bool:
                """Scan the beat's relevance-ranked alternates and promote the first acceptable one.
                downgrade=False → the ORIGINAL strict promotion (verify at the beat's own strictness,
                accept only an explicit verdict==keep). downgrade=True → the EXACT→CONTEXTUAL rung:
                verify LENIENTLY and accept a right-subject / on-topic clip that simply isn't the
                exact moment (wrong-show/era/character still fail and are skipped). Returns True on a
                swap. All the production safeguards (reuse-ledger cap, Window-QC, beat_windows rewrite,
                re-cut) are shared by both modes."""
                nonlocal swapped, replaced
                tried = 0
                for alt in sel.alternates:
                    if tried >= max_replacements:
                        break
                    tried += 1
                    ashot = get_shot(alt.source_id, alt.shot_index)
                    if ashot is None:
                        continue
                    anames = ashot.face_ids or []
                    av = _verify_ctx(ashot.keyframe_path, ashot, seg,
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
                        av["downgraded"] = "exact→contextual"
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
                elif not _confirmed_wrong_character(seg, faceid_names, _ok_toks):
                    # No exact/contextual, no POSITIVE era signal — but Face-ID does NOT confirm a
                    # different, unrelated character (empty/unconfirmed Face-ID, or a CO-MENTIONED
                    # character like Tywin on a 'Joffrey calls Tywin a coward' beat, or — in a
                    # single-scene deep-dive — any main-cast member). A thematic scene clip is a
                    # legitimate GENERIC-FILLER last resort. NEVER a CONFIRMED wrong character.
                    v["verdict"] = "keep"
                    v["downgraded"] = "exact→generic_filler(character last-resort)"
                    v["relevance_class"] = "generic_filler"
                    sel.verifier = v
                    replaced += 1
                    swapped = True
                    log(f"verify: seg{sel.segment_index} exact→generic_filler (character; no confirmed "
                        f"wrong character — thematic scene clip, honestly labeled)")

            # EXACT→GENERIC-FILLER (the last hierarchy tier before an honest gap). When neither the
            # exact moment NOR a right-subject contextual clip exists, a NON-CHARACTER beat
            # (scene / event / object / location / abstract) may air its thematic clip as honestly
            # labelled generic_filler — a thematic same-show clip is NOT contradictory and beats a
            # frozen still / black. A CHARACTER/actor beat is NOT filler-eligible: a clip that does
            # not show the required person risks a WRONG-CHARACTER read (contradictory), so it stays
            # unresolved for the still / hold / honest release-block. env-gated (default ON).
            _filler_on = _os_ms.environ.get(
                "VIDLORE_CLIPSTUDIO_GENERIC_FILLER_DOWNGRADE", "1").strip() \
                not in ("0", "false", "no")
            if not swapped and _exact and _downgrade_on and _filler_on \
                    and (getattr(seg, "required_kind", "") or "").lower() not in ("character", "actor"):
                v["verdict"] = "keep"
                v["downgraded"] = "exact→generic_filler"
                v["relevance_class"] = "generic_filler"
                sel.verifier = v
                replaced += 1
                swapped = True
                log(f"verify: seg{sel.segment_index} exact→generic_filler "
                    f"(non-character beat; exact+contextual absent — thematic clip, honestly labeled)")

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
    log(f"verify: done — {verified} verified ({_reused} reused), {_errored} ERRORED, "
        f"{replaced} replaced, {failed} unresolved")
    # LIVENESS. Reported as its own fact, never folded into 'unresolved': a run that checked
    # nothing must not read like a run that found nothing wrong. This is a SECOND line of defence —
    # the primary one is per-beat (an unverifiable exact beat is already unresolved above), because
    # a global ratio alone would happily pass a render whose 20% of failures were all exact beats.
    if _errored:
        log(f"verify: ⚠ {_errored}/{_attempted} beats could not be verified "
            f"(backend errors){' — CIRCUIT BREAKER OPEN' if _breaker_open else ''}")
    return {"verified": verified, "replaced": replaced, "failed": failed, "available": True,
            "errored": _errored, "reused": _reused, "attempted": _attempted,
            "verifier_down": bool(_breaker_open),
            "verified_frac": (verified / _attempted) if _attempted else 1.0}
