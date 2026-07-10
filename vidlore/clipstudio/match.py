"""Stage 4 — match each script segment to the best source clip.

Score(segment, shot) = w_clip*clip01 + w_trans*trans + w_face*face + w_obj*obj - penalties,
clamped to [0,1]. CLIP visual similarity (engine's local ONNX model) is the base signal;
transcript keyword/entity overlap and face presence are additive bonuses. Selection is greedy
in script order under anti-reuse / source-diversity / pacing constraints, and keeps alternates
for the review surface. Every signal is recorded on the selection for the ledger.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .models import Shot, ScriptSegment, ClipCandidate, ClipSelection, ClipProject, SOURCE_OK
from .config import ClipConfig
from .segment import _STOP
from . import index as _index
from . import policy as _policy


def _f_env(name: str, default: float) -> float:
    import os
    try:
        return float(os.environ.get(name, "").strip() or default)
    except (TypeError, ValueError):
        return default


_BRIGHT_CACHE: dict = {}


def _shot_brightness(keyframe_path: str) -> float:
    """Mean luma 0..1 of a shot's keyframe (cached). Used to keep a DARK/night/candlelit single-scene
    deep-dive from cutting to a bright daytime-exterior shot of the right character (the daylight
    forest Bronn that clashed with the night-tavern narration). Returns 0.5 (neutral) if unreadable."""
    if not keyframe_path:
        return 0.5
    if keyframe_path in _BRIGHT_CACHE:
        return _BRIGHT_CACHE[keyframe_path]
    val = 0.5
    try:
        from PIL import Image
        import os
        if os.path.exists(keyframe_path):
            im = Image.open(keyframe_path).convert("L").resize((32, 18))
            px = list(im.getdata())
            val = (sum(px) / len(px)) / 255.0 if px else 0.5
    except Exception:
        val = 0.5
    while len(_BRIGHT_CACHE) >= 4096:    # bounded (portal jobs share this module-global); evict
        try:                             # oldest-first — clearing ALL would thrash to 0% hits on
            _BRIGHT_CACHE.pop(next(iter(_BRIGHT_CACHE)))      # pools larger than the bound
        except (StopIteration, KeyError):
            break
    _BRIGHT_CACHE[keyframe_path] = val
    return val


# Words in an anchor scene's description that mark it as a DARK / night / interior moment — when
# present, a bright daytime shot is the wrong setting even if it's the right character.
_DARK_SCENE = re.compile(r"\b(tavern|night|candlelit|candle|dark|dungeon|cell|torch|firelit|"
                         r"interior|indoor|brothel|crypt|throne room|chamber|inn|cellar|nighttime)\b", re.I)


@dataclass
class _PoolShot:
    sid: str
    shot: Shot
    embed: object = None        # numpy vector or None


# A persistent rival-channel WATERMARK (a corner logo on every frame) is only OCR-legible on SOME
# frames, so the per-frame gate misses the rest. If a source shows junk text on a meaningful
# fraction of its shots, the watermark is persistent → drop the WHOLE source (env-tunable).
def _source_is_watermarked(shots, *, min_frac: float = 0.12, min_hits: int = 4) -> bool:
    if not shots:
        return False
    hits = sum(1 for sh in shots if _OCR_JUNK.search((getattr(sh, "ocr_text", "") or "")))
    return hits >= min_hits and (hits / len(shots)) >= min_frac


_CORNER_LOGO_CACHE: dict = {}


def _source_corner_logo(shots, *, samples: int = 16, min_kf: int = 6) -> str:
    """PIXEL-level static-corner-logo detector — the OCR-independent backstop for
    _source_is_watermarked. A stylized/graffiti/semi-transparent channel bug OCRs as garbage
    (or not at all), so keyword matching can't be the only detector.

    v3 (calibrated on 38 real sources — 7/7 true bugs found incl. a tiny semi-transparent 'BOC'
    and a vertical 'SociopathMD' edge bug, 0 false positives): EDGE-PERSISTENCE voting at 2× res.
    Scene edges move between shots; a logo's edges sit on the SAME pixels. Per keyframe, threshold
    the corner's edge map; a corner where the edged-pixel mask is PRESENT on ≥25% of keyframes,
    positionally CONSISTENT (mean IoU vs the majority mask ≥0.45), and 2D-clustered (not a
    letterbox line) is a bug. The frame centre must vary across shots — a static card is
    _source_is_static's call, not a logo.

    Returns 'tl'|'tr'|'bl'|'br' or ''. Memoized per source (keyed on the first keyframe path).
    Kill switch: VIDLORE_CLIPSTUDIO_CORNER_LOGO_GATE=0 (checked by callers).

    PERSISTED-FIRST: when the index computed multi-frame corner masks (3 samples/shot, OR'd),
    aggregate THOSE — no image IO, and intermittent bugs that fade in mid-shot are visible in
    the OR'd masks. Keyframe decoding below is the fallback for old indexes only."""
    import numpy as np
    from PIL import Image
    kfs = [getattr(sh, "keyframe_path", "") or "" for sh in shots]
    kfs = [k for k in kfs if k]
    if len(kfs) < min_kf:
        return ""
    ck = (kfs[0], len(kfs))
    if ck in _CORNER_LOGO_CACHE:
        return _CORNER_LOGO_CACHE[ck]
    while len(_CORNER_LOGO_CACHE) >= 256:
        _CORNER_LOGO_CACHE.pop(next(iter(_CORNER_LOGO_CACHE)))
    _CORNER_LOGO_CACHE[ck] = ""                       # default while computing (also on early-outs)
    def _pfv(sh):
        v = getattr(sh, "subs_flag", -1)
        return -1 if v is None else int(v)             # NOTE: `v or -1` would turn a valid 0 into -1

    flagged = [sh for sh in shots if _pfv(sh) >= 0]
    if len(flagged) >= min_kf:                        # multi-frame flags present → mask aggregation
        # static-card guard (ports the keyframe path's centre-variance early-out): a static
        # image + audio source has identical masks everywhere — that's _source_is_static's
        # call, not a corner bug (observed FP: the audiobook card fired 'br' here).
        if _source_is_static(shots):
            return ""
        from .index import _mask_from_hex
        best, best_score = "", 0.0
        for name in ("tl", "tr", "bl", "br"):
            masks = []
            for sh in flagged:
                h = (getattr(sh, "corner_masks", None) or {}).get(name)
                m = _mask_from_hex(h) if h else None
                masks.append(m)
            present = [m for m in masks if m is not None and m.mean() > 0.02]
            pf = len(present) / max(1, len(masks))
            if len(present) < 4 or pf < 0.25:
                continue
            maj = np.stack(present).mean(axis=0) >= 0.5
            if maj.sum() < 3 or maj.mean() > 0.85:
                continue
            iou = float(np.mean([float((m & maj).sum()) / max(1.0, float((m | maj).sum()))
                                 for m in present]))
            ys, xs = np.nonzero(maj)
            spread = min(ys.std(), xs.std()) if len(ys) > 2 else 0.0
            score = pf * iou
            if pf >= 0.25 and iou >= 0.45 and spread > 0.6 and score >= 0.20 \
                    and score > best_score:
                best, best_score = name, score
        _CORNER_LOGO_CACHE[ck] = best
        return best
    kfs = [k for k in kfs if Path(k).exists()]
    if len(kfs) < min_kf:
        return ""
    step = max(1, len(kfs) // samples)
    frames = []
    for k in kfs[::step][:samples]:
        try:
            frames.append(np.asarray(Image.open(k).convert("L").resize((640, 360)),
                                     dtype="float32"))
        except Exception:
            continue
    if len(frames) < min_kf:
        return ""
    small = np.stack([f[::4, ::4] for f in frames])   # 160×90 for the cheap centre-variance test
    if float(small[:, 14:76, 24:136].std(axis=0).mean()) < 8.0:
        return ""                                      # centre never changes → static card, not a bug
    edges = []
    for f in frames:
        gy, gx = np.gradient(f)
        edges.append(np.hypot(gx, gy))
    regions = {          # 18% × 12% patches at 640×360, flush to each corner
        "tl": (slice(0, 44), slice(0, 116)),   "tr": (slice(0, 44), slice(524, 640)),
        "bl": (slice(316, 360), slice(0, 116)), "br": (slice(316, 360), slice(524, 640)),
    }
    best, best_score = "", 0.0
    for name, (rs, cs) in regions.items():
        masks = [(e[rs, cs] > 18.0) for e in edges]
        present = [m for m in masks if m.mean() > 0.02]
        pf = len(present) / len(masks)
        if len(present) < 4 or pf < 0.25:
            continue
        maj = np.stack(present).mean(axis=0) >= 0.5   # majority mask = the bug's footprint
        if maj.sum() < 25 or maj.mean() > 0.85:
            continue                                   # too small, or the WHOLE corner "persists"
                                                       # (busy texture/noise, not a bounded logo)
        iou = float(np.mean([float((m & maj).sum()) / max(1.0, float((m | maj).sum()))
                             for m in present]))
        ys, xs = np.nonzero(maj)
        spread = min(ys.std(), xs.std()) if len(ys) > 12 else 0.0
        score = pf * iou
        if pf >= 0.25 and iou >= 0.45 and spread > 2.5 and score >= 0.20 and score > best_score:
            best, best_score = name, score
    _CORNER_LOGO_CACHE[ck] = best
    return best


def _source_is_static(shots, *, min_shots: int = 4) -> bool:
    """A still-image / lyric source — almost every shot is a phash near-duplicate of the others (a
    static portrait + audio, e.g. a 'X sings Rains of Castamere' card). It's not scene footage and,
    worse, it REPEATS the same still across the video. Detect a near-zero-variance phash set so the
    whole source can be dropped from the pool."""
    phs = [getattr(sh, "phash", "") for sh in shots if getattr(sh, "phash", "")]
    if len(phs) < min_shots:
        return False
    base = phs[0]
    near = sum(1 for p in phs if _index._hamming(base, p) <= 8)
    return (near / len(phs)) >= 0.85         # ≥85% of shots look identical → a static image


def _source_is_nonphotographic(proj, shots, *, sample: int = 6, frac: float = 0.55) -> bool:
    """A NON-LIVE-ACTION source — Playmobil/toy stop-motion, claymation, or AI/CGI-rendered
    'recreation' of the scene. These read instantly as fake in a serious analysis. Sample a
    few keyframes and run the CLIP photo-vs-art test (the same one the image fallback uses);
    if most are art/toy/render, drop the whole source."""
    try:
        from .image_fallback import _photographic_ok
        from . import index as _ix
        if not _ix.clip_available():
            return False
    except Exception:
        return False
    paths = [getattr(sh, "keyframe_path", "") for sh in shots if getattr(sh, "keyframe_path", "")]
    paths = [p for p in paths if p and Path(p).exists()]
    if len(paths) < 4:
        return False
    step = max(1, len(paths) // sample)
    chosen = paths[::step][:sample]
    art = sum(1 for p in chosen if not _photographic_ok(Path(p)))
    return art / max(1, len(chosen)) >= frac


# A modern PODCAST / on-camera VLOG / makeup-BTS / interview LOOKS nothing like a medieval-fantasy
# scene — that VISUAL gap is the reliable signal. (Face-ID cast-MATCHING is NOT: it under-matches
# real cast badly — on one render only 13% of face-shots matched, so 'few cast hits' flagged real
# dialogue scenes like Tyrion's trial. Never gate footage on cast-absence.) These CLIP prompts let
# us tell a modern talking-head from a period scene; the show-agnostic wording keeps it reusable.
_MODERN_TH_PROMPTS = (
    "a person talking into a podcast microphone",
    "a youtuber talking directly to the camera",
    "two people wearing headphones in a recording studio",
    "a modern television interview",
    "behind the scenes makeup with prosthetics being applied",
    "a person speaking to the camera with a microphone",
)
_PERIOD_SCENE_PROMPTS = (
    "a scene from a medieval fantasy television show",
    "actors in medieval costumes in a film scene",
    "a cinematic period drama scene",
    "a fantasy battle scene",
    "a dark castle interior scene",
)
_TH_TXT_CACHE: dict = {}     # prompt -> L2-normalized text embed (computed once per process)


def _source_is_modern_talkinghead(shots, embeds, *, min_face_shots: int = 8,
                                   min_face_density: float = 0.60, sample: int = 8,
                                   modern_frac: float = 0.70) -> bool:
    """The robust backstop for non-scene uploads whose TITLE dodges the keyword gates: a face-DENSE
    source that VISUALLY reads as a modern talking-head (podcast / vlog / interview / makeup-BTS)
    rather than a medieval-fantasy scene. Catches the two-host GoT podcast + the White-Walker
    makeup clip that leaked, while SPARING real dialogue scenes (Tyrion's trial, Old Nan, Hardhome
    all scored 0.0-0.5 modern vs the 0.70 bar in validation).

    Conservative on purpose — drops only when ALL hold:
      * face-DENSE (>=60% of shots show a face) — limits the (cheap) CLIP pass to suspicious sources;
      * enough evidence (>=8 face-bearing shots) — never judge a tiny source;
      * a strong MAJORITY (>=70%) of sampled face frames look more like a modern talking-head than a
        period scene under CLIP. If CLIP is unavailable we DON'T guess (return False) — the title
        gate still applies. `embeds` is the source's per-shot CLIP image-embed array (or None)."""
    if not shots or embeds is None:
        return False
    face_shots = [s for s in shots if int(getattr(s, "faces", 0) or 0) >= 1
                  and 0 <= getattr(s, "embed_row", -1) < len(embeds)]
    if len(face_shots) < min_face_shots:
        return False
    if len(face_shots) / len(shots) < min_face_density:
        return False
    try:
        import numpy as np
        from .image_fallback import _vr
        vr = _vr()
        if vr is None:
            return False
        def _txt(p):
            v = _TH_TXT_CACHE.get(p)
            if v is None:
                v = np.asarray(vr._txt_embed(p), "float32")
                v = v / (np.linalg.norm(v) + 1e-6)
                _TH_TXT_CACHE[p] = v
            return v
        mods = [_txt(p) for p in _MODERN_TH_PROMPTS]
        scns = [_txt(p) for p in _PERIOD_SCENE_PROMPTS]
        step = max(1, len(face_shots) // sample)
        chosen = face_shots[::step][:sample]
        modern = 0
        for s in chosen:
            ie = np.asarray(embeds[s.embed_row], "float32")
            ie = ie / (np.linalg.norm(ie) + 1e-6)
            m = max(float(np.dot(ie, t)) for t in mods)
            sc = max(float(np.dot(ie, t)) for t in scns)
            if m > sc + 0.005:
                modern += 1
        return (modern / len(chosen)) >= modern_frac
    except Exception:
        return False


def _load_pool(proj: ClipProject, cfg: Optional[ClipConfig] = None, progress=None,
               show_title: str = "") -> list[_PoolShot]:
    import os
    gate_on = os.environ.get("VIDLORE_CLIPSTUDIO_OCR_GATE", "1").strip() not in ("0", "false", "no", "")
    # "crop" (default) KEEPS a watermarked source — build.py punch-in-crops the logo off-frame, so
    # its footage stays available (relevance). "drop" excludes the whole source from the pool.
    # cfg.watermark_mode already defaults from the same env var, so a programmatic setting wins
    # and the env flag still works.
    wm_mode = ((getattr(cfg, "watermark_mode", "") or
                os.environ.get("VIDLORE_CLIPSTUDIO_WATERMARK_MODE", "crop")).strip().lower())
    nonshow_on = os.environ.get("VIDLORE_CLIPSTUDIO_NONSHOW_GATE", "1").strip() \
        not in ("0", "false", "no")
    from .discover import _NONSHOW_TITLE, _REJECT_TITLE, _REACTION_TITLE, _wrong_installment
    wrongshow_on = os.environ.get("VIDLORE_CLIPSTUDIO_WRONGSHOW_GATE", "1").strip() \
        not in ("0", "false", "no")
    pool: list[_PoolShot] = []
    # Visual footage gate (kill switch: VIDLORE_CLIPSTUDIO_FACE_FOOTAGE_GATE=0): a face-dense source
    # that CLIP reads as a modern talking-head (podcast / vlog / interview / makeup-BTS) rather than
    # a period scene is dropped — the robust backstop for non-scene uploads whose TITLE dodged the
    # keyword gates. Deliberately NOT based on Face-ID cast-matching, which under-matches real cast.
    face_gate_on = os.environ.get("VIDLORE_CLIPSTUDIO_FACE_FOOTAGE_GATE", "1").strip() \
        not in ("0", "false", "no", "")
    for src in proj.sources:
        if src.status != SOURCE_OK:
            continue
        # belt-and-suspenders for sources downloaded BEFORE the discovery wrong-show rule existed:
        # a franchise sibling/prequel (House of the Dragon in a Game of Thrones video) is the wrong
        # production — right world, wrong cast/era — and must never enter the footage pool.
        if wrongshow_on and show_title and _wrong_installment(show_title, src.title or ""):
            if progress:
                progress(f"match: dropping wrong-show source {src.id} "
                         f"(franchise sibling/prequel: {(src.title or '')[:48]!r})")
            continue
        if nonshow_on and _NONSHOW_TITLE.search(src.title or ""):
            if progress:
                progress(f"match: dropping non-show source {src.id} "
                         f"(game/AMV/animated: {(src.title or '')[:48]!r})")
            continue
        # REACTION/facecam video that slipped in (e.g. dialogue-verified back during discovery):
        # its footage is people on a couch over a tiny show inset — never let it into the pool.
        if nonshow_on and _REACTION_TITLE.search(src.title or ""):
            if progress:
                progress(f"match: dropping reaction/facecam source {src.id} "
                         f"({(src.title or '')[:48]!r})")
            continue
        # belt-and-suspenders: a talking-head / interview / featurette / promo source that was
        # downloaded BEFORE the discovery reject rule existed must still be kept OUT of the
        # footage pool (no scene footage — just a presenter / channel branding).
        if nonshow_on and _REJECT_TITLE.search(src.title or ""):
            if progress:
                progress(f"match: dropping talking-head/promo source {src.id} "
                         f"({(src.title or '')[:48]!r})")
            continue
        shots = _index.load_shots(proj, src.id)
        embeds = _index.load_embeds(proj, src.id)
        if face_gate_on and _source_is_modern_talkinghead(shots, embeds):
            if progress:
                progress(f"match: dropping modern talking-head source {src.id} "
                         f"(podcast/vlog/interview/makeup-BTS look, not a scene: {(src.title or '')[:48]!r})")
            continue
        if gate_on and wm_mode == "drop" and \
                (_source_is_watermarked(shots)
                 or (os.environ.get("VIDLORE_CLIPSTUDIO_CORNER_LOGO_GATE", "1").strip()
                     not in ("0", "false", "no") and _source_corner_logo(shots))):
            if progress:
                progress(f"match: dropping watermarked source {src.id} (persistent channel logo)")
            continue
        if _source_is_static(shots):              # still-image / lyric card — not scene footage
            if progress:
                progress(f"match: dropping static-image source {src.id} (repeating still, not footage)")
            continue
        if nonshow_on and _source_is_nonphotographic(proj, shots):
            if progress:
                progress(f"match: dropping non-live-action source {src.id} "
                         f"(toy/claymation/AI-render — not real footage)")
            continue
        for sh in shots:                              # `embeds` loaded once above (reused here)
            vec = None
            if embeds is not None and 0 <= sh.embed_row < len(embeds):
                vec = embeds[sh.embed_row]
            pool.append(_PoolShot(src.id, sh, vec))
    return pool


def _clip01(cos: float, cfg: ClipConfig) -> float:
    lo, hi = cfg.clip_cos_lo, cfg.clip_cos_hi
    if hi <= lo:
        return max(0.0, min(1.0, cos))
    return max(0.0, min(1.0, (cos - lo) / (hi - lo)))


def _norm_words(s: str) -> list[str]:
    return [w for w in re.findall(r"[a-z']+", (s or "").lower()) if len(w) > 1 and w not in _STOP]


def _dialogue_match(seg: ScriptSegment, transcript: str) -> float:
    """SCENE-LOCK: is the beat's iconic quote actually SPOKEN in this clip's ASR transcript? If the
    exact line ('break the wheel', 'I am the dragon's daughter') shows up in the speech, this clip IS
    the moment the script is about. Scores a contiguous-phrase run high, scattered words lower."""
    quote = (getattr(seg, "quote", "") or "").strip()
    if not quote or not transcript:
        return 0.0
    qw = _norm_words(quote)
    tw = _norm_words(transcript)
    if not qw or not tw:
        return 0.0
    if len(qw) == 1:                                               # a single iconic word (Dracarys, Mhysa)
        return 1.0 if (len(qw[0]) >= 5 and qw[0] in set(tw)) else 0.0
    tset = set(tw)
    overlap = sum(1 for w in qw if w in tset) / len(qw)            # word recall
    tstr = " " + " ".join(tw) + " "
    best_run = 0                                                   # longest contiguous quote-phrase in ASR
    for i in range(len(qw)):
        for j in range(len(qw), i + 1, -1):
            if (j - i) <= best_run:
                break
            if (" " + " ".join(qw[i:j]) + " ") in tstr:
                best_run = j - i
                break
    run_frac = best_run / len(qw)
    return max(0.0, min(1.0, 0.35 * overlap + 0.65 * run_frac))


def _text_sim(seg: ScriptSegment, transcript: str) -> float:
    """Recall-oriented overlap of the segment's content words/entities with the clip's speech."""
    if not transcript:
        return 0.0
    shot_tokens = {w.lower() for w in re.findall(r"[A-Za-z'][A-Za-z']+", transcript)
                   if w.lower() not in _STOP and len(w) > 2}
    if not shot_tokens:
        return 0.0
    seg_tokens = set(seg.keywords)
    ent_tokens = {w.lower() for e in seg.entities for w in e.split() if len(w) > 2}
    denom = len(seg_tokens) + len(ent_tokens)
    if denom == 0:
        return 0.0
    kw_hits = len(seg_tokens & shot_tokens)
    ent_hits = len(ent_tokens & shot_tokens)        # entity hits weighted double
    return min(1.0, (kw_hits + 2 * ent_hits) / max(1, denom))


def _trim_window(shot: Shot, seg: ScriptSegment, cfg: ClipConfig) -> tuple[float, float]:
    """Pick an [in,out] inside the shot ~matching the segment's screen time, centered on it."""
    L = max(cfg.min_clip_sec, min(cfg.max_clip_sec, shot.duration, seg.est_duration + 0.6))
    center = (shot.start + shot.end) / 2.0
    a = max(shot.start, center - L / 2.0)
    b = min(shot.end, a + L)
    a = max(shot.start, b - L)                      # re-pin if we hit the tail
    return round(a, 3), round(b, 3)


def _face_targets(seg: ScriptSegment, char2actor: dict) -> set:
    """Lowercased names to look for in a shot's Face-ID / OCR when this beat needs a person."""
    if seg.required_kind not in ("actor", "character") or not seg.required_entity:
        return set()
    t = seg.required_entity.strip().lower()
    targets = {t}
    if t in char2actor:
        targets.add(char2actor[t].lower())
    return targets


# On-frame text that marks a shot as an AD / NEWS-banner / CTA-card / channel-WATERMARK — NOT clean
# in-show footage. KEYWORD-based on purpose (never a word-count): a real dialogue subtitle like
# "we will not lay down our spears..." must pass, but "E! NEWS", "SUBSCRIBE", a ripper's channel
# watermark, or a donation card must be rejected. Frames whose OCR matches this are dropped from the
# candidate pool entirely (env VIDLORE_CLIPSTUDIO_OCR_GATE=0 disables).
_OCR_JUNK = re.compile(
    r"subscrib|e!?\s?news|\bcnn\b|\bbbc\b|click (to|here)|watch more|read more|link in (bio|desc)|"
    r"like and sub|comment below|hit the bell|donat|fundrais|red nose|wake ?a ?difference|"
    r"salary rumor|addresses?[a-z' ]{0,18}rumor|coming soon|official trailer|in theaters|"
    r"blacktr|reaction|react(s|ing)|patreon|onlyfans|\.com\b|www\.|@[a-z0-9_]{3,}|"
    r"plz do|please subscribe|new video|click to watch", re.I)


def _ocr_is_junk(shot) -> bool:
    txt = (getattr(shot, "ocr_text", "") or "").strip()
    return bool(txt) and bool(_OCR_JUNK.search(txt))


_SUBBAND_CACHE: dict = {}


def _shot_subtitle_band(shot) -> bool:
    """SCRIPT-AGNOSTIC burned-subtitle detector for one shot's keyframe. The OCR text gate only
    catches text RapidOCR can read — Arabic/Turkish burned subs sailed through (observed: a
    Turkish 'Oğlumsun.' aired over the privy scene). Visual heuristic instead: subtitles are a
    horizontal STRIPE of small high-contrast strokes in the bottom band — high edge density
    there vs the mid-frame, spanning many columns, concentrated in few rows. Calibrated on real
    sources (Turkish/Arabic/English-subbed positives, 30+ clean negatives).

    Kill switch: VIDLORE_CLIPSTUDIO_SUBBAND_GATE=0 (checked by callers). Memoized per keyframe.

    PERSISTED-FIRST: when the index computed multi-frame flags (3 samples/shot), trust them —
    they see subs that appear MID-shot which the keyframe instant misses (the remaining Turkish
    'Gel, geçip odamda konuşalım.' leak was exactly this). Keyframe heuristic is the fallback
    for old indexes only."""
    _pf = int(getattr(shot, "subs_flag", -1) if getattr(shot, "subs_flag", -1) is not None else -1)
    if _pf >= 0:
        return bool(_pf)
    kf = getattr(shot, "keyframe_path", "") or ""
    if not kf:
        return False
    if kf in _SUBBAND_CACHE:
        return _SUBBAND_CACHE[kf]
    while len(_SUBBAND_CACHE) >= 8192:
        _SUBBAND_CACHE.pop(next(iter(_SUBBAND_CACHE)))
    _SUBBAND_CACHE[kf] = False
    if not Path(kf).exists():
        return False
    try:
        import numpy as np
        from PIL import Image
        fr = np.asarray(Image.open(kf).convert("L").resize((320, 180)), dtype="float32")
        gy, gx = np.gradient(fr)
        E = np.hypot(gx, gy)
        band = E[137:175, 38:282] > 42.0              # bottom 76–97% height, x 12–88%
        bf = float(band.mean())
        mf = float((E[72:126, 38:282] > 42.0).mean())  # mid-frame reference density
        ys, xs = np.nonzero(band)
        colcov = len(np.unique(xs // 8)) / (244 // 8) if len(xs) else 0.0
        rowspread = ys.std() if len(ys) > 20 else 99.0
        ok = bf > 0.05 and bf > 2.0 * max(mf, 0.008) and colcov >= 0.28 and rowspread < 11.0
        _SUBBAND_CACHE[kf] = bool(ok)
        return bool(ok)
    except Exception:
        return False


def _shot_unreadable(shot) -> bool:
    """UNREADABLE-DARK shot — near-black across its WHOLE span, not just at one instant.
    Uses the persisted multi-frame luma (start/mid/end samples): `luma_avg` low AND even the
    brightest sample's 95th-percentile pixel (`luma_hi`) dim. A valid dark cinematic scene
    (candlelit privy, torch in the catacombs) keeps bright highlights → high p95 → passes.
    Old indexes (sentinel -1) are never gated here — the quality floor still applies.
    Env: VIDLORE_CLIPSTUDIO_UNREADABLE_GATE=0 disables; _AVG/_HI tune thresholds."""
    import os as _os_u
    if _os_u.environ.get("VIDLORE_CLIPSTUDIO_UNREADABLE_GATE", "1").strip() in ("0", "false", "no"):
        return False
    la = float(getattr(shot, "luma_avg", -1.0) or -1.0)
    lh = float(getattr(shot, "luma_hi", -1.0) or -1.0)
    if la < 0 or lh < 0:
        return False                                   # not computed (old index) → fail open
    # CONSERVATIVE by design (measured on real GoT footage): the readable dark privy/crossbow
    # shots sit at avg 12-14 / hi 127-147, true murk at avg 6-10 / hi 46-98. Overlap exists, so
    # the gate only fires on clear murk — a borderline dark scene stays (relevance-first).
    try:
        t_avg = float(_os_u.environ.get("VIDLORE_CLIPSTUDIO_UNREADABLE_AVG", "11") or 11)
        t_hi = float(_os_u.environ.get("VIDLORE_CLIPSTUDIO_UNREADABLE_HI", "90") or 90)
    except (TypeError, ValueError):
        t_avg, t_hi = 11.0, 90.0
    return la < t_avg and lh < t_hi


def _source_subs_frac(shots) -> float:
    """Fraction of a source's shots showing a burned-subtitle band — the SOURCE-level cleanliness
    signal for clean-copy arbitration. Burned subs are intermittent (they appear whenever dialogue
    plays), so ANY clip from a source with a meaningful fraction risks subs mid-clip even when its
    own keyframe happens to be clean."""
    if not shots:
        return 0.0
    hits = sum(1 for sh in shots if _shot_subtitle_band(sh))
    return hits / len(shots)


_LOGO_TOKENS = {"HBO", "MAX", "AMC", "FOX", "TNT", "ITV", "SKY", "CW"}


def _ocr_text_heavy(shot) -> bool:
    """Burned-in OVERLAY text — tweet/quote cards, lyric cards, hard subtitles, title slates,
    word-by-word animated captions. USER RULE: footage with readable on-screen text NEVER airs,
    no matter how relevant (env VIDLORE_CLIPSTUDIO_TEXT_GATE=0 disables). Signals: 3+ readable
    words, OR any ALL-CAPS word (big stylized captions OCR one word per keyframe — observed
    'TWO'/'BRAVOS'/'EVERY SINGLESTRIKE' on one essay). Known channel-bug tokens stay — the
    watermark crop handles corner logos."""
    txt = (getattr(shot, "ocr_text", "") or "")
    words = re.findall(r"[A-Za-z']{3,}", txt)
    if len(words) >= 3:
        return True
    if any(w.isupper() and w.upper() not in _LOGO_TOKENS for w in words):
        return True
    # OCR frequently concatenates a burned-in line WITHOUT spaces (hard subtitles, recap overlays,
    # foreign dubs, outro cards: "TheRedWeddingwasunexpected", "Ihavealwaysbeenyourson",
    # "VisitourWebsite", "THANKYOUFORWATCHING") — these scan as 1-2 long "words" and used to slip the
    # >=3-word check, so a source's red recap caption / channel outro / hard subtitle AIRED. A long
    # alpha run, or lots of letters overall, IS a readable text card/subtitle and must never air.
    if any(len(w) >= 14 for w in words):
        return True
    if sum(len(w) for w in words) >= 18:
        return True
    return False


def _score_pool(seg: ScriptSegment, pool: list[_PoolShot], text_vec, cfg: ClipConfig,
                face_targets: set, anchor_sids: set | None = None,
                anchor_bonus: float = 0.0,
                all_faces: set | None = None) -> list[tuple[float, float, dict, _PoolShot]]:
    import numpy as np
    import os
    gate_on = os.environ.get("VIDLORE_CLIPSTUDIO_OCR_GATE", "1").strip() not in ("0", "false", "no", "")
    tgate_on = os.environ.get("VIDLORE_CLIPSTUDIO_TEXT_GATE", "1").strip() not in ("0", "false", "no", "")
    subband_on = os.environ.get("VIDLORE_CLIPSTUDIO_SUBBAND_GATE", "1").strip() \
        not in ("0", "false", "no", "")
    wrongface_on = os.environ.get("VIDLORE_CLIPSTUDIO_WRONGFACE_GATE", "1").strip() \
        not in ("0", "false", "no", "")
    # BLACK-FRAME floor: a near-black / unusable keyframe (transition fade, outro background, a
    # source's blacked-out card) must never air. `quality` (0..1, brightness+sharpness+resolution)
    # is ~0 for those; floor drops only the worst, fail-open on uncomputed quality (defaults 1.0).
    try:
        _black_floor = float(os.environ.get("VIDLORE_CLIPSTUDIO_BLACK_FLOOR", "0.10") or 0.10)
    except (TypeError, ValueError):
        _black_floor = 0.10
    all_faces = all_faces or set()
    scored = []
    for ps in pool:
        if gate_on and _ocr_is_junk(ps.shot):
            continue                                      # drop ad/news/CTA/watermark frames
        if tgate_on and _ocr_text_heavy(ps.shot):
            continue                                      # readable overlay text NEVER airs
        if subband_on and _shot_subtitle_band(ps.shot):
            continue                                      # burned subs (any script) NEVER air
        if _black_floor > 0 and float(getattr(ps.shot, "quality", 1.0) or 1.0) < _black_floor:
            continue                                      # near-black / unusable frame never airs
        if _shot_unreadable(ps.shot):
            continue                                      # near-black across the WHOLE shot span
                                                          # (multi-frame luma; keeps candlelit scenes)
        clip_cos = 0.0
        clip01 = 0.0
        if text_vec is not None and ps.embed is not None:
            clip_cos = float(np.dot(text_vec, ps.embed))
            clip01 = _clip01(clip_cos, cfg)
        trans = _text_sim(seg, ps.shot.transcript)
        # Face-ID: does this shot actually show the required actor/character?
        faceid = 0.0
        wrongface = 0.0
        ids = {f.lower() for f in ps.shot.face_ids}
        if face_targets:
            ocrn = {o.lower() for o in ps.shot.ocr_names}
            if face_targets & ids:
                faceid = 1.0                          # confirmed by face recognition
            elif face_targets & ocrn:
                faceid = 0.8                          # confirmed by an on-screen name card
        # WRONG-CHARACTER GATE: this beat names a specific person, and the shot CONFIDENTLY
        # shows a DIFFERENT main character (Robb's face on a "Tywin" line). The matcher used
        # to only ADD a bonus for the right face and never penalize the wrong one, so when the
        # right character's footage was scarce the wrong character's clips won on CLIP alone.
        # Hard-penalize so a confirmed-wrong face loses to generic-but-not-wrong footage.
        if wrongface_on and face_targets and all_faces:
            conf_here = {(d.get("name", "") or "").lower()
                         for d in (getattr(ps.shot, "identities", None) or [])
                         if d.get("confident")}
            conf_here |= ids
            wrong = (conf_here & all_faces) - face_targets
            if wrong and not (conf_here & face_targets):
                wrongface = 1.0
        obj = 0.0
        if seg.keywords and ps.shot.tags:
            obj = min(1.0, len(set(seg.keywords) & {t.lower() for t in ps.shot.tags}) / max(1, len(seg.keywords)))
        dlg = _dialogue_match(seg, ps.shot.transcript)    # SCENE-LOCK: the line is spoken in this clip
        base = (cfg.w_clip * clip01 + cfg.w_trans * trans + cfg.w_face * faceid
                + cfg.w_obj * obj + cfg.w_dialogue * dlg)
        # ANCHOR CONTINUITY: for a single-scene deep-dive the editor must STAY on the one scene the
        # video is about (like a real essayist re-watching it), cutting only between shots WITHIN that
        # scene — not jumping to an unrelated battle for a transition line. Footage from an anchor
        # source gets a continuity bonus so the through-line holds, while dialogue-lock / CLIP still
        # decide WHICH shot of that scene fits each beat. The bonus drives RANKING only — it is kept
        # out of `base` so a zero-signal anchor shot can't report 0.45 confidence and dodge the
        # low-confidence review flag.
        bonus = anchor_bonus if (anchor_bonus and anchor_sids and ps.sid in anchor_sids) else 0.0
        # SOFT preference for clean frames: a burned-in source dialogue subtitle (readable on-frame
        # text that survived the junk-gate) clashes with our own narration caption. Small penalty so
        # a clean frame wins a near-tie, but a subtitled frame still wins if it's clearly more
        # relevant — never trade real relevance for cleanliness.
        otext = (getattr(ps.shot, "ocr_text", "") or "").strip()
        if gate_on and len(otext) >= 10:
            base -= cfg.subtitle_penalty
        # confirmed WRONG character → heavy penalty (drives it below any non-wrong shot,
        # but it is still a fallback if literally nothing else exists for the beat)
        if wrongface:
            base -= cfg.wrongface_penalty
        sig = {"clip": round(clip01, 4), "clip_cos": round(clip_cos, 4),
               "transcript": round(trans, 4), "faceid": round(faceid, 3),
               "object": round(obj, 3), "dialogue": round(dlg, 3),
               "quality": round(ps.shot.quality, 3)}
        if bonus:
            sig["anchor_bonus"] = round(bonus, 3)
        if wrongface:
            sig["wrongface"] = True
        scored.append((base, bonus, sig, ps))
    scored.sort(key=lambda x: x[0] + x[1], reverse=True)
    return scored


def _res_tier(h: int) -> int:
    """Resolution TIER (not raw pixels) so a 718p and a 720p copy tie: 1080-class=3, 720-class=2,
    480-class=1, sub-SD=0, unknown=1 (benefit of the doubt, below HD)."""
    if h >= 1000:
        return 3
    if h >= 640:
        return 2
    if h >= 440:
        return 1
    return 0 if h else 1


def _cleanliness_key(sid: str, shot, src_dirty: dict, src_height: dict) -> tuple:
    """Sort key for clean-copy arbitration — LOWER is cleaner. Ordered exactly per policy:
    watermark first, then burned-sub risk, then resolution tier, then luma/sharpness."""
    d = src_dirty.get(sid) or {}
    subs_risk = (float(d.get("subs", 0.0)) >= 0.12) or _shot_subtitle_band(shot)
    return (1 if d.get("corner") else 0,
            1 if subs_risk else 0,
            -_res_tier(int(src_height.get(sid, 0) or 0)),
            -float(getattr(shot, "quality", 0.5) or 0.5))


def _clean_copy_swap(seg, best, scored, src_dirty: dict, src_height: dict, cfg,
                     *, eps: float = 0.03):
    """SAME-SCENE CLEAN-COPY ARBITRATION. Scene compilations upload the same iconic moment many
    times — one copy clean 1080p, another with a channel bug / burned Turkish subs / 360p. The
    greedy pick takes the highest-scoring copy, which is blind to cleanliness (observed: a
    watermarked copy of the Tywin scene aired 16 beats while a clean 1080p copy of the SAME
    moment sat unused). When another source holds a NEAR-DUPLICATE of the winning shot
    (phash/CLIP-embed/ASR overlap) at practically the same relevance (within eps), prefer the
    cleanest copy: no watermark → no burned subs → higher resolution tier → sharper.

    Relevance-first is preserved by construction: only same-scene duplicates within eps swap —
    exact relevant footage is never replaced by an irrelevant HD shot.
    Kill switch: VIDLORE_CLIPSTUDIO_CLEAN_COPY_GATE=0. Returns (best, note|None)."""
    import numpy as np
    if best is None:
        return best, None
    _adj, base_b, ps_b, cand_b = best
    key_b = _cleanliness_key(ps_b.sid, ps_b.shot, src_dirty, src_height)
    if key_b[0] == 0 and key_b[1] == 0 and key_b[2] == -3:
        return best, None                          # already a clean 1080-class copy — nothing to win
    tb = (getattr(ps_b.shot, "transcript", "") or "").lower().split()
    alt, alt_key, alt_base = None, key_b, 0.0
    for base, bonus, sig, ps in scored:
        if ps.sid == ps_b.sid:
            continue                               # a different SOURCE = a different copy
        if base < base_b - eps:
            continue                               # relevance-first: never trade relevance away
        same = False
        if ps.shot.phash and ps_b.shot.phash:
            same = _index._hamming(ps.shot.phash, ps_b.shot.phash) <= cfg.dup_hamming
        if not same and ps.embed is not None and ps_b.embed is not None:
            same = float(np.dot(ps.embed, ps_b.embed)) >= cfg.near_dup_cos
        if not same and len(tb) >= 6:
            ta = (getattr(ps.shot, "transcript", "") or "").lower().split()
            if len(ta) >= 6:
                sa, sb = set(ta), set(tb)
                same = len(sa & sb) / max(1, min(len(sa), len(sb))) >= 0.6
        if not same:
            continue
        k = _cleanliness_key(ps.sid, ps.shot, src_dirty, src_height)
        if k < alt_key or (k == alt_key and alt is not None and base > alt_base):
            alt, alt_key, alt_base = (base, bonus, sig, ps), k, base
    if alt is None:
        return best, None
    base, bonus, sig, ps = alt
    in_p, out_p = _trim_window(ps.shot, seg, cfg)
    cand = ClipCandidate(segment_index=seg.index, source_id=ps.sid, shot_index=ps.shot.index,
                         score=round(max(0.0, min(1.0, base)), 4),
                         in_point=in_p, out_point=out_p, signals=sig)
    note = (f"match: clean-copy swap seg{seg.index} {ps_b.sid[:28]}→{ps.sid[:28]} "
            f"(dirty{key_b[:3]}→clean{alt_key[:3]}, Δrel={base_b - base:+.3f})")
    return (best[0], base, ps, cand), note


def match_segments(proj: ClipProject, segments: list[ScriptSegment], cfg: ClipConfig,
                   *, analysis=None, progress=None) -> list[ClipSelection]:
    """Greedy, constraint-aware selection. Fills and returns proj.selections.
    `analysis` (ScriptAnalysis) lets the matcher resolve a required character → its actor for
    Face-ID matching."""
    import numpy as np
    pool = _load_pool(proj, cfg, progress=progress,
                      show_title=(getattr(analysis, "movie_title", "") or "") if analysis is not None else "")
    if not pool:
        proj.selections = [ClipSelection(segment_index=s.index, source_id="", shot_index=-1,
                                         in_point=0, out_point=0, confidence=0.0) for s in segments]
        return proj.selections

    vr = _index._vr() if _index.clip_available() else None
    char2actor = analysis.char_to_actor() if analysis is not None else {}
    # the full set of MAIN-character + actor names (lowercased) — a shot confidently showing
    # one of these that ISN'T the beat's target is a wrong-character mismatch
    all_faces: set = set()
    for _c, _a in char2actor.items():
        if _c:
            all_faces.add(_c.lower())
        if _a:
            all_faces.add(_a.lower())

    # ANCHOR SOURCES — which downloaded sources ARE the core scene(s) the video is about (title echoes
    # an anchor scene by ≥2 scene-specific tokens). For a single-scene deep-dive we keep the cut ON
    # those sources so the editor walks through the one scene instead of scattering clips.
    import os as _os
    anchor_sids: set = set()
    anchor_bonus = 0.0
    anchor_desc = ""                  # the one scene's SETTING — biases every beat's CLIP query
    single_scene = (analysis is not None and getattr(analysis, "video_type", "") == "single_scene")
    # SINGLE-SCENE PURITY: an essay / "best scenes" / compilation source spans ALL seasons, so
    # its cutaway B-roll drops WRONG-ERA / WRONG-CHARACTER shots into a single-scene deep-dive
    # (e.g. short-hair S6 Cersei in an S1E5 scene). Prefer the raw scene clips — drop those
    # multi-season sources from the footage pool when >=2 cleaner sources remain. env-gated.
    # ERA FILTER: for a single-scene deep-dive whose episode is known (e.g. S01E05), a source
    # whose TITLE declares a DIFFERENT season ("Cersei 6x10", "Tyrion 8x05", "... S03E..") is
    # off-era footage (short-hair S6 Cersei in an S1 scene) — drop it. Title season labels are
    # explicit, so this is high-precision; sources with no season label are kept.
    def _title_season(t: str):
        t = (t or "").lower()
        m = (re.search(r"s0*(\d{1,2})\s*[ex]\d", t)            # S03E01 / s3e1
             or re.search(r"\b(\d{1,2})\s*x\s*\d{1,2}\b", t)   # 3x01
             or re.search(r"season\s*0*(\d{1,2})\b", t)        # season 3
             # bare season code: "Game of Thrones S3", "GoT S4" — common on clip titles and a
             # real era leak (a Pycelle-S3 source slipped onto an S01E07 stag-scene video). Require
             # a word boundary + 's' + 1-2 digits NOT followed by another digit/e/x (so S01E07
             # still parses via the first pattern, not as season 1 here).
             or re.search(r"\bs0*(\d{1,2})\b(?![\dex])", t))
        return int(m.group(1)) if m else None
    _ep = (getattr(analysis, "episode_hint", "") or "") if analysis is not None else ""
    _tgt_season = _title_season(_ep)
    if single_scene and _tgt_season and _os.environ.get(
            "VIDLORE_CLIPSTUDIO_ERA_FILTER", "1").strip() not in ("0", "false", "no"):
        _t2 = {s.id: (s.title or "") for s in proj.sources}
        _ok_ids = {ps.sid for ps in pool
                   if (_title_season(_t2.get(ps.sid, "")) in (None, _tgt_season))}
        if len({ps.sid for ps in pool}) - len(_ok_ids) >= 1 and len(_ok_ids) >= 1:
            _before = len(pool)
            pool = [ps for ps in pool if ps.sid in _ok_ids]
            if progress and len(pool) < _before:
                progress(f"match: era filter — dropped {_before - len(pool)} off-season shot(s) "
                         f"(target season {_tgt_season})")
    _purity = _os.environ.get("VIDLORE_CLIPSTUDIO_SINGLE_SCENE_PURITY", "1").strip() \
        not in ("0", "false", "no")
    if single_scene and _purity:
        _essay_comp = re.compile(
            r"psycholog|toxic|best scenes|supercut|\banalysis\b|breakdown|explained|video essay|"
            r"all .{0,20}scenes|every .{0,20}scene|do you want to know|the truth (about|behind)|"
            # editorial-essay HOOK titles: phrased as a thesis/claim, not a scene label. These
            # narrate over MULTI-ERA / MULTI-SHOW cutaways (an essay on the S1 chamber scene splices
            # in S6 short-hair Cersei and even House of the Dragon B-roll), which is exactly the
            # within-source contamination the title season-filter can't see.
            r"the scene that|that (changed|defined|broke|made)|"
            r"changed (game of thrones|the show|everything)|"
            r"\bwhy [a-z]+(?:'s)?\b.{0,40}\b(scene|moment|matters|works|is|was)\b|"
            r"the (real )?(meaning|genius|brilliance) of|here'?s (why|what|how)", re.I)
        _title = {s.id: (s.title or "") for s in proj.sources}
        _clean = [ps for ps in pool if not _essay_comp.search(_title.get(ps.sid, ""))]
        _clean_src = {ps.sid for ps in _clean}
        # only prune when the CLEAN pool is still rich enough to fill every beat with variety
        # (>=3 clean sources AND >=2.5x as many clean shots as beats) — otherwise dropping the
        # essay/compilation footage starves the matcher and forces heavy shot REPETITION, which
        # hurts retention more than an occasional off-era cutaway. Keep everything in that case.
        if len(_clean_src) >= 3 and len(_clean) >= 2.5 * max(1, len(segments)):
            _before = len(pool)
            pool = _clean
            if progress and len(pool) < _before:
                progress(f"match: single-scene purity — dropped {_before - len(pool)} "
                         f"essay/compilation shot(s) (rich clean pool remains)")
        elif progress:
            progress("match: single-scene purity skipped — clean pool too small "
                     "(keeping essay/compilation footage to avoid repetition)")
    if analysis is not None and getattr(analysis, "anchor_scenes", None):
        anchor_desc = " ".join(((sc.get("name", "") + " " + sc.get("query", "")).strip())
                               for sc in analysis.anchor_scenes).strip()[:160]
        mtoks = {w for w in re.findall(r"\w+", (analysis.movie_title or "").lower()) if len(w) > 2}
        atok_sets = []
        for sc in analysis.anchor_scenes:
            ts = {w for w in re.findall(r"\w+", (sc.get("query", "") + " " + sc.get("name", "")).lower())
                  if len(w) > 2 and w not in mtoks and w not in _STOP}
            if ts:
                atok_sets.append(ts)
        # entity tokens (character/actor names) — a title-only anchor match must include one,
        # or a song/location word shared across scenes makes a FALSE anchor (the Red Wedding
        # "Rains of Castamere" clip hijacked a Bronn-tavern video this way)
        ent_toks = set()
        for ch in (getattr(analysis, "characters", None) or []):
            for w in re.findall(r"\w+", (ch.get("name") or "").lower()):
                if len(w) > 2 and w not in _STOP:
                    ent_toks.add(w)
        for ac in (getattr(analysis, "actors", None) or []):
            for w in re.findall(r"\w+", (ac or "").lower()):
                if len(w) > 2 and w not in _STOP:
                    ent_toks.add(w)
        # the scene's spoken lines — CONTENT-level anchor proof via each source's own ASR
        # (anchor dialogue from analysis is VERBATIM; per-beat quotes are often paraphrases)
        _quotes = []
        for sc in (getattr(analysis, "anchor_scenes", None) or []):
            for d in (sc.get("dialogue") or []):
                if isinstance(d, str) and len(d.split()) >= 3:
                    _quotes.append(d)
        for s in segments:
            qt = (getattr(s, "quote", "") or "").strip().strip('"“”')
            if len(qt) >= 12 and len(qt.split()) >= 3:
                _quotes.append(qt)
        _tr_by_sid: dict = {}
        for ps in pool:
            if ps.shot.transcript:
                _tr_by_sid.setdefault(ps.sid, []).append(ps.shot.transcript)

        def _raw_words(s):
            # NO stopword filter here: iconic quotes are made of common words
            # ("you're just like me, only smaller") — filtering would erase them
            return re.findall(r"[a-z']+", (s or "").lower())

        _ep = (getattr(analysis, "episode_hint", "") or "").strip().lower().replace(" ", "")
        for src in proj.sources:
            # (a) dialogue-verified at discovery time = anchor, full stop
            if (getattr(src, "extra", None) or {}).get("anchor_verified"):
                anchor_sids.add(src.id)
                continue
            # (a2) the exact EPISODE CODE in the title = the raw scene itself (a "chair scraping
            # scene S03E03" upload has no character name but IS the scene) — strong enough alone
            if _ep and _ep in (src.title or "").lower().replace(" ", ""):
                anchor_sids.add(src.id)
                continue
            # (b) the source's own ASR SPEAKS one of the scene's DIALOGUE lines (scene-specific →
            # one solid hit is decisive). Music sources are already dropped from the pool upstream.
            joined = " " + " ".join(_raw_words(" ".join(_tr_by_sid.get(src.id, [])))) + " "
            spoken = False
            for qt in _quotes:
                qw = _raw_words(qt)[:6]
                if len(qw) >= 3 and (" " + " ".join(qw) + " ") in joined:
                    spoken = True
                    break
            if spoken:
                anchor_sids.add(src.id)
                continue
            # (c) title fallback: ≥2 scene tokens AND at least one is a character/actor name.
            # Token match, not raw substring — prefix only for short plural/possessive suffixes.
            twords = set(re.findall(r"\w+", (src.title or "").lower()))
            for ts in atok_sets:
                hits = {t for t in ts
                        if any(w == t or (w.startswith(t) and len(w) - len(t) <= 2)
                               for w in twords)}
                if len(hits) >= 2 and (not ent_toks or hits & ent_toks):
                    anchor_sids.add(src.id)
                    break
        anchor_bonus = _f_env("VIDLORE_CLIPSTUDIO_ANCHOR_BONUS", 0.45 if single_scene else 0.12)
    # DARK-SCENE guard — if the one scene is a night/candlelit/interior moment, a bright daytime shot
    # is the wrong setting (the daylight-forest Bronn that clashed with a night-tavern narration).
    # Scoped to the ANCHOR description with movie-title words blanked IN PLACE — deleting tokens
    # and re-joining would splice non-adjacent words ("the throne. The room" → "throne room") into
    # false multi-word matches; and only meaningful title words are blanked ("the" never is).
    _ttoks = {w for w in re.findall(r"\w+", (getattr(analysis, "movie_title", "") or "").lower())
              if len(w) > 2 and w not in _STOP}
    _adesc = anchor_desc.lower()
    for _t in _ttoks:
        _adesc = re.sub(rf"\b{re.escape(_t)}\b", " ", _adesc)
    dark_scene = single_scene and bool(_DARK_SCENE.search(_adesc))
    if anchor_sids and progress:
        progress(f"match: {len(anchor_sids)} anchor source(s) · bonus={anchor_bonus} "
                 f"· dark_scene={dark_scene} · type={getattr(analysis,'video_type','')}")

    # FOOTAGE-ABUNDANCE GUARD: the per-shot `relax` below lets anchor sources recur freely — right for
    # a TRUE one-scene deep-dive with only a few uploads, but when the long-form source floor pulls in
    # MANY anchor sources, relaxing reuse re-airs a few of them dozens of times while equally-relevant
    # context sources sit unused (observed: a Cleganebowl deep-dive aired one source 26× / one window
    # 13× with 14 relevant sources never touched). With abundant anchor footage keep the NORMAL
    # anti-reuse so the cut spreads across what discovery actually found.
    _anchor_abundant = len(anchor_sids) > int(_f_env("VIDLORE_CLIPSTUDIO_ANCHOR_SCARCE_MAX", 6))
    _src_height = {s.id: int(getattr(s, "height", 0) or 0) for s in proj.sources}
    # CLEANLINESS MAP for same-scene clean-copy arbitration: per-source corner-bug + burned-sub
    # fraction, computed once over the pooled shots (both detectors are memoized).
    import os as _os_cc
    _clean_gate = _os_cc.environ.get("VIDLORE_CLIPSTUDIO_CLEAN_COPY_GATE", "1").strip() \
        not in ("0", "false", "no")
    _corner_gate = _os_cc.environ.get("VIDLORE_CLIPSTUDIO_CORNER_LOGO_GATE", "1").strip() \
        not in ("0", "false", "no")
    _src_dirty: dict = {}
    if _clean_gate:
        _by_src_shots: dict = {}
        for ps in pool:
            _by_src_shots.setdefault(ps.sid, []).append(ps.shot)
        for _sid, _shs in _by_src_shots.items():
            _src_dirty[_sid] = {"corner": (_source_corner_logo(_shs) if _corner_gate else ""),
                                "subs": _source_subs_frac(_shs)}
        _dirty_n = sum(1 for v in _src_dirty.values() if v["corner"] or v["subs"] >= 0.12)
        if progress and _dirty_n:
            progress(f"match: cleanliness map — {_dirty_n}/{len(_src_dirty)} source(s) carry a "
                     f"corner bug or burned subs (clean-copy arbitration active)")
    source_uses: dict[str, int] = {}
    shot_uses: dict[tuple[str, int], int] = {}
    recent_sources: list[str] = []
    recent: list[dict] = []          # recency window: {pos, key, phash, embed} of recent picks
    selections: list[ClipSelection] = []

    _hist_win = max(cfg.recency_cooldown, cfg.source_recency_window)
    for seg in segments:
        # recency/source history only matters within the cooldown window — prune so candidate
        # scoring stays O(pool × window) instead of O(pool × all prior segments)
        if recent:
            recent[:] = [h for h in recent if (seg.index - h["pos"]) < _hist_win]
        if len(recent_sources) > 4 * max(1, cfg.max_consecutive_same_source):
            del recent_sources[:-cfg.max_consecutive_same_source]
        text_vec = None
        if vr is not None:
            try:
                # build the CLIP query from the MOST concrete description of the exact moment:
                # the scene_query + expected_visual (LLM), falling back to the narration line.
                _q = " ".join(x for x in (getattr(seg, "scene_query", ""),
                                          seg.expected_visual) if x).strip() or seg.text
                # SINGLE-SCENE SETTING BIAS: an abstract analysis line ("he's just like the Hound")
                # has no setting words, so plain CLIP grabbed ANY shot of the right character — even a
                # bright daytime exterior that clashes with a dark-tavern narration. Appending the
                # anchor scene's description pulls EVERY beat toward the one scene's setting/look, so
                # the cut stays in the candlelit tavern instead of wandering across the character's arc.
                if single_scene and anchor_desc:
                    _q = f"{_q} {anchor_desc}".strip()
                text_vec = np.asarray(vr._txt_embed(_q), dtype="float32")
            except Exception:
                text_vec = None
        face_targets = _face_targets(seg, char2actor)
        scored = _score_pool(seg, pool, text_vec, cfg, face_targets, anchor_sids, anchor_bonus,
                             all_faces=all_faces)

        # `base` = match QUALITY (drives the reported confidence + flagging).
        # `adj`  = base + anchor bonus minus diversity penalties (drives only WHICH is picked).
        # Decoupled so anti-reuse/anchor-continuity never distort the reported confidence.
        best = None    # (adj, base, ps, cand)
        alt_best: dict[str, ClipCandidate] = {}    # best candidate per source → alternates
        # VISUAL-POLICY variety (req. 2/5): filler/character beats should SPREAD across sources, so
        # their reuse + source-recency penalties are amplified. exact_scene beats keep the multiplier
        # at 1.0 — the precise scene must win even from a recently-used source.
        _variety = 1.6 if _policy.maximize_variety(seg) else 1.0
        for base, bonus, sig, ps in scored:
            key = (ps.sid, ps.shot.index)
            # SINGLE-SCENE DEEP-DIVE: the anchor scene IS the video — an expert essayist keeps
            # cutting BETWEEN that one scene's shots (Bronn → the mug → the Hound → reaction), so the
            # limited anchor footage must be allowed to recur. We relax reuse/source/recency penalties
            # for anchor sources (but still block the IDENTICAL shot back-to-back via same-shot
            # recency, so it never looks like a freeze/loop).
            relax = (single_scene and bool(anchor_sids) and ps.sid in anchor_sids
                     and not _anchor_abundant)    # abundant anchor footage → normal anti-reuse, spread
            shot_cap = cfg.max_reuse_per_shot * (3 if relax else 1)
            if shot_uses.get(key, 0) >= shot_cap:
                continue                              # this exact shot is exhausted
            pen = cfg.reuse_penalty * source_uses.get(ps.sid, 0) * (0.25 if relax else _variety)
            if not relax and len(recent_sources) >= cfg.max_consecutive_same_source and \
               all(s == ps.sid for s in recent_sources[-cfg.max_consecutive_same_source:]):
                pen += 0.15                           # break long runs from one source
            if ps.shot.dup_of >= 0 and not relax:
                pen += 0.10                            # prefer unique scenes over near-duplicates
            if dark_scene:                            # a night/candlelit scene wants the candlelit
                b = _shot_brightness(getattr(ps.shot, "keyframe_path", ""))   # sweet spot, not daylight
                if b > 0.44:                          # bright daytime exterior — strongly wrong setting
                    pen += min(0.6, (b - 0.44) * 2.4)
                elif b < 0.10:                        # unreadably black filler — also avoid
                    pen += 0.15
            # RECENCY: penalize this shot, the same continuous SCENE, and visual near-duplicates if
            # shown recently — decaying to 0 across recency_cooldown beats. "Same scene" = the exact
            # shot, OR another shot from the SAME source whose time range is within scene_gap_sec
            # (adjacent cuts of one continuous take look identical to a viewer), OR a phash/CLIP
            # near-duplicate (catches the same iconic shot reused across different source files).
            rec_pen = 0.0
            src_pen = 0.0
            for h in recent:
                gap = seg.index - h["pos"]
                if gap <= 0:
                    continue
                if ps.sid == h["sid"] and gap < cfg.source_recency_window and not relax:
                    src_pen = max(src_pen, cfg.source_recency_weight * _variety
                                  * (1.0 - gap / cfg.source_recency_window))
                if gap >= cfg.recency_cooldown:
                    continue
                same_shot = (key == h["key"])
                near = same_shot
                # The ADJACENT-TIME "same scene" block over-suppresses legit DIFFERENT shots of the one
                # anchor scene — relax drops only THAT. But phash/embed VISUAL near-duplicate blocking
                # stays ON even in relax, so a near-identical frame (e.g. a static lyric card, or the
                # exact same iconic shot from two files) never repeats across the deep-dive.
                if not relax and not near and ps.sid == h["sid"] and \
                        not (ps.shot.end < h["t0"] - cfg.scene_gap_sec or
                             ps.shot.start > h["t1"] + cfg.scene_gap_sec):
                    near = True                       # same source, overlapping/adjacent time = same scene
                if not near and ps.shot.phash and h["phash"]:
                    near = _index._hamming(ps.shot.phash, h["phash"]) <= cfg.dup_hamming
                if not near and ps.embed is not None and h["embed"] is not None:
                    near = float(np.dot(ps.embed, h["embed"])) >= cfg.near_dup_cos
                if near:
                    # always keep a STRONG block on the identical shot back-to-back (even in relax
                    # mode) so the scene never freezes/loops; relax only softens the broader scene block
                    w = cfg.recency_weight * (0.5 if (relax and same_shot) else 1.0)
                    rec_pen = max(rec_pen, w * (1.0 - gap / cfg.recency_cooldown))
            pen += rec_pen + src_pen
            # RESOLUTION preference (selection only, not reported confidence): among similarly
            # relevant shots, prefer an HD source — an SD (≤480p) exact-scene upload upscales to
            # a soft 1080p, so a 1080p clip of the same scene should win more beats. Sub-SD
            # (<480p) additionally pays a small flat penalty: a 360p stream on the 1080p canvas
            # is visibly soft even after detail-enhance (observed: 17% of beats aired 360p while
            # HD alternates existed). Both terms stay SMALL — an exact scene at 360p must still
            # beat an irrelevant 1080p shot (relevance-first policy).
            _sh = _src_height.get(ps.sid, 0)
            hd_pref = 0.06 * min(1.0, _sh / 1080.0) if _sh else 0.0
            if 0 < _sh < 480:
                hd_pref -= 0.04
            adj = max(0.0, base + bonus - pen + hd_pref
                      + 0.08 * (ps.shot.quality - 0.5))  # mild quality pref
            qual = round(max(0.0, min(1.0, base)), 4)
            in_p, out_p = _trim_window(ps.shot, seg, cfg)
            cand = ClipCandidate(segment_index=seg.index, source_id=ps.sid,
                                 shot_index=ps.shot.index, score=qual,
                                 in_point=in_p, out_point=out_p, signals=sig)
            if best is None or adj > best[0]:
                if best is not None:
                    prev = best[3]
                    cur = alt_best.get(prev.source_id)
                    if cur is None or prev.score > cur.score:
                        alt_best[prev.source_id] = prev
                best = (adj, base, ps, cand)
            else:
                cur = alt_best.get(ps.sid)
                if cur is None or cand.score > cur.score:
                    alt_best[ps.sid] = cand

        # SAME-SCENE CLEAN-COPY ARBITRATION: if another source holds a near-duplicate of the
        # winning shot at ~equal relevance, air the cleanest copy (no watermark → no burned subs →
        # higher resolution → sharper). The displaced pick stays available as an alternate.
        if _clean_gate and best is not None:
            _old_cand = best[3]
            best, _swap_note = _clean_copy_swap(seg, best, scored, _src_dirty, _src_height, cfg)
            if _swap_note:
                _pc = alt_best.get(_old_cand.source_id)
                if _pc is None or _old_cand.score > _pc.score:
                    alt_best[_old_cand.source_id] = _old_cand
                if progress:
                    progress(_swap_note)

        # alternates: best candidate per source, ORDERED best-first — the verifier's repair only
        # tries the first few, and beat_windows inherit this order. Rank by score PLUS the anchor
        # bonus (recorded in signals): a single-scene deep-dive's windows must keep leading with
        # the anchor scene's shots, while the reported score stays pure quality.
        if best is not None:
            _ckey = (best[3].source_id, best[3].shot_index)
            alternates = sorted(
                (c for c in alt_best.values() if (c.source_id, c.shot_index) != _ckey),
                key=lambda c: c.score + float((c.signals or {}).get("anchor_bonus", 0.0)),
                reverse=True)[:cfg.candidates_per_segment]
        else:
            alternates = []

        if best is None:
            sel = ClipSelection(segment_index=seg.index, source_id="", shot_index=-1,
                                in_point=0, out_point=0, confidence=0.0)
        else:
            adj, base_q, ps, cand = best
            sel = ClipSelection(
                segment_index=seg.index, source_id=ps.sid, shot_index=ps.shot.index,
                in_point=cand.in_point, out_point=cand.out_point,
                confidence=round(max(0.0, min(1.0, base_q)), 4),
                signals=cand.signals, reuse_count=source_uses.get(ps.sid, 0),
                alternates=alternates[:cfg.candidates_per_segment],
                source_url=(proj.source(ps.sid).url if proj.source(ps.sid) else ""),
            )
            if ps.shot.face_ids:
                sel.identity = ps.shot.face_ids[0]
                if ps.shot.identities:
                    sel.identity_score = float(ps.shot.identities[0].get("score", 0.0))
            # multi-clip windows: chosen shot + distinct alternates (already source-diverse &
            # recency-filtered above) → the engine's intra-scene sub-beats each get a DIFFERENT clip.
            wins, seen_w = [], set()
            for c in [cand] + alternates[:cfg.candidates_per_segment]:
                wk = (c.source_id, c.shot_index)
                if wk in seen_w:
                    continue
                seen_w.add(wk)
                wins.append([c.source_id, round(c.in_point, 3), round(c.out_point, 3)])
            sel.beat_windows = wins
            source_uses[ps.sid] = source_uses.get(ps.sid, 0) + 1
            shot_uses[(ps.sid, ps.shot.index)] = shot_uses.get((ps.sid, ps.shot.index), 0) + 1
            recent_sources.append(ps.sid)
            recent.append({"pos": seg.index, "key": (ps.sid, ps.shot.index), "sid": ps.sid,
                           "t0": ps.shot.start, "t1": ps.shot.end,
                           "phash": ps.shot.phash, "embed": ps.embed})
        selections.append(sel)
        if progress and (seg.index % 20 == 0):
            progress(f"match: {seg.index+1}/{len(segments)} conf={sel.confidence}")

    proj.selections = selections
    return selections
