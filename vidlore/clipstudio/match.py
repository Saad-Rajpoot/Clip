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


def banned_source_ids(proj, *, include_auto: bool = True) -> set:
    """Source-ids banned from EVERY pool for this render.

    A ban means "this upload is not authentic show footage" (fan film, AI recreation,
    alternate-ending re-cut) — so it must be excluded everywhere a frame or a line can reach
    the timeline, not just from match's moving-clip pool. Three consumers share this list:
    match (moving clips), the still/image-fallback pool, and breakout selection (real-audio
    dialogue). Honouring it in only one of them is how an AI-recreation still keeps airing —
    or worse, keeps its real-audio BREAKOUT — after the operator has already banned it.

    Sources: proj.meta['banned_sources'] (persisted, so a re-render reproduces the ban
    deterministically) plus VIDLORE_CLIPSTUDIO_BANNED_SOURCES (comma-separated, ad hoc), plus
    proj.meta['auto_rejected_sources'] — the SOURCE-LEVEL rejections _load_pool makes while
    building the match pool (subtitled copy, static image, non-photographic, watermarked, modern
    talking-head).

    That last one closes a real leak. Those rejections used to be bare `continue`s inside
    _load_pool, so they removed the source from match's pool ONLY — the still/image-fallback pool
    and build's shot-walk read proj.sources directly and happily aired it anyway. Measured on job
    69d80e9dd4: 'How Game of Thrones Filmed Arya And Brienne's Sword Fight' was dropped by
    _load_pool as a subtitled copy, contributed ZERO beats to selections, and still put
    behind-the-scenes stunt-rehearsal footage (modern t-shirts, gym mats, Nike trainers) on screen
    three times. Same doctrine as the operator ban: reject once, hold everywhere.

    include_auto=False is for _load_pool itself: it RE-DERIVES those rejections from the shots on
    every call, so reading its own persisted output back would make the auto-bans sticky and a gate
    kill-switch could never re-admit a source."""
    import os as _os_bl
    _m = getattr(proj, "meta", None) or {}
    out = {str(x) for x in (_m.get("banned_sources") or [])}
    if include_auto:
        out |= {str(x) for x in (_m.get("auto_rejected_sources") or [])}
    out |= {x.strip() for x in
            _os_bl.environ.get("VIDLORE_CLIPSTUDIO_BANNED_SOURCES", "").split(",") if x.strip()}
    return out


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


# CHANNEL-PROMO OVERLAY: a burned "SUBSCRIBE"/"SUBSCRIBED"/bell/"thanks for watching" card is proof
# the upload is a re-packaged fan compilation (quiz, listicle, reaction-adjacent) rather than a clean
# scene rip — and such uploads carry the rest of the packaging too: numbered listicle titles, end
# cards, promo lower-thirds. The tokens are already sitting in the per-shot OCR text from indexing,
# so this costs nothing. Measured on job 69d80e9dd4 (1983 shots / 43 sources): exactly 3 sources fire,
# all three genuinely promo-bearing — including 'Can you recognize all Valyrian steel weapons?', whose
# "2. NEEDLE" listicle title aired over a beat about Arya's execution of Littlefinger.
_PROMO_OVERLAY_RX = re.compile(
    r"subscrib\w*|smash the (like|bell)|bell icon|turn on notification|hit the bell|"
    r"thank ?you ?for ?watching|thanks ?for ?watching|like ?(and|&) ?subscribe|"
    r"link in (the )?(bio|description)|patreon\.com|join this channel", re.I)


def _source_has_promo_overlay(shots, *, min_hits: int = 1) -> bool:
    """True when a source burns channel-promo furniture into its BODY (not intro/outro cards).

    The boundary exclusion is the whole point. Nearly every clean scene rip begins or ends with a
    dedicated channel card. Measured across two finished jobs, 4 of the 5 promo-bearing sources had
    their ONLY promo shot in the last seconds; the 101-beat incident also had an otherwise-usable
    170.9s 1080p scene rip whose sole ``LIKE / COMMENT / SUBSCRIBE`` card occupied 8.22-15.06s,
    wholly inside its first 10%. The source-level verdict discarded the exact Catspaw scene at
    95-108s even though the existing per-shot text gate already excludes the intro card itself.

    A promo overlay in the BODY is different: it means the upload is packaged content
    (quiz/listicle/compilation) that also carries numbered titles and lower-thirds — that is the
    class that put another channel's '2. NEEDLE' title on screen. The intro exemption is therefore
    deliberately narrow: at most the first 10% and never more than 20 seconds, and the entire promo
    shot must end inside that zone. An overlay which starts near the boundary but continues into the
    programme remains a body hit. Per-shot text gates still reject every exempt card's own pixels.

    Per-shot text gates still handle the outro shots themselves; this is only the source verdict."""
    if not shots:
        return False
    try:
        end = max(float(getattr(s, "end", 0.0) or 0.0) for s in shots)
    except ValueError:
        return False
    # intro = first 10%, capped at 20s.  Require the WHOLE shot to fit: a persistent overlay that
    # starts during the intro but continues over footage is body furniture, not an intro card.
    intro_end = min(end * 0.10, 20.0) if end > 0 else 0.0
    # tail = last 20% or last 30s, whichever starts earlier (long uploads get the seconds rule)
    tail_start = min(end * 0.80, end - 30.0) if end > 0 else 0.0
    hits = 0
    for sh in shots:
        t = (getattr(sh, "ocr_text", "") or "")
        if not t or not _PROMO_OVERLAY_RX.search(t):
            continue
        if float(getattr(sh, "end", 0.0) or 0.0) <= intro_end:
            continue                                   # dedicated intro card — shot gate owns it
        if float(getattr(sh, "start", 0.0) or 0.0) >= tail_start:
            continue                                   # outro end-card — not a packaging signal
        hits += 1
        if hits >= min_hits:
            return True
    return False


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


_EDGE_LOGO_CACHE: dict = {}


def _source_edge_logo(shots, *, samples: int = 16, min_kf: int = 8) -> str:
    """Persistent overlay in a vertical EDGE BAND rather than a corner. Returns 'l' | 'r' | ''.

    MEASURED on portal job 409e284b60. A media-player badge — an orange rounded square with a
    white 'm' and a running timer ('53:5…', '54:0…', '54:4…') — aired on four beats, burned into
    four different re-uploads of the same screen recording. `_source_corner_logo` could not see it
    for two independent reasons, and both are geometric rather than statistical:

      * it sits at MID-HEIGHT on the right border (y 46-56% of frame), so no corner patch — the
        outer 18%x12% boxes — contains a single pixel of it;
      * its digits change every second, so any test that asks the whole patch to be static fails
        even where the patch does overlap.

    Same statistic as the corner detector, different geometry: threshold each keyframe's edge map
    inside the outer 9% of width over the FULL height, and look for a footprint that is present on
    nearly every keyframe (pf), lands on the same pixels (IoU vs the majority mask), and is a
    COMPACT blob rather than a full-height line — the last one is what separates a badge from a
    pillarbox seam, which is otherwise a perfect static edge.

    Calibrated on that render's 74 indexed sources: fires on 5, and all 5 carry a real burned-in
    overlay ('SPHINX TV', 'FAVORITE FLASHBACKS FRENZY', a bottom-edge channel mark, and the two
    player-badge re-uploads) — 0 false positives. The remaining two badge sources show it only
    intermittently (pf 0.25), so a source-level detector cannot reach them; that is a per-shot
    question and is left to the per-shot overlay rule.

    Kill switch: VIDLORE_CLIPSTUDIO_EDGE_LOGO_GATE=0 (checked by callers)."""
    import numpy as np
    from PIL import Image
    kfs = [k for k in ((getattr(sh, "keyframe_path", "") or "") for sh in shots) if k]
    if len(kfs) < min_kf:
        return ""
    ck = (kfs[0], len(kfs))
    if ck in _EDGE_LOGO_CACHE:
        return _EDGE_LOGO_CACHE[ck]
    while len(_EDGE_LOGO_CACHE) >= 256:
        _EDGE_LOGO_CACHE.pop(next(iter(_EDGE_LOGO_CACHE)))
    _EDGE_LOGO_CACHE[ck] = ""
    if _source_is_static(shots):
        return ""                                   # a static card's edges persist everywhere
    kfs = [k for k in kfs if Path(k).exists()]
    if len(kfs) < min_kf:
        return ""
    W, H, BAND = 640, 360, 0.09
    step = max(1, len(kfs) // samples)
    frames = []
    for k in kfs[::step][:samples]:
        try:
            frames.append(np.asarray(Image.open(k).convert("L").resize((W, H)), dtype="float32"))
        except Exception:                            # noqa: BLE001, PERF203
            continue
    if len(frames) < min_kf:
        return ""
    bw = int(W * BAND)
    best, best_score = "", 0.0
    for side in ("l", "r"):
        masks = []
        for f in frames:
            band = f[:, W - bw:] if side == "r" else f[:, :bw]
            gx = np.abs(np.diff(band, axis=1, prepend=band[:, :1]))
            gy = np.abs(np.diff(band, axis=0, prepend=band[:1, :]))
            e = np.maximum(gx, gy)
            masks.append(e >= max(18.0, float(np.percentile(e, 96))))
        present = [m for m in masks if m.mean() > 0.004]
        pf = len(present) / max(1, len(masks))
        if len(present) < 4 or pf < 0.90:
            continue
        maj = np.stack(present).mean(axis=0) >= 0.5
        if maj.sum() < 40 or maj.mean() > 0.5:
            continue
        ys, xs = np.nonzero(maj)
        if (ys.max() - ys.min()) > 0.25 * H:
            continue                                 # full-height seam = pillarbox, not a badge
        iou = float(np.mean([float((m & maj).sum()) / max(1.0, float((m | maj).sum()))
                             for m in present]))
        score = pf * iou
        if iou >= 0.55 and score > best_score:
            best, best_score = side, score
    _EDGE_LOGO_CACHE[ck] = best
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
    # OPERATOR / AUTO source BAN-LIST: explicit source-ids kept out of the pool wholesale.
    # Used for fan-film / AI-recreation productions the per-frame verifier under-detects (a
    # high-production Tower-of-Joy fan film reads as HBO to a fast vision model, so it is banned
    # at the SOURCE — never entering match — and its beats fall to real footage / stills / holds).
    # Persisted on the project (proj.meta['banned_sources']) or via env (comma-separated ids), so
    # a re-render reproduces the ban deterministically.
    # include_auto=False: this pass re-derives its own source-level rejections below, so reading
    # back what a previous pass wrote would make them sticky and defeat every gate kill-switch.
    _banned = banned_source_ids(proj, include_auto=False)
    # SOURCE-LEVEL rejections recorded here are promoted to the shared ban-list (persisted as
    # proj.meta['auto_rejected_sources']) so they also hold in the still/image-fallback pool, the
    # breakout pool and build's shot-walk. Before this, they were bare `continue`s that only
    # emptied match's pool, and a rejected upload could still air as a still or a walked shot.
    _auto_rej: set[str] = set()
    # WHY each source was rejected. Two very different kinds hide in this set: a QUALITY reject
    # (subtitled re-upload / burned watermark / promo card / screener) means "right footage, unusable
    # copy" and is worth replacing, while a CONTENT reject (interview, reaction, wrong show, fan art)
    # is footage we never wanted. The backfill pass needs the distinction — without it, it spends
    # searches looking for a cleaner copy of a talking-head interview.
    _auto_why: dict[str, str] = {}

    def _reject(sid: str, code: str) -> None:
        _auto_rej.add(sid)
        _auto_why[sid] = code

    for src in proj.sources:
        if src.status != SOURCE_OK:
            continue
        if src.id in _banned:
            if progress:
                progress(f"match: dropping BANNED source {src.id} "
                         f"(fan-film / non-authentic, operator/auto ban: {(src.title or '')[:44]!r})")
            continue
        # belt-and-suspenders for sources downloaded BEFORE the discovery wrong-show rule existed:
        # a franchise sibling/prequel (House of the Dragon in a Game of Thrones video) is the wrong
        # production — right world, wrong cast/era — and must never enter the footage pool.
        if wrongshow_on and show_title and _wrong_installment(show_title, src.title or ""):
            if progress:
                progress(f"match: dropping wrong-show source {src.id} "
                         f"(franchise sibling/prequel: {(src.title or '')[:48]!r})")
            _reject(src.id, "wrong_show")
            continue
        if nonshow_on and _NONSHOW_TITLE.search(src.title or ""):
            if progress:
                progress(f"match: dropping non-show source {src.id} "
                         f"(game/AMV/animated: {(src.title or '')[:48]!r})")
            _reject(src.id, "non_show")
            continue
        # REACTION/facecam video that slipped in (e.g. dialogue-verified back during discovery):
        # its footage is people on a couch over a tiny show inset — never let it into the pool.
        if nonshow_on and _REACTION_TITLE.search(src.title or ""):
            if progress:
                progress(f"match: dropping reaction/facecam source {src.id} "
                         f"({(src.title or '')[:48]!r})")
            _reject(src.id, "reaction")
            continue
        # belt-and-suspenders: a talking-head / interview / featurette / promo source that was
        # downloaded BEFORE the discovery reject rule existed must still be kept OUT of the
        # footage pool (no scene footage — just a presenter / channel branding).
        if nonshow_on and _REJECT_TITLE.search(src.title or ""):
            if progress:
                progress(f"match: dropping talking-head/promo source {src.id} "
                         f"({(src.title or '')[:48]!r})")
            _reject(src.id, "talking_head_title")
            continue
        # NATIVE PUBLICATION FLOOR — unconditional and based on the downloaded BYTES, never the
        # discovery/download metadata.  The final build gate already refuses anything below a real
        # 1280x720 raster, but match used to admit 360p sources and verifier could install them after
        # clean-copy arbitration.  Reject before loading the index: an unpublishable source should
        # neither consume expensive visual gates nor enter match/still/breakout/shot-walk pools.
        # Quote classification intentionally scans ASR separately because SD audio can still prove
        # that an authored phrase is real while its pixels are never allowed to air.
        from .quality_contract import native_video_ok as _native_video_ok
        from .quality_contract import probe_native_video_info as _probe_native
        _native_path = str(getattr(src, "local_path", "") or "")
        _native_info = dict(_probe_native(_native_path) or {})
        if not _native_info.get("width") or not _native_info.get("height"):
            # Unknown is a PIPELINE/probe fault, not evidence that the footage itself is SD. Stop
            # before persisting a replaceable quality rejection; Resume must get a genuine retry.
            raise RuntimeError(
                f"native-resolution probe unavailable for source {src.id} ({_native_path or 'no path'})")
        if not _native_video_ok(_native_info):
            try:
                _nw = int(_native_info.get("width") or 0)
                _nh = int(_native_info.get("height") or 0)
            except (TypeError, ValueError):
                _nw = _nh = 0
            _why_native = f"{_nw}x{_nh}" if _nw and _nh else "unprobeable"
            if progress:
                progress(f"match: dropping sub-native-HD source {src.id} "
                         f"({_why_native} actual bytes; publication requires 1280x720)")
            _reject(src.id, "sub_native_hd")
            continue
        shots = _index.load_shots(proj, src.id)
        embeds = _index.load_embeds(proj, src.id)
        if face_gate_on and _source_is_modern_talkinghead(shots, embeds):
            if progress:
                progress(f"match: dropping modern talking-head source {src.id} "
                         f"(podcast/vlog/interview/makeup-BTS look, not a scene: {(src.title or '')[:48]!r})")
            _reject(src.id, "talking_head_visual")
            continue
        if gate_on and wm_mode == "drop" and \
                (_source_is_watermarked(shots)
                 or (os.environ.get("VIDLORE_CLIPSTUDIO_CORNER_LOGO_GATE", "1").strip()
                     not in ("0", "false", "no") and _source_corner_logo(shots))):
            if progress:
                progress(f"match: dropping watermarked source {src.id} (persistent channel logo)")
            _reject(src.id, "watermarked")
            continue
        # HEAVILY-SUBTITLED COPY: when ≥20% of a source's shots carry a burned-sub band, subs
        # appear on ANY dialogue moment — per-shot flags and window QC keep missing lines that
        # flash between samples (observed: three different Turkish lines aired across three
        # renders from such copies). These uploads are re-subtitled COPIES of scenes that clean
        # duplicates cover, so drop the whole source. VIDLORE_CLIPSTUDIO_SUBBED_SOURCE_MAX_FRAC
        # tunes the threshold (1 disables).
        try:
            _sub_max = float(os.environ.get("VIDLORE_CLIPSTUDIO_SUBBED_SOURCE_MAX_FRAC",
                                            "0.20") or 0.20)
        except (TypeError, ValueError):
            _sub_max = 0.20
        if _sub_max < 1.0:
            _sf = _source_subs_frac(shots)
            if _sf >= _sub_max:
                if progress:
                    progress(f"match: dropping subtitled-copy source {src.id} "
                             f"({_sf:.0%} of shots carry a burned-sub band)")
                _reject(src.id, "subtitled_copy")
                continue
        # CHANNEL-PROMO furniture burned into the frames = a re-packaged fan compilation, which
        # brings its listicle titles / end cards along with it. Kill switch: PROMO_OVERLAY_GATE=0.
        if os.environ.get("VIDLORE_CLIPSTUDIO_PROMO_OVERLAY_GATE", "1").strip() \
                not in ("0", "false", "no") and _source_has_promo_overlay(shots):
            if progress:
                progress(f"match: dropping promo-overlay source {src.id} "
                         f"(burned subscribe/bell card — repackaged compilation, not a clean rip)")
            _reject(src.id, "promo_overlay")
            continue
        if _source_is_static(shots):              # still-image / lyric card — not scene footage
            if progress:
                progress(f"match: dropping static-image source {src.id} (repeating still, not footage)")
            _reject(src.id, "static_image")
            continue
        if nonshow_on and _source_is_nonphotographic(proj, shots):
            if progress:
                progress(f"match: dropping non-live-action source {src.id} "
                         f"(toy/claymation/AI-render — not real footage)")
            _reject(src.id, "non_live_action")
            continue
        # NON-SHOW GRAPHICS — per-shot AND source-level. _source_is_nonphotographic above samples
        # only ~6 keyframes at a 55% bar, so a 40%-illustrated book-essay source passes it and,
        # with no per-shot gate at all, contributes every fan-art/CGI shot to the pool (observed:
        # a news-CGI intro aired on a connector beat; fan art aired via a secondary beat window).
        _gfx_on = nonshow_on and os.environ.get("VIDLORE_CLIPSTUDIO_GRAPHICS_GATE", "1").strip() \
            not in ("0", "false", "no")
        _gfx_idx: set = set()
        if _gfx_on:
            _tiers = {}
            for _pos, sh in enumerate(shots):
                _row = getattr(sh, "embed_row", -1)
                _v = (embeds[_row] if embeds is not None and 0 <= _row < len(embeds) else None)
                _tiers[getattr(sh, "index", _pos)] = _shot_graphics_tier(sh, _v)
                # WRITE-THROUGH backfill: stamp the computed tier on the in-memory shot so every
                # later consumer of THIS shots list (window-QC dirty reasons, breakout gating)
                # sees it persisted-first; _persist_graphics_flags below writes it to shots.json
                # so old projects converge to the new-index behaviour on their next load.
                if getattr(sh, "graphics_flag", -1) in (-1, None) and _tiers[
                        getattr(sh, "index", _pos)] >= 0:
                    try:
                        sh.graphics_flag = _tiers[getattr(sh, "index", _pos)]
                    except Exception:
                        pass
            _n_hard = sum(1 for t in _tiers.values() if t >= 2)
            # hard tier always gates; band tier gates only when the SAME source shows SOLID hard
            # evidence (>=3 hard shots — a lone intro/outro title card must not arm the band and
            # cost real dark/stylized footage; the illustrated/parody sources all carry 4+)
            _band_armed = _n_hard >= 3
            _gfx_idx = {i for i, t in _tiers.items() if t >= 2 or (t == 1 and _band_armed)}
            if _graphics_source_verdict(_n_hard, len(shots)):
                if progress:
                    progress(f"match: dropping illustrated/graphics source {src.id} "
                             f"({_n_hard}/{len(shots)} shots are designed graphics — "
                             f"parody/fan-art/news content, not footage)")
                _persist_graphics_flags(proj, src.id, shots)
                _reject(src.id, "graphics")
                continue
            if _gfx_idx and progress:
                progress(f"match: {src.id} — {len(_gfx_idx)} designed-graphics shot(s) gated "
                         f"(news CGI / game UI / illustration never airs)")
            _persist_graphics_flags(proj, src.id, shots)
        # LISTICLE COUNTDOWN NUMERALS — a 'Top 5'-style essay parks a giant burned '1'/'3' on
        # otherwise-real footage; digits defeat every text rule (no junk keyword, one token,
        # no letters for the badge rule). Measured on job 5462677f95: the listicle source has
        # numeral OCR on 75% of its shots; every legit source <=2.7% (OCR noise) — so the gate
        # is SOURCE-level: >=3 numeral shots AND >=10% of the source. Sparse hits never gate.
        if nonshow_on and os.environ.get("VIDLORE_CLIPSTUDIO_NUMERAL_GATE", "1").strip() \
                not in ("0", "false", "no"):
            _n_num = sum(1 for sh in shots if _shot_numeral_overlay(sh))
            if _n_num >= 3 and _n_num / max(1, len(shots)) >= float(
                    os.environ.get("VIDLORE_CLIPSTUDIO_NUMERAL_SRC_FRAC", "0.10") or 0.10):
                if progress:
                    progress(f"match: dropping listicle/countdown source {src.id} "
                             f"({_n_num}/{len(shots)} shots carry a burned countdown numeral)")
                _reject(src.id, "numeral_overlay")
                continue
        # SLIDESHOW / AI-ART ESSAY source — most shots barely move (art cards under narration).
        # Its occasional real-footage shots can't be trusted between keyframes either, same
        # doctrine as the graphics-source drop. Content reject: backfill must not chase a
        # "cleaner copy" of an essay.
        if nonshow_on and os.environ.get("VIDLORE_CLIPSTUDIO_STATIC_GATE", "1").strip() \
                not in ("0", "false", "no") and _slideshow_source_verdict(shots):
            if progress:
                progress(f"match: dropping slideshow/AI-art essay source {src.id} "
                         f"(most shots are near-static art cards, not scene footage)")
            _reject(src.id, "slideshow_source")
            continue
        # BONUS-TAIL guard — '... + BONUS Scene' uploads append interviews/featurettes after the
        # real footage; a press-junket frame aired mid-beat from such a tail. Stamp trailing-span
        # shots (last 25% of the file) so BOTH candidate scoring and window-QC treat them dirty
        # (the stamp rides on shot.scores, so _shot_dirty_reason needs no source context).
        try:
            from .discover import source_has_bonus_tail as _has_bonus
            _dur_bt = float(getattr(src, "duration", 0) or 0)
            if _has_bonus(getattr(src, "title", "") or "") and _dur_bt > 60:
                _cut_bt = 0.75 * _dur_bt
                _n_bt = 0
                for sh in shots:
                    if float(getattr(sh, "start", 0) or 0) >= _cut_bt:
                        try:
                            sh.scores = dict(getattr(sh, "scores", None) or {})
                            sh.scores["bonus_tail"] = 1
                            _n_bt += 1
                        except Exception:                # noqa: BLE001
                            pass
                if _n_bt and progress:
                    progress(f"match: {src.id} — {_n_bt} trailing shot(s) gated as bonus-tail "
                             f"(post-scene interview/featurette risk)")
        except Exception:                                # noqa: BLE001
            pass
        # SCREEN-RECORDING source — a burned mouse cursor sat mid-frame for a whole 12.6s
        # breakout AND aired again on a regular beat (job 5462677f95 / the shorttest). One
        # whole-source probe: 4 frames at 30/50/70/90% of the file — a solid-white frozen
        # blob at the SAME pixels across four DIFFERENT scenes is necessarily an overlay
        # (the within-clip probe's calibrated core/ring/edge rules apply unchanged). Verdict
        # persisted in proj.meta so each source is decoded once. Quality reject: the same
        # scenes exist in clean uploads, so backfill may chase a replacement.
        if nonshow_on and os.environ.get("VIDLORE_CLIPSTUDIO_CURSOR_SRC_GATE", "1").strip() \
                not in ("0", "false", "no") and getattr(src, "local_path", None):
            _cur_cache = proj.meta.setdefault("cursor_scan", {}) if hasattr(proj, "meta") \
                and isinstance(getattr(proj, "meta", None), dict) else {}
            _cv = _cur_cache.get(src.id)
            if _cv is None:
                try:
                    from .build import _breakout_cursor_probe as _cprobe
                    _cv = bool(_cprobe(Path(src.local_path),
                                       float(getattr(src, "duration", 0.0) or 0.0)))
                except Exception:                        # noqa: BLE001
                    _cv = False
                _cur_cache[src.id] = _cv
            if _cv:
                if progress:
                    progress(f"match: dropping screen-recording source {src.id} "
                             f"(burned mouse cursor detected across the file)")
                _reject(src.id, "screen_recording")
                continue
        for _pos, sh in enumerate(shots):             # `embeds` loaded once above (reused here)
            if getattr(sh, "index", _pos) in _gfx_idx:
                continue
            vec = None
            _row = getattr(sh, "embed_row", -1)
            if embeds is not None and 0 <= _row < len(embeds):
                vec = embeds[_row]
            pool.append(_PoolShot(src.id, sh, vec))
    # PROMOTE this pass's source-level rejections to the shared ban-list so they hold in the
    # still/image-fallback pool, the breakout pool and build's shot-walk too (see
    # banned_source_ids). Persisted on the project, so a resume/re-render reproduces them.
    try:
        _meta = getattr(proj, "meta", None)
        if _meta is None:
            _meta = {}
            proj.meta = _meta
        _prev = {str(x) for x in (_meta.get("auto_rejected_sources") or [])}
        # REPLACE, never union: the list must always describe the CURRENT gate configuration, so
        # flipping a gate off (or a re-index changing a verdict) actually re-admits the source
        # everywhere instead of leaving a stale ban behind.
        if _auto_rej != _prev:
            if _auto_rej:
                _meta["auto_rejected_sources"] = sorted(_auto_rej)
            else:
                _meta.pop("auto_rejected_sources", None)
            if progress and _auto_rej:
                progress(f"match: {len(_auto_rej)} source-level rejection(s) promoted to the "
                         f"shared ban-list (also excluded from stills / breakouts / shot-walk)")
        # Reasons travel with the list, same REPLACE semantics — written every pass, not only when
        # the id set changes, so a re-gated source can't keep a stale reason. The backfill pass reads
        # this to tell a QUALITY reject (right footage, unusable copy — worth replacing) from a
        # CONTENT reject (interview / reaction / wrong show — footage we never wanted).
        if _auto_why:
            _meta["auto_rejected_reasons"] = dict(sorted(_auto_why.items()))
        else:
            _meta.pop("auto_rejected_reasons", None)
    except Exception:
        pass                                       # never fail a render over bookkeeping
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


# ---------------------------------------------------------------------------
# DIALOGUE-BASED MOMENT MATCHING
#
# `_dialogue_match` above asks "is the beat's quote inside THIS SHOT's transcript?" — and
# index._assign_transcript bins words into shots by MIDPOINT, so a line spoken across a cut belongs
# to no single shot. index.py documents this ("per-shot transcripts useless for locating a QUOTE").
# Measured on the v2 render: 85 of 268 beats carry the exact line they are about, yet the dialogue
# signal — the heaviest weight in the scorer at w_dialogue=0.55 — fired on only 20 of them (23.5%,
# mean 0.105). Sixty-five beats named their moment and got nothing for it.
#
# The fix is to locate the line in the source's CONTINUOUS word stream (index.find_quote_span, the
# same primitive breakouts already use) and then score each shot by how close it sits to that span.
# That is moment-level, not source-level: it tells us WHERE INSIDE a 6-minute upload the line is
# spoken, so the cut lands on that frame instead of anywhere in the right source. It is also what an
# editor actually does — find the line, cut to it.
# ---------------------------------------------------------------------------

_QSPAN_CACHE: dict = {}          # (sid, normalized quote) -> (t0, t1, ratio) | None
# find_quote_span's own floor is 0.72; require a little more before a located span is allowed to
# DECIDE a pick, so a loose phrase match never drags a beat onto the wrong second.
_MOMENT_MIN_RATIO = 0.78
# mean luma at/above which a copy is considered comfortably watchable; below it the moment bonus
# is scaled down so a brighter copy of the SAME moment can win (see the damper in _score_pool).
_MOMENT_LUMA_OK = 34.0


def quote_span_in_source(proj, sid: str, quote: str):
    """Locate `quote` in source `sid`'s word stream. Memoized — the sliding-window align is not
    free and the same (beat quote x source) pair is asked for on every candidate shot."""
    q = " ".join(_norm_words(quote or ""))
    if not q or not sid:
        return None
    key = (sid, q)
    if key in _QSPAN_CACHE:
        return _QSPAN_CACHE[key]
    span = None
    try:
        words = _index.load_words(proj, sid)
        if words:
            span = _index.find_quote_span(words, quote)
    except Exception:
        span = None
    _QSPAN_CACHE[key] = span
    return span


def _moment_proximity(shot, span, *, pre_roll: float = 1.5, decay: float = None) -> float:
    """How well does this shot sit ON the located line? 1.0 when the shot overlaps the spoken span,
    decaying to 0 across `decay` seconds either side.

    `pre_roll` lets a shot that starts just BEFORE the line still score 1.0 — an editor cuts to the
    speaker a beat before they talk, and the reaction shot that precedes a line is part of the
    moment. Nothing here depends on shot transcripts, so a line straddling a cut is fine."""
    if not span:
        return 0.0
    if decay is None:
        # 12s, not a couple of seconds, and the reason is editorial. Measured: 40 of the 85 quoted
        # beats in one essay share their line with another beat ("chaos is a ladder" is referenced
        # by 5 separate beats), so anti-reuse necessarily denies most of them the exact same two
        # seconds. A wide neighbourhood means a returning beat lands on ANOTHER SHOT OF THE SAME
        # SCENE — the reaction, the other angle — which is what an editor cuts, instead of falling
        # through to unrelated footage. Swept on the real pool: decay 2.5 -> 12 traded 2 exact hits
        # for 5 same-scene hits and cut unrelated picks 17 -> 14 (66% -> 72% on-or-in-scene).
        decay = _f_env("VIDLORE_CLIPSTUDIO_MOMENT_DECAY", 12.0)
    t0, t1 = float(span[0]), float(span[1])
    s0, s1 = float(getattr(shot, "start", 0.0)), float(getattr(shot, "end", 0.0))
    if s1 >= (t0 - pre_roll) and s0 <= t1:
        return 1.0                                     # the shot is ON the line
    gap = (t0 - s1) if s1 < t0 else (s0 - t1)
    if gap <= 0:
        return 1.0
    return max(0.0, 1.0 - gap / decay)


def _anchor_echo(seg, anchor_lines) -> str:
    """The anchor-scene line this beat most clearly echoes, or "".

    Scored by COVERAGE of the anchor line's content words, not a raw hit count. `_norm_words` strips
    stopwords, so a perfectly good line — "I did it to protect the woman I love." — reduces to three
    content words (protect / woman / love); any fixed ">= N shared words" rule silently ignores every
    short line, which is most quoted dialogue."""
    if not anchor_lines:
        return ""
    txt = set(_norm_words(getattr(seg, "text", "") or "")) | \
        set(_norm_words(getattr(seg, "expected_visual", "") or "")) | \
        set(_norm_words(getattr(seg, "quote", "") or ""))
    if not txt:
        return ""
    best, best_score = "", 0.0
    for line in anchor_lines:
        lw = set(_norm_words(line))
        if len(lw) < 2:
            continue
        hits = len(txt & lw)
        cov = hits / float(len(lw))
        # >=2 shared content words AND most of the line accounted for — enough to say the beat is
        # talking about THIS line rather than merely sharing a word with it.
        if hits >= 2 and cov >= 0.6 and cov > best_score:
            best, best_score = line, cov
    return best


def beat_quote_candidates(seg, anchor_lines=None) -> list:
    """Every phrasing worth looking for, best first.

    The analyzer's quote is a PARAPHRASE as often as a transcription, and find_quote_span scores the
    whole phrase, so extra words sink it below the 0.72 floor. Measured: the aired line is "I did it
    to protect you"; the analyzer wrote it two ways across neighbouring beats —
      "I did it to protect Sansa."        -> located, ratio 0.909
      "I did what I did to protect Sansa." -> not located at all
    Same moment, same source, one phrasing finds it and one does not. So try the beat's own quote
    first, then the anchor scene's VERBATIM line that the beat echoes — the analyzer records those
    separately and they are transcriptions, not paraphrases."""
    out, seen = [], set()
    for cand in ((getattr(seg, "quote", "") or "").strip(),
                 _anchor_echo(seg, anchor_lines)):
        if not cand:
            continue
        k = " ".join(_norm_words(cand))
        if k and k not in seen:
            seen.add(k)
            out.append(cand)
    return out


def _beat_quote(seg, anchor_lines=None) -> str:
    """Back-compat single-value view of `beat_quote_candidates` (tests call this)."""
    c = beat_quote_candidates(seg, anchor_lines)
    return c[0] if c else ""


def locate_beat_moment(proj, sid: str, seg, anchor_lines=None):
    """First phrasing of this beat that can be found in source `sid`. -> (t0, t1, ratio) | None."""
    for q in beat_quote_candidates(seg, anchor_lines):
        sp = quote_span_in_source(proj, sid, q)
        if sp:
            return sp
    return None


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


def _trim_window(shot: Shot, seg: ScriptSegment, cfg: ClipConfig,
                 moment: tuple | None = None) -> tuple[float, float]:
    """Pick an [in,out] inside the shot ~matching the segment's screen time.

    Centred on the shot midpoint normally — but on the MOMENT when we know one. Picking the right
    shot is only half of "hold on that scene": a 12-second shot trimmed to 2.5s around its midpoint
    can easily miss the second the line is actually spoken, which is how a correctly-chosen shot
    still aired the wrong instant. When find_quote_span has located the line, centre the window on
    it so the cut CONTAINS the words the narration is talking about."""
    L = max(cfg.min_clip_sec, min(cfg.max_clip_sec, shot.duration, seg.est_duration + 0.6))
    center = (shot.start + shot.end) / 2.0
    if moment:
        m0, m1 = float(moment[0]), float(moment[1])
        if m1 >= shot.start and m0 <= shot.end:      # the line lies in (or overlaps) this shot
            mc = (max(m0, shot.start) + min(m1, shot.end)) / 2.0
            # a line longer than the beat's screen time: start ON the line rather than centring,
            # so the cut opens on the words instead of arriving halfway through them.
            center = mc if (m1 - m0) <= L else (max(m0, shot.start) + L / 2.0)
    a = max(shot.start, center - L / 2.0)
    b = min(shot.end, a + L)
    a = max(shot.start, b - L)                      # re-pin if we hit the tail
    return round(a, 3), round(b, 3)


# NB: 'high' and 'septon' are NOT here. 'High Sparrow' IS a roster name, and stripping 'high'
# turned it into 'sparrow', which resolves to nothing. 'the High Septon' is kept out by the
# contiguous-run rule instead ('high sparrow' is not a run inside 'high septon').
_HONORIFICS = ("ser", "king", "queen", "lord", "lady", "prince", "princess",
               "maester", "grand", "the")
_ENT_SPLIT = re.compile(r"\s*(?:,|;|/|&|\band\b|\bwith\b|\bvs\.?\b|\bversus\b)\s*", re.I)


def _strip_honorifics(part: str) -> str:
    """Drop leading titles from BOTH the query and the roster alias, so 'Ser Gregor Clegane' can
    meet a roster entry stored as 'Ser Gregor Clegane / The Mountain'."""
    toks = [t for t in (part or "").split() if t]
    while toks and toks[0] in _HONORIFICS:
        toks = toks[1:]
    return " ".join(toks)


def resolve_face_targets(required_entity: str, char2actor: dict) -> tuple:
    """(actor/character names to look for, did EVERY named part resolve?).

    The old version was an exact dict lookup, so a beat saying "Cersei" or "Ser Gregor Clegane" or
    "Tommen" matched nothing — `char2actor` is keyed on the full canonical name. Measured on a real
    render: 27 of 107 character-beats (25%) could never match a Face-ID result, which contains ACTOR
    names only. On those beats the +0.30 face bonus never fired AND the -0.50 wrongface penalty fired
    on every named shot INCLUDING shots of the correct person — 1,293 (beat, shot) pairs took a 0.80
    swing in the wrong direction, on the protagonist.

    Resolution is deliberately CONSERVATIVE, because a false target is what lets a wrong character
    through. A part matches a roster name when it equals an alias, contains the full alias as a
    contiguous run of tokens, or is a single token equal to a roster GIVEN name that is unique across
    the roster. That last rule is why 'Cersei' resolves and 'Tyrell' (a surname shared by three
    characters) and 'the High Septon' (a prefix of no alias) both correctly resolve to nothing.

    The second return value matters for the wrong-character decision: when a beat names someone the
    roster does not know ("Missandei, Jon, Olenna"), we cannot conclude a shot shows the WRONG
    person, only that we do not know. Callers must not claim 'wrong' on a partial resolution."""
    ent = (required_entity or "").strip().lower()
    if not ent:
        return set(), False
    # roster alias -> actor, plus the unique-given-name index
    alias2actor: dict = {}
    for ch, ac in (char2actor or {}).items():
        for alias in str(ch or "").lower().split("/"):
            alias = _strip_honorifics(alias.strip())
            if alias:
                alias2actor[alias] = str(ac or "").lower()
    given: dict = {}
    for alias in alias2actor:
        toks = alias.split()
        if toks:
            given.setdefault(toks[0], set()).add(alias)

    out: set = set()
    parts = [p for p in (_strip_honorifics(x.strip()) for x in _ENT_SPLIT.split(ent)) if p]
    if not parts:
        return set(), False
    full = True
    for p in parts:
        hit = None
        if p in alias2actor:
            hit = p
        else:
            ptoks = p.split()
            for alias in alias2actor:                    # contiguous-run containment
                a = alias.split()
                if len(a) <= len(ptoks) and any(
                        ptoks[i:i + len(a)] == a for i in range(len(ptoks) - len(a) + 1)):
                    hit = alias
                    break
            if hit is None and len(ptoks) == 1 and len(given.get(p, ())) == 1:
                hit = next(iter(given[p]))               # unique given name, e.g. 'cersei'
        if hit is None:
            full = False
            continue
        out.add(hit)
        if alias2actor.get(hit):
            out.add(alias2actor[hit])
    return out, (full and bool(out))


def _face_targets(seg: ScriptSegment, char2actor: dict) -> set:
    """Lowercased names to look for in a shot's Face-ID / OCR when this beat needs a person."""
    if seg.required_kind not in ("actor", "character") or not seg.required_entity:
        return set()
    return resolve_face_targets(seg.required_entity, char2actor)[0]


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


# A commenter/channel AVATAR BADGE — "reads a comment" overlay: profile pic + a personal name
# in a bordered box, parked in one corner for a stretch of the video. It defeats every other
# text rule at once: the name matches no _OCR_JUNK keyword, two mixed-case words stay under
# _ocr_text_heavy's 3-word/ALL-CAPS floors, and the source-level pixel corner detector needs the
# bug on ≥25% of shots (an intermittent overlay never qualifies — and cropping the WHOLE source
# for a part-time overlay would be wrong anyway). Observed: a 'Jacquelyn Sutphen' fox-avatar
# badge aired on 2 beats and forced a caption-dodge dropout.
_NAME_BADGE_RX = re.compile(r"\b[A-Z][a-z]{2,}\s+[A-Z][a-z]{2,}\b")


def _shot_overlay_badge(sh) -> bool:
    """Persisted-first per-SHOT detector for the avatar-badge class: 1-2-word NAME-LIKE OCR text
    (3+ words is _ocr_text_heavy's call) coinciding with a DENSE edge mask in some corner (the
    badge's border/avatar/text edges — mean ≥0.25 of the corner grid). Calibrated on the 50-source
    Qyburn project index: flags exactly the 4 real badge shots of 1957, zero false positives.
    Old indexes without corner_masks return False (fail-open, same doctrine as the other flags)."""
    txt = (getattr(sh, "ocr_text", "") or "").strip()
    if not txt or not _NAME_BADGE_RX.search(txt):
        return False
    if len(re.findall(r"[A-Za-z']{3,}", txt)) >= 3:
        return False                                   # text-heavy overlay — already gated
    masks = getattr(sh, "corner_masks", None) or {}
    if not masks:
        return False
    from .index import _mask_from_hex
    for h in masks.values():
        m = _mask_from_hex(h)
        if m is not None and float(m.mean()) >= 0.25:
            return True
    return False


def _shot_static_collage(sh) -> bool:
    """A FROZEN IMAGE airing as footage — thumbnail collage, AI-art card, promo composite.
    Two calibrated tiers (job 5462677f95: 8 true statics, 220 control shots, 0 confirmed
    live-action FPs):
      FREEZE (standalone): >=75% of sample pairs diff < 0.9 (the persisted static_frac), with
        a luma guard — near-black live footage can sit that low (control floor: dmax 0.97 at
        luma 3.2). NOT corroboration-gated: half the true statics are single-face freezes with
        graphics_flag 0 and no OCR.
      STILL (corroborated): every pair diff < 2.3 AND (graphics band+ OR OCR text OR >=2
        faces) — slower slideshow pans over designed art.
    Old indexes (-1 sentinels) fail open. Env: VIDLORE_CLIPSTUDIO_STATIC_GATE=0 disables."""
    import os as _os_st
    if _os_st.environ.get("VIDLORE_CLIPSTUDIO_STATIC_GATE", "1").strip() in ("0", "false", "no"):
        return False
    # FREEZE tier keys on pair_diff_max, NOT static_frac: HD-clean slow dark scenes sit under
    # the 0.9 per-pair threshold (measured on a Ramsay bedroom shot: pmax 0.61, real footage),
    # while a genuinely frozen digital card is EXACTLY zero-diff at any resolution (outro card
    # pmax 0.00). 0.20 splits them with wide margins on both sides.
    pmx = getattr(sh, "pair_diff_max", -1.0)
    pmx = -1.0 if pmx is None else float(pmx)
    la = float(getattr(sh, "luma_avg", -1.0) or -1.0)
    if 0.0 <= pmx < 0.20 and la >= 5.0:
        return True
    # STILL tier: near-frozen (digital stills read ~0-0.3 at any resolution; HD real footage
    # floors ~0.4) AND corroborated by designed-graphics/text evidence. faces>=2 was measured
    # OUT as a corroborator: at HD, slow two-person dialogue sits under the old 2.3 bound and
    # every real conversation has two faces — it flagged 39 real shots on one 6-source test.
    if 0.0 <= pmx < 0.60:
        gfx = int(getattr(sh, "graphics_flag", -1) or -1)
        has_ocr = bool((getattr(sh, "ocr_text", "") or "").strip())
        if gfx >= 1 or has_ocr:
            return True
    return False


def _slideshow_source_verdict(shots) -> bool:
    """True when a WHOLE source is a slideshow/AI-art essay (drop it): >=65% of its shots have
    a mean pair diff < 6 (>=12 measured) AND the slowness is CORROBORATED by designed-graphics
    evidence — >=3 HARD graphics shots or >=3 frozen digital cards. Slowness alone does NOT
    transfer across source quality: the 0.65 floor was calibrated on 360p uploads, and clean HD
    downloads put real slow candlelit dialogue (Cersei dinner 0.72, Ramsay bedroom 0.72) right
    where the 360p AI-art essays sat — while those essays carry hard-graphics shots real scenes
    never do ('When Roses' 8 hard, 'The Strangler' 3 hard vs 0 across every real HD source)."""
    vals = []
    hard = 0
    frozen = 0
    for sh in shots:
        pm = getattr(sh, "pair_diff_mean", -1.0)
        pm = -1.0 if pm is None else float(pm)
        if pm >= 0.0:
            vals.append(pm)
        try:
            if int(getattr(sh, "graphics_flag", -1) or -1) >= 2:
                hard += 1
        except (TypeError, ValueError):
            pass
        if _shot_static_collage(sh):
            frozen += 1
    if len(vals) < 12:
        return False
    slow = sum(1 for v in vals if v < 6.0) / len(vals) >= 0.65
    return slow and (hard >= 3 or frozen >= 3)


_NUMERAL_RX = re.compile(r"^\s*#?\d{1,2}\s*[.):]?\s*$")
_EPCODE_CACHE: dict = {}                # sid -> parsed title episode code (per-process memo)

# EDITORIAL-ESSAY / COMPILATION TITLES. Hoisted verbatim from the single-scene purity filter so the
# scene-title affinity can discount them too: an essay ABOUT a scene matches the beat's query as
# well as the scene upload does, but its footage is interleaved with talking heads, text cards and
# cross-era B-roll. Purity (which excludes) and affinity (which merely halves a bonus) read the
# same definition — one pattern, two strictnesses.
_ESSAY_TITLE_RX = re.compile(
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
_SENTINEL = object()


def _shot_numeral_overlay(sh) -> bool:
    """Listicle COUNTDOWN NUMERAL burned into the frame — 'Top 5'-style essays park a giant
    '1'/'3' on the edge of otherwise-real footage. The numeral defeats every text rule: digits
    match no junk keyword, one token is under _ocr_text_heavy's floors, and _NAME_BADGE_RX wants
    letters. Rule: the shot's OCR is a LONE 1-2-digit token (optionally '#'/'.'/')'/':') and
    nothing else. NOT a safe per-shot gate on its own — measured on job 5462677f95 (4267 shots),
    OCR noise puts 1-2 stray digit reads in legit scene packs too (63 raw hits total). The
    listicle source carried them on 75% of its shots vs <=2.7% everywhere else, so the CALLER
    gates at source level (>=3 hits and >=10% of shots; see _load_pool)."""
    txt = (getattr(sh, "ocr_text", "") or "").strip()
    return bool(txt) and bool(_NUMERAL_RX.match(txt))


def _shot_graphics_tier(sh, vec=None) -> int:
    """Tiered designed-graphics verdict for a pool shot — persisted-first (graphics_flag from a
    new index), else computed from the shot's CLIP embedding (`vec`) via index.graphics_flag_of.
    -1 when neither is available (old index + no embed): fail-open, same doctrine as the other
    persisted flags. Tiers: 2 hard (always excluded) · 1 band-art (excluded only alongside hard
    evidence in the same source) · 0 photographic. Calibrated on 1957 real shots — the hard tier
    has 0 live-action FPs; the band tier's one observed live FP (a stylized drawbridge aerial)
    is why band alone never gates."""
    pf = getattr(sh, "graphics_flag", -1)
    try:
        pf = -1 if pf is None else int(pf)
    except (TypeError, ValueError):
        pf = -1
    if pf >= 0:
        return pf
    if vec is None:
        return -1
    from .index import graphics_flag_of
    # multi-frame luma_avg is only a PROXY for the keyframe's own luma (which the embed was
    # computed from and which old indexes never stored) — good enough for the near-black guard.
    # NOTE: no `or -1.0` — that would collapse a legitimate 0.0 (pure black) to "unknown" and
    # judge a degenerate embed instead of guarding it.
    _la = getattr(sh, "luma_avg", -1.0)
    _la = -1.0 if _la is None else float(_la)
    return graphics_flag_of(vec, _la)


def _shot_is_graphics(sh, vec=None) -> bool:
    """HARD-tier check only — safe without source context (see _shot_graphics_tier)."""
    return _shot_graphics_tier(sh, vec) >= 2


def _persist_graphics_flags(proj, source_id: str, shots) -> None:
    """Write freshly-computed graphics tiers back to shots.json (old-index BACKFILL) so the
    persisted-flag consumers — window-QC dirty reasons, breakout gating, build mirrors — see
    them on every later load, not just in this process. Atomic tmp+replace; best-effort (a
    failure leaves the old file intact and the live-compute path still governs)."""
    try:
        import json
        f = proj.shots_path(source_id)
        if not f.exists():
            return
        recs = json.loads(f.read_text(encoding="utf-8"))
        if not isinstance(recs, list) or len(recs) != len(shots):
            return
        changed = False
        for r, sh in zip(recs, shots):
            _t = getattr(sh, "graphics_flag", -1)
            _t = -1 if _t is None else int(_t)
            if _t >= 0 and r.get("graphics_flag", -1) != _t:
                r["graphics_flag"] = _t
                changed = True
        if not changed:
            return
        tmp = f.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(recs), encoding="utf-8")
        tmp.replace(f)
    except Exception:
        pass


def _graphics_source_verdict(n_gfx: int, n_shots: int) -> bool:
    """True when a WHOLE source is an illustrated/parody upload (drop it entirely): >=20% of its
    shots are HARD-tier designed graphics with at least 3 such shots. Calibrated on a real
    50-source render: the four parody/illustrated sources score 28.6-94.6% hard; every
    real-footage source <=6.1%. A source above the bar can't be trusted between keyframes either
    (a 15s shot of an illustrated essay aired fan art that its single keyframe never showed).
    Counts HARD shots only — band-tier shots may include stylized live-action, and letting them
    tip the fraction could kill a short real source over one title card + two moody frames."""
    return n_gfx >= 3 and n_gfx / max(1, n_shots) >= 0.20


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
        if not ok and len(ys) >= 80 and mf <= 0.004 and rowspread < 8.0:
            # SHORT-LINE branch — see index._flags_from_frames (kept in sync)
            _h = ys.max() - ys.min() + 1
            _w = xs.max() - xs.min() + 1
            ok = _w >= 150 and _h <= 34
        _SUBBAND_CACHE[kf] = bool(ok)
        return bool(ok)
    except Exception:
        return False


def _luma(shot, name: str) -> float:
    """A shot's luma field, with a genuine 0.0 kept DISTINCT from "not computed".

    `float(getattr(shot, name, -1.0) or -1.0)` reads a real 0.0 as the -1 sentinel, so a literally
    black shot took the fail-open path out of every luma gate. Measured across 10,013 indexed
    shots: 50 (0.50%) escape `_shot_unreadable` for this reason alone, and 25 of those are black
    outright (luma_hi <= 2) — including one whose whole frame is black (0.0 / 1.0 / 0.0, black
    fraction 1.0). This is the one change here that RAISES rejection; all of it is pure black."""
    v = getattr(shot, name, None)
    if v is None:
        return -1.0
    try:
        return float(v)
    except (TypeError, ValueError):
        return -1.0


def _shot_featureless(shot) -> bool:
    """FLAT shot — bright enough that detail SHOULD be visible, yet carrying almost none.

    The dark gates below cannot see this class, and it aired: the intro A/B judged one beat 0/10
    on a mid-grey wall (luma_avg 61.4, quality 0.32 — squarely mid-pool, so the quality floor had
    no opinion either). The same rule also catches the white "SUBSCRIBED" end-cards, a channel
    banner card, and washed-out blizzard frames.

    The discriminator is LUMA RANGE, not brightness: a real frame has highlights somewhere, so
    `luma_hi` sits well above `luma_avg`; a flat card's histogram is a spike. It is deliberately
    conjoined with a BRIGHTNESS floor, because a dark frame legitimately has a narrow range — the
    Long Night measures avg ~10-20 / hi 36-53, a spread as narrow as the grey card's. Requiring
    luma_avg >= 45 exempts every night scene, which matters more here than anywhere: the video
    this was found on is about the Long Night.

    The third arm asks the same question across TIME (`luma_min`, the darkest sample's mean): a
    card does not change over its span. MEASURED on this job's 4840 indexed shots: 17 flagged
    (0.35%), 16 of them plainly unusable on inspection and the 17th a near-white blizzard wide.
    Env: VIDLORE_CLIPSTUDIO_FLAT_GATE=0 disables."""
    import os as _os_f
    if _os_f.environ.get("VIDLORE_CLIPSTUDIO_FLAT_GATE", "1").strip() in ("0", "false", "no"):
        return False
    la = _luma(shot, "luma_avg")
    lh = _luma(shot, "luma_hi")
    lm = _luma(shot, "luma_min")
    if la < 0 or lh < 0 or lm < 0:
        return False                                   # not computed (old index) → fail open
    try:
        t_lit = float(_os_f.environ.get("VIDLORE_CLIPSTUDIO_FLAT_LIT", "45") or 45)
        t_span = float(_os_f.environ.get("VIDLORE_CLIPSTUDIO_FLAT_SPAN", "35") or 35)
    except (TypeError, ValueError):
        t_lit, t_span = 45.0, 35.0
    return la >= t_lit and (lh - la) < t_span and abs(lm - la) <= 0.10 * la


def _shot_unreadable(shot) -> bool:
    """UNREADABLE-DARK shot — near-black across its WHOLE span, not just at one instant.
    Uses the persisted multi-frame luma (start/mid/end samples): `luma_avg` low AND even the
    brightest sample's ~99.8th-percentile pixel (`luma_hi`) dim. A valid dark cinematic scene
    (candlelit privy, torch in the catacombs) keeps bright highlights → high luma_hi → passes.
    Old indexes (sentinel -1) are never gated here — the quality floor still applies.
    Env: VIDLORE_CLIPSTUDIO_UNREADABLE_GATE=0 disables; _AVG/_HI tune thresholds."""
    import os as _os_u
    if _os_u.environ.get("VIDLORE_CLIPSTUDIO_UNREADABLE_GATE", "1").strip() in ("0", "false", "no"):
        return False
    la = _luma(shot, "luma_avg")
    lh = _luma(shot, "luma_hi")
    if la < 0 or lh < 0:
        return False                                   # not computed (old index) → fail open
    # CONSERVATIVE by design (measured on real GoT footage): the readable dark privy/crossbow
    # shots sit at avg 12-14 / hi 127-147, true murk at avg 6-10 / hi 46-98. Overlap exists, so
    # the gate only fires on clear murk — a borderline dark scene stays (relevance-first).
    try:
        t_avg = float(_os_u.environ.get("VIDLORE_CLIPSTUDIO_UNREADABLE_AVG", "11") or 11)
        t_hi = float(_os_u.environ.get("VIDLORE_CLIPSTUDIO_UNREADABLE_HI", "90") or 90)
        t_hi_hard = float(_os_u.environ.get("VIDLORE_CLIPSTUDIO_UNREADABLE_HI_HARD", "60") or 60)
    except (TypeError, ValueError):
        t_avg, t_hi, t_hi_hard = 11.0, 90.0, 60.0
    # Two arms: (1) the original avg-AND-hi murk gate; (2) NO HIGHLIGHT AT ALL — even the brightest
    # ~99.8th-pct pixel is below t_hi_hard (60), so nothing in the frame is legible regardless of the
    # average. Arm 2 catches the near-black Hall-of-Faces shot that aired (avg 19 cleared t_avg=11 but
    # its luma_hi was 49 — no pixel brighter than 49). Readable dark scenes (torch/candle privy) keep
    # bright highlights (hi 127-147) and pass both.
    if (la < t_avg and lh < t_hi) or (lh < t_hi_hard):
        return True
    # THIRD ARM — intra-shot dark SPAN. Both arms above are shot-wide, so a shot that is black for
    # part of its run but carries a bright highlight somewhere passes: the v2 render aired a beat
    # whose shot measured avg 39.4 / hi 255 while its delivered frames sat at mean luma 2.5 with
    # 100% of pixels under 16. luma_min is that shot's DARKEST sample, and black_frac says how much
    # of it is genuinely black — together they separate a low-key lit frame from an unusable one.
    # Sentinel -1 (old index, not recomputed) fails open, exactly like the arms above.
    lmin = _luma(shot, "luma_min")
    lbf = float(getattr(shot, "luma_min_black_frac", -1.0) or -1.0)
    if lmin >= 0.0 and lbf >= 0.0:
        try:
            t_min = float(_os_u.environ.get("VIDLORE_CLIPSTUDIO_UNREADABLE_MIN", "9") or 9)
            t_bf = float(_os_u.environ.get("VIDLORE_CLIPSTUDIO_UNREADABLE_BLACKFRAC", "0.90") or 0.90)
        except (TypeError, ValueError):
            t_min, t_bf = 9.0, 0.90
        if lmin < t_min and lbf >= t_bf:
            return True
    return False


# ---------------------------------------------------------------------------
# CUT-WINDOW FLAG VALIDATION — the rendered cut can extend past the selected
# shot's boundaries (cut_selection pads short shots to min_clip_sec; build's
# beat-window walk advances a playhead), so a shot that is clean at its own
# sampled frames can still AIR an adjacent shot's burned subs / logo / murk
# (observed: a 1.4s clean shot padded into the next shot's Turkish subtitle).
# Validate the ENTIRE final [t0, t1] window against every overlapping indexed
# shot's persisted flags; prefer SHORTENING to a clean sub-window that still
# contains the chosen moment; never trade the exact scene for unrelated filler.
# ---------------------------------------------------------------------------

_PARTIAL_CORNER_CACHE: dict = {}


def _partial_corner_shots(shots) -> dict:
    """{shot_index: corner} for shots carrying POSITIONALLY-CONSISTENT corner-logo evidence even
    when the source-level detector stays below its 25% presence threshold (an intermittent bug
    that fades in on a minority of shots — observed airing on a STILL). A cluster of ≥3 shots
    whose masks for the same corner mutually agree (IoU vs the cluster majority ≥0.45, bounded
    footprint, 2D-clustered) marks exactly THOSE shots dirty — not the whole source."""
    import numpy as np
    if not shots:
        return {}
    ck = (getattr(shots[0], "keyframe_path", "") or getattr(shots[0], "source_id", ""), len(shots))
    if ck in _PARTIAL_CORNER_CACHE:
        return _PARTIAL_CORNER_CACHE[ck]
    while len(_PARTIAL_CORNER_CACHE) >= 256:
        _PARTIAL_CORNER_CACHE.pop(next(iter(_PARTIAL_CORNER_CACHE)))
    _PARTIAL_CORNER_CACHE[ck] = {}
    if _source_is_static(shots):
        return {}
    from .index import _mask_from_hex
    out: dict = {}
    for corner in ("tl", "tr", "bl", "br"):
        members = []
        for sh in shots:
            h = (getattr(sh, "corner_masks", None) or {}).get(corner)
            m = _mask_from_hex(h) if h else None
            if m is not None and m.mean() > 0.02:
                members.append((sh.index, m))
        if len(members) < 3:
            continue
        maj = np.stack([m for _, m in members]).mean(axis=0) >= 0.5
        if maj.sum() < 3 or maj.mean() > 0.85:
            continue
        ys, xs = np.nonzero(maj)
        if len(ys) <= 2 or min(ys.std(), xs.std()) <= 0.6:
            continue
        ious = [(idx, float((m & maj).sum()) / max(1.0, float((m | maj).sum())))
                for idx, m in members]
        consistent = [idx for idx, i in ious if i >= 0.45]
        if len(consistent) >= 3 and float(np.mean([i for _, i in ious])) >= 0.40:
            for idx in consistent:
                out.setdefault(idx, corner)
    _PARTIAL_CORNER_CACHE[ck] = out
    return out


def _shot_dirty_reason(sh, partial_corner: dict | None = None,
                       band_graphics: bool = False) -> str:
    """'' when the shot may air, else the reason it must not — persisted-first, no image IO on
    a flagged index. Mirrors the _score_pool hard gates so window validation and candidate
    gating can never disagree. `band_graphics=True` = the caller established this source has
    solid hard-graphics evidence (>=3 persisted hard shots), arming the band tier here too."""
    if _shot_subtitle_band(sh):
        return "subs"
    if _shot_unreadable(sh):
        return "unreadable"
    if _ocr_is_junk(sh) or _ocr_text_heavy(sh):
        return "ocr-text"
    if _shot_overlay_badge(sh):
        return "overlay-badge"
    if _shot_static_collage(sh):
        return "static-collage"
    if int((getattr(sh, "scores", None) or {}).get("bonus_tail", 0) or 0):
        return "bonus-tail"
    try:
        import os as _os_g
        if int(getattr(sh, "graphics_flag", -1) or -1) >= 2 \
                and _os_g.environ.get("VIDLORE_CLIPSTUDIO_GRAPHICS_GATE", "1").strip() \
                not in ("0", "false", "no"):
            return "graphics"                          # persisted HARD tier only (no embeds in
    except (TypeError, ValueError):                    # scope); the pool gate computes live tiers
        pass
    if band_graphics and int(getattr(sh, "graphics_flag", -1) or -1) == 1:
        # band tier dirties ONLY with source context (the caller saw >=3 persisted hard shots in
        # this source) — mirrors the pool rule so a window shift can't re-admit what the pool gated
        return "graphics-band"
    if partial_corner:
        c = partial_corner.get(getattr(sh, "index", -1))
        if c:
            return f"logo-{c}"
    return ""


def _moment_kept(nt0: float, nt1: float, moment: tuple) -> bool:
    """Does the window [nt0, nt1] still SHOW the originally selected moment? Primary arm:
    the window contains the moment's midpoint. The overlap arm (keep ≥ OVERLAP × the
    moment's range, VIDLORE_CLIPSTUDIO_WQC_MOMENT_OVERLAP default 0.6) only participates
    for env values < 0.5: a window that misses the midpoint lies entirely on one side of
    it, so it can keep at most HALF the moment — at the strict default the midpoint rule
    governs alone. Deliberate: strict is the relevance-safe direction; lower the env
    below 0.5 to accept midpoint-less partial overlaps."""
    pm = (moment[0] + moment[1]) / 2.0
    if nt0 - 0.05 <= pm <= nt1 + 0.05:
        return True
    pdur = max(0.2, moment[1] - moment[0])
    ov = min(nt1, moment[1]) - max(nt0, moment[0])
    return ov >= _f_env("VIDLORE_CLIPSTUDIO_WQC_MOMENT_OVERLAP", 0.6) * pdur


def _wrong_face_spans(shots, t0: float, t1: float, face_guard) -> list:
    """Spans inside [t0, t1] where a CONFIDENTLY-NAMED main-cast face is not the beat's person.

    MEASURED on portal job 409e284b60: two beats whose chosen shot carried the RIGHT actor —
    faceid 1.0, identity 'Joseph Mawle' for a Benjen beat, 'Kit Harington' for a Jon Snow beat —
    still aired seconds showing somebody else. The shot was right and the WINDOW was wrong, which
    no shot-level score can see: by the time the window is cut, the scoring is over.

    `face_guard` is (targets, all_faces, fully_resolved) from resolve_face_targets. The three
    states are kept strictly apart, as everywhere else: a span is wrong only when the shot names
    someone CONFIDENTLY, that name is main cast, the beat's entity resolves FULLY, and the named
    person is not the target. An unnamed face is unknown, never wrong."""
    targets, all_faces, full = face_guard
    if not (targets and all_faces and full):
        return []
    out = []
    for sh in shots:
        if not (sh.end > t0 and sh.start < t1):
            continue
        conf = {(d.get("name", "") or "").lower()
                for d in (getattr(sh, "identities", None) or []) if d.get("confident")}
        conf = {c for c in conf if c}
        if not conf or (conf & targets):
            continue
        if conf & all_faces:
            ds, de = max(t0, float(sh.start)), min(t1, float(sh.end))
            if de - ds > 0.01:
                out.append((ds, de, f"wrong-face(shot {getattr(sh, 'index', '?')})"))
    return out


def clean_cut_window(shots, t0: float, t1: float, min_len: float,
                     anchor: tuple | None = None, partial_corner: dict | None = None,
                     preserve: tuple | None = None, face_guard: tuple | None = None):
    """Validate the FINAL render window [t0, t1] against all overlapping indexed shots.
    Returns (nt0, nt1, action, reason):
      action 'ok'        — window clean as-is
             'shortened' — [nt0, nt1] is the longest clean sub-window (≥ min_len) that still
                           overlaps `anchor` (the chosen shot's span → the exact scene survives)
             'rejected'  — no clean anchor-overlapping sub-window of ≥ min_len exists
    `preserve` — the ORIGINAL candidate range for a moment-locked beat (exact scene / quote /
    character-specific). A shortened window must still SHOW that moment (_moment_kept — the
    moment's midpoint at the default threshold); a clean span elsewhere in the same shot is
    'rejected' with a 'moment-lost:' reason so the caller falls back to relevance-ranked
    alternates instead of silently airing a DIFFERENT moment of the same shot.
    `face_guard` — (targets, all_faces, fully_resolved) for a beat that names a person. Seconds
    showing a confidently-named DIFFERENT main-cast member are treated as dirty, so the cut moves
    off them. STRICT SHORTEN-ONLY: the whole window is computed twice, once with those spans and
    once exactly as before, and the identity pass is used only when it does not reject. It can
    therefore move a cut but can never lose a shot — footage is never starved by it.
    Old indexes (no flags) report every shot clean → 'ok' (fail-open, keyframe gates still ran)."""
    anchor = anchor or (t0, t1)
    if face_guard:
        extra = _wrong_face_spans(shots, t0, t1, face_guard)
        if extra:
            r0, r1, act, why = _clean_cut_window_inner(
                shots, t0, t1, min_len, anchor, partial_corner, preserve, extra)
            if act != "rejected":
                return r0, r1, act, why
            #  fall through to the identical call WITHOUT the identity spans: a wrong face is a
            #  reason to prefer other seconds, never a reason to have no footage.
    return _clean_cut_window_inner(shots, t0, t1, min_len, anchor, partial_corner, preserve, [])


def _clean_cut_window_inner(shots, t0: float, t1: float, min_len: float,
                            anchor: tuple, partial_corner, preserve, extra_dirty: list):
    """The window algorithm, unchanged. `extra_dirty` is empty on the legacy path."""
    # A designed on-screen TEXT card — a channel/CTA/outro slate or a promo card — routinely FADES
    # in a few frames BEFORE its indexed shot boundary, so a window cleared to end exactly at that
    # boundary still airs a frame or two of the card (the Max/WarnerMedia outro aired from a ~0.15s
    # fade-in that begins just before the shot-33 boundary; a time-neutral cut ending at 147.9 still
    # showed slate luma 61). Pad only TEXT-card dirty spans (ocr-text) by a small safety margin so the
    # cleared window never abuts one — and WIDEN the shot-consideration window by that same margin so a
    # card whose indexed shot starts just BEYOND [t0,t1] (its fade bleeds back in) is still caught.
    # Burned dialogue subtitles (subs), dark murk, and corner logos keep their EXACT bounds — margining
    # subs would over-trim ordinary subtitled scenes and change long-standing shortening behavior; and
    # a shot that starts past t1 for a non-text reason yields a clamped-empty span, so widening `over`
    # is behaviour-neutral except for the ocr-text margin it enables.
    _edge = _f_env("VIDLORE_CLIPSTUDIO_WQC_EDGE_MARGIN", 0.35)
    over = [sh for sh in shots if sh.end > t0 - _edge and sh.start < t1 + _edge]
    # band-graphics context: mirror the pool rule (>=3 persisted hard shots arm the band tier)
    # so a shifted/padded window can't re-admit a band-art span the pool already refused
    _band_gfx = sum(1 for sh in shots
                    if int(getattr(sh, "graphics_flag", -1) or -1) >= 2) >= 3
    dirty = []
    for sh in over:
        r = _shot_dirty_reason(sh, partial_corner, band_graphics=_band_gfx)
        if r:
            ds, de = float(sh.start), float(sh.end)
            # avatar badges fade in/out around their shot bounds exactly like text cards do
            if r in ("ocr-text", "overlay-badge") and _edge > 0:
                ds -= _edge
                de += _edge
            ds, de = max(t0, ds), min(t1, de)
            if de - ds > 0.01:                        # skip clamped-empty spans (shot fully outside)
                dirty.append((ds, de, f"{r}(shot {getattr(sh, 'index', '?')})"))
    dirty.extend(extra_dirty)
    if not dirty:
        return t0, t1, "ok", ""
    dirty.sort()
    reasons = ", ".join(r for _, _, r in dirty)
    # subtract dirty spans → candidate clean spans
    spans, cur = [], t0
    for ds, de, _r in dirty:
        if ds - cur >= 0.2:
            spans.append((cur, ds))
        cur = max(cur, de)
    if t1 - cur >= 0.2:
        spans.append((cur, t1))
    # best clean span: must overlap the anchor (the exact chosen moment) and fit min_len
    best = None
    moment_lost = False
    for s0, s1 in spans:
        if s1 <= anchor[0] + 0.05 or s0 >= anchor[1] - 0.05:
            continue                                   # doesn't contain the chosen moment
        if (s1 - s0) < min_len - 1e-6:
            continue
        if preserve is not None and not _moment_kept(s0, s1, preserve):
            moment_lost = True                         # clean, but a DIFFERENT moment
            continue
        if best is None or (s1 - s0) > (best[1] - best[0]):
            best = (s0, s1)
    if best is None:
        return t0, t1, "rejected", (f"moment-lost: {reasons}" if moment_lost else reasons)
    # 'shortened' returns the FULL best clean span: spans are built strictly inside [t0, t1],
    # so the span never exceeds the requested duration and no sub-positioning happens here.
    # When `preserve` is set the span already passed _moment_kept in the loop above; a caller
    # that airs a SHORTER slice of this span (build.wqc_render_window) positions that slice
    # on the moment and re-checks it there.
    return round(best[0], 3), round(best[1], 3), "shortened", reasons


def wqc_moment_policy(seg) -> str:
    """Window-QC MOMENT policy for a beat. 'exact' — the beat is locked to the originally
    selected moment (exact_scene / character_specific / any dialogue-quote beat): QC may trim
    around that moment but never slide to a different one. 'generic' — filler/abstract beats
    may shift within the same on-topic shot when that improves cleanliness. Unknown beats
    (seg=None) protect the moment — the safe default."""
    if seg is None:
        return "exact"
    p = _policy.policy_of(seg)
    if p in (_policy.EXACT, _policy.CHARACTER) or _policy.is_breakout_candidate(seg):
        return "exact"
    return "generic"


def face_guard_for(seg, char2actor: dict, all_faces: set) -> tuple | None:
    """(targets, all_faces, fully_resolved) for a beat that names a person, else None.

    One builder so every window-QC call site applies the same three-state rule, and so a beat
    whose entity does not fully resolve can never be used to call a face WRONG."""
    if not (seg is not None and char2actor and all_faces):
        return None
    tgt, full = resolve_face_targets(getattr(seg, "required_entity", "") or "", char2actor)
    return (tgt, set(all_faces), bool(full)) if (tgt and full) else None


def validate_candidate_window(cand, shot, shots, cfg, seg=None, face_guard=None):
    """PRODUCTION candidate window-QC — the single path behind match selections, beat windows
    AND the verifier's promotion repair. Validates cand's PADDED render window (cut_selection
    pads short shots to min_clip_sec) against the source's indexed `shots`; mutates cand
    in/out on 'shortened'. The original range is captured BEFORE any mutation.

    Moment policy (wqc_moment_policy): 'exact' anchors QC to the ORIGINAL candidate range and
    requires any shortened window to keep that moment (midpoint or strong overlap) — otherwise
    'rejected' so the caller uses relevance-ranked alternates or keeps the dirty exact scene,
    never a silently different moment. 'generic' keeps the whole-shot anchor.

    Returns (action, reason, meta) — meta = {policy, orig, final, preserved} for audit logs."""
    orig = (float(cand.in_point), float(cand.out_point))
    pol = wqc_moment_policy(seg)
    meta = {"policy": pol, "orig": orig, "final": orig, "preserved": True}
    if not shots:
        return "ok", "", meta                          # nothing indexed → fail-open
    pad_end = cand.in_point + max(cfg.min_clip_sec, cand.out_point - cand.in_point)
    if pol == "exact":
        anchor, preserve = orig, orig
    else:
        # stub-tolerant: verifier tests promote bare SimpleNamespace shots without spans
        _ss, _se = getattr(shot, "start", None), getattr(shot, "end", None)
        anchor = (float(_ss), float(_se)) if _ss is not None and _se is not None else None
        preserve = None
    nt0, nt1, act, why = clean_cut_window(shots, cand.in_point, pad_end, cfg.min_clip_sec,
                                          anchor=anchor, preserve=preserve,
                                          face_guard=face_guard)
    if act == "shortened":
        cand.in_point, cand.out_point = nt0, nt1
        meta["final"] = (nt0, nt1)
    meta["preserved"] = (act != "rejected") if pol == "exact" else True
    return act, why, meta


def _wqc_log_line(act: str, meta: dict, why: str) -> str:
    """Uniform audit tail: policy + original candidate range + final range + preservation."""
    o, f = meta["orig"], meta["final"]
    return (f"policy={meta['policy']} orig=[{o[0]:.1f}-{o[1]:.1f}] "
            f"final=[{f[0]:.1f}-{f[1]:.1f}] moment-preserved="
            f"{'yes' if meta['preserved'] else 'NO'} reason={why}")


def wqc_arbitrate_selection(best, alternates, by_src_shots, ps_by_key, cfg, seg,
                            stats=None, progress=None, shot_uses=None, shot_cap=None,
                            face_guard=None):
    """PRODUCTION selection-level window-QC arbitration, one beat at a time (module-level so
    tests drive the real path). Validates the winning candidate's final window; on 'rejected'
    scans the relevance-ranked `alternates` for the first whose window validates (fallback);
    when none does, the original stays (a dirty exact scene still beats unrelated footage).
    Returns (best, alternates) — best is (adj, base, ps, cand)."""
    stats = stats if stats is not None else {}
    _adjq, _baseq, _psq, _candq = best
    shs = by_src_shots.get(_candq.source_id) or []
    act, why, meta = validate_candidate_window(_candq, _psq.shot, shs, cfg, seg,
                                               face_guard=face_guard)
    seg_i = getattr(seg, "index", _candq.segment_index)
    if act == "shortened":
        stats["shortened"] = stats.get("shortened", 0) + 1
        if progress:
            progress(f"window-qc: shortened seg={seg_i} src={_candq.source_id[:28]} "
                     f"{_wqc_log_line(act, meta, why)}")
    elif act == "rejected":
        # Prefer alternates that are still UNDER the reuse cap — this promotion runs after the
        # greedy loop and would otherwise re-air an exhausted window. Over-cap alternates stay
        # available as a last resort (an already-seen shot beats an unusable window).
        _under, _over = [], []
        for _c in alternates:
            if shot_uses is not None and shot_cap is not None and \
                    shot_uses.get((_c.source_id, _c.shot_index), 0) >= shot_cap:
                _over.append(_c)
            else:
                _under.append(_c)
        for _alt in (_under + _over):
            _aps = ps_by_key.get((_alt.source_id, _alt.shot_index))
            if _aps is None:
                continue
            _a_shs = by_src_shots.get(_alt.source_id) or []
            _a_act, _a_why, _a_meta = validate_candidate_window(_alt, _aps.shot, _a_shs, cfg, seg,
                                                                face_guard=face_guard)
            if _a_act != "rejected":
                stats["fallback"] = stats.get("fallback", 0) + 1
                if progress:
                    progress(f"window-qc: fallback seg={seg_i} "
                             f"{_candq.source_id[:24]}→{_alt.source_id[:24]} "
                             f"{_wqc_log_line(_a_act, _a_meta, why)}")
                best = (_adjq, float(_alt.score), _aps, _alt)
                alternates = [c for c in alternates
                              if (c.source_id, c.shot_index) != (_alt.source_id,
                                                                 _alt.shot_index)]
                return best, alternates
        stats["kept-dirty"] = stats.get("kept-dirty", 0) + 1
        if progress:
            progress(f"window-qc: rejected seg={seg_i} src={_candq.source_id[:28]} "
                     f"kept-dirty (no clean window keeping the moment, no valid alternate) "
                     f"{_wqc_log_line(act, meta, why)}")
    return best, alternates


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


# ── DEICTIC-TARGET visibility probe (no vision LLM) ──────────────────────────────────────────
# "Watch the chalice" must outrank "right people, no chalice". A bare CLIP cosine cannot carry
# this (measured ~0.02 discriminative range across 4301 shots — see the note above _clip01), so
# the probe copies the CONTRASTIVE-anchor pattern the graphics gate proved out: positive prompts
# about the named target vs generic-scene negatives, margin rank-normalized WITHIN the beat's
# own pool (adaptive — no absolute calibration to drift).
_TGT_VEC_CACHE: dict = {}
_TGT_NEG_PROMPTS = ("a wide shot of a medieval hall with many people",
                    "two people in conversation, faces visible",
                    "a dark castle corridor",
                    "soldiers standing in a courtyard")


def _target_pool_scores(target: str, pool: list, vr) -> dict | None:
    """{(sid, shot_index): 0..1 rank-normalized contrastive margin} for 'is TARGET visible in
    this shot', or None when undecidable (no embeds / degenerate spread). Milliseconds: two
    text-tower passes per unique target (cached) + one matrix product over persisted embeds."""
    try:
        import numpy as np
        key = (target or "").strip().lower()
        if not key:
            return None
        if key not in _TGT_VEC_CACHE:
            pos = (f"a close-up photograph of {target}",
                   f"{target}, clearly visible in the frame")
            P = np.asarray([vr._txt_embed(p) for p in pos], dtype="float32")
            N = np.asarray([vr._txt_embed(p) for p in _TGT_NEG_PROMPTS], dtype="float32")
            _TGT_VEC_CACHE[key] = (P, N)
        P, N = _TGT_VEC_CACHE[key]
        margins, vals = {}, []
        for ps in pool:
            if ps.embed is None:
                continue
            m = float((P @ ps.embed).max() - (N @ ps.embed).max())
            margins[(ps.sid, ps.shot.index)] = m
            vals.append(m)
        if len(vals) < 8:
            return None
        vals.sort()
        lo = vals[int(0.50 * (len(vals) - 1))]
        hi = vals[int(0.95 * (len(vals) - 1))]
        if hi - lo < 1e-6:
            return None
        return {k: max(0.0, min(1.0, (v - lo) / (hi - lo))) for k, v in margins.items()}
    except Exception:                                    # noqa: BLE001
        return None


def _score_pool(seg: ScriptSegment, pool: list[_PoolShot], text_vec, cfg: ClipConfig,
                face_targets: set, anchor_sids: set | None = None,
                anchor_bonus: float = 0.0,
                all_faces: set | None = None,
                title_toks: dict | None = None,
                mv_toks: set | None = None,
                beat_era: str = "",
                src_titles: dict | None = None,
                proj=None,
                anchor_lines: list | None = None,
                tgt01: dict | None = None,
                anchor_ep=None,
                beat_era_soft: bool = False) -> list[tuple[float, float, dict, _PoolShot]]:
    import numpy as np
    import os
    # DETERMINISTIC ERA PENALTY. The vision verifier cannot read a season off a torch-lit hall —
    # measured: an S7/S8 Winterfell great-hall shot (Bran's wheelchair in frame) aired under S4
    # Purple Wedding narration, sandwiched between two correct S4E2 shots, because the only era
    # gates were prompt-side. When the beat DECLARES an era (its own words, an event mapping, or
    # anchor-scene inheritance — see era.beat_era) and a source's TITLE declares seasons that do
    # NOT include it (range-aware: 'Seasons 1-8' includes 4), the shot is hard-penalized, same
    # doctrine as wrongface: it loses to any non-conflicting shot but remains a last resort.
    # Undeclared on either side never penalizes (era is never guessed).
    _era_pen = _f_env("VIDLORE_CLIPSTUDIO_ERA_PENALTY", 0.5)
    _era_conf_sids: set = set()
    if _era_pen > 0 and beat_era and src_titles:
        from . import era as _era_sp
        for _sid, _t in src_titles.items():
            try:
                if _era_sp.title_era_conflicts(beat_era, _t):
                    _era_conf_sids.add(_sid)
            except Exception:
                pass
    # ERA AGREEMENT — the positive half of the test above, which until now was one-sided: a title
    # declaring the WRONG season was punished while one declaring the RIGHT season earned nothing.
    # MEASURED on beat 15 ("The show wrote this ending in season one"): the correct upload, titled
    # "Game of Thrones S01E01 White Walkers Attack", lost to a season-less compilation, "Game of
    # Thrones || The White Walkers". Scene-title affinity could not save it because a beat's era is
    # INVISIBLE to token matching — titles have their digits stripped and 'season'/'episode' are
    # stopwords, so "season one" can never match "S01E01" as text however hard the tokens are
    # weighted. Exact_scene beats only, and only for a title naming EXACTLY that season: a
    # "Seasons 1-8" compilation that merely includes it is not evidence of anything.
    _era_match_sids: set = set()
    _era_bonus_w = _f_env("VIDLORE_CLIPSTUDIO_ERA_MATCH_BONUS", 0.12)
    if _era_bonus_w > 0 and beat_era and src_titles and not beat_era_soft:
        from . import era as _era_mt
        _bs = _era_mt.parse_season(beat_era or "")
        if _bs:
            for _sid, _t in src_titles.items():
                try:
                    if _era_mt.title_seasons(_t) == {_bs}:
                        _era_match_sids.add(_sid)
                except Exception:
                    pass
    # SCENE-TITLE AFFINITY (ranking-only, the same decoupling as the anchor-continuity bonus).
    # CLIP + transcript are blind to MICRO-moments: a beat about plucking a tiny stone off a
    # necklace has no distinctive thumbnail (the action is 1-2s and the object centimetres wide)
    # and the scene is near-silent, so neither signal fires — measured: a source whose TITLE
    # literally named the beat's action (4 shared scene-query tokens) never entered the beat's
    # alternates, and the beat release-blocked while the right footage sat downloaded in the pool.
    # An uploader's title is human scene-labeling — use it: shots from a source whose title shares
    # >=2 scene-specific query tokens get a small ranking bonus (kept OUT of `base`, so reported
    # confidence is undistorted). Specific/exact beats only; env VIDLORE_CLIPSTUDIO_TITLE_AFFINITY=0
    # disables.
    #
    # MAGNITUDE (measured, job 5cab63d801 beat 0 — "a hand closes around Arya Stark's throat and
    # lifts her off the ground", scene_query naming the Night King, policy exact_scene). The pool
    # held 25 Night-King-titled sources. The correct one ranked #1 and LOST by 0.0084 to an S7E4
    # "Arya returns to Winterfell" shot. The decomposition is the whole story:
    #   CLIP could not discriminate  — clip_cos 0.3351 (wrong) vs 0.3354 (right); the right scene
    #                                  was fractionally BETTER on CLIP and it changed nothing.
    #   faceid gave both 0.30        — Arya is in most of the pool, so it ranks nothing.
    #   transcript gave the wrong    — 0.5 x w_trans = +0.10, earned purely by the clip SPEAKING
    #     shot the margin              "Arya Stark" (entities count double in _text_sim), i.e. the
    #                                  same character-presence evidence faceid already scored.
    #   title affinity, the ONLY     — capped at 0.12, and awarded 0.09.
    #     scene-identity evidence
    # So character presence was worth 0.50 and scene identity 0.12. On an exact_scene beat that
    # ordering is backwards: every candidate shares the subject, only one shares the SCENE.
    #
    # Two changes, both ranking-only (no gate, no rejection — the pool is never narrowed):
    #  1. INFORMATION-WEIGHTED hits. A raw count treats 'arya' (41% of this pool's titles) as equal
    #     evidence to 'throat' (0%). Weight each hit by its inverse title-frequency over THIS pool,
    #     so the measure self-calibrates per video instead of needing a hand-tuned constant.
    #  2. A ceiling worth having on exact_scene beats, where naming the scene is the entire job.
    #     is_specific_claim beats keep the old modest cap — they assert a fact, not a shot.
    _aff = _f_env("VIDLORE_CLIPSTUDIO_TITLE_AFFINITY", 0.12)
    _aff_exact = _f_env("VIDLORE_CLIPSTUDIO_TITLE_AFFINITY_EXACT", 0.34)
    _ta_full = _f_env("VIDLORE_CLIPSTUDIO_TITLE_AFFINITY_FULL", 0.90)
    # ONE switch back to the pre-fix behaviour (raw hit count, single 0.12 cap) — it is what the
    # A/B's OFF arm runs, and it stays as the production escape hatch.
    _aff_itf = os.environ.get("VIDLORE_CLIPSTUDIO_TITLE_AFFINITY_ITF", "1").strip() \
        not in ("0", "false", "no")
    # COVERAGE vs ABSOLUTE MASS. `_ta_full` is an absolute ITF mass, and on a 62-title pool almost
    # every token is "rare": 0 titles -> 1.000, 1 -> 0.833, 2 -> 0.735. So ONE lucky collision very
    # nearly saturates the 0.90 threshold and buys the whole cap.
    #
    # Measured, job 6a26707939 beat 82, "Game of Thrones Bolton banner removed Winterfell Stark
    # banner raised". The winning source is "GoT 5x2 - Stannis offers Jon Snow to become Jon STARK,
    # Lord of WINTERFELL" — a legitimisation scene with no banner in it. It hits {stark, winterfell}
    # for mass 1.254, saturates, and takes the full 0.34. The correct upload, "Jon Snow takes back
    # Winterfell", whose shot 64 IS the flayed-man banners lying in the snow and which is CLIP rank
    # 1 of 1497 with base 0.80, hits only {winterfell} and is zeroed outright by the >=2 floor. The
    # rank-261 shot wins 0.8875 to 0.7944 and the beat release-blocked the render.
    # 'stark' scored ITF 0.833 because it appears in exactly ONE title in the pool — the wrong one.
    # Rarity in a 62-title corpus is not evidence about a scene.
    #
    # COVERAGE asks the question the bonus is actually for: of everything that makes this scene
    # identifiable, how much does this title actually name? Normalise by the query's OWN total ITF
    # mass instead of a constant. Beat 82 then pays the winner 1.254/4.989 = 25% of the cap (0.085)
    # and the correct source 8% (0.029). It also retires the >=2 count floor: one common token
    # becomes a small fraction rather than a coin flip between zero and the maximum.
    #
    # MEASURED AND NOT SHIPPED (default stays "absolute"). On the offline replay of this job the
    # coverage arm moved 119 of 167 picks and STILL did not land the banner shot on a single one of
    # the eight beats it was built for. Damping the bonus only hands the decision to the reuse and
    # recency penalties, which are just as large as the CLIP range — so the video is re-rolled
    # wholesale for no gain, which is the exact trade this project keeps making and must stop
    # making. The real defect on those beats turned out to be the one-shot-per-source bench below.
    # Kept, documented and measurable, because the reasoning is still right and it may matter once
    # the penalty stack is bounded. Control arm reproduces 0/167 changed.
    _ta_mode = os.environ.get("VIDLORE_CLIPSTUDIO_TITLE_AFFINITY_MODE", "absolute").strip().lower()
    _ta_damp = os.environ.get("VIDLORE_CLIPSTUDIO_TITLE_AFFINITY_DAMP", "hard").strip().lower()
    _exact_beat = (getattr(seg, "visual_policy", "") or "") == "exact_scene"
    _aff_cap = _aff_exact if (_exact_beat and _aff_itf) else _aff
    _sq_toks: set = set()
    _sq_itf: dict = {}
    _sq_itf_total: float = 0.0             # bound unconditionally — never a NameError under a gate
    if _aff > 0 and title_toks and (
            _exact_beat or bool(getattr(seg, "is_specific_claim", False))):
        import re as _re_ta
        try:
            from .discover import _STOPQ as _TA_STOP
        except Exception:
            _TA_STOP = set()
        _sq_toks = {w for w in _re_ta.findall(
                        r"[a-z']+", (getattr(seg, "scene_query", "") or "").lower())
                    if len(w) > 2 and w not in (mv_toks or set()) and w not in _TA_STOP}
        if _sq_toks and src_titles and _aff_itf:
            import math as _m_ta
            _df_ta: dict = {}
            for _t_ta in src_titles.values():
                for _w_ta in set(_re_ta.findall(r"[a-z']+", (_t_ta or "").lower())):
                    _df_ta[_w_ta] = _df_ta.get(_w_ta, 0) + 1
            _n_ta = max(1, len(src_titles))
            _ln_ta = _m_ta.log(_n_ta + 1) or 1.0
            _sq_itf = {w: _m_ta.log((_n_ta + 1) / (_df_ta.get(w, 0) + 1)) / _ln_ta
                       for w in _sq_toks}
            # the query's OWN identifying mass — the denominator coverage mode measures against.
            # A token no title contains still counts here: failing to name the scene's rarest word
            # is exactly the evidence coverage is meant to price in.
            _sq_itf_total = sum(_sq_itf.values())
    gate_on = os.environ.get("VIDLORE_CLIPSTUDIO_OCR_GATE", "1").strip() not in ("0", "false", "no", "")
    tgate_on = os.environ.get("VIDLORE_CLIPSTUDIO_TEXT_GATE", "1").strip() not in ("0", "false", "no", "")
    # MOMENT-LOCK: resolve the beat to a dialogue line ONCE per beat (not per candidate shot).
    _mm_on = proj is not None and os.environ.get(
        "VIDLORE_CLIPSTUDIO_MOMENT_LOCK", "1").strip() not in ("0", "false", "no")
    _bq = _beat_quote(seg, anchor_lines) if _mm_on else ""
    # big enough to beat CLIP's noise swing (w_clip 0.80 over a ~0.02-cosine real range), and only
    # ever awarded when find_quote_span matched the phrase at >= _MOMENT_MIN_RATIO.
    _mom_w = _f_env("VIDLORE_CLIPSTUDIO_MOMENT_WEIGHT", 0.9)
    # the graphics mirror keys on ITS OWN switch (same as the pool builder) — keying it on
    # TEXT_GATE made VIDLORE_CLIPSTUDIO_GRAPHICS_GATE=0 a dead switch (the pool re-admitted
    # shots that this mirror immediately dropped again)
    ggate_on = os.environ.get("VIDLORE_CLIPSTUDIO_GRAPHICS_GATE", "1").strip() \
        not in ("0", "false", "no") \
        and os.environ.get("VIDLORE_CLIPSTUDIO_NONSHOW_GATE", "1").strip() \
        not in ("0", "false", "no")
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
        if tgate_on and _shot_overlay_badge(ps.shot):
            continue                                      # commenter-avatar badge NEVER airs
        if ggate_on and _shot_is_graphics(ps.shot, getattr(ps, "embed", None)):
            continue                                      # designed graphics/illustration NEVER airs
        if _shot_static_collage(ps.shot):
            continue                                      # frozen collage/AI-art card NEVER airs
        if int((getattr(ps.shot, "scores", None) or {}).get("bonus_tail", 0) or 0):
            continue                                      # '+ BONUS' tail (interview/featurette)
        if subband_on and _shot_subtitle_band(ps.shot):
            continue                                      # burned subs (any script) NEVER air
        if _black_floor > 0 and float(getattr(ps.shot, "quality", 1.0) or 1.0) < _black_floor:
            continue                                      # near-black / unusable frame never airs
        if _shot_unreadable(ps.shot):
            continue                                      # near-black across the WHOLE shot span
                                                          # (multi-frame luma; keeps candlelit scenes)
        if _shot_featureless(ps.shot):
            continue                                      # lit but empty: grey/white card, CTA
                                                          # end-slate, washed-out fog
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
        # MOMENT-LOCK: where inside this SOURCE is that line actually spoken? The shot-transcript
        # test above misses any line that straddles a cut (midpoint binning), which is most of them.
        _mom = 0.0
        _mom_ratio = 0.0
        if _mm_on and _bq:
            _sp = locate_beat_moment(proj, ps.sid, seg, anchor_lines)
            if _sp:
                _mom = _moment_proximity(ps.shot, _sp)
                _mom_ratio = float(_sp[2])
                if _mom > dlg:
                    dlg = _mom                        # the stream located it; trust that over the bin
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
        _ebonus = 0.0
        if _exact_beat and ps.sid in _era_match_sids:
            _ebonus = _era_bonus_w
            bonus += _ebonus
        # MOMENT-LOCK is EVIDENCE, not similarity: the words are demonstrably spoken here, in this
        # source, at this second. CLIP is a guess whose whole discriminative range on this material
        # is ~0.02 cosine (measured across 4301 shots) yet whose weight is 0.80 — so without a
        # decisive term a shot sitting exactly on the line loses to CLIP noise. Rides on `bonus`
        # (ranking only) so reported confidence stays an honest match-quality number.
        _mom_bonus = 0.0
        if _mom > 0.0 and _mom_ratio >= _MOMENT_MIN_RATIO:
            _mom_bonus = _mom_w * _mom * min(1.0, _mom_ratio)
            # LEGIBILITY DAMPER. Finding the moment is worthless if the copy holding it is
            # unwatchable. Measured on the v3 render: moment-lock beats averaged 6.48 relevance vs
            # 6.12 for the rest — the mechanism works — but 3 of 8 criticals came from it, all
            # "too_dark_illegible", because a 0.9 bonus happily out-ranked a bright HD alternative
            # of the same scene. "Keep your eye on the dagger" landed on the right handover in a
            # copy where 95% of pixels sat under luma 48. So scale the bonus by how watchable this
            # copy is: an unreadable shot gets none of it, a dim one gets part, and a clean copy of
            # the same moment wins.
            _la = float(getattr(ps.shot, "luma_avg", -1.0) or -1.0)
            if _shot_unreadable(ps.shot):
                _mom_bonus = 0.0
            elif 0.0 <= _la < _MOMENT_LUMA_OK:
                _mom_bonus *= max(0.25, _la / _MOMENT_LUMA_OK)
            bonus += _mom_bonus
        _tbonus = 0.0
        if _sq_toks:
            _tw = (title_toks or {}).get(ps.sid) or set()
            _thit_w = [w for w in _sq_toks
                       if any(t == w or (t.startswith(w) and len(t) - len(w) <= 2)
                              for t in _tw)]
            # COVERAGE mode needs no count floor — see the note at _ta_mode. ABSOLUTE mode keeps it,
            # because there one rare token saturates and the floor is the only thing stopping it.
            _coverage = bool(_sq_itf) and _ta_mode == "coverage"
            if len(_thit_w) >= (1 if _coverage else 2):
                if _coverage:
                    _tbonus = _aff_cap * min(
                        1.0, sum(_sq_itf.get(w, 0.0) for w in _thit_w) / max(1e-6, _sq_itf_total))
                elif _sq_itf:
                    _tbonus = _aff_cap * min(
                        1.0, sum(_sq_itf.get(w, 0.0) for w in _thit_w) / max(1e-6, _ta_full))
                else:
                    _tbonus = _aff_cap * min(len(_thit_w), 4) / 4.0
                # A VIDEO ESSAY ABOUT the scene is not the scene. Titles like "Why Arya Killing the
                # Night King is Perfect" match the query as well as the scene upload does, but their
                # footage is interleaved with talking heads, text cards and cross-era B-roll — on
                # this very job an essay source outranked "GoT S08E03 - Arya kills the Night King".
                # Halved, not excluded: essays often hold the only copy of a moment, and dropping
                # them would narrow the pool, which is exactly what we must not do.
                if _aff_itf and _ESSAY_TITLE_RX.search((src_titles or {}).get(ps.sid, "") or ""):
                    _tbonus *= 0.5
                # LEGIBILITY DAMPER — naming the scene is not the same as showing it, so a title
                # match must not buy an UNWATCHABLE copy over a legible one of the same moment.
                # MEASURED (intro A/B, job 5cab63d801) in two strengths:
                #   "full" (unreadable -> 0, dim -> scaled, the moment-lock damper verbatim) fixed
                #         the blurry beat 6 but BROKE beat 18: "Arya kills the Night King in the
                #         godswood" is a NIGHT BATTLE, so the correct S08E03 source was dim by
                #         nature, lost its bonus, and a bright S08E01 reunion won instead (10->5).
                #   "hard" damps only what the pipeline already calls unreadable, leaving
                #         legitimately dark scenes their evidence.
                # Dark is not the same as illegible, and the Long Night punishes any rule that
                # confuses them.
                if _aff_itf and _ta_damp != "off":
                    _la_ta = float(getattr(ps.shot, "luma_avg", -1.0) or -1.0)
                    if _shot_unreadable(ps.shot):
                        _tbonus = 0.0
                    elif _ta_damp == "full" and 0.0 <= _la_ta < _MOMENT_LUMA_OK:
                        _tbonus *= max(0.25, _la_ta / _MOMENT_LUMA_OK)
                    # TRIED AND REJECTED — scaling the bonus by the shot's own quality
                    # (`_tbonus *= 0.55 + 0.45*quality`) to steer WITHIN a title-matched source
                    # toward its watchable shots. It fixed the beat it was designed for (6: 5->9)
                    # and cost more than it earned: 8 beats moved instead of 5 and the changed-beat
                    # mean fell 7.25 -> 6.50, with beat 1 collapsing 9 -> 0 onto "dark silhouettes
                    # in a fiery haze". Quality is already in `base` via w_clip's own inputs; a
                    # second helping of it here just re-ranks on brightness.
                bonus += _tbonus
        # DEICTIC-TARGET signal — "watch the chalice": the CLIP probe's per-shot visibility rank
        # is RECORDED (sig['target_vis'] below) for the verifier's deep-bench ordering and the
        # still pass, but it does NOT move ranking by default. MEASURED TWICE on job 5462677f95
        # (gate_ab, 24-25 changed beats, same-stage arms): a flat bonus scored −0.21 and a
        # clip01-multiplied bonus −0.54 — both imported LOOK-ALIKE WRONG SCENES (Red Wedding /
        # S3-wedding banquets under Purple Wedding beats; 7→2 regressions), because CLIP cannot
        # separate two candlelit feasts at the granularity that matters. Same lesson as look-gate
        # v1: searching harder for the named thing finds a worse clip. The probe's value survives
        # on the FAILURE paths only (bench order when the pick is already unusable; still query),
        # which never gamble a usable pick. VIDLORE_CLIPSTUDIO_LOOK_MATCH_BONUS=1 re-enables the
        # (measured-negative) ranking bonus for experiments.
        _tgvis = 0.0
        if tgt01 is not None:
            _tgvis = float(tgt01.get((ps.sid, ps.shot.index), 0.0))
            if _tgvis > 0.0 and clip01 > 0.0 and os.environ.get(
                    "VIDLORE_CLIPSTUDIO_LOOK_MATCH_BONUS", "0").strip() in ("1", "true", "yes"):
                try:
                    _tgw = float(os.environ.get("VIDLORE_CLIPSTUDIO_LOOK_MATCH_W",
                                                "0.12") or 0.12)
                except (TypeError, ValueError):
                    _tgw = 0.12
                bonus += _tgw * _tgvis * clip01
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
        _era_conf = ps.sid in _era_conf_sids
        if _era_conf:
            # soft (anchor-inherited) era = HALF penalty: a nudge toward the anchor's season for
            # era-silent beats, never the hard wrong-era hammer reserved for explicit claims
            base -= _era_pen * (0.5 if beat_era_soft else 1.0)
        # WRONG-EPISODE soft nudge (single_scene, EXACT beats only): an S8E2-titled upload under
        # an S8E4-anchored exact beat aired look-alike wrong-scene footage on 44 beats. SOFT by
        # design: small penalty, only when the title DECLARES a different episode code, never on
        # codeless titles and never on generic/abstract/character beats — exactness is demanded
        # only where the narration demands it.
        if anchor_ep is not None and _policy.policy_of(seg) == _policy.EXACT:
            _t_ep = _EPCODE_CACHE.get(ps.sid, _SENTINEL)
            if _t_ep is _SENTINEL:
                try:
                    from .era import parse_episode as _pe_sp
                    _t_ep = _pe_sp((src_titles or {}).get(ps.sid, "") or "")
                except Exception:
                    _t_ep = None
                _EPCODE_CACHE[ps.sid] = _t_ep
            if _t_ep is not None and tuple(_t_ep) != tuple(anchor_ep):
                try:
                    base -= float(os.environ.get("VIDLORE_CLIPSTUDIO_EPCODE_PENALTY",
                                                 "0.12") or 0.12)
                except (TypeError, ValueError):
                    base -= 0.12
        sig = {"clip": round(clip01, 4), "clip_cos": round(clip_cos, 4),
               "transcript": round(trans, 4), "faceid": round(faceid, 3),
               "object": round(obj, 3), "dialogue": round(dlg, 3),
               "quality": round(ps.shot.quality, 3)}
        if _mom:
            sig["moment_lock"] = round(_mom, 3)
            sig["moment_ratio"] = round(_mom_ratio, 3)
        if _mom_bonus:
            sig["moment_bonus"] = round(_mom_bonus, 3)
        if bonus:
            sig["anchor_bonus"] = round(bonus, 3)
        if _tbonus:
            sig["title_affinity"] = round(_tbonus, 3)
        if _ebonus:
            sig["era_match"] = round(_ebonus, 3)
        if _tgvis > 0.0:
            sig["target_vis"] = round(_tgvis, 3)
        if _era_conf:
            sig["era_conflict"] = 1.0        # numeric: ledger rounds every signal via float()
        if wrongface:
            sig["wrongface"] = True
        scored.append((base, bonus, sig, ps))
    scored.sort(key=lambda x: x[0] + x[1], reverse=True)
    return scored


def usable_shot_yield(proj, source_id: str, cfg: Optional[ClipConfig] = None) -> tuple:
    """(usable, total) shots for one source, applying the SHOT-level gates match will apply.

    A source can clear every SOURCE-level gate and still contribute nothing: the backfill pass
    fetched a replacement titled "Littlefinger gives Catspaw dagger to Bran Stark" — exactly the
    footage 8 beats were asking for — and it turned out to be another screener with
    "FOR INTERNAL VIEWING ONLY" burned into the picture, so all 11 of its shots died at
    _ocr_text_heavy and it won zero beats. The pass reported "+3 clean sources" and the pool had
    gained nothing. A replacement has to prove it can actually air."""
    import os as _os_y
    shots = _index.load_shots(proj, source_id)
    if not shots:
        return (0, 0)
    embeds = _index.load_embeds(proj, source_id)
    _on = lambda k, d="1": _os_y.environ.get(k, d).strip() not in ("0", "false", "no", "")  # noqa: E731
    gate_on, tgate_on = _on("VIDLORE_CLIPSTUDIO_OCR_GATE"), _on("VIDLORE_CLIPSTUDIO_TEXT_GATE")
    ggate_on = _on("VIDLORE_CLIPSTUDIO_GRAPHICS_GATE")
    subband_on = _on("VIDLORE_CLIPSTUDIO_SUBBAND_GATE")
    usable = 0
    for pos, sh in enumerate(shots):
        row = getattr(sh, "embed_row", -1)
        vec = embeds[row] if (embeds is not None and 0 <= row < len(embeds)) else None
        if gate_on and _ocr_is_junk(sh):
            continue
        if tgate_on and (_ocr_text_heavy(sh) or _shot_overlay_badge(sh)):
            continue
        if ggate_on and _shot_is_graphics(sh, vec):
            continue
        if subband_on and _shot_subtitle_band(sh):
            continue
        if _shot_unreadable(sh):
            continue
        usable += 1
    return (usable, len(shots))


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
    """Sort key for clean-copy arbitration — LOWER is cleaner. Order: watermark first, then burned-
    sub risk, then DECODED-QUALITY tier, then container resolution tier, then fine quality.

    Decoded quality (shot.quality = Laplacian-variance sharpness + brightness + resolution, measured
    on decoded frames) is the CODEC-NEUTRAL picture-quality signal and ranks ABOVE raw container
    resolution — so a genuinely sharp 720p beats a SOFT/upscaled 1080p, and a higher H.264 bitrate is
    never blindly equated with better decoded quality than an efficient VP9/AV1 copy. shot.quality
    already includes a resolution term, so a REAL sharp 1080p still wins; container resolution only
    breaks near-ties (0.25 quality bins keep sharpness noise from flipping genuine resolution ties)."""
    d = src_dirty.get(sid) or {}
    subs_risk = (float(d.get("subs", 0.0)) >= 0.12) or _shot_subtitle_band(shot)
    q = float(getattr(shot, "quality", 0.5) or 0.5)
    q_bin = round(q * 4) / 4.0                          # coarse decoded-quality tier (0.25 steps)
    return (1 if d.get("corner") else 0,
            1 if subs_risk else 0,
            -q_bin,
            -_res_tier(int(src_height.get(sid, 0) or 0)),
            -q)


def _clean_copy_swap(seg, best, scored, src_dirty: dict, src_height: dict, cfg,
                     *, eps: float = 0.03, shot_uses: dict | None = None,
                     shot_cap: int | None = None, proj=None, beat_quote: str = "",
                     anchor_lines: list | None = None):
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
    # Early-out ONLY when the current best is UNBEATABLE: no watermark (key[0]==0), no subs
    # (key[1]==0), top decoded-quality tier (key[2] <= -1.0, i.e. q_bin==1.0 — reachable only by a
    # pristine HD frame), AND top resolution tier (key[3]==-3 = 1080-class). NOTE the tuple order is
    # now (corner, subs, -decoded_quality_tier, -res_tier, -quality): decoded quality comes BEFORE
    # resolution, so the stale `key_b[2] == -3` (which used to test the 1080p res tier) no longer
    # applies — resolution is key[3]. A clean-but-SOFT 1080p is intentionally NOT skipped here, so a
    # sharp 720p copy can still win the arbitration below.
    if key_b[0] == 0 and key_b[1] == 0 and key_b[2] <= -1.0 and key_b[3] == -3:
        return best, None                          # already a clean, pristine 1080-class copy
    # A clean-copy substitution promises the SAME MOMENT in a cleaner source.  If a located quote
    # contributed any material moment-lock evidence to the current winner, remember both its
    # proximity and its stronger containment category: direct quote footage may not degrade into a
    # pre-roll-only copy, and a pre-roll/reaction winner may not jump to a far-away similar frame.
    _orig_mom = locate_beat_moment(proj, ps_b.sid, seg, anchor_lines) \
        if (proj and beat_quote) else None
    _orig_moment_proximity = (_moment_proximity(ps_b.shot, _orig_mom) if _orig_mom else 0.0)
    try:
        _recorded_moment = float((getattr(cand_b, "signals", None) or {}).get(
            "moment_lock", _orig_moment_proximity) or 0.0)
        _recorded_ratio = float((getattr(cand_b, "signals", None) or {}).get(
            "moment_ratio", (_orig_mom[2] if _orig_mom else 0.0)) or 0.0)
    except (TypeError, ValueError):
        _recorded_moment, _recorded_ratio = _orig_moment_proximity, 0.0
    _orig_moment_proximity = max(_orig_moment_proximity, _recorded_moment)
    _preserve_quote_moment = bool(
        _orig_mom and min(float(_orig_mom[2]), _recorded_ratio) >= _MOMENT_MIN_RATIO
        and _orig_moment_proximity > 0.0
    )
    try:
        from .relevance_contract import QUOTE_WINDOW_TOLERANCE_SEC as _quote_window_tol
        _quote_window_tol = float(_quote_window_tol)
    except Exception:
        _quote_window_tol = 0.75

    def _contract_contains(window, moment) -> bool:
        """Use the publication contract's exact tolerated-containment rule.

        A clean-copy swap must not turn a window which currently satisfies the quote contract into
        one which only overlaps the dialogue shot.  Strict geometric containment is not enough as
        the comparison predicate because publication deliberately permits 0.75s of ASR/editorial
        boundary tolerance.
        """
        if not moment or not window:
            return False
        try:
            w0, w1 = float(window[0]), float(window[1])
            q0, q1 = float(moment[0]), float(moment[1])
            return (w1 > w0 >= 0.0 and q1 >= q0 >= 0.0
                    and q0 >= w0 - _quote_window_tol
                    and q1 <= w1 + _quote_window_tol)
        except (TypeError, ValueError, IndexError):
            return False

    _orig_direct = bool(_orig_mom and float(ps_b.shot.end) >= float(_orig_mom[0])
                        and float(ps_b.shot.start) <= float(_orig_mom[1]))
    _orig_contains = _contract_contains((cand_b.in_point, cand_b.out_point), _orig_mom)
    tb = (getattr(ps_b.shot, "transcript", "") or "").lower().split()
    alt, alt_key, alt_base, alt_moment = None, key_b, 0.0, None
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
        _copy_mom = None
        if _preserve_quote_moment:
            _copy_mom = locate_beat_moment(proj, ps.sid, seg, anchor_lines)
            # The target source may contain the same scene more than once.  It is only a valid
            # clean copy for this quote-locked winner when THIS candidate has at least the same
            # moment proximity in its own source.  Otherwise _trim_window ignores the far-away
            # moment and silently cuts the shot midpoint — the beat-85 failure this guard closes.
            _copy_proximity = _moment_proximity(ps.shot, _copy_mom) if _copy_mom else 0.0
            _copy_direct = bool(_copy_mom and float(ps.shot.end) >= float(_copy_mom[0])
                                and float(ps.shot.start) <= float(_copy_mom[1]))
            _copy_window = (_trim_window(ps.shot, seg, cfg, _copy_mom)
                            if _copy_mom else (0.0, 0.0))
            _copy_contains = _contract_contains(_copy_window, _copy_mom)
            if not (_copy_mom and float(_copy_mom[2]) >= _MOMENT_MIN_RATIO
                    and _copy_proximity >= _orig_moment_proximity - 1e-6
                    and (not _orig_direct or _copy_direct)
                    and (not _orig_contains or _copy_contains)):
                continue
        # REUSE-AWARE: this swap runs AFTER the greedy loop that owns the reuse ledger, so without
        # this check it happily funnels many beats onto one "cleanest copy" and silently defeats the
        # per-shot cap (measured: one window won 6 beats against a cap of 2).
        if shot_uses is not None and shot_cap is not None and \
                shot_uses.get((ps.sid, ps.shot.index), 0) >= shot_cap:
            continue
        k = _cleanliness_key(ps.sid, ps.shot, src_dirty, src_height)
        if k < alt_key or (k == alt_key and alt is not None and base > alt_base):
            alt, alt_key, alt_base, alt_moment = (base, bonus, sig, ps), k, base, _copy_mom
    if alt is None:
        return best, None
    base, bonus, sig, ps = alt
    # the swap moves to a DIFFERENT copy of the scene, where the line sits at a different
    # timestamp — re-locate it in the new source, or the window falls back to the shot midpoint
    # and can miss the very words the beat is about.
    _sw_mom = alt_moment if _preserve_quote_moment else (       # already located in guarded path
        locate_beat_moment(proj, ps.sid, seg, anchor_lines) if (proj and beat_quote) else None)
    in_p, out_p = _trim_window(ps.shot, seg, cfg, _sw_mom)
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
    #  WINDOW-level wrong-character guard. The shot-level wrongface penalty cannot see this: two
    #  beats in job 409e284b60 chose a shot carrying the RIGHT actor (faceid 1.0, 'Joseph Mawle'
    #  for a Benjen beat, 'Kit Harington' for a Jon Snow beat) and still aired seconds showing
    #  someone else. Shorten-only by construction — see clean_cut_window's face_guard.
    import os as _os_wf          # this module has no module-level `os`; a bare one is a NameError
    _wf_win_gate = _os_wf.environ.get("VIDLORE_CLIPSTUDIO_WRONGFACE_WINDOW_GATE", "1").strip() \
        not in ("0", "false", "no")
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
    _ep_verified = bool(getattr(analysis, "episode_hint_verified", False))
    _tgt_season = _title_season(_ep)
    # ERA PURGE — deletes shots outright, so it demands the strongest evidence in the pipeline.
    # Measured: an UNVERIFIED hint of "S04E01" (the scene is S03E10) dropped 354 shots on every
    # run, including 'Game Of Thrones S03E10 Red Wedding Aftermath scene' — the correct episode,
    # zero shots used. Sources with no season in the title survived, so the filter punished honest
    # labelling and rewarded vague titles.
    #
    # Two conditions now: the hint must be CORROBORATED, and a source whose own audio speaks the
    # anchor dialogue can never be purged on a title string. Content beats labels.
    if single_scene and _tgt_season and _ep_verified and _os.environ.get(
            "VIDLORE_CLIPSTUDIO_ERA_FILTER", "1").strip() not in ("0", "false", "no"):
        _t2 = {s.id: (s.title or "") for s in proj.sources}
        _dialogue_ok = {s.id for s in proj.sources
                        if (getattr(s, "extra", None) or {}).get("anchor_verified")}
        _ok_ids = {ps.sid for ps in pool
                   if (_title_season(_t2.get(ps.sid, "")) in (None, _tgt_season)
                       or ps.sid in _dialogue_ok)}
        if len({ps.sid for ps in pool}) - len(_ok_ids) >= 1 and len(_ok_ids) >= 1:
            _before = len(pool)
            pool = [ps for ps in pool if ps.sid in _ok_ids]
            if progress and len(pool) < _before:
                progress(f"match: era filter — dropped {_before - len(pool)} off-season shot(s) "
                         f"(target season {_tgt_season}, hint corroborated)")
    elif single_scene and _tgt_season and not _ep_verified and progress:
        progress(f"match: era filter SKIPPED — episode hint {_ep!r} is uncorroborated; "
                 f"an unverified hint may not purge sources")
    _purity = _os.environ.get("VIDLORE_CLIPSTUDIO_SINGLE_SCENE_PURITY", "1").strip() \
        not in ("0", "false", "no")
    if single_scene and _purity:
        _essay_comp = _ESSAY_TITLE_RX
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
            # (a2) an EPISODE CODE in the title proves which EPISODE, never which SCENE — an
            # episode is ~56 minutes of unrelated material. Granting anchor status on the code
            # ALONE (and `continue`-ing past the dialogue check below) is how
            # 'S04E01 - Arya Stark and the Hound' and 'S04E01 Jon Snow meets Janos Slynt' became
            # anchors of a small-council video and collected the +0.45 bonus. The code is now a
            # corroborating hint: it still has to pass the dialogue/title evidence below, and only
            # counts at all once the hint itself is verified.
            _code_hit = bool(_ep and _ep_verified
                             and _ep in (src.title or "").lower().replace(" ", ""))
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
            # A verified episode code corroborates (drops the bar to 1 scene token) but can never
            # substitute for the entity evidence: 'S04E01 - Arya Stark and the Hound' carries the
            # code and no small-council entity, so it stays a non-anchor no matter how right the
            # code is.
            _need = 1 if _code_hit else 2
            twords = set(re.findall(r"\w+", (src.title or "").lower()))
            for ts in atok_sets:
                hits = {t for t in ts
                        if any(w == t or (w.startswith(t) and len(w) - len(t) <= 2)
                               for w in twords)}
                if len(hits) >= _need and (not ent_toks or hits & ent_toks):
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
    # the anchor scene's own dialogue — the fallback referent for a beat that DISCUSSES a line
    # without quoting it, so moment-lock still resolves (see _beat_quote).
    _anchor_lines: list = []
    for _as in ((getattr(proj, "meta", None) or {}).get("analysis", {}) or {}).get(
            "anchor_scenes", []) or []:
        for _dl in (_as.get("dialogue") or []):
            if isinstance(_dl, str) and _dl.strip():
                _anchor_lines.append(_dl.strip())
    import os as _os_mm
    _mm_gate = _os_mm.environ.get("VIDLORE_CLIPSTUDIO_MOMENT_LOCK", "1").strip() \
        not in ("0", "false", "no")
    _src_height = {s.id: int(getattr(s, "height", 0) or 0) for s in proj.sources}
    # CLEANLINESS MAP for same-scene clean-copy arbitration: per-source corner-bug + burned-sub
    # fraction, computed once over the pooled shots (both detectors are memoized).
    import os as _os_cc
    _clean_gate = _os_cc.environ.get("VIDLORE_CLIPSTUDIO_CLEAN_COPY_GATE", "1").strip() \
        not in ("0", "false", "no")
    _corner_gate = _os_cc.environ.get("VIDLORE_CLIPSTUDIO_CORNER_LOGO_GATE", "1").strip() \
        not in ("0", "false", "no")
    _by_src_shots: dict = {}
    _ps_by_key: dict = {}
    for ps in pool:
        _by_src_shots.setdefault(ps.sid, []).append(ps.shot)
        _ps_by_key[(ps.sid, ps.shot.index)] = ps
    _src_dirty: dict = {}
    if _clean_gate:
        for _sid, _shs in _by_src_shots.items():
            _src_dirty[_sid] = {"corner": (_source_corner_logo(_shs) if _corner_gate else ""),
                                "subs": _source_subs_frac(_shs)}
        _dirty_n = sum(1 for v in _src_dirty.values() if v["corner"] or v["subs"] >= 0.12)
        if progress and _dirty_n:
            progress(f"match: cleanliness map — {_dirty_n}/{len(_src_dirty)} source(s) carry a "
                     f"corner bug or burned subs (clean-copy arbitration active)")
    _wqc_gate = _os_cc.environ.get("VIDLORE_CLIPSTUDIO_WINDOW_QC", "1").strip() \
        not in ("0", "false", "no")
    _wqc_stats = {"shortened": 0, "fallback": 0, "kept-dirty": 0, "windows-dropped": 0}

    # partial-corner evidence is deliberately NOT a window-dirty reason: measured on real
    # footage it fires on scene-static elements (a candle sconce framed identically across
    # a scene's shots) — shrinking exact-scene cuts around those is a relevance regression.
    # Full-source corner bugs are handled by the punch-in crop instead.
    def _validate_cand_window(cand, shot, seg=None):
        """Thin adapter over the module-level PRODUCTION validator (tests call that directly)."""
        return validate_candidate_window(cand, shot, _by_src_shots.get(cand.source_id) or [],
                                         cfg, seg)
    source_uses: dict[str, int] = {}
    shot_uses: dict[tuple[str, int], int] = {}
    recent_sources: list[str] = []
    recent: list[dict] = []          # recency window: {pos, key, phash, embed} of recent picks
    selections: list[ClipSelection] = []
    # PER-WINDOW ledger for the anti-repeat block below. `shot_uses` counts airings but forgets
    # WHEN — and beat index is the wrong clock for "when" (a 272-beat essay runs 22 minutes, a
    # 189-beat one 14). These remember the timeline second and beat of each window's last airing.
    win_last_t: dict[tuple[str, int], float] = {}
    win_last_pos: dict[tuple[str, int], int] = {}
    # timeline clock: prefix sum over the planned beat durations, so "90 seconds ago" is real
    # seconds of finished video rather than a beat count.
    t_of: dict[int, float] = {}
    _acc = 0.0
    for _s in segments:
        t_of[_s.index] = _acc
        _acc += max(0.0, float(getattr(_s, "est_duration", 0.0) or 0.0))

    _hist_win = max(cfg.recency_cooldown, cfg.source_recency_window)
    for seg in segments:
        t_now = t_of.get(seg.index, 0.0)
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
        # DEICTIC-TARGET probe — when the narration instructs the viewer ("watch the chalice"),
        # score candidate shots for whether the named target is visible. SIGNAL-ONLY by default
        # (sig['target_vis'] feeds the verifier's deep-bench ordering + the still pass); the
        # ranking bonus is opt-in and measured negative — see the note in _score_pool.
        # Uses persisted embeds + a cached text probe; no vision calls, adaptive per-beat scaling.
        _tgt01 = None
        import os as _os_tgt
        if vr is not None and _os_tgt.environ.get(
                "VIDLORE_CLIPSTUDIO_LOOK_PROBE", "1").strip() not in ("0", "false", "no"):
            _tgt_phrase = _policy.deictic_target(seg)
            if _tgt_phrase:
                _tgt01 = _target_pool_scores(_tgt_phrase, pool, vr)
        face_targets = _face_targets(seg, char2actor)
        _ta_titles = {src.id: set(re.findall(r"[a-z']+", (src.title or "").lower()))
                      for src in proj.sources}
        _ana_m = (getattr(proj, "meta", None) or {}).get("analysis", {}) or {}
        _ta_mv = {w for w in re.findall(
                      r"[a-z']+", (_ana_m.get("movie_title", "") or "").lower()) if len(w) > 2}
        # the beat's declared era (own words / event mapping / anchor inheritance) for the
        # deterministic era penalty — computed with the same machinery verify uses
        try:
            from . import era as _era_mm
            _ana_shim_m = type("A", (), {"anchor_scenes": _ana_m.get("anchor_scenes"),
                                         "movie_title": _ana_m.get("movie_title", "")})()
            _beat_era_m = _era_mm.beat_era(
                seg, str(_ana_m.get("episode_hint") or ""),
                single_scene=(_ana_m.get("video_type") == "single_scene"),
                global_verified=bool(_ana_m.get("episode_hint_verified")),
                event_eras=_era_mm.event_eras_from(_ana_shim_m),
                anchor_eras=_era_mm.anchor_token_eras(_ana_shim_m))
        except Exception:
            _beat_era_m = ""
        _src_titles_m = {src.id: (src.title or "") for src in proj.sources}
        # single_scene ANCHOR EPISODE CODE — for the soft wrong-episode nudge on EXACT beats
        # only (see _score_pool). Deliberately NARROW: never a gate, never on generic/abstract
        # beats, never on codeless titles — strictness beyond the exact-scene beats would
        # starve the pool ("scene milna band ho jayega").
        _anchor_ep_m = None
        _era_soft_m = False
        if single_scene:
            try:
                from .era import parse_episode as _pe_m, parse_season as _ps_m
                _anc0 = ((_ana_m.get("anchor_scenes") or [{}])[0] or {})
                _anc_txt = (str(_anc0.get("episode", "") or "")
                            or str(_anc0.get("query", "") or ""))
                _anchor_ep_m = _pe_m(_anc_txt)
                # ANCHOR-ERA SOFT AFFINITY: a single_scene essay's era-SILENT beats default to
                # the anchor's season at HALF penalty — measured: 35 wrong-era beats (S1 Ned/
                # S3 captive-Jaime under S8E4-night narration) filled anchor beats era-blind.
                # Beats that carry their OWN era (the S1/S2/S6 backstory crimes) keep it and
                # are untouched; sources with no declared season are never penalized.
                if not _beat_era_m:
                    _a_season = _ps_m(_anc_txt)
                    if _a_season:
                        _beat_era_m = f"season {_a_season}"
                        _era_soft_m = True
            except Exception:
                _anchor_ep_m = None
        scored = _score_pool(seg, pool, text_vec, cfg, face_targets, anchor_sids, anchor_bonus,
                             all_faces=all_faces, title_toks=_ta_titles, mv_toks=_ta_mv,
                             beat_era=_beat_era_m, src_titles=_src_titles_m,
                             proj=proj, anchor_lines=_anchor_lines, tgt01=_tgt01,
                             anchor_ep=_anchor_ep_m, beat_era_soft=_era_soft_m)

        # `base` = match QUALITY (drives the reported confidence + flagging).
        # `adj`  = base + anchor bonus minus diversity penalties (drives only WHICH is picked).
        # Decoupled so anti-reuse/anchor-continuity never distort the reported confidence.
        best = None    # (adj, base, ps, cand)
        alt_best: dict[str, ClipCandidate] = {}    # best candidate per source → alternates
        # ...and the best candidate per SHOT, for the deep bench only. One-per-source is right for
        # `alternates` (their job is spread), and structurally unable to find a moment: a 21-strong
        # bench spans 21 sources × 1 shot each, so the second-best shot of the RIGHT file is
        # invisible. Measured on job 6a26707939, beats 24 and 82 ("the flayed man banners brought
        # down to the ground" / "the flayed man comes down off the walls"): the bench carried
        # game_of_thrones_jon_sn_123ebf87 shot 66 — the STARK banner going UP — so the verifier
        # correctly refused it and gave up, while shot 64 of the same file, the Bolton banners lying
        # in the snow, was never a candidate on any beat. Both beats release-blocked the render.
        shot_best: dict[tuple, ClipCandidate] = {}
        # VISUAL-POLICY variety (req. 2/5): filler/character beats should SPREAD across sources, so
        # their reuse + source-recency penalties are amplified. exact_scene beats keep the multiplier
        # at 1.0 — the precise scene must win even from a recently-used source.
        _variety = 1.6 if _policy.maximize_variety(seg) else 1.0
        # the dialogue line this beat is about, resolved once per beat (mirrors _score_pool)
        _bq_seg = _beat_quote(seg, _anchor_lines) if _mm_gate else ""
        # The near-adjacent replay block below is a HARD skip, so it can in principle starve a beat
        # of every candidate. `_hard_gap` is dropped on a second pass if that happens — a beat must
        # never go unselected (that escalates to the still/release-block path).
        _hard_gap = True
        for _pass in (0, 1):
          for base, bonus, sig, ps in scored:
            key = (ps.sid, ps.shot.index)
            # SINGLE-SCENE DEEP-DIVE: the anchor scene IS the video — an expert essayist keeps
            # cutting BETWEEN that one scene's shots (Bronn → the mug → the Hound → reaction), so the
            # limited anchor footage must be allowed to recur. We relax reuse/source/recency penalties
            # for anchor sources (but still block the IDENTICAL shot back-to-back via same-shot
            # recency, so it never looks like a freeze/loop).
            relax = (single_scene and bool(anchor_sids) and ps.sid in anchor_sids
                     and not _anchor_abundant)    # abundant anchor footage → normal anti-reuse, spread
            shot_cap = cfg.max_reuse_per_shot * (cfg.relax_reuse_mult if relax else 1)
            if shot_uses.get(key, 0) >= shot_cap:
                continue                              # this exact shot is exhausted
            # HARD near-adjacent block — applies EVEN under relax. Measured on the delivered file, the
            # ugliest repeats were the same window either side of a cut 2-5s apart ("cut to nowhere");
            # relax had disabled the same-scene arm of the recency rule that used to stop them.
            if _hard_gap and key in win_last_pos and (
                    (seg.index - win_last_pos[key]) <= cfg.window_min_gap_beats
                    or (t_now - win_last_t.get(key, -1e9)) < cfg.window_min_gap_sec):
                continue
            pen = cfg.reuse_penalty * source_uses.get(ps.sid, 0) * (0.25 if relax else _variety)
            # PER-WINDOW reuse cost: scales with how OFTEN this exact window already aired and how
            # RECENTLY in timeline seconds. Capped so a genuinely exact scene with no alternative can
            # still win rather than being replaced by something irrelevant.
            _nw = shot_uses.get(key, 0)
            if _nw:
                _wp = cfg.window_reuse_penalty * _nw * (0.5 if relax else _variety)
                _dt = t_now - win_last_t.get(key, -1e9)
                if _dt < cfg.window_reuse_gap_sec:
                    _wp += cfg.window_reuse_recency_weight * (
                        1.0 - max(0.0, _dt) / cfg.window_reuse_gap_sec)
                pen += min(_wp, 0.55)
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
            # centre the cut on the located line when there is one (see _trim_window)
            _cand_mom = locate_beat_moment(proj, ps.sid, seg, _anchor_lines) if _bq_seg else None
            in_p, out_p = _trim_window(ps.shot, seg, cfg, _cand_mom)
            cand = ClipCandidate(segment_index=seg.index, source_id=ps.sid,
                                 shot_index=ps.shot.index, score=qual,
                                 in_point=in_p, out_point=out_p, signals=sig)
            _sk = (ps.sid, ps.shot.index)
            _sc = shot_best.get(_sk)
            if _sc is None or cand.score > _sc.score:
                shot_best[_sk] = cand
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
          # the hard near-adjacent block starved this beat of every candidate — retry once without
          # it rather than leave the beat unselected (which escalates to the still/release path).
          if best is not None or not _hard_gap:
              break
          _hard_gap = False

        # SAME-SCENE CLEAN-COPY ARBITRATION: if another source holds a near-duplicate of the
        # winning shot at ~equal relevance, air the cleanest copy (no watermark → no burned subs →
        # higher resolution → sharper). The displaced pick stays available as an alternate.
        if _clean_gate and best is not None:
            _old_cand = best[3]
            best, _swap_note = _clean_copy_swap(seg, best, scored, _src_dirty, _src_height, cfg,
                                                shot_uses=shot_uses,
                                                shot_cap=cfg.max_reuse_per_shot,
                                                proj=proj, beat_quote=_bq_seg,
                                                anchor_lines=_anchor_lines)
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
            # Ranked DEEPER than the beat will use. The first `candidates_per_segment` are the
            # alternates every stage sees; the tail is the verifier's deep bench (see
            # ClipSelection.deep_alternates), read only when it is about to settle for a contextual
            # fallback. Truncating to 6 here left that bench empty and the rescue path dead.
            import os as _os_alt
            try:
                _keep_n = max(cfg.candidates_per_segment,
                              int(_os_alt.environ.get("VIDLORE_CLIPSTUDIO_DEEP_ALTERNATES", "20")
                                  or 20))
            except (TypeError, ValueError):
                _keep_n = max(cfg.candidates_per_segment, 20)
            _rank = (lambda c: c.score + float((c.signals or {}).get("anchor_bonus", 0.0)))
            alternates = sorted(
                (c for c in alt_best.values() if (c.source_id, c.shot_index) != _ckey),
                key=_rank, reverse=True)[:_keep_n]
            # SIBLING SHOTS are APPENDED past the one-per-source list. The head of `alternates` —
            # everything up to candidates_per_segment, which is what every other stage reads — is
            # untouched and still one-per-source, so the spread the alternates exist to provide is
            # unchanged. Only the TAIL grows, and the tail is read solely when the verifier is about
            # to give up on an exact beat and settle for a contextual fallback. At that point
            # diversity is worth nothing and finding the actual moment is worth everything: the
            # right file is usually already on the bench, at the wrong second.
            # Drawn from every source that made the bench, a couple of shots each, rather than a
            # deep dig into the leading few. Measured on this job's beats 24 and 82: the file that
            # holds the shot they need ranked 10th, not top-6, so a leading-sources-only rule pulled
            # nothing for exactly the beats it was written for. Breadth is what finds a moment; the
            # per-source depth stays small so the bench does not fill with one file's runtime.
            try:
                _sib_src = max(0, int(_os_alt.environ.get(
                    "VIDLORE_CLIPSTUDIO_SIBLING_SOURCES", "20") or 20))
                _sib_per = max(0, int(_os_alt.environ.get(
                    "VIDLORE_CLIPSTUDIO_SIBLING_SHOTS", "2") or 2))
            except (TypeError, ValueError):
                _sib_src, _sib_per = 20, 2
            _have = {(c.source_id, c.shot_index) for c in alternates} | {_ckey}
            _sib: list = []
            for _lead in alternates[:_sib_src]:
                _sib += sorted(
                    (c for k, c in shot_best.items()
                     if k[0] == _lead.source_id and k not in _have),
                    key=_rank, reverse=True)[:_sib_per]
            _sib.sort(key=_rank, reverse=True)
            alternates += _sib
            _n_sib = len(_sib)
        else:
            alternates = []
            _n_sib = 0

        # CUT-WINDOW FLAG VALIDATION — the rendered cut pads/extends past shot bounds, so the
        # FULL final window must be clean, not just the chosen shot's own samples. Moment-locked
        # beats (exact/quote/character) anchor to the ORIGINAL candidate range — shorten only
        # around that moment; else fall back to the first (already relevance-ranked) alternate
        # whose window validates; else keep the original (a dirty exact scene still beats
        # unrelated footage — verifier/still recovery may replace it later).
        if _wqc_gate and best is not None:
            best, alternates = wqc_arbitrate_selection(
                best, alternates, _by_src_shots, _ps_by_key, cfg, seg,
                stats=_wqc_stats, progress=progress,
                shot_uses=shot_uses, shot_cap=cfg.max_reuse_per_shot,
                face_guard=(face_guard_for(seg, char2actor, all_faces)
                            if _wf_win_gate else None))

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
            # DEEP BENCH for the verifier's last resort. `alternates` is 6 deep out of a ~4000-shot
            # pool, and when none of those 6 is the exact moment the verifier keeps the original and
            # relabels it contextual_fallback. Measured on job 69d80e9dd4_v4: that path took 113 of
            # 268 beats, and those beats score 4.34 on the frame eval against 5.92 for the beats
            # where a replacement WAS found — the worst group in the video. So keep a deeper ranked
            # bench the verifier can reach for before it settles. Ranking-only: nothing else reads
            # it, so a beat the verifier never questions is byte-identical.
            import os as _os_deep
            try:
                _deep_n = int(_os_deep.environ.get(
                    "VIDLORE_CLIPSTUDIO_DEEP_ALTERNATES", "20") or 20)
            except (TypeError, ValueError):
                _deep_n = 20
            # + the sibling shots appended above: they sit past _deep_n by construction, and the
            # whole point of adding them is that the verifier can reach them.
            if _deep_n > cfg.candidates_per_segment:
                sel.deep_alternates = alternates[cfg.candidates_per_segment:_deep_n + _n_sib]
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
                # window-QC every beat_window too — build plays these directly (the chosen cand
                # at wins[0] is already validated; alternates' windows get the same treatment:
                # shorten in place — moment-locked beats only around their own moment — or drop
                # the window when no clean moment-keeping sub-window exists)
                if _wqc_gate and wk != (cand.source_id, cand.shot_index):
                    _wps = _ps_by_key.get(wk)
                    _w_act, _, _ = _validate_cand_window(c, _wps.shot if _wps else None, seg)
                    if _w_act == "rejected":
                        _wqc_stats["windows-dropped"] += 1
                        continue
                wins.append([c.source_id, round(c.in_point, 3), round(c.out_point, 3)])
            sel.beat_windows = wins
            source_uses[ps.sid] = source_uses.get(ps.sid, 0) + 1
            shot_uses[(ps.sid, ps.shot.index)] = shot_uses.get((ps.sid, ps.shot.index), 0) + 1
            # per-window ledger keyed the same way, remembering WHEN (both clocks) so the next beat
            # can price the repeat by real elapsed video, not beat count.
            win_last_t[(ps.sid, ps.shot.index)] = t_now
            win_last_pos[(ps.sid, ps.shot.index)] = seg.index
            recent_sources.append(ps.sid)
            recent.append({"pos": seg.index, "key": (ps.sid, ps.shot.index), "sid": ps.sid,
                           "t0": ps.shot.start, "t1": ps.shot.end,
                           "phash": ps.shot.phash, "embed": ps.embed})
        selections.append(sel)
        if progress and (seg.index % 20 == 0):
            progress(f"match: {seg.index+1}/{len(segments)} conf={sel.confidence}")

    if _wqc_gate and progress and any(_wqc_stats.values()):
        progress(f"window-qc: summary — {_wqc_stats['shortened']} shortened, "
                 f"{_wqc_stats['fallback']} fallback, {_wqc_stats['kept-dirty']} kept-dirty, "
                 f"{_wqc_stats['windows-dropped']} beat-windows dropped")
    proj.selections = selections
    return selections
