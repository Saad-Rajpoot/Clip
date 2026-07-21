"""Exact-scene IMAGE fallback — when no YouTube clip fits a beat, fetch the real scene
as a still from the open web (Bing/DDG/Wikimedia) and validate it HARD before use.

The guiding rule (user): a wrong image is worse than a generic clip, so we only accept an
image we can VERIFY is the right scene/person — CLIP relevance to the beat's scene
description AND, when the beat names a character, that character's face actually present
(and no OTHER main character's face dominating). Otherwise we skip and keep the footage.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

from . import web_images as _wi

_STOP = {"the", "a", "an", "of", "and", "or", "in", "on", "at", "to", "is", "was",
         "his", "her", "their", "he", "she", "they", "it", "this", "that", "with",
         "for", "as", "by", "from", "who", "what", "when", "where", "scene", "moment"}


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w' ]", " ", s or "")).strip()


def build_query(seg, analysis) -> str:
    """A specific, image-search-friendly query for this beat: show + the named character +
    the canonical scene description — never the raw narration sentence."""
    movie = _clean(getattr(analysis, "movie_title", "") or "")
    char = ""
    if getattr(seg, "required_kind", "") in ("character", "actor") and getattr(seg, "required_entity", ""):
        char = _clean(seg.required_entity)
    desc = _clean(getattr(seg, "scene_query", "") or getattr(seg, "expected_visual", ""))
    # drop words already in movie/char to avoid a bloated query
    drop = set((movie + " " + char).lower().split()) | _STOP
    desc_words = [w for w in desc.split() if w.lower() not in drop][:8]
    # ERA anchor: for a known-episode video, append the season so a character search leans to the
    # RIGHT era (a bare "Cersei" query returns short-hair S6 stills for an S1 scene). Only added
    # when the description doesn't already name a season.
    era = ""
    ep = (getattr(analysis, "episode_hint", "") or "")
    m = re.search(r"s0*(\d{1,2})\s*[ex]\d", ep.lower()) or re.search(r"season\s*0*(\d{1,2})", ep.lower())
    if m and "season" not in (desc + " " + char).lower():
        era = f"season {int(m.group(1))}"
    parts = [p for p in (movie, char, " ".join(desc_words), era) if p]
    q = " ".join(parts).strip()
    return q or (movie + " scene")


def _aspect(p: Path) -> float:
    try:
        import cv2
        im = cv2.imread(str(p))
        h, w = im.shape[:2]
        return w / max(1.0, h)
    except Exception:
        return 1.0


def _has_center_seam(p: Path) -> bool:
    """Detect a hard collage stitch: a single near-full-length high-edge line in the central
    band — checked on BOTH axes (side-by-side AND stacked/split-screen panels). Conservative
    threshold: a false positive only costs a skip (the beat keeps its footage)."""
    try:
        import cv2
        import numpy as np
        gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            return False
        h, w = gray.shape[:2]
        if min(h, w) < 160:
            return False
        # vertical seam (columns) — side-by-side panels (scan a wide central band so an
        # OFF-CENTER stitch, e.g. unequal panels, is still caught)
        colE = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)).mean(axis=0)
        cband = colE[int(w * 0.34):int(w * 0.66)]
        if cband.size and float(cband.max()) > 7.0 * (float(np.median(colE)) + 1e-3):
            return True
        # horizontal seam (rows) — stacked / split-screen panels
        rowE = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)).mean(axis=1)
        rband = rowE[int(h * 0.34):int(h * 0.66)]
        return rband.size > 0 and float(rband.max()) > 7.0 * (float(np.median(rowE)) + 1e-3)
    except Exception:
        return False


_WATERMARK_RX = re.compile(r"@[A-Za-z0-9_]{3,}|\bunbox|\bcosplay\b|©|\bshutterstock\b|\bgetty\b",
                           re.I)


def _has_watermark_text(img_path: Path) -> bool:
    """Reject fan/cosplay/stock photos carrying a social-handle or watermark caption (e.g.
    '@UnBoxPHD') — those are real-world photos, not show footage."""
    try:
        from . import ocr as _ocr
        if not _ocr.available():
            return False
        return bool(_WATERMARK_RX.search(_ocr.read_text(str(img_path)) or ""))
    except Exception:
        return False


def _vr():
    try:
        from . import index as _index
        if _index.clip_available():
            return _index._vr()
    except Exception:
        pass
    return None


_PHOTO_PROMPTS = ["a photograph from a live-action television show",
                  "a real film still photograph of actors"]
_ART_PROMPTS = ["a digital illustration or concept art", "a video game screenshot",
                "a cartoon or animated drawing", "a poster or wallpaper artwork",
                "a photo of plastic toy figurines or a Playmobil or Lego scene",
                "a photo of collectible action figures or a model diorama",
                "an AI-generated image", "a 3D rendered animation"]


def _photographic_ok(img_path: Path) -> bool:
    """CLIP zero-shot: reject illustrated / concept-art / game-screenshot / cartoon images —
    we want LIVE-ACTION scene stills only. True = looks photographic (or CLIP unavailable)."""
    vr = _vr()
    if vr is None:
        return True
    try:
        import numpy as np
        from PIL import Image
        ie = np.asarray(vr._img_embed(Image.open(str(img_path))), dtype="float32")
        photo = max(float(np.dot(ie, np.asarray(vr._txt_embed(p), "float32")))
                    for p in _PHOTO_PROMPTS)
        art = max(float(np.dot(ie, np.asarray(vr._txt_embed(p), "float32")))
                  for p in _ART_PROMPTS)
        return photo >= art - 0.01            # tie goes to "photo" only if not clearly art
    except Exception:
        return True


def _clip_relevance(img_path: Path, scene_text: str) -> float:
    """0..1 CLIP cosine of the image vs the scene description (mapped from the ~0.15-0.34
    cosine band that separates on-topic from off-topic for this CLIP)."""
    vr = _vr()
    if vr is None or not scene_text.strip():
        return -1.0                                  # unavailable → "unknown", not a vote
    try:
        import numpy as np
        from PIL import Image
        ie = np.asarray(vr._img_embed(Image.open(str(img_path))), dtype="float32")
        te = np.asarray(vr._txt_embed(scene_text), dtype="float32")
        cos = float(np.dot(ie, te))
        return max(0.0, min(1.0, (cos - 0.15) / (0.34 - 0.15)))
    except Exception:
        return -1.0


def _shot_relevance(shot, kf, scene_text: str, embeds_of=None, rel_memo: dict | None = None) -> float:
    """CLIP relevance of a pool SHOT's keyframe vs the beat text — numerically identical to
    _clip_relevance(kf, text), but served from the index's PERSISTED embedding when available.

    The persisted row IS the vector _clip_relevance would recompute: index_source stores
    float32(_img_embed(keyframe)) at index time from the very same keyframe file, and both
    paths apply the same float32 dot + (cos-0.15)/(0.34-0.15) clamp — so a valid row swaps a
    full CLIP image forward pass (~tens of ms × up to scan_cap × candidates × beats) for one
    dot product with zero score change. Falls back to the LIVE _clip_relevance path whenever
    the persisted data is missing, stale, out of bounds, or the model is unavailable — and
    the text-embedding still comes from the live (already process-cached) model, so
    availability semantics are identical: no model → -1.0 exactly as before.

    `rel_memo` (per-render dict) memoizes results by (source_id, shot_index, text): the
    still pass re-scans the same pool head for every candidate round, recomputing identical
    scores. Only REAL scores are memoized — a -1.0 "unavailable" is never cached, so the
    baseline's per-call retry behavior on transient failure is preserved."""
    from . import perf_metrics as _pm
    sid = getattr(shot, "source_id", "") or ""
    key = (sid, getattr(shot, "index", -1), scene_text)
    if rel_memo is not None and key in rel_memo:
        _pm.incr("imgfb.rel.memo_hit")
        return rel_memo[key]
    rel = None
    if embeds_of is not None:
        try:
            row = getattr(shot, "embed_row", -1)
            row = -1 if row is None else int(row)
            if row >= 0 and scene_text.strip():
                vr = _vr()
                if vr is not None:
                    bundle = embeds_of(sid)              # (matrix, manifest row_map) — verified
                    mat, row_map = bundle if isinstance(bundle, tuple) else (None, None)
                    info = (row_map or {}).get(str(row))
                    if (mat is not None and info is not None and row < int(mat.shape[0])
                            and int(info.get("shot", -1)) == int(getattr(shot, "index", -2))
                            and str(info.get("kf", "")) == Path(kf).name
                            and _kf_md5_memo(kf, rel_memo) == str(info.get("kf_md5", ""))):
                        import numpy as np
                        te = np.asarray(vr._txt_embed(scene_text), dtype="float32")
                        cos = float(np.dot(np.asarray(mat[row], dtype="float32"), te))
                        rel = max(0.0, min(1.0, (cos - 0.15) / (0.34 - 0.15)))
                        _pm.incr("imgfb.rel.persisted")
                    elif mat is not None:
                        _pm.incr("imgfb.rel.stale_row")  # certified matrix, uncertified row
        except Exception:
            rel = None                                 # any persisted-path failure → live path
    if rel is None:
        rel = _clip_relevance(Path(kf), scene_text)
        _pm.incr("imgfb.rel.live")
    if rel_memo is not None and rel >= 0.0:
        rel_memo[key] = rel
    return rel


def _kf_md5_memo(kf, rel_memo: dict | None) -> str:
    """md5 of the CURRENT keyframe bytes — the per-row content identity the manifest is
    checked against. Memoized in the caller's per-render dict (a keyframe file does not
    change mid-still-pass), so the hash is paid once per frame, not once per scan."""
    mkey = ("_kfmd5", str(kf))
    if rel_memo is not None and mkey in rel_memo:
        return rel_memo[mkey]
    import hashlib
    try:
        h = hashlib.md5(Path(kf).read_bytes()).hexdigest()
    except Exception:
        h = ""
    if rel_memo is not None:
        rel_memo[mkey] = h
    return h


def _face_verdict(img_path: Path, target_actors: set, all_actors: set,
                  faceid_obj, refs: dict) -> str:
    """'match' (target actor present), 'wrong' (a DIFFERENT main character dominates),
    'unknown' (clear face(s) present but NONE match a real actor — i.e. AI-generated fakes
    or the wrong person), 'none' (no face at all — wide/establishing shot), or 'skip'.
    The 'unknown' verdict is the AI-image catch: an uncanny AI 'Arya'/'Tywin' two-shot has
    detectable faces that don't match Maisie Williams / Charles Dance."""
    if faceid_obj is None or not refs:
        return "skip"
    try:
        from . import faceid as _F
        faces = _F.identify_faces(img_path, faceid_obj, refs)
    except Exception:
        return "skip"
    if not faces:
        return "none"
    conf = [(f.get("name") or "").lower() for f in faces if f.get("confident")]
    if any(c in target_actors for c in conf):
        return "match"
    if conf and conf[0] in all_actors and conf[0] not in target_actors:
        return "wrong"                                # a different real main character
    # face(s) present but none confidently match ANY known actor → likely AI/wrong person.
    # Require a reasonably LARGE face (>=2% of the frame) so small background extras in a real
    # wide shot don't trip this — `area` from faceid is pixels, so normalize by frame size.
    frame_px = 1.0
    try:
        import cv2
        im = cv2.imread(str(img_path))
        frame_px = float(im.shape[0] * im.shape[1]) if im is not None else 1.0
    except Exception:
        pass
    big = [f for f in faces if float(f.get("area", 0) or 0) / max(1.0, frame_px) >= 0.02]
    return "unknown" if big else "none"


def _md5(p: Path) -> str:
    import hashlib
    try:
        return hashlib.md5(Path(p).read_bytes()).hexdigest()
    except Exception:
        return ""


def phash_hamming(a: str, b: str) -> int:
    """Hamming distance between two 16-hex (64-bit) perceptual hashes. 64 = max (totally
    different); <~10 = visually near-identical. Returns 64 on any parse failure (treat as
    different so it never falsely suppresses a candidate)."""
    try:
        return bin(int(a, 16) ^ int(b, 16)).count("1")
    except Exception:
        return 64


def _min_hamming(ph: str, against) -> int:
    if not ph or not against:
        return 64
    return min((phash_hamming(ph, x) for x in against if x), default=64)


# ---------------------------------------------------------------------------
# THEN-AND-NOW support. A "where are they now" / cast video pairs each subject's MOVIE footage
# (the THEN) with how they look TODAY (the NOW). The movie footage can never show the actor in the
# present, so a present-day ("now/today/retired/in his eighties") beat must use a RECENT real photo
# of the ACTOR. This is the one piece the scene-matcher can't supply.
# ---------------------------------------------------------------------------
_NOW_BEAT_RX = re.compile(
    r"\b(now|today|nowadays|these days|currently|recently|present[- ]?day|modern[- ]?day|"
    r"is now|are now|has since|have since|went on to|has become|reinvented|retired|"
    r"grew up|all grown up|stepped away|walked away|years later|decades later|still alive|"
    r"passed away|died in 20\d\d|in (his|her|their) (twenties|thirties|forties|fifties|sixties|"
    r"seventies|eighties|nineties)|at age \d|in 20[12]\d|as of 20[12]\d|"
    r"where (are|is) (they|he|she) now|out of the spotlight|looks (happy|different))\b", re.I)


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (s or "").lower()).strip("_")[:24] or "x"


def actor_in_beat(text: str, analysis, char2actor: dict | None) -> str:
    """The real ACTOR a present-day beat is talking about — found by an actor name or a character
    name appearing in the beat text. '' if none (then we leave the beat to normal footage)."""
    tl = (text or "").lower()
    for a in (getattr(analysis, "actors", None) or []):
        if a and len(a) > 2 and a.lower() in tl:
            return a
    for ch, ac in (char2actor or {}).items():
        if ch and len(ch) > 2 and ch.lower() in tl and ac:
            return ac
    return ""


def fetch_recent_actor_photo(actor: str, dest_dir: Path, *, faceid_obj=None, refs: dict | None = None,
                             seen_hashes: set | None = None, log=None, budget: int = 10) -> Optional[dict]:
    """Fetch a RECENT, present-day photo of `actor` (the 'NOW'). We do NOT require a Face-ID match
    against the movie reference — faces age decades — but we DO require a real photographic portrait
    that CONTAINS a face, with no watermark/collage/AI-source, deduped across beats. Returns
    {path, actor, source, query} or None."""
    refs = refs or {}
    seen_hashes = seen_hashes if seen_hashes is not None else set()
    if not actor:
        return None
    # MERGE a few CLEAN queries. A noisy query like "<actor> 2024 2025 recent photo today" gets
    # hijacked by year-matching junk (broadband/tech pages); "<actor> recent" / "<actor> actor now"
    # return real entertainment-press portraits. Dedupe by image URL, preserve order.
    seen_urls: set = set()
    cands: list = []
    for q in (f"{actor} recent", f"{actor} actor now", f"{actor} latest photo"):
        try:
            for c in _wi.search_images(q, n=16):
                u = c.get("image_url") or ""
                if u and u not in seen_urls:
                    seen_urls.add(u)
                    cands.append(c)
        except Exception:
            pass
    q = f"{actor} recent"
    if not cands:
        if log:
            log(f"now-photo: no candidates for {actor!r}")
        return None
    from . import ocr as _ocr
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    tried = 0
    for c in cands:
        if tried >= budget:
            break
        if _wi.is_ai_generated_source(c.get("source_site", "")):
            continue
        tmp = dest_dir / f"_now_{_slug(actor)}_{tried}.jpg"
        p = _wi.download_image(c["image_url"], tmp)
        if p is None:
            continue
        tried += 1
        ar = _aspect(p)
        if ar > 1.95 or ar < 0.5 or _has_center_seam(p):     # banners / collages
            p.unlink(missing_ok=True); continue
        try:
            if _ocr.available() and (_ocr.has_big_text(p) or _has_watermark_text(p)):
                p.unlink(missing_ok=True); continue          # captioned / watermarked tabloid art
        except Exception:
            pass
        if not _photographic_ok(p):                           # illustration / AI / poster
            p.unlink(missing_ok=True); continue
        h = _md5(p)
        if h and h in seen_hashes:
            p.unlink(missing_ok=True); continue
        # require a face (a real portrait of a person), and don't accept a DIFFERENT known actor.
        verdict = _face_verdict(p, {actor.lower()}, set(), faceid_obj, refs)
        if verdict in ("none", "wrong"):
            p.unlink(missing_ok=True); continue
        final = dest_dir / f"now_{_slug(actor)}.jpg"
        Path(p).replace(final)
        if h:
            seen_hashes.add(h)
        if log:
            log(f"now-photo: ✓ {actor} ← {c.get('source_site', 'web')} (face={verdict})")
        return {"path": final, "actor": actor, "source": c.get("source_site", ""), "query": q}
    if log:
        log(f"now-photo: no usable recent photo for {actor!r} ({tried} tried)")
    return None


def _shot_has_overlay_text(shot) -> bool:
    """A still keyframe carrying on-screen text/names is a logo, title card, reactor-name overlay
    ('AmberReacts', 'Kristian Harloff'), or burned-in subtitle — NOT a clean scene still. Reject it
    from the still pool (defense-in-depth behind the source-title gate, and catches text frames that
    appear inside otherwise-clean uploads: intro logos, lower-thirds, 'GAME OF THRONES' cards)."""
    if getattr(shot, "ocr_names", None):                       # roster/name cards read off the frame
        return True
    t = (getattr(shot, "ocr_text", "") or "").strip()
    if len(t) >= 6:                                             # any real readable overlay text
        return True
    # script-agnostic visual band check — Arabic/Turkish burned subs OCR to nothing readable but
    # must never air as a FROZEN still (they'd sit on screen for seconds)
    from .match import _shot_subtitle_band
    return _shot_subtitle_band(shot)


def pick_source_still(sel, shots_by_key: dict, used_keys: set, used_phash: set,
                      *, distinct_from=None, min_dist: int = 14, min_quality: float = 0.42):
    """Pick a REAL source-footage still (a shot keyframe) for a repeated filler beat from the
    matcher's own ranked `alternates`. The safe, exact alternative to a web image: 100% real
    footage, era-correct (pool already era/purity/wrong-show filtered), zero AI/watermark risk,
    relevance straight from the matcher's ranking.

    Picks the highest-RELEVANCE alternate that is also visually DISTINCT — different shot than the
    assigned moving clip, not already used as a still, and perceptually far (Hamming > `min_dist`)
    from everything already on screen (`distinct_from`). So a filler beat that would re-air the same
    look instead gets a genuinely different, still-relevant composition (often contextual B-roll
    the matcher ranked highly). If nothing distinct exists, returns None → the beat keeps its
    moving footage (no forced slideshow).

    Returns (keyframe_path, source_id, shot_index, score, phash) or None.
    """
    distinct_from = distinct_from or set()
    assigned = (getattr(sel, "source_id", ""), getattr(sel, "shot_index", -1))
    for c in (getattr(sel, "alternates", None) or []):     # already relevance-ranked
        key = (getattr(c, "source_id", ""), getattr(c, "shot_index", -1))
        shot = shots_by_key.get(key)
        if shot is None:
            continue
        kf = getattr(shot, "keyframe_path", "") or ""
        if not kf or not Path(kf).exists():
            continue
        if float(getattr(shot, "quality", 1.0) or 0.0) < min_quality:   # skip blurry/dark frames
            continue
        from .match import _shot_unreadable
        if _shot_unreadable(shot):                 # near-black across the shot (multi-frame luma)
            continue
        if _shot_has_overlay_text(shot):           # logo / title card / reactor-name / burned subtitle
            continue
        if key == assigned or key in used_keys:   # different than the moving clip + still-dedup
            continue
        ph = getattr(shot, "phash", "") or ""
        if ph and ph in used_phash:                # exact near-duplicate already used as a still
            continue
        if ph and _min_hamming(ph, distinct_from) <= min_dist:   # looks like something already aired
            continue
        return (kf, key[0], key[1], float(getattr(c, "score", 0.0) or 0.0), ph)
    return None


def pick_pool_still(seg, shots_by_key: dict, used_keys: set, used_phash: set,
                    *, distinct_from=None, min_rel: float = 0.30, min_quality: float = 0.42,
                    scan_cap: int = 220, want_faces=None, embeds_of=None,
                    rel_memo: dict | None = None):
    """Best RELEVANT downloaded source-video keyframe for a beat that has NO usable selection /
    alternates — CLIP-ranked over the indexed pool (real footage frames ONLY, never web/AI). Bounded
    by `scan_cap` so a no-clip beat can't scan an unbounded pool. `want_faces` (a list of accepted
    NAME token-sets from orchestrate.entity_name_variants) makes the pick FACE-AWARE for a named-
    character beat: a shot whose Face-ID confirms the required person strictly outranks any
    face-less/other shot — without it the picker fed 3 no-face candidates to the still checks and
    every one was (correctly) rejected, leaving the beat for the release gate. Returns
    (kf, sid, shot, rel, phash) or None when nothing in the pool is relevant enough."""
    from . import perf_metrics as _pm_ps
    _pm_ps.incr("imgfb.pool_scan")
    text = _clean(getattr(seg, "scene_query", "") or getattr(seg, "expected_visual", "")
                  or getattr(seg, "text", ""))
    if not text:
        return None
    distinct_from = distinct_from or set()
    variants = [v for v in (want_faces or []) if v]

    def _face_hit(sh):
        ftoks = set(re.findall(r"[a-z0-9]+",
                               " ".join(getattr(sh, "face_ids", []) or []).lower()))
        return any(v <= ftoks for v in variants)

    items = list(shots_by_key.items())
    if variants:
        # face-confirmed shots FIRST in scan order — the scan_cap bounds CLIP calls, and dict
        # order is per-source, so the pool's few confirmed shots of the required character could
        # sit entirely past the cap and never be scanned (observed: 1 surviving confirmed shot,
        # never reached). The cheap Face-ID test is evaluated on every shot; only CLIP is capped.
        items.sort(key=lambda kv: 0 if _face_hit(kv[1]) else 1)
    best = None                                    # (face_hit, rel, kf, sid, sidx, ph)
    scanned = 0
    for key, shot in items:
        if scanned >= scan_cap:
            break
        if key in used_keys:
            continue
        kf = getattr(shot, "keyframe_path", "") or ""
        if not kf or not Path(kf).exists():
            continue
        if float(getattr(shot, "quality", 1.0) or 0.0) < min_quality:
            continue
        from .match import _shot_unreadable
        if _shot_unreadable(shot):                 # near-black across the shot (multi-frame luma)
            continue
        if _shot_has_overlay_text(shot):           # logo / title card / reactor-name / burned subtitle
            continue
        ph = getattr(shot, "phash", "") or ""
        if ph and ph in used_phash:
            continue
        if ph and _min_hamming(ph, distinct_from) <= 14:
            continue
        scanned += 1
        hit = 1 if (variants and _face_hit(shot)) else 0
        rel = _shot_relevance(shot, kf, text, embeds_of, rel_memo)
        if rel >= min_rel and (best is None or (hit, rel) > (best[0], best[1])):
            best = (hit, rel, kf, key[0], key[1], ph)
    return (best[2], best[3], best[4], best[1], best[5]) if best else None


# Web-image SOURCE/CONTEXT guard — reject candidates that are NOT a real in-show scene still:
# red-carpet / premiere / interview / press / behind-the-scenes / blooper / fan-art / cosplay /
# poster / promo-portrait / headshot / awards / editorial-stock / then-and-now pages.
_WEB_REJECT_RX = re.compile(
    r"red.?carpet|premiere|photo.?call|step.?and.?repeat|"
    r"interview|press.?(junket|conference|tour)|junket|panel|"
    r"behind.the.scenes|\bbts\b|making.of|blooper|gag.?reel|on.?set|set photo|"
    r"fan.?art|fanart|deviantart|artstation|cosplay|fan.?edit|fan.?page|"
    r"pinterest|wallpaper|\bposter\b|key.?art|promo(tional)?.?(poster|portrait|still|shot|photo)|"
    r"head.?shot|portrait|net.?worth|then.and.now|where are they now|"
    r"getty|shutterstock|alamy|wireimage|gala|awards?|oscars?|emmys?|golden.?globe|"
    r"comic.?con|convention", re.I)
_GENERIC_TITLE_WORDS = {"the", "of", "and", "a", "an", "movie", "film", "series", "show",
                        "season", "episode", "part", "scene", "tv", "hd"}


def _web_candidate_ok(c: dict, analysis) -> bool:
    """A web image may air ONLY if it is a real still FROM THE TARGET movie/series. Reject off-context
    pages (red-carpet/interview/BTS/fan/cosplay/poster/portrait/stock) and wrong-franchise / sibling-
    show / spinoff images, and require the candidate to actually REFERENCE the target title (token
    overlap) — so a generic image that merely shows the same actor is rejected (final image policy)."""
    blob = " ".join(str(c.get(k, "")) for k in ("title", "source_site", "source_page", "image_url")).lower()
    if _WEB_REJECT_RX.search(blob):
        return False
    mv = (getattr(analysis, "movie_title", "") or "").strip()
    if mv:
        title = c.get("title", "") or ""
        try:                                           # wrong franchise / sibling show / spinoff
            from .discover import _wrong_installment
            if _wrong_installment(mv, title) or _wrong_installment(mv, blob):
                return False
        except Exception:
            pass
        mvtoks = [w for w in re.findall(r"[a-z0-9]+", mv.lower())
                  if len(w) > 2 and w not in _GENERIC_TITLE_WORDS]
        if mvtoks and not any(t in blob for t in mvtoks):
            return False                               # never references the target title → reject
    return True


def fetch_scene_image(seg, analysis, dest_dir: Path, *, faceid_obj=None, refs: dict | None = None,
                      char2actor: dict | None = None, log=None, budget: int = 12,
                      seen_hashes: set | None = None) -> Optional[dict]:
    """Search + validate an exact-scene still for one beat. Returns
    {path, score, query, source, clip, face} or None if nothing verified out.
    `seen_hashes` (shared across beats) prevents the SAME image being used at two beats."""
    refs = refs or {}
    char2actor = char2actor or {}
    seen_hashes = seen_hashes if seen_hashes is not None else set()
    if not _wi._env_flag("VIDLORE_CLIPSTUDIO_IMAGE_FALLBACK", True):
        return None
    q = build_query(seg, analysis)
    scene_text = _clean(getattr(seg, "scene_query", "") or getattr(seg, "expected_visual", "")
                        or getattr(seg, "text", ""))
    # who must be in frame (actor names) when the beat names a person
    target_actors: set = set()
    if getattr(seg, "required_kind", "") in ("character", "actor") and getattr(seg, "required_entity", ""):
        t = seg.required_entity.strip().lower()
        target_actors.add(t)
        if t in char2actor:
            target_actors.add(char2actor[t].lower())
    all_actors = set()
    for c, a in char2actor.items():
        all_actors.add(c.lower())
        all_actors.add(a.lower())

    cands = _wi.search_images(q, n=30)
    if not cands:
        if log:
            log(f"image-fallback: no candidates for {q!r}")
        return None

    from . import ocr as _ocr
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    scored = []
    tried = 0
    for c in cands:
        if tried >= budget:
            break
        # the IMAGE host is already filtered in search_images; here also drop candidates whose
        # SOURCE page is a pure AI generator (shedevrum.ai etc.) — the file may sit on a neutral
        # CDN but it is synthetic art, not a real still.
        if _wi.is_ai_generated_source(c.get("source_site", "")):
            continue
        if not _web_candidate_ok(c, analysis):     # wrong franchise / BTS / red-carpet / portrait / off-title
            continue
        tmp = dest_dir / f"_imgcand_{seg.index:03d}_{tried}.jpg"
        p = _wi.download_image(c["image_url"], tmp)
        if p is None:
            continue
        tried += 1
        # reject wide banners / side-by-side collages (a 2:1 "10 opinions" panel is two
        # images, not one clean scene still) and a vertical center seam (two stitched panels)
        ar = _aspect(p)
        if ar > 1.95 or ar < 0.55 or _has_center_seam(p):
            p.unlink(missing_ok=True)
            continue
        # reject posters / title cards / meme overlays with big burned text, and
        # fan/cosplay/stock photos with a social-handle or watermark caption
        try:
            if _ocr.available() and (_ocr.has_big_text(p) or _has_watermark_text(p)):
                p.unlink(missing_ok=True)
                continue
        except Exception:
            pass
        # reject illustrated / concept-art / game-screenshot / cartoon — live-action only
        if not _photographic_ok(p):
            p.unlink(missing_ok=True)
            continue
        # cross-beat dedup — never reuse the same image at two beats
        _h = _md5(p)
        if _h and _h in seen_hashes:
            p.unlink(missing_ok=True)
            continue
        clip_rel = _clip_relevance(p, scene_text)
        face = _face_verdict(p, target_actors, all_actors, faceid_obj, refs)
        # wrong real character → never; on a CHARACTER beat, faces that match NO real actor
        # ('unknown') are AI fakes or the wrong person (the uncanny AI 'Arya+Tywin' two-shot) →
        # reject too, the beat keeps its footage instead.
        if face == "wrong" or (target_actors and face == "unknown"):
            p.unlink(missing_ok=True)
            continue
        # score: CLIP relevance is the spine; a confirmed target face is a strong boost,
        # a missing face on a person-beat is a real penalty
        base = clip_rel if clip_rel >= 0 else 0.30    # CLIP unavailable → neutral prior
        if target_actors:
            base += {"match": 0.45, "none": -0.12, "unknown": -0.30,
                     "skip": 0.0}.get(face, 0.0)
        scored.append({"path": p, "score": round(base, 4), "query": q, "hash": _h,
                       "source": c.get("source_site", ""), "clip": round(clip_rel, 3),
                       "face": face, "title": c.get("title", "")})

    if not scored:
        if log:
            log(f"image-fallback: 0/{tried} candidates usable for {q!r}")
        return None
    scored.sort(key=lambda d: -d["score"])
    best = scored[0]
    # ACCEPTANCE BAR — strict on purpose. A person-beat needs the face OR very high CLIP;
    # a scene/location/event beat needs solid CLIP relevance.
    floor = float(os.environ.get("VIDLORE_CLIPSTUDIO_IMAGE_MIN_SCORE", "0.42"))
    # CHARACTER-beat (non-exact) web stills must clear a HIGHER bar AND show the CONFIRMED target
    # actor — no generic portrait / random character still slipping in (req. 3).
    from . import policy as _pol
    _exact = _pol.is_exact(seg)
    if not _exact:
        floor = max(floor, float(os.environ.get("VIDLORE_CLIPSTUDIO_IMAGE_MIN_SCORE_CHAR", "0.60")))
    ok = best["score"] >= floor
    if not _exact and best["face"] != "match":
        ok = False                                    # non-exact web still needs the confirmed actor
    if target_actors and best["face"] not in ("match", "skip") and best["clip"] < 0.55:
        ok = False
    # keep only the winner; clean the rest
    for d in scored:
        if d["path"] != best["path"]:
            Path(d["path"]).unlink(missing_ok=True)
    if not ok:
        Path(best["path"]).unlink(missing_ok=True)
        if log:
            log(f"image-fallback: best below bar ({best['score']:.2f}, face={best['face']}, "
                f"clip={best['clip']}) for {q!r} — skipped")
        return None
    final = dest_dir / f"scene_img_{seg.index:03d}.jpg"
    Path(best["path"]).replace(final)
    best["path"] = final
    if best.get("hash"):
        seen_hashes.add(best["hash"])                  # claim it so no later beat reuses it
    if log:
        log(f"image-fallback: ✓ beat {seg.index} ← {best['source']} "
            f"(score {best['score']:.2f}, clip {best['clip']}, face {best['face']}) · {q!r}")
    return best
