"""Stage 2 — index.

For each source: detect shots, transcribe (word-level), extract a keyframe per shot, embed it
with the engine's CLIP, and record face presence. Output per source:
  index/<sid>.shots.json   — Shot[]  (in/out, transcript, keyframe ref, face/embed refs)
  index/<sid>.embeds.npy   — float32 [n_shots, dim] CLIP keyframe embeddings
  index/<sid>/keyframes/   — shot_NNN.jpg

Reuses the engine's local ONNX CLIP (vidlore.visual_relevance) — no torch, no API. Heavy
imports (faster_whisper, scenedetect, numpy, PIL) are done lazily so importing this module is
cheap. Re-indexing is skipped when shots.json already exists (resume), unless force=True.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

from .models import Shot, SourceVideo, ClipProject, SOURCE_OK
from .config import ClipConfig, ffmpeg_exe
from .ingest import probe

# Index schema. Bump to invalidate caches whose SHAPE changed (not merely their capabilities).
#   1 → shots + per-shot transcript only
#   2 → + <sid>.words.json (word-level ASR start/end/text) for quote-span location
INDEX_SCHEMA = 2

_WHISPER = {}   # cache: model_name -> WhisperModel


# ---------------------------------------------------------------------------
# CLIP bridge
# ---------------------------------------------------------------------------

def clip_available() -> bool:
    try:
        from vidlore import visual_relevance as vr
        return bool(vr.available())
    except Exception:
        return False


def _vr():
    from vidlore import visual_relevance as vr
    return vr


# ---------------------------------------------------------------------------
# Shot detection
# ---------------------------------------------------------------------------

def detect_shots(path: Path, cfg: ClipConfig, total_dur: float = 0.0) -> list[tuple[float, float]]:
    """Content-aware shot boundaries → [(start,end)]. Falls back to fixed chunks on failure."""
    scenes: list[tuple[float, float]] = []
    try:
        from scenedetect import open_video, SceneManager, ContentDetector
        video = open_video(str(path))
        sm = SceneManager()
        sm.add_detector(ContentDetector(threshold=cfg.scene_threshold))
        sm.detect_scenes(video, show_progress=False)
        for s, e in sm.get_scene_list():
            scenes.append((float(s.get_seconds()), float(e.get_seconds())))
    except Exception:
        scenes = []
    if not scenes:
        # fallback: uniform chunks of ~2*target so matching still has candidates
        dur = total_dur or probe(path).get("duration", 0.0)
        step = max(2.0, cfg.target_clip_sec * 2)
        t = 0.0
        while t < dur:
            scenes.append((t, min(dur, t + step)))
            t += step
    # floor at min_clip_sec too: a shot shorter than the minimum cuttable clip would force
    # cut_selection to extend past the shot boundary into the NEXT, unrelated shot
    return _merge_short(scenes, max(cfg.min_shot_sec, cfg.min_clip_sec))


def _merge_short(scenes: list[tuple[float, float]], min_sec: float) -> list[tuple[float, float]]:
    """Merge sub-min_sec shots into the previous shot so every shot is usable."""
    if not scenes:
        return scenes
    out = [list(scenes[0])]
    for s, e in scenes[1:]:
        if (e - s) < min_sec:
            out[-1][1] = e                       # absorb into previous
        else:
            out.append([s, e])
    # ensure the very first shot meets min length too
    if len(out) > 1 and (out[0][1] - out[0][0]) < min_sec:
        out[1][0] = out[0][0]
        out.pop(0)
    return [(a, b) for a, b in out]


# ---------------------------------------------------------------------------
# Transcription (word-level)
# ---------------------------------------------------------------------------

def _whisper(cfg: ClipConfig):
    key = cfg.whisper_model
    if key not in _WHISPER:
        from faster_whisper import WhisperModel
        # cpu_threads: 0 = ctranslate2 default; a positive value pins the ASR pass to that many
        # cores (auto-scaled from the machine in config) so a powerful box transcribes much faster.
        threads = int(getattr(cfg, "whisper_cpu_threads", 0) or 0)
        _WHISPER[key] = WhisperModel(cfg.whisper_model, device="cpu",
                                     compute_type=cfg.whisper_compute,
                                     cpu_threads=max(0, threads))
    return _WHISPER[key]


def transcribe_words(path: Path, cfg: ClipConfig) -> list[tuple[float, float, str]]:
    """Return [(start,end,word)] for the whole source. Empty list if ASR unavailable."""
    try:
        model = _whisper(cfg)
        segments, _info = model.transcribe(str(path), word_timestamps=True, vad_filter=True)
        words: list[tuple[float, float, str]] = []
        for seg in segments:
            for w in (seg.words or []):
                words.append((float(w.start), float(w.end), w.word.strip()))
        return words
    except Exception:
        return []


def _assign_transcript(shots: list[Shot], words: list[tuple[float, float, str]]) -> None:
    """Attach to each shot the words whose midpoint falls inside it.

    NOTE: this bins by midpoint, so a single spoken SENTENCE routinely splits across two shots
    whenever the cut lands mid-line. That is correct for "what is said during this shot" but makes
    per-shot transcripts useless for locating a QUOTE — see `find_quote_span`, which matches the
    word stream instead. The (ws, we) timings are preserved separately by `save_words`."""
    if not words:
        return
    for sh in shots:
        toks = []
        for (ws, we, wt) in words:
            mid = (ws + we) / 2.0
            if sh.start <= mid < sh.end:
                toks.append(wt)
        sh.transcript = " ".join(toks).strip()


# --- word-level ASR: persistence + quote location -------------------------------------------
# The shipped render proved why this must survive indexing. `transcribe_words` already produced
# word timings, then _assign_transcript threw the timecodes away and kept only per-shot strings.
# Two measured consequences in tywin_lannister_dismis_f4c81b75:
#   * "Any man who must say I am the king is no true king" straddles a cut, so it lands as
#     "...Any man who must" + "say I am the king is no true king." — no single shot contains it,
#     so a per-shot substring search can never find the most iconic line in the video.
#   * The breakout in-point could only be a SHOT boundary (130.05s), 0.41s after that line ends
#     (129.64s) — and the window only extends forward, so the line was unreachable by construction.
def _norm_tok(t: str) -> str:
    return re.sub(r"[^a-z0-9']", "", (t or "").lower())


def save_words(proj: ClipProject, source_id: str, words: list[tuple[float, float, str]]) -> None:
    f = proj.index_dir / f"{source_id}.words.json"
    tmp = f.with_suffix(".json.tmp")
    tmp.write_text(json.dumps([[round(float(a), 3), round(float(b), 3), str(c)] for a, b, c in words]),
                   encoding="utf-8")
    tmp.replace(f)


def load_words(proj: ClipProject, source_id: str) -> list[tuple[float, float, str]]:
    f = proj.index_dir / f"{source_id}.words.json"
    if not f.exists():
        return []
    try:
        return [(float(a), float(b), str(c)) for a, b, c in
                json.loads(f.read_text(encoding="utf-8"))]
    except Exception:
        return []


def _tok_close(a: str, b: str, thresh: float = 0.8) -> bool:
    """Two tokens are 'the same word' if ASR merely garbled it. Never used to accept a match on its
    own — only ever as one term inside the PHRASE score in `find_quote_span`."""
    if a == b:
        return True
    if not a or not b or abs(len(a) - len(b)) > 3:
        return False
    return SequenceMatcher(None, a, b).ratio() >= thresh


def find_quote_span(words, quote: str, *, min_ratio: float = 0.72,
                    window_slack: int = 3):
    """Locate `quote` in a source's word stream. -> (start_s, end_s, ratio) or None.

    Aligns the quote against a sliding window of the word stream and scores the WHOLE PHRASE. A
    single garbled word cannot carry a match (that would let "sleep" alone anchor anywhere), and a
    single garbled word cannot break one either — which is what the measured data demands:

        source ASR : "Maester. Perhaps" | "a messence of nightshade to help him" | "sleep."
        script quote: "Perhaps some essence of nightshade to help him sleep."

    "messence"/"essence" and "a"/"some" differ, yet the phrase is unmistakably the same line. Shot
    boundaries are irrelevant here by construction — the stream is continuous."""
    qt = [t for t in (_norm_tok(w) for w in re.findall(r"[\w']+", quote or "")) if t]
    if len(qt) < 2 or not words:
        return None
    st = [(_norm_tok(w[2]), w[0], w[1]) for w in words]
    st = [x for x in st if x[0]]
    if not st:
        return None
    n = len(qt)
    best = None
    for i in range(len(st)):
        if not _tok_close(st[i][0], qt[0]) and not any(_tok_close(st[i][0], q) for q in qt[:2]):
            continue                      # cheap anchor: only start where the head plausibly begins
        for L in range(max(2, n - window_slack), min(len(st) - i, n + window_slack) + 1):
            win = [x[0] for x in st[i:i + L]]
            m = SequenceMatcher(None, win, qt)
            hits = sum(bl.size for bl in m.get_matching_blocks())
            # credit near-miss tokens the block matcher rejected (ASR garble), positionally
            fuzzy = sum(1 for a, b in zip(win, qt) if a != b and _tok_close(a, b))
            ratio = (2.0 * (hits + fuzzy)) / float(L + n)
            if ratio >= min_ratio and (best is None or ratio > best[2]):
                best = (st[i][1], st[i + L - 1][2], round(min(1.0, ratio), 3))
    return best


# ---------------------------------------------------------------------------
# Keyframes
# ---------------------------------------------------------------------------

def extract_keyframe(path: Path, t: float, dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [ffmpeg_exe(), "-y", "-ss", f"{max(0.0, t):.3f}", "-i", str(path),
           "-frames:v", "1", "-q:v", "3", "-vf", "scale=512:-2", str(dest)]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=60)
        return dest.exists() and dest.stat().st_size > 0
    except Exception:
        return False


# ---------------------------------------------------------------------------
# MULTI-FRAME shot flags (start/mid/end samples) — the single keyframe misses
# INTERMITTENT defects: burned subs that appear mid-shot (observed: a Turkish
# "Gel, geçip odamda konuşalım." aired although the shot's keyframe was clean),
# fading channel bugs, and shots that are readable at the keyframe instant but
# near-black for most of their span. Computed once at index time, persisted on
# the Shot, and reused by every gate (match/build/stills/breakouts) without
# re-running detection.
# ---------------------------------------------------------------------------

# corner-bug mask geometry — MUST stay in sync with match._source_corner_logo (v3, calibrated):
# analysis at 640×360, 4 corner regions of 44×116 px, edge threshold 18, presence >2% of pixels.
_CORNER_REGIONS = {"tl": (slice(0, 44), slice(0, 116)), "tr": (slice(0, 44), slice(524, 640)),
                   "bl": (slice(316, 360), slice(0, 116)), "br": (slice(316, 360), slice(524, 640))}
_MASK_GRID = (11, 29)          # 44×116 region → 4×4 blocks


def _mask_to_hex(mask) -> str:
    import numpy as np
    bits = np.packbits(mask.astype("uint8").flatten())
    return bits.tobytes().hex()


def _mask_from_hex(h: str):
    import numpy as np
    try:
        bits = np.unpackbits(np.frombuffer(bytes.fromhex(h), dtype="uint8"))
        n = _MASK_GRID[0] * _MASK_GRID[1]
        if len(bits) < n:
            return None
        return bits[:n].reshape(_MASK_GRID).astype(bool)
    except Exception:
        return None


def _flags_from_frames(frames: list) -> dict:
    """Pure flag computation from up to 3 grayscale 640×360 float arrays (start/mid/end).
    Returns {subs_flag, text_conf, luma_avg, luma_hi, corner_masks}. Deterministic and
    frame-source-agnostic so it is unit-testable without video files."""
    import numpy as np
    out = {"subs_flag": 0, "text_conf": 0.0, "luma_avg": -1.0, "luma_hi": -1.0,
           "corner_masks": {}}
    frames = [f for f in frames if f is not None]
    if not frames:
        out["subs_flag"] = -1
        return out
    lumas, hi_vals, confs = [], [], []
    corner_or = {k: None for k in _CORNER_REGIONS}
    subs_any = False
    for fr in frames:
        lumas.append(float(fr.mean()))
        hi_vals.append(float(np.percentile(fr, 99.8)))
        # subtitle band on the 320×180 downsample — SAME calibrated geometry/thresholds as
        # match._shot_subtitle_band (Turkish/Arabic/English positives, 30+ clean negatives)
        small = fr[::2, ::2]
        gy, gx = np.gradient(small)
        E = np.hypot(gx, gy)
        band = E[137:175, 38:282] > 42.0
        bf = float(band.mean())
        mf = float((E[72:126, 38:282] > 42.0).mean())
        ys, xs = np.nonzero(band)
        colcov = len(np.unique(xs // 8)) / (244 // 8) if len(xs) else 0.0
        rowspread = ys.std() if len(ys) > 20 else 99.0
        confs.append(bf / max(mf, 0.008))
        if bf > 0.05 and bf > 2.0 * max(mf, 0.008) and colcov >= 0.28 and rowspread < 11.0:
            subs_any = True
        elif len(ys) >= 80 and mf <= 0.004 and rowspread < 8.0:
            # SHORT-LINE branch (calibrated on a leaked 2-word Turkish sub over a dark scene,
            # 0 FPs on 180 clean frames): a sparse but WIDE, row-tight stroke box in an
            # otherwise-empty band — too few edge pixels for the density path above.
            _h = ys.max() - ys.min() + 1
            _w = xs.max() - xs.min() + 1
            if _w >= 150 and _h <= 34:
                subs_any = True
        # corner edge masks at full 640×360 (same threshold as the source-level detector)
        gy2, gx2 = np.gradient(fr)
        E2 = np.hypot(gx2, gy2)
        for name, (rs, cs) in _CORNER_REGIONS.items():
            m = (E2[rs, cs] > 18.0)
            grid = m.reshape(_MASK_GRID[0], 4, _MASK_GRID[1], 4).mean(axis=(1, 3)) > 0.25
            corner_or[name] = grid if corner_or[name] is None else (corner_or[name] | grid)
    out["subs_flag"] = 1 if subs_any else 0
    out["text_conf"] = round(max(confs), 2) if confs else 0.0
    out["luma_avg"] = round(float(np.mean(lumas)), 1)
    out["luma_hi"] = round(float(max(hi_vals)), 1)
    for name, g in corner_or.items():
        if g is not None and g.sum() >= 2:            # store only corners with real edge content
            out["corner_masks"][name] = _mask_to_hex(g)
    return out


def _sample_times(start: float, end: float) -> list:
    """Sample timestamps for one shot's flag pass. SHORT shots (<2s) get FIVE spread samples —
    they are exactly where a brief burned sub / bug flash slips between 3 samples (observed: a
    rare Turkish sub inside a 1.4s shot missed by 3 samples), and where the render pads the cut
    past the shot boundary, so their verdict must be the most reliable. Longer shots keep 3.
    Never returns a single mid-frame-only sample."""
    d = max(0.0, end - start)
    fr = (0.1, 0.3, 0.5, 0.7, 0.9) if d < 2.0 else (0.15, 0.5, 0.85)
    return [start + f * d for f in fr]


def _band_ocr_hit(pil_frame) -> bool:
    """OCR the SUBTITLE BAND of one full-colour sample frame — the definitive catch for THIN
    Latin subs the edge-density heuristic under-detects at 320×180 (observed: a thin yellow
    'poison your three.' slipped both the density and short-line branches). ≥2 readable words
    of ≥3 letters in the band = burned text. Non-Latin scripts stay covered by the visual
    heuristic; OCR-unavailable environments simply skip this (fail-open)."""
    try:
        from . import ocr as _ocr
        if not _ocr.available():
            return False
        import re as _re
        import tempfile
        import os as _os
        w, h = pil_frame.size
        band = pil_frame.crop((int(w * 0.10), int(h * 0.74), int(w * 0.90), int(h * 0.99)))
        band = band.resize((band.width * 2, band.height * 2))
        fd, tmp = tempfile.mkstemp(suffix=".jpg")
        _os.close(fd)
        try:
            band.save(tmp, quality=88)
            txt = _ocr.read_text(tmp) or ""
        finally:
            try:
                _os.remove(tmp)
            except OSError:
                pass
        return len(_re.findall(r"[A-Za-zÀ-ÿĀ-žğışöüçÇĞİŞÖÜ]{3,}", txt)) >= 2
    except Exception:
        return False


def compute_shot_flags(path, shots: list, *, progress=None) -> int:
    """Decode 3-5 frames per shot (start..end) via PyAV and persist multi-frame flags on each
    Shot. Returns the number of shots flagged. Tolerant: a source that can't be decoded leaves
    the sentinel values (-1) in place and every gate falls back to keyframe heuristics."""
    import numpy as np
    try:
        import av
    except Exception:
        return 0
    done = 0
    try:
        c = av.open(str(path))
    except Exception:
        return 0
    try:
        for k, sh in enumerate(shots):
            ts = _sample_times(sh.start, sh.end)
            frames = []
            pil_mid = None
            for j, t in enumerate(ts):
                try:
                    c.seek(int(t * 1e6))
                    got = None
                    for fr in c.decode(video=0):
                        got = fr
                        if float(fr.time or 0.0) >= t - 0.05:
                            break
                    if got is not None:
                        pim = got.to_image()
                        if j == len(ts) // 2:
                            pil_mid = pim
                        im = pim.convert("L").resize((640, 360))
                        frames.append(np.asarray(im, dtype="float32"))
                except Exception:
                    continue
            flags = _flags_from_frames(frames)
            # band OCR on the mid sample — catches thin Latin subs the edge heuristic misses.
            # Only consulted when the visual pass reads clean AND the shot has dialogue (burned
            # subs track speech), keeping the OCR cost to the shots that can actually leak.
            if flags["subs_flag"] == 0 and pil_mid is not None \
                    and (getattr(sh, "transcript", "") or "").strip() \
                    and _band_ocr_hit(pil_mid):
                flags["subs_flag"] = 1
                flags["text_conf"] = max(flags["text_conf"], 9.9)
            sh.subs_flag = flags["subs_flag"]
            sh.text_conf = flags["text_conf"]
            sh.luma_avg = flags["luma_avg"]
            sh.luma_hi = flags["luma_hi"]
            sh.corner_masks = flags["corner_masks"]
            if flags["subs_flag"] >= 0:
                done += 1
            if progress and (k % 50 == 0) and k:
                progress(f"index: multi-frame flags {k}/{len(shots)}")
    finally:
        try:
            c.close()
        except Exception:
            pass
    return done


# ---------------------------------------------------------------------------
# Shot quality + perceptual hash + duplicate-scene detection
# ---------------------------------------------------------------------------

def _ahash(img_bgr) -> str:
    import cv2
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (8, 8), interpolation=cv2.INTER_AREA)
    mean = g.mean()
    val = 0
    for b in (g > mean).flatten():
        val = (val << 1) | int(b)
    return f"{val:016x}"


def _hamming(h1: str, h2: str) -> int:
    if not h1 or not h2:
        return 64
    try:
        return bin(int(h1, 16) ^ int(h2, 16)).count("1")
    except ValueError:
        return 64


def _quality(img_bgr, source_height: int = 0) -> float:
    """0..1 shot quality from sharpness (Laplacian var) + brightness + resolution."""
    import cv2
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(g, cv2.CV_64F).var()
    sharp = max(0.0, min(1.0, (lap - 25.0) / 155.0))
    m = float(g.mean()) / 255.0
    bright = 1.0 - min(1.0, abs(m - 0.5) / 0.5) * 0.6
    h = source_height or img_bgr.shape[0]
    res = 1.0 if h >= 720 else (0.8 if h >= 480 else (0.6 if h >= 360 else 0.4))
    return round(max(0.05, 0.5 * sharp + 0.2 * bright + 0.3 * res), 3)


def _dedup_shots(shots: list[Shot], hamming_thresh: int) -> None:
    """Mark near-identical keyframes (within a source) → sh.dup_of = representative index."""
    reps: list[Shot] = []
    for sh in shots:
        if not sh.phash:
            reps.append(sh)
            continue
        dup = next((k for k in reps if _hamming(sh.phash, k.phash) <= hamming_thresh), None)
        if dup is not None:
            sh.dup_of = dup.index
        else:
            reps.append(sh)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def index_source(proj: ClipProject, source: SourceVideo, cfg: ClipConfig,
                 *, references=None, faceid=None, roster=None,
                 force: bool = False, progress=None) -> list[Shot]:
    import numpy as np
    from PIL import Image

    def log(m):
        if progress:
            progress(m)

    shots_file = proj.shots_path(source.id)
    meta_file = proj.index_dir / f"{source.id}.index.meta.json"
    # capabilities this call wants — a cache built WITHOUT them (e.g. by the manual pipeline,
    # roster-less) must not satisfy an auto-mode call that needs Face-ID/OCR signals. "Wanted"
    # includes availability, else a missing OCR lib would force a futile re-index every run.
    want_caps = {"faceid": bool(faceid and references), "ocr": False}
    if cfg.detect_ocr:
        from . import ocr as _ocr_probe
        want_caps["ocr"] = _ocr_probe.available()
    # roster gives ocr_names (name-card corroboration) — a roster-less OCR cache must not
    # satisfy a roster-bearing auto run; only meaningful when OCR itself can run
    want_caps["roster"] = bool(roster) and want_caps["ocr"]
    # word-level ASR timings (INDEX_SCHEMA 2). A pre-schema cache has shots but no <sid>.words.json,
    # so quote location would silently fall back to per-shot substring search — the exact failure
    # that lost "…is no true king". Treat it as a missing capability so those sources re-index.
    want_caps["words"] = True
    if shots_file.exists() and not force:
        try:
            have_caps = json.loads(meta_file.read_text(encoding="utf-8"))
        except Exception:
            have_caps = {}
        if not isinstance(have_caps, dict):      # 'null'/list/string parses fine but isn't caps
            have_caps = {}
        if int(have_caps.get("schema", 1) or 1) < INDEX_SCHEMA:
            have_caps["words"] = False           # schema bump invalidates the word cache
        if want_caps["words"] and not (proj.index_dir / f"{source.id}.words.json").exists():
            have_caps["words"] = False           # meta may claim words while the file is gone
        missing = [k for k, v in want_caps.items() if v and not have_caps.get(k)]
        if missing:
            log(f"index: {source.id} re-indexing (cache lacks {'+'.join(missing)})")
        else:
            try:
                data = json.loads(shots_file.read_text(encoding="utf-8"))
                log(f"index: {source.id} cached ({len(data)} shots)")
                return [Shot.from_dict(d) for d in data]
            except Exception:
                log(f"index: ⚠ {source.id} cache unreadable — re-indexing")

    if source.status != SOURCE_OK or not source.local_path:
        log(f"index: {source.id} skipped (status={source.status})")
        return []

    path = Path(source.local_path)
    proj.index_dir.mkdir(parents=True, exist_ok=True)
    kf_dir = proj.keyframes_dir(source.id)
    kf_dir.mkdir(parents=True, exist_ok=True)

    # 1) shots
    bounds = detect_shots(path, cfg, total_dur=source.duration)
    if not bounds:
        # zero shots = unreadable/zero-duration media; do NOT cache, so a later run retries.
        # Reaching here WITH an existing cache means a forced/capability re-index — drop the
        # stale files too, or match's load_shots silently serves the old index of media that
        # can no longer be read.
        for _stale in (shots_file, meta_file):
            try:
                _stale.unlink(missing_ok=True)
            except OSError:
                pass
        log(f"index: ⚠ {source.id} produced 0 shots (unreadable media?) — skipped, not cached")
        return []
    log(f"index: {source.id} {len(bounds)} shots detected")

    # 2) transcript
    words = transcribe_words(path, cfg)
    log(f"index: {source.id} transcript words={len(words)}")

    # 3) keyframes + CLIP embeds + faces + Face-ID + OCR + quality + phash
    import cv2
    use_clip = clip_available()
    vr = _vr() if use_clip else None
    do_faceid = bool(faceid and references)
    # OCR runs whenever enabled — the junk/watermark gates in match.py need ocr_text on EVERY
    # pipeline (the roster only adds name-card corroboration on top)
    do_ocr = bool(cfg.detect_ocr)
    _ocr = None
    if do_ocr:
        from . import ocr as _ocr
        do_ocr = _ocr.available()
    _fid = None
    if do_faceid:
        from . import faceid as _fid
    shots: list[Shot] = []
    embeds = []
    for i, (s, e) in enumerate(bounds):
        sh = Shot(source_id=source.id, index=i, start=round(s, 3), end=round(e, 3))
        kf = kf_dir / f"shot_{i:04d}.jpg"
        if extract_keyframe(path, (s + e) / 2.0, kf):
            sh.keyframe_path = str(kf)
            try:
                im = Image.open(kf)
                if use_clip:
                    v = vr._img_embed(im)
                    sh.embed_row = len(embeds)
                    embeds.append(np.asarray(v, dtype="float32"))
                if cfg.detect_faces and use_clip:
                    frac = float(vr._face_frac(im))
                    sh.scores["face_frac"] = round(frac, 4)
                    sh.faces = 1 if frac > 0.02 else 0
            except Exception:
                pass
            img_bgr = cv2.imread(str(kf))
            if img_bgr is not None:
                sh.phash = _ahash(img_bgr)
                sh.quality = _quality(img_bgr, source.height)
                if do_faceid:
                    try:
                        ids = _fid.identify_faces(kf, faceid, references)
                        sh.identities = ids[:4]
                        sh.face_ids = [d["name"] for d in ids if d.get("name")]
                        if ids:
                            sh.faces = max(sh.faces, len(ids))
                    except Exception:
                        pass
                if do_ocr:
                    try:
                        txt = _ocr.read_text(kf)
                        sh.ocr_text = txt[:200]
                        sh.ocr_names = _ocr.names_in_text(txt, roster)
                    except Exception:
                        pass
        shots.append(sh)
        if progress and (i % 25 == 0):
            log(f"index: {source.id} keyframed {i+1}/{len(bounds)}")

    _assign_transcript(shots, words)
    _dedup_shots(shots, cfg.dup_hamming)

    # 3b) MULTI-FRAME flags (start/mid/end sample per shot) — catches intermittent burned subs /
    # corner bugs / mid-shot darkness the single keyframe misses. Persisted on the Shot so every
    # downstream gate reuses them without re-decoding. env VIDLORE_CLIPSTUDIO_MULTIFRAME_FLAGS=0
    # disables (gates then fall back to keyframe heuristics).
    if os.environ.get("VIDLORE_CLIPSTUDIO_MULTIFRAME_FLAGS", "1").strip() \
            not in ("0", "false", "no"):
        try:
            _nf = compute_shot_flags(path, shots, progress=(log if progress else None))
            log(f"index: {source.id} multi-frame flags on {_nf}/{len(shots)} shots")
        except Exception as _e:
            log(f"index: {source.id} multi-frame flags skipped ({type(_e).__name__})")

    # 4) persist (atomic tmp+replace — an interrupted write must not brick later runs' resume).
    # embeds first, shots last: shots.json presence gates the cache, and match.py bounds-checks
    # embed_row, so a kill between the two degrades gracefully instead of pairing a valid cache
    # with a truncated matrix.
    if embeds:
        _etmp = proj.embeds_path(source.id).with_suffix(".tmp.npy")
        np.save(_etmp, np.vstack(embeds))
        _etmp.replace(proj.embeds_path(source.id))
    # words BEFORE shots: shots.json presence gates the cache, so writing words first means a kill
    # between the two re-indexes (safe) rather than serving shots with no word stream (silent
    # degradation back to per-shot quote search).
    save_words(proj, source.id, words)
    _tmp = shots_file.with_suffix(".json.tmp")
    _tmp.write_text(json.dumps([s.to_dict() for s in shots], indent=1), encoding="utf-8")
    _tmp.replace(shots_file)
    _mtmp = meta_file.with_suffix(".json.tmp")
    # "words" records that the word-level PASS RAN, never that it found anything — a genuinely
    # silent source yields [] and must still be cacheable, or it re-indexes on every run forever.
    _mtmp.write_text(json.dumps({"faceid": do_faceid, "ocr": do_ocr,
                                 "roster": bool(roster), "words": True,
                                 "schema": INDEX_SCHEMA}), encoding="utf-8")
    _mtmp.replace(meta_file)
    log(f"index: {source.id} done — {len(shots)} shots, {len(embeds)} embeds, clip={use_clip}")
    return shots


def index_all(proj: ClipProject, cfg: ClipConfig, *, references=None, faceid=None, roster=None,
              force: bool = False, progress=None) -> dict[str, list[Shot]]:
    out: dict[str, list[Shot]] = {}
    for src in proj.sources:
        if src.status != SOURCE_OK:
            continue
        out[src.id] = index_source(proj, src, cfg, references=references, faceid=faceid,
                                   roster=roster, force=force, progress=progress)
    return out


def load_shots(proj: ClipProject, source_id: str) -> list[Shot]:
    f = proj.shots_path(source_id)
    if not f.exists():
        return []
    return [Shot.from_dict(d) for d in json.loads(f.read_text(encoding="utf-8"))]


def load_embeds(proj: ClipProject, source_id: str):
    import numpy as np
    f = proj.embeds_path(source_id)
    return np.load(f) if f.exists() else None
