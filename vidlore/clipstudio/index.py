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
import subprocess
from pathlib import Path
from typing import Optional

from .models import Shot, SourceVideo, ClipProject, SOURCE_OK
from .config import ClipConfig, ffmpeg_exe
from .ingest import probe

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
    """Attach to each shot the words whose midpoint falls inside it."""
    if not words:
        return
    wi = 0
    for sh in shots:
        toks = []
        for (ws, we, wt) in words:
            mid = (ws + we) / 2.0
            if sh.start <= mid < sh.end:
                toks.append(wt)
        sh.transcript = " ".join(toks).strip()


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
    if shots_file.exists() and not force:
        try:
            have_caps = json.loads(meta_file.read_text(encoding="utf-8"))
        except Exception:
            have_caps = {}
        if not isinstance(have_caps, dict):      # 'null'/list/string parses fine but isn't caps
            have_caps = {}
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

    # 4) persist (atomic tmp+replace — an interrupted write must not brick later runs' resume).
    # embeds first, shots last: shots.json presence gates the cache, and match.py bounds-checks
    # embed_row, so a kill between the two degrades gracefully instead of pairing a valid cache
    # with a truncated matrix.
    if embeds:
        _etmp = proj.embeds_path(source.id).with_suffix(".tmp.npy")
        np.save(_etmp, np.vstack(embeds))
        _etmp.replace(proj.embeds_path(source.id))
    _tmp = shots_file.with_suffix(".json.tmp")
    _tmp.write_text(json.dumps([s.to_dict() for s in shots], indent=1), encoding="utf-8")
    _tmp.replace(shots_file)
    _mtmp = meta_file.with_suffix(".json.tmp")
    _mtmp.write_text(json.dumps({"faceid": do_faceid, "ocr": do_ocr,
                                 "roster": bool(roster)}), encoding="utf-8")
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
