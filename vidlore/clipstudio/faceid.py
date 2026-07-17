"""Phase 4 — Face-ID (actor / character recognition).

Uses OpenCV's built-in YuNet (detect + 5-point landmarks) and SFace (128-d recognition embed,
with alignment handled by cv2.alignCrop). Reference faces are built automatically from Wikipedia
page images for the actor roster the analyzer produced; characters map to their actor's face.
Each detected source face is matched to the roster → (identity, cosine score, confident?).
Uncertain matches are returned non-confident so the matcher/ledger can flag them.

No insightface / dlib needed — only cv2 + two small ONNX models in ~/.cache/clipstudio_models.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

MODELS_DIR = Path(os.environ.get("VIDLORE_CLIPSTUDIO_MODELS",
                                 os.path.expanduser("~/.cache/clipstudio_models")))
YUNET = MODELS_DIR / "yunet.onnx"
SFACE = MODELS_DIR / "sface.onnx"
SFACE_COS_THRESH = 0.363          # cv2 SFace "same identity" cosine threshold
MARGIN = 0.06                     # best must beat 2nd-best by this for a confident ID
DET_MAX_SIDE = 1024               # retry detection here when a big still yields no faces (see _detect)


def available() -> bool:
    return YUNET.exists() and SFACE.exists()


class FaceID:
    """Detector + recognizer pair. Construct once and reuse (models load on init)."""

    def __init__(self, det_size=(320, 320), score_thresh=0.7):
        import cv2
        self.cv2 = cv2
        self.det = cv2.FaceDetectorYN_create(str(YUNET), "", det_size, score_thresh, 0.3, 5000)
        self.rec = cv2.FaceRecognizerSF_create(str(SFACE), "")

    def _detect_at(self, img):
        h, w = img.shape[:2]
        self.det.setInputSize((w, h))
        _n, faces = self.det.detect(img)
        return faces if faces is not None else []

    def _detect(self, img):
        """YuNet is a fixed-scale detector: fed a multi-thousand-pixel still it returns NOTHING
        (measured: 3858x4804 → 0 faces, the same photo at ≤1024 → 1). Wiki reference photos are
        routinely that big, so leads silently became unidentifiable — and an unidentifiable lead
        makes every "no CONFIRMED wrong character" gate vacuously true. Detect at native size
        first (so nothing that works today changes), and only on a MISS retry downscaled, mapping
        the boxes + landmarks back to original coordinates so alignCrop still uses full-res pixels."""
        faces = self._detect_at(img)
        if len(faces) or max(img.shape[:2]) <= DET_MAX_SIDE:
            return faces
        h, w = img.shape[:2]
        s = DET_MAX_SIDE / float(max(h, w))
        small = self.cv2.resize(img, (max(1, int(round(w * s))), max(1, int(round(h * s)))),
                                interpolation=self.cv2.INTER_AREA)
        faces = self._detect_at(small)
        if len(faces):
            faces = faces.copy()
            faces[:, :14] /= s          # cols 0-13 are bbox + the 5 landmarks; col 14 is the score
        return faces

    def embeddings(self, img):
        """[(bbox(x,y,w,h), area, embed[np.float32,128 L2-normed]), ...] for every face."""
        import numpy as np
        out = []
        for f in self._detect(img):
            try:
                aligned = self.rec.alignCrop(img, f)
                feat = self.rec.feature(aligned).flatten().astype("float32")
            except Exception:
                continue
            n = float(np.linalg.norm(feat)) + 1e-8
            x, y, w, h = (int(v) for v in f[:4])
            out.append(((x, y, w, h), int(w * h), feat / n))
        return out

    def largest_embedding(self, img):
        es = self.embeddings(img)
        if not es:
            return None
        es.sort(key=lambda e: e[1], reverse=True)
        return es[0][2]


def _cos(a, b) -> float:
    import numpy as np
    return float(np.dot(a, b))


# ---------------------------------------------------------------------------
# Reference set (Wikipedia page images for the roster)
# ---------------------------------------------------------------------------

# Wikimedia REQUIRES a descriptive User-Agent with contact info; the old "ClipStudio/0.1
# (testing)" UA got 429/403 from upload.wikimedia.org (the image CDN), so reference-image
# downloads SILENTLY failed for every actor whose first attempt hit the rate limiter —
# only Charles Dance happened to slip through, leaving Robb/Jaime with NO face reference
# and thus no way for the matcher to tell those characters apart. Compliant UA + retry fix.
_WIKI_UA = ("ClipStudio/1.0 (https://github.com/vidlore/clipstudio; "
            "clipstudio@vidlore.local) python-requests")
_WIKI_HEADERS = {"User-Agent": _WIKI_UA,
                 "Accept": "image/avif,image/webp,image/*,*/*",
                 "Referer": "https://en.wikipedia.org/"}


def _wiki_image_url(name: str) -> str:
    import requests
    try:
        r = requests.get("https://en.wikipedia.org/w/api.php",
                         params={"action": "query", "titles": name, "prop": "pageimages",
                                 "piprop": "original", "format": "json", "redirects": 1},
                         timeout=20, headers={"User-Agent": _WIKI_UA})
        for p in r.json().get("query", {}).get("pages", {}).values():
            u = p.get("original", {}).get("source", "")
            if u:
                return u
    except Exception:
        pass
    return ""


def _wiki_more_image_urls(name: str, limit: int = 6) -> list:
    """Other images on the page (the lead pageimage sometimes has no detectable frontal
    face — e.g. an angled convention photo). Returns candidate jpg/png URLs, portrait-ish
    filenames first, so a face can still be found for the reference."""
    import requests
    try:
        r = requests.get("https://en.wikipedia.org/w/api.php",
                         params={"action": "query", "titles": name, "prop": "images",
                                 "imlimit": "40", "format": "json", "redirects": 1},
                         timeout=20, headers={"User-Agent": _WIKI_UA})
        files = []
        for p in r.json().get("query", {}).get("pages", {}).values():
            for im in (p.get("images") or []):
                t = im.get("title", "")
                if re.search(r"\.(jpg|jpeg|png)$", t, re.I) and not re.search(
                        r"\b(logo|icon|commons|wikimedia|edit|symbol|flag|map|signature)\b", t, re.I):
                    files.append(t)
        # resolve File: titles to URLs
        urls = []
        for i in range(0, min(len(files), 20), 10):
            chunk = files[i:i + 10]
            ri = requests.get("https://en.wikipedia.org/w/api.php",
                              params={"action": "query", "titles": "|".join(chunk),
                                      "prop": "imageinfo", "iiprop": "url", "format": "json"},
                              timeout=20, headers={"User-Agent": _WIKI_UA})
            for p in ri.json().get("query", {}).get("pages", {}).values():
                for ii in (p.get("imageinfo") or []):
                    if ii.get("url"):
                        urls.append(ii["url"])
        return urls[:limit]
    except Exception:
        return []


def _download_image(url: str, dest: Path) -> bool:
    import requests
    import time
    for attempt in range(4):
        try:
            r = requests.get(url, timeout=25, headers=_WIKI_HEADERS)
            if r.status_code == 200 and len(r.content) > 2000:
                dest.write_bytes(r.content)
                return True
            if r.status_code in (429, 503):              # rate-limited — back off and retry
                time.sleep(1.5 * (attempt + 1))
                continue
            return False                                 # 403/404/etc — retrying won't help
        except Exception:
            time.sleep(1.0)
    return False


def build_references(identities: list[dict], cache_dir, faceid: "FaceID", *, progress=None) -> dict:
    """identities: [{name, kind, actor}] → {name: {embed, actor, kind, ref_image}}.
    Characters use their actor's photo. Skips identities with no findable face."""
    import cv2
    refs: dict[str, dict] = {}
    ref_dir = Path(cache_dir) / "face_refs"
    ref_dir.mkdir(parents=True, exist_ok=True)
    for idt in identities:
        name = (idt.get("name") or "").strip()
        actor = (idt.get("actor") or name).strip()
        if not name:
            continue
        slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
        # cache the chosen image next to its slug
        existing = next((p for p in ref_dir.glob(f"{slug}.*")), None)
        emb = None
        if existing is not None:
            img = cv2.imread(str(existing))
            emb = faceid.largest_embedding(img) if img is not None else None
        if emb is None:
            # try the actor's photo first, then the character page as a fallback (a wiki
            # rate-limit or a faceless lead image on one query must not lose the character)
            queries = []
            if idt.get("kind") == "character" and actor:
                queries = [actor, name]
            else:
                queries = [name] + ([actor] if actor and actor != name else [])
            # candidate image URLs: each query's lead image, then OTHER images on the
            # actor's page (lead photo can be faceless) — first one with a detectable face wins
            cand_urls = []
            for query in queries:
                u = _wiki_image_url(query)
                if u:
                    cand_urls.append(u)
            for query in queries:
                cand_urls.extend(_wiki_more_image_urls(query))
            seen_u = set()
            for url in cand_urls:
                if url in seen_u:
                    continue
                seen_u.add(url)
                ext = (Path(url.split("?")[0]).suffix or ".jpg")[:5]
                img_path = ref_dir / f"{slug}{ext}"
                if not _download_image(url, img_path):
                    continue
                img = cv2.imread(str(img_path))
                emb = faceid.largest_embedding(img) if img is not None else None
                if emb is not None:
                    existing = img_path
                    break
        if emb is None:
            if progress:
                progress(f"faceid: ⚠ NO REFERENCE for {name!r} — this character cannot be "
                         f"identified, footage of them may be mismatched")
            continue
        refs[name] = {"embed": emb, "actor": actor, "kind": idt.get("kind", ""),
                      "ref_image": str(existing)}
        if progress:
            progress(f"faceid: reference built · {name}")
    return refs


def match(emb, references: dict, *, thresh: float = SFACE_COS_THRESH) -> tuple[str, float, bool]:
    """(best_name, cosine_score, confident). Confident = above threshold AND clearly beats 2nd."""
    best_name, best, second = "", 0.0, 0.0
    for nm, r in references.items():
        s = _cos(emb, r["embed"])
        if s > best:
            second = best
            best, best_name = s, nm
        elif s > second:
            second = s
    confident = (best >= thresh) and ((best - second) >= MARGIN)
    return best_name, round(best, 4), confident


def identify_faces(img_path, faceid: "FaceID", references: dict) -> list[dict]:
    """For one image: [{bbox, name, score, confident, area}] per detected face (largest first)."""
    import cv2
    img = cv2.imread(str(img_path))
    if img is None:
        return []
    res = []
    for bbox, area, emb in faceid.embeddings(img):
        nm, score, conf = match(emb, references) if references else ("", 0.0, False)
        res.append({"bbox": list(bbox), "area": area,
                    "name": (nm if conf else ""), "best_name": nm,
                    "score": score, "confident": conf})
    res.sort(key=lambda r: r["area"], reverse=True)
    return res
