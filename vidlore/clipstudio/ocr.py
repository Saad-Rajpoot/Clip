"""Phase 4 — OCR (on-screen text / name-card reader).

Reads burned-in text from keyframes (actor name cards, era/date cards, titles) to corroborate
Face-ID and add a strong identity signal — the sample's "RICHARD JORDAN" / "FARRAH FAWCETT"
lower-thirds are exactly this. Uses rapidocr-onnxruntime from the isolated clone-local lib dir
(onnxruntime models, no torch). Degrades to "" if OCR isn't installed.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

# isolated install target (keeps the engine venv pristine)
_LIBS = Path(__file__).resolve().parents[2] / ".clipstudio_libs"
if _LIBS.exists() and str(_LIBS) not in sys.path:
    sys.path.append(str(_LIBS))

_engine = None


def available() -> bool:
    try:
        import rapidocr_onnxruntime  # noqa: F401
        return True
    except Exception:
        return False


def _get_engine():
    global _engine
    if _engine is None:
        from rapidocr_onnxruntime import RapidOCR
        _engine = RapidOCR()
    return _engine


def read_text(img_path) -> str:
    """All readable text in the image as one string ('' if OCR unavailable / nothing found)."""
    if not available():
        return ""
    try:
        result, _elapse = _get_engine()(str(img_path))
        if not result:
            return ""
        return " ".join(str(line[1]) for line in result).strip()
    except Exception:
        return ""


def has_big_text(img_path, min_h_frac: float = 0.065) -> bool:
    """True when the frame carries a LARGE text overlay (a burned caption / title card /
    animated word) — any OCR box taller than `min_h_frac` of the frame height with 2+
    letters. Small corner logos/bugs ('max', 'HBO') sit well under the size cut."""
    if not available():
        return False
    try:
        import cv2
        img = cv2.imread(str(img_path))
        if img is None:
            return False
        H = float(img.shape[0])
        result, _elapse = _get_engine()(str(img_path))
        for line in (result or []):
            box, txt = line[0], str(line[1])
            if len(re.findall(r"[A-Za-z]", txt)) < 2:
                continue
            ys = [p[1] for p in box]
            if (max(ys) - min(ys)) / H >= min_h_frac:
                return True
        return False
    except Exception:
        return False


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def names_in_text(text: str, roster: list[str]) -> list[str]:
    """Which roster names appear in the OCR text (space/case-insensitive — robust to the way
    stylized name-cards OCR as 'FARRAHFAWCETT'). This is the corroboration signal Face-ID uses."""
    if not text:
        return []
    nt = _norm(text)
    return [name for name in (roster or []) if len(_norm(name)) >= 6 and _norm(name) in nt]


def names_on_card(img_path, roster: list[str]) -> list[str]:
    """Roster names burned into this frame (e.g. a lower-third name card)."""
    return names_in_text(read_text(img_path), roster)


_NAME_RX = re.compile(r"\b([A-Z][a-zA-Z.'-]+(?:\s+[A-Z][a-zA-Z.'-]+){1,2})\b")


def read_names(img_path) -> list[str]:
    """Likely person names from on-screen text (2–3 capitalized words, e.g. a name card)."""
    txt = read_text(img_path)
    if not txt:
        return []
    # normalize ALLCAPS cards ("RICHARD JORDAN") into Titlecase so the name regex matches
    norm = " ".join(w.capitalize() if w.isupper() and len(w) > 1 else w for w in txt.split())
    out, seen = [], set()
    for m in _NAME_RX.finditer(norm):
        nm = m.group(1).strip()
        if nm.lower() not in seen and len(nm) > 4:
            seen.add(nm.lower())
            out.append(nm)
    return out[:6]
