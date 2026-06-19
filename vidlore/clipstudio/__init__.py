"""ClipStudio — intelligent movie-clip editing module for Vidlore.

Takes a narration script + permitted source videos and assembles a faceless movie-niche video
by matching each line to the source clip that best shows what it's about, then rendering through
Vidlore's existing `assemble()` engine. Purely additive; see DESIGN.md for the full design,
the reuse map, and the honest limits / manual-review checkpoints.

Light imports only at package level — the heavy stages (index/match pull in onnxruntime,
faster-whisper, OpenCV) are imported lazily by the CLI / callers.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

# isolated deps (yt-dlp, scenedetect, google-genai, rapidocr …) live in .clipstudio_libs to keep
# the engine venv pristine — register it ONCE here so every entrypoint sees them, not just the
# modules (llm/ocr) that happened to append it themselves
_LIBS = _Path(__file__).resolve().parents[2] / ".clipstudio_libs"
if _LIBS.exists() and str(_LIBS) not in _sys.path:
    _sys.path.append(str(_LIBS))

__version__ = "0.1.0"

from .models import (
    ClipProject, SourceVideo, Shot, ScriptSegment, ClipCandidate, ClipSelection,
    PERMISSIONS, PERMISSION_BLOCKS,
)
from .config import ClipConfig, load_clip_config

__all__ = [
    "__version__",
    "ClipProject", "SourceVideo", "Shot", "ScriptSegment", "ClipCandidate", "ClipSelection",
    "PERMISSIONS", "PERMISSION_BLOCKS",
    "ClipConfig", "load_clip_config",
]
