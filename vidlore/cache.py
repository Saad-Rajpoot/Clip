"""Per-scene content-addressed cache.

Lets `--resume` regenerate only the scenes that actually changed (the
thing Vidlore can't do — there a single edit forces a full ~50-60 min
re-render). The cache key is a hash of everything that affects a scene's
output, so an unedited scene is a cache hit and is skipped entirely.
"""
from __future__ import annotations

import hashlib


def scene_key(*parts: object) -> str:
    """Stable short hash of the inputs that determine a scene artifact.
    Order matters; None is allowed."""
    h = hashlib.sha256()
    for p in parts:
        h.update(repr(p).encode("utf-8"))
        h.update(b"\x1f")  # unit separator so parts can't run together
    return h.hexdigest()[:16]
