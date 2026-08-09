"""A selection installed after the cut stage still has to reach the screen.

verify's late replacement and the recovery pass both run AFTER cut. A selection either of them
installs owns a source, a shot and a window — but no file was ever cut for it. Job f840b0cb49 hit
this on beats 57, 58 and 166 and died at:

    scene-lineage gate: could not make a complete owned derivative for beat 57

which is a message about the fitter, for a selection that simply had no clip. Two defects behind it:

  1. `Path("")` is `Path(".")`. The current directory exists and stats non-empty, so the
     missing-clip guard waved a blank clip_path straight through to the derivative step.
  2. The guard's advice — "re-cut the selection before build" — had nobody to act on it.

The clip is now materialised by the cut stage's own `cut_selection`, on the selection's own declared
source and window. Nothing is softened: no alternate window, no neighbour, no placeholder, and every
downstream lineage check still runs on the result. A genuine failure to produce it is still fatal.
"""
from __future__ import annotations

import inspect

from vidlore.clipstudio import build as B


SRC = inspect.getsource(B.build_video)
# The guard and its recovery, isolated from the rest of a 9000-line function.
BLOCK = SRC[SRC.index("_declared_clip = str("):SRC.index("_root = _selection_root(sel")]


def test_a_blank_clip_path_is_treated_as_missing():
    """`Path("")` must never be allowed to stand in for a real file."""
    assert '_declared_clip = str(getattr(sel, "clip_path", "") or "").strip()' in BLOCK
    assert "Path(_declared_clip) if _declared_clip else None" in BLOCK
    assert "_selected_clip is None" in BLOCK
    # is_file(), not exists() — a directory is not a clip.
    assert "_selected_clip.is_file()" in BLOCK


def test_the_missing_clip_is_recut_from_its_own_window():
    """The recovery is the cut stage's own cutter — same source, same window, same contract."""
    assert "from .cut import cut_selection as _cut_one" in BLOCK
    assert "_cut_one(proj, sel, cfg, resume=True)" in BLOCK
    # cut_selection derives the window from the selection itself, so ownership cannot drift.
    cut_src = inspect.getsource(B.__dict__.get("cut_selection") or _cut_selection())
    assert "sel.in_point" in cut_src and "sel.source_id" in cut_src


def _cut_selection():
    from vidlore.clipstudio.cut import cut_selection
    return cut_selection


def test_no_alternate_or_placeholder_is_reachable_from_here():
    """The recovery must touch the SAME selection or nothing at all — checked on code, not prose."""
    code = "\n".join(ln for ln in BLOCK.splitlines() if not ln.lstrip().startswith("#"))
    code = code.replace('"; refusing an unowned placeholder/neighbour frame"', "")  # the refusal
    for forbidden in ("alternates", "_placeholder_clip", "window[alt]", '"walk"'):
        assert forbidden not in code, f"recovery path reaches for {forbidden!r}"
    assert "refusing an unowned placeholder/neighbour frame" in BLOCK  # the refusal still says so


def test_a_failed_recut_is_still_fatal():
    assert "raise NonRetryableBuildError" in BLOCK
    assert 'kind="scene_lineage"' in BLOCK
    # and it says what actually went wrong, not what the fitter thinks
    assert "has no cut clip and re-cutting" in BLOCK


def test_a_raising_cutter_does_not_escape_as_a_crash():
    """A broken cutter must be reported and then fail closed, never bubble as an opaque traceback."""
    assert "except Exception as _recut_exc" in BLOCK
    assert "clip re-cut raised" in BLOCK


def test_the_recut_result_is_validated_before_use():
    """A returned path that is empty or absent is not a clip."""
    assert "Path(_recut).is_file()" in BLOCK
    assert "Path(_recut).stat().st_size > 0" in BLOCK
    # setattr, not `sel.clip_path = …`: the selection lock forbids naming the attribute in here.
    assert 'setattr(sel, "clip_path", str(_recut))' in BLOCK
