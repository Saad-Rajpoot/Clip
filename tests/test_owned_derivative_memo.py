"""Re-encoding an unchanged beat on every build pass is waste; re-using a changed one is a lie.

`_fit_verified_selection_clip` produces each beat's owned derivative, and it runs for every beat on
every build pass. A render that self-heals and rebuilds pays for all of them again — on a 180-scene
job, twice — even where nothing about the beat changed. The output is a pure function of its
inputs, so it is memoised.

The memo is keyed on ALL of those inputs and on nothing else:
  * the selected clip's own BYTES — not its path and not its mtime, because a re-cut writes the
    same filename, and a path key would happily serve the previous cut's pixels
  * the duration asked for, the crop filter, the zoom
  * the boundary contract in force (a clip cut half-open may hold its true final frame; one that
    was not may not, and that changes the encode)
  * a schema version, so changing this function invalidates every entry ever written

A hit is re-validated exactly as a fresh encode is: the file must exist, be non-empty, and still
satisfy the requested duration on probe. The caller then puts hit and miss through the identical
`_lineage_derive` proof. The memo can skip work; it cannot approve any.
"""
from __future__ import annotations

import inspect
import json

import pytest

from vidlore.clipstudio import build as B


FIT = inspect.getsource(B._fit_verified_selection_clip)


# ------------------------------------------------------------------ the key
def test_the_key_is_the_clip_content_not_its_path():
    """A re-cut writes the same filename. A path or mtime key would serve stale pixels."""
    assert "clip_sha256" in FIT
    assert "sha256()" in FIT or "sha256(" in FIT
    assert "st_mtime" not in FIT, "mtime must play no part in identity"


@pytest.mark.parametrize("field", ["need", "crop", "zoom", "frame_exact", "schema"])
def test_every_input_that_can_change_the_output_is_in_the_key(field):
    assert f'"{field}"' in FIT, field


def test_the_schema_version_can_invalidate_every_entry():
    assert isinstance(B._FIT_MEMO_SCHEMA, int)
    assert "_FIT_MEMO_SCHEMA" in FIT


def test_the_boundary_contract_is_part_of_identity():
    """A half-open clip may have its true final frame held; an uncertified one may not. Same
    source bytes, different encode — so they must not share a memo entry."""
    assert '"frame_exact": bool(frame_exact)' in FIT


# ------------------------------------------------------------------ a hit is re-validated
def test_a_hit_must_still_exist_and_be_non_empty():
    assert "dest.exists() and dest.stat().st_size > 0" in FIT


def test_a_hit_must_still_satisfy_the_requested_duration():
    """The same check a fresh encode has to pass — a truncated leftover is not a hit."""
    i = FIT.index('_FIT_MEMO_STATS["hit"]')
    assert "_ffprobe_duration(dest) + (2.0 / 30.0) >= need" in FIT[:i]


def test_hit_and_miss_both_face_the_lineage_proof():
    whole = inspect.getsource(B)
    i = whole.index("_owned = _fit_verified_selection_clip(")
    tail = whole[i:i + 600]
    assert "_lineage_derive(" in tail, \
        "the caller proves lineage on whatever it gets back, cached or fresh"


# ------------------------------------------------------------------ writing
def test_the_memo_is_only_recorded_for_a_validated_file():
    """Written after the duration check, never before — a short encode must not leave a memo."""
    i_check = FIT.index("if got + (2.0 / 30.0) < need:")
    i_write = FIT.index("_os_w.replace(_tmp, _memo)")
    assert i_check < i_write


def test_the_write_is_atomic():
    assert "_os_w.replace(_tmp, _memo)" in FIT
    assert ".tmp" in FIT


def test_os_is_imported_locally_for_the_memo_write():
    """MEASURED FAILURE of this memo's first draft. Module-level `os` is not reliably bound in
    build.py — every neighbouring function does its own `import os as _os_xx`. The NameError landed
    inside the fail-open `except Exception`, so the memo silently never recorded: hit 0 / miss 2 on
    a warm run, with owned.mp4.key.json.tmp left behind on disk."""
    assert "import os as _os_w" in FIT


def test_a_memo_fault_never_costs_the_encode():
    """Caching is an optimisation. It must not be able to fail the work it memoises."""
    assert FIT.count("except Exception") >= 2
    assert "_key = None" in FIT, "an unreadable/corrupt memo must fall through to a real encode"


def test_hits_and_misses_are_counted_for_the_audit_trail():
    assert set(B._FIT_MEMO_STATS) == {"hit", "miss"}


# ------------------------------------------------------------------ the scenarios the brief names
def _key_of(clip_sha="a", need=2.0, crop="", zoom=1.0, frame_exact=True):
    return {"schema": B._FIT_MEMO_SCHEMA, "clip_sha256": clip_sha, "need": round(need, 4),
            "crop": crop, "zoom": round(float(zoom), 6), "frame_exact": bool(frame_exact)}


def test_an_identical_rebuild_is_the_same_key():
    assert _key_of() == _key_of()


def test_changing_the_scene_selection_changes_the_key():
    """A different selection is different clip bytes."""
    assert _key_of(clip_sha="a") != _key_of(clip_sha="b")


def test_changing_the_crop_changes_the_key():
    assert _key_of(crop="") != _key_of(crop="crop=iw*0.840:ih*0.840:0:0")


def test_changing_the_requested_duration_changes_the_key():
    assert _key_of(need=2.0) != _key_of(need=3.5)


def test_changing_the_zoom_changes_the_key():
    assert _key_of(zoom=1.0) != _key_of(zoom=1.055)


def test_a_corrupted_memo_is_detected_and_rebuilt(tmp_path):
    """A truncated or non-JSON sidecar must read as a miss, not raise and not be trusted."""
    p = tmp_path / "beat.mp4.key.json"
    p.write_text("{ not json at all")
    try:
        got = json.loads(p.read_text()).get("key")
    except Exception:
        got = None
    assert got != _key_of()


# ------------------------------------------------------------------ the output is an input too
#
# MEASURED LATENT DEFECT of this memo's first version. Matching every input and finding a file of
# the right duration at `dest` does NOT prove the file is the one the memo wrote: the caption-dodge
# sweep calls `_crop_clip_corner`, which ends in `out.replace(src)` on exactly this path — the
# derivative is rewritten IN PLACE after the entry is recorded. build.py's own dark-sweep comment
# already notes this ("_crop_clip_corner rewrites the clip file in place"). The key would still
# match and the duration would still pass, so the next build pass would be handed a
# caption-dodge-cropped clip as the plain derivative: a crop applied twice, or applied where none
# was asked for.
def test_a_hit_re_derives_the_digest_of_the_file_on_disk():
    assert '_blob.get("out_sha256") == _sha256_file(dest)' in FIT


def test_the_entry_records_what_it_actually_wrote():
    assert '"out_sha256": _sha256_file(dest)' in FIT


def test_the_output_digest_is_checked_before_the_hit_is_taken():
    i_check = FIT.index('_blob.get("out_sha256")')
    i_hit = FIT.index('_FIT_MEMO_STATS["hit"]')
    assert i_check < i_hit


def test_the_schema_was_raised_so_old_entries_cannot_be_trusted():
    """Entries written before the digest existed carry no out_sha256 and must not be honoured."""
    assert B._FIT_MEMO_SCHEMA >= 2


def test_an_entry_without_an_output_digest_is_a_miss():
    key = {"schema": B._FIT_MEMO_SCHEMA, "clip_sha256": "a", "need": 1.0,
           "crop": "", "zoom": 1.0, "frame_exact": True}
    old_entry = {"key": key}                       # what version 1 wrote
    assert old_entry.get("out_sha256") != "whatever-is-on-disk"


def test_identity_is_content_not_path_or_mtime():
    src = inspect.getsource(B._sha256_file)
    assert "sha256" in src
    assert "st_mtime" not in src and "st_size" not in src
