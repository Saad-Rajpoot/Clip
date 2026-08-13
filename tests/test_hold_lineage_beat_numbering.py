"""An editorial hold must name its donor in the numbering the lineage gate reads.

Job d835faa83e died at the scene-lineage gate after six hours — twice in a row at the end of a
render, the second time with:

    scene-lineage gate: 12 clip(s) do not derive from their own verified selection
      (first: scene hold claims donor beat 177 but its root owner is 176)

Six holds, twelve violations, and every one of them the same off-by-the-breakout-count. The gate
compares `hold_of_beat` against the record's `owner` / `original_beat`, which are ORIGINAL beat
numbers. The hold was writing `_last_clean_idx` — a FINAL SCENE index. A breakout inserted earlier
in the timeline shifts every later final scene past its beat by one, so the donor named a different
beat than the one whose frame was actually frozen, and it simultaneously tripped the second rule:
`donor == original`, "names itself as its own donor".

The tell is two lines below the defect: the rejected-footage AUDIT record has always written
`_orig(_last_clean_idx)`. Only the lineage record forgot the conversion.

Nothing about the gate is relaxed here. The frame still has to trace to the donor's own verified
selection; it is now merely asked about under the donor's real name.
"""
from __future__ import annotations

import inspect
import re

from vidlore.clipstudio import build as B
from vidlore.clipstudio import scene_lineage as SL


def _hold_record(**over):
    rec = {
        "kind": "scene_hold", "via": "editorial_hold",
        "original_beat": 177, "owner": 176, "hold_of_beat": 176,
        "hold_kind": "editorial_hold", "hold_duration_s": 1.9,
        "hold_compat_evidence": {"scene_overlap": 0.71},
    }
    rec.update(over)
    return rec


def _violations(rec) -> list:
    """Run the two hold-identity rules the way the gate does."""
    out = []
    donor, owner, original = rec.get("hold_of_beat"), rec.get("owner"), rec.get("original_beat")
    if donor is not None and donor != owner:
        out.append(f"claims donor beat {donor} but its root owner is {owner}")
    if donor is not None and donor == original:
        out.append("names itself as its own donor")
    return out


# ---------------------------------------------------------------- the render that died
def test_the_shipped_record_is_rejected_and_the_corrected_one_is_not():
    """Beat 178 of that render: final scene 178, original beat 177, one breakout before it."""
    shipped = _hold_record(hold_of_beat=177)         # what the code wrote: a FINAL scene index
    assert len(_violations(shipped)) == 2, "this is the pair that killed the render"

    corrected = _hold_record(hold_of_beat=176)       # the same donor, named as an ORIGINAL beat
    assert _violations(corrected) == []


def test_the_lineage_record_converts_the_donor_index():
    src = inspect.getsource(B.build_video)
    m = re.search(r'"hold_of_beat":\s*int\(([^)]*\))?\)?', src)
    assert m, "the hold record no longer declares its donor"
    assert "_orig(" in m.group(0), \
        "hold_of_beat is a FINAL scene index again — the gate reads original beat numbers"


def test_it_agrees_with_the_audit_record_beside_it():
    """Both records describe the same donation; disagreeing on its name is how this got shipped."""
    src = inspect.getsource(B.build_video)
    i = src.index('"hold_of_beat"')
    nearby = src[i:i + 2000]
    assert '"held_from_beat": _orig(_last_clean_idx)' in nearby, \
        "the audit's converted form is the reference; keep them side by side"


# ---------------------------------------------------------------- the gate itself is intact
def test_both_identity_rules_still_exist_and_still_fail_closed():
    # the rules live in the module's record walker, not in the raise wrapper
    src = inspect.getsource(SL)
    assert "scene hold claims donor beat" in src
    assert "scene hold names itself as its own donor" in src
    for guard in ('via != "editorial_hold"', "hold_compat_evidence", "hold_duration_s"):
        assert guard in src, f"a hold rule disappeared: {guard}"


def test_a_hold_that_really_does_smuggle_a_foreign_owner_still_fails():
    """The fix must not make the rule unfalsifiable: a donor that is NOT the record's owner is
    still a violation, whatever numbering it is written in."""
    assert _violations(_hold_record(hold_of_beat=42)) == [
        "claims donor beat 42 but its root owner is 176"]


def test_a_self_donating_hold_still_fails():
    assert "names itself as its own donor" in _violations(_hold_record(hold_of_beat=177,
                                                                      owner=177))


def test_a_render_with_no_breakouts_is_unaffected():
    """_orig is identity when nothing shifted, so the overwhelming majority of renders — and every
    one that ever passed this gate — behave exactly as before."""
    src = inspect.getsource(B.build_video)
    assert "_final_to_orig.get(i, i)" in src, \
        "_orig must fall back to the index itself when there is no remap"
