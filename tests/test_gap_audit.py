"""A machine audit may say "not there" only when it actually looked everywhere and every look answered.

This is the producer for the evidence the specificity ladder is gated behind. It does not pretend to
be the human frame review — it declares `audit_kind: machine_exhaustive_strict_verifier` and rests on
a narrower, checkable claim: every eligible window in a recorded, countable universe went through the
SAME judge the publication gate uses, and none passed.

Three properties carry that claim, and each is enforced rather than asserted:

  ONE JUDGE — decisions come from `verify.strict_window_verdict`, the function `_try_promote_inner`
  itself calls. No second implementation of the bar exists to drift from it.

  CLIP MAY ORDER, NEVER EXCLUDE — beat 134's answer sat at CLIP rank 563 of 4942. An auditor that
  trusted retrieval would have called that beat a footage gap and been wrong. Exclusions here are
  structural only, and every one carries a recorded reason.

  A CALL THAT DID NOT ANSWER IS NOT A REJECTION — one incomplete verdict anywhere makes the whole
  audit `audit_incomplete`, which authorizes nothing. Incomplete verdicts are also never cached,
  because a cached incomplete would become a rejection on the next pass.
"""
from __future__ import annotations

import inspect
import json

import pytest

from vidlore.clipstudio import gap_audit as GA


SRC = inspect.getsource(GA.exhaustive_gap_audit)


# ------------------------------------------------------------------ it is not the human review
def test_it_declares_what_kind_of_audit_it_is():
    assert GA.GAP_AUDIT_KIND == "machine_exhaustive_strict_verifier"
    assert "audit_kind" in SRC


def test_it_never_claims_pipeline_bugs_are_ruled_out():
    """That claim belongs to a human frame review. This one must not borrow it."""
    whole = inspect.getsource(GA)
    assert "pipeline_bug_ruled_out" not in whole


# ------------------------------------------------------------------ one judge
def test_windows_are_decided_by_the_gate_s_own_function():
    assert "_V.strict_window_verdict(" in SRC


def test_there_is_no_local_reimplementation_of_the_bar():
    whole = inspect.getsource(GA)
    for forbidden in ("_strict_keep_rejection_reason", "_exact_contextual_ok",
                      "_character_keep_rejection_reason"):
        assert forbidden not in whole, forbidden


# ------------------------------------------------------------------ ranking may order, never exclude
def test_ordering_is_optional_and_cannot_shrink_the_universe():
    assert "if len(ordered) == universe_n:" in SRC, \
        "a reorder that changes the count must be discarded, not trusted"


def test_eligibility_excludes_only_structurally_and_records_every_reason():
    src = inspect.getsource(GA.eligible_universe)
    for reason in ("source_status_not_ok", "pool_gate:", "media_missing", "keyframe_missing",
                   "index_unreadable"):
        assert reason in src, reason
    for ranking in ("clip", "rank", "score", "top_k", "bench"):
        assert ranking not in src.lower().replace("ranking low", ""), ranking


def test_every_exclusion_carries_a_reason_field():
    src = inspect.getsource(GA.eligible_universe)
    assert src.count('"reason"') >= 5
    assert 'exclusions.append({"source_id": sid})' not in src


# ------------------------------------------------------------------ incomplete is not rejection
def test_one_incomplete_window_blocks_the_whole_authorization():
    assert "complete = covered and incomplete == 0" in SRC
    assert 'status = "footage_gap", "complete"' in SRC.replace("classification, ", "")


def test_a_capped_scan_can_never_report_exhaustion():
    assert "covered = examined + incomplete >= universe_n and not cap" in SRC


def test_authorization_requires_both_a_gap_and_a_complete_scan():
    assert 'bool(classification == "footage_gap" and status == "complete")' in SRC


def test_an_incomplete_verdict_is_never_cached():
    src = inspect.getsource(GA._VerdictCache.put)
    assert 'decision.get("status") != "judged"' in src and "return" in src


def test_only_judged_entries_are_served_from_cache():
    src = inspect.getsource(GA._VerdictCache.get)
    assert 'got.get("status") == "judged"' in src


def test_a_corrupt_cache_reads_as_empty_rather_than_raising(tmp_path):
    p = tmp_path / "v.json"
    p.write_text("{ not json")
    c = GA._VerdictCache(p)
    assert c.get("anything") is None


# ------------------------------------------------------------------ the key binds everything
@pytest.mark.parametrize("field", ["keyframe_sha256", "contract", "judge", "schema"])
def test_the_verdict_key_binds_every_input(field):
    src = inspect.getsource(GA._verdict_key)
    assert field in src


def test_a_changed_contract_is_a_different_key():
    j = {"m": 1}
    a = GA._verdict_key("aa", {"text": "one"}, j)
    b = GA._verdict_key("aa", {"text": "two"}, j)
    assert a != b


def test_a_changed_judge_is_a_different_key():
    c = {"text": "one"}
    assert GA._verdict_key("aa", c, {"m": 1}) != GA._verdict_key("aa", c, {"m": 2})


def test_a_changed_window_is_a_different_key():
    c, j = {"text": "one"}, {"m": 1}
    assert GA._verdict_key("aa", c, j) != GA._verdict_key("bb", c, j)


def test_the_judge_identity_covers_the_decision_source_and_the_model():
    src = inspect.getsource(GA.judge_identity)
    assert "strict_window_verdict" in src and "vision_model" in src


@pytest.mark.parametrize("field", [
    "index", "text", "visual_policy", "required_kind", "required_entity",
    "scene_query", "expected_visual", "quote", "is_specific_claim"])
def test_the_beat_contract_covers_what_changes_satisfiability(field):
    assert field in inspect.getsource(GA.beat_contract)


# ------------------------------------------------------------------ the report is auditable
@pytest.mark.parametrize("field", [
    "universe_size", "universe_fingerprint", "windows_examined", "windows_incomplete",
    "exclusions", "exclusion_count", "pool_fingerprint", "beat_fingerprint",
    "contract_fingerprint", "judge", "verdict_cache", "authorizes_softening", "passing_window"])
def test_the_result_records_what_a_reviewer_would_have_to_check(field):
    assert f'"{field}"' in SRC, field


def test_a_pass_stops_the_scan_and_is_reported():
    """Finding footage is the more valuable outcome — it means the beat was never a gap."""
    assert "break" in SRC
    assert '"not_a_footage_gap"' in SRC
