"""ONE table decides which terminal failures a review draft may carry.

Every render death is one of three things, and the question that separates them is not "how bad is
this?" but "if I delete this raise and ship the file anyway, what is the file?":

  CONTENT    this render's honest output with a beat a human can look at and recut — an unresolved
             beat, a scene that is not the exact moment, a verdict we could not obtain on an
             otherwise-healthy backend, a sub-HD still. This is exactly what a review draft is FOR.
  INTEGRITY  an artifact whose ownership, binding, or frame-to-verdict correspondence is unproven.
             "This may not be our footage at all." Fatal in BOTH modes, forever.
  TECHNICAL  a measurement or a mechanism that did not run — ffprobe, ffmpeg, ASR, JSON, a NameError,
             a malformed receipt. Fatal in BOTH modes. "We could not measure it" is NEVER content,
             even when the thing being measured is content: tolerating these is how a dead recovery
             path hid for months behind a `skipped (NameError)` line nobody read.

WHY A TABLE, AND WHY KEYED ON `kind`
------------------------------------
Four portal renders died in a row, each on a DIFFERENT terminal raise from the same family, because
each driver decided eligibility for itself and nothing enumerated the set. The two mechanisms that
existed disagreed on their axis: `build.content_defect_is_deliverable` matched message PREFIXES,
and `verify.is_content_stop` matched exception class. A message is prose — it gets reworded,
f-string-interpolated, concatenated from a sub-gate — and prefix matching has already misfiled a
real gate because the word "owner" appeared in its text. `kind` is a machine field with one writer,
it survives rewording, it is greppable, and `NonRetryableBuildError`'s own docstring already
mandates it: "routing MUST dispatch on it, never on message substrings".

A table is also the only thing a test can ENUMERATE. A gate that forgot to declare a prefix is
indistinguishable from a gate that has none; a gate that forgot to declare a kind is caught by
tests/test_terminal_raise_census.py before a render finds it.

FAIL CLOSED
-----------
An exception with no kind, or with a kind nobody has registered, is INTEGRITY. Not content. A
maintainer who invents a gate next year and ships it undeclared gets a hard failure and a census
test telling them to classify it — never a silent review draft.
"""
from __future__ import annotations

import os

# --------------------------------------------------------------------------------------------
# The registry. Adding a row to CONTENT widens what a review draft may carry, so it is deliberately
# a visible, reviewed diff — tests/test_terminal_raise_census.py asserts the CONTENT set against a
# literal list, and a new member fails that test until someone edits it on purpose.
# --------------------------------------------------------------------------------------------
KIND_CLASS: dict[str, str] = {
    # ---- CONTENT — an honest artifact with a flaw a human can see and fix -------------------
    # beats that fail the semantic publication contract, and the repair machinery that ran out of
    # moves trying to fix them (pagination exhaustion, inconclusive scoped re-verification)
    "selection_relevance": "content",
    # verifier-rejected beats with no valid fallback — they air flagged
    "rejected_footage": "content",
    # a MEASURED sub-HD selection. NOT an unprobeable one — see native_resolution_probe.
    "native_resolution": "content",
    # a still whose semantic verdict is missing, stale or negative
    "image_semantic": "content",
    # narration faster than the reading ceiling
    "caption_readability": "content",
    # a per-breakout editorial verdict (black window, wrong clip, near-silent). The driver's
    # resume drops the offending insert via breakout_qa_exclude.json and re-composes.
    "breakout_qa": "content",
    # a per-still non-verdict on a backend that is otherwise answering
    "still_verdict": "content",
    # exact beats left unverified. Its own warn branch already ships them flagged and only the
    # block branch raises, so this row preserves that decision rather than making a new one — it
    # exists so the DRIVER can see what the gate had already concluded.
    "unverified_exact": "content",
    # the pre-assembly feasibility check: the SOUND subset of the rejected-footage gate, predicted
    # before ~20 min of doomed encoding. Same verdict, same class, cheaper.
    "preassemble_feasibility": "content",

    # ---- INTEGRITY — the artifact may not be ours; never deliverable -----------------------
    "scene_lineage": "integrity",
    "selection_relevance_audit": "integrity",
    "voiceover_alignment": "integrity",
    "montage": "integrity",
    "animal": "integrity",

    # ---- TECHNICAL — a measurement or mechanism did not run; never deliverable -------------
    "down": "technical",        # VisionBackendError: backend outage
    "billing": "technical",     # VisionBackendError: quota/credits
    "auth": "technical",        # VisionBackendError: bad key
    "native_resolution_probe": "technical",   # ffprobe gave nothing / no source bound
    "breakout_qa_probe": "technical",         # a breakout QA probe could not run
    "final_ad_unverified": "technical",       # the ad scan itself could not complete
    "breakout_caption": "technical",
}

CONTENT_KINDS = frozenset(k for k, v in KIND_CLASS.items() if v == "content")


def gate_class(exc) -> str:
    """'content' | 'integrity' | 'technical' for a terminal exception. Unclassified is integrity."""
    kind = str(getattr(exc, "kind", "") or "")
    if not kind:
        return "integrity"
    return KIND_CLASS.get(kind, "integrity")


def is_content_verdict(exc) -> bool:
    """The artifact is honestly ours and imperfect — a review draft may carry it."""
    return gate_class(exc) == "content"


def review_draft_mode() -> bool:
    """True when this render is a review draft rather than a publication candidate."""
    return os.environ.get("VIDLORE_CLIPSTUDIO_RELEASE_BLOCK_MODE", "block").strip().lower() == "warn"


def deliverable(exc) -> bool:
    """Whether THIS render may continue past THIS failure and hand back a marked draft.

    Both halves are required: production mode ('block', the default) delivers nothing, and review
    mode delivers only registered content verdicts.
    """
    return review_draft_mode() and is_content_verdict(exc)
