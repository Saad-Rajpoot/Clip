"""Every terminal raise must declare whether a review draft may carry it.

Four portal renders died in a row, each on a DIFFERENT terminal raise, each one a member of the same
family that nobody had classified:

    0ca9dc4c2f x3   semantic scoped re-verification remained technically inconclusive
    f840b0cb49      semantic recovery pagination reached its finite 16-page guard
    0321078108      semantic recovery pagination reached its finite 14-page guard
    229233891e      source fingerprint unavailable before indexing <source>

Each was found the same way: a render burned six to eight hours, died, and a human read the
traceback. Fixing them one at a time never converged, because the set was never enumerated — a gate
that forgot to declare itself looked exactly like a gate that had nothing to declare.

This test enumerates the set. It walks the AST of every shipped module, finds every construction of
a terminal exception, and requires each one to be classified — either by carrying a `kind` that
release_policy.KIND_CLASS knows, or by being listed in the untyped inventory below with a reason.
A new raise fails this test with instructions, before it can ever fail a render.

It is fail-closed in both directions: an unclassified raise fails, and an unknown kind fails.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

from vidlore.clipstudio import release_policy as RP


ROOT = pathlib.Path(__file__).resolve().parents[1]
PKG = ROOT / "vidlore"

# The exception types whose construction ENDS a render.
TERMINAL = {"NonRetryableBuildError", "VisionBackendError", "PipelineError",
            "InconclusiveAcquisitionError"}

# PipelineError predates `kind` and is used for ~65 plumbing faults — a missing script, no sources
# discovered, an unreadable checkpoint. Those are TECHNICAL by construction and correctly fatal in
# both modes, so they are not required to carry a kind. The two that ARE content verdicts have been
# typed (the pagination guard and the inconclusive re-verification); this test pins that count so a
# third content-bearing PipelineError cannot appear untyped and unnoticed.
TYPED_PIPELINE_ERRORS_EXPECTED = 2

# The CONTENT set is asserted literally. Widening what a review draft may carry is always a visible
# two-file diff: the registry, and this list.
CONTENT_KINDS_EXPECTED = sorted([
    "breakout_qa",
    "caption_readability",
    "image_semantic",
    "native_resolution",
    "preassemble_feasibility",
    "rejected_footage",
    "selection_relevance",
    "still_verdict",
    "unverified_exact",
])


def _terminal_sites():
    """(module, lineno, exc_name, kind_or_None) for every terminal construction we ship."""
    out = []
    for path in sorted(PKG.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:                                   # not ours to police here
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = (node.func.id if isinstance(node.func, ast.Name)
                    else getattr(node.func, "attr", ""))
            if name not in TERMINAL:
                continue
            kind = None
            for kw in node.keywords:
                if kw.arg == "kind":
                    kind = (kw.value.value if isinstance(kw.value, ast.Constant)
                            else "<dynamic>")
            # VisionBackendError takes kind positionally as its 2nd arg
            if kind is None and name == "VisionBackendError" and len(node.args) >= 2:
                a = node.args[1]
                kind = a.value if isinstance(a, ast.Constant) else "<dynamic>"
            out.append((str(path.relative_to(ROOT)), node.lineno, name, kind))
    return out


SITES = _terminal_sites()


def test_the_census_finds_the_raises_at_all():
    """A walker that silently matches nothing would make every assertion below vacuous."""
    assert len(SITES) > 100, f"only {len(SITES)} terminal sites found — the AST walk is broken"
    mods = {s[0] for s in SITES}
    assert any("build.py" in m for m in mods) and any("orchestrate.py" in m for m in mods)


def test_every_typed_raise_uses_a_registered_kind():
    """A kind nobody registered resolves to integrity — silently undeliverable. Catch it here."""
    unknown = [s for s in SITES
               if s[3] not in (None, "<dynamic>") and s[3] not in RP.KIND_CLASS]
    assert not unknown, (
        "terminal raise(s) carry a kind that release_policy.KIND_CLASS does not know:\n  "
        + "\n  ".join(f"{f}:{ln} {exc}(kind={k!r})" for f, ln, exc, k in unknown)
        + "\nAdd each kind to KIND_CLASS as 'content', 'integrity' or 'technical'.")


def test_no_build_error_ships_without_a_kind():
    """NonRetryableBuildError means 'a content verdict'. Untyped, it is treated as integrity and
    stops offering a review draft — which is how a render that could have delivered, died."""
    bare = [s for s in SITES if s[2] == "NonRetryableBuildError" and s[3] is None]
    assert not bare, (
        "NonRetryableBuildError raised without kind=:\n  "
        + "\n  ".join(f"{f}:{ln}" for f, ln, _e, _k in bare)
        + "\nDeclare it: kind='<name>' plus a row in release_policy.KIND_CLASS.")


def test_no_vision_error_ships_without_a_kind():
    bare = [s for s in SITES if s[2] == "VisionBackendError" and s[3] is None]
    assert not bare, ("VisionBackendError raised without a kind:\n  "
                      + "\n  ".join(f"{f}:{ln}" for f, ln, _e, _k in bare))


def test_content_bearing_pipeline_errors_are_pinned():
    """PipelineError has no kind= parameter; the content-verdict ones are tagged after construction.
    Pinning the count means a third one cannot appear untyped and go unnoticed for four renders."""
    src = (PKG / "clipstudio" / "orchestrate.py").read_text(encoding="utf-8")
    tagged = src.count('.kind = "selection_relevance"')
    assert tagged == TYPED_PIPELINE_ERRORS_EXPECTED, (
        f"{tagged} PipelineError(s) are tagged as content verdicts, expected "
        f"{TYPED_PIPELINE_ERRORS_EXPECTED}. If you added one on purpose, bump the constant and say "
        f"why in the commit; if you removed one, a render that used to deliver now dies.")


def test_the_content_set_is_a_reviewed_diff():
    assert sorted(RP.CONTENT_KINDS) == CONTENT_KINDS_EXPECTED, (
        "the set of failures a REVIEW DRAFT may carry changed.\n"
        f"  registry: {sorted(RP.CONTENT_KINDS)}\n  expected: {CONTENT_KINDS_EXPECTED}\n"
        "Widening this ships imperfect video; narrowing it kills renders that used to deliver. "
        "Update this list deliberately, in the same commit, with the reason.")


def test_unclassified_is_integrity_not_content():
    """The whole safety property in one assertion."""
    class _Anon(RuntimeError):
        pass
    assert RP.gate_class(_Anon("x")) == "integrity"
    assert RP.gate_class(RuntimeError("x")) == "integrity"
    e = RuntimeError("x")
    e.kind = "a_kind_invented_next_year"
    assert RP.gate_class(e) == "integrity"
    assert RP.is_content_verdict(e) is False


@pytest.mark.parametrize("kind", CONTENT_KINDS_EXPECTED)
def test_production_mode_delivers_nothing(kind, monkeypatch):
    monkeypatch.setenv("VIDLORE_CLIPSTUDIO_RELEASE_BLOCK_MODE", "block")
    e = RuntimeError("x")
    e.kind = kind
    assert RP.is_content_verdict(e) is True        # it IS a content verdict …
    assert RP.deliverable(e) is False              # … and production still refuses to deliver it


def test_message_prefix_matching_is_gone():
    """It mis-filed a gate because the word 'owner' appeared in its text, and it could not be
    enumerated. Nobody may reintroduce it."""
    build = (PKG / "clipstudio" / "build.py").read_text(encoding="utf-8")
    assert "_DELIVERABLE_GATE_PREFIXES" not in build
    assert "startswith(p) for p in" not in build
    import inspect
    from vidlore.clipstudio import build as B
    sig = inspect.signature(B.content_defect_is_deliverable)
    assert list(sig.parameters) == ["exc"], "it must take the exception, not a message string"
