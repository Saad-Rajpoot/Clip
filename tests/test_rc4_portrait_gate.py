"""RC4 — portrait IDENTITY gate (pure unit tests, NO network).

Root cause guarded: a documentary card labelled "AYATOLLAH KHOMEINI" rendered a
DIFFERENT cleric's face. Tier-1 Wikipedia lead-image is page-title identity-
verified, but Tier-2 Wikimedia Commons and Tier-3 Library of Congress used to
accept a NAME-SEARCH hit gated only by `_looks_like_portrait` (a face-STRUCTURE
check, NOT whose face). So a wrong person's face could be baked. This proves the
fix:
  (1) a PAGE-VERIFIED portrait is still allowed,
  (2) a Commons name-search hit whose title does NOT name-match → None
      (the wrong face is NEVER used),
  (3) returning None is the documented monogram-fallback signal,
  (4) `_person_name_variants("Ayatollah Khomeini")` includes "Khomeini" and
      `_person_name_variants("President Saddam Hussein")` includes
      "Saddam Hussein",
  (5) no crash when every source is missing (returns None),
  (6) provenance records identity_verified=False on rejection,
  (7) the legacy classified card uses a NEUTRAL header (never a fabricated
      "DEPARTMENT OF INTERNAL AFFAIRS") when no agency is supplied.

All network functions are monkeypatched — nothing touches the network.

Run:  PYTHONPATH=. .venv/bin/python tests/test_rc4_portrait_gate.py
"""
import contextlib
import inspect
import os
import tempfile
from pathlib import Path

import vidlore.footage as footage
import vidlore.portrait_intel as pintel

_passed = 0


def check(name, cond):
    global _passed
    assert cond, "FAIL: " + name
    print("  ok ", name)
    _passed += 1


# --------------------------------------------------------------------------- #
# tiny monkeypatch helper (no pytest in this venv)
# --------------------------------------------------------------------------- #
@contextlib.contextmanager
def patched(targets):
    """targets: list of (obj, attr, value). Restores originals on exit.
    A sentinel value `_DEL` deletes the attr (and removes the env var)."""
    saved = []
    for obj, attr, val in targets:
        if obj is os.environ:
            saved.append((obj, attr, os.environ.get(attr, _DEL), True))
            if val is _DEL:
                os.environ.pop(attr, None)
            else:
                os.environ[attr] = val
        else:
            saved.append((obj, attr, getattr(obj, attr, _DEL), False))
            setattr(obj, attr, val)
    try:
        yield
    finally:
        for obj, attr, old, is_env in reversed(saved):
            if is_env:
                if old is _DEL:
                    os.environ.pop(attr, None)
                else:
                    os.environ[attr] = old
            elif old is _DEL:
                with contextlib.suppress(AttributeError):
                    delattr(obj, attr)
            else:
                setattr(obj, attr, old)


_DEL = object()


def _tmp():
    return Path(tempfile.mkdtemp(prefix="rc4_portrait_"))


def _fake_jpeg(path: Path) -> None:
    """Write a real tiny JPEG so dest.exists() is true (pixels never inspected
    by the identity gate)."""
    try:
        from PIL import Image
        Image.new("RGB", (640, 800), (90, 90, 90)).save(path, "JPEG")
    except Exception:                                          # pragma: no cover
        path.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 4096 + b"\xff\xd9")


def _reset_globals():
    footage._LAST_WIKIMEDIA_TITLE = None
    footage._LAST_LOC_TITLE = None
    footage._PORTRAIT_PROVENANCE.clear()


# base env every person test runs under: force public-figure path, wikimedia on
_BASE_ENV = [
    (os.environ, "VIDLORE_REAL_PERSON", "all"),
    (os.environ, "VIDLORE_WIKIMEDIA", "1"),
    (os.environ, "VIDLORE_UNVERIFIED_PORTRAIT", _DEL),
    (footage, "_wikimedia_on", lambda: True),
]


# =========================================================================== #
# (4) honorific-normalized name variants
# =========================================================================== #
def test_name_variants():
    v1 = footage._person_name_variants("Ayatollah Khomeini")
    check("variants(Ayatollah Khomeini) includes 'Khomeini'", "Khomeini" in v1)
    check("variants keep the original", "Ayatollah Khomeini" in v1)

    v2 = footage._person_name_variants("President Saddam Hussein")
    check("variants(President Saddam Hussein) includes 'Saddam Hussein'",
          "Saddam Hussein" in v2)
    check("variants keep original (Saddam)",
          "President Saddam Hussein" in v2)

    check("multi-word honorific 'Supreme Leader Khamenei' -> 'Khamenei'",
          "Khamenei" in footage._person_name_variants("Supreme Leader Khamenei"))
    check("no honorific -> single deduped entry",
          footage._person_name_variants("Winston Churchill")
          == ["Winston Churchill"])
    check("never strip to zero tokens (bare 'General')",
          footage._person_name_variants("General") == ["General"])
    check("blank in -> empty out (no crash)",
          footage._person_name_variants("") == [])
    check("'Dr Strangelove' -> 'Strangelove' variant",
          "Strangelove" in footage._person_name_variants("Dr Strangelove"))


# =========================================================================== #
# (1) a PAGE-VERIFIED portrait is allowed
# =========================================================================== #
def test_page_verified_allowed():
    _reset_globals()
    d = _tmp() / "p.jpg"

    def fake_lead(name, dest):
        _fake_jpeg(dest)
        return str(dest), {"person": name, "source": "Wikipedia lead image",
                           "name_match": 1.0, "validator": "grayscale_portrait",
                           "fallback_reason": None}

    def boom(*a, **k):
        raise AssertionError("Tier-2 must NOT run on a Tier-1 win")

    with patched(_BASE_ENV + [(footage, "_wikipedia_lead_portrait", fake_lead),
                              (footage, "_wikimedia_image", boom)]):
        out = footage._real_person_image("Ayatollah Khomeini", "supreme leader",
                                         d, cache_dir=d.parent)
    check("page-verified portrait returned", out == str(d))
    check("page-verified file exists", d.exists())
    prov = footage._PORTRAIT_PROVENANCE["Ayatollah Khomeini"]
    check("page-verified identity_verified True", prov["identity_verified"] is True)
    check("page-verified source is wikipedia",
          "Wikipedia lead image" in str(prov.get("source")))


def test_tier1_retries_over_variant():
    """Honorific full name fails page-verify; stripped surname wins — the exact
    Khomeini fix path."""
    _reset_globals()
    d = _tmp() / "p.jpg"
    seen = []

    def fake_lead(name, dest):
        seen.append(name)
        if name == "Khomeini":
            _fake_jpeg(dest)
            return str(dest), {"person": name, "source": "Wikipedia lead image",
                               "name_match": 1.0, "validator": "grayscale_portrait"}
        return None, {"person": name, "name_match": 0.0,
                      "fallback_reason": "name mismatch (page='Wrong')"}

    with patched(_BASE_ENV + [(footage, "_wikipedia_lead_portrait", fake_lead)]):
        out = footage._real_person_image("Ayatollah Khomeini", "cleric", d,
                                         cache_dir=d.parent)
    check("tier1 variant retry returns portrait", out == str(d))
    check("tier1 tried both original and stripped",
          "Ayatollah Khomeini" in seen and "Khomeini" in seen)


# =========================================================================== #
# (2) wrong-person Commons name-search hit -> None  +  (6) provenance honest
# =========================================================================== #
def test_commons_wrong_person_rejected():
    _reset_globals()
    d = _tmp() / "p.jpg"

    def no_lead(name, dest):
        return None, {"person": name, "name_match": None,
                      "fallback_reason": "no wikipedia lead image"}

    def wrong_commons(queries, dest, variant=0):
        _fake_jpeg(dest)
        footage._LAST_WIKIMEDIA_TITLE = "Mohammad Sadeghi Tehrani portrait"
        return True

    with patched(_BASE_ENV + [
        (footage, "_wikipedia_lead_portrait", no_lead),
        (footage, "_wikimedia_image", wrong_commons),
        (footage, "_looks_like_portrait", lambda p: (True, "grayscale_portrait")),
        (pintel, "prefers_artwork", lambda *a, **k: False),
    ]):
        out = footage._real_person_image("Ayatollah Khomeini", "cleric", d,
                                         cache_dir=d.parent)
    check("wrong-person Commons face -> None (never baked)", out is None)
    check("rejected candidate file removed", not d.exists())
    prov = footage._PORTRAIT_PROVENANCE["Ayatollah Khomeini"]
    check("rejection provenance identity_verified not True",
          prov.get("identity_verified") in (False, None))
    check("rejection provenance claims no real source",
          prov.get("source") is None)


def test_commons_correct_person_accepted():
    """Same Commons path, but the captured title NAMES the person → allowed.
    Proves the gate is identity-based, not a blanket Commons ban."""
    _reset_globals()
    d = _tmp() / "p.jpg"

    def no_lead(name, dest):
        return None, {"person": name, "name_match": None, "fallback_reason": "x"}

    def right_commons(queries, dest, variant=0):
        _fake_jpeg(dest)
        footage._LAST_WIKIMEDIA_TITLE = "Ruhollah Khomeini official portrait"
        return True

    with patched(_BASE_ENV + [
        (footage, "_wikipedia_lead_portrait", no_lead),
        (footage, "_wikimedia_image", right_commons),
        (footage, "_looks_like_portrait", lambda p: (True, "grayscale_portrait")),
        (pintel, "prefers_artwork", lambda *a, **k: False),
    ]):
        out = footage._real_person_image("Khomeini", "cleric", d, cache_dir=d.parent)
    check("correct-person Commons accepted", out == str(d))
    prov = footage._PORTRAIT_PROVENANCE["Khomeini"]
    check("accepted identity_verified True", prov["identity_verified"] is True)
    check("accepted name_match >= 0.5", prov["name_match"] >= 0.5)


def test_commons_untitled_rejected_for_named():
    """No capturable title → a NAMED person must not use that source."""
    _reset_globals()
    d = _tmp() / "p.jpg"

    def no_lead(name, dest):
        return None, {"person": name, "name_match": None, "fallback_reason": "x"}

    def untitled_commons(queries, dest, variant=0):
        _fake_jpeg(dest)
        footage._LAST_WIKIMEDIA_TITLE = None     # could not capture a title
        return True

    with patched(_BASE_ENV + [
        (footage, "_wikipedia_lead_portrait", no_lead),
        (footage, "_wikimedia_image", untitled_commons),
        (footage, "_looks_like_portrait", lambda p: (True, "grayscale_portrait")),
        (pintel, "prefers_artwork", lambda *a, **k: False),
    ]):
        out = footage._real_person_image("Some Named Person", "leader", d,
                                         cache_dir=d.parent)
    check("untitled Commons hit -> None for named person", out is None)


def test_unverified_env_override():
    """VIDLORE_UNVERIFIED_PORTRAIT=1 re-allows the legacy name-search face
    (escape hatch, default OFF). Provenance stays honest (NOT verified)."""
    _reset_globals()
    d = _tmp() / "p.jpg"

    def no_lead(name, dest):
        return None, {"person": name, "name_match": None, "fallback_reason": "x"}

    def wrong_commons(queries, dest, variant=0):
        _fake_jpeg(dest)
        footage._LAST_WIKIMEDIA_TITLE = "Wrong Person portrait"
        return True

    with patched(_BASE_ENV + [
        (os.environ, "VIDLORE_UNVERIFIED_PORTRAIT", "1"),
        (footage, "_wikipedia_lead_portrait", no_lead),
        (footage, "_wikimedia_image", wrong_commons),
        (footage, "_looks_like_portrait", lambda p: (True, "grayscale_portrait")),
        (pintel, "prefers_artwork", lambda *a, **k: False),
    ]):
        out = footage._real_person_image("Ayatollah Khomeini", "cleric", d,
                                         cache_dir=d.parent)
    check("override accepts unverified face", out == str(d))
    prov = footage._PORTRAIT_PROVENANCE["Ayatollah Khomeini"]
    check("override provenance still identity_verified False",
          prov["identity_verified"] is False)


def test_prephoto_artwork_preserved():
    """Pre-photographic figure (Napoleon) keeps the curated-artwork Commons
    path even when the painting title lacks the surname — provenance honest."""
    _reset_globals()
    d = _tmp() / "p.jpg"

    def no_lead(name, dest):
        return None, {"person": name, "name_match": None, "fallback_reason": "x"}

    def art_commons(queries, dest, variant=0):
        _fake_jpeg(dest)
        footage._LAST_WIKIMEDIA_TITLE = "The Coronation of 1804 (detail)"
        return True

    with patched(_BASE_ENV + [
        (footage, "_wikipedia_lead_portrait", no_lead),
        (footage, "_wikimedia_image", art_commons),
        (footage, "_looks_like_portrait", lambda p: (True, "portrait_structure")),
        (pintel, "prefers_artwork", lambda *a, **k: True),
        (pintel, "portrait_queries", lambda *a, **k: ["Napoleon portrait painting"]),
    ]):
        out = footage._real_person_image("Napoleon Bonaparte", "emperor", d,
                                         cache_dir=d.parent)
    check("pre-photographic curated artwork still works", out == str(d))
    prov = footage._PORTRAIT_PROVENANCE["Napoleon Bonaparte"]
    check("artwork provenance honest (identity_verified False)",
          prov["identity_verified"] is False)
    check("artwork fallback_reason notes artwork",
          "artwork" in str(prov.get("fallback_reason")).lower())


# =========================================================================== #
# (3) None == monogram-fallback signal   +   (5) all-missing -> None no crash
# =========================================================================== #
def test_all_sources_missing_returns_none():
    _reset_globals()
    d = _tmp() / "p.jpg"

    def no_lead(name, dest):
        return None, {"person": name, "name_match": None,
                      "fallback_reason": "no wikipedia lead image"}

    with patched(_BASE_ENV + [
        (footage, "_wikipedia_lead_portrait", no_lead),
        (footage, "_wikimedia_image", lambda *a, **k: False),
        (pintel, "prefers_artwork", lambda *a, **k: False),
    ]):
        out = footage._real_person_image("Nonexistent Person", "leader", d,
                                         cache_dir=d.parent)
    check("all sources missing -> None (no crash)", out is None)


def test_none_is_monogram_signal():
    """Real caller `resolve_legend_portrait`: None from `_real_person_image`
    must degrade to monogram_pedestal with portrait_path=None — documenting the
    None contract callers rely on."""
    _reset_globals()
    d = _tmp() / "p.jpg"
    footage._PORTRAIT_PROVENANCE["Jane Doe"] = {
        "person": "Jane Doe", "source": None,
        "identity_verified": False, "fallback_reason": "name mismatch"}

    with patched([(footage, "_real_person_image", lambda *a, **k: None)]):
        out = footage.resolve_legend_portrait("Jane Doe", role="agent", dest=d,
                                              allow_ai=False)
    check("monogram: portrait_path is None", out["portrait_path"] is None)
    check("monogram: source_type monogram_pedestal",
          out["portrait_source_type"] == "monogram_pedestal")
    check("monogram: fallback reason surfaced",
          bool(out["portrait_fallback_reason"]))


# =========================================================================== #
# FIX 2 — (7) classified card header is NEUTRAL (never fabricated)
# =========================================================================== #
def test_classified_header_source_neutral():
    src = inspect.getsource(footage._render_classified_dossier_card)
    check("fabricated 'DEPARTMENT OF INTERNAL AFFAIRS' removed",
          "DEPARTMENT OF INTERNAL AFFAIRS" not in src)
    check("neutral 'CLASSIFIED BRIEFING' present", "CLASSIFIED BRIEFING" in src)


def test_classified_card_renders_neutral_header():
    """Render the card with NO agency and capture every drawn string; assert the
    neutral header is baked and the fabricated agency is absent. Skips only if
    the heavyweight renderer can't run in this environment (fonts/PIL)."""
    try:
        from PIL import ImageDraw
        from vidlore.script_gen import Scene
    except Exception as e:                                     # pragma: no cover
        print("  skip  classified render (env):", e)
        return

    drawn = []
    orig_text = ImageDraw.ImageDraw.text

    def spy_text(self, xy, text, *a, **k):
        with contextlib.suppress(Exception):
            drawn.append(str(text))
        return orig_text(self, xy, text, *a, **k)

    sc = Scene(index=1, narration="A secret file surfaces.",
               keywords=["classified", "dossier"])
    d = _tmp() / "card.jpg"
    theme = {"accent": "#d4c396", "bg": "#0e0e12"}
    try:
        with patched([(ImageDraw.ImageDraw, "text", spy_text)]):
            ok = footage._render_classified_dossier_card(
                sc, theme, d, file_no="7595-DA",
                title="OPERATION X", body="Details withheld.")
    except Exception as e:                                     # pragma: no cover
        print("  skip  classified render (renderer):", e)
        return

    blob = " ".join(drawn).upper()
    check("rendered header has NO fabricated agency",
          "DEPARTMENT OF INTERNAL AFFAIRS" not in blob)
    if ok:
        check("rendered header is the neutral CLASSIFIED BRIEFING",
              "CLASSIFIED BRIEFING" in blob)
    else:                                                      # pragma: no cover
        print("  note  renderer returned False; source-level check already passed")


if __name__ == "__main__":
    test_name_variants()
    test_page_verified_allowed()
    test_tier1_retries_over_variant()
    test_commons_wrong_person_rejected()
    test_commons_correct_person_accepted()
    test_commons_untitled_rejected_for_named()
    test_unverified_env_override()
    test_prephoto_artwork_preserved()
    test_all_sources_missing_returns_none()
    test_none_is_monogram_signal()
    test_classified_header_source_neutral()
    test_classified_card_renders_neutral_header()
    print("\nALL %d CHECKS PASSED" % _passed)
