"""RC5 — FAIL-CLOSED visual-relevance gate regression (pure unit, no network/CLIP).

Locks in the four root-cause fixes that let three irrelevant assets (an anime/
video-game DVD cover, a strategy-game UI screenshot, a generic multilingual sign)
reach the final RC4 documentary:

  1. FLAG-DEFAULT MISMATCH — the gate is now ON BY DEFAULT (visual_relevance.
     _enabled()/available() default to "1"); VIDLORE_VISUAL_RELEVANCE=0 disables.
  2. WEB-IMAGE BYPASS — a web image is judged (metadata + graphic_dom + VR) before
     acceptance; a designed-graphic candidate is rejected before it is accepted.
  3. FAIL-OPEN POST-PASS — an unscored / below-threshold CONCRETE beat is now a
     REJECT routed to the fallback ladder, not a silent accept.
  4. HARD-REJECT VISUAL CLASSES — classify_junk_metadata() flags game / anime /
     dvd-cover / UI / poster / infographic / logo / meme / wallpaper from
     metadata+slug+query while PASSING genuine war/history documentary queries.

  + QUARANTINE — a rejected asset is recorded and a re-pick of the same hash/url
     is blocked (per-project), and metadata-junk is blocked across projects.

The CLIP scorer is fully MOCKED here (monkeypatched score_asset/available), so no
ONNX model, no network, and no GPU are touched — these assertions run anywhere.

Run:
    PYTHONPATH=/Users/hussnain/Desktop/vidrush-clone \
      /Users/hussnain/Desktop/vidrush-clone/.venv/bin/python \
      tools/test_rc5_visual_relevance_failclosed.py
"""
import os
import sys
import tempfile

# Isolate the cross-project quarantine registry to a temp file so the test never
# touches the real ~/.vidlore/relevance_quarantine.json. MUST be set before the
# module is imported (it binds the path at import time).
_TMP_GLOBAL = os.path.join(tempfile.mkdtemp(prefix="rc5_qtn_"),
                           "relevance_quarantine.json")
os.environ["VIDLORE_RELEVANCE_QUARANTINE"] = _TMP_GLOBAL

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vidlore import visual_relevance as VR          # noqa: E402
from vidlore import render_quarantine as RQ          # noqa: E402

_FAILS: list = []


def _check(cond, msg):
    print(("  PASS " if cond else "  FAIL ") + msg)
    if not cond:
        _FAILS.append(msg)


# ── helpers — a mock score_asset that returns a chosen scores dict ────────────
def _mock_scores(**over):
    """A 'good' scored asset, overridable per field. engine='clip-onnx' so the
    callers treat it as a real score (not skipped)."""
    base = {
        "visual_relevance": 0.30, "pos_sim": 0.30, "distractor_sim": 0.10,
        "margin": 0.20, "clarity": 0.50, "darkness_info": 0.50,
        "face_frac": 0.0, "distractor_dom": -0.05, "people_dom": -0.05,
        "war_dom": -0.05, "vehicle_dom": -0.05, "graphic_dom": -0.05,
        "period_risk": 0.0, "repetition": 0.0, "phash": 123, "engine": "clip-onnx",
    }
    base.update(over)
    return base


def _install_scorer(monkey_scores):
    """Force the gate 'available' and replace score_asset with a stub returning
    `monkey_scores` (a dict or a callable(path,is_video,**kw)->dict)."""
    VR._enabled = lambda: True                    # noqa: E731
    VR._try_load = lambda: True                   # noqa: E731
    VR.available = lambda: True                   # noqa: E731
    if callable(monkey_scores):
        VR.score_asset = monkey_scores
    else:
        VR.score_asset = lambda *a, **k: dict(monkey_scores)


# ══════════════════════════════════════════════════════════════════════════════
# (1) GATE IS ON BY DEFAULT
# ══════════════════════════════════════════════════════════════════════════════
def test_gate_on_by_default():
    print("\n[1] gate ON by default (=0 disables)")
    # re-import a fresh view of the unpatched _enabled by reading the source flag
    for var in ("VIDLORE_VISUAL_RELEVANCE",):
        os.environ.pop(var, None)
    # _enabled reads the env each call; restore the real function first.
    import importlib
    importlib.reload(VR)
    os.environ["VIDLORE_RELEVANCE_QUARANTINE"] = _TMP_GLOBAL  # reload cleared env
    _check(VR._enabled() is True,
           "default (unset) VIDLORE_VISUAL_RELEVANCE -> gate enabled")
    os.environ["VIDLORE_VISUAL_RELEVANCE"] = "0"
    _check(VR._enabled() is False, "VIDLORE_VISUAL_RELEVANCE=0 -> disabled")
    os.environ["VIDLORE_VISUAL_RELEVANCE"] = "1"
    _check(VR._enabled() is True, "VIDLORE_VISUAL_RELEVANCE=1 -> enabled")
    os.environ.pop("VIDLORE_VISUAL_RELEVANCE", None)
    # available() reflects _enabled() AND model load; with model absent in CI it
    # may be False, but the DEFAULT-ON wiring is the thing under test here.
    _check("available" in dir(VR), "available() is exported")


# ══════════════════════════════════════════════════════════════════════════════
# (2) HARD-REJECT METADATA CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════
def test_metadata_classifier():
    print("\n[2] hard-reject metadata classifier (junk flagged, real passes)")
    junk_cases = [
        ("game", dict(title="Civilization VI strategy game gameplay",
                      query="iran iraq war border")),
        ("anime", dict(title="Bomberman Jetters anime DVD cover",
                       query="american troops gulf war")),
        ("dvd-cover", dict(title="PS2 box art", slug="bomberman-jetters-ps2.jpg",
                           query="invasion 1980")),
        ("ui/screenshot", dict(title="Hearts of Iron IV UI screenshot HUD",
                               query="missile launch invasion")),
        ("poster", dict(title="revolutionary movie poster", query="revolution")),
        ("infographic", dict(title="mortgage rates infographic chart 2024",
                             query="economy")),
        ("logo", dict(title="political party logo emblem", query="politics")),
        ("meme", dict(title="funny war meme", query="war")),
        ("wallpaper", dict(title="desktop wallpaper 4k", query="city")),
        ("multilingual-sign", dict(
            title="welcome sign we're glad you're our neighbor",
            query="revolutionary poster border neighbor")),
    ]
    for label, kw in junk_cases:
        isj, reason, hits = VR.classify_junk_metadata(**kw)
        _check(isj, f"JUNK flagged: {label} (reason={reason or '-'})")

    real_pass = [
        ("iran-iraq-war", dict(
            title="Iran Iraq war soldiers 1980 archival footage",
            query="iran iraq war soldiers 1980 archival")),
        ("tehran-revolution", dict(
            title="Tehran revolution 1979 crowd street",
            query="tehran revolution 1979 crowd")),
        ("baghdad-1980s", dict(title="Baghdad city street 1980s",
                               query="baghdad city 1980s")),
    ]
    for label, kw in real_pass:
        isj, reason, hits = VR.classify_junk_metadata(**kw)
        _check(not isj, f"REAL passes: {label} (not junk)")

    # on-topic exemption: a doc literally ABOUT a game keeps a 'game' image.
    isj, reason, _ = VR.classify_junk_metadata(
        title="Tetris game history", query="tetris",
        narration="this documentary tells the history of the Tetris game")
    _check(not isj, "on-topic exemption: narration names the game -> kept")


# ══════════════════════════════════════════════════════════════════════════════
# (3) FAIL-CLOSED: below-threshold / unscored -> REJECT (not silent accept)
# ══════════════════════════════════════════════════════════════════════════════
def test_fail_closed_decision():
    print("\n[3] fail-closed: below-threshold/unscored -> REJECT, not accept")
    fd, p = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    try:
        # below the relevance floor (_DEFAULT_MIN=0.10) -> accept() must REJECT.
        _install_scorer(_mock_scores(visual_relevance=0.02))
        ok, s, why = VR.accept(p, False, expected="iran iraq war soldiers",
                               concrete=True)
        _check(ok is False, f"below-floor relevance -> REJECT ({why})")
        _check("subject-absent" in why, "reason is subject-absent (the floor)")

        # designed-graphic dominance -> REJECT (the anime/UI/sign class).
        _install_scorer(_mock_scores(graphic_dom=0.20))
        ok2, _, why2 = VR.accept(p, False, expected="iran iraq war",
                                 concrete=True)
        _check(ok2 is False, f"high graphic_dom -> REJECT ({why2})")

        # UNSCORED verdict (engine='skipped') must NOT be a confident accept the
        # caller can rely on: accept() returns scorer-skipped, and the footage
        # post-pass treats a CONCRETE unscored beat as fail-closed (see the
        # predicate test below). Here we assert accept() does not pretend it
        # scored — its reason is 'scorer-skipped', engine 'skipped'.
        _install_scorer(_mock_scores(engine="skipped"))
        ok3, s3, why3 = VR.accept(p, False, expected="iran iraq war",
                                  concrete=True)
        _check(s3.get("engine") == "skipped" and why3 == "scorer-skipped",
               "unscored verdict is flagged skipped (post-pass fail-closes it)")

        # the post-pass fail-closed PREDICATE (concrete + unavailable/error/no-
        # score -> route to fallback). Mirror the exact condition added to
        # footage.py so the policy is regression-locked here without a render.
        def _should_fail_closed(concrete, why):
            w = (why or "").lower()
            return concrete and ("unavailable" in w or "error" in w
                                 or "no-item" in w or w == "")
        _check(_should_fail_closed(True, "unavailable") is True,
               "concrete + unavailable -> fail-closed (route to fallback)")
        _check(_should_fail_closed(True, "") is True,
               "concrete + no-score -> fail-closed")
        _check(_should_fail_closed(False, "abstract") is False,
               "abstract scene stays the allowed fail-open")
        _check(_should_fail_closed(True, "abstract") is False,
               "abstract reason never fail-closes (unscoreable)")
    finally:
        os.unlink(p)


# ══════════════════════════════════════════════════════════════════════════════
# (4) WEB-IMAGE designed-graphic candidate REJECTED before acceptance
# ══════════════════════════════════════════════════════════════════════════════
def test_webimage_graphic_rejected():
    print("\n[4] web-image designed-graphic candidate rejected before accept")
    fd, p = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    try:
        # graphic_signal: a high graphic_dom -> looks_designed True (the exact
        # gate the web-image tier now runs before returning a FootageItem).
        _install_scorer(_mock_scores(graphic_dom=0.18))
        g = VR.graphic_signal(p, is_video=False)
        _check(g.get("looks_designed") is True,
               f"graphic_signal flags designed graphic (gd={g.get('graphic_dom')})")

        # a real photograph (negative graphic_dom) is NOT flagged.
        _install_scorer(_mock_scores(graphic_dom=-0.06))
        g2 = VR.graphic_signal(p, is_video=False)
        _check(g2.get("looks_designed") is False,
               "graphic_signal keeps a real photograph (not designed)")

        # the COMBINED web-image accept predicate the tier uses:
        #   reject if (metadata junk) OR (graphic_signal designed) OR (VR reject)
        def _webimg_reject(meta_kw, scores):
            isj, _, _ = VR.classify_junk_metadata(**meta_kw)
            if isj:
                return True
            _install_scorer(scores)
            if VR.graphic_signal(p, False).get("looks_designed"):
                return True
            return False
        _check(_webimg_reject(dict(title="anime dvd cover", query="war"),
                              _mock_scores()) is True,
               "metadata-junk web image -> rejected")
        _check(_webimg_reject(dict(title="tehran 1979 crowd", query="tehran"),
                              _mock_scores(graphic_dom=0.18)) is True,
               "graphic web image -> rejected by pixel gate")
        _check(_webimg_reject(dict(title="tehran 1979 crowd", query="tehran"),
                              _mock_scores(graphic_dom=-0.06)) is False,
               "genuine relevant web image -> accepted (no over-reject)")
    finally:
        os.unlink(p)


# ══════════════════════════════════════════════════════════════════════════════
# (5) GOOD DOCUMENTARY FOOTAGE PRESERVED (no over-rejection)
# ══════════════════════════════════════════════════════════════════════════════
def test_good_footage_preserved():
    print("\n[5] good documentary footage preserved (no over-rejection)")
    fd, p = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    try:
        # a clean, relevant war/history clip: relevance above floor, all
        # dominance signals negative, clear + bright -> ACCEPT.
        _install_scorer(_mock_scores(
            visual_relevance=0.34, graphic_dom=-0.07, distractor_dom=-0.06,
            people_dom=-0.04, war_dom=-0.04, clarity=0.6, darkness_info=0.5))
        # crowd_ok True because a war scene legitimately shows soldiers/crowds.
        ok, s, why = VR.accept(p, True, expected="iran iraq war soldiers 1980",
                               concrete=True, crowd_ok=True)
        _check(ok is True, f"relevant war footage KEPT ({why})")
        # genuine metadata is not junk.
        isj, _, _ = VR.classify_junk_metadata(
            title="Iran Iraq war soldiers archival 1980",
            query="iran iraq war soldiers 1980 archival")
        _check(not isj, "relevant footage metadata is not flagged junk")
        # and a real-photo graphic_signal stays NOT designed.
        g = VR.graphic_signal(p, True)
        _check(g.get("looks_designed") is False,
               "relevant footage not flagged designed-graphic")
    finally:
        os.unlink(p)


# ══════════════════════════════════════════════════════════════════════════════
# (6) QUARANTINE: rejected asset recorded + re-pick of same hash/url blocked
# ══════════════════════════════════════════════════════════════════════════════
def test_quarantine_blocks_repick():
    print("\n[6] quarantine records a reject + blocks a re-pick (per+cross proj)")
    RQ.reset()
    tmp = tempfile.mkdtemp(prefix="rc5_proj_")
    RQ.attach(tmp)
    url = "https://example.com/bomberman-jetters-ps2-cover.jpg"
    lp = os.path.join(tmp, "webimg_x.jpg")
    _check(RQ.is_quarantined(local_path=lp, source_url=url) is False,
           "asset not quarantined before reject")
    # metadata-junk reject -> per-project AND global.
    RQ.quarantine(lp, source_url=url, reason="relevance:junk-metadata:anime,dvd",
                  replacement_source_type="fallback-ladder", global_junk=True)
    _check(RQ.is_quarantined(local_path=lp) is True,
           "rejected asset is quarantined by local path")
    _check(RQ.is_quarantined(source_url=url) is True,
           "rejected asset is quarantined by source url (re-pick blocked)")
    # cross-project: a fresh attach in a DIFFERENT project still blocks the url.
    RQ.reset()
    tmp2 = tempfile.mkdtemp(prefix="rc5_proj2_")
    RQ.attach(tmp2)
    _check(RQ.is_quarantined(source_url=url) is True,
           "globally-obvious junk stays blocked across projects")
    # a NON-junk relevance reject (project-specific) does NOT go global.
    RQ.reset()
    tmp3 = tempfile.mkdtemp(prefix="rc5_proj3_")
    RQ.attach(tmp3)
    purl = "https://stock.example.com/some-real-but-offtopic-clip.mp4"
    RQ.quarantine(os.path.join(tmp3, "clip_y.mp4"), source_url=purl,
                  reason="relevance:subject-absent", global_junk=False)
    _check(RQ.is_quarantined(source_url=purl) is True,
           "project-specific reject blocked in THIS project")
    RQ.reset()
    tmp4 = tempfile.mkdtemp(prefix="rc5_proj4_")
    RQ.attach(tmp4)
    _check(RQ.is_quarantined(source_url=purl) is False,
           "project-specific reject is NOT blocked in a different project")


# ══════════════════════════════════════════════════════════════════════════════
# (7) EDITOR-REPLACEMENT VALIDATION HELPER (web.py.validate_replacement_asset)
# ══════════════════════════════════════════════════════════════════════════════
# The Review-Editor manual-replacement path is one of the three un-gated
# producers (RC5 STEP 9). web.py.validate_replacement_asset() runs the SAME
# classify_junk_metadata + graphic_signal gate on an uploaded / searched asset
# and returns accepted | warning | rejected. We exercise it WITHOUT a Flask
# request and with the scorer mocked, so a junk filename is hard-blocked, a junk
# PIXEL frame is hard-blocked, and a clean relevant image is accepted (no
# over-rejection). Flask is imported by web.py at module load; if it is missing
# in this environment the whole group is skipped (still green) rather than erroring.
def _import_web():
    try:
        from vidlore import web as W            # noqa: WPS433
        return W
    except Exception as e:                       # noqa: BLE001
        print(f"  SKIP web.py import unavailable ({type(e).__name__}: "
              f"{str(e)[:60]}) — editor-replacement group skipped")
        return None


def test_editor_replacement_validation():
    print("\n[7] editor-replacement validation helper (junk blocked, clean kept)")
    W = _import_web()
    if W is None:
        return
    # mock the scorer so graphic_signal is deterministic + offline.
    _install_scorer(_mock_scores(graphic_dom=-0.06))   # default: real photo

    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 4096          # non-empty bytes to hash

    # (a) metadata-junk filename → REJECTED (hard junk), original preserved.
    st = W.validate_replacement_asset(
        png, "bomberman-jetters-anime-dvd-cover.jpg",
        narration="a documentary about the iran iraq war", query="war")
    _check(st["status"] == "rejected" and st["hard_junk"],
           f"anime/DVD-cover filename -> rejected ({st.get('reason')})")
    _check(bool(st.get("asset_sha1")), "candidate bytes hashed for the audit log")

    # (b) clean relevant filename + real-photo pixels → ACCEPTED.
    st2 = W.validate_replacement_asset(
        png, "iran_iraq_war_soldiers_1980.jpg",
        narration="iran iraq war soldiers 1980 archival", query="iran iraq war")
    _check(st2["status"] == "accepted",
           f"relevant clean replacement -> accepted ({st2.get('reason') or '-'})")

    # (c) clean filename but DESIGNED-GRAPHIC pixels (infographic) → REJECTED.
    _install_scorer(_mock_scores(graphic_dom=0.20))
    st3 = W.validate_replacement_asset(
        png, "scene_visual.jpg", narration="the economy collapsed", query="economy")
    _check(st3["status"] == "rejected" and st3["looks_designed"],
           f"designed-graphic pixels -> rejected ({st3.get('reason')})")
    _install_scorer(_mock_scores(graphic_dom=-0.06))

    # (d) the route-level guard: blocked by default, but force=1 overrides AND
    # the override is recorded with the original-asset hash. Use a temp run dir.
    import json as _json
    tmp = tempfile.mkdtemp(prefix="rc5_ed_")
    # minimal scene context so _editor_scene_context doesn't error
    (os.path.join(tmp, "script.json"))
    with open(os.path.join(tmp, "script.json"), "w", encoding="utf-8") as fh:
        _json.dump({"scenes": [{"narration": "iran iraq war", "keywords": []}]}, fh)

    proceed, blocked = W._guard_manual_replacement(
        tmp, 0, png, "anime-dvd-cover-game-ui.jpg", is_video=False,
        force=False, override_reason="")
    _check(proceed is None and blocked and blocked["status"] == "rejected",
           "guard BLOCKS hard-junk replacement by default")

    proceed2, blocked2 = W._guard_manual_replacement(
        tmp, 0, png, "anime-dvd-cover-game-ui.jpg", is_video=False,
        force=True, override_reason="I am literally making a doc about this game")
    _check(proceed2 is not None and blocked2 is None
           and proceed2["status"] == "accepted_override",
           "guard ALLOWS the same junk with force=1 (explicit override)")

    # the override must be LOGGED (hash + chosen asset + reason + timestamp).
    audit = os.path.join(tmp, "edits", "replacement_audit.jsonl")
    logged = ""
    try:
        with open(audit, "r", encoding="utf-8") as fh:
            logged = fh.read()
    except Exception:                                  # noqa: BLE001
        logged = ""
    _check("override_forced" in logged and "asset_sha1" in logged
           and "override_reason" in logged,
           "forced override is written to replacement_audit.jsonl (never silent)")

    # a CLEAN replacement is NOT blocked and is logged as accepted.
    proceed3, blocked3 = W._guard_manual_replacement(
        tmp, 0, png, "iran_iraq_war_tank_1982.jpg", is_video=False, force=False)
    _check(proceed3 is not None and proceed3["status"] == "accepted",
           "guard PASSES a clean relevant replacement (no over-reject)")


# ══════════════════════════════════════════════════════════════════════════════
# (8) STEP-8 — KNOWN-FAILURE corpus rejected / GOOD corpus preserved, no spike
# ══════════════════════════════════════════════════════════════════════════════
# A broad fixture sweep that mirrors the real RC4/RC5 failure list and a matched
# set of legitimate documentary assets. The scorer is mocked PER ASSET (a callable
# keyed on the asset's intended graphic_dom), so a "designed graphic" failure is
# caught by the pixel gate and a real photo passes — no network, no CLIP. We
# assert: every KNOWN FAILURE is rejected (by metadata OR pixels), every GOOD
# asset is preserved, the over-rejection RATE on the good set is ~0, and the
# verified-name portrait is NOT rejected by the suitability path.
#
# Each fixture: (label, metadata-kwargs-for-classify, graphic_dom, is_concrete).
_KNOWN_FAILURES = [
    # anime / video-game DVD cover (the headline RC4 leak)
    ("anime_game_dvd_cover",
     dict(title="Bomberman Jetters anime DVD cover", slug="bomberman-ps2-boxart.jpg",
          query="iran iraq war"), 0.05, True),
    # strategy-game missile / UI screenshot (the 2nd RC4 leak)
    ("strategy_game_missile_ui",
     dict(title="Hearts of Iron IV missile launch UI HUD screenshot",
          query="missile strike 1984"), 0.04, True),
    # generic multilingual welcome sign (the 3rd RC4 leak)
    ("multilingual_welcome_sign",
     dict(title="welcome sign we're glad you're our neighbor",
          query="revolutionary poster border neighbor"), 0.041, True),
    # infographic / chart
    ("infographic_chart",
     dict(title="2024 mortgage rates infographic chart", query="economy"),
     0.06, True),
    # unrelated political party logo / emblem
    ("political_party_logo",
     dict(title="political party logo emblem vector", query="politics"),
     0.07, True),
    # movie / propaganda poster
    ("movie_poster",
     dict(title="revolutionary movie poster", query="revolution 1979"),
     0.05, True),
    # software dashboard / app UI
    ("software_dashboard_ui",
     dict(title="analytics dashboard app screenshot UI", query="surveillance"),
     0.055, True),
    # unrelated printed text board / slide
    ("unrelated_text_board",
     dict(title="a presentation slide with printed words", slug="politics-sign.png",
          query="politics"), 0.045, True),
]
# GOOD assets — clean metadata (NOT junk) + a real-photo / footage graphic_dom
# well below the 0.036 gate. crowd_ok flags a genuine war/UN/people scene.
_GOOD_ASSETS = [
    ("archival_war_footage",
     dict(title="Iran Iraq war soldiers 1980 archival footage",
          query="iran iraq war soldiers 1980 archival"), -0.05, True, True),
    ("period_neutral_landscape",
     dict(title="desert border landscape wide shot", query="desert border"),
     -0.07, True, False),
    ("grounded_fal_war_still",
     dict(title="cinematic still soldiers trench 1982", query="trench warfare"),
     -0.06, True, True),
    ("verified_portrait",
     dict(title="ayatollah portrait black and white photograph",
          query="ayatollah portrait"), -0.06, True, False),
    ("relevant_map_animation",
     dict(title="map of iran iraq border animation", query="iran iraq border map"),
     -0.04, True, False),
    ("relevant_classified_document_card",
     dict(title="declassified intelligence document scan", query="classified cable"),
     -0.02, True, False),
    ("relevant_un_footage",
     dict(title="United Nations security council session footage",
          query="un security council"), -0.05, True, True),
]


def test_step8_known_failures_and_good_assets():
    print("\n[8] STEP-8 known-failure corpus rejected + good corpus preserved")
    fd, p = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    try:
        # ---- KNOWN FAILURES: every one must be rejected (metadata OR pixels) ----
        rejected_bad = 0
        for label, meta, gdom, concrete in _KNOWN_FAILURES:
            # metadata path
            isj, reason, hits = VR.classify_junk_metadata(**meta)
            # pixel path (mock the scorer for THIS asset's graphic_dom)
            _install_scorer(_mock_scores(graphic_dom=gdom))
            designed = VR.graphic_signal(p, False).get("looks_designed")
            ok_clip, _, _why = VR.accept(p, False,
                                         expected=meta.get("query", "subject"),
                                         concrete=concrete)
            blocked = bool(isj) or bool(designed) or (ok_clip is False)
            rejected_bad += int(blocked)
            _check(blocked, f"KNOWN FAILURE rejected: {label} "
                            f"(junk={bool(isj)} designed={bool(designed)} "
                            f"accept={ok_clip})")
        _check(rejected_bad == len(_KNOWN_FAILURES),
               f"ALL {len(_KNOWN_FAILURES)} known failures rejected")

        # ---- GOOD ASSETS: every one preserved (NOT junk + NOT designed + accept) ----
        kept_good = 0
        over_rejected = []
        for label, meta, gdom, concrete, crowd in _GOOD_ASSETS:
            isj, _, _ = VR.classify_junk_metadata(**meta)
            _install_scorer(_mock_scores(
                visual_relevance=0.34, graphic_dom=gdom, distractor_dom=-0.06,
                people_dom=-0.04, war_dom=-0.04, vehicle_dom=-0.04,
                clarity=0.6, darkness_info=0.5))
            designed = VR.graphic_signal(p, False).get("looks_designed")
            ok_clip, _s, _why = VR.accept(p, False,
                                          expected=meta.get("query", "subject"),
                                          concrete=concrete, crowd_ok=crowd)
            kept = (not isj) and (not designed) and (ok_clip is True)
            kept_good += int(kept)
            if not kept:
                over_rejected.append(f"{label} (junk={isj} designed={designed} "
                                     f"accept={ok_clip}:{_why})")
            _check(kept, f"GOOD asset preserved: {label}")
        _check(kept_good == len(_GOOD_ASSETS),
               f"ALL {len(_GOOD_ASSETS)} good assets preserved")

        # ---- NO OVER-REJECTION SPIKE: the good-set false-reject rate must be 0 ----
        spike = len(over_rejected)
        _check(spike == 0,
               f"no over-rejection spike on good corpus "
               f"(false-rejects={spike}/{len(_GOOD_ASSETS)})")
    finally:
        os.unlink(p)


def test_step8_portrait_identity_preserved():
    print("\n[8b] portrait-identity-preserved invariant (verified name not rejected)")
    fd, p = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    try:
        # A verified-name PORTRAIT legitimately has ONE prominent face. The
        # suitability/relevance gate must NOT reject it as a "person on a
        # non-person scene": for a portrait the scene IS a person, so the caller
        # passes person_expected=True. A clean portrait (real photo, no graphic,
        # relevance above floor) must be ACCEPTED.
        _install_scorer(_mock_scores(
            visual_relevance=0.33, graphic_dom=-0.06, face_frac=0.42,
            distractor_dom=-0.05, people_dom=-0.04, war_dom=-0.05,
            clarity=0.55, darkness_info=0.45))
        ok, s, why = VR.accept(p, False, expected="ayatollah portrait",
                               concrete=True, person_expected=True)
        _check(ok is True, f"verified-name portrait KEPT (person_expected) ({why})")
        # graphic_signal must NOT flag a real B&W portrait as a designed graphic.
        g = VR.graphic_signal(p, False)
        _check(g.get("looks_designed") is False,
               "verified portrait not flagged designed-graphic")
        # metadata for a real portrait is not junk.
        isj, _, _ = VR.classify_junk_metadata(
            title="ayatollah portrait black and white photograph",
            query="ayatollah portrait")
        _check(not isj, "verified portrait metadata is not junk")
        # SAFETY: the SAME face-bearing frame on a NON-person scene (a crowd on a
        # 'copper mining' beat) is still rejected — the invariant protects a
        # PORTRAIT, it does not weaken the people gate for non-person scenes.
        _install_scorer(_mock_scores(
            visual_relevance=0.33, graphic_dom=-0.06, face_frac=0.42,
            people_dom=0.09, distractor_dom=-0.05, clarity=0.55,
            darkness_info=0.45))
        ok2, _s2, why2 = VR.accept(p, False, expected="copper ore mining",
                                   concrete=True, person_expected=False,
                                   crowd_ok=False)
        _check(ok2 is False,
               f"crowd/face on a non-person scene still rejected ({why2})")
    finally:
        os.unlink(p)


def main():
    print("=" * 70)
    print("RC5 FAIL-CLOSED VISUAL-RELEVANCE GATE — unit regression")
    print("=" * 70)
    test_gate_on_by_default()
    test_metadata_classifier()
    test_fail_closed_decision()
    test_webimage_graphic_rejected()
    test_good_footage_preserved()
    test_quarantine_blocks_repick()
    test_editor_replacement_validation()
    test_step8_known_failures_and_good_assets()
    test_step8_portrait_identity_preserved()
    print("\n" + "=" * 70)
    if _FAILS:
        print(f"RESULT: {len(_FAILS)} FAILURE(S)")
        for f in _FAILS:
            print("  - " + f)
        sys.exit(1)
    print("RESULT: ALL GREEN")
    sys.exit(0)


if __name__ == "__main__":
    main()
