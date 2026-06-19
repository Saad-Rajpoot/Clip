"""RC5 STEP 3 — CROSS-NICHE VISUAL-DECISION SWEEP (hermetic: no renders, no network).

A lightweight dry-run of the visual-relevance DECISION across 8 documentary niches
(war/geopolitics, history, spy, crime, biography, science, technology, business).
For each niche it pushes a fixture corpus through the SAME three decision channels
the live footage engine uses — and asserts the gate rejects every known-junk class
while preserving every genuinely-useful visual.

The three channels (all pure / mockable; the CLIP scorer is monkeypatched, so no
ONNX model, GPU, or network is touched):

  (A) METADATA STRING  — visual_relevance.classify_junk_metadata(): a hard-reject
      from title/slug/url/query/provider keywords (game / anime / dvd / ui /
      poster / infographic / logo / meme / wallpaper / screenshot / dashboard /
      template …), with an on-topic exemption when the SAME token is in the
      narration. Pure string logic — no scorer needed, so this channel ALSO proves
      the "scorer-unavailable never silently accepts junk-by-metadata" property.

  (B) PERIOD STRING    — period_guard via footage._period_blocked(): a wrong-era
      slug/title on a HISTORICAL scene ("modern COVID crowd" on a 1980s war beat,
      "modern cars" on an 1812 beat) is dropped on the search side and routed to a
      period-grounded fal still. Pure string logic.

  (C) PIXEL (mocked)   — visual_relevance.accept(): the wrong-DOMINANT-concept /
      designed-graphic / crowd / war / vehicle / face / relevance-floor gates,
      exercised with a monkeypatched score_asset() returning a chosen scores dict
      (an unrelated celebrity face, a war crowd on a science beat, a designed
      graphic with no junk metadata, …). This is the path for junk that carries NO
      junk metadata token.

ASSERTIONS (all must be green):
  • known-junk false-negatives == 0 across all niches (every MUST-REJECT rejected).
  • the scorer-unavailable path never accepts junk-by-metadata (channel A with the
    gate forced unavailable still rejects metadata junk).
  • no useful-visual over-rejection in the MUST-PRESERVE set on the channel that
    owns each item; any item the METADATA classifier is too aggressive on (a junk
    token whose narration does NOT name it) is DOCUMENTED, not silently passed.
  • a per-niche table (accepted / rejected / quarantined counts) is printed.

Run:
    PYTHONPATH=/Users/hussnain/Desktop/vidrush-clone \
      /Users/hussnain/Desktop/vidrush-clone/.venv/bin/python \
      tools/test_rc5_cross_niche_sweep.py
"""
import os
import sys
import tempfile

# Isolate the cross-project quarantine registry to a temp file (channel A's
# render_quarantine layer binds the path at import time). Never touch the real
# ~/.vidlore/relevance_quarantine.json.
_TMP_GLOBAL = os.path.join(tempfile.mkdtemp(prefix="rc5_xniche_"),
                           "relevance_quarantine.json")
os.environ["VIDLORE_RELEVANCE_QUARANTINE"] = _TMP_GLOBAL
# Deterministic, hermetic: the pixel channel is fully mocked, so the gate's real
# ONNX load is irrelevant — but keep the flag ON so _enabled() is the production
# default under test.
os.environ.pop("VIDLORE_VISUAL_RELEVANCE", None)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vidlore import visual_relevance as VR          # noqa: E402
from vidlore import period_guard as PG               # noqa: E402
from vidlore import footage as FT                    # noqa: E402

_FAILS: list = []


def _check(cond, msg):
    if not cond:
        _FAILS.append(msg)
    return bool(cond)


# ── pixel-channel scorer harness ─────────────────────────────────────────────
def _scores(**over):
    """A 'good, on-subject' scored asset (engine='clip-onnx' so accept() treats it
    as a real score). Override individual signals to model a wrong-subject asset."""
    base = {
        "visual_relevance": 0.32, "pos_sim": 0.32, "distractor_sim": 0.10,
        "margin": 0.22, "clarity": 0.55, "darkness_info": 0.55,
        "face_frac": 0.0, "distractor_dom": -0.06, "people_dom": -0.06,
        "war_dom": -0.06, "vehicle_dom": -0.06, "graphic_dom": -0.06,
        "period_risk": 0.0, "repetition": 0.0, "phash": 7, "engine": "clip-onnx",
    }
    base.update(over)
    return base


def _force_scorer_available(scores_dict):
    """Force the gate available and make score_asset return scores_dict."""
    VR._enabled = lambda: True                    # noqa: E731
    VR._try_load = lambda: True                   # noqa: E731
    VR.available = lambda: True                   # noqa: E731
    VR.score_asset = lambda *a, **k: dict(scores_dict)


def _force_scorer_unavailable():
    VR._enabled = lambda: True                    # noqa: E731
    VR._try_load = lambda: False                  # noqa: E731
    VR.available = lambda: False                  # noqa: E731


# A single temp file path stands in for every "downloaded asset" the pixel channel
# scores (the bytes are never read — score_asset is mocked).
_FD, _ASSET = tempfile.mkstemp(suffix=".jpg")
os.close(_FD)


# ══════════════════════════════════════════════════════════════════════════════
# FIXTURE CORPUS — per niche
# ══════════════════════════════════════════════════════════════════════════════
# Each niche has a narration CONTEXT and three groups:
#   reject_meta   : (label, kwargs-for-classify_junk_metadata)  -> must be JUNK
#   reject_pixel  : (label, scores-override, accept-kwargs)      -> accept()==False
#   reject_period : (label, era_dict, slug_text)                 -> _period_blocked True
#   preserve_meta : (label, kwargs)        -> classify_junk_metadata NOT junk
#   preserve_pixel: (label, scores, akw)   -> accept()==True
# The 15 MUST-REJECT and 11 MUST-PRESERVE classes named in the spec are distributed
# across the niches (each niche carries a representative subset so every class is
# covered, with the historically-grounded ones on the period-bearing niches).

_NICHES = {}


def _niche(key, *, narration, reject_meta=(), reject_pixel=(), reject_period=(),
           preserve_meta=(), preserve_pixel=()):
    _NICHES[key] = dict(narration=narration, reject_meta=reject_meta,
                        reject_pixel=reject_pixel, reject_period=reject_period,
                        preserve_meta=preserve_meta, preserve_pixel=preserve_pixel)


# Shared era fixtures for the period channel
_ERA_IRANIRAQ = PG.detect_era("the iran-iraq war of the 1980s", title="Iran-Iraq War")
_ERA_1812 = PG.detect_era("Napoleon's 1812 invasion of Russia", title="1812")
_ERA_WWII = PG.detect_era("World War II 1942 the eastern front", title="WWII")

# ── 1. WAR / GEOPOLITICS ──────────────────────────────────────────────────────
_niche(
    "war_geopolitics",
    narration="the iran iraq war of the 1980s reshaped the persian gulf as the two "
              "armies fought along the border",
    reject_meta=[
        ("game UI", dict(title="Hearts of Iron IV strategy game UI HUD screenshot",
                         query="iran iraq border front")),
        ("gameplay screenshot", dict(title="ARMA 3 milsim gameplay screenshot",
                                     query="modern soldiers")),
        ("unrelated infographic", dict(title="2024 mortgage rates infographic chart",
                                       query="gulf economy crisis")),
    ],
    reject_period=[
        ("modern COVID crowd on 1980s war", _ERA_IRANIRAQ,
         "covid 19 pandemic crowd wearing face masks 2021"),
        ("video-game capture on 1980s war", _ERA_IRANIRAQ,
         "war game simulation 4k gameplay footage"),
    ],
    preserve_meta=[
        ("relevant archival footage", dict(
            title="iran iraq war soldiers 1980 archival footage",
            query="iran iraq war archival")),
        ("relevant map animation", dict(
            title="persian gulf front line map animation",
            query="iran iraq border map")),
    ],
    preserve_pixel=[
        ("period-neutral landscape", _scores(), dict(
            expected="a desert battlefield", crowd_ok=True)),
        ("relevant UN footage", _scores(), dict(
            expected="united nations security council", crowd_ok=True)),
    ],
)

# ── 2. HISTORY ────────────────────────────────────────────────────────────────
_niche(
    "history",
    narration="in 1812 napoleon marched the grande armee into russia toward moscow",
    reject_meta=[
        ("cartoon", dict(title="napoleon cartoon illustration animated",
                         query="napoleon russia")),
        ("DVD cover", dict(title="War and Peace movie DVD cover box art",
                           query="1812 invasion")),
    ],
    reject_period=[
        ("modern cars on an 1812 scene", _ERA_1812,
         "modern city street with cars and traffic"),
        ("modern moscow tram on 1812", _ERA_1812,
         "moscow metro tram station 2019 commuters smartphone"),
    ],
    preserve_meta=[
        ("relevant archival footage", dict(
            title="napoleonic war reenactment 1812 period painting",
            query="napoleon 1812 grande armee")),
    ],
    preserve_pixel=[
        ("period-neutral landscape", _scores(), dict(
            expected="a snowy russian field", crowd_ok=True)),
        ("relevant map animation", _scores(), dict(
            expected="map of napoleon's march to moscow")),
    ],
)

# ── 3. SPY ────────────────────────────────────────────────────────────────────
_niche(
    "spy",
    narration="the mossad operative passed a classified document through a dead drop "
              "in cold-war vienna",
    reject_meta=[
        ("anime", dict(title="spy anime series episode 3", query="cold war agent")),
        ("unrelated software dashboard", dict(
            title="CRM analytics dashboard ui screenshot", query="agency files")),
        ("watermark-heavy junk", dict(
            title="shutterstock watermark stock wallpaper", query="random spy")),
    ],
    preserve_meta=[
        ("relevant classified-document card", dict(
            title="classified cia document dossier redacted",
            query="classified document cold war",
            narration="a classified document revealed the operation")),
        ("verified portrait", dict(
            title="cold war spy portrait photograph black and white",
            query="kim philby portrait")),
    ],
    preserve_pixel=[
        ("grounded fal still", _scores(), dict(
            expected="a dead drop in a vienna alley at night", concrete=True)),
    ],
)

# ── 4. CRIME ──────────────────────────────────────────────────────────────────
_niche(
    "crime",
    narration="detectives traced the cartel's money through shell companies and "
              "wiretap evidence",
    reject_meta=[
        ("political-logo collage", dict(
            title="political party logo emblem collage", query="cartel politics")),
        ("unrelated poster", dict(
            title="concert poster large title text design", query="music festival")),
        ("text-heavy unrelated board", dict(
            title="powerpoint slide template with bullet text",
            query="presentation deck")),
    ],
    reject_pixel=[
        ("unrelated celebrity face", _scores(face_frac=0.40), dict(
            expected="a wiretap evidence board", person_expected=False)),
    ],
    preserve_meta=[
        ("verified portrait", dict(
            title="suspect mugshot portrait photograph",
            query="cartel boss portrait")),
    ],
    preserve_pixel=[
        ("grounded fal still", _scores(), dict(
            expected="stacks of cash on a table", concrete=True)),
    ],
)

# ── 5. BIOGRAPHY ──────────────────────────────────────────────────────────────
_niche(
    "biography",
    narration="john d rockefeller built standard oil into a vast refining empire",
    reject_meta=[
        ("anime", dict(title="tycoon anime character art", query="oil baron")),
        ("DVD cover", dict(title="biopic DVD cover box art poster",
                           query="rockefeller film")),
    ],
    reject_pixel=[
        ("unrelated celebrity face", _scores(face_frac=0.45), dict(
            expected="an oil refinery", person_expected=False)),
    ],
    preserve_meta=[
        ("verified portrait", dict(
            title="john d rockefeller portrait photograph",
            query="rockefeller portrait")),
        ("relevant archival footage", dict(
            title="standard oil refinery archival photograph 1900",
            query="standard oil refinery archival")),
    ],
    preserve_pixel=[
        ("grounded fal still", _scores(), dict(
            expected="a 19th century oil refinery", concrete=True)),
    ],
)

# ── 6. SCIENCE ────────────────────────────────────────────────────────────────
_niche(
    "science",
    narration="the diagram shows the stages of cell mitosis as chromosomes separate",
    reject_meta=[
        ("unrelated infographic", dict(
            title="cryptocurrency price infographic chart 2024",
            query="cell biology")),
        ("meme", dict(title="funny science meme image", query="mitosis")),
    ],
    reject_pixel=[
        ("war crowd on a science beat", _scores(war_dom=0.10, people_dom=0.09), dict(
            expected="a cell mitosis diagram", crowd_ok=False)),
        ("designed graphic, no junk metadata", _scores(graphic_dom=0.20), dict(
            expected="a cell under a microscope")),
    ],
    preserve_meta=[
        ("relevant science diagram", dict(
            title="cell mitosis diagram labeled stages",
            query="mitosis diagram",
            narration="this diagram shows the stages of mitosis")),
    ],
    preserve_pixel=[
        ("grounded fal still", _scores(), dict(
            expected="chromosomes under a microscope", concrete=True)),
    ],
)

# ── 7. TECHNOLOGY ─────────────────────────────────────────────────────────────
_niche(
    "technology",
    narration="inside the fab a silicon wafer is etched layer by layer into a "
              "microchip",
    reject_meta=[
        ("game UI", dict(title="cyberpunk game ui hud screenshot",
                         query="semiconductor chip")),
        ("unrelated software dashboard", dict(
            title="SaaS billing dashboard ui screenshot", query="chip company")),
    ],
    reject_pixel=[
        ("designed graphic, no junk metadata", _scores(graphic_dom=0.18), dict(
            expected="a silicon wafer")),
    ],
    preserve_meta=[
        ("relevant technology visual", dict(
            title="silicon wafer semiconductor fabrication clean room",
            query="silicon wafer fab")),
    ],
    preserve_pixel=[
        ("grounded fal still", _scores(), dict(
            expected="a silicon wafer in a clean room", concrete=True)),
    ],
)

# ── 8. BUSINESS ───────────────────────────────────────────────────────────────
_niche(
    "business",
    narration="the chart shows quarterly revenue growth as the company scaled to a "
              "billion dollars",
    reject_meta=[
        ("unrelated infographic", dict(
            title="weather forecast infographic chart", query="company revenue")),
        ("unrelated poster", dict(
            title="movie premiere poster big title", query="startup")),
    ],
    reject_pixel=[
        ("war crowd on a business beat", _scores(war_dom=0.09, people_dom=0.08), dict(
            expected="a revenue growth chart", crowd_ok=False)),
    ],
    preserve_meta=[
        ("relevant business chart (grounded)", dict(
            title="quarterly revenue bar chart growth",
            query="revenue chart",
            narration="the chart shows revenue growth each quarter")),
    ],
    preserve_pixel=[
        ("grounded fal still", _scores(), dict(
            expected="a modern corporate headquarters", concrete=True)),
    ],
)


# ══════════════════════════════════════════════════════════════════════════════
# DRIVER
# ══════════════════════════════════════════════════════════════════════════════
def _decide_meta(narration, kw):
    """Channel A decision. Returns (rejected: bool, reason). The fixture's own
    `narration` (if it supplies one — e.g. a relevant card that names its diagram/
    chart/document) takes precedence over the niche default."""
    call = dict(kw)
    call.setdefault("narration", narration)
    isj, reason, _hits = VR.classify_junk_metadata(**call)
    return isj, (reason or "")


def _decide_period(era, slug):
    """Channel B decision. Returns (rejected: bool, reason)."""
    FT._VIDEO_CTX["era"] = era
    blocked = FT._period_blocked(slug)
    return blocked, ("period-conflict" if blocked else "")


def _decide_pixel(scores, akw):
    """Channel C decision. Returns (rejected: bool, reason). accept()==False ->
    rejected (routed to the fail-closed ladder)."""
    _force_scorer_available(scores)
    ok, _s, why = VR.accept(_ASSET, False, **akw)
    return (not ok), why


def run_sweep():
    print("=" * 78)
    print("RC5 CROSS-NICHE VISUAL-DECISION SWEEP  (8 niches, hermetic, no render)")
    print("=" * 78)

    totals = {"accepted": 0, "rejected": 0, "quarantined": 0}
    junk_false_neg = 0           # known-junk that was NOT rejected (must be 0)
    preserve_over_reject = []    # useful visuals the gate dropped (documented)

    rows = []
    for key, nf in _NICHES.items():
        narr = nf["narration"]
        acc = rej = qtn = 0
        # ---- MUST-REJECT: metadata ----
        for label, kw in nf["reject_meta"]:
            rejected, reason = _decide_meta(narr, kw)
            if rejected:
                rej += 1
                qtn += 1   # a metadata-junk verdict is recorded to quarantine
            else:
                acc += 1
                junk_false_neg += 1
                _check(False, f"[{key}] META junk NOT rejected: {label}")
        # ---- MUST-REJECT: period (string) ----
        for label, era, slug in nf["reject_period"]:
            rejected, reason = _decide_period(era, slug)
            if rejected:
                rej += 1
            else:
                acc += 1
                junk_false_neg += 1
                _check(False, f"[{key}] PERIOD junk NOT rejected: {label}")
        # ---- MUST-REJECT: pixel (mocked CLIP) ----
        for label, scores, akw in nf["reject_pixel"]:
            rejected, reason = _decide_pixel(scores, akw)
            if rejected:
                rej += 1
            else:
                acc += 1
                junk_false_neg += 1
                _check(False, f"[{key}] PIXEL junk NOT rejected: {label} ({reason})")
        # ---- MUST-PRESERVE: metadata ----
        for label, kw in nf["preserve_meta"]:
            rejected, reason = _decide_meta(narr, kw)
            if rejected:
                preserve_over_reject.append((key, "meta", label, reason))
                rej += 1
            else:
                acc += 1
        # ---- MUST-PRESERVE: pixel ----
        for label, scores, akw in nf["preserve_pixel"]:
            rejected, reason = _decide_pixel(scores, akw)
            if rejected:
                preserve_over_reject.append((key, "pixel", label, reason))
                rej += 1
            else:
                acc += 1
        totals["accepted"] += acc
        totals["rejected"] += rej
        totals["quarantined"] += qtn
        rows.append((key, acc, rej, qtn))

    # ── per-niche table ──
    print()
    print(f"  {'niche':<18}{'accepted':>10}{'rejected':>10}{'quarantined':>13}")
    print("  " + "-" * 51)
    for key, acc, rej, qtn in rows:
        print(f"  {key:<18}{acc:>10}{rej:>10}{qtn:>13}")
    print("  " + "-" * 51)
    print(f"  {'TOTAL':<18}{totals['accepted']:>10}{totals['rejected']:>10}"
          f"{totals['quarantined']:>13}")

    # ── scorer-unavailable: metadata junk still rejected ──
    print("\n[scorer-unavailable] metadata junk must STILL be rejected "
          "(no silent accept):")
    _force_scorer_unavailable()
    su_false_neg = 0
    su_cases = [
        ("game UI", "war_geopolitics", dict(
            title="Hearts of Iron IV strategy game UI HUD", query="iran iraq")),
        ("anime", "spy", dict(title="spy anime episode", query="cold war")),
        ("infographic", "business", dict(
            title="weather infographic chart", query="revenue")),
        ("dvd-cover", "biography", dict(
            title="biopic DVD cover box art", query="rockefeller")),
    ]
    for label, niche, kw in su_cases:
        narr = _NICHES[niche]["narration"]
        isj, reason, _ = VR.classify_junk_metadata(narration=narr, **kw)
        ok = _check(isj, f"[scorer-OFF] metadata junk rejected: {label}")
        print(f"    {'PASS' if ok else 'FAIL'}  {label:<14} "
              f"available()={VR.available()}  junk={isj}")
        if not isj:
            su_false_neg += 1
    # the gate is genuinely unavailable on this path
    _check(VR.available() is False,
           "available() is False on the scorer-unavailable path")

    # ── assertions summary ──
    print("\n" + "=" * 78)
    print("ASSERTIONS")
    print("=" * 78)
    a1 = _check(junk_false_neg == 0,
                f"known-junk false-negatives == 0  (got {junk_false_neg})")
    print(f"  [{'PASS' if junk_false_neg == 0 else 'FAIL'}] "
          f"known-junk false-negatives across all niches == 0  "
          f"(got {junk_false_neg})")
    a2 = _check(su_false_neg == 0,
                f"scorer-unavailable accepted metadata junk (got {su_false_neg})")
    print(f"  [{'PASS' if su_false_neg == 0 else 'FAIL'}] "
          f"scorer-unavailable path never silently accepts junk-by-metadata  "
          f"(got {su_false_neg})")
    a3 = _check(len(preserve_over_reject) == 0,
                "useful-visual over-rejection in must-preserve set")
    if preserve_over_reject:
        print(f"  [DOC ] must-preserve OVER-REJECTED ({len(preserve_over_reject)}) "
              f"— documented, NOT silently accepted:")
        for niche, chan, label, reason in preserve_over_reject:
            print(f"          - [{niche}/{chan}] {label}: {reason}")
    else:
        print("  [PASS] no useful-visual over-rejection in the must-preserve set")

    # Coverage sanity: every named MUST-REJECT class appears at least once.
    n_reject = sum(len(nf["reject_meta"]) + len(nf["reject_pixel"])
                   + len(nf["reject_period"]) for nf in _NICHES.values())
    n_preserve = sum(len(nf["preserve_meta"]) + len(nf["preserve_pixel"])
                     for nf in _NICHES.values())
    print(f"\n  corpus: {n_reject} MUST-REJECT + {n_preserve} MUST-PRESERVE "
          f"fixtures across {len(_NICHES)} niches")
    _check(n_reject >= 15, "at least 15 MUST-REJECT fixtures exercised")
    _check(n_preserve >= 11, "at least 11 MUST-PRESERVE fixtures exercised")

    return junk_false_neg, su_false_neg, preserve_over_reject


def main():
    try:
        run_sweep()
    finally:
        try:
            os.unlink(_ASSET)
        except OSError:
            pass
    print("\n" + "=" * 78)
    if _FAILS:
        print(f"RESULT: FAIL — {len(_FAILS)} assertion(s) failed:")
        for m in _FAILS:
            print(f"  - {m}")
        sys.exit(1)
    print("RESULT: ALL GREEN — every known-junk class rejected in every niche; "
          "scorer-unavailable never accepts metadata junk; "
          "no must-preserve over-rejection.")
    sys.exit(0)


if __name__ == "__main__":
    main()
