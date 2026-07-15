#!/usr/bin/env python3
"""Behavioral tests for the NO-vision-model deterministic recovery-still gate (R4-6).

Drives the real pure decision `_deterministic_still_ok` (+ `_norm_title_toks`, `_title_season`) with
real ScriptSegment objects. Proves the three named requirements — GoT ≠ The Last of Us, wrong-season
GoT rejected, correct same-era GoT accepted — plus the supporting title-identity / era / CLIP /
Face-ID gates. No source-greps."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vidlore.clipstudio.orchestrate import (          # noqa: E402
    _deterministic_still_ok, _norm_title_toks, _title_season)
from vidlore.clipstudio.models import ScriptSegment    # noqa: E402

PASS = FAIL = 0
GOT = {"game", "of", "thrones"}                         # raw movie_title tokens (stopwords included)


def _say(ok, msg):
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  ok   {msg}")
    else:
        FAIL += 1
        print(f"  FAIL {msg}")


def seg(entity="", kind="", scene_query="", text=""):
    return ScriptSegment(index=1, text=text or "a line", required_entity=entity, required_kind=kind,
                         scene_query=scene_query)


def det(title, score=0.6, s=None, faces=None, era="", single=True):
    return _deterministic_still_ok(source_title=title, score=score, seg=s or seg(),
                                   faces=faces or [], movie_toks=GOT, global_era=era,
                                   single_scene=single, min_clip=0.30)


def main():
    # ---- helpers ----
    _say(_norm_title_toks("Game of Thrones - Season 3 (Official)") == {"game", "thrones"},
         f"_norm_title_toks strips generic/season words → {_norm_title_toks('Game of Thrones - Season 3 (Official)')}")
    _say(_norm_title_toks("The Last of Us Episode 3") == {"last"},
         f"_norm_title_toks('The Last of Us Episode 3') = {_norm_title_toks('The Last of Us Episode 3')}")
    _say(_title_season("Game of Thrones S03E10") == "season 3" and _title_season("Cersei 6x10") == "season 6"
         and _title_season("Tywin scene") == "",
         "_title_season reads S03E10→season 3, 6x10→season 6, none→''")

    # ---- (A) GoT ≠ The Last of Us (the headline requirement) ----
    ok, why = det("The Last of Us - Episode 3 - Joel and Ellie")
    _say(not ok and "show" in why, f"(A) GoT still is NOT installed onto a Last-of-Us source ({why})")

    # ...and unrelated shows in general
    for bad in ["Breaking Bad S03E07", "The Witcher — Geralt fights", "House of the Dragon S1E10"]:
        ok_b, why_b = det(bad)
        _say(not ok_b, f"(A) rejects unrelated show {bad!r} ({why_b})")

    # ---- (B) wrong-SEASON GoT rejected (beat era season 3, source season 5) ----
    ok2, why2 = det("Game of Thrones Season 5 — Cersei's walk", era="season 3")
    _say(not ok2 and "wrong era" in why2, f"(B) wrong-season GoT source is REJECTED ({why2})")

    # ---- (C) correct same-era GoT accepted ----
    ok3, why3 = det("Game of Thrones S03E10 — Tywin & Tyrion", era="season 3")
    _say(ok3, f"(C) correct same-show same-era GoT source is ACCEPTED ({why3})")

    # a GoT source with NO season label and an unconstrained beat is accepted (era unconstrained)
    ok4, why4 = det("Game of Thrones — Tywin small council", era="")
    _say(ok4, f"(C) same-show, era-unconstrained GoT source is ACCEPTED ({why4})")

    # ---- (D) CLIP relevance floor ----
    ok5, why5 = det("Game of Thrones S03E10", score=0.10, era="season 3")
    _say(not ok5 and "CLIP" in why5, f"(D) below-floor CLIP relevance is REJECTED ({why5})")

    # ---- (E) named character must be Face-ID present ----
    char = seg(entity="Tywin Lannister", kind="character")
    ok6, why6 = det("Game of Thrones S03E10", s=char, faces=["Cersei Lannister"], era="season 3")
    _say(not ok6 and "not Face-ID-confirmed" in why6,
         f"(E) character beat with the WRONG face present is REJECTED ({why6})")
    ok7, why7 = det("Game of Thrones S03E10", s=char, faces=["Tywin Lannister"], era="season 3")
    _say(ok7, f"(E) character beat with the RIGHT face present is ACCEPTED ({why7})")

    # ---- (F) a title with no meaningful tokens can't be confirmed same-show ----
    ok8, why8 = det("Season 3 Episode 10 [HD]")
    _say(not ok8, f"(F) a generic-only title cannot confirm same show ({why8})")

    # ---- (G) REVIEW FIX C: a SINGLE shared common token is too weak for same-show ----
    # 'The Game' shares only 'game' with 'Game of Thrones' → NOT the same show.
    okG1, whyG1 = det("The Game S03E05 — poker night")
    _say(not okG1 and "need >= 2" in whyG1, f"(G) 'The Game' shares only 'game' → REJECTED ({whyG1})")
    # 'House of the Dragon' vs 'Dragon Ball' — shares only 'dragon' → NOT the same show.
    HOTD = {"house", "of", "the", "dragon"}
    okG2, whyG2 = _deterministic_still_ok(source_title="Dragon Ball Super Episode 3", score=0.6,
                                          seg=seg(), faces=[], movie_toks=HOTD, global_era="",
                                          single_scene=True)
    _say(not okG2, f"(G) 'Dragon Ball' shares only 'dragon' with 'House of the Dragon' → REJECTED ({whyG2})")
    okG3, whyG3 = _deterministic_still_ok(source_title="House of the Dragon S01E10 — Rhaenyra", score=0.6,
                                          seg=seg(), faces=[], movie_toks=HOTD, global_era="",
                                          single_scene=True)
    _say(okG3, f"(G) a real HotD source (shares house+dragon) → ACCEPTED ({whyG3})")
    _say("s03e10" not in _norm_title_toks("Game of Thrones S03E10")
         and "s01" not in _norm_title_toks("House of the Dragon S01E10"),
         "(G) episode codes (S03E10 / S01) are stripped from title tokens")

    # ---- (H) REVIEW FIX D: whole-word, article-dropped, ALL-token Face-ID match ----
    hound = seg(entity="The Hound", kind="character")
    okH1, whyH1 = det("Game of Thrones S04E01", s=hound, faces=["Theon Greyjoy"], era="")
    _say(not okH1, f"(H) 'The Hound' beat is NOT satisfied by a 'Theon' face (no 'the' substring hit) ({whyH1})")
    okH2, whyH2 = det("Game of Thrones S04E01", s=hound, faces=["The Hound"], era="")
    _say(okH2, f"(H) 'The Hound' beat IS satisfied by a 'Hound' face ({whyH2})")
    jon = seg(entity="Jon Snow", kind="character")
    okH3, whyH3 = det("Game of Thrones S04E01", s=jon, faces=["Jon Arryn"], era="")
    _say(not okH3, f"(H) 'Jon Snow' beat is NOT satisfied by 'Jon Arryn' (shared given name only) ({whyH3})")
    okH4, whyH4 = det("Game of Thrones S04E01", s=jon, faces=["Jon Snow"], era="")
    _say(okH4, f"(H) 'Jon Snow' beat IS satisfied by a 'Jon Snow' face ({whyH4})")

    print(f"\n{PASS} passed · {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
