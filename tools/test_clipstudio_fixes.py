"""ClipStudio bug-fix regression suite.

Pins the fixes from the 2026-06-10 deep review (research/clipstudio_review/REVIEW_FINDINGS.md)
so none of them silently regresses:

    python3 tools/test_clipstudio_fixes.py

Each section is independent; any failed assertion = regression. Tests are pure-logic /
monkeypatched — no network, no LLM, no ffmpeg run.
"""
import json
import os
import sys
import tempfile
import types
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASS = 0
FAIL = 0


def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}")


# ===========================================================================
# 1) verify.py — promoting an alternate must rewrite beat_windows
#    (rejected pick was FIRST in beat_windows; build_video plays from there)
# ===========================================================================

def test_verifier_promotion_rewrites_beat_windows():
    print("[1] verifier promotion rewrites beat_windows")
    from vidlore.clipstudio import verify as V
    from vidlore.clipstudio import models as M
    from vidlore.clipstudio.config import ClipConfig
    import vidlore.clipstudio.llm as L

    tmp = tempfile.mkdtemp(prefix="csfix_")
    proj = M.ClipProject(name="t", root=tmp)
    sel = M.ClipSelection(segment_index=0, source_id="srcA", shot_index=1,
                          in_point=10.0, out_point=14.0, confidence=0.9)
    alt = M.ClipCandidate(segment_index=0, source_id="srcB", shot_index=2,
                          score=0.8, in_point=20.0, out_point=24.0)
    sel.alternates = [alt]
    # match.py order: chosen FIRST, then alternates (srcB already present), then others
    sel.beat_windows = [["srcA", 10.0, 14.0], ["srcB", 20.0, 24.0], ["srcC", 30.0, 34.0]]
    proj.selections = [sel]
    seg = M.ScriptSegment(index=0, text="the hero enters the vault")

    fake_shot = types.SimpleNamespace(index=2, keyframe_path="kf.jpg", face_ids=[],
                                      identities=[])

    calls = {"n": 0, "cut": 0}

    def fake_verify_frame(kf, narration, ent, kind, names, eng_cfg, model="", is_specific=True,
                          **kwargs):
        calls["n"] += 1
        # first call = the chosen pick -> replace; second = the alternate -> keep
        if calls["n"] == 1:
            return {"verdict": "replace", "confidence": 0.2, "reason": "wrong subject"}
        return {"verdict": "keep", "confidence": 0.9, "reason": "good"}

    orig_vf, orig_lookup = V.verify_frame, V._shot_lookup
    orig_cut, orig_has = V._cut.cut_selection, L.has_llm
    try:
        V.verify_frame = fake_verify_frame
        V._shot_lookup = lambda p: (lambda sid, idx: fake_shot)
        V._cut.cut_selection = lambda p, s, c: calls.__setitem__("cut", calls["cut"] + 1)
        L.has_llm = lambda eng_cfg=None: True
        eng = types.SimpleNamespace(anthropic_model="test-model")
        out = V.verify_and_repair(proj, [seg], ClipConfig(), eng, progress=None)
    finally:
        V.verify_frame, V._shot_lookup = orig_vf, orig_lookup
        V._cut.cut_selection, L.has_llm = orig_cut, orig_has

    check("alternate promoted", sel.source_id == "srcB" and out["replaced"] == 1)
    check("re-cut invoked", calls["cut"] == 1)
    wins = sel.beat_windows
    check("promoted window leads", bool(wins) and wins[0][0] == "srcB"
          and abs(float(wins[0][1]) - 20.0) < 0.01)
    check("rejected window dropped",
          not any(w[0] == "srcA" and abs(float(w[1]) - 10.0) < 0.05 for w in wins))
    check("promoted window not duplicated",
          sum(1 for w in wins if w[0] == "srcB" and abs(float(w[1]) - 20.0) < 0.05) == 1)
    check("unrelated window preserved",
          any(w[0] == "srcC" for w in wins))


# ===========================================================================
# 2) build.py — visual-budget loop must fall back to the stage-3 beat counts
#    when plan_beats raises (was: overwrote them with zeros -> ZeroDivisionError)
# ===========================================================================

def _extract_budget_loop_src():
    src = (Path(__file__).resolve().parent.parent / "vidlore" / "clipstudio" / "build.py") \
        .read_text(encoding="utf-8")
    start = src.index("    _energies_eff = list(_eng)")
    end_marker = "    nbeats = [max(1, kk) for kk in _nb]"
    end = src.index(end_marker)
    end = src.index("\n", end) + 1
    block = src[start:end]
    # dedent one level (the block lives inside build_video)
    return "\n".join(ln[4:] if ln.startswith("    ") else ln for ln in block.splitlines())


def test_budget_loop_survives_plan_beats_failure():
    print("[2] build.py budget loop fallback (no zero beats)")
    block = _extract_budget_loop_src()
    seg0, seg1 = types.SimpleNamespace(index=0), types.SimpleNamespace(index=1)

    def boom(*a, **k):
        raise ValueError("plan_beats unavailable")

    ns = {"_eng": [3, 3], "segments": [seg0, seg1], "plan_beats": boom,
          "_durs": [6.0, 8.0], "_rol": ["body", "body"], "_D": {0: 2, 1: 2},
          "_gbudget": 99, "nbeats": [1, 1], "hold_pos": set()}
    exec(block, ns)
    check("plan_beats failure keeps stage-3 counts", ns["nbeats"] == [1, 1])
    check("no zero beat count on failure", all(k >= 1 for k in ns["nbeats"]))

    # healthy plan_beats: build keeps assemble's k (NEVER caps below it — that desyncs into
    # assemble's first-clip replay); energy lowering is the only reduction mechanism
    def ok_pb(durs, target, bmin, energies, roles):
        return [(0, 0, 3.0), (0, 3.0, 6.0), (1, 0, 8.0)]

    ns2 = {"_eng": [1, 1], "segments": [seg0, seg1], "plan_beats": ok_pb,
           "_durs": [6.0, 8.0], "_rol": ["body", "body"], "_D": {0: 1, 1: 1},
           "_gbudget": 99, "nbeats": [1, 1], "hold_pos": set()}
    exec(block, ns2)
    check("energy-1 scene keeps assemble's beat count (sync, pad covers repeats)",
          ns2["nbeats"] == [2, 1])

    ns3 = {"_eng": [3, 3], "segments": [seg0, seg1], "plan_beats": ok_pb,
           "_durs": [6.0, 8.0], "_rol": ["body", "body"], "_D": {0: 4, 1: 4},
           "_gbudget": 99, "nbeats": [1, 1], "hold_pos": set()}
    exec(block, ns3)
    check("healthy loop keeps planned counts", ns3["nbeats"] == [2, 1])


# ===========================================================================
# 3) download integrity — produced-file detection must prefer the merge target
#    and never accept a video-only .fNNN DASH fragment; ffmpeg rc respected
# ===========================================================================

def test_find_produced_video():
    print("[3] produced-file detection (fragments/stale files)")
    from vidlore.clipstudio.ingest import find_produced_video

    d = Path(tempfile.mkdtemp(prefix="csdl_"))

    # fragment + merge target both present -> merge target wins (was: fragment, 'f' < 'm')
    (d / "vid1.f616.mp4").write_bytes(b"frag")
    (d / "vid1.mp4").write_bytes(b"merged")
    check("merge target preferred over fragment",
          (find_produced_video(d, "vid1") or Path("")).name == "vid1.mp4")

    # only a leftover fragment -> None (was: accepted as the source, silent/no-audio)
    (d / "vid2.f616.mp4").write_bytes(b"frag")
    check("lone DASH fragment rejected", find_produced_video(d, "vid2") is None)

    # partial merge temp -> None
    (d / "vid3.temp.mp4").write_bytes(b"tmp")
    check("merge temp rejected", find_produced_video(d, "vid3") is None)

    # zero-byte merge target -> None
    (d / "vid4.mp4").write_bytes(b"")
    check("zero-byte file rejected", find_produced_video(d, "vid4") is None)

    # normal alternate container still found
    (d / "vid5.mkv").write_bytes(b"x")
    check("alt container accepted",
          (find_produced_video(d, "vid5") or Path("")).name == "vid5.mkv")

    # a similarly-prefixed sibling never bleeds in
    (d / "vid6_other.mp4").write_bytes(b"x")
    check("sibling stem not matched", find_produced_video(d, "vid6") is None)


def test_cut_checks_ffmpeg_rc():
    print("[4] cut_selection respects ffmpeg return code")
    from vidlore.clipstudio import cut as C
    from vidlore.clipstudio import models as M
    from vidlore.clipstudio.config import ClipConfig

    tmp = tempfile.mkdtemp(prefix="cscut_")
    proj = M.ClipProject(name="t", root=tmp)
    src_file = Path(tmp) / "src.mp4"
    src_file.write_bytes(b"x" * 64)
    proj.sources = [M.SourceVideo(id="s1", url="", title="t", local_path=str(src_file),
                                  duration=60.0, permission="owner")]
    sel = M.ClipSelection(segment_index=0, source_id="s1", shot_index=0,
                          in_point=1.0, out_point=4.0, confidence=0.9)

    class FakeProc:
        def __init__(self, rc):
            self.returncode = rc

    orig_run = C.subprocess.run
    try:
        # rc != 0 but a (truncated) output file appears -> must return None
        def fail_run(cmd, **kw):
            out = Path(cmd[-1])
            out.write_bytes(b"truncated")
            return FakeProc(1)
        C.subprocess.run = fail_run
        check("rc!=0 rejected even with file on disk",
              C.cut_selection(proj, sel, ClipConfig()) is None)

        def ok_run(cmd, **kw):
            out = Path(cmd[-1])
            out.write_bytes(b"good")
            return FakeProc(0)
        C.subprocess.run = ok_run
        got = C.cut_selection(proj, sel, ClipConfig())
        check("rc==0 accepted", got is not None and got.exists())
    finally:
        C.subprocess.run = orig_run


# ===========================================================================
# 5) orchestrate.py — coverage/anchor force-includes must survive the download
#    cap, and the caller's ClipConfig must never be mutated
# ===========================================================================

def test_discover_budget_and_cfg_copy():
    print("[5] discover budget honors force-includes; cfg not mutated")
    import dataclasses
    from vidlore.clipstudio.config import ClipConfig
    src = (Path(__file__).resolve().parent.parent / "vidlore" / "clipstudio" /
           "orchestrate.py").read_text(encoding="utf-8")
    check("download cap covers the whole discovered list", "dl_limit = len(candidates)" in src)
    check("cfg copied, not mutated", "_dc.replace(cfg" in src
          and "cfg.discover_target = max(" not in src)
    cfg = ClipConfig()
    cfg2 = dataclasses.replace(cfg, discover_target=6)
    check("dataclasses.replace works on ClipConfig",
          cfg2.discover_target == 6 and cfg.discover_target != 6 or cfg.discover_target == 18)


# ===========================================================================
# 6) match.py — scoring/relevance fixes
# ===========================================================================

def _mk_shot(sid, idx, start, end, transcript="", quality=1.0):
    from vidlore.clipstudio.models import Shot
    return Shot(source_id=sid, index=idx, start=start, end=end, transcript=transcript,
                quality=quality)


def test_match_scoring_fixes():
    print("[6] match.py scoring: anchor bonus, clamp, alternates order, dark-scene scope")
    from vidlore.clipstudio import match as MM
    from vidlore.clipstudio import models as M
    from vidlore.clipstudio.config import ClipConfig

    cfg = ClipConfig()

    # --- (a) anchor bonus drives ranking but NOT the reported base/confidence ---
    seg = M.ScriptSegment(index=0, text="he raises his sword", keywords=["sword"])
    anchor_shot = MM._PoolShot("anch", _mk_shot("anch", 0, 0, 5))            # zero signals
    good_shot = MM._PoolShot("other", _mk_shot("other", 0, 10, 15, transcript="sword fight"))
    scored = MM._score_pool(seg, [anchor_shot, good_shot], None, cfg,
                            set(), {"anch"}, 0.45)
    by_sid = {s[3].sid: s for s in scored}
    check("zero-signal anchor base stays ~0 (bonus decoupled)",
          by_sid["anch"][0] < cfg.min_confidence and abs(by_sid["anch"][1] - 0.45) < 1e-9)
    check("anchor bonus recorded in signals", by_sid["anch"][2].get("anchor_bonus") == 0.45)
    check("ranking still honors the bonus (anchor first)",
          scored[0][3].sid == "anch")
    check("non-anchor keeps real base, no bonus",
          by_sid["other"][0] > 0 and by_sid["other"][1] == 0.0)

    # --- (b)+(c) full selection: confidence clamped >= 0, alternates score-ordered ---
    tmp = tempfile.mkdtemp(prefix="csmatch_")
    proj = M.ClipProject(name="t", root=tmp)
    proj.sources = [M.SourceVideo(id=s, url=f"u://{s}", title=s, permission="owner",
                                  status="ok") for s in ("A", "B", "C")]
    pool = [
        MM._PoolShot("A", _mk_shot("A", 0, 0, 6, transcript="sword battle charge")),
        MM._PoolShot("B", _mk_shot("B", 0, 0, 6, transcript="sword")),
        MM._PoolShot("C", _mk_shot("C", 0, 0, 6, transcript="")),
    ]
    segs = [M.ScriptSegment(index=0, text="x", keywords=["sword", "battle"]),
            M.ScriptSegment(index=1, text="y", keywords=[])]   # zero-signal beat
    orig_pool, orig_avail = MM._load_pool, MM._index.clip_available
    try:
        MM._load_pool = lambda p, c=None, progress=None, show_title="": pool
        MM._index.clip_available = lambda: False
        sels = MM.match_segments(proj, segs, cfg)
    finally:
        MM._load_pool, MM._index.clip_available = orig_pool, orig_avail
    s0 = sels[0]
    check("best source chosen", s0.source_id == "A")
    alts = s0.alternates
    check("alternates score-ordered desc",
          [a.source_id for a in alts][:2] == ["B", "C"]
          and all(alts[i].score >= alts[i + 1].score for i in range(len(alts) - 1)))
    check("confidence never negative", all(s.confidence >= 0.0 for s in sels))
    check("beat_windows lead with the chosen pick",
          s0.beat_windows and s0.beat_windows[0][0] == "A")

    # --- (d) dark-scene guard: movie-title/dark words in beat queries must NOT trigger it ---
    def run_dark(anchor_query, scene_query, movie_title):
        msgs = []
        ana = types.SimpleNamespace(
            video_type="single_scene", movie_title=movie_title,
            anchor_scenes=[{"name": "anchor", "query": anchor_query}],
            char_to_actor=lambda: {})
        prj = M.ClipProject(name="t", root=tmp)
        prj.sources = [M.SourceVideo(id="A", url="u", title=anchor_query,
                                     permission="owner", status="ok")]
        sg = [M.ScriptSegment(index=0, text="x", scene_query=scene_query)]
        orig_p, orig_a = MM._load_pool, MM._index.clip_available
        try:
            MM._load_pool = lambda p, c=None, progress=None, show_title="": pool
            MM._index.clip_available = lambda: False
            MM.match_segments(prj, sg, cfg, analysis=ana, progress=msgs.append)
        finally:
            MM._load_pool, MM._index.clip_available = orig_p, orig_a
        line = next((m for m in msgs if "dark_scene=" in m), "")
        return "dark_scene=True" in line

    check("title word 'dark' in beat query does NOT trigger dark guard",
          run_dark("rooftop duel daytime confrontation",
                   "The Dark Knight rooftop confrontation", "The Dark Knight") is False)
    check("genuinely dark anchor scene DOES trigger dark guard",
          run_dark("hound tavern night candlelit argument", "", "Game of Thrones") is True)

    # --- (e) anchor-source detection is token-based, not substring ---
    ana = types.SimpleNamespace(
        video_type="single_scene", movie_title="Show",
        anchor_scenes=[{"name": "", "query": "arya hound farewell"}],
        char_to_actor=lambda: {})
    prj = M.ClipProject(name="t", root=tmp)
    prj.sources = [M.SourceVideo(id="bad", url="u", title="caryatid houndstooth pattern",
                                 permission="owner", status="ok"),
                   M.SourceVideo(id="good", url="u", title="Arya and the Hound farewell scene",
                                 permission="owner", status="ok")]
    msgs = []
    orig_p, orig_a = MM._load_pool, MM._index.clip_available
    try:
        MM._load_pool = lambda p, c=None, progress=None, show_title="": pool
        MM._index.clip_available = lambda: False
        MM.match_segments(prj, [M.ScriptSegment(index=0, text="x")], cfg,
                          analysis=ana, progress=msgs.append)
    finally:
        MM._load_pool, MM._index.clip_available = orig_p, orig_a
    line = next((m for m in msgs if "anchor source(s)" in m), "")
    check("substring title ('caryatid houndstooth') not an anchor; real title is",
          line.startswith("match: 1 anchor source(s)"))


# ===========================================================================
# 13) exact-scene recall + anti-false-anchor + repetition cap
# ===========================================================================

def test_exact_scene_recall_fixes():
    print("[13] exact-scene recall: anchor variants, dialogue verify, false-anchor kill")
    from vidlore.clipstudio import discover as D
    from vidlore.clipstudio import match as MM
    from vidlore.clipstudio import models as M
    from vidlore.clipstudio.analyze import ScriptAnalysis
    from vidlore.clipstudio.config import ClipConfig

    ana = ScriptAnalysis(
        movie_title="Game of Thrones", episode_hint="S02E09", video_type="single_scene",
        anchor_scenes=[{"name": "Bronn and The Hound's tavern confrontation",
                        "query": "Bronn sings Rains of Castamere tavern The Hound confrontation"}],
        characters=[{"name": "Bronn"}, {"name": "The Hound"}, {"name": "Sandor Clegane"}],
        actors=["Jerome Flynn", "Rory McCann"])
    segs = [M.ScriptSegment(index=0, text="x",
                            quote="You're just like me, only smaller"),
            M.ScriptSegment(index=1, text="y", quote="I like killing")]

    qs = D.anchor_queries(ana, segs)
    j = " | ".join(qs).lower()
    check("anchor variants include character pair", "bronn and the hound scene" in j)
    check("anchor variants include episode code", "s02e09" in j)
    check("anchor variants include the iconic quote", "just like me" in j)
    check("variant count bounded", 1 < len(qs) <= 18)   # raised: angle/version variants added

    # dialogue containment: one scene-specific DIALOGUE line is decisive; a different scene with
    # none of the lines does not verify. (Lyric poisoning is handled by music-title exclusion in
    # verify_anchor_candidates, NOT by hit count — the LLM often gets only 1 of N lines verbatim.)
    subs_good = "you fight for gold I fight for honor ... you're just like me only smaller hmm"
    subs_bad = "the rains of castamere plays as the doors close lannister regards"
    quotes = D._scene_quotes(segs)
    check("subs speaking a scene line verify", D._subs_contain_quote(subs_good, quotes) is True)
    check("different-scene subs do NOT verify", D._subs_contain_quote(subs_bad, quotes) is False)
    # music/cover/lyric uploads are excluded from the verify pool by title (the real lyric guard)
    check("music-title videos flagged for verify-pool exclusion",
          bool(D._MUSIC_ONLY_RX.search("Tywin sings Rains of Castamere [Extended Version]"))
          and bool(D._MUSIC_ONLY_RX.search("The Rains of Castamere (Best Version)")))

    # match.py anchor sources: false title anchor dies; ASR-spoken + verified pass
    tmp = tempfile.mkdtemp(prefix="csanc_")
    proj = M.ClipProject(name="t", root=tmp)
    proj.sources = [
        M.SourceVideo(id="redwed", url="u", title="The Red Wedding - The Rains of Castamere plays",
                      permission="owner", status="ok"),
        M.SourceVideo(id="tavern", url="u", title="Some random clip title",
                      permission="owner", status="ok"),
        M.SourceVideo(id="verified", url="u", title="whatever",
                      permission="owner", status="ok", extra={"anchor_verified": True}),
        M.SourceVideo(id="titlechar", url="u", title="Bronn and the Hound tavern standoff",
                      permission="owner", status="ok"),
    ]
    pool = [
        MM._PoolShot("redwed", _mk_shot("redwed", 0, 0, 5, transcript="music plays loudly")),
        MM._PoolShot("tavern", _mk_shot("tavern", 0, 0, 5,
                                        transcript="I like killing hmm you're just like me only smaller")),
        MM._PoolShot("verified", _mk_shot("verified", 0, 0, 5)),
        MM._PoolShot("titlechar", _mk_shot("titlechar", 0, 0, 5)),
    ]
    msgs = []
    ana2 = types.SimpleNamespace(
        video_type="single_scene", movie_title="Game of Thrones",
        anchor_scenes=[{"name": "Bronn tavern confrontation",
                        "query": "Bronn sings Rains of Castamere tavern Hound confrontation"}],
        characters=[{"name": "Bronn"}, {"name": "The Hound"}], actors=[],
        char_to_actor=lambda: {})
    op, oa = MM._load_pool, MM._index.clip_available
    try:
        MM._load_pool = lambda p, c=None, progress=None, show_title="": pool
        MM._index.clip_available = lambda: False
        MM.match_segments(proj, segs, ClipConfig(), analysis=ana2, progress=msgs.append)
    finally:
        MM._load_pool, MM._index.clip_available = op, oa
    line = next((m for m in msgs if "anchor source(s)" in m), "")
    check("3 true anchors (ASR + verified + title-with-character), false one dead",
          line.startswith("match: 3 anchor source(s)"))

    # build.py: hard cap pinned
    bsrc = (Path(__file__).resolve().parent.parent / "vidlore" / "clipstudio" /
            "build.py").read_text(encoding="utf-8")
    check("cross-scene window air cap exists (once, ever)",
          "_air_ct.get(_wkey(w[0], w[1]), 0) >= 1" in bsrc)
    check("scene-walk playhead wired",
          "_playhead" in bsrc and "max(_playhead.get(src.id, 0.0), start + src_need" in bsrc)
    # analyze: season/episode split pinned
    asrc = (Path(__file__).resolve().parent.parent / "vidlore" / "clipstudio" /
            "analyze.py").read_text(encoding="utf-8")
    check("movie title season/episode split", "episode_hint = f\"S{int(" in asrc)

    # canonical-scene recall: data-driven scene phrase, compilation demotion, episode anchoring
    dsrc2 = (Path(__file__).resolve().parent.parent / "vidlore" / "clipstudio" /
             "discover.py").read_text(encoding="utf-8")
    check("dominant scene phrase mined from beats", "_dominant_scene_phrases" in dsrc2)
    check("single-scene compilation demotion", "all scenes" in dsrc2 and "score -= 0.30" in dsrc2)
    check("episode-code relevance boost", 'ep in t.replace(" ", "")' in dsrc2)
    msrc2 = (Path(__file__).resolve().parent.parent / "vidlore" / "clipstudio" /
             "match.py").read_text(encoding="utf-8")
    check("episode-code anchoring in match", "_ep in (src.title" in msrc2)
    # FOOTAGE-ABUNDANCE GUARD: a long single-scene deep-dive whose long-form floor pulled in MANY anchor
    # sources must NOT relax anti-reuse — relaxing re-aired one Cleganebowl source 26× / one window 13×
    # while 14 relevant context sources sat unused. Verified by re-match: 20→26 sources, 26×→13×, 13×→3×.
    check("single-scene reuse-relax disabled when anchor footage is abundant",
          "_anchor_abundant" in msrc2
          and "len(anchor_sids) > int(_f_env" in msrc2
          and "and not _anchor_abundant" in msrc2)

    # pacing + quality: Ken Burns hold/zoom, SD sharpen-upscale, HD selection preference, CRF
    from vidlore.clipstudio.build import _ken_burns_filter, _upscale_filter
    kb = _ken_burns_filter(3.0, 640)
    check("ken burns is a slow push-in", "zoompan" in kb and "min(1.0+" in kb)
    check("ken burns pre-upscales (lanczos)", "lanczos" in kb)
    check("SD source gets sharpen-upscale", "unsharp" in _upscale_filter(640))
    check("every cut gets contrast-adaptive sharpen (fake-HD uploads are soft)",
          "cas=" in _upscale_filter(1920) and "cas=" in _upscale_filter(640)
          and "cas=" in kb)
    bsrc3 = (Path(__file__).resolve().parent.parent / "vidlore" / "clipstudio" /
             "build.py").read_text(encoding="utf-8")
    check("main-moment hold+zoom wired", "main-moment hold" in bsrc3.lower()
          and "pos in hold_pos" in bsrc3)
    check("recut CRF raised to 18 (clips start crisp)",
          '"-c:v", "libx264", "-preset", "medium", "-crf", "18"' in bsrc3)
    msrc3 = (Path(__file__).resolve().parent.parent / "vidlore" / "clipstudio" /
             "match.py").read_text(encoding="utf-8")
    check("HD resolution preference in selection", "hd_pref" in msrc3 and "_src_height" in msrc3)

    # _dominant_scene_phrases: surfaces 'small council' from beat queries
    from vidlore.clipstudio.discover import _dominant_scene_phrases
    from vidlore.clipstudio.analyze import ScriptAnalysis
    segs_dsp = [M.ScriptSegment(index=i, text="x",
                                scene_query=q) for i, q in enumerate([
        "Littlefinger small council chairs", "Tywin small council table",
        "council meeting chairs scramble", "small council room Varys"])]
    ana_dsp = ScriptAnalysis(movie_title="Game of Thrones")
    phrases = _dominant_scene_phrases(segs_dsp, ana_dsp)
    check("mines 'small council' as dominant scene phrase",
          any("council" in p for p in phrases))


# ===========================================================================
# 12) round-2 adversarial-review fixes
# ===========================================================================

def test_round2_review_fixes():
    print("[12] round-2: failed-alt windows, anchor-ordered alternates, dark splice, misc")
    from vidlore.clipstudio import verify as V
    from vidlore.clipstudio import match as MM
    from vidlore.clipstudio import models as M
    from vidlore.clipstudio.config import ClipConfig
    import vidlore.clipstudio.llm as L

    # (a) a verifier-FAILED alternate's window must leave beat_windows too
    tmp = tempfile.mkdtemp(prefix="csr2_")
    proj = M.ClipProject(name="t", root=tmp)
    sel = M.ClipSelection(segment_index=0, source_id="srcA", shot_index=1,
                          in_point=10.0, out_point=14.0, confidence=0.9)
    altA = M.ClipCandidate(segment_index=0, source_id="srcB", shot_index=2,
                           score=0.8, in_point=20.0, out_point=24.0)
    altB = M.ClipCandidate(segment_index=0, source_id="srcC", shot_index=3,
                           score=0.7, in_point=30.0, out_point=34.0)
    sel.alternates = [altA, altB]
    sel.beat_windows = [["srcA", 10.0, 14.0], ["srcB", 20.0, 24.0],
                        ["srcC", 30.0, 34.0], ["srcD", 40.0, 44.0]]
    proj.selections = [sel]
    seg = M.ScriptSegment(index=0, text="x")
    # Distinct shot per (source_id, shot_index) — the verdict-rung cache keys on the shot's
    # identity (source_id / bounds / keyframe), which every REAL pair of alternates has. A single
    # shared mock shot would make altA and altB the byte-identical cached question, so altB would
    # (correctly, per the cache doctrine) reuse altA's verdict and never reach the mock.
    def fake_shot_of(sid, idx):
        return types.SimpleNamespace(index=idx, source_id=sid, start=10.0 * idx,
                                     end=10.0 * idx + 4.0, keyframe_path=f"kf_{sid}_{idx}.jpg",
                                     face_ids=[], identities=[])
    calls = {"n": 0}

    def fake_vf(kf, narration, ent, kind, names, eng_cfg, model="", is_specific=True, **kwargs):
        calls["n"] += 1
        # chosen -> replace; altA -> replace (FAILS); altB -> keep
        return {"verdict": ["replace", "replace", "keep"][min(calls["n"] - 1, 2)],
                "confidence": 0.5, "reason": "r"}

    orig = (V.verify_frame, V._shot_lookup, V._cut.cut_selection, L.has_llm)
    try:
        V.verify_frame = fake_vf
        V._shot_lookup = lambda p: fake_shot_of
        V._cut.cut_selection = lambda p, s, c: None
        L.has_llm = lambda eng_cfg=None: True
        V.verify_and_repair(proj, [seg], ClipConfig(),
                            types.SimpleNamespace(anthropic_model="m"))
    finally:
        V.verify_frame, V._shot_lookup, V._cut.cut_selection, L.has_llm = orig
    sids = [w[0] for w in sel.beat_windows]
    check("promoted altB leads", sids and sids[0] == "srcC" and sel.source_id == "srcC")
    check("rejected primary gone", "srcA" not in sids)
    check("verifier-FAILED altA gone too", "srcB" not in sids)
    check("never-judged srcD window preserved", "srcD" in sids)

    # (b) alternates ordering keeps anchor-bonus ranking (single-scene continuity)
    seg2 = M.ScriptSegment(index=0, text="y", keywords=["sword", "fight"])
    anch = MM._PoolShot("anch", _mk_shot("anch", 0, 0, 5, transcript="sword"))
    other = MM._PoolShot("plain", _mk_shot("plain", 0, 0, 5, transcript="sword fight"))
    scored = MM._score_pool(seg2, [anch, other], None, ClipConfig(), set(), {"anch"}, 0.45)
    # ranking: anchor first (bonus), though 'plain' has the higher raw base
    check("anchor outranks higher-base plain shot", scored[0][3].sid == "anch"
          and scored[0][0] < scored[1][0])

    # (c) dark-scene title removal must not splice words into false matches
    cfg = ClipConfig()
    pool = [MM._PoolShot("A", _mk_shot("A", 0, 0, 6))]
    def run_dark(anchor_query, movie_title):
        msgs = []
        ana = types.SimpleNamespace(video_type="single_scene", movie_title=movie_title,
                                    anchor_scenes=[{"name": "anchor", "query": anchor_query}],
                                    char_to_actor=lambda: {})
        prj = M.ClipProject(name="t", root=tmp)
        prj.sources = [M.SourceVideo(id="A", url="u", title=anchor_query,
                                     permission="owner", status="ok")]
        op, oa = MM._load_pool, MM._index.clip_available
        try:
            MM._load_pool = lambda p, c=None, progress=None, show_title="": pool
            MM._index.clip_available = lambda: False
            MM.match_segments(prj, [M.ScriptSegment(index=0, text="x")], cfg,
                              analysis=ana, progress=msgs.append)
        finally:
            MM._load_pool, MM._index.clip_available = op, oa
        return "dark_scene=True" in next((m for m in msgs if "dark_scene=" in m), "")
    # 'Throne' removed from title would splice 'the throne golden room' -> 'the golden room';
    # with boundary-preserving blanking 'throne room' must NOT be fabricated from
    # 'throne ... room' across a removed word
    check("no fabricated adjacency after title removal",
          run_dark("the throne golden room duel", "Throne Golden") is False)
    check("real dark words still trigger", run_dark("tavern night candlelit", "Show") is True)

    # (d) misc: prefix matcher bound, hd-bonus default tiers, empty-[] retry, finished TTL
    src_d = (Path(__file__).resolve().parent.parent / "vidlore" / "clipstudio")
    dsrc = (src_d / "discover.py").read_text(encoding="utf-8")
    check("coverage prefix bounded to short suffixes", "len(w) - len(tok) <= 2" in dsrc)
    check("anchor prefix bounded too", "len(t) - len(w) <= 2" in dsrc)
    asrc = (src_d / "analyze.py").read_text(encoding="utf-8")
    check("empty-[] enrich reply spends the retry", "'[]' is no enrichment" in asrc)
    ssrc = (src_d / "segment.py").read_text(encoding="utf-8")
    check("segment enrich guards non-numeric i", "_bi(o)" in ssrc and "isinstance(o, dict)" in ssrc)
    wsrc = (src_d / "web.py").read_text(encoding="utf-8")
    check("job TTL anchors on completion", 'j["finished"] = time.time()' in wsrc
          and '.get("finished", _JOBS[jid].get("started"' in wsrc)
    bsrc = (src_d / "build.py").read_text(encoding="utf-8")
    check("final plan recompute after budget loop", "FINAL recompute from the FINAL energies" in bsrc)
    check("beat clips cut to real planned lengths", "max(cfg.min_clip_sec, _lens[m]) + 0.5" in bsrc)
    osrc = (src_d / "orchestrate.py").read_text(encoding="utf-8")
    check("download covers full discovered list", "dl_limit = len(candidates)" in osrc)
    hsrc2 = (src_d / "hd_download.py").read_text(encoding="utf-8")
    check("hd out_stem resolved absolute (cwd-relative -o nested the output)",
          "out_stem = str(Path(out_stem).expanduser().resolve())" in hsrc2)
    check("hd reports the produced file's real dimensions",
          "from .ingest import probe as _probe" in hsrc2)
    dlsrc = (src_d / "download.py").read_text(encoding="utf-8")
    check("HD fallback never silent", 'sv.extra["hd_fallback"]' in dlsrc)


# ===========================================================================
# 11) low-severity fixes
# ===========================================================================

def test_low_severity_fixes():
    print("[11] lows: shot-merge floor, local-path detection, ffprobe order, prefer-height")
    from vidlore.clipstudio import ingest as IN
    from vidlore.clipstudio import index as IX
    from vidlore.clipstudio import models as M
    from vidlore.clipstudio.config import ClipConfig

    # shots shorter than min_clip_sec must merge — cut would otherwise cross the boundary
    cfg = ClipConfig()
    scenes = [(0.0, 1.05), (1.05, 5.0)]      # 1.05s < min_clip_sec 1.2
    merged = IX._merge_short(scenes, max(cfg.min_shot_sec, cfg.min_clip_sec))
    check("sub-min_clip shot merged away",
          all((e - s) >= cfg.min_clip_sec for s, e in merged))
    src = (Path(__file__).resolve().parent.parent / "vidlore" / "clipstudio" /
           "index.py").read_text(encoding="utf-8")
    check("detect_shots uses the min_clip floor",
          "_merge_short(scenes, max(cfg.min_shot_sec, cfg.min_clip_sec))" in src)

    # mistyped local path → clean 'file not found', not a yt-dlp extractor error
    proj = M.ClipProject(name="t", root=tempfile.mkdtemp(prefix="csloc_"))
    sv = IN.ingest_one(IN.SourceSpec(ref="/no/such/file.mp4", permission="owner"),
                       proj, ClipConfig())
    check("missing local path reported as such",
          sv.status == "download_failed" and "local file not found" in (sv.error or ""))
    # …while URL-shaped refs still go to the URL path (status failed is fine offline;
    # the point is it is NOT 'local file not found')
    check("scheme-less domain still treated as URL",
          IN.SourceSpec(ref="www.youtube.com/watch?v=x", permission="owner").is_local is False)

    # ffprobe: explicit env override must win
    from vidlore.clipstudio import config as C
    probe_src = (Path(__file__).resolve().parent.parent / "vidlore" / "clipstudio" /
                 "config.py").read_text(encoding="utf-8")
    check("VIDLORE_FFPROBE override first in candidates",
          probe_src.index('os.environ.get("VIDLORE_FFPROBE"')
          < probe_src.index('_op.join(_op.dirname(ffmpeg_exe()), "ffprobe"'))
    check("no hardcoded home dir in ffprobe candidates",
          "/Users/hussnain" not in probe_src and 'expanduser("~/pinokio' in probe_src)

    # discover_prefer_height is live (default tiers unchanged: 1080/720/480)
    dsrc = (Path(__file__).resolve().parent.parent / "vidlore" / "clipstudio" /
            "discover.py").read_text(encoding="utf-8")
    check("prefer-height knob wired into HD bonus", "cfg.discover_prefer_height or 720" in dsrc)

    # pot server: lock + closed log fd + no hardcoded /tmp
    hsrc = (Path(__file__).resolve().parent.parent / "vidlore" / "clipstudio" /
            "hd_download.py").read_text(encoding="utf-8")
    check("pot server start is lock-guarded", "with _POT_LOCK:" in hsrc)
    check("pot log uses tempdir, fd closed",
          "gettempdir()" in hsrc and "logf.close()" in hsrc and '"/tmp"' not in hsrc)


# ===========================================================================
# 10) ingest merge + permission validation + download dedup + web job eviction
# ===========================================================================

def test_ingest_and_web_hygiene():
    print("[10] ingest merge/permissions, download dedup, web eviction")
    from vidlore.clipstudio import ingest as IN
    from vidlore.clipstudio import download as DL
    from vidlore.clipstudio import models as M
    from vidlore.clipstudio.config import ClipConfig
    from vidlore.clipstudio.discover import SourceCandidate

    # incremental ingest must PRESERVE previously-ingested sources
    tmp = Path(tempfile.mkdtemp(prefix="csing_"))
    proj = M.ClipProject(name="t", root=str(tmp))
    proj.sources = [M.SourceVideo(id="old_src", url="u://old", title="old",
                                  permission="owner", status="ok")]
    orig_one = IN.ingest_one
    try:
        IN.ingest_one = lambda sp, p, c: M.SourceVideo(
            id=IN._source_id(sp), url=sp.ref, title=sp.title or "new",
            permission=sp.permission, status="ok")
        IN.ingest_sources(proj, [IN.SourceSpec(ref="u://new", permission="owner")],
                          ClipConfig())
    finally:
        IN.ingest_one = orig_one
    ids = [s.id for s in proj.sources]
    check("incremental ingest keeps prior sources",
          "old_src" in ids and len(ids) == 2)

    # a typo'd permission must BLOCK, not pass as an assertion
    sv = IN.ingest_one(IN.SourceSpec(ref="u://x", permission="fair-use"),
                       proj, ClipConfig())
    check("unknown permission string blocks ingest",
          sv.status == "blocked_no_permission" and sv.permission == "unverified")

    # the same video discovered twice must download once
    calls = []
    orig_dl = DL._download_one
    try:
        def fake_dl(c, sid, perm, note, p, cfg, progress=None):
            calls.append(sid)
            return M.SourceVideo(id=sid, url=c.url, title=c.title, permission=perm, status="ok")
        DL._download_one = fake_dl
        proj2 = M.ClipProject(name="t2", root=str(Path(tempfile.mkdtemp(prefix="csdl2_"))))
        cand = SourceCandidate(url="https://example.com/v1", title="Same Video")
        DL.download_candidates(proj2, [cand, SourceCandidate(url="https://example.com/v1",
                                                             title="Same Video")],
                               ClipConfig(), policy="approved_testing")
        check("duplicate candidate submitted once", len(calls) == 1)
    finally:
        DL._download_one = orig_dl

    # web job registry eviction is bounded and only touches DONE jobs
    from vidlore.clipstudio import web as W
    import time as _t
    with W._LOCK:
        W._JOBS.clear()
    now = _t.time()
    with W._LOCK:
        W._JOBS["stale"] = {"done": True, "started": now - 100000}
        W._JOBS["fresh_done"] = {"done": True, "started": now - 10}
        W._JOBS["running"] = {"done": False, "started": now - 100000}
    W._evict_jobs()
    with W._LOCK:
        keys = set(W._JOBS)
    check("stale done job evicted, running + fresh kept",
          keys == {"fresh_done", "running"})
    with W._LOCK:
        W._JOBS.clear()


# ===========================================================================
# 9) build.py media fixes — watermark crop, logo vote, theme map, env parses
# ===========================================================================

def test_build_media_fixes():
    print("[9] build.py: watermark crop, logo vote, theme buckets, env hygiene")
    from vidlore.clipstudio import build as B
    from vidlore.clipstudio import hd_download as HD

    # watermark crop must not hard-rescale (assemble normalizes AR-safely to 1920x1080)
    f = B._watermark_crop_filter("br")
    check("no hard scale in watermark crop", "scale=" not in f and f.startswith("crop="))
    check("br crop keeps top-left region", f.endswith(":0:0"))
    f2 = B._watermark_crop_filter("tl")
    check("tl crop keeps bottom-right region", "iw*0.160" in f2 and "ih*0.160" in f2)

    src = (Path(__file__).resolve().parent.parent / "vidlore" / "clipstudio" /
           "build.py").read_text(encoding="utf-8")
    check("logo vote requires a true corner (both axes off-center)",
          "(cx < 0.3 or cx > 0.7) and (cy < 0.3 or cy > 0.7)" in src)
    check("logo vote skips frame-wide subtitle strips", "bw > 0.45" in src)
    check("caption-dodge is per-beat", "PER-BEAT windows" in src)
    check("slow-mo failure re-cuts full length", "re-cut the full beat length un-slowed" in src)

    # every REAL engine theme must map to a music bucket (crime must not get history music)
    from vidlore.themes import THEMES
    missing = [t for t in THEMES if t not in B._MUSIC_BUCKET]
    check(f"all engine themes mapped to music buckets {missing}", not missing)
    check("crime maps to dark bucket", B._MUSIC_BUCKET["crime"] == "dark_investigation")

    # env parsing hygiene
    check("HD_DOWNLOAD parse is case-insensitive with empty=on",
          '.strip().lower()' in (Path(__file__).resolve().parent.parent / "vidlore" /
                                 "clipstudio" / "hd_download.py").read_text(encoding="utf-8")
          and isinstance(HD.HD_ENABLED, bool))
    saved = os.environ.get("VIDLORE_CLIPSTUDIO_QUERY_CAP")
    try:
        os.environ["VIDLORE_CLIPSTUDIO_QUERY_CAP"] = "garbage"
        from vidlore.clipstudio.config import _i
        check("bad numeric env falls back to default", _i("VIDLORE_CLIPSTUDIO_QUERY_CAP", 60) == 60)
    finally:
        if saved is None:
            os.environ.pop("VIDLORE_CLIPSTUDIO_QUERY_CAP", None)
        else:
            os.environ["VIDLORE_CLIPSTUDIO_QUERY_CAP"] = saved


# ===========================================================================
# 7) pipeline correctness — PipelineError, source gate, LLM hygiene
# ===========================================================================

def test_pipeline_error_and_gates():
    print("[7] PipelineError + produce() source gate + LLM hygiene")
    import time
    from vidlore.clipstudio import orchestrate as O
    from vidlore.clipstudio import models as M
    from vidlore.clipstudio.config import ClipConfig
    from vidlore.clipstudio import llm as L

    check("PipelineError is a normal Exception (catchable by the web worker)",
          issubclass(O.PipelineError, Exception) and not issubclass(O.PipelineError, SystemExit))
    # VISUAL-FILTER fail-CLOSED: with no CLIP model the animated / video-game / toy footage gate can't
    # run, so a render would SILENTLY ship cartoon/game footage (a fresh Windows box did — full of
    # Telltale 'Game of Thrones' game cut-scenes). produce_auto now REFUSES up front, not fails open.
    import vidlore.clipstudio.index as _ix2
    _save_clip = _ix2.clip_available
    _ix2.clip_available = lambda: False
    _noclip_refused = False
    try:
        O.produce_auto(tempfile.mkdtemp(prefix="noclip_"), topic="x", script_text="a b c",
                       movie_hint="Y", do_build=False)
    except O.PipelineError as _e:
        _noclip_refused = "CLIP visual model not loaded" in str(_e)
    except Exception:
        _noclip_refused = False
    finally:
        _ix2.clip_available = _save_clip
    check("no-CLIP render fails CLOSED (won't silently ship animated/game footage)", _noclip_refused)

    # produce() must abort when every permitted source failed to download
    tmp = Path(tempfile.mkdtemp(prefix="csgate_"))
    proj = M.ClipProject(name="t", root=str(tmp))
    proj.sources = [M.SourceVideo(id="s1", url="u://x", title="t", permission="owner",
                                  status="download_failed")]
    proj.save()
    try:
        O.produce(tmp, script_text="hello world", cfg=ClipConfig(), do_build=False)
        check("produce() aborts on all-failed sources", False)
    except O.PipelineError:
        check("produce() aborts on all-failed sources", True)
    except SystemExit:
        check("produce() aborts on all-failed sources", False)

    # keyless (no Claude / DeepSeek / Gemini) must fail FAST (no ~5.6s of empty-string retries)
    saved = os.environ.pop("ANTHROPIC_API_KEY", None)
    saved_ds = os.environ.pop("DEEPSEEK_API_KEY", None)   # DeepSeek is now a provider too
    orig_gem = L.gemini_available
    try:
        L.gemini_available = lambda: False
        t0 = time.time()
        out = L.complete(messages=[{"role": "user", "content": "hi"}], eng_cfg=None)
        dt = time.time() - t0
        check("keyless LLM call returns '' fast", out == "" and dt < 1.0)
    finally:
        L.gemini_available = orig_gem
        if saved_ds is not None:
            os.environ["DEEPSEEK_API_KEY"] = saved_ds
    # DeepSeek provider: selectable as PRIMARY (OpenAI-compatible, text-only). When primary, a
    # VISION call (image message) must skip it so a vision-capable provider (Claude/Gemini) serves.
    _sv_p = os.environ.get("VIDLORE_CLIPSTUDIO_LLM_PROVIDER")
    _sv_k = os.environ.get("DEEPSEEK_API_KEY")
    try:
        os.environ["DEEPSEEK_API_KEY"] = "sk-unit-test"
        os.environ["VIDLORE_CLIPSTUDIO_LLM_PROVIDER"] = "deepseek"
        check("DeepSeek selectable as primary provider",
              "deepseek" in L.active_provider().lower() and L.deepseek_available())
        check("DeepSeek (text-only) skips vision calls → fallback serves images",
              L._msgs_have_image([{"role": "user", "content": [{"type": "image", "source": {}}]}])
              and L._deepseek_complete("", [{"role": "user", "content": [{"type": "image", "source": {}}]}], 64, "") == "")
        check("OpenAI message shape conversion (system + text)",
              L._to_openai_messages("SYS", [{"role": "user", "content": "hi"}])
              == [{"role": "system", "content": "SYS"}, {"role": "user", "content": "hi"}])
    finally:
        if _sv_p is None:
            os.environ.pop("VIDLORE_CLIPSTUDIO_LLM_PROVIDER", None)
        else:
            os.environ["VIDLORE_CLIPSTUDIO_LLM_PROVIDER"] = _sv_p
        if _sv_k is None:
            os.environ.pop("DEEPSEEK_API_KEY", None)
        else:
            os.environ["DEEPSEEK_API_KEY"] = _sv_k
        if saved is not None:
            os.environ["ANTHROPIC_API_KEY"] = saved

    # ANTHROPIC_MODEL read at call time (the engine loads .env after import)
    saved_m = os.environ.get("ANTHROPIC_MODEL")
    try:
        os.environ["ANTHROPIC_MODEL"] = "claude-test-late-bind"
        check("ANTHROPIC_MODEL late-bound", L._claude_model() == "claude-test-late-bind")
    finally:
        if saved_m is None:
            os.environ.pop("ANTHROPIC_MODEL", None)
        else:
            os.environ["ANTHROPIC_MODEL"] = saved_m

    # a non-numeric verifier confidence must not crash the QC report
    from vidlore.clipstudio import review as R
    proj2 = M.ClipProject(name="t2", root=str(Path(tempfile.mkdtemp(prefix="csrev_"))))
    seg = M.ScriptSegment(index=0, text="line")
    sel = M.ClipSelection(segment_index=0, source_id="s", shot_index=0, in_point=0,
                          out_point=2, confidence=0.5,
                          verifier={"status": "ok", "verdict": "keep", "confidence": None})
    proj2.selections = [sel]
    try:
        p = R.write_review(proj2, [seg])
        check("write_review survives null verifier confidence", Path(p).exists())
    except Exception as e:
        check(f"write_review survives null verifier confidence ({e})", False)


# ===========================================================================
# 8) index cache + atomic saves
# ===========================================================================

def test_index_cache_and_atomic_saves():
    print("[8] index resume cache (capabilities, corrupt cache) + atomic project save")
    from vidlore.clipstudio import index as IX
    from vidlore.clipstudio import models as M
    from vidlore.clipstudio.config import ClipConfig

    tmp = Path(tempfile.mkdtemp(prefix="csidx_"))
    proj = M.ClipProject(name="t", root=str(tmp))
    proj.ensure_dirs()
    src = M.SourceVideo(id="s1", url="u", title="t", permission="owner",
                        status="download_failed")        # never indexable fresh
    cfg = ClipConfig()

    # a corrupt cached shots.json must not crash — it falls through to the (skipped) fresh path
    proj.shots_path("s1").write_text("{not json", encoding="utf-8")
    try:
        out = IX.index_source(proj, src, cfg)
        check("corrupt shots.json cache tolerated", out == [])
    except Exception as e:
        check(f"corrupt shots.json cache tolerated ({e})", False)

    # a roster-less cache must NOT satisfy a Face-ID-wanting call (forces re-index path)
    shot = M.Shot(source_id="s1", index=0, start=0.0, end=2.0)
    proj.shots_path("s1").write_text(json.dumps([shot.to_dict()]), encoding="utf-8")
    (proj.index_dir / "s1.index.meta.json").write_text(
        json.dumps({"faceid": False, "ocr": False}), encoding="utf-8")
    msgs = []
    out = IX.index_source(proj, src, cfg, references={"actor": object()}, faceid=object(),
                          progress=msgs.append)
    check("faceid-less cache rejected for a faceid call",
          out == [] and any("re-indexing" in m and "faceid" in m for m in msgs))

    # a pre-schema-2 cache (no word-level ASR) must NOT be served: quote location would silently
    # fall back to per-shot substring search, which cannot see a line that straddles a cut.
    msgs_w = []
    cfg_noocr = ClipConfig()
    cfg_noocr.detect_ocr = False
    out_w = IX.index_source(proj, src, cfg_noocr, progress=msgs_w.append)
    check("pre-schema-2 (word-less) cache rejected",
          out_w == [] and any("re-indexing" in m and "words" in m for m in msgs_w))

    # same cache, same (no-cap) call, now schema-2 complete → still served from cache
    (proj.index_dir / "s1.index.meta.json").write_text(
        json.dumps({"faceid": False, "ocr": False, "words": True,
                    "schema": IX.INDEX_SCHEMA}), encoding="utf-8")
    (proj.index_dir / "s1.words.json").write_text("[]", encoding="utf-8")
    msgs2 = []
    out2 = IX.index_source(proj, src, cfg_noocr, progress=msgs2.append)
    check("matching-capability cache still served",
          len(out2) == 1 and any("cached" in m for m in msgs2))

    # a silent source legitimately has ZERO words; its cache must still be served, never re-indexed
    # forever (meta records that the pass RAN, not that it found anything)
    msgs3 = []
    out3 = IX.index_source(proj, src, cfg_noocr, progress=msgs3.append)
    check("silent source (0 words) cache is stable, not re-indexed forever",
          len(out3) == 1 and not any("re-indexing" in m for m in msgs3))

    # meta claims words but the file is gone → re-index (don't trust the claim alone)
    (proj.index_dir / "s1.words.json").unlink()
    msgs4 = []
    out4 = IX.index_source(proj, src, cfg_noocr, progress=msgs4.append)
    check("words cache with a missing file is rejected",
          out4 == [] and any("re-indexing" in m and "words" in m for m in msgs4))
    (proj.index_dir / "s1.words.json").write_text("[]", encoding="utf-8")

    # atomic save: project.json gets replaced, no .tmp left behind
    proj.save()
    check("project.json saved", proj.manifest_path.exists())
    check("no tmp manifest left", not proj.manifest_path.with_suffix(".json.tmp").exists())

    import vidlore.clipstudio.ocr as OCR
    check("names_in_text tolerates None roster", OCR.names_in_text("FARRAH FAWCETT", None) == [])



# ===========================================================================
# 14) real-audio breakouts — narration pauses, the scene speaks
# ===========================================================================

def test_breakout_intelligence():
    print("[14] real-audio breakouts")
    from vidlore.clipstudio.build import _quote_run_in
    qw = "you're just like me only smaller".split()
    check("verbatim quote located in ASR",
          _quote_run_in(qw, "the killing you're just like me only smaller hmm".split()) >= 5)
    check("3-run drift tolerated",
          _quote_run_in(qw, "just like me he said quietly".split()) == 3)
    check("unrelated ASR rejected",
          _quote_run_in(qw, "the rains of castamere plays loudly".split()) == 0)
    src = (Path(__file__).resolve().parent.parent / "vidlore" / "clipstudio" /
           "build.py").read_text(encoding="utf-8")
    check("breakout audio spliced into narration track", "_splice_audio" in src
          and "narration_breakouts.wav" in src)
    check("words shifted after insertion", "w.start += shift" in src)
    # ATOMICITY: _apply_breakouts shifts word-times then splices audio. If the splice fails it RAISES,
    # but it used to have already mutated the words IN PLACE → captions left shifted while the audio
    # was un-spliced = a constant caption lag (observed ~4.4s on an uploaded-VO render when a t=0
    # cold-open splice failed and the caller swallowed the exception). Now it works on COPIES.
    check("breakout apply is ATOMIC — copies so a failed splice can't desync captions",
          "ATOMIC: reindex + word-shift on COPIES" in src
          and "ns.words = [_copy.copy(w) for w in" in src
          and "seg = _copy.copy(seg)" in src)
    check("splice handles a t=0 (cold-open) prepend without an empty atrim segment",
          "emit the original slice only if NON-empty" in src)
    # READ==WRITE: a 2nd splice pass (a cold-open after the mid-breakout pass) fed ffmpeg the same
    # narration_breakouts.wav as input AND output → it failed, which silently disabled the cold-open
    # on uploaded voiceover. Now a fresh numbered sibling is used when input == default dest.
    check("splice never read==writes its input (cold-open can fire after mid-breakouts)",
          "NEVER read==write" in src
          and "Path(full).resolve() == dest.resolve()" in src
          and 'f"narration_breakouts_{k}.wav"' in src)
    # FUNCTIONAL: a failed audio splice must leave the narration word-times PRISTINE (no desync).
    import types as _tt
    from vidlore.clipstudio import build as B
    from vidlore.tts import WordTiming as _WT, NarratedScene as _NSc, Narration as _Narr
    _w = lambda t, s, e: _WT(t, s, e)
    _ns0 = _NSc(index=0, audio=Path("/dev/null"), duration=3.0, words=[_w("hello", 0.0, 0.5), _w("world", 0.5, 1.0)])
    _ns1 = _NSc(index=1, audio=Path("/dev/null"), duration=3.0, words=[_w("foo", 3.0, 3.5), _w("bar", 3.5, 4.0)])
    _nar = _Narr(scenes=[_ns0, _ns1], audio=Path("/dev/null"))
    _segs = [_tt.SimpleNamespace(index=0, text="hello world", est_duration=3.0),
             _tt.SimpleNamespace(index=1, text="foo bar", est_duration=3.0)]
    _scs = [_tt.SimpleNamespace(index=0), _tt.SimpleNamespace(index=1)]
    _pick = [{"seg_index": 1, "dur": 4.0, "audio": "/dev/null", "video": "/dev/null"}]
    import tempfile as _tf
    _wd = Path(_tf.mkdtemp(prefix="atom_"))
    _save_sa, _save_rp = B._splice_audio, B._refine_pause_times
    B._splice_audio = lambda *a, **k: None              # simulate a splice FAILURE
    B._refine_pause_times = lambda *a, **k: {}          # no whisper in the test
    _raised = False
    try:
        B._apply_breakouts(_tt.SimpleNamespace(), _segs, _scs, _nar, _pick, _wd, lambda *a: None)
    except RuntimeError:
        _raised = True
    except Exception:
        _raised = False
    finally:
        B._splice_audio, B._refine_pause_times = _save_sa, _save_rp
    check("failed splice raises AND original word-times stay pristine (no caption desync)",
          _raised and _nar.scenes[0].words[0].start == 0.0
          and _nar.scenes[1].words[0].start == 3.0)      # NOT 7.0 (would be the +4.0 corruption)
    # ERA-COHERENCE: a breakout plays a full scene with its own audio; a bearded S7 Tyrion on a beach
    # over an S4E10 privy scene reads as a continuity break. Later-season-only sources are barred from
    # breakouts (earlier seasons stay — the script narrates S1/S3 backstory). Aired on Tyrion render.
    check("breakout era-coherence gate bars later-season-only sources",
          "_core_seasons9" in src and "min(_ss9b) > _core_max9" in src
          and "later-season source" in src and "anchor_scenes" in src)
    check("breakout scenes excluded from hold/freeze",
          "_breakout_clip}" in src or "_breakout_clip" in src and "hold_pos = _hold" in src)
    check("tiered mining: exact episode first, scene-titled needs overlap",
          "_mine_tier([s for s in srcs if s.id in _tier1], _min_ov9)" in src
          and "_mine_tier([s for s in srcs if s.id in _tier2], _min_ov9)" in src)
    check("evidence mining merges with located quotes (not fallback-only)",
          "mined = (_mine_tier(" in src and "_seen_shot" in src)
    # the overlap floor (default 2, env BREAKOUT_MINE_MIN_OV) now holds EVERYWHERE — the old
    # zero-overlap last-resort tier aired an S2 war-council clip mid-trial-argument (3/10 twice)
    check("tier-1 breakout overlap tightened to >=2 (no 1-word tangents)",
          "_mine_tier([s for s in srcs if s.id in _tier1], 1)" not in src
          and "_mine_tier([s for s in srcs if s.id in _tier1], 0)" not in src
          and "VIDLORE_CLIPSTUDIO_BREAKOUT_MINE_MIN_OV" in src)
    check("miner overlap ignores semantically-empty words (never/said/know class)",
          '"never", "ever", "always", "said"' in src)
    check("competitor evidence density cap (~1 per 28s, max 8)",
          "n_max = max(1, min(8, 1 + int(total // 28)))" in src)
    check("verbatim quote outranks mined evidence", "run + 3 +" in src)
    # scene 0 is the title overlay, so a GENERIC scene-0 breakout is excluded — but a verbatim
    # COLD-OPEN at scene 0 (the opening hook quote) is allowed through this filter (see
    # test_intro_coldopen_breakout for the behavioral coverage)
    check("scene-0 allowed only for a verbatim cold-open (generic scene-0 still excluded)",
          "if c[1] >= 1 or (c[1], c[2].id, round(float(c[3].start), 1)) in _verbatim_strong" in src
          and "_cold_key" in src)
    check("breakout length is dialogue-aware (3-10s), not a fixed shot-length chop",
          "dur = 10.0" in src and "_dialogue_aware_dur" in src)
    check("breakout look registered in air-guard",
          "_aired_hashes.append(_frame_hash(str(_bclip), 1.0))" in src)
    # LEGIBILITY GATE: a near-black / degraded shot (e.g. the VHS 'Angels with Even Filthier Souls'
    # in-movie film, mean-luma ~55) must not air as a breakout even when its ASR overlaps the
    # narration — it reads as a dead frame. Reject on air-window mean luma; beat keeps its footage.
    from vidlore.clipstudio.build import _breakout_window_luma as _bwl
    check("breakout legibility (luma) gate present + env-tunable",
          callable(_bwl) and "_breakout_window_luma" in src
          and "VIDLORE_CLIPSTUDIO_BREAKOUT_LUMA_FLOOR" in src
          and "too dark/illegible" in src)
    check("breakout luma gate fails-open (unreadable probe never rejects)",
          "0.0 <= _blum9 < _lflo9" in src)   # a negative/unknown probe is not < floor → kept
    # COMMENTARY-AUDIO gate: a breakout plays the shot's OWN audio — if that is a YouTuber/essayist
    # analyzing the scene (3rd-person commentary), not in-character movie dialogue, it must not air.
    # Observed: 3 of 4 breakouts on a Tywin S01E07 video were essay narration that slipped the
    # title/coverage gates (their commentary about the scene overlaps the script about the scene).
    from vidlore.clipstudio.build import _is_narration as _isn, _ESSAYISH_RX as _ess
    check("commentary breakout audio detected (essayist, not in-character dialogue)",
          _isn("Basically what the red wedding is to Tywin, he's writing a new verse in the same song")
          and _isn("all that warmth, all that passion he once had for her got buried inside an idea")
          and _isn("Tywin has just orchestrated the end of the Starks, so here he is ready to discuss"))
    # REGRESSION — the v1 FIRST-breakout bug: a commentary clip ("...the black water proved Tyrion's
    # strategic brilliance") aired over the Tysha narration. Analytical/evaluative 3rd-person phrasing
    # ("proved X's strategic brilliance", "strategic genius", "demonstrates his downfall") must flag.
    check("analytical-commentary breakout audio detected (v1 first-breakout regression)",
          _isn("He doubles Bronn's pay to stay loyal. The black water proved Tyrion's strategic brilliance.")
          and _isn("Tyrion's strategic brilliance") and _isn("This scene proves his cold genius")
          and _isn("It demonstrates the character's downfall"))
    check("real in-character movie dialogue NOT flagged as commentary",
          not _isn("I still don't see why I should let you back on my council, my lord")
          and not _isn("You raped her, you killed her children, and now you bring me his bones")
          and not _isn("A Lannister always pays his debts")
          and not _isn("Say her name again")
          and not _isn("I am your son and you sentenced me to die"))
    check("breakout picking loop gates on the airing shot transcript",
          "_is_narration(getattr(c[3]" in src
          and "source audio is commentary" in src)
    check("essay/edit/reaction breakout-source titles barred (the slipped ones)",
          bool(_ess.search("The Scene Where Tywin Lannister Almost Cries"))
          and bool(_ess.search("Tywin Lannister - Motivational Speech"))
          and bool(_ess.search("Tywin educating Jaime (watching 7th episode)"))
          and not _ess.search("Game of Thrones - Jaime & Tywin Lannister Conversation"))


# ===========================================================================
# 15) breakout audio-trust gate — never air another narrator's voice
# ===========================================================================

def test_breakout_audio_gate():
    print("[15] breakout audio-trust gate")
    from vidlore.clipstudio.build import _breakout_src_ok, _english_ish

    class _Src:
        def __init__(self, title): self.title = title

    class _Shot:
        def __init__(self, words): self.transcript = " ".join(["word"] * words)

    bursty = [_Shot(0), _Shot(0), _Shot(8), _Shot(2), _Shot(0), _Shot(12), _Shot(0), _Shot(0)]
    wall = [_Shot(14)] * 8
    check("essay-ish title rejected (why/really)",
          not _breakout_src_ok(_Src("Why Was Jaqen Really in the Black Cells?"), bursty))
    check("essay-ish title rejected (you missed/real plan)",
          not _breakout_src_ok(_Src("Jaqen Used Arya Here's His Real Plan (You Missed It)"), bursty))
    check("documentary title rejected (history of)",
          not _breakout_src_ok(_Src("Building the Red Keep | History of King's Landing"), bursty))
    check("wall-to-wall narration rejected on coverage even with clean title",
          not _breakout_src_ok(_Src("Arya meets Jaqen in the black cells"), wall))
    check("real scene clip accepted (clean title + bursty dialogue)",
          _breakout_src_ok(_Src("Game of thrones | Arya meets Jaqen H'ghar"), bursty))
    # REACTION video must be barred from breakouts — breakout selection scans proj.sources directly
    # (not the match pool), and _ESSAYISH_RX's \\b...\\b misses the PLURAL "Reactions". A profanity-
    # laced reaction ("...Reactions!") leaked as a breakout on the pro Tyrion render until fixed.
    check("reaction video (plural 'Reactions') barred from breakouts",
          not _breakout_src_ok(_Src('TOP "TYRION DEMANDS TRIAL BY COMBAT" Reactions! Game of Thrones'), bursty)
          and not _breakout_src_ok(_Src("Couple Reacts to Tyrion's Trial"), bursty)
          and not _breakout_src_ok(_Src("First Time Watching Game of Thrones"), bursty))
    check("dialogue-heavy REAL scene accepted (0.67 coverage, clean title)",
          _breakout_src_ok(_Src("Varys Tells Tyrion Lannister A Riddle About Power"),
                           [_Shot(10)] * 4 + [_Shot(0)] * 2))
    check("english ASR accepted",
          _english_ish("you trusted him to fight with us and he ran".split()))
    check("french ASR rejected",
          not _english_ish("je veux retourner aux cuisines tais toi maintenant".split()))
    src = (Path(__file__).resolve().parent.parent / "vidlore" / "clipstudio" /
           "build.py").read_text(encoding="utf-8")
    check("gate wired into quote path and tiers", src.count("ok_audio") >= 4)
    check("foreign-audio guard wired into mining", "_english_ish(raw9)" in src)
    # freeze style variety (competitor: B&W+click only on the biggest punchlines,
    # quiet color stills for the rest)
    import inspect as _ins
    from vidlore.clipstudio.build import _freeze_punchline
    sig = _ins.signature(_freeze_punchline)
    check("freeze styles: bw + color still", "style" in sig.parameters
          and 'style == "bw"' in src and "saturation=1.05" in src)
    check("B&W reserved for top punchlines", "_bw_pos = set(sorted(hold_pos," in src)
    check("click SFX only on B&W freezes",
          'if _style == "bw":' in src and "freeze_marks.append" in src)
    check("hold continuation independent of click marks",
          "_last_frz_seg == seg.index" in src and "_last_frz_seg = seg.index" in src)
    # clause-safe pauses + audio polish (competitor: pause never lands mid-sentence)
    check("breakout pause delayed to the sentence end",
          "_deltas[old] = _rel + 0.10" in src
          and "splices.append((t_cursor + delta_here," in src)
    check("pause snapped to REAL word end via whisper",
          "_refine_pause_times(Path(narration.audio), _needs)" in src
          and "word_timestamps=True" in src)
    check("sentence end detected without punctuation (ASR scripts)",
          "gap >= 0.28" in src and 'nxt[:1].isupper() and nxt.lower().strip(".,!?")' in src)
    check("previous scene extended, next beat shortened",
          "new_ns[-2].duration = float(new_ns[-2].duration) + delta_here" in src
          and "ns.duration = max(0.6, _orig_dur - delta_here)" in src)
    check("pre-pause words keep earlier-shift only",
          "w.start += shift_before" in src)
    check("2-pass loudnorm on breakout audio",
          "measured_I=" in src and "linear=true" in src and "I=-16.5" in src)
    check("leading-silence trim on breakout",
          "silencedetect=noise=-30dB:d=0.4" in src and "start + _lead - 0.25" in src)
    # dialogue-aware breakout length — end on a complete spoken line, 3-10s, cap 10s
    check("breakout length ends on a complete sentence (3-10s)",
          "def _dialogue_aware_dur" in src and "lo: float = 3.0" in src
          and "hi: float = 10.0" in src)
    check("breakout passes 10s max cap (not a hard 5.5s chop)",
          "dur = 10.0" in src and "_dialogue_aware_dur(str(src_path), start, lo=_lo, hi=_hi)" in src
          # the 3s floor still holds; `lo` is now raised to the QUOTE's own length on a
          # quote-anchored window so the search can't end on an EARLIER complete line and truncate
          # the iconic one (how the nightshade payoff was lost)
          and "_lo = max(3.0, min(float(min_dur or 0.0), _hi))" in src)
    check("breakout content dedup (no repeated spoken line)",
          "_picked_word_sets" in src and "never airs the SAME moment twice" in src
          and "_same_src_win" in src and "_substr" in src)


# ===========================================================================
# 16) burned-in text gate — footage with readable on-screen text never airs
# ===========================================================================

def test_text_gate():
    print("[16] burned-in text gate")
    from vidlore.clipstudio.match import _ocr_text_heavy

    class _Sh:
        def __init__(self, t): self.ocr_text = t

    check("tweet/quote overlay rejected",
          _ocr_text_heavy(_Sh('"Jaime\'s hand magically healed in Episode 5," one fan groused')))
    check("hard subtitle rejected",
          _ocr_text_heavy(_Sh("I want to go back to the kitchens")))
    check("title slate rejected", _ocr_text_heavy(_Sh("THE RED KEEP")))
    check("corner logo kept (watermark crop handles it)",
          not _ocr_text_heavy(_Sh("max")))
    check("known channel bug kept", not _ocr_text_heavy(_Sh("HBO")))
    check("ALL-CAPS animated caption rejected (one word per keyframe)",
          _ocr_text_heavy(_Sh("TWO")) and _ocr_text_heavy(_Sh("EVERY SINGLESTRIKE")))
    check("clean frame kept", not _ocr_text_heavy(_Sh("")))
    msrc = (Path(__file__).resolve().parent.parent / "vidlore" / "clipstudio" /
            "match.py").read_text(encoding="utf-8")
    check("hard gate in candidate pool",
          "tgate_on and _ocr_text_heavy(ps.shot)" in msrc)
    bsrc = (Path(__file__).resolve().parent.parent / "vidlore" / "clipstudio" /
            "build.py").read_text(encoding="utf-8")
    check("window air-check gated", "_shot_has_text(w[0], float(w[1]))" in bsrc)
    check("fill-walk gated", "_TGATE and _txt_heavy(sh)" in bsrc)
    # breakout shots are gated through _texty9 (OCR text gate OR the script-agnostic
    # subtitle band — Arabic/Turkish burned subs OCR to nothing readable)
    check("breakout shots gated", bsrc.count("if _texty9(sh):") >= 2
          and "_tgate9 and _txt9(sh)" in bsrc and "_sbgate9 and _sub9(sh)" in bsrc)
    check("air-time frame OCR probe exists",
          "def _frame_has_burned_text" in bsrc
          and "_frame_has_burned_text(_wsp, float(_wh[0][1]) + _q9)" in bsrc
          and "has_big_text(png)" in bsrc)
    check("fill-walk air-time probe", "air-time probe (keyframe can miss)" in bsrc)
    check("breakout picks probed", "air-time probe: text never airs" in bsrc)
    # the cartoon/tweet-card flash root cause: the 0.2s lead-in crossed a shot boundary
    # into the PREVIOUS (unrelated) shot — OCR can't read stylized fake text, so the cut
    # must simply never cross backwards
    check("lead-in snapped to containing shot start",
          "start = max(0.0, in_p - 0.2, (float(_csh.start) if _csh else 0.0))" in bsrc)


# ===========================================================================
# 17) character relevance — Face-ID refs, wrong-character penalty, non-show gate
# ===========================================================================

def test_character_relevance():
    print("[17] character relevance (face refs + wrong-char penalty + non-show gate)")
    # --- Fix A: Face-ID reference download robustness ---
    fsrc = (Path(__file__).resolve().parent.parent / "vidlore" / "clipstudio" /
            "faceid.py").read_text(encoding="utf-8")
    check("wikimedia-compliant User-Agent (contact URL)",
          "github.com/vidlore/clipstudio" in fsrc and "_WIKI_UA" in fsrc)
    check("download retries on 429/503", "if r.status_code in (429, 503)" in fsrc)
    check("page-images fallback when lead photo is faceless",
          "_wiki_more_image_urls" in fsrc and "cand_urls.extend(_wiki_more_image_urls" in fsrc)
    check("missing-reference logged loudly", "NO REFERENCE for" in fsrc)

    # --- Fix C: non-show / wrong-medium footage gate ---
    from vidlore.clipstudio.discover import _NONSHOW_TITLE, _reject
    nonshow = ["TYWIN LANNISTER VS ROOSE BOLTON l Battle on the Green Fork",
               "Battle Of The Green Fork(299 AC) | House Lannister VS House Stark",
               "Battle of Winterfell Two Steps from Hell - Archangel",
               "Game of Thrones Total War mod gameplay", "GoT AMV - Warriors"]
    real = ["Robb Stark captures Jaime Lannister",
            "Game of thrones | Robb stark threatens Jaime Lannister",
            "Rob Stark the king in the north | Scene | Game of thrones S01E08",
            "Game of Thrones - Varys and Ned dungeon scene"]
    check("non-show titles rejected (game/AMV/animated/(NNN AC)/house-vs)",
          all(_NONSHOW_TITLE.search(t) for t in nonshow))
    check("real scene titles kept", not any(_NONSHOW_TITLE.search(t) for t in real))
    # RENDERED/CUT-SCENE GAME footage — the Telltale 'Game of Thrones' game looks cartoonish, not the
    # live-action show; its cut-scene compilations ('Telltale' / 'all cutscenes' / 'game movie', no
    # 'gameplay' word) slipped the gate and flooded a Theon/Ironborn render (the game is set in that era).
    check("Telltale / cut-scene GAME footage rejected (rendered, not the live-action show)",
          bool(_NONSHOW_TITLE.search("Game of Thrones Telltale - All Cutscenes (Game Movie)"))
          and bool(_NONSHOW_TITLE.search("GoT Telltale Walkthrough Part 1"))
          and bool(_NONSHOW_TITLE.search("Game of Thrones video game cutscene"))
          and not _NONSHOW_TITLE.search("Theon rescues Sansa - Game of Thrones S05E10"))

    class _C:
        def __init__(s, title): s.title = title; s.duration = 300; s.height = 1080
    from vidlore.clipstudio.config import ClipConfig
    cfg = ClipConfig()
    check("_reject flags non-show",
          _reject(_C("Battle (299 AC) House Stark VS House Lannister"), cfg).startswith("non-show"))
    check("_reject keeps real scene", _reject(_C("Robb Stark captures Jaime Lannister"), cfg) == "")
    msrc = (Path(__file__).resolve().parent.parent / "vidlore" / "clipstudio" /
            "match.py").read_text(encoding="utf-8")
    check("non-show source dropped from match pool",
          "_NONSHOW_TITLE.search(src.title" in msrc)

    # --- Fix B: wrong-character penalty in _score_pool ---
    from vidlore.clipstudio.match import _score_pool
    from vidlore.clipstudio.models import Shot, ScriptSegment
    seg = ScriptSegment(index=0, text="Tywin Lannister won the battle",
                        required_entity="Tywin Lannister", required_kind="character")
    # a shot confidently identified as ROBB (wrong character for a Tywin beat)
    robb = Shot(source_id="s", index=0, start=0.0, end=3.0)
    robb.identities = [{"name": "Robb Stark", "confident": True, "score": 0.8}]
    # a clean shot with no identity (generic battlefield)
    generic = Shot(source_id="s", index=1, start=3.0, end=6.0)
    from vidlore.clipstudio.match import _PoolShot
    pool = [_PoolShot("s", robb, None), _PoolShot("s", generic, None)]
    faces = {"tywin lannister", "charles dance", "robb stark", "richard madden",
             "jaime lannister", "nikolaj coster-waldau"}
    scored = _score_pool(seg, pool, None, cfg, {"tywin lannister", "charles dance"},
                         all_faces=faces)
    by_shot = {ps.shot.index: (base, sig) for base, _b, sig, ps in scored}
    check("confirmed wrong character flagged", by_shot[0][1].get("wrongface") is True)
    check("wrong character penalized below generic shot",
          by_shot[0][0] < by_shot[1][0])
    check("wrongface_penalty config knob exists", hasattr(cfg, "wrongface_penalty"))


# ===========================================================================
# 18) exact-scene IMAGE fallback (web search + validation + Ken-Burns render)
# ===========================================================================

def test_image_fallback():
    print("[18] exact-scene image fallback")
    from vidlore.clipstudio import web_images as W
    # domain hygiene
    check("blacklist domains rejected",
          W.is_unusable_domain("https://www.pinterest.com/x.jpg")
          and W.is_unusable_domain("https://i.pinimg.com/y.jpg"))
    check("watermark stock rejected", W.is_unusable_domain("https://www.alamy.com/z.jpg"))
    check("real domain kept", not W.is_unusable_domain("https://screenrant.com/a.jpg"))
    check("bing default-on, env-gated",
          W._env_flag("VIDLORE_CLIPSTUDIO_IMG_BING", True))

    from vidlore.clipstudio import image_fallback as IF
    from vidlore.clipstudio.models import ScriptSegment
    from vidlore.clipstudio.analyze import ScriptAnalysis
    ana = ScriptAnalysis(topic="t", movie_title="Game of Thrones",
                         characters=[{"name": "Robb Stark", "actor": "Richard Madden"}])
    seg = ScriptSegment(index=3, text="Robb releases the scout",
                        required_entity="Robb Stark", required_kind="character",
                        scene_query="Robb Stark in the woods with his army")
    q = IF.build_query(seg, ana)
    check("query uses show + character + scene, not raw narration",
          "game of thrones" in q.lower() and "robb stark" in q.lower()
          and "releases" not in q.lower())

    bsrc = (Path(__file__).resolve().parent.parent / "vidlore" / "clipstudio" /
            "build.py").read_text(encoding="utf-8")
    check("image beats render as Ken-Burns stills",
          "_image_kenburns_clip" in bsrc and "image-still beat" in bsrc)
    osrc = (Path(__file__).resolve().parent.parent / "vidlore" / "clipstudio" /
            "orchestrate.py").read_text(encoding="utf-8")
    check("fallback breaks repeated FILLER beats with distinct source-frame stills",
          "_fill_image_fallbacks" in osrc and "aired_shots" in osrc
          and "pick_source_still" in osrc and "_policy.FILLER" in osrc)
    check("selection carries image_path",
          'image_path: str = ""' in (Path(__file__).resolve().parent.parent / "vidlore" /
                                      "clipstudio" / "models.py").read_text(encoding="utf-8"))
    ifsrc = (Path(__file__).resolve().parent.parent / "vidlore" / "clipstudio" /
             "image_fallback.py").read_text(encoding="utf-8")
    check("validation: clip relevance + face verdict + text/collage reject",
          "_clip_relevance" in ifsrc and "_face_verdict" in ifsrc
          and "has_big_text" in ifsrc and "_aspect" in ifsrc)
    check("wrong-character image hard-rejected", 'face == "wrong"' in ifsrc)
    check("live-action guard rejects illustration/game/cartoon",
          "_photographic_ok" in ifsrc and "video game screenshot" in ifsrc)
    check("collage seam + aspect guard", "_has_center_seam" in ifsrc and "ar > 1.95" in ifsrc)
    check("game-art / wallpaper / merch domains blocked",
          W.is_unusable_domain("https://www.gotconquest.com/x.jpg")
          and W.is_unusable_domain("https://wallpapercave.com/y.jpg")
          and not W.is_unusable_domain("https://gamerant.com/z.jpg"))
    check("AI-art / clipart-farm domains blocked (subdomains too)",
          W.is_unusable_domain("https://animalia-life.club/x.jpg")
          and W.is_unusable_domain("https://www.animalia-life.club/x.jpg")
          and W.is_unusable_domain("https://ar.inspiredpencil.com/y.jpg")
          and W.is_unusable_domain("https://craiyon.com/z.jpg"))
    # WARM-CACHE re-filter: a search cache written before a domain was blacklisted must NOT
    # resurrect that domain on a later run (this is what leaked animalia-life.club 3x).
    import json as _json, time as _time, hashlib as _hl, tempfile as _tf, os as _os3
    _os3.environ["VIDLORE_CLIPSTUDIO_IMG_CACHE"] = _tf.mkdtemp()
    _q = "warm cache blacklist regression"
    _c = W._cache_root() / "search" / (_hl.sha256(_q.encode()).hexdigest()[:24] + ".json")
    _c.write_text(_json.dumps({"t": _time.time(), "r": [
        {"image_url": "https://animalia-life.club/a.jpg"},
        {"image_url": "https://ar.inspiredpencil.com/b.jpg"},
        {"image_url": "https://www.refinery29.com/ok.jpg"}]}))
    check("warm search cache re-applies the (grown) domain blacklist",
          [r["image_url"] for r in W.search_images(_q)] == ["https://www.refinery29.com/ok.jpg"])
    # AI-generator SOURCE page rejected even when the image sits on a neutral CDN (shedevrum.ai
    # parked on avatars.mds.yandex.net); a clipart aggregator that hot-links a real still is NOT
    # an AI source (animalia-life.club → static.hbo.com stays judged by host + photographic guard).
    check("AI-generator source page rejected (CDN-hosted synthetic art)",
          W.is_ai_generated_source("shedevrum.ai")
          and W.is_ai_generated_source("https://www.craiyon.com/result/x")
          and not W.is_ai_generated_source("www.animalia-life.club")
          and not W.is_ai_generated_source("static.hbo.com"))
    check("image-fallback drops AI-generator source candidates",
          "is_ai_generated_source" in ifsrc)
    # era anchor in the image query — a character beat in a known-episode video leans to the
    # right season so a bare 'Cersei' search can't return short-hair S6 stills for an S1 scene
    from vidlore.clipstudio.image_fallback import build_query as _bq
    from vidlore.clipstudio.analyze import ScriptAnalysis as _SA
    from vidlore.clipstudio.models import ScriptSegment as _SS
    _an = _SA(topic="t", movie_title="Game of Thrones", episode_hint="S01E05")
    _sg = _SS(index=1, text="Cersei watches", scene_query="Cersei Lannister watchful chamber",
              required_kind="character", required_entity="Cersei Lannister")
    check("image query carries the era hint (season 1)", "season 1" in _bq(_sg, _an).lower())
    # SOURCE-FRAME stills: a weak beat draws a REAL keyframe from the matcher's ranked alternates
    # (100% real footage, no web/AI risk), preferring a shot DISTINCT from the assigned moving clip
    # and not already used — this is the safe replacement for web images + breaks filler repetition.
    from vidlore.clipstudio.image_fallback import pick_source_still as _pss
    from vidlore.clipstudio.models import ClipSelection as _CS, ClipCandidate as _CC, Shot as _SH
    import tempfile as _tf2, os as _os4
    _kf = _os4.path.join(_tf2.mkdtemp(), "kf.jpg"); open(_kf, "wb").write(b"\xff\xd8\xff\xe0jpgdata" * 50)
    _shots = {
        ("srcA", 5): _SH(source_id="srcA", index=5, start=0, end=4, keyframe_path=_kf,
                         quality=0.9, phash="abc"),
        ("srcA", 2): _SH(source_id="srcA", index=2, start=0, end=4, keyframe_path=_kf,
                         quality=0.1, phash="lowq"),   # too blurry → skipped
    }
    _sel = _CS(segment_index=1, source_id="srcA", shot_index=9, in_point=0, out_point=3,
               confidence=0.3,
               alternates=[_CC(segment_index=1, source_id="srcA", shot_index=9, score=0.8),  # == assigned → skip
                           _CC(segment_index=1, source_id="srcA", shot_index=2, score=0.7),  # low quality → skip
                           _CC(segment_index=1, source_id="srcA", shot_index=5, score=0.6)]) # ✓ distinct, sharp
    _r = _pss(_sel, _shots, set(), set())
    check("source-still picks a distinct, sharp alternate keyframe (not the assigned shot)",
          _r is not None and _r[1] == "srcA" and _r[2] == 5)
    check("source-still respects cross-beat dedup (already-used shot skipped)",
          _pss(_sel, _shots, {("srcA", 5)}, set()) is None)
    # ANGLE / VERSION query variants: surface DISTINCT uploads (and thus angles) of the SAME scene
    # so the matcher has more visually-different REAL footage → less repetition without leaving it.
    from vidlore.clipstudio.discover import anchor_queries as _aq
    _aa = _SA(topic="t", movie_title="Game of Thrones", episode_hint="S01E05",
              characters=[{"name": "Robert Baratheon"}, {"name": "Cersei Lannister"}],
              anchor_scenes=[{"name": "Robert and Cersei chamber",
                              "query": "Robert Cersei royal chambers talk"}])
    _aqs = _aq(_aa)
    _variants = [q for q in _aqs if any(v in q.lower() for v in
                 ("full scene", "scene hd", "4k", "extended", "no music", "complete scene"))]
    check("angle/version anchor-query variants generated (more distinct same-scene uploads)",
          len(_variants) >= 4
          and any("robert" in q.lower() and "cersei" in q.lower() for q in _variants))
    from vidlore.clipstudio.image_fallback import phash_hamming as _ph
    check("phash hamming distance (perceptual-repeat detection)",
          _ph("0000000000000000", "0000000000000000") == 0
          and _ph("ffffffffffffffff", "0000000000000000") == 64)
    # REACTION/facecam guard: a reaction video plays the real scene audio (so dialogue-verify WOULD
    # re-admit it) but its footage is people on a couch — must stay rejected. Caught the leak in the
    # Tyrion-trial test render ("Reactors Reacting to...", "...Reactions!").
    from vidlore.clipstudio.discover import _REACTION_TITLE as _RT
    check("reaction/facecam titles flagged",
          bool(_RT.search("Reactors Reacting to TYRION DEMANDS TRIAL BY COMBAT"))
          and bool(_RT.search('TOP "TYRION DEMANDS TRIAL BY COMBAT" Reactions!'))
          and bool(_RT.search("Couple Reacts to Tyrion Trial"))
          and bool(_RT.search("First Time Watching Game of Thrones")))
    check("clean scene titles NOT flagged as reaction",
          not _RT.search("Tyrion Demands a Trial by Combat | Game of Thrones")
          and not _RT.search("Tyrion's Trial. Game Of Thrones S04E06.")
          and not _RT.search("Betrayal of Shae with Tyrion Lannister"))
    _dsrc_r = (Path(__file__).resolve().parent.parent / "vidlore" / "clipstudio" /
               "discover.py").read_text(encoding="utf-8")
    _msrc_r = (Path(__file__).resolve().parent.parent / "vidlore" / "clipstudio" /
               "match.py").read_text(encoding="utf-8")
    _dsrc_r = (Path(__file__).resolve().parent.parent / "vidlore" / "clipstudio" /
               "discover.py").read_text(encoding="utf-8")  # (re-read; defensive)
    check("reaction videos excluded from dialogue-verify re-admit + match pool",
          "_REACTION_TITLE" in _dsrc_r and "_REACTION_TITLE.search(c.title)" in _dsrc_r
          and "_REACTION_TITLE" in _msrc_r)
    # THEN-AND-NOW: a present-day beat about a named actor fetches a RECENT real photo (the NOW)
    # instead of decades-old movie footage. Detect now-beats + resolve the actor (incl. countdown
    # context where the actor is named once then referenced).
    from vidlore.clipstudio.image_fallback import _NOW_BEAT_RX as _NOWRX, actor_in_beat as _aib
    _aN = _SA(topic="t", movie_title="Home Alone",
              actors=["Joe Pesci", "Daniel Stern", "Macaulay Culkin"],
              characters=[{"name": "Harry", "actor": "Joe Pesci"},
                          {"name": "Kevin McCallister", "actor": "Macaulay Culkin"}])
    _c2a = _aN.char_to_actor()
    check("now-beat detector flags present-day lines, not 'then' lines",
          bool(_NOWRX.search("But today Joe Pesci is in his eighties, mostly retired"))
          and bool(_NOWRX.search("Now Daniel Stern has reinvented himself as a sculptor"))
          and not _NOWRX.search("On screen he was all rage and slapstick")
          and not _NOWRX.search("Number three: Joe Pesci who played Harry"))
    check("now-beat resolves the actor (by actor or character name)",
          _aib("today Joe Pesci is in his eighties", _aN, _c2a) == "Joe Pesci"
          and _aib("Kevin McCallister is now a father", _aN, _c2a) == "Macaulay Culkin")
    _osrc3 = (Path(__file__).resolve().parent.parent / "vidlore" / "clipstudio" /
              "orchestrate.py").read_text(encoding="utf-8")
    _ifsrc3 = (Path(__file__).resolve().parent.parent / "vidlore" / "clipstudio" /
               "image_fallback.py").read_text(encoding="utf-8")
    # FINAL IMAGE POLICY: NO NOW-photos / recent-actor photos in the auto fallback path (disabled)
    check("NOW-photo / recent-actor-photo fallback NOT called in auto path",
          "fetch_recent_actor_photo" not in _osrc3 and "now-photo" not in _osrc3)
    osrc2 = osrc
    check("fallback prefers real source-frames; web is a later, gated pass",
          osrc2.index("PASS 1 — SOURCE-FRAME STILLS") < osrc2.index("PASS 2 — WEB-EXACT-SCENE")
          and "pick_pool_still" in osrc2 and "VIDLORE_CLIPSTUDIO_SOURCE_STILL_FRACTION" in osrc2)
    check("web image only for exact/character + tagged web-exact-scene (never generic filler / AI)",
          "_policy.allows_web_image(seg)" in osrc
          and '"source": "web-exact-scene"' in osrc
          and "fetch_scene_image" in osrc)
    from vidlore.clipstudio.image_fallback import _has_watermark_text, _WATERMARK_RX
    check("watermark/social-handle/cosplay caption rejected",
          bool(_WATERMARK_RX.search("photo by @UnBoxPHD"))
          and bool(_WATERMARK_RX.search("cosplay shoot"))
          and not _WATERMARK_RX.search("Tywin Lannister war tent"))
    check("collage seam checks BOTH axes (side-by-side + stacked)",
          "rowE" in ifsrc and "horizontal seam" in ifsrc)
    # non-live-action VIDEO footage (Playmobil/toy/claymation/AI-render) dropped
    from vidlore.clipstudio.discover import _NONSHOW_TITLE
    check("title gate rejects toy/AI recreations",
          bool(_NONSHOW_TITLE.search("Oberyn vs Mountain Playmobil stop motion"))
          and bool(_NONSHOW_TITLE.search("GoT death AI recreation"))
          and not _NONSHOW_TITLE.search("Oberyn Martell death scene 4K"))
    check("art prompts include toy/AI/3D-render", "Playmobil or Lego" in ifsrc
          and "AI-generated image" in ifsrc)
    _msrc = (Path(__file__).resolve().parent.parent / "vidlore" / "clipstudio" /
             "match.py").read_text(encoding="utf-8")
    check("match drops non-photographic source via keyframe CLIP vote",
          "_source_is_nonphotographic" in _msrc and "_photographic_ok" in _msrc)
    # AI-generated image leak: faces present but matching no real actor -> 'unknown' -> rejected
    check("face verdict distinguishes unknown (AI/wrong) from none (no face)",
          '"unknown" if big else "none"' in ifsrc
          and 'target_actors and face == "unknown"' in ifsrc)
    # VERIFIER non-show HARD RULE covers BOTH new leak classes seen on the Arthur-Dayne render:
    # photorealistic AI/deepfake VIDEO and different-production FAN-FILM live-action, while still
    # sparing genuine-but-degraded show footage (authenticity, not resolution).
    _vsrc = (Path(__file__).resolve().parent.parent / "vidlore" / "clipstudio" /
             "verify.py").read_text(encoding="utf-8")
    check("verifier rejects AI/deepfake VIDEO (not just still images)",
          "deepfake image OR VIDEO" in _vsrc and "morph" in _vsrc)
    check("verifier rejects a DIFFERENT PRODUCTION (fan film / re-enactment)",
          "DIFFERENT PRODUCTION" in _vsrc and "fan film" in _vsrc)
    check("verifier non-show rule spares genuine low-res/dark footage (authenticity not resolution)",
          "AUTHENTICITY, not resolution" in _vsrc)
    check("AI/unknown-face image penalized + rejected on character beat",
          '"unknown": -0.30' in ifsrc)
    check("image fallback dedups across beats (no reuse)",
          "seen_hashes" in ifsrc and "seen_hashes.add" in ifsrc
          and "seen_hashes=seen_hashes" in osrc)
    # word-by-word breakout captions (engine untouched, clipstudio ASS post-pass). The BK Style
    # line now comes from the caption-preset registry (breakout_style_line) — same design family.
    _cpsrc = (Path(__file__).resolve().parent.parent / "vidlore" / "clipstudio"
              / "caption_presets.py").read_text(encoding="utf-8")
    check("breakout caption ASS builder (word-by-word karaoke)",
          "_breakout_caption_ast" not in bsrc and "_breakout_caption_ass" in bsrc
          and "\\\\kf" in bsrc and "preset.breakout_style_line()" in bsrc
          and "Style: BK," in _cpsrc)
    check("breakout captions burned after assemble",
          "_burn_breakout_captions" in bsrc and "_breakout_caps" in bsrc)
    check("breakout caption start = original + prior-breakout shift",
          't_cursor + delta_here + shift_before' in bsrc)
    # COMPETITOR-VOICEOVER guard: a breakout must be the movie's own dialogue, never essay narration
    from vidlore.clipstudio.build import _is_narration, _ESSAYISH_RX
    check("breakout audio narration/CTA rejected (content guard)",
          _is_narration("if you enjoyed this breakdown hit that like button subscribe")
          and _is_narration("the philosophical core of the scene in psychology")
          and _is_narration("this is the most devastating line in the scene"))
    check("real movie dialogue kept (not narration)",
          not _is_narration("I only know she was the one thing I ever wanted")
          and not _is_narration("What was she like"))
    check("narration guard wired into breakout extraction",
          "_is_narration(_bk_text)" in bsrc and "BREAKOUT_VOICE_GUARD" in bsrc
          and "_dialogue_aware_dur(str(src_path), start, lo=_lo, hi=_hi)" in bsrc)
    check("essay/interview titles excluded from breakout sources",
          bool(_ESSAYISH_RX.search("The Toxic Psychology of Robert & Cersei"))
          and bool(_ESSAYISH_RX.search("Best Scenes: Robert and Cersei"))
          and bool(_ESSAYISH_RX.search("Lena Headey Featurette")))
    dsrc = (Path(__file__).resolve().parent.parent / "vidlore" / "clipstudio" /
            "discover.py").read_text(encoding="utf-8")
    from vidlore.clipstudio.discover import _REJECT_TITLE as _RT
    check("discovery drops talking-head/featurette/promo sources",
          bool(_RT.search("Lena Headey on Playing Cersei Featurette"))
          and bool(_RT.search("FilmIsNow Movie Extras"))
          and bool(_RT.search("GRRM vs Fan Plot Hole"))
          and not _RT.search("King Robert talks to Cersei - Game of Thrones 1x05 (HD)"))
    # BTS 'anatomy of a scene' / making-of featurettes (show film crew + lights) and joke re-edits
    # (fart/crack/YTP) aired junk on the Tyrion render — an 'Anatomy of a Scene' clip showed studio
    # equipment mid-conclusion. They must be barred from the footage pool by title.
    check("discovery drops BTS 'anatomy of a scene' / making-of / joke re-edits",
          bool(_RT.search("Game of Thrones Season 4: Anatomy of a Scene"))
          and bool(_RT.search("The Making of Game of Thrones"))
          and bool(_RT.search("Tyrion Kills Tywin (Fart Edition)"))
          and bool(_RT.search("Game of Thrones YTP"))
          and not _RT.search("Tyrion kills Tywin Lannister - Tywin's death"))
    # COMMENTARY VIDEO-ESSAY ('Why Omitting ... Ruins ... Character Arc') aired 37x on the Tyrion
    # render — it narrates OVER clips, so it is not clean scene footage. Gate it by title; keep the
    # raw scene clip.
    check("discovery drops 'why ... ruins character arc' commentary essays",
          bool(_RT.search("Game of Thrones: Why Omitting the Tysha Confession Ruins Tyrion's Character Arc"))
          and bool(_RT.search("Why Tyrion Killed Tywin Explained"))
          and bool(_RT.search("The Tyrion Character Arc, Explained"))
          and not _RT.search("Tyrion kills Tywin - Game of Thrones 4x10")
          and not _RT.search("Tyrion's Trial Scene | Game of Thrones S04E10"))
    # branding/social-links/CTA card removal (freeze-replace, even when source had no OCR index)
    from vidlore.clipstudio.build import _BRANDING_RX
    check("branding/CTA slate detected (ExploreWesteros / FilmIsNow / 'links in description')",
          bool(_BRANDING_RX.search("www.ExploreWesteros.com /FollowHouseStark @ExploreWesteros"))
          and bool(_BRANDING_RX.search("Welcome to the world of FILMISNOW movie extras"))
          and bool(_BRANDING_RX.search("All links are in the description, don't forget to like"))
          and not _BRANDING_RX.search("I only know she was the one thing I ever wanted"))
    check("branding clips freeze-replaced from previous clean clip",
          "_clip_branding_text" in bsrc and "_freeze_replace" in bsrc
          and "branding-card removal" in bsrc and "BRANDING_GATE" in bsrc)
    check("match drops talking-head sources already downloaded",
          "_REJECT_TITLE.search(src.title" in _msrc)
    # AI-upscaler watermark (Magnific etc.) treated as branding/render → removed
    from vidlore.clipstudio.build import _BRANDING_RX as _BR
    check("AI-upscaler watermark (magnific/topaz/gigapixel) rejected",
          bool(_BR.search("Magnific")) and bool(_BR.search("Topaz Gigapixel"))
          and not _BR.search("Robert and Cersei talk in the room"))
    # retention pacing — fewer, shorter holds; never hold a long scene
    check("retention pacing: hold fraction + max-length cap",
          "VIDLORE_CLIPSTUDIO_HOLD_FRACTION" in bsrc and "VIDLORE_CLIPSTUDIO_HOLD_MAX_SEC" in bsrc
          and "_too_long" in bsrc)
    # single-scene purity: essay/compilation multi-season B-roll dropped
    check("single-scene purity drops essay/compilation footage",
          "single-scene purity" in _msrc and "SINGLE_SCENE_PURITY" in _msrc)
    # discovery variety + era-anchoring (kills repetition + wrong-era footage)
    from vidlore.clipstudio.discover import _era_hint, build_queries
    from vidlore.clipstudio.analyze import ScriptAnalysis
    _a = ScriptAnalysis(topic="t", movie_title="Game of Thrones", episode_hint="S01E05",
                        locations=["King's Landing", "Westeros"],
                        events=["Death of Lyanna Stark", "Fall of the Mad King"],
                        characters=[{"name": "Cersei Lannister", "actor": "Lena Headey"}])
    check("era hint parsed from episode code", _era_hint(_a) == "season 1")
    # era filter must parse BARE season codes ("Game of Thrones S3", "GoT S4") — a Pycelle-S3
    # source slipped onto an S01E07 stag-scene video because "S3" (no episode/no 'season') wasn't
    # parsed. The bare-season pattern must NOT mis-read S01E07 as season 1 (first pattern handles it).
    import re as _re_es
    def _ts(t):
        t = (t or "").lower()
        m = (_re_es.search(r"s0*(\d{1,2})\s*[ex]\d", t) or _re_es.search(r"\b(\d{1,2})\s*x\s*\d{1,2}\b", t)
             or _re_es.search(r"season\s*0*(\d{1,2})\b", t) or _re_es.search(r"\bs0*(\d{1,2})\b(?![\dex])", t))
        return int(m.group(1)) if m else None
    _msrc_es = (Path(__file__).resolve().parent.parent / "vidlore" / "clipstudio" /
                "match.py").read_text(encoding="utf-8")
    check("era filter parses bare season codes (S3/S4) without breaking SxxEyy",
          r"\bs0*(\d{1,2})\b(?![\dex])" in _msrc_es
          and _ts("Tywin exposes Maester Pycelle: Game of Thrones S3") == 3
          and _ts("GoT S4") == 4 and _ts("Game of Thrones S01E07") == 1
          and _ts("Game of Thrones - Jaime & Tywin Conversation") is None)
    _qs = build_queries(_a)
    check("on-era character B-roll query generated",
          any("cersei" in q.lower() and "season 1" in q.lower() for q in _qs))
    check("location B-roll queries generated (safe establishing shots)",
          any("king's landing" in q.lower() for q in _qs))
    # event-name B-roll queries are DELIBERATELY NOT generated: a franchise event like "Fall of
    # the Mad King" / "Death of Lyanna" returns House of the Dragon (the prequel) footage, which
    # is the wrong production. The era/character queries cover variety without that cross-show risk.
    check("event-name B-roll queries suppressed (avoid wrong-show contamination)",
          not any("lyanna" in q.lower() for q in _qs)
          and not any("mad king" in q.lower() for q in _qs))
    dsrc2 = (Path(__file__).resolve().parent.parent / "vidlore" / "clipstudio" /
             "discover.py").read_text(encoding="utf-8")
    check("'Westeros' too-generic location excluded", "westeros" in dsrc2)
    # wrong-show / franchise-sibling guard (GoT video must never pull House of the Dragon clips)
    from vidlore.clipstudio.discover import _wrong_installment
    check("wrong-show guard rejects franchise sibling (HotD in a GoT video)",
          _wrong_installment("Game of Thrones", "House of the Dragon - Aegon II coronation") is True
          and _wrong_installment("Game of Thrones", "A Knight of the Seven Kingdoms") is True)
    check("wrong-show guard keeps real target footage + comparison videos",
          _wrong_installment("Game of Thrones", "Game of Thrones S1 Robert & Cersei") is False
          and _wrong_installment("Game of Thrones", "Game of Thrones vs House of the Dragon") is False)
    check("wrong-show guard is symmetric (GoT clips rejected for a HotD video)",
          _wrong_installment("House of the Dragon", "Game of Thrones S3 Red Wedding") is True
          and _wrong_installment("House of the Dragon", "House of the Dragon - Rhaenyra") is False)
    check("wrong-show gate wired at discover + match",
          "_wrong_installment" in dsrc2 and "wrong-show" in dsrc2 and "_wrong_installment" in _msrc)
    # HotD sibling titled by CHARACTER (not show name) must also be caught — the real miss that put
    # an Aemond clip in the cold-open of a Daenerys render (title had no "House of the Dragon").
    check("wrong-show guard catches HotD sibling titled by character (Aemond / High Valyrian)",
          _wrong_installment("Game of Thrones",
                             "Aemond Destroys King Aegon at the Small Council Meeting in High Valyrian") is True
          and _wrong_installment("Game of Thrones", "Rhaenyra claims the Iron Throne") is True)
    check("wrong-show guard does NOT false-positive on shared-backstory / High Valyrian GoT titles",
          _wrong_installment("Game of Thrones", "Aegon Targaryen revealed — Jon Snow's real name") is False
          and _wrong_installment("Game of Thrones", "Daenerys speaks High Valyrian to the Unsullied") is False)
    # breakout miner must apply the same cross-show gate (it reads proj.sources directly, not the
    # match-filtered pool) — otherwise an HotD clip that never enters the pool can still open the video.
    _bsrc_ws = (Path(__file__).resolve().parents[1] / "vidlore" / "clipstudio" /
                "build.py").read_text(encoding="utf-8")
    check("breakout selection applies the wrong-show gate on proj.sources",
          "_wrong_installment" in _bsrc_ws and "breakout wrong-show gate" in _bsrc_ws)
    # hook (dynamic open) + dynamic music arc
    check("hook: opening beats forced dynamic (no slow hold) + stronger push",
          "_hold -= set(range(max(0, _hook_n))" in bsrc and "pos < _hook_n" in bsrc)
    check("retention music arc (hook→ease→build→climax swell→outro)",
          "MUSIC_ARC" in bsrc and "swell into the climax" in bsrc
          and "(_t * 0.82, 5)" in bsrc          # climax point at intensity 5
          and '"start": p[0]' in bsrc)          # compose_score SEGMENT contract (the {t} points
                                                # never matched it — KeyError on every render)
    # Option B: analyze prompt now asks for ALL named characters
    asrc = (Path(__file__).resolve().parent.parent / "vidlore" / "clipstudio" /
            "analyze.py").read_text(encoding="utf-8")
    check("analyze prompt requests full cast roster for face recognition",
          "OTHER major recurring characters" in asrc
          and "WRONG character is never shown" in asrc)


# ===========================================================================
# 19) AI narration voice (engine's neural TTS wired into clipstudio + dashboard)
# ===========================================================================

def test_ai_voice():
    print("[19] AI narration voice")
    from vidlore.clipstudio.config import ClipConfig
    cfg = ClipConfig()
    check("config has voice provider + preset (default neural)",
          getattr(cfg, "voice_provider", "") == "kokoro"
          and getattr(cfg, "voice_preset", "") == "deep_male_documentary")
    bsrc = (Path(__file__).resolve().parent.parent / "vidlore" / "clipstudio" /
            "build.py").read_text(encoding="utf-8")
    check("build routes neural backends via narrate_premium",
          "narrate_premium(script" in bsrc and 'in ("chatterbox", "kokoro")' in bsrc)
    check("build degrades AI→edge→silent (resilient)",
          "edge-tts fallback" in bsrc and "silent narration" in bsrc)
    check("build supports ElevenLabs cloud voice when key set",
          '"elevenlabs" if (_prov == "elevenlabs" and eng.elevenlabs_api_key)' in bsrc)
    osrc = (Path(__file__).resolve().parent.parent / "vidlore" / "clipstudio" /
            "orchestrate.py").read_text(encoding="utf-8")
    check("produce_auto threads voice_provider/preset to build",
          "voice_provider=voice_provider" in osrc and "voice_preset=voice_preset" in osrc)
    wsrc = (Path(__file__).resolve().parent.parent / "vidlore" / "clipstudio" /
            "web.py").read_text(encoding="utf-8")
    check("dashboard exposes AI voice + character selectors",
          'name=voice_provider' in wsrc and 'name=voice_preset' in wsrc
          and "voice_provider=voice_provider" in wsrc)
    csrc = (Path(__file__).resolve().parent.parent / "vidlore" / "clipstudio" /
            "cli.py").read_text(encoding="utf-8")
    check("CLI exposes --voice-provider / --voice-preset",
          "--voice-provider" in csrc and "--voice-preset" in csrc
          and "voice_provider=args.voice_provider" in csrc)
    # the engine's neural stack is actually reachable (no vidlore edits needed)
    from vidlore.tts import narrate_premium  # noqa: F401
    from vidlore.voice_presets import PRESETS
    check("engine neural presets available", cfg.voice_preset in PRESETS)


def test_cpu_parallelism():
    print("[cpu] worker counts auto-scale to the machine + turbo switch")
    import os as _os
    from vidlore.clipstudio.config import ClipConfig, _NCPU
    from vidlore.clipstudio import cut as C
    from vidlore.clipstudio import models as M

    keys = ("VIDLORE_CLIPSTUDIO_MAX_CPU", "VIDLORE_CLIPSTUDIO_CUT_WORKERS",
            "VIDLORE_CLIPSTUDIO_WHISPER_THREADS", "VIDLORE_CLIPSTUDIO_CONCURRENCY")
    saved = {k: _os.environ.get(k) for k in keys}
    try:
        for k in keys:
            _os.environ.pop(k, None)
        # --- defaults: scale with cores, leave headroom, never the old fixed 4 on a big box ---
        cfg = ClipConfig()
        check("cut_workers default scales with cores",
              cfg.cut_workers == min(12, max(2, _NCPU - 1)))
        check("whisper_cpu_threads default ~half the cores",
              cfg.whisper_cpu_threads == max(2, _NCPU // 2))
        check("download_concurrency default modest",
              cfg.download_concurrency == min(4, max(2, _NCPU // 4)))

        # --- explicit env override always wins over auto/turbo ---
        _os.environ["VIDLORE_CLIPSTUDIO_CUT_WORKERS"] = "3"
        check("explicit cut_workers env overrides auto", ClipConfig().cut_workers == 3)
        _os.environ.pop("VIDLORE_CLIPSTUDIO_CUT_WORKERS", None)

        # --- turbo: VIDLORE_CLIPSTUDIO_MAX_CPU saturates every core ---
        _os.environ["VIDLORE_CLIPSTUDIO_MAX_CPU"] = "1"
        turbo = ClipConfig()
        check("turbo cut_workers == all cores", turbo.cut_workers == _NCPU)
        check("turbo whisper threads == all cores", turbo.whisper_cpu_threads == _NCPU)
        check("turbo download bumped (capped)",
              turbo.download_concurrency == min(6, max(2, _NCPU // 2)))
        check("turbo never below 1", turbo.cut_workers >= 1)
        _os.environ.pop("VIDLORE_CLIPSTUDIO_MAX_CPU", None)

        # --- cut_all resolves workers from cfg.cut_workers when none passed (no ffmpeg: empty proj) ---
        tmp = tempfile.mkdtemp(prefix="cscpu_")
        proj = M.ClipProject(name="t", root=tmp)
        proj.selections = []            # nothing to cut → exercises worker resolution, runs no ffmpeg
        check("cut_all(workers=None) uses cfg without crashing",
              C.cut_all(proj, ClipConfig(), workers=None) == 0)
    finally:
        for k, v in saved.items():
            if v is None:
                _os.environ.pop(k, None)
            else:
                _os.environ[k] = v

    # whisper model is actually constructed with cpu_threads (source-pinned wiring)
    isrc = (Path(__file__).resolve().parent.parent / "vidlore" / "clipstudio" /
            "index.py").read_text(encoding="utf-8")
    check("whisper model wired with cpu_threads",
          "cpu_threads=" in isrc and "whisper_cpu_threads" in isrc)


def test_deepseek_primary_brain():
    print("[brain] DeepSeek primary (v4-pro → v4-flash → Gemini → Claude LAST) + portal selector")
    import os as _os
    import vidlore.clipstudio.llm as L

    keys = ("VIDLORE_CLIPSTUDIO_LLM_PROVIDER", "VIDLORE_CLIPSTUDIO_DEEPSEEK_MODEL",
            "VIDLORE_CLIPSTUDIO_DEEPSEEK_FAST_MODEL", "DEEPSEEK_API_KEY",
            "ANTHROPIC_API_KEY", "VIDLORE_CLIPSTUDIO_LLM_RETRIES")
    saved = {k: _os.environ.get(k) for k in keys}
    try:
        for k in keys:
            _os.environ.pop(k, None)
        # defaults: DeepSeek IS the brain; v4-pro primary, v4-flash the fast fallback
        check("default provider is deepseek", L._provider() == "deepseek")
        check("primary model = deepseek-v4-pro", L._deepseek_model() == "deepseek-v4-pro")
        check("fast fallback = deepseek-v4-flash", L.fast_deepseek_model() == "deepseek-v4-flash")

        # structural: in complete()'s deepseek chain, Claude is the LAST attempt (after flash + gemini)
        lsrc = (Path(__file__).resolve().parent.parent / "vidlore" / "clipstudio" /
                "llm.py").read_text(encoding="utf-8")
        block = lsrc.split('if prov in ("deepseek", "ds"):', 1)[1].split("elif", 1)[0]
        i_flash, i_gem, i_claude = (block.find("fast_deepseek_model()"),
                                    block.find("_try_gemini"), block.find("_try_claude"))
        check("deepseek chain includes a flash fallback", i_flash > 0)
        check("deepseek chain: Claude is the LAST entry",
              i_claude > i_flash and i_claude > i_gem)

        # behavioral: pro returns empty → deepseek-v4-flash serves it (Claude never reached)
        _os.environ["DEEPSEEK_API_KEY"] = "sk-test"
        _os.environ["VIDLORE_CLIPSTUDIO_LLM_PROVIDER"] = "deepseek"
        _os.environ["VIDLORE_CLIPSTUDIO_LLM_RETRIES"] = "1"      # no slow retries on the empty pro
        calls = []
        orig = L._deepseek_complete

        def fake_ds(system, messages, max_tokens, model):
            calls.append(model or "")
            return "FLASH-OK" if "flash" in (model or "") else ""   # pro empty → fall to flash
        L._deepseek_complete = fake_ds
        try:
            out = L.complete(system="s", messages=[{"role": "user", "content": "hi"}], max_tokens=50)
        finally:
            L._deepseek_complete = orig
        check("pro-empty falls to deepseek-v4-flash (not Claude)", out == "FLASH-OK")
        check("flash model id actually requested", any("flash" in c for c in calls))
    finally:
        for k, v in saved.items():
            if v is None:
                _os.environ.pop(k, None)
            else:
                _os.environ[k] = v

    # portal: brain selector wired, defaults to DeepSeek V4 Pro, validates + routes provider/model
    wsrc = (Path(__file__).resolve().parent.parent / "vidlore" / "clipstudio" /
            "web.py").read_text(encoding="utf-8")
    check("portal form has a brain selector", "name=brain" in wsrc and "__BRAIN_OPTS__" in wsrc)
    check("portal offers v4-pro (default) + v4-flash",
          "deepseek-v4-pro" in wsrc and "deepseek-v4-flash" in wsrc and "(default)" in wsrc)
    check("portal validates brain + routes provider/model to the job",
          "provider=provider" in wsrc and "ds_model=ds_model" in wsrc
          and "VIDLORE_CLIPSTUDIO_LLM_PROVIDER" in wsrc)
    # renders are SERIALIZED through a lock so concurrent jobs can't clobber each other's brain env,
    # and the env is set FULLY (both keys) so no stale deepseek model bleeds into a non-deepseek job
    check("portal serializes renders (no cross-job brain bleed)",
          "_RENDER_LOCK" in wsrc and "with _RENDER_LOCK" in wsrc)
    check("portal sets BOTH brain env keys (symmetric, no stale model)",
          'os.environ["VIDLORE_CLIPSTUDIO_DEEPSEEK_MODEL"] = ds_model or "deepseek-v4-pro"' in wsrc)
    # the job page must not ship the literal __BRAIN__ placeholder to the browser
    check("job page strips the __BRAIN__ placeholder",
          'replace("__BRAIN__", "")' in wsrc)
    # Turbo (all-CPU-cores) toggle is exposed in the portal + applied per job (race-safe in the lock)
    check("portal exposes a Turbo selector",
          "name=turbo" in wsrc and "__TURBO_OPTS__" in wsrc and "_turbo_options" in wsrc)
    check("portal applies Turbo via VIDLORE_CLIPSTUDIO_MAX_CPU per job",
          'os.environ["VIDLORE_CLIPSTUDIO_MAX_CPU"] = "1" if turbo else "0"' in wsrc
          and "turbo=turbo" in wsrc)


def test_source_budget_scales_with_script():
    print("[relevance] footage budget scales with script length (long essays get more sources)")
    from vidlore.clipstudio.orchestrate import _scaled_source_budget as B
    # short clip: unchanged (respect user's small budget)
    check("short script keeps the requested budget", B(8, 12, "essay") == 8)
    # long 181-beat essay: scales up well past 8 (~1 src / 4 beats since the 2026-07-26 raise —
    # the old 1-per-5 / cap-48 budget starved long scripts, which the frame-by-frame audit of job
    # 69d80e9dd4 measured as 5% exact-scene hits and 46% near-duplicate scenes)
    check("181-beat essay scales up", B(8, 181, "essay") == min(72, (181 + 3) // 4))
    check("181-beat essay budget is much larger than 8", B(8, 181, "essay") >= 30)
    # never below the user's explicit ask
    check("explicit high budget is never lowered", B(40, 60, "essay") == 40)
    # SHORT single-scene deep-dive is NOT scaled (wants one scene's footage, not breadth)
    check("short single_scene is not scaled", B(8, 40, "single_scene") == 8)
    # LONG single-scene deep-dive gets a >=20 source floor (avoids airing one clip 57x — the Cersei
    # 20-min render used its top source 57x because single_scene returned base=10)
    check("long single_scene gets a >=20 floor", B(10, 211, "single_scene") >= 20)
    check("long single_scene scales gently with length",
          B(10, 211, "single_scene") == min(56, 24 + (211 - 100) // 8))
    # long-form floor applies to multi-scene too, but scaling can exceed it
    check("100-beat video reaches the 20 floor", B(6, 100, "essay") >= 20)
    # capped so it can't explode (cap raised 48 -> 72; SOURCE_BUDGET_CAP env is the hard ceiling)
    check("budget is capped at 72", B(8, 9999, "essay") == 72)
    # the operator can trade render time for footage breadth
    import os as _os_b
    _plain = B(8, 272, "single_scene")
    _os_b.environ["VIDLORE_CLIPSTUDIO_SOURCE_BUDGET_MULT"] = "1.5"
    try:
        check("SOURCE_BUDGET_MULT scales the budget up", B(8, 272, "single_scene") > _plain)
        check("MULT never drops below the operator's explicit ask", B(40, 12, "essay") >= 40)
    finally:
        _os_b.environ.pop("VIDLORE_CLIPSTUDIO_SOURCE_BUDGET_MULT", None)
    check("budget returns to the default once MULT is unset", B(8, 272, "single_scene") == _plain)
    _os_b.environ["VIDLORE_CLIPSTUDIO_SOURCE_BUDGET_CAP"] = "20"
    try:
        check("SOURCE_BUDGET_CAP is the hard ceiling", B(8, 9999, "essay") == 20)
    finally:
        _os_b.environ.pop("VIDLORE_CLIPSTUDIO_SOURCE_BUDGET_CAP", None)


def test_caption_sync_per_scene_tolerant():
    print("[caption] uploaded-voiceover word-sync is PER-SCENE tolerant (engine gate is all-or-nothing)")
    import types as _t
    import tempfile
    import vidlore.tts as _tts
    import vidlore.ffmpeg_tool as _ff
    from vidlore.clipstudio import build as B

    script = _t.SimpleNamespace(scenes=[
        _t.SimpleNamespace(index=0, narration="one two three"),
        _t.SimpleNamespace(index=1, narration="four five"),
        _t.SimpleNamespace(index=2, narration="six seven eight nine"),
    ])
    # UNEVEN aligned word times (proportional would be evenly spaced — so this proves real alignment
    # is used). Scene 1's word "five" balloons to 50s: the engine's gate would DISCARD the whole
    # alignment; ours must clamp just that scene and keep the rest.
    aligned = [(0.0, 0.3), (0.3, 1.4), (1.4, 1.5),          # scene 0 (uneven)
               (1.5, 2.0), (2.0, 50.0),                     # scene 1 — "five" balloons
               (7.0, 7.3), (7.3, 7.6), (7.6, 7.9), (7.9, 8.0)]  # scene 2
    flat = "one two three four five six seven eight nine".split()
    hyp = [(flat[k], aligned[k][0], aligned[k][1]) for k in range(len(flat))]
    total = 8.0
    save = (_tts._wav_duration, _tts._slice_scene, _ff.run, B._chunked_whisper_words)
    try:
        _tts._wav_duration = lambda p: total
        _tts._slice_scene = lambda *a, **k: None
        _ff.run = lambda *a, **k: None
        B._chunked_whisper_words = lambda audio, tot, **k: hyp     # locally-accurate chunked stream
        wd = Path(tempfile.mkdtemp(prefix="csync_"))
        nar = B._synced_narration_from_file(script, "/dev/null", wd, None)
        check("sync returns a narration despite a ballooned scene (engine would discard ALL)",
              nar is not None)
        if nar is not None:
            words = nar.all_words()
            check("all 9 words timed", len(words) == 9)
            check("uses REAL aligned timing, not proportional spread",
                  abs(words[1].end - 1.4) < 0.05)         # 'two' ends at the aligned 1.4, not ~1.0
            check("all 3 scenes survive the clamp", len(nar.scenes) == 3)
            check("ballooned scene clamped (no multi-second freeze)",
                  all(s.duration <= total + 0.5 for s in nar.scenes))
    finally:
        _tts._wav_duration, _tts._slice_scene, _ff.run, B._chunked_whisper_words = save

    bsrc = (Path(__file__).resolve().parent.parent / "vidlore" / "clipstudio" /
            "build.py").read_text(encoding="utf-8")
    check("build tries word-sync BEFORE the engine narrate_from_file fallback",
          bsrc.index("_synced_narration_from_file(script") <
          bsrc.index("narrate_from_file(script, str(Path(voiceover)"))
    # Real-audio breakouts pause narration. They STAY ON (the feature is wanted, incl. on uploaded
    # voiceovers); caption sync is held by (a) _group cutting a cue at the breakout's silent gap and
    # (b) suppressing the main caption over the breakout window — NOT by disabling breakouts.
    check("breakouts stay ON by default (feature kept; env still overrides)",
          'os.environ.get("VIDLORE_CLIPSTUDIO_BREAKOUTS", "1")' in bsrc
          and '_bk_default' not in bsrc)
    check("breakout windows feed the caption-suppress list (main caption stays voice-locked)",
          '_breakout_caps' in bsrc and 'suppress_wins.append' in bsrc
          and 'real-audio breakout window' in bsrc)
    # FUNCTIONAL — the root-cause fix: a caption cue must NOT hold across a breakout-length silence.
    from vidlore.captions import _group as _grp, _CUE_GAP_BREAK
    from vidlore.tts import WordTiming as _WT
    check("caption gap-break threshold is breakout-scaled (>breath, <breakout)",
          0.8 <= _CUE_GAP_BREAK <= 1.6)
    _bk_words = [
        _WT("She", 0.0, 0.3), _WT("didn't", 0.3, 0.6), _WT("turn", 0.6, 0.9),
        _WT("evil.", 0.9, 1.2),                       # last word BEFORE the breakout
        _WT("This", 9.2, 9.5), _WT("is", 9.5, 9.7),   # 8s breakout gap, then narration resumes
        _WT("how", 9.7, 9.9),
    ]
    _cues = _grp(_bk_words)
    check("no cue spans the breakout gap (kills the 13s freeze)",
          all((c[-1].end - c[0].start) < 5.0 for c in _cues))
    check("post-breakout word 'This' opens a fresh cue (not merged with 'evil.')",
          any(c[0].word == "This" for c in _cues)
          and not any({"evil.", "This"} <= {w.word for w in c} for c in _cues))
    # a NORMAL breath gap (<=0.8s) must NOT cut — cinematic holds preserved (legacy behaviour)
    _smooth = [_WT("a", 0.0, 0.3), _WT("b", 0.5, 0.8), _WT("c", 1.0, 1.3)]
    check("normal speech rhythm still groups into one cue (no over-cutting)",
          len(_grp(_smooth)) == 1)
    # CAPTION-FROM-TRANSCRIPTION: when the pasted script can't be aligned to the uploaded voiceover
    # (script != voiceover — wrong file / edited draft), captions must come from the voiceover's OWN
    # transcription (locked to the voice), NOT the engine's drifting proportional split. Seen on a
    # 22-min Hound render whose script ("chickens") and voiceover ("you want to be like me") were
    # different content → only 162/2853 align anchors → drift.
    import vidlore.tts as _ttsmod
    _save_ss = _ttsmod._slice_scene
    _ttsmod._slice_scene = lambda *a, **k: None          # no ffmpeg in the unit test
    try:
        _hyp = [("hello", 0.0, 0.4), ("world", 0.5, 0.9), ("foo", 1.0, 1.4),
                ("bar", 1.5, 1.9), ("baz", 2.0, 2.4), ("qux", 2.5, 2.9)]
        _nar = B._narration_from_hyp(_hyp, 3, 3.0, Path("/dev/null"), Path(tempfile.mkdtemp()))
    finally:
        _ttsmod._slice_scene = _save_ss
    check("transcription-narration: caption words ARE the voiceover transcription at real times",
          _nar is not None
          and [w.word for w in _nar.all_words()] == ["hello", "world", "foo", "bar", "baz", "qux"]
          and abs(_nar.all_words()[0].start) < 0.01 and abs(_nar.all_words()[-1].end - 2.9) < 0.06)
    check("transcription-narration keeps the scene count (scene->footage mapping intact)",
          _nar is not None and len(_nar.scenes) == 3)
    check("aligner captions from transcription (not engine drift) when script != voiceover",
          "from the voiceover transcription" in bsrc
          and "_narration_from_hyp(hyp, n, total" in bsrc)


def test_final_image_policy():
    print("[image] final policy: source-frames preferred · AI banned · web only validated-exact")
    import types, tempfile
    from vidlore.clipstudio import orchestrate as O
    from vidlore.clipstudio import image_fallback as IF
    from vidlore.clipstudio import web_images as WI
    from vidlore.clipstudio import index as IDX
    from vidlore.clipstudio import models as M
    from vidlore.clipstudio import policy as PL

    # 1) AI generator domains/source pages always rejected
    check("AI-generator sources rejected (real host kept)",
          WI.is_ai_generated_source("shedevrum.ai") and not WI.is_ai_generated_source("hbo.com"))

    # 2) behavioral _fill_image_fallbacks — source-frame first, web only for exact, exact flagged
    tmp = tempfile.mkdtemp(prefix="csimg_")
    proj = M.ClipProject(name="t", root=tmp)
    proj.sources = [M.SourceVideo(id="s1", url="", title="t", local_path="x", duration=60.0, permission="owner")]
    proj.selections = []
    segs = [M.ScriptSegment(index=0, text="and the realm held its breath", visual_policy=PL.FILLER),
            M.ScriptSegment(index=1, text="nothing was ever the same", visual_policy=PL.ABSTRACT),
            M.ScriptSegment(index=2, text="Tyrion shoots Tywin with a crossbow",
                            visual_policy=PL.EXACT, is_specific_claim=True)]
    analysis = types.SimpleNamespace(char_to_actor=lambda: {})
    web_calls = []
    fake_still = ("/tmp/kf_x.jpg", "s1", 5, 0.7, "ph0")
    import os as _os
    _sf = _os.environ.get("VIDLORE_CLIPSTUDIO_SOURCE_STILL_FRACTION")
    _os.environ["VIDLORE_CLIPSTUDIO_SOURCE_STILL_FRACTION"] = "1.0"   # allow a still on every test beat
    save = (IF.pick_source_still, IF.pick_pool_still, IF.fetch_scene_image, IDX.load_shots)
    try:
        IF.pick_source_still = lambda *a, **k: None
        IF.pick_pool_still = lambda seg, *a, **k: (fake_still if PL.policy_of(seg) in (PL.FILLER, PL.ABSTRACT) else None)

        def _web(seg, *a, **k):
            web_calls.append(PL.policy_of(seg))
            return None                                    # nothing verified out → exact stays flagged
        IF.fetch_scene_image = _web
        IDX.load_shots = lambda p, sid: []
        O._fill_image_fallbacks(proj, segs, analysis, None, {}, lambda m: None)
    finally:
        IF.pick_source_still, IF.pick_pool_still, IF.fetch_scene_image, IDX.load_shots = save
        if _sf is None:
            _os.environ.pop("VIDLORE_CLIPSTUDIO_SOURCE_STILL_FRACTION", None)
        else:
            _os.environ["VIDLORE_CLIPSTUDIO_SOURCE_STILL_FRACTION"] = _sf
    by = {s.segment_index: s for s in proj.selections}
    check("generic_filler / abstract no-clip beats filled from a SOURCE-FRAME (not web)",
          by.get(0) and by[0].image_meta.get("source") == "source-frame"
          and by.get(1) and by[1].image_meta.get("source") == "source-frame")
    check("web image NEVER attempted for filler/abstract (no random web decoration)",
          PL.FILLER not in web_calls and PL.ABSTRACT not in web_calls)
    check("web image attempted ONLY for the exact/character beat", web_calls == [PL.EXACT])
    check("exact beat with no footage + no validated web still → exact_scene_missing (review)",
          by.get(2) and M.FLAG_EXACT_MISSING in by[2].flag_reasons)

    # 3) strict web validation gates present (watermark/text/collage/poster/fan-art/AI rejection)
    ifsrc = (Path(__file__).resolve().parent.parent / "vidlore" / "clipstudio" /
             "image_fallback.py").read_text(encoding="utf-8")
    check("web validation: collage/seam + big-text/watermark + photographic-only + AI-source reject",
          "_has_center_seam" in ifsrc and "has_big_text" in ifsrc and "_has_watermark_text" in ifsrc
          and "_photographic_ok" in ifsrc and "is_ai_generated_source" in ifsrc)
    # 4) ledger records the image source clearly
    lsrc = (Path(__file__).resolve().parent.parent / "vidlore" / "clipstudio" /
            "ledger.py").read_text(encoding="utf-8")
    check("ledger records image_source (source-frame / web-exact-scene)", '"image_source"' in lsrc)


def test_image_policy_edge_gaps():
    print("[image] edge gaps: web upgrades recovery · ledger coverage · franchise/BTS web guard")
    import types, tempfile
    import os as _os
    from vidlore.clipstudio import orchestrate as O
    from vidlore.clipstudio import image_fallback as IF
    from vidlore.clipstudio import index as IDX
    from vidlore.clipstudio import ledger as LG
    from vidlore.clipstudio import models as M
    from vidlore.clipstudio import policy as PL
    from vidlore.clipstudio.config import ClipConfig

    # --- req 4: web candidate guard (same movie/series + no BTS/red-carpet/franchise) ---
    an = types.SimpleNamespace(movie_title="Game of Thrones")

    def C(title="", site="x.com"):
        return {"title": title, "source_site": site, "source_page": "http://" + site + "/p",
                "image_url": "http://x/y.jpg"}
    check("web guard accepts a real in-show still naming the title",
          IF._web_candidate_ok(C("Game of Thrones Tyrion trial scene"), an))
    check("web guard rejects wrong franchise / spinoff (House of the Dragon)",
          not IF._web_candidate_ok(C("House of the Dragon Rhaenyra scene"), an))
    check("web guard rejects red-carpet / interview / BTS / fan / cosplay pages",
          not IF._web_candidate_ok(C("Peter Dinklage red carpet premiere"), an)
          and not IF._web_candidate_ok(C("Game of Thrones cast interview"), an)
          and not IF._web_candidate_ok(C("GoT behind the scenes making of"), an)
          and not IF._web_candidate_ok(C("Tyrion cosplay fan art"), an))
    check("web guard rejects an image that never references the target title",
          not IF._web_candidate_ok(C("random medieval castle wallpaper", "pinterest.com"), an))
    ifsrc = (Path(__file__).resolve().parent.parent / "vidlore" / "clipstudio" /
             "image_fallback.py").read_text(encoding="utf-8")
    check("character web still needs a higher floor + confirmed actor (no generic portrait)",
          "IMAGE_MIN_SCORE_CHAR" in ifsrc and 'best["face"] != "match"' in ifsrc)

    # --- req 2: ledger treats image coverage correctly (not no_candidate) ---
    cfg = ClipConfig()

    def mk(idx, src):
        s = M.ClipSelection(segment_index=idx, source_id="", shot_index=-1,
                            in_point=0, out_point=0, confidence=0.0)
        s.image_path = "/tmp/x.jpg"
        s.image_meta = {"source": src}
        return s
    segE = M.ScriptSegment(index=0, text="Tyrion shoots Tywin", visual_policy=PL.EXACT, is_specific_claim=True)
    segF = M.ScriptSegment(index=1, text="the realm watched", visual_policy=PL.FILLER)
    check("ledger: source-frame coverage is NOT no_candidate",
          M.FLAG_NO_CANDIDATE not in LG.evaluate_flags(mk(1, "source-frame"), segF, None, cfg))
    check("ledger: web-exact-scene coverage is NOT no_candidate",
          M.FLAG_NO_CANDIDATE not in LG.evaluate_flags(mk(0, "web-exact-scene"), segE, None, cfg))
    _recov = LG.evaluate_flags(mk(0, "source-frame-recovery"), segE, None, cfg)
    check("ledger: exact source-frame-recovery is review-flagged but NOT no_candidate",
          M.FLAG_EXACT_MISSING in _recov and M.FLAG_NO_CANDIDATE not in _recov)
    _empty = LG.evaluate_flags(
        M.ClipSelection(segment_index=2, source_id="", shot_index=-1, in_point=0, out_point=0, confidence=0.0),
        M.ScriptSegment(index=2, text="x"), None, cfg)
    check("ledger: a truly empty beat IS no_candidate", M.FLAG_NO_CANDIDATE in _empty)

    # --- behavioral _fill_image_fallbacks harness ---
    def run_fallback(segs, sels, *, web_ok, pool_ok, src_still_ok, frac="1.0"):
        proj = M.ClipProject(name="t", root=tempfile.mkdtemp(prefix="cseg_"))
        proj.sources = [M.SourceVideo(id="s1", url="", title="t", local_path="x", duration=60.0, permission="owner")]
        proj.selections = sels
        an2 = types.SimpleNamespace(char_to_actor=lambda: {}, movie_title="Game of Thrones")
        sf = _os.environ.get("VIDLORE_CLIPSTUDIO_SOURCE_STILL_FRACTION")
        _os.environ["VIDLORE_CLIPSTUDIO_SOURCE_STILL_FRACTION"] = frac
        sv = (IF.pick_source_still, IF.pick_pool_still, IF.fetch_scene_image, IDX.load_shots)
        try:
            IF.pick_source_still = lambda *a, **k: (("/tmp/sf.jpg", "s1", 7, 0.7, "pha") if src_still_ok else None)
            IF.pick_pool_still = lambda *a, **k: (("/tmp/pp.jpg", "s1", 8, 0.6, "phb") if pool_ok else None)
            IF.fetch_scene_image = ((lambda *a, **k: {"path": "/tmp/web.jpg", "score": 0.9, "clip": 0.7,
                                                      "face": "match", "query": "q"}) if web_ok
                                    else (lambda *a, **k: None))
            IDX.load_shots = lambda p, sid: []
            O._fill_image_fallbacks(proj, segs, an2, None, {}, lambda m: None)
        finally:
            IF.pick_source_still, IF.pick_pool_still, IF.fetch_scene_image, IDX.load_shots = sv
            if sf is None:
                _os.environ.pop("VIDLORE_CLIPSTUDIO_SOURCE_STILL_FRACTION", None)
            else:
                _os.environ["VIDLORE_CLIPSTUDIO_SOURCE_STILL_FRACTION"] = sf
        return {s.segment_index: s for s in proj.selections}

    def _exact_beat():
        return M.ScriptSegment(index=0, text="Tyrion shoots Tywin with a crossbow", visual_policy=PL.EXACT,
                               is_specific_claim=True, scene_query="Tyrion shoots Tywin crossbow")

    def _weak_exact_sel():
        s = M.ClipSelection(segment_index=0, source_id="s1", shot_index=2, in_point=0, out_point=3, confidence=0.2)
        s.flag_reasons = ["verifier_failed"]
        return s

    # req 1: exact recovery + valid web still → web replaces recovery
    by = run_fallback([_exact_beat()], [_weak_exact_sel()], web_ok=True, pool_ok=False, src_still_ok=True)
    check("exact recovery is UPGRADED to web-exact-scene when a valid web still exists",
          by[0].image_meta.get("source") == "web-exact-scene")
    # req 1: web fails → keep recovery + stay exact_scene_missing
    by2 = run_fallback([_exact_beat()], [_weak_exact_sel()], web_ok=False, pool_ok=False, src_still_ok=True)
    check("web fails → source-frame-recovery kept + exact_scene_missing stays flagged",
          by2[0].image_meta.get("source") == "source-frame-recovery"
          and M.FLAG_EXACT_MISSING in by2[0].flag_reasons)
    # req 5: no-clip generic/abstract fill even when optional cap is exhausted (frac=0 → cap=1)
    by3 = run_fallback([M.ScriptSegment(index=0, text="the realm held its breath", visual_policy=PL.FILLER),
                        M.ScriptSegment(index=1, text="nothing was the same", visual_policy=PL.ABSTRACT),
                        M.ScriptSegment(index=2, text="and time passed", visual_policy=PL.FILLER)],
                       [], web_ok=False, pool_ok=True, src_still_ok=False, frac="0")
    check("no-clip filler/abstract beats ALL fill from source-frames despite the optional cap",
          sum(1 for i in (0, 1, 2)
              if by3.get(i) and by3[i].image_meta.get("source") == "source-frame") == 3)


def test_unified_visual_policy():
    print("[policy] unified per-beat visual policy + downstream wiring")
    from vidlore.clipstudio import policy as PL
    from vidlore.clipstudio.models import ScriptSegment as Seg

    # --- classifier: the five treatments ---
    exact = Seg(index=0, text="Tyrion shoots Tywin with a crossbow", required_kind="event", is_specific_claim=True)
    quote = Seg(index=1, text='He says the line', quote="I demand a trial by combat")
    char = Seg(index=2, text="Tyrion was always the clever one", required_kind="character",
               required_entity="Tyrion Lannister")
    filler = Seg(index=3, text="and the whole realm held its breath")
    abstr = Seg(index=4, text="but in the end none of it really mattered")
    check("exact_scene from a specific event", PL.policy_of(exact) == PL.EXACT)
    check("exact_scene from an iconic quote (+breakout)",
          PL.policy_of(quote) == PL.EXACT and PL.is_breakout_candidate(quote))
    check("character_specific from a named person", PL.policy_of(char) == PL.CHARACTER)
    check("generic_filler from a generic line", PL.policy_of(filler) == PL.FILLER)
    check("abstract_effect from a reflective line", PL.policy_of(abstr) == PL.ABSTRACT)

    # --- explicit LLM value wins; bad value falls back to heuristic ---
    char.visual_policy = "exact_scene"
    check("explicit LLM policy overrides heuristic", PL.policy_of(char) == PL.EXACT)
    check("invalid policy string falls back to heuristic", PL.normalize("garbage") == "")
    char.visual_policy = ""

    # --- per-stage helpers (the contracts every stage obeys) ---
    check("AI images banned for ALL policies",
          not any(PL.allows_ai_image(s) for s in [exact, char, filler, abstr]))
    char_q = Seg(index=9, text="Tyrion in the throne room", required_kind="character",
                 required_entity="Tyrion Lannister", scene_query="Tyrion stands trial in the throne room")
    check("web image: exact always; character only with a strong scene query; NEVER filler/abstract",
          PL.allows_web_image(exact) and PL.allows_web_image(char_q)
          and not PL.allows_web_image(char)            # bare character portrait → no web
          and not PL.allows_web_image(filler) and not PL.allows_web_image(abstr))
    check("abstract: prefers image/effect", PL.prefers_image(abstr) and not PL.prefers_image(exact))
    check("discovery query only for exact/character (not filler/abstract)",
          PL.wants_discovery_query(exact) and PL.wants_discovery_query(char)
          and not PL.wants_discovery_query(filler) and not PL.wants_discovery_query(abstr))
    check("discovery tiers high/medium/low/none",
          (PL.discovery_tier(exact), PL.discovery_tier(char), PL.discovery_tier(filler),
           PL.discovery_tier(abstr)) == ("high", "medium", "low", "none"))
    check("variety maximized for filler/character, not exact",
          PL.maximize_variety(filler) and PL.maximize_variety(char) and not PL.maximize_variety(exact))
    check("verify strict only for exact", PL.verify_strict(exact)
          and not PL.verify_strict(filler) and not PL.verify_strict(abstr))

    # --- finalize persists policy + breakout on every beat ---
    tally = PL.finalize_beats([exact, quote, char, filler, abstr])
    check("finalize persists visual_policy on all beats",
          all(s.visual_policy in PL.POLICIES for s in [exact, quote, char, filler, abstr]))
    check("finalize tallies the classes", sum(tally.values()) == 5)

    # --- downstream wiring is actually present in each stage ---
    def src(mod):
        return (Path(__file__).resolve().parent.parent / "vidlore" / "clipstudio" / mod).read_text(encoding="utf-8")
    check("orchestrate classifies beats at analyze stage",
          "_policy.finalize_beats(segs)" in src("orchestrate.py"))
    check("orchestrate image-fallback: source-frame preferred, web gated, exact flagged",
          "pick_pool_still" in src("orchestrate.py")
          and '"source": "web-exact-scene"' in src("orchestrate.py")
          and "FLAG_EXACT_MISSING" in src("orchestrate.py"))
    check("match.py applies policy variety multiplier",
          "_policy.maximize_variety(seg)" in src("match.py") and "_variety" in src("match.py"))
    check("discover.py gates per-beat queries by policy",
          "_policy.wants_discovery_query(s)" in src("discover.py"))
    check("verify.py: exact strict + exact_scene_missing flag",
          "_policy.verify_strict(seg)" in src("verify.py") and "FLAG_EXACT_MISSING" in src("verify.py"))
    check("ledger.py re-derives exact_scene_missing (survives apply_flags overwrite)",
          "_policy.is_exact(segment)" in src("ledger.py") and "FLAG_EXACT_MISSING" in src("ledger.py"))


def test_generic_beat_filler_leniency():
    print("[relevance] verifier: exact for SPECIFIC beats, relevant FILLER ok for GENERIC beats")
    from vidlore.clipstudio import verify as V
    from vidlore.clipstudio import models as M
    from vidlore.clipstudio.config import ClipConfig
    import vidlore.clipstudio.llm as L

    tmp = tempfile.mkdtemp(prefix="csgen_")
    proj = M.ClipProject(name="t", root=tmp)
    # beat 0 = GENERIC narration (non-exact, on-topic filler ok);
    # beat 1 = EXACT beat, REQUIRED SUBJECT confirmed on screen → exact→contextual downgrade;
    # beat 2 = EXACT beat, WRONG subject (correct_subject_visible False) → contradictory, NOT downgraded
    gen = M.ClipSelection(segment_index=0, source_id="srcA", shot_index=1,
                          in_point=1.0, out_point=4.0, confidence=0.8)
    ctx = M.ClipSelection(segment_index=1, source_id="srcA", shot_index=2,
                          in_point=5.0, out_point=8.0, confidence=0.8)
    wrong = M.ClipSelection(segment_index=2, source_id="srcA", shot_index=3,
                            in_point=9.0, out_point=12.0, confidence=0.8)
    proj.selections = [gen, ctx, wrong]
    segs = [M.ScriptSegment(index=0, text="and everything was about to change", is_specific_claim=False),
            M.ScriptSegment(index=1, text="Tywin at the small council table",
                            visual_policy="exact_scene", is_specific_claim=True),   # over-marked spec
            M.ScriptSegment(index=2, text="the wrong character entirely",
                            visual_policy="exact_scene", required_kind="character",
                            required_entity="Jon Snow", is_specific_claim=True)]
    fake_shot = types.SimpleNamespace(index=1, keyframe_path="kf.jpg", face_ids=[], identities=[])

    # META/commentary narration → matches_narration False even when the right subject IS on screen
    # (mirrors the real Gemini behaviour). Beat 2's footage shows the WRONG subject.
    def fake_vf(kf, narration, ent, kind, names, eng_cfg, model="", is_specific=True, **kwargs):
        subj = False if "wrong character" in narration else True
        return {"verdict": "replace", "matches_narration": False, "correct_subject_visible": subj,
                "specific_enough": False, "confidence": 0.6, "reason": "not the exact moment"}

    # the WRONG beat (shot_index 3) Face-IDs a CONFIRMED unrelated character (Cersei on a Jon Snow
    # beat) so it is genuinely contradictory; the other shots have no Face-ID.
    def _shot_for(sid, idx):
        return types.SimpleNamespace(index=idx, keyframe_path="kf.jpg", identities=[],
                                     face_ids=(["Cersei Lannister"] if idx == 3 else []))
    orig = (V.verify_frame, V._shot_lookup, V._cut.cut_selection, L.has_llm)
    try:
        V.verify_frame = fake_vf
        V._shot_lookup = lambda p: _shot_for
        V._cut.cut_selection = lambda p, s, c: None
        L.has_llm = lambda eng_cfg=None: True
        eng = types.SimpleNamespace(anthropic_model="test-model")
        V.verify_and_repair(proj, segs, ClipConfig(), eng, progress=None)
    finally:
        V.verify_frame, V._shot_lookup, V._cut.cut_selection, L.has_llm = orig

    check("GENERIC beat: on-topic filler KEPT (not flagged)",
          gen.verifier.get("verdict") == "keep" and not gen.flagged)
    # EXACT beat, right subject on screen (but META narration → matches_narration False): the
    # downgrade keys on the SUBJECT, not literal narration match, so it is kept as contextual.
    check("EXACT beat, right subject visible (meta narration) → DOWNGRADED to contextual",
          ctx.verifier.get("verdict") == "keep" and ctx.verifier.get("downgraded") == "exact→contextual"
          and not ctx.flagged and "verifier_failed" not in (ctx.flag_reasons or []))
    # EXACT character beat whose shot Face-IDs a CONFIRMED unrelated character (Cersei on a Jon Snow
    # beat) → contradictory → NOT downgraded (not contextual, not generic_filler) → flagged/blocked.
    check("EXACT character beat, CONFIRMED wrong character in Face-ID → NOT downgraded (blocked)",
          wrong.verifier.get("verdict") == "replace" and wrong.flagged
          and "verifier_failed" in (wrong.flag_reasons or []))

    # GENERIC-FILLER tier: an EXACT NON-CHARACTER beat (scene/event/object) whose subject is absent
    # (correct_subject_visible False) airs its thematic clip as honest generic_filler, not a block.
    tmp2 = tempfile.mkdtemp(prefix="csfill_")
    proj2 = M.ClipProject(name="t2", root=tmp2)
    fill = M.ClipSelection(segment_index=0, source_id="srcA", shot_index=1,
                           in_point=1.0, out_point=4.0, confidence=0.8)
    proj2.selections = [fill]
    segs2 = [M.ScriptSegment(index=0, text="Days earlier, at a wedding, an army was butchered",
                             visual_policy="exact_scene", required_kind="event",
                             required_entity="Red Wedding", is_specific_claim=True)]
    orig2 = (V.verify_frame, V._shot_lookup, V._cut.cut_selection, L.has_llm)
    try:
        V.verify_frame = lambda *a, **k: {"verdict": "replace", "matches_narration": False,
                                          "correct_subject_visible": False, "specific_enough": False,
                                          "confidence": 0.5, "reason": "off-pool event not shown"}
        V._shot_lookup = lambda p: (lambda sid, idx: fake_shot)
        V._cut.cut_selection = lambda p, s, c: None
        L.has_llm = lambda eng_cfg=None: True
        V.verify_and_repair(proj2, segs2, ClipConfig(),
                            types.SimpleNamespace(anthropic_model="m"), progress=None)
    finally:
        V.verify_frame, V._shot_lookup, V._cut.cut_selection, L.has_llm = orig2
    # The stubbed verifier rejects this footage on BOTH passes (strict AND lenient:
    # matches_narration=False, "off-pool event not shown"). Generic filler is a claim that the clip
    # is at least on-topic — nothing here proves that. The old code relabelled the SAME rejecting
    # verdict as "keep" and aired it, overwriting the verifier's own judgment.
    check("NON-CHARACTER beat the lenient pass ALSO rejects → NOT filler-eligible (unresolved)",
          fill.verifier.get("verdict") != "keep"
          and fill.verifier.get("relevance_class") != "generic_filler" and fill.flagged)
    # the prompt actually carries the specific/generic instruction
    vsrc = (Path(__file__).resolve().parent.parent / "vidlore" / "clipstudio" /
            "verify.py").read_text(encoding="utf-8")
    check("verify_frame prompt distinguishes specific vs generic",
          "is_specific" in vsrc and "GENERIC narration line" in vsrc and "Be STRICT" in vsrc)


def test_dark_patch_prepass():
    print("[dark] pre-pass catches a dark PATCH (same run-rule as the final black gate)")
    import subprocess
    from vidlore.clipstudio.build import _clip_too_dark
    from vidlore.clipstudio.config import ffmpeg_exe
    ff = ffmpeg_exe()
    td = Path(tempfile.mkdtemp(prefix="csdark_"))

    def _seg(color, dur, dest):
        subprocess.run([ff, "-y", "-loglevel", "error", "-f", "lavfi",
                        "-i", f"color=c={color}:s=640x360:rate=30", "-t", str(dur),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(dest)],
                       check=True, capture_output=True)

    def _cat(parts, dest):
        lst = td / f"l_{dest.stem}.txt"
        lst.write_text("".join(f"file '{p.name}'\n" for p in parts))
        subprocess.run([ff, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                        "-i", str(lst), "-c", "copy", str(dest)], check=True, capture_output=True)

    bright, dark, blip = td / "b.mp4", td / "d.mp4", td / "x.mp4"
    _seg("gray", 2, bright); _seg("0x060606", 1.2, dark); _seg("0x060606", 0.3, blip)
    patch = td / "patch.mp4"; _cat([bright, dark, bright], patch)
    allbright = td / "ok.mp4"; _seg("gray", 4, allbright)
    alldark = td / "all.mp4"; _seg("0x060606", 3, alldark)
    fade = td / "fade.mp4"; _cat([bright, blip, bright], fade)

    # THE BUG: the old pre-pass asked "dark THROUGHOUT?" (max of 4 samples), so a clip with a dark
    # PATCH passed here and then quarantined the whole render at the final gate (a 1s region at
    # 274.5s failed a 15-minute render). It must use the gate's own sustained-run rule.
    check("dark-clip pre-pass: a dark PATCH inside a bright clip is CAUGHT (the 274.5s render bug)",
          _clip_too_dark(patch) is True)
    check("dark-clip pre-pass: a fully bright clip passes", _clip_too_dark(allbright) is False)
    check("dark-clip pre-pass: a fully dark clip is still caught", _clip_too_dark(alldark) is True)
    check("dark-clip pre-pass: a short 0.3s dark blip (fade) is NOT over-flagged",
          _clip_too_dark(fade) is False)


def test_breakout_qa_whisper_generator():
    print("[breakout] post-render QA: whisper GENERATOR must be materialised once (speech_frac)")
    import subprocess
    import sys as _sys
    import types as _t
    from vidlore.clipstudio.build import _postrender_breakout_qa
    from vidlore.clipstudio.config import ffmpeg_exe
    ff = ffmpeg_exe()
    tmp = Path(tempfile.mkdtemp(prefix="csqagen_"))
    # a 4s video WITH audible audio so the loudness floor passes and only ASR decides
    vid = tmp / "v.mp4"
    subprocess.run([ff, "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=s=320x240:rate=30",
                    "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000", "-t", "4",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(vid)],
                   check=True, capture_output=True)

    class _Word:
        def __init__(self, w, b, e):
            self.word, self.start, self.end = w, b, e

    class _Seg:
        def __init__(self, words):
            self.words = words

    class _FakeModel:
        def __init__(self, *a, **k):
            pass

        def transcribe(self, path, **k):
            # faster-whisper returns a GENERATOR for segments — reproduce that exactly.
            def _gen():
                yield _Seg([_Word("real", 0.0, 0.5), _Word("war", 0.6, 1.2), _Word("king", 1.3, 2.4)])
            return _gen(), None

    fake = _t.ModuleType("faster_whisper")
    fake.WhisperModel = _FakeModel
    _old = _sys.modules.get("faster_whisper")
    _sys.modules["faster_whisper"] = fake
    try:
        probs = _postrender_breakout_qa(vid, [{"start": 0.5, "dur": 3.0, "line": "real war king",
                                               "video": str(vid)}], tmp, log=lambda m: None)
    finally:
        if _old is not None:
            _sys.modules["faster_whisper"] = _old
        else:
            _sys.modules.pop("faster_whisper", None)
    _no_speech = [p for p in probs if "NO detectable speech" in (p.get("reason") or "")]
    # On the buggy code the _wds comprehension consumed the generator, leaving _dur empty →
    # speech_frac 0.00 → EVERY breakout video was quarantined as "no detectable speech".
    check("post-render QA: whisper generator consumed ONCE → speech detected (no false 'no speech')",
          not _no_speech)


def test_character_present_unconfirmed():
    print("[relevance] character beat, subject present-but-unconfirmed → era+FaceID-gated contextual")
    from vidlore.clipstudio.verify import (_present_unconfirmed_ok, _confirmed_wrong_character,
                                           _beat_mention_tokens)
    from vidlore.clipstudio import models as M

    joff = M.ScriptSegment(index=1, text="Joffrey is carried out screaming",
                           required_kind="character", required_entity="Joffrey Baratheon")
    V0 = {"correct_subject_visible": False}   # rejected pick, subject not face-confirmed

    # --- CO-MENTIONED character guard (_confirmed_wrong_character) ---
    joff_tywin = M.ScriptSegment(index=2, text="Joffrey calls Tywin a coward", required_kind="character",
                                 required_entity="Joffrey Baratheon", entities=["Tywin Lannister"])
    check("CO-MENTIONED character (Tywin on a 'Joffrey calls Tywin' beat) is NOT a wrong character",
          _confirmed_wrong_character(joff_tywin, ["Tywin Lannister"]) is False)
    check("EMPTY / unconfirmed Face-ID is NOT a confirmed wrong character",
          _confirmed_wrong_character(joff, []) is False)
    check("a CONFIRMED unrelated person (Sansa on a Joffrey beat) IS a wrong character",
          _confirmed_wrong_character(joff, ["Sansa Stark"]) is True)
    check("single-scene ROSTER member (Cersei) is allowed via extra_ok_tokens",
          _confirmed_wrong_character(joff, ["Cersei Lannister"], {"cersei", "lannister"}) is False)
    check("_beat_mention_tokens gathers required + co-mentioned entities",
          _beat_mention_tokens(joff_tywin) >= {"joffrey", "baratheon", "tywin", "lannister"})

    # Right era + no wrong Face-ID but an EMPTY one → BLOCKS. Absence of a wrong face is not
    # presence of the right one, and this rule was vacuously true in the render that exposed it:
    # Face-ID had NO REFERENCE for Joffrey, Varys or Pycelle (oversized wiki stills defeated
    # YuNet), so "no confirmed wrong character" held for every frame in existence and 121 exact
    # beats were downgraded to contextual over whatever happened to be on screen.
    check("right era but EMPTY Face-ID → present-unconfirmed contextual BLOCKS (unknown ≠ present)",
          _present_unconfirmed_ok(V0, joff, "Game of Thrones S03E10 small council", [],
                                  "season 3") is False)
    # POSITIVE evidence — Face-ID actually places the required entity in the shot → allowed
    check("right era + Face-ID CONFIRMS the required entity → contextual ALLOWED",
          _present_unconfirmed_ok(V0, joff, "Game of Thrones S03E10 small council",
                                  ["Joffrey Baratheon"], "season 3") is True)

    # REGRESSION CORNER (the exact hole the reviewer flagged): UNCONSTRAINED era + empty Face-ID +
    # no wrong_subject_visible → MUST still block (never gamble a wrong character through).
    check("UNCONSTRAINED era + empty Face-ID → STILL BLOCKS (fails closed, no gamble)",
          _present_unconfirmed_ok(V0, joff, "Game of Thrones small council", [], "") is False)

    # a DIFFERENT identified main character in the shot → contradictory → block
    check("wrong character in Face-ID (Cersei on a Joffrey beat) → BLOCKS",
          _present_unconfirmed_ok(V0, joff, "Game of Thrones S03E10", ["Cersei Lannister"], "season 3") is False)

    # a source declaring a DIFFERENT season → wrong era → block
    check("wrong-era source (S05) on a season-3 beat → BLOCKS",
          _present_unconfirmed_ok(V0, joff, "Game of Thrones S05E08", [], "season 3") is False)

    # vision explicitly saw a different main subject → block even with matching era
    check("wrong_subject_visible=True → BLOCKS regardless of era/FaceID",
          _present_unconfirmed_ok({"correct_subject_visible": False, "wrong_subject_visible": True},
                                  joff, "Game of Thrones S03E10", [], "season 3") is False)

    # the SAME required character face present is fine (matches the required entity token)
    check("same character in Face-ID (Joffrey) + right era → ALLOWED",
          _present_unconfirmed_ok(V0, joff, "Game of Thrones S03E10", ["Joffrey Baratheon"], "season 3") is True)

    # --- behavioral: a character beat, empty Face-ID, UNCONSTRAINED era → GENERIC-FILLER last resort
    #     (not a block), because no CONFIRMED wrong character is present ---
    import vidlore.clipstudio.verify as V
    import vidlore.clipstudio.llm as L
    from vidlore.clipstudio.config import ClipConfig
    tmpc = tempfile.mkdtemp(prefix="cschar_")
    projc = M.ClipProject(name="tc", root=tmpc)
    projc.meta["analysis"] = {"video_type": "multi_scene"}          # roster NOT auto-allowed
    cb = M.ClipSelection(segment_index=0, source_id="srcA", shot_index=1, in_point=1, out_point=4,
                         confidence=0.8)
    projc.selections = [cb]
    csegs = [M.ScriptSegment(index=0, text="He calls Tywin a coward", required_kind="character",
                             required_entity="Joffrey Baratheon", entities=["Tywin Lannister"],
                             visual_policy="exact_scene", is_specific_claim=True)]
    fshot = types.SimpleNamespace(index=1, keyframe_path="kf.jpg", face_ids=[], identities=[])
    origc = (V.verify_frame, V._shot_lookup, V._cut.cut_selection, L.has_llm)
    try:
        V.verify_frame = lambda *a, **k: {"verdict": "replace", "matches_narration": False,
                                          "correct_subject_visible": False, "wrong_subject_visible": True,
                                          "specific_enough": False, "confidence": 0.5, "reason": "not Joffrey"}
        V._shot_lookup = lambda p: (lambda sid, idx: fshot)
        V._cut.cut_selection = lambda p, s, c: None
        L.has_llm = lambda eng_cfg=None: True
        V.verify_and_repair(projc, csegs, ClipConfig(),
                            types.SimpleNamespace(anthropic_model="m"), progress=None)
    finally:
        V.verify_frame, V._shot_lookup, V._cut.cut_selection, L.has_llm = origc
    # The stub says wrong_subject_visible=True and reason="not Joffrey" — the verifier LOOKED and
    # saw the wrong person. The old rung kept it as generic filler regardless, because Face-ID was
    # empty and so `not _confirmed_wrong_character(...)` held: an absent accusation outranked an
    # explicit one. With the leads unresolvable that was true of every frame in existence.
    check("character beat the verifier says shows the WRONG subject → NOT filler-eligible",
          cb.verifier.get("verdict") != "keep"
          and cb.verifier.get("relevance_class") != "generic_filler" and cb.flagged)


def test_verify_only_indices_subset():
    print("[recovery] verify_and_repair only_indices restricts the re-verify to a beat subset")
    from vidlore.clipstudio import verify as V
    from vidlore.clipstudio import models as M
    from vidlore.clipstudio.config import ClipConfig
    import vidlore.clipstudio.llm as L

    tmp = tempfile.mkdtemp(prefix="csonly_")
    proj = M.ClipProject(name="t", root=tmp)
    a = M.ClipSelection(segment_index=0, source_id="srcA", shot_index=1, in_point=1, out_point=4,
                        confidence=0.8, verifier={"status": "ok", "verdict": "keep", "sentinel": "OLD"})
    b = M.ClipSelection(segment_index=1, source_id="srcA", shot_index=2, in_point=5, out_point=8,
                        confidence=0.8, verifier={"status": "ok", "verdict": "keep", "sentinel": "OLD"})
    proj.selections = [a, b]
    segs = [M.ScriptSegment(index=0, text="beat zero", is_specific_claim=False),
            M.ScriptSegment(index=1, text="beat one", is_specific_claim=False)]
    fake_shot = types.SimpleNamespace(index=1, keyframe_path="kf.jpg", face_ids=[], identities=[])
    calls = []

    def fake_vf(kf, narration, ent, kind, names, eng_cfg, model="", is_specific=True, **kwargs):
        calls.append(narration)
        return {"verdict": "keep", "matches_narration": True, "correct_subject_visible": True,
                "specific_enough": True, "confidence": 0.9, "reason": "new-verify"}

    orig = (V.verify_frame, V._shot_lookup, V._cut.cut_selection, L.has_llm)
    try:
        V.verify_frame = fake_vf
        V._shot_lookup = lambda p: (lambda sid, idx: fake_shot)
        V._cut.cut_selection = lambda p, s, c: None
        L.has_llm = lambda eng_cfg=None: True
        eng = types.SimpleNamespace(anthropic_model="test-model")
        V.verify_and_repair(proj, segs, ClipConfig(), eng, only_indices={1}, progress=None)
    finally:
        V.verify_frame, V._shot_lookup, V._cut.cut_selection, L.has_llm = orig

    check("only_indices verified ONLY beat 1 (single verify call)", calls == ["beat one"])
    check("beat 0 (outside subset) keeps its prior verdict untouched",
          a.verifier.get("sentinel") == "OLD")
    check("beat 1 (in subset) was re-verified fresh", b.verifier.get("sentinel") is None
          and b.verifier.get("reason") == "new-verify")


def test_reaction_still_pool_gate():
    print("[image] reaction/essay frames excluded from the still-recovery pool (Ned-render regression)")
    import types
    from vidlore.clipstudio import discover as D
    from vidlore.clipstudio import image_fallback as IF

    # (a) source-title gate: reaction / essay / review / interview / non-show uploads are excluded
    #     from the still pool, while clean scene uploads are kept. These are the ACTUAL titles that
    #     leaked reactor frames into the Ned Stark render.
    check("source gate EXCLUDES reaction uploads (reactor facecam keyframes)",
          D.is_unwanted_source_title('MORE Reactors Reactions to NED STARK LOSING HIS HEAD')
          and D.is_unwanted_source_title('Best Reactions to "Eddard Stark\'s Execution" | Game Of Thrones'))
    check("source gate EXCLUDES essay/explained uploads",
          D.is_unwanted_source_title('Ned Stark\'s Execution | Game of Thrones S1E9 Explained'))
    check("source gate KEEPS clean raw scene uploads",
          not D.is_unwanted_source_title('Ned Stark\'s Execution | Game of Thrones: Season 1, Episode 9')
          and not D.is_unwanted_source_title('Game of Thrones - Varys and Ned dungeon scene')
          and not D.is_unwanted_source_title('Sansa Stark scene pack | Game of thrones season six'))

    # (b) OCR overlay guard (defense-in-depth): a keyframe carrying reactor-name / logo / burned text
    #     is rejected; a clean scene keyframe is accepted.
    def sh(ocr_text="", ocr_names=None, quality=0.9, phash="", keyframe_path="kf.jpg"):
        return types.SimpleNamespace(ocr_text=ocr_text, ocr_names=ocr_names or [],
                                     quality=quality, phash=phash, keyframe_path=keyframe_path)
    check("OCR guard rejects reactor-name / logo overlay frames",
          IF._shot_has_overlay_text(sh(ocr_text="AmberReacts No!Stop!"))
          and IF._shot_has_overlay_text(sh(ocr_text="Kristian Harloff GAMEOF HRONES"))
          and IF._shot_has_overlay_text(sh(ocr_names=["Snaxan"])))
    check("OCR guard accepts a clean scene frame (no on-screen text)",
          not IF._shot_has_overlay_text(sh(ocr_text="", ocr_names=[]))
          and not IF._shot_has_overlay_text(sh(ocr_text="ok")))   # 1-2 char OCR noise tolerated

    # (c) both wiring points present
    osrc = (Path(__file__).resolve().parent.parent / "vidlore" / "clipstudio" /
            "orchestrate.py").read_text(encoding="utf-8")
    ifsrc = (Path(__file__).resolve().parent.parent / "vidlore" / "clipstudio" /
             "image_fallback.py").read_text(encoding="utf-8")
    check("orchestrate excludes unwanted-title sources when building the still pool",
          "is_unwanted_source_title" in osrc and "shots_by_key" in osrc)
    check("both still pickers apply the OCR overlay guard",
          ifsrc.count("_shot_has_overlay_text(shot)") >= 2)


def test_nonscene_footage_gate():
    print("[match] non-scene sources (podcast/BTS-makeup/concert/theory) blocked: title gate + Face-ID footage gate")
    import types, tempfile, os
    from datetime import datetime
    from vidlore.clipstudio import discover as D
    from vidlore.clipstudio import match as MM
    from vidlore.clipstudio import models as M
    from vidlore.clipstudio.models import SOURCE_OK

    # (a) TITLE gate — the EXACT titles that leaked NON-scene footage into the Night-King render
    #     (a two-host podcast, a White-Walker makeup/BTS clip, an actor retrospective, a table-read,
    #     a theory upload, a single-song music clip). All must now be rejected.
    leak_titles = [
        "Bran's Elaborate Plan to Beat the Night King (Game of Thrones)",
        "Is The Night King A Targaryen? (SPIRAL SYMBOL Meaning)",
        "I Got Transformed Into the Night King From Game Of Thrones",
        "A Decade of Game of Thrones | John Bradley on Samwell Tarly",
        "GoT8 FINAL Script Reading - ARYA kills Night King!!",
        "The Long Night Song Game of Thrones Season 8 E03 Song",
        "Game of Thrones White Walker Makeup Transformation",
        "Game of Thrones Night King Theme - Live Concert (Ramin Djawadi)",
        "Game of Thrones Podcast: the Night King vlog",
        # FABRICATED-OUTCOME / AI recreations — a '🔥 Oberyn Martell Kills The Mountain | ...
        # Alternate Ending ⚔️' AI source fed a breakout + 3 beats of photorealistic AI footage
        # (the graphics gate caught only 1/48 of its shots). In the real duel the Mountain kills
        # Oberyn, so any 'Oberyn kills the Mountain / alternate ending' is a rewrite, not footage.
        "🔥 Oberyn Martell Kills The Mountain | Game of Thrones Alternate Ending ⚔️",
        "What If Oberyn Killed The Mountain - Alternate Ending",
        "Game of Thrones Alternate Timeline: If Oberyn Won",
        "If Ned Stark Lived | Game of Thrones",
        "Oberyn Survived - Alternate Universe (GoT)",
        "GoT AI Recreation: The Red Viper's Revenge",
    ]
    missed = [t for t in leak_titles if not D.is_unwanted_source_title(t)]
    check(f"title gate BLOCKS all leaked non-scene titles (missed={missed})", not missed)

    # legit raw scene clips stay ALLOWED — incl. the \bsong\b lookahead guard so the SHOW's own name
    # ('A Song of Ice and Fire') never false-trips the new music rule.
    keep_titles = [
        "Game of Thrones S06E05 - The Children of the forest",
        "Arya Stark kills Night King",
        "Battle of Hardhome - Full Scene | Game of Thrones (S5E8)",
        "The Night King raises the dead at Hardhome | Game of Thrones",
        "Game of Thrones: The Night King [1080p HD]",
        "Game of Thrones - A Song of Ice and Fire (opening scene)",
        # real duel/scene uploads that share tokens with the fabricated ones must still pass
        "Oberyn Martell Fights The Mountain | Game of Thrones | HBO Max",
        "Game Of Thrones - S04E07 - Oberyn as a Tyrion's Champion",
        "The Scene Oberyn Martell Exposed Tywin Lannister's Weakness",
    ]
    wrongly = [t for t in keep_titles if D.is_unwanted_source_title(t)]
    check(f"title gate KEEPS clean raw scene clips (wrongly blocked={wrongly})", not wrongly)

    # (b) VISUAL footage gate — the robust backstop for non-scene uploads whose TITLE dodges the
    #     keyword gates: a face-DENSE source that CLIP reads as a modern talking-head (podcast /
    #     vlog / interview / makeup-BTS) is dropped, a period scene is spared. (NOT cast-matching:
    #     Face-ID under-matched real cast 87% of the time on the leaked render, so cast-absence
    #     wrongly flagged Tyrion's trial etc.) CLIP is faked so the math is deterministic:
    #     modern prompt -> [1,0], period prompt -> [0,1].
    import numpy as np
    from vidlore.clipstudio import image_fallback as IF

    class _FakeVR:
        def _txt_embed(self, p):
            return [1.0, 0.0] if p in MM._MODERN_TH_PROMPTS else [0.0, 1.0]

    def mk(n_face, n_plain, vec):
        sh = [types.SimpleNamespace(faces=1, embed_row=i, source_id="x") for i in range(n_face)]
        sh += [types.SimpleNamespace(faces=0, embed_row=-1, source_id="x") for _ in range(n_plain)]
        return sh, np.array([list(vec) for _ in range(n_face)], dtype="float32")
    pod_sh,  pod_emb  = mk(9, 1, (1, 0))     # face-dense, frames look MODERN  -> drop
    scn_sh,  scn_emb  = mk(7, 3, (0, 1))     # face-dense, frames look PERIOD  -> keep (the FP we must avoid)
    act_sh,  act_emb  = mk(2, 8, (1, 0))     # 'modern' but NOT face-dense     -> keep
    tiny_sh, tiny_emb = mk(4, 0, (1, 0))     # 'modern' but too few face shots -> keep

    _savevr = IF._vr
    MM._TH_TXT_CACHE.clear()
    try:
        IF._vr = lambda: _FakeVR()
        check("visual gate FLAGS a modern talking-head/podcast source (CLIP-modern + face-dense)",
              MM._source_is_modern_talkinghead(pod_sh, pod_emb))
        check("visual gate SPARES a real period scene (CLIP-period, even when face-dense)",
              not MM._source_is_modern_talkinghead(scn_sh, scn_emb))
        check("visual gate SPARES a non-face-dense source (action/landscape)",
              not MM._source_is_modern_talkinghead(act_sh, act_emb))
        check("visual gate never judges a tiny source (<8 face-bearing shots)",
              not MM._source_is_modern_talkinghead(tiny_sh, tiny_emb))
        IF._vr = lambda: None
        check("visual gate is SAFE without CLIP (returns False so the title gate still governs)",
              not MM._source_is_modern_talkinghead(pod_sh, pod_emb))

        # (c) _load_pool integration — DROP the talking-head source, KEEP the period scene; kill switch.
        tmp = Path(tempfile.mkdtemp())
        proj = M.ClipProject(name="t", root=tmp, created_at=datetime.now(),
                             sources=[], segments=[], selections=[], meta={})
        proj.sources = [
            M.SourceVideo(id="pod", url="u", title="neutral title one", local_path="x",
                          duration=60.0, permission="owner", status=SOURCE_OK),
            M.SourceVideo(id="scene", url="u", title="neutral title two", local_path="x",
                          duration=60.0, permission="owner", status=SOURCE_OK),
        ]
        shots_by  = {"pod": pod_sh,  "scene": scn_sh}
        embeds_by = {"pod": pod_emb, "scene": scn_emb}
        save = (MM._index.load_shots, MM._index.load_embeds, MM._source_is_nonphotographic)
        try:
            MM._index.load_shots = lambda p, sid: shots_by.get(sid, [])
            MM._index.load_embeds = lambda p, sid: embeds_by.get(sid)
            MM._source_is_nonphotographic = lambda *a, **k: False
            IF._vr = lambda: _FakeVR()
            ids = {ps.sid for ps in MM._load_pool(proj)}
            check("_load_pool DROPS the title-clean modern talking-head source", "pod" not in ids)
            check("_load_pool KEEPS the period scene source", "scene" in ids)
            os.environ["VIDLORE_CLIPSTUDIO_FACE_FOOTAGE_GATE"] = "0"
            ids2 = {ps.sid for ps in MM._load_pool(proj)}
            check("kill switch VIDLORE_CLIPSTUDIO_FACE_FOOTAGE_GATE=0 disables the visual gate", "pod" in ids2)
        finally:
            os.environ.pop("VIDLORE_CLIPSTUDIO_FACE_FOOTAGE_GATE", None)
            MM._index.load_shots, MM._index.load_embeds, MM._source_is_nonphotographic = save
    finally:
        IF._vr = _savevr


def test_breakout_evidence_mining_reachable():
    print("[breakout] evidence-mining runs even when the quote pool is empty (essay scripts)")
    src = (Path(__file__).resolve().parent.parent / "vidlore" / "clipstudio" /
           "build.py").read_text(encoding="utf-8")
    fn = src.split("def _select_breakouts")[1].split("\ndef ")[0]
    # the structural bug: an early `if not quote_segs: return []` blocked the evidence-miner
    # (whose own comment says it "always runs") — so analytical/essay scripts got 0 breakouts.
    check("no early-return on empty quote pool (evidence-mining stays reachable)",
          "if not quote_segs:" not in fn)
    check("evidence-mining block still present and unconditional",
          "EVIDENCE MINING" in fn and "_mine_tier(" in fn)
    check("terminal bail-out on zero candidates is the real exit",
          "if not cands:" in fn)


def test_intro_coldopen_breakout():
    print("[breakout] intro COLD-OPEN: scene-0 verbatim quote aired; generic scene-0 rejected; gates hold")
    import types
    from pathlib import Path as _P
    from vidlore.clipstudio import build as B
    import vidlore.clipstudio.index as _idxmod

    _P("/tmp/x.mp4").write_bytes(b"x")                  # source path must exist (helpers are mocked)

    def _run(*, quote, asr, aired=None, face_ids=(), luma=80.0, burned=False,
             src_title="Cersei vs Littlefinger throne scene HD", n_segs=6):
        shots = [
            types.SimpleNamespace(index=0, start=12.0, end=20.0, transcript=asr,
                                  face_ids=list(face_ids), local_path="/tmp/x.mp4"),
            types.SimpleNamespace(index=1, start=22.0, end=26.0, transcript="",
                                  face_ids=[], local_path="/tmp/x.mp4"),
            types.SimpleNamespace(index=2, start=30.0, end=34.0, transcript="he walks in",
                                  face_ids=[], local_path="/tmp/x.mp4"),
        ]
        src = types.SimpleNamespace(id="s1", status="ok", local_path="/tmp/x.mp4",
                                    title=src_title, width=1920, extra={"anchor_verified": True})
        proj = types.SimpleNamespace(sources=[src], meta={"analysis": {
            "characters": [{"name": "Cersei Lannister", "actor": "Lena Headey"}],
            "anchor_scenes": [{"name": "Cersei power scene",
                               "query": "cersei littlefinger power", "dialogue": []}],
            "movie_title": "Game of Thrones", "episode_hint": ""}})
        segs = [types.SimpleNamespace(
                    index=i,
                    text=("seize him cut his throat power king realm" if i == 0
                          else f"narration filler beat number {i} words here now"),
                    quote=(quote if i == 0 else "")) for i in range(n_segs)]
        _orig_ls = _idxmod.load_shots
        _orig = (B._extract_breakout, B._breakout_window_luma, B._frame_has_burned_text,
                 B._asr_wav_words, B._breakout_window_admissible)
        _idxmod.load_shots = lambda p, sid: shots
        # These cases exercise COLD-OPEN mechanics (quote coverage, hook stitching, audit
        # provenance) on synthetic beats whose narration is deliberately nonsense filler. The
        # post-extract admission judge is a separate concern with its own suite
        # (tests/test_breakout_relevance.py) and would correctly refuse this fixture, so stub it
        # here exactly as the extraction pipeline above is stubbed.
        B._breakout_window_admissible = lambda *a, **k: (True, "stubbed for the cold-open cases", [])
        B._extract_breakout = lambda *a, **k: 5.0
        B._breakout_window_luma = lambda *a, **k: luma
        B._frame_has_burned_text = lambda *a, **k: burned
        # test-double for the aired-audio RE-ASR (the mocked _extract_breakout writes no wav): by
        # default the aired audio equals the scene ASR (contains the quote → coverage passes); pass
        # `aired` to simulate a mismatched aired transcript (low ordered coverage → dropped).
        _airtx = aired if aired is not None else asr
        B._asr_wav_words = lambda p: (_airtx.split(), _airtx, 5.0)
        logs = []
        try:
            out = B._select_breakouts(proj, segs, 200.0, _P("/tmp"), lambda m: logs.append(str(m)))
        finally:
            _idxmod.load_shots = _orig_ls
            (B._extract_breakout, B._breakout_window_luma, B._frame_has_burned_text,
             B._asr_wav_words, B._breakout_window_admissible) = _orig
        return out, "\n".join(logs)

    def _has0(out):
        return any(o["seg_index"] == 0 for o in out)

    VQ = "Seize him. Cut his throat."                    # opening hook quote (5 content words)
    SC = "seize him cut his throat stop wait i have changed my mind"   # scene ASR contains it verbatim

    # 1) scene-0 verbatim cold-open ACCEPTED + logged as COLD-OPEN
    out, logs = _run(quote=VQ, asr=SC)
    check("scene-0 verbatim quote accepted as cold-open", _has0(out))
    check("[BREAKOUT-OK] logs COLD-OPEN provenance", "COLD-OPEN" in logs)

    # 2) generic / non-verbatim scene-0 candidate REJECTED (no quote → mined-only → dropped at scene 0)
    out2, _ = _run(quote="", asr="seize him cut his throat power king realm and the crown")
    check("scene-0 generic (non-verbatim) candidate rejected", not _has0(out2))

    # 3) verbatim cold-open OVERRIDES Face-ID (empty face_ids would normally be wrong-character)
    out3, logs3 = _run(quote=VQ, asr=SC, face_ids=[])
    check("verbatim cold-open overrides Face-ID gate",
          _has0(out3) and "audio-match overrides face guard" in logs3)

    # 4) safety gates STILL reject a verbatim cold-open:
    check("verbatim cold-open still rejected on near-black",
          not _has0(_run(quote=VQ, asr=SC, luma=10.0)[0]))
    check("verbatim cold-open still rejected on burned-in text",
          not _has0(_run(quote=VQ, asr=SC, burned=True)[0]))
    _cm = "basically what this scene shows is cersei seize him cut his throat then changes her mind to prove power"
    check("verbatim cold-open still rejected on commentary audio",
          not _has0(_run(quote="Seize him. Cut his throat.", asr=_cm)[0]))

    # 5) R3-2: a COLD-OPEN must meet the SAME >=70% ordered-coverage floor — an unrelated aired
    #    transcript (shot has the quote so the candidate is created, but the extracted audio does NOT
    #    contain it) is REJECTED even though it is the cold-open (no `not _is_cold` exemption).
    out5, logs5 = _run(quote=VQ, asr=SC,
                       aired="the weather today is quite pleasant nothing at all like the quote")
    check("unrelated low-coverage COLD-OPEN is rejected (>=70% floor applies to cold-opens)",
          not _has0(out5) and "COLD-OPEN) DROPPED" in logs5)
    # a cold-open whose aired audio DOES contain the line still airs (regression guard)
    check("faithful cold-open still airs (coverage passes)", _has0(_run(quote=VQ, asr=SC)[0]))
    # candidate origin is explicit + persisted in the audit (not inferred from _verbatim_strong)
    _oc = _run(quote=VQ, asr=SC)[0]
    _c0 = next((o for o in _oc if o["seg_index"] == 0), None)
    check("candidate origin + type + accepted floor persisted in the audit",
          _c0 is not None and _c0["_audit"].get("candidate_origin") == "verbatim_quote"
          and _c0["_audit"].get("candidate_type") == "verbatim"
          and _c0["_audit"].get("accepted_coverage_floor") == 0.70)
    check("verbatim cold-open still rejected on recap line",
          not _has0(_run(quote="last time I was here I killed",
                         asr="last time i was here i killed my father with a crossbow")[0]))
    check("verbatim cold-open still rejected on later-era / wrong source",
          not _has0(_run(quote=VQ, asr=SC,
                         src_title="Game of Thrones FINALE Breakdown Explained reaction")[0]))

    # A) COLD-OPEN FRAGMENTATION — analyze splits the opening hook into tiny per-beat fragments;
    # they must combine into ONE scene-0 hook candidate (the user's exact A.4 case).
    fr = [types.SimpleNamespace(index=0, quote="Seize him.", text="x"),
          types.SimpleNamespace(index=1, quote="Cut his throat.", text="x"),
          types.SimpleNamespace(index=2, quote="Stop.", text="x"),
          types.SimpleNamespace(index=3, quote="Wait — I've changed my mind.", text="x"),
          types.SimpleNamespace(index=4, quote="", text="x")]
    hook = B._combine_opening_hook(fr)
    check("fragmented opening hook combined into one scene-0 candidate",
          hook is not None and hook[0].index == 0
          and hook[1] == "Seize him. Cut his throat. Stop. Wait — I've changed my mind."
          and hook[2] == {0, 1, 2, 3})
    check("no opening hook when the first beats carry no quotes",
          B._combine_opening_hook([types.SimpleNamespace(index=i, quote="", text="x")
                                   for i in range(5)]) is None)
    # end-to-end: fragmented opening beats + footage carrying the full hook → scene-0 COLD-OPEN airs
    def _run_frag():
        shots = [types.SimpleNamespace(index=0, start=10.0, end=20.0,
                     transcript="seize him cut his throat stop wait i have changed my mind let him go",
                     face_ids=[], local_path="/tmp/x.mp4"),
                 types.SimpleNamespace(index=1, start=40.0, end=44.0, transcript="",
                     face_ids=[], local_path="/tmp/x.mp4"),
                 types.SimpleNamespace(index=2, start=50.0, end=54.0, transcript="he walks in",
                     face_ids=[], local_path="/tmp/x.mp4")]
        src = types.SimpleNamespace(id="s1", status="ok", local_path="/tmp/x.mp4",
                 title="Knowledge Is Power scene HD", width=1920, extra={"anchor_verified": True})
        proj = types.SimpleNamespace(sources=[src], meta={"analysis": {
            "characters": [{"name": "Cersei Lannister", "actor": "Lena Headey"}],
            "anchor_scenes": [{"name": "power scene", "query": "cersei power", "dialogue": []}],
            "movie_title": "Game of Thrones", "episode_hint": ""}})
        segs = [fr[0], fr[1], fr[2], fr[3]] + [types.SimpleNamespace(
            index=i, quote="", text=f"narration filler beat {i} words here") for i in range(4, 9)]
        _ls = _idxmod.load_shots
        _o = (B._extract_breakout, B._breakout_window_luma, B._frame_has_burned_text,
              B._asr_wav_words, B._breakout_window_admissible)
        _idxmod.load_shots = lambda p, sid: shots
        # same reasoning as _run above: this case is about HOOK STITCHING, not relevance
        B._breakout_window_admissible = lambda *a, **k: (True, "stubbed for the hook case", [])
        B._extract_breakout = lambda *a, **k: 6.0
        B._breakout_window_luma = lambda *a, **k: 80.0
        B._frame_has_burned_text = lambda *a, **k: False
        _hooktx = "seize him cut his throat stop wait i have changed my mind let him go"
        B._asr_wav_words = lambda p: (_hooktx.split(), _hooktx, 6.0)
        lg = []
        try:
            o = B._select_breakouts(proj, segs, 200.0, _P("/tmp"), lambda m: lg.append(str(m)))
        finally:
            _idxmod.load_shots = _ls
            (B._extract_breakout, B._breakout_window_luma, B._frame_has_burned_text,
             B._asr_wav_words, B._breakout_window_admissible) = _o
        return o, "\n".join(lg)
    of, lf = _run_frag()
    check("fragmented hook airs as a scene-0 COLD-OPEN breakout",
          any(o["seg_index"] == 0 for o in of) and "COLD-OPEN before-scene=0" in lf)


def test_breakout_window_commentary_gate():
    print("[breakout] essay-title source rejection + post-extraction window-commentary gate")
    import types
    from pathlib import Path as _P
    from vidlore.clipstudio import build as B
    from vidlore.clipstudio.build import _ESSAYISH_RX, _breakout_src_ok
    import vidlore.clipstudio.index as _idxmod

    # B.4 — essay-analysis TITLE rejection (the real bad case + the user's listed phrases)
    check("essay title 'Cersei's Fatal Mistake with Littlefinger' caught",
          bool(_ESSAYISH_RX.search("Cersei's Fatal Mistake with Littlefinger")))
    check("_breakout_src_ok rejects the fatal-mistake source",
          not _breakout_src_ok(types.SimpleNamespace(title="Cersei's Fatal Mistake with Littlefinger"), []))
    check("'lost ... because' analysis title caught",
          bool(_ESSAYISH_RX.search("How Cersei Lost Her Throne Because Of Littlefinger")))
    check("'explained' / 'breakdown' / 'analysis' titles caught",
          all(bool(_ESSAYISH_RX.search(t)) for t in
              ("Iron Throne Scene Explained", "Finale Breakdown", "Cersei Power Analysis")))
    check("clean real-scene source still allowed (not over-blocked)",
          _breakout_src_ok(types.SimpleNamespace(
              title="Littlefinger vs Cersei - Knowledge Is Power scene HD"), []))

    # B.1/B.2/B.3 — post-extraction window-audio validation: matched shot line is clean
    # ("chaos is a ladder"), but the extracted window bleeds into the NEXT shot's commentary.
    _P("/tmp/x.mp4").write_bytes(b"x")
    shots = [
        types.SimpleNamespace(index=0, start=575.0, end=579.0, transcript="",
                              face_ids=[], local_path="/tmp/x.mp4"),
        types.SimpleNamespace(index=1, start=581.0, end=584.0,
                              transcript="chaos isn't a pit chaos is a ladder",
                              face_ids=[], local_path="/tmp/x.mp4"),
        types.SimpleNamespace(index=2, start=584.0, end=591.0,
                              transcript="and she walked out untouched most people think cersei "
                                         "lost the throne because of this",
                              face_ids=[], local_path="/tmp/x.mp4"),
    ]
    src = types.SimpleNamespace(id="s1", status="ok", local_path="/tmp/x.mp4",
                               title="Littlefinger Chaos Is A Ladder scene HD", width=1920,
                               extra={"anchor_verified": True})
    proj = types.SimpleNamespace(sources=[src], meta={"analysis": {
        "characters": [{"name": "Petyr Baelish", "actor": "Aidan Gillen"}],
        "anchor_scenes": [{"name": "chaos is a ladder",
                           "query": "littlefinger chaos ladder", "dialogue": []}],
        "movie_title": "Game of Thrones", "episode_hint": ""}})
    segs = [types.SimpleNamespace(
                index=i,
                text=("chaos is a ladder climb the realm" if i == 5 else f"narration filler beat {i} words"),
                quote=("Chaos isn't a pit. Chaos is a ladder." if i == 5 else "")) for i in range(10)]
    _orig_ls = _idxmod.load_shots
    _orig = (B._extract_breakout, B._breakout_window_luma, B._frame_has_burned_text)
    _idxmod.load_shots = lambda p, sid: shots
    B._extract_breakout = lambda *a, **k: 7.5            # window [581, 588.5] spans the commentary shot
    B._breakout_window_luma = lambda *a, **k: 80.0
    B._frame_has_burned_text = lambda *a, **k: False
    logs = []
    try:
        out = B._select_breakouts(proj, segs, 800.0, _P("/tmp"), lambda m: logs.append(str(m)))
    finally:
        _idxmod.load_shots = _orig_ls
        B._extract_breakout, B._breakout_window_luma, B._frame_has_burned_text = _orig
    lt = "\n".join(logs)
    check("aired-window commentary rejected post-extraction (matched line was clean)",
          not out and "window_commentary" in lt and "REJECTED post-extract" in lt)


def test_coldopen_vocut():
    print("[breakout] cold-open VO word-cut (default ON, uploaded-VO only): replace hook, trim straddle, fallback")
    import types
    from pathlib import Path as _P
    from vidlore.clipstudio import build as B

    def _w(word, s, e):
        return types.SimpleNamespace(word=word, start=s, end=e)

    def _mk():
        # beat0 [0,2] fully in cut; beat1 [2,5] STRADDLES cut@4 (throat/mind<4 cut, thats/cersei>=4
        # kept); beat2 [5,9] + beat3 [9,13] survive
        ns = [types.SimpleNamespace(index=0, duration=2.0, audio=_P("/tmp/x.wav"),
                  words=[_w("seize", 0.0, 0.5), _w("him", 0.5, 1.0), _w("cut", 1.0, 1.6)]),
              types.SimpleNamespace(index=1, duration=3.0, audio=_P("/tmp/x.wav"),
                  words=[_w("throat", 2.0, 2.5), _w("mind", 3.0, 3.6),
                         _w("thats", 4.2, 4.6), _w("cersei", 4.6, 5.0)]),
              types.SimpleNamespace(index=2, duration=4.0, audio=_P("/tmp/x.wav"),
                  words=[_w("queen", 5.2, 5.6), _w("regent", 6.0, 6.6)]),
              types.SimpleNamespace(index=3, duration=4.0, audio=_P("/tmp/x.wav"),
                  words=[_w("the", 9.2, 9.4), _w("most", 9.4, 9.9)])]
        narr = types.SimpleNamespace(audio=_P("/tmp/vo.wav"), scenes=ns, total=13.0, _breakout_caps=[])
        segs = [types.SimpleNamespace(index=i, text=f"b{i}", est_duration=ns[i].duration, quote="")
                for i in range(4)]
        scs = [types.SimpleNamespace(index=i, role="x", visual="") for i in range(4)]
        co = {"seg_index": 0, "dur": 6.0, "video": _P("/tmp/clip.mp4"), "audio": _P("/tmp/clip.wav"),
              "cold_open": True, "hook_quote": "seize him cut his throat mind", "hook_beats": [0, 1]}
        return types.SimpleNamespace(sources=[]), segs, scs, narr, co

    _ol, _oa = B._locate_hook_span, B._audio_trim_prepend
    B._audio_trim_prepend = lambda *a, **k: _P("/tmp/narration_vocut.wav")
    B._locate_hook_span = lambda *a, **k: (0.0, 4.0, 1.0)          # cut at 4.0s
    proj, segs, scs, narr, co = _mk()
    try:
        res = B._apply_coldopen_vocut(proj, segs, scs, narr, co, _P("/tmp"), lambda m: None)
    finally:
        B._locate_hook_span, B._audio_trim_prepend = _ol, _oa
    check("VO-cut returns a transform (gate passed)", res is not None)
    if res:
        _ns2, _sc2, nnarr, bmap, imap, dropped = res
        delta = 6.0 - 4.0                                          # D_clip - h1 = 2.0
        check("cold-open clip is scene 0 (dur=D_clip)",
              bmap.get(0) is not None and abs(nnarr.scenes[0].duration - 6.0) < 1e-6)
        check("beat fully inside the cut is dropped (beat0 gone)", dropped == 1 and 0 not in imap)
        check("straddle beat trimmed to its post-cut tail",
              [w.word for w in nnarr.scenes[1].words] == ["thats", "cersei"]
              and abs(nnarr.scenes[1].duration - 1.0) < 1e-6)
        check("straddle tail word shifted by Δ", abs(nnarr.scenes[1].words[0].start - (4.2 + delta)) < 0.02)
        check("surviving beat words shifted by Δ", abs(nnarr.scenes[2].words[0].start - (5.2 + delta)) < 0.02)
        check("idx_map maps surviving originals → new indices", imap.get(2) == 2 and imap.get(3) == 3)
        check("cold-open caption added at t=0",
              bool(nnarr._breakout_caps) and nnarr._breakout_caps[0]["start"] == 0.0)
        check("narration audio replaced by the trimmed+prepended track",
              str(nnarr.audio).endswith("narration_vocut.wav"))

    # FALLBACK: hook not located → None (caller uses the proven insert path); fresh objects
    B._locate_hook_span = lambda *a, **k: None
    proj2, segs2, scs2, narr2, co2 = _mk()
    try:
        res2 = B._apply_coldopen_vocut(proj2, segs2, scs2, narr2, co2, _P("/tmp"), lambda m: None)
    finally:
        B._locate_hook_span = _ol
    check("VO-cut falls back (None) when the hook isn't confidently located", res2 is None)
    # the build wiring is default-ON for word-aligned uploaded voiceover (kill switch VO_CUT=0)
    # and falls back to insert on any uncertainty
    bsrc = (Path(__file__).resolve().parent.parent / "vidlore" / "clipstudio" /
            "build.py").read_text(encoding="utf-8")
    check("build falls back to insert when VO-cut returns None",
          "if not _done:" in bsrc and "_apply_coldopen_vocut(" in bsrc)

    # --- default-ON + safety-gate proofs (the env parse + uploaded-VO gate, mirrored from build()) ---
    check("VO-cut default is ON (env unset → '1')",
          'os.environ.get("VIDLORE_CLIPSTUDIO_VO_CUT", "1")' in bsrc)
    check("VO-cut runs ONLY on word-aligned uploaded voiceover (never TTS)",
          'getattr(narration, "_vo_word_aligned", False)' in bsrc)

    def _vocut_on(val):
        import os as _os
        _old = _os.environ.get("VIDLORE_CLIPSTUDIO_VO_CUT")
        if val is None:
            _os.environ.pop("VIDLORE_CLIPSTUDIO_VO_CUT", None)
        else:
            _os.environ["VIDLORE_CLIPSTUDIO_VO_CUT"] = val
        try:
            return _os.environ.get("VIDLORE_CLIPSTUDIO_VO_CUT", "1").strip() not in ("0", "false", "no")
        finally:
            if _old is None:
                _os.environ.pop("VIDLORE_CLIPSTUDIO_VO_CUT", None)
            else:
                _os.environ["VIDLORE_CLIPSTUDIO_VO_CUT"] = _old
    check("default (unset) → VO-cut ON", _vocut_on(None) is True)
    check("explicit '0' → VO-cut OFF (kill switch)", _vocut_on("0") is False)
    check("explicit 'false'/'no' → VO-cut OFF", _vocut_on("false") is False and _vocut_on("no") is False)
    check("explicit '1' → VO-cut ON", _vocut_on("1") is True)
    # the build gate is (_vocut AND narration._vo_word_aligned) — TTS narration lacks the marker
    _tts = types.SimpleNamespace()                              # TTS narration: no _vo_word_aligned
    _vo = types.SimpleNamespace(_vo_word_aligned=True)          # uploaded, word-aligned
    check("TTS narration → VO-cut gate is False (uses insert)",
          (_vocut_on(None) and getattr(_tts, "_vo_word_aligned", False)) is False)
    check("uploaded-VO narration → VO-cut gate is True",
          bool(_vocut_on(None) and getattr(_vo, "_vo_word_aligned", False)) is True)


def test_discovery_plural_gates_and_purge():
    print("[discover] PLURAL title-gate fix (reactions/reactors/reviews/...) + cached-source purge")
    from vidlore.clipstudio import discover as D
    from vidlore.clipstudio.discover import SourceCandidate
    from vidlore.clipstudio.config import load_clip_config
    cfg = load_clip_config()

    def rej(t):
        return D._reject(SourceCandidate(url="x", id="x", title=t, duration=300, height=720), cfg)

    # (a) the EXACT class of bug: \b(...)\b trailing boundary made singular alternatives miss plurals.
    #     Every one of these slipped through before the fix and would put on-camera/logo junk in a render.
    plural_junk = ["Game of Thrones Reactions", "MORE Reactors Reactions to Ned",
                   "Best Reactions to Eddard", "Movie Reviews", "Honest Reviews", "Reviewers react",
                   "Season Recaps", "Scene Breakdowns", "Ending Breakdowns", "Cast Interviews",
                   "Video Essays on Ned", "Movie Podcasts", "Biggest Plot Holes", "Featurettes",
                   "Best Parodies", "All Character Arcs", "Funny Skits", "Casting Auditions",
                   "Best Deepfakes", "Cast Reunions", "Ned Stark Revealed",
                   "Couples React to GoT", "Couples Reaction", "First Watches"]
    leaks = [t for t in plural_junk if not rej(t)]
    check(f"all {len(plural_junk)} plural-form junk titles now rejected at discovery", not leaks)
    if leaks:
        print("     STILL LEAKING:", leaks)

    # (b) no false-positives: real raw scene uploads must STILL pass the gate
    clean = ["Ned Stark Execution | Game of Thrones S1E9", "Varys and Ned dungeon scene",
             "Sansa Stark scene pack season six", "Joffrey becomes king",
             "Arya Stark on the streets", "Tywin Lannister small council scene",
             "Daenerys throne room speech"]
    fps = [(t, rej(t)) for t in clean if rej(t)]
    check("clean raw scene uploads still pass (no false-positives)", not fps)
    if fps:
        print("     FALSE POSITIVES:", fps)

    # (c) the specific singular forms that already worked must keep working
    check("singular forms still rejected (no regression)",
          rej("a reaction video") and rej("movie review") and rej("scene breakdown"))

    # (d) cached-source purge: a reaction source downloaded BEFORE the fix is blocked on next run
    import types
    from vidlore.clipstudio import orchestrate as O
    from vidlore.clipstudio.models import SOURCE_OK, SOURCE_BLOCKED
    proj = types.SimpleNamespace(sources=[
        types.SimpleNamespace(id="a", title="Ned Stark Execution | Game of Thrones S1E9", status=SOURCE_OK),
        types.SimpleNamespace(id="b", title="MORE Reactors Reactions to Ned", status=SOURCE_OK),
        types.SimpleNamespace(id="c", title="Best Reactions to Eddard", status=SOURCE_OK),
        types.SimpleNamespace(id="d", title="Varys and Ned dungeon scene", status=SOURCE_OK),
    ])
    n = O._purge_unwanted_sources(proj, None)
    blocked = {s.id for s in proj.sources if s.status == SOURCE_BLOCKED}
    kept_ok = {s.id for s in proj.sources if s.status == SOURCE_OK}
    check("purge blocks exactly the cached reaction sources, keeps clean ones",
          n == 2 and blocked == {"b", "c"} and kept_ok == {"a", "d"})
    check("purge is idempotent (second run blocks nothing new)",
          O._purge_unwanted_sources(proj, None) == 0)
    osrc = (Path(__file__).resolve().parent.parent / "vidlore" / "clipstudio" /
            "orchestrate.py").read_text(encoding="utf-8")
    check("produce_auto calls the purge after loading the project",
          "_purged = _purge_unwanted_sources(proj, log)" in osrc
          and osrc.count("_purge_unwanted_sources") >= 2)


def test_breakout_era_scene_gate():
    print("[breakout] wrong-era / wrong-character / recap gating (v1 breakout #2 regression)")
    import types
    from vidlore.clipstudio.build import (_is_recap_line as _recap, _script_wants_comparison as _cmp)
    bsrc = (Path(__file__).resolve().parent.parent / "vidlore" / "clipstudio" /
            "build.py").read_text(encoding="utf-8")

    # (a) S7 retrospective recap line rejected in an S4-targeted video — the ACTUAL breakout #2 text
    check("S7 recap dialogue flagged (rejected): 'last time I was here, I killed my father with a crossbow'",
          _recap("Last time I was here, I killed my father with a crossbow.")
          and _recap("Last time I was here, you killed my son with wildfire.")
          and _recap("all those years ago he chose the lion"))
    # same character but the SCENE'S OWN S4 dialogue must NOT be flagged
    check("the S4 scene's own in-character dialogue is NOT flagged as recap",
          not _recap("I am your son and you sentenced me to die")
          and not _recap("You're no son of mine") and not _recap("Say her name again"))

    # (b) comparison footage only when the script explicitly compares
    seg = lambda x: types.SimpleNamespace(text=x)
    check("comparison clip allowed ONLY when the script explicitly compares",
          _cmp([seg("This is calm, unlike House of the Dragon's chaos")])
          and not _cmp([seg("Tyrion shoots Tywin with a crossbow")]))

    # (c) guards are wired into the breakout selector
    check("era-gate bars purely-later-season SOURCES in S1/S2/S4-targeted video (unless comparison)",
          "min(_ss9b) > _core_max9" in bsrc and "_core_max9 and not _allow_compare9" in bsrc)
    check("recap-line gate wired in the breakout picker (wrong-season in-character dialogue)",
          '_is_recap_line(getattr(c[3]' in bsrc)
    # CONTRACT CHANGED, deliberately. `_fids9 & _main_faces9` READ like a wrong-character test and
    # was not one: measured on a real render, 0 of 625 Face-ID name instances fall outside the main
    # cast (match() is an argmax over a roster of main cast only), so the expression was
    # `bool(face_ids)` in disguise. It rejected 25 candidates that were merely UNIDENTIFIED — 10 of
    # the 18 distinct shots plainly show main cast — while PASSING 7 candidates whose confident face
    # is a main-cast member the beat is not about, 2 of which aired. Three states now, never
    # conflated: right / wrong / unknown.
    check("identity gate separates WRONG from UNKNOWN (not `bool(face_ids)` in disguise)",
          "_conf9 and _tgt9 and _full9 and not (_conf9 & _tgt9)" in bsrc
          and '_rej["unidentified"]' in bsrc
          and "_fids9 & _main_faces9" not in bsrc)
    check("EXACT episode / anchor-verified source PREFERRED (tier1 mined before tier2)",
          "anchor_verified" in bsrc
          and "_mine_tier([s for s in srcs if s.id in _tier1], _min_ov9)" in bsrc
          and "_mine_tier([s for s in srcs if s.id in _tier2], _min_ov9)" in bsrc)
    # (d) every render must REPORT a greppable breakout audit: candidates + per-reason counts + kept
    check("greppable [BREAKOUT-AUDIT] summary with per-reason rejection counts",
          "[BREAKOUT-AUDIT]" in bsrc and "candidates=" in bsrc and "accepted=" in bsrc
          and all(k in bsrc for k in ("commentary=", "recap/wrong-era=", "wrong-character=",
                                      "dark=", "later-era-source=", "essay/foreign-source=")))
    check("greppable [BREAKOUT-OK] accepted-provenance line (source title + timestamp + line)",
          "[BREAKOUT-OK]" in bsrc and "src={(src.title" in bsrc and "src@" in bsrc)
    check("self-diagnostic flags when the conservative Face-ID gate is the limiter",
          "the conservative" in bsrc and "Face-ID gate is the MAIN limiter" in bsrc
          and 'max(_rej, key=_rej.get)' in bsrc)


def test_burned_text_black_and_recap_gates():
    print("[footage] burned-text (no-space) + black-frame + recap/fan-edit source gates")
    import types
    from vidlore.clipstudio.match import _ocr_text_heavy as H
    from vidlore.clipstudio.discover import _REJECT_TITLE as RJ

    def sh(t):
        return types.SimpleNamespace(ocr_text=t)
    # Fix A — no-space concatenated OCR (the bug that aired a source's red recap caption / hard
    # subtitle / outro card / social handle) is now caught even though it scans as 1-2 "words".
    check("burned recap/subtitle/outro/social text dropped even with no spaces",
          H(sh("TheRedWeddingwasunexpected.. butwasthetreasonunexpected?"))
          and H(sh("Ihavealwaysbeenyourson.")) and H(sh("Tuveuxteconfesser?"))
          and H(sh("THANKYOUFORWATCHING")) and H(sh("VisitourWebsite&Social Media")))
    check("clean frames (OCR noise / corner logo) still pass",
          not H(sh("")) and not H(sh("心")) and not H(sh("M")) and not H(sh("7"))
          and not H(sh("HBO")) and not H(sh("max")))

    # Fix B — black-frame floor wired into the candidate scorer
    msrc = (Path(__file__).resolve().parent.parent / "vidlore" / "clipstudio" /
            "match.py").read_text(encoding="utf-8")
    check("near-black frames dropped from the footage pool (quality floor)",
          "VIDLORE_CLIPSTUDIO_BLACK_FLOOR" in msrc and "near-black" in msrc
          and 'getattr(ps.shot, "quality", 1.0)' in msrc)

    # Fix C — recap-clue + fan-music-edit source titles excluded from the pool
    check("recap 'clues' + fan-music-edit titles excluded from footage pool",
          bool(RJ.search("Red Wedding - The clues"))
          and bool(RJ.search("Everything is Sadder with The Leftovers Music: Game Of Thrones"))
          and bool(RJ.search("GoT edit set to music")))
    check("legit scene titles still allowed",
          not RJ.search("Game of Thrones S04E10 - Tyrion kills Tywin")
          and not RJ.search("Ned Stark Execution Scene Baelor S1E9"))


# ===========================================================================
# Expert-editor quality pass (2026-07-10 audit of the 70284fa2c7 render):
# pixel corner-logo detector, still-pool purity, sub-SD selection penalty,
# audiobook gate, download quality audit, persistent breakout audit.
# ===========================================================================

def test_corner_logo_and_quality_gates():
    print("[expert-quality] pixel corner-logo detector + quality gates")
    import numpy as np
    from PIL import Image
    from vidlore.clipstudio.match import _source_corner_logo

    from vidlore.clipstudio.match import _CORNER_LOGO_CACHE

    def mkshots(tmp, n=10, logo=False, static=False, letterbox=False):
        """Smooth low-frequency 'scene' frames (real footage has sparse edges) at the detector's
        640×360 analysis size; the logo is a bounded structured patch inside the br corner."""
        _CORNER_LOGO_CACHE.clear()
        rng = np.random.RandomState(7)
        logo_patch = (np.random.RandomState(3).rand(28, 64) > 0.5).astype("float32") * 255
        shots = []
        for i in range(n):
            base = (rng.rand(5, 8) * 200 + 20).astype("uint8")
            fr = np.asarray(Image.fromarray(base, "L").resize((640, 360), Image.BILINEAR),
                            dtype="float32")
            if static:
                fr = np.full((360, 640), 90.0, dtype="float32")
            if letterbox:
                fr[0:36, :] = 0
                fr[324:360, :] = 0
            if logo:
                fr[326:354, 550:614] = logo_patch                  # bounded patch in br corner
            p = os.path.join(tmp, f"kf_{i}_{logo}_{static}_{letterbox}.png")
            Image.fromarray(fr.astype("uint8"), "L").save(p)
            shots.append(types.SimpleNamespace(keyframe_path=p))
        return shots

    with tempfile.TemporaryDirectory() as tmp:
        check("corner-logo: structured static br logo over changing scenes → 'br'",
              _source_corner_logo(mkshots(tmp, logo=True)) == "br")
        check("corner-logo: clean changing scenes → no detection",
              _source_corner_logo(mkshots(tmp)) == "")
        check("corner-logo: static-card source (centre never changes) → not this detector's call",
              _source_corner_logo(mkshots(tmp, static=True)) == "")
        check("corner-logo: flat letterbox bars are NOT a logo (needs spatial structure)",
              _source_corner_logo(mkshots(tmp, letterbox=True)) == "")

    root = Path(__file__).resolve().parents[1] / "vidlore" / "clipstudio"
    bsrc = (root / "build.py").read_text(encoding="utf-8")
    msrc = (root / "match.py").read_text(encoding="utf-8")
    osrc = (root / "orchestrate.py").read_text(encoding="utf-8")
    dsrc = (root / "download.py").read_text(encoding="utf-8")
    check("watermark-crop ORs the pixel detector (OCR-garbage logos get cropped too)",
          "_source_corner_logo" in bsrc and "pixel-static" in bsrc)
    check("match drop-mode also drops pixel-detected corner-logo sources",
          msrc.count("_source_corner_logo") >= 2)
    check("still pool excludes watermarked + sub-SD sources (stills bypass the crop)",
          "_src_wm" in osrc and "STILL_MIN_SRC_HEIGHT" in osrc)
    check("selection: sub-SD sources pay a small penalty, HD pref stays mild",
          "hd_pref -= 0.04" in msrc and "_sh < 480" in msrc)
    # audiobook uploads (narrator over a static image) never enter the pool
    from vidlore.clipstudio.discover import _REJECT_TITLE
    check("audiobook titles rejected (singular + plural)",
          bool(_REJECT_TITLE.search("Tyrion kills Tywin || Audiobook"))
          and bool(_REJECT_TITLE.search("GoT Audio Books full")))
    check("legit scene titles still pass the audiobook gate",
          not _REJECT_TITLE.search("Tyrion kills Tywin - Game of Thrones S04E10"))
    check("download logs requested vs ACTUAL height with a low-res reason",
          "LOW-RES" in dsrc and "quality_audit" in dsrc and "source_has_no_hd_stream" in dsrc)
    check("breakout audit persisted to work/breakout_audit.json with final air times",
          bsrc.count("breakout_audit.json") >= 2 and "aired_at_s" in bsrc
          and "log_lines" in bsrc)


# ===========================================================================
# Clean-copy arbitration (2026-07-10): corner-logo v3 (small/semi-transparent
# bugs, all corners), script-agnostic subtitle band, same-scene dup arbitration.
# Test cases mirror the REAL remaining defects of the 70284fa2c7 audit:
# a tiny semi-transparent 'BOC' bug, a 'SociopathMD' bottom-LEFT edge bug,
# Arabic/Turkish burned subs, and a clean-HD dup beating a dirty SD dup.
# ===========================================================================

def _smooth_scene(rng, w=640, h=360):
    """Low-frequency 'scene' frame (real footage has sparse edges, unlike noise)."""
    import numpy as np
    from PIL import Image
    base = (rng.rand(5, 8) * 200 + 20).astype("uint8")
    return np.asarray(Image.fromarray(base, "L").resize((w, h), Image.BILINEAR),
                      dtype="float32")


def test_clean_copy_arbitration():
    print("[clean-copy] corner-logo v3 + subtitle band + same-scene arbitration")
    import numpy as np
    from PIL import Image
    from vidlore.clipstudio.match import (_source_corner_logo, _shot_subtitle_band,
                                          _clean_copy_swap, _cleanliness_key, _res_tier,
                                          _CORNER_LOGO_CACHE, _SUBBAND_CACHE)
    from vidlore.clipstudio.config import load_clip_config
    from vidlore.clipstudio.models import ScriptSegment

    rng = np.random.RandomState(11)

    def mkcorner(tmp, tag, bug=None, alpha=1.0, n=10):
        """bug: (rows, cols) region at 640x360; alpha<1 = semi-transparent blend."""
        _CORNER_LOGO_CACHE.clear()
        pat = (np.random.RandomState(3).rand(360, 640) > 0.5).astype("float32") * 255
        shots = []
        for i in range(n):
            fr = _smooth_scene(rng)
            if bug is not None:
                rs, cs = bug
                fr[rs, cs] = (1 - alpha) * fr[rs, cs] + alpha * pat[rs, cs]
            p = os.path.join(tmp, f"cl_{tag}_{i}.png")
            Image.fromarray(fr.astype("uint8"), "L").save(p)
            shots.append(types.SimpleNamespace(keyframe_path=p))
        return shots

    with tempfile.TemporaryDirectory() as tmp:
        # BOC-style: SMALL, SEMI-TRANSPARENT bottom-right bug (v1/v2 missed exactly this)
        got = _source_corner_logo(mkcorner(tmp, "boc",
                                           bug=(slice(330, 352), slice(586, 624)), alpha=0.55))
        check("BOC-style small semi-transparent br bug detected", got == "br")
        # SociopathMD-style: narrow VERTICAL strip on the bottom-LEFT edge
        got = _source_corner_logo(mkcorner(tmp, "socio",
                                           bug=(slice(318, 358), slice(2, 14)), alpha=1.0))
        check("SociopathMD-style bottom-left edge bug detected", got == "bl")
        check("clean smooth scenes → no corner bug",
              _source_corner_logo(mkcorner(tmp, "clean")) == "")

        # Arabic/Turkish-style burned subs: script-agnostic stroke band (OCR reads nothing here)
        _SUBBAND_CACHE.clear()
        fr = _smooth_scene(rng, 320, 180)
        for k in range(60):                       # dense short strokes across the subtitle band
            r = 150 + (k % 3) * 6
            c = 45 + int(rng.rand() * 210)
            fr[r:r + 2, c:c + 6] = 255 if k % 2 else 5
        p_sub = os.path.join(tmp, "band_sub.png")
        Image.fromarray(fr.astype("uint8"), "L").save(p_sub)
        check("Arabic/Turkish-style stroke band flagged (script-agnostic)",
              _shot_subtitle_band(types.SimpleNamespace(keyframe_path=p_sub)))
        p_clean = os.path.join(tmp, "band_clean.png")
        Image.fromarray(_smooth_scene(rng, 320, 180).astype("uint8"), "L").save(p_clean)
        check("clean frame not flagged as subtitled",
              not _shot_subtitle_band(types.SimpleNamespace(keyframe_path=p_clean)))

    # ---- same-scene arbitration: clean HD dup must beat dirty low-res dup ----
    cfg = load_clip_config()
    seg = ScriptSegment(index=0, text="tywin dies on the privy", est_duration=4.0)

    def mkps(sid, ph, q=0.8, txt=""):
        shot = types.SimpleNamespace(index=1, phash=ph, transcript=txt, quality=q,
                                     start=10.0, end=16.0, duration=6.0, keyframe_path="")
        return types.SimpleNamespace(sid=sid, shot=shot, embed=None)

    dirty = mkps("dirty_src", "aa" * 8)
    clean = mkps("clean_src", "aa" * 8)           # identical phash = same scene
    other = mkps("other_src", "55" * 8, txt="completely different scene words entirely")
    src_dirty = {"dirty_src": {"corner": "br", "subs": 0.4},
                 "clean_src": {"corner": "", "subs": 0.0},
                 "other_src": {"corner": "", "subs": 0.0}}
    src_height = {"dirty_src": 360, "clean_src": 1080, "other_src": 1080}
    scored = [(0.80, 0.0, {}, dirty), (0.79, 0.0, {}, clean), (0.95, 0.0, {}, other)]
    best = (0.80, 0.80, dirty, types.SimpleNamespace(source_id="dirty_src", shot_index=1,
                                                     score=0.80, in_point=10, out_point=14,
                                                     signals={}, segment_index=0))
    new_best, note = _clean_copy_swap(seg, best, scored, src_dirty, src_height, cfg)
    check("clean 1080p duplicate beats dirty watermarked 360p duplicate",
          new_best[2].sid == "clean_src" and note is not None)
    # rule 5: the higher-scoring but DIFFERENT-scene HD shot must NOT hijack the beat
    check("different-scene HD footage never replaces the exact relevant pick",
          new_best[2].sid != "other_src")
    # relevance-first: a same-scene copy far below in relevance never swaps in
    scored2 = [(0.80, 0.0, {}, dirty), (0.70, 0.0, {}, clean)]
    nb2, note2 = _clean_copy_swap(seg, best, scored2, src_dirty, src_height, cfg)
    check("same-scene copy outside the relevance window does not swap",
          nb2[2].sid == "dirty_src" and note2 is None)

    # BEHAVIORAL regression (decoded-quality-before-resolution tuple): a SOFT 1080p best is swapped
    # for a same-scene SHARP 720p (codec-neutral decoded quality wins), and the early-return uses the
    # correct tuple fields — a PRISTINE clean 1080p best is never swapped away.
    soft1080 = mkps("soft1080", "cc" * 8, q=0.35)
    sharp720 = mkps("sharp720", "cc" * 8, q=0.62)      # identical phash = same scene
    sq_dirty = {"soft1080": {"corner": "", "subs": 0.0}, "sharp720": {"corner": "", "subs": 0.0}}
    sq_h = {"soft1080": 1080, "sharp720": 720}
    best_soft = (0.80, 0.80, soft1080, types.SimpleNamespace(source_id="soft1080", shot_index=1,
                 score=0.80, in_point=10, out_point=14, signals={}, segment_index=0))
    scored_sq = [(0.80, 0.0, {}, soft1080), (0.79, 0.0, {}, sharp720)]
    nb3, note3 = _clean_copy_swap(seg, best_soft, scored_sq, sq_dirty, sq_h, cfg)
    check("SOFT 1080p best is swapped for a same-scene SHARP 720p (decoded quality > resolution)",
          nb3[2].sid == "sharp720" and note3 is not None)
    # a PRISTINE clean 1080p best (q=1.0) early-outs and is never downgraded to a lower copy
    prist = mkps("prist1080", "dd" * 8, q=1.0)
    lowcopy = mkps("low720", "dd" * 8, q=0.9)
    pr_dirty = {"prist1080": {"corner": "", "subs": 0.0}, "low720": {"corner": "", "subs": 0.0}}
    pr_h = {"prist1080": 1080, "low720": 720}
    best_pr = (0.80, 0.80, prist, types.SimpleNamespace(source_id="prist1080", shot_index=1,
               score=0.80, in_point=10, out_point=14, signals={}, segment_index=0))
    nb4, note4 = _clean_copy_swap(seg, best_pr, [(0.80, 0.0, {}, prist), (0.80, 0.0, {}, lowcopy)],
                                  pr_dirty, pr_h, cfg)
    check("pristine clean 1080p best early-outs (never downgraded)", nb4[2].sid == "prist1080")
    from vidlore.clipstudio.match import _cleanliness_key as _ck2
    kp = _ck2("prist1080", prist.shot, pr_dirty, pr_h)
    check("early-return tests the intended fields (clean, q_bin==1.0, res tier 1080)",
          kp[0] == 0 and kp[1] == 0 and kp[2] <= -1.0 and kp[3] == -3)
    # ordering sanity of the cleanliness key itself
    k_dirty = _cleanliness_key("dirty_src", dirty.shot, src_dirty, src_height)
    k_clean = _cleanliness_key("clean_src", clean.shot, src_dirty, src_height)
    check("cleanliness key orders watermark > subs > decoded-quality > resolution",
          k_clean < k_dirty and _res_tier(1080) == 3 and _res_tier(718) == 2
          and _res_tier(360) == 0)
    # CODEC-AWARE: a genuinely SHARP 720p beats a SOFT/upscaled 1080p (decoded quality, not raw
    # container resolution/bitrate). A REAL sharp 1080p still beats the sharp 720p.
    soft_1080 = types.SimpleNamespace(quality=0.35, ocr_text="", subs_flag=0)
    sharp_720 = types.SimpleNamespace(quality=0.60, ocr_text="", subs_flag=0)
    real_1080 = types.SimpleNamespace(quality=0.72, ocr_text="", subs_flag=0)
    sh = {"soft1080": 1080, "sharp720": 720, "real1080": 1080}
    k_soft = _cleanliness_key("soft1080", soft_1080, {}, sh)
    k_sharp = _cleanliness_key("sharp720", sharp_720, {}, sh)
    k_real = _cleanliness_key("real1080", real_1080, {}, sh)
    check("sharp 720p beats SOFT/upscaled 1080p (codec-neutral decoded quality)", k_sharp < k_soft)
    check("REAL sharp 1080p still beats the sharp 720p", k_real < k_sharp)

    # wiring: gates + breakout crop + still-pool band check
    root = Path(__file__).resolve().parents[1] / "vidlore" / "clipstudio"
    msrc2 = (root / "match.py").read_text(encoding="utf-8")
    bsrc2 = (root / "build.py").read_text(encoding="utf-8")
    isrc2 = (root / "image_fallback.py").read_text(encoding="utf-8")
    check("subtitle-band gate wired into _score_pool",
          "_shot_subtitle_band(ps.shot)" in msrc2 and "SUBBAND_GATE" in msrc2)
    check("clean-copy arbitration wired into the selection loop",
          "_clean_copy_swap(seg, best, scored" in msrc2 and "CLEAN_COPY_GATE" in msrc2)
    check("breakouts gate subtitle bands + crop corner-bug sources",
          "_texty9" in bsrc2 and "crop_corner=_bk_corner" in bsrc2)
    check("still pool rejects visually-subtitled keyframes",
          "_shot_subtitle_band" in isrc2)


# ===========================================================================
# Multi-frame shot sampling (2026-07-10): 3 frames/shot at index time persist
# subs_flag / luma / corner masks so gates catch INTERMITTENT defects the
# keyframe instant misses (the t=258 Turkish sub aired from a clean-keyframe
# shot). Cases: mid-shot subtitle, late-appearing corner bug, valid dark
# cinematic scene vs unreadable black, and persisted-flag reuse without IO.
# ===========================================================================

def test_multiframe_shot_flags():
    print("[multi-frame] index-time flags: mid-shot subs, late bugs, luma, persistence")
    import numpy as np
    from vidlore.clipstudio.index import _flags_from_frames, _mask_from_hex, _mask_to_hex
    from vidlore.clipstudio.match import (_shot_subtitle_band, _shot_unreadable,
                                          _source_corner_logo, _CORNER_LOGO_CACHE,
                                          _SUBBAND_CACHE)
    from PIL import Image

    rng = np.random.RandomState(19)

    def scene():
        base = (rng.rand(5, 8) * 200 + 20).astype("uint8")
        return np.asarray(Image.fromarray(base, "L").resize((640, 360), Image.BILINEAR),
                          dtype="float32")

    def with_sub(fr):
        fr = fr.copy()
        r2 = rng.rand(220)
        for k in range(60):                     # stroke band at 320x180-scale rows 150-168 → ×2
            r = 300 + (k % 3) * 12
            c = 90 + int(r2[k] * 420)
            fr[r:r + 4, c:c + 12] = 255 if k % 2 else 5
        return fr

    def with_bug(fr):
        fr = fr.copy()
        pat = (np.random.RandomState(3).rand(28, 64) > 0.5).astype("float32") * 255
        fr[326:354, 550:614] = pat
        return fr

    # (a) clean keyframe, subtitle appears MID-SHOT (start sample carries it)
    f = _flags_from_frames([with_sub(scene()), scene(), scene()])
    check("mid-shot subtitle flagged although the (mid) keyframe instant is clean",
          f["subs_flag"] == 1 and f["text_conf"] > 2.0)
    f2 = _flags_from_frames([scene(), scene(), scene()])
    check("all-clean samples → subs_flag=0", f2["subs_flag"] == 0)
    check("no frames → sentinel -1 (gates fall back to keyframe heuristics)",
          _flags_from_frames([])["subs_flag"] == -1)

    # persisted-first: flag wins over the keyframe file (nonexistent path proves ZERO IO)
    _SUBBAND_CACHE.clear()
    sub_shot = types.SimpleNamespace(subs_flag=1, keyframe_path="/nonexistent/kf.jpg")
    clean_shot = types.SimpleNamespace(subs_flag=0, keyframe_path="/nonexistent/kf2.jpg")
    check("persisted subs_flag reused with no keyframe IO (file doesn't exist)",
          _shot_subtitle_band(sub_shot) and not _shot_subtitle_band(clean_shot))

    # (b) corner bug appears only in the LATER shots (intermittent across the source)
    _CORNER_LOGO_CACHE.clear()
    def shot_of(frames, i):
        fl = _flags_from_frames(frames)
        return types.SimpleNamespace(index=i, keyframe_path=f"/x/kf_{i}.jpg",
                                     subs_flag=fl["subs_flag"], corner_masks=fl["corner_masks"],
                                     luma_avg=fl["luma_avg"], luma_hi=fl["luma_hi"])
    shots = [shot_of([scene(), scene(), scene()], i) for i in range(6)] + \
            [shot_of([scene(), with_bug(scene()), with_bug(scene())], 6 + i) for i in range(4)]
    check("corner bug appearing only in later shots detected from persisted masks",
          _source_corner_logo(shots) == "br")
    _CORNER_LOGO_CACHE.clear()
    clean_shots = [shot_of([scene(), scene(), scene()], i) for i in range(10)]
    check("clean source → no corner bug from persisted masks",
          _source_corner_logo(clean_shots) == "")
    # mask hex round-trip
    m = np.zeros((11, 29), dtype=bool); m[3:6, 4:9] = True
    check("corner mask hex round-trip", (_mask_from_hex(_mask_to_hex(m)) == m).all())

    # (c) valid dark cinematic scene vs unreadable black (multi-frame luma). The candle case
    # sits BELOW the avg threshold (~9.5 mean) so it exercises the luma_hi discriminator:
    # a lit face/candle covering ~3.5% of the frame keeps the 99.8th-percentile pixel bright
    # (mirrors the measured real privy/crossbow shots: avg 12-14, hi 127-147).
    candle = np.full((360, 640), 3.0, dtype="float32")                   # dark base
    candle[130:190, 280:415] = 190.0                                     # lit region ~3.5%
    murk = np.full((360, 640), 5.0, dtype="float32")
    murk += rng.rand(360, 640) * 6.0                                     # near-black noise
    fc = _flags_from_frames([candle, candle, candle])
    fm = _flags_from_frames([murk, murk, murk])
    cand_shot = types.SimpleNamespace(luma_avg=fc["luma_avg"], luma_hi=fc["luma_hi"])
    murk_shot = types.SimpleNamespace(luma_avg=fm["luma_avg"], luma_hi=fm["luma_hi"])
    check("valid dark cinematic scene (bright highlights) NOT gated",
          not _shot_unreadable(cand_shot))
    check("unreadable near-black shot gated across its whole span",
          _shot_unreadable(murk_shot))
    old_shot = types.SimpleNamespace(luma_avg=-1.0, luma_hi=-1.0)
    check("old index without flags fails OPEN (never gated here)",
          not _shot_unreadable(old_shot))

    # wiring: index integration + pool gate + stills
    root = Path(__file__).resolve().parents[1] / "vidlore" / "clipstudio"
    isrc = (root / "index.py").read_text(encoding="utf-8")
    msrc = (root / "match.py").read_text(encoding="utf-8")
    fsrc = (root / "image_fallback.py").read_text(encoding="utf-8")
    check("index_source computes multi-frame flags (env-gated)",
          "compute_shot_flags(path, shots" in isrc and "MULTIFRAME_FLAGS" in isrc)
    check("_score_pool gates unreadable-dark shots via persisted luma",
          "_shot_unreadable(ps.shot)" in msrc)
    check("still pickers gate unreadable-dark shots",
          fsrc.count("_shot_unreadable(shot)") >= 2)
    check("corner detector consumes persisted masks before touching keyframes",
          "_mask_from_hex" in msrc)




# ===========================================================================
# Cut-window flag validation (2026-07-10 finalization): the rendered cut can
# extend past the selected shot (min_clip padding, playhead walks, breakout
# line-extension) — validate the ENTIRE final window against every overlapping
# shot's persisted flags. Shorten > fallback-to-relevant-alternate > keep;
# never swap the exact scene for unrelated footage.
# ===========================================================================

def test_cut_window_validation():
    print("[window-qc] final-cut window validation across shot boundaries")
    from vidlore.clipstudio.match import clean_cut_window, _partial_corner_shots
    from vidlore.clipstudio.index import _sample_times, _flags_from_frames
    import numpy as np
    from PIL import Image

    def shot(i, s, e, subs=0, la=60.0, lh=200.0, masks=None):
        return types.SimpleNamespace(index=i, start=s, end=e, subs_flag=subs,
                                     luma_avg=la, luma_hi=lh, corner_masks=masks or {},
                                     ocr_text="", ocr_names=[], keyframe_path="")

    MINL = 1.2
    # (1) clean selected shot; PADDED extension overlaps a subtitle-flagged adjacent shot
    A, B = shot(0, 0.0, 1.4), shot(1, 1.4, 4.0, subs=1)
    n0, n1, act, why = clean_cut_window([A, B], 0.0, 2.4, MINL, anchor=(0.0, 1.4))
    check("extension into subtitle-flagged neighbour → shortened",
          act == "shortened" and "subs" in why)
    check("shortening preserves the exact-scene cut (stays inside the chosen shot)",
          n0 >= -1e-6 and n1 <= 1.4 + 1e-6 and (n1 - n0) >= MINL - 1e-6)
    # (2) corner-logo evidence only in the extension window (fresh CLEAN neighbour — the
    # partial-corner dict alone must dirty it)
    B_logo = shot(1, 1.4, 4.0)
    _, _, act2, why2 = clean_cut_window([A, B_logo], 0.0, 2.4, MINL, anchor=(0.0, 1.4),
                                        partial_corner={1: "br"})
    check("corner-logo in the extension window → shortened with logo reason",
          act2 == "shortened" and "logo-br" in why2)
    # (3) unreadable-black adjacent shot
    C = shot(1, 1.4, 4.0, la=5.0, lh=30.0)
    _, _, act3, why3 = clean_cut_window([A, C], 0.0, 2.4, MINL, anchor=(0.0, 1.4))
    check("unreadable-black neighbour → shortened with unreadable reason",
          act3 == "shortened" and "unreadable" in why3)
    # (4) window entirely clean → untouched
    D = shot(1, 1.4, 4.0)
    r = clean_cut_window([A, D], 0.0, 2.4, MINL, anchor=(0.0, 1.4))
    check("clean window passes unchanged", r[2] == "ok" and r[0] == 0.0 and r[1] == 2.4)
    # (5) NO clean anchor window → rejected — a clean NON-anchor span must never be returned
    #     (that would swap the exact scene for a different moment; fallback flow handles it)
    A_dirty = shot(0, 0.0, 1.4, subs=1)
    _, _, act5, _ = clean_cut_window([A_dirty, D], 0.0, 2.4, MINL, anchor=(0.0, 1.4))
    check("anchor itself dirty → rejected (clean non-anchor span NOT hijacked)",
          act5 == "rejected")
    # old index (sentinels) → every shot reads clean → fail-open
    old = types.SimpleNamespace(index=0, start=0.0, end=4.0, subs_flag=-1, luma_avg=-1.0,
                                luma_hi=-1.0, corner_masks={}, ocr_text="", ocr_names=[],
                                keyframe_path="")
    check("old index without flags fails open", clean_cut_window([old], 0, 2.4, MINL)[2] == "ok")

    # (6) short <2s shots get FIVE distinct sample timestamps (never mid-only)
    ts_short = _sample_times(10.0, 11.4)
    ts_long = _sample_times(10.0, 16.0)
    check("<2s shot sampled at 5 DISTINCT timestamps",
          len(ts_short) == 5 and len(set(round(t, 4) for t in ts_short)) == 5
          and all(10.0 <= t <= 11.4 for t in ts_short))
    check(">=2s shot keeps 3 samples", len(ts_long) == 3)

    # partial-corner evidence: an intermittent bug on a MINORITY of shots flags those shots only
    rng = np.random.RandomState(23)

    def scene640():
        base = (rng.rand(5, 8) * 200 + 20).astype("uint8")
        return np.asarray(Image.fromarray(base, "L").resize((640, 360), Image.BILINEAR),
                          dtype="float32")

    pat = (np.random.RandomState(3).rand(28, 64) > 0.5).astype("float32") * 255

    def bugged(fr):
        fr = fr.copy()
        fr[326:354, 550:614] = pat
        return fr

    def mk(i, dirty):
        fl = _flags_from_frames([bugged(scene640()) if dirty else scene640()
                                 for _ in range(3)])
        return types.SimpleNamespace(index=i, start=float(i), end=float(i + 1),
                                     subs_flag=fl["subs_flag"], corner_masks=fl["corner_masks"],
                                     luma_avg=fl["luma_avg"], luma_hi=fl["luma_hi"],
                                     ocr_text="", ocr_names=[], keyframe_path=f"/x/pc_{i}.jpg",
                                     phash=f"{(i * 2654435761) % (2 ** 64):016x}")
    shots_pc = [mk(i, dirty=(i >= 9)) for i in range(12)]      # bug only on the last 3 (25%<thr)
    pc = _partial_corner_shots(shots_pc)
    check("partial-corner: exactly the bugged minority shots flagged",
          set(pc.keys()) == {9, 10, 11} and set(pc.values()) == {"br"})

    # (7) upgrade tool counts clean subs_flag=0 as ALREADY upgraded (falsy-zero bug fixed)
    tool = (Path(__file__).resolve().parents[1] / "tools" / "upgrade_index_flags.py").read_text(
        encoding="utf-8")
    check("upgrade tool skip-logic is falsy-zero safe",
          "all(_flagged(sh) for sh in shots)" in tool
          and 'subs_flag", -1) or -1' not in tool)

    # (8) all four paths wired: normal selection, verifier repair, still pool, breakouts (+cut)
    root = Path(__file__).resolve().parents[1] / "vidlore" / "clipstudio"
    msrc = (root / "match.py").read_text(encoding="utf-8")
    vsrc = (root / "verify.py").read_text(encoding="utf-8")
    csrc = (root / "cut.py").read_text(encoding="utf-8")
    bsrc = (root / "build.py").read_text(encoding="utf-8")
    osrc = (root / "orchestrate.py").read_text(encoding="utf-8")
    isrc = (root / "index.py").read_text(encoding="utf-8")
    check("normal selections window-validated (+fallback to relevance-ranked alternates)",
          "_validate_cand_window" in msrc and "window-qc: fallback" in msrc
          and "window-qc: summary" in msrc)
    check("verifier promotions window-validated",
          "window-qc: rejected verify-promotion" in vsrc
          and "window-qc: shortened verify-promotion" in vsrc)
    check("cut padding never bleeds into a dirty neighbour",
          "clean_cut_window" in csrc and "don't pad into dirt" in csrc)
    check("build render windows + breakout extension window-validated",
          "_wqc_render_start" in bsrc and "window-qc: rejected breakout" in bsrc)
    check("partial-corner deliberately NOT a production dirty-reason (candle-sconce FP)",
          "candle sconce" in msrc and "_pcs_o" not in osrc
          and "partial_corner=_pcs" not in bsrc)
    check("stale p95 naming gone from luma implementation",
          "p95s" not in isrc and "luma_hi" in isrc)


def test_wqc_moment_preservation():
    print("[window-qc] exact-moment preservation — policy-anchored PRODUCTION validation paths")
    from vidlore.clipstudio.match import (validate_candidate_window, wqc_arbitrate_selection,
                                          wqc_moment_policy)
    from vidlore.clipstudio.build import wqc_render_window, wqc_render_log_line
    from vidlore.clipstudio.models import ClipCandidate
    from vidlore.clipstudio.config import ClipConfig
    cfg = ClipConfig()

    def shot(i, s, e, subs=0):
        return types.SimpleNamespace(index=i, start=s, end=e, subs_flag=subs,
                                     luma_avg=60.0, luma_hi=200.0, corner_masks={},
                                     ocr_text="", ocr_names=[], keyframe_path="")

    def seg_of(policy, quote="", idx=3):
        return types.SimpleNamespace(index=idx, text="Tywin tells Tyrion the truth",
                                     quote=quote, required_kind="", required_entity="",
                                     is_specific_claim=False, visual_policy=policy,
                                     breakout_candidate=bool(quote))

    def cand_of(sid="srcA", si=0, inp=2.0, outp=8.0, score=0.8):
        return ClipCandidate(segment_index=3, source_id=sid, shot_index=si, score=score,
                             in_point=inp, out_point=outp)

    # (0) policy resolution: exact/character/quote → moment-locked; filler/abstract → generic
    check("policy: exact_scene / character_specific / dialogue-quote → moment-locked",
          wqc_moment_policy(seg_of("exact_scene")) == "exact"
          and wqc_moment_policy(seg_of("character_specific")) == "exact"
          and wqc_moment_policy(seg_of("generic_filler", quote="I am your son")) == "exact"
          and wqc_moment_policy(None) == "exact")
    check("policy: generic_filler / abstract_effect → may shift within the shot",
          wqc_moment_policy(seg_of("generic_filler")) == "generic"
          and wqc_moment_policy(seg_of("abstract_effect")) == "generic")

    # Geometry: candidate [2,8] (moment mid=5); head of the window dirty (subs shot to 6.5),
    # a clean span [6.5,8] exists — big enough to air, but it is a DIFFERENT moment (25%
    # overlap, no midpoint). Selected shot spans the whole window.
    D, C = shot(0, 0.0, 6.5, subs=1), shot(1, 6.5, 10.0)
    S_sel = shot(9, 0.0, 10.0)

    # (1) exact beat: same-shot clean span that loses the moment → REJECTED, cand untouched
    c1 = cand_of()
    act1, why1, m1 = validate_candidate_window(c1, S_sel, [D, C], cfg, seg_of("exact_scene"))
    check("exact beat NEVER shifts to a different moment in the same shot",
          act1 == "rejected" and why1.startswith("moment-lost")
          and c1.in_point == 2.0 and c1.out_point == 8.0
          and m1["policy"] == "exact" and m1["preserved"] is False)
    # (2) SAME inputs, generic beat → whole-shot anchor allows the shift (mutates in place)
    c2 = cand_of()
    act2, _, m2 = validate_candidate_window(c2, S_sel, [D, C], cfg, seg_of("generic_filler"))
    check("generic filler MAY shift within the same shot (same inputs, other policy)",
          act2 == "shortened" and c2.in_point >= 6.5 - 1e-6
          and m2["policy"] == "generic" and m2["preserved"] is True)
    # (3) exact beat may still SHORTEN when the original moment survives
    D1, C1 = shot(0, 0.0, 3.0, subs=1), shot(1, 3.0, 10.0)
    c3 = cand_of()
    act3, _, m3 = validate_candidate_window(c3, S_sel, [D1, C1], cfg, seg_of("exact_scene"))
    check("exact beat shortens ONLY around its own moment (midpoint kept)",
          act3 == "shortened" and c3.in_point <= 5.0 <= c3.out_point
          and m3["preserved"] is True)

    # (4) selection-level arbitration (REAL production function): contaminated exact beat
    # falls back to the first relevance-ranked alternate whose window validates ...
    by_src = {"srcA": [D, C], "srcB": [shot(0, 0.0, 10.0)], "srcC": [shot(0, 0.0, 10.0, subs=1)]}
    psA = types.SimpleNamespace(shot=S_sel, sid="srcA")
    psB = types.SimpleNamespace(shot=shot(0, 0.0, 10.0), sid="srcB")
    psC = types.SimpleNamespace(shot=shot(0, 0.0, 10.0, subs=1), sid="srcC")
    ps_by_key = {("srcB", 0): psB, ("srcC", 0): psC}
    altB = cand_of(sid="srcB", si=0, inp=1.0, outp=4.0, score=0.7)
    stats = {}
    best, alts = wqc_arbitrate_selection((0.9, 0.8, psA, cand_of()), [altB], by_src,
                                         ps_by_key, cfg, seg_of("exact_scene"), stats=stats)
    check("exact beat w/ contaminated moment → falls back to a VALID alternate",
          best[3] is altB and stats.get("fallback") == 1 and altB not in alts)
    # ... and stays (dirty) on the ORIGINAL when no alternate validates
    altC = cand_of(sid="srcC", si=0, inp=1.0, outp=4.0, score=0.7)
    cA, stats2 = cand_of(), {}
    best2, _ = wqc_arbitrate_selection((0.9, 0.8, psA, cA), [altC], by_src,
                                       ps_by_key, cfg, seg_of("exact_scene"), stats=stats2)
    check("exact beat w/ NO valid alternate → kept-dirty on the original moment",
          best2[3] is cA and stats2.get("kept-dirty") == 1
          and cA.in_point == 2.0 and cA.out_point == 8.0)

    # (5) build render-window (REAL production function): neighbourhood shift obeys policy.
    # `moment` = the chosen window's OWN [in, out]; here it fills the whole render window.
    # Window [10,14] dirty; the only clean span keeps just 0.5s of the moment.
    sh_b = [shot(0, 9.0, 13.5, subs=1), shot(1, 13.5, 30.0)]
    ns_g, act_g, _, mg = wqc_render_window(sh_b, 10.0, 4.0, seg_of("generic_filler"),
                                           (10.0, 14.0))
    ns_e, act_e, _, me = wqc_render_window(sh_b, 10.0, 4.0, seg_of("exact_scene"),
                                           (10.0, 14.0))
    check("build shift: generic beat may slide to the clean neighbour span",
          act_g == "shifted" and ns_g >= 13.4)
    # kept-dirty = cleanliness failed, NOT moment lost: the unchanged final window still
    # contains the candidate, so moment-preserved reports the honest yes
    check("build shift: exact beat REFUSES a moment-losing slide → kept-dirty at the original",
          act_e == "kept-dirty" and ns_e == 10.0 and me["preserved"] is True)
    kd_line = wqc_render_log_line(act_e, me, "subs(shot 0)")
    check("audit log: kept-dirty exact beat that still airs its candidate says preserved=yes",
          "moment-preserved=yes" in kd_line and "action=kept-dirty" in kd_line
          and "candidate=[10.0-14.0]" in kd_line and "render-request=[10.0-14.0]" in kd_line)
    # playhead-walk fill (NO selected moment) → even an exact beat may take the clean span
    ns_w, act_w, _, mw = wqc_render_window(sh_b, 10.0, 4.0, seg_of("exact_scene"), None)
    check("build shift: walk-fill window (no selected moment) may slide even on exact beats",
          act_w == "shifted" and ns_w >= 13.4)
    wf_line = wqc_render_log_line(act_w, mw, "subs(shot 0)")
    check("audit log: walk-fill (no candidate) logs moment-preserved=n-a, never yes/no",
          mw["preserved"] is None and "moment-preserved=n-a" in wf_line
          and "candidate=none" in wf_line and "action=shifted" in wf_line)
    g_line = wqc_render_log_line(act_g, mg, "subs(shot 0)")
    check("audit log: generic beat (guarantee not applicable) also logs n-a",
          mg["preserved"] is None and "moment-preserved=n-a" in g_line)
    # a moment-KEEPING shift is still allowed for exact beats (dirt only at the head)
    sh_b2 = [shot(0, 9.5, 11.0, subs=1), shot(1, 11.0, 30.0)]
    ns_e2, act_e2, _, me2 = wqc_render_window(sh_b2, 10.0, 4.0, seg_of("exact_scene"),
                                              (10.0, 14.0))
    check("build shift: exact beat shifts when the aired window still shows the moment",
          act_e2 == "shifted" and abs(ns_e2 - 11.0) < 0.2 and me2["preserved"] is True)
    # the AIRED slice (need-length) must show the moment — not just the longer clean span.
    # Clean span [6,13] contains the moment mid (12) but a span-start aired slice [6,10]
    # would NOT: the aired start must slide forward inside the clean span to cover it.
    sh_b3 = [shot(0, 5.0, 6.0, subs=1), shot(1, 6.0, 13.0), shot(2, 13.0, 14.0, subs=1),
             shot(3, 14.0, 30.0)]
    ns_e3, act_e3, _, me3 = wqc_render_window(sh_b3, 10.0, 4.0, seg_of("exact_scene"),
                                              (10.0, 14.0))
    check("build shift: exact beat positions the AIRED slice on the moment inside the span",
          act_e3 == "shifted" and ns_e3 <= 12.0 <= ns_e3 + 4.0 and me3["preserved"] is True)
    # REGRESSION (observed beat 127): a SHORT chosen moment [93.9-95.3] padded to a 4.4s
    # render window [93.9-98.3]; subs sit in the PADDING. Preserving the padded window
    # refused the clean shift — preserving the chosen MOMENT takes the clean span before
    # it and keeps the full moment on screen.
    sh_b4 = [shot(0, 85.0, 95.3), shot(1, 95.3, 96.5, subs=1), shot(2, 96.5, 97.0),
             shot(3, 97.0, 98.5, subs=1), shot(4, 98.5, 110.0)]
    ns_e4, act_e4, _, me4 = wqc_render_window(sh_b4, 93.9, 4.4, seg_of("exact_scene"),
                                              (93.9, 95.3))
    check("build shift: padding is NOT the moment — short chosen moment shifts clean",
          act_e4 == "shifted" and ns_e4 <= 93.9 and ns_e4 + 4.4 >= 95.25
          and me4["preserved"] is True)
    e4_line = wqc_render_log_line(act_e4, me4, "subs(shot 1)")
    check("audit log: exact shift logs candidate and render-request as DISTINCT fields",
          "candidate=[93.9-95.3]" in e4_line and "render-request=[93.9-98.3]" in e4_line
          and "moment-preserved=yes" in e4_line and "action=shifted" in e4_line
          and me4["candidate"] == (93.9, 95.3)
          and abs(me4["render_request"][0] - 93.9) < 1e-6
          and abs(me4["render_request"][1] - 98.3) < 1e-6)

    # (5b) pre-commit adversarial-review fixes for the shift machinery.
    # The widened neighbourhood search must NOT "shift" past the indexed extent:
    # clean_cut_window counts un-indexed time as clean, so an unclamped search proposed a
    # span past the last shot (≈ source end) and the caller's duration clamp dragged the
    # aired window straight back into the dirt while the log claimed a clean shift.
    sh_b5 = [shot(0, 0.0, 9.0, subs=1), shot(1, 9.0, 10.5)]     # index ends at 10.5
    ns_x, act_x, _, _ = wqc_render_window(sh_b5, 8.0, 4.0, seg_of("generic_filler"), None)
    check("build shift: widened search clamped to the indexed extent (no phantom-clean tail)",
          act_x == "kept-dirty" and ns_x == 8.0)
    # Generic/walk-fill shifts: the AIRED need-slice — not just the validated span — must
    # overlap the original window. Span [96,101] vs orig [100,104]: airing the span head
    # [96,100] misses the original window entirely; the slice must sit at [97,101].
    sh_b6 = [shot(0, 96.0, 101.0), shot(1, 101.0, 112.0, subs=1)]
    ns_y, act_y, _, my = wqc_render_window(sh_b6, 100.0, 4.0, seg_of("generic_filler"), None)
    check("build shift: generic AIRED slice positioned to overlap the original window",
          act_y == "shifted" and abs(ns_y - 97.0) < 1e-6 and my["final"][1] > 100.0 + 1e-6)
    # Module-level callers may pass an empty shot list → explicit fail-open.
    ns_z, act_z, _, _ = wqc_render_window([], 5.0, 4.0, seg_of("exact_scene"), (5.0, 9.0))
    check("build shift: empty shot list fails open", act_z == "ok" and ns_z == 5.0)

    # (5c) duration-clamp mirror: indexed shot ends can exceed the integer-rounded source
    # duration metadata, so a shift into the index's tail gets dragged back by the caller's
    # duration clamp — `final=` must describe what actually AIRS, not the phantom shift.
    # Decision-neutral: the caller applies the identical clamp to whatever is returned.
    sh_b7 = [shot(0, 80.0, 91.9, subs=1), shot(1, 91.9, 95.9)]     # index ends past dur=95
    ns_d, act_d, why_d, md = wqc_render_window(sh_b7, 91.0, 4.0, seg_of("generic_filler"),
                                               None, 95.0)
    check("audit log: duration-clamped evaporated shift reports kept-dirty (final = request)",
          act_d == "kept-dirty" and ns_d == 91.0 and md["final"] == (91.0, 95.0)
          and "duration-clamped" in why_d)
    ns_d2, act_d2, _, md2 = wqc_render_window(sh_b7, 91.0, 4.0, seg_of("exact_scene"),
                                              (92.0, 94.0), 95.0)
    check("audit log: duration-clamped exact beat still reports the honest moment answer",
          act_d2 == "kept-dirty" and md2["preserved"] is True
          and "moment-preserved=yes" in wqc_render_log_line(act_d2, md2, "x"))
    ns_d3, act_d3, _, md3 = wqc_render_window(sh_b7, 90.5, 4.0, seg_of("generic_filler"),
                                              None, 95.0)
    check("audit log: partially clamped shift logs the clamped aired window as final",
          act_d3 == "shifted" and abs(ns_d3 - 91.0) < 1e-6
          and abs(md3["final"][1] - 95.0) < 1e-6)

    # (6) verifier promotion path uses the SAME production validator (stub shots tolerated)
    alt_v = cand_of(sid="srcA", si=0)
    ashot_bare = types.SimpleNamespace(index=0)          # verify stubs may lack start/end
    act_v, why_v, _ = validate_candidate_window(alt_v, ashot_bare, [D, C], cfg,
                                                seg_of("exact_scene"))
    check("verify-promotion path: contaminated exact moment rejected (alternate skipped)",
          act_v == "rejected" and why_v.startswith("moment-lost"))
    act_v2, _, _ = validate_candidate_window(cand_of(), ashot_bare, [D, C], cfg,
                                             seg_of("generic_filler"))
    check("verify-promotion path: span-less stub shot fails open to the window anchor",
          act_v2 in ("shortened", "rejected"))

    # (7) wiring: every path routes through the policy-aware production functions
    root = Path(__file__).resolve().parents[1] / "vidlore" / "clipstudio"
    msrc = (root / "match.py").read_text(encoding="utf-8")
    vsrc = (root / "verify.py").read_text(encoding="utf-8")
    bsrc = (root / "build.py").read_text(encoding="utf-8")
    csrc = (root / "cut.py").read_text(encoding="utf-8")
    check("match selections arbitrate via wqc_arbitrate_selection (policy audit logs)",
          "wqc_arbitrate_selection(" in msrc and "moment-preserved=" in msrc)
    check("verifier promotions use validate_candidate_window",
          "validate_candidate_window(" in vsrc)
    check("build render shifts use wqc_render_window with the beat's seg + chosen moment",
          "wqc_render_window(shots, start, need, seg, moment, _sdur)" in bsrc
          and "_wqc_render_start(sid, start, src_need, seg, _wqc_moment)" in bsrc
          and "_wqc_moment = (in_p, float(chosen_w[2]))" in bsrc)
    check("breakouts reject, never shift (quote-locked = exact by definition)",
          "never shifted; trying next candidate" in bsrc)
    check("cut padding never relocates the moment (refuse-the-pad only)",
          "never relocates the aired moment" in csrc)


def test_gemini_timeout_hardening():
    print("[llm] gemini hard timeout — env parsing never crashes the client build")
    from vidlore.clipstudio.llm import _gemini_http_options
    key = "VIDLORE_GEMINI_TIMEOUT_SEC"
    old = os.environ.get(key)
    try:
        got = {}
        for val in ("", "abc", "nan", "inf", "-5", "0", "30", "1.5"):
            os.environ[key] = val
            got[val] = _gemini_http_options().timeout
        check("garbage / nan / inf / non-positive env values fall back to the 120s default",
              all(got[v] == 120000 for v in ("", "abc", "nan", "inf", "-5", "0")))
        check("valid env values convert seconds → SDK milliseconds",
              got["30"] == 30000 and got["1.5"] == 1500)
    finally:
        if old is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old


def test_breakout_atomic_composition():
    print("[breakout] atomic composition — clip-map/bidx/captions/audit stay in lockstep")
    import tempfile
    import os as _os_bq
    # these are VISUAL-dimension QA tests on synthetic silent clips; the final-mix AUDIO QA (a
    # separate real gate that requires audible dialogue) is exercised via the fresh render + its own
    # unit checks, so disable it here so silent test fixtures don't trip the fail-closed audio path.
    _os_bq.environ["VIDLORE_CLIPSTUDIO_BREAKOUT_AUDIO_QA"] = "0"
    import subprocess as _sp
    import vidlore.clipstudio.build as B
    from vidlore.tts import NarratedScene
    from vidlore.script_gen import Scene
    from vidlore.clipstudio.models import ScriptSegment
    from vidlore.clipstudio.build import (_compose_breakout_state, _compose_breakouts,
                                          _validate_breakout_assembly, _finalize_breakout_audit,
                                          _postrender_breakout_qa)

    tmp = Path(tempfile.mkdtemp(prefix="bktest_"))
    FF = B.ffmpeg_exe()

    def _mk_mp4(p, d=2.0, black=False):
        col = "black" if black else "white"
        _sp.run([FF, "-y", "-f", "lavfi", "-i", f"color=c={col}:s=160x90:d={d}",
                 "-pix_fmt", "yuv420p", "-r", "10", str(p)], capture_output=True)

    def _mk_wav(p, d=2.0):
        _sp.run([FF, "-y", "-f", "lavfi", "-i", f"sine=frequency=300:duration={d}",
                 str(p)], capture_output=True)

    # ── Part A: _compose_breakout_state — the CORE remap fix (pure logic) ──────────────────────
    cm, bx = _compose_breakout_state({}, {}, None, {5: "v"}, {4: 5})
    check("compose: mid-only merges new clip/bidx unchanged", cm == {5: "v"} and bx == {4: 5})
    # THE BUG: a later insertion reindexes every scene → the prior mid clip key MUST be remapped
    _idxmap = {i: i + 1 for i in range(12)}            # cold-open shifted every scene by +1
    cm2, bx2 = _compose_breakout_state({5: "midvid"}, {4: 5}, _idxmap, {0: "covid"}, {2: 0})
    check("compose: cold-open reindex remaps the mid clip key (no stale audio-over-black)",
          5 not in cm2 and cm2.get(6) == "midvid" and cm2.get(0) == "covid"
          and bx2.get(4) == 6 and bx2.get(2) == 0)
    _logs = []
    cm3, _ = _compose_breakout_state({9: "gone"}, {}, {0: 0, 1: 1}, {}, {}, log=_logs.append)
    check("compose: a dropped-scene clip key is removed and logged",
          cm3 == {} and any("stale" in L.lower() for L in _logs))

    # ── factories for the REAL _apply_breakouts / _compose_breakouts path ─────────────────────
    dummy_proj = types.SimpleNamespace()

    def _state(n=10):
        segs = [ScriptSegment(index=i, text=f"beat {i}") for i in range(n)]
        scs = [Scene(index=i, narration=f"beat {i}", visual="footage") for i in range(n)]
        nss = [NarratedScene(index=i, audio=tmp / f"n{i}.wav", duration=4.0, words=[])
               for i in range(n)]
        narr = types.SimpleNamespace(scenes=nss, audio=str(tmp / "narr.wav"), total=n * 4.0)
        return segs, scs, narr

    _bkn = [0]

    def _pick(beat, cold=False, line=None, dur=2.0, black=False, make=True):
        _bkn[0] += 1
        tag = f"bk{_bkn[0]}"
        v, a = tmp / f"{tag}.mp4", tmp / f"{tag}.wav"
        if make:
            _mk_mp4(v, dur, black=black)
            _mk_wav(a, dur)
        p = {"seg_index": beat, "dur": dur, "video": str(v), "audio": str(a),
             "_audit": {"seg_index": beat, "source_id": f"src{beat}", "source_t": 1.0,
                        "line": line or f"line {beat}"}}
        if cold:
            p["cold_open"] = True
            p["hook_quote"] = "the opening hook line"
            p["hook_beats"] = [beat]
        return p

    _orig_splice = B._splice_audio
    B._splice_audio = lambda full, splices, work: Path(full)   # skip the ffmpeg concat
    _NULL = lambda m: None
    try:
        # CASE 1 — mids only
        s, sc, nr = _state()
        s2, sc2, nr2, cm, bx, ents = _compose_breakouts(
            dummy_proj, s, sc, nr, [_pick(4), _pick(7)], tmp, True, log=_NULL)
        _bk = {x.index for x in sc2 if getattr(x, "visual", "") == "breakout"}
        check("mids-only: clip-map keys == breakout pseudo-scenes (2)",
              set(cm.keys()) == _bk and len(_bk) == 2)
        _ok, _ = _validate_breakout_assembly(sc2, nr2, cm, nr2._breakout_caps, True)
        check("mids-only: assembly invariant passes", _ok)

        # CASE 2 — cold-open + mids via VO-cut FALLBACK (insert): the exact production bug
        s, sc, nr = _state()
        nr._vo_word_aligned = False                    # no word alignment → insert fallback
        picks = [_pick(4, line="mid four"), _pick(7, line="mid seven"),
                 _pick(2, cold=True, line="cold hook")]
        s2, sc2, nr2, cm, bx, ents = _compose_breakouts(
            dummy_proj, s, sc, nr, picks, tmp, True, log=_NULL)
        _bk = {x.index for x in sc2 if getattr(x, "visual", "") == "breakout"}
        check("cold-open+mids (fallback): all 3 breakout scenes mapped, keys == scenes, videos exist",
              set(cm.keys()) == _bk and len(_bk) == 3 and all(Path(cm[k]).exists() for k in cm))
        _ok, _probs = _validate_breakout_assembly(sc2, nr2, cm, nr2._breakout_caps, True)
        check("cold-open+mids (fallback): invariant passes (WAS audio-over-black)", _ok and not _probs)
        _caps = nr2._breakout_caps
        check("cold-open+mids: all 3 captions preserved (mids NOT overwritten by cold-open)",
              len(_caps) == 3 and len({round(c["start"], 2) for c in _caps}) == 3)
        # every caption's video points at a real breakout scene in the clip-map
        check("cold-open+mids: every caption maps to a mapped breakout video",
              all(str(c.get("video")) in {str(v) for v in cm.values()} for c in _caps))
        # REGRESSION (review finding #1): bidx must NOT map the cold-open's "insert-before" beat to
        # the pseudo-scene — that beat is a REAL narration beat that keeps its own footage. Every
        # bidx value must be a DISTINCT scene, and no bidx value may point at a breakout pseudo-scene
        # (breakouts are consumed via the clip-map, never proj.selections).
        check("cold-open+mids: bidx preserves ordinary beats (no clip-map key stolen from footage)",
              len(set(bx.values())) == len(bx)
              and not (set(bx.values()) & set(cm.keys())))
        check("cold-open+mids: cold-open's insert-before beat still maps to a real (non-breakout) scene",
              bx.get(2) is not None and bx.get(2) not in cm)

        # CASE 3 — cold-open + mids with VO_CUT=0 (env off → insert path)
        os.environ["VIDLORE_CLIPSTUDIO_VO_CUT"] = "0"
        try:
            s, sc, nr = _state()
            nr._vo_word_aligned = True                 # even aligned, env off forces insert
            s2, sc2, nr2, cm, bx, ents = _compose_breakouts(
                dummy_proj, s, sc, nr,
                [_pick(4), _pick(7), _pick(2, cold=True)], tmp, True, log=_NULL)
            _bk = {x.index for x in sc2 if getattr(x, "visual", "") == "breakout"}
            _ok, _ = _validate_breakout_assembly(sc2, nr2, cm, nr2._breakout_caps, True)
            check("VO_CUT=0: cold-open takes the insert path, all mapped, invariant passes",
                  set(cm.keys()) == _bk and len(_bk) == 3 and _ok)
        finally:
            os.environ.pop("VIDLORE_CLIPSTUDIO_VO_CUT", None)

        # CASE 4 — cold-open + mids via VO-cut SUCCESS (word-aligned uploaded VO)
        _orig_loc, _orig_trim = B._locate_hook_span, B._audio_trim_prepend
        B._locate_hook_span = lambda vo, q, log, **k: (2.0, 10.0, 0.92)
        B._audio_trim_prepend = lambda vo, h1, ca, d, work: Path(vo)
        try:
            s, sc, nr = _state(12)
            nr._vo_word_aligned = True
            picks = [_pick(6, line="mid six"), _pick(9, line="mid nine"),
                     _pick(3, cold=True, line="cold hook")]
            s2, sc2, nr2, cm, bx, ents = _compose_breakouts(
                dummy_proj, s, sc, nr, picks, tmp, True, log=_NULL)
            _bk = {x.index for x in sc2 if getattr(x, "visual", "") == "breakout"}
            _ok, _probs = _validate_breakout_assembly(sc2, nr2, cm, nr2._breakout_caps, True)
            check("cold-open+mids (VO-cut success): all mapped, keys==scenes, invariant passes",
                  set(cm.keys()) == _bk and len(_bk) == 3 and _ok and not _probs)
            check("VO-cut success: cold-open pseudo-scene is index 0 (replaces the hook)",
                  0 in cm)
        finally:
            B._locate_hook_span, B._audio_trim_prepend = _orig_loc, _orig_trim

        # CASE 5 — captions OFF: invariant does not require a caption per scene
        s, sc, nr = _state()
        nr._vo_word_aligned = False
        s2, sc2, nr2, cm, bx, ents = _compose_breakouts(
            dummy_proj, s, sc, nr, [_pick(4), _pick(2, cold=True)], tmp, False, log=_NULL)
        _ok, _ = _validate_breakout_assembly(sc2, nr2, cm, [], False)
        check("captions OFF: invariant passes without caption specs", _ok)

        # CASE 6 — MISSING breakout video blocks publication → whole feature rolled back
        s, sc, nr = _state()
        nr._vo_word_aligned = False
        _bad = _pick(5, make=False)                    # video/audio files never created
        Path(_bad["audio"]).write_bytes(b"")           # audio exists but video is missing
        s2, sc2, nr2, cm, bx, ents = _compose_breakouts(
            dummy_proj, s, sc, nr, [_bad], tmp, True, log=_NULL)
        _bk = [x for x in sc2 if getattr(x, "visual", "") == "breakout"]
        check("missing breakout video → rolled back (no breakout scene, no audio-over-black)",
              cm == {} and ents == [] and not _bk)

        # CASE 7 — audit finalized to FINAL aired times keyed by stable identity
        _baud = tmp / "breakout_audit.json"
        _baud.write_text(json.dumps({"accepted": [
            {"seg_index": 4, "source_id": "srcX", "source_t": 1.0, "line": "the exact quote"}]}),
            encoding="utf-8")
        _caps = [{"start": 152.6, "dur": 7.5, "audio": str(tmp / "x.wav"),
                  "video": str(tmp / "x.mp4"), "final_index": 8,
                  "source_id": "srcX", "source_t": 1.0, "line": "the exact quote"}]
        _finalize_breakout_audit(tmp, _caps, {8: tmp / "x.mp4"},
                                 types.SimpleNamespace(total=800.0), log=_NULL)
        _fin = json.loads(_baud.read_text(encoding="utf-8"))["accepted"][0]
        check("audit finalized: aired_at_s/end + final_index from stable identity",
              abs(_fin.get("aired_at_s", 0) - 152.6) < 1e-6 and _fin.get("final_index") == 8
              and abs(_fin.get("aired_end_s", 0) - 160.1) < 1e-6)

        # CASE 8 — post-render QA + HARD gate: black / wrong / undecodable fail; correct passes.
        from vidlore.clipstudio.build import _breakout_qa_gate

        def _mk_pat(p, d=4.0, kind="testsrc"):
            _sp.run([FF, "-y", "-f", "lavfi", "-i", f"{kind}=size=320x180:rate=10:duration={d}",
                     "-pix_fmt", "yuv420p", str(p)], capture_output=True)

        # (a) a BLACK breakout window is flagged (luma gate) even when the "source" is also black
        _blk = tmp / "qa_black.mp4"; _mk_mp4(_blk, 4.0, black=True)
        _pb = _postrender_breakout_qa(_blk, [{"start": 0.5, "dur": 3.0, "line": "over black",
                                              "video": str(_blk)}], tmp, log=_NULL)
        check("post-render QA: a sustained-black breakout window is flagged",
              len(_pb) == 1 and "BLACK" in _pb[0]["reason"])
        # (b) CORRECT footage (final == prepared clip, patterned) passes
        _pat = tmp / "qa_pat.mp4"; _mk_pat(_pat, 4.0, "testsrc")
        _pc = _postrender_breakout_qa(_pat, [{"start": 0.5, "dur": 3.0, "line": "real",
                                              "video": str(_pat)}], tmp, log=_NULL)
        check("post-render QA: a correct non-black breakout (final matches prepared clip) passes",
              _pc == [])
        # (c) WRONG footage — final is a different pattern than the prepared clip → gross mismatch
        _pat2 = tmp / "qa_pat2.mp4"; _mk_pat(_pat2, 4.0, "testsrc2")
        _pw = _postrender_breakout_qa(_pat, [{"start": 0.5, "dur": 3.0, "line": "wrong",
                                              "video": str(_pat2)}], tmp, log=_NULL)
        check("post-render QA: wrong (grossly different) footage fails",
              len(_pw) == 1 and "does NOT match" in _pw[0]["reason"])
        # (d) UNDECODABLE final frames fail CLOSED (not a silent valid=[] pass)
        _junk = tmp / "qa_junk.mp4"; _junk.write_bytes(b"not a video" * 50)
        _pd = _postrender_breakout_qa(_junk, [{"start": 0.5, "dur": 3.0, "line": "junk",
                                               "video": str(_pat)}], tmp, log=_NULL)
        check("post-render QA: undecodable final frames fail closed (probe error)",
              len(_pd) == 1 and "decoded" in _pd[0]["reason"] and _pd[0].get("probe_errors"))
        # (e) MISSING prepared clip fails closed
        _pm = _postrender_breakout_qa(_pat, [{"start": 0.5, "dur": 3.0, "line": "nosrc",
                                              "video": str(tmp / "missing.mp4")}], tmp, log=_NULL)
        check("post-render QA: missing prepared clip fails closed",
              len(_pm) == 1 and "could not verify" in _pm[0]["reason"])
        # (f) FLAT non-black final vs FLAT source (both near-uniform) must NOT false-match — the
        # low-texture crops can't be trusted, so it fails closed (review finding #2).
        _flat_a = tmp / "qa_flat_a.mp4"; _mk_mp4(_flat_a, 4.0, black=False)   # solid white
        _flat_b = tmp / "qa_flat_b.mp4"
        _sp.run([FF, "-y", "-f", "lavfi", "-i", "color=c=gray:s=320x180:d=4",
                 "-pix_fmt", "yuv420p", "-r", "10", str(_flat_b)], capture_output=True)
        _pf = _postrender_breakout_qa(_flat_a, [{"start": 0.5, "dur": 3.0, "line": "flat",
                                                 "video": str(_flat_b)}], tmp, log=_NULL)
        check("post-render QA: flat non-black final vs flat source fails closed (no false-match)",
              len(_pf) == 1 and "could not verify" in _pf[0]["reason"])
        # (g) BLACK check fails CLOSED when luma is unmeasurable (review finding #1) — monkeypatch
        # _qa_crop_stats to return an unreadable luma on a decodable frame.
        _orig_stats = B._qa_crop_stats
        B._qa_crop_stats = lambda img: (12345, -1.0, 200)   # decodes, hash ok, luma unreadable
        try:
            _pg = _postrender_breakout_qa(_pat, [{"start": 0.5, "dur": 3.0, "line": "nolum",
                                                  "video": str(_pat)}], tmp, log=_NULL)
        finally:
            B._qa_crop_stats = _orig_stats
        check("post-render QA: unmeasurable luma on decoded frames fails closed (no fail-open)",
              len(_pg) == 1 and "could not measure luma" in _pg[0]["reason"])

        # HARD GATE — a failing QA quarantines the final and raises; final.mp4 no longer exists.
        # Mirror the real layout: work=<out>/work, final + failures json live in <out> (work.parent).
        import shutil as _sh
        _out = tmp / "gateout"; _wd = _out / "work"; _wd.mkdir(parents=True)
        _final = _out / "gate_final.mp4"; _sh.copy(_blk, _final)
        (_wd / "breakout_audit.json").write_text(json.dumps({"accepted": [
            {"seg_index": 4, "source_id": "s", "source_t": 1.0, "line": "x"}]}), encoding="utf-8")
        _raised = False
        try:
            _breakout_qa_gate(_final, [{"start": 0.5, "dur": 3.0, "line": "over black",
                                        "video": str(_blk)}], _wd, log=_NULL)
        except RuntimeError:
            _raised = True
        check("hard gate: black breakout RAISES and quarantines (final.mp4 not publishable)",
              _raised and not _final.exists()
              and (_out / "gate_final.FAILED_BREAKOUT_QA.mp4").exists()
              and (_out / "breakout_qa_failures.json").exists())
        _qp = json.loads((_wd / "breakout_audit.json").read_text())
        check("hard gate: audit stamped qa_passed=false on failure", _qp.get("qa_passed") is False)
        # a wrong (non-black) clip also RAISES via the gate
        _final2 = _out / "gate_final2.mp4"; _sh.copy(_pat, _final2)
        _r2 = False
        try:
            _breakout_qa_gate(_final2, [{"start": 0.5, "dur": 3.0, "line": "wrong",
                                         "video": str(_pat2)}], _wd, log=_NULL)
        except RuntimeError:
            _r2 = True
        check("hard gate: wrong-footage breakout RAISES and quarantines",
              _r2 and not _final2.exists())
        # a CORRECT render returns normally and stays publishable
        _final3 = _out / "gate_final3.mp4"; _sh.copy(_pat, _final3)
        _ret = _breakout_qa_gate(_final3, [{"start": 0.5, "dur": 3.0, "line": "real",
                                            "video": str(_pat)}], _wd, log=_NULL)
        check("hard gate: a valid render returns the result and stays publishable",
              _ret == _final3 and _final3.exists())

        # CROSS-WIRED caption (right count, wrong final_index/video) fails the invariant
        _sc_bk = [types.SimpleNamespace(index=3, visual="breakout"),
                  types.SimpleNamespace(index=6, visual="breakout")]
        _nr_bk = types.SimpleNamespace(scenes=[
            types.SimpleNamespace(index=3, audio=str(tmp / "a3.wav")),
            types.SimpleNamespace(index=6, audio=str(tmp / "a6.wav"))])
        for _p9 in (tmp / "v3.mp4", tmp / "v6.mp4"):
            _mk_pat(_p9, 2.0)
        for _a9 in (tmp / "a3.wav", tmp / "a6.wav"):
            _a9.write_bytes(b"x")
        _cm_ok = {3: tmp / "v3.mp4", 6: tmp / "v6.mp4"}
        _caps_ok = [{"final_index": 3, "video": str(tmp / "v3.mp4"), "audio": str(tmp / "a3.wav")},
                    {"final_index": 6, "video": str(tmp / "v6.mp4"), "audio": str(tmp / "a6.wav")}]
        _ok9, _ = _validate_breakout_assembly(_sc_bk, _nr_bk, _cm_ok, _caps_ok, True)
        check("invariant: correctly-wired captions pass", _ok9)
        _caps_xwire = [{"final_index": 3, "video": str(tmp / "v6.mp4"), "audio": str(tmp / "a3.wav")},
                       {"final_index": 6, "video": str(tmp / "v3.mp4"), "audio": str(tmp / "a6.wav")}]
        _okx, _px = _validate_breakout_assembly(_sc_bk, _nr_bk, _cm_ok, _caps_xwire, True)
        check("invariant: cross-wired caption video (count correct) FAILS", not _okx and _px)
        _caps_dupidx = [{"final_index": 3, "video": str(tmp / "v3.mp4"), "audio": str(tmp / "a3.wav")},
                        {"final_index": 3, "video": str(tmp / "v3.mp4"), "audio": str(tmp / "a3.wav")}]
        _okd, _ = _validate_breakout_assembly(_sc_bk, _nr_bk, _cm_ok, _caps_dupidx, True)
        check("invariant: duplicate caption final_index FAILS", not _okd)
    finally:
        B._splice_audio = _orig_splice

    # ── wiring: the production build routes breakouts through the single tested entrypoint ──────
    bsrc = (Path(__file__).resolve().parents[1] / "vidlore" / "clipstudio"
            / "build.py").read_text(encoding="utf-8")
    check("build_video composes breakouts via _compose_breakouts (no inline .update())",
          "_compose_breakouts(" in bsrc
          and "segments, scenes, narration, _breakout_clip, _bidx, _breakout_entries = _compose_breakouts("
          in bsrc)
    check("black-frame repair receives breakout windows (never preserves them as fades)",
          "breakout_windows=" in bsrc and "breakout_window_black" in
          (Path(__file__).resolve().parents[1] / "vidlore" / "assemble.py").read_text(encoding="utf-8"))
    check("post-render breakout QA is a HARD gate (quarantine + raise on failure)",
          "_breakout_qa_gate(result" in bsrc and "FAILED_BREAKOUT_QA" in bsrc
          and "raise RuntimeError(" in bsrc)
    check("final audit persists explicit counts (accepted/validated/qa_passed)",
          'data["accepted_count"]' in bsrc and 'data["validated_count"]' in bsrc
          and '"qa_passed"' in bsrc)


def test_caption_presets():
    print("[captions] preset system — 5 designs, ON/OFF, ASS validity, wiring, OFF-safety")
    import tempfile
    from pathlib import Path as _P
    from vidlore.clipstudio import caption_presets as CP
    from vidlore.captions import write_ass
    from vidlore.tts import WordTiming

    # (1) every valid preset resolves to itself; (2) invalid falls back + flags invalid
    for name in CP.VALID_STYLES:
        p, inv = CP.resolve_style(name)
        check(f"preset '{name}' resolves to itself", p.name == name and inv is False)
    _pf, _invf = CP.resolve_style("totally-bogus-value")
    check("invalid style falls back to professional and is flagged",
          _pf.name == "professional" and _invf is True)
    # (3) existing projects / no settings → safe defaults
    check("no style (None/'') → professional, not flagged invalid",
          CP.resolve_style(None)[0].name == "professional" and CP.resolve_style(None)[1] is False
          and CP.resolve_style("")[0].name == "professional")
    check("captions enabled defaults ON; explicit False honoured",
          CP.captions_enabled(None) is True and CP.captions_enabled(False) is False
          and CP.captions_enabled(True) is True)
    check("exactly five presets, professional is the default",
          len(CP.VALID_STYLES) == 5 and CP.DEFAULT_STYLE == "professional"
          and set(CP.VALID_STYLES) == {"professional", "minimal", "cinematic", "documentary", "focus"})

    # env fallback: VIDLORE_CLIPSTUDIO_CAPTION_STYLE + _CAPTIONS
    _oldS = os.environ.get("VIDLORE_CLIPSTUDIO_CAPTION_STYLE")
    _oldC = os.environ.get("VIDLORE_CLIPSTUDIO_CAPTIONS")
    try:
        os.environ["VIDLORE_CLIPSTUDIO_CAPTION_STYLE"] = "cinematic"
        check("env VIDLORE_CLIPSTUDIO_CAPTION_STYLE used when no explicit style",
              CP.resolve_style(None)[0].name == "cinematic"
              and CP.resolve_style("focus")[0].name == "focus")   # explicit still wins
        os.environ["VIDLORE_CLIPSTUDIO_CAPTIONS"] = "0"
        check("env VIDLORE_CLIPSTUDIO_CAPTIONS=0 → captions off (fallback path)",
              CP.captions_enabled(None) is False)
    finally:
        for k, v in (("VIDLORE_CLIPSTUDIO_CAPTION_STYLE", _oldS), ("VIDLORE_CLIPSTUDIO_CAPTIONS", _oldC)):
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    # (9) every preset produces VALID ASS for BOTH narration and breakout captions.
    _words = [WordTiming("Power", 0.0, 0.4), WordTiming("is", 0.4, 0.6),
              WordTiming("never", 0.6, 1.1), WordTiming("given", 1.1, 1.7),
              WordTiming("to", 1.7, 1.9), WordTiming("the", 1.9, 2.1),
              WordTiming("patient", 2.1, 2.7)]   # long-ish → two-line wrap territory
    tmp = _P(tempfile.mkdtemp(prefix="captest_"))
    _req_keys = {"font", "size", "primary", "outline", "back", "bold",
                 "border_style", "outline_w", "shadow", "margin_v"}
    for name in CP.VALID_STYLES:
        preset = CP.CAPTION_PRESETS[name]
        capd, accent = preset.theme_caption()
        # theme caption dict carries every key write_ass reads
        check(f"preset '{name}' theme-caption dict has all write_ass keys",
              _req_keys.issubset(set(capd.keys())) and isinstance(accent, tuple) and len(accent) == 3)
        # drive the REAL engine caption writer with this preset's style
        _ass = write_ass(_words, tmp / f"{name}.ass", style=capd, accent=accent,
                         emphasis_words={"never"})
        _txt = _ass.read_text(encoding="utf-8")
        check(f"preset '{name}' → valid narration ASS (header + Main style + events)",
              "[V4+ Styles]" in _txt and "Style: Main," in _txt and "[Events]" in _txt
              and capd["primary"] in _txt and "Dialogue:" in _txt)
        # breakout Style line: the 17 ASS columns (Name..Encoding), Name=BK
        _bk = preset.breakout_style_line()
        _fields = _bk[len("Style: "):].split(",")
        check(f"preset '{name}' → valid breakout BK style (17 ASS fields, sung≠unsung)",
              _bk.startswith("Style: BK,") and len(_fields) == 17
              and CP._ass_color(preset.bk_sung_rgb) != CP._ass_color(preset.bk_unsung_rgb))

    # (10) MAIN + BREAKOUT captions come from the SAME preset family (documentary box → both boxed)
    _doc = CP.CAPTION_PRESETS["documentary"]
    check("documentary: both narration + breakout use the translucent-box border (family match)",
          _doc.theme_caption()[0]["border_style"] == 3 and _doc.bk_border_style == 3)
    _min = CP.CAPTION_PRESETS["minimal"]
    check("minimal: quiet accent close to white (least distracting emphasis)",
          _min.accent_rgb[0] > 230 and _min.accent_rgb[1] > 220)
    check("focus: bright energetic active-word gold (strong word-sync emphasis)",
          CP.CAPTION_PRESETS["focus"].accent_rgb == (255, 178, 40))

    # (15/16) presets stay within a lower safe region and clear the cinematic letterbox bars.
    for name in CP.VALID_STYLES:
        capd = CP.CAPTION_PRESETS[name].theme_caption()[0]
        check(f"preset '{name}' margin_v in a sane lower-safe range + max 2 lines",
              40 <= capd["margin_v"] <= 220 and capd["max_lines"] == 2)

    # portal preview payload is well-formed for every preset
    _choices = CP.preset_choices()
    check("portal preset_choices lists all five with preview typography",
          [c["name"] for c in _choices] == list(CP.VALID_STYLES)
          and all({"color", "accent", "weight", "backplate", "text_shadow", "family"}
                  <= set(c["preview"].keys()) for c in _choices))

    # (4/20) portal validation: unknown/manipulated preset value falls back to the default; the
    # server never forwards a raw value. Drive the REAL portal card generator + validation constants.
    from vidlore.clipstudio import web as _web
    _cards = _web._caption_cards()
    check("portal renders a selectable card for every preset with a sample + accent word",
          all(f'data-name="{n}"' in _cards for n in CP.VALID_STYLES)
          and _cards.count("Power is") == 5 and _cards.count("never") == 5)
    # the /create route clamps caption_style to VALID_STYLES (mirror its logic on a hostile value)
    _hostile = "professional; DROP TABLE"
    check("portal clamps an unknown/hostile caption_style to the default",
          (_hostile.strip().lower() if _hostile.strip().lower() in CP.VALID_STYLES
           else CP.DEFAULT_STYLE) == CP.DEFAULT_STYLE)

    # (5) per-job settings do not leak across jobs (the registry/resolver is pure — no shared state)
    _a = CP.resolve_style("minimal")[0]
    _b = CP.resolve_style("focus")[0]
    check("resolver is stateless — two jobs keep independent presets",
          _a.name == "minimal" and _b.name == "focus" and _a is not _b)

    # wiring greps: OFF removes visible layers but NEVER the breakout safety metadata.
    root = _P(__file__).resolve().parents[1] / "vidlore" / "clipstudio"
    bsrc = (root / "build.py").read_text(encoding="utf-8")
    wsrc = (root / "web.py").read_text(encoding="utf-8")
    osrc = (root / "orchestrate.py").read_text(encoding="utf-8")
    check("build resolves ONE caption on/off + preset and logs it",
          "_cap_on" in bsrc and "_cap_preset" in bsrc
          and 'log(f"build: captions=' in bsrc and "invalid caption style" in bsrc)
    check("narration + breakout burn + compose all gate on the resolved _cap_on",
          "captions=_cap_on" in bsrc and "if _cap_on and os.environ" in bsrc
          and "work, _cap_on, log=log)" in bsrc)
    check("selected preset drives BOTH narration theme caption AND breakout burn",
          '"caption": {**th.get("caption", {}), **_cap_dict}' in bsrc
          and "_burn_breakout_captions(result, _caps, work, log, preset=_cap_preset)" in bsrc)
    # the active-word colour is caption-SCOPED (caption_accent) so it never recolours title/graphic
    # overlays or the key-phrase stabs (which fire only when captions are OFF).
    asmsrc = (Path(__file__).resolve().parents[1] / "vidlore" / "assemble.py").read_text(encoding="utf-8")
    check("caption accent is caption-scoped (theme['accent'] not clobbered → no overlay recolour)",
          '"caption_accent": _cap_accent' in bsrc and '"accent": _cap_accent' not in bsrc
          and 'theme.get("caption_accent", theme.get("accent"' in asmsrc)
    check("OFF-safety: breakout metadata (_breakout_caps) + QA gate are NOT gated on captions",
          "_final_caps = list(getattr(narration, \"_breakout_caps\", None) or [])" in bsrc
          and "_breakout_qa_gate(result, _final_caps, work, log=log)" in bsrc)
    check("caption settings persisted to project meta for rebuild",
          'proj.meta["caption_settings"]' in bsrc)
    check("portal threads captions_enabled + caption_style to the worker (validated)",
          "captions_enabled=captions_enabled, caption_style=caption_style" in wsrc
          and "if caption_style not in VALID_STYLES" in wsrc)
    check("portal status page shows Captions: <Style> / Off",
          'Captions: {_p.label' in wsrc and '"Captions: Off"' in wsrc)
    check("caption_style threaded through produce_auto + produce",
          osrc.count("caption_style=caption_style") >= 2)


def test_caption_correctness():
    print("[captions] correctness pass — persist, precedence, preset lock, two-line, portal POST")
    import tempfile
    from pathlib import Path as _P
    from vidlore.clipstudio import caption_presets as CP

    # ── (1) PERSIST caption_settings to project.json — real save + reload roundtrip ──────────────
    from vidlore.clipstudio.models import ClipProject
    _root = _P(tempfile.mkdtemp(prefix="capproj_"))
    proj = ClipProject(name="capttest", root=str(_root))
    # replicate build_video's exact resolution + persist (real resolvers + real ClipProject.save)
    _on = CP.captions_enabled(False)                       # explicit OFF
    _preset, _ = CP.resolve_style("cinematic")
    proj.meta["caption_settings"] = {"enabled": bool(_on), "style": _preset.name}
    proj.save()
    reloaded = ClipProject.load(_root)
    _cs = reloaded.meta.get("caption_settings")
    check("caption_settings survives save→reload exactly (enabled=False, style=cinematic)",
          _cs == {"enabled": False, "style": "cinematic"})
    # rerender_project consumes the reloaded values (mirror its read logic)
    _capset = (reloaded.meta.get("caption_settings") or {})
    check("rerender_project reads persisted caption settings from the reloaded project",
          bool(_capset.get("enabled", True)) is False and str(_capset.get("style", "")) == "cinematic")
    # build_video actually owns the write (its persist calls proj.save AFTER setting the meta)
    _bsrc = (_P(__file__).resolve().parents[1] / "vidlore" / "clipstudio"
             / "build.py").read_text(encoding="utf-8")
    check("build_video persists atomically (proj.save after caption_settings)",
          'proj.meta["caption_settings"] = {"enabled": bool(_cap_on), "style": _cap_preset.name}'
          in _bsrc and "proj.save()" in _bsrc.split('proj.meta["caption_settings"]')[1][:80])

    # ── (2) PRECEDENCE — explicit > env > default (centralized captions_enabled) ─────────────────
    _oldC = os.environ.get("VIDLORE_CLIPSTUDIO_CAPTIONS")
    try:
        os.environ["VIDLORE_CLIPSTUDIO_CAPTIONS"] = "1"
        check("env=1 + explicit False → OFF (explicit wins)", CP.captions_enabled(False) is False)
        os.environ["VIDLORE_CLIPSTUDIO_CAPTIONS"] = "0"
        check("env=0 + explicit True → ON (explicit wins)", CP.captions_enabled(True) is True)
        check("no explicit + env=0 → OFF", CP.captions_enabled(None) is False)
        os.environ.pop("VIDLORE_CLIPSTUDIO_CAPTIONS", None)
        check("no explicit + no env → ON (default)", CP.captions_enabled(None) is True)
    finally:
        if _oldC is None:
            os.environ.pop("VIDLORE_CLIPSTUDIO_CAPTIONS", None)
        else:
            os.environ["VIDLORE_CLIPSTUDIO_CAPTIONS"] = _oldC
    check("build_video routes captions through captions_enabled (bool|None), not inline env logic",
          "_cap_on = _captions_enabled(captions)" in _bsrc
          and "captions: Optional[bool] = None" in _bsrc)

    # ── (3) PRESET LOCK — Look-DNA / subtitle_style must NOT re-skin a locked preset ─────────────
    from vidlore.captions import write_ass
    from vidlore.tts import WordTiming
    import vidlore.look_dna as _ld
    _tmp = _P(tempfile.mkdtemp(prefix="caplock_"))
    _words = [WordTiming(w, i * 0.3, (i + 1) * 0.3) for i, w in enumerate("Power is never given".split())]
    _oldcur, _oldget, _oldss = _ld.current, _ld.look_get, os.environ.get("VIDLORE_SUBTITLE_STYLE")

    def _style_line(ass_path):
        return next(l for l in ass_path.read_text().splitlines() if l.startswith("Style: Main,"))
    try:
        # activate an aggressive Look-DNA that would blow up size + swap the font, + a subtitle_style
        _ld.current = lambda: {"active": True}
        _ld.look_get = lambda k, d=None: {"captions.size_mult": 1.6,
                                          "captions.font_family": ["Impact"],
                                          "captions.margin_v_mult": 1.5,
                                          "subtitle_style": "bold_lower"}.get(k, d)
        os.environ["VIDLORE_SUBTITLE_STYLE"] = "1"
        for name in ("professional", "minimal", "focus"):
            capd, acc = CP.CAPTION_PRESETS[name].theme_caption()
            _sl = _style_line(write_ass(_words, _tmp / f"lock_{name}.ass", style=capd, accent=acc))
            _f = _sl.split(",")
            # Fontname (field 1 after 'Style: Main') must be the preset's font, Fontsize its own size
            _expect_size = int(CP.CAPTION_PRESETS[name].size * 1.06)
            check(f"locked preset '{name}' keeps its font ({capd['font']}) + size ({_expect_size}) "
                  f"despite active Look-DNA",
                  _f[1] == capd["font"] and abs(int(_f[2]) - _expect_size) <= 1)
        # a NON-locked caption dict (other engine callers) still gets the Look-DNA override
        _unlocked = {k: v for k, v in CP.CAPTION_PRESETS["professional"].theme_caption()[0].items()
                     if k != "preset_locked"}
        _slu = _style_line(write_ass(_words, _tmp / "unlocked.ass", style=_unlocked, accent=(255, 196, 84)))
        _fu = _slu.split(",")
        check("non-locked caller still honours Look-DNA (font swapped / size scaled)",
              _fu[1] == "Impact" and int(_fu[2]) > int(CP.CAPTION_PRESETS["professional"].size * 1.06))
    finally:
        _ld.current, _ld.look_get = _oldcur, _oldget
        if _oldss is None:
            os.environ.pop("VIDLORE_SUBTITLE_STYLE", None)
        else:
            os.environ["VIDLORE_SUBTITLE_STYLE"] = _oldss

    # ── (4) TWO-LINE ENFORCEMENT — real generated ASS, never a third line ────────────────────────
    from vidlore.captions import split_wide_cells, layout_two_lines
    capd, acc = CP.CAPTION_PRESETS["professional"].theme_caption()
    _big = int(capd["size"] * 1.06)
    _safe = 1920 - 90 - 90

    def _max_breaks(ass_path):
        return max((l.count("\\N") for l in ass_path.read_text().splitlines()
                    if l.startswith("Dialogue")), default=0)
    # normal short sentence → one line
    _n = [WordTiming(w, i * 0.3, (i + 1) * 0.3) for i, w in enumerate("Power is never given".split())]
    check("two-line: a normal short cue stays on one line",
          _max_breaks(write_ass(_n, _tmp / "n.ass", style=capd, accent=acc)) == 0)
    # a WIDE single cue (6 long words) → exactly one break, never two
    _wide = ["Extraordinary", "revolutionary", "transformation", "fundamentally", "reshaping", "everything"]
    _w = [WordTiming(x, i * 0.2, (i + 1) * 0.2) for i, x in enumerate(_wide)]
    check("two-line: a wide cue breaks into exactly two lines (one \\N, never a third)",
          _max_breaks(write_ass(_w, _tmp / "w.ass", style=capd, accent=acc, emphasis_words={"revolutionary"})) == 1)
    # an EXTREME single word → fit-scaled (\fs), never split, never a third line
    _xl = [WordTiming("Supercalifragilisticexpialidociousantidisestablishmentarianism", 0, 1.0),
           WordTiming("indeed", 1.0, 1.4)]
    _xa = write_ass(_xl, _tmp / "xl.ass", style=capd, accent=acc)
    check("two-line: an extreme long word is fit-scaled (\\fs), never a third line",
          any("\\fs" in l for l in _xa.read_text().splitlines() if l.startswith("Dialogue"))
          and _max_breaks(_xa) <= 1)
    # punctuation + CJK + RTL never exceed one break
    _p = [WordTiming(w, i * 0.3, (i + 1) * 0.3) for i, w in enumerate(["Power,", "is", "never—", "given!"])]
    _cjk = [WordTiming("权力从来不是别人赐予的东西啊", 0, 0.5), WordTiming("而是自己夺来的", 0.5, 1.0)]
    _rtl = [WordTiming("القوة", 0, 0.4), WordTiming("لا", 0.4, 0.6), WordTiming("تُمنح", 0.6, 1.0)]
    check("two-line: punctuation / CJK / RTL never produce a third line",
          _max_breaks(write_ass(_p, _tmp / "p.ass", style=capd, accent=acc)) <= 1
          and _max_breaks(write_ass(_cjk, _tmp / "cjk.ass", style=capd, accent=acc)) <= 1
          and _max_breaks(write_ass(_rtl, _tmp / "rtl.ass", style=capd, accent=acc)) <= 1)
    _wc, _wi = split_wide_cells(_wide, float(_big), float(_safe))
    _wb, _wf, _wsq = layout_two_lines(_wc, float(_big), float(_safe))
    check("two-line: split picks a balanced midpoint boundary for a wide cue",
          _wb in (2, 3, 4) and _wsq == 100)

    # ── (6) REAL Flask POST — the portal actually sends validated caption settings to the worker ──
    import vidlore.clipstudio.web as _web
    _portal_root = _P(tempfile.mkdtemp(prefix="portal_"))
    _orig_root, _orig_run = _web._ROOT, _web._run_job
    _captured = {}

    def _fake_run_job(jid, proj, **kw):                    # capture kwargs; never actually render
        _captured["jid"] = jid
        _captured.update(kw)
    _web._ROOT = _portal_root
    _web._run_job = _fake_run_job
    try:
        _web.app.config["TESTING"] = True
        _c = _web.app.test_client()
        import time as _t
        _captured.clear()
        _c.post("/create", data={"topic": "T", "script": "hello world", "captions": "0",
                                 "caption_style": "cinematic"})
        for _ in range(40):
            if "caption_style" in _captured:
                break
            _t.sleep(0.05)
        check("portal POST: captions OFF + cinematic reach the worker (real test_client)",
              _captured.get("captions_enabled") is False and _captured.get("caption_style") == "cinematic")
        _jid = _captured.get("jid")
        check("portal POST: per-job caption settings stored on the job (no cross-job state)",
              _web._JOBS.get(_jid, {}).get("captions_enabled") is False
              and _web._JOBS.get(_jid, {}).get("caption_style") == "cinematic")
        # a manipulated / unknown style clamps to the default before reaching the worker
        _captured.clear()
        _c.post("/create", data={"topic": "T", "script": "hi", "captions": "1",
                                 "caption_style": "bogus'; DROP TABLE"})
        for _ in range(40):
            if "caption_style" in _captured:
                break
            _t.sleep(0.05)
        check("portal POST: hostile/unknown caption_style is clamped to professional server-side",
              _captured.get("caption_style") == "professional"
              and _captured.get("captions_enabled") is True)
    finally:
        _web._ROOT, _web._run_job = _orig_root, _orig_run


def test_caption_motion_presets():
    print("[captions] per-preset word MOTION (gap 4) — locked, distinct, tasteful")
    import re
    import tempfile
    from pathlib import Path as _P
    from vidlore.clipstudio import caption_presets as CP
    from vidlore.tts import WordTiming
    from vidlore.captions import write_ass
    _tmp = _P(tempfile.mkdtemp(prefix="capmotion_"))
    _words = [WordTiming("Power", 0.0, 0.4), WordTiming("corrupts", 0.4, 1.0),
              WordTiming("always", 1.0, 1.5)]

    def _max_fscx(txt):
        vals = [int(x) for x in re.findall(r"\\fscx(\d+)", txt)]
        return max(vals) if vals else 100

    def _has_bounce(txt):                                  # the punch bounce is an animated \t
        return "\\t(" in txt

    _peak, _bounce = {}, {}
    for name in CP.VALID_STYLES:
        cap, acc = CP.CAPTION_PRESETS[name].theme_caption()
        _txt = write_ass(_words, _tmp / f"{name}.ass", style=cap, accent=acc,
                         emphasis_words={"corrupts"}).read_text()
        _peak[name], _bounce[name] = _max_fscx(_txt), _has_bounce(_txt)
    check("motion: focus is the strongest word-synced emphasis",
          _peak["focus"] == max(_peak.values()) and _peak["focus"] > _peak["professional"])
    check("motion: minimal is the most restrained pop",
          _peak["minimal"] == min(_peak.values()) and _peak["minimal"] < _peak["professional"])
    check("motion: cinematic + documentary sit below professional (subtle, readable)",
          _peak["cinematic"] < _peak["professional"] and _peak["documentary"] < _peak["professional"])
    check("motion: ONLY focus bounces; minimal/cinematic/documentary/professional stay calm",
          _bounce["focus"] and not any(_bounce[n] for n in
                                       ("minimal", "cinematic", "documentary", "professional")))
    _m = {n: CP.CAPTION_PRESETS[n].theme_caption()[0].get("motion") for n in CP.VALID_STYLES}
    check("motion: every preset emits a 'motion' (emphasis+bounce) in its caption dict",
          all(isinstance(_m[n], dict) and "emphasis" in _m[n] and "bounce" in _m[n]
              for n in CP.VALID_STYLES))
    # PRESET-LOCK also locks motion: an active subtitle_style must not alter a locked preset's motion
    _old = os.environ.get("VIDLORE_SUBTITLE_STYLE")
    os.environ["VIDLORE_SUBTITLE_STYLE"] = "1"
    try:
        cap, acc = CP.CAPTION_PRESETS["focus"].theme_caption()      # locked
        _lk = write_ass(_words, _tmp / "focus_locked.ass", style=cap, accent=acc,
                        emphasis_words={"corrupts"}).read_text()
        check("motion: preset-lock locks motion too (subtitle_style can't override)",
              _max_fscx(_lk) == _peak["focus"] and _has_bounce(_lk))
    finally:
        if _old is None:
            os.environ.pop("VIDLORE_SUBTITLE_STYLE", None)
        else:
            os.environ["VIDLORE_SUBTITLE_STYLE"] = _old


def test_caption_pixel_bbox():
    print("[captions] PIXEL validation (gaps 1-3) — real libass render, measured bbox in safe margins")
    import tempfile
    from pathlib import Path as _P
    try:
        sys.path.insert(0, str(_P(__file__).resolve().parent))
        from caption_pixel_probe import (have_libass, render_frame, measure,
                                          _write_narration, _cases, FFMPEG)
    except Exception as _e:                                # probe/deps missing → honest skip
        check(f"pixel: probe unavailable, skipped ({type(_e).__name__})", True)
        return
    if not have_libass():
        check("pixel: libass subtitles filter unavailable — skipped (documented limitation)", True)
        return
    from vidlore.clipstudio import caption_presets as CP
    style, accent = CP.CAPTION_PRESETS["professional"].theme_caption()
    _tmp = _P(tempfile.mkdtemp(prefix="cappix_"))
    ml, _bad, _measured = 90, [], 0
    for label, words, emph, t in _cases():                 # normal + peak-animation sample times
        for (w, h, tag) in ((1920, 1080, "1080p"), (1280, 720, "720p")):
            ass = _tmp / f"{label}.ass"
            _write_narration(words, style, accent, emph, ass)
            png = _tmp / f"{label}_{tag}.png"
            if not render_frame(ass, png, w=w, h=h, t=t):
                continue
            m = measure(png)
            _measured += 1
            safe_l = ml * (w / 1920.0)
            if m["empty"] or m["rows"] > 2 or m["margin_l"] < safe_l - 6 or m["margin_r"] < safe_l - 6:
                _bad.append((label, tag, m))
    check(f"pixel: {_measured} rendered caption frames all ≤2 rows AND inside safe margins (0 clips)",
          _measured >= 15 and not _bad)
    if _bad:
        print("   OUT-OF-SAFE:", [(b[0], b[1], b[2].get("rows"), b[2].get("margin_l"),
                                   b[2].get("margin_r")) for b in _bad[:4]])
    # Caption OFF → zero visible caption ink (no subtitle burn at all)
    import subprocess as _sp
    _blank = _tmp / "off.png"
    _sp.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
             "-i", "color=c=black:s=1280x720:d=1", "-frames:v", "1", str(_blank)],
            check=True, capture_output=True)
    _mo = measure(_blank)
    check("pixel: Caption OFF renders zero visible caption ink", _mo.get("empty") or _mo.get("px") == 0)


def test_breakout_caption_layout():
    print("[captions] breakout width-aware karaoke (gap 1) — ≤2 rows, timing preserved, ASR floor")
    import re
    import tempfile
    import types as _ty
    from pathlib import Path as _P
    try:
        import faster_whisper as _fw
    except Exception:
        check("breakout-layout: faster_whisper unavailable — skipped", True)
        return
    import vidlore.clipstudio.build as B
    from vidlore.clipstudio import caption_presets as CP

    class _FW:                                             # canned word timestamps (no audio/model)
        def __init__(self, word, start, end, probability=0.99):
            self.word, self.start, self.end, self.probability = word, start, end, probability

    class _Seg:
        def __init__(self, words):
            self.words = words

    _canned = {"segs": []}

    class _FakeModel:
        def __init__(self, *a, **k):
            pass

        def transcribe(self, audio, **k):
            return _canned["segs"], _ty.SimpleNamespace()

    _orig, _fw.WhisperModel = _fw.WhisperModel, _FakeModel
    _tmp = _P(tempfile.mkdtemp(prefix="bkcap_"))
    _pre = CP.CAPTION_PRESETS["professional"]
    caps = [{"audio": str(_tmp / "a.wav"), "start": 10.0, "dur": 4.0}]
    try:
        # FIVE long W-heavy words → wrap into ≤2 rows per karaoke line, never a clipped third
        _five = [_FW("WWWWWWWWWWWW", i * 0.4, i * 0.4 + 0.35) for i in range(5)]
        _canned["segs"] = [_Seg(_five)]
        out = _tmp / "bk.ass"
        B._breakout_caption_ass(caps, out, log=None, preset=_pre)
        _dlg = [l for l in out.read_text().splitlines() if l.startswith("Dialogue")]
        check("breakout: five long words wrap ≤2 rows per line (one \\N max, never a third)",
              _dlg and all(l.count("\\N") <= 1 for l in _dlg))
        check("breakout: word-sync karaoke \\kf tags preserved on every line",
              _dlg and all("\\kf" in l for l in _dlg))
        _cs = sum(int(x) for l in _dlg for x in re.findall(r"\\kf(\d+)", l))
        _exp = sum(int(round((w.end - w.start) * 100)) for w in _five)
        check("breakout: total karaoke fill preserves the spoken word durations (timing unchanged)",
              abs(_cs - _exp) <= len(_five) * 3 + 6)
        # a PATHOLOGICAL 80-char unbroken word → grapheme-split, still ≤2 rows, never truncated
        _canned["segs"] = [_Seg([_FW("W" * 80, 0.0, 1.0)])]
        out2 = _tmp / "bk2.ass"
        B._breakout_caption_ass(caps, out2, log=None, preset=_pre)
        _d2 = [l for l in out2.read_text().splitlines() if l.startswith("Dialogue")]
        check("breakout: an 80-char unbroken word wraps ≤2 rows (grapheme-split, never a third)",
              _d2 and all(l.count("\\N") <= 1 for l in _d2))
        # ASR-confidence floor still drops a low-confidence line (no wrong word burned)
        _canned["segs"] = [_Seg([_FW("garbled", 0.0, 0.5, probability=0.10)])]
        _r3 = B._breakout_caption_ass(caps, _tmp / "bk3.ass", log=None, preset=_pre)
        check("breakout: a sub-floor ASR line is dropped (a missing line beats a wrong one)",
              _r3 is None)
    finally:
        _fw.WhisperModel = _orig


def test_era_policy_and_still_verification():
    print("[gap-2] beat-local era policy + semantic recovery-still verification")
    from vidlore.clipstudio.verify import _beat_era
    from vidlore.clipstudio import models as M
    # (1) single-scene video → a CORROBORATED global episode hint may be used
    s = M.ScriptSegment(index=0, text="Tywin dismisses Joffrey")
    check("single-scene uses the global episode hint once corroborated",
          _beat_era(s, "S03E04", single_scene=True, global_verified=True) == "S03E04")
    # (1b) an UNVERIFIED hint constrains nothing, even single-scene. A wrong hint doesn't merely
    # mis-tag a beat: it purges the right footage. Measured — "S04E01" for a scene that is S03E10
    # dropped 354 shots including the correct episode's own upload.
    check("single-scene IGNORES an uncorroborated global hint",
          _beat_era(s, "S03E04", single_scene=True, global_verified=False) == "")
    # (2) multi-scene → NO global hint; era must come from the beat's own evidence
    check("multi-scene ignores the unsafe global hint when the beat has no local era",
          _beat_era(s, "S03E04", single_scene=False) == "")
    s2 = M.ScriptSegment(index=1, text="Arya wakes up blind",
                         scene_query="Arya blinded House of Black and White season 6")
    check("multi-scene derives beat-local era from scene_query",
          _beat_era(s2, "S02E07", single_scene=False) == "season 6")
    s3 = M.ScriptSegment(index=2, text="the coin handoff", expected_visual="Jaqen 2x10 gives the coin")
    check("multi-scene derives beat-local era from expected_visual (SxxExx)",
          _beat_era(s3, "", single_scene=False) == "season 2")
    # (3) wiring: recovery stills semantically verified; era policy threaded
    vsrc = (Path(__file__).resolve().parents[1] / "vidlore" / "clipstudio" / "verify.py").read_text()
    osrc = (Path(__file__).resolve().parents[1] / "vidlore" / "clipstudio" / "orchestrate.py").read_text()
    bsrc = (Path(__file__).resolve().parents[1] / "vidlore" / "clipstudio" / "build.py").read_text()
    check("verifier uses beat-local era (not a global multi-scene hint)",
          "_era_of(_seg)" in vsrc and '_vtype == "single_scene"' in vsrc
          and "global_verified=_global_ok" in vsrc)
    check("recovery stills semantically verified before install (tri-state; real face_ids)",
          "_still_verdict" in osrc and "_shot_face_ids" in osrc
          and "NOT the beat's requested entities" in osrc)
    check("unverified/rejected still is NOT installed; rejected moving footage never airs",
          "no wrong still airs" in osrc and "verifier-rejected clip(s) replaced with a validated" in bsrc
          and "verifier_failed" in bsrc)
    check("rejected footage → validated same-scene editorial HOLD (capped) or release-block",
          "editorial_hold" in bsrc and "_scene_compat" in bsrc and "MAX_CONSEC_HOLD" in bsrc
          and "FAILED_REJECTED_FOOTAGE" in bsrc and "rejected_footage_audit.json" in bsrc)
    check("no-vision-model exact/char still requires DETERMINISTIC same-show/era/CLIP/Face-ID",
          "_still_deterministic_ok" in osrc and "no arbitrary unverified still installed" in osrc
          and "DET_STILL_MIN_CLIP" in osrc)


def test_verifier_context_and_fallback():
    print("[stage-2] verifier storyboard context + honest fallback relevance-class")
    from vidlore.clipstudio import verify as V
    import vidlore.clipstudio.llm as L
    from vidlore.clipstudio import models as M
    from vidlore.clipstudio import ledger as LG
    from PIL import Image

    # (1) verify_frame feeds the beat's storyboard (expected_visual / scene_query / era) to the LLM,
    #     and instructs it that the right character alone is NOT enough (kills the "Jon = the most
    #     powerful man" rationalized keep).
    kf = Path(tempfile.mktemp(suffix=".jpg")); Image.new("RGB", (64, 36), (30, 30, 30)).save(kf)
    captured = {}

    def fake_complete(system=None, max_tokens=None, messages=None, eng_cfg=None, model=None):
        captured["text"] = messages[0]["content"][-1]["text"]
        return '{"verdict":"replace","confidence":0.3,"reason":"wrong scene"}'
    orig = L.complete
    try:
        L.complete = fake_complete
        V.verify_frame(str(kf), "the most powerful man in Westeros", "Arya Stark", "character", [],
                       types.SimpleNamespace(), model="m", is_specific=True,
                       expected_visual="an 11-year-old girl looks up at Tywin at the dinner table",
                       scene_query="Arya looks up at Tywin Harrenhal season 2", era_hint="S2")
    finally:
        L.complete = orig
    kf.unlink(missing_ok=True)
    t = captured.get("text", "")
    check("verifier prompt carries expected_visual", "looks up at Tywin at the dinner table" in t)
    check("verifier prompt carries scene_query + era", "Harrenhal season 2" in t and "S2" in t)
    check("verifier told right character alone is NOT enough",
          "correct character ALONE is not enough" in t or "different scene" in t.lower())

    # (2) action beats get a start/mid/end contact sheet; storyboard context is threaded from segs
    vsrc = (Path(__file__).resolve().parents[1] / "vidlore" / "clipstudio" / "verify.py").read_text()
    check("action contact-sheet builder exists + multiframe wired",
          "_action_contact_sheet" in vsrc and "multiframe=is_mf" in vsrc)
    check("verify passes storyboard to both primary + alternate checks",
          vsrc.count("expected_visual=getattr(_seg") >= 1 and "_verify_ctx(ashot.keyframe_path" in vsrc)

    # (3) honest relevance_class in the ledger
    tmp = Path(tempfile.mkdtemp(prefix="rc_"))
    proj = M.ClipProject(name="t", root=str(tmp))
    # a) exact beat, verifier-kept moving footage → exact_scene
    s0 = M.ClipSelection(segment_index=0, source_id="srcA", shot_index=1, in_point=1, out_point=4,
                         confidence=0.8)
    s0.verifier = {"status": "ok", "verdict": "keep"}
    # b) recovery still on exact beat → contextual_fallback (labelled at fill time)
    s1 = M.ClipSelection(segment_index=1, source_id="", shot_index=-1, in_point=0, out_point=0,
                         confidence=0.0)
    s1.image_meta = {"source": "source-frame-recovery", "relevance_class": "contextual_fallback"}
    # c) exact beat, verifier failed, no footage → exact_scene_missing
    s2 = M.ClipSelection(segment_index=2, source_id="srcB", shot_index=3, in_point=1, out_point=4,
                         confidence=0.5)
    s2.flag_reasons = ["verifier_failed", "exact_scene_missing"]
    proj.selections = [s0, s1, s2]
    g0 = M.ScriptSegment(index=0, text="x", quote="a line"); g0.visual_policy = "exact_scene"
    g1 = M.ScriptSegment(index=1, text="y"); g1.visual_policy = "exact_scene"
    g2 = M.ScriptSegment(index=2, text="z", quote="q"); g2.visual_policy = "exact_scene"
    LG.write_ledger(proj, [g0, g1, g2])
    recs = [json.loads(l) for l in proj.ledger_path.read_text().splitlines() if l.strip()]
    rc = {r["segment_index"]: r.get("relevance_class") for r in recs}
    check("verifier-kept exact footage → relevance_class exact_scene", rc.get(0) == "exact_scene")
    check("recovery still on exact beat → contextual_fallback", rc.get(1) == "contextual_fallback")
    check("verifier-failed exact beat → exact_scene_missing", rc.get(2) == "exact_scene_missing")


def test_source_quality_and_repetition():
    print("[stage-5] source quality (relevance>resolution) + repetition control")
    root = Path(__file__).resolve().parents[1] / "vidlore" / "clipstudio"
    hsrc = (root / "hd_download.py").read_text()
    msrc = (root / "match.py").read_text()
    vsrc = (root / "verify.py").read_text()
    bsrc = (root / "build.py").read_text()

    # (1) format sort is BITRATE-first (decoded-quality proxy), never fps/H.264-first
    check("format sort puts bitrate ahead of fps/codec",
          'res:{max_h},br' in hsrc and 'res:{max_h},fps,vcodec:h264' not in hsrc)

    # (2) _shot_unreadable second arm: no-highlight (luma_hi alone < 60) = unreadable
    from vidlore.clipstudio.match import _shot_unreadable
    def _sh(avg, hi):
        return types.SimpleNamespace(luma_avg=avg, luma_hi=hi)
    check("near-black no-highlight shot (avg 19, hi 49) is UNREADABLE", _shot_unreadable(_sh(19.0, 49.0)))
    check("torch-lit readable dark shot (avg 13, hi 140) stays readable",
          not _shot_unreadable(_sh(13.0, 140.0)))
    check("bright normal shot readable", not _shot_unreadable(_sh(80.0, 220.0)))
    check("old-index sentinel fails open", not _shot_unreadable(_sh(-1.0, -1.0)))

    # (3) verify_and_repair has a reuse ledger seeded from selections, capped per shot
    check("verify reuse ledger seeded from selections",
          "_reuse = _Counter()" in vsrc and "for _s in proj.selections:" in vsrc
          and "_reuse[(alt.source_id, alt.shot_index)] >= _reuse_cap" in vsrc)
    check("verify reuse counter updated on promotion",
          "_reuse[(alt.source_id, alt.shot_index)] += 1" in vsrc and "_reuse[_old_key] -= 1" in vsrc)

    # (4) whole-timeline aired-look cap + wired _near_recent (was dead code)
    check("whole-timeline look cap present",
          "_look_aired_count" in bsrc and "_LOOK_CAP" in bsrc and "VIDLORE_CLIPSTUDIO_LOOK_CAP" in bsrc)
    check("_near_recent is now WIRED into fresh-window selection (no longer dead)",
          "_near_recent(_win_embed(" in bsrc)
    check("look-cap prefers under-cap looks but never drops the last option",
          "_under if _under else _fresh" in bsrc)

    # (5) Gap 3: no near-black/unreadable beat airs; final sustained-black release gate
    check("build window picker skips unreadable shots (near-black never airs; no dark last-resort)",
          "_win_unreadable(w[0]" in bsrc and "near-black/unreadable NEVER airs" in bsrc
          and "_fresh_dark" not in bsrc)
    check("full window (not just in-point shot) validated for legibility + dark-clip removal",
          "Validate the ENTIRE rendered source window" in bsrc and "_clip_too_dark" in bsrc
          and "unreadable-clip removal" in bsrc)
    check("final-video sustained-black/legibility gate wired as a release gate",
          "def _final_video_black_gate" in bsrc and "_final_video_black_gate(result, work" in bsrc
          and "FAILED_BLACK_QA" in bsrc and "sustained unusable-dark" in bsrc)
    check("black gate distinguishes short fades from sustained unusable dark",
          "FINAL_BLACK_MINDUR" in bsrc and "short fades allowed" in bsrc)


def test_breakout_correctness():
    print("[stage-4] breakout: early-stop, strict Face-ID bypass, dup detect, audit fields")
    import vidlore.clipstudio.build as B

    # (1) _pick_breakout_stop: a short complete line + long silence ends at ITS stop, not at hi
    # a short complete line, then SILENCE (whisper transcribes no further words in the window)
    short = [("Anyone", 0.0, 0.4), ("can", 0.4, 0.7), ("be", 0.7, 0.9), ("killed.", 0.9, 1.4)]
    d = B._pick_breakout_stop(short, lo=3.0, hi=10.0)
    check("short complete line + silence → ~2s window (not 10s dead air)", 1.6 <= d <= 2.5)
    # a two-part exchange with a stop inside [lo,hi] keeps the later complete line
    two = [("They", 0.0, 0.3), ("say", 0.3, 0.6), ("he", 0.6, 0.8), ("cant", 0.8, 1.1),
           ("be", 1.1, 1.3), ("killed.", 1.3, 1.7),
           ("Do", 4.5, 4.7), ("you", 4.7, 4.9), ("believe", 4.9, 5.3), ("them?", 5.3, 5.8)]
    d2 = B._pick_breakout_stop(two, lo=3.0, hi=10.0)
    check("two-part exchange keeps the later complete line", 5.5 <= d2 <= 6.2)

    # (2) _verbatim_bypass_ok: a generic 3-4-word prefix does NOT bypass Face-ID; a strong,
    #     high-coverage distinctive quote does
    q_generic = ["do", "you", "know", "the", "story", "of", "harrenhal"]     # 7-word quote
    check("generic 4-word prefix of a long quote does NOT bypass Face-ID (0.57 cov)",
          not B._verbatim_bypass_ok(q_generic, 4))
    q_iconic = ["anyone", "can", "be", "killed"]
    check("iconic short line fully matched DOES bypass Face-ID (1.0 cov, content word)",
          B._verbatim_bypass_ok(q_iconic, 4))
    check("a 3-word run never bypasses (needs >= 4)", not B._verbatim_bypass_ok(q_iconic, 3))
    q_allfunc = ["do", "you", "did", "that"]                 # every token is a function word
    check("a 4-word all-function-word quote does NOT bypass (no content word)",
          not B._verbatim_bypass_ok(q_allfunc, 4))

    # (3) _narr_dup_run: narrator repeating the breakout line in an adjacent beat is detected
    from vidlore.clipstudio import models as M
    segs = [M.ScriptSegment(index=0, text="He asks where she is from"),
            M.ScriptSegment(index=1, text="watch what he knew and when"),
            M.ScriptSegment(index=2, text="she tells him that anyone can be killed today")]
    dup = B._narr_dup_run(["anyone", "can", "be", "killed"], segs, idx=1)
    check("narrator duplication detected (>=4-word overlap in a neighbour beat)", dup >= 4)
    nodup = B._narr_dup_run(["valar", "morghulis", "all", "men"], segs, idx=1)
    check("no false duplication when narration differs", nodup < 4)

    # (4) ordered coverage: in-order subsequence, not unordered word presence
    check("ordered coverage matches an in-order quote", B._ordered_coverage(
        ["anyone", "can", "be", "killed"], ["anyone", "can", "be", "killed"]) == 1.0)
    check("ordered coverage rejects a REVERSED/shuffled quote", B._ordered_coverage(
        ["alpha", "beta", "gamma"], ["gamma", "beta", "alpha"]) < 0.5)
    check("ordered coverage is 0 for an absent quote", B._ordered_coverage(
        ["dragon", "fire", "wall"], ["the", "king", "sits", "on", "a", "chair"]) == 0.0)
    # R4-1: contractions are CANONICALIZED (not discarded) on both sides
    check("contraction canonicalized: 'i've changed my mind' matches aired 'i have changed my mind'",
          B._ordered_coverage(["i've", "changed", "my", "mind"],
                              ["i", "have", "changed", "my", "mind"]) == 1.0)
    check("_canon_tokens expands don't/can't/possessive",
          B._canon_tokens(["don't"]) == ["do", "not"] and B._canon_tokens(["can't"]) == ["cannot"]
          and B._canon_tokens(["tywin's"]) == ["tywin"])
    # DEGENERATE quotes (no >=2 content words) are INSUFFICIENT EVIDENCE — cannot pass ANY audio,
    # including audio that literally contains the canonical phrase
    check("'don't' quote is insufficient — 0.0 even vs 'do not do it'",
          B._ordered_coverage(["don't"], ["do", "not", "do", "it"]) == 0.0)
    check("'can't' quote is insufficient — 0.0 even vs 'i cannot do this'",
          B._ordered_coverage(["can't"], ["i", "cannot", "do", "this"]) == 0.0)
    check("'i've' quote is insufficient — 0.0 even vs 'i have seen it'",
          B._ordered_coverage(["i've"], ["i", "have", "seen", "it"]) == 0.0)
    check("possessive-only quote insufficient vs UNRELATED audio",
          B._ordered_coverage(["tywin's"], ["the", "weather", "is", "nice", "today"]) == 0.0)
    check("generic 'do you know' prefix is insufficient (1 content word)",
          B._ordered_coverage(["do", "you", "know"], ["do", "you", "know", "the", "story"]) == 0.0)
    check("a real 2-content-word quote still passes with contractions present",
          B._ordered_coverage(["they", "can't", "be", "killed"],
                              ["they", "cannot", "be", "killed"]) == 1.0)

    # (5) wiring / audit fields / gates present
    bsrc = (Path(__file__).resolve().parents[1] / "vidlore" / "clipstudio" / "build.py").read_text()
    check("audit persists aired_transcript + line_coverage + speaker + standalone + dup",
          all(k in bsrc for k in ('"aired_transcript"', '"line_coverage"', '"speaker"',
                                  '"standalone_utterance"', '"narrator_duplication_words"')))
    check("aired_transcript is RE-ASR of the extracted audio (ground truth)",
          "_asr_wav_words(a)" in bsrc and "_atext if _aw else" in bsrc)
    check("genuine wrong-occurrence breakout is DROPPED on low ordered coverage",
          "BREAKOUT_MIN_COVERAGE" in bsrc and "ordered coverage" in bsrc)
    check("speaker is always 'unknown' (Face-ID proves visible, not speaking); visible_faces separate",
          '"speaker": "unknown"' in bsrc and '"visible_faces"' in bsrc
          and "who is VISIBLE, not who is SPEAKING" in bsrc)
    check("breakout re-ASR REQUIRED (no indexed-transcript fallback) + type-aware ordered coverage",
          "could not be re-ASR'd" in bsrc and "MIN_COVERAGE_VERBATIM" in bsrc
          and "MIN_COVERAGE_MINED" in bsrc)
    check("final-mix breakout QA enforces a PER-CANDIDATE coverage floor + documented ASR tolerance",
          "BREAKOUT_AUDIO_ASR_TOLERANCE" in bsrc and "accepted_coverage_floor" in bsrc
          and "final-mix ASR unavailable (no Whisper)" in bsrc)
    check("cold-open uses the same >=70% coverage floor (no not-_is_cold exemption)",
          "applies to COLD-OPENS too" in bsrc and "not _is_cold) and _ocov < _mincov" not in bsrc)
    check("candidate ORIGIN is explicit at creation, not inferred from _verbatim_strong",
          "_cand_origin[_k] = \"verbatim_quote\"" in bsrc and '"evidence_mined")' in bsrc
          and "_cand_origin.get((idx, src.id" in bsrc)
    check("final-mix breakout AUDIO QA fails CLOSED (probe failure = UNVERIFIED, not pass)",
          "BREAKOUT_AUDIO_QA" in bsrc and "UNVERIFIED (failing closed)" in bsrc
          and "could NOT be extracted" in bsrc)
    check("duplicated mid-video breakout is SKIPPED BY DEFAULT (VO-cut stays off)",
          'VIDLORE_CLIPSTUDIO_SKIP_DUP_BREAKOUT", "1"' in bsrc
          and 'not in ("0", "false", "no")' in bsrc.split("SKIP_DUP_BREAKOUT")[1][:120])


def test_release_gate_recovery_alignment():
    print("[release-gate] recovery sees every gate-blockable beat + hold roster mapping")
    from vidlore.clipstudio.orchestrate import _beat_is_unresolved
    from vidlore.clipstudio.build import _hold_scene_compat
    from vidlore.clipstudio import policy as _pol

    def seg_of(policy, ent="", kind=""):
        return types.SimpleNamespace(visual_policy=policy, required_entity=ent,
                                     required_kind=kind, quote="", is_specific_claim=False,
                                     text="", breakout_candidate=False)

    def sel_of(verdict="", img="", img_src="", src="src1"):
        return types.SimpleNamespace(
            verifier=({"status": "ok", "verdict": verdict} if verdict else {}),
            image_path=img, image_meta=({"source": img_src} if img_src else {}),
            source_id=src)

    # (1) the gate blocks ANY policy's verifier-rejected beat — recovery must see the same set.
    # Observed failure: 7 character_specific beats FATALed a finished render while recovery
    # reported zero unresolved (it filtered to exact_scene only).
    check("rejected character beat IS unresolved (was invisible to recovery)",
          _beat_is_unresolved(sel_of("replace"), seg_of("character_specific",
                                                        "Sansa Stark", "character"), _pol))
    check("rejected generic beat IS unresolved",
          _beat_is_unresolved(sel_of("replace"), seg_of("generic_filler"), _pol))
    check("kept character beat is NOT unresolved",
          not _beat_is_unresolved(sel_of("keep"), seg_of("character_specific"), _pol))
    check("rejected character beat WITH a still is NOT unresolved (gate skips image beats)",
          not _beat_is_unresolved(sel_of("replace", img="/x/a.jpg", img_src="web-portrait"),
                                  seg_of("character_specific"), _pol))
    check("exact beat requires a REAL still (web-portrait does not cover it)",
          _beat_is_unresolved(sel_of("replace", img="/x/a.jpg", img_src="web-portrait"),
                              seg_of("exact_scene"), _pol))
    check("exact beat with a source-frame still is covered",
          not _beat_is_unresolved(sel_of("replace", img="/x/a.jpg", img_src="source-frame"),
                                  seg_of("exact_scene"), _pol))
    check("missing selection: unresolved for exact only",
          _beat_is_unresolved(None, seg_of("exact_scene"), _pol)
          and not _beat_is_unresolved(None, seg_of("character_specific"), _pol))

    # (2) hold Face-ID identity check maps character↔actor through the roster: a
    # 'sophie turner' held frame IS 'sansa stark' (was rejected as a different person).
    def hseg(ent="", kind=""):
        return types.SimpleNamespace(
            scene_query="sansa stark red keep courtyard pleads", required_entity=ent,
            required_kind=kind, expected_visual="", text="")
    def hsel(identity, src="s1"):
        return types.SimpleNamespace(identity=identity, source_id=src)
    _c2a = {"Sansa Stark": "Sophie Turner", "Cersei Lannister": "Lena Headey"}
    ok_map, ev_map = _hold_scene_compat(
        hseg(), hseg("Sansa Stark", "character"), hsel("Sophie Turner"), hsel(""),
        single_scene=False, global_era="", char2actor=_c2a)
    check(f"hold accepts the SAME person across actor/character naming (roster-mapped) "
          f"[{ev_map.get('reason', 'ok')}]", ok_map)
    ok_wrong, _ = _hold_scene_compat(
        hseg(), hseg("Sansa Stark", "character"), hsel("Lena Headey"), hsel(""),
        single_scene=False, global_era="", char2actor=_c2a)
    check("hold still rejects a genuinely DIFFERENT person", not ok_wrong)
    ok_nomap, _ = _hold_scene_compat(
        hseg(), hseg("Sansa Stark", "character"), hsel("Sophie Turner"), hsel(""),
        single_scene=False, global_era="", char2actor=None)
    check("without a roster the old conservative rejection stands (fail-closed)", not ok_nomap)

    # (3) wiring: gate passes the roster; recovery prioritizes exact beats at the cap;
    # the surgical re-render tool runs the recovery stage (it FATALed without it)
    root = Path(__file__).resolve().parents[1]
    bsrc = (root / "vidlore" / "clipstudio" / "build.py").read_text(encoding="utf-8")
    osrc = (root / "vidlore" / "clipstudio" / "orchestrate.py").read_text(encoding="utf-8")
    tsrc = (root / "tools" / "rerender_project.py").read_text(encoding="utf-8")
    check("release gate passes the analysis roster into hold compat",
          "char2actor=_c2a_rf" in bsrc)
    check("recovery cap keeps GATE-VULNERABLE beats first (character-ranked)",
          "unresolved.sort(key=_rec_rank)" in osrc)
    check("surgical re-render tool runs bounded recovery before stills",
          "_recover_unresolved_beats(" in tsrc)

    # (4) era strings arrive in MIXED formats — canonical comparison, never string !=.
    # Observed: every still candidate rejected as "wrong era (beat S04E01 vs source season 4)"
    # (the SAME era), which release-blocked a finished render.
    from vidlore.clipstudio.verify import _era_conflict
    from vidlore.clipstudio.orchestrate import _deterministic_still_ok
    check("S04E01 vs 'season 4' is the SAME era (no conflict)",
          not _era_conflict("S04E01", "season 4"))
    check("S04E01 vs 'season 3' conflicts", _era_conflict("S04E01", "season 3"))
    check("same season, different EPISODE conflicts (S04E01 vs S04E10)",
          _era_conflict("S04E01", "S04E10"))
    check("a one-sided era never conflicts",
          not _era_conflict("season 4", "") and not _era_conflict("", "S04E01"))
    check("identical eras never conflict",
          not _era_conflict("season 4", "season 4") and not _era_conflict("S04E01", "s04e01"))
    _seg_e = types.SimpleNamespace(scene_query="", expected_visual="", text="",
                                   required_entity="", required_kind="")
    ok_e, why_e = _deterministic_still_ok(
        source_title="Game of Thrones Season 4 Tywin scenes", score=0.9, seg=_seg_e,
        faces=[], movie_toks={"game", "thrones"}, global_era="S04E01", single_scene=True,
        global_verified=True)
    check(f"deterministic still accepts a same-era source (S04E01 beat, 'Season 4' title) "
          f"[{why_e[:40]}]", ok_e)
    ok_e2, _ = _deterministic_still_ok(
        source_title="Game of Thrones Season 3 Tywin scenes", score=0.9, seg=_seg_e,
        faces=[], movie_toks={"game", "thrones"}, global_era="S04E01", single_scene=True,
        global_verified=True)
    check("deterministic still still rejects a genuinely WRONG season (hint corroborated)",
          not ok_e2)
    # …but an UNCORROBORATED hint may not reject on era grounds. "Wrong era" on the strength of a
    # guess is exactly how 'Game Of Thrones S03E10 Red Wedding Aftermath scene' -- the CORRECT
    # episode -- contributed zero shots to a video about S03E10. The other gates still apply.
    ok_e3, _ = _deterministic_still_ok(
        source_title="Game of Thrones Season 3 Tywin scenes", score=0.9, seg=_seg_e,
        faces=[], movie_toks={"game", "thrones"}, global_era="S04E01", single_scene=True,
        global_verified=False)
    check("an UNCORROBORATED era hint cannot reject a still on era grounds", ok_e3)

    # (5) Face-ID identities are ACTOR names; beats name CHARACTERS. The still checks and the
    # pool picker must map through the roster (a perfect Joffrey frame carries 'jack gleeson').
    from vidlore.clipstudio.orchestrate import entity_name_variants
    _c2a2 = {"Joffrey Baratheon": "Jack Gleeson"}
    _vars = entity_name_variants("joffrey baratheon", _c2a2)
    check("entity variants include the roster ACTOR name",
          {"jack", "gleeson"} in _vars and {"joffrey", "baratheon"} in _vars)
    check("actor-named beat maps back to the CHARACTER",
          {"joffrey", "baratheon"} in entity_name_variants("jack gleeson", _c2a2))
    _seg_j = types.SimpleNamespace(scene_query="", expected_visual="", text="",
                                   required_entity="Joffrey Baratheon", required_kind="character")
    ok_j, _ = _deterministic_still_ok(
        source_title="Game of Thrones Season 4", score=0.9, seg=_seg_j,
        faces=["Jack Gleeson"], movie_toks={"game", "thrones"}, global_era="S04E01",
        single_scene=True, char2actor=_c2a2)
    check("deterministic still accepts the character's ACTOR Face-ID (roster-mapped)", ok_j)
    ok_j2, _ = _deterministic_still_ok(
        source_title="Game of Thrones Season 4", score=0.9, seg=_seg_j,
        faces=["Lena Headey"], movie_toks={"game", "thrones"}, global_era="S04E01",
        single_scene=True, char2actor=_c2a2)
    check("deterministic still rejects a DIFFERENT person's face (roster active)", not ok_j2)

    # (6) pool picker prefers a Face-ID-confirmed shot of the required character over a
    # higher-CLIP shot with no face (3 no-face candidates previously all failed the checks)
    from vidlore.clipstudio import image_fallback as _imgfb
    from PIL import Image as _Img
    with tempfile.TemporaryDirectory() as _td:
        _kfa, _kfb = os.path.join(_td, "a.jpg"), os.path.join(_td, "b.jpg")
        _Img.new("RGB", (32, 18), (40, 40, 40)).save(_kfa)
        _Img.new("RGB", (32, 18), (90, 90, 90)).save(_kfb)

        def _stub_shot(kf, faces, ph):
            return types.SimpleNamespace(keyframe_path=kf, quality=0.9, phash=ph,
                                         face_ids=faces, subs_flag=0, luma_avg=60.0,
                                         luma_hi=200.0, corner_masks={}, ocr_text="",
                                         ocr_names=[])
        _pool = {("s1", 0): _stub_shot(_kfa, ["Jack Gleeson"], "a" * 16),
                 ("s2", 0): _stub_shot(_kfb, [], "f" * 16)}
        _seg_p = types.SimpleNamespace(scene_query="joffrey throne room", expected_visual="",
                                       text="")
        _old_rel = _imgfb._clip_relevance
        _imgfb._clip_relevance = lambda p, t: 0.9 if str(p).endswith("b.jpg") else 0.5
        try:
            _pick_f = _imgfb.pick_pool_still(_seg_p, _pool, set(), set(),
                                             want_faces=[{"jack", "gleeson"}])
            _pick_n = _imgfb.pick_pool_still(_seg_p, _pool, set(), set())
        finally:
            _imgfb._clip_relevance = _old_rel
        check("face-aware pick: the required character's shot outranks a higher-CLIP no-face shot",
              _pick_f is not None and _pick_f[1] == "s1")
        check("without want_faces the pick stays pure CLIP-ranked (no behavior change)",
              _pick_n is not None and _pick_n[1] == "s2")
        # scan_cap must never skip the pool's few face-confirmed shots: 3 no-face shots fill a
        # cap of 3, with the ONLY confirmed shot inserted LAST in dict order
        _pool_cap = {(f"n{i}", 0): _stub_shot(_kfb, [], f"{i:x}" * 16) for i in range(3)}
        _pool_cap[("hit", 0)] = _stub_shot(_kfa, ["Jack Gleeson"], "a" * 16)
        _imgfb._clip_relevance = lambda p, t: 0.6
        try:
            _pick_c = _imgfb.pick_pool_still(_seg_p, _pool_cap, set(), set(), scan_cap=3,
                                             want_faces=[{"jack", "gleeson"}])
        finally:
            _imgfb._clip_relevance = _old_rel
        check("face-confirmed shots are scanned FIRST (scan_cap can't hide them)",
              _pick_c is not None and _pick_c[1] == "hit")
    root2 = Path(__file__).resolve().parents[1]
    osrc2 = (root2 / "vidlore" / "clipstudio" / "orchestrate.py").read_text(encoding="utf-8")
    tsrc2 = (root2 / "tools" / "rerender_project.py").read_text(encoding="utf-8")
    check("PASS-1 pool picks are face-aware for character beats",
          "want_faces=_wf" in osrc2)
    check("re-render tool enables the vision still verifier (eng_cfg)",
          "eng_cfg=eng" in tsrc2)
    # (7) sub-HD LAST-RESORT still pool: low-res sources are retained in a separate pool and
    # retried ONLY when the HD pool produced nothing installable, under the SAME semantic
    # verification; watermarked/reaction sources stay excluded from BOTH pools (content safety).
    check("low-res sources retained as a last-resort still pool",
          "shots_lowres_by_key[(sh.source_id, sh.index)]" in osrc2
          and "LAST-RESORT sub-HD still" in osrc2)
    check("last-resort retry verifies semantically (vision verdict / deterministic gate)",
          "_vd == \"ok\" or (_vd == \"disabled\"" in osrc2)
    check("watermark check runs BEFORE the low-res split (never in either pool)",
          osrc2.index("_src_wm(_shots) or (_corner_on and _src_logo(_shots))")
          < osrc2.index("shots_lowres_by_key[(sh.source_id, sh.index)]"))
    check("low-res installs are labelled honestly for the ledger",
          "\"lowres_still\": bool(_lowres_pick)" in osrc2)
    # (8) rejected beats' stills are COVERAGE (never starved by the variety cap — an abstract
    # outro beat hit `cap 24` and release-blocked), and the recovery cap ranks CHARACTER beats
    # first (every observed blocker was one), skipping beats with no query material.
    check("verifier-rejected beats' stills are essential (variety cap can't starve them)",
          "essential = no_clip or weak" in osrc2)
    check("recovery ranks CHARACTER beats first and skips query-less beats",
          "_policy.CHARACTER: 0" in osrc2 and "can't be rediscovered" in osrc2)
    # (8b) a web-exact-scene still for an EXACT footage-gap beat is ESSENTIAL coverage and must
    # BYPASS the still cap — else 37 essential source-frame stills exhaust the cap-24 and the whole
    # web pass breaks on iteration 1, so barely-filmed backstory beats (Lyanna/Rhaegar/Aerys) get
    # no real still and the render re-airs verifier-REJECTED fan-film/AI on them (Arthur-Dayne render).
    check("essential web-exact-scene coverage bypasses the still cap (optional web still stays capped)",
          "_essential_web" in osrc2
          and "if not _essential_web and (src_filled + web_filled) >= cap:" in osrc2)

    # (9) a validated same-scene hold blocked ONLY by the frozen-frame duration caps becomes a
    # Ken-Burns MOTION hold (the caps exist against long FROZEN frames; a push-in is the tool's
    # own sanctioned still treatment). Any other blocker still release-blocks.
    from vidlore.clipstudio.build import _hold_block_reason
    _dur_block = _hold_block_reason(clips_present=True, has_predecessor=True, compat_ok=True,
                                    compat_reason="", consec_holds=0, hold_cap=1,
                                    beat_hold_dur=4.6, hold_total=0.0,
                                    single_cap=2.5, total_cap=3.0)
    _dur_only = _hold_block_reason(clips_present=True, has_predecessor=True, compat_ok=True,
                                   compat_reason="", consec_holds=0, hold_cap=1,
                                   beat_hold_dur=0.0, hold_total=0.0,
                                   single_cap=2.5, total_cap=3.0)
    _not_dur = _hold_block_reason(clips_present=True, has_predecessor=True, compat_ok=False,
                                  compat_reason="scene tokens differ", consec_holds=0, hold_cap=1,
                                  beat_hold_dur=0.0, hold_total=0.0,
                                  single_cap=2.5, total_cap=3.0)
    check("duration-capped hold is detected as DURATION-ONLY (zeroed re-check passes)",
          _dur_block is not None and "cap" in _dur_block and _dur_only is None)
    check("a compat-failed hold is NOT duration-only (still release-blocks)",
          _not_dur is not None)
    bsrc2 = (root2 / "vidlore" / "clipstudio" / "build.py").read_text(encoding="utf-8")
    check("gate wires the Ken-Burns motion hold for duration-only blocks",
          "_kenburns_hold(Path(_last_clean_r)" in bsrc2
          and "same-scene Ken-Burns MOTION hold" in bsrc2)
    # CONTRACT CHANGED, deliberately. Motion holds used to be exempt from the frozen-seconds caps
    # on the theory that a Ken-Burns push-in is not really a freeze. A frame audit priced that: a
    # 6.68s hold shipped while rejected_footage_audit.json reported total_hold_seconds 2.38, and
    # two adjacent beats spent 10.7 CONSECUTIVE seconds on one web JPEG — 2.8x the median beat and
    # longer than any real shot in the film. The push-in did not make it stop being a held frame;
    # it only made the caps blind to it. Every held second now counts.
    check("EVERY held second counts against the caps, motion or frozen",
          "_hold_total += _beat_hold_dur" in bsrc2
          and "if not _motion_hold:                      # only FROZEN" not in bsrc2)

    # (10) cast-interview / press-junket recap sources never enter any pool ('Richard Madden
    # Relives the Red Wedding' matched a Tyrion beat and release-blocked it)
    from vidlore.clipstudio.discover import is_unwanted_source_title as _unw
    check("actor-interview recap titles are unwanted sources",
          _unw("Richard Madden Relives the Red Wedding | Game of Thrones")
          and _unw("Jaime Lannister Breaks Down His Best Scenes")
          and _unw("Sophie Turner Looks Back at Sansa's Journey"))
    check("REAL scene uploads still pass the source gate",
          not _unw("Tywin Lannister Dismisses King Joffrey | Game Of Thrones")
          and not _unw("The Scene Tywin Lannister proved his power | GoT S3E10")
          and not _unw("Tyrion kills Tywin Lannister - Game of Thrones"))
    # (11) candidate breadth: the first ACCEPTED frame for a starved beat was #4 — the cap of 3
    # release-blocked it every run (measured offline: 6 candidates → keeps at #2/#3/#4/#6)
    check("still search offers ≥6 verified candidates (env-tunable, default 6)",
          "VIDLORE_CLIPSTUDIO_STILL_CANDIDATES" in osrc2
          and "for _ in range(_cand_n)" in osrc2 and "for _ in range(3)" not in osrc2)
    # (12) a vision-API outage (ALL candidates 'unverified', ZERO semantic rejections) gets ONE
    # bounded backoff retry — a 2-minute blip must not permanently starve a beat's coverage;
    # any real 'reject' in the batch means the API works and verdicts stand (no retry)
    check("vision-outage signature gets a bounded backoff retry (fail-closed if it persists)",
          "_unv > 0 and _rej == 0" in osrc2
          and "VIDLORE_CLIPSTUDIO_VISION_RETRY_SEC" in osrc2)


def test_music_dynamics_wiring():
    print("[stage-3] natural music dynamics — envelope + wiring (VO never touched)")
    import vidlore.clipstudio.build as B
    # (1) pure expr: breakout dip + reveal boost, no-op when empty, smooth trapezoid (2-arg min)
    e = B._music_envelope_expr([(4.0, 7.0)], [(9.0, 11.0)])
    check("expr has breakout dip + reveal boost", ("1-0.85" in e) and ("1+0.15" in e))
    check("expr uses smooth 2-arg-min trapezoid ramps", "min(clip(" in e)
    check("empty windows → no-op 1.0", B._music_envelope_expr([], []) == "1.0")
    # (2) wiring: build shapes the resolved music and passes music=_music_track (not raw resolve)
    bsrc = (Path(__file__).resolve().parents[1] / "vidlore" / "clipstudio" / "build.py").read_text()
    check("build shapes music before assemble",
          "_shape_music_envelope(" in bsrc and "music=_music_track" in bsrc)
    check("reveal windows keyed on reveal/climax beats",
          '_role in ("reveal", "climax")' in bsrc)
    check("music shaping is engine-side-only (uploaded VO never touched)",
          "uploaded voiceover is never altered" in bsrc or "never touched" in bsrc)
    check("music dynamics env-gated", "VIDLORE_CLIPSTUDIO_MUSIC_DYNAMICS" in bsrc)


def test_discovery_query_stratification():
    print("[stage-2] discovery: timeline-stratified query coverage + equivalent-query dedup")
    import os, re
    from vidlore.clipstudio.discover import build_queries
    from vidlore.clipstudio.analyze import ScriptAnalysis
    from vidlore.clipstudio import models as M
    ana = ScriptAnalysis(movie_title="Game of Thrones")
    ana.characters = []; ana.locations = []; ana.key_scenes = []
    # 90 EXACT beats spread across the timeline, each a unique scene
    segs = []
    for i in range(90):
        s = M.ScriptSegment(index=i, text=f"beat {i}",
                            scene_query=f"unique moment number {i} distinct scene {i}")
        s.visual_policy = "exact_scene"
        segs.append(s)
    # + three EQUIVALENT queries (same tokens, reordered) → must collapse to ONE
    for j, q in enumerate(["Arya dinner Harrenhal Tywin", "Tywin Harrenhal dinner Arya",
                           "dinner at Harrenhal with Tywin and Arya"]):
        s = M.ScriptSegment(index=45, text="dup", scene_query=q)
        s.visual_policy = "exact_scene"
        segs.append(s)

    qs = build_queries(ana, segs)
    j = " || ".join(qs).lower()
    # (1) with cap scaling, EVERY beat's scene is searched — including the LAST beats (no starvation)
    check("late beats are searched (no segment-order starvation)",
          "number 88" in j and "number 89" in j and "number 2 " in j)
    # (2) equivalent scene queries collapse to a single search
    n_harrenhal = sum(1 for x in qs if "harrenhal" in x.lower())
    check("equivalent scene queries deduped to one", n_harrenhal == 1)

    # (3) when the HARD cap forces truncation, coverage is STRATIFIED across early/middle/late,
    #     never all-early (the old flat truncation dropped every late exact beat)
    os.environ["VIDLORE_CLIPSTUDIO_QUERY_CAP"] = "24"
    os.environ["VIDLORE_CLIPSTUDIO_QUERY_CAP_MAX"] = "24"
    try:
        qs2 = build_queries(ana, segs)
    finally:
        os.environ.pop("VIDLORE_CLIPSTUDIO_QUERY_CAP", None)
        os.environ.pop("VIDLORE_CLIPSTUDIO_QUERY_CAP_MAX", None)
    j2 = " || ".join(qs2).lower()
    nums = [int(re.search(r"number (\d+)", x).group(1)) for x in qs2 if re.search(r"number (\d+)", x)]
    check("hard cap honored", len(qs2) <= 24)
    check("truncation is stratified: early AND middle AND late beats all represented",
          bool(nums) and min(nums) < 30 and any(30 <= n < 60 for n in nums) and max(nums) >= 60)


def test_fps_and_ad_protection():
    print("[stage-1] FPS time-neutrality + ad/branding release gate")
    import vidlore.clipstudio.build as B
    from vidlore.clipstudio.match import clean_cut_window
    root = Path(__file__).resolve().parents[1] / "vidlore" / "clipstudio"
    bsrc = (root / "build.py").read_text(encoding="utf-8")
    msrc = (root / "match.py").read_text(encoding="utf-8")

    # (1) Ken-Burns is FPS time-neutral: fps=30 is normalized BEFORE zoompan so 1 output second ==
    #     1 source second for every source frame rate (stops the non-30fps source over/under-consume
    #     that ran the cut into the Max/WarnerMedia outro slate).
    kb = B._ken_burns_filter(6.0, zoom_to=1.10)
    check("ken-burns normalizes fps BEFORE zoompan (time-neutral)",
          kb.startswith("fps=30,") and kb.index("fps=30") < kb.index("zoompan"))

    # (2) _PROMO_RX: strong promo/CTA/outro tokens only — never ordinary narration words
    for s in ["max PLANS START AT $9.99/MONTH mox.com", "subscribe now", "only on HBO Max",
              "www.foo.tv", "discover more", "now streaming"]:
        check(f"promo-rx flags {s!r}", bool(B._PROMO_RX.search(s)))
    for s in ["watch how she moves", "follow her north", "the dragons happened",
              "he catches her by the throat"]:
        check(f"promo-rx ignores narration {s!r}", not B._PROMO_RX.search(s))

    # (3) branding probe covers the clip TAIL (end-slates live there), not just the head
    offs = B._branding_probe_offsets(8.0)
    check("branding probe samples the TAIL", max(offs) >= 7.0 and any(o >= 6.0 for o in offs))
    check("branding probe head-only fallback on unknown duration",
          B._branding_probe_offsets(0) == [0.2, 0.8, 1.6, 2.6, 3.6])

    # (4) card-geometry: flat card high, photographic frame low
    try:
        from PIL import Image
        import numpy as _np
        _flat = Path(tempfile.mktemp(suffix=".png")); _rnd = Path(tempfile.mktemp(suffix=".png"))
        Image.new("RGB", (960, 540), (12, 40, 150)).save(_flat)
        Image.fromarray((_np.random.RandomState(5).rand(540, 960, 3) * 255).astype("uint8")).save(_rnd)
        uf, ur = B._frame_card_uniformity(_flat), B._frame_card_uniformity(_rnd)
        _flat.unlink(missing_ok=True); _rnd.unlink(missing_ok=True)
        check("card-uniformity: flat slate >= 0.9, photographic < 0.25", uf >= 0.9 and ur < 0.25)
    except Exception as _e:
        check(f"card-uniformity PIL test ran ({_e})", False)

    # (5) clean_cut_window keeps a safety MARGIN before an ocr-text (branding/card) shot, but NOT
    #     before a subtitle shot (subs keep exact bounds — long-standing shortening behavior)
    def _shot(i, s, e, **kw):
        d = dict(index=i, start=s, end=e, subs_flag=0, luma_avg=60.0, luma_hi=200.0,
                 corner_masks={}, ocr_text="", ocr_names=[], keyframe_path="")
        d.update(kw)
        return types.SimpleNamespace(**d)
    A = _shot(0, 0.0, 2.0)
    Btext = _shot(1, 2.0, 5.0, ocr_text="max PLANS START AT WATCH")     # ocr-text card
    _, n1_txt, act_t, _ = clean_cut_window([A, Btext], 0.0, 2.4, 1.2, anchor=(0.0, 2.0))
    check("ocr-text neighbour → clean window ends with a safety margin before it",
          act_t == "shortened" and n1_txt <= 2.0 - 0.3 + 1e-6)
    Bsub = _shot(1, 2.0, 5.0, subs_flag=1)                              # burned subtitle
    _, n1_sub, act_s, _ = clean_cut_window([A, Bsub], 0.0, 2.4, 1.2, anchor=(0.0, 2.0))
    check("subs neighbour → exact boundary kept (no margin, unchanged behavior)",
          act_s == "shortened" and abs(n1_sub - 2.0) <= 1e-6)
    # avatar badges fade in/out like text cards, so overlay-badge shares the ocr-text margin;
    # subs/unreadable/logo keep exact bounds (the over-trim protection this test exists for)
    check("edge-margin applies to ocr-text + overlay-badge reasons only",
          'r in ("ocr-text", "overlay-badge")' in msrc and 'r == "subs"' not in msrc)
    Bbadge = _shot(1, 2.0, 5.0, ocr_text="Jacquelyn Sutphen",
                   corner_masks={"bl": "300c0381801ff00c000000600000037fffff9bfffffcdeffbfe6"
                                       "0000003ffffff9ffffffc0000000"})
    _, n1_bdg, act_b, _ = clean_cut_window([A, Bbadge], 0.0, 2.4, 1.2, anchor=(0.0, 2.0))
    check("overlay-badge neighbour → clean window ends with the same safety margin",
          act_b == "shortened" and n1_bdg <= 2.0 - 0.3 + 1e-6)

    # (6) the final-video ad gate is WIRED into build_video as a release gate
    check("final-video ad gate wired into build_video",
          "_final_video_ad_gate(result, work, _ocr_eng" in bsrc
          and "def _final_video_ad_gate" in bsrc and "FAILED_AD_QA" in bsrc)
    # (7) ad gate FAILS CLOSED (no OCR / zero frames / excessive errors → unverified → block unless
    #     explicit override) and has a SECOND non-uniform layout-heavy detection path
    check("ad gate fails closed on unverifiable scans",
          '"status": "unverified"' in bsrc and "AD_GATE_OVERRIDE" in bsrc
          and "zero decoded scan frames" in bsrc and "excessive OCR errors" in bsrc)
    check("ad gate has a second (layout-heavy, non-uniform card) detection path",
          "_ocr_layout_metrics" in bsrc and "layout_heavy" in bsrc and "strong_single" in bsrc)

    # (8) COMPLETE-TIMELINE coverage: the scan must reach the FINAL timestamp, not merely 95% of the
    #     frames (a 5% shortfall on a 15-min video silently drops the last ~45s where outros live)
    stride, dur = 0.5, 900.0
    full = int(dur / stride) + 1
    check("full-timeline scan is covered (last frame reaches the end)",
          B._scan_coverage_reason(full, stride, dur) is None)
    n_2s = int((dur - 2.0) / stride) + 1                 # final 2 seconds missing
    check("final 2s missing → NOT covered (fails closed)",
          B._scan_coverage_reason(n_2s, stride, dur) is not None)
    n_4pct = int((dur * 0.96) / stride) + 1              # final 4% missing
    check("final 4% missing → NOT covered (fails closed)",
          B._scan_coverage_reason(n_4pct, stride, dur) is not None)
    n_half = int((dur - 1.0) / stride) + 1               # final 1.0s missing (> one stride)
    check("final 1.0s missing → NOT covered (LITERAL within-one-stride guarantee)",
          B._scan_coverage_reason(n_half, stride, dur) is not None)
    check("tolerance is ~one stride (0.65), NOT max(1s, 2 strides)",
          "stride + eps" in bsrc and "max(1.0, 2.0 * stride)" not in bsrc)
    check("gates also PTS-check the FINAL frame decodes (not just the count)",
          "_final_timestamp_reachable" in bsrc and "does not decode — tail not covered" in bsrc)
    check("the old 5%-tolerance would have WRONGLY passed the 2s-short case",
          n_2s >= full * 0.95)                           # proves the heuristic was insufficient
    check("zero duration → unverified", B._scan_coverage_reason(full, stride, 0.0) is not None)
    check("rc!=0 tolerated ONLY with proven full coverage",
          B._scan_coverage_reason(full, stride, dur, rc=1) is None
          and B._scan_coverage_reason(n_2s, stride, dur, rc=1) is not None)
    check("dense ad rescan is tri-state (confirmed/clean/unverified) → unverified blocks",
          "'unverified'" in bsrc and "dense rescan around a real promo candidate" in bsrc
          and "cannot rule it out (fail-closed)" in bsrc)
    check("BOTH ad + black gates use the same complete-timeline coverage guarantee",
          bsrc.count("_scan_coverage_reason(len(frames), stride, dur") >= 2)


def main():
    test_fps_and_ad_protection()
    test_release_gate_recovery_alignment()
    test_source_quality_and_repetition()
    test_breakout_correctness()
    test_music_dynamics_wiring()
    test_discovery_query_stratification()
    test_verifier_context_and_fallback()
    test_era_policy_and_still_verification()
    test_verifier_promotion_rewrites_beat_windows()
    test_budget_loop_survives_plan_beats_failure()
    test_find_produced_video()
    test_cut_checks_ffmpeg_rc()
    test_discover_budget_and_cfg_copy()
    test_match_scoring_fixes()
    test_pipeline_error_and_gates()
    test_index_cache_and_atomic_saves()
    test_build_media_fixes()
    test_ingest_and_web_hygiene()
    test_low_severity_fixes()
    test_round2_review_fixes()
    test_breakout_intelligence()
    test_breakout_audio_gate()
    test_text_gate()
    test_character_relevance()
    test_image_fallback()
    test_ai_voice()
    test_exact_scene_recall_fixes()
    test_cpu_parallelism()
    test_deepseek_primary_brain()
    test_caption_sync_per_scene_tolerant()
    test_source_budget_scales_with_script()
    test_generic_beat_filler_leniency()
    test_dark_patch_prepass()
    test_breakout_qa_whisper_generator()
    test_character_present_unconfirmed()
    test_verify_only_indices_subset()
    test_unified_visual_policy()
    test_final_image_policy()
    test_image_policy_edge_gaps()
    test_reaction_still_pool_gate()
    test_nonscene_footage_gate()
    test_discovery_plural_gates_and_purge()
    test_breakout_evidence_mining_reachable()
    test_intro_coldopen_breakout()
    test_breakout_window_commentary_gate()
    test_coldopen_vocut()
    test_breakout_era_scene_gate()
    test_burned_text_black_and_recap_gates()
    test_corner_logo_and_quality_gates()
    test_clean_copy_arbitration()
    test_multiframe_shot_flags()
    test_cut_window_validation()
    test_wqc_moment_preservation()
    test_gemini_timeout_hardening()
    test_breakout_atomic_composition()
    test_caption_presets()
    test_caption_correctness()
    test_caption_motion_presets()
    test_breakout_caption_layout()
    test_caption_pixel_bbox()
    print(f"\n{PASS} passed · {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
