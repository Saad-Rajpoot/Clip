"""Adversarial tests for the P0 performance work: the speedups must be DECISION-NEUTRAL.

Covers:
  * verdict_fingerprint: the venue_fallback question-variant is in the key, and venue=False
    keys are byte-identical to pre-change keys (golden re-derivation) so no cache is orphaned;
  * rung caching: a repeated verify pass (the review-draft / resume scenario) reproduces the
    SAME decisions with ZERO fallback vision calls, and strict vs lenient questions on the
    same frame never share a key;
  * cache hygiene: corrupt cache files are ignored; transport failures are never stored;
  * prefetch concurrency: workers=4 produces byte-identical decisions to workers=1, and a
    failing backend under workers=4 aborts the pool to the serial path with the breaker
    contract intact;
  * pick_pool_still: the persisted-embedding path returns identical picks/scores to the live
    CLIP path, including the sliding scan_cap replay across successive candidate rounds.

    python3 tests/test_perf_neutral_caching.py

No network, no LLM, no ffmpeg.
"""
import hashlib
import json
import os
import shutil
import sys
import tempfile
from types import SimpleNamespace as NS

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vidlore.clipstudio import verify as V                      # noqa: E402
from vidlore.clipstudio import image_fallback as IF             # noqa: E402

FAILS = []
BASE = dict(src_hash="a", source_id="s1", shot_start=1.0, shot_end=3.0,
            beat_text="Tywin dismisses Joffrey", required_entity="Tywin Lannister",
            required_kind="character", expected_visual="Tywin at the council table",
            scene_query="Game of Thrones small council", era="S03E10",
            visual_policy="exact_scene", is_specific=True,
            faceid_names=["Charles Dance"], multiframe=False, image_id="kf:abc",
            model="gemini:gemini-2.5-flash:apikey")


# ---------------------------------------------------------------------------
# K1 — the venue question-variant is part of the verdict identity
# ---------------------------------------------------------------------------
def test_venue_fallback_is_in_the_key_and_false_preserves_legacy_keys():
    fp0 = V.verdict_fingerprint(**BASE)
    assert fp0 == V.verdict_fingerprint(**BASE, venue_fallback=False), \
        "venue_fallback=False must be byte-identical to the pre-change key (no cache orphaned)"
    assert fp0 != V.verdict_fingerprint(**BASE, venue_fallback=True), \
        "the venue holding-image question is a DIFFERENT question — it must never share a key"
    # GOLDEN re-derivation of the legacy algorithm (main's exact part order): a change to the
    # part list or separators silently orphans every existing verdict_cache.json.
    h = hashlib.sha256()
    for part in ("a", "s1", "1.000", "3.000", "Tywin dismisses Joffrey", "tywin lannister",
                 "character", "Tywin at the council table", "Game of Thrones small council",
                 "s03e10", "exact_scene", "1", "charles dance", "sf", "kf:abc",
                 "gemini:gemini-2.5-flash:apikey", V.PROMPT_VERSION, V.SHEET_VERSION):
        h.update(str(part).encode("utf-8", "replace"))
        h.update(b"\x1f")
    assert fp0 == h.hexdigest()[:32], "venue=False key must equal the legacy derivation exactly"


def test_corrupt_cache_file_is_ignored_not_fatal():
    tmp = tempfile.mkdtemp()
    try:
        proj = NS(root=tmp)
        with open(os.path.join(tmp, "verdict_cache.json"), "w") as fh:
            fh.write("{not json at all")
        assert V._load_verdict_cache(proj) == {}
        with open(os.path.join(tmp, "verdict_cache.json"), "w") as fh:
            fh.write('["a list, not a dict"]')
        assert V._load_verdict_cache(proj) == {}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# K2 — rung caching: a repeated pass replays decisions with zero fallback calls
# ---------------------------------------------------------------------------
def _mini_project_with_alternates(tmp, n_beats=6):
    """n_beats=6 -> 36-call fallback ladder; failure tests pass n_beats>=10 so the
    8-consecutive-error breaker can actually trip."""
    from vidlore.clipstudio.models import (ClipProject, ScriptSegment, ClipSelection,
                                           SourceVideo, Shot, ClipCandidate)
    proj = ClipProject(name="t", root=str(tmp))
    proj.ensure_dirs()
    media = os.path.join(tmp, "src.mp4")
    with open(media, "wb") as fh:
        fh.write(b"\0" * 2048)
    proj.sources = [SourceVideo(id="s1", url="u", title="Game of Thrones S03E10 council",
                                permission="owner", status="ok", local_path=media)]
    segs, shots = [], []
    n_shots = n_beats * 3                                 # primary + 2 alternates per beat
    for j in range(n_shots):
        kf = os.path.join(tmp, f"kf{j}.jpg")
        with open(kf, "wb") as fh:
            fh.write(b"\xff\xd8\xff" + bytes([j]))        # distinct bytes -> distinct image_id
        shots.append(Shot(source_id="s1", index=j, start=float(j), end=float(j) + 2.0,
                          keyframe_path=kf))
    for i in range(n_beats):
        segs.append(ScriptSegment(index=i, text=f"Tywin dismisses Joffrey {i}",
                                  required_entity="Tywin Lannister", required_kind="character",
                                  visual_policy="exact_scene", is_specific_claim=True))
        sel = ClipSelection(segment_index=i, source_id="s1", shot_index=i * 3,
                            in_point=float(i * 3), out_point=float(i * 3) + 2.0, confidence=0.8)
        sel.alternates = [ClipCandidate(segment_index=i, source_id="s1", shot_index=i * 3 + k,
                                        score=0.5, in_point=float(i * 3 + k),
                                        out_point=float(i * 3 + k) + 2.0)
                          for k in (1, 2)]
        proj.selections.append(sel)
    proj.meta["analysis"] = {"video_type": "single_scene", "episode_hint": "S03E10",
                             "episode_hint_verified": True, "characters": [], "actors": []}
    return proj, segs, shots


_REJECT_ALL = {"verdict": "replace", "correct_subject_visible": False,
               "matches_narration": False, "wrong_subject_visible": False,
               "quality_ok": True, "confidence": 0.8, "reason": "t"}


def _drive(tmp, verdict_fn, *, workers=None, n_beats=6):
    from vidlore.clipstudio.config import ClipConfig
    proj, segs, shots = _mini_project_with_alternates(tmp, n_beats)
    by = {(s.source_id, s.index): s for s in shots}
    calls = {"n": 0}

    def fake_verify_frame(*a, **k):
        calls["n"] += 1
        return verdict_fn(calls["n"])

    def lookup(p):
        def get(sid, ix):
            return by.get((sid, ix))
        get.all_shots = lambda sid: [s for s in shots if s.source_id == sid]
        return get

    orig_vf, orig_lookup = V.verify_frame, V._shot_lookup
    orig_env = os.environ.get("VIDLORE_CLIPSTUDIO_VERIFY_ACTION_SHEET")
    orig_wk = os.environ.get("VIDLORE_CLIPSTUDIO_VERIFY_WORKERS")
    os.environ["VIDLORE_CLIPSTUDIO_VERIFY_ACTION_SHEET"] = "0"
    if workers is not None:
        os.environ["VIDLORE_CLIPSTUDIO_VERIFY_WORKERS"] = str(workers)
    V.verify_frame = fake_verify_frame
    V._shot_lookup = lookup
    try:
        summ = V.verify_and_repair(proj, segs, ClipConfig(),
                                   NS(anthropic_model="m", anthropic_key="k"), progress=None)
    finally:
        V.verify_frame, V._shot_lookup = orig_vf, orig_lookup
        for var, old in (("VIDLORE_CLIPSTUDIO_VERIFY_ACTION_SHEET", orig_env),
                         ("VIDLORE_CLIPSTUDIO_VERIFY_WORKERS", orig_wk)):
            if old is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = old
    return summ, calls["n"], proj


def _decision_state(proj):
    """Everything editorial about the outcome — what parity must preserve."""
    out = []
    for s in sorted(proj.selections, key=lambda x: x.segment_index):
        out.append((s.segment_index, s.source_id, s.shot_index, round(s.in_point, 3),
                    round(s.out_point, 3), bool(s.flagged), tuple(sorted(s.flag_reasons or [])),
                    (s.verifier or {}).get("verdict"), (s.verifier or {}).get("downgraded"),
                    (s.verifier or {}).get("relevance_class")))
    return out


def test_repeated_pass_replays_decisions_from_cache_with_zero_calls():
    """The review-draft / resume scenario: same questions -> answers from the cache, decisions
    identical, ZERO vision calls. Before P0-2 only the primary verdicts replayed; the fallback
    chain (strict promotion x2 alternates + lenient re-ask per beat) re-paid every time."""
    t1 = tempfile.mkdtemp()
    t2 = tempfile.mkdtemp()
    try:
        summ1, n1, proj1 = _drive(t1, lambda n: dict(_REJECT_ALL))
        # 6 beats x (1 primary + 2 strict-promotion + 2 contextual-promotion + 1 lenient) = 36
        assert n1 == 36, f"cold pass must pay the full ladder, paid {n1}"
        cache1 = json.load(open(os.path.join(t1, "verdict_cache.json")))
        assert len(cache1) == 36, f"every distinct question must be cached once, got {len(cache1)}"

        # same project rebuilt bit-identically elsewhere + the cache carried over = a resume
        shutil.copy(os.path.join(t1, "verdict_cache.json"), os.path.join(t2, "verdict_cache.json"))
        summ2, n2, proj2 = _drive(t2, lambda n: dict(_REJECT_ALL))
        assert n2 == 0, f"a warm pass must make ZERO vision calls, made {n2}"
        assert _decision_state(proj1) == _decision_state(proj2), \
            "cached replay must reproduce the decisions byte-for-byte"
        assert summ2["failed"] == summ1["failed"]
    finally:
        shutil.rmtree(t1, ignore_errors=True)
        shutil.rmtree(t2, ignore_errors=True)


def test_strict_and_lenient_questions_never_share_a_key():
    """The same alternate frame is asked strictly (promotion) and leniently (downgrade) —
    opposite rules, so 4 distinct cached entries per alternate pair, never 2."""
    tmp = tempfile.mkdtemp()
    try:
        _, n, _ = _drive(tmp, lambda n: dict(_REJECT_ALL))
        cache = json.load(open(os.path.join(tmp, "verdict_cache.json")))
        assert len(cache) == n == 36, \
            "strict/lenient collisions would cache fewer entries than calls"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_transport_failures_are_never_cached():
    tmp = tempfile.mkdtemp()
    try:
        # primary succeeds ('replace'), every fallback call errors (None)
        summ, n, _ = _drive(tmp, lambda n: dict(_REJECT_ALL) if (n % 6) == 1 else None)
        cache = json.load(open(os.path.join(tmp, "verdict_cache.json")))
        assert all(V._verdict_schema_ok(v) for v in cache.values()), \
            "a None/transport outcome must never be stored"
        assert len(cache) < n, "errored calls must not add entries"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# K3 — prefetch concurrency parity + failure fallback
# ---------------------------------------------------------------------------
def test_workers4_is_decision_identical_to_serial():
    t1 = tempfile.mkdtemp()
    t2 = tempfile.mkdtemp()
    keep = {"verdict": "keep", "correct_subject_visible": True, "matches_narration": True,
            "wrong_subject_visible": False, "contradicts_narration": False,
            "specific_enough": True, "quality_ok": True, "confidence": 0.9, "reason": "t"}
    try:
        summ1, n1, proj1 = _drive(t1, lambda n: dict(keep), workers=1)
        summ2, n2, proj2 = _drive(t2, lambda n: dict(keep), workers=4)
        assert _decision_state(proj1) == _decision_state(proj2), \
            "the prefetch pool must not change any decision"
        assert summ1["failed"] == summ2["failed"] == 0
        assert n1 == n2 == 6, "healthy pass: one call per beat either way"
        c1 = json.load(open(os.path.join(t1, "verdict_cache.json")))
        c2 = json.load(open(os.path.join(t2, "verdict_cache.json")))
        assert set(c1.keys()) == set(c2.keys()), "same questions, same keys, either path"
    finally:
        shutil.rmtree(t1, ignore_errors=True)
        shutil.rmtree(t2, ignore_errors=True)


def test_workers4_backend_failure_aborts_pool_and_keeps_breaker_contract():
    t1 = tempfile.mkdtemp()
    t2 = tempfile.mkdtemp()
    try:
        s1, n1, p1 = _drive(t1, lambda n: None, workers=1, n_beats=12)
        s2, n2, p2 = _drive(t2, lambda n: None, workers=4, n_beats=12)
        assert s1["verifier_down"] is True and s2["verifier_down"] is True, \
            "a dead backend must open the breaker under either mode"
        assert s1["failed"] == s2["failed"], "fail-closed outcome identical"
        assert _decision_state(p1) == _decision_state(p2)
    finally:
        shutil.rmtree(t1, ignore_errors=True)
        shutil.rmtree(t2, ignore_errors=True)


# ---------------------------------------------------------------------------
# K4 — pick_pool_still: persisted embeddings == live CLIP, sliding scan preserved
# ---------------------------------------------------------------------------
def _fake_pool(tmp, n=10):
    """Real (tiny) JPEGs: the LIVE path PIL-opens the keyframe before embedding, so the
    equality test must exercise a genuine decode, not an unreadable stub file. Returns the
    matrix, the shot map, and a certified manifest row_map (P1.3 contract)."""
    import numpy as np
    from PIL import Image
    rng = __import__("random").Random(7)
    mat = np.zeros((n, 8), dtype="float32")
    shots = {}
    row_map = {}
    for j in range(n):
        v = np.array([rng.random() for _ in range(8)], dtype="float32")
        mat[j] = v / (float(np.linalg.norm(v)) + 1e-8)
        kf = os.path.join(tmp, f"pool{j}.jpg")
        Image.new("RGB", (8, 8), (j * 20 % 255, 30, 40)).save(kf, "JPEG")
        shots[("s1", j)] = NS(source_id="s1", index=j, keyframe_path=kf, quality=1.0,
                              phash="", face_ids=[], ocr_names=[], ocr_text="",
                              embed_row=j, luma_avg=-1.0, luma_hi=-1.0, subs_frac=-1.0)
        row_map[str(j)] = {"shot": j, "kf": os.path.basename(kf),
                           "kf_md5": hashlib.md5(open(kf, "rb").read()).hexdigest()}
    return mat, shots, row_map


def test_persisted_embeddings_reproduce_live_picks_and_scores():
    import numpy as np
    tmp = tempfile.mkdtemp()
    try:
        mat, shots, row_map = _fake_pool(tmp)
        te = np.ones(8, dtype="float32") / np.sqrt(np.float32(8.0))
        by_path = {shots[("s1", j)].keyframe_path: mat[j] for j in range(len(mat))}

        class FakeVR:                                     # live path reads the keyframe file
            @staticmethod
            def _img_embed(im):
                return by_path[getattr(im, "filename", "")]

            @staticmethod
            def _txt_embed(text):
                return te

        orig_vr = IF._vr
        IF._vr = lambda: FakeVR
        seg = NS(index=0, scene_query="tywin at the council", expected_visual="", text="")
        try:
            def run(embeds_of, rounds=3):
                picks, used = [], set()
                memo = {} if embeds_of else None
                for _ in range(rounds):                  # the sliding-scan candidate rounds
                    p = IF.pick_pool_still(seg, shots, used, set(),
                                           embeds_of=embeds_of, rel_memo=memo)
                    if not p:
                        break
                    picks.append(p)
                    used.add((p[1], p[2]))
                return picks

            live = run(None)
            fast = run(lambda sid: (mat, row_map))
            assert live and len(live) == len(fast), "same number of candidate picks"
            for a, b in zip(live, fast):
                assert a[:3] == b[:3] and a[4] == b[4], f"pick differs: {a} vs {b}"
                assert abs(a[3] - b[3]) == 0.0, f"score differs: {a[3]} vs {b[3]}"
        finally:
            IF._vr = orig_vr
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_persisted_path_falls_back_to_live_on_missing_or_stale_rows():
    import numpy as np
    tmp = tempfile.mkdtemp()
    try:
        mat, shots, row_map = _fake_pool(tmp, n=6)
        shots[("s1", 2)].embed_row = -1                   # never embedded
        shots[("s1", 3)].embed_row = 99                   # stale: past the matrix bounds
        te = np.ones(8, dtype="float32") / np.sqrt(np.float32(8.0))
        by_path = {shots[("s1", j)].keyframe_path: mat[j] for j in range(6)}
        live_calls = {"n": 0}

        class FakeVR:
            @staticmethod
            def _img_embed(im):
                live_calls["n"] += 1
                return by_path[getattr(im, "filename", "")]

            @staticmethod
            def _txt_embed(text):
                return te

        orig_vr = IF._vr
        IF._vr = lambda: FakeVR
        try:
            seg = NS(index=0, scene_query="tywin at the council", expected_visual="", text="")
            p = IF.pick_pool_still(seg, shots, set(), set(),
                                   embeds_of=lambda sid: (mat, row_map))
            assert p is not None
            assert live_calls["n"] == 2, \
                f"exactly the missing/stale rows fall back to live embeds, got {live_calls['n']}"
        finally:
            IF._vr = orig_vr
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


TESTS = [
    test_venue_fallback_is_in_the_key_and_false_preserves_legacy_keys,
    test_corrupt_cache_file_is_ignored_not_fatal,
    test_repeated_pass_replays_decisions_from_cache_with_zero_calls,
    test_strict_and_lenient_questions_never_share_a_key,
    test_transport_failures_are_never_cached,
    test_workers4_is_decision_identical_to_serial,
    test_workers4_backend_failure_aborts_pool_and_keeps_breaker_contract,
    test_persisted_embeddings_reproduce_live_picks_and_scores,
    test_persisted_path_falls_back_to_live_on_missing_or_stale_rows,
]

if __name__ == "__main__":
    for fn in TESTS:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception as e:
            FAILS.append(fn.__name__)
            print(f"FAIL  {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(TESTS) - len(FAILS)}/{len(TESTS)} passed")
    sys.exit(1 if FAILS else 0)
