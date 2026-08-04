"""Zero-risk API-cost fixes — every one must be DECISION-NEUTRAL.

Three fixes, from the 42-agent cost audit of jobs 9e3a5b9bfd ($1.26 / 2106 flash calls) and
5462677f95 ($1.88 / 3123):

  A. ACCOUNTING — `_USAGE` was never reset (the portal is long-lived, so every job's report also
     contained the previous jobs'), the cost dump sat after build (so a gate-blocked render — the
     single most expensive failure mode — recorded NOTHING), and `stage=` was never passed by any
     call site (`stages: {}` in every report ever written).
  B. STAGED RUNG PREFETCH — phase-2 warmed all max_replacements alternates plus a lenient question
     per beat; the serial loop stops at its first accepted alternate, so most were never read.
  C. SELF-HEAL CACHE — `_venue_verify` was the last uncached vision path: self-heal runs up to 3
     rounds (twice over on a review-draft retry) and re-bought identical verdicts every round.

The bar for all three: warming/booking FEWER questions may never change a decision, because
anything unwarmed is simply asked fresh by the serial loop — today's exact behaviour.

    python3 -m pytest tests/test_cost_zero_risk.py -q

No network, no LLM, no ffmpeg.
"""
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from vidlore.clipstudio import llm as L                          # noqa: E402
from vidlore.clipstudio import orchestrate as O                  # noqa: E402
from vidlore.clipstudio import selfheal as SH                    # noqa: E402
from vidlore.clipstudio import verify as V                       # noqa: E402
from vidlore.clipstudio.config import ClipConfig                 # noqa: E402

import test_perf_neutral_caching as H                            # noqa: E402  (shared fixture)

SRC = ROOT / "vidlore" / "clipstudio"
_REJECT = dict(H._REJECT_ALL)
_KEEP = {"verdict": "keep", "correct_subject_visible": True, "matches_narration": True,
         "wrong_subject_visible": False, "contradicts_narration": False,
         "specific_enough": True, "quality_ok": True, "confidence": 0.9, "reason": "ok"}


# ─────────────────────────────────────────────────────────── A. accounting
class TestAccounting(unittest.TestCase):
    def setUp(self):
        L.reset_usage()

    def tearDown(self):
        L.reset_usage()

    def test_reset_clears_usage_stage_and_claude_counter(self):
        L.set_stage("verify")
        L.record_usage("gemini-2.5-flash", prompt=10, completion=2)
        L._CLAUDE_VISION_CALLS[0] = 7
        L.reset_usage()
        self.assertEqual(L.usage_summary()["calls"], 0)
        self.assertEqual(L._CLAUDE_VISION_CALLS[0], 0)
        self.assertEqual(L.current_stage(), "")

    def test_stage_attribution_records_calls_and_tokens(self):
        L.set_stage("AI verify + repair")
        L.record_usage("gemini-2.5-flash", prompt=1000, completion=100)
        with L.usage_stage("assemble final video"):
            L.record_usage("gemini-2.5-flash", prompt=500, completion=50)
        L.record_usage("gemini-2.5-flash", prompt=200, completion=20)   # stage restored
        m = L.usage_summary()["models"]["gemini-2.5-flash"]
        self.assertEqual(m["stages"], {"AI verify + repair": 2, "assemble final video": 1})
        self.assertEqual(m["stage_tokens"]["AI verify + repair"], {"prompt": 1200,
                                                                   "completion": 120})
        self.assertEqual(m["stage_tokens"]["assemble final video"], {"prompt": 500,
                                                                     "completion": 50})

    def test_stage_label_survives_worker_threads(self):
        """verify/still/self-heal all fan out to pools — a ContextVar alone would lose the label."""
        import concurrent.futures as cf
        L.set_stage("AI verify + repair")
        with cf.ThreadPoolExecutor(max_workers=4) as ex:
            list(ex.map(lambda _i: L.record_usage("gemini-2.5-flash", prompt=1, completion=1),
                        range(8)))
        self.assertEqual(L.usage_summary()["models"]["gemini-2.5-flash"]["stages"],
                         {"AI verify + repair": 8})

    def test_flash_lite_has_a_price_row(self):
        L.record_usage("gemini-2.5-flash-lite", prompt=1_000_000, completion=1_000_000)
        usd = L.usage_summary()["models"]["gemini-2.5-flash-lite"]["usd"]
        self.assertAlmostEqual(usd, 0.50, places=4)      # 0.10 in + 0.40 out, not $0

    def test_thinking_tokens_are_billed_as_output(self):
        resp = NS(usage_metadata=NS(prompt_token_count=100, candidates_token_count=20,
                                    thoughts_token_count=900))
        self.assertEqual(L._usage_from(resp), (100, 920))

    def test_claude_vision_budget_guards_the_price_cliff(self):
        img = [{"content": [{"type": "image"}]}]
        text = [{"content": [{"type": "text"}]}]
        with unittest_env(VIDLORE_CLIPSTUDIO_MAX_CLAUDE_VISION="2"):
            L.reset_usage()
            self.assertTrue(L._claude_vision_budget_ok(img))     # 1
            self.assertTrue(L._claude_vision_budget_ok(img))     # 2
            self.assertFalse(L._claude_vision_budget_ok(img))    # 3 → refused, fail-closed
            self.assertTrue(L._claude_vision_budget_ok(text))    # text calls never counted

    def test_default_cap_is_generous_enough_for_transient_fallbacks(self):
        os.environ.pop("VIDLORE_CLIPSTUDIO_MAX_CLAUDE_VISION", None)
        L.reset_usage()
        img = [{"content": [{"type": "image"}]}]
        self.assertTrue(all(L._claude_vision_budget_ok(img) for _ in range(50)))

    def test_persist_cost_writes_partial_report_and_never_raises(self):
        with tempfile.TemporaryDirectory() as td:
            L.reset_usage()
            L.record_usage("gemini-2.5-flash", prompt=1000, completion=100)
            c = O._persist_cost(td, partial=True)
            import json
            d = json.loads((Path(td) / "output" / "cost_report.json").read_text())
            self.assertTrue(d["partial"])
            self.assertEqual(d["calls"], 1)
            self.assertEqual(c["calls"], 1)
        self.assertEqual(O._persist_cost("/nonexistent/\0bad", partial=True), {})

    def test_wrapper_resets_per_job_but_accumulates_a_same_job_resume(self):
        """A retry's true price INCLUDES the attempt that failed — that abort→retry cycle is the
        most expensive pattern the audit found and must not be reset away."""
        calls = {"n": 0}

        def fake_impl(project_dir, **kw):
            calls["n"] += 1
            L.record_usage("gemini-2.5-flash", prompt=1000, completion=100)
            return {"ok": True}

        orig = O._produce_auto
        O._produce_auto = fake_impl
        try:
            with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
                O.produce_auto(a)
                self.assertEqual(L.usage_summary()["calls"], 1)
                O.produce_auto(a, resume=True)                  # same job resumed → accumulate
                self.assertEqual(L.usage_summary()["calls"], 2)
                O.produce_auto(b, resume=True)                  # DIFFERENT job → fresh scope
                self.assertEqual(L.usage_summary()["calls"], 1)
                O.produce_auto(b)                               # fresh run → fresh scope
                self.assertEqual(L.usage_summary()["calls"], 1)
        finally:
            O._produce_auto = orig
            L.reset_usage()

    def test_wrapper_records_on_a_raise_and_reraises_the_original(self):
        def boom(project_dir, **kw):
            L.record_usage("gemini-2.5-flash", prompt=2000, completion=200)
            raise V.NonRetryableBuildError("rejected-footage gate: ...", kind="rejected_footage")

        orig = O._produce_auto
        O._produce_auto = boom
        try:
            with tempfile.TemporaryDirectory() as td:
                with self.assertRaises(V.NonRetryableBuildError) as ctx:
                    O.produce_auto(td)
                self.assertEqual(getattr(ctx.exception, "kind", ""), "rejected_footage")
                import json
                d = json.loads((Path(td) / "output" / "cost_report.json").read_text())
                self.assertTrue(d["partial"])
                self.assertEqual(d["calls"], 1)
        finally:
            O._produce_auto = orig
            L.reset_usage()

    def test_public_signature_stays_introspectable(self):
        import inspect
        p = inspect.signature(O.produce_auto).parameters
        for name in ("resume", "verify", "do_build", "policy"):
            self.assertIn(name, p)


class unittest_env:
    """Tiny scoped-env helper (env restored even on failure)."""

    def __init__(self, **kv):
        self.kv = kv
        self.old = {}

    def __enter__(self):
        for k, v in self.kv.items():
            self.old[k] = os.environ.get(k)
            os.environ[k] = v
        return self

    def __exit__(self, *exc):
        for k, v in self.old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        return False


# ─────────────────────────────────────────────────────────── B. staged rung prefetch
def _drive_verify(workers, rule, n_beats=6):
    """Run the real verify_and_repair over the shared fixture with an ORDER-INDEPENDENT stub
    (the verdict depends on the frame, never on call order — otherwise prefetch reordering
    alone would look like a decision change)."""
    tmp = tempfile.mkdtemp()
    calls = [0]
    proj, segs, shots = H._mini_project_with_alternates(tmp, n_beats)
    by = {(s.source_id, s.index): s for s in shots}

    def lookup(_p):
        def get(sid, ix):
            return by.get((sid, ix))
        get.all_shots = lambda sid: [s for s in shots if s.source_id == sid]
        return get

    def fake_verify_frame(*a, **k):
        calls[0] += 1
        m = re.search(r"kf(\d+)\.jpg", str(a[0]) if a else "")
        return rule(int(m.group(1)) % 3 if m else 0)

    o_vf, o_lk = V.verify_frame, V._shot_lookup
    V.verify_frame, V._shot_lookup = fake_verify_frame, lookup
    with unittest_env(VIDLORE_CLIPSTUDIO_VERIFY_ACTION_SHEET="0",
                      VIDLORE_CLIPSTUDIO_VERIFY_WORKERS=str(workers)):
        try:
            V.verify_and_repair(proj, segs, ClipConfig(),
                                NS(anthropic_model="m", anthropic_key="k"), progress=None)
        finally:
            V.verify_frame, V._shot_lookup = o_vf, o_lk
    return calls[0], H._decision_state(proj)


class TestStagedPrefetch(unittest.TestCase):
    def test_promotion_succeeds_costs_no_more_than_serial(self):
        """alt #1 accepted → waves 2/3 skipped AND the lenient warm dropped. Eagerly this
        fixture paid 18 calls for 12 questions the loop actually asked."""
        rule = lambda off: dict(_REJECT) if off == 0 else dict(_KEEP)   # noqa: E731
        c_serial, s_serial = _drive_verify(1, rule)
        c_staged, s_staged = _drive_verify(4, rule)
        self.assertEqual(s_serial, s_staged, "decisions must be identical")
        self.assertLessEqual(c_staged, c_serial,
                             f"staged prefetch must not cost more than serial "
                             f"({c_staged} vs {c_serial})")

    def test_worst_case_full_ladder_is_also_cost_neutral(self):
        rule = lambda off: dict(_REJECT)                                # noqa: E731
        c_serial, s_serial = _drive_verify(1, rule)
        c_staged, s_staged = _drive_verify(4, rule)
        self.assertEqual(s_serial, s_staged)
        self.assertLessEqual(c_staged, c_serial)

    def test_wiring_contracts(self):
        src = (SRC / "verify.py").read_text()
        seg = src.split("PHASE-2 RUNG PREFETCH")[1].split("for sel in proj.selections:")[0]
        self.assertIn("_alt_waves", seg)                      # depth paid one wave at a time
        self.assertIn("_alt_done", seg)
        self.assertIn("_exact_contextual_ok(_v0", seg)        # keep-contextual beats: no lenient
        self.assertIn("if not _alt_done.get(_bi)", seg)       # promoted beats: no lenient either
        self.assertIn("VIDLORE_CLIPSTUDIO_VERIFY_PREFETCH_RUNGS", seg)


# ─────────────────────────────────────────────────────────── C. self-heal verdict cache
def _seg(idx=0):
    from vidlore.clipstudio.models import ScriptSegment
    return ScriptSegment(index=idx, text="Ned kneels on the steps.",
                         scene_query="Ned Stark execution Sept of Baelor",
                         required_entity="Ned Stark", required_kind="character",
                         visual_policy="exact_scene")


def _proj_with_kf(td):
    from vidlore.clipstudio.models import ClipProject, SourceVideo
    proj = ClipProject(name="t", root=td)
    proj.ensure_dirs()
    media = Path(td) / "src.mp4"
    media.write_bytes(b"\0" * 2048)
    proj.sources = [SourceVideo(id="s1", url="u", title="GoT", permission="owner",
                                status="ok", local_path=str(media))]
    kf = proj.index_dir / "s1" / "keyframes" / "shot_0000.jpg"
    kf.parent.mkdir(parents=True, exist_ok=True)
    kf.write_bytes(b"\xff\xd8\xff\x01")
    return proj, str(kf)


class TestSelfhealCache(unittest.TestCase):
    def setUp(self):
        SH._VV_CACHE["root"], SH._VV_CACHE["data"] = "", {}

    def test_identical_question_is_answered_once(self):
        with tempfile.TemporaryDirectory() as td:
            proj, kf = _proj_with_kf(td)
            seg, eng = _seg(), NS(anthropic_model="m", anthropic_key="k")
            calls = [0]

            def fake(*a, **k):
                calls[0] += 1
                return dict(_KEEP)

            o = V.verify_frame
            V.verify_frame = fake
            try:
                cache = SH._venue_cache(proj)
                v1 = SH._venue_verify(kf, seg, [], eng, proj=proj, cache=cache)
                v2 = SH._venue_verify(kf, seg, [], eng, proj=proj, cache=cache)
            finally:
                V.verify_frame = o
            self.assertEqual(calls[0], 1, "the second identical question must be free")
            self.assertEqual(v1["verdict"], v2["verdict"])

    def test_malformed_positive_hit_is_reasked_and_repaired(self):
        with tempfile.TemporaryDirectory() as td:
            proj, kf = _proj_with_kf(td)
            seg, eng = _seg(), NS(anthropic_model="m", anthropic_key="k")
            cache = SH._venue_cache(proj)
            fp = SH._venue_fp(proj, kf, seg, [], eng)
            poisoned = dict(_KEEP)
            poisoned["matches_naration"] = poisoned.pop("matches_narration")
            cache[fp] = poisoned
            calls = [0]

            def fake(*a, **k):
                calls[0] += 1
                return dict(_KEEP)

            o = V.verify_frame
            V.verify_frame = fake
            try:
                verdict = SH._venue_verify(kf, seg, [], eng, proj=proj, cache=cache)
            finally:
                V.verify_frame = o
            self.assertEqual(calls[0], 1, "a malformed cached keep must not suppress vision")
            self.assertEqual(verdict.get("matches_narration"), True)
            self.assertNotIn("matches_naration", cache[fp])

    def test_uncached_when_no_cache_is_supplied(self):
        """Back-compat: the bare call signature still works and simply pays."""
        with tempfile.TemporaryDirectory() as td:
            _proj, kf = _proj_with_kf(td)
            calls = [0]

            def fake(*a, **k):
                calls[0] += 1
                return dict(_KEEP)

            o = V.verify_frame
            V.verify_frame = fake
            try:
                SH._venue_verify(kf, _seg(), [], NS(anthropic_model="m"))
                SH._venue_verify(kf, _seg(), [], NS(anthropic_model="m"))
            finally:
                V.verify_frame = o
            self.assertEqual(calls[0], 2)

    def test_key_does_not_collide_with_the_still_layer_question(self):
        """selfheal sends NO era_hint; the still layer bakes era into its key. Sharing entries
        would answer one layer's question with the other layer's verdict."""
        with tempfile.TemporaryDirectory() as td:
            proj, kf = _proj_with_kf(td)
            seg, eng = _seg(), NS(anthropic_model="m")
            fp_selfheal = SH._venue_fp(proj, kf, seg, [], eng)
            fp_with_era = V.verdict_fingerprint(
                src_hash=SH._src_hash_of(proj, kf), source_id="s1",
                shot_start=0.0, shot_end=0.0, beat_text=seg.text,
                required_entity=seg.required_entity, required_kind=seg.required_kind,
                expected_visual="", scene_query=seg.scene_query, era="S01E09",
                visual_policy="exact_scene", is_specific=False, faceid_names=[],
                multiframe=False, image_id=f"kf:{V._file_fingerprint(kf)}",
                model=SH._vision_model(eng), venue_fallback=True, must_see="")
            self.assertTrue(fp_selfheal)
            self.assertNotEqual(fp_selfheal, fp_with_era)

    def test_only_successful_verdicts_are_stored(self):
        with tempfile.TemporaryDirectory() as td:
            proj, kf = _proj_with_kf(td)
            o = V.verify_frame
            V.verify_frame = lambda *a, **k: None          # transport failure
            try:
                cache = SH._venue_cache(proj)
                SH._venue_verify(kf, _seg(), [], NS(anthropic_model="m"),
                                 proj=proj, cache=cache)
            finally:
                V.verify_frame = o
            self.assertEqual(cache, {}, "errors must never be cached (retry contract)")

    def test_save_merges_instead_of_clobbering_other_layers(self):
        with tempfile.TemporaryDirectory() as td:
            proj, _kf = _proj_with_kf(td)
            V._save_verdict_cache(proj, {"from_verify": {"verdict": "keep"}})
            SH._venue_cache(proj)                          # loads {'from_verify': ...}
            SH._VV_CACHE["data"]["from_selfheal"] = {"verdict": "keep"}
            V._save_verdict_cache(proj, {"from_verify": {"verdict": "keep"},
                                         "written_later": {"verdict": "replace"}})
            SH._venue_cache_save(proj)
            on_disk = V._load_verdict_cache(proj)
            for k in ("from_verify", "written_later", "from_selfheal"):
                self.assertIn(k, on_disk, f"{k} was clobbered")

    def test_wave_warming_stops_after_a_keep(self):
        src = (SRC / "selfheal.py").read_text()
        seg = src.split("def still_recover")[1].split("def _frame_text_dirty")[0]
        self.assertIn("_warm_wave", seg)
        self.assertIn("VIDLORE_CLIPSTUDIO_SELFHEAL_VERIFY_WAVE", seg)
        self.assertIn("break", seg.split("for _i in range(0, len(cands), _wave):")[1][:200])
        # the accept walk itself is untouched: ranked order, first keep wins
        self.assertIn("for rel, sid, sh in cands:", seg)
        self.assertIn('v.get("verdict") == "keep"', seg)

    def test_heal_pass_persists_the_cache(self):
        src = (SRC / "selfheal.py").read_text()
        self.assertIn("_venue_cache_save(proj)", src.split("def heal_blocked_beats")[1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
