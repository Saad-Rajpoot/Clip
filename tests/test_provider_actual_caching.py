"""P1.2 — provider-ACTUAL verdict caching: a verdict is keyed by the provider that really
served it, never by the prediction; a Gemini-keyed lookup can never return a Claude answer.

    python3 tests/test_provider_actual_caching.py

No network. The provider ladder in llm.complete_ex is exercised for real with stubbed
provider transports; verify_and_repair runs for real over a stub project.
"""
import json
import os
import shutil
import sys
import tempfile
from types import SimpleNamespace as NS

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vidlore.clipstudio import llm as L                          # noqa: E402
from vidlore.clipstudio import verify as V                       # noqa: E402

FAILS = []
_VERDICT_TXT = json.dumps({"matches_narration": True, "correct_subject_visible": True,
                           "wrong_subject_visible": False, "specific_enough": True,
                           "quality_ok": True, "confidence": 0.9, "verdict": "keep",
                           "reason": "stub"})


class _Providers:
    """Stub every provider transport under llm; script per-provider health."""

    def __init__(self, gemini_ok=True, claude_ok=True):
        self.gemini_ok = gemini_ok
        self.claude_ok = claude_ok
        self.calls = {"gemini": 0, "claude": 0, "deepseek": 0}

    def install(self):
        self._orig = (L._gemini_complete, L._claude_complete, L._deepseek_complete,
                      L.gemini_available, L._claude_key, L._deepseek_key,
                      L._provider, L._retries, L._gemini_model, L._claude_model,
                      L._gemini_api_key)

        def gem(system, messages, max_tokens, model):
            self.calls["gemini"] += 1
            if not self.gemini_ok:
                raise RuntimeError("503 service unavailable")
            return _VERDICT_TXT

        def cla(system, messages, max_tokens, eng_cfg, model):
            self.calls["claude"] += 1
            if not self.claude_ok:
                raise RuntimeError("503 service unavailable")
            return _VERDICT_TXT

        def dsk(system, messages, max_tokens, model):
            self.calls["deepseek"] += 1
            raise AssertionError("vision calls must never reach DeepSeek")

        L._gemini_complete, L._claude_complete, L._deepseek_complete = gem, cla, dsk
        L.gemini_available = lambda: True
        L._claude_key = lambda eng_cfg=None: "k"
        L._deepseek_key = lambda eng_cfg=None: ""        # text brain irrelevant here
        L._provider = lambda: "deepseek"                 # production default ladder
        L._retries = lambda: 1                           # fail fast in tests
        L._gemini_model = lambda: "gemini-2.5-flash"
        L._claude_model = lambda: "claude-x"
        L._gemini_api_key = lambda: "gk"
        return self

    def restore(self):
        (L._gemini_complete, L._claude_complete, L._deepseek_complete,
         L.gemini_available, L._claude_key, L._deepseek_key,
         L._provider, L._retries, L._gemini_model, L._claude_model,
         L._gemini_api_key) = self._orig


GEM_ID = "gemini:gemini-2.5-flash:apikey"
CLA_ID = "anthropic:claude-x"


# ---------------------------------------------------------------------------
# A — complete_ex reports the ACTUAL server
# ---------------------------------------------------------------------------
def test_complete_ex_reports_actual_server_and_fallback():
    p = _Providers(gemini_ok=True).install()
    try:
        txt, meta = L.complete_ex(messages=[{"role": "user", "content": [
            {"type": "image", "source": {}}, {"type": "text", "text": "q"}]}])
        assert txt and meta["served"] == GEM_ID, meta
        p.gemini_ok = False
        txt2, meta2 = L.complete_ex(messages=[{"role": "user", "content": [
            {"type": "image", "source": {}}, {"type": "text", "text": "q"}]}])
        assert txt2 and meta2["served"] == CLA_ID, "fallback must be reported as Claude"
        p.gemini_ok = False
        p.claude_ok = False
        txt3, meta3 = L.complete_ex(messages=[{"role": "user", "content": [
            {"type": "image", "source": {}}, {"type": "text", "text": "q"}]}])
        assert txt3 == "" and meta3["served"] == "none", "total failure -> none"
        assert L.complete(messages=[{"role": "user", "content": [
            {"type": "text", "text": "q"}]}]) == "", "legacy wrapper intact (deepseek off)"
    finally:
        p.restore()


def test_verify_frame_attaches_actual_provenance():
    p = _Providers(gemini_ok=False).install()
    try:
        tmp = tempfile.mkdtemp()
        kf = os.path.join(tmp, "kf.jpg")
        from PIL import Image
        Image.new("RGB", (8, 8)).save(kf, "JPEG")
        v = V.verify_frame(kf, "line", "Tywin", "character", [], NS(anthropic_model="m"))
        assert v and v["vision_served_by"] == CLA_ID, v
        shutil.rmtree(tmp, ignore_errors=True)
    finally:
        p.restore()


# ---------------------------------------------------------------------------
# B — end-to-end: gemini fail -> claude fallback -> gemini recovery
# ---------------------------------------------------------------------------
def _mini(tmp, n=4):
    from vidlore.clipstudio.models import ClipProject, ScriptSegment, ClipSelection, SourceVideo, Shot
    from PIL import Image
    proj = ClipProject(name="t", root=str(tmp))
    proj.ensure_dirs()
    media = os.path.join(tmp, "src.mp4")
    with open(media, "wb") as fh:
        fh.write(b"\0" * 2048)
    proj.sources = [SourceVideo(id="s1", url="u", title="Game of Thrones S03E10",
                                permission="owner", status="ok", local_path=media)]
    segs, shots = [], []
    for i in range(n):
        kf = os.path.join(tmp, f"kf{i}.jpg")
        Image.new("RGB", (8, 8), (i * 30 % 255, 10, 10)).save(kf, "JPEG")
        segs.append(ScriptSegment(index=i, text=f"beat {i}", required_entity="Tywin Lannister",
                                  required_kind="character", visual_policy="exact_scene",
                                  is_specific_claim=True))
        shots.append(Shot(source_id="s1", index=i, start=float(i), end=float(i) + 2.0,
                          keyframe_path=kf))
        proj.selections.append(ClipSelection(segment_index=i, source_id="s1", shot_index=i,
                                             in_point=float(i), out_point=float(i) + 2.0,
                                             confidence=0.8))
    proj.meta["analysis"] = {"video_type": "single_scene", "episode_hint": "S03E10",
                             "episode_hint_verified": True, "characters": [], "actors": []}
    return proj, segs, shots


def _verify_pass(tmp, providers):
    from vidlore.clipstudio.config import ClipConfig
    proj, segs, shots = _mini(tmp)
    by = {(s.source_id, s.index): s for s in shots}
    orig_lookup = V._shot_lookup
    orig_env = os.environ.get("VIDLORE_CLIPSTUDIO_VERIFY_ACTION_SHEET")
    os.environ["VIDLORE_CLIPSTUDIO_VERIFY_ACTION_SHEET"] = "0"
    V._shot_lookup = lambda p: (lambda sid, ix: by.get((sid, ix)))
    try:
        summ = V.verify_and_repair(proj, segs, ClipConfig(),
                                   NS(anthropic_model="m", anthropic_key="k"), progress=None)
    finally:
        V._shot_lookup = orig_lookup
        if orig_env is None:
            os.environ.pop("VIDLORE_CLIPSTUDIO_VERIFY_ACTION_SHEET", None)
        else:
            os.environ["VIDLORE_CLIPSTUDIO_VERIFY_ACTION_SHEET"] = orig_env
    return summ, proj


def test_fallback_verdicts_key_under_actual_provider_and_never_cross_reuse():
    tmp = tempfile.mkdtemp()
    try:
        # PASS 1 — Gemini DOWN: Claude serves every verdict
        p = _Providers(gemini_ok=False).install()
        try:
            summ1, _ = _verify_pass(tmp, p)
        finally:
            p.restore()
        c1, g1 = p.calls["claude"], p.calls["gemini"]
        assert c1 == 4 and summ1["verified"] == 4, (p.calls, summ1)
        cache = json.load(open(os.path.join(tmp, "verdict_cache.json")))
        assert len(cache) == 4
        assert all(v.get("vision_served_by") == CLA_ID for v in cache.values()), \
            "every stored verdict must record the ACTUAL (Claude) server"
        # no entry may be keyed under the predicted-Gemini fingerprint: every stored key must
        # be reproducible with model=CLA_ID
        # PASS 2 — Gemini RECOVERED: predicted-Gemini lookups must MISS the Claude entries
        # (no cross-provider reuse) and re-ask Gemini fresh
        p2 = _Providers(gemini_ok=True).install()
        try:
            summ2, _ = _verify_pass(tmp, p2)
        finally:
            p2.restore()
        assert p2.calls["gemini"] == 4, \
            f"recovered Gemini must be re-asked all 4 questions, asked {p2.calls['gemini']}"
        assert p2.calls["claude"] == 0, "Claude must not serve when Gemini is healthy"
        assert summ2["reused"] == 0, "Claude-keyed entries must never satisfy Gemini lookups"
        cache2 = json.load(open(os.path.join(tmp, "verdict_cache.json")))
        assert len(cache2) == 8, "4 Claude-keyed + 4 Gemini-keyed entries coexist"
        # PASS 3 — Gemini healthy again: now the Gemini-keyed entries DO serve
        p3 = _Providers(gemini_ok=True).install()
        try:
            summ3, _ = _verify_pass(tmp, p3)
        finally:
            p3.restore()
        assert p3.calls["gemini"] == 0 and summ3["reused"] == 4, \
            "same-provider warm pass must be pure cache hits"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_poisoned_cross_provider_entry_is_dropped_at_lookup():
    """Belt-and-suspenders: an entry sitting under a Gemini key but RECORDING a Claude server
    (a mislabeled legacy write) must not be served."""
    tmp = tempfile.mkdtemp()
    try:
        p = _Providers(gemini_ok=True).install()
        try:
            _verify_pass(tmp, p)                        # populate 4 gemini-keyed entries
            cache = json.load(open(os.path.join(tmp, "verdict_cache.json")))
            for k in cache:                             # poison: claim Claude served them
                cache[k]["vision_served_by"] = CLA_ID
            json.dump(cache, open(os.path.join(tmp, "verdict_cache.json"), "w"))
            p2 = _Providers(gemini_ok=True).install()
            try:
                summ, _ = _verify_pass(tmp, p2)
            finally:
                p2.restore()
            assert p2.calls["gemini"] == 4 and summ["reused"] == 0, \
                "a provider-mismatched entry must be re-asked, not served"
        finally:
            p.restore()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


TESTS = [
    test_complete_ex_reports_actual_server_and_fallback,
    test_verify_frame_attaches_actual_provenance,
    test_fallback_verdicts_key_under_actual_provider_and_never_cross_reuse,
    test_poisoned_cross_provider_entry_is_dropped_at_lookup,
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
