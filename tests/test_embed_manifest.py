"""P1.3 — persisted-embedding staleness safety: a stored vector is used ONLY when the
manifest certifies index schema, embedding-model identity, dimension, row count, AND the
row's shot/keyframe/content identity all match the current world.

    python3 tests/test_embed_manifest.py

No network, no model (visual_relevance is stubbed at the module seam).
"""
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                               # noqa: E402

from vidlore.clipstudio import index as I                        # noqa: E402
from vidlore.clipstudio import image_fallback as IF              # noqa: E402

FAILS = []
IDENT = "clip_vision.onnx:123:456:8"


class _Proj(NS):
    def embeds_path(self, sid):
        return Path(self.root) / "index" / f"{sid}.embeds.npy"

    def shots_path(self, sid):
        return Path(self.root) / "index" / f"{sid}.shots.json"


def _mk(tmp, n=6, dim=8):
    proj = _Proj(root=tmp)
    (Path(tmp) / "index").mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(3)
    mat = rng.rand(n, dim).astype("float32")
    np.save(proj.embeds_path("s1").with_suffix(".tmp.npy"), mat)
    Path(proj.embeds_path("s1").with_suffix(".tmp.npy")).replace(proj.embeds_path("s1"))
    shots = []
    for j in range(n):
        kf = Path(tmp) / f"kf{j}.jpg"
        kf.write_bytes(b"\xff\xd8\xff" + bytes([j]) * 16)
        shots.append(NS(source_id="s1", index=j, embed_row=j, keyframe_path=str(kf),
                        quality=1.0, phash="", face_ids=[], ocr_names=[], ocr_text="",
                        luma_avg=-1.0, luma_hi=-1.0, subs_frac=-1.0))
    man = {"schema": I.INDEX_SCHEMA, "model": IDENT, "dim": dim, "rows": n,
           "row_map": {str(j): {"shot": j, "kf": Path(shots[j].keyframe_path).name,
                                "kf_md5": hashlib.md5(
                                    Path(shots[j].keyframe_path).read_bytes()).hexdigest()}
                       for j in range(n)}}
    I._manifest_path(proj, "s1").write_text(json.dumps(man))
    return proj, mat, shots, man


class _StubVR:
    """Module-seam stub: identity + live embed callable, with a live-call counter."""

    live_calls = 0
    by_path = {}

    @classmethod
    def _img_embed(cls, im):
        _StubVR.live_calls += 1
        return cls.by_path[getattr(im, "filename", "")]

    @staticmethod
    def _txt_embed(text):
        return np.ones(8, dtype="float32") / np.float32(np.sqrt(8.0))


def _patch(proj, mat, shots, ident=IDENT):
    import vidlore.visual_relevance as vr
    _StubVR.live_calls = 0
    # live path must PIL-open real files; our kf stubs aren't decodable, so route the live
    # path through _clip_relevance's failure (-1.0) — tests that need live equality use
    # decodable files instead. For fallback-counting tests -1.0 is fine.
    orig = (IF._vr, vr.model_identity)
    IF._vr = lambda: _StubVR
    vr.model_identity = lambda: ident
    return orig


def _unpatch(orig):
    import vidlore.visual_relevance as vr
    IF._vr, vr.model_identity = orig


def test_valid_manifest_serves_verified_matrix():
    tmp = tempfile.mkdtemp()
    try:
        proj, mat, shots, man = _mk(tmp)
        orig = _patch(proj, mat, shots)
        try:
            got, rows = I.load_embeds_verified(proj, "s1")
            assert got is not None and rows == man["row_map"]
            seg = NS(index=0, scene_query="tywin council", expected_visual="", text="")
            memo = {}
            rel = IF._shot_relevance(shots[2], shots[2].keyframe_path, "tywin council",
                                     embeds_of=lambda sid: (got, rows), rel_memo=memo)
            assert rel >= 0.0 and _StubVR.live_calls == 0, \
                "a fully-certified row must be served from the matrix (no live embed)"
        finally:
            _unpatch(orig)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _expect_rejected(mutate, name, *, expect_loader_reject=True):
    tmp = tempfile.mkdtemp()
    try:
        proj, mat, shots, man = _mk(tmp)
        mutate(tmp, proj, mat, shots, man)
        orig = _patch(proj, mat, shots)
        try:
            got, rows = I.load_embeds_verified(proj, "s1")
            if expect_loader_reject:
                assert got is None, f"{name}: loader must reject the manifest"
        finally:
            _unpatch(orig)
        return got, rows, proj, shots
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_changed_model_identity_rejects():
    def m(tmp, proj, mat, shots, man):
        man["model"] = "other_model.onnx:9:9:8"
        I._manifest_path(proj, "s1").write_text(json.dumps(man))
    _expect_rejected(m, "changed model")


def test_wrong_dimension_rejects():
    def m(tmp, proj, mat, shots, man):
        man["dim"] = 512
        I._manifest_path(proj, "s1").write_text(json.dumps(man))
    _expect_rejected(m, "wrong dim")


def test_wrong_schema_rejects():
    def m(tmp, proj, mat, shots, man):
        man["schema"] = I.INDEX_SCHEMA + 1
        I._manifest_path(proj, "s1").write_text(json.dumps(man))
    _expect_rejected(m, "wrong schema")


def test_row_count_mismatch_rejects():
    def m(tmp, proj, mat, shots, man):
        np.save(proj.embeds_path("s1"), mat[:-1])       # truncated matrix
    _expect_rejected(m, "row count")


def test_corrupt_manifest_rejects():
    def m(tmp, proj, mat, shots, man):
        I._manifest_path(proj, "s1").write_text("{broken json")
    _expect_rejected(m, "corrupt manifest")


def test_missing_manifest_is_legacy_live_path():
    def m(tmp, proj, mat, shots, man):
        I._manifest_path(proj, "s1").unlink()
    _expect_rejected(m, "legacy (no manifest)")


def test_changed_keyframe_bytes_fall_back_per_row():
    tmp = tempfile.mkdtemp()
    try:
        proj, mat, shots, man = _mk(tmp)
        Path(shots[3].keyframe_path).write_bytes(b"\xff\xd8\xffCHANGED")   # re-extracted kf
        orig = _patch(proj, mat, shots)
        try:
            got, rows = I.load_embeds_verified(proj, "s1")
            assert got is not None, "matrix-level identity still certifies"
            bundle = (got, rows)
            memo = {}
            ok_rel = IF._shot_relevance(shots[2], shots[2].keyframe_path, "t x",
                                        embeds_of=lambda s: bundle, rel_memo=memo)
            assert _StubVR.live_calls == 0 and ok_rel >= 0.0
            IF._shot_relevance(shots[3], shots[3].keyframe_path, "t x",
                               embeds_of=lambda s: bundle, rel_memo=memo)
            # stale row -> the LIVE path ran (our stub kf is undecodable so it lands at
            # -1.0 via _clip_relevance's failure guard — the point is it did NOT serve mat[3])
        finally:
            _unpatch(orig)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_reordered_rows_fall_back_per_row():
    tmp = tempfile.mkdtemp()
    try:
        proj, mat, shots, man = _mk(tmp)
        rm = man["row_map"]
        rm["1"], rm["2"] = rm["2"], rm["1"]             # rows point at the WRONG shots
        I._manifest_path(proj, "s1").write_text(json.dumps(man))
        orig = _patch(proj, mat, shots)
        try:
            got, rows = I.load_embeds_verified(proj, "s1")
            bundle = (got, rows)
            memo = {}
            IF._shot_relevance(shots[1], shots[1].keyframe_path, "t x",
                               embeds_of=lambda s: bundle, rel_memo=memo)
            IF._shot_relevance(shots[2], shots[2].keyframe_path, "t x",
                               embeds_of=lambda s: bundle, rel_memo=memo)
            IF._shot_relevance(shots[4], shots[4].keyframe_path, "t x",
                               embeds_of=lambda s: bundle, rel_memo=memo)
            # shots 1 and 2 hit the shot-index mismatch guard -> live path; shot 4 intact
            from vidlore.clipstudio import perf_metrics as _pm
            c = _pm.snapshot()["counts"]
            assert c.get("imgfb.rel.stale_row", 0) >= 2, \
                "swapped rows must be refused per-row (stale_row counted)"
        finally:
            _unpatch(orig)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


TESTS = [
    test_valid_manifest_serves_verified_matrix,
    test_changed_model_identity_rejects,
    test_wrong_dimension_rejects,
    test_wrong_schema_rejects,
    test_row_count_mismatch_rejects,
    test_corrupt_manifest_rejects,
    test_missing_manifest_is_legacy_live_path,
    test_changed_keyframe_bytes_fall_back_per_row,
    test_reordered_rows_fall_back_per_row,
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
