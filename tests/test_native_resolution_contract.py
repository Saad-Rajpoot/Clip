from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from vidlore.clipstudio.quality_contract import native_video_ok, assert_native_hd_selections


@pytest.mark.parametrize("width,height,ok", [
    (640, 360, False), (1280, 719, False), (1279, 720, False),
    (640, 720, False), (1280, 720, True), (1920, 1080, True),
    (720, 1280, True), (0, 0, False), (None, None, False), ("bad", 720, False),
])
def test_native_floor_requires_real_1280x720_raster(width, height, ok):
    assert native_video_ok({"width": width, "height": height}) is ok


def test_upscaled_container_metadata_cannot_override_native_probe(monkeypatch, tmp_path):
    src_file = tmp_path / "source.mp4"
    src_file.write_bytes(b"native bytes")
    src = NS(id="sd", local_path=str(src_file), title="SD source", height=1080)
    proj = NS(source=lambda sid: src if sid == "sd" else None)
    sel = NS(segment_index=7, source_id="sd", image_path="")
    monkeypatch.setattr("vidlore.clipstudio.ingest.probe",
                        lambda _p: {"width": 640, "height": 360})
    audit = tmp_path / "native.json"
    with pytest.raises(Exception, match="native-resolution gate"):
        assert_native_hd_selections(proj, [sel], audit)
    assert '"height": 360' in audit.read_text()


def test_verified_image_is_deferred_to_fullres_rescue(monkeypatch, tmp_path):
    proj = NS(source=lambda _sid: None)
    img = tmp_path / "verified.jpg"
    img.write_bytes(b"verified image bytes")
    sel = NS(segment_index=2, source_id="", image_path=str(img))
    got = assert_native_hd_selections(proj, [sel], tmp_path / "native.json")
    assert got["passed"] and got["selections"] == []


def test_missing_image_cannot_bypass_video_native_probe(monkeypatch, tmp_path):
    src_file = tmp_path / "source.mp4"
    src_file.write_bytes(b"sd")
    src = NS(id="sd", local_path=str(src_file), title="SD", height=1080)
    proj = NS(source=lambda _sid: src)
    sel = NS(segment_index=3, source_id="sd", image_path=str(tmp_path / "missing.jpg"))
    monkeypatch.setattr("vidlore.clipstudio.ingest.probe",
                        lambda _p: {"width": 640, "height": 360})
    with pytest.raises(Exception, match="native-resolution gate"):
        assert_native_hd_selections(proj, [sel], tmp_path / "native.json")


def test_build_wires_native_probe_before_selection_derivatives():
    build = (Path(__file__).resolve().parents[1] / "vidlore" / "clipstudio" /
             "build.py").read_text()
    gate = build.index("_assert_native_hd(")
    owned = build.index("_owned = _fit_verified_selection_clip(")
    assert gate < owned
    assert 'proj.output_dir / "native_resolution_audit.json"' in build[gate:owned]
